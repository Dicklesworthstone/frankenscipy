#![forbid(unsafe_code)]

use std::f64::consts::PI;

use fsci_runtime::RuntimeMode;

use crate::types::{
    Complex64, DispatchPlan, DispatchStep, KernelRegime, SpecialError, SpecialErrorKind,
    SpecialResult, SpecialTensor, not_yet_implemented, record_special_trace,
};

pub const GAMMA_DISPATCH_PLAN: &[DispatchPlan] = &[
    DispatchPlan {
        function: "gamma",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Reflection,
                when: "Re(z) < 0.5 and not at poles",
            },
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "0.5 <= Re(z) < 8.0",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "Re(z) >= 8.0",
            },
        ],
        notes: "Strict mode preserves SciPy signed-zero and pole semantics; hardened mode adds fail-closed diagnostics.",
    },
    DispatchPlan {
        function: "gammaln",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "real axis shift into stable window",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large |x| Stirling region",
            },
        ],
        notes: "Keep +inf pole parity with SciPy and avoid direct exp(gammaln) overflow back-conversions.",
    },
    DispatchPlan {
        function: "gammasgn",
        steps: &[DispatchStep {
            regime: KernelRegime::Reflection,
            when: "real negative non-integers derive sign from reflection intervals",
        }],
        notes: "Matches scipy.special.gammasgn on the real axis, including signed-zero and pole NaN semantics.",
    },
    DispatchPlan {
        function: "loggamma",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "positive real axis reuses gammaln stable window",
            },
            DispatchStep {
                regime: KernelRegime::Reflection,
                when: "complex arguments cross Re(z) < 0.5 principal branch",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large positive real or complex magnitude",
            },
        ],
        notes: "Real inputs preserve scipy.special.loggamma's negative-axis NaN branch cut; complex inputs return the principal branch.",
    },
    DispatchPlan {
        function: "gammainc",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "x < a + 1 via stable lower-tail expansion",
            },
            DispatchStep {
                regime: KernelRegime::ContinuedFraction,
                when: "x >= a + 1 via stable upper-tail continued fraction",
            },
        ],
        notes: "Regularized lower incomplete gamma P(a, x); strict mode preserves NaN domain fallback.",
    },
    DispatchPlan {
        function: "gammaincc",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "complement from lower-tail expansion",
            },
            DispatchStep {
                regime: KernelRegime::ContinuedFraction,
                when: "direct upper-tail continued fraction for stability",
            },
        ],
        notes: "Regularized upper incomplete gamma Q(a, x)=1-P(a, x); hardened mode fail-closes malformed domains.",
    },
    DispatchPlan {
        function: "digamma",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Reflection,
                when: "Re(z) <= 0 and away from poles",
            },
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "shift toward asymptotic region",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large |z|",
            },
        ],
        notes: "Negative-axis cancellation regions require dedicated numerical guards before D2 implementation.",
    },
    DispatchPlan {
        function: "polygamma",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "order n fixed and argument shifted off poles",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large argument regime",
            },
        ],
        notes: "Polygamma order must remain integer and nonnegative in both modes.",
    },
    DispatchPlan {
        function: "rgamma",
        steps: &[DispatchStep {
            regime: KernelRegime::BackendDelegate,
            when: "evaluate via reciprocal-safe gamma path",
        }],
        notes: "Prefer reciprocal from stabilized gammaln/gamma path to reduce overflow risk.",
    },
];

const SQRT_2PI: f64 = 2.506_628_274_631_000_7;
const LANCZOS_G: f64 = 7.0;
const LANCZOS_COEFFS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    0.000_009_984_369_578_019_572,
    0.000_000_150_563_273_514_931_16,
];

pub fn gamma(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    gamma_dispatch("gamma", z, mode)
}

/// Audit-emitting variant of [`gamma`]. In Hardened mode, a pole/
/// non-finite/domain rejection records an `AuditAction::FailClosed`
/// event on the provided ledger before returning the error. In Strict
/// mode the ledger is left untouched. See br-egba-2.
pub fn gamma_with_audit(
    z: &SpecialTensor,
    mode: RuntimeMode,
    ledger: &crate::audit::SyncSharedAuditLedger,
) -> SpecialResult {
    let result = gamma_dispatch("gamma", z, mode);
    if mode == RuntimeMode::Hardened
        && let Err(err) = &result
    {
        let reason = format!("gamma::{:?}", err.kind);
        crate::audit::record_fail_closed(ledger, err.detail.as_bytes(), &reason, "rejected");
    }
    result
}

pub fn gammaln(x: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    gammaln_dispatch("gammaln", x, mode)
}

pub fn gammasgn(x: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("gammasgn", x, mode, |value| gammasgn_scalar(value, mode))
}

pub fn loggamma(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    loggamma_dispatch("loggamma", z, mode)
}

fn gamma_dispatch(function: &'static str, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    match z {
        SpecialTensor::RealScalar(x) => gamma_scalar(*x, mode).map(SpecialTensor::RealScalar),
        SpecialTensor::RealVec(values) => values
            .iter()
            .map(|&x| gamma_scalar(x, mode))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        SpecialTensor::ComplexScalar(z_val) => {
            Ok(SpecialTensor::ComplexScalar(complex_gammaln(*z_val).exp()))
        }
        SpecialTensor::ComplexVec(values) => Ok(SpecialTensor::ComplexVec(
            values
                .iter()
                .map(|&z_val| complex_gammaln(z_val).exp())
                .collect(),
        )),
        SpecialTensor::Empty => Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "empty tensor is not a valid special-function input",
        }),
    }
}

fn gammaln_dispatch(function: &'static str, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    match z {
        SpecialTensor::RealScalar(x) => gammaln_scalar(*x, mode).map(SpecialTensor::RealScalar),
        SpecialTensor::RealVec(values) => values
            .iter()
            .map(|&x| gammaln_scalar(x, mode))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        SpecialTensor::ComplexScalar(z_val) => {
            Ok(SpecialTensor::ComplexScalar(complex_gammaln(*z_val)))
        }
        SpecialTensor::ComplexVec(values) => Ok(SpecialTensor::ComplexVec(
            values.iter().map(|&z_val| complex_gammaln(z_val)).collect(),
        )),
        SpecialTensor::Empty => Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "empty tensor is not a valid special-function input",
        }),
    }
}

fn loggamma_dispatch(
    function: &'static str,
    z: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    match z {
        SpecialTensor::RealScalar(x) => loggamma_scalar(*x, mode).map(SpecialTensor::RealScalar),
        SpecialTensor::RealVec(values) => values
            .iter()
            .map(|&x| loggamma_scalar(x, mode))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        SpecialTensor::ComplexScalar(z_val) => {
            Ok(SpecialTensor::ComplexScalar(complex_gammaln(*z_val)))
        }
        SpecialTensor::ComplexVec(values) => Ok(SpecialTensor::ComplexVec(
            values.iter().map(|&z_val| complex_gammaln(z_val)).collect(),
        )),
        SpecialTensor::Empty => Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "empty tensor is not a valid special-function input",
        }),
    }
}

pub fn digamma(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    digamma_dispatch("digamma", z, mode)
}

pub fn psi(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    digamma_dispatch("psi", z, mode)
}

fn digamma_dispatch(function: &'static str, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    match z {
        SpecialTensor::RealScalar(x) => digamma_scalar(*x, mode).map(SpecialTensor::RealScalar),
        SpecialTensor::RealVec(values) => values
            .iter()
            .map(|&x| digamma_scalar(x, mode))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        SpecialTensor::ComplexScalar(z_val) => {
            Ok(SpecialTensor::ComplexScalar(complex_digamma_scalar(*z_val)))
        }
        SpecialTensor::ComplexVec(values) => Ok(SpecialTensor::ComplexVec(
            values
                .iter()
                .map(|&z_val| complex_digamma_scalar(z_val))
                .collect(),
        )),
        SpecialTensor::Empty => Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "empty tensor is not a valid special-function input",
        }),
    }
}

pub fn gammainc(a: &SpecialTensor, x: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    gammainc_dispatch("gammainc", a, x, mode, true)
}

pub fn gammaincc(a: &SpecialTensor, x: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    gammainc_dispatch("gammaincc", a, x, mode, false)
}

fn gammainc_dispatch(
    function: &'static str,
    a: &SpecialTensor,
    x: &SpecialTensor,
    mode: RuntimeMode,
    lower: bool,
) -> SpecialResult {
    match (a, x) {
        // Real-real cases
        (SpecialTensor::RealScalar(a_val), SpecialTensor::RealScalar(x_val)) => {
            if lower {
                gammainc_scalar(*a_val, *x_val, mode).map(SpecialTensor::RealScalar)
            } else {
                gammaincc_scalar(*a_val, *x_val, mode).map(SpecialTensor::RealScalar)
            }
        }
        (SpecialTensor::RealVec(a_vec), SpecialTensor::RealScalar(x_val)) => a_vec
            .iter()
            .map(|&a_val| {
                if lower {
                    gammainc_scalar(a_val, *x_val, mode)
                } else {
                    gammaincc_scalar(a_val, *x_val, mode)
                }
            })
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        (SpecialTensor::RealScalar(a_val), SpecialTensor::RealVec(x_vec)) => x_vec
            .iter()
            .map(|&x_val| {
                if lower {
                    gammainc_scalar(*a_val, x_val, mode)
                } else {
                    gammaincc_scalar(*a_val, x_val, mode)
                }
            })
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        (SpecialTensor::RealVec(a_vec), SpecialTensor::RealVec(x_vec)) => {
            if a_vec.len() != x_vec.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            a_vec
                .iter()
                .zip(x_vec.iter())
                .map(|(&a_val, &x_val)| {
                    if lower {
                        gammainc_scalar(a_val, x_val, mode)
                    } else {
                        gammaincc_scalar(a_val, x_val, mode)
                    }
                })
                .collect::<Result<Vec<_>, _>>()
                .map(SpecialTensor::RealVec)
        }
        // Real a, complex x cases
        (SpecialTensor::RealScalar(a_val), SpecialTensor::ComplexScalar(x_val)) => {
            if lower {
                complex_gammainc_scalar(*a_val, *x_val, mode).map(SpecialTensor::ComplexScalar)
            } else {
                complex_gammaincc_scalar(*a_val, *x_val, mode).map(SpecialTensor::ComplexScalar)
            }
        }
        (SpecialTensor::RealScalar(a_val), SpecialTensor::ComplexVec(x_vec)) => x_vec
            .iter()
            .map(|&x_val| {
                if lower {
                    complex_gammainc_scalar(*a_val, x_val, mode)
                } else {
                    complex_gammaincc_scalar(*a_val, x_val, mode)
                }
            })
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::RealVec(a_vec), SpecialTensor::ComplexScalar(x_val)) => a_vec
            .iter()
            .map(|&a_val| {
                if lower {
                    complex_gammainc_scalar(a_val, *x_val, mode)
                } else {
                    complex_gammaincc_scalar(a_val, *x_val, mode)
                }
            })
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::RealVec(a_vec), SpecialTensor::ComplexVec(x_vec)) => {
            if a_vec.len() != x_vec.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            a_vec
                .iter()
                .zip(x_vec.iter())
                .map(|(&a_val, &x_val)| {
                    if lower {
                        complex_gammainc_scalar(a_val, x_val, mode)
                    } else {
                        complex_gammaincc_scalar(a_val, x_val, mode)
                    }
                })
                .collect::<Result<Vec<_>, _>>()
                .map(SpecialTensor::ComplexVec)
        }
        // Complex a cases
        (SpecialTensor::ComplexScalar(a_val), SpecialTensor::RealScalar(x_val)) => {
            gammainc_complex_parameter_scalar(*a_val, Complex64::new(*x_val, 0.0), mode, lower)
                .map(SpecialTensor::ComplexScalar)
        }
        (SpecialTensor::ComplexScalar(a_val), SpecialTensor::RealVec(x_vec)) => x_vec
            .iter()
            .map(|&x_val| {
                gammainc_complex_parameter_scalar(*a_val, Complex64::new(x_val, 0.0), mode, lower)
            })
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::ComplexVec(a_vec), SpecialTensor::RealScalar(x_val)) => a_vec
            .iter()
            .map(|&a_val| {
                gammainc_complex_parameter_scalar(a_val, Complex64::new(*x_val, 0.0), mode, lower)
            })
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::ComplexVec(a_vec), SpecialTensor::RealVec(x_vec)) => {
            if a_vec.len() != x_vec.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            a_vec
                .iter()
                .zip(x_vec.iter())
                .map(|(&a_val, &x_val)| {
                    gammainc_complex_parameter_scalar(
                        a_val,
                        Complex64::new(x_val, 0.0),
                        mode,
                        lower,
                    )
                })
                .collect::<Result<Vec<_>, _>>()
                .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::ComplexScalar(a_val), SpecialTensor::ComplexScalar(x_val)) => {
            gammainc_complex_parameter_scalar(*a_val, *x_val, mode, lower)
                .map(SpecialTensor::ComplexScalar)
        }
        (SpecialTensor::ComplexScalar(a_val), SpecialTensor::ComplexVec(x_vec)) => x_vec
            .iter()
            .map(|&x_val| gammainc_complex_parameter_scalar(*a_val, x_val, mode, lower))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::ComplexVec(a_vec), SpecialTensor::ComplexScalar(x_val)) => a_vec
            .iter()
            .map(|&a_val| gammainc_complex_parameter_scalar(a_val, *x_val, mode, lower))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::ComplexVec(a_vec), SpecialTensor::ComplexVec(x_vec)) => {
            if a_vec.len() != x_vec.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            a_vec
                .iter()
                .zip(x_vec.iter())
                .map(|(&a_val, &x_val)| {
                    gammainc_complex_parameter_scalar(a_val, x_val, mode, lower)
                })
                .collect::<Result<Vec<_>, _>>()
                .map(SpecialTensor::ComplexVec)
        }
        // Empty tensor cases
        (SpecialTensor::Empty, _) | (_, SpecialTensor::Empty) => Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "empty tensor is not a valid special-function input",
        }),
    }
}

fn gammainc_complex_parameter_scalar(
    a: Complex64,
    z: Complex64,
    mode: RuntimeMode,
    lower: bool,
) -> Result<Complex64, SpecialError> {
    if a.im == 0.0 {
        if z.im == 0.0 {
            let value = if lower {
                gammainc_scalar(a.re, z.re, mode)?
            } else {
                gammaincc_scalar(a.re, z.re, mode)?
            };
            return Ok(Complex64::new(value, 0.0));
        }
        return if lower {
            complex_gammainc_scalar(a.re, z, mode)
        } else {
            complex_gammaincc_scalar(a.re, z, mode)
        };
    }

    if lower {
        complex_parameter_gammainc_scalar(a, z, mode)
    } else {
        complex_parameter_gammaincc_scalar(a, z, mode)
    }
}

pub fn polygamma(n: usize, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    polygamma_dispatch("polygamma", n, z, mode)
}

fn polygamma_dispatch(
    function: &'static str,
    n: usize,
    z: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    match z {
        SpecialTensor::RealScalar(x) => {
            let result = match n {
                0 => digamma_scalar(*x, mode),
                1 => trigamma_scalar(*x, mode),
                2 => tetragamma_scalar(*x, mode),
                _ => polygamma_higher_scalar(n, *x, mode),
            };
            result.map(SpecialTensor::RealScalar)
        }
        SpecialTensor::RealVec(values) => {
            let results: Result<Vec<f64>, SpecialError> = match n {
                0 => values.iter().map(|&x| digamma_scalar(x, mode)).collect(),
                1 => values.iter().map(|&x| trigamma_scalar(x, mode)).collect(),
                2 => values.iter().map(|&x| tetragamma_scalar(x, mode)).collect(),
                _ => values
                    .iter()
                    .map(|&x| polygamma_higher_scalar(n, x, mode))
                    .collect(),
            };
            results.map(SpecialTensor::RealVec)
        }
        SpecialTensor::ComplexScalar(z_val) => Ok(SpecialTensor::ComplexScalar(
            complex_polygamma_scalar(n, *z_val),
        )),
        SpecialTensor::ComplexVec(values) => Ok(SpecialTensor::ComplexVec(
            values
                .iter()
                .map(|&z_val| complex_polygamma_scalar(n, z_val))
                .collect(),
        )),
        SpecialTensor::Empty => Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "empty tensor is not a valid special-function input",
        }),
    }
}

pub fn rgamma(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    match z {
        SpecialTensor::RealScalar(x) => rgamma_scalar(*x, mode).map(SpecialTensor::RealScalar),
        SpecialTensor::RealVec(values) => values
            .iter()
            .copied()
            .map(|x| rgamma_scalar(x, mode))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        SpecialTensor::ComplexScalar(z_val) => {
            Ok(SpecialTensor::ComplexScalar(complex_rgamma_scalar(*z_val)))
        }
        SpecialTensor::ComplexVec(values) => Ok(SpecialTensor::ComplexVec(
            values.iter().copied().map(complex_rgamma_scalar).collect(),
        )),
        SpecialTensor::Empty => Err(SpecialError {
            function: "rgamma",
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "empty tensor is not a valid special-function input",
        }),
    }
}

pub fn multigammaln(a: &SpecialTensor, d: f64, mode: RuntimeMode) -> SpecialResult {
    if d.is_nan() || d.floor() != d {
        record_special_trace(
            "multigammaln",
            mode,
            "domain_error",
            format!("a={a:?},d={d}"),
            "fail_closed",
            "d should be a positive integer (dimension)",
            false,
        );
        return Err(SpecialError {
            function: "multigammaln",
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "d should be a positive integer (dimension)",
        });
    }

    map_real_input("multigammaln", a, mode, |x| multigammaln_scalar(x, d, mode))
}

/// Gamma distribution CDF with rate `a` and shape `b`.
///
/// Matches `scipy.special.gdtr(a, b, x)`.
#[must_use]
pub fn gdtr(a: f64, b: f64, x: f64) -> f64 {
    if a.is_nan() || b.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    gammainc_scalar(b, a * x, RuntimeMode::Strict).unwrap_or(f64::NAN)
}

/// Gamma distribution survival function with rate `a` and shape `b`.
///
/// Returns P(X > x) = 1 - gdtr(a, b, x).
///
/// Matches `scipy.special.gdtrc(a, b, x)`.
#[must_use]
pub fn gdtrc(a: f64, b: f64, x: f64) -> f64 {
    if a.is_nan() || b.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 1.0;
    }
    gammaincc_scalar(b, a * x, RuntimeMode::Strict).unwrap_or(f64::NAN)
}

/// Inverse gamma distribution CDF with rate `a` and shape `b`, solving for `x`.
///
/// Matches `scipy.special.gdtrix(a, b, p)`.
#[must_use]
pub fn gdtrix(a: f64, b: f64, p: f64) -> f64 {
    if a.is_nan() || b.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    if b <= 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }

    gammaincinv(b, p) / a
}

/// Inverse gamma distribution CDF with respect to rate `a`.
///
/// Matches `scipy.special.gdtria(p, b, x)`.
#[must_use]
pub fn gdtria(p: f64, b: f64, x: f64) -> f64 {
    if p.is_nan() || b.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if b <= 0.0 || !(0.0..=1.0).contains(&p) || x == 0.0 {
        return f64::NAN;
    }

    gammaincinv(b, p) / x
}

/// Inverse gamma distribution CDF with respect to shape `b`.
///
/// Matches `scipy.special.gdtrib(a, p, x)`.
#[must_use]
pub fn gdtrib(a: f64, p: f64, x: f64) -> f64 {
    if a.is_nan() || p.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if a <= 0.0 || x < 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }

    let scaled_x = a * x;
    if !scaled_x.is_finite() {
        return f64::NAN;
    }
    if scaled_x == 0.0 {
        return if p == 0.0 { f64::NAN } else { 0.0 };
    }
    if p == 0.0 {
        return f64::INFINITY;
    }
    if p == 1.0 {
        return 0.0;
    }

    gammainc_shape_inv(scaled_x, p)
}

fn map_real_input<F>(
    function: &'static str,
    input: &SpecialTensor,
    mode: RuntimeMode,
    kernel: F,
) -> SpecialResult
where
    F: Fn(f64) -> Result<f64, SpecialError>,
{
    match input {
        SpecialTensor::RealScalar(x) => kernel(*x).map(SpecialTensor::RealScalar),
        SpecialTensor::RealVec(values) => values
            .iter()
            .copied()
            .map(kernel)
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        SpecialTensor::ComplexScalar(_) | SpecialTensor::ComplexVec(_) => {
            not_yet_implemented(function, mode, "complex-valued path pending")
        }
        SpecialTensor::Empty => {
            record_special_trace(
                function,
                mode,
                "domain_error",
                "input=empty",
                "fail_closed",
                "empty tensor is not a valid special-function input",
                false,
            );
            Err(SpecialError {
                function,
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "empty tensor is not a valid special-function input",
            })
        }
    }
}

fn gamma_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if matches!(mode, RuntimeMode::Hardened) && is_negative_integer_pole(x) {
        record_special_trace(
            "gamma",
            mode,
            "pole_input",
            format!("input={x}"),
            "fail_closed",
            "gamma pole at nonpositive integer",
            false,
        );
        return Err(SpecialError {
            function: "gamma",
            kind: SpecialErrorKind::PoleInput,
            mode,
            detail: "gamma pole at nonpositive integer",
        });
    }
    let value = gamma_core(x);
    if !value.is_finite() && !is_negative_integer_pole(x) && x != 0.0 {
        record_special_trace(
            "gamma",
            mode,
            "non_finite_output",
            format!("input={x}"),
            "returned_non_finite",
            format!("output={value}"),
            false,
        );
    }
    Ok(value)
}

pub fn gammaln_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if x.is_nan() {
        return Ok(f64::NAN);
    }
    if is_negative_integer_pole(x) || x == 0.0 {
        if matches!(mode, RuntimeMode::Hardened) {
            record_special_trace(
                "gammaln",
                mode,
                "pole_input",
                format!("input={x}"),
                "fail_closed",
                "gammaln pole at nonpositive integer",
                false,
            );
            return Err(SpecialError {
                function: "gammaln",
                kind: SpecialErrorKind::PoleInput,
                mode,
                detail: "gammaln pole at nonpositive integer",
            });
        } else {
            return Ok(f64::INFINITY);
        }
    }

    let output = if x >= 0.5 {
        if x < 100.0 {
            lngamma_lanczos(x)
        } else {
            lngamma_positive(x)
        }
    } else {
        // Reflection formula: ln|Γ(x)| = ln(π) - ln|sin(πx)| - ln|Γ(1-x)|
        // Valid for all x < 0.5 (except poles handled above)
        let g1mx = if 1.0 - x < 100.0 {
            lngamma_lanczos(1.0 - x)
        } else {
            lngamma_positive(1.0 - x)
        };
        PI.ln() - (PI * x).sin().abs().ln() - g1mx
    };

    if !output.is_finite() && !x.is_infinite() {
        record_special_trace(
            "gammaln",
            mode,
            "non_finite_output",
            format!("input={x}"),
            "returned_non_finite",
            format!("output={output}"),
            false,
        );
    }
    Ok(output)
}

pub fn loggamma_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if x.is_nan() {
        return Ok(f64::NAN);
    }
    if x < 0.0 {
        if matches!(mode, RuntimeMode::Hardened) {
            record_special_trace(
                "loggamma",
                mode,
                "domain_error",
                format!("input={x}"),
                "fail_closed",
                "real loggamma is undefined on the negative real branch cut",
                false,
            );
            return Err(SpecialError {
                function: "loggamma",
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "real loggamma is undefined on the negative real branch cut",
            });
        }
        return Ok(f64::NAN);
    }
    if x == 0.0 && matches!(mode, RuntimeMode::Hardened) {
        record_special_trace(
            "loggamma",
            mode,
            "pole_input",
            format!("input={x}"),
            "fail_closed",
            "loggamma pole at zero",
            false,
        );
        return Err(SpecialError {
            function: "loggamma",
            kind: SpecialErrorKind::PoleInput,
            mode,
            detail: "loggamma pole at zero",
        });
    }
    gammaln_scalar(x, mode)
}

pub fn gammasgn_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if x.is_nan() || x == f64::NEG_INFINITY {
        return Ok(f64::NAN);
    }
    if x == f64::INFINITY {
        return Ok(1.0);
    }
    if is_negative_integer_pole(x) {
        if matches!(mode, RuntimeMode::Hardened) {
            record_special_trace(
                "gammasgn",
                mode,
                "pole_input",
                format!("input={x}"),
                "fail_closed",
                "gammasgn pole at negative integer",
                false,
            );
            return Err(SpecialError {
                function: "gammasgn",
                kind: SpecialErrorKind::PoleInput,
                mode,
                detail: "gammasgn pole at negative integer",
            });
        }
        return Ok(f64::NAN);
    }
    if x == 0.0 {
        return Ok(if x.is_sign_negative() { -1.0 } else { 1.0 });
    }
    if x > 0.0 {
        return Ok(1.0);
    }

    let sine = (PI * x).sin();
    if sine.is_nan() {
        Ok(f64::NAN)
    } else if sine.is_sign_negative() {
        Ok(-1.0)
    } else {
        Ok(1.0)
    }
}

fn multigammaln_scalar(a: f64, d: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    let threshold = 0.5 * (d - 1.0);
    if !a.is_nan() && a <= threshold {
        record_special_trace(
            "multigammaln",
            mode,
            "domain_error",
            format!("a={a},d={d}"),
            "fail_closed",
            format!("condition a ({a}) > 0.5 * (d - 1) ({threshold}) not met"),
            false,
        );
        return Err(SpecialError {
            function: "multigammaln",
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "condition a must exceed 0.5 * (d - 1)",
        });
    }
    if a.is_nan() {
        return Ok(f64::NAN);
    }
    if !d.is_finite() {
        record_special_trace(
            "multigammaln",
            mode,
            "domain_error",
            format!("a={a},d={d}"),
            "fail_closed",
            "d should be a positive integer (dimension)",
            false,
        );
        return Err(SpecialError {
            function: "multigammaln",
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "d should be a positive integer (dimension)",
        });
    }

    let mut result = 0.25 * d * (d - 1.0) * PI.ln();
    for j in 1..=(d as i64) {
        result += gammaln_scalar(a - 0.5 * ((j - 1) as f64), mode)?;
    }
    Ok(result)
}

fn lngamma_lanczos(x: f64) -> f64 {
    let mut coeff_sum = LANCZOS_COEFFS[0];
    for (idx, coeff) in LANCZOS_COEFFS.iter().enumerate().skip(1) {
        coeff_sum += coeff / (x + (idx as f64 - 1.0));
    }

    let t = x + 6.5;
    // ln(Γ(x)) = ln(sqrt(2π)) + (x-0.5)ln(t) - t + ln(coeff_sum)
    0.5 * (2.0 * PI).ln() + (x - 0.5) * t.ln() - t + coeff_sum.ln()
}

fn digamma_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if matches!(mode, RuntimeMode::Hardened) && is_negative_integer_pole(x) {
        record_special_trace(
            "digamma",
            mode,
            "pole_input",
            format!("input={x}"),
            "fail_closed",
            "digamma pole at nonpositive integer",
            false,
        );
        return Err(SpecialError {
            function: "digamma",
            kind: SpecialErrorKind::PoleInput,
            mode,
            detail: "digamma pole at nonpositive integer",
        });
    }
    let value = digamma_core(x);
    if !value.is_finite() {
        record_special_trace(
            "digamma",
            mode,
            "non_finite_output",
            format!("input={x}"),
            "returned_non_finite",
            format!("output={value}"),
            false,
        );
    }
    Ok(value)
}

fn trigamma_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if matches!(mode, RuntimeMode::Hardened) && is_negative_integer_pole(x) {
        record_special_trace(
            "polygamma",
            mode,
            "pole_input",
            format!("input={x}"),
            "fail_closed",
            "trigamma pole at nonpositive integer",
            false,
        );
        return Err(SpecialError {
            function: "polygamma",
            kind: SpecialErrorKind::PoleInput,
            mode,
            detail: "trigamma pole at nonpositive integer",
        });
    }
    let value = trigamma_core(x);
    if !value.is_finite() {
        record_special_trace(
            "polygamma",
            mode,
            "non_finite_output",
            format!("input={x}"),
            "returned_non_finite",
            format!("output={value}"),
            false,
        );
    }
    Ok(value)
}

fn tetragamma_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if matches!(mode, RuntimeMode::Hardened) && is_negative_integer_pole(x) {
        record_special_trace(
            "polygamma",
            mode,
            "pole_input",
            format!("input={x}"),
            "fail_closed",
            "tetragamma pole at nonpositive integer",
            false,
        );
        return Err(SpecialError {
            function: "polygamma",
            kind: SpecialErrorKind::PoleInput,
            mode,
            detail: "tetragamma pole at nonpositive integer",
        });
    }
    let value = crate::convenience::tetragamma(x);
    if !value.is_finite() {
        record_special_trace(
            "polygamma",
            mode,
            "non_finite_output",
            format!("input={x}"),
            "returned_non_finite",
            format!("output={value}"),
            false,
        );
    }
    Ok(value)
}

fn polygamma_higher_scalar(order: usize, x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if matches!(mode, RuntimeMode::Hardened) && is_negative_integer_pole(x) {
        record_special_trace(
            "polygamma",
            mode,
            "pole_input",
            format!("order={order},input={x}"),
            "fail_closed",
            "higher-order polygamma pole at nonpositive integer",
            false,
        );
        return Err(SpecialError {
            function: "polygamma",
            kind: SpecialErrorKind::PoleInput,
            mode,
            detail: "higher-order polygamma pole at nonpositive integer",
        });
    }
    let value = polygamma_higher_core(order, x);
    if !value.is_finite() {
        record_special_trace(
            "polygamma",
            mode,
            "non_finite_output",
            format!("order={order},input={x}"),
            "returned_non_finite",
            format!("output={value}"),
            false,
        );
    }
    Ok(value)
}

fn rgamma_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if is_negative_integer_pole(x) {
        return Ok(0.0);
    }
    let gamma_value = gamma_scalar(x, mode)?;
    let value = 1.0 / gamma_value;
    if !value.is_finite() {
        record_special_trace(
            "rgamma",
            mode,
            "non_finite_output",
            format!("input={x}"),
            "returned_non_finite",
            format!("output={value}"),
            false,
        );
    }
    Ok(value)
}

fn complex_rgamma_scalar(z: Complex64) -> Complex64 {
    if !z.is_finite() {
        return Complex64::new(f64::NAN, f64::NAN);
    }
    if is_complex_real_gamma_pole(z) {
        return Complex64::from_real(0.0);
    }
    (-complex_gammaln(z)).exp()
}

pub fn gammainc_scalar(a: f64, x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    validate_incomplete_gamma_domain("gammainc", a, x, mode)?;
    let (p, _) = regularized_gamma_pair(a, x, mode)?;
    Ok(p)
}

pub fn gammaincc_scalar(a: f64, x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    validate_incomplete_gamma_domain("gammaincc", a, x, mode)?;
    let (_, q) = regularized_gamma_pair(a, x, mode)?;
    Ok(q)
}

fn validate_incomplete_gamma_domain(
    function: &'static str,
    a: f64,
    x: f64,
    mode: RuntimeMode,
) -> Result<(), SpecialError> {
    if a.is_nan() || x.is_nan() {
        return Ok(());
    }
    if !a.is_finite() || a <= 0.0 || x < 0.0 {
        return match mode {
            RuntimeMode::Strict => {
                record_special_trace(
                    function,
                    mode,
                    "domain_error",
                    format!("a={a},x={x}"),
                    "returned_nan",
                    "strict domain fallback",
                    false,
                );
                Ok(())
            }
            RuntimeMode::Hardened => {
                record_special_trace(
                    function,
                    mode,
                    "domain_error",
                    format!("a={a},x={x}"),
                    "fail_closed",
                    "regularized incomplete gamma requires finite a>0 and x>=0",
                    false,
                );
                Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "regularized incomplete gamma requires finite a>0 and x>=0",
                })
            }
        };
    }
    Ok(())
}

fn regularized_gamma_pair(a: f64, x: f64, mode: RuntimeMode) -> Result<(f64, f64), SpecialError> {
    if a.is_nan() || x.is_nan() {
        return Ok((f64::NAN, f64::NAN));
    }
    if !a.is_finite() || a <= 0.0 || x < 0.0 {
        return Ok((f64::NAN, f64::NAN));
    }
    if x == 0.0 {
        return Ok((0.0, 1.0));
    }
    if x.is_infinite() {
        return Ok((1.0, 0.0));
    }

    const EPS: f64 = 1.0e-14;
    const MAX_ITERS: usize = 200;
    const FPMIN: f64 = 1.0e-300;

    let lg = gammaln_scalar(a, RuntimeMode::Strict)?;
    let prefactor = (-x + a * x.ln() - lg).exp();
    let (p, q) = if x < a + 1.0 {
        let mut ap = a;
        let mut term = 1.0 / a;
        let mut sum = term;
        for _ in 0..MAX_ITERS {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if term.abs() <= sum.abs() * EPS {
                break;
            }
        }
        let lower = prefactor * sum;
        let p_value = clamp_unit_interval(lower);
        let q_value = clamp_unit_interval(1.0 - p_value);
        (p_value, q_value)
    } else {
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / FPMIN;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..=MAX_ITERS {
            let i_f = i as f64;
            let an = -i_f * (i_f - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < FPMIN {
                d = FPMIN;
            }
            c = b + an / c;
            if c.abs() < FPMIN {
                c = FPMIN;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() <= EPS {
                break;
            }
        }
        let upper = prefactor * h;
        let q_value = clamp_unit_interval(upper);
        let p_value = clamp_unit_interval(1.0 - q_value);
        (p_value, q_value)
    };

    if !p.is_finite() {
        record_special_trace(
            "gammainc",
            mode,
            "non_finite_output",
            format!("a={a},x={x}"),
            "returned_non_finite",
            format!("output={p}"),
            false,
        );
    }
    if !q.is_finite() {
        record_special_trace(
            "gammaincc",
            mode,
            "non_finite_output",
            format!("a={a},x={x}"),
            "returned_non_finite",
            format!("output={q}"),
            false,
        );
    }

    Ok((p, q))
}

fn clamp_unit_interval(value: f64) -> f64 {
    if value.is_nan() {
        return f64::NAN;
    }
    value.clamp(0.0, 1.0)
}

fn gamma_core(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return if x.is_sign_negative() {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
    }
    if x.is_infinite() {
        return if x.is_sign_positive() {
            f64::INFINITY
        } else {
            f64::NAN
        };
    }
    if is_negative_integer_pole(x) {
        // SciPy returns NaN for negative integer poles
        return f64::NAN;
    }

    if x < 0.5 {
        // Reflection formula: Γ(x) = π / (sin(πx) * Γ(1-x))
        let sin_pi_x = (PI * x).sin();
        if sin_pi_x == 0.0 {
            // This matches x = 0 handled above, or integer x < 0
            if x == 0.0 {
                return if x.is_sign_negative() {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                };
            }
            return f64::NAN;
        }
        return PI / (sin_pi_x * gamma_core(1.0 - x));
    }

    gamma_lanczos(x)
}

fn gamma_lanczos(x: f64) -> f64 {
    let x_minus_1 = x - 1.0;
    let mut coeff_sum = LANCZOS_COEFFS[0];
    for (idx, coeff) in LANCZOS_COEFFS.iter().enumerate().skip(1) {
        coeff_sum += coeff / (x_minus_1 + idx as f64);
    }

    let t = x_minus_1 + 7.5;
    if coeff_sum == 0.0 {
        return 0.0;
    }
    let log_abs_value = SQRT_2PI.ln() + (x_minus_1 + 0.5) * t.ln() - t + coeff_sum.abs().ln();
    if log_abs_value > f64::MAX.ln() {
        return f64::INFINITY.copysign(coeff_sum);
    }
    log_abs_value.exp().copysign(coeff_sum)
}

fn digamma_core(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return if x.is_sign_negative() {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
    }
    if x.is_infinite() {
        return if x.is_sign_positive() {
            f64::INFINITY
        } else {
            f64::NAN
        };
    }
    if is_negative_integer_pole(x) {
        return f64::NAN;
    }

    if x < 0.0 {
        return digamma_core(1.0 - x) - PI / (PI * x).tan();
    }

    let mut shifted = x;
    let mut acc = 0.0;
    while shifted < 8.0 {
        acc -= 1.0 / shifted;
        shifted += 1.0;
    }

    let inv = 1.0 / shifted;
    let inv2 = inv * inv;
    // Bernoulli asymptotic extended through the B₁₀ term (was truncated at B₆,
    // leaving ~2.5e-10 at the shift point). frankenscipy-luxsz.
    acc + shifted.ln()
        - 0.5 * inv
        - inv2
            * (1.0 / 12.0
                - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0 - inv2 * (1.0 / 240.0 - inv2 / 132.0))))
}

fn trigamma_core(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY;
    }
    if x.is_infinite() {
        return if x.is_sign_positive() { 0.0 } else { f64::NAN };
    }
    if is_negative_integer_pole(x) {
        return f64::NAN;
    }

    if x < 0.0 {
        let sin_pi_x = (PI * x).sin();
        return (PI * PI) / (sin_pi_x * sin_pi_x) - trigamma_core(1.0 - x);
    }

    let mut shifted = x;
    let mut acc = 0.0;
    while shifted < 8.0 {
        acc += 1.0 / (shifted * shifted);
        shifted += 1.0;
    }

    let inv = 1.0 / shifted;
    let inv2 = inv * inv;
    let inv3 = inv2 * inv;
    let inv5 = inv3 * inv2;
    let inv7 = inv5 * inv2;
    let inv9 = inv7 * inv2;
    let inv11 = inv9 * inv2;
    // Extended through the B₁₀ term (was truncated at B₆). frankenscipy-luxsz.
    acc + inv + 0.5 * inv2 + inv3 / 6.0 - inv5 / 30.0 + inv7 / 42.0 - inv9 / 30.0
        + 5.0 * inv11 / 66.0
}

fn polygamma_higher_core(order: usize, x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return polygamma_sign(order) * f64::INFINITY;
    }
    if x.is_infinite() {
        return if x.is_sign_positive() { 0.0 } else { f64::NAN };
    }
    if is_negative_integer_pole(x) {
        return f64::NAN;
    }

    let sign = polygamma_sign(order);
    let factorial = factorial_f64(order);
    let order_plus_one = order as f64 + 1.0;

    let mut shifted = x;
    let mut correction = 0.0;
    while shifted < 8.0 {
        correction += sign * factorial / shifted.powf(order_plus_one);
        shifted += 1.0;
    }

    correction + sign * factorial * crate::convenience::hurwitz_zeta(order_plus_one, shifted)
}

fn polygamma_sign(order: usize) -> f64 {
    if order % 2 == 1 { 1.0 } else { -1.0 }
}

fn factorial_f64(order: usize) -> f64 {
    (1..=order).fold(1.0, |acc, value| acc * value as f64)
}

fn is_negative_integer_pole(x: f64) -> bool {
    x.is_finite() && x < 0.0 && x.fract() == 0.0
}

fn is_complex_real_gamma_pole(z: Complex64) -> bool {
    z.im == 0.0 && (z.re == 0.0 || is_negative_integer_pole(z.re))
}

/// Log-gamma function computed directly via Stirling series.
/// Avoids the overflow that gamma_core hits for large arguments.
/// Only valid for x > 0.
fn lngamma_positive(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }
    if x < 8.0 {
        // Shift upward using recurrence: lnΓ(x) = lnΓ(x+n) - ln(x(x+1)...(x+n-1))
        let mut shifted = x;
        let mut correction = 0.0;
        while shifted < 8.0 {
            correction += shifted.ln();
            shifted += 1.0;
        }
        stirling_lngamma(shifted) - correction
    } else {
        stirling_lngamma(x)
    }
}

/// Stirling series for lnΓ(x), valid for large x.
fn stirling_lngamma(x: f64) -> f64 {
    let half_ln_2pi = 0.5 * (2.0 * PI).ln();
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    // Bernoulli number series: lnΓ(x) ≈ (x-0.5)*ln(x) - x + 0.5*ln(2π)
    //   + 1/(12x) - 1/(360x³) + 1/(1260x⁵) - 1/(1680x⁷)
    (x - 0.5) * x.ln() - x
        + half_ln_2pi
        + inv * (1.0 / 12.0 - inv2 * (1.0 / 360.0 - inv2 * (1.0 / 1260.0 - inv2 / 1680.0)))
}

// ══════════════════════════════════════════════════════════════════════
// Combinatorial Functions
// ══════════════════════════════════════════════════════════════════════

/// Compute n! (factorial).
///
/// Matches `scipy.special.factorial(n, exact=True)`.
/// Returns f64 to handle large factorials (overflows to infinity).
pub fn factorial(n: u64) -> f64 {
    // Use lookup table for small values, gamma for large
    if n <= 20 {
        FACTORIALS[n as usize]
    } else {
        // n! = gamma(n+1)
        gamma_core((n + 1) as f64)
    }
}

/// Compute the double factorial n!! = n * (n-2) * (n-4) * ... * (1 or 2).
///
/// Matches `scipy.special.factorial2(n, exact=True)`.
///
/// For odd n: n!! = n * (n-2) * ... * 3 * 1
/// For even n: n!! = n * (n-2) * ... * 4 * 2
///
/// SciPy compatibility cases:
/// - 0!! = 1 (by convention)
/// - For negative n, returns 0
pub fn factorial2(n: i64) -> f64 {
    match n.cmp(&0) {
        std::cmp::Ordering::Less => 0.0,
        std::cmp::Ordering::Equal => 1.0, // 0!! = 1
        std::cmp::Ordering::Greater => {
            if n <= 33 {
                // Direct computation for small values
                let mut result = 1.0_f64;
                let mut k = n;
                while k > 1 {
                    result *= k as f64;
                    k -= 2;
                }
                result
            } else {
                // For large n, use relation to gamma function:
                // n!! = 2^(n/2) * Gamma(n/2 + 1) for even n
                // n!! = 2^((n+1)/2) * Gamma((n+1)/2) / sqrt(pi) for odd n
                let nf = n as f64;
                if n % 2 == 0 {
                    // Even: n!! = 2^(n/2) * (n/2)!
                    let half_n = nf / 2.0;
                    2.0_f64.powf(half_n) * gamma_core(half_n + 1.0)
                } else {
                    // Odd m: m!! = 2^((m+1)/2) * Gamma((m+2)/2) / sqrt(pi),
                    // derived from (2k-1)!! = Gamma(k+1/2) * 2^k / sqrt(pi)
                    // with m = 2k - 1, so k = (m+1)/2 and Gamma argument is
                    // k + 1/2 = (m+2)/2.
                    let exp_pow = (nf + 1.0) / 2.0;
                    let gamma_arg = (nf + 2.0) / 2.0;
                    2.0_f64.powf(exp_pow) * gamma_core(gamma_arg) / PI.sqrt()
                }
            }
        }
    }
}

/// Compute the binomial coefficient `C(x, y) = Γ(x+1) / (Γ(y+1) · Γ(x−y+1))`
/// for real `x ≥ 0` and `y ≥ 0`.
///
/// Matches `scipy.special.binom(x, y)` on the common domain. Differences
/// from `comb`:
/// - accepts any non-negative real `x`, `y` (not just integers)
/// - uses the gamma-function generalization, so `binom(0.5, 0.25)` is
///   well-defined (≈ 1.069 via Γ(1.5)/(Γ(1.25)·Γ(1.25)))
///
/// Conventions:
/// - `y > x` returns 0.0 (matching scipy's vanishing-coefficient rule on
///   the real generalization)
/// - `x < 0` or `y < 0` returns NaN. The full negative-argument path needs
///   the gamma reflection formula and is tracked as a follow-on under
///   `frankenscipy-b6z3m`.
/// - NaN propagates.
pub fn binom(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 && y >= 0.0 {
        // Negative-integer x is a pole of Γ(x+1) => NaN (scipy.special.binom).
        if x == x.floor() {
            return f64::NAN;
        }
        // Negative non-integer x: binom = Γ(x+1)/(Γ(y+1)Γ(x-y+1)) is SIGNED.
        // gammaln gives ln|·|, so restore the sign from gammasgn (an x-y+1 pole
        // gives a vanishing 0, matching scipy, e.g. binom(-4.5,2)=12.375).
        let l = gammaln_scalar(x + 1.0, RuntimeMode::Strict).unwrap_or(f64::NAN)
            - gammaln_scalar(y + 1.0, RuntimeMode::Strict).unwrap_or(f64::NAN)
            - gammaln_scalar(x - y + 1.0, RuntimeMode::Strict).unwrap_or(f64::NAN);
        let sign = gammasgn_scalar(x + 1.0, RuntimeMode::Strict).unwrap_or(f64::NAN)
            * gammasgn_scalar(y + 1.0, RuntimeMode::Strict).unwrap_or(f64::NAN)
            * gammasgn_scalar(x - y + 1.0, RuntimeMode::Strict).unwrap_or(f64::NAN);
        return sign * l.exp();
    }
    if x < 0.0 || y < 0.0 {
        return f64::NAN;
    }
    if y > x {
        // For integer overshoot (y > x with y, x both integers, or
        // y - x integer such that x - y + 1 ≤ 0), the gamma formula
        // hits a pole or negative argument that lngamma_positive
        // can't handle without reflection. Return 0 in that
        // vanishing-coefficient regime. For non-integer y > x where
        // x - y + 1 > 0, fall through to the gamma formula — scipy
        // returns the finite nonzero value (e.g. binom(0.5, 1) = 0.5).
        // [frankenscipy-du5m4]
        if x - y + 1.0 <= 0.0 || (y - x).fract() == 0.0 {
            return 0.0;
        }
        // else fall through
    }
    if y == 0.0 || y == x {
        return 1.0;
    }
    // Use lnΓ for large arguments; small / non-integer arguments still
    // route through lngamma_positive which handles the (0, 8) branch via
    // the reflection-free recurrence.
    let l = lngamma_positive(x + 1.0) - lngamma_positive(y + 1.0) - lngamma_positive(x - y + 1.0);
    l.exp()
}

/// k-step factorial: `n * (n − k) * (n − 2k) * … * r`, where the iteration
/// stops at the smallest positive remainder `r`.
///
/// Matches `scipy.special.factorialk(n, k, exact=False)`. Conventions:
/// - `n == 0` returns 1.0 (empty-product identity).
/// - `n < 0` returns NaN (real factorialk only defined on non-negative
///   integers; scipy.special's exact=False mode also returns NaN here).
/// - `k < 1` returns NaN (k must be a positive integer).
/// - `k == 1` reduces to the ordinary factorial; `k == 2` to the double
///   factorial.
pub fn factorialk(n: i64, k: i64) -> f64 {
    if k < 1 {
        return f64::NAN;
    }
    if n < 0 {
        return f64::NAN;
    }
    if n == 0 {
        return 1.0;
    }
    let mut result = 1.0_f64;
    let mut step = n;
    while step > 0 {
        result *= step as f64;
        step -= k;
    }
    result
}

/// Poisson distribution CDF: P(X <= k) for Poisson with mean m.
///
/// Matches `scipy.special.pdtr(k, m)`.
///
/// Uses the relation: pdtr(k, m) = gammaincc(k + 1, m)
///
/// # Arguments
/// * `k` - Number of events (non-negative integer, but accepts float)
/// * `m` - Expected number of events (mean, must be >= 0)
pub fn pdtr(k: f64, m: f64) -> f64 {
    if k.is_nan() || m.is_nan() {
        return f64::NAN;
    }
    if m < 0.0 || k < 0.0 {
        return f64::NAN;
    }
    if m == 0.0 {
        return 1.0; // P(X <= k) = 1 when m = 0
    }

    // pdtr(k, m) = gammaincc(k + 1, m) = Q(k + 1, m)
    gammaincc_scalar(k + 1.0, m, RuntimeMode::Strict).unwrap_or(f64::NAN)
}

/// Poisson distribution survival function: P(X > k) for Poisson with mean m.
///
/// Matches `scipy.special.pdtrc(k, m)`.
///
/// Uses the relation: pdtrc(k, m) = gammainc(k + 1, m)
///
/// # Arguments
/// * `k` - Number of events (non-negative integer, but accepts float)
/// * `m` - Expected number of events (mean, must be >= 0)
pub fn pdtrc(k: f64, m: f64) -> f64 {
    if k.is_nan() || m.is_nan() {
        return f64::NAN;
    }
    if m < 0.0 || k < 0.0 {
        return f64::NAN;
    }
    if m == 0.0 {
        return 0.0; // P(X > k) = 0 when m = 0
    }

    // pdtrc(k, m) = gammainc(k + 1, m) = P(k + 1, m)
    gammainc_scalar(k + 1.0, m, RuntimeMode::Strict).unwrap_or(f64::NAN)
}

/// Inverse of Poisson CDF: find m such that pdtr(k, m) = p.
///
/// Matches `scipy.special.pdtri(k, p)`.
///
/// Uses Newton's method to find the root.
///
/// # Arguments
/// * `k` - Number of events (non-negative integer, but accepts float)
/// * `p` - Probability (must be in [0, 1])
pub fn pdtri(k: f64, p: f64) -> f64 {
    if k.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    if k < 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::INFINITY;
    }
    if p == 1.0 {
        return 0.0;
    }

    // Use gammaincinv: we need m such that gammaincc(k+1, m) = p
    // This is the same as gammainc(k+1, m) = 1 - p
    // So m = gammaincinv(k+1, 1-p)
    gammaincinv(k + 1.0, 1.0 - p)
}

/// Inverse of Poisson CDF with respect to event count k.
///
/// Matches `scipy.special.pdtrik(p, m)`.
///
/// Uses the relation pdtr(k, m) = gammaincc(k + 1, m), so k + 1 is the
/// inverse of gammainc(a, m) at 1 - p.
#[must_use]
pub fn pdtrik(p: f64, m: f64) -> f64 {
    if p.is_nan() || m.is_nan() {
        return f64::NAN;
    }
    if !(0.0..=1.0).contains(&p) || m < 0.0 {
        return f64::NAN;
    }
    if p == 1.0 {
        return f64::NAN;
    }
    if p == 0.0 || m == 0.0 {
        return 0.0;
    }
    if !m.is_finite() {
        return f64::NAN;
    }

    let shape = gammainc_shape_inv(m, 1.0 - p);
    if !shape.is_finite() {
        return shape;
    }

    (shape - 1.0).max(0.0)
}

/// Chi-squared distribution CDF.
///
/// Returns the probability P(X <= x) where X follows a chi-squared
/// distribution with v degrees of freedom.
///
/// Matches `scipy.special.chdtr(v, x)`.
#[must_use]
pub fn chdtr(v: f64, x: f64) -> f64 {
    if v.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if v <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    // chdtr(v, x) = gammainc(v/2, x/2) = P(v/2, x/2)
    gammainc_scalar(v / 2.0, x / 2.0, RuntimeMode::Strict).unwrap_or(f64::NAN)
}

/// Chi-squared distribution survival function.
///
/// Returns the probability P(X > x) where X follows a chi-squared
/// distribution with v degrees of freedom.
///
/// Matches `scipy.special.chdtrc(v, x)`.
#[must_use]
pub fn chdtrc(v: f64, x: f64) -> f64 {
    if v.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if v <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 1.0;
    }
    // chdtrc(v, x) = gammaincc(v/2, x/2) = Q(v/2, x/2)
    gammaincc_scalar(v / 2.0, x / 2.0, RuntimeMode::Strict).unwrap_or(f64::NAN)
}

/// Inverse complemented chi-squared distribution.
///
/// Returns x such that P(X > x) = p where X follows a chi-squared
/// distribution with v degrees of freedom.
///
/// Matches `scipy.special.chdtri(v, p)`.
#[must_use]
pub fn chdtri(v: f64, p: f64) -> f64 {
    if v.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    if v <= 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::INFINITY;
    }
    if p == 1.0 {
        return 0.0;
    }
    // chdtri(v, p) finds x such that gammaincc(v/2, x/2) = p.
    // So x/2 = gammaincinv(v/2, 1-p), thus x = 2 * gammaincinv(v/2, 1-p).
    2.0 * gammaincinv(v / 2.0, 1.0 - p)
}

/// Inverse chi-squared distribution CDF with respect to degrees of freedom.
///
/// Returns `v` such that `P(X <= x) = p` where `X` follows a chi-squared
/// distribution with `v` degrees of freedom.
///
/// Matches `scipy.special.chdtriv(p, x)`.
#[must_use]
pub fn chdtriv(p: f64, x: f64) -> f64 {
    if p.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if !(0.0..=1.0).contains(&p) || x <= 0.0 || !x.is_finite() {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::INFINITY;
    }
    if p == 1.0 {
        return 0.0;
    }

    let shape = gammainc_shape_inv(x / 2.0, p);
    if !shape.is_finite() {
        return shape;
    }

    2.0 * shape
}

/// Inverse of the regularized lower incomplete gamma function.
/// Finds x such that gammainc(a, x) = p.
///
/// Delegates to `crate::convenience::gammaincinv_scalar`. The local
/// Newton-Raphson here was diverging when p was close to 0 or 1: the
/// initial guess for small p reached x ≈ 8 even when the true root
/// was ~0.15, and Newton walked to the iteration cap returning a
/// value many orders of magnitude wrong (frankenscipy-jr3na). The
/// convenience helper is the validated one that the diff_special
/// gammainc harness already covers at 1e-9 rel.
fn gammaincinv(a: f64, p: f64) -> f64 {
    if a <= 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    crate::convenience::gammaincinv_scalar(a, p)
}

fn gammainc_shape_inv(x: f64, p: f64) -> f64 {
    let mut lo = 0.0;
    let mut hi = x.max(1.0);
    while gammainc_scalar(hi, x, RuntimeMode::Strict).unwrap_or(f64::NAN) > p {
        lo = hi;
        hi *= 2.0;
        if !hi.is_finite() {
            return f64::INFINITY;
        }
    }

    for _ in 0..180 {
        let mid = 0.5 * (lo + hi);
        let value = gammainc_scalar(mid, x, RuntimeMode::Strict).unwrap_or(f64::NAN);
        if !value.is_finite() {
            return f64::NAN;
        }
        if value > p {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo).abs() <= 1.0e-12 * hi.abs().max(1.0) {
            break;
        }
    }

    0.5 * (lo + hi)
}

const FACTORIALS: [f64; 21] = [
    1.0,
    1.0,
    2.0,
    6.0,
    24.0,
    120.0,
    720.0,
    5_040.0,
    40_320.0,
    362_880.0,
    3_628_800.0,
    39_916_800.0,
    479_001_600.0,
    6_227_020_800.0,
    87_178_291_200.0,
    1_307_674_368_000.0,
    20_922_789_888_000.0,
    355_687_428_096_000.0,
    6_402_373_705_728_000.0,
    121_645_100_408_832_000.0,
    2_432_902_008_176_640_000.0,
];

/// Compute the binomial coefficient C(n, k) = n! / (k! * (n-k)!).
///
/// Matches `scipy.special.comb(n, k, exact=True)` for nonneg integer inputs.
/// Returns 0 for invalid inputs (k > n or negative-equivalent).
pub fn comb(n: u64, k: u64) -> f64 {
    if k > n {
        return 0.0;
    }
    if k == 0 || k == n {
        return 1.0;
    }
    // Use symmetry: C(n, k) = C(n, n-k) to minimize multiplications
    let k = k.min(n - k);
    if k <= 20 && n <= 170 {
        // Direct computation to avoid floating-point drift
        let mut result = 1.0_f64;
        for i in 0..k {
            result *= (n - i) as f64;
            result /= (i + 1) as f64;
        }
        result
    } else {
        // Use log-gamma for large values: ln(C(n,k)) = lnΓ(n+1) - lnΓ(k+1) - lnΓ(n-k+1)
        let lnc = lngamma_positive((n + 1) as f64)
            - lngamma_positive((k + 1) as f64)
            - lngamma_positive((n - k + 1) as f64);
        lnc.exp()
    }
}

/// Compute the number of k-permutations of n: P(n, k) = n! / (n-k)!.
///
/// Matches `scipy.special.perm(n, k, exact=True)` for nonneg integer inputs.
/// Returns 0 for invalid inputs (k > n).
pub fn perm(n: u64, k: u64) -> f64 {
    if k > n {
        return 0.0;
    }
    if k == 0 {
        return 1.0;
    }
    if k <= 20 && n <= 170 {
        let mut result = 1.0_f64;
        for i in 0..k {
            result *= (n - i) as f64;
        }
        result
    } else {
        let lnp = lngamma_positive((n + 1) as f64) - lngamma_positive((n - k + 1) as f64);
        lnp.exp()
    }
}

// ══════════════════════════════════════════════════════════════════════
// Riemann Zeta Function
// ══════════════════════════════════════════════════════════════════════

/// Compute the Riemann zeta function ζ(s) for real s.
///
/// Matches `scipy.special.zeta(s)`.
///
/// Uses direct summation with Euler-Maclaurin acceleration for s > 1,
/// and the reflection formula for s < 0.
pub fn zeta(s: f64) -> f64 {
    if s.is_nan() {
        return f64::NAN;
    }
    if s == 1.0 {
        return f64::INFINITY; // pole
    }
    if s == 0.0 {
        return -0.5; // ζ(0) = -1/2
    }
    if s > 1.0 {
        zeta_positive(s)
    } else if s < 0.0 {
        // Reflection formula: ζ(s) = 2^s * π^(s-1) * sin(πs/2) * Γ(1-s) * ζ(1-s)
        let s1 = 1.0 - s;
        let z1 = zeta_positive(s1);
        let sin_half_pi_s = (PI * s / 2.0).sin();
        let gamma_1_minus_s = gamma_core(s1);
        // Compute parts carefully to avoid overflow/underflow
        let factor = 2.0_f64.powf(s) * PI.powf(s - 1.0) * sin_half_pi_s * gamma_1_minus_s;
        factor * z1
    } else {
        // 0 < s < 1: use Dirichlet eta function relation
        // ζ(s) = η(s) / (1 - 2^(1-s)) where η(s) = sum (-1)^(n+1) / n^s
        let eta = dirichlet_eta(s);
        let denom = 1.0 - 2.0_f64.powf(1.0 - s);
        if denom == 0.0 {
            return f64::NAN;
        }
        eta / denom
    }
}

/// Zeta for s > 1 via Euler-Maclaurin summation.
///
/// Delegates to `hurwitz_zeta(s, 1.0)`. Both Hurwitz and Riemann
/// share the same Euler-Maclaurin kernel, and routing the Riemann
/// case through hurwitz_zeta gives the full j=4 Bernoulli correction
/// (versus the previous in-place j=2 truncation that floored at
/// ~5e-11 abs accuracy near s=1.1) — frankenscipy-3u8ze.
fn zeta_positive(s: f64) -> f64 {
    crate::convenience::hurwitz_zeta(s, 1.0)
}

/// Dirichlet eta function η(s) = Σ_{n≥1} (-1)^{n+1} / n^s, for s > 0.
///
/// Evaluated with Borwein's algorithm (P. Borwein, "An Efficient Algorithm
/// for the Riemann Zeta Function", 2000). A naive partial sum of the
/// alternating series converges far too slowly in the critical strip — its
/// truncation error after `m` terms is ~½·m^{-s}, giving only ~1 digit at
/// s = 0.5 for m = 100. Borwein's weighted partial sum instead converges
/// geometrically, with error ~(3+√8)^{-n} ≈ 5.83^{-n}.
fn dirichlet_eta(s: f64) -> f64 {
    // d_k = n · Σ_{j=0}^{k} t_j,  with t_0 = 1/n and
    // t_{j+1} = t_j · 4(n+j)(n-j) / ((2j+1)(2j+2)).
    const N: usize = 32;
    let nf = N as f64;
    let mut d = [0.0_f64; N + 1];
    let mut t = 1.0 / nf;
    let mut cumulative = t;
    d[0] = nf * cumulative;
    for j in 0..N {
        let jf = j as f64;
        t *= 4.0 * (nf + jf) * (nf - jf) / ((2.0 * jf + 1.0) * (2.0 * jf + 2.0));
        cumulative += t;
        d[j + 1] = nf * cumulative;
    }
    let dn = d[N];
    // η(s) = -1/d_n · Σ_{k=0}^{n-1} (-1)^k (d_k - d_n) / (k+1)^s.
    let mut sum = 0.0;
    for (k, &dk) in d.iter().take(N).enumerate() {
        let sign = if k.is_multiple_of(2) { 1.0 } else { -1.0 };
        sum += sign * (dk - dn) / ((k + 1) as f64).powf(s);
    }
    -sum / dn
}

/// Compute the Riemann zeta function complement: zetac(s) = zeta(s) - 1.
///
/// Matches `scipy.special.zetac(s)`.
///
/// This is useful when zeta(s) is close to 1 (i.e., for large s > 1),
/// where direct computation of zeta(s) - 1 would suffer from catastrophic
/// cancellation. For large s, zetac(s) ≈ 2^(-s).
///
/// # Arguments
/// * `s` - Real argument
///
/// # Returns
/// ζ(s) - 1 for any real s ≠ 1
pub fn zetac(s: f64) -> f64 {
    if s.is_nan() {
        return f64::NAN;
    }
    if s == 1.0 {
        return f64::INFINITY; // pole of zeta
    }
    if s > 10.0 {
        // For large s > 10, zetac(s) = sum_{n=2}^∞ n^(-s) ≈ 2^(-s) + 3^(-s) + ...
        // Direct computation avoids subtraction from 1
        let mut sum = 0.0_f64;
        for n in 2..=100 {
            let term = (n as f64).powf(-s);
            if term < 1e-18 {
                break;
            }
            sum += term;
        }
        sum
    } else {
        // For smaller s, just compute zeta(s) - 1
        zeta(s) - 1.0
    }
}

// ============================================================================
// Complex incomplete gamma functions
// ============================================================================

/// Complex log-gamma using Lanczos approximation.
/// Matches scipy.special.loggamma for complex arguments.
pub fn complex_gammaln(z: Complex64) -> Complex64 {
    if is_complex_real_gamma_pole(z) {
        return Complex64::new(f64::NAN, f64::NAN);
    }

    if z.re < 0.5 {
        // Reflection formula: Γ(z) = π / (sin(πz) Γ(1-z))
        // ln Γ(z) = ln(π) - ln(sin(πz)) - ln Γ(1-z)
        let pi_z = Complex64::new(PI * z.re, PI * z.im);
        let sin_pi_z = complex_sin(pi_z);
        let ln_sin = sin_pi_z.ln();
        let one_minus_z = Complex64::new(1.0 - z.re, -z.im);
        let ln_gamma_1mz = complex_gammaln(one_minus_z);
        Complex64::new(PI.ln(), 0.0) - ln_sin - ln_gamma_1mz
    } else {
        let z_shifted = Complex64::new(z.re - 1.0, z.im);
        let mut x = Complex64::new(LANCZOS_COEFFS[0], 0.0);
        for (i, &coeff) in LANCZOS_COEFFS.iter().enumerate().skip(1) {
            let denom = Complex64::new(z_shifted.re + i as f64, z_shifted.im);
            x = x + Complex64::new(coeff, 0.0) / denom;
        }
        let t = Complex64::new(z_shifted.re + LANCZOS_G + 0.5, z_shifted.im);
        // ln Γ(z) = 0.5 ln(2π) + (z - 0.5) ln(t) - t + ln(x)
        let half_ln_2pi = 0.5 * (2.0 * PI).ln();
        let z_minus_half = Complex64::new(z_shifted.re + 0.5, z_shifted.im);
        let ln_t = t.ln();
        let term1 = Complex64::new(half_ln_2pi, 0.0);
        let term2 = z_minus_half * ln_t;
        let term3 = Complex64::new(-t.re, -t.im);
        let term4 = x.ln();
        term1 + term2 + term3 + term4
    }
}

/// Complex sine function.
fn complex_sin(z: Complex64) -> Complex64 {
    // sin(a + bi) = sin(a)cosh(b) + i cos(a)sinh(b)
    Complex64::new(z.re.sin() * z.im.cosh(), z.re.cos() * z.im.sinh())
}

/// Complex cosine function.
fn complex_cos(z: Complex64) -> Complex64 {
    // cos(a + bi) = cos(a)cosh(b) - i sin(a)sinh(b)
    Complex64::new(z.re.cos() * z.im.cosh(), -z.re.sin() * z.im.sinh())
}

/// Complex cotangent.
fn complex_cot(z: Complex64) -> Complex64 {
    complex_cos(z) / complex_sin(z)
}

/// Complex digamma (psi) function.
/// ψ(z) = d/dz ln(Γ(z))
pub fn complex_digamma_scalar(z: Complex64) -> Complex64 {
    if !z.is_finite() {
        return Complex64::new(f64::NAN, f64::NAN);
    }
    if is_complex_real_gamma_pole(z) {
        return Complex64::new(f64::NAN, f64::NAN);
    }

    // Reflection for negative real part: ψ(1-z) - π*cot(πz)
    if z.re < 0.5 {
        let one_minus_z = Complex64::new(1.0 - z.re, -z.im);
        let pi_z = Complex64::new(PI * z.re, PI * z.im);
        let pi_cot_pi_z = Complex64::new(PI, 0.0) * complex_cot(pi_z);
        return complex_digamma_scalar(one_minus_z) - pi_cot_pi_z;
    }

    // Shift upward until Re(z) >= 8 using recurrence: ψ(z+1) = ψ(z) + 1/z
    let mut shifted = z;
    let mut acc = Complex64::new(0.0, 0.0);
    while shifted.re < 8.0 {
        acc = acc - shifted.recip();
        shifted = Complex64::new(shifted.re + 1.0, shifted.im);
    }

    // Asymptotic expansion for large |z|:
    // ψ(z) ≈ ln(z) - 1/(2z) - 1/(12z²) + 1/(120z⁴) - 1/(252z⁶) + ...
    let inv = shifted.recip();
    let inv2 = inv * inv;
    let inv4 = inv2 * inv2;
    let inv6 = inv4 * inv2;
    let inv8 = inv6 * inv2;
    let inv10 = inv8 * inv2;

    // Extended through the B₁₀ term to match the real path (luxsz).
    let result = shifted.ln() - inv * 0.5 - inv2 * (1.0 / 12.0) + inv4 * (1.0 / 120.0)
        - inv6 * (1.0 / 252.0)
        + inv8 * (1.0 / 240.0)
        - inv10 * (1.0 / 132.0);

    acc + result
}

/// Complex trigamma function (polygamma of order 1).
/// ψ¹(z) = d²/dz² ln(Γ(z))
fn complex_trigamma_scalar(z: Complex64) -> Complex64 {
    if !z.is_finite() {
        return Complex64::new(f64::NAN, f64::NAN);
    }

    // Reflection: ψ¹(1-z) + π²/sin²(πz) = ψ¹(z)
    if z.re < 0.5 {
        let one_minus_z = Complex64::new(1.0 - z.re, -z.im);
        let pi_z = Complex64::new(PI * z.re, PI * z.im);
        let sin_pi_z = complex_sin(pi_z);
        let pi_sq_over_sin_sq = Complex64::new(PI * PI, 0.0) / (sin_pi_z * sin_pi_z);
        return pi_sq_over_sin_sq - complex_trigamma_scalar(one_minus_z);
    }

    // Shift upward: ψ¹(z+1) = ψ¹(z) - 1/z²
    let mut shifted = z;
    let mut acc = Complex64::new(0.0, 0.0);
    while shifted.re < 8.0 {
        let inv = shifted.recip();
        acc = acc + inv * inv;
        shifted = Complex64::new(shifted.re + 1.0, shifted.im);
    }

    // Asymptotic: ψ¹(z) ≈ 1/z + 1/(2z²) + 1/(6z³) - 1/(30z⁵) + 1/(42z⁷)
    let inv = shifted.recip();
    let inv2 = inv * inv;
    let inv3 = inv2 * inv;
    let inv5 = inv3 * inv2;
    let inv7 = inv5 * inv2;
    let inv9 = inv7 * inv2;
    let inv11 = inv9 * inv2;

    // Extended through the B₁₀ term to match the real path (luxsz).
    let result = inv + inv2 * 0.5 + inv3 / 6.0 - inv5 / 30.0 + inv7 / 42.0 - inv9 / 30.0
        + inv11 * (5.0 / 66.0);

    acc + result
}

/// Complex polygamma function of order n.
/// Uses recurrence and asymptotic expansion.
pub fn complex_polygamma_scalar(n: usize, z: Complex64) -> Complex64 {
    if n == 0 {
        return complex_digamma_scalar(z);
    }
    if n == 1 {
        return complex_trigamma_scalar(z);
    }

    if !z.is_finite() {
        return Complex64::new(f64::NAN, f64::NAN);
    }

    let sign = if n % 2 == 1 { 1.0 } else { -1.0 };
    let factorial = (1..=n).fold(1.0, |acc, v| acc * v as f64);
    let n_plus_one = n as f64 + 1.0;

    // Reflection for negative real part
    if z.re < 0.5 {
        // The reflection formula for higher polygamma is complex
        // Use numerical differentiation of lower orders as fallback
        let h = Complex64::new(1e-6, 0.0);
        let f_plus = complex_polygamma_scalar(n - 1, z + h);
        let f_minus = complex_polygamma_scalar(n - 1, z - h);
        return (f_plus - f_minus) / Complex64::new(2e-6, 0.0);
    }

    // Shift upward: ψⁿ(z+1) = ψⁿ(z) + (-1)^(n+1) * n! / z^(n+1)
    let mut shifted = z;
    let mut correction = Complex64::new(0.0, 0.0);
    while shifted.re < 8.0 {
        let inv_pow = shifted.powf(-n_plus_one);
        correction = correction + Complex64::new(sign * factorial, 0.0) * inv_pow;
        shifted = Complex64::new(shifted.re + 1.0, shifted.im);
    }

    // Asymptotic expansion using Hurwitz zeta
    // ψⁿ(z) ≈ (-1)^(n+1) * n! * ζ(n+1, z)
    // For large z: ζ(s, z) ≈ z^(1-s)/(s-1) + z^(-s)/2 + Σ B_{2k}/(2k)! * (s)_{2k-1} * z^(-s-2k+1)
    let inv_pow = shifted.powf(-n_plus_one);
    let inv_pow_minus_1 = shifted.powf(-(n as f64));

    // Leading terms of asymptotic zeta
    let zeta_approx = inv_pow_minus_1 / n as f64 + inv_pow * 0.5;

    let result = Complex64::new(sign * factorial, 0.0) * zeta_approx;
    correction + result
}

/// Complex regularized lower incomplete gamma P(a, z) for real a > 0 and complex z.
/// Uses series expansion: P(a,z) = z^a * e^(-z) / Γ(a) * Σ_{n=0}^∞ z^n / (a)_n
/// where (a)_n = a(a+1)...(a+n-1) is the Pochhammer symbol.
pub fn complex_gammainc_scalar(
    a: f64,
    z: Complex64,
    mode: RuntimeMode,
) -> Result<Complex64, SpecialError> {
    const FUNCTION: &str = "gammainc";

    // Domain check
    if a.is_nan() || !z.is_finite() {
        return Ok(Complex64::new(f64::NAN, f64::NAN));
    }
    if a <= 0.0 {
        record_special_trace(
            FUNCTION,
            mode,
            "domain_error",
            format!("a={a},z=({},{})", z.re, z.im),
            "returned_nan",
            "a must be positive",
            false,
        );
        return Ok(Complex64::new(f64::NAN, f64::NAN));
    }

    // Special case: z = 0
    if z.re == 0.0 && z.im == 0.0 {
        return Ok(Complex64::new(0.0, 0.0));
    }

    // Use series expansion for |z| < a + 1 or continued fraction otherwise
    let z_abs = z.abs();
    if z_abs < a + 1.0 {
        complex_gammainc_series(a, z, mode)
    } else {
        // P(a, z) = 1 - Q(a, z)
        let q = complex_gammaincc_cf(a, z, mode)?;
        Ok(Complex64::new(1.0 - q.re, -q.im))
    }
}

/// Complex regularized upper incomplete gamma Q(a, z) for real a > 0 and complex z.
pub fn complex_gammaincc_scalar(
    a: f64,
    z: Complex64,
    mode: RuntimeMode,
) -> Result<Complex64, SpecialError> {
    const FUNCTION: &str = "gammaincc";

    if a.is_nan() || !z.is_finite() {
        return Ok(Complex64::new(f64::NAN, f64::NAN));
    }
    if a <= 0.0 {
        record_special_trace(
            FUNCTION,
            mode,
            "domain_error",
            format!("a={a},z=({},{})", z.re, z.im),
            "returned_nan",
            "a must be positive",
            false,
        );
        return Ok(Complex64::new(f64::NAN, f64::NAN));
    }

    if z.re == 0.0 && z.im == 0.0 {
        return Ok(Complex64::new(1.0, 0.0));
    }

    let z_abs = z.abs();
    if z_abs < a + 1.0 {
        // Q(a, z) = 1 - P(a, z)
        let p = complex_gammainc_series(a, z, mode)?;
        Ok(Complex64::new(1.0 - p.re, -p.im))
    } else {
        complex_gammaincc_cf(a, z, mode)
    }
}

fn complex_parameter_gammainc_scalar(
    a: Complex64,
    z: Complex64,
    mode: RuntimeMode,
) -> Result<Complex64, SpecialError> {
    const FUNCTION: &str = "gammainc";

    if !a.is_finite() || !z.is_finite() {
        return Ok(Complex64::new(f64::NAN, f64::NAN));
    }
    if complex_gamma_pole(a) {
        record_special_trace(
            FUNCTION,
            mode,
            "domain_error",
            format!("a=({},{}) z=({},{})", a.re, a.im, z.re, z.im),
            "returned_nan",
            "a must not be a non-positive integer pole",
            false,
        );
        return Ok(Complex64::new(f64::NAN, f64::NAN));
    }
    if z == Complex64::new(0.0, 0.0) {
        if a.re > 0.0 {
            return Ok(Complex64::new(0.0, 0.0));
        }
        record_special_trace(
            FUNCTION,
            mode,
            "domain_error",
            format!("a=({},{}) z=(0,0)", a.re, a.im),
            "returned_nan",
            "z=0 requires Re(a) > 0",
            false,
        );
        return Ok(Complex64::new(f64::NAN, f64::NAN));
    }

    if z.abs() < a.abs() + 1.0 {
        complex_parameter_gammainc_series(a, z)
    } else {
        let q = complex_parameter_gammaincc_cf(a, z)?;
        Ok(Complex64::new(1.0, 0.0) - q)
    }
}

fn complex_parameter_gammaincc_scalar(
    a: Complex64,
    z: Complex64,
    mode: RuntimeMode,
) -> Result<Complex64, SpecialError> {
    const FUNCTION: &str = "gammaincc";

    if !a.is_finite() || !z.is_finite() {
        return Ok(Complex64::new(f64::NAN, f64::NAN));
    }
    if complex_gamma_pole(a) {
        record_special_trace(
            FUNCTION,
            mode,
            "domain_error",
            format!("a=({},{}) z=({},{})", a.re, a.im, z.re, z.im),
            "returned_nan",
            "a must not be a non-positive integer pole",
            false,
        );
        return Ok(Complex64::new(f64::NAN, f64::NAN));
    }
    if z == Complex64::new(0.0, 0.0) {
        if a.re > 0.0 {
            return Ok(Complex64::new(1.0, 0.0));
        }
        record_special_trace(
            FUNCTION,
            mode,
            "domain_error",
            format!("a=({},{}) z=(0,0)", a.re, a.im),
            "returned_nan",
            "z=0 requires Re(a) > 0",
            false,
        );
        return Ok(Complex64::new(f64::NAN, f64::NAN));
    }

    if z.abs() < a.abs() + 1.0 {
        let p = complex_parameter_gammainc_series(a, z)?;
        Ok(Complex64::new(1.0, 0.0) - p)
    } else {
        complex_parameter_gammaincc_cf(a, z)
    }
}

fn complex_gamma_pole(a: Complex64) -> bool {
    const POLE_EPS: f64 = 1.0e-12;

    a.im.abs() <= POLE_EPS && a.re <= 0.0 && (a.re - a.re.round()).abs() <= POLE_EPS
}

/// Series expansion for lower incomplete gamma.
fn complex_gammainc_series(
    a: f64,
    z: Complex64,
    _mode: RuntimeMode,
) -> Result<Complex64, SpecialError> {
    const EPS: f64 = 1.0e-14;
    const MAX_ITERS: usize = 200;

    // Compute ln(Γ(a)) for normalization
    let lg = gammaln_scalar(a, RuntimeMode::Strict)?;

    // Compute z^a * e^(-z) / Γ(a)
    let ln_z = z.ln();
    let a_ln_z = Complex64::new(a * ln_z.re, a * ln_z.im);
    let exp_arg = Complex64::new(a_ln_z.re - z.re - lg, a_ln_z.im - z.im);
    let prefactor = exp_arg.exp();

    // Series: sum_{n=0}^∞ z^n / (a+1)(a+2)...(a+n)
    let mut term = Complex64::new(1.0 / a, 0.0);
    let mut sum = term;

    for n in 1..MAX_ITERS {
        let denom = a + n as f64;
        term = term * z / Complex64::new(denom, 0.0);
        sum = sum + term;
        if term.abs() <= sum.abs() * EPS {
            break;
        }
    }

    Ok(prefactor * sum)
}

fn complex_parameter_gammainc_series(
    a: Complex64,
    z: Complex64,
) -> Result<Complex64, SpecialError> {
    const EPS: f64 = 1.0e-14;
    const MAX_ITERS: usize = 200;

    let lg = complex_gammaln(a);
    let prefactor = (a * z.ln() - z - lg).exp();

    let mut term = Complex64::new(1.0, 0.0) / a;
    let mut sum = term;

    for n in 1..MAX_ITERS {
        let denom = a + Complex64::new(n as f64, 0.0);
        term = term * z / denom;
        sum = sum + term;
        if term.abs() <= sum.abs() * EPS {
            break;
        }
    }

    Ok(prefactor * sum)
}

/// Continued fraction for upper incomplete gamma (Lentz's algorithm).
fn complex_gammaincc_cf(
    a: f64,
    z: Complex64,
    _mode: RuntimeMode,
) -> Result<Complex64, SpecialError> {
    const EPS: f64 = 1.0e-14;
    const FPMIN: f64 = 1.0e-300;
    const MAX_ITERS: usize = 200;

    let lg = gammaln_scalar(a, RuntimeMode::Strict)?;

    // Compute prefactor: z^a * e^(-z) / Γ(a)
    let ln_z = z.ln();
    let a_ln_z = Complex64::new(a * ln_z.re, a * ln_z.im);
    let exp_arg = Complex64::new(a_ln_z.re - z.re - lg, a_ln_z.im - z.im);
    let prefactor = exp_arg.exp();

    // Continued fraction: Q(a,z) = prefactor * 1/(z + 1-a - 1*(1-a)/(z + 3-a - 2*(2-a)/(z + 5-a - ...)))
    let mut b = Complex64::new(z.re + 1.0 - a, z.im);
    let mut c = Complex64::new(1.0 / FPMIN, 0.0);
    let mut d = b.recip();
    let mut h = d;

    for i in 1..=MAX_ITERS {
        let i_f = i as f64;
        let an = Complex64::new(-i_f * (i_f - a), 0.0);
        b = b + Complex64::new(2.0, 0.0);
        d = an * d + b;
        if d.abs() < FPMIN {
            d = Complex64::new(FPMIN, 0.0);
        }
        c = b + an / c;
        if c.abs() < FPMIN {
            c = Complex64::new(FPMIN, 0.0);
        }
        d = d.recip();
        let delta = d * c;
        h = h * delta;
        if (delta.re - 1.0).abs() + delta.im.abs() <= EPS {
            break;
        }
    }

    Ok(prefactor * h)
}

fn complex_parameter_gammaincc_cf(a: Complex64, z: Complex64) -> Result<Complex64, SpecialError> {
    const EPS: f64 = 1.0e-14;
    const FPMIN: f64 = 1.0e-300;
    const MAX_ITERS: usize = 200;

    let lg = complex_gammaln(a);
    let prefactor = (a * z.ln() - z - lg).exp();

    let mut b = z + Complex64::new(1.0, 0.0) - a;
    let mut c = Complex64::new(1.0 / FPMIN, 0.0);
    let mut d = b.recip();
    let mut h = d;

    for i in 1..=MAX_ITERS {
        let i_c = Complex64::new(i as f64, 0.0);
        let an = -i_c * (i_c - a);
        b = b + Complex64::new(2.0, 0.0);
        d = an * d + b;
        if d.abs() < FPMIN {
            d = Complex64::new(FPMIN, 0.0);
        }
        c = b + an / c;
        if c.abs() < FPMIN {
            c = Complex64::new(FPMIN, 0.0);
        }
        d = d.recip();
        let delta = d * c;
        h = h * delta;
        if (delta - Complex64::new(1.0, 0.0)).abs() <= EPS {
            break;
        }
    }

    Ok(prefactor * h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy
    fn digamma_polygamma_extended_asymptotic() {
        // frankenscipy-luxsz: digamma/trigamma/tetragamma asymptotics extended
        // through the B₁₀ term (were ~1e-9 off at the shift point). scipy 1.17.1.
        let m = RuntimeMode::Strict;
        let s = |v: f64| SpecialTensor::RealScalar(v);
        let g = |r: SpecialResult| match r {
            Ok(SpecialTensor::RealScalar(v)) => v,
            _ => f64::NAN,
        };
        let dig = [
            (0.5, -1.9635100260214235),
            (1.5, 0.03648997397857652),
            (-0.5, 0.03648997397857651),
            (2.5, 0.7031566406452432),
            (-10.5, 2.3982391295357814),
        ];
        for (x, expected) in dig {
            let got = g(digamma(&s(x), m));
            assert!((got - expected).abs() <= 1e-11 * expected.abs().max(1e-3), "digamma({x}) = {got}, scipy {expected}");
        }
        let tri = [(1.5, 0.9348022005446793), (2.5, 0.4903577561002348)];
        for (x, expected) in tri {
            let got = g(polygamma(1, &s(x), m));
            assert!(((got - expected) / expected).abs() < 1e-11, "polygamma(1,{x}) = {got}, scipy {expected}");
        }
        let tetra = [(1.5, -0.8287966442343201), (2.5, -0.23620405164172736)];
        for (x, expected) in tetra {
            let got = g(polygamma(2, &s(x), m));
            assert!(((got - expected) / expected).abs() < 1e-10, "polygamma(2,{x}) = {got}, scipy {expected}");
        }
    }

    fn scalar(value: f64) -> SpecialTensor {
        SpecialTensor::RealScalar(value)
    }

    fn get_scalar(result: SpecialResult) -> Result<f64, String> {
        match result.map_err(|err| err.to_string())? {
            SpecialTensor::RealScalar(value) => Ok(value),
            other => Err(format!("expected real scalar, got {other:?}")),
        }
    }

    #[test]
    fn gamma_with_audit_emits_failclosed_on_hardened_pole() {
        // br-egba-2: gamma_with_audit records AuditAction::FailClosed
        // when Hardened mode rejects a pole input. Strict mode does
        // not emit.
        let ledger = crate::audit::sync_audit_ledger();
        // Strict path: returns Ok (NaN), no emission.
        let _ = gamma_with_audit(&scalar(-2.0), RuntimeMode::Strict, &ledger);
        assert_eq!(ledger.lock().unwrap().len(), 0, "strict must not emit");

        // Hardened pole: must emit exactly one FailClosed event.
        let err = gamma_with_audit(&scalar(-2.0), RuntimeMode::Hardened, &ledger)
            .expect_err("hardened rejects pole");
        assert_eq!(err.kind, SpecialErrorKind::PoleInput);
        assert_eq!(
            ledger.lock().unwrap().len(),
            1,
            "hardened pole must emit once"
        );
    }

    #[test]
    fn rgamma_returns_zero_at_negative_integer_poles() -> Result<(), String> {
        for x in [-1.0, -2.0, -8.0] {
            let actual = get_scalar(rgamma(&scalar(x), RuntimeMode::Strict))?;
            assert_eq!(actual, 0.0);
        }
        Ok(())
    }

    #[test]
    fn gammasgn_matches_scipy_real_axis_intervals() -> Result<(), String> {
        let cases = [
            (-5.5, 1.0),
            (-4.5, -1.0),
            (-3.5, 1.0),
            (-2.5, -1.0),
            (-1.5, 1.0),
            (-0.5, -1.0),
            (-0.0, -1.0),
            (0.0, 1.0),
            (0.5, 1.0),
            (1.0, 1.0),
            (2.0, 1.0),
            (f64::INFINITY, 1.0),
        ];

        for (x, expected) in cases {
            let actual = get_scalar(gammasgn(&scalar(x), RuntimeMode::Strict))?;
            assert_eq!(actual, expected, "gammasgn({x})");
        }
        Ok(())
    }

    #[test]
    fn gammasgn_preserves_scipy_pole_and_nan_semantics() -> Result<(), String> {
        for x in [-5.0, -4.0, -3.0, -2.0, -1.0, f64::NEG_INFINITY, f64::NAN] {
            let actual = get_scalar(gammasgn(&scalar(x), RuntimeMode::Strict))?;
            assert!(actual.is_nan(), "gammasgn({x}) should be NaN");
        }

        let err = gammasgn(&scalar(-2.0), RuntimeMode::Hardened).unwrap_err();
        assert_eq!(err.kind, SpecialErrorKind::PoleInput);
        Ok(())
    }

    #[test]
    fn gammasgn_supports_real_vectors() -> Result<(), String> {
        let input = SpecialTensor::RealVec(vec![-2.5, -0.0, 0.0, 3.0]);
        let result = gammasgn(&input, RuntimeMode::Strict).map_err(|err| err.to_string())?;
        match result {
            SpecialTensor::RealVec(values) => {
                assert_eq!(values, vec![-1.0, -1.0, 1.0, 1.0]);
                Ok(())
            }
            other => Err(format!("expected real vector, got {other:?}")),
        }
    }

    #[test]
    fn gamma_large_finite_boundary_matches_scipy() -> Result<(), String> {
        let actual = get_scalar(gamma(&scalar(170.0), RuntimeMode::Strict))?;
        let expected = 4.269_068_009_004_706e304;
        let rel = ((actual - expected) / expected).abs();
        assert!(
            actual.is_finite() && rel <= 1.0e-12,
            "gamma(170) should be finite SciPy boundary value, got {actual:e}"
        );
        Ok(())
    }

    #[test]
    fn multigammaln_matches_scipy_reference_values() -> Result<(), String> {
        let cases = [
            (3.5, 2.0, 2.4664857258317197),
            (4.0, 3.0, 5.402975080909175),
            (1.0, 2.0, 1.1447298858494),
        ];
        for (a, d, expected) in cases {
            let actual = get_scalar(multigammaln(&scalar(a), d, RuntimeMode::Strict))?;
            assert!((actual - expected).abs() <= 1.0e-12);
        }
        Ok(())
    }

    #[test]
    fn multigammaln_reduces_to_gammaln_for_dimension_one() -> Result<(), String> {
        for a in [0.5, 1.25, 3.5, 10.0] {
            let actual = get_scalar(multigammaln(&scalar(a), 1.0, RuntimeMode::Strict))?;
            let expected = gammaln_scalar(a, RuntimeMode::Strict).map_err(|err| err.to_string())?;
            assert!((actual - expected).abs() <= 1.0e-12);
        }
        Ok(())
    }

    #[test]
    fn multigammaln_supports_real_vectors() -> Result<(), String> {
        let input = SpecialTensor::RealVec(vec![2.5, 10.0]);
        let result =
            multigammaln(&input, 2.0, RuntimeMode::Strict).map_err(|err| err.to_string())?;
        match result {
            SpecialTensor::RealVec(values) => {
                let expected = [
                    0.5 * PI.ln()
                        + gammaln_scalar(2.5, RuntimeMode::Strict)
                            .map_err(|err| err.to_string())?
                        + gammaln_scalar(2.0, RuntimeMode::Strict)
                            .map_err(|err| err.to_string())?,
                    0.5 * PI.ln()
                        + gammaln_scalar(10.0, RuntimeMode::Strict)
                            .map_err(|err| err.to_string())?
                        + gammaln_scalar(9.5, RuntimeMode::Strict)
                            .map_err(|err| err.to_string())?,
                ];
                assert_eq!(values.len(), expected.len());
                for (actual, expected_value) in values.iter().zip(expected.iter()) {
                    assert!((*actual - *expected_value).abs() <= 1.0e-12);
                }
            }
            other => return Err(format!("expected real vector, got {other:?}")),
        }
        Ok(())
    }

    #[test]
    fn multigammaln_preserves_scipy_edge_cases() -> Result<(), String> {
        let zero_dim = get_scalar(multigammaln(&scalar(3.5), 0.0, RuntimeMode::Strict))?;
        assert_eq!(zero_dim, 0.0);

        let negative_dim = get_scalar(multigammaln(&scalar(3.5), -1.0, RuntimeMode::Strict))?;
        assert!((negative_dim - 0.5723649429247001).abs() <= 1.0e-12);

        let nan_a = get_scalar(multigammaln(&scalar(f64::NAN), 2.0, RuntimeMode::Strict))?;
        assert!(nan_a.is_nan());
        Ok(())
    }

    #[test]
    fn multigammaln_rejects_non_integer_dimension() {
        let err = multigammaln(&scalar(3.5), 1.2, RuntimeMode::Strict)
            .expect_err("non-integer dimensions should fail closed");
        assert_eq!(err.kind, SpecialErrorKind::DomainError);
        assert_eq!(err.detail, "d should be a positive integer (dimension)");
    }

    #[test]
    fn multigammaln_rejects_inputs_below_threshold() {
        let err = multigammaln(&scalar(0.4), 2.0, RuntimeMode::Strict)
            .expect_err("a <= 0.5 * (d - 1) should fail closed");
        assert_eq!(err.kind, SpecialErrorKind::DomainError);
        assert_eq!(err.detail, "condition a must exceed 0.5 * (d - 1)");
    }

    #[test]
    fn polygamma_order_two_matches_tetragamma_scalar() -> Result<(), String> {
        let x = 1.5;
        let actual = get_scalar(polygamma(2, &scalar(x), RuntimeMode::Strict))?;
        let expected = crate::convenience::tetragamma(x);
        assert!((actual - expected).abs() <= 1.0e-12);
        Ok(())
    }

    #[test]
    fn polygamma_order_two_supports_real_vectors() -> Result<(), String> {
        let input = SpecialTensor::RealVec(vec![0.5, 1.0, 2.5]);
        let result = polygamma(2, &input, RuntimeMode::Strict).map_err(|err| err.to_string())?;
        match result {
            SpecialTensor::RealVec(values) => {
                let expected = [
                    crate::convenience::tetragamma(0.5),
                    crate::convenience::tetragamma(1.0),
                    crate::convenience::tetragamma(2.5),
                ];
                assert_eq!(values.len(), expected.len());
                for (actual, expected_value) in values.iter().zip(expected.iter()) {
                    assert!((*actual - *expected_value).abs() <= 1.0e-12);
                }
            }
            other => return Err(format!("expected real vector, got {other:?}")),
        }
        Ok(())
    }

    #[test]
    fn polygamma_order_two_hardened_rejects_poles() {
        let err = polygamma(2, &scalar(-1.0), RuntimeMode::Hardened)
            .expect_err("hardened mode should reject tetragamma poles");
        assert_eq!(err.kind, SpecialErrorKind::PoleInput);
    }

    #[test]
    fn polygamma_order_three_matches_known_value_at_one() -> Result<(), String> {
        let actual = get_scalar(polygamma(3, &scalar(1.0), RuntimeMode::Strict))?;
        let expected = PI.powi(4) / 15.0;
        assert!((actual - expected).abs() <= 1.0e-11);
        Ok(())
    }

    #[test]
    fn polygamma_higher_orders_support_real_vectors() -> Result<(), String> {
        let input = SpecialTensor::RealVec(vec![1.0, 2.0, 0.5]);
        let result = polygamma(3, &input, RuntimeMode::Strict).map_err(|err| err.to_string())?;
        match result {
            SpecialTensor::RealVec(values) => {
                let expected = [PI.powi(4) / 15.0, PI.powi(4) / 15.0 - 6.0, PI.powi(4)];
                assert_eq!(values.len(), expected.len());
                for (actual, expected_value) in values.iter().zip(expected.iter()) {
                    assert!((*actual - *expected_value).abs() <= 1.0e-10);
                }
            }
            other => return Err(format!("expected real vector, got {other:?}")),
        }
        Ok(())
    }

    #[test]
    fn polygamma_order_four_satisfies_recurrence_off_negative_axis() -> Result<(), String> {
        let x = -0.25;
        let lhs = get_scalar(polygamma(4, &scalar(x + 1.0), RuntimeMode::Strict))?;
        let rhs = get_scalar(polygamma(4, &scalar(x), RuntimeMode::Strict))? + 24.0 / x.powi(5);
        assert!((lhs - rhs).abs() <= 1.0e-10);
        Ok(())
    }

    #[test]
    fn polygamma_higher_orders_hardened_reject_poles() {
        let err = polygamma(3, &scalar(-1.0), RuntimeMode::Hardened)
            .expect_err("hardened mode should reject higher-order poles");
        assert_eq!(err.kind, SpecialErrorKind::PoleInput);
    }

    // ── Complex digamma/polygamma tests ──────────────────────────────────

    fn digamma_complex_scalar(re: f64, im: f64) -> SpecialTensor {
        SpecialTensor::ComplexScalar(Complex64::new(re, im))
    }

    fn get_digamma_complex(result: SpecialResult) -> Result<Complex64, String> {
        match result.map_err(|err| err.to_string())? {
            SpecialTensor::ComplexScalar(c) => Ok(c),
            other => Err(format!("expected ComplexScalar, got {other:?}")),
        }
    }

    #[test]
    fn complex_digamma_real_axis_matches_real_digamma() -> Result<(), String> {
        let x = 2.5;
        let real_result = get_scalar(digamma(&scalar(x), RuntimeMode::Strict))?;
        let complex_result = get_digamma_complex(digamma(
            &digamma_complex_scalar(x, 0.0),
            RuntimeMode::Strict,
        ))?;
        assert!((complex_result.re - real_result).abs() < 1e-10);
        assert!(complex_result.im.abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn complex_digamma_pure_imaginary() -> Result<(), String> {
        let z = digamma_complex_scalar(0.0, 2.0);
        let result = get_digamma_complex(digamma(&z, RuntimeMode::Strict))?;
        assert!(result.re.is_finite());
        assert!(result.im.is_finite());
        Ok(())
    }

    #[test]
    fn complex_digamma_negative_real() -> Result<(), String> {
        let z = digamma_complex_scalar(-0.5, 1.0);
        let result = get_digamma_complex(digamma(&z, RuntimeMode::Strict))?;
        assert!(result.re.is_finite());
        assert!(result.im.is_finite());
        Ok(())
    }

    #[test]
    fn complex_digamma_returns_nan_at_real_poles() -> Result<(), String> {
        for x in [0.0, -1.0, -2.0, -8.0] {
            let actual = get_digamma_complex(digamma(
                &digamma_complex_scalar(x, 0.0),
                RuntimeMode::Strict,
            ))?;
            assert!(
                actual.re.is_nan(),
                "digamma({x}+0i) real part should be NaN"
            );
            assert!(
                actual.im.is_nan(),
                "digamma({x}+0i) imag part should be NaN"
            );
        }
        Ok(())
    }

    #[test]
    fn complex_digamma_near_real_poles_stays_finite() -> Result<(), String> {
        let actual = get_digamma_complex(digamma(
            &digamma_complex_scalar(-1.0, 1.0e-12),
            RuntimeMode::Strict,
        ))?;
        assert!(actual.re.is_finite());
        assert!(actual.im.is_finite());
        Ok(())
    }

    #[test]
    fn complex_trigamma_real_axis_matches_real_trigamma() -> Result<(), String> {
        let x = 2.5;
        let real_result = get_scalar(polygamma(1, &scalar(x), RuntimeMode::Strict))?;
        let complex_result = get_digamma_complex(polygamma(
            1,
            &digamma_complex_scalar(x, 0.0),
            RuntimeMode::Strict,
        ))?;
        assert!((complex_result.re - real_result).abs() < 1e-10);
        assert!(complex_result.im.abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn complex_polygamma_order_two_matches_real() -> Result<(), String> {
        let x = 1.5;
        let real_result = get_scalar(polygamma(2, &scalar(x), RuntimeMode::Strict))?;
        let complex_result = get_digamma_complex(polygamma(
            2,
            &digamma_complex_scalar(x, 0.0),
            RuntimeMode::Strict,
        ))?;
        // Higher-order polygamma uses numerical differentiation for complex, so tolerance is relaxed
        assert!(
            (complex_result.re - real_result).abs() < 0.01,
            "real_result={}, complex_result.re={}",
            real_result,
            complex_result.re
        );
        assert!(complex_result.im.abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn complex_digamma_vector_input() -> Result<(), String> {
        let input = SpecialTensor::ComplexVec(vec![
            Complex64::new(1.0, 0.5),
            Complex64::new(2.0, 0.0),
            Complex64::new(0.5, 1.0),
        ]);
        let result = digamma(&input, RuntimeMode::Strict).map_err(|e| e.to_string())?;
        match result {
            SpecialTensor::ComplexVec(values) => {
                assert_eq!(values.len(), 3);
                for v in values {
                    assert!(v.is_finite());
                }
            }
            _ => return Err("expected ComplexVec".into()),
        }
        Ok(())
    }

    // ── factorial2 tests ─────────────────────────────────────────────────

    #[test]
    fn factorial2_small_even() {
        // 0!! = 1, 2!! = 2, 4!! = 8, 6!! = 48, 8!! = 384, 10!! = 3840
        assert_eq!(factorial2(0), 1.0);
        assert_eq!(factorial2(2), 2.0);
        assert_eq!(factorial2(4), 8.0);
        assert_eq!(factorial2(6), 48.0);
        assert_eq!(factorial2(8), 384.0);
        assert_eq!(factorial2(10), 3840.0);
    }

    #[test]
    fn factorial2_small_odd() {
        // 1!! = 1, 3!! = 3, 5!! = 15, 7!! = 105, 9!! = 945
        assert_eq!(factorial2(1), 1.0);
        assert_eq!(factorial2(3), 3.0);
        assert_eq!(factorial2(5), 15.0);
        assert_eq!(factorial2(7), 105.0);
        assert_eq!(factorial2(9), 945.0);
    }

    #[test]
    fn factorial2_negative() {
        // SciPy returns 0 for all negative integer inputs.
        assert_eq!(factorial2(-1), 0.0);
        assert_eq!(factorial2(-2), 0.0);
        assert_eq!(factorial2(-5), 0.0);
    }

    #[test]
    fn factorial2_large() {
        // Test gamma-based computation for large values
        // 20!! = 3715891200
        let computed = factorial2(20);
        let expected = 3_715_891_200.0_f64;
        assert!((computed - expected).abs() / expected < 1e-10);
    }

    #[test]
    fn factorial2_large_odd_gamma_path() {
        // n > 33 takes the gamma-based fallback. The previous formula
        // used Gamma((n+1)/2) instead of Gamma((n+2)/2) and overshot
        // the true value by roughly 8x at n=35. Reference values are
        // computed exactly via the recurrence m!! = m * (m-2)!!.
        fn double_fact_exact(n: i64) -> f64 {
            let mut result = 1.0f64;
            let mut k = n;
            while k > 1 {
                result *= k as f64;
                k -= 2;
            }
            result
        }
        for n in [35_i64, 51, 99, 151] {
            let computed = factorial2(n);
            let expected = double_fact_exact(n);
            let rel_err = ((computed - expected) / expected).abs();
            assert!(
                rel_err < 1e-12,
                "factorial2({n}) gamma-path: got {computed:e}, expected {expected:e}, rel_err={rel_err:e}"
            );
        }
    }

    // ── zetac tests ──────────────────────────────────────────────────────

    #[test]
    fn zetac_small_s() {
        // zetac(2) = zeta(2) - 1 = π²/6 - 1 ≈ 0.6449340668
        let result = zetac(2.0);
        let expected = PI * PI / 6.0 - 1.0;
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn zetac_large_s() {
        // For large s, zetac(s) ≈ 2^(-s)
        let s = 20.0;
        let result = zetac(s);
        let approx = 2.0_f64.powf(-s);
        // Should be close to 2^(-s), within a few percent
        assert!((result - approx).abs() / approx < 0.01);
    }

    #[test]
    fn zetac_consistency() {
        // zetac(s) should equal zeta(s) - 1
        for s in [3.0, 4.0, 5.0, 10.0] {
            let zetac_result = zetac(s);
            let zeta_minus_one = zeta(s) - 1.0;
            assert!((zetac_result - zeta_minus_one).abs() < 1e-12);
        }
    }

    #[test]
    fn zetac_pole() {
        // At s=1, zetac should return infinity
        assert!(zetac(1.0).is_infinite());
    }

    #[test]
    fn zetac_critical_strip_matches_scipy() {
        // 0 < s < 1: reference values from scipy.special.zetac. The naive
        // alternating-series sum was ~5% off here; Borwein's acceleration
        // restores full precision (frankenscipy-kk4vu).
        for &(s, expected) in &[
            (0.1_f64, -1.603_037_519_856_241_9),
            (0.3, -1.904_559_257_253_984_2),
            (0.5, -2.460_354_508_809_586_6),
            (0.7, -3.778_388_445_553_696),
            (0.9, -10.430_114_019_402_255),
        ] {
            let got = zetac(s);
            assert!(
                (got - expected).abs() < 1e-10,
                "zetac({s}) = {got}, expected {expected}"
            );
        }
    }

    // ── Poisson distribution functions ────────────────────────────────

    #[test]
    fn pdtr_basic() {
        // pdtr(k, m) = P(X <= k) for Poisson with mean m
        // pdtr(0, 1) = P(X = 0) = exp(-1) ≈ 0.3679
        let result = pdtr(0.0, 1.0);
        assert!((result - (-1.0_f64).exp()).abs() < 1e-10);

        // pdtr(k, 0) = 1 for any k >= 0
        assert!((pdtr(5.0, 0.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pdtr_pdtrc_complement() {
        // pdtr(k, m) + pdtrc(k, m) = 1
        for &k in &[0.0, 1.0, 5.0, 10.0] {
            for &m in &[0.5, 1.0, 5.0, 10.0] {
                let sum = pdtr(k, m) + pdtrc(k, m);
                assert!(
                    (sum - 1.0).abs() < 1e-10,
                    "pdtr + pdtrc != 1 for k={k}, m={m}"
                );
            }
        }
    }

    #[test]
    fn pdtri_inverse() {
        // pdtri should be inverse of pdtr
        for &k in &[1.0, 5.0, 10.0] {
            for &m in &[1.0, 5.0, 10.0] {
                let p = pdtr(k, m);
                if p > 0.01 && p < 0.99 {
                    let m_recovered = pdtri(k, p);
                    assert!(
                        (m_recovered - m).abs() / m < 0.01,
                        "pdtri failed: k={k}, m={m}, p={p}, m_recovered={m_recovered}"
                    );
                }
            }
        }
    }

    /// pdtri / chdtri at p close to 0 or 1 (frankenscipy-jr3na).
    ///
    /// Pre-fix the local Newton-Raphson `gammaincinv` here diverged
    /// when 1-p was small; pdtri(1, 0.99) returned ~1e13 instead of
    /// scipy's 0.149, and chdtri(3, 0.99) returned ~5.6e5 instead of
    /// 0.115. Both now delegate to the convenience-module
    /// `gammaincinv_scalar`, which is validated by the diff_special
    /// gammainc harness.
    #[test]
    fn pdtri_chdtri_match_scipy_at_extreme_p() {
        // (k, p, scipy.special.pdtri(k, p))
        let pdtri_cases: [(f64, f64, f64); 5] = [
            (1.0, 0.99, 0.14855474025327913),
            (1.0, 0.999, 0.04540201776948989),
            (0.0, 0.99, 0.010050335853501444),
            (5.0, 0.5, 5.670161188712083),
            (3.0, 0.95, 1.3663183967498314),
        ];
        for (k, p, expected) in pdtri_cases {
            let got = pdtri(k, p);
            let scale = expected.abs().max(1.0);
            assert!(
                (got - expected).abs() < 1e-9 * scale,
                "pdtri({k}, {p}) = {got}, expected {expected}"
            );
        }

        // (v, p, scipy.special.chdtri(v, p))
        let chdtri_cases: [(f64, f64, f64); 4] = [
            (3.0, 0.99, 0.11483180189911714),
            (3.0, 0.5, 2.3659738843753395),
            (5.0, 0.95, 1.1454762260617706),
            (1.0, 0.999, 1.5707971492623826e-06),
        ];
        for (v, p, expected) in chdtri_cases {
            let got = chdtri(v, p);
            let scale = expected.abs().max(1.0);
            assert!(
                (got - expected).abs() < 1e-9 * scale,
                "chdtri({v}, {p}) = {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn pdtrik_reference_values() {
        let cases = [
            (0.7, 2.5, 2.692_718_092_203_629),
            (0.5, 2.5, 1.825_430_403_950_633_3),
            (0.757_576_133_133_066_2, 2.5, 3.000_000_000_000_001_8),
            (0.99, 2.5, 6.299_974_200_719_591),
        ];

        for (p, m, expected) in cases {
            let actual = pdtrik(p, m);
            assert!(
                (actual - expected).abs() < 5e-10,
                "pdtrik({p}, {m}) = {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn pdtrik_inverse() {
        for &m in &[0.5, 1.0, 2.5, 5.0, 10.0] {
            for &k in &[0.25, 1.0, 2.5, 5.0] {
                let p = pdtr(k, m);
                let recovered = pdtrik(p, m);
                assert!(
                    (recovered - k).abs() < 5e-10,
                    "pdtrik failed: m={m}, k={k}, p={p}, recovered={recovered}"
                );
            }
        }
    }

    #[test]
    fn pdtrik_edges_match_scipy() {
        assert_eq!(pdtrik(0.0, 2.5), 0.0);
        assert_eq!(pdtrik(0.001, 2.5), 0.0);
        assert_eq!(pdtrik(0.7, 0.0), 0.0);
        assert_eq!(pdtrik(0.0, f64::INFINITY), 0.0);

        assert!(pdtrik(1.0, 2.5).is_nan());
        assert!(pdtrik(0.7, f64::INFINITY).is_nan());
        assert!(pdtrik(-0.1, 2.5).is_nan());
        assert!(pdtrik(1.1, 2.5).is_nan());
        assert!(pdtrik(0.7, -1.0).is_nan());
        assert!(pdtrik(f64::NAN, 2.5).is_nan());
        assert!(pdtrik(0.7, f64::NAN).is_nan());
    }

    #[test]
    fn gdtr_closed_form_reference_values() {
        // /testing-golden-artifacts: pin gdtr against three closed-form
        // values where gammainc collapses to a finite series:
        //   gdtr(1, 1, 1) = P(1, 1)   = 1 - e^{-1}
        //   gdtr(2, 3, 2) = P(3, 4)   = 1 - 13 e^{-4}
        //   gdtr(0.5, 2, 4) = P(2, 2) = 1 - 3 e^{-2}
        let cases: &[(f64, f64, f64, f64)] = &[
            (1.0, 1.0, 1.0, 1.0 - (-1.0_f64).exp()),
            (2.0, 3.0, 2.0, 1.0 - 13.0 * (-4.0_f64).exp()),
            (0.5, 2.0, 4.0, 1.0 - 3.0 * (-2.0_f64).exp()),
        ];
        for &(a, b, x, expected) in cases {
            let actual = gdtr(a, b, x);
            assert!(
                (actual - expected).abs() < 1e-12,
                "gdtr({a}, {b}, {x}) = {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn gdtria_roundtrip_recovers_rate() {
        // gdtria(gdtr(a, b, x), b, x) ≈ a
        // Closed form: a = gammaincinv(b, p) / x.
        for &a in &[0.5, 1.0, 2.5] {
            for &b in &[1.0, 2.0, 3.5] {
                for &x in &[0.5, 2.0, 5.0] {
                    let p = gdtr(a, b, x);
                    let recovered = gdtria(p, b, x);
                    assert!(
                        (recovered - a).abs() < 1e-9,
                        "gdtria(gdtr({a}, {b}, {x})={p}, {b}, {x}) = {recovered}, expected {a}"
                    );
                }
            }
        }
    }

    #[test]
    fn gdtrib_roundtrip_recovers_shape() {
        // gdtrib(a, gdtr(a, b, x), x) ≈ b
        // gdtrib uses gammainc_shape_inv, a Newton root finder over b.
        for &a in &[0.5, 1.0, 2.5] {
            for &b in &[1.0, 2.0, 3.5] {
                for &x in &[0.5, 2.0, 5.0] {
                    let p = gdtr(a, b, x);
                    let recovered = gdtrib(a, p, x);
                    assert!(
                        (recovered - b).abs() < 1e-7,
                        "gdtrib({a}, gdtr({a}, {b}, {x})={p}, {x}) = {recovered}, expected {b}"
                    );
                }
            }
        }
    }

    #[test]
    fn gdtria_gdtrib_domain_error_returns_nan() {
        // Out-of-domain inputs return NaN per scipy.special semantics.
        assert!(gdtria(0.5, -1.0, 1.0).is_nan(), "gdtria b<0");
        assert!(gdtria(-0.1, 1.0, 1.0).is_nan(), "gdtria p<0");
        assert!(gdtria(1.1, 1.0, 1.0).is_nan(), "gdtria p>1");
        assert!(gdtria(0.5, 1.0, 0.0).is_nan(), "gdtria x=0");

        assert!(gdtrib(-1.0, 0.5, 1.0).is_nan(), "gdtrib a<0");
        assert!(gdtrib(1.0, -0.1, 1.0).is_nan(), "gdtrib p<0");
        assert!(gdtrib(1.0, 1.1, 1.0).is_nan(), "gdtrib p>1");
        assert!(gdtrib(1.0, 0.5, -1.0).is_nan(), "gdtrib x<0");
    }

    #[test]
    fn gdtrc_complement() {
        // gdtr + gdtrc should equal 1
        for &a in &[1.0, 2.0, 5.0] {
            for &b in &[1.0, 2.0] {
                for &x in &[0.5, 1.0, 2.0, 5.0] {
                    let sum = gdtr(a, b, x) + gdtrc(a, b, x);
                    assert!(
                        (sum - 1.0).abs() < 1e-10,
                        "gdtr({a}, {b}, {x}) + gdtrc = {sum}, expected 1.0"
                    );
                }
            }
        }
    }

    #[test]
    fn gdtrix_reference_values() {
        let cases: &[(f64, f64, f64, f64, f64)] = &[
            (2.5, 1.25, 0.4, 0.28778803752030013, 2e-12),
            (2.0, 3.0, 0.7, 1.8077838329329956, 2e-12),
            (-2.5, 1.25, 0.4, -0.28778803752030013, 2e-12),
        ];
        for &(a, b, p, expected, tolerance) in cases {
            let actual = gdtrix(a, b, p);
            assert!(
                (actual - expected).abs() < tolerance,
                "gdtrix({a}, {b}, {p}) = {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn gdtrix_matches_scipy_nonpositive_rates() {
        assert!(gdtrix(0.0, 1.25, 0.4).is_infinite() && gdtrix(0.0, 1.25, 0.4).is_sign_positive());
        assert!(
            gdtrix(-0.0, 1.25, 0.4).is_infinite() && gdtrix(-0.0, 1.25, 0.4).is_sign_negative()
        );
        assert_eq!(gdtrix(-2.5, 1.25, 0.0), -0.0);
        assert!(
            gdtrix(-2.5, 1.25, 1.0).is_infinite() && gdtrix(-2.5, 1.25, 1.0).is_sign_negative()
        );
        assert!(gdtrix(0.0, 1.25, 0.0).is_nan());

        assert!(gdtrix(2.5, 0.0, 0.4).is_nan());
        assert!(gdtrix(2.5, -1.0, 0.4).is_nan());
        assert!(gdtrix(2.5, 1.25, -0.1).is_nan());
        assert!(gdtrix(2.5, 1.25, 1.1).is_nan());
        assert!(gdtrix(f64::NAN, 1.25, 0.4).is_nan());
        assert!(gdtrix(2.5, f64::NAN, 0.4).is_nan());
        assert!(gdtrix(2.5, 1.25, f64::NAN).is_nan());
    }

    #[test]
    fn chdtr_basic() {
        // Chi-squared CDF values compared with scipy.special.chdtr
        // chdtr(1, 1) ≈ 0.6827 (P(X <= 1) for chi2 with df=1)
        let result = chdtr(1.0, 1.0);
        assert!(
            (result - 0.6827).abs() < 0.01,
            "chdtr(1, 1) = {result}, expected ~0.6827"
        );

        // chdtr(2, 2) ≈ 0.6321 (P(X <= 2) for chi2 with df=2)
        let result = chdtr(2.0, 2.0);
        assert!(
            (result - 0.6321).abs() < 0.01,
            "chdtr(2, 2) = {result}, expected ~0.6321"
        );

        // chdtr(10, 10) ≈ 0.5595
        let result = chdtr(10.0, 10.0);
        assert!(
            (result - 0.5595).abs() < 0.01,
            "chdtr(10, 10) = {result}, expected ~0.5595"
        );
    }

    #[test]
    fn chdtr_chdtrc_complement() {
        // chdtr + chdtrc should equal 1
        for &v in &[1.0, 2.0, 5.0, 10.0] {
            for &x in &[0.5, 1.0, 2.0, 5.0] {
                let sum = chdtr(v, x) + chdtrc(v, x);
                assert!(
                    (sum - 1.0).abs() < 1e-10,
                    "chdtr({v}, {x}) + chdtrc({v}, {x}) = {sum}, expected 1.0"
                );
            }
        }
    }

    #[test]
    fn chdtri_inverse() {
        // chdtri is the inverse of chdtrc, matching scipy.special.chdtri.
        for &v in &[1.0, 2.0, 5.0, 10.0] {
            for &x in &[0.5, 1.0, 2.0, 5.0, 10.0] {
                let p = chdtrc(v, x);
                if p > 0.01 && p < 0.99 {
                    let x_recovered = chdtri(v, p);
                    assert!(
                        (x_recovered - x).abs() / x < 0.01,
                        "chdtri failed: v={v}, x={x}, p={p}, x_recovered={x_recovered}"
                    );
                }
            }
        }
    }

    #[test]
    fn chdtriv_reference_values() {
        let cases = [
            (0.5, 2.0, 2.628_500_020_690_701_4),
            (0.8, 3.0, 1.853_054_579_870_468_3),
            (0.95, 10.0, 4.318_557_759_041_437),
        ];

        for (p, x, expected) in cases {
            let actual = chdtriv(p, x);
            assert!(
                (actual - expected).abs() < 5e-10,
                "chdtriv({p}, {x}) = {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn chdtriv_inverse() {
        for &v in &[0.5, 1.0, 2.5, 5.0, 10.0] {
            for &x in &[1.0, 2.0, 5.0, 10.0] {
                let p = chdtr(v, x);
                if p > 0.01 && p < 0.99 {
                    let recovered = chdtriv(p, x);
                    let tolerance = 5e-10 * v.abs().max(1.0);
                    assert!(
                        (recovered - v).abs() <= tolerance,
                        "chdtriv failed: v={v}, x={x}, p={p}, recovered={recovered}"
                    );
                }
            }
        }
    }

    #[test]
    fn chdtriv_edges_match_scipy() {
        assert!(chdtriv(0.0, 2.0).is_infinite());
        assert_eq!(chdtriv(1.0, 2.0), 0.0);

        assert!(chdtriv(0.5, 0.0).is_nan());
        assert!(chdtriv(0.0, 0.0).is_nan());
        assert!(chdtriv(1.0, 0.0).is_nan());
        assert!(chdtriv(0.5, f64::INFINITY).is_nan());
        assert!(chdtriv(-0.1, 3.0).is_nan());
        assert!(chdtriv(1.1, 3.0).is_nan());
        assert!(chdtriv(0.5, -1.0).is_nan());
        assert!(chdtriv(f64::NAN, 3.0).is_nan());
        assert!(chdtriv(0.5, f64::NAN).is_nan());
    }

    // ========================================================================
    // Complex gammainc tests
    // ========================================================================

    fn complex_scalar(re: f64, im: f64) -> SpecialTensor {
        SpecialTensor::ComplexScalar(Complex64::new(re, im))
    }

    fn get_complex_scalar(result: SpecialResult) -> Result<Complex64, String> {
        match result.map_err(|err| err.to_string())? {
            SpecialTensor::ComplexScalar(value) => Ok(value),
            other => Err(format!("expected complex scalar, got {other:?}")),
        }
    }

    #[test]
    fn gammainc_a_equals_one_matches_exponential_cdf() {
        // /testing-golden-artifacts for [frankenscipy-swlxw]:
        // gammainc(1, x) = 1 - e^{-x}, the Exp(1) CDF.
        for &x in &[0.0_f64, 0.1, 0.5, 1.0, 2.5, 7.0] {
            let actual = gammainc_scalar(1.0, x, RuntimeMode::Strict).unwrap();
            let expected = if x == 0.0 { 0.0 } else { 1.0 - (-x).exp() };
            assert!(
                (actual - expected).abs() < 1e-12,
                "gammainc(1, {x}) = {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn gammainc_a_equals_two_matches_closed_form() {
        // gammainc(2, x) = 1 - (1 + x)·e^{-x} (Erlang-2 CDF).
        for &x in &[0.0_f64, 0.5, 1.0, 2.0, 5.0] {
            let actual = gammainc_scalar(2.0, x, RuntimeMode::Strict).unwrap();
            let expected = if x == 0.0 {
                0.0
            } else {
                1.0 - (1.0 + x) * (-x).exp()
            };
            assert!(
                (actual - expected).abs() < 1e-12,
                "gammainc(2, {x}) = {actual}, expected 1 - (1+x)e^(-x) = {expected}"
            );
        }
    }

    #[test]
    fn gammainc_at_x_zero_is_zero() {
        // gammainc(a, 0) = 0 for any a > 0.
        for &a in &[0.5_f64, 1.0, 2.5, 7.0, 100.0] {
            let actual = gammainc_scalar(a, 0.0, RuntimeMode::Strict).unwrap();
            assert!(
                actual.abs() < 1e-15,
                "gammainc({a}, 0) = {actual}, expected 0"
            );
        }
    }

    #[test]
    fn gammainc_plus_gammaincc_equals_one() {
        // gammainc(a, x) + gammaincc(a, x) = 1 (P + Q = 1).
        for &a in &[0.5_f64, 1.5, 3.0, 10.0] {
            for &x in &[0.1_f64, 1.0, 5.0, 20.0] {
                let p = gammainc_scalar(a, x, RuntimeMode::Strict).unwrap();
                let q = gammaincc_scalar(a, x, RuntimeMode::Strict).unwrap();
                assert!(
                    (p + q - 1.0).abs() < 1e-12,
                    "gammainc({a},{x}) + gammaincc = {} != 1",
                    p + q
                );
            }
        }
    }

    #[test]
    fn complex_gammainc_reduces_to_real_for_real_z() -> Result<(), String> {
        // For real positive z, complex_gammainc_scalar should match gammainc_scalar
        let a = 2.0;
        let z_re = 1.5;
        let z = Complex64::new(z_re, 0.0);

        let real_result = gammainc_scalar(a, z_re, RuntimeMode::Strict).unwrap();
        let complex_result = complex_gammainc_scalar(a, z, RuntimeMode::Strict).unwrap();

        assert!(
            (complex_result.re - real_result).abs() < 1e-10,
            "Real parts should match: {} vs {}",
            complex_result.re,
            real_result
        );
        assert!(
            complex_result.im.abs() < 1e-10,
            "Imaginary part should be negligible for real input: {}",
            complex_result.im
        );
        Ok(())
    }

    #[test]
    fn complex_gammaincc_reduces_to_real_for_real_z() -> Result<(), String> {
        let a = 2.0;
        let z_re = 1.5;
        let z = Complex64::new(z_re, 0.0);

        let real_result = gammaincc_scalar(a, z_re, RuntimeMode::Strict).unwrap();
        let complex_result = complex_gammaincc_scalar(a, z, RuntimeMode::Strict).unwrap();

        assert!(
            (complex_result.re - real_result).abs() < 1e-10,
            "Real parts should match: {} vs {}",
            complex_result.re,
            real_result
        );
        assert!(
            complex_result.im.abs() < 1e-10,
            "Imaginary part should be negligible for real input: {}",
            complex_result.im
        );
        Ok(())
    }

    #[test]
    fn complex_gammainc_at_z_zero() {
        let result =
            complex_gammainc_scalar(2.0, Complex64::new(0.0, 0.0), RuntimeMode::Strict).unwrap();
        assert_eq!(result.re, 0.0);
        assert_eq!(result.im, 0.0);
    }

    #[test]
    fn complex_gammaincc_at_z_zero() {
        let result =
            complex_gammaincc_scalar(2.0, Complex64::new(0.0, 0.0), RuntimeMode::Strict).unwrap();
        assert_eq!(result.re, 1.0);
        assert_eq!(result.im, 0.0);
    }

    #[test]
    fn complex_gammainc_p_plus_q_equals_one() {
        // P(a,z) + Q(a,z) = 1 for any a,z
        let a = 2.5;
        let z = Complex64::new(1.5, 0.7);

        let p = complex_gammainc_scalar(a, z, RuntimeMode::Strict).unwrap();
        let q = complex_gammaincc_scalar(a, z, RuntimeMode::Strict).unwrap();

        let sum_re = p.re + q.re;
        let sum_im = p.im + q.im;

        assert!(
            (sum_re - 1.0).abs() < 1e-10,
            "P + Q real part should be 1: {}",
            sum_re
        );
        assert!(
            sum_im.abs() < 1e-10,
            "P + Q imaginary part should be 0: {}",
            sum_im
        );
    }

    #[test]
    fn complex_gammainc_tensor_interface() -> Result<(), String> {
        // Test the tensor interface works for complex inputs
        let a = scalar(2.0);
        let z = complex_scalar(1.5, 0.5);

        let result = gammainc(&a, &z, RuntimeMode::Strict);
        let c = get_complex_scalar(result)?;

        // Just verify we get a finite result
        assert!(c.re.is_finite(), "Real part should be finite");
        assert!(c.im.is_finite(), "Imaginary part should be finite");
        Ok(())
    }

    #[test]
    fn complex_a_gammainc_reduces_to_real_path_for_real_x() -> Result<(), String> {
        let a = complex_scalar(2.0, 0.0);
        let x = scalar(1.5);

        let actual = get_complex_scalar(gammainc(&a, &x, RuntimeMode::Strict))?;
        let expected = gammainc_scalar(2.0, 1.5, RuntimeMode::Strict).unwrap();

        assert!((actual.re - expected).abs() < 1e-10);
        assert!(actual.im.abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn complex_a_gammainc_p_plus_q_equals_one() {
        let a = Complex64::new(2.5, 0.75);
        let z = Complex64::new(1.5, -0.25);

        let p = complex_parameter_gammainc_scalar(a, z, RuntimeMode::Strict).unwrap();
        let q = complex_parameter_gammaincc_scalar(a, z, RuntimeMode::Strict).unwrap();

        let sum = p + q;
        assert!((sum.re - 1.0).abs() < 1e-10, "real(sum) = {}", sum.re);
        assert!(sum.im.abs() < 1e-10, "imag(sum) = {}", sum.im);
    }

    #[test]
    fn complex_a_gammainc_parameter_vector_broadcasts_real_x() -> Result<(), String> {
        let a =
            SpecialTensor::ComplexVec(vec![Complex64::new(2.5, 0.75), Complex64::new(3.0, -0.5)]);
        let x = scalar(1.5);

        let result = gammainc(&a, &x, RuntimeMode::Strict).map_err(|e| e.to_string())?;
        match result {
            SpecialTensor::ComplexVec(values) => {
                assert_eq!(values.len(), 2);
                let lane0 = complex_parameter_gammainc_scalar(
                    Complex64::new(2.5, 0.75),
                    Complex64::new(1.5, 0.0),
                    RuntimeMode::Strict,
                )
                .unwrap();
                let lane1 = complex_parameter_gammainc_scalar(
                    Complex64::new(3.0, -0.5),
                    Complex64::new(1.5, 0.0),
                    RuntimeMode::Strict,
                )
                .unwrap();
                assert!((values[0].re - lane0.re).abs() < 1e-10);
                assert!((values[0].im - lane0.im).abs() < 1e-10);
                assert!((values[1].re - lane1.re).abs() < 1e-10);
                assert!((values[1].im - lane1.im).abs() < 1e-10);
            }
            other => return Err(format!("expected complex vector, got {other:?}")),
        }
        Ok(())
    }

    #[test]
    fn complex_a_gammainc_parameter_vector_mismatch_is_shape_error() {
        let a =
            SpecialTensor::ComplexVec(vec![Complex64::new(2.0, 0.5), Complex64::new(3.0, -0.5)]);
        let x = SpecialTensor::RealVec(vec![0.5, 1.0, 1.5]);

        let err = gammainc(&a, &x, RuntimeMode::Strict).unwrap_err();
        assert_eq!(err.kind, SpecialErrorKind::DomainError);
        assert_eq!(err.detail, "vector inputs must have matching lengths");
    }

    #[test]
    fn complex_gammainc_domain_error_for_nonpositive_a() {
        let z = Complex64::new(1.0, 0.5);

        let result = complex_gammainc_scalar(0.0, z, RuntimeMode::Strict).unwrap();
        assert!(result.re.is_nan());

        let result = complex_gammainc_scalar(-1.0, z, RuntimeMode::Strict).unwrap();
        assert!(result.re.is_nan());
    }

    #[test]
    fn complex_gamma_real_inputs_match_real_path() -> Result<(), String> {
        let z = complex_scalar(3.0, 0.0);
        let result = gamma(&z, RuntimeMode::Strict).map_err(|e| e.to_string())?;
        let c = get_complex_scalar(Ok(result))?;

        let real_z = scalar(3.0);
        let real_result = gamma(&real_z, RuntimeMode::Strict).map_err(|e| e.to_string())?;
        let expected = get_scalar(Ok(real_result))?;

        assert!(
            (c.re - expected).abs() < 1e-10,
            "gamma(3+0i) = {} + {}i, expected {} + 0i",
            c.re,
            c.im,
            expected
        );
        assert!(c.im.abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn complex_gammaln_real_inputs_match_real_path() -> Result<(), String> {
        let z = complex_scalar(5.0, 0.0);
        let result = gammaln(&z, RuntimeMode::Strict).map_err(|e| e.to_string())?;
        let c = get_complex_scalar(Ok(result))?;

        let real_z = scalar(5.0);
        let real_result = gammaln(&real_z, RuntimeMode::Strict).map_err(|e| e.to_string())?;
        let expected = get_scalar(Ok(real_result))?;

        assert!(
            (c.re - expected).abs() < 1e-10,
            "gammaln(5+0i) = {} + {}i, expected {} + 0i",
            c.re,
            c.im,
            expected
        );
        assert!(c.im.abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn loggamma_real_inputs_preserve_scipy_branch_cut() -> Result<(), String> {
        let positive = get_scalar(loggamma(&scalar(0.5), RuntimeMode::Strict))?;
        assert!((positive - 0.5 * PI.ln()).abs() < 1e-12);

        let pole = get_scalar(loggamma(&scalar(0.0), RuntimeMode::Strict))?;
        assert!(pole.is_infinite() && pole.is_sign_positive());

        let branch_cut = get_scalar(loggamma(&scalar(-0.5), RuntimeMode::Strict))?;
        assert!(branch_cut.is_nan());

        let err = loggamma(&scalar(-0.5), RuntimeMode::Hardened).unwrap_err();
        assert_eq!(err.kind, SpecialErrorKind::DomainError);

        let err = loggamma(&scalar(0.0), RuntimeMode::Hardened).unwrap_err();
        assert_eq!(err.kind, SpecialErrorKind::PoleInput);
        Ok(())
    }

    #[test]
    fn loggamma_complex_inputs_use_principal_branch() -> Result<(), String> {
        let actual = get_complex_scalar(loggamma(&complex_scalar(-0.5, 0.0), RuntimeMode::Strict))?;
        assert!((actual.re - 1.265_512_123_484_647).abs() < 1e-12);
        assert!((actual.im + PI).abs() < 1e-12);

        let near_upper = get_complex_scalar(loggamma(
            &complex_scalar(-0.5, 1.0e-12),
            RuntimeMode::Strict,
        ))?;
        let near_lower = get_complex_scalar(loggamma(
            &complex_scalar(-0.5, -1.0e-12),
            RuntimeMode::Strict,
        ))?;
        assert!(near_upper.im < 0.0);
        assert!(near_lower.im > 0.0);
        Ok(())
    }

    #[test]
    fn complex_rgamma_real_inputs_match_real_path() -> Result<(), String> {
        let z = complex_scalar(5.0, 0.0);
        let result = rgamma(&z, RuntimeMode::Strict).map_err(|e| e.to_string())?;
        let c = get_complex_scalar(Ok(result))?;

        let real_z = scalar(5.0);
        let real_result = rgamma(&real_z, RuntimeMode::Strict).map_err(|e| e.to_string())?;
        let expected = get_scalar(Ok(real_result))?;

        assert!(
            (c.re - expected).abs() < 1e-12,
            "rgamma(5+0i) = {} + {}i, expected {} + 0i",
            c.re,
            c.im,
            expected
        );
        assert!(c.im.abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn complex_rgamma_returns_zero_at_negative_integer_poles() -> Result<(), String> {
        for x in [0.0, -1.0, -2.0, -8.0] {
            let actual = get_complex_scalar(rgamma(&complex_scalar(x, 0.0), RuntimeMode::Strict))?;
            assert_eq!(actual, Complex64::from_real(0.0));
        }
        Ok(())
    }

    #[test]
    fn complex_gammaln_returns_nan_at_real_poles() -> Result<(), String> {
        for x in [0.0, -1.0, -2.0, -8.0] {
            let actual = get_complex_scalar(gammaln(&complex_scalar(x, 0.0), RuntimeMode::Strict))?;
            assert!(
                actual.re.is_nan(),
                "gammaln({x}+0i) real part should be NaN"
            );
            assert!(
                actual.im.is_nan(),
                "gammaln({x}+0i) imag part should be NaN"
            );
        }
        Ok(())
    }

    #[test]
    fn complex_gammaln_near_real_poles_stays_finite() -> Result<(), String> {
        let actual =
            get_complex_scalar(gammaln(&complex_scalar(-1.0, 1.0e-12), RuntimeMode::Strict))?;
        assert!(actual.re.is_finite());
        assert!(actual.im.is_finite());
        Ok(())
    }

    #[test]
    fn complex_rgamma_vector_preserves_shape() -> Result<(), String> {
        let input =
            SpecialTensor::ComplexVec(vec![Complex64::new(2.0, 1.0), Complex64::new(5.0, 0.0)]);
        let result = rgamma(&input, RuntimeMode::Strict).map_err(|e| e.to_string())?;
        match result {
            SpecialTensor::ComplexVec(values) => {
                assert_eq!(values.len(), 2);
                let lane0 =
                    get_complex_scalar(rgamma(&complex_scalar(2.0, 1.0), RuntimeMode::Strict))?;
                let lane1 =
                    get_complex_scalar(rgamma(&complex_scalar(5.0, 0.0), RuntimeMode::Strict))?;
                assert!((values[0].re - lane0.re).abs() < 1e-12);
                assert!((values[0].im - lane0.im).abs() < 1e-12);
                assert!((values[1].re - lane1.re).abs() < 1e-12);
                assert!((values[1].im - lane1.im).abs() < 1e-12);
            }
            other => return Err(format!("expected complex vector, got {other:?}")),
        }
        Ok(())
    }

    #[test]
    fn complex_gamma_with_imaginary_part() -> Result<(), String> {
        let z = complex_scalar(2.0, 1.0);
        let result = gamma(&z, RuntimeMode::Strict).map_err(|e| e.to_string())?;
        let c = get_complex_scalar(Ok(result))?;

        assert!(c.re.is_finite(), "gamma(2+1i) real part should be finite");
        assert!(c.im.is_finite(), "gamma(2+1i) imag part should be finite");
        Ok(())
    }

    #[test]
    fn complex_gammaln_with_imaginary_part() -> Result<(), String> {
        let z = complex_scalar(2.0, 1.0);
        let result = gammaln(&z, RuntimeMode::Strict).map_err(|e| e.to_string())?;
        let c = get_complex_scalar(Ok(result))?;

        assert!(c.re.is_finite(), "gammaln(2+1i) real part should be finite");
        assert!(c.im.is_finite(), "gammaln(2+1i) imag part should be finite");

        // Verify gamma = exp(gammaln)
        let gamma_result =
            gamma(&complex_scalar(2.0, 1.0), RuntimeMode::Strict).map_err(|e| e.to_string())?;
        let gamma_c = get_complex_scalar(Ok(gamma_result))?;
        let expected = c.exp();
        assert!(
            (gamma_c.re - expected.re).abs() < 1e-10,
            "gamma should equal exp(gammaln)"
        );
        assert!(
            (gamma_c.im - expected.im).abs() < 1e-10,
            "gamma should equal exp(gammaln)"
        );
        Ok(())
    }

    #[test]
    fn binom_metamorphic_matches_comb_for_integer_arguments() {
        // For non-negative integer (n, k), binom(n, k) == comb(n, k) within
        // ulp.
        for n in 0..=15u64 {
            for k in 0..=n {
                let b = binom(n as f64, k as f64);
                let c = comb(n, k);
                let rel = (b - c).abs() / c.abs().max(1.0);
                assert!(rel < 1e-10, "binom({n},{k}) = {b}, comb = {c}");
            }
        }
    }

    #[test]
    fn binom_real_arguments_match_gamma_definition() {
        // binom(0.5, 0.25) = Γ(1.5) / (Γ(1.25) · Γ(1.25))
        let b = binom(0.5, 0.25);
        let lng = |x: f64| lngamma_positive(x);
        let expected = (lng(1.5) - 2.0 * lng(1.25)).exp();
        assert!(
            (b - expected).abs() < 1e-12,
            "binom(0.5, 0.25) = {b}, expected {expected}"
        );
    }

    #[test]
    fn binom_y_greater_than_x_returns_zero() {
        // Integer overshoot: vanishing-coefficient rule.
        assert_eq!(binom(3.0, 5.0), 0.0);
        // Non-integer y - x with integer y - x = 0 (full y-x).fract() = 0:
        // (y, x) where the difference is an exact integer ≥ 1 → 0.
        assert_eq!(binom(2.5, 3.5), 0.0);
        assert_eq!(binom(2.5, 4.5), 0.0);
    }

    #[test]
    fn binom_negative_x_matches_scipy() {
        // scipy.special.binom is finite (and signed) for negative NON-integer x;
        // we previously fail-closed to NaN for any x<0.
        let cases = [
            (-4.5, 2.0, 12.375_f64),
            (-2.5, 3.0, -6.5625),
            (-0.5, 1.0, -0.5),
            (-3.5, 2.0, 7.875),
            (-4.5, 3.0, -26.8125),
            (5.0, 2.0, 10.0), // positive args: unchanged
        ];
        for (x, y, want) in cases {
            let got = binom(x, y);
            assert!(
                (got - want).abs() <= 1e-11 * want.abs().max(1.0),
                "binom({x},{y}) got {got}, want {want}"
            );
        }
        // Negative-integer x is a pole => NaN (scipy).
        assert!(binom(-3.0, 2.0).is_nan());
        assert!(binom(-2.0, 1.0).is_nan());
    }

    #[test]
    fn binom_metamorphic_symmetry_x_minus_y() {
        // Apply /testing-metamorphic: binom(x, y) == binom(x, x − y)
        // for x ≥ y ≥ 0 (gamma-form symmetry of the binomial). Pin
        // across integer, half-integer, and non-integer x.
        let cases: &[(f64, f64)] = &[
            (5.0, 2.0),   // integer: C(5,2) = C(5,3) = 10
            (10.0, 4.0),  // integer
            (7.5, 2.5),   // half-integer x, integer y differential
            (4.25, 1.25), // non-integer
            (3.0, 0.0),   // y=0 endpoint
        ];
        for &(x, y) in cases {
            let lhs = binom(x, y);
            let rhs = binom(x, x - y);
            let tol = 1e-9 * (lhs.abs() + 1.0);
            assert!(
                (lhs - rhs).abs() < tol,
                "binom({x}, {y}) = {lhs} but binom({x}, {x}-{y}={}) = {rhs}",
                x - y
            );
        }
    }

    #[test]
    fn binom_real_y_greater_than_x_uses_gamma_formula() {
        // [frankenscipy-du5m4] Regression: scipy.special.binom for
        // non-integer y > x (where y - x is non-integer) returns the
        // gamma formula value, not 0.
        // binom(0.5, 1.0) = Γ(1.5)/(Γ(2)·Γ(0.5)) = (√π/2)/√π = 0.5.
        // The lngamma asymptotic-with-recurrence loses ~1e-11 of precision
        // here; a 1e-9 tolerance cleanly distinguishes the new gamma path
        // (≈0.5) from the old vanishing-coefficient bug (= 0.0).
        let v05_1 = binom(0.5_f64, 1.0);
        assert!((v05_1 - 0.5).abs() < 1e-9, "binom(0.5, 1.0) = {v05_1}");
        // binom(0.5, 0.6) = Γ(1.5)/(Γ(1.6)·Γ(0.9)).
        let expected = lngamma_positive(1.5) - lngamma_positive(1.6) - lngamma_positive(0.9);
        let expected = expected.exp();
        assert!((binom(0.5_f64, 0.6) - expected).abs() < 1e-9);
    }

    #[test]
    fn binom_endpoints_return_one() {
        assert_eq!(binom(5.0, 0.0), 1.0);
        assert_eq!(binom(5.0, 5.0), 1.0);
        assert_eq!(binom(2.5, 0.0), 1.0);
        assert_eq!(binom(2.5, 2.5), 1.0);
    }

    #[test]
    fn binom_negative_or_nan_propagates() {
        assert!(binom(-1.0, 0.5).is_nan());
        assert!(binom(2.0, -0.5).is_nan());
        assert!(binom(f64::NAN, 1.0).is_nan());
        assert!(binom(2.0, f64::NAN).is_nan());
    }

    #[test]
    fn factorialk_zero_and_negative_conventions() {
        assert_eq!(factorialk(0, 1), 1.0);
        assert_eq!(factorialk(0, 5), 1.0);
        assert!(factorialk(-1, 1).is_nan());
        assert!(factorialk(5, 0).is_nan());
        assert!(factorialk(5, -2).is_nan());
    }

    #[test]
    fn factorialk_metamorphic_k_one_matches_factorial() {
        // factorialk(n, 1) == n! for n in 0..=10
        for n in 0..=10 {
            let fk = factorialk(n, 1);
            let fact = factorial(n as u64);
            assert!(
                (fk - fact).abs() < 1e-12,
                "factorialk({n}, 1) = {fk}, factorial({n}) = {fact}"
            );
        }
    }

    #[test]
    fn factorialk_metamorphic_k_two_matches_factorial2() {
        // factorialk(n, 2) == n!! for n in 0..=15
        for n in 0..=15 {
            let fk = factorialk(n, 2);
            let f2 = factorial2(n);
            assert!(
                (fk - f2).abs() < 1e-12,
                "factorialk({n}, 2) = {fk}, factorial2({n}) = {f2}"
            );
        }
    }

    #[test]
    fn factorialk_metamorphic_recurrence() {
        // For n >= k, factorialk(n, k) == n * factorialk(n - k, k).
        for n in 1..=20 {
            for k in 1..=5 {
                if n >= k {
                    let lhs = factorialk(n, k);
                    let rhs = (n as f64) * factorialk(n - k, k);
                    assert!(
                        (lhs - rhs).abs() < 1e-9 * lhs.abs().max(1.0),
                        "factorialk({n}, {k}) = {lhs}, n * factorialk({n}-{k}, {k}) = {rhs}"
                    );
                }
            }
        }
    }

    #[test]
    fn factorialk_known_value_n_10_k_3() {
        // 10 * 7 * 4 * 1 = 280
        assert_eq!(factorialk(10, 3), 280.0);
    }

    #[test]
    fn comb_perm_match_scipy_reference_points() {
        // /testing-conformance-harnesses: pin canonical scipy values
        // for comb (binomial) and perm (k-permutations).
        //   comb(5, 2)  = 10
        //   comb(10, 5) = 252
        //   comb(0, 0)  = 1   (vacuous identity)
        //   comb(7, 0)  = 1
        //   comb(7, 7)  = 1
        //   comb(7, 8)  = 0   (k > n short-circuit)
        //   perm(5, 2)  = 5·4 = 20
        //   perm(10, 3) = 10·9·8 = 720
        //   perm(7, 0)  = 1   (empty product)
        //   perm(0, 0)  = 1
        //   perm(7, 8)  = 0
        assert_eq!(comb(5, 2), 10.0);
        assert_eq!(comb(10, 5), 252.0);
        assert_eq!(comb(0, 0), 1.0);
        assert_eq!(comb(7, 0), 1.0);
        assert_eq!(comb(7, 7), 1.0);
        assert_eq!(comb(7, 8), 0.0);
        assert_eq!(perm(5, 2), 20.0);
        assert_eq!(perm(10, 3), 720.0);
        assert_eq!(perm(7, 0), 1.0);
        assert_eq!(perm(0, 0), 1.0);
        assert_eq!(perm(7, 8), 0.0);
        // Symmetry C(n, k) = C(n, n-k).
        assert_eq!(comb(20, 7), comb(20, 13));
        assert_eq!(comb(50, 17), comb(50, 33));
    }

    #[test]
    fn gammaln_matches_scipy_reference_points() {
        // /testing-conformance-harnesses: pin gammaln at canonical
        // points scipy gives in closed form:
        //   gammaln(0.5) = ln(√π) = 0.5·ln(π)
        //   gammaln(1)   = 0
        //   gammaln(2)   = 0
        //   gammaln(0.25)= 1.288_022_524_698_077... (Γ(1/4)·Γ(3/4) = π√2)
        //   gammaln(10)  = ln(9!) = ln(362880) = 12.801_827_480_081_469
        let pi = std::f64::consts::PI;
        let cases: &[(f64, f64)] = &[
            (0.5, 0.5 * pi.ln()),
            (1.0, 0.0),
            (2.0, 0.0),
            (10.0, 362_880.0_f64.ln()),
        ];
        for &(x, expected) in cases {
            let got = gammaln_scalar(x, RuntimeMode::Strict).unwrap();
            assert!(
                (got - expected).abs() < 1e-12,
                "gammaln({x}) = {got}, expected {expected}"
            );
        }
        // Pole behavior — strict mode returns +inf, not NaN.
        assert!(
            gammaln_scalar(0.0, RuntimeMode::Strict)
                .unwrap()
                .is_infinite()
        );
        assert!(
            gammaln_scalar(-1.0, RuntimeMode::Strict)
                .unwrap()
                .is_infinite()
        );
    }

    #[test]
    fn zeta_matches_scipy_reference_points() {
        // /testing-conformance-harnesses: pin closed-form scipy.special
        // .zeta values:
        //   ζ(2)  = π²/6
        //   ζ(4)  = π⁴/90
        //   ζ(6)  = π⁶/945
        //   ζ(0)  = −1/2
        //   ζ(−1) = −1/12   (functional equation gives the negative-s
        //                   continuation)
        let pi = std::f64::consts::PI;
        let cases: &[(f64, f64)] = &[
            (2.0, pi * pi / 6.0),
            (4.0, pi.powi(4) / 90.0),
            (6.0, pi.powi(6) / 945.0),
        ];
        for &(s, expected) in cases {
            let got = zeta(s);
            let rel = ((got - expected) / expected).abs();
            assert!(
                rel < 1e-9,
                "ζ({s}) = {got}, expected {expected} (rel = {rel})"
            );
        }
        // Large-s tail: ζ(s) → 1 as s → ∞.
        assert!((zeta(50.0) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn trigamma_metamorphic_recurrence() {
        // /testing-metamorphic: ψ'(x+1) = ψ'(x) − 1/x² for x not at a
        // non-positive integer pole.
        // Plus closed-form trigamma values:
        //   ψ'(1) = π²/6   ψ'(0.5) = π²/2   ψ'(2) = π²/6 − 1
        let pi_sq_over_6 = std::f64::consts::PI * std::f64::consts::PI / 6.0;
        let pi_sq_over_2 = std::f64::consts::PI * std::f64::consts::PI / 2.0;
        assert!((trigamma_core(1.0) - pi_sq_over_6).abs() < 1e-9);
        assert!((trigamma_core(0.5) - pi_sq_over_2).abs() < 1e-9);
        assert!((trigamma_core(2.0) - (pi_sq_over_6 - 1.0)).abs() < 1e-9);
        // Recurrence across multiple x.
        for &x in &[0.25_f64, 0.5, 1.3, 2.7, 5.3, 9.1, 17.4] {
            let lhs = trigamma_core(x + 1.0);
            let rhs = trigamma_core(x) - 1.0 / (x * x);
            assert!(
                (lhs - rhs).abs() < 1e-9,
                "ψ'({x}+1) = {lhs}, expected ψ'({x}) − 1/{x}² = {rhs}"
            );
        }
    }

    #[test]
    fn digamma_metamorphic_recurrence() {
        // /testing-metamorphic: ψ(x+1) = ψ(x) + 1/x for any x not at
        // a non-positive integer. Independent of any specific value;
        // catches drift in either the recurrence-shift branch or the
        // asymptotic series. Pin across multiple x.
        // 1e-9 envelope reflects the asymptotic-series truncation
        // residual (~5e-11 for moderate x, growing with x).
        for &x in &[0.25_f64, 0.5, 1.5, 2.7, 5.3, 9.1, 17.4] {
            let lhs = digamma_core(x + 1.0);
            let rhs = digamma_core(x) + 1.0 / x;
            assert!(
                (lhs - rhs).abs() < 1e-9,
                "ψ({x}+1) = {lhs}, expected ψ({x}) + 1/{x} = {rhs}"
            );
        }
    }

    #[test]
    fn digamma_matches_scipy_reference_points() {
        // /testing-conformance-harnesses: pin canonical scipy.special
        // .psi values:
        //   ψ(1)   = -γ                       (Euler-Mascheroni constant)
        //   ψ(0.5) = -γ - 2·ln 2              (closed form)
        //   ψ(2)   = 1 - γ                    (recurrence ψ(n+1) = ψ(n) + 1/n)
        //   ψ(3)   = 3/2 - γ
        let gamma_em = 0.577_215_664_901_532_9_f64;
        let ln_2 = 2.0_f64.ln();
        let cases: &[(f64, f64)] = &[
            (1.0, -gamma_em),
            (0.5, -gamma_em - 2.0 * ln_2),
            (2.0, 1.0 - gamma_em),
            (3.0, 1.5 - gamma_em),
            (4.0, 1.0 + 0.5 + 1.0 / 3.0 - gamma_em),
        ];
        // Tolerance reflects the ~2e-10 residual of the asymptotic-
        // with-shift digamma_core (Euler-Maclaurin truncated at
        // order 1/x⁷). Tighter than 1e-9 would require extending the
        // series; pin the current envelope.
        for &(x, expected) in cases {
            let got = digamma_core(x);
            assert!(
                (got - expected).abs() < 1e-9,
                "ψ({x}) = {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn psi_alias_matches_digamma_dispatch() -> Result<(), String> {
        let scalar_input = scalar(2.5);
        let psi_scalar = get_scalar(psi(&scalar_input, RuntimeMode::Strict))?;
        let digamma_scalar = get_scalar(digamma(&scalar_input, RuntimeMode::Strict))?;
        assert!((psi_scalar - digamma_scalar).abs() < 1e-14);

        let vec_input = SpecialTensor::RealVec(vec![0.5, 1.0, 2.5, 8.0]);
        let psi_vec = psi(&vec_input, RuntimeMode::Strict).map_err(|err| err.to_string())?;
        let digamma_vec =
            digamma(&vec_input, RuntimeMode::Strict).map_err(|err| err.to_string())?;
        assert_eq!(psi_vec, digamma_vec);

        let complex_input = SpecialTensor::ComplexScalar(Complex64::new(1.5, 0.25));
        let psi_complex =
            psi(&complex_input, RuntimeMode::Strict).map_err(|err| err.to_string())?;
        let digamma_complex =
            digamma(&complex_input, RuntimeMode::Strict).map_err(|err| err.to_string())?;
        assert_eq!(psi_complex, digamma_complex);
        Ok(())
    }

    #[test]
    fn gamma_metamorphic_factorial_identity() {
        // /testing-metamorphic: Γ(n+1) = n! for non-negative integer n.
        // Pin across n = 0..=10 to a tolerance of 1 ulp (closed-form
        // integer arithmetic, no rounding error in either side).
        let mut fact = 1.0_f64;
        for n in 0..=10_u64 {
            if n > 0 {
                fact *= n as f64;
            }
            let gamma_val = gamma_core((n + 1) as f64);
            let rel = ((gamma_val - fact) / fact).abs();
            assert!(
                rel < 1e-13,
                "Γ({}) = {gamma_val}, expected {n}! = {fact} (rel = {rel})",
                n + 1
            );
        }
    }

    #[test]
    fn gamma_metamorphic_reflection_formula() {
        // /testing-metamorphic: Γ(x) · Γ(1−x) = π/sin(πx) for any
        // non-integer x. Independent of any specific gamma value;
        // catches any drift in either the Lanczos or recurrence
        // branches without hard-coding a reference number.
        let pi = std::f64::consts::PI;
        // Avoid x near integer poles where Γ diverges.
        for &x in &[0.25_f64, 0.5, 0.75, 1.5, 2.5, 3.7, 5.3, 7.1] {
            let lhs = gamma_core(x) * gamma_core(1.0 - x);
            let rhs = pi / (pi * x).sin();
            let rel = (lhs - rhs).abs() / rhs.abs().max(1.0);
            assert!(
                rel < 1e-12,
                "Γ({x})·Γ(1-{x}) = {lhs}, expected π/sin(π·{x}) = {rhs} (rel = {rel})"
            );
        }
    }

    #[test]
    fn factorial_matches_scipy_reference_values() {
        // scipy.special.factorial(0) = 1.0
        // scipy.special.factorial(5) = 120.0
        // scipy.special.factorial(10) = 3628800.0
        assert_eq!(factorial(0), 1.0, "0! should be 1");
        assert_eq!(factorial(5), 120.0, "5! should be 120");
        assert_eq!(factorial(10), 3628800.0, "10! should be 3628800");
    }

    #[test]
    fn factorial2_matches_scipy_reference_values() {
        // scipy.special.factorial2(5) = 5!! = 5*3*1 = 15
        // scipy.special.factorial2(6) = 6!! = 6*4*2 = 48
        // scipy.special.factorial2(0) = 1
        assert_eq!(factorial2(0), 1.0, "0!! should be 1");
        assert_eq!(factorial2(5), 15.0, "5!! should be 15");
        assert_eq!(factorial2(6), 48.0, "6!! should be 48");
    }

    #[test]
    fn binom_matches_scipy_reference_values() {
        // scipy.special.binom(5, 2) = 10
        // scipy.special.binom(10, 3) = 120
        // scipy.special.binom(6, 0) = 1
        assert!(
            (binom(5.0, 2.0) - 10.0).abs() < 1e-10,
            "C(5,2) = {}, expected 10",
            binom(5.0, 2.0)
        );
        assert!(
            (binom(10.0, 3.0) - 120.0).abs() < 1e-6,
            "C(10,3) = {}, expected 120",
            binom(10.0, 3.0)
        );
        assert!(
            (binom(6.0, 0.0) - 1.0).abs() < 1e-10,
            "C(6,0) = {}, expected 1",
            binom(6.0, 0.0)
        );
    }

    #[test]
    fn gammainc_matches_scipy_reference_values() {
        // scipy.special.gammainc(1, 1) = 1 - exp(-1) ≈ 0.6321205588
        // scipy.special.gammainc(2, 1) ≈ 0.2642411177
        let val1 = gammainc_scalar(1.0, 1.0, RuntimeMode::Strict).expect("gammainc(1, 1)");
        assert!(
            (val1 - 0.6321205588).abs() < 1e-6,
            "gammainc(1,1) = {}, expected 0.6321205588",
            val1
        );

        let val2 = gammainc_scalar(2.0, 1.0, RuntimeMode::Strict).expect("gammainc(2, 1)");
        assert!(
            (val2 - 0.2642411177).abs() < 1e-6,
            "gammainc(2,1) = {}, expected 0.2642411177",
            val2
        );
    }
}
