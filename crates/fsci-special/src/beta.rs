#![forbid(unsafe_code)]

use fsci_runtime::RuntimeMode;

use crate::gamma;
use crate::types::{
    Complex64, DispatchPlan, DispatchStep, KernelRegime, SpecialError, SpecialErrorKind,
    SpecialResult, SpecialTensor, not_yet_implemented, record_special_trace,
};

pub const BETA_DISPATCH_PLAN: &[DispatchPlan] = &[
    DispatchPlan {
        function: "beta",
        steps: &[
            DispatchStep {
                regime: KernelRegime::BackendDelegate,
                when: "use gamma/gammaln composition in stable space",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large-parameter regime uses logspace stabilization",
            },
        ],
        notes: "Symmetry beta(a,b)=beta(b,a) must hold in strict mode and hardened mode.",
    },
    DispatchPlan {
        function: "betaln",
        steps: &[
            DispatchStep {
                regime: KernelRegime::BackendDelegate,
                when: "direct logspace composition",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "a+b sufficiently large",
            },
        ],
        notes: "Primary path for underflow-prone beta regions.",
    },
    DispatchPlan {
        function: "betainc",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "x in lower-tail region",
            },
            DispatchStep {
                regime: KernelRegime::ContinuedFraction,
                when: "x in upper-tail region",
            },
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "parameter shifts for stability",
            },
        ],
        notes: "Strict mode preserves SciPy endpoint behavior at x=0 and x=1.",
    },
    DispatchPlan {
        function: "betaincc",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Reflection,
                when: "evaluate the complementary tail as I_(1-x)(b, a)",
            },
            DispatchStep {
                regime: KernelRegime::ContinuedFraction,
                when: "delegates to the stable betainc tail path after swapping parameters",
            },
        ],
        notes: "Strict mode preserves SciPy endpoint behavior at x=0 and x=1.",
    },
];

const DISTRIBUTION_INVERSE_ITERS: usize = 160;
const DISTRIBUTION_INVERSE_UPPER_SENTINEL: f64 = 1.0e100;

pub fn beta(a: &SpecialTensor, b: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    beta_dispatch(a, b, mode)
}

pub fn betaln(a: &SpecialTensor, b: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    betaln_dispatch(a, b, mode)
}

fn beta_dispatch(a: &SpecialTensor, b: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    if matches!(
        (a, b),
        (
            SpecialTensor::RealScalar(_) | SpecialTensor::RealVec(_),
            SpecialTensor::RealScalar(_) | SpecialTensor::RealVec(_)
        )
    ) {
        map_real_binary("beta", a, b, mode, |av, bv| beta_scalar(av, bv, mode))
    } else {
        map_complex_binary("beta", a, b, mode, complex_beta_scalar)
    }
}

fn betaln_dispatch(a: &SpecialTensor, b: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    if matches!(
        (a, b),
        (
            SpecialTensor::RealScalar(_) | SpecialTensor::RealVec(_),
            SpecialTensor::RealScalar(_) | SpecialTensor::RealVec(_)
        )
    ) {
        map_real_binary("betaln", a, b, mode, |av, bv| betaln_scalar(av, bv, mode))
    } else {
        map_complex_binary("betaln", a, b, mode, complex_betaln_scalar)
    }
}

fn map_complex_binary<F>(
    function: &'static str,
    a: &SpecialTensor,
    b: &SpecialTensor,
    mode: RuntimeMode,
    kernel: F,
) -> SpecialResult
where
    F: Fn(Complex64, Complex64) -> Complex64,
{
    match (a, b) {
        (SpecialTensor::ComplexScalar(lhs), SpecialTensor::ComplexScalar(rhs)) => {
            Ok(SpecialTensor::ComplexScalar(kernel(*lhs, *rhs)))
        }
        (SpecialTensor::ComplexVec(lhs), SpecialTensor::ComplexScalar(rhs)) => Ok(
            SpecialTensor::ComplexVec(lhs.iter().map(|value| kernel(*value, *rhs)).collect()),
        ),
        (SpecialTensor::ComplexScalar(lhs), SpecialTensor::ComplexVec(rhs)) => Ok(
            SpecialTensor::ComplexVec(rhs.iter().map(|value| kernel(*lhs, *value)).collect()),
        ),
        (SpecialTensor::ComplexVec(lhs), SpecialTensor::ComplexVec(rhs)) => {
            if lhs.len() != rhs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            Ok(SpecialTensor::ComplexVec(
                lhs.iter()
                    .zip(rhs.iter())
                    .map(|(left, right)| kernel(*left, *right))
                    .collect(),
            ))
        }
        (SpecialTensor::RealScalar(lhs), SpecialTensor::ComplexScalar(rhs)) => Ok(
            SpecialTensor::ComplexScalar(kernel(Complex64::from_real(*lhs), *rhs)),
        ),
        (SpecialTensor::ComplexScalar(lhs), SpecialTensor::RealScalar(rhs)) => Ok(
            SpecialTensor::ComplexScalar(kernel(*lhs, Complex64::from_real(*rhs))),
        ),
        (SpecialTensor::RealVec(lhs), SpecialTensor::ComplexScalar(rhs)) => {
            Ok(SpecialTensor::ComplexVec(
                lhs.iter()
                    .map(|value| kernel(Complex64::from_real(*value), *rhs))
                    .collect(),
            ))
        }
        (SpecialTensor::ComplexScalar(lhs), SpecialTensor::RealVec(rhs)) => {
            Ok(SpecialTensor::ComplexVec(
                rhs.iter()
                    .map(|value| kernel(*lhs, Complex64::from_real(*value)))
                    .collect(),
            ))
        }
        (SpecialTensor::RealScalar(lhs), SpecialTensor::ComplexVec(rhs)) => {
            Ok(SpecialTensor::ComplexVec(
                rhs.iter()
                    .map(|value| kernel(Complex64::from_real(*lhs), *value))
                    .collect(),
            ))
        }
        (SpecialTensor::ComplexVec(lhs), SpecialTensor::RealScalar(rhs)) => {
            Ok(SpecialTensor::ComplexVec(
                lhs.iter()
                    .map(|value| kernel(*value, Complex64::from_real(*rhs)))
                    .collect(),
            ))
        }
        (SpecialTensor::RealVec(lhs), SpecialTensor::ComplexVec(rhs)) => {
            if lhs.len() != rhs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            Ok(SpecialTensor::ComplexVec(
                lhs.iter()
                    .zip(rhs.iter())
                    .map(|(left, right)| kernel(Complex64::from_real(*left), *right))
                    .collect(),
            ))
        }
        (SpecialTensor::ComplexVec(lhs), SpecialTensor::RealVec(rhs)) => {
            if lhs.len() != rhs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            Ok(SpecialTensor::ComplexVec(
                lhs.iter()
                    .zip(rhs.iter())
                    .map(|(left, right)| kernel(*left, Complex64::from_real(*right)))
                    .collect(),
            ))
        }
        _ => Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "unsupported tensor combination",
        }),
    }
}

pub fn betainc(
    a: &SpecialTensor,
    b: &SpecialTensor,
    x: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    betainc_dispatch(a, b, x, mode)
}

fn betainc_dispatch(
    a: &SpecialTensor,
    b: &SpecialTensor,
    x: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    match (a, b, x) {
        (
            SpecialTensor::RealScalar(av),
            SpecialTensor::RealScalar(bv),
            SpecialTensor::RealScalar(xv),
        ) => betainc_scalar(*av, *bv, *xv, mode).map(SpecialTensor::RealScalar),
        (
            SpecialTensor::ComplexScalar(av),
            SpecialTensor::ComplexScalar(bv),
            SpecialTensor::ComplexScalar(xv),
        ) => Ok(SpecialTensor::ComplexScalar(complex_betainc_scalar(
            *av, *bv, *xv,
        ))),
        (
            SpecialTensor::RealScalar(av),
            SpecialTensor::RealScalar(bv),
            SpecialTensor::ComplexScalar(xv),
        ) => Ok(SpecialTensor::ComplexScalar(complex_betainc_scalar(
            Complex64::from_real(*av),
            Complex64::from_real(*bv),
            *xv,
        ))),
        (
            SpecialTensor::ComplexScalar(av),
            SpecialTensor::ComplexScalar(bv),
            SpecialTensor::RealScalar(xv),
        ) => Ok(SpecialTensor::ComplexScalar(complex_betainc_scalar(
            *av,
            *bv,
            Complex64::from_real(*xv),
        ))),
        _ => map_real_ternary("betainc", a, b, x, mode, |av, bv, xv| {
            betainc_scalar(av, bv, xv, mode)
        }),
    }
}

pub fn betaincc(
    a: &SpecialTensor,
    b: &SpecialTensor,
    x: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    map_real_ternary("betaincc", a, b, x, mode, |av, bv, xv| {
        betaincc_scalar(av, bv, xv, mode)
    })
}

pub fn betainccinv(
    a: &SpecialTensor,
    b: &SpecialTensor,
    y: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    map_real_ternary("betainccinv", a, b, y, mode, |av, bv, yv| {
        Ok(betainccinv_scalar(av, bv, yv))
    })
}

/// Beta distribution CDF.
///
/// Matches `scipy.special.btdtr(a, b, x)`.
#[must_use]
pub fn btdtr(a: f64, b: f64, x: f64) -> f64 {
    betainc_scalar(a, b, x, RuntimeMode::Strict).unwrap_or(f64::NAN)
}

/// Beta distribution survival function.
///
/// Returns P(X > x) = 1 - btdtr(a, b, x).
///
/// Matches `scipy.special.btdtrc(a, b, x)`.
#[must_use]
pub fn btdtrc(a: f64, b: f64, x: f64) -> f64 {
    if a.is_nan() || b.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 1.0;
    }
    if x >= 1.0 {
        return 0.0;
    }
    // btdtrc(a, b, x) = 1 - betainc(a, b, x) = betainc(b, a, 1-x)
    betainc_scalar(b, a, 1.0 - x, RuntimeMode::Strict).unwrap_or(f64::NAN)
}

/// Inverse beta distribution CDF.
///
/// Matches `scipy.special.btdtri(a, b, y)`.
#[must_use]
pub fn btdtri(a: f64, b: f64, y: f64) -> f64 {
    if a.is_nan() || b.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if a <= 0.0 || b <= 0.0 || !(0.0..=1.0).contains(&y) {
        return f64::NAN;
    }
    if y == 0.0 {
        return 0.0;
    }
    if y == 1.0 {
        return 1.0;
    }

    invert_monotone_unit_interval(|x| btdtr(a, b, x), y)
}

/// Inverse beta distribution CDF with respect to shape parameter `a`.
///
/// Returns `a` such that `betainc(a, b, x) = p`.
///
/// Matches `scipy.special.btdtria(p, b, x)`.
#[must_use]
pub fn btdtria(p: f64, b: f64, x: f64) -> f64 {
    if p.is_nan() || b.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if b <= 0.0 || !(0.0..=1.0).contains(&p) || x <= 0.0 {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::INFINITY;
    }
    if p == 1.0 {
        return f64::MIN_POSITIVE;
    }
    if x >= 1.0 || !x.is_finite() {
        return f64::NAN;
    }
    // I_x(a, b) = 1 - I_{1-x}(b, a); use the smaller tail when p is near 1.
    if p > 0.5 {
        return btdtrib(b, 1.0 - p, 1.0 - x);
    }

    invert_monotone_positive(|a| btdtr(a, b, x), p, false)
}

/// Inverse beta distribution CDF with respect to shape parameter `b`.
///
/// Returns `b` such that `betainc(a, b, x) = p`.
///
/// Matches `scipy.special.btdtrib(a, p, x)`.
#[must_use]
pub fn btdtrib(a: f64, p: f64, x: f64) -> f64 {
    if a.is_nan() || p.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if a <= 0.0 || !(0.0..=1.0).contains(&p) || x <= 0.0 {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::MIN_POSITIVE;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    if x >= 1.0 || !x.is_finite() {
        return f64::NAN;
    }
    // I_x(a, b) = 1 - I_{1-x}(b, a); use the smaller tail when p is near 1.
    if p > 0.5 {
        return btdtria(1.0 - p, a, 1.0 - x);
    }

    invert_monotone_positive(|b| btdtr(a, b, x), p, true)
}

/// F-distribution CDF.
///
/// Matches `scipy.special.fdtr(dfn, dfd, x)`.
#[must_use]
pub fn fdtr(dfn: f64, dfd: f64, x: f64) -> f64 {
    if dfn.is_nan() || dfd.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if dfn <= 0.0 || dfd <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    let z = dfn * x / (dfn * x + dfd);
    btdtr(0.5 * dfn, 0.5 * dfd, z)
}

/// F-distribution survival function.
///
/// Returns P(X > x) = 1 - fdtr(dfn, dfd, x).
///
/// Matches `scipy.special.fdtrc(dfn, dfd, x)`.
#[must_use]
pub fn fdtrc(dfn: f64, dfd: f64, x: f64) -> f64 {
    if dfn.is_nan() || dfd.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if dfn <= 0.0 || dfd <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 1.0;
    }
    // fdtrc(dfn, dfd, x) = 1 - fdtr(dfn, dfd, x)
    // = 1 - btdtr(dfn/2, dfd/2, z) = btdtr(dfd/2, dfn/2, 1-z)
    let z = dfn * x / (dfn * x + dfd);
    btdtr(0.5 * dfd, 0.5 * dfn, 1.0 - z)
}

/// Inverse F-distribution CDF.
///
/// Matches `scipy.special.fdtri(dfn, dfd, y)`.
#[must_use]
pub fn fdtri(dfn: f64, dfd: f64, y: f64) -> f64 {
    if dfn.is_nan() || dfd.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if dfn <= 0.0 || dfd <= 0.0 || !(0.0..=1.0).contains(&y) {
        return f64::NAN;
    }
    if y == 0.0 {
        return 0.0;
    }
    if y == 1.0 {
        return f64::INFINITY;
    }

    let z = btdtri(0.5 * dfn, 0.5 * dfd, y);
    dfd * z / (dfn * (1.0 - z))
}

/// Inverse F-distribution CDF with respect to denominator degrees of freedom.
///
/// Returns dfd such that P(F <= x) = p for an F distribution with dfn and dfd
/// degrees of freedom.
///
/// Matches `scipy.special.fdtridfd(dfn, p, x)`, including CDFlib sentinel
/// values for no-solution boundary cases.
#[must_use]
pub fn fdtridfd(dfn: f64, p: f64, x: f64) -> f64 {
    const LOWER_SENTINEL: f64 = 1.0e-100;
    const UPPER_SENTINEL: f64 = 1.0e100;

    if dfn.is_nan() || p.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if dfn <= 0.0 || !(0.0..=1.0).contains(&p) || x < 0.0 {
        return f64::NAN;
    }
    if p == 1.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return if p == 0.0 {
            5.0
        } else if p <= 0.5 {
            LOWER_SENTINEL
        } else {
            UPPER_SENTINEL
        };
    }
    if x.is_infinite() {
        return UPPER_SENTINEL;
    }
    if p == 0.0 {
        return LOWER_SENTINEL;
    }

    let upper_limit = gamma::chdtr(dfn, dfn * x);
    if !upper_limit.is_finite() {
        return f64::NAN;
    }
    if p >= upper_limit {
        return UPPER_SENTINEL;
    }

    let mut hi = 1.0;
    while fdtr(dfn, hi, x) < p {
        hi *= 2.0;
        if hi >= UPPER_SENTINEL {
            return UPPER_SENTINEL;
        }
    }

    let mut lo = 0.0;
    for _ in 0..240 {
        let mid = lo + (hi - lo) * 0.5;
        let value = fdtr(dfn, mid, x);
        if !value.is_finite() || value < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let dfd = lo + (hi - lo) * 0.5;
    if dfd == 0.0 { LOWER_SENTINEL } else { dfd }
}

/// Student's t distribution CDF.
///
/// Returns P(T <= t) where T follows a Student's t distribution
/// with v degrees of freedom.
///
/// Matches `scipy.special.stdtr(v, t)`.
#[must_use]
pub fn stdtr(v: f64, t: f64) -> f64 {
    if v.is_nan() || t.is_nan() {
        return f64::NAN;
    }
    if v <= 0.0 {
        return f64::NAN;
    }

    // Use the relation with incomplete beta:
    // For t >= 0: stdtr(v, t) = 1 - 0.5 * I(v/(v+t²); v/2, 1/2)
    // For t < 0:  stdtr(v, t) = 0.5 * I(v/(v+t²); v/2, 1/2)
    let x = v / (v + t * t);
    let half_beta = 0.5 * btdtr(0.5 * v, 0.5, x);

    if t >= 0.0 { 1.0 - half_beta } else { half_beta }
}

/// Student's t distribution survival function.
///
/// Returns P(T > t) = 1 - stdtr(v, t) where T follows a Student's t
/// distribution with v degrees of freedom.
///
/// Matches `scipy.special.stdtrc(v, t)`.
#[must_use]
pub fn stdtrc(v: f64, t: f64) -> f64 {
    if v.is_nan() || t.is_nan() {
        return f64::NAN;
    }
    if v <= 0.0 {
        return f64::NAN;
    }

    // Use symmetry: stdtrc(v, t) = 1 - stdtr(v, t) = stdtr(v, -t)
    let x = v / (v + t * t);
    let half_beta = 0.5 * btdtr(0.5 * v, 0.5, x);

    if t >= 0.0 { half_beta } else { 1.0 - half_beta }
}

/// Inverse Student's t distribution CDF.
///
/// Returns t such that P(T <= t) = p where T follows a Student's t
/// distribution with v degrees of freedom.
///
/// Matches `scipy.special.stdtrit(v, p)`.
#[must_use]
pub fn stdtrit(v: f64, p: f64) -> f64 {
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
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-15 {
        return 0.0;
    }

    // Use the inverse beta to find z = v/(v+t²)
    // For p > 0.5: z = btdtri(v/2, 1/2, 2*(1-p))
    // For p < 0.5: z = btdtri(v/2, 1/2, 2*p)
    let (z, sign) = if p > 0.5 {
        (btdtri(0.5 * v, 0.5, 2.0 * (1.0 - p)), 1.0)
    } else {
        (btdtri(0.5 * v, 0.5, 2.0 * p), -1.0)
    };

    // z = v/(v+t²) => t² = v*(1-z)/z => t = sign * sqrt(v*(1-z)/z)
    if z <= 0.0 {
        return sign * f64::INFINITY;
    }
    if z >= 1.0 {
        return 0.0;
    }

    sign * (v * (1.0 - z) / z).sqrt()
}

/// Inverse Student's t distribution CDF with respect to degrees of freedom.
///
/// Returns v such that P(T <= t) = p where T follows a Student's t
/// distribution with v degrees of freedom.
///
/// Matches `scipy.special.stdtridf(p, t)`, including CDFlib sentinel
/// values for no-solution boundary cases.
#[must_use]
pub fn stdtridf(p: f64, t: f64) -> f64 {
    const LOWER_SENTINEL: f64 = -1.0e100;
    const UPPER_SENTINEL: f64 = 1.0e10;
    const MIN_DF_SENTINEL: f64 = 5.0e-51;

    if p.is_nan() || t.is_nan() {
        return f64::NAN;
    }
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if t == 0.0 {
        return if p == 0.5 { 5.0 } else { UPPER_SENTINEL };
    }
    if t.is_infinite() {
        return if t.is_sign_positive() {
            if p == 1.0 {
                5.0
            } else if p > 0.5 {
                LOWER_SENTINEL
            } else {
                UPPER_SENTINEL
            }
        } else if p == 0.0 {
            5.0
        } else if p < 0.5 {
            LOWER_SENTINEL
        } else {
            UPPER_SENTINEL
        };
    }
    if p == 0.5 {
        return MIN_DF_SENTINEL;
    }
    if t < 0.0 {
        return stdtridf(1.0 - p, -t);
    }
    if p < 0.5 {
        return LOWER_SENTINEL;
    }

    let upper_value = stdtr(UPPER_SENTINEL, t);
    if !upper_value.is_finite() {
        return f64::NAN;
    }
    if p >= upper_value {
        return UPPER_SENTINEL;
    }

    let mut lo = 0.0;
    let mut hi = UPPER_SENTINEL;

    for _ in 0..240 {
        let mid = lo + (hi - lo) * 0.5;
        let value = stdtr(mid, t);
        if !value.is_finite() || value < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let df = lo + (hi - lo) * 0.5;
    if df == 0.0 { MIN_DF_SENTINEL } else { df }
}

/// Binomial distribution CDF.
///
/// Returns P(X <= k) where X follows a binomial distribution
/// with n trials and success probability p.
///
/// Matches `scipy.special.bdtr(k, n, p)`.
#[must_use]
pub fn bdtr(k: f64, n: f64, p: f64) -> f64 {
    if k.is_nan() || n.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    if n < 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if k < 0.0 {
        return 0.0;
    }
    if k >= n {
        return 1.0;
    }

    // bdtr(k, n, p) = I(1-p; n-k, k+1) = betainc(n-k, k+1, 1-p)
    // Or equivalently: 1 - betainc(k+1, n-k, p)
    btdtr(n - k, k + 1.0, 1.0 - p)
}

/// Binomial distribution survival function.
///
/// Returns P(X > k) where X follows a binomial distribution
/// with n trials and success probability p.
///
/// Matches `scipy.special.bdtrc(k, n, p)`.
#[must_use]
pub fn bdtrc(k: f64, n: f64, p: f64) -> f64 {
    if k.is_nan() || n.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    if n < 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if k < 0.0 {
        return 1.0;
    }
    if k >= n {
        return 0.0;
    }

    // bdtrc(k, n, p) = betainc(k+1, n-k, p)
    btdtr(k + 1.0, n - k, p)
}

/// Inverse binomial distribution CDF.
///
/// Returns p such that P(X <= k) = y where X follows a binomial distribution
/// with n trials.
///
/// Matches `scipy.special.bdtri(k, n, y)`.
#[must_use]
pub fn bdtri(k: f64, n: f64, y: f64) -> f64 {
    if k.is_nan() || n.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if n < 0.0 || !(0.0..=1.0).contains(&y) || k < 0.0 || k > n {
        return f64::NAN;
    }
    if y == 0.0 {
        return 1.0;
    }
    if y == 1.0 {
        return 0.0;
    }
    if k >= n {
        return 0.0;
    }

    // bdtr(k, n, p) = betainc(n-k, k+1, 1-p) = y
    // So 1-p = btdtri(n-k, k+1, y)
    // Thus p = 1 - btdtri(n-k, k+1, y)
    1.0 - btdtri(n - k, k + 1.0, y)
}

/// Inverse binomial CDF with respect to `k`.
///
/// Returns `k` such that `bdtr(k, n, p) = y`.
/// Matches `scipy.special.bdtrik(y, n, p)` for the regular interior domain.
#[must_use]
pub fn bdtrik(y: f64, n: f64, p: f64) -> f64 {
    if y.is_nan() || n.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    if n <= 0.0 || !(0.0..=1.0).contains(&y) || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NAN;
    }
    if y == 0.0 {
        return 0.0;
    }
    if y == 1.0 || p == 1.0 {
        return n;
    }

    bisect_increasing(0.0, n, y, |k| bdtr(k, n, p))
}

/// Inverse binomial CDF with respect to `n`.
///
/// Returns `n` such that `bdtr(k, n, p) = y`.
/// Matches `scipy.special.bdtrin(k, y, p)` for the regular interior domain.
#[must_use]
pub fn bdtrin(k: f64, y: f64, p: f64) -> f64 {
    if k.is_nan() || y.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    if k < 0.0 || !(0.0..=1.0).contains(&y) || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NAN;
    }
    if y == 0.0 {
        return DISTRIBUTION_INVERSE_UPPER_SENTINEL;
    }
    if y == 1.0 || p == 1.0 {
        return k;
    }

    let lo = k;
    let mut hi = (k + 1.0).max(1.0);
    while bdtr(k, hi, p) > y {
        hi *= 2.0;
        if hi >= DISTRIBUTION_INVERSE_UPPER_SENTINEL {
            return DISTRIBUTION_INVERSE_UPPER_SENTINEL;
        }
    }

    bisect_decreasing(lo, hi, y, |n| bdtr(k, n, p))
}

/// Negative binomial distribution CDF.
///
/// Returns P(X <= k) where X is the number of failures before n successes
/// in a sequence of Bernoulli trials with success probability p.
///
/// Matches `scipy.special.nbdtr(k, n, p)`.
#[must_use]
pub fn nbdtr(k: f64, n: f64, p: f64) -> f64 {
    if k.is_nan() || n.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    if n <= 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return 1.0;
    }
    if k < 0.0 {
        return 0.0;
    }

    // nbdtr(k, n, p) = I_p(n, k+1) = betainc(n, k+1, p)
    btdtr(n, k + 1.0, p)
}

/// Negative binomial distribution survival function.
///
/// Returns P(X > k) where X is the number of failures before n successes.
///
/// Matches `scipy.special.nbdtrc(k, n, p)`.
#[must_use]
pub fn nbdtrc(k: f64, n: f64, p: f64) -> f64 {
    if k.is_nan() || n.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    if n <= 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return 1.0;
    }
    if p == 1.0 {
        return 0.0;
    }
    if k < 0.0 {
        return 1.0;
    }

    // nbdtrc(k, n, p) = 1 - betainc(n, k+1, p) = betainc(k+1, n, 1-p)
    btdtr(k + 1.0, n, 1.0 - p)
}

/// Inverse negative binomial distribution CDF.
///
/// Returns p such that nbdtr(k, n, p) = y.
///
/// Matches `scipy.special.nbdtri(k, n, y)`.
#[must_use]
pub fn nbdtri(k: f64, n: f64, y: f64) -> f64 {
    if k.is_nan() || n.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if n <= 0.0 || !(0.0..=1.0).contains(&y) || k < 0.0 {
        return f64::NAN;
    }
    if y == 0.0 {
        return 0.0;
    }
    if y == 1.0 {
        return 1.0;
    }

    // nbdtr(k, n, p) = betainc(n, k+1, p) = y
    // So p = btdtri(n, k+1, y)
    btdtri(n, k + 1.0, y)
}

/// Inverse negative-binomial CDF with respect to `k`.
///
/// Returns `k` such that `nbdtr(k, n, p) = y`.
/// Matches `scipy.special.nbdtrik(y, n, p)` for the regular interior domain.
#[must_use]
pub fn nbdtrik(y: f64, n: f64, p: f64) -> f64 {
    if y.is_nan() || n.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    if n <= 0.0 || !(0.0..=1.0).contains(&y) || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if y == 0.0 || p == 0.0 {
        return 0.0;
    }
    if y == 1.0 {
        return f64::NAN;
    }
    if p == 1.0 {
        return DISTRIBUTION_INVERSE_UPPER_SENTINEL;
    }

    let mut hi = 1.0;
    while nbdtr(hi, n, p) < y {
        hi *= 2.0;
        if hi >= DISTRIBUTION_INVERSE_UPPER_SENTINEL {
            return DISTRIBUTION_INVERSE_UPPER_SENTINEL;
        }
    }

    bisect_increasing(0.0, hi, y, |k| nbdtr(k, n, p))
}

/// Inverse negative-binomial CDF with respect to `n`.
///
/// Returns `n` such that `nbdtr(k, n, p) = y`.
/// Matches `scipy.special.nbdtrin(k, y, p)` for the regular interior domain.
#[must_use]
pub fn nbdtrin(k: f64, y: f64, p: f64) -> f64 {
    if k.is_nan() || y.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    if k < 0.0 || !(0.0..=1.0).contains(&y) || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if y == 0.0 {
        return DISTRIBUTION_INVERSE_UPPER_SENTINEL;
    }
    if y == 1.0 {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return DISTRIBUTION_INVERSE_UPPER_SENTINEL;
    }

    let mut hi = 1.0;
    while nbdtr(k, hi, p) > y {
        hi *= 2.0;
        if hi >= DISTRIBUTION_INVERSE_UPPER_SENTINEL {
            return DISTRIBUTION_INVERSE_UPPER_SENTINEL;
        }
    }

    bisect_decreasing(0.0, hi, y, |n| nbdtr(k, n, p))
}

fn bisect_increasing<F>(mut lo: f64, mut hi: f64, target: f64, f: F) -> f64
where
    F: Fn(f64) -> f64,
{
    for _ in 0..DISTRIBUTION_INVERSE_ITERS {
        let mid = lo + (hi - lo) * 0.5;
        let value = f(mid);
        if !value.is_finite() || value < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo + (hi - lo) * 0.5
}

fn bisect_decreasing<F>(mut lo: f64, mut hi: f64, target: f64, f: F) -> f64
where
    F: Fn(f64) -> f64,
{
    for _ in 0..DISTRIBUTION_INVERSE_ITERS {
        let mid = lo + (hi - lo) * 0.5;
        let value = f(mid);
        if !value.is_finite() || value > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo + (hi - lo) * 0.5
}

fn map_real_binary<F>(
    function: &'static str,
    a: &SpecialTensor,
    b: &SpecialTensor,
    mode: RuntimeMode,
    kernel: F,
) -> SpecialResult
where
    F: Fn(f64, f64) -> Result<f64, SpecialError>,
{
    match (a, b) {
        (SpecialTensor::RealScalar(lhs), SpecialTensor::RealScalar(rhs)) => {
            kernel(*lhs, *rhs).map(SpecialTensor::RealScalar)
        }
        (SpecialTensor::RealVec(lhs), SpecialTensor::RealScalar(rhs)) => lhs
            .iter()
            .copied()
            .map(|value| kernel(value, *rhs))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        (SpecialTensor::RealScalar(lhs), SpecialTensor::RealVec(rhs)) => rhs
            .iter()
            .copied()
            .map(|value| kernel(*lhs, value))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        (SpecialTensor::RealVec(lhs), SpecialTensor::RealVec(rhs)) => {
            if lhs.len() != rhs.len() {
                record_special_trace(
                    function,
                    mode,
                    "domain_error",
                    format!("lhs_len={},rhs_len={}", lhs.len(), rhs.len()),
                    "fail_closed",
                    "vector inputs must have matching lengths",
                    false,
                );
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            lhs.iter()
                .copied()
                .zip(rhs.iter().copied())
                .map(|(left, right)| kernel(left, right))
                .collect::<Result<Vec<_>, _>>()
                .map(SpecialTensor::RealVec)
        }
        (SpecialTensor::ComplexScalar(_), _)
        | (SpecialTensor::ComplexVec(_), _)
        | (_, SpecialTensor::ComplexScalar(_))
        | (_, SpecialTensor::ComplexVec(_)) => {
            not_yet_implemented(function, mode, "complex-valued path pending")
        }
        _ => {
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

fn map_real_ternary<F>(
    function: &'static str,
    a: &SpecialTensor,
    b: &SpecialTensor,
    c: &SpecialTensor,
    mode: RuntimeMode,
    kernel: F,
) -> SpecialResult
where
    F: Fn(f64, f64, f64) -> Result<f64, SpecialError>,
{
    match (a, b, c) {
        (
            SpecialTensor::RealScalar(av),
            SpecialTensor::RealScalar(bv),
            SpecialTensor::RealScalar(cv),
        ) => kernel(*av, *bv, *cv).map(SpecialTensor::RealScalar),
        (
            SpecialTensor::RealVec(av),
            SpecialTensor::RealScalar(bv),
            SpecialTensor::RealScalar(cv),
        ) => av
            .iter()
            .copied()
            .map(|lhs| kernel(lhs, *bv, *cv))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        (
            SpecialTensor::RealScalar(av),
            SpecialTensor::RealVec(bv),
            SpecialTensor::RealScalar(cv),
        ) => bv
            .iter()
            .copied()
            .map(|rhs| kernel(*av, rhs, *cv))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        (
            SpecialTensor::RealScalar(av),
            SpecialTensor::RealScalar(bv),
            SpecialTensor::RealVec(cv),
        ) => cv
            .iter()
            .copied()
            .map(|xv| kernel(*av, *bv, xv))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        (SpecialTensor::ComplexScalar(_), _, _)
        | (SpecialTensor::ComplexVec(_), _, _)
        | (_, SpecialTensor::ComplexScalar(_), _)
        | (_, SpecialTensor::ComplexVec(_), _)
        | (_, _, SpecialTensor::ComplexScalar(_))
        | (_, _, SpecialTensor::ComplexVec(_)) => {
            not_yet_implemented(function, mode, "complex-valued path pending")
        }
        _ => {
            record_special_trace(
                function,
                mode,
                "domain_error",
                "unsupported_broadcast_pattern",
                "fail_closed",
                "unsupported broadcast pattern for ternary inputs",
                false,
            );
            Err(SpecialError {
                function,
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "unsupported broadcast pattern for ternary inputs",
            })
        }
    }
}

/// Sign of Γ(x) on the real line. Γ > 0 for x > 0; for x < 0 it alternates on
/// each interval between consecutive negative integers. (Nonpositive-integer
/// poles are carried by the ±inf log path, so the value here is harmless there.)
fn gamma_sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if (-x).ceil() as i64 % 2 == 0 {
        1.0
    } else {
        -1.0
    }
}

fn beta_scalar(a: f64, b: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    // Symmetry beta(a,b)=beta(b,a)
    let (a, b) = if a < b { (b, a) } else { (a, b) };

    let log_value = betaln_scalar(a, b, mode)?;
    // B(a,b) = Γ(a)Γ(b)/Γ(a+b) is signed: betaln gives ln|B|, so restore the sign
    // from the gamma factors (scipy.special.beta(-2.5,3)=-1.0667). For positive
    // a,b every gamma is positive => sign = +1 (unchanged).
    let sign = gamma_sign(a) * gamma_sign(b) * gamma_sign(a + b);
    const LN_MAX: f64 = 709.782_712_893_384;
    const LN_MIN: f64 = -745.133_219_101_941_1;

    if log_value > LN_MAX {
        if matches!(mode, RuntimeMode::Hardened) {
            record_special_trace(
                "beta",
                mode,
                "overflow_risk",
                format!("a={a},b={b},log_beta={log_value}"),
                "fail_closed",
                "beta overflow risk",
                true,
            );
            return Err(SpecialError {
                function: "beta",
                kind: SpecialErrorKind::OverflowRisk,
                mode,
                detail: "beta overflow risk",
            });
        }
        record_special_trace(
            "beta",
            mode,
            "overflow_risk",
            format!("a={a},b={b},log_beta={log_value}"),
            "returned_inf",
            "strict overflow fallback",
            true,
        );
        return Ok(sign * f64::INFINITY);
    }
    if log_value < LN_MIN {
        record_special_trace(
            "beta",
            mode,
            "underflow_risk",
            format!("a={a},b={b},log_beta={log_value}"),
            "returned_zero",
            "underflow-safe clamp to zero",
            true,
        );
        return Ok(sign * 0.0);
    }

    Ok(sign * log_value.exp())
}

pub fn betaln_scalar(a: f64, b: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if a.is_nan() || b.is_nan() {
        return Ok(f64::NAN);
    }
    if matches!(mode, RuntimeMode::Hardened) && (a <= 0.0 || b <= 0.0) {
        record_special_trace(
            "betaln",
            mode,
            "domain_error",
            format!("a={a},b={b}"),
            "fail_closed",
            "betaln principal domain requires positive parameters",
            false,
        );
        return Err(SpecialError {
            function: "betaln",
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "betaln principal domain requires positive parameters",
        });
    }
    // `betaln(a,b) = ln|B(a,b)| = gammaln(a) + gammaln(b) - gammaln(a+b)` is valid
    // for ALL real a, b, not just positives: SciPy returns finite values for
    // negative non-integer arguments (e.g. betaln(-2.5, 3.0) = 0.0645) and the
    // pole limits (+/-inf) elsewhere. `gammaln_scalar(_, Strict)` returns the
    // reflection-formula value for negative non-integers and +inf at nonpositive-
    // integer poles, so the sum reproduces SciPy across the real line. (Only the
    // rare both-nonpositive-integer pole-cancellation, e.g. betaln(-3, 2), is left
    // as NaN; SciPy resolves it via a finite gamma-ratio, see frankenscipy notes.)
    let lg_a = gammaln_scalar(a, RuntimeMode::Strict)?;
    let lg_b = gammaln_scalar(b, RuntimeMode::Strict)?;
    let lg_ab = gammaln_scalar(a + b, RuntimeMode::Strict)?;
    Ok(lg_a + lg_b - lg_ab)
}

pub fn complex_betaln_scalar(a: Complex64, b: Complex64) -> Complex64 {
    let lg_a = gamma::complex_gammaln(a);
    let lg_b = gamma::complex_gammaln(b);
    let lg_ab = gamma::complex_gammaln(a + b);
    lg_a + lg_b - lg_ab
}

pub fn complex_beta_scalar(a: Complex64, b: Complex64) -> Complex64 {
    complex_betaln_scalar(a, b).exp()
}

pub fn complex_betainc_scalar(a: Complex64, b: Complex64, x: Complex64) -> Complex64 {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);

    if x.re == 0.0 && x.im == 0.0 {
        return zero;
    }
    if x.re == 1.0 && x.im == 0.0 {
        return one;
    }

    let ln_beta = complex_betaln_scalar(a, b);
    let ln_front = a * x.ln() + b * (one - x).ln() - ln_beta;
    let front = ln_front.exp();

    let threshold = (a + one) / (a + b + Complex64::new(2.0, 0.0));
    if x.re < threshold.re {
        front * complex_betacf(a, b, x) / a
    } else {
        one - front * complex_betacf(b, a, one - x) / b
    }
}

fn complex_betacf(a: Complex64, b: Complex64, x: Complex64) -> Complex64 {
    const MAX_ITERS: usize = 200;
    const EPS: f64 = 3.0e-14;
    const MIN_NUM: f64 = 1.0e-300;

    let one = Complex64::new(1.0, 0.0);
    let qab = a + b;
    let qap = a + one;
    let qam = a - one;
    let mut c = one;
    let mut d = one - qab * x / qap;
    if d.abs() < MIN_NUM {
        d = Complex64::new(MIN_NUM, 0.0);
    }
    d = d.recip();
    let mut h = d;

    for m in 1..=MAX_ITERS {
        let m_f = Complex64::new(m as f64, 0.0);
        let m2 = m_f * Complex64::new(2.0, 0.0);
        let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = one + aa * d;
        if d.abs() < MIN_NUM {
            d = Complex64::new(MIN_NUM, 0.0);
        }
        c = one + aa / c;
        if c.abs() < MIN_NUM {
            c = Complex64::new(MIN_NUM, 0.0);
        }
        d = d.recip();
        h = h * d * c;

        let aa2 = (a + m_f) * (qab + m_f) * x * Complex64::new(-1.0, 0.0) / ((a + m2) * (qap + m2));
        d = one + aa2 * d;
        if d.abs() < MIN_NUM {
            d = Complex64::new(MIN_NUM, 0.0);
        }
        c = one + aa2 / c;
        if c.abs() < MIN_NUM {
            c = Complex64::new(MIN_NUM, 0.0);
        }
        d = d.recip();
        let delta = d * c;
        h = h * delta;
        if (delta - one).abs() <= EPS {
            break;
        }
    }

    h
}

pub fn betainc_scalar(a: f64, b: f64, x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if a.is_nan() || b.is_nan() || x.is_nan() {
        return Ok(f64::NAN);
    }
    if !(0.0..=1.0).contains(&x) {
        return match mode {
            RuntimeMode::Strict => {
                record_special_trace(
                    "betainc",
                    mode,
                    "domain_error",
                    format!("a={a},b={b},x={x}"),
                    "returned_nan",
                    "strict domain fallback",
                    false,
                );
                Ok(f64::NAN)
            }
            RuntimeMode::Hardened => {
                record_special_trace(
                    "betainc",
                    mode,
                    "domain_error",
                    format!("a={a},b={b},x={x}"),
                    "fail_closed",
                    "betainc domain requires x in [0, 1]",
                    false,
                );
                Err(SpecialError {
                    function: "betainc",
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "betainc domain requires x in [0, 1]",
                })
            }
        };
    }
    if x == 0.0 {
        return Ok(0.0);
    }
    if x == 1.0 {
        return Ok(1.0);
    }
    if a <= 0.0 || b <= 0.0 {
        return match mode {
            RuntimeMode::Strict => {
                record_special_trace(
                    "betainc",
                    mode,
                    "domain_error",
                    format!("a={a},b={b},x={x}"),
                    "returned_nan",
                    "strict domain fallback",
                    false,
                );
                Ok(f64::NAN)
            }
            RuntimeMode::Hardened => {
                record_special_trace(
                    "betainc",
                    mode,
                    "domain_error",
                    format!("a={a},b={b},x={x}"),
                    "fail_closed",
                    "betainc requires positive shape parameters",
                    false,
                );
                Err(SpecialError {
                    function: "betainc",
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "betainc requires positive shape parameters",
                })
            }
        };
    }

    let ln_beta = betaln_scalar(a, b, RuntimeMode::Strict)?;
    let front = (a * x.ln() + b * (1.0 - x).ln() - ln_beta).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        Ok(front * betacf(a, b, x) / a)
    } else {
        Ok(1.0 - front * betacf(b, a, 1.0 - x) / b)
    }
}

pub fn betaincc_scalar(a: f64, b: f64, x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if a.is_nan() || b.is_nan() || x.is_nan() {
        return Ok(f64::NAN);
    }
    if !(0.0..=1.0).contains(&x) {
        return match mode {
            RuntimeMode::Strict => {
                record_special_trace(
                    "betaincc",
                    mode,
                    "domain_error",
                    format!("a={a},b={b},x={x}"),
                    "returned_nan",
                    "strict domain fallback",
                    false,
                );
                Ok(f64::NAN)
            }
            RuntimeMode::Hardened => {
                record_special_trace(
                    "betaincc",
                    mode,
                    "domain_error",
                    format!("a={a},b={b},x={x}"),
                    "fail_closed",
                    "betaincc domain requires x in [0, 1]",
                    false,
                );
                Err(SpecialError {
                    function: "betaincc",
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "betaincc domain requires x in [0, 1]",
                })
            }
        };
    }
    if x == 0.0 {
        return Ok(1.0);
    }
    if x == 1.0 {
        return Ok(0.0);
    }
    if a <= 0.0 || b <= 0.0 {
        return match mode {
            RuntimeMode::Strict => {
                record_special_trace(
                    "betaincc",
                    mode,
                    "domain_error",
                    format!("a={a},b={b},x={x}"),
                    "returned_nan",
                    "strict domain fallback",
                    false,
                );
                Ok(f64::NAN)
            }
            RuntimeMode::Hardened => {
                record_special_trace(
                    "betaincc",
                    mode,
                    "domain_error",
                    format!("a={a},b={b},x={x}"),
                    "fail_closed",
                    "betaincc requires positive shape parameters",
                    false,
                );
                Err(SpecialError {
                    function: "betaincc",
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "betaincc requires positive shape parameters",
                })
            }
        };
    }

    betainc_scalar(b, a, 1.0 - x, mode)
}

#[must_use]
pub fn betainccinv_scalar(a: f64, b: f64, y: f64) -> f64 {
    if a.is_nan() || b.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if a <= 0.0 || b <= 0.0 || !(0.0..=1.0).contains(&y) {
        return f64::NAN;
    }
    if y == 0.0 {
        return 1.0;
    }
    if y == 1.0 {
        return 0.0;
    }

    1.0 - crate::convenience::betaincinv_scalar(b, a, y)
}

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAX_ITERS: usize = 200;
    const EPS: f64 = 3.0e-14;
    const MIN_NUM: f64 = 1.0e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < MIN_NUM {
        d = MIN_NUM;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=MAX_ITERS {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;
        let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < MIN_NUM {
            d = MIN_NUM;
        }
        c = 1.0 + aa / c;
        if c.abs() < MIN_NUM {
            c = MIN_NUM;
        }
        d = 1.0 / d;
        h *= d * c;

        let aa2 = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa2 * d;
        if d.abs() < MIN_NUM {
            d = MIN_NUM;
        }
        c = 1.0 + aa2 / c;
        if c.abs() < MIN_NUM {
            c = MIN_NUM;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() <= EPS {
            break;
        }
    }

    h
}

fn invert_monotone_unit_interval(cdf: impl Fn(f64) -> f64, target: f64) -> f64 {
    let mut lo = 0.0;
    let mut hi = 1.0;

    for _ in 0..100 {
        let mid = lo + (hi - lo) * 0.5;
        let value = cdf(mid);
        if !value.is_finite() {
            hi = mid;
            continue;
        }
        if value < target {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo) < f64::EPSILON * 2.0 {
            break;
        }
    }

    lo + (hi - lo) * 0.5
}

fn invert_monotone_positive(cdf: impl Fn(f64) -> f64, target: f64, increasing: bool) -> f64 {
    let mut lo = f64::MIN_POSITIVE;
    let mut hi = 1.0;
    let mut hi_value = cdf(hi);

    if !hi_value.is_finite() {
        return f64::NAN;
    }

    if increasing {
        while hi_value < target {
            lo = hi;
            hi *= 2.0;
            if !hi.is_finite() {
                return f64::INFINITY;
            }
            hi_value = cdf(hi);
            if !hi_value.is_finite() {
                return f64::NAN;
            }
        }
    } else {
        while hi_value > target {
            lo = hi;
            hi *= 2.0;
            if !hi.is_finite() {
                return f64::INFINITY;
            }
            hi_value = cdf(hi);
            if !hi_value.is_finite() {
                return f64::NAN;
            }
        }
    }

    for _ in 0..180 {
        let mid = 0.5 * (lo + hi);
        let value = cdf(mid);
        if !value.is_finite() {
            return f64::NAN;
        }
        if increasing {
            if value < target {
                lo = mid;
            } else {
                hi = mid;
            }
        } else if value > target {
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

fn gammaln_scalar(value: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    let tensor = SpecialTensor::RealScalar(value);
    let result = gamma::gammaln(&tensor, mode)?;
    match result {
        SpecialTensor::RealScalar(v) => Ok(v),
        _ => Err(SpecialError {
            function: "betaln",
            kind: SpecialErrorKind::NotYetImplemented,
            mode,
            detail: "unexpected non-scalar gammaln output",
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type DistributionInverseCase = (fn(f64, f64, f64) -> f64, f64, f64, f64, f64);

    #[test]
    fn betaincc_complements_betainc_and_preserves_endpoints() {
        for &(a, b, x) in &[
            (0.5_f64, 0.5, 0.2),
            (2.0, 3.0, 0.25),
            (5.0, 2.0, 0.75),
            (10.0, 10.0, 0.5),
        ] {
            let lower = betainc_scalar(a, b, x, RuntimeMode::Strict).unwrap_or(f64::NAN);
            let upper = betaincc_scalar(a, b, x, RuntimeMode::Strict).unwrap_or(f64::NAN);
            assert!(
                (lower + upper - 1.0).abs() < 1.0e-12,
                "betainc + betaincc must sum to 1 for ({a}, {b}, {x})"
            );
        }

        assert_eq!(
            betaincc_scalar(2.0, 3.0, 0.0, RuntimeMode::Strict).unwrap_or(f64::NAN),
            1.0
        );
        assert_eq!(
            betaincc_scalar(2.0, 3.0, 1.0, RuntimeMode::Strict).unwrap_or(f64::NAN),
            0.0
        );
        assert!(betaincc_scalar(2.0, 3.0, -0.1, RuntimeMode::Strict).is_ok_and(f64::is_nan));
    }

    #[test]
    fn betainccinv_matches_scipy_reference_and_inverts_tail() {
        let reference = betainccinv_scalar(2.0, 3.0, 0.25);
        assert!(
            (reference - 0.543_678_285_419_080_3).abs() < 1.0e-12,
            "betainccinv(2, 3, 0.25) = {reference}"
        );

        for &y in &[0.001_f64, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999] {
            let x = betainccinv_scalar(2.0, 3.0, y);
            let tail = betaincc_scalar(2.0, 3.0, x, RuntimeMode::Strict).unwrap_or(f64::NAN);
            assert!(
                (tail - y).abs() < 1.0e-9,
                "betaincc(2, 3, betainccinv(..., {y})) = {tail}"
            );
        }

        assert_eq!(betainccinv_scalar(2.0, 3.0, 0.0), 1.0);
        assert_eq!(betainccinv_scalar(2.0, 3.0, 1.0), 0.0);
        assert!(betainccinv_scalar(2.0, 3.0, -0.1).is_nan());
        assert!(betainccinv_scalar(-1.0, 3.0, 0.5).is_nan());
    }

    #[test]
    fn betaincc_tensor_dispatch_broadcasts_real_vectors() {
        let result = betaincc(
            &SpecialTensor::RealScalar(2.0),
            &SpecialTensor::RealScalar(3.0),
            &SpecialTensor::RealVec(vec![0.0, 0.25, 0.5, 1.0]),
            RuntimeMode::Strict,
        )
        .expect("betaincc vector");
        let values = match result {
            SpecialTensor::RealVec(values) => values,
            _ => Vec::new(),
        };
        assert_eq!(values.len(), 4);
        assert_eq!(values[0], 1.0);
        assert!((values[1] - 0.738_281_25).abs() < 1.0e-12);
        assert!((values[2] - 0.312_5).abs() < 1.0e-12);
        assert_eq!(values[3], 0.0);
    }

    #[test]
    fn btdtr_closed_form_reference_values() {
        // /testing-golden-artifacts for [frankenscipy-1ulgv]:
        // btdtr (regularized incomplete beta I_x(a, b)) has multiple
        // analytic closed-form arms:
        //
        //   btdtr(a, b, 0) = 0
        //   btdtr(a, b, 1) = 1
        //   btdtr(1, 1, x) = x                    (uniform CDF)
        //   btdtr(a, 1, x) = x^a                  (power-law)
        //   btdtr(1, b, x) = 1 - (1 - x)^b
        //
        // Catches subtle sign or exponent errors in the underlying
        // incomplete-beta core.
        for &a in &[0.5_f64, 1.0, 2.5, 7.0] {
            for &b in &[0.5_f64, 1.0, 2.5, 7.0] {
                assert_eq!(btdtr(a, b, 0.0), 0.0, "btdtr({a}, {b}, 0) must be 0");
                assert!(
                    (btdtr(a, b, 1.0) - 1.0).abs() < 1e-12,
                    "btdtr({a}, {b}, 1) = {} != 1",
                    btdtr(a, b, 1.0)
                );
            }
        }

        for &x in &[0.05_f64, 0.25, 0.5, 0.75, 0.95] {
            assert!(
                (btdtr(1.0, 1.0, x) - x).abs() < 1e-12,
                "btdtr(1, 1, {x}) = {} != {x}",
                btdtr(1.0, 1.0, x)
            );
        }

        for &a in &[0.5_f64, 2.0, 5.0] {
            for &x in &[0.1_f64, 0.5, 0.9] {
                let expected = x.powf(a);
                assert!(
                    (btdtr(a, 1.0, x) - expected).abs() < 1e-12,
                    "btdtr({a}, 1, {x}) = {} != x^a = {expected}",
                    btdtr(a, 1.0, x)
                );
            }
        }

        for &b in &[0.5_f64, 2.0, 5.0] {
            for &x in &[0.1_f64, 0.5, 0.9] {
                let expected = 1.0 - (1.0 - x).powf(b);
                assert!(
                    (btdtr(1.0, b, x) - expected).abs() < 1e-12,
                    "btdtr(1, {b}, {x}) = {} != 1 - (1-x)^b = {expected}",
                    btdtr(1.0, b, x)
                );
            }
        }
    }

    #[test]
    fn btdtr_swap_symmetry_identity() {
        // I_x(a, b) = 1 - I_{1-x}(b, a). Catches swapped-argument
        // bugs in the regularized incomplete beta core that the
        // closed-form arms above wouldn't detect.
        for &a in &[0.5_f64, 1.5, 3.0] {
            for &b in &[0.5_f64, 1.5, 3.0] {
                for &x in &[0.1_f64, 0.4, 0.7] {
                    let lhs = btdtr(a, b, x);
                    let rhs = 1.0 - btdtr(b, a, 1.0 - x);
                    assert!(
                        (lhs - rhs).abs() < 1e-10,
                        "btdtr({a},{b},{x}) = {lhs}, but 1 - btdtr({b},{a},{}) = {rhs}",
                        1.0 - x
                    );
                }
            }
        }
    }

    #[test]
    fn btdtrc_complement() {
        // btdtr + btdtrc should equal 1
        for &a in &[0.5, 1.0, 2.0, 5.0] {
            for &b in &[0.5, 1.0, 2.0, 5.0] {
                for &x in &[0.1, 0.3, 0.5, 0.7, 0.9] {
                    let sum = btdtr(a, b, x) + btdtrc(a, b, x);
                    assert!(
                        (sum - 1.0).abs() < 1e-10,
                        "btdtr({a}, {b}, {x}) + btdtrc = {sum}, expected 1.0"
                    );
                }
            }
        }
    }

    #[test]
    fn btdtria_inverse() {
        for &a in &[0.25, 0.5, 1.0, 2.0, 5.0] {
            for &b in &[0.5, 2.0, 5.0] {
                for &x in &[0.2, 0.3, 0.7] {
                    let p = btdtr(a, b, x);
                    let recovered = btdtria(p, b, x);
                    assert!(
                        (recovered - a).abs() <= 1e-8 * a.abs().max(1.0),
                        "btdtria/btdtr failed: a={a}, b={b}, x={x}, p={p}, recovered={recovered}"
                    );
                }
            }
        }
    }

    #[test]
    fn btdtria_reference_values() {
        let cases: &[(f64, f64, f64, f64, f64)] = &[
            (0.5, 2.0, 0.3, 1.0249306894715173, 2e-12),
            (0.8, 2.0, 0.3, 0.38229690978762904, 2e-12),
            (0.2, 2.0, 0.3, 2.084034825279176, 2e-12),
            (0.5, 5.0, 0.7, 11.231150488078322, 2e-11),
            (
                0.9999999999999999,
                13.584377534422599,
                0.9654253202433963,
                5.915066688085389,
                2e-12,
            ),
        ];
        for &(p, b, x, expected, tolerance) in cases {
            let actual = btdtria(p, b, x);
            assert!(
                (actual - expected).abs() < tolerance,
                "btdtria({p}, {b}, {x}) = {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn btdtria_edges_match_scipy() {
        assert!(btdtria(0.0, 2.0, 0.3).is_infinite());
        assert_eq!(btdtria(1.0, 2.0, 0.3), f64::MIN_POSITIVE);
        assert_eq!(btdtria(1.0, 2.0, 1.1), f64::MIN_POSITIVE);
        assert!(btdtria(0.0, 2.0, 1.1).is_infinite());

        assert!(btdtria(0.5, 2.0, 0.0).is_nan());
        assert!(btdtria(0.5, 2.0, 1.0).is_nan());
        assert!(btdtria(0.0, 2.0, 0.0).is_nan());
        assert!(btdtria(1.0, 2.0, 0.0).is_nan());
        assert!(btdtria(0.5, 0.0, 0.3).is_nan());
        assert!(btdtria(0.5, -1.0, 0.3).is_nan());
        assert!(btdtria(-0.1, 2.0, 0.3).is_nan());
        assert!(btdtria(1.1, 2.0, 0.3).is_nan());
        assert!(btdtria(f64::NAN, 2.0, 0.3).is_nan());
        assert!(btdtria(0.5, f64::NAN, 0.3).is_nan());
        assert!(btdtria(0.5, 2.0, f64::NAN).is_nan());
    }

    #[test]
    fn btdtrib_inverse() {
        for &a in &[0.25, 0.5, 1.0, 2.0, 5.0] {
            for &b in &[0.5, 2.0, 5.0] {
                for &x in &[0.2, 0.3, 0.7] {
                    let p = btdtr(a, b, x);
                    let recovered = btdtrib(a, p, x);
                    assert!(
                        (recovered - b).abs() <= 1e-8 * b.abs().max(1.0),
                        "btdtrib/btdtr failed: a={a}, b={b}, x={x}, p={p}, recovered={recovered}"
                    );
                }
            }
        }
    }

    #[test]
    fn btdtrib_reference_values() {
        let cases: &[(f64, f64, f64, f64, f64)] = &[
            (2.0, 0.5, 0.3, 4.246702175718102, 2e-12),
            (2.0, 0.8, 0.3, 7.924719144223477, 2e-11),
            (2.0, 0.2, 0.3, 1.8790372491805813, 2e-12),
            (0.5, 0.7, 0.2, 2.6396222094025554, 2e-12),
            (
                0.04395757565162919,
                0.9999999999999999,
                0.7611845908830768,
                21.60864146301538,
                2e-11,
            ),
        ];
        for &(a, p, x, expected, tolerance) in cases {
            let actual = btdtrib(a, p, x);
            assert!(
                (actual - expected).abs() < tolerance,
                "btdtrib({a}, {p}, {x}) = {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn btdtrib_edges_match_scipy() {
        assert_eq!(btdtrib(2.0, 0.0, 0.3), f64::MIN_POSITIVE);
        assert!(btdtrib(2.0, 1.0, 0.3).is_infinite());
        assert_eq!(btdtrib(2.0, 0.0, 1.1), f64::MIN_POSITIVE);
        assert!(btdtrib(2.0, 1.0, 1.1).is_infinite());

        assert!(btdtrib(2.0, 0.5, 0.0).is_nan());
        assert!(btdtrib(2.0, 0.5, 1.0).is_nan());
        assert!(btdtrib(2.0, 0.0, 0.0).is_nan());
        assert!(btdtrib(2.0, 1.0, 0.0).is_nan());
        assert!(btdtrib(0.0, 0.5, 0.3).is_nan());
        assert!(btdtrib(-1.0, 0.5, 0.3).is_nan());
        assert!(btdtrib(2.0, -0.1, 0.3).is_nan());
        assert!(btdtrib(2.0, 1.1, 0.3).is_nan());
        assert!(btdtrib(f64::NAN, 0.5, 0.3).is_nan());
        assert!(btdtrib(2.0, f64::NAN, 0.3).is_nan());
        assert!(btdtrib(2.0, 0.5, f64::NAN).is_nan());
    }

    #[test]
    fn fdtrc_complement() {
        // fdtr + fdtrc should equal 1
        for &dfn in &[1.0, 5.0, 10.0] {
            for &dfd in &[1.0, 5.0, 10.0] {
                for &x in &[0.5, 1.0, 2.0, 5.0] {
                    let sum = fdtr(dfn, dfd, x) + fdtrc(dfn, dfd, x);
                    assert!(
                        (sum - 1.0).abs() < 1e-10,
                        "fdtr({dfn}, {dfd}, {x}) + fdtrc = {sum}, expected 1.0"
                    );
                }
            }
        }
    }

    #[test]
    fn fdtr_metamorphic_boundary_zero_and_monotone() {
        // /testing-metamorphic for [frankenscipy-xybu7]:
        // fdtr at x=0 must be exactly 0; fdtr must be monotonically
        // non-decreasing in x for any (dfn, dfd) > 0.
        for &dfn in &[1.0_f64, 2.0, 5.0, 10.0] {
            for &dfd in &[1.0_f64, 2.0, 5.0, 10.0] {
                assert_eq!(fdtr(dfn, dfd, 0.0), 0.0, "fdtr({dfn},{dfd},0) must be 0");

                let xs = [0.1_f64, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0];
                let mut prev = 0.0_f64;
                for &x in &xs {
                    let p = fdtr(dfn, dfd, x);
                    assert!(
                        p >= prev - 1e-12,
                        "fdtr({dfn}, {dfd}, {x}) = {p} < prev {prev} (not monotone)"
                    );
                    assert!(
                        (0.0..=1.0).contains(&p),
                        "fdtr({dfn}, {dfd}, {x}) = {p} outside [0, 1]"
                    );
                    prev = p;
                }
            }
        }
    }

    #[test]
    fn fdtr_dfn_equals_dfd_median_at_x_one() {
        // For F(d, d) the distribution is symmetric around 1 in the
        // sense that 1/F ~ F(d, d). Thus the median is exactly 1:
        //   fdtr(d, d, 1) = 0.5 for any d > 0.
        for &d in &[1.0_f64, 2.0, 5.0, 10.0, 50.0] {
            let p = fdtr(d, d, 1.0);
            assert!(
                (p - 0.5).abs() < 1e-10,
                "fdtr({d}, {d}, 1) = {p}, expected 0.5"
            );
        }
    }

    #[test]
    fn fdtri_roundtrip_recovers_x() {
        // fdtri(dfn, dfd, fdtr(dfn, dfd, x)) ≈ x for any positive x
        // and any (dfn, dfd) > 0.
        for &dfn in &[1.0_f64, 2.0, 5.0] {
            for &dfd in &[1.0_f64, 2.0, 5.0] {
                for &x in &[0.5_f64, 1.0, 2.0, 5.0] {
                    let p = fdtr(dfn, dfd, x);
                    let recovered = fdtri(dfn, dfd, p);
                    assert!(
                        (recovered - x).abs() < 1e-7,
                        "fdtri({dfn}, {dfd}, {p}={fdtr_val}) = {recovered}, expected {x}",
                        fdtr_val = p
                    );
                }
            }
        }
    }

    #[test]
    fn fdtridfd_inverse() {
        for &dfn in &[1.0, 2.0, 5.0, 10.0] {
            for &dfd in &[0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
                for &x in &[1.5, 2.0, 5.0] {
                    let p = fdtr(dfn, dfd, x);
                    let dfd_recovered = fdtridfd(dfn, p, x);
                    assert!(
                        (dfd_recovered - dfd).abs() < 1e-8,
                        "fdtridfd/fdtr failed: dfn={dfn}, dfd={dfd}, x={x}, p={p}, dfd_recovered={dfd_recovered}"
                    );
                }
            }
        }
    }

    #[test]
    fn fdtridfd_reference_values() {
        let cases: &[(f64, f64, f64, f64, f64)] = &[
            (5.0, 0.7, 1.5, 7.1205455518861855, 2e-10),
            (5.0, 0.5537887707581542, 1.5, 2.0, 2e-12),
            (5.0, 0.5, 1.5, 1.3789296276108034, 2e-12),
            (10.0, 0.8, 2.0, 6.2203600193018, 2e-10),
            (2.0, 0.9, 5.0, 3.3085663860076275, 2e-10),
        ];
        for &(dfn, p, x, expected, tolerance) in cases {
            let result = fdtridfd(dfn, p, x);
            assert!(
                (result - expected).abs() < tolerance,
                "fdtridfd({dfn}, {p}, {x}) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn fdtridfd_scipy_sentinels() {
        assert_eq!(fdtridfd(5.0, 0.0, 1.5), 1.0e-100);
        assert_eq!(fdtridfd(5.0, 0.0, 0.0), 5.0);
        assert_eq!(fdtridfd(5.0, 0.1, 0.0), 1.0e-100);
        assert_eq!(fdtridfd(5.0, 0.5, 0.0), 1.0e-100);
        assert_eq!(fdtridfd(5.0, 0.5000000001, 0.0), 1.0e100);
        assert_eq!(fdtridfd(5.0, 0.7, 0.0), 1.0e100);
        assert_eq!(fdtridfd(5.0, 0.9, 1.5), 1.0e100);
        assert_eq!(fdtridfd(5.0, 0.7, f64::INFINITY), 1.0e100);
        assert!(fdtridfd(5.0, 1.0, 1.5).is_nan());
        assert!(fdtridfd(f64::NAN, 0.7, 1.0).is_nan());
        assert!(fdtridfd(5.0, f64::NAN, 1.0).is_nan());
        assert!(fdtridfd(5.0, 0.7, f64::NAN).is_nan());
        assert!(fdtridfd(-1.0, 0.7, 1.0).is_nan());
        assert!(fdtridfd(5.0, -0.1, 1.0).is_nan());
        assert!(fdtridfd(5.0, 1.1, 1.0).is_nan());
        assert!(fdtridfd(5.0, 0.7, -1.0).is_nan());
    }

    #[test]
    fn stdtr_basic() {
        // stdtr(v, 0) = 0.5 for any v > 0 (symmetric around 0)
        assert!((stdtr(1.0, 0.0) - 0.5).abs() < 1e-10);
        assert!((stdtr(10.0, 0.0) - 0.5).abs() < 1e-10);
        assert!((stdtr(100.0, 0.0) - 0.5).abs() < 1e-10);

        // For large v, approaches normal distribution
        // stdtr(1000, 1.96) ≈ 0.975
        let result = stdtr(1000.0, 1.96);
        assert!(
            (result - 0.975).abs() < 0.01,
            "stdtr(1000, 1.96) = {result}, expected ~0.975"
        );

        // stdtr(1, t) = 0.5 + arctan(t)/π (Cauchy distribution)
        let t = 1.0_f64;
        let expected = 0.5 + t.atan() / std::f64::consts::PI;
        let result = stdtr(1.0, t);
        assert!(
            (result - expected).abs() < 0.001,
            "stdtr(1, 1) = {result}, expected {expected}"
        );
    }

    #[test]
    fn stdtr_symmetry() {
        // stdtr(v, -t) = 1 - stdtr(v, t)
        for &v in &[1.0, 2.0, 5.0, 10.0, 30.0] {
            for &t in &[0.5, 1.0, 2.0, 3.0] {
                let left = stdtr(v, -t);
                let right = 1.0 - stdtr(v, t);
                assert!(
                    (left - right).abs() < 1e-10,
                    "stdtr({v}, -{t}) = {left}, expected {right}"
                );
            }
        }
    }

    #[test]
    fn stdtrc_complement() {
        // stdtr + stdtrc should equal 1
        for &v in &[1.0, 2.0, 5.0, 10.0, 30.0] {
            for &t in &[-2.0, -1.0, 0.0, 1.0, 2.0] {
                let sum = stdtr(v, t) + stdtrc(v, t);
                assert!(
                    (sum - 1.0).abs() < 1e-10,
                    "stdtr({v}, {t}) + stdtrc = {sum}, expected 1.0"
                );
            }
        }
    }

    #[test]
    fn stdtrit_inverse() {
        // stdtrit should be inverse of stdtr
        for &v in &[1.0, 2.0, 5.0, 10.0, 30.0] {
            for &p in &[0.1, 0.25, 0.5, 0.75, 0.9, 0.95] {
                let t = stdtrit(v, p);
                let p_recovered = stdtr(v, t);
                assert!(
                    (p_recovered - p).abs() < 1e-8,
                    "stdtrit/stdtr failed: v={v}, p={p}, t={t}, p_recovered={p_recovered}"
                );
            }
        }
    }

    #[test]
    fn stdtrit_endpoints() {
        // SciPy returns +inf at both exact endpoints.
        assert!(stdtrit(5.0, 0.0).is_infinite() && stdtrit(5.0, 0.0).is_sign_positive());
        assert!(stdtrit(5.0, 1.0).is_infinite() && stdtrit(5.0, 1.0).is_sign_positive());
        assert!((stdtrit(5.0, 0.5) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn stdtridf_inverse() {
        for &v in &[0.25, 0.5, 1.0, 2.0, 5.0, 10.0] {
            for &t in &[0.75, 1.0, 1.5, 2.0, 4.0] {
                let p = stdtr(v, t);
                let v_recovered = stdtridf(p, t);
                assert!(
                    (v_recovered - v).abs() < 1e-8,
                    "stdtridf/stdtr failed: v={v}, t={t}, p={p}, v_recovered={v_recovered}"
                );
            }
        }
    }

    #[test]
    fn stdtridf_reference_values() {
        let cases: &[(f64, f64, f64, f64)] = &[
            (0.8, 1.25, 1.2176499408295116, 2e-12),
            (0.6, 1.0, 0.1321140237878431, 2e-12),
            (0.4, -1.0, 0.1321140237878431, 2e-12),
            (0.75, 1.0, 1.0000000000000093, 2e-12),
            (0.95, 2.0, 5.176135682782311, 2e-10),
        ];
        for &(p, t, expected, tolerance) in cases {
            let result = stdtridf(p, t);
            assert!(
                (result - expected).abs() < tolerance,
                "stdtridf({p}, {t}) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn stdtridf_scipy_sentinels() {
        assert_eq!(stdtridf(0.4, 1.0), -1.0e100);
        assert_eq!(stdtridf(0.8, -1.0), -1.0e100);
        assert_eq!(stdtridf(0.8, 0.0), 1.0e10);
        assert_eq!(stdtridf(0.5, 0.0), 5.0);
        assert_eq!(stdtridf(0.5, 1.0), 5.0e-51);
        assert_eq!(stdtridf(1.0, 1.0), 1.0e10);
        assert!(stdtridf(f64::NAN, 1.0).is_nan());
        assert!(stdtridf(0.8, f64::NAN).is_nan());
        assert!(stdtridf(-0.1, 1.0).is_nan());
        assert!(stdtridf(1.1, 1.0).is_nan());
    }

    #[test]
    fn bdtr_basic() {
        // bdtr(0, 1, 0.5) = 0.5 (1 trial, k=0: P(X=0) = 1-p)
        let result = bdtr(0.0, 1.0, 0.5);
        assert!(
            (result - 0.5).abs() < 1e-10,
            "bdtr(0, 1, 0.5) = {result}, expected 0.5"
        );

        // bdtr(0, 10, 0.5) = P(X=0) = 0.5^10 = 0.0009765625
        let result = bdtr(0.0, 10.0, 0.5);
        assert!(
            (result - 0.0009765625).abs() < 1e-8,
            "bdtr(0, 10, 0.5) = {result}, expected 0.0009765625"
        );

        // bdtr(5, 10, 0.5) = 0.623046875 (median of symmetric binomial)
        let result = bdtr(5.0, 10.0, 0.5);
        assert!(
            (result - 0.623046875).abs() < 1e-6,
            "bdtr(5, 10, 0.5) = {result}, expected 0.623046875"
        );
    }

    #[test]
    fn bdtr_bdtrc_complement() {
        // bdtr(k, n, p) + bdtrc(k, n, p) = 1
        for &n in &[5.0, 10.0, 20.0] {
            for &p in &[0.2, 0.5, 0.8] {
                for k in 0..=(n as i32) {
                    let kf = k as f64;
                    let sum = bdtr(kf, n, p) + bdtrc(kf, n, p);
                    assert!(
                        (sum - 1.0).abs() < 1e-10,
                        "bdtr({kf}, {n}, {p}) + bdtrc = {sum}"
                    );
                }
            }
        }
    }

    #[test]
    fn bdtri_inverse() {
        // bdtri should be inverse of bdtr
        for &n in &[5.0, 10.0, 20.0] {
            for &k in &[1.0, 3.0, 5.0] {
                if k >= n {
                    continue;
                }
                for &p in &[0.2, 0.5, 0.8] {
                    let y = bdtr(k, n, p);
                    if y > 0.01 && y < 0.99 {
                        let p_recovered = bdtri(k, n, y);
                        assert!(
                            (p_recovered - p).abs() < 0.01,
                            "bdtri failed: k={k}, n={n}, p={p}, y={y}, p_recovered={p_recovered}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn binomial_inverse_shape_parameters_round_trip() {
        let y = bdtr(4.5, 10.0, 0.5);
        assert!((bdtrik(y, 10.0, 0.5) - 4.5).abs() < 1.0e-10);

        let y = bdtr(5.0, 11.0, 0.5);
        assert!((bdtrin(5.0, y, 0.5) - 11.0).abs() < 1.0e-9);

        assert!(bdtrik(-0.1, 10.0, 0.5).is_nan());
        assert!(bdtrin(5.0, 1.1, 0.5).is_nan());
    }

    #[test]
    fn binomial_inverse_shape_parameters_match_scipy_reference_values() {
        let cases: &[DistributionInverseCase] = &[
            (bdtrik, 0.5, 10.0, 0.5, 4.5),
            (bdtrik, 0.9, 10.0, 0.5, 6.517_443_355_854_331),
            (bdtrik, 0.5, 5.0, 0.2, 0.385_541_504_319_304_9),
            (bdtrin, 5.0, 0.5, 0.5, 10.999_999_999_998_705),
            (bdtrin, 5.0, 0.9, 0.5, 7.507_360_823_906_628),
            (bdtrin, 0.0, 0.000_976_562_5, 0.5, 10.000_000_000_015_802),
        ];
        for &(func, a, b, c, expected) in cases {
            let actual = func(a, b, c);
            assert!(
                (actual - expected).abs() <= 2.0e-6 * expected.abs().max(1.0),
                "binomial inverse reference mismatch: got {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn nbdtr_basic() {
        // nbdtr(0, 1, 0.5) = P(0 failures before 1 success) = p = 0.5
        let result = nbdtr(0.0, 1.0, 0.5);
        assert!(
            (result - 0.5).abs() < 1e-10,
            "nbdtr(0, 1, 0.5) = {result}, expected 0.5"
        );

        // nbdtr(0, 1, p) = p for all p (geometric: first trial is success)
        for &p in &[0.1, 0.5, 0.9] {
            let result = nbdtr(0.0, 1.0, p);
            assert!(
                (result - p).abs() < 1e-10,
                "nbdtr(0, 1, {p}) = {result}, expected {p}"
            );
        }

        // As k -> infinity, nbdtr(k, n, p) -> 1
        let result = nbdtr(100.0, 1.0, 0.5);
        assert!(
            (result - 1.0).abs() < 1e-10,
            "nbdtr(100, 1, 0.5) = {result}, expected ~1.0"
        );
    }

    #[test]
    fn nbdtr_nbdtrc_complement() {
        // nbdtr(k, n, p) + nbdtrc(k, n, p) = 1
        for &n in &[1.0, 3.0, 5.0] {
            for &p in &[0.2, 0.5, 0.8] {
                for &k in &[0.0, 1.0, 5.0, 10.0] {
                    let sum = nbdtr(k, n, p) + nbdtrc(k, n, p);
                    assert!(
                        (sum - 1.0).abs() < 1e-10,
                        "nbdtr({k}, {n}, {p}) + nbdtrc = {sum}"
                    );
                }
            }
        }
    }

    #[test]
    fn nbdtri_inverse() {
        // nbdtri should be inverse of nbdtr
        for &n in &[1.0, 3.0, 5.0] {
            for &k in &[0.0, 2.0, 5.0] {
                for &p in &[0.2, 0.5, 0.8] {
                    let y = nbdtr(k, n, p);
                    if y > 0.01 && y < 0.99 {
                        let p_recovered = nbdtri(k, n, y);
                        assert!(
                            (p_recovered - p).abs() < 0.01,
                            "nbdtri failed: k={k}, n={n}, p={p}, y={y}, p_recovered={p_recovered}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn negative_binomial_inverse_shape_parameters_round_trip() {
        let y = nbdtr(9.0, 10.0, 0.5);
        assert!((nbdtrik(y, 10.0, 0.5) - 9.0).abs() < 1.0e-9);

        let y = nbdtr(5.0, 6.0, 0.5);
        assert!((nbdtrin(5.0, y, 0.5) - 6.0).abs() < 1.0e-9);

        assert!(nbdtrik(1.0, 10.0, 0.5).is_nan());
        assert!(nbdtrin(-1.0, 0.5, 0.5).is_nan());
    }

    #[test]
    fn negative_binomial_inverse_shape_parameters_match_scipy_reference_values() {
        let cases: &[DistributionInverseCase] = &[
            (nbdtrik, 0.5, 10.0, 0.5, 9.000_000_000_001_473),
            (nbdtrik, 0.9, 10.0, 0.5, 15.452_848_668_084_998),
            (nbdtrik, 0.5, 5.0, 0.2, 18.017_180_776_964_413),
            (nbdtrin, 5.0, 0.5, 0.5, 5.999_999_999_997_906),
            (nbdtrin, 5.0, 0.9, 0.5, 2.507_360_823_906_642_7),
            (nbdtrin, 0.0, 0.5, 0.5, 1.000_000_000_000_667_7),
        ];
        for &(func, a, b, c, expected) in cases {
            let actual = func(a, b, c);
            assert!(
                (actual - expected).abs() <= 2.0e-6 * expected.abs().max(1.0),
                "negative-binomial inverse reference mismatch: got {actual}, expected {expected}"
            );
        }
    }

    fn complex_scalar(tensor: &SpecialTensor) -> Complex64 {
        match tensor {
            SpecialTensor::ComplexScalar(c) => *c,
            _ => panic!("expected ComplexScalar"),
        }
    }

    fn complex_vec(tensor: &SpecialTensor) -> &[Complex64] {
        match tensor {
            SpecialTensor::ComplexVec(values) => values,
            _ => panic!("expected ComplexVec"),
        }
    }

    fn assert_complex_close(actual: Complex64, expected: Complex64) {
        assert!(
            (actual.re - expected.re).abs() < 1e-10,
            "real parts differ: actual={}, expected={}",
            actual.re,
            expected.re
        );
        assert!(
            (actual.im - expected.im).abs() < 1e-10,
            "imaginary parts differ: actual={}, expected={}",
            actual.im,
            expected.im
        );
    }

    #[test]
    fn complex_betaln_real_inputs_match_real_path() {
        let a = SpecialTensor::ComplexScalar(Complex64::new(2.0, 0.0));
        let b = SpecialTensor::ComplexScalar(Complex64::new(3.0, 0.0));
        let result = betaln(&a, &b, RuntimeMode::Strict).unwrap();
        let c = complex_scalar(&result);

        let real_a = SpecialTensor::RealScalar(2.0);
        let real_b = SpecialTensor::RealScalar(3.0);
        let real_result = betaln(&real_a, &real_b, RuntimeMode::Strict).unwrap();
        let expected = match real_result {
            SpecialTensor::RealScalar(v) => v,
            _ => panic!("expected RealScalar"),
        };

        assert!(
            (c.re - expected).abs() < 1e-10,
            "complex betaln(2+0i, 3+0i) = {} + {}i, expected {} + 0i",
            c.re,
            c.im,
            expected
        );
        assert!(
            c.im.abs() < 1e-10,
            "imaginary part should be ~0, got {}",
            c.im
        );
    }

    #[test]
    fn complex_beta_real_inputs_match_real_path() {
        let a = SpecialTensor::ComplexScalar(Complex64::new(2.0, 0.0));
        let b = SpecialTensor::ComplexScalar(Complex64::new(3.0, 0.0));
        let result = beta(&a, &b, RuntimeMode::Strict).unwrap();
        let c = complex_scalar(&result);

        let real_a = SpecialTensor::RealScalar(2.0);
        let real_b = SpecialTensor::RealScalar(3.0);
        let real_result = beta(&real_a, &real_b, RuntimeMode::Strict).unwrap();
        let expected = match real_result {
            SpecialTensor::RealScalar(v) => v,
            _ => panic!("expected RealScalar"),
        };

        assert!(
            (c.re - expected).abs() < 1e-10,
            "complex beta(2+0i, 3+0i) = {} + {}i, expected {} + 0i",
            c.re,
            c.im,
            expected
        );
        assert!(
            c.im.abs() < 1e-10,
            "imaginary part should be ~0, got {}",
            c.im
        );
    }

    #[test]
    fn complex_betaln_with_imaginary_parts() {
        // Verify complex betaln produces finite results for complex inputs
        let a = SpecialTensor::ComplexScalar(Complex64::new(1.0, 1.0));
        let b = SpecialTensor::ComplexScalar(Complex64::new(2.0, 0.5));
        let result = betaln(&a, &b, RuntimeMode::Strict).unwrap();
        let c = complex_scalar(&result);

        assert!(c.re.is_finite(), "betaln result should be finite");
        assert!(c.im.is_finite(), "betaln result should be finite");
        // Verify symmetry: betaln(a, b) = betaln(b, a)
        let result_sym = betaln(&b, &a, RuntimeMode::Strict).unwrap();
        let c_sym = complex_scalar(&result_sym);
        assert!(
            (c.re - c_sym.re).abs() < 1e-10,
            "betaln should be symmetric"
        );
        assert!(
            (c.im - c_sym.im).abs() < 1e-10,
            "betaln should be symmetric"
        );
    }

    #[test]
    fn complex_beta_with_imaginary_parts() {
        // Verify complex beta produces finite results
        let a = SpecialTensor::ComplexScalar(Complex64::new(1.0, 1.0));
        let b = SpecialTensor::ComplexScalar(Complex64::new(2.0, 0.5));
        let result = beta(&a, &b, RuntimeMode::Strict).unwrap();
        let c = complex_scalar(&result);

        assert!(c.re.is_finite(), "beta result should be finite");
        assert!(c.im.is_finite(), "beta result should be finite");
        // beta = exp(betaln)
        let ln_result = betaln(&a, &b, RuntimeMode::Strict).unwrap();
        let ln_c = complex_scalar(&ln_result);
        let expected = ln_c.exp();
        assert!(
            (c.re - expected.re).abs() < 1e-10,
            "beta should equal exp(betaln)"
        );
        assert!(
            (c.im - expected.im).abs() < 1e-10,
            "beta should equal exp(betaln)"
        );
    }

    #[test]
    fn complex_betaln_vector() {
        let a = SpecialTensor::ComplexVec(vec![Complex64::new(2.0, 0.0), Complex64::new(1.0, 1.0)]);
        let b = SpecialTensor::ComplexScalar(Complex64::new(3.0, 0.0));
        let result = betaln(&a, &b, RuntimeMode::Strict).unwrap();
        match result {
            SpecialTensor::ComplexVec(v) => {
                assert_eq!(v.len(), 2);
                // first entry (real inputs) should have near-zero imaginary part
                assert!(v[0].im.abs() < 1e-10, "real inputs should have im~0");
                assert!(v[0].re.is_finite());
                assert!(v[1].re.is_finite());
                assert!(v[1].im.is_finite());
            }
            _ => panic!("expected ComplexVec"),
        }
    }

    #[test]
    fn beta_vector_mismatch_returns_domain_error() {
        let err = beta(
            &SpecialTensor::RealVec(vec![1.0, 2.0]),
            &SpecialTensor::RealVec(vec![1.0]),
            RuntimeMode::Hardened,
        )
        .expect_err("mismatched beta vectors should fail");
        assert_eq!(err.kind, SpecialErrorKind::DomainError);
        assert_eq!(err.detail, "vector inputs must have matching lengths");
    }

    #[test]
    fn betaln_complex_vector_mismatch_returns_domain_error() {
        let err = betaln(
            &SpecialTensor::ComplexVec(vec![Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)]),
            &SpecialTensor::ComplexVec(vec![Complex64::new(1.0, 0.0)]),
            RuntimeMode::Hardened,
        )
        .expect_err("mismatched betaln vectors should fail");
        assert_eq!(err.kind, SpecialErrorKind::DomainError);
        assert_eq!(err.detail, "vector inputs must have matching lengths");
    }

    #[test]
    fn complex_beta_mixed_real_complex() {
        let a = SpecialTensor::RealScalar(2.0);
        let b = SpecialTensor::ComplexScalar(Complex64::new(3.0, 0.5));
        let result = beta(&a, &b, RuntimeMode::Strict).unwrap();
        let c = complex_scalar(&result);
        // Should produce a complex result
        assert!(c.re.is_finite());
        assert!(c.im.is_finite());
    }

    #[test]
    fn complex_betainc_real_inputs_match_real() {
        // Complex betainc with real inputs should match real path
        let a = SpecialTensor::ComplexScalar(Complex64::new(2.0, 0.0));
        let b = SpecialTensor::ComplexScalar(Complex64::new(3.0, 0.0));
        let x = SpecialTensor::ComplexScalar(Complex64::new(0.5, 0.0));
        let result = betainc(&a, &b, &x, RuntimeMode::Strict).unwrap();
        let c = complex_scalar(&result);

        let real_a = SpecialTensor::RealScalar(2.0);
        let real_b = SpecialTensor::RealScalar(3.0);
        let real_x = SpecialTensor::RealScalar(0.5);
        let real_result = betainc(&real_a, &real_b, &real_x, RuntimeMode::Strict).unwrap();
        let expected = match real_result {
            SpecialTensor::RealScalar(v) => v,
            _ => panic!("expected RealScalar"),
        };

        assert!(
            (c.re - expected).abs() < 1e-8,
            "complex betainc(2,3,0.5) = {} + {}i, expected {} + 0i",
            c.re,
            c.im,
            expected
        );
        assert!(
            c.im.abs() < 1e-10,
            "imaginary part should be ~0, got {}",
            c.im
        );
    }

    #[test]
    fn complex_betainc_endpoints() {
        // betainc(a, b, 0) = 0
        let a = SpecialTensor::ComplexScalar(Complex64::new(2.0, 0.5));
        let b = SpecialTensor::ComplexScalar(Complex64::new(3.0, 0.0));
        let x = SpecialTensor::ComplexScalar(Complex64::new(0.0, 0.0));
        let result = betainc(&a, &b, &x, RuntimeMode::Strict).unwrap();
        let c = complex_scalar(&result);
        assert!(c.re.abs() < 1e-10, "betainc(a,b,0) should be 0");
        assert!(c.im.abs() < 1e-10);

        // betainc(a, b, 1) = 1
        let x1 = SpecialTensor::ComplexScalar(Complex64::new(1.0, 0.0));
        let result1 = betainc(&a, &b, &x1, RuntimeMode::Strict).unwrap();
        let c1 = complex_scalar(&result1);
        assert!(
            (c1.re - 1.0).abs() < 1e-10,
            "betainc(a,b,1) should be 1, got {}",
            c1.re
        );
        assert!(c1.im.abs() < 1e-10);
    }

    #[test]
    fn complex_betainc_with_complex_args() {
        // Just verify it produces finite results
        let a = SpecialTensor::ComplexScalar(Complex64::new(2.0, 0.5));
        let b = SpecialTensor::ComplexScalar(Complex64::new(3.0, -0.3));
        let x = SpecialTensor::ComplexScalar(Complex64::new(0.5, 0.0));
        let result = betainc(&a, &b, &x, RuntimeMode::Strict).unwrap();
        let c = complex_scalar(&result);
        assert!(c.re.is_finite(), "betainc should produce finite result");
        assert!(c.im.is_finite(), "betainc should produce finite result");
    }

    #[test]
    fn beta_real_scalar_complex_vector_broadcasts() {
        let a = SpecialTensor::RealScalar(2.0);
        let b = SpecialTensor::ComplexVec(vec![Complex64::new(3.0, 0.0), Complex64::new(1.5, 0.5)]);
        let result = beta(&a, &b, RuntimeMode::Strict).unwrap();
        let values = complex_vec(&result);

        assert_eq!(values.len(), 2);
        assert_complex_close(
            values[0],
            complex_beta_scalar(Complex64::from_real(2.0), Complex64::new(3.0, 0.0)),
        );
        assert_complex_close(
            values[1],
            complex_beta_scalar(Complex64::from_real(2.0), Complex64::new(1.5, 0.5)),
        );
    }

    #[test]
    fn betaln_complex_scalar_real_vector_broadcasts() {
        let a = SpecialTensor::ComplexScalar(Complex64::new(2.5, 0.25));
        let b = SpecialTensor::RealVec(vec![1.0, 2.0]);
        let result = betaln(&a, &b, RuntimeMode::Strict).unwrap();
        let values = complex_vec(&result);

        assert_eq!(values.len(), 2);
        assert_complex_close(
            values[0],
            complex_betaln_scalar(Complex64::new(2.5, 0.25), Complex64::from_real(1.0)),
        );
        assert_complex_close(
            values[1],
            complex_betaln_scalar(Complex64::new(2.5, 0.25), Complex64::from_real(2.0)),
        );
    }

    #[test]
    fn beta_mixed_real_complex_vectors_preserve_symmetry() {
        let a = SpecialTensor::RealVec(vec![2.0, 3.0]);
        let b =
            SpecialTensor::ComplexVec(vec![Complex64::new(1.5, 0.5), Complex64::new(2.5, -0.25)]);
        let forward = beta(&a, &b, RuntimeMode::Strict).unwrap();
        let reverse = beta(&b, &a, RuntimeMode::Strict).unwrap();
        let forward_values = complex_vec(&forward);
        let reverse_values = complex_vec(&reverse);

        assert_eq!(forward_values.len(), 2);
        assert_eq!(reverse_values.len(), 2);
        for (forward_value, reverse_value) in forward_values.iter().zip(reverse_values.iter()) {
            assert_complex_close(*forward_value, *reverse_value);
        }
    }

    #[test]
    fn betaln_mixed_vector_mismatch_returns_domain_error() {
        let err = betaln(
            &SpecialTensor::RealVec(vec![2.0, 3.0]),
            &SpecialTensor::ComplexVec(vec![Complex64::new(1.0, 0.0)]),
            RuntimeMode::Hardened,
        )
        .expect_err("mismatched mixed betaln vectors should fail");
        assert_eq!(err.kind, SpecialErrorKind::DomainError);
        assert_eq!(err.detail, "vector inputs must have matching lengths");
    }

    #[test]
    fn btdtr_matches_scipy_reference_values() {
        // scipy.special.btdtr(2, 3, 0.5) - beta distribution CDF
        // B(2,3) at x=0.5: I_0.5(2,3) = 0.6875
        let result = btdtr(2.0, 3.0, 0.5);
        assert!(
            (result - 0.6875).abs() < 1e-10,
            "btdtr(2,3,0.5) got {result}, expected 0.6875"
        );
    }

    #[test]
    fn betaln_negative_args_match_scipy() {
        // scipy.special.betaln returns finite values for negative non-integer
        // arguments (= gammaln(a)+gammaln(b)-gammaln(a+b)); we previously
        // fail-closed to NaN for any nonpositive parameter.
        let cases = [
            (-2.5, 3.0, 0.06453852113757116_f64),
            (2.0, -3.5, -2.169053700369523),
            (-4.3, -1.2, 3.814037380330781),
            (0.3, -0.7, 1.2337463436314935),
            (-1.5, 2.5, 1.1447298858494004),
            (5.0, -4.5, -0.2073951943460706),
        ];
        for (a, b, want) in cases {
            let got = betaln_scalar(a, b, RuntimeMode::Strict).unwrap();
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                "betaln({a},{b}) got {got}, want {want}"
            );
        }
        // Pole limits also match SciPy: a+b a nonpositive integer => -inf.
        assert_eq!(
            betaln_scalar(-0.5, 0.5, RuntimeMode::Strict).unwrap(),
            f64::NEG_INFINITY
        );
        assert_eq!(
            betaln_scalar(-2.5, -2.5, RuntimeMode::Strict).unwrap(),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn beta_negative_args_match_scipy_signed() {
        // scipy.special.beta is SIGNED for negative args (B = Γ(a)Γ(b)/Γ(a+b)),
        // not just |B|; the gamma-sign factor restores it.
        let cases = [
            (-2.5, 3.0, -1.0666666666666667_f64),
            (2.0, -3.5, 0.11428571428571427),
            (-4.3, -1.2, -45.33309684186555),
            (-1.5, 2.5, 3.1415926535897936),
            (0.3, -0.7, 3.4340706764177225),
            (3.0, 4.0, 0.016666666666666666), // positive args: sign=+1, unchanged
        ];
        for (a, b, want) in cases {
            let got = beta_scalar(a, b, RuntimeMode::Strict).unwrap();
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                "beta({a},{b}) got {got}, want {want}"
            );
        }
    }

    #[test]
    fn bdtr_matches_scipy_reference_values() {
        // scipy.special.bdtr(5, 10, 0.5) - binomial distribution CDF
        // P(X <= 5) for X ~ Binom(10, 0.5)
        let result = bdtr(5.0, 10.0, 0.5);
        let expected = 0.623046875;
        assert!(
            (result - expected).abs() < 1e-10,
            "bdtr(5,10,0.5) got {result}, expected {expected}"
        );
    }

    #[test]
    fn stdtr_matches_scipy_reference_values() {
        // scipy.special.stdtr(df, t) - Student's t CDF
        // stdtr(5, 0) = 0.5 (symmetric at 0)
        // stdtr(5, 1.0) ≈ 0.8183 (df=5, t=1)
        let result0 = stdtr(5.0, 0.0);
        assert!(
            (result0 - 0.5).abs() < 1e-10,
            "stdtr(5, 0) got {result0}, expected 0.5"
        );

        let result1 = stdtr(5.0, 1.0);
        assert!(
            (result1 - 0.8183).abs() < 1e-3,
            "stdtr(5, 1) got {result1}, expected ~0.8183"
        );
    }

    #[test]
    fn fdtr_matches_scipy_reference_values() {
        // scipy.special.fdtr(dfn, dfd, x) - F distribution CDF
        // fdtr(5, 10, 1.0) ≈ 0.5348
        let result = fdtr(5.0, 10.0, 1.0);
        assert!(
            (result - 0.5349).abs() < 1e-3,
            "fdtr(5, 10, 1) got {result}, expected ~0.5349"
        );

        // fdtr at 0 should be 0
        let result0 = fdtr(5.0, 10.0, 0.0);
        assert!(
            result0.abs() < 1e-10,
            "fdtr(5, 10, 0) got {result0}, expected 0"
        );
    }

    #[test]
    fn fdtri_matches_scipy_reference_values() {
        // scipy.special.fdtri(5, 10, 0.5) ≈ 0.931933160851048
        let result = fdtri(5.0, 10.0, 0.5);
        assert!(
            (result - 0.931933160851048).abs() < 1e-6,
            "fdtri(5, 10, 0.5) got {result}, expected 0.931933160851048"
        );
    }

    #[test]
    fn stdtrit_matches_scipy_reference_values() {
        // scipy.special.stdtrit(10, 0.95) ≈ 1.8124611228116756
        let result = stdtrit(10.0, 0.95);
        assert!(
            (result - 1.8124611228116756).abs() < 1e-6,
            "stdtrit(10, 0.95) got {result}, expected 1.8124611228116756"
        );
    }
}
