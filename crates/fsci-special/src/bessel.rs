#![forbid(unsafe_code)]

use std::f64::consts::{FRAC_2_PI, LN_2, PI};

use fsci_runtime::RuntimeMode;

use crate::types::{
    Complex64, DispatchPlan, DispatchStep, KernelRegime, SpecialError, SpecialErrorKind,
    SpecialResult, SpecialTensor, not_yet_implemented, record_special_trace,
};

pub const BESSEL_DISPATCH_PLAN: &[DispatchPlan] = &[
    DispatchPlan {
        function: "j0",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "small |x| rational/series blend",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large |x| oscillatory asymptotics",
            },
        ],
        notes: "Strict mode tracks SciPy-style finite-real behavior for integer-order Bessel J0.",
    },
    DispatchPlan {
        function: "j1",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "small |x| rational/series blend",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large |x| oscillatory asymptotics",
            },
        ],
        notes: "Odd parity j1(-x)=-j1(x) must remain intact.",
    },
    DispatchPlan {
        function: "jn",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "integer order n>=2 via upward recurrence",
            },
            DispatchStep {
                regime: KernelRegime::Reflection,
                when: "negative-order parity reconstruction",
            },
        ],
        notes: "Order input must be integral; hardened mode fail-closes non-integral orders.",
    },
    DispatchPlan {
        function: "y0",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "small positive x with logarithmic correction",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large x oscillatory asymptotics",
            },
        ],
        notes: "Negative real axis is domain-invalid for this real-only wrapper.",
    },
    DispatchPlan {
        function: "y1",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "small positive x with logarithmic correction",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large x oscillatory asymptotics",
            },
        ],
        notes: "x=0 preserves SciPy-style singular divergence in strict mode.",
    },
    DispatchPlan {
        function: "yn",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "integer order n>=2 via upward recurrence",
            },
            DispatchStep {
                regime: KernelRegime::Reflection,
                when: "negative-order parity reconstruction",
            },
        ],
        notes: "Order input must be integral; hardened mode fail-closes non-integral orders.",
    },
    DispatchPlan {
        function: "jv",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "small |z|",
            },
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "integer-order and moderate |z| windows",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large |z| oscillatory region",
            },
        ],
        notes: "Negative-order reconstruction uses SciPy-compatible cancellation-safe branches.",
    },
    DispatchPlan {
        function: "yv",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "small |z| away from singular neighborhoods",
            },
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "moderate argument / order coupling",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large |z|",
            },
        ],
        notes: "Origin-adjacent singularity behavior must preserve SciPy divergence semantics.",
    },
    DispatchPlan {
        function: "iv",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "small |z|",
            },
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "order shifting in stable window",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large |z| or large order",
            },
        ],
        notes: "Real-negative argument handling enforces strict integer-order branch rules.",
    },
    DispatchPlan {
        function: "kv",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "small positive |z|",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large positive real z",
            },
            DispatchStep {
                regime: KernelRegime::BackendDelegate,
                when: "complex continuation branch",
            },
        ],
        notes: "Hardened mode adds singular-neighborhood diagnostics near zero.",
    },
    DispatchPlan {
        function: "hankel1",
        steps: &[
            DispatchStep {
                regime: KernelRegime::BackendDelegate,
                when: "AMOS-compatible principal branch composition",
            },
            DispatchStep {
                regime: KernelRegime::Reflection,
                when: "negative-order phase mapping",
            },
        ],
        notes: "Outgoing-wave sign and phase conventions are contract-critical.",
    },
    DispatchPlan {
        function: "hankel2",
        steps: &[
            DispatchStep {
                regime: KernelRegime::BackendDelegate,
                when: "AMOS-compatible principal branch composition",
            },
            DispatchStep {
                regime: KernelRegime::Reflection,
                when: "negative-order phase mapping",
            },
        ],
        notes: "Incoming-wave sign and phase conventions are contract-critical.",
    },
    DispatchPlan {
        function: "wright_bessel",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "nonnegative arguments summed in the log-domain series",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large-a windows collapse to early terms through log-gamma damping",
            },
        ],
        notes: "SciPy only exposes the nonnegative real domain; strict mode returns NaN outside it.",
    },
];

const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_9;
const BESSEL_SERIES_MAX_TERMS: usize = 96;

pub fn j0(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("j0", z, mode, |x| Ok(j0_core(x)))
}

pub fn j1(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("j1", z, mode, |x| Ok(j1_core(x)))
}

pub fn jn(n: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_binary("jn", n, z, mode, |order, x| jn_scalar(order, x, mode))
}

pub fn y0(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("y0", z, mode, |x| y0_scalar(x, mode))
}

pub fn y1(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("y1", z, mode, |x| y1_scalar(x, mode))
}

pub fn yn(n: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_binary("yn", n, z, mode, |order, x| yn_scalar(order, x, mode))
}

/// Bessel function of the first kind for real order v: J_v(z).
///
/// Uses power series for small z, asymptotic expansion for large z,
/// and reduces to j0/j1/jn for integer orders.
/// Supports both real and complex z arguments.
pub fn jv(v: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    bessel_dispatch("jv", v, z, mode, BesselKind::Jv)
}

/// Bessel function of the second kind for real order v: Y_v(z).
///
/// Y_v(z) = (J_v(z) cos(vπ) - J_{-v}(z)) / sin(vπ) for non-integer v.
/// For integer v, uses the integer-order y0/y1/yn implementations.
/// Supports both real and complex z arguments.
pub fn yv(v: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    bessel_dispatch("yv", v, z, mode, BesselKind::Yv)
}

/// Exponentially scaled Bessel function of the first kind.
///
/// Matches `scipy.special.jve(v, z)`, scaling complex-valued inputs by
/// `exp(-abs(Im(z)))`. Real-valued inputs are unchanged.
pub fn jve(v: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    bessel_dispatch("jve", v, z, mode, BesselKind::Jve)
}

/// Exponentially scaled Bessel function of the second kind.
///
/// Matches `scipy.special.yve(v, z)`, scaling complex-valued inputs by
/// `exp(-abs(Im(z)))`. Real-valued inputs are unchanged.
pub fn yve(v: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    bessel_dispatch("yve", v, z, mode, BesselKind::Yve)
}

/// Modified Bessel function of the first kind for real order v: I_v(z).
///
/// I_v(z) = (z/2)^v Σ (z²/4)^k / (k! Γ(v+k+1))
/// Supports both real and complex z arguments.
pub fn iv(v: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    bessel_dispatch("iv", v, z, mode, BesselKind::Iv)
}

/// Modified Bessel function of the first kind of order 0: I_0(z).
///
/// Convenience wrapper for iv(0, z). Matches `scipy.special.i0(z)`.
pub fn i0(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("i0", z, mode, |x| Ok(iv_scalar(0.0, x)))
}

/// Modified Bessel function of the first kind of order 1: I_1(z).
///
/// Convenience wrapper for iv(1, z). Matches `scipy.special.i1(z)`.
pub fn i1(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("i1", z, mode, |x| Ok(iv_scalar(1.0, x)))
}

/// Scalar convenience function for I_0(x).
#[must_use]
pub fn i0_scalar(x: f64) -> f64 {
    iv_scalar(0.0, x)
}

/// Scalar convenience function for I_1(x).
#[must_use]
pub fn i1_scalar(x: f64) -> f64 {
    iv_scalar(1.0, x)
}

/// Modified Bessel function of the second kind for real order v: K_v(z).
///
/// K_v(z) = π/2 (I_{-v}(z) - I_v(z)) / sin(vπ) for non-integer v.
/// Supports both real and complex z arguments.
pub fn kv(v: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    bessel_dispatch("kv", v, z, mode, BesselKind::Kv)
}

/// Modified Bessel function of the second kind of order 0: K_0(z).
///
/// Convenience wrapper for kv(0, z). Matches `scipy.special.k0(z)`.
pub fn k0(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("k0", z, mode, |x| kv_scalar(0.0, x, mode))
}

/// Modified Bessel function of the second kind of order 1: K_1(z).
///
/// Convenience wrapper for kv(1, z). Matches `scipy.special.k1(z)`.
pub fn k1(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("k1", z, mode, |x| kv_scalar(1.0, x, mode))
}

/// Modified Bessel function of the second kind for integer order n: K_n(z).
///
/// Convenience wrapper for kv(n, z). Matches `scipy.special.kn(n, z)`.
pub fn kn(n: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    bessel_dispatch("kn", n, z, mode, BesselKind::Kv)
}

/// Scalar convenience function for K_0(x).
///
/// Matches `scipy.special.k0(x)` for positive real x.
#[must_use]
pub fn k0_scalar(x: f64) -> f64 {
    kv_scalar(0.0, x, RuntimeMode::Strict).unwrap_or(f64::NAN)
}

/// Scalar convenience function for K_1(x).
///
/// Matches `scipy.special.k1(x)` for positive real x.
#[must_use]
pub fn k1_scalar(x: f64) -> f64 {
    kv_scalar(1.0, x, RuntimeMode::Strict).unwrap_or(f64::NAN)
}

/// Scalar convenience function for K_n(x) with integer order n.
///
/// Matches `scipy.special.kn(n, x)` for positive real x.
#[must_use]
pub fn kn_scalar(n: i32, x: f64) -> f64 {
    kv_scalar(f64::from(n), x, RuntimeMode::Strict).unwrap_or(f64::NAN)
}

// ═══════════════════════════════════════════════════════════════════════════
// Exponentially Scaled Bessel Functions
// ═══════════════════════════════════════════════════════════════════════════

/// Exponentially scaled modified Bessel function I_0: i0e(x) = I_0(x) * exp(-|x|).
///
/// Matches `scipy.special.i0e(x)`.
pub fn i0e(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("i0e", z, mode, |x| Ok(i0e_scalar(x)))
}

/// Exponentially scaled modified Bessel function I_1: i1e(x) = I_1(x) * exp(-|x|).
///
/// Matches `scipy.special.i1e(x)`.
pub fn i1e(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("i1e", z, mode, |x| Ok(i1e_scalar(x)))
}

/// Exponentially scaled modified Bessel function I_v: ive(v, x) = I_v(x) * exp(-|x|).
///
/// Matches `scipy.special.ive(v, x)`.
pub fn ive(v: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    bessel_dispatch("ive", v, z, mode, BesselKind::Ive)
}

/// Exponentially scaled modified Bessel function K_0: k0e(x) = K_0(x) * exp(x).
///
/// Matches `scipy.special.k0e(x)`.
pub fn k0e(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("k0e", z, mode, |x| Ok(k0e_scalar(x)))
}

/// Exponentially scaled modified Bessel function K_1: k1e(x) = K_1(x) * exp(x).
///
/// Matches `scipy.special.k1e(x)`.
pub fn k1e(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("k1e", z, mode, |x| Ok(k1e_scalar(x)))
}

/// Exponentially scaled modified Bessel function K_v: kve(v, x) = K_v(x) * exp(x).
///
/// Matches `scipy.special.kve(v, x)`.
pub fn kve(v: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    bessel_dispatch("kve", v, z, mode, BesselKind::Kve)
}

/// Scalar: i0e(x) = I_0(x) * exp(-|x|).
#[must_use]
pub fn i0e_scalar(x: f64) -> f64 {
    iv_scalar(0.0, x) * (-x.abs()).exp()
}

/// Scalar: i1e(x) = I_1(x) * exp(-|x|).
#[must_use]
pub fn i1e_scalar(x: f64) -> f64 {
    iv_scalar(1.0, x) * (-x.abs()).exp()
}

/// Scalar: ive(v, x) = I_v(x) * exp(-|x|).
#[must_use]
pub fn ive_scalar(v: f64, x: f64) -> f64 {
    iv_scalar(v, x) * (-x.abs()).exp()
}

/// Scalar: k0e(x) = K_0(x) * exp(x).
#[must_use]
pub fn k0e_scalar(x: f64) -> f64 {
    k0_scalar(x) * x.exp()
}

/// Scalar: k1e(x) = K_1(x) * exp(x).
#[must_use]
pub fn k1e_scalar(x: f64) -> f64 {
    k1_scalar(x) * x.exp()
}

/// Scalar: kve(v, x) = K_v(x) * exp(x).
#[must_use]
pub fn kve_scalar(v: f64, x: f64) -> f64 {
    // Scaled directly so it stays finite past z ≈ 745, where K_v underflows to 0
    // and the old kv·e^x form gave 0·∞ = NaN. SciPy's kve is finite there.
    if x > 0.0 {
        return kv_scaled_value(v.abs(), x);
    }
    kv_scalar(v, x, RuntimeMode::Strict).unwrap_or(f64::NAN) * x.exp()
}

/// Hankel function of the first kind: H1_v(z) = J_v(z) + i·Y_v(z).
pub fn hankel1(v: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    hankel_dispatch("hankel1", v, z, mode, HankelKind::H1)
}

/// Hankel function of the second kind: H2_v(z) = J_v(z) - i·Y_v(z).
pub fn hankel2(v: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    hankel_dispatch("hankel2", v, z, mode, HankelKind::H2)
}

/// Exponentially scaled Hankel function of the first kind.
///
/// Matches `scipy.special.hankel1e(v, z) = hankel1(v, z) * exp(-i z)`.
pub fn hankel1e(v: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    scaled_hankel_dispatch("hankel1e", v, z, mode, HankelKind::H1, HankelScale::First)
}

/// Exponentially scaled Hankel function of the second kind.
///
/// Matches `scipy.special.hankel2e(v, z) = hankel2(v, z) * exp(i z)`.
pub fn hankel2e(v: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    scaled_hankel_dispatch("hankel2e", v, z, mode, HankelKind::H2, HankelScale::Second)
}

/// Wright's generalized Bessel function.
///
/// Matches `scipy.special.wright_bessel(a, b, x)` on SciPy's supported
/// nonnegative real domain.
pub fn wright_bessel(
    a: &SpecialTensor,
    b: &SpecialTensor,
    x: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    map_real_ternary("wright_bessel", a, b, x, mode, |av, bv, xv| {
        wright_bessel_scalar(av, bv, xv, mode)
    })
}

/// Logarithm of Wright's generalized Bessel function.
///
/// Matches `scipy.special.log_wright_bessel(a, b, x)` on SciPy's supported
/// nonnegative real domain while evaluating directly in log space.
pub fn log_wright_bessel(
    a: &SpecialTensor,
    b: &SpecialTensor,
    x: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    map_real_ternary("log_wright_bessel", a, b, x, mode, |av, bv, xv| {
        log_wright_bessel_scalar(av, bv, xv, mode)
    })
}

/// Derivative of the Bessel function `J_v(z)`.
///
/// Matches `scipy.special.jvp(v, z, n)` for real-valued inputs. The derivative
/// order `n` is represented by `derivative_order` and defaults to 1 in SciPy.
pub fn jvp(
    v: &SpecialTensor,
    z: &SpecialTensor,
    derivative_order: usize,
    mode: RuntimeMode,
) -> SpecialResult {
    bessel_derivative_dispatch(
        "jvp",
        v,
        z,
        derivative_order,
        mode,
        BesselKind::Jv,
        DerivativeRule::Alternating,
    )
}

/// Derivative of the Bessel function `Y_v(z)`.
///
/// Matches `scipy.special.yvp(v, z, n)` for real-valued inputs.
pub fn yvp(
    v: &SpecialTensor,
    z: &SpecialTensor,
    derivative_order: usize,
    mode: RuntimeMode,
) -> SpecialResult {
    bessel_derivative_dispatch(
        "yvp",
        v,
        z,
        derivative_order,
        mode,
        BesselKind::Yv,
        DerivativeRule::Alternating,
    )
}

/// Derivative of the modified Bessel function `I_v(z)`.
///
/// Matches `scipy.special.ivp(v, z, n)` for real-valued inputs.
pub fn ivp(
    v: &SpecialTensor,
    z: &SpecialTensor,
    derivative_order: usize,
    mode: RuntimeMode,
) -> SpecialResult {
    bessel_derivative_dispatch(
        "ivp",
        v,
        z,
        derivative_order,
        mode,
        BesselKind::Iv,
        DerivativeRule::Positive,
    )
}

/// Derivative of the modified Bessel function `K_v(z)`.
///
/// Matches `scipy.special.kvp(v, z, n)` for real-valued inputs.
pub fn kvp(
    v: &SpecialTensor,
    z: &SpecialTensor,
    derivative_order: usize,
    mode: RuntimeMode,
) -> SpecialResult {
    bessel_derivative_dispatch(
        "kvp",
        v,
        z,
        derivative_order,
        mode,
        BesselKind::Kv,
        DerivativeRule::NegativeByOrder,
    )
}

/// Derivative of the Hankel function `H_v^(1)(z)`.
///
/// Returns a complex tensor: `h1vp = jvp + i*yvp`.
pub fn h1vp(
    v: &SpecialTensor,
    z: &SpecialTensor,
    derivative_order: usize,
    mode: RuntimeMode,
) -> SpecialResult {
    hankel_derivative_dispatch("h1vp", v, z, derivative_order, mode, HankelKind::H1)
}

/// Derivative of the Hankel function `H_v^(2)(z)`.
///
/// Returns a complex tensor: `h2vp = jvp - i*yvp`.
pub fn h2vp(
    v: &SpecialTensor,
    z: &SpecialTensor,
    derivative_order: usize,
    mode: RuntimeMode,
) -> SpecialResult {
    hankel_derivative_dispatch("h2vp", v, z, derivative_order, mode, HankelKind::H2)
}

#[derive(Debug, Clone, Copy)]
enum DerivativeRule {
    Alternating,
    Positive,
    NegativeByOrder,
}

#[derive(Debug, Clone, Copy)]
enum HankelKind {
    H1,
    H2,
}

fn bessel_derivative_sum<F>(
    function: &'static str,
    order: f64,
    x: f64,
    derivative_order: usize,
    mode: RuntimeMode,
    rule: DerivativeRule,
    mut kernel: F,
) -> Result<f64, SpecialError>
where
    F: FnMut(f64, f64) -> Result<f64, SpecialError>,
{
    if derivative_order == 0 {
        return kernel(order, x);
    }
    if derivative_order > 128 {
        return domain_error_by_mode(
            function,
            mode,
            format!("n={derivative_order}"),
            "derivative order is too large for stable recurrence",
        );
    }

    let mut binom = 1.0;
    let mut sum = 0.0;
    for k in 0..=derivative_order {
        let shifted_order = order - derivative_order as f64 + 2.0 * k as f64;
        let mut term = binom * kernel(shifted_order, x)?;
        if matches!(rule, DerivativeRule::Alternating) && k % 2 == 1 {
            term = -term;
        }
        sum += term;

        if k < derivative_order {
            binom *= (derivative_order - k) as f64 / (k + 1) as f64;
        }
    }

    let scale = 0.5_f64.powi(derivative_order as i32);
    let sign = if matches!(rule, DerivativeRule::NegativeByOrder) && derivative_order % 2 == 1 {
        -1.0
    } else {
        1.0
    };
    Ok(sign * scale * sum)
}

fn bessel_derivative_sum_complex<F>(
    function: &'static str,
    order: f64,
    z: Complex64,
    derivative_order: usize,
    mode: RuntimeMode,
    rule: DerivativeRule,
    mut kernel: F,
) -> Result<Complex64, SpecialError>
where
    F: FnMut(f64, Complex64) -> Result<Complex64, SpecialError>,
{
    if derivative_order == 0 {
        return kernel(order, z);
    }
    if derivative_order > 128 {
        return complex_domain_error_by_mode(
            function,
            mode,
            format!("n={derivative_order}"),
            "derivative order is too large for stable recurrence",
        );
    }

    let mut binom = 1.0;
    let mut sum = Complex64::new(0.0, 0.0);
    for k in 0..=derivative_order {
        let shifted_order = order - derivative_order as f64 + 2.0 * k as f64;
        let mut term = kernel(shifted_order, z)? * binom;
        if matches!(rule, DerivativeRule::Alternating) && k % 2 == 1 {
            term = -term;
        }
        sum = sum + term;

        if k < derivative_order {
            binom *= (derivative_order - k) as f64 / (k + 1) as f64;
        }
    }

    let scale = 0.5_f64.powi(derivative_order as i32);
    let sign = if matches!(rule, DerivativeRule::NegativeByOrder) && derivative_order % 2 == 1 {
        -1.0
    } else {
        1.0
    };
    Ok(sum * (sign * scale))
}

fn bessel_derivative_dispatch(
    function: &'static str,
    v: &SpecialTensor,
    z: &SpecialTensor,
    derivative_order: usize,
    mode: RuntimeMode,
    kind: BesselKind,
    rule: DerivativeRule,
) -> SpecialResult {
    match (v, z) {
        (SpecialTensor::RealScalar(order), SpecialTensor::RealScalar(x)) => {
            bessel_derivative_real_scalar(function, *order, *x, derivative_order, mode, kind, rule)
                .map(SpecialTensor::RealScalar)
        }
        (SpecialTensor::RealVec(orders), SpecialTensor::RealScalar(x)) => orders
            .iter()
            .map(|&order| {
                bessel_derivative_real_scalar(
                    function,
                    order,
                    *x,
                    derivative_order,
                    mode,
                    kind,
                    rule,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        (SpecialTensor::RealScalar(order), SpecialTensor::RealVec(xs)) => xs
            .iter()
            .map(|&x| {
                bessel_derivative_real_scalar(
                    function,
                    *order,
                    x,
                    derivative_order,
                    mode,
                    kind,
                    rule,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        (SpecialTensor::RealVec(orders), SpecialTensor::RealVec(xs)) => {
            if orders.len() != xs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            orders
                .iter()
                .zip(xs.iter())
                .map(|(&order, &x)| {
                    bessel_derivative_real_scalar(
                        function,
                        order,
                        x,
                        derivative_order,
                        mode,
                        kind,
                        rule,
                    )
                })
                .collect::<Result<Vec<_>, _>>()
                .map(SpecialTensor::RealVec)
        }
        (SpecialTensor::RealScalar(order), SpecialTensor::ComplexScalar(z_val)) => {
            bessel_derivative_complex_scalar(
                function,
                *order,
                *z_val,
                derivative_order,
                mode,
                kind,
                rule,
            )
            .map(SpecialTensor::ComplexScalar)
        }
        (SpecialTensor::RealScalar(order), SpecialTensor::ComplexVec(zs)) => zs
            .iter()
            .map(|&z_val| {
                bessel_derivative_complex_scalar(
                    function,
                    *order,
                    z_val,
                    derivative_order,
                    mode,
                    kind,
                    rule,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::RealVec(orders), SpecialTensor::ComplexScalar(z_val)) => orders
            .iter()
            .map(|&order| {
                bessel_derivative_complex_scalar(
                    function,
                    order,
                    *z_val,
                    derivative_order,
                    mode,
                    kind,
                    rule,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::RealVec(orders), SpecialTensor::ComplexVec(zs)) => {
            if orders.len() != zs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            orders
                .iter()
                .zip(zs.iter())
                .map(|(&order, &z_val)| {
                    bessel_derivative_complex_scalar(
                        function,
                        order,
                        z_val,
                        derivative_order,
                        mode,
                        kind,
                        rule,
                    )
                })
                .collect::<Result<Vec<_>, _>>()
                .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::ComplexScalar(_), _) | (SpecialTensor::ComplexVec(_), _) => {
            not_yet_implemented(function, mode, "complex-valued order not supported")
        }
        (SpecialTensor::Empty, _) | (_, SpecialTensor::Empty) => Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "empty tensor is not a valid special-function input",
        }),
    }
}

fn bessel_derivative_real_scalar(
    function: &'static str,
    order: f64,
    x: f64,
    derivative_order: usize,
    mode: RuntimeMode,
    kind: BesselKind,
    rule: DerivativeRule,
) -> Result<f64, SpecialError> {
    // Exponentially-scaled kinds differentiate the product `B_v(x)·e^{σx}` via the Leibniz
    // rule, reusing the unscaled per-order recurrence and the closed-form derivatives of the
    // scaling factor:
    //
    //   d^n/dx^n [B_v(x)·e^{σx}] = e^{σx} · Σ_{j=0}^n C(n,j) · σ^{n-j} · B_v^{(j)}(x)
    //
    // Each `B_v^{(j)}` reuses `bessel_derivative_sum`, so it is bit-identical to the unscaled
    // derivative path (`jvp`/`yvp`/`ivp`/`kvp`); the scaling is layered on top in closed form.
    // Genuinely-unsupported domains stay fail-closed because the base kernel (e.g. `K_v` for
    // `x ≤ 0`) propagates its own error.
    if let Some((base_kind, base_rule, sigma)) = scaled_real_derivative_params(kind, x) {
        let mut binom = 1.0;
        let mut sum = 0.0;
        for j in 0..=derivative_order {
            let unscaled = bessel_derivative_sum(
                function,
                order,
                x,
                j,
                mode,
                base_rule,
                |shifted_order, value| match base_kind {
                    BesselKind::Jv => Ok(jv_scalar(shifted_order, value)),
                    BesselKind::Yv => yv_scalar(shifted_order, value, mode),
                    BesselKind::Iv => Ok(iv_scalar(shifted_order, value)),
                    BesselKind::Kv => kv_scalar(shifted_order, value, mode),
                    // Unreachable: `scaled_real_derivative_params` only yields unscaled kinds.
                    BesselKind::Jve | BesselKind::Yve | BesselKind::Ive | BesselKind::Kve => {
                        Err(SpecialError {
                            function,
                            kind: SpecialErrorKind::NotYetImplemented,
                            mode,
                            detail: "scaled base kind has no unscaled recurrence",
                        })
                    }
                },
            )?;
            sum += binom * sigma.powi((derivative_order - j) as i32) * unscaled;
            if j < derivative_order {
                binom *= (derivative_order - j) as f64 / (j + 1) as f64;
            }
        }
        return Ok((sigma * x).exp() * sum);
    }

    bessel_derivative_sum(
        function,
        order,
        x,
        derivative_order,
        mode,
        rule,
        |shifted_order, value| match kind {
            BesselKind::Jv => Ok(jv_scalar(shifted_order, value)),
            BesselKind::Yv => yv_scalar(shifted_order, value, mode),
            BesselKind::Iv => Ok(iv_scalar(shifted_order, value)),
            BesselKind::Kv => kv_scalar(shifted_order, value, mode),
            BesselKind::Jve | BesselKind::Yve | BesselKind::Ive | BesselKind::Kve => {
                // Intercepted above; kept as a defensive fail-closed arm.
                Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::NotYetImplemented,
                    mode,
                    detail: "scaled Bessel derivatives are not implemented",
                })
            }
        },
    )
}

/// For an exponentially-scaled real Bessel kind, return the unscaled base kind, its
/// derivative recurrence rule, and the scale exponent `σ` such that the scaling factor
/// on the real axis is `e^{σ·x}`. Returns `None` for unscaled kinds.
///
/// On the real axis `J_v`/`Y_v` are scaled by `e^{-|Im z|} = 1`, so their scaled
/// derivative coincides with the unscaled one (`σ = 0`). `ive(v, x) = I_v(x)·e^{-|x|}`
/// gives `σ = -sign(x)`, and `kve(v, x) = K_v(x)·e^{x}` (defined for `x > 0`) gives `σ = +1`.
fn scaled_real_derivative_params(
    kind: BesselKind,
    x: f64,
) -> Option<(BesselKind, DerivativeRule, f64)> {
    match kind {
        BesselKind::Jve => Some((BesselKind::Jv, DerivativeRule::Alternating, 0.0)),
        BesselKind::Yve => Some((BesselKind::Yv, DerivativeRule::Alternating, 0.0)),
        BesselKind::Ive => {
            let sigma = if x < 0.0 { 1.0 } else { -1.0 };
            Some((BesselKind::Iv, DerivativeRule::Positive, sigma))
        }
        BesselKind::Kve => Some((BesselKind::Kv, DerivativeRule::NegativeByOrder, 1.0)),
        BesselKind::Jv | BesselKind::Yv | BesselKind::Iv | BesselKind::Kv => None,
    }
}

fn bessel_derivative_complex_scalar(
    function: &'static str,
    order: f64,
    z: Complex64,
    derivative_order: usize,
    mode: RuntimeMode,
    kind: BesselKind,
    rule: DerivativeRule,
) -> Result<Complex64, SpecialError> {
    if z.im == 0.0 && z.re >= 0.0 {
        return bessel_derivative_real_scalar(
            function,
            order,
            z.re,
            derivative_order,
            mode,
            kind,
            rule,
        )
        .map(|value| Complex64::new(value, 0.0));
    }

    bessel_derivative_sum_complex(
        function,
        order,
        z,
        derivative_order,
        mode,
        rule,
        |shifted_order, value| bessel_complex_scalar(function, shifted_order, value, mode, kind),
    )
}

fn hankel_derivative_dispatch(
    function: &'static str,
    v: &SpecialTensor,
    z: &SpecialTensor,
    derivative_order: usize,
    mode: RuntimeMode,
    kind: HankelKind,
) -> SpecialResult {
    match (v, z) {
        (SpecialTensor::RealScalar(order), SpecialTensor::RealScalar(x)) => {
            hankel_derivative_real_scalar(function, *order, *x, derivative_order, mode, kind)
                .map(SpecialTensor::ComplexScalar)
        }
        (SpecialTensor::RealVec(orders), SpecialTensor::RealScalar(x)) => orders
            .iter()
            .map(|&order| {
                hankel_derivative_real_scalar(function, order, *x, derivative_order, mode, kind)
            })
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::RealScalar(order), SpecialTensor::RealVec(xs)) => xs
            .iter()
            .map(|&x| {
                hankel_derivative_real_scalar(function, *order, x, derivative_order, mode, kind)
            })
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::RealVec(orders), SpecialTensor::RealVec(xs)) => {
            if orders.len() != xs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            orders
                .iter()
                .zip(xs.iter())
                .map(|(&order, &x)| {
                    hankel_derivative_real_scalar(function, order, x, derivative_order, mode, kind)
                })
                .collect::<Result<Vec<_>, _>>()
                .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::RealScalar(order), SpecialTensor::ComplexScalar(z_val)) => {
            hankel_derivative_complex_scalar(function, *order, *z_val, derivative_order, mode, kind)
                .map(SpecialTensor::ComplexScalar)
        }
        (SpecialTensor::RealScalar(order), SpecialTensor::ComplexVec(zs)) => zs
            .iter()
            .map(|&z_val| {
                hankel_derivative_complex_scalar(
                    function,
                    *order,
                    z_val,
                    derivative_order,
                    mode,
                    kind,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::RealVec(orders), SpecialTensor::ComplexScalar(z_val)) => orders
            .iter()
            .map(|&order| {
                hankel_derivative_complex_scalar(
                    function,
                    order,
                    *z_val,
                    derivative_order,
                    mode,
                    kind,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::RealVec(orders), SpecialTensor::ComplexVec(zs)) => {
            if orders.len() != zs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            orders
                .iter()
                .zip(zs.iter())
                .map(|(&order, &z_val)| {
                    hankel_derivative_complex_scalar(
                        function,
                        order,
                        z_val,
                        derivative_order,
                        mode,
                        kind,
                    )
                })
                .collect::<Result<Vec<_>, _>>()
                .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::ComplexScalar(_), _) | (SpecialTensor::ComplexVec(_), _) => {
            not_yet_implemented(function, mode, "complex-valued order not supported")
        }
        (SpecialTensor::Empty, _) | (_, SpecialTensor::Empty) => Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "empty tensor is not a valid special-function input",
        }),
    }
}

fn hankel_derivative_real_scalar(
    function: &'static str,
    order: f64,
    x: f64,
    derivative_order: usize,
    mode: RuntimeMode,
    kind: HankelKind,
) -> Result<Complex64, SpecialError> {
    let real = bessel_derivative_sum(
        function,
        order,
        x,
        derivative_order,
        mode,
        DerivativeRule::Alternating,
        |shifted_order, value| Ok(jv_scalar(shifted_order, value)),
    )?;
    let imag = bessel_derivative_sum(
        function,
        order,
        x,
        derivative_order,
        mode,
        DerivativeRule::Alternating,
        |shifted_order, value| yv_scalar(shifted_order, value, mode),
    )?;
    Ok(match kind {
        HankelKind::H1 => Complex64::new(real, imag),
        HankelKind::H2 => Complex64::new(real, -imag),
    })
}

fn hankel_derivative_complex_scalar(
    function: &'static str,
    order: f64,
    z: Complex64,
    derivative_order: usize,
    mode: RuntimeMode,
    kind: HankelKind,
) -> Result<Complex64, SpecialError> {
    if z.im == 0.0 && z.re >= 0.0 {
        return hankel_derivative_real_scalar(function, order, z.re, derivative_order, mode, kind);
    }

    let j = bessel_derivative_sum_complex(
        function,
        order,
        z,
        derivative_order,
        mode,
        DerivativeRule::Alternating,
        |shifted_order, value| {
            bessel_complex_scalar(function, shifted_order, value, mode, BesselKind::Jv)
        },
    )?;
    let y = bessel_derivative_sum_complex(
        function,
        order,
        z,
        derivative_order,
        mode,
        DerivativeRule::Alternating,
        |shifted_order, value| {
            bessel_complex_scalar(function, shifted_order, value, mode, BesselKind::Yv)
        },
    )?;
    Ok(match kind {
        HankelKind::H1 => j + Complex64::new(-y.im, y.re),
        HankelKind::H2 => j + Complex64::new(y.im, -y.re),
    })
}

/// J_v(z) for real order v via power series.
fn jv_scalar(v: f64, z: f64) -> f64 {
    if z.is_nan() || v.is_nan() {
        return f64::NAN;
    }

    // Integer order: delegate to integer routines
    if v.fract() == 0.0 && v.abs() <= i32::MAX as f64 {
        let n = v as i32;
        return if n < 0 {
            let sign = if (-n) & 1 == 0 { 1.0 } else { -1.0 };
            sign * jn_nonnegative((-n) as u32, z)
        } else {
            jn_nonnegative(n as u32, z)
        };
    }

    if z == 0.0 {
        return if v > 0.0 { 0.0 } else { f64::INFINITY };
    }

    let az = z.abs();
    let av = v.abs();

    // Power series: J_v(z) = (z/2)^v Σ (-z²/4)^k / (k! Γ(v+k+1)). Its largest
    // term is ~e^z while |J_v| ~ z^{-1/2}, so it loses ~0.43·z digits to
    // cancellation once z grows — catastrophic for large z (the old z < 20+|v|
    // cutoff gave jv(30.5,50)=1.48 vs scipy -0.0084, and jv(200.5,198.5) ~1e29
    // off). Keep it only for small argument (z < 20, ≲9 digits lost); larger z
    // is covered by the recurrences/asymptotic below for any order.
    if az < 20.0 {
        return jv_series(v, z);
    }

    // z ≥ 20. The DLMF 10.17.3 asymptotic only converges for z > v² (term ratio
    // ~v²/(2z)); in the band z ≤ v² it diverges (jv(10.5,50) was 0.0296 vs
    // -0.0848). There J_v is reached by a stable Miller backward recurrence from
    // a small base order — valid for both the oscillatory z > |v| and the
    // recessive z ≤ |v| regions (the latter via the rescaling in jv_miller).
    // Negative order: J_{-p} = cos(pπ)J_p − sin(pπ)Y_p, Y_p from the (stable)
    // upward recurrence. frankenscipy-goaov / frankenscipy-87poa.
    if av < 1.0 || az > av * av {
        return jv_asymptotic(v, az);
    }
    let jav = jv_miller(av, az);
    if v > 0.0 {
        jav
    } else {
        let (cos_av, sin_av) = bessel_reflection_trig(av);
        cos_av * jav - sin_av * yv_upward(av, az)
    }
}

/// Power series for J_v(z).
fn jv_series(v: f64, z: f64) -> f64 {
    // For non-integer v and negative z, J_v(z) involves complex values
    if z < 0.0 && v.fract() != 0.0 {
        return f64::NAN;
    }

    let half_z_abs = (z / 2.0).abs();

    // J_v(z) = Σ_{k=0}^∞ (-1)^k (z/2)^{v+2k} / (k! Γ(v+k+1)).
    // log|term_k| evolves via the recurrence
    //   log|term_{k+1}| = log|term_k| + log(z²/4) - log(k+1) - log|v+k+1|.
    // sign(term_k) = (-1)^k · sign(Γ(v+k+1)). Pre-fix the recurrence used
    // log(v+k+1) which is NaN whenever v+k+1<0 — this killed yv at every
    // non-integer v > 1 (frankenscipy-6avjb).
    let log_first = v * half_z_abs.ln() - lgamma(v + 1.0);
    let mut sum = 0.0;
    let mut log_term = log_first;
    let mut gamma_sign = gamma_sign_fn(v + 1.0);

    for k in 0..200 {
        let term = log_term.exp();
        let alternating = if k % 2 == 0 { 1.0 } else { -1.0 };
        sum += alternating * gamma_sign * term;

        if term < 1e-16 * sum.abs().max(1e-300) && k > 5 {
            break;
        }

        let kf = k as f64;
        let v_k_1 = v + kf + 1.0;
        log_term += (z * z / 4.0).ln() - (kf + 1.0).ln() - v_k_1.abs().ln();
        // sign(Γ(v+k+2)) = sign(v+k+1) · sign(Γ(v+k+1)).
        if v_k_1 < 0.0 {
            gamma_sign = -gamma_sign;
        }
    }

    sum
}

/// Sign of Γ(x) for non-integer x. Returns +1 at non-negative arguments
/// (Γ(0) is taken as +∞ which the caller should guard) and 0 at negative
/// integer poles. For negative non-integer x, uses the reflection
/// identity sign(Γ(x)) = sign(sin(πx)).
fn gamma_sign_fn(x: f64) -> f64 {
    if x >= 0.0 {
        return 1.0;
    }
    if x.fract() == 0.0 {
        // Pole — caller is responsible for guarding before reaching here.
        return 0.0;
    }
    if (PI * x).sin() >= 0.0 { 1.0 } else { -1.0 }
}

/// Asymptotic expansion for J_v(z) for large |z|.
fn jv_asymptotic(v: f64, z: f64) -> f64 {
    // DLMF 10.17.3 large-z asymptotic for J_v(z):
    //   J_v(z) ~ sqrt(2/(πz)) [cos(ω) P(v,z) - sin(ω) Q(v,z)],  ω = z - vπ/2 - π/4,
    //   P = Σ_j (-1)^j a_{2j}/z^{2j},  Q = Σ_j (-1)^j a_{2j+1}/z^{2j+1},
    //   a_0 = 1,  a_k = a_{k-1}(4v² - (2k-1)²)/(8k).
    // The earlier leading-order form (P = 1, Q = 0) was accurate only to O(1/z),
    // leaving non-integer-order J_v — and the Y_v built from J_{±v} — ~1% off
    // SciPy at large z. Sum the (divergent) asymptotic series to its smallest
    // term. frankenscipy-rbmy5.
    let mu = 4.0 * v * v;
    let mut term = 1.0_f64; // a_k / z^k, a_0 = 1
    let mut prev_abs = 1.0_f64;
    let mut p = 1.0_f64; // k = 0 term of P
    let mut q = 0.0_f64;
    for k in 1..64 {
        let kf = k as f64;
        term *= (mu - (2.0 * kf - 1.0).powi(2)) / (8.0 * kf * z);
        let abs_term = term.abs();
        if abs_term > prev_abs {
            break; // asymptotic series past its smallest term — truncate
        }
        if k % 2 == 0 {
            let sign = if (k / 2) % 2 == 0 { 1.0 } else { -1.0 };
            p += sign * term;
        } else {
            let sign = if ((k - 1) / 2) % 2 == 0 { 1.0 } else { -1.0 };
            q += sign * term;
        }
        prev_abs = abs_term;
        if abs_term <= f64::EPSILON {
            break;
        }
    }
    let omega = z - v * PI / 2.0 - PI / 4.0;
    (FRAC_2_PI / z).sqrt() * (omega.cos() * p - omega.sin() * q)
}

/// Shared P, Q sums for the DLMF 10.17.3/10.17.4 large-z Bessel asymptotics.
/// Valid only when z is large relative to the order (z ≳ v²); the series is
/// divergent and summed to its smallest term.
fn bessel_asymptotic_pq(v: f64, z: f64) -> (f64, f64) {
    let mu = 4.0 * v * v;
    let mut term = 1.0_f64;
    let mut prev_abs = 1.0_f64;
    let mut p = 1.0_f64;
    let mut q = 0.0_f64;
    for k in 1..64 {
        let kf = k as f64;
        term *= (mu - (2.0 * kf - 1.0).powi(2)) / (8.0 * kf * z);
        let abs_term = term.abs();
        if abs_term > prev_abs {
            break;
        }
        if k % 2 == 0 {
            let sign = if (k / 2) % 2 == 0 { 1.0 } else { -1.0 };
            p += sign * term;
        } else {
            let sign = if ((k - 1) / 2) % 2 == 0 { 1.0 } else { -1.0 };
            q += sign * term;
        }
        prev_abs = abs_term;
        if abs_term <= f64::EPSILON {
            break;
        }
    }
    (p, q)
}

/// Y_v(z) large-z asymptotic (DLMF 10.17.4), companion to [`jv_asymptotic`].
/// Only valid for z ≳ v²; used here for the small base orders of the recurrence.
fn yv_asymptotic(v: f64, z: f64) -> f64 {
    let (p, q) = bessel_asymptotic_pq(v, z);
    let omega = z - v * PI / 2.0 - PI / 4.0;
    (FRAC_2_PI / z).sqrt() * (omega.sin() * p + omega.cos() * q)
}

/// J_v(z) by Miller's backward recurrence, for non-integer order av ≥ 1 with
/// av < z (the oscillatory transition band v < z ≤ v² where the plain large-z
/// asymptotic diverges). J is the recessive solution as order increases, so the
/// downward recurrence J_{ν−1} = (2ν/z)J_ν − J_{ν+1} is stable; the unnormalized
/// run is rescaled by the accurate small-order asymptotic value (whichever base
/// order has the larger magnitude, to avoid normalizing through a J zero).
fn jv_miller(av: f64, z: f64) -> f64 {
    let frac = av - av.floor();
    let n = (av - frac).round() as usize;
    // Start well above both the target order n and the argument z. The extra
    // 2·n margin (and the rescaling below) lets the recurrence also serve the
    // recessive region av ≥ z, where J grows steeply going downward from the
    // start order and would otherwise overflow before reaching the base.
    let m_start = n + z as usize + 60.max(2 * n);
    let mut jp1 = 0.0_f64;
    let mut jc = 1e-300_f64;
    let mut order = frac + m_start as f64;
    let mut j_at_n = 0.0_f64;
    let mut j_at_0 = 0.0_f64;
    let mut j_at_1 = 0.0_f64;
    let mut m = m_start as isize;
    while m >= 0 {
        let mu = m as usize;
        if mu == n {
            j_at_n = jc;
        }
        if m == 0 {
            j_at_0 = jc;
        }
        if m == 1 {
            j_at_1 = jc;
        }
        let jm1 = (2.0 * order / z) * jc - jp1;
        jp1 = jc;
        jc = jm1;
        order -= 1.0;
        m -= 1;
        // Rescale the whole running state (including the captured values, so
        // their ratios are preserved) when the recurrence grows too large.
        if jc.abs() > 1e250 {
            jc *= 1e-250;
            jp1 *= 1e-250;
            j_at_n *= 1e-250;
            j_at_0 *= 1e-250;
            j_at_1 *= 1e-250;
        }
    }
    let t0 = jv_asymptotic(frac, z);
    let t1 = jv_asymptotic(frac + 1.0, z);
    // Normalize against whichever base order has the larger magnitude, so we
    // never divide through a near-zero of J.
    let scale = if j_at_0.abs() >= j_at_1.abs() {
        t0 / j_at_0
    } else {
        t1 / j_at_1
    };
    j_at_n * scale
}

/// cos(avπ), sin(avπ) for the negative-order reflection, with cos forced to an
/// exact zero at half-integer order. There cos(avπ) is mathematically 0 but
/// rounds to ~1e-15; multiplied by a huge Y_av (order ≫ argument) that spurious
/// term would otherwise swamp the true (tiny) value.
fn bessel_reflection_trig(av: f64) -> (f64, f64) {
    let frac = av - av.floor();
    let cos_av = if (frac - 0.5).abs() < 1e-12 {
        0.0
    } else {
        (av * PI).cos()
    };
    (cos_av, (av * PI).sin())
}

/// Y_v(z) by upward recurrence for non-integer order av ≥ 1. Y is the dominant
/// solution as order increases, so the upward recurrence
/// Y_{ν+1} = (2ν/z)Y_ν − Y_{ν−1} is stable for any av (including the recessive
/// av ≥ z region, where Y grows large); the base orders use [`yv_asymptotic`].
fn yv_upward(av: f64, z: f64) -> f64 {
    let frac = av - av.floor();
    let n = (av - frac).round() as usize;
    if n == 0 {
        return yv_asymptotic(frac, z);
    }
    let mut ym1 = yv_asymptotic(frac, z);
    let mut ym = yv_asymptotic(frac + 1.0, z);
    let mut order = frac + 1.0;
    for _ in 0..(n - 1) {
        let next = (2.0 * order / z) * ym - ym1;
        ym1 = ym;
        ym = next;
        order += 1.0;
    }
    ym
}

/// Y_v(z) for real order v.
pub(crate) fn yv_scalar(v: f64, z: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if v.is_nan() || z.is_nan() {
        return Ok(f64::NAN);
    }
    if z <= 0.0 {
        if z == 0.0 {
            return Ok(f64::NEG_INFINITY);
        }
        return domain_error_by_mode(
            "yv",
            mode,
            format!("v={v},z={z}"),
            "yv requires z > 0 for real-valued output",
        );
    }

    // Integer order: delegate
    if v.fract() == 0.0 && v.abs() <= i32::MAX as f64 {
        let n = v as i32;
        return yn_scalar(n as f64, z, mode);
    }

    // Large order in the band z ≤ v²: the J_{±v} reflection below cancels
    // catastrophically because for z ≤ |v| the J_{-v} term carries the huge
    // dominant Y component (yv(100.5,95.5) was ~1000× off). Compute Y directly
    // by its (stable) upward recurrence instead; negative order via
    // Y_{-p} = cos(pπ)Y_p + sin(pπ)J_p, with cos forced to 0 at half-integers so
    // a ~1e-15 rounding does not swamp a tiny true value. frankenscipy-87poa.
    let av = v.abs();
    if av >= 1.0 && z >= 20.0 && z <= av * av {
        let yav = yv_upward(av, z);
        if v > 0.0 {
            return Ok(yav);
        }
        let (cos_av, sin_av) = bessel_reflection_trig(av);
        return Ok(cos_av * yav + sin_av * jv_miller(av, z));
    }

    // Non-integer order: Y_v = (J_v cos(vπ) - J_{-v}) / sin(vπ)
    let sin_vpi = (v * PI).sin();
    if sin_vpi.abs() < 1e-15 {
        // Near integer order — should have been caught above
        return Ok(f64::NAN);
    }

    let jv_pos = jv_scalar(v, z);
    let jv_neg = jv_scalar(-v, z);
    Ok((jv_pos * (v * PI).cos() - jv_neg) / sin_vpi)
}

/// I_v(z) for real order v via power series.
fn iv_scalar(v: f64, z: f64) -> f64 {
    if z.is_nan() || v.is_nan() {
        return f64::NAN;
    }
    let az = z.abs();
    if az == 0.0 {
        return if v == 0.0 {
            1.0
        } else if v > 0.0 {
            0.0
        } else if v.fract() == 0.0 {
            // I_{-n}(0) = I_n(0)
            if v.abs() == 0.0 { 1.0 } else { 0.0 }
        } else {
            f64::INFINITY
        };
    }

    if v < 0.0 && v.fract() == 0.0 {
        return iv_scalar(v.abs(), z);
    }
    // Negative non-integer order: the power-series term Γ(v+k+1) passes through
    // ln(v+k+1) for v+k+1 <= 0 (any v <= -1), producing NaN. SciPy instead uses
    // the reflection identity I_{-p}(z) = I_p(z) + (2/π) sin(pπ) K_p(z) with p=|v|.
    // K_v is symmetric (K_p = K_{|v|}); for z < 0 the result is complex, matching
    // the NaN returned by the power-series branch below.
    if v < 0.0 && v.fract() != 0.0 && z > 0.0 {
        let p = -v;
        let kp = kv_scaled_value(p, z) * (-z).exp();
        return iv_scalar(p, z) + (2.0 / PI) * (p * PI).sin() * kp;
    }

    // The large-ARGUMENT asymptotic I_v(z) ~ e^z/√(2πz)·Σ(4v²-…)/(8z)^k is only
    // valid (and convergent under optimal truncation) when z is large relative
    // to the order: its term ratio ~ v²/(2z), so for z ≲ v² the series diverges
    // and produced huge NEGATIVE garbage (iv(50,100) was -8.3e44 vs scipy
    // 4.8e36). Require z > v² so the asymptotic is firmly in its valid regime;
    // otherwise fall through to the (everywhere-valid) ascending power series.
    if az > 50.0 && az > v * v {
        return iv_asymptotic(v, az);
    }

    // Power series: I_v(z) = (z/2)^v Σ (z²/4)^k / (k! Γ(v+k+1))
    let half_z = az / 2.0;
    let quarter_z2 = az * az / 4.0;

    let log_first = v * half_z.ln() - lgamma(v + 1.0);
    let mut sum = 0.0;
    let mut log_term = log_first;

    // The summand peaks near k ≈ (√(v²+z²) − v)/2, which reaches a few hundred
    // for large order with z ≲ v² (iv(100,500) peaks at ~305 terms); cap well
    // past that so the tail is captured before the relative-convergence break.
    for k in 0..1000 {
        let term = log_term.exp();
        sum += term;

        if term < 1e-16 * sum && k > 10 {
            break;
        }

        let kf = k as f64;
        log_term += quarter_z2.ln() - (kf + 1.0).ln() - (v + kf + 1.0).ln();
    }

    // Parity for negative z: I_v(-z) = (-1)^v I_v(z) for integer v
    if z < 0.0 {
        if v.fract() == 0.0 {
            let n = v.abs() as i64;
            if n % 2 != 0 {
                return -sum;
            }
        } else {
            return f64::NAN; // Complex for non-integer v and negative z
        }
    }

    sum
}

fn iv_asymptotic(v: f64, z: f64) -> f64 {
    // I_v(z) ~ e^z / sqrt(2*pi*z) * [ 1 - (4v^2-1)/8z + (4v^2-1)(4v^2-9)/(2! (8z)^2) - ... ]
    let mu = 4.0 * v * v;
    let mut sum = 1.0;
    let mut term = 1.0;
    let iz8 = 1.0 / (8.0 * z);

    for k in 1..20 {
        let kf = k as f64;
        term *= -(mu - (2.0 * kf - 1.0).powi(2)) * iz8 / kf;
        sum += term;
        if term.abs() < 1e-14 * sum.abs() {
            break;
        }
    }

    (z * 2.0 * PI).sqrt().recip() * z.exp() * sum
}

/// K_v(z) for real order v.
fn kv_scalar(v: f64, z: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if v.is_nan() || z.is_nan() {
        return Ok(f64::NAN);
    }
    if z <= 0.0 {
        if z == 0.0 {
            return Ok(f64::INFINITY);
        }
        return domain_error_by_mode("kv", mode, format!("v={v},z={z}"), "kv requires z > 0");
    }

    // K_v = e^{-z} · (K_v·e^z). The scaled value is computed without ever forming
    // the tiny K_v directly; for z ≳ 745 the e^{-z} underflows to 0, matching
    // SciPy (kve stays finite via kve_scalar). K_v is symmetric: K_{-v} = K_v.
    Ok(kv_scaled_value(v.abs(), z) * (-z).exp())
}

/// Exponentially scaled K_v: returns K_v(z)·e^z for z > 0, staying O(1)/finite
/// for all z. Underpins both kv (×e^{-z}) and kve (directly), avoiding the
/// underflow/overflow round-trip and the tiny-magnitude integral that defeated
/// the absolute-tolerance quadrature. frankenscipy-j3bw7.
fn kv_scaled_value(v_abs: f64, z: f64) -> f64 {
    // Large z relative to v²: DLMF 10.40.2 asymptotic (scaled form, no e^{-z}).
    if z >= 30.0 && z >= 0.5 * v_abs * v_abs {
        return kv_asymptotic_scaled(v_abs, z);
    }
    // Non-integer order: scaled integral directly.
    if v_abs.fract() != 0.0 {
        return kv_integral_scaled(v_abs, z);
    }
    // Integer order: the recurrence K_{n+1} = K_{n-1} + (2n/z)K_n is identical for
    // the e^z-scaled values, so build them from the scaled K_0, K_1.
    let k0 = kv_integral_scaled(0.0, z);
    let n = v_abs.round() as u32;
    if n == 0 {
        return k0;
    }
    let k1 = kv_integral_scaled(1.0, z);
    if n == 1 {
        return k1;
    }
    let mut k_prev = k0;
    let mut k_curr = k1;
    for i in 1..n {
        let k_next = k_prev + 2.0 * i as f64 / z * k_curr;
        k_prev = k_curr;
        k_curr = k_next;
    }
    k_curr
}

/// Scaled DLMF 10.40.2 asymptotic: K_v(z)·e^z ~ sqrt(π/(2z)) Σ_k a_k/z^k,
/// a_0 = 1, a_k = a_{k-1}(4v² − (2k-1)²)/(8k) (all-positive). The series is
/// asymptotic (divergent); sum to its smallest term.
fn kv_asymptotic_scaled(v: f64, z: f64) -> f64 {
    let mu = 4.0 * v * v;
    let mut term = 1.0_f64; // a_k / z^k, a_0 = 1
    let mut prev_abs = 1.0_f64;
    let mut sum = 1.0_f64;
    for k in 1..64 {
        let kf = k as f64;
        term *= (mu - (2.0 * kf - 1.0).powi(2)) / (8.0 * kf * z);
        let abs_term = term.abs();
        if abs_term > prev_abs {
            break; // asymptotic series past its smallest term — truncate
        }
        sum += term;
        prev_abs = abs_term;
        if abs_term <= f64::EPSILON {
            break;
        }
    }
    (PI / (2.0 * z)).sqrt() * sum
}

/// Scaled integral form: K_v(z)·e^z = ∫_0^∞ e^{-z(cosh t − 1)} cosh(vt) dt.
/// Factoring e^{-z} out of the integrand makes it O(1) (peak ≈ 1) so the
/// absolute-tolerance adaptive Simpson resolves it correctly — the unfactored
/// integrand was ≈ e^{-z}, defeating the tolerance and giving the coarse value
/// of a spike. The integrand's saddle sits at t* = asinh(v/z) (0 for v = 0); we
/// split the interval there and extend the upper limit until the exponent has
/// fallen ~50 below the peak.
fn kv_integral_scaled(v: f64, z: f64) -> f64 {
    let t_star = (v / z).asinh();
    // Exponent φ(t) = z(cosh t − 1) − v t; integrand ≈ e^{−(φ(t)−φ(t*))} near peak.
    let phi = |t: f64| z * (t.cosh() - 1.0) - v * t;
    let base = phi(t_star);
    let mut upper = t_star + 1.0;
    while phi(upper) - base < 50.0 && upper < 40.0 {
        upper += 0.5;
    }
    let integrand = |t: f64| (-z * (t.cosh() - 1.0)).exp() * (v * t).cosh();
    if t_star > 1.0e-9 && t_star < upper {
        adaptive_simpson(&integrand, 0.0, t_star, 1.0e-13, 24)
            + adaptive_simpson(&integrand, t_star, upper, 1.0e-13, 24)
    } else {
        adaptive_simpson(&integrand, 0.0, upper, 1.0e-13, 24)
    }
}

fn adaptive_simpson(f: &impl Fn(f64) -> f64, a: f64, b: f64, tol: f64, depth: u32) -> f64 {
    let fa = f(a);
    let fb = f(b);
    let c = 0.5 * (a + b);
    let fc = f(c);
    let whole = simpson_estimate(a, b, fa, fb, fc);
    adaptive_simpson_inner(f, a, b, tol, whole, (fa, fb, fc), depth)
}

fn adaptive_simpson_inner(
    f: &impl Fn(f64) -> f64,
    a: f64,
    b: f64,
    tol: f64,
    whole: f64,
    samples: (f64, f64, f64),
    depth: u32,
) -> f64 {
    let (fa, fb, fc) = samples;
    let c = 0.5 * (a + b);
    let left_mid = 0.5 * (a + c);
    let right_mid = 0.5 * (c + b);
    let f_left_mid = f(left_mid);
    let f_right_mid = f(right_mid);
    let left = simpson_estimate(a, c, fa, fc, f_left_mid);
    let right = simpson_estimate(c, b, fc, fb, f_right_mid);
    let delta = left + right - whole;

    if depth == 0 || delta.abs() <= 15.0 * tol {
        return left + right + delta / 15.0;
    }

    adaptive_simpson_inner(f, a, c, tol / 2.0, left, (fa, fc, f_left_mid), depth - 1)
        + adaptive_simpson_inner(f, c, b, tol / 2.0, right, (fc, fb, f_right_mid), depth - 1)
}

fn simpson_estimate(a: f64, b: f64, fa: f64, fb: f64, fm: f64) -> f64 {
    (b - a) * (fa + 4.0 * fm + fb) / 6.0
}

/// Log-gamma function for Bessel series computation.
fn lgamma(x: f64) -> f64 {
    if x <= 0.0 && x.fract().abs() < 1e-14 {
        return f64::INFINITY;
    }
    // Stirling's approximation with Lanczos correction
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    const G: f64 = 7.0;

    if x < 0.5 {
        // Reflection formula: lgamma(x) = ln(π/|sin(πx)|) - lgamma(1-x)
        let sin_px = (PI * x).sin();
        if sin_px.abs() < 1e-300 {
            return f64::INFINITY; // pole
        }
        return (PI / sin_px.abs()).ln() - lgamma(1.0 - x);
    }

    let z = x - 1.0;
    let mut s = COEFFS[0];
    for (idx, coeff) in COEFFS.iter().enumerate().skip(1) {
        s += coeff / (z + idx as f64);
    }
    let t = z + G + 0.5;
    0.5 * (2.0 * PI).ln() + (z + 0.5) * t.ln() - t + s.ln()
}

// ── Spherical Bessel functions ──────────────────────────────────────

/// Spherical Bessel function of the first kind j_n(z).
///
/// j_n(z) = √(π/(2z)) J_{n+1/2}(z)
///
/// Computed via recurrence from:
///   j_0(z) = sin(z)/z,  j_1(z) = sin(z)/z² - cos(z)/z
pub fn spherical_jn(n: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    spherical_bessel_dispatch("spherical_jn", n, z, mode, SphericalKind::Jn)
}

/// Spherical Bessel function of the second kind y_n(z).
///
/// y_n(z) = √(π/(2z)) Y_{n+1/2}(z)
///
/// Computed via recurrence from:
///   y_0(z) = -cos(z)/z,  y_1(z) = -cos(z)/z² - sin(z)/z
pub fn spherical_yn(n: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    spherical_bessel_dispatch("spherical_yn", n, z, mode, SphericalKind::Yn)
}

/// Modified spherical Bessel function of the first kind i_n(z).
///
/// i_n(z) = √(π/(2z)) I_{n+1/2}(z)
///
/// Computed via recurrence from:
///   i_0(z) = sinh(z)/z,  i_1(z) = cosh(z)/z - sinh(z)/z²
pub fn spherical_in(n: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    spherical_bessel_dispatch("spherical_in", n, z, mode, SphericalKind::In)
}

/// Modified spherical Bessel function of the second kind k_n(z).
///
/// k_n(z) = √(π/(2z)) K_{n+1/2}(z)
///
/// Computed via recurrence from:
///   k_0(z) = π exp(-z)/(2z),  k_1(z) = π exp(-z)/(2z) * (1 + 1/z)
pub fn spherical_kn(n: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    spherical_bessel_dispatch("spherical_kn", n, z, mode, SphericalKind::Kn)
}

#[derive(Clone, Copy)]
enum SphericalKind {
    Jn,
    Yn,
    In,
    Kn,
}

fn spherical_bessel_dispatch(
    function: &'static str,
    n: &SpecialTensor,
    z: &SpecialTensor,
    mode: RuntimeMode,
    kind: SphericalKind,
) -> SpecialResult {
    match (n, z) {
        // Real-real cases - use existing scalar implementations
        (SpecialTensor::RealScalar(order), SpecialTensor::RealScalar(x)) => {
            let result = match kind {
                SphericalKind::Jn => spherical_jn_scalar(*order, *x, mode),
                SphericalKind::Yn => spherical_yn_scalar(*order, *x, mode),
                SphericalKind::In => spherical_in_scalar(*order, *x, mode),
                SphericalKind::Kn => spherical_kn_scalar(*order, *x, mode),
            };
            result.map(SpecialTensor::RealScalar)
        }
        (SpecialTensor::RealVec(orders), SpecialTensor::RealScalar(x)) => orders
            .iter()
            .map(|&order| match kind {
                SphericalKind::Jn => spherical_jn_scalar(order, *x, mode),
                SphericalKind::Yn => spherical_yn_scalar(order, *x, mode),
                SphericalKind::In => spherical_in_scalar(order, *x, mode),
                SphericalKind::Kn => spherical_kn_scalar(order, *x, mode),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        (SpecialTensor::RealScalar(order), SpecialTensor::RealVec(xs)) => xs
            .iter()
            .map(|&x| match kind {
                SphericalKind::Jn => spherical_jn_scalar(*order, x, mode),
                SphericalKind::Yn => spherical_yn_scalar(*order, x, mode),
                SphericalKind::In => spherical_in_scalar(*order, x, mode),
                SphericalKind::Kn => spherical_kn_scalar(*order, x, mode),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        (SpecialTensor::RealVec(orders), SpecialTensor::RealVec(xs)) => {
            if orders.len() != xs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            orders
                .iter()
                .zip(xs.iter())
                .map(|(&order, &x)| match kind {
                    SphericalKind::Jn => spherical_jn_scalar(order, x, mode),
                    SphericalKind::Yn => spherical_yn_scalar(order, x, mode),
                    SphericalKind::In => spherical_in_scalar(order, x, mode),
                    SphericalKind::Kn => spherical_kn_scalar(order, x, mode),
                })
                .collect::<Result<Vec<_>, _>>()
                .map(SpecialTensor::RealVec)
        }
        // Real n, complex z - use complex implementations
        (SpecialTensor::RealScalar(order), SpecialTensor::ComplexScalar(z_val)) => {
            spherical_bessel_complex_scalar(function, *order, *z_val, mode, kind)
                .map(SpecialTensor::ComplexScalar)
        }
        (SpecialTensor::RealScalar(order), SpecialTensor::ComplexVec(zs)) => zs
            .iter()
            .map(|&z_val| spherical_bessel_complex_scalar(function, *order, z_val, mode, kind))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::RealVec(orders), SpecialTensor::ComplexScalar(z_val)) => orders
            .iter()
            .map(|&order| spherical_bessel_complex_scalar(function, order, *z_val, mode, kind))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::RealVec(orders), SpecialTensor::ComplexVec(zs)) => {
            if orders.len() != zs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            orders
                .iter()
                .zip(zs.iter())
                .map(|(&order, &z_val)| {
                    spherical_bessel_complex_scalar(function, order, z_val, mode, kind)
                })
                .collect::<Result<Vec<_>, _>>()
                .map(SpecialTensor::ComplexVec)
        }
        // Complex n - not supported
        (SpecialTensor::ComplexScalar(_), _) | (SpecialTensor::ComplexVec(_), _) => {
            not_yet_implemented(function, mode, "complex-valued order not supported")
        }
        // Empty tensor
        (SpecialTensor::Empty, _) | (_, SpecialTensor::Empty) => Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "empty tensor is not a valid special-function input",
        }),
    }
}

fn spherical_bessel_complex_scalar(
    function: &'static str,
    order: f64,
    z: Complex64,
    mode: RuntimeMode,
    kind: SphericalKind,
) -> Result<Complex64, SpecialError> {
    if order.is_nan() || !z.is_finite() {
        return Ok(Complex64::new(f64::NAN, f64::NAN));
    }

    let n = match integer_order_or_nan(function, order, mode)? {
        Some(v) => v,
        None => return Ok(Complex64::new(f64::NAN, f64::NAN)),
    };

    if n < 0 {
        return complex_domain_error_by_mode(
            function,
            mode,
            format!("n={n}"),
            "spherical Bessel order must be non-negative",
        );
    }

    Ok(match kind {
        SphericalKind::Jn => complex_spherical_jn(n as u32, z),
        SphericalKind::Yn => complex_spherical_yn(n as u32, z),
        SphericalKind::In => complex_spherical_in(n as u32, z),
        SphericalKind::Kn => complex_spherical_kn(n as u32, z),
    })
}

fn spherical_jn_scalar(order: f64, x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if order.is_nan() || x.is_nan() {
        return Ok(f64::NAN);
    }
    let n = match integer_order_or_nan("spherical_jn", order, mode)? {
        Some(v) => v,
        None => return Ok(f64::NAN),
    };
    if n < 0 {
        return domain_error_by_mode(
            "spherical_jn",
            mode,
            format!("n={n}"),
            "spherical Bessel order must be non-negative",
        );
    }
    Ok(spherical_jn_nonneg(n as u32, x))
}

fn spherical_yn_scalar(order: f64, x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if order.is_nan() || x.is_nan() {
        return Ok(f64::NAN);
    }
    if x <= 0.0 {
        if x == 0.0 {
            return Ok(f64::NEG_INFINITY);
        }
        return domain_error_by_mode(
            "spherical_yn",
            mode,
            format!("x={x}"),
            "spherical_yn requires x > 0",
        );
    }
    let n = match integer_order_or_nan("spherical_yn", order, mode)? {
        Some(v) => v,
        None => return Ok(f64::NAN),
    };
    if n < 0 {
        return domain_error_by_mode(
            "spherical_yn",
            mode,
            format!("n={n}"),
            "spherical Bessel order must be non-negative",
        );
    }
    Ok(spherical_yn_nonneg(n as u32, x))
}

fn spherical_in_scalar(order: f64, x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if order.is_nan() || x.is_nan() {
        return Ok(f64::NAN);
    }
    let n = match integer_order_or_nan("spherical_in", order, mode)? {
        Some(v) => v,
        None => return Ok(f64::NAN),
    };
    if n < 0 {
        return domain_error_by_mode(
            "spherical_in",
            mode,
            format!("n={n}"),
            "spherical Bessel order must be non-negative",
        );
    }
    Ok(spherical_in_nonneg(n as u32, x))
}

fn spherical_kn_scalar(order: f64, x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if order.is_nan() || x.is_nan() {
        return Ok(f64::NAN);
    }
    if x <= 0.0 {
        if x == 0.0 {
            return Ok(f64::INFINITY);
        }
        return domain_error_by_mode(
            "spherical_kn",
            mode,
            format!("x={x}"),
            "spherical_kn requires x > 0",
        );
    }
    let n = match integer_order_or_nan("spherical_kn", order, mode)? {
        Some(v) => v,
        None => return Ok(f64::NAN),
    };
    if n < 0 {
        return domain_error_by_mode(
            "spherical_kn",
            mode,
            format!("n={n}"),
            "spherical Bessel order must be non-negative",
        );
    }
    Ok(spherical_kn_nonneg(n as u32, x))
}

/// j_0(z) = sin(z)/z, j_1(z) = sin(z)/z² - cos(z)/z
/// Forward recurrence: j_{k+1}(z) = (2k+1)/z * j_k(z) - j_{k-1}(z)
fn spherical_jn_nonneg(n: u32, x: f64) -> f64 {
    if x.is_infinite() {
        return 0.0;
    }
    if x == 0.0 {
        return if n == 0 { 1.0 } else { 0.0 };
    }
    let ax = x.abs();
    if ax < 1e-15 {
        // Small-x Taylor: j_n(x) ≈ x^n / (2n+1)!!
        return if n == 0 { 1.0 } else { 0.0 };
    }

    // For x >= n, forward recurrence is stable; below that threshold,
    // forward recurrence catastrophically cancels (e.g., x=0.1, n=6
    // gives the wrong sign, n=9 yields O(1) results when the true
    // value is ~1e-19). Switch to Miller's downward recurrence in
    // that regime — it's unconditionally stable for j_n. Resolves
    // [frankenscipy-xlci0].
    let n_f = n as f64;
    if n >= 2 && ax < n_f {
        // Start the backward recurrence well above the order so
        // truncation error decays exponentially down to k=0.
        // m_start needs enough headroom that the truncation error from
        // setting j_{m_start+1} = 0 decays away by the time we reach
        // k=n. For small x, j_{m+1}/j_m ~ x/(2m+3); pick m_start so
        // that 30+ recurrence steps separate it from n.
        let m_start = (2 * n + 30).max((n_f + 8.0 * (40.0 * n_f).sqrt().ceil()) as u32);
        let mut j_kplus1 = 0.0_f64;
        let mut j_k = 1.0e-30_f64;
        let mut j_at_n = 0.0_f64;
        // Iterate k = m_start down to 1, computing j_{k-1} each step;
        // after the loop j_k holds j_0 (un-normalized). Rescale on
        // overflow — Miller's recurrence grows by ~(2k+1)/x per step
        // for small x and would overflow f64 within a few dozen orders.
        for k in (1..=m_start).rev() {
            let kf = k as f64;
            let j_kminus1 = (2.0 * kf + 1.0) / x * j_k - j_kplus1;
            if k == n {
                j_at_n = j_k;
            }
            j_kplus1 = j_k;
            j_k = j_kminus1;
            if j_k.abs() > 1.0e100 {
                let scale = 1.0e-100;
                j_k *= scale;
                j_kplus1 *= scale;
                j_at_n *= scale;
            }
        }
        // Normalize using the analytic j_0 = sin(x)/x.
        let true_j0 = x.sin() / x;
        return j_at_n * (true_j0 / j_k);
    }

    let mut j_prev = x.sin() / x; // j_0
    if n == 0 {
        return j_prev;
    }
    let mut j_curr = x.sin() / (x * x) - x.cos() / x; // j_1
    if n == 1 {
        return j_curr;
    }

    for k in 1..n {
        let next = (2.0 * k as f64 + 1.0) / x * j_curr - j_prev;
        j_prev = j_curr;
        j_curr = next;
    }
    j_curr
}

/// y_0(z) = -cos(z)/z, y_1(z) = -cos(z)/z² - sin(z)/z
/// Forward recurrence: y_{k+1}(z) = (2k+1)/z * y_k(z) - y_{k-1}(z)
fn spherical_yn_nonneg(n: u32, x: f64) -> f64 {
    if x.is_infinite() {
        return 0.0;
    }
    let mut y_prev = -x.cos() / x; // y_0
    if n == 0 {
        return y_prev;
    }
    let mut y_curr = -x.cos() / (x * x) - x.sin() / x; // y_1
    if n == 1 {
        return y_curr;
    }

    for k in 1..n {
        let next = (2.0 * k as f64 + 1.0) / x * y_curr - y_prev;
        y_prev = y_curr;
        y_curr = next;
    }
    y_curr
}

/// i_0(z) = sinh(z)/z, i_1(z) = cosh(z)/z - sinh(z)/z²
/// Forward recurrence: i_{k+1}(z) = i_{k-1}(z) - (2k+1)/z * i_k(z),
/// stable when |z| ≥ k. For small z the forward recurrence
/// catastrophically cancels (i_n decays super-exponentially while
/// (2k+1)/z grows), so we switch to Miller's downward recurrence
/// exactly as in spherical_jn_nonneg. Resolves [frankenscipy-0j009].
fn spherical_in_nonneg(n: u32, x: f64) -> f64 {
    if x == 0.0 {
        return if n == 0 { 1.0 } else { 0.0 };
    }
    if x.is_infinite() {
        return f64::INFINITY;
    }
    let ax = x.abs();

    // Miller's downward recurrence in the small-x regime.
    let n_f = n as f64;
    if n >= 2 && ax < n_f {
        let m_start = (2 * n + 30).max((n_f + 8.0 * (40.0 * n_f).sqrt().ceil()) as u32);
        let mut i_kplus1 = 0.0_f64;
        let mut i_k = 1.0e-30_f64;
        let mut i_at_n = 0.0_f64;
        // i_{k-1} = (2k+1)/x · i_k + i_{k+1}
        for k in (1..=m_start).rev() {
            let kf = k as f64;
            let i_kminus1 = (2.0 * kf + 1.0) / ax * i_k + i_kplus1;
            if k == n {
                i_at_n = i_k;
            }
            i_kplus1 = i_k;
            i_k = i_kminus1;
            if i_k.abs() > 1.0e100 {
                let scale = 1.0e-100;
                i_k *= scale;
                i_kplus1 *= scale;
                i_at_n *= scale;
            }
        }
        // Normalize using the analytic i_0(x) = sinh(x)/x.
        let true_i0 = ax.sinh() / ax;
        let result = i_at_n * (true_i0 / i_k);
        return if x < 0.0 && !n.is_multiple_of(2) {
            -result
        } else {
            result
        };
    }

    let sh = ax.sinh();
    let ch = ax.cosh();

    let mut i_prev = sh / ax; // i_0
    if n == 0 {
        return i_prev;
    }
    let mut i_curr = ch / ax - sh / (ax * ax); // i_1
    if n == 1 {
        // For negative x, i_n(-x) = (-1)^n * i_n(x)
        return if x < 0.0 { -i_curr } else { i_curr };
    }

    for k in 1..n {
        let next = i_prev - (2.0 * k as f64 + 1.0) / ax * i_curr;
        i_prev = i_curr;
        i_curr = next;
    }
    // Parity: i_n(-x) = (-1)^n i_n(x)
    if x < 0.0 && !n.is_multiple_of(2) {
        -i_curr
    } else {
        i_curr
    }
}

/// k_0(z) = π exp(-z)/(2z), k_1(z) = π exp(-z)/(2z) * (1 + 1/z)
/// Recurrence: k_{k+1}(z) = k_{k-1}(z) + (2k+1)/z * k_k(z)
fn spherical_kn_nonneg(n: u32, x: f64) -> f64 {
    if x.is_infinite() {
        return 0.0;
    }
    let emx = (-x).exp();
    let scale = PI / 2.0;
    let mut k_prev = scale * emx / x; // k_0
    if n == 0 {
        return k_prev;
    }
    let mut k_curr = scale * emx / x * (1.0 + 1.0 / x); // k_1
    if n == 1 {
        return k_curr;
    }

    for k in 1..n {
        let next = k_prev + (2.0 * k as f64 + 1.0) / x * k_curr;
        k_prev = k_curr;
        k_curr = next;
    }
    k_curr
}

// ============================================================================
// Complex spherical Bessel functions
// ============================================================================

/// Complex spherical Bessel function of the first kind j_n(z).
///
/// Uses Miller's downward recurrence in the |z| < n regime to avoid
/// the same forward-recurrence catastrophic cancellation that affects
/// the real-valued spherical_jn (resolves [frankenscipy-tt2v2]).
fn complex_spherical_jn(n: u32, z: Complex64) -> Complex64 {
    if z.re == 0.0 && z.im == 0.0 {
        return if n == 0 {
            Complex64::new(1.0, 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        };
    }

    let sin_z = z.sin();
    let z_inv = z.recip();
    let n_f = n as f64;

    if n >= 2 && z.abs() < n_f {
        let m_start = (2 * n + 30).max((n_f + 8.0 * (40.0 * n_f).sqrt().ceil()) as u32);
        let mut j_kplus1 = Complex64::new(0.0, 0.0);
        let mut j_k = Complex64::new(1.0e-30, 0.0);
        let mut j_at_n = Complex64::new(0.0, 0.0);
        for k in (1..=m_start).rev() {
            let coeff = Complex64::new(2.0 * k as f64 + 1.0, 0.0) * z_inv;
            let j_kminus1 = coeff * j_k - j_kplus1;
            if k == n {
                j_at_n = j_k;
            }
            j_kplus1 = j_k;
            j_k = j_kminus1;
            if j_k.abs() > 1.0e100 {
                let scale = Complex64::new(1.0e-100, 0.0);
                j_k = j_k * scale;
                j_kplus1 = j_kplus1 * scale;
                j_at_n = j_at_n * scale;
            }
        }
        // Normalize so j_0 matches sin(z)/z.
        let true_j0 = sin_z * z_inv;
        return j_at_n * (true_j0 * j_k.recip());
    }

    let cos_z = z.cos();
    let mut j_prev = sin_z * z_inv; // j_0
    if n == 0 {
        return j_prev;
    }

    let z_inv2 = z_inv * z_inv;
    let mut j_curr = sin_z * z_inv2 - cos_z * z_inv; // j_1
    if n == 1 {
        return j_curr;
    }

    for k in 1..n {
        let coeff = Complex64::new(2.0 * k as f64 + 1.0, 0.0) * z_inv;
        let next = coeff * j_curr - j_prev;
        j_prev = j_curr;
        j_curr = next;
    }
    j_curr
}

/// Complex spherical Bessel function of the second kind y_n(z).
fn complex_spherical_yn(n: u32, z: Complex64) -> Complex64 {
    if z.re == 0.0 && z.im == 0.0 {
        return Complex64::new(f64::NEG_INFINITY, 0.0);
    }

    let sin_z = z.sin();
    let cos_z = z.cos();
    let z_inv = z.recip();

    let mut y_prev = Complex64::new(-1.0, 0.0) * cos_z * z_inv; // y_0
    if n == 0 {
        return y_prev;
    }

    let z_inv2 = z_inv * z_inv;
    let mut y_curr = Complex64::new(-1.0, 0.0) * cos_z * z_inv2 - sin_z * z_inv; // y_1
    if n == 1 {
        return y_curr;
    }

    for k in 1..n {
        let coeff = Complex64::new(2.0 * k as f64 + 1.0, 0.0) * z_inv;
        let next = coeff * y_curr - y_prev;
        y_prev = y_curr;
        y_curr = next;
    }
    y_curr
}

/// Complex modified spherical Bessel function of the first kind i_n(z).
///
/// Same Miller's-recurrence treatment as complex_spherical_jn for the
/// |z| < n regime (resolves [frankenscipy-tt2v2]).
fn complex_spherical_in(n: u32, z: Complex64) -> Complex64 {
    if z.re == 0.0 && z.im == 0.0 {
        return if n == 0 {
            Complex64::new(1.0, 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        };
    }

    let sinh_z = z.sinh();
    let z_inv = z.recip();
    let n_f = n as f64;

    if n >= 2 && z.abs() < n_f {
        let m_start = (2 * n + 30).max((n_f + 8.0 * (40.0 * n_f).sqrt().ceil()) as u32);
        let mut i_kplus1 = Complex64::new(0.0, 0.0);
        let mut i_k = Complex64::new(1.0e-30, 0.0);
        let mut i_at_n = Complex64::new(0.0, 0.0);
        for k in (1..=m_start).rev() {
            let coeff = Complex64::new(2.0 * k as f64 + 1.0, 0.0) * z_inv;
            let i_kminus1 = coeff * i_k + i_kplus1;
            if k == n {
                i_at_n = i_k;
            }
            i_kplus1 = i_k;
            i_k = i_kminus1;
            if i_k.abs() > 1.0e100 {
                let scale = Complex64::new(1.0e-100, 0.0);
                i_k = i_k * scale;
                i_kplus1 = i_kplus1 * scale;
                i_at_n = i_at_n * scale;
            }
        }
        let true_i0 = sinh_z * z_inv;
        return i_at_n * (true_i0 * i_k.recip());
    }

    let cosh_z = z.cosh();
    let mut i_prev = sinh_z * z_inv; // i_0
    if n == 0 {
        return i_prev;
    }

    let z_inv2 = z_inv * z_inv;
    let mut i_curr = cosh_z * z_inv - sinh_z * z_inv2; // i_1
    if n == 1 {
        return i_curr;
    }

    for k in 1..n {
        let coeff = Complex64::new(2.0 * k as f64 + 1.0, 0.0) * z_inv;
        let next = i_prev - coeff * i_curr;
        i_prev = i_curr;
        i_curr = next;
    }
    i_curr
}

/// Complex modified spherical Bessel function of the second kind k_n(z).
fn complex_spherical_kn(n: u32, z: Complex64) -> Complex64 {
    if z.re == 0.0 && z.im == 0.0 {
        return Complex64::new(f64::INFINITY, 0.0);
    }

    let neg_z = Complex64::new(-z.re, -z.im);
    let emz = neg_z.exp();
    let scale = Complex64::new(std::f64::consts::PI / 2.0, 0.0);
    let z_inv = z.recip();

    let mut k_prev = scale * emz * z_inv; // k_0
    if n == 0 {
        return k_prev;
    }

    let one_plus_zinv = Complex64::new(1.0, 0.0) + z_inv;
    let mut k_curr = scale * emz * z_inv * one_plus_zinv; // k_1
    if n == 1 {
        return k_curr;
    }

    for k in 1..n {
        let coeff = Complex64::new(2.0 * k as f64 + 1.0, 0.0) * z_inv;
        let next = k_prev + coeff * k_curr;
        k_prev = k_curr;
        k_curr = next;
    }
    k_curr
}

fn wright_bessel_scalar(a: f64, b: f64, x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    const LN_MAX: f64 = 709.782_712_893_384;
    const LN_MIN: f64 = -745.133_219_101_941_1;

    if a.is_nan() || b.is_nan() || x.is_nan() {
        return Ok(f64::NAN);
    }
    if !a.is_finite() || !b.is_finite() || !x.is_finite() || a < 0.0 || b < 0.0 || x < 0.0 {
        return domain_error_by_mode(
            "wright_bessel",
            mode,
            format!("a={a},b={b},x={x}"),
            "wright_bessel requires finite a>=0, b>=0, x>=0",
        );
    }
    if x == 0.0 {
        return rgamma_nonnegative(b);
    }
    let log_value = if a == 0.0 {
        x + rgamma_log_nonnegative(b)?
    } else {
        log_wright_bessel_series(a, b, x)?
    };
    Ok(exp_from_log(log_value, LN_MIN, LN_MAX))
}

fn log_wright_bessel_scalar(
    a: f64,
    b: f64,
    x: f64,
    mode: RuntimeMode,
) -> Result<f64, SpecialError> {
    if a.is_nan() || b.is_nan() || x.is_nan() {
        return Ok(f64::NAN);
    }
    if !a.is_finite() || !b.is_finite() || !x.is_finite() || a < 0.0 || b < 0.0 || x < 0.0 {
        return domain_error_by_mode(
            "log_wright_bessel",
            mode,
            format!("a={a},b={b},x={x}"),
            "log_wright_bessel requires finite a>=0, b>=0, x>=0",
        );
    }
    if x == 0.0 {
        return rgamma_log_nonnegative(b);
    }
    if a == 0.0 {
        return Ok(x + rgamma_log_nonnegative(b)?);
    }
    log_wright_bessel_series(a, b, x)
}

fn log_wright_bessel_series(a: f64, b: f64, x: f64) -> Result<f64, SpecialError> {
    const RELATIVE_LOG_EPS: f64 = -40.0;
    const MIN_DECREASING_TERMS: usize = 8;

    let ln_x = x.ln();
    let mut max_log = f64::NEG_INFINITY;
    let mut scaled_sum = 0.0;
    let mut prev_log = f64::NEG_INFINITY;
    let mut decreasing_terms = 0usize;
    let max_terms = if a >= 10.0 {
        256usize
    } else if a >= 1.0 {
        1024usize
    } else if a >= 0.1 {
        4096usize
    } else {
        16_384usize
    };

    for k in 0..max_terms {
        let kf = k as f64;
        let akb = a.mul_add(kf, b);
        let log_rgamma = rgamma_log_nonnegative(akb)?;
        let log_term =
            kf * ln_x - crate::gamma::gammaln_scalar(kf + 1.0, RuntimeMode::Strict)? + log_rgamma;

        if !log_term.is_finite() {
            prev_log = log_term;
            continue;
        }

        if log_term > max_log {
            scaled_sum = if max_log.is_finite() {
                scaled_sum * (max_log - log_term).exp() + 1.0
            } else {
                1.0
            };
            max_log = log_term;
        } else {
            scaled_sum += (log_term - max_log).exp();
        }

        if prev_log.is_finite() && log_term < prev_log {
            decreasing_terms += 1;
        } else {
            decreasing_terms = 0;
        }
        prev_log = log_term;

        let current_log_sum = max_log + scaled_sum.ln();
        if decreasing_terms >= MIN_DECREASING_TERMS
            && log_term - current_log_sum <= RELATIVE_LOG_EPS
        {
            return Ok(current_log_sum);
        }
    }

    if max_log.is_finite() {
        Ok(max_log + scaled_sum.ln())
    } else {
        Ok(f64::NEG_INFINITY)
    }
}

fn rgamma_nonnegative(x: f64) -> Result<f64, SpecialError> {
    Ok(exp_from_log(
        rgamma_log_nonnegative(x)?,
        f64::NEG_INFINITY,
        709.782_712_893_384,
    ))
}

fn rgamma_log_nonnegative(x: f64) -> Result<f64, SpecialError> {
    if x == 0.0 {
        return Ok(f64::NEG_INFINITY);
    }
    Ok(-crate::gamma::gammaln_scalar(x, RuntimeMode::Strict)?)
}

fn exp_from_log(log_value: f64, ln_min: f64, ln_max: f64) -> f64 {
    if log_value.is_nan() {
        f64::NAN
    } else if log_value == f64::NEG_INFINITY {
        0.0
    } else if log_value > ln_max {
        f64::INFINITY
    } else if log_value < ln_min {
        0.0
    } else {
        log_value.exp()
    }
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

fn map_real_binary<F>(
    function: &'static str,
    lhs: &SpecialTensor,
    rhs: &SpecialTensor,
    mode: RuntimeMode,
    kernel: F,
) -> SpecialResult
where
    F: Fn(f64, f64) -> Result<f64, SpecialError>,
{
    match (lhs, rhs) {
        (SpecialTensor::RealScalar(left), SpecialTensor::RealScalar(right)) => {
            kernel(*left, *right).map(SpecialTensor::RealScalar)
        }
        (SpecialTensor::RealVec(left), SpecialTensor::RealScalar(right)) => left
            .iter()
            .copied()
            .map(|value| kernel(value, *right))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        (SpecialTensor::RealScalar(left), SpecialTensor::RealVec(right)) => right
            .iter()
            .copied()
            .map(|value| kernel(*left, value))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        (SpecialTensor::RealVec(left), SpecialTensor::RealVec(right)) => {
            if left.len() != right.len() {
                record_special_trace(
                    function,
                    mode,
                    "domain_error",
                    format!("lhs_len={},rhs_len={}", left.len(), right.len()),
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
            left.iter()
                .copied()
                .zip(right.iter().copied())
                .map(|(l, r)| kernel(l, r))
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

fn jn_scalar(order: f64, x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if order.is_nan() || x.is_nan() {
        return Ok(f64::NAN);
    }
    let n = match integer_order_or_nan("jn", order, mode)? {
        Some(value) => value,
        None => return Ok(f64::NAN),
    };

    if n < 0 {
        let sign = if n & 1 == 0 { 1.0 } else { -1.0 };
        return Ok(sign * jn_nonnegative((-n) as u32, x));
    }
    Ok(jn_nonnegative(n as u32, x))
}

fn yn_scalar(order: f64, x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if order.is_nan() || x.is_nan() {
        return Ok(f64::NAN);
    }
    if x < 0.0 {
        return domain_error_by_mode(
            "yn",
            mode,
            format!("order={order},x={x}"),
            "yn real-valued wrapper requires x >= 0",
        );
    }
    if x == 0.0 {
        return Ok(f64::NEG_INFINITY);
    }
    if x.is_infinite() {
        return Ok(0.0);
    }
    let n = match integer_order_or_nan("yn", order, mode)? {
        Some(value) => value,
        None => return Ok(f64::NAN),
    };
    if n < 0 {
        let sign = if n & 1 == 0 { 1.0 } else { -1.0 };
        return Ok(sign * yn_nonnegative((-n) as u32, x));
    }
    Ok(yn_nonnegative(n as u32, x))
}

fn integer_order_or_nan(
    function: &'static str,
    order: f64,
    mode: RuntimeMode,
) -> Result<Option<i32>, SpecialError> {
    if order.is_finite()
        && order.fract() == 0.0
        && order >= i32::MIN as f64
        && order <= i32::MAX as f64
    {
        return Ok(Some(order as i32));
    }
    match mode {
        RuntimeMode::Strict => {
            record_special_trace(
                function,
                mode,
                "domain_error",
                format!("order={order}"),
                "returned_nan",
                "strict non-integral-order fallback",
                false,
            );
            Ok(None)
        }
        RuntimeMode::Hardened => {
            record_special_trace(
                function,
                mode,
                "domain_error",
                format!("order={order}"),
                "fail_closed",
                "Bessel order must be a finite integer",
                false,
            );
            Err(SpecialError {
                function,
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "Bessel order must be a finite integer",
            })
        }
    }
}

fn y0_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if x.is_nan() {
        return Ok(f64::NAN);
    }
    if x == 0.0 {
        return Ok(f64::NEG_INFINITY);
    }
    if x < 0.0 {
        return domain_error_by_mode(
            "y0",
            mode,
            format!("x={x}"),
            "y0 real-valued wrapper requires x >= 0",
        );
    }
    if x.is_infinite() {
        return Ok(0.0);
    }
    Ok(y0_core_positive(x))
}

fn y1_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if x.is_nan() {
        return Ok(f64::NAN);
    }
    if x == 0.0 {
        return Ok(f64::NEG_INFINITY);
    }
    if x < 0.0 {
        return domain_error_by_mode(
            "y1",
            mode,
            format!("x={x}"),
            "y1 real-valued wrapper requires x >= 0",
        );
    }
    if x.is_infinite() {
        return Ok(0.0);
    }
    Ok(y1_core_positive(x))
}

fn domain_error_by_mode(
    function: &'static str,
    mode: RuntimeMode,
    input_summary: String,
    detail: &'static str,
) -> Result<f64, SpecialError> {
    match mode {
        RuntimeMode::Strict => {
            record_special_trace(
                function,
                mode,
                "domain_error",
                input_summary,
                "returned_nan",
                "strict domain fallback",
                false,
            );
            Ok(f64::NAN)
        }
        RuntimeMode::Hardened => {
            record_special_trace(
                function,
                mode,
                "domain_error",
                input_summary,
                "fail_closed",
                detail,
                false,
            );
            Err(SpecialError {
                function,
                kind: SpecialErrorKind::DomainError,
                mode,
                detail,
            })
        }
    }
}

fn complex_domain_error_by_mode(
    function: &'static str,
    mode: RuntimeMode,
    input_summary: String,
    detail: &'static str,
) -> Result<Complex64, SpecialError> {
    match mode {
        RuntimeMode::Strict => {
            record_special_trace(
                function,
                mode,
                "domain_error",
                input_summary,
                "returned_nan",
                "strict domain fallback",
                false,
            );
            Ok(Complex64::new(f64::NAN, f64::NAN))
        }
        RuntimeMode::Hardened => {
            record_special_trace(
                function,
                mode,
                "domain_error",
                input_summary,
                "fail_closed",
                detail,
                false,
            );
            Err(SpecialError {
                function,
                kind: SpecialErrorKind::DomainError,
                mode,
                detail,
            })
        }
    }
}

fn jn_nonnegative(n: u32, x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return 0.0;
    }
    if n == 0 {
        return j0_core(x);
    }
    if n == 1 {
        return j1_core(x);
    }
    if x == 0.0 {
        return 0.0;
    }

    // Forward recurrence J_{k+1} = 2k/x · J_k − J_{k-1} is unstable when
    // x ≪ 2k (J_n decays super-exponentially while the prefactor grows).
    // Switch to Miller's downward recurrence in that regime. Resolves
    // [frankenscipy-ls0q1].
    let n_f = n as f64;
    let ax = x.abs();
    if n >= 2 && ax < n_f {
        let m_start = (2 * n + 30).max((n_f + 8.0 * (40.0 * n_f).sqrt().ceil()) as u32);
        let mut j_kplus1 = 0.0_f64;
        let mut j_k = 1.0e-30_f64;
        let mut j_at_n = 0.0_f64;
        // J_{k-1} = 2k/x · J_k − J_{k+1}
        for k in (1..=m_start).rev() {
            let kf = k as f64;
            let j_kminus1 = (2.0 * kf / x) * j_k - j_kplus1;
            if k == n {
                j_at_n = j_k;
            }
            j_kplus1 = j_k;
            j_k = j_kminus1;
            if j_k.abs() > 1.0e100 {
                let scale = 1.0e-100;
                j_k *= scale;
                j_kplus1 *= scale;
                j_at_n *= scale;
            }
        }
        // Normalize against the analytic J_0(x).
        let true_j0 = j0_core(x);
        return j_at_n * (true_j0 / j_k);
    }

    let mut j_prev = j0_core(x);
    let mut j_curr = j1_core(x);
    for k in 1..n {
        let next = (2.0 * k as f64 / x) * j_curr - j_prev;
        j_prev = j_curr;
        j_curr = next;
    }
    j_curr
}

fn yn_nonnegative(n: u32, x: f64) -> f64 {
    if n == 0 {
        return y0_core_positive(x);
    }
    if n == 1 {
        return y1_core_positive(x);
    }

    let mut y_prev = y0_core_positive(x);
    let mut y_curr = y1_core_positive(x);
    for k in 1..n {
        let next = (2.0 * k as f64 / x) * y_curr - y_prev;
        y_prev = y_curr;
        y_curr = next;
    }
    y_curr
}

fn j0_core(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return 0.0;
    }

    let ax = x.abs();
    if ax == 0.0 {
        // J_0(0) = 1 exactly; keep the analytic value at the origin.
        return 1.0;
    }
    if ax < 8.0 {
        j0_series_small(ax)
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - PI / 4.0;
        let p = 1.0
            + y * (-0.001_098_628_627
                + y * (0.000_027_345_104_07
                    + y * (-0.000_002_073_370_639 + y * 0.000_000_209_388_721_1)));
        let q = -0.015_624_999_95
            + y * (0.000_143_048_876_5
                + y * (-0.000_006_911_147_651
                    + y * (0.000_000_762_109_516_1 + y * (-0.000_000_093_494_515_2))));
        (FRAC_2_PI / ax).sqrt() * (xx.cos() * p - z * xx.sin() * q)
    }
}

fn j0_series_small(x: f64) -> f64 {
    let z = x * x * 0.25;
    let mut term = 1.0;
    let mut sum = 1.0;
    for k in 1..=BESSEL_SERIES_MAX_TERMS {
        let kf = k as f64;
        term *= -z / (kf * kf);
        sum += term;
        if term.abs() <= f64::EPSILON * sum.abs().max(1.0) {
            break;
        }
    }
    sum
}

fn j1_core(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return 0.0;
    }

    let ax = x.abs();
    let ans = if ax < 8.0 {
        let y = ax * ax;
        let numer = ax
            * (72_362_614_232.0
                + y * (-7_895_059_235.0
                    + y * (242_396_853.1
                        + y * (-2_972_611.439 + y * (15_704.482_60 + y * (-30.160_366_06))))));
        let denom = 144_725_228_442.0
            + y * (2_300_535_178.0
                + y * (18_583_304.74 + y * (99_447.433_94 + y * (376.999_139_7 + y))));
        numer / denom
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 3.0 * PI / 4.0;
        let p = 1.0
            + y * (0.001_831_05
                + y * (-0.000_035_163_964_96
                    + y * (0.000_002_457_520_174 + y * (-0.000_000_240_337_019))));
        let q = 0.046_874_999_95
            + y * (-0.000_200_269_087_3
                + y * (0.000_008_449_199_096
                    + y * (-0.000_000_882_289_87 + y * 0.000_000_105_787_412)));
        (FRAC_2_PI / ax).sqrt() * (xx.cos() * p - z * xx.sin() * q)
    };

    if x < 0.0 { -ans } else { ans }
}

fn y0_core_positive(x: f64) -> f64 {
    if x < 8.0 {
        y0_series_small(x)
    } else {
        let z = 8.0 / x;
        let y = z * z;
        let xx = x - PI / 4.0;
        let p = 1.0
            + y * (-0.001_098_628_627
                + y * (0.000_027_345_104_07
                    + y * (-0.000_002_073_370_639 + y * 0.000_000_209_388_721_1)));
        let q = -0.015_624_999_95
            + y * (0.000_143_048_876_5
                + y * (-0.000_006_911_147_651
                    + y * (0.000_000_762_109_516_1 + y * (-0.000_000_093_494_515_2))));
        (FRAC_2_PI / x).sqrt() * (xx.sin() * p + z * xx.cos() * q)
    }
}

fn y0_series_small(x: f64) -> f64 {
    let z = x * x * 0.25;
    let j0 = j0_series_small(x);
    let mut harmonic = 0.0;
    let mut term = 1.0;
    let mut correction = 0.0;
    for k in 1..=BESSEL_SERIES_MAX_TERMS {
        let kf = k as f64;
        harmonic += 1.0 / kf;
        term *= -z / (kf * kf);
        let addend = -harmonic * term;
        correction += addend;
        if addend.abs() <= f64::EPSILON * correction.abs().max(1.0) {
            break;
        }
    }
    FRAC_2_PI * ((x.ln() - LN_2 + EULER_MASCHERONI) * j0 + correction)
}

fn y1_core_positive(x: f64) -> f64 {
    if x < 8.0 {
        let y = x * x;
        let numer = x
            * (-4.900_604_943e12
                + y * (1.275_274_39e12
                    + y * (-5.153_438_139e10
                        + y * (7.349_264_551e8 + y * (-4.237_922_726e6 + y * 8_511.937_935)))));
        let denom = 2.499_580_57e13
            + y * (4.244_419_664e11
                + y * (3.733_650_367e9
                    + y * (2.245_904_002e7 + y * (1.020_426_05e5 + y * (354.963_288_5 + y)))));
        numer / denom + FRAC_2_PI * (j1_core(x) * x.ln() - 1.0 / x)
    } else {
        let z = 8.0 / x;
        let y = z * z;
        let xx = x - 3.0 * PI / 4.0;
        let p = 1.0
            + y * (0.001_831_05
                + y * (-0.000_035_163_964_96
                    + y * (0.000_002_457_520_174 + y * (-0.000_000_240_337_019))));
        let q = 0.046_874_999_95
            + y * (-0.000_200_269_087_3
                + y * (0.000_008_449_199_096
                    + y * (-0.000_000_882_289_87 + y * 0.000_000_105_787_412)));
        (FRAC_2_PI / x).sqrt() * (xx.sin() * p + z * xx.cos() * q)
    }
}

// ============================================================================
// Complex cylindrical Bessel functions J_v, Y_v, I_v, K_v
// ============================================================================

/// Complex J_v(z) via power series.
/// J_v(z) = (z/2)^v Σ_{k=0}^∞ (-z²/4)^k / (k! Γ(v+k+1))
fn negative_integer_order(v: f64) -> Option<u32> {
    if !v.is_finite() || v >= 0.0 || v.fract() != 0.0 || v.abs() > u32::MAX as f64 {
        return None;
    }

    Some((-v) as u32)
}

/// J_v(z) by Miller's backward recurrence for the turning band v < |z| ≤ v²
/// (where the power series cancels and the asymptotic diverges). J is recessive
/// as order grows, so the downward recurrence J_{ν−1}=(2ν/z)J_ν−J_{ν+1} is
/// stable; the unnormalized run is rescaled to a small base order whose value
/// comes from the (convergent) jv asymptotic. frankenscipy-8eiog.
fn complex_jv_miller(v: f64, z: Complex64) -> Complex64 {
    let frac = v - v.floor();
    // Target order is frac + off; off (= round(v − frac)) is negative for the
    // J_{-v}/I_{-v} terms that yv/kv need, so keep it signed and let the loop
    // descend through 0 to it. Base orders frac (m=0) and frac+1 (m=1) supply
    // the normalization.
    let off = (v - frac).round() as isize;
    let lo = off.min(0);
    let m_start = z.abs() as isize + off.max(0) + 60 + 2 * off.unsigned_abs() as isize;
    let z_inv = z.recip();
    let mut jp1 = Complex64::new(0.0, 0.0);
    let mut jc = Complex64::new(1e-300, 0.0);
    let mut order = frac + m_start as f64;
    let mut j_at_n = Complex64::new(0.0, 0.0);
    let mut j_at_0 = Complex64::new(0.0, 0.0);
    let mut j_at_1 = Complex64::new(0.0, 0.0);
    let mut m = m_start;
    while m >= lo {
        if m == off {
            j_at_n = jc;
        }
        if m == 0 {
            j_at_0 = jc;
        }
        if m == 1 {
            j_at_1 = jc;
        }
        let jm1 = Complex64::new(2.0 * order, 0.0) * z_inv * jc - jp1;
        jp1 = jc;
        jc = jm1;
        order -= 1.0;
        m -= 1;
        if jc.abs() > 1e250 {
            let s = Complex64::new(1e-250, 0.0);
            jc = jc * s;
            jp1 = jp1 * s;
            j_at_n = j_at_n * s;
            j_at_0 = j_at_0 * s;
            j_at_1 = j_at_1 * s;
        }
    }
    // The unnormalized run can leave the captured values at ~1e-257, where the
    // naive complex division below would form |j|² ≈ 1e-514 and underflow to 0
    // (→ inf/NaN). Rescale all three by the robust hypot magnitude first so the
    // division operates on O(1) values; the common factor cancels in the ratio.
    let mref = j_at_0.abs().max(j_at_1.abs()).max(j_at_n.abs());
    if mref > 0.0 && mref.is_finite() {
        j_at_0 = j_at_0 / mref;
        j_at_1 = j_at_1 / mref;
        j_at_n = j_at_n / mref;
    }
    let t0 = complex_jv_asymptotic(frac, z);
    let t1 = complex_jv_asymptotic(frac + 1.0, z);
    let scale = if j_at_0.abs() >= j_at_1.abs() {
        t0 / j_at_0
    } else {
        t1 / j_at_1
    };
    j_at_n * scale
}

/// J_v(z) large-|z| Hankel asymptotic (DLMF 10.17.3): √(2/πz)[cos ω·P − sin ω·Q],
/// ω = z − vπ/2 − π/4. The plain form is on its branch cut at the negative real
/// axis, so reflect Re(z) < 0 via the connection J_v(z) = e^{±ivπ}J_v(−z) (upper
/// sign for Im(z) ≥ 0) onto the accurate Re(z) ≥ 0 side. frankenscipy-oisri.
fn complex_jv_asymptotic(v: f64, z: Complex64) -> Complex64 {
    if z.re < 0.0 {
        let s = (v * PI).sin();
        let factor = if z.im >= 0.0 {
            Complex64::new((v * PI).cos(), s)
        } else {
            Complex64::new((v * PI).cos(), -s)
        };
        return factor * complex_jv_asymptotic(v, Complex64::new(-z.re, -z.im));
    }
    let mu = 4.0 * v * v;
    let z_inv = z.recip();
    let mut term = Complex64::new(1.0, 0.0);
    let mut p = term;
    let mut q = Complex64::new(0.0, 0.0);
    let mut prev = 1.0_f64;
    for k in 1..60 {
        let kf = k as f64;
        term = term * Complex64::new((mu - (2.0 * kf - 1.0).powi(2)) / (8.0 * kf), 0.0) * z_inv;
        let at = term.abs();
        if at > prev {
            break;
        }
        if k % 2 == 0 {
            let s = if (k / 2) % 2 == 0 { 1.0 } else { -1.0 };
            p = p + term * Complex64::new(s, 0.0);
        } else {
            let s = if ((k - 1) / 2) % 2 == 0 { 1.0 } else { -1.0 };
            q = q + term * Complex64::new(s, 0.0);
        }
        prev = at;
    }
    let omega = z - Complex64::new(v * PI / 2.0 + PI / 4.0, 0.0);
    let pref = (Complex64::new(2.0 / PI, 0.0) * z_inv).powf(0.5);
    pref * (omega.cos() * p - omega.sin() * q)
}

/// I_v(z) for large |z| via I_v(z) = e^{−iπv/2}J_v(iz) (e^{3iπv/2}J_v(iz) for
/// arg z > π/2), routing through the everywhere-accurate jv asymptotic so the
/// Stokes line on the imaginary axis is dodged (iz rotates it to the real axis).
fn complex_iv_asymptotic(v: f64, z: Complex64) -> Complex64 {
    let iz = Complex64::new(-z.im, z.re);
    // Full jv router (asymptotic for |iz| > v², Miller for the turning band).
    let jiz = if iz.abs() > v * v {
        complex_jv_asymptotic(v, iz)
    } else {
        complex_jv_miller(v, iz)
    };
    if z.arg() <= PI / 2.0 {
        Complex64::new((v * PI / 2.0).cos(), -(v * PI / 2.0).sin()) * jiz
    } else {
        Complex64::new((3.0 * v * PI / 2.0).cos(), (3.0 * v * PI / 2.0).sin()) * jiz
    }
}

pub(crate) fn complex_jv_scalar(v: f64, z: Complex64) -> Complex64 {
    if !z.is_finite() || v.is_nan() {
        return Complex64::new(f64::NAN, f64::NAN);
    }

    if let Some(n) = negative_integer_order(v) {
        let value = complex_jv_scalar(-v, z);
        return if n % 2 == 0 { value } else { -value };
    }

    if z.re == 0.0 && z.im == 0.0 {
        return if v == 0.0 {
            Complex64::new(1.0, 0.0)
        } else if v > 0.0 {
            Complex64::new(0.0, 0.0)
        } else {
            Complex64::new(f64::INFINITY, 0.0)
        };
    }

    // Large |z|: the 200-term power series truncates and cancels (max term
    // ~e^{|z|}). The Hankel asymptotic converges only for |z| > v²; in the
    // turning band v < |z| ≤ v² it diverges, so use Miller's backward recurrence
    // there — but ONLY for v ≥ 0: J is recessive going up in order, so the
    // downward recurrence is stable, whereas for J_{-v} (used by yv/kv) the run
    // descends past order 0 into the growing-Y regime and loses accuracy
    // (~1e-3). frankenscipy-8eiog. frankenscipy-oisri.
    if z.abs() >= 20.0 && (z.abs() > v * v || v >= 0.0) {
        return if z.abs() > v * v {
            complex_jv_asymptotic(v, z)
        } else {
            complex_jv_miller(v, z)
        };
    }

    let half_z = z / 2.0;
    let neg_quarter_z2 = Complex64::new(-1.0, 0.0) * half_z * half_z;

    // First term: (z/2)^v / Γ(v+1)
    let log_half_z = half_z.ln();
    let v_c = Complex64::new(v, 0.0);
    let log_first = v_c * log_half_z - crate::gamma::complex_gammaln(Complex64::new(v + 1.0, 0.0));
    let mut sum = log_first.exp();
    let mut term = sum;

    for k in 1..200 {
        let kf = k as f64;
        // term *= (-z²/4) / (k * (v + k))
        term = term * neg_quarter_z2 / Complex64::new(kf * (v + kf), 0.0);
        sum = sum + term;

        if term.abs() < 1e-15 * sum.abs() && k > 5 {
            break;
        }
    }

    sum
}

/// Complex I_v(z) via power series.
/// I_v(z) = (z/2)^v Σ_{k=0}^∞ (z²/4)^k / (k! Γ(v+k+1))
pub(crate) fn complex_iv_scalar(v: f64, z: Complex64) -> Complex64 {
    if !z.is_finite() || v.is_nan() {
        return Complex64::new(f64::NAN, f64::NAN);
    }

    if negative_integer_order(v).is_some() {
        return complex_iv_scalar(-v, z);
    }

    if z.re == 0.0 && z.im == 0.0 {
        return if v == 0.0 {
            Complex64::new(1.0, 0.0)
        } else if v > 0.0 {
            Complex64::new(0.0, 0.0)
        } else {
            Complex64::new(f64::INFINITY, 0.0)
        };
    }

    // Large |z|: power series cancels; route through the jv relation, which
    // itself picks the asymptotic (|z| > v²) or Miller's recurrence (the
    // turning band). Restricted to v ≥ 0 in the band (the negative-order Miller
    // that I_{-v} would need loses accuracy); larger |z| handles any sign.
    // frankenscipy-oisri / frankenscipy-8eiog.
    if z.abs() >= 20.0 && (z.abs() > v * v || v >= 0.0) {
        return complex_iv_asymptotic(v, z);
    }

    let half_z = z / 2.0;
    let quarter_z2 = half_z * half_z;

    // First term: (z/2)^v / Γ(v+1)
    let log_half_z = half_z.ln();
    let v_c = Complex64::new(v, 0.0);
    let log_first = v_c * log_half_z - crate::gamma::complex_gammaln(Complex64::new(v + 1.0, 0.0));
    let mut sum = log_first.exp();
    let mut term = sum;

    for k in 1..200 {
        let kf = k as f64;
        // term *= (z²/4) / (k * (v + k))
        term = term * quarter_z2 / Complex64::new(kf * (v + kf), 0.0);
        sum = sum + term;

        if term.abs() < 1e-15 * sum.abs() && k > 5 {
            break;
        }
    }

    sum
}

/// Complex Y_v(z) for real order v.
/// Y_v = (J_v cos(vπ) - J_{-v}) / sin(vπ) for non-integer v.
fn complex_yv_scalar(v: f64, z: Complex64, _mode: RuntimeMode) -> Result<Complex64, SpecialError> {
    if v.is_nan() || !z.is_finite() {
        return Ok(Complex64::new(f64::NAN, f64::NAN));
    }

    if z.re == 0.0 && z.im == 0.0 {
        return Ok(Complex64::new(f64::NEG_INFINITY, 0.0));
    }

    // Integer order: use recurrence
    if v.fract() == 0.0 && v.abs() <= i32::MAX as f64 {
        let n = v.abs() as u32;
        let result = complex_yn_integer(n, z);
        return if v < 0.0 && n % 2 == 1 {
            Ok(Complex64::new(-result.re, -result.im))
        } else {
            Ok(result)
        };
    }

    // For |z| ≥ 20 the (J_v cos(vπ) − J_{-v})/sin(vπ) form needs J_{-v}, which in
    // the turning band v < |z| ≤ v² goes through the negative-order Miller
    // recurrence (only ~1e-3, since it descends past order 0 into the growing-Y
    // regime). Instead compute Y_v from the Hankel relation
    //   H₁_v(z) = (2/π) i^{-(v+1)} K_v(z e^{-iπ/2}),  Y_v = (H₁_v − J_v)/i,
    // using the now-accurate complex J_v and K_v (K is exact across the whole
    // plane). Im(z) < 0 uses Schwarz reflection Y_v(z̄) = conj(Y_v(z)).
    // frankenscipy-d2s72.
    if z.abs() >= 20.0 {
        return Ok(complex_yv_hankel(v, z, _mode));
    }

    // Non-integer order, moderate |z|: Y_v = (J_v cos(vπ) - J_{-v}) / sin(vπ)
    let sin_vpi = (v * PI).sin();
    if sin_vpi.abs() < 1e-15 {
        return Ok(Complex64::new(f64::NAN, f64::NAN));
    }

    let jv_pos = complex_jv_scalar(v, z);
    let jv_neg = complex_jv_scalar(-v, z);
    let cos_vpi = Complex64::new((v * PI).cos(), 0.0);
    let sin_vpi_c = Complex64::new(sin_vpi, 0.0);

    Ok((jv_pos * cos_vpi - jv_neg) / sin_vpi_c)
}

/// Y_v(z) for real order v at large |z| via the Hankel/K relation (DLMF 10.27.8):
///   H₁_v(z) = (2/π) i^{-(v+1)} K_v(z e^{-iπ/2}),  Y_v(z) = (H₁_v(z) − J_v(z))/i.
/// Valid for Im(z) ≥ 0; the lower half plane uses Y_v(z̄) = conj(Y_v(z)).
fn complex_yv_hankel(v: f64, z: Complex64, mode: RuntimeMode) -> Complex64 {
    if z.im < 0.0 {
        let yc = complex_yv_hankel(v, Complex64::new(z.re, -z.im), mode);
        return Complex64::new(yc.re, -yc.im);
    }
    // z·e^{-iπ/2} = -i·z = (Im(z), -Re(z)).
    let rot = Complex64::new(z.im, -z.re);
    let kv = complex_kv_scalar(v, rot, mode).unwrap_or(Complex64::new(f64::NAN, f64::NAN));
    let ang = -(v + 1.0) * PI / 2.0;
    let ifac = Complex64::new(ang.cos(), ang.sin()); // i^{-(v+1)} = e^{-i(v+1)π/2}
    let h1 = Complex64::new(2.0 / PI, 0.0) * ifac * kv;
    let jv = complex_jv_scalar(v, z);
    let d = h1 - jv;
    // (H₁ − J_v)/i = -i·(H₁ − J_v): (a+bi)·(-i) = b − a i.
    Complex64::new(d.im, -d.re)
}

/// Complex Y_n(z) for integer order via recurrence.
fn complex_yn_integer(n: u32, z: Complex64) -> Complex64 {
    // Y_0 and Y_1 via formula, then recurrence
    let y0 = complex_y0_series(z);
    if n == 0 {
        return y0;
    }
    let y1 = complex_y1_series(z);
    if n == 1 {
        return y1;
    }

    let z_inv = z.recip();
    let mut y_prev = y0;
    let mut y_curr = y1;
    for k in 1..n {
        let coeff = Complex64::new(2.0 * k as f64, 0.0) * z_inv;
        let next = coeff * y_curr - y_prev;
        y_prev = y_curr;
        y_curr = next;
    }
    y_curr
}

const BESSEL_EULER_GAMMA: f64 = 0.577_215_664_901_532_9;
/// Crossover |z| between the ascending log series and the large-z asymptotic for
/// the integer-order complex K_v/Y_v. Below it the series converges without much
/// cancellation; above it the I/J terms cancel to the recessive value and the
/// asymptotic takes over.
const COMPLEX_KY_ASYMP: f64 = 11.0;

/// K_v(z) large-|z| asymptotic √(π/2z)·e^{-z}·Σ a_k/z^k, a_k=a_{k-1}(4v²−(2k−1)²)/(8k).
fn complex_kv_asymptotic(v: f64, z: Complex64) -> Complex64 {
    let mu = 4.0 * v * v;
    let z_inv = z.recip();
    let mut a = Complex64::new(1.0, 0.0);
    let mut sum = a;
    let mut zpow = z_inv;
    let mut prev = 1.0_f64;
    for k in 1..40 {
        let kf = k as f64;
        a = a * Complex64::new((mu - (2.0 * kf - 1.0).powi(2)) / (8.0 * kf), 0.0);
        let term = a * zpow;
        let at = term.abs();
        if at > prev {
            break;
        }
        sum = sum + term;
        prev = at;
        zpow = zpow * z_inv;
    }
    let pref = (Complex64::new(PI / 2.0, 0.0) * z_inv).powf(0.5);
    pref * Complex64::new(-z.re, -z.im).exp() * sum
}

/// K_v(z) for non-integer order in the turning band v < |z| ≤ v² (Re(z) ≥ 0):
/// seed the K asymptotic at the small base orders frac and frac+1 (where it is
/// machine-accurate since |z| ≫ frac²) and walk up by K_{ν+1}=K_{ν-1}+(2ν/z)K_ν.
/// K grows monotonically with order, so the upward recurrence is stable through
/// the turning point order = |z| (unlike the oscillatory Y). frankenscipy-d2s72.
fn complex_kv_band(v: f64, z: Complex64) -> Complex64 {
    let frac = v - v.floor();
    let n = (v - frac).round() as usize;
    let z_inv = z.recip();
    let mut km1 = complex_kv_asymptotic(frac, z);
    if n == 0 {
        return km1;
    }
    let mut km = complex_kv_asymptotic(frac + 1.0, z);
    let mut order = frac + 1.0;
    for _ in 1..n {
        let next = km1 + Complex64::new(2.0 * order, 0.0) * z_inv * km;
        km1 = km;
        km = next;
        order += 1.0;
    }
    km
}

/// Y_v(z) large-|z| asymptotic √(2/πz)·[P·sin ω + Q·cos ω], ω=z−vπ/2−π/4
/// (DLMF 10.17.4), with the same a_k as the J asymptotic.
fn complex_yv_asymptotic(v: f64, z: Complex64) -> Complex64 {
    let mu = 4.0 * v * v;
    let z_inv = z.recip();
    let mut term = Complex64::new(1.0, 0.0);
    let mut p = term;
    let mut q = Complex64::new(0.0, 0.0);
    let mut prev = 1.0_f64;
    for k in 1..60 {
        let kf = k as f64;
        term = term * Complex64::new((mu - (2.0 * kf - 1.0).powi(2)) / (8.0 * kf), 0.0) * z_inv;
        let at = term.abs();
        if at > prev {
            break;
        }
        if k % 2 == 0 {
            let s = if (k / 2) % 2 == 0 { 1.0 } else { -1.0 };
            p = p + term * Complex64::new(s, 0.0);
        } else {
            let s = if ((k - 1) / 2) % 2 == 0 { 1.0 } else { -1.0 };
            q = q + term * Complex64::new(s, 0.0);
        }
        prev = at;
    }
    let omega = z - Complex64::new(v * PI / 2.0 + PI / 4.0, 0.0);
    let pref = (Complex64::new(2.0 / PI, 0.0) * z_inv).powf(0.5);
    pref * (omega.sin() * p + omega.cos() * q)
}

/// Complex Y_0(z) — DLMF 10.8.1 ascending series for |z| < 11, asymptotic above.
fn complex_y0_series(z: Complex64) -> Complex64 {
    if z.abs() >= COMPLEX_KY_ASYMP {
        return complex_yv_asymptotic(0.0, z);
    }
    // Y_0 = (2/π)[(ln(z/2)+γ)J_0(z) − Σ_{k≥1} H_k (−z²/4)^k/(k!)²].
    let j0 = complex_jv_scalar(0.0, z);
    let neg_z2_4 = Complex64::new(-1.0, 0.0) * z * z / 4.0;
    let mut t = Complex64::new(1.0, 0.0);
    let mut s = Complex64::new(0.0, 0.0);
    let mut h = 0.0_f64;
    for k in 1..200 {
        let kf = k as f64;
        t = t * neg_z2_4 / (kf * kf);
        h += 1.0 / kf;
        let term = t * Complex64::new(h, 0.0);
        s = s + term;
        if k > 3 && term.abs() < 1e-17 * s.abs().max(1e-300) {
            break;
        }
    }
    let log_term = (z / 2.0).ln() + Complex64::new(BESSEL_EULER_GAMMA, 0.0);
    Complex64::new(FRAC_2_PI, 0.0) * (log_term * j0 - s)
}

/// Complex Y_1(z) — DLMF 10.8.1 ascending series for |z| < 11, asymptotic above.
fn complex_y1_series(z: Complex64) -> Complex64 {
    if z.abs() >= COMPLEX_KY_ASYMP {
        return complex_yv_asymptotic(1.0, z);
    }
    // Y_1 = (2/π)(ln(z/2)+γ)J_1(z) − 2/(πz)
    //       − (1/π)(z/2) Σ_{k≥0} (H_k+H_{k+1})(−z²/4)^k/(k!(k+1)!).
    let j1 = complex_jv_scalar(1.0, z);
    let neg_z2_4 = Complex64::new(-1.0, 0.0) * z * z / 4.0;
    let mut t = Complex64::new(1.0, 0.0);
    let mut s = Complex64::new(0.0, 0.0);
    let mut hk = 0.0_f64;
    for k in 0..200 {
        let kf = k as f64;
        let hk1 = hk + 1.0 / (kf + 1.0);
        let term = t * Complex64::new(hk + hk1, 0.0);
        s = s + term;
        if k > 3 && term.abs() < 1e-17 * s.abs().max(1e-300) {
            break;
        }
        t = t * neg_z2_4 / ((kf + 1.0) * (kf + 2.0));
        hk = hk1;
    }
    let log_term = (z / 2.0).ln() + Complex64::new(BESSEL_EULER_GAMMA, 0.0);
    Complex64::new(FRAC_2_PI, 0.0) * log_term * j1 - Complex64::new(FRAC_2_PI, 0.0) * z.recip()
        - Complex64::new(1.0 / PI, 0.0) * (z / 2.0) * s
}

/// Complex K_v(z) for real order v.
/// K_v = π/2 * (I_{-v} - I_v) / sin(vπ) for non-integer v.
pub(crate) fn complex_kv_scalar(v: f64, z: Complex64, _mode: RuntimeMode) -> Result<Complex64, SpecialError> {
    if v.is_nan() || !z.is_finite() {
        return Ok(Complex64::new(f64::NAN, f64::NAN));
    }

    if z.re == 0.0 && z.im == 0.0 {
        return Ok(Complex64::new(f64::INFINITY, 0.0));
    }

    // Re(z) < 0 is across the K_v branch cut, where every Re(z) ≥ 0 method here
    // (asymptotic / band recurrence / I_{-v}−I_v) is on the wrong sheet. Reflect
    // onto Re(w) > 0 via the connection K_v(w e^{±iπ}) = e^{∓ivπ}K_v(w) ∓ iπI_v(w)
    // (w = −z; upper sign for Im(z) ≥ 0), using the now-accurate Re≥0 K_v and
    // I_v. frankenscipy-d2s72.
    if z.re < 0.0 {
        let w = Complex64::new(-z.re, -z.im);
        let kw = complex_kv_scalar(v, w, _mode)?;
        let iw = complex_iv_scalar(v, w);
        let s = (v * PI).sin();
        let c = (v * PI).cos();
        let (phase, i_pi) = if z.im >= 0.0 {
            (Complex64::new(c, -s), Complex64::new(0.0, -PI))
        } else {
            (Complex64::new(c, s), Complex64::new(0.0, PI))
        };
        return Ok(phase * kw + i_pi * iw);
    }

    // Large |z| (Re(z) ≥ 0): the non-integer K_v = π/2·(I_{-v}−I_v)/sin(vπ) form
    // cancels catastrophically (both I's ~ e^z, their difference is the
    // recessive ~e^{-z}). For |z| > v² the direct K asymptotic converges; in the
    // turning band v < |z| ≤ v² it diverges, so seed the asymptotic at a small
    // order and walk up by the recurrence K_{ν+1}=K_{ν-1}+(2ν/z)K_ν, which is
    // stable since K grows monotonically with order (no oscillation/turning the
    // way Y has). frankenscipy-oisri / frankenscipy-d2s72.
    if z.re >= 0.0 && z.abs() >= 15.0 {
        return Ok(if z.abs() > v * v {
            complex_kv_asymptotic(v, z)
        } else {
            complex_kv_band(v, z)
        });
    }

    // For non-integer v: K_v = π/2 * (I_{-v} - I_v) / sin(vπ)
    let sin_vpi = (v * PI).sin();
    if sin_vpi.abs() > 1e-10 {
        let iv_neg = complex_iv_scalar(-v, z);
        let iv_pos = complex_iv_scalar(v, z);
        let pi_half = Complex64::new(PI / 2.0, 0.0);
        let sin_vpi_c = Complex64::new(sin_vpi, 0.0);
        return Ok(pi_half * (iv_neg - iv_pos) / sin_vpi_c);
    }

    // Integer order: use recurrence
    let n = v.abs().round() as u32;
    Ok(complex_kn_integer(n, z))
}

/// Complex K_n(z) for integer order: accurate K_0/K_1 (DLMF 10.31.2 series for
/// |z| < 11, asymptotic above), then the upward recurrence
/// K_{n+1} = K_{n-1} + (2n/z)K_n, which is stable (K grows with order).
fn complex_kn_integer(n: u32, z: Complex64) -> Complex64 {
    let z_inv = z.recip();
    let mut k_prev = complex_k0(z);
    if n == 0 {
        return k_prev;
    }
    let mut k_curr = complex_k1(z);
    if n == 1 {
        return k_curr;
    }
    for k in 1..n {
        let coeff = Complex64::new(2.0 * k as f64, 0.0) * z_inv;
        let next = k_prev + coeff * k_curr;
        k_prev = k_curr;
        k_curr = next;
    }
    k_curr
}

/// Complex K_0(z) — DLMF 10.31.2 ascending series for |z| < 11, asymptotic above.
fn complex_k0(z: Complex64) -> Complex64 {
    if z.abs() >= COMPLEX_KY_ASYMP {
        return complex_kv_asymptotic(0.0, z);
    }
    // K_0 = −(ln(z/2)+γ)I_0(z) + Σ_{k≥1} H_k (z²/4)^k/(k!)².
    let i0 = complex_iv_scalar(0.0, z);
    let z2_4 = z * z / 4.0;
    let mut t = Complex64::new(1.0, 0.0);
    let mut s = Complex64::new(0.0, 0.0);
    let mut h = 0.0_f64;
    for k in 1..200 {
        let kf = k as f64;
        t = t * z2_4 / (kf * kf);
        h += 1.0 / kf;
        let term = t * Complex64::new(h, 0.0);
        s = s + term;
        if k > 3 && term.abs() < 1e-17 * s.abs().max(1e-300) {
            break;
        }
    }
    let log_term = (z / 2.0).ln() + Complex64::new(BESSEL_EULER_GAMMA, 0.0);
    Complex64::new(-1.0, 0.0) * log_term * i0 + s
}

/// Complex K_1(z) — DLMF 10.31.2 ascending series for |z| < 11, asymptotic above.
fn complex_k1(z: Complex64) -> Complex64 {
    if z.abs() >= COMPLEX_KY_ASYMP {
        return complex_kv_asymptotic(1.0, z);
    }
    // K_1 = (ln(z/2)+γ)I_1(z) + 1/z
    //       − (1/2)(z/2) Σ_{k≥0} (H_k+H_{k+1})(z²/4)^k/(k!(k+1)!).
    let i1 = complex_iv_scalar(1.0, z);
    let z2_4 = z * z / 4.0;
    let mut t = Complex64::new(1.0, 0.0);
    let mut s = Complex64::new(0.0, 0.0);
    let mut hk = 0.0_f64;
    for k in 0..200 {
        let kf = k as f64;
        let hk1 = hk + 1.0 / (kf + 1.0);
        let term = t * Complex64::new(hk + hk1, 0.0);
        s = s + term;
        if k > 3 && term.abs() < 1e-17 * s.abs().max(1e-300) {
            break;
        }
        t = t * z2_4 / ((kf + 1.0) * (kf + 2.0));
        hk = hk1;
    }
    let log_term = (z / 2.0).ln() + Complex64::new(BESSEL_EULER_GAMMA, 0.0);
    log_term * i1 + z.recip() - Complex64::new(0.5, 0.0) * (z / 2.0) * s
}

fn hankel_dispatch(
    function: &'static str,
    v: &SpecialTensor,
    z: &SpecialTensor,
    mode: RuntimeMode,
    kind: HankelKind,
) -> SpecialResult {
    match (v, z) {
        (SpecialTensor::RealScalar(order), SpecialTensor::RealScalar(x)) => {
            hankel_real_scalar(function, *order, *x, mode, kind).map(SpecialTensor::ComplexScalar)
        }
        (SpecialTensor::RealVec(orders), SpecialTensor::RealScalar(x)) => orders
            .iter()
            .map(|&order| hankel_real_scalar(function, order, *x, mode, kind))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::RealScalar(order), SpecialTensor::RealVec(xs)) => xs
            .iter()
            .map(|&x| hankel_real_scalar(function, *order, x, mode, kind))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::RealVec(orders), SpecialTensor::RealVec(xs)) => {
            if orders.len() != xs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            orders
                .iter()
                .zip(xs.iter())
                .map(|(&order, &x)| hankel_real_scalar(function, order, x, mode, kind))
                .collect::<Result<Vec<_>, _>>()
                .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::RealScalar(order), SpecialTensor::ComplexScalar(z_val)) => {
            hankel_complex_scalar(function, *order, *z_val, mode, kind)
                .map(SpecialTensor::ComplexScalar)
        }
        (SpecialTensor::RealScalar(order), SpecialTensor::ComplexVec(zs)) => zs
            .iter()
            .map(|&z_val| hankel_complex_scalar(function, *order, z_val, mode, kind))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::RealVec(orders), SpecialTensor::ComplexScalar(z_val)) => orders
            .iter()
            .map(|&order| hankel_complex_scalar(function, order, *z_val, mode, kind))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::RealVec(orders), SpecialTensor::ComplexVec(zs)) => {
            if orders.len() != zs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            orders
                .iter()
                .zip(zs.iter())
                .map(|(&order, &z_val)| hankel_complex_scalar(function, order, z_val, mode, kind))
                .collect::<Result<Vec<_>, _>>()
                .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::ComplexScalar(_), _) | (SpecialTensor::ComplexVec(_), _) => {
            not_yet_implemented(function, mode, "complex-valued order not supported")
        }
        (SpecialTensor::Empty, _) | (_, SpecialTensor::Empty) => Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "empty tensor is not a valid special-function input",
        }),
    }
}

#[derive(Clone, Copy)]
enum HankelScale {
    First,
    Second,
}

fn scaled_hankel_dispatch(
    function: &'static str,
    v: &SpecialTensor,
    z: &SpecialTensor,
    mode: RuntimeMode,
    kind: HankelKind,
    scale: HankelScale,
) -> SpecialResult {
    match (v, z) {
        (SpecialTensor::RealScalar(order), SpecialTensor::RealScalar(x)) => {
            hankel_real_scalar(function, *order, *x, mode, kind)
                .map(|value| scale_hankel(value, Complex64::new(*x, 0.0), scale))
                .map(SpecialTensor::ComplexScalar)
        }
        (SpecialTensor::RealVec(orders), SpecialTensor::RealScalar(x)) => orders
            .iter()
            .map(|&order| {
                hankel_real_scalar(function, order, *x, mode, kind)
                    .map(|value| scale_hankel(value, Complex64::new(*x, 0.0), scale))
            })
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::RealScalar(order), SpecialTensor::RealVec(xs)) => xs
            .iter()
            .map(|&x| {
                hankel_real_scalar(function, *order, x, mode, kind)
                    .map(|value| scale_hankel(value, Complex64::new(x, 0.0), scale))
            })
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::RealVec(orders), SpecialTensor::RealVec(xs)) => {
            if orders.len() != xs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            orders
                .iter()
                .zip(xs.iter())
                .map(|(&order, &x)| {
                    hankel_real_scalar(function, order, x, mode, kind)
                        .map(|value| scale_hankel(value, Complex64::new(x, 0.0), scale))
                })
                .collect::<Result<Vec<_>, _>>()
                .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::RealScalar(order), SpecialTensor::ComplexScalar(z_val)) => {
            hankel_complex_scalar(function, *order, *z_val, mode, kind)
                .map(|value| scale_hankel(value, *z_val, scale))
                .map(SpecialTensor::ComplexScalar)
        }
        (SpecialTensor::RealScalar(order), SpecialTensor::ComplexVec(zs)) => zs
            .iter()
            .map(|&z_val| {
                hankel_complex_scalar(function, *order, z_val, mode, kind)
                    .map(|value| scale_hankel(value, z_val, scale))
            })
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::RealVec(orders), SpecialTensor::ComplexScalar(z_val)) => orders
            .iter()
            .map(|&order| {
                hankel_complex_scalar(function, order, *z_val, mode, kind)
                    .map(|value| scale_hankel(value, *z_val, scale))
            })
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::RealVec(orders), SpecialTensor::ComplexVec(zs)) => {
            if orders.len() != zs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            orders
                .iter()
                .zip(zs.iter())
                .map(|(&order, &z_val)| {
                    hankel_complex_scalar(function, order, z_val, mode, kind)
                        .map(|value| scale_hankel(value, z_val, scale))
                })
                .collect::<Result<Vec<_>, _>>()
                .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::ComplexScalar(_), _) | (SpecialTensor::ComplexVec(_), _) => {
            not_yet_implemented(function, mode, "complex-valued order not supported")
        }
        (SpecialTensor::Empty, _) | (_, SpecialTensor::Empty) => Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "empty tensor is not a valid special-function input",
        }),
    }
}

fn scale_hankel(value: Complex64, z: Complex64, scale: HankelScale) -> Complex64 {
    let exponent = match scale {
        HankelScale::First => Complex64::new(z.im, -z.re),
        HankelScale::Second => Complex64::new(-z.im, z.re),
    };
    value * exponent.exp()
}

fn hankel_real_scalar(
    function: &'static str,
    order: f64,
    x: f64,
    mode: RuntimeMode,
    kind: HankelKind,
) -> Result<Complex64, SpecialError> {
    let j = jv_scalar(order, x);
    let y = yv_scalar(order, x, mode)?;
    let imag = match kind {
        HankelKind::H1 => y,
        HankelKind::H2 => -y,
    };
    if !j.is_finite() || !imag.is_finite() {
        record_special_trace(
            function,
            mode,
            "non_finite_output",
            format!("order={order},x={x}"),
            "returned_non_finite",
            format!("real={j},imag={imag}"),
            false,
        );
    }
    Ok(Complex64::new(j, imag))
}

fn hankel_complex_scalar(
    function: &'static str,
    order: f64,
    z: Complex64,
    mode: RuntimeMode,
    kind: HankelKind,
) -> Result<Complex64, SpecialError> {
    let j = bessel_complex_scalar(function, order, z, mode, BesselKind::Jv)?;
    let y = bessel_complex_scalar(function, order, z, mode, BesselKind::Yv)?;
    Ok(match kind {
        HankelKind::H1 => j + Complex64::new(-y.im, y.re),
        HankelKind::H2 => j + Complex64::new(y.im, -y.re),
    })
}

#[derive(Clone, Copy, Debug)]
enum BesselKind {
    Jv,
    Yv,
    Jve,
    Yve,
    Iv,
    Kv,
    Ive,
    Kve,
}

fn bessel_dispatch(
    function: &'static str,
    v: &SpecialTensor,
    z: &SpecialTensor,
    mode: RuntimeMode,
    kind: BesselKind,
) -> SpecialResult {
    match (v, z) {
        // Real-real: delegate to existing real scalars
        (SpecialTensor::RealScalar(order), SpecialTensor::RealScalar(x)) => {
            let result = match kind {
                BesselKind::Jv => Ok(jv_scalar(*order, *x)),
                BesselKind::Yv => yv_scalar(*order, *x, mode),
                BesselKind::Jve => Ok(jv_scalar(*order, *x)),
                BesselKind::Yve => yv_scalar(*order, *x, mode),
                BesselKind::Iv => Ok(iv_scalar(*order, *x)),
                BesselKind::Kv => kv_scalar(*order, *x, mode),
                BesselKind::Ive => Ok(ive_scalar(*order, *x)),
                BesselKind::Kve => Ok(kve_scalar(*order, *x)),
            };
            result.map(SpecialTensor::RealScalar)
        }
        (SpecialTensor::RealVec(orders), SpecialTensor::RealScalar(x)) => orders
            .iter()
            .map(|&order| match kind {
                BesselKind::Jv => Ok(jv_scalar(order, *x)),
                BesselKind::Yv => yv_scalar(order, *x, mode),
                BesselKind::Jve => Ok(jv_scalar(order, *x)),
                BesselKind::Yve => yv_scalar(order, *x, mode),
                BesselKind::Iv => Ok(iv_scalar(order, *x)),
                BesselKind::Kv => kv_scalar(order, *x, mode),
                BesselKind::Ive => Ok(ive_scalar(order, *x)),
                BesselKind::Kve => Ok(kve_scalar(order, *x)),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        (SpecialTensor::RealScalar(order), SpecialTensor::RealVec(xs)) => xs
            .iter()
            .map(|&x| match kind {
                BesselKind::Jv => Ok(jv_scalar(*order, x)),
                BesselKind::Yv => yv_scalar(*order, x, mode),
                BesselKind::Jve => Ok(jv_scalar(*order, x)),
                BesselKind::Yve => yv_scalar(*order, x, mode),
                BesselKind::Iv => Ok(iv_scalar(*order, x)),
                BesselKind::Kv => kv_scalar(*order, x, mode),
                BesselKind::Ive => Ok(ive_scalar(*order, x)),
                BesselKind::Kve => Ok(kve_scalar(*order, x)),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        (SpecialTensor::RealVec(orders), SpecialTensor::RealVec(xs)) => {
            if orders.len() != xs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::ShapeMismatch,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            orders
                .iter()
                .zip(xs.iter())
                .map(|(&order, &x)| match kind {
                    BesselKind::Jv => Ok(jv_scalar(order, x)),
                    BesselKind::Yv => yv_scalar(order, x, mode),
                    BesselKind::Jve => Ok(jv_scalar(order, x)),
                    BesselKind::Yve => yv_scalar(order, x, mode),
                    BesselKind::Iv => Ok(iv_scalar(order, x)),
                    BesselKind::Kv => kv_scalar(order, x, mode),
                    BesselKind::Ive => Ok(ive_scalar(order, x)),
                    BesselKind::Kve => Ok(kve_scalar(order, x)),
                })
                .collect::<Result<Vec<_>, _>>()
                .map(SpecialTensor::RealVec)
        }
        // Real order, complex z
        (SpecialTensor::RealScalar(order), SpecialTensor::ComplexScalar(z_val)) => {
            bessel_complex_scalar(function, *order, *z_val, mode, kind)
                .map(SpecialTensor::ComplexScalar)
        }
        (SpecialTensor::RealScalar(order), SpecialTensor::ComplexVec(zs)) => zs
            .iter()
            .map(|&z_val| bessel_complex_scalar(function, *order, z_val, mode, kind))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::RealVec(orders), SpecialTensor::ComplexScalar(z_val)) => orders
            .iter()
            .map(|&order| bessel_complex_scalar(function, order, *z_val, mode, kind))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        (SpecialTensor::RealVec(orders), SpecialTensor::ComplexVec(zs)) => {
            if orders.len() != zs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::ShapeMismatch,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            orders
                .iter()
                .zip(zs.iter())
                .map(|(&order, &z_val)| bessel_complex_scalar(function, order, z_val, mode, kind))
                .collect::<Result<Vec<_>, _>>()
                .map(SpecialTensor::ComplexVec)
        }
        // Complex order not supported
        (SpecialTensor::ComplexScalar(_), _) | (SpecialTensor::ComplexVec(_), _) => {
            not_yet_implemented(function, mode, "complex-valued order not supported")
        }
        // Empty
        (SpecialTensor::Empty, _) | (_, SpecialTensor::Empty) => Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "empty tensor is not a valid special-function input",
        }),
    }
}

fn bessel_complex_scalar(
    _function: &'static str,
    order: f64,
    z: Complex64,
    mode: RuntimeMode,
    kind: BesselKind,
) -> Result<Complex64, SpecialError> {
    match kind {
        BesselKind::Jv => Ok(complex_jv_scalar(order, z)),
        BesselKind::Yv => complex_yv_scalar(order, z, mode),
        BesselKind::Jve => Ok(scale_cylindrical_bessel(complex_jv_scalar(order, z), z)),
        BesselKind::Yve => {
            complex_yv_scalar(order, z, mode).map(|value| scale_cylindrical_bessel(value, z))
        }
        BesselKind::Iv => Ok(complex_iv_scalar(order, z)),
        BesselKind::Kv => complex_kv_scalar(order, z, mode),
        BesselKind::Ive => Ok(complex_iv_scalar(order, z) * (-z.re.abs()).exp()),
        BesselKind::Kve => complex_kv_scalar(order, z, mode).map(|k| k * z.re.exp()),
    }
}

fn scale_cylindrical_bessel(value: Complex64, z: Complex64) -> Complex64 {
    value * (-z.im.abs()).exp()
}

/// Riccati-Bessel function of the first kind, returning
/// `(S_k(x), S'_k(x))` for `k = 0..=n`.
///
/// Matches `scipy.special.riccati_jn(n, x)`. Defined as
///   S_k(x) = x · j_k(x)
/// where j_k is the spherical Bessel function of the first kind.
/// The values follow the same upward recurrence as j_k; the derivatives
/// use S'_0 = cos(x) and S'_k = -k S_k(x)/x + S_{k-1}(x) for k ≥ 1.
pub fn riccati_jn(n: u32, x: f64) -> (Vec<f64>, Vec<f64>) {
    let len = n as usize + 1;
    let mut s = Vec::with_capacity(len);
    let mut sp = Vec::with_capacity(len);
    for k in 0..=n {
        let jk = spherical_jn_nonneg(k, x);
        s.push(x * jk);
    }
    sp.push(x.cos());
    for k in 1..=n as usize {
        let inv_x = if x.abs() < 1.0e-300 { 0.0 } else { 1.0 / x };
        let val = -((k as f64) * s[k] * inv_x) + s[k - 1];
        sp.push(val);
    }
    (s, sp)
}

/// Riccati-Bessel function of the second kind, returning
/// `(C_k(x), C'_k(x))` for `k = 0..=n`.
///
/// Matches `scipy.special.riccati_yn(n, x)`. Defined as
///   C_k(x) = x · y_k(x)
/// where y_k is the spherical Bessel function of the second kind.
/// `C_0(x) = -cos(x)` and `C'_0(x) = sin(x)`; higher orders use the
/// derivative identity C'_k = -k C_k(x)/x + C_{k-1}(x). Pre-fix the
/// implementation seeded `C_0 = +cos(x)`, opposite of scipy and of
/// the upstream Riccati-Bessel convention — frankenscipy-gt5x9.
pub fn riccati_yn(n: u32, x: f64) -> (Vec<f64>, Vec<f64>) {
    let len = n as usize + 1;
    let mut c = Vec::with_capacity(len);
    let mut cp = Vec::with_capacity(len);
    for k in 0..=n {
        let yk = spherical_yn_nonneg(k, x);
        c.push(x * yk);
    }
    cp.push(x.sin());
    for k in 1..=n as usize {
        let inv_x = if x.abs() < 1.0e-300 { 0.0 } else { 1.0 / x };
        let val = -((k as f64) * c[k] * inv_x) + c[k - 1];
        cp.push(val);
    }
    (c, cp)
}

/// Olver/Tricomi initial-guess for the first zero of J_n at large n,
/// from DLMF 10.21.40 (asymptotic expansion in n^(-1/3)):
///   j_{n,1} ≈ n + a·n^(1/3) + b·n^(-1/3) − c·n^(-1) − d·n^(-5/3)
/// with a ≈ 1.855_757, b ≈ 1.033_150, c ≈ 0.003_972, d ≈ 0.090_8.
fn olver_jn_first_zero(n_f: f64) -> f64 {
    let n_one_third = n_f.cbrt();
    let inv_n_one_third = 1.0 / n_one_third;
    let inv_n = 1.0 / n_f;
    let inv_n_five_thirds = inv_n * inv_n_one_third * inv_n_one_third;
    n_f + 1.855_757_081_206_722 * n_one_third + 1.033_150_071_8 * inv_n_one_third
        - 0.003_972 * inv_n
        - 0.090_8 * inv_n_five_thirds
}

/// Olver/Tricomi initial-guess for the first zero of Y_n at large n
/// (DLMF 10.21.40 second-kind variant).
fn olver_yn_first_zero(n_f: f64) -> f64 {
    let n_one_third = n_f.cbrt();
    let inv_n_one_third = 1.0 / n_one_third;
    let inv_n = 1.0 / n_f;
    let inv_n_five_thirds = inv_n * inv_n_one_third * inv_n_one_third;
    n_f + 0.931_577_4 * n_one_third + 0.260_086 * inv_n_one_third
        - 0.011_911 * inv_n
        - 0.043_4 * inv_n_five_thirds
}

/// Refine a bessel-zero initial guess by bracket expansion + bisection.
/// Walks outward from `initial` in steps of `step`, bounded by [`floor`,
/// initial+max_outward], until a sign change of `f_at` is found, then
/// bisects to ~1e-12 precision. Returns `initial` unchanged if no
/// bracket can be found within `max_outward`.
fn bracket_and_bisect_zero(
    f_at: impl Fn(f64) -> f64,
    initial: f64,
    floor: f64,
    step: f64,
    max_outward: f64,
) -> f64 {
    let mut lo = (initial - step).max(floor);
    let mut hi = (initial + step).max(lo + step);
    let mut f_lo = f_at(lo);
    let mut f_hi = f_at(hi);
    let mut walked = step;
    while walked < max_outward {
        if f_lo.is_finite() && f_hi.is_finite() && f_lo.signum() != f_hi.signum() {
            break;
        }
        let new_lo = (lo - step).max(floor);
        let new_hi = hi + step;
        if new_lo == lo && new_hi == hi {
            break;
        }
        lo = new_lo;
        hi = new_hi;
        f_lo = f_at(lo);
        f_hi = f_at(hi);
        walked += step;
    }
    if !f_lo.is_finite() || !f_hi.is_finite() || f_lo.signum() == f_hi.signum() {
        return initial;
    }
    for _ in 0..120 {
        let mid = 0.5 * (lo + hi);
        let f_mid = f_at(mid);
        if !f_mid.is_finite() {
            break;
        }
        if f_mid.signum() == f_lo.signum() {
            lo = mid;
            f_lo = f_mid;
        } else {
            hi = mid;
        }
        if (hi - lo) < 1.0e-12 {
            break;
        }
    }
    0.5 * (lo + hi)
}

fn bisect_bracketed_zero(f_at: impl Fn(f64) -> f64, mut lo: f64, mut hi: f64) -> f64 {
    let mut f_lo = f_at(lo);
    let f_hi = f_at(hi);
    if !lo.is_finite()
        || !hi.is_finite()
        || !f_lo.is_finite()
        || !f_hi.is_finite()
        || f_lo.signum() == f_hi.signum()
    {
        return 0.5 * (lo + hi);
    }

    for _ in 0..120 {
        let mid = 0.5 * (lo + hi);
        let f_mid = f_at(mid);
        if !f_mid.is_finite() {
            break;
        }
        if f_mid == 0.0 {
            return mid;
        }
        if f_mid.signum() == f_lo.signum() {
            lo = mid;
            f_lo = f_mid;
        } else {
            hi = mid;
        }
        if (hi - lo) < 1.0e-12 {
            break;
        }
    }
    0.5 * (lo + hi)
}

/// First `k` positive zeros of the Bessel function Y_n(x), for integer
/// order `n ≥ 0`. Returns a Vec of length `k`, sorted ascending.
///
/// Matches `scipy.special.yn_zeros(n, k)`. Initial guess is piecewise:
/// for k=1 with n ≥ 5 use Olver/Tricomi (DLMF 10.21.40); otherwise use
/// McMahon's asymptotic
///   μ = (4 k + 2 n - 3) · π / 4
///   y_{n,k} ≈ μ - (4 n² - 1)/(8 μ) - (4 n² - 1)(28 n² - 31)/(384 μ³)
/// (DLMF 10.21.19 second-kind variant — offset differs from jn_zeros by
/// 2 because zeros of Y_n interlace those of J_n). Refines via bracket
/// expansion + bisection.
pub fn yn_zeros(n: u32, k: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(k);
    let n_f = n as f64;
    let four_n_sq = 4.0 * n_f * n_f;
    let mut prev_zero = 0.0_f64;
    for ki in 1..=k {
        let mu = (4.0 * ki as f64 + 2.0 * n_f - 3.0) * std::f64::consts::PI / 4.0;
        let inv_mu = 1.0 / mu;
        let inv_mu2 = inv_mu * inv_mu;
        let mcmahon = mu
            - (four_n_sq - 1.0) / (8.0 * mu)
            - (four_n_sq - 1.0) * (28.0 * four_n_sq - 31.0) / 384.0 * inv_mu * inv_mu2;
        let initial = if ki == 1 && n >= 5 {
            olver_yn_first_zero(n_f)
        } else if ki >= 2 {
            // Successive zeros are separated by ~π; if the asymptotic
            // mcmahon falls below prev_zero + π/4 (sign that mcmahon is
            // too low for this regime), nudge it up.
            mcmahon.max(prev_zero + 0.5 * std::f64::consts::PI)
        } else {
            mcmahon
        };
        let f_at = |x: f64| -> f64 { yn_scalar(n_f, x, RuntimeMode::Strict).unwrap_or(f64::NAN) };
        let floor = if ki >= 2 { prev_zero + 1.0e-6 } else { 1.0e-6 };
        let zero = bracket_and_bisect_zero(f_at, initial, floor, 0.5, (n_f * 0.25).max(20.0));
        out.push(zero);
        prev_zero = zero;
    }
    out
}

/// First `nt` positive zeros of `Y_0(x)` and SciPy's companion values.
///
/// Matches the real branch of `scipy.special.y0_zeros(nt)`. SciPy returns
/// the zeros of `Y_0` and the corresponding `Y_1(zero)` values.
pub fn y0_zeros(nt: usize) -> (Vec<f64>, Vec<f64>) {
    let zeros = yn_zeros(0, nt);
    let values = zeros
        .iter()
        .map(|&z| y1_scalar(z, RuntimeMode::Strict).unwrap_or(f64::NAN))
        .collect();
    (zeros, values)
}

/// First `nt` positive zeros of `Y_1(x)` and derivative values.
///
/// Matches the real branch of `scipy.special.y1_zeros(nt)`. At a zero
/// of `Y_1`, the derivative value is `Y_0(zero)`.
pub fn y1_zeros(nt: usize) -> (Vec<f64>, Vec<f64>) {
    let zeros = yn_zeros(1, nt);
    let values = zeros
        .iter()
        .map(|&z| y0_scalar(z, RuntimeMode::Strict).unwrap_or(f64::NAN))
        .collect();
    (zeros, values)
}

/// First `nt` positive zeros of `J_n'(x)`.
///
/// Matches `scipy.special.jnp_zeros(n, nt)` on the real nonnegative integer
/// order branch. For `n = 0`, `J_0'(x) = -J_1(x)`, so the roots are exactly
/// the roots of `J_1`; otherwise derivative roots interlace the roots of
/// `J_n`, with the first derivative root before the first function root.
pub fn jnp_zeros(n: u32, nt: usize) -> Vec<f64> {
    if n == 0 {
        return jn_zeros(1, nt);
    }

    let function_zeros = jn_zeros(n, nt);
    let n_f = n as f64;
    let f_at = |x: f64| -> f64 {
        bessel_derivative_real_scalar(
            "jnp_zeros",
            n_f,
            x,
            1,
            RuntimeMode::Strict,
            BesselKind::Jv,
            DerivativeRule::Alternating,
        )
        .unwrap_or(f64::NAN)
    };

    let mut out = Vec::with_capacity(nt);
    for idx in 0..nt {
        let lo = if idx == 0 {
            1.0e-6
        } else {
            function_zeros[idx - 1] + 1.0e-6
        };
        let hi = function_zeros[idx] - 1.0e-6;
        out.push(bisect_bracketed_zero(f_at, lo, hi));
    }
    out
}

/// First `nt` positive zeros of `Y_n'(x)`.
///
/// Matches `scipy.special.ynp_zeros(n, nt)` on the real nonnegative integer
/// order branch. For `n = 0`, `Y_0'(x) = -Y_1(x)`; otherwise derivative
/// roots interlace successive roots of `Y_n`.
pub fn ynp_zeros(n: u32, nt: usize) -> Vec<f64> {
    if n == 0 {
        return yn_zeros(1, nt);
    }

    let function_zeros = yn_zeros(n, nt + 1);
    let n_f = n as f64;
    let f_at = |x: f64| -> f64 {
        bessel_derivative_real_scalar(
            "ynp_zeros",
            n_f,
            x,
            1,
            RuntimeMode::Strict,
            BesselKind::Yv,
            DerivativeRule::Alternating,
        )
        .unwrap_or(f64::NAN)
    };

    let mut out = Vec::with_capacity(nt);
    for idx in 0..nt {
        let lo = function_zeros[idx] + 1.0e-6;
        let hi = function_zeros[idx + 1] - 1.0e-6;
        out.push(bisect_bracketed_zero(f_at, lo, hi));
    }
    out
}

/// First `nt` positive zeros of `Y_1'(x)` and `Y_1` values there.
///
/// Matches the real branch of `scipy.special.y1p_zeros(nt)`.
pub fn y1p_zeros(nt: usize) -> (Vec<f64>, Vec<f64>) {
    let zeros = ynp_zeros(1, nt);
    let values = zeros
        .iter()
        .map(|&z| y1_scalar(z, RuntimeMode::Strict).unwrap_or(f64::NAN))
        .collect();
    (zeros, values)
}

/// First `k` positive zeros of the Bessel function J_n(x), for integer
/// order `n ≥ 0`. Returns a Vec of length `k`, sorted ascending.
///
/// Matches `scipy.special.jn_zeros(n, k)`. Initial guess is piecewise:
/// for k=1 with n ≥ 5 use Olver/Tricomi (DLMF 10.21.40); otherwise use
/// McMahon's asymptotic
///   μ = (4 k + 2 n - 1) · π / 4 = (k + n/2 − 1/4) · π
///   j_{n,k} ≈ μ - (4 n² - 1)/(8 μ) - (4 n² - 1)(28 n² - 31)/(384 μ³)
/// (DLMF 10.21.19). Refines via bracket expansion + bisection. Bisection
/// is preferred over Newton for the same basin-flip robustness reason
/// as ai_zeros / bi_zeros. Without the Olver fallback, McMahon's
/// expansion produces useless guesses for moderate-large n at small k
/// (e.g. n=100, k=1 places McMahon at ~120 vs the true 108.84).
pub fn jn_zeros(n: u32, k: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(k);
    let n_f = n as f64;
    let four_n_sq = 4.0 * n_f * n_f;
    let mut prev_zero = 0.0_f64;
    for ki in 1..=k {
        let mu = (4.0 * ki as f64 + 2.0 * n_f - 1.0) * std::f64::consts::PI / 4.0;
        let inv_mu = 1.0 / mu;
        let inv_mu2 = inv_mu * inv_mu;
        let mcmahon = mu
            - (four_n_sq - 1.0) / (8.0 * mu)
            - (four_n_sq - 1.0) * (28.0 * four_n_sq - 31.0) / 384.0 * inv_mu * inv_mu2;
        let initial = if ki == 1 && n >= 5 {
            olver_jn_first_zero(n_f)
        } else if ki >= 2 {
            mcmahon.max(prev_zero + 0.5 * std::f64::consts::PI)
        } else {
            mcmahon
        };
        let f_at = |x: f64| -> f64 { jn_scalar(n_f, x, RuntimeMode::Strict).unwrap_or(f64::NAN) };
        let floor = if ki >= 2 { prev_zero + 1.0e-6 } else { 1.0e-6 };
        let zero = bracket_and_bisect_zero(f_at, initial, floor, 0.5, (n_f * 0.25).max(20.0));
        out.push(zero);
        prev_zero = zero;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy
    #[allow(clippy::type_complexity)] // flat (v,re,im,Y.re,Y.im) golden rows
    fn complex_yv_turning_band_matches_scipy() {
        // frankenscipy-d2s72: non-integer complex Y_v in the turning band
        // v < |z| ≤ v² was wrong (the (J cos − J_{-v})/sin form needs J_{-v},
        // whose negative-order Miller is only ~1e-3; Y's own recurrence is
        // unstable through the turning point). Computed instead from the Hankel
        // relation H₁_v(z)=(2/π)i^{-(v+1)}K_v(ze^{-iπ/2}), Y_v=(H₁_v−J_v)/i,
        // using the now-exact complex J_v and K_v; Im(z)<0 via Y_v(z̄)=conj.
        // (v, re, im, Y.re, Y.im) — scipy 1.17.1.
        let cases: [(f64, f64, f64, f64, f64); 7] = [
            (10.5, 30.0, 9.3, 425.31459982514065, -197.0281497959184),
            (20.5, 25.0, 9.0, -1.7446466112634633, 31.83128401682869),
            (50.5, 33.0, 11.0, -405.9697322325668, -182.52477151349686),
            (5.5, 15.0, 5.0, 4.13393906623684, 10.44359307054551),
            (2.5, -30.0, 10.0, -1394.4536042627415, -600.5226490392904),
            (20.5, -25.0, -12.0, -247.5694151344469, -235.6656094173197),
            (0.5, -8.0, 3.0, -2.5906873013164318, -0.8761557188492526),
        ];
        for (v, re, im, yr, yi) in cases {
            let z = Complex64::new(re, im);
            let y = complex_yv_scalar(v, z, RuntimeMode::Strict).unwrap();
            let err = (y.re - yr).hypot(y.im - yi) / yr.hypot(yi);
            assert!(err <= 1e-7, "yv({v},{re}{im:+}i) = {y:?}, scipy ({yr},{yi}), rel {err:e}");
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy
    #[allow(clippy::type_complexity)] // flat (v,re,im,K.re,K.im) golden rows
    fn complex_kv_turning_band_matches_scipy() {
        // frankenscipy-d2s72: non-integer complex K_v in the turning band
        // v < |z| ≤ v² (Re(z) ≥ 0) was computed by (I_{-v}−I_v)/sin(vπ), which
        // cancels (both I ~ e^z). K grows monotonically with order, so seeding
        // the K asymptotic at small order and walking up the recurrence is
        // stable (unlike oscillatory Y). Re(z) < 0 (across the branch cut) is
        // reflected via K_v(we^{±iπ})=e^{∓ivπ}K_v(w)∓iπI_v(w).
        // (v, re, im, K.re, K.im) — scipy 1.17.1.
        let cases: [(f64, f64, f64, f64, f64); 11] = [
            (10.5, 30.0, 9.3, -9.37954079129976e-14, 5.344341257316493e-14),
            (20.5, 25.0, 9.0, 2.080674029070471e-09, 3.6929289885929565e-09),
            (30.5, 45.0, 11.0, 4.9377593759569705e-17, -4.154102966627179e-17),
            (50.5, 55.0, 18.0, 5.0678599670482056e-17, 9.429501885540872e-17),
            (5.5, 15.0, 5.0, 1.5307782517384106e-07, 1.7251233742654897e-07),
            (20.5, 15.0, 12.0, 0.0003422838796743065, 0.00031697965000494787),
            // Re(z) < 0 (branch-cut reflection):
            (0.5, -3.0, 2.0, -13.136389329139432, 1.7869727121447632),
            (5.5, -20.0, 8.0, -66248408.63984236, 14978308.422065388),
            (10.5, -25.0, 15.0, 1037005.5000110489, 3260618052.855438),
            (2.5, -50.0, -10.0, 4.0274493691085514e20, -7.581988572080506e20),
            (20.5, -30.0, -12.0, -5781507165.759946, 200323296.50961077),
        ];
        for (v, re, im, kr, ki) in cases {
            let z = Complex64::new(re, im);
            let k = complex_kv_scalar(v, z, RuntimeMode::Strict).unwrap();
            let err = (k.re - kr).hypot(k.im - ki) / kr.hypot(ki);
            assert!(err <= 1e-7, "kv({v},{re}{im:+}i) = {k:?}, scipy ({kr},{ki}), rel {err:e}");
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy
    #[allow(clippy::type_complexity)] // flat (v,re,im,J,I) golden rows
    fn complex_jv_iv_turning_band_matches_scipy() {
        // frankenscipy-8eiog: for large order in the turning band v < |z| ≤ v²
        // the power series cancels and the Hankel/I asymptotics diverge. J is
        // recessive as order grows, so Miller's backward recurrence (normalized
        // by a small-order asymptotic) is stable; I follows via I_v(z)=
        // e^{-iπv/2}J_v(iz). (v, re, im, J.re,J.im, I.re,I.im) from scipy 1.17.1.
        let cases: [(f64, f64, f64, f64, f64, f64, f64); 5] = [
            (10.5, 30.0, 9.3, -197.02815202783466, -425.31457727367575, -136745728601.30203, -34365056209.05175),
            (20.5, 25.0, 9.0, 31.831636545639334, 1.7443462385888058, 1050761.4293616053, -3459922.56166499),
            (30.5, 38.0, 11.0, 28.8899240212221, -89.82134451310553, 9940418282.002474, 30996195925.81287),
            (50.5, 55.0, 18.0, -900.4238940608079, 26.820654557258933, 19190288570304.97, -59319339392746.0),
            (20.5, -50.0, 35.0, -10625161413727.812, 2880710047906.2197, -1.3356153802331707e19, 7.901885428551122e18),
        ];
        for (v, re, im, jr, ji, ir, ii) in cases {
            let z = Complex64::new(re, im);
            let j = complex_jv_scalar(v, z);
            let i = complex_iv_scalar(v, z);
            let je = (j.re - jr).hypot(j.im - ji) / jr.hypot(ji);
            let ie = (i.re - ir).hypot(i.im - ii) / ir.hypot(ii);
            assert!(je <= 1e-7, "jv({v},{re}{im:+}i) = {j:?}, scipy ({jr},{ji}), rel {je:e}");
            assert!(ie <= 1e-7, "iv({v},{re}{im:+}i) = {i:?}, scipy ({ir},{ii}), rel {ie:e}");
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy
    #[allow(clippy::type_complexity)] // flat (v,re,im,J,I,K,Y) golden rows
    fn complex_bessel_noninteger_large_z_matches_scipy() {
        // frankenscipy-oisri: complex_jv/iv used a 200-term power series for all
        // z that truncates+cancels for |z|≳30; the non-integer kv = π/2(I_{-v}−I_v)
        // /sin(vπ) form cancelled further (both I~e^z). Added the Hankel J
        // asymptotic (with the Re<0 reflection), I via J(iz), and the direct K
        // asymptotic. (v, re, im, J.re,J.im, I.re,I.im, K.re,K.im, Y.re,Y.im) — scipy 1.17.1.
        let cases: [(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64); 5] = [
            (0.5, 20.0, 5.0, 12.466521336498948, 3.8267134139998005, 7006986.415091062, -42048916.84546594, 2.2684059886175998e-10, 5.21771888934066e-10, -3.8273250606136826, 12.465507499599681),
            (2.5, -15.0, 25.0, 4820258766.728313, -670823000.802485, -121420.56920674515, 194474.00807032152, 381453.9682146355, -610958.115067951, 670823000.802485, 4820258766.728313),
            (3.5, 40.0, -10.0, -1109.4573598197273, 720.8276509954086, -1.119458545990043e16, 5954167327505190.0, -7.091076212196013e-19, -6.373437988023484e-19, 720.8276552862686, 1109.4573558385662),
            (0.5, 3.0, 80.0, -1.4455656554351904e33, -2.0033277552023882e33, -0.6900384488490181, -0.5742502926122344, 0.004254020290227443, 0.005526238447046366, 2.0033277552023882e33, -1.4455656554351904e33),
            (2.5, 50.0, 3.0, 0.19792798151697677, -1.1086923774604178, -2.7126068652301323e20, 4.59770369446714e19, -3.6026674628426933e-23, -3.9101945736867964e-24, 1.1141728356305414, 0.19661157971907242),
        ];
        for (v, re, im, jr, ji, ir, ii, kr, ki, yr, yi) in cases {
            let z = Complex64::new(re, im);
            let checks = [
                (complex_jv_scalar(v, z), jr, ji, "jv"),
                (complex_iv_scalar(v, z), ir, ii, "iv"),
                (complex_kv_scalar(v, z, RuntimeMode::Strict).unwrap(), kr, ki, "kv"),
                (complex_yv_scalar(v, z, RuntimeMode::Strict).unwrap(), yr, yi, "yv"),
            ];
            for (got, wr, wi, name) in checks {
                let err = (got.re - wr).hypot(got.im - wi) / wr.hypot(wi);
                assert!(err <= 1e-7, "{name}({v},{re}{im:+}i) = {got:?}, scipy ({wr},{wi}), rel {err:e}");
            }
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy
    #[allow(clippy::type_complexity)] // flat (v,re,im,K.re,K.im,Y.re,Y.im) golden rows
    fn complex_kv_yv_integer_order_matches_scipy() {
        // frankenscipy-wan3w: integer-order complex K_n/Y_n used crude
        // placeholders — complex_y0/y1_series dropped the entire harmonic series,
        // complex_kn_integer used only the leading asymptotic — so they were
        // wrong even at small |z| (kv(2,1.9+0.6i) ~22% off). Rebuilt from DLMF
        // 10.8.1 / 10.31.2 ascending series (|z|<11) and the large-z asymptotics,
        // with the stable upward recurrence for n≥2.
        // (v, re, im, K.re, K.im, Y.re, Y.im) from scipy.special 1.17.1.
        let cases: [(u32, f64, f64, f64, f64, f64, f64); 8] = [
            (0, 2.0, 1.0, 0.037987722915986476, -0.10171357546139093, 0.800451120409994, 0.07563855028639382),
            (1, 1.5, 2.5, -0.17583976180149624, -0.005975889337860949, -0.6409702450434701, 2.4512074350972797),
            (2, 3.0, 1.0, 0.015275965061514699, -0.05547174952705652, -0.1433635707870101, 0.4576996364039667),
            (2, 0.5, 4.0, 0.18495000454282312, 0.38087077161509364, -3.137626774336867, -5.632589456759534),
            (5, 8.0, 3.0, -0.000460035945850506, 0.00023640292674529355, 1.4290304684892552, 0.9410036389912462),
            (0, 15.0, 5.0, 4.129869836775042e-08, 8.634770156118545e-08, 14.529566891182334, -3.4121917632978587),
            (1, 5.0, 30.0, 0.001172778369302158, 0.0009898467040185962, -157028907008.00497, -747235851320.0383),
            (3, 2.0, -3.0, -0.06909578637591238, -0.1727382937957655, -1.0614832363273328, 0.8158048933143148),
        ];
        for (v, re, im, kr, ki, yr, yi) in cases {
            let z = Complex64::new(re, im);
            let k = complex_kv_scalar(v as f64, z, RuntimeMode::Strict).unwrap();
            let y = complex_yv_scalar(v as f64, z, RuntimeMode::Strict).unwrap();
            let ke = (k.re - kr).hypot(k.im - ki) / kr.hypot(ki);
            let ye = (y.re - yr).hypot(y.im - yi) / yr.hypot(yi);
            assert!(ke <= 1e-7, "kv({v},{re}{im:+}i) = {k:?}, scipy ({kr},{ki}), rel {ke:e}");
            assert!(ye <= 1e-7, "yv({v},{re}{im:+}i) = {y:?}, scipy ({yr},{yi}), rel {ye:e}");
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy
    fn wofz_near_real_axis_matches_scipy() {
        // frankenscipy-wsv5b: the upper-half-plane contour integral has a pole at
        // t=Re(z) that fixed-step Simpson misses for small Im (h≈0.03), so
        // wofz(0.5+0.001i) was ~10× off, error growing as Im→0 across |Im|≲0.15.
        // For |z|<4 use the exact relation w(z)=e^{-z²}erfc(-iz) (pole-free erf
        // series). (re, im, w.re, w.im) from scipy.special.wofz 1.17.1.
        let cases: [(f64, f64, f64, f64); 7] = [
            (0.001, 0.001, 0.9988716223354106, 0.001126380671599899),
            (0.5, 0.001, 0.7781517183125487, 0.47814717512158444),
            (1.0, 0.005, 0.3682999758137983, 0.6034919113460572),
            (2.0, 0.02, 0.022898797594497745, 0.33851256446284966),
            (0.5, -0.05, 0.8122678146281099, 0.519958322379095),
            (3.0, 0.0005, 0.00016269164507317786, 0.20115693817442962),
            (-2.0, 0.01, 0.020620065445569127, -0.3392813705802114),
        ];
        for (re, im, wr, wi) in cases {
            let w = crate::convenience::wofz_scalar(Complex64::new(re, im), RuntimeMode::Strict)
                .unwrap();
            let err = (w.re - wr).hypot(w.im - wi) / wr.hypot(wi);
            assert!(err <= 1e-9, "wofz({re}{im:+}i) = {w:?}, scipy ({wr},{wi}), rel {err:e}");
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy
    fn jv_yv_turning_point_and_recessive_matches_scipy() {
        // frankenscipy-87poa: for z ≤ |v| the old power-series cutoff (z < 20+|v|)
        // ran the ascending series deep into its cancellation region — jv(30.5,50)
        // gave 1.48 vs -0.0084 and jv(200.5,198.5) was ~1e29 off — while the
        // J_{±v} reflection made yv lose the dominant Y term (yv(100.5,95.5) ~1000×
        // off). Miller backward recurrence (with rescaling) for J and the upward
        // recurrence for the dominant Y now cover the turning-point and recessive
        // regions. (v, z, jv, yv) from scipy.special 1.17.1.
        let cases: [(f64, f64, f64, f64); 10] = [
            (100.5, 90.5, 0.0026209176462256884, -2.803268172454871),
            (100.5, 99.5, 0.07758801076002489, -0.2006774017433331),
            (200.5, 198.5, 0.05333846277237729, -0.17645897377188122),
            (100.5, 50.3, 9.689355774601216e-22, -3.775863610873226e18),
            (200.5, 140.4, 5.556422162397472e-18, -400245671792266.44),
            (50.5, 48.0, 0.054388185233499184, -0.3759710492003596),
            (30.5, 29.0, 0.08422010831116396, -0.37454996333301277),
            (-100.5, 90.5, 2.803268172454871, 0.0026209176462256884),
            (-30.5, 29.0, 0.37454996333301277, 0.08422010831116396),
            (100.5, 80.4, 4.3861274964605445e-06, -1204.4817952546605),
        ];
        for (v, z, jref, yref) in cases {
            let j = jv_scalar(v, z);
            let y = yv_scalar(v, z, RuntimeMode::Strict).unwrap();
            assert!((j - jref).abs() <= 1e-9 * jref.abs().max(1e-12), "jv({v},{z}) = {j:e}, scipy {jref:e}");
            assert!((y - yref).abs() <= 1e-9 * yref.abs().max(1e-12), "yv({v},{z}) = {y:e}, scipy {yref:e}");
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy
    fn jv_yv_oscillatory_large_order_matches_scipy() {
        // frankenscipy-goaov: in the oscillatory band |v| < z ≤ v² the DLMF
        // 10.17.3 large-z asymptotic diverges (term ratio ~v²/(2z)>1), so
        // jv(10.5,50) was 0.0296 vs scipy -0.0848 and yv was 43% off. Stable
        // recurrences (Miller backward for J, upward for the dominant Y) now
        // track scipy. (v, z, jv, yv) from scipy.special 1.17.1.
        let cases: [(f64, f64, f64, f64); 10] = [
            (10.5, 50.0, -0.08484972094355323, 0.07630487814534202),
            (20.5, 50.0, -0.08905749444593426, 0.07762984235393049),
            (10.5, 80.0, 0.07504844558951558, 0.04893610385008825),
            (30.5, 80.0, 0.08105179077034935, -0.045144937097620484),
            (50.5, 150.0, -0.06698277098437858, 0.004528086280292741),
            (100.5, 300.0, 0.014234331967893813, 0.04527228395681286),
            (10.25, 50.0, -0.10540198817035573, 0.043568700883024794),
            (15.75, 80.0, 0.07816700781318475, -0.044791887989034496),
            (-10.5, 50.0, -0.07630487814534202, -0.08484972094355323),
            (-20.5, 80.0, 0.052184524915355836, 0.07422369444265345),
        ];
        for (v, z, jref, yref) in cases {
            let j = jv_scalar(v, z);
            let y = yv_scalar(v, z, RuntimeMode::Strict).unwrap();
            assert!((j - jref).abs() <= 1e-9 * jref.abs().max(1e-3), "jv({v},{z}) = {j}, scipy {jref}");
            assert!((y - yref).abs() <= 1e-9 * yref.abs().max(1e-3), "yv({v},{z}) = {y}, scipy {yref}");
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy
    fn iv_large_order_power_series_matches_scipy() {
        // frankenscipy-4icoi: the large-argument asymptotic (valid only for
        // z >> v²) was used whenever z>50, so for large order its 4v²/(8z) series
        // diverged into huge NEGATIVE garbage (iv(50,100) was -8.3e44 vs scipy
        // 4.8e36; iv must be positive). Gating it to z>v² and routing the rest
        // through the ascending power series fixes it. (v, z, scipy iv).
        let cases: [(f64, f64, f64); 7] = [
            (50.0, 100.0, 4.821958085594079e36),
            (100.0, 100.0, 4.641534941616278e21),
            (100.0, 500.0, 1.1637732868603707e211),
            (20.0, 200.0, 7.49106766376834e84),
            (75.0, 100.0, 1.8288935933501197e30),
            (50.0, 200.0, 4.003924798366755e82),
            (30.0, 80.0, 9.1987338426633e30),
        ];
        for (v, z, want) in cases {
            let got = iv_scalar(v, z);
            assert!(got > 0.0, "iv({v},{z}) = {got} must be positive");
            assert!(
                (got - want).abs() <= 1e-10 * want.abs(),
                "iv({v},{z}) = {got:e}, scipy {want:e}"
            );
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy/mpmath
    fn jv_yv_noninteger_order_large_z_matches_scipy() {
        // Non-integer order large-z now uses the full DLMF 10.17.3 J_v asymptotic
        // (frankenscipy-rbmy5): the leading-order form was ~1% off (yv ~2.6%).
        // (v, z, jv_scipy, yv_scipy) from scipy.special 1.17.1.
        let cases = [
            (2.5, 200.0, 0.04885452923635855, 0.02822361750823702),
            (3.5, 500.0, -0.031335750692154954, -0.017068709147445744),
            (2.5, 50.0, 0.02303721950962553, 0.11053044455625441),
            (4.5, 300.0, -0.046065538475612275, -0.0005175841159171315),
            (0.5, 100.0, -0.04040213271625212, -0.0688030914687281),
            (10.5, 400.0, 0.03650518806158982, -0.016108011911492963),
        ];
        for (v, z, jref, yref) in cases {
            let jval = real_value(tensor_result(jv(&scalar(v), &scalar(z), RuntimeMode::Strict)).unwrap())
                .unwrap();
            let yval = real_value(tensor_result(yv(&scalar(v), &scalar(z), RuntimeMode::Strict)).unwrap())
                .unwrap();
            assert!(((jval - jref) / jref).abs() < 1e-10, "jv({v},{z})={jval:e} vs {jref:e}");
            assert!(((yval - yref) / yref).abs() < 1e-10, "yv({v},{z})={yval:e} vs {yref:e}");
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy/mpmath
    fn kv_kve_large_z_matches_scipy() {
        // Large-z K_v via the DLMF 10.40.2 asymptotic (frankenscipy-c66a3): the
        // integral path was 12-39% off (its tiny magnitude defeated the
        // absolute-tolerance adaptive Simpson). Integer and non-integer order.
        let rv = |r: SpecialResult| match r {
            Ok(SpecialTensor::RealScalar(v)) => v,
            _ => f64::NAN,
        };
        let s = SpecialTensor::RealScalar;
        let m = RuntimeMode::Strict;
        let cases = [
            (0.0, 100.0, 4.656628229175903e-45, 0.1251756216591266),
            (1.0, 100.0, 4.67985373563691e-45, 0.12579995047957854),
            (3.0, 200.0, 1.253501761543211e-88, 0.0905777084721066),
            (5.0, 500.0, 4.093284751762465e-219, 0.05745302623029479),
            (2.5, 200.0, 1.2449350429724718e-88, 0.08995867963539583),
            (0.5, 50.0, 3.418620095457075e-23, 0.1772453850905516),
            (10.5, 400.0, 1.376812560442129e-175, 0.07188985052835141),
        ];
        for (v, z, kref, keref) in cases {
            let kval = rv(kv(&s(v), &s(z), m));
            let keval = rv(kve(&s(v), &s(z), m));
            assert!(((kval - kref) / kref).abs() < 1e-10, "kv({v},{z})={kval:e} vs {kref:e}");
            assert!(((keval - keref) / keref).abs() < 1e-10, "kve({v},{z})={keval:e} vs {keref:e}");
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy/mpmath
    fn kv_kve_integral_window_and_scaled_overflow() {
        // frankenscipy-j3bw7: (a) large-v moderate-z still uses the integral
        // (below the asymptotic threshold) — the e^{-z}-factored, saddle-aware
        // form fixes it; (b) kve stays finite past z≈745 where kv underflows.
        let rv = |r: SpecialResult| match r {
            Ok(SpecialTensor::RealScalar(v)) => v,
            _ => f64::NAN,
        };
        let s = SpecialTensor::RealScalar;
        let m = RuntimeMode::Strict;
        let cases = [
            (15.0, 100.0, 1.4234832511447141e-44, 0.3826489728490269),
            (20.0, 150.0, 2.765588292853233e-66, 0.3854426899928329),
            (10.0, 40.0, 2.868029311367192e-18, 0.6750918447525612),
            (30.0, 40.0, 3.6670011340654733e-14, 8631.58040433656),
            (2.5, 10.0, 2.3931325864627893e-05, 0.5271225305815995),
        ];
        for (v, z, kref, keref) in cases {
            let kval = rv(kv(&s(v), &s(z), m));
            let keval = rv(kve(&s(v), &s(z), m));
            assert!(((kval - kref) / kref).abs() < 1e-10, "kv({v},{z})={kval:e} vs {kref:e}");
            assert!(((keval - keref) / keref).abs() < 1e-10, "kve({v},{z})={keval:e} vs {keref:e}");
        }
        // z > 745: kv underflows to 0 (matching scipy), kve stays finite.
        for (v, z, keref) in [
            (0.0, 750.0, 0.045756939928889066),
            (2.0, 800.0, 0.04441525775942454),
            (1.0, 900.0, 0.04179453901303371),
        ] {
            assert_eq!(rv(kv(&s(v), &s(z), m)), 0.0, "kv({v},{z}) should underflow to 0");
            let keval = rv(kve(&s(v), &s(z), m));
            assert!(((keval - keref) / keref).abs() < 1e-10, "kve({v},{z})={keval:e} vs {keref:e}");
        }
    }

    fn scalar(value: f64) -> SpecialTensor {
        SpecialTensor::RealScalar(value)
    }

    fn tensor_result(result: SpecialResult) -> Result<SpecialTensor, String> {
        result.map_err(|err| err.to_string())
    }

    fn real_value(tensor: SpecialTensor) -> Result<f64, String> {
        match tensor {
            SpecialTensor::RealScalar(value) => Ok(value),
            other => Err(format!("expected RealScalar, got {other:?}")),
        }
    }

    fn complex_value(tensor: SpecialTensor) -> Result<Complex64, String> {
        match tensor {
            SpecialTensor::ComplexScalar(value) => Ok(value),
            other => Err(format!("expected ComplexScalar, got {other:?}")),
        }
    }

    fn assert_complex_close(actual: Complex64, expected: Complex64, tol: f64) {
        assert!(
            (actual.re - expected.re).abs() < tol,
            "real mismatch: actual={} expected={}",
            actual.re,
            expected.re
        );
        assert!(
            (actual.im - expected.im).abs() < tol,
            "imag mismatch: actual={} expected={}",
            actual.im,
            expected.im
        );
    }

    #[test]
    fn j0_y0_small_arguments_match_scipy_reference_points() -> Result<(), String> {
        let cases: &[(f64, f64, f64)] = &[
            (1.0e-12, 1.0, -17.664_258_668_214_952),
            (0.001, 0.999_999_750_000_015_5, -4.471_416_611_375_923),
            (0.5, 0.938_469_807_240_813, -0.444_518_733_506_706_6),
            (1.0, 0.765_197_686_557_966_5, 0.088_256_964_215_676_97),
            (7.5, 0.266_339_657_880_378_4, 0.117_313_286_148_208_63),
            (7.999, 0.171_885_372_282_320_45, 0.223_363_307_307_185_32),
        ];

        for &(x, expected_j0, expected_y0) in cases {
            let got_j0 = real_value(tensor_result(j0(&scalar(x), RuntimeMode::Strict))?)?;
            let got_y0 = real_value(tensor_result(y0(&scalar(x), RuntimeMode::Strict))?)?;
            assert!(
                (got_j0 - expected_j0).abs() < 2.0e-14,
                "j0({x}) = {got_j0}, expected {expected_j0}"
            );
            assert!(
                (got_y0 - expected_y0).abs() < 2.0e-13,
                "y0({x}) = {got_y0}, expected {expected_y0}"
            );
        }
        Ok(())
    }

    #[test]
    fn jvp_zero_order_matches_jv() -> Result<(), String> {
        let v = scalar(2.0);
        let x = scalar(1.25);
        let derivative = real_value(tensor_result(jvp(&v, &x, 0, RuntimeMode::Strict))?)?;
        let base = real_value(tensor_result(jv(&v, &x, RuntimeMode::Strict))?)?;
        assert!((derivative - base).abs() < 1e-14);
        Ok(())
    }

    #[test]
    fn jvp_first_and_second_derivatives_match_recurrences() -> Result<(), String> {
        let x = 1.25;
        let first = real_value(tensor_result(jvp(
            &scalar(0.0),
            &scalar(x),
            1,
            RuntimeMode::Strict,
        ))?)?;
        let expected_first = -real_value(tensor_result(j1(&scalar(x), RuntimeMode::Strict))?)?;
        assert!((first - expected_first).abs() < 1e-12);

        let second = real_value(tensor_result(jvp(
            &scalar(0.0),
            &scalar(x),
            2,
            RuntimeMode::Strict,
        ))?)?;
        let j0_x = real_value(tensor_result(j0(&scalar(x), RuntimeMode::Strict))?)?;
        let j2_x = real_value(tensor_result(jn(
            &scalar(2.0),
            &scalar(x),
            RuntimeMode::Strict,
        ))?)?;
        let expected_second = 0.5 * (j2_x - j0_x);
        assert!((second - expected_second).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn yvp_first_derivative_matches_y0_identity() -> Result<(), String> {
        let x = 1.75;
        let derivative = real_value(tensor_result(yvp(
            &scalar(0.0),
            &scalar(x),
            1,
            RuntimeMode::Strict,
        ))?)?;
        let expected = -real_value(tensor_result(y1(&scalar(x), RuntimeMode::Strict))?)?;
        assert!((derivative - expected).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn ivp_and_kvp_first_derivatives_match_order_zero_identities() -> Result<(), String> {
        let x = 1.1;
        let iv_derivative = real_value(tensor_result(ivp(
            &scalar(0.0),
            &scalar(x),
            1,
            RuntimeMode::Strict,
        ))?)?;
        let iv_expected = real_value(tensor_result(iv(
            &scalar(1.0),
            &scalar(x),
            RuntimeMode::Strict,
        ))?)?;
        assert!((iv_derivative - iv_expected).abs() < 1e-12);

        let kv_derivative = real_value(tensor_result(kvp(
            &scalar(0.0),
            &scalar(x),
            1,
            RuntimeMode::Strict,
        ))?)?;
        let kv_expected = -real_value(tensor_result(kv(
            &scalar(1.0),
            &scalar(x),
            RuntimeMode::Strict,
        ))?)?;
        assert!((kv_derivative - kv_expected).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn complex_bessel_derivative_zero_order_matches_base_functions() -> Result<(), String> {
        let z = complex_scalar(1.25, 0.75);

        let jvp_zero = complex_value(tensor_result(jvp(
            &scalar(0.5),
            &z,
            0,
            RuntimeMode::Strict,
        ))?)?;
        let jv_base = complex_value(tensor_result(jv(&scalar(0.5), &z, RuntimeMode::Strict))?)?;
        assert!((jvp_zero.re - jv_base.re).abs() < 1e-12);
        assert!((jvp_zero.im - jv_base.im).abs() < 1e-12);

        let yvp_zero = complex_value(tensor_result(yvp(
            &scalar(0.5),
            &z,
            0,
            RuntimeMode::Strict,
        ))?)?;
        let yv_base = complex_value(tensor_result(yv(&scalar(0.5), &z, RuntimeMode::Strict))?)?;
        assert!((yvp_zero.re - yv_base.re).abs() < 1e-12);
        assert!((yvp_zero.im - yv_base.im).abs() < 1e-12);

        let ivp_zero = complex_value(tensor_result(ivp(
            &scalar(0.5),
            &z,
            0,
            RuntimeMode::Strict,
        ))?)?;
        let iv_base = complex_value(tensor_result(iv(&scalar(0.5), &z, RuntimeMode::Strict))?)?;
        assert!((ivp_zero.re - iv_base.re).abs() < 1e-12);
        assert!((ivp_zero.im - iv_base.im).abs() < 1e-12);

        let kvp_zero = complex_value(tensor_result(kvp(
            &scalar(0.5),
            &z,
            0,
            RuntimeMode::Strict,
        ))?)?;
        let kv_base = complex_value(tensor_result(kv(&scalar(0.5), &z, RuntimeMode::Strict))?)?;
        assert!((kvp_zero.re - kv_base.re).abs() < 1e-12);
        assert!((kvp_zero.im - kv_base.im).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn complex_bessel_negative_integer_orders_stay_finite() {
        let z = Complex64::new(1.0, 0.0);

        assert_complex_close(
            complex_jv_scalar(-1.0, z),
            Complex64::new(-0.440_050_585_744_933_5, 0.0),
            1.0e-12,
        );
        assert_complex_close(
            complex_jv_scalar(-2.0, z),
            Complex64::new(0.114_903_484_931_900_5, 0.0),
            1.0e-12,
        );
        assert_complex_close(
            complex_iv_scalar(-1.0, z),
            Complex64::new(0.565_159_103_992_485, 0.0),
            1.0e-12,
        );
        assert_complex_close(
            complex_jv_scalar(-1.0, Complex64::new(0.0, 0.0)),
            Complex64::new(0.0, 0.0),
            1.0e-12,
        );
        assert_complex_close(
            complex_iv_scalar(-1.0, Complex64::new(0.0, 0.0)),
            Complex64::new(0.0, 0.0),
            1.0e-12,
        );
    }

    #[test]
    fn complex_bessel_derivative_reduces_to_real_axis_values() -> Result<(), String> {
        let x = 1.8;

        let j_real = real_value(tensor_result(jvp(
            &scalar(0.0),
            &scalar(x),
            1,
            RuntimeMode::Strict,
        ))?)?;
        let j_complex = complex_value(tensor_result(jvp(
            &scalar(0.0),
            &complex_scalar(x, 0.0),
            1,
            RuntimeMode::Strict,
        ))?)?;
        assert!((j_real - j_complex.re).abs() < 1e-12);
        assert!(j_complex.im.abs() < 1e-12);

        let y_real = real_value(tensor_result(yvp(
            &scalar(0.0),
            &scalar(x),
            1,
            RuntimeMode::Strict,
        ))?)?;
        let y_complex = complex_value(tensor_result(yvp(
            &scalar(0.0),
            &complex_scalar(x, 0.0),
            1,
            RuntimeMode::Strict,
        ))?)?;
        assert!((y_real - y_complex.re).abs() < 1e-12);
        assert!(y_complex.im.abs() < 1e-12);

        let i_real = real_value(tensor_result(ivp(
            &scalar(0.0),
            &scalar(x),
            1,
            RuntimeMode::Strict,
        ))?)?;
        let i_complex = complex_value(tensor_result(ivp(
            &scalar(0.0),
            &complex_scalar(x, 0.0),
            1,
            RuntimeMode::Strict,
        ))?)?;
        assert!((i_real - i_complex.re).abs() < 1e-12);
        assert!(i_complex.im.abs() < 1e-12);

        let k_real = real_value(tensor_result(kvp(
            &scalar(0.0),
            &scalar(x),
            1,
            RuntimeMode::Strict,
        ))?)?;
        let k_complex = complex_value(tensor_result(kvp(
            &scalar(0.0),
            &complex_scalar(x, 0.0),
            1,
            RuntimeMode::Strict,
        ))?)?;
        assert!((k_real - k_complex.re).abs() < 1e-12);
        assert!(k_complex.im.abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn complex_bessel_derivative_broadcasts_complex_vectors() -> Result<(), String> {
        let zs = SpecialTensor::ComplexVec(vec![
            Complex64::new(1.0, 0.5),
            Complex64::new(1.5, -0.25),
            Complex64::new(2.0, 0.0),
        ]);
        let result = tensor_result(jvp(&scalar(0.5), &zs, 1, RuntimeMode::Strict))?;
        match result {
            SpecialTensor::ComplexVec(values) => {
                assert_eq!(values.len(), 3);
                for (index, value) in values.iter().enumerate() {
                    let lane = match &zs {
                        SpecialTensor::ComplexVec(items) => items[index],
                        _ => unreachable!(),
                    };
                    let expected = complex_value(tensor_result(jvp(
                        &scalar(0.5),
                        &SpecialTensor::ComplexScalar(lane),
                        1,
                        RuntimeMode::Strict,
                    ))?)?;
                    assert!((value.re - expected.re).abs() < 1e-12);
                    assert!((value.im - expected.im).abs() < 1e-12);
                }
            }
            other => return Err(format!("expected ComplexVec, got {other:?}")),
        }
        Ok(())
    }

    #[test]
    fn hankel_derivatives_return_complex_conjugate_parts() -> Result<(), String> {
        let x = 2.0;
        let h1 = complex_value(tensor_result(h1vp(
            &scalar(0.0),
            &scalar(x),
            1,
            RuntimeMode::Strict,
        ))?)?;
        let h2 = complex_value(tensor_result(h2vp(
            &scalar(0.0),
            &scalar(x),
            1,
            RuntimeMode::Strict,
        ))?)?;
        let j_part = real_value(tensor_result(jvp(
            &scalar(0.0),
            &scalar(x),
            1,
            RuntimeMode::Strict,
        ))?)?;
        let y_part = real_value(tensor_result(yvp(
            &scalar(0.0),
            &scalar(x),
            1,
            RuntimeMode::Strict,
        ))?)?;

        assert!((h1.re - j_part).abs() < 1e-12);
        assert!((h1.im - y_part).abs() < 1e-12);
        assert!((h2.re - j_part).abs() < 1e-12);
        assert!((h2.im + y_part).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn complex_hankel_derivative_zero_order_matches_base_hankel() -> Result<(), String> {
        let z = complex_scalar(2.0, 1.0);
        let h1_derivative = complex_value(tensor_result(h1vp(
            &scalar(1.0),
            &z,
            0,
            RuntimeMode::Strict,
        ))?)?;
        let h1_base = complex_value(tensor_result(hankel1(
            &scalar(1.0),
            &z,
            RuntimeMode::Strict,
        ))?)?;
        assert!((h1_derivative.re - h1_base.re).abs() < 1e-12);
        assert!((h1_derivative.im - h1_base.im).abs() < 1e-12);

        let h2_derivative = complex_value(tensor_result(h2vp(
            &scalar(1.0),
            &z,
            0,
            RuntimeMode::Strict,
        ))?)?;
        let h2_base = complex_value(tensor_result(hankel2(
            &scalar(1.0),
            &z,
            RuntimeMode::Strict,
        ))?)?;
        assert!((h2_derivative.re - h2_base.re).abs() < 1e-12);
        assert!((h2_derivative.im - h2_base.im).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn complex_hankel_derivative_reduces_to_real_axis_values() -> Result<(), String> {
        let x = 2.0;
        let h1_real = complex_value(tensor_result(h1vp(
            &scalar(0.0),
            &scalar(x),
            1,
            RuntimeMode::Strict,
        ))?)?;
        let h1_complex = complex_value(tensor_result(h1vp(
            &scalar(0.0),
            &complex_scalar(x, 0.0),
            1,
            RuntimeMode::Strict,
        ))?)?;
        assert!((h1_real.re - h1_complex.re).abs() < 1e-12);
        assert!((h1_real.im - h1_complex.im).abs() < 1e-12);

        let h2_real = complex_value(tensor_result(h2vp(
            &scalar(0.0),
            &scalar(x),
            1,
            RuntimeMode::Strict,
        ))?)?;
        let h2_complex = complex_value(tensor_result(h2vp(
            &scalar(0.0),
            &complex_scalar(x, 0.0),
            1,
            RuntimeMode::Strict,
        ))?)?;
        assert!((h2_real.re - h2_complex.re).abs() < 1e-12);
        assert!((h2_real.im - h2_complex.im).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn complex_hankel_derivatives_broadcast_vectors() -> Result<(), String> {
        let zs = SpecialTensor::ComplexVec(vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(2.0, 1.0),
            Complex64::new(3.0, 0.0),
        ]);
        let h1_result = tensor_result(h1vp(&scalar(0.0), &zs, 1, RuntimeMode::Strict))?;
        match h1_result {
            SpecialTensor::ComplexVec(values) => {
                assert_eq!(values.len(), 3);
                for (index, value) in values.iter().enumerate() {
                    let z = match &zs {
                        SpecialTensor::ComplexVec(items) => items[index],
                        _ => unreachable!(),
                    };
                    let expected = complex_value(tensor_result(h1vp(
                        &scalar(0.0),
                        &SpecialTensor::ComplexScalar(z),
                        1,
                        RuntimeMode::Strict,
                    ))?)?;
                    assert!(
                        (value.re - expected.re).abs() < 1e-12
                            || (value.re.is_nan() && expected.re.is_nan())
                    );
                    assert!(
                        (value.im - expected.im).abs() < 1e-12
                            || (value.im.is_nan() && expected.im.is_nan())
                    );
                }
            }
            other => return Err(format!("expected ComplexVec, got {other:?}")),
        }

        let h2_result = tensor_result(h2vp(&scalar(0.0), &zs, 1, RuntimeMode::Strict))?;
        match h2_result {
            SpecialTensor::ComplexVec(values) => {
                assert_eq!(values.len(), 3);
                for (index, value) in values.iter().enumerate() {
                    let z = match &zs {
                        SpecialTensor::ComplexVec(items) => items[index],
                        _ => unreachable!(),
                    };
                    let expected = complex_value(tensor_result(h2vp(
                        &scalar(0.0),
                        &SpecialTensor::ComplexScalar(z),
                        1,
                        RuntimeMode::Strict,
                    ))?)?;
                    assert!(
                        (value.re - expected.re).abs() < 1e-12
                            || (value.re.is_nan() && expected.re.is_nan())
                    );
                    assert!(
                        (value.im - expected.im).abs() < 1e-12
                            || (value.im.is_nan() && expected.im.is_nan())
                    );
                }
            }
            other => return Err(format!("expected ComplexVec, got {other:?}")),
        }
        Ok(())
    }

    #[test]
    fn derivative_helpers_broadcast_vectors() -> Result<(), String> {
        let values = SpecialTensor::RealVec(vec![1.0, 2.0]);
        let result = tensor_result(jvp(&scalar(0.0), &values, 1, RuntimeMode::Strict))?;
        match result {
            SpecialTensor::RealVec(items) => {
                assert_eq!(items.len(), 2);
                assert!((items[0] + j1_core(1.0)).abs() < 1e-12);
                assert!((items[1] + j1_core(2.0)).abs() < 1e-12);
            }
            other => return Err(format!("expected RealVec, got {other:?}")),
        }
        Ok(())
    }

    #[test]
    fn yvp_hardened_rejects_negative_real_argument() -> Result<(), String> {
        let Err(err) = yvp(&scalar(0.0), &scalar(-1.0), 1, RuntimeMode::Hardened) else {
            return Err("negative argument should fail closed".to_string());
        };
        assert_eq!(err.kind, SpecialErrorKind::DomainError);
        Ok(())
    }

    #[test]
    fn hankel_tensor_interface_returns_full_complex_parts() -> Result<(), String> {
        let x = scalar(2.0);
        let h1 = complex_value(tensor_result(hankel1(
            &scalar(0.0),
            &x,
            RuntimeMode::Strict,
        ))?)?;
        let h2 = complex_value(tensor_result(hankel2(
            &scalar(0.0),
            &x,
            RuntimeMode::Strict,
        ))?)?;
        let j = real_value(tensor_result(jv(&scalar(0.0), &x, RuntimeMode::Strict))?)?;
        let y = real_value(tensor_result(yv(&scalar(0.0), &x, RuntimeMode::Strict))?)?;

        assert!((h1.re - j).abs() < 1e-12);
        assert!((h1.im - y).abs() < 1e-12);
        assert!((h2.re - j).abs() < 1e-12);
        assert!((h2.im + y).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn complex_hankel_tensor_interface_matches_jv_plus_minus_i_yv() -> Result<(), String> {
        let z = complex_scalar(2.0, 1.0);
        let j = complex_value(tensor_result(jv(&scalar(1.0), &z, RuntimeMode::Strict))?)?;
        let y = complex_value(tensor_result(yv(&scalar(1.0), &z, RuntimeMode::Strict))?)?;
        let h1 = complex_value(tensor_result(hankel1(
            &scalar(1.0),
            &z,
            RuntimeMode::Strict,
        ))?)?;
        let h2 = complex_value(tensor_result(hankel2(
            &scalar(1.0),
            &z,
            RuntimeMode::Strict,
        ))?)?;

        let iy = Complex64::new(-y.im, y.re);
        assert!((h1.re - (j + iy).re).abs() < 1e-12);
        assert!((h1.im - (j + iy).im).abs() < 1e-12);

        let minus_iy = Complex64::new(y.im, -y.re);
        assert!((h2.re - (j + minus_iy).re).abs() < 1e-12);
        assert!((h2.im - (j + minus_iy).im).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn hankel_broadcasts_real_vector_inputs_to_complex_vec() -> Result<(), String> {
        let xs = SpecialTensor::RealVec(vec![1.0, 2.0]);
        let result = tensor_result(hankel1(&scalar(0.0), &xs, RuntimeMode::Strict))?;
        match result {
            SpecialTensor::ComplexVec(values) => {
                assert_eq!(values.len(), 2);
                for value in values {
                    assert!(value.re.is_finite());
                    assert!(value.im.is_finite());
                }
            }
            other => return Err(format!("expected ComplexVec, got {other:?}")),
        }
        Ok(())
    }

    #[test]
    fn wright_bessel_zero_matches_rgamma_boundary() -> Result<(), String> {
        let result = real_value(tensor_result(wright_bessel(
            &scalar(0.0),
            &scalar(0.0),
            &scalar(0.0),
            RuntimeMode::Strict,
        ))?)?;
        assert_eq!(result, 0.0);

        let exp_result = real_value(tensor_result(wright_bessel(
            &scalar(0.0),
            &scalar(1.0),
            &scalar(2.0),
            RuntimeMode::Strict,
        ))?)?;
        assert!((exp_result - 2.0_f64.exp()).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn log_wright_bessel_zero_and_exponential_boundaries() -> Result<(), String> {
        let zero = real_value(tensor_result(log_wright_bessel(
            &scalar(0.0),
            &scalar(0.0),
            &scalar(0.0),
            RuntimeMode::Strict,
        ))?)?;
        assert_eq!(zero, f64::NEG_INFINITY);

        let exp_log = real_value(tensor_result(log_wright_bessel(
            &scalar(0.0),
            &scalar(1.0),
            &scalar(2.0),
            RuntimeMode::Strict,
        ))?)?;
        assert!((exp_log - 2.0).abs() < 1e-14);
        Ok(())
    }

    #[test]
    fn log_wright_bessel_matches_existing_log_domain_series() -> Result<(), String> {
        let cases: [(f64, f64, f64); 4] = [
            (1.0, 1.0, 3.0),
            (2.0, 4.0, 5.0),
            (10.0, 10.0, 100.0),
            (20.0, 1.5, 2.0),
        ];

        for (a, b, x) in cases {
            let logged = real_value(tensor_result(log_wright_bessel(
                &scalar(a),
                &scalar(b),
                &scalar(x),
                RuntimeMode::Strict,
            ))?)?;
            let direct = real_value(tensor_result(wright_bessel(
                &scalar(a),
                &scalar(b),
                &scalar(x),
                RuntimeMode::Strict,
            ))?)?;
            assert!(
                (logged.exp() - direct).abs() <= 1e-12 * direct.abs().max(1.0),
                "log_wright_bessel({a}, {b}, {x}) did not match wright_bessel"
            );
        }
        Ok(())
    }

    #[test]
    fn wright_bessel_matches_large_a_scipy_reference_values() -> Result<(), String> {
        let cases: [(f64, f64, f64, f64, f64); 4] = [
            (20.0, 1.5, 2.0, std::f64::consts::FRAC_2_SQRT_PI, 1e-13),
            (50.0, 0.25, 0.1, 0.275_815_662_830_209_3, 1e-13),
            (100.0, 0.0, 1e20, 1.071_510_288_125_446_2e-136, 1e-12),
            (100.0, 100.0, 1e20, 1.071_510_288_125_468_3e-156, 1e-12),
        ];

        for (a, b, x, expected, rel_tol) in cases {
            let actual = real_value(tensor_result(wright_bessel(
                &scalar(a),
                &scalar(b),
                &scalar(x),
                RuntimeMode::Strict,
            ))?)?;
            let scale = expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= rel_tol * scale,
                "wright_bessel({a}, {b}, {x}) mismatch: actual={actual}, expected={expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn wright_bessel_rejects_negative_domain_in_hardened_mode() -> Result<(), String> {
        let strict = real_value(tensor_result(wright_bessel(
            &scalar(-0.5),
            &scalar(2.0),
            &scalar(0.1),
            RuntimeMode::Strict,
        ))?)?;
        assert!(strict.is_nan());

        let Err(err) = wright_bessel(
            &scalar(-0.5),
            &scalar(2.0),
            &scalar(0.1),
            RuntimeMode::Hardened,
        ) else {
            return Err("negative wright_bessel input should fail closed".to_string());
        };
        assert_eq!(err.kind, SpecialErrorKind::DomainError);
        Ok(())
    }

    // ========================================================================
    // Complex spherical Bessel tests
    // ========================================================================

    fn complex_scalar(re: f64, im: f64) -> SpecialTensor {
        SpecialTensor::ComplexScalar(Complex64::new(re, im))
    }

    fn get_complex(result: SpecialResult) -> Result<Complex64, String> {
        match result.map_err(|e| e.to_string())? {
            SpecialTensor::ComplexScalar(c) => Ok(c),
            other => Err(format!("expected complex scalar, got {other:?}")),
        }
    }

    #[test]
    fn spherical_jn_small_x_large_n_matches_series() {
        // [frankenscipy-xlci0] regression: forward recurrence
        // catastrophically cancels when x << 2n+1. The previous
        // algorithm returned values off by 10× to many orders of
        // magnitude (e.g. n=9 gave -5.62 when the true value is ~3e-19).
        // Verify the Miller's-recurrence path agrees with the
        // small-x Taylor series j_n(x) ≈ x^n / (2n+1)!! * (1 -
        // x²/(2(2n+3))) which is accurate to ~1e-3 at x=0.1.
        let x = 0.1_f64;
        for n in 5_u32..=10 {
            let got = super::spherical_jn_nonneg(n, x);
            // Series truth: j_n ≈ x^n / (2n+1)!! · (1 − x²/(2(2n+3))).
            let mut double_fact = 1.0_f64;
            for k in 1..=n {
                double_fact *= (2 * k + 1) as f64;
            }
            let leading = x.powi(n as i32) / double_fact;
            let series_truth = leading * (1.0 - x * x / (2.0 * (2 * n + 3) as f64));
            let rel = ((got - series_truth) / series_truth).abs();
            // 1% accuracy is enough to reject the broken forward
            // recurrence (which returned wrong-sign or 10⁵× wrong
            // results).  Series itself is only accurate to a few
            // parts in 10⁴ at x=0.1.
            assert!(
                rel < 1e-2,
                "spherical_jn({n}, 0.1) = {got}, series truth ≈ {series_truth} (rel = {rel})"
            );
            // Also pin the sign — the catastrophic-cancellation
            // failure mode often flipped the sign.
            assert!(
                got > 0.0,
                "spherical_jn({n}, 0.1) = {got} should be positive"
            );
        }
    }

    #[test]
    fn spherical_jn_recurrence_threshold_continuity() {
        // The Miller-vs-forward switch happens at x = n. Verify the
        // result is continuous across that boundary by checking n=4
        // at three x values straddling x=n: 3.9, 4.0, 4.1.
        let prev = super::spherical_jn_nonneg(4, 3.9);
        let mid = super::spherical_jn_nonneg(4, 4.0);
        let next = super::spherical_jn_nonneg(4, 4.1);
        // Continuity: the three values should be close (within 5%
        // since j_n is smooth in x).
        assert!((mid - prev).abs() < 0.05_f64.max(prev.abs() * 0.05));
        assert!((next - mid).abs() < 0.05_f64.max(mid.abs() * 0.05));
    }

    #[test]
    fn complex_spherical_jn_reduces_to_real() -> Result<(), String> {
        // For real positive z, complex j_n should match real j_n
        let n = 2.0;
        let x = 1.5;

        let real_result = real_value(tensor_result(spherical_jn(
            &scalar(n),
            &scalar(x),
            RuntimeMode::Strict,
        ))?)?;

        let complex_result = get_complex(spherical_jn(
            &scalar(n),
            &complex_scalar(x, 0.0),
            RuntimeMode::Strict,
        ))?;

        assert!(
            (complex_result.re - real_result).abs() < 1e-10,
            "Real parts should match: {} vs {}",
            complex_result.re,
            real_result
        );
        assert!(
            complex_result.im.abs() < 1e-10,
            "Imaginary part should be negligible: {}",
            complex_result.im
        );
        Ok(())
    }

    #[test]
    fn complex_spherical_jn_at_zero() {
        let j0 = complex_spherical_jn(0, Complex64::new(0.0, 0.0));
        assert_eq!(j0.re, 1.0);
        assert_eq!(j0.im, 0.0);

        let j1 = complex_spherical_jn(1, Complex64::new(0.0, 0.0));
        assert_eq!(j1.re, 0.0);
        assert_eq!(j1.im, 0.0);
    }

    #[test]
    fn complex_spherical_yn_at_zero_is_negative_inf() {
        let y0 = complex_spherical_yn(0, Complex64::new(0.0, 0.0));
        assert!(y0.re.is_infinite() && y0.re < 0.0);
    }

    #[test]
    fn spherical_in_small_x_large_n_matches_series() {
        // [frankenscipy-0j009] regression: forward recurrence for
        // spherical i_n catastrophically cancels at small x (e.g.
        // n=10, x=0.1 gave −213 when the true value is ~7e−21).
        // Verify the Miller's-recurrence path matches the small-x
        // Taylor series i_n(x) ≈ x^n/(2n+1)!! · (1 + x²/(2(2n+3))).
        let x = 0.1_f64;
        for n in 5_u32..=10 {
            let got = super::spherical_in_nonneg(n, x);
            let mut double_fact = 1.0_f64;
            for k in 1..=n {
                double_fact *= (2 * k + 1) as f64;
            }
            let leading = x.powi(n as i32) / double_fact;
            let series_truth = leading * (1.0 + x * x / (2.0 * (2 * n + 3) as f64));
            let rel = ((got - series_truth) / series_truth).abs();
            assert!(
                rel < 1e-2,
                "spherical_in({n}, 0.1) = {got}, series truth ≈ {series_truth} (rel = {rel})"
            );
            assert!(
                got > 0.0,
                "spherical_in({n}, 0.1) = {got} should be positive"
            );
        }
    }

    #[test]
    fn spherical_in_negative_x_parity() {
        // i_n(-x) = (-1)^n · i_n(x); verify both even and odd n
        // round-trip through the Miller's path.
        let x = 0.1_f64;
        for n in 5_u32..=8 {
            let pos = super::spherical_in_nonneg(n, x);
            let neg = super::spherical_in_nonneg(n, -x);
            let expected = if n.is_multiple_of(2) { pos } else { -pos };
            assert!(
                (neg - expected).abs() < 1e-30 + 1e-10 * pos.abs(),
                "spherical_in({n}, -0.1) = {neg}, expected {expected}"
            );
        }
    }

    #[test]
    fn complex_spherical_jn_small_z_large_n_matches_real() {
        // [frankenscipy-tt2v2] regression: complex_spherical_jn forward
        // recurrence had the same catastrophic-cancellation defect as
        // the real path. For z on the real axis with small |z| and
        // moderate n, the complex result must match the real result
        // to high accuracy.
        for n in 5_u32..=10 {
            let real = super::spherical_jn_nonneg(n, 0.1);
            let complex = super::complex_spherical_jn(n, Complex64::new(0.1, 0.0));
            let rel = ((complex.re - real) / real).abs();
            assert!(
                rel < 1e-9,
                "complex_spherical_jn({n}, 0.1+0i).re = {}, real = {real} (rel = {rel})",
                complex.re
            );
            assert!(
                complex.im.abs() < 1e-25,
                "complex_spherical_jn({n}, 0.1+0i).im = {} should be ≈ 0",
                complex.im
            );
        }
    }

    #[test]
    fn complex_spherical_in_small_z_large_n_matches_real() {
        for n in 5_u32..=10 {
            let real = super::spherical_in_nonneg(n, 0.1);
            let complex = super::complex_spherical_in(n, Complex64::new(0.1, 0.0));
            let rel = ((complex.re - real) / real).abs();
            assert!(
                rel < 1e-9,
                "complex_spherical_in({n}, 0.1+0i).re = {}, real = {real} (rel = {rel})",
                complex.re
            );
            assert!(
                complex.im.abs() < 1e-25,
                "complex_spherical_in({n}, 0.1+0i).im = {} should be ≈ 0",
                complex.im
            );
        }
    }

    #[test]
    fn complex_spherical_in_reduces_to_real() -> Result<(), String> {
        let n = 1.0;
        let x = 2.0;

        let real_result = real_value(tensor_result(spherical_in(
            &scalar(n),
            &scalar(x),
            RuntimeMode::Strict,
        ))?)?;

        let complex_result = get_complex(spherical_in(
            &scalar(n),
            &complex_scalar(x, 0.0),
            RuntimeMode::Strict,
        ))?;

        assert!(
            (complex_result.re - real_result).abs() < 1e-8,
            "Real parts should match: {} vs {}",
            complex_result.re,
            real_result
        );
        Ok(())
    }

    #[test]
    fn complex_spherical_kn_reduces_to_real() -> Result<(), String> {
        let n = 0.0;
        let x = 1.0;

        let real_result = real_value(tensor_result(spherical_kn(
            &scalar(n),
            &scalar(x),
            RuntimeMode::Strict,
        ))?)?;

        let complex_result = get_complex(spherical_kn(
            &scalar(n),
            &complex_scalar(x, 0.0),
            RuntimeMode::Strict,
        ))?;

        assert!(
            (complex_result.re - real_result).abs() < 1e-10,
            "Real parts should match: {} vs {}",
            complex_result.re,
            real_result
        );
        Ok(())
    }

    #[test]
    fn complex_spherical_jn_tensor_interface() -> Result<(), String> {
        let result = spherical_jn(&scalar(2.0), &complex_scalar(1.0, 0.5), RuntimeMode::Strict);
        let c = get_complex(result)?;
        assert!(c.re.is_finite());
        assert!(c.im.is_finite());
        Ok(())
    }

    // ── Complex cylindrical Bessel function tests ───────────────────────

    #[test]
    fn complex_jv_real_matches_real_jv() -> Result<(), String> {
        let v = scalar(1.5);
        let z_real = scalar(2.0);
        let z_complex = complex_scalar(2.0, 0.0);

        let real_result = real_value(tensor_result(jv(&v, &z_real, RuntimeMode::Strict))?)?;
        let complex_result =
            complex_value(tensor_result(jv(&v, &z_complex, RuntimeMode::Strict))?)?;

        assert!((complex_result.re - real_result).abs() < 1e-10);
        assert!(complex_result.im.abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn complex_jv_pure_imaginary() -> Result<(), String> {
        let v = scalar(0.0);
        let z = complex_scalar(0.0, 2.0);
        let result = complex_value(tensor_result(jv(&v, &z, RuntimeMode::Strict))?)?;
        assert!(result.re.is_finite());
        assert!(result.im.is_finite());
        Ok(())
    }

    #[test]
    fn complex_iv_real_matches_real_iv() -> Result<(), String> {
        let v = scalar(1.0);
        let z_real = scalar(2.0);
        let z_complex = complex_scalar(2.0, 0.0);

        let real_result = real_value(tensor_result(iv(&v, &z_real, RuntimeMode::Strict))?)?;
        let complex_result =
            complex_value(tensor_result(iv(&v, &z_complex, RuntimeMode::Strict))?)?;

        assert!((complex_result.re - real_result).abs() < 1e-10);
        assert!(complex_result.im.abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn complex_yv_tensor_interface() -> Result<(), String> {
        let v = scalar(1.0);
        let z = complex_scalar(2.0, 1.0);
        let result = yv(&v, &z, RuntimeMode::Strict);
        let c = complex_value(tensor_result(result)?)?;
        assert!(c.re.is_finite());
        assert!(c.im.is_finite());
        Ok(())
    }

    #[test]
    fn scaled_cylindrical_bessel_real_inputs_match_base() -> Result<(), String> {
        let v = scalar(0.5);
        let x = scalar(1.25);

        let j_scaled = real_value(tensor_result(jve(&v, &x, RuntimeMode::Strict))?)?;
        let j_base = real_value(tensor_result(jv(&v, &x, RuntimeMode::Strict))?)?;
        let y_scaled = real_value(tensor_result(yve(&v, &x, RuntimeMode::Strict))?)?;
        let y_base = real_value(tensor_result(yv(&v, &x, RuntimeMode::Strict))?)?;

        assert!((j_scaled - j_base).abs() < 1e-14);
        assert!((y_scaled - y_base).abs() < 1e-14);
        Ok(())
    }

    #[test]
    fn scaled_cylindrical_bessel_complex_inputs_apply_scipy_scale() -> Result<(), String> {
        let v = scalar(0.5);
        let z_value = Complex64::new(1.25, 0.5);
        let z = SpecialTensor::ComplexScalar(z_value);
        let scale = (-z_value.im.abs()).exp();

        let j_scaled = complex_value(tensor_result(jve(&v, &z, RuntimeMode::Strict))?)?;
        let j_base = complex_value(tensor_result(jv(&v, &z, RuntimeMode::Strict))?)?;
        assert_complex_close(j_scaled, j_base * scale, 1e-12);

        let y_scaled = complex_value(tensor_result(yve(&v, &z, RuntimeMode::Strict))?)?;
        let y_base = complex_value(tensor_result(yv(&v, &z, RuntimeMode::Strict))?)?;
        assert_complex_close(y_scaled, y_base * scale, 1e-12);
        Ok(())
    }

    #[test]
    fn scaled_hankel_real_inputs_apply_scipy_phase() -> Result<(), String> {
        let v = scalar(1.0);
        let x_value = 1.25;
        let x = scalar(x_value);
        let z = Complex64::new(x_value, 0.0);

        let h1_scaled = complex_value(tensor_result(hankel1e(&v, &x, RuntimeMode::Strict))?)?;
        let h1_base = complex_value(tensor_result(hankel1(&v, &x, RuntimeMode::Strict))?)?;
        assert_complex_close(h1_scaled, h1_base * Complex64::new(0.0, -z.re).exp(), 1e-12);

        let h2_scaled = complex_value(tensor_result(hankel2e(&v, &x, RuntimeMode::Strict))?)?;
        let h2_base = complex_value(tensor_result(hankel2(&v, &x, RuntimeMode::Strict))?)?;
        assert_complex_close(h2_scaled, h2_base * Complex64::new(0.0, z.re).exp(), 1e-12);
        Ok(())
    }

    #[test]
    fn complex_kv_tensor_interface() -> Result<(), String> {
        let v = scalar(1.0);
        let z = complex_scalar(2.0, 1.0);
        let result = kv(&v, &z, RuntimeMode::Strict);
        let c = complex_value(tensor_result(result)?)?;
        assert!(c.re.is_finite());
        assert!(c.im.is_finite());
        Ok(())
    }

    #[test]
    fn complex_bessel_vector_broadcast() -> Result<(), String> {
        let v = scalar(0.0);
        let zs = SpecialTensor::ComplexVec(vec![
            Complex64::new(1.0, 0.5),
            Complex64::new(2.0, 0.0),
            Complex64::new(0.5, 1.0),
        ]);
        let result = tensor_result(jv(&v, &zs, RuntimeMode::Strict))?;
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

    #[test]
    fn riccati_jn_zero_order_is_sin_cos() {
        // S_0(x) = sin(x), S'_0(x) = cos(x).
        for &x in &[0.0_f64, 0.5, 1.0, 2.5, 5.0] {
            let (s, sp) = riccati_jn(0, x);
            assert_eq!(s.len(), 1);
            assert!((s[0] - x.sin()).abs() < 1e-12);
            assert!((sp[0] - x.cos()).abs() < 1e-12);
        }
    }

    #[test]
    fn riccati_jn_metamorphic_recurrence_consistency() {
        // S'_k = -k S_k / x + S_{k-1} for k ≥ 1. The function builds the
        // derivative using this identity, so verify it holds independently.
        let x = 2.5_f64;
        let n = 6;
        let (s, sp) = riccati_jn(n, x);
        for k in 1..=n as usize {
            let predicted = -(k as f64) * s[k] / x + s[k - 1];
            assert!(
                (sp[k] - predicted).abs() < 1e-12,
                "S'_{k}({x}) = {} vs recurrence {predicted}",
                sp[k]
            );
        }
    }

    #[test]
    fn riccati_yn_zero_order_matches_scipy_convention() {
        // scipy uses C_n(x) = x · y_n(x), so C_0(x) = -cos(x) and
        // C'_0(x) = sin(x). frankenscipy-gt5x9.
        for &x in &[0.5_f64, 1.0, 2.5, 5.0] {
            let (c, cp) = riccati_yn(0, x);
            assert!((c[0] - (-x.cos())).abs() < 1e-12);
            assert!((cp[0] - x.sin()).abs() < 1e-12);
        }
    }

    /// Anchor riccati_yn against scipy.special.riccati_yn for a grid of
    /// (n, x) — frankenscipy-gt5x9.
    #[test]
    fn riccati_yn_matches_scipy() {
        // (n, x, scipy C_n, scipy C'_n)
        let cases: [(u32, f64, f64, f64); 8] = [
            (0, 1.0, -0.5403023058681398, 0.8414709848078965),
            (0, 2.0, 0.4161468365471424, 0.9092974268256817),
            (1, 1.0, -1.3817732906760363, 0.8414709848078965),
            (1, 2.0, -0.7012240086157488, 0.7667588408010407),
            (1, 5.0, 0.9021918376441532, -0.46410055305030),
            (2, 1.0, -3.6050175661733055, 5.828261841594828),
            (2, 5.0, 0.8249772880, 0.5722009224),
            (3, 5.0, -0.0772145496, 0.8713060177),
        ];
        for (n, x, exp_c, exp_cp) in cases {
            let (c, cp) = riccati_yn(n, x);
            let last = n as usize;
            let scale_c = exp_c.abs().max(1.0);
            let scale_cp = exp_cp.abs().max(1.0);
            assert!(
                (c[last] - exp_c).abs() < 1e-9 * scale_c,
                "C_{n}({x}) = {}, expected {exp_c}",
                c[last]
            );
            assert!(
                (cp[last] - exp_cp).abs() < 1e-9 * scale_cp,
                "C'_{n}({x}) = {}, expected {exp_cp}",
                cp[last]
            );
        }
    }

    #[test]
    fn riccati_metamorphic_pythagorean_at_k_zero() {
        // S_0² + C_0² = sin²(x) + cos²(x) = 1.
        for &x in &[0.5_f64, 1.0, 2.5, 5.0] {
            let (s, _) = riccati_jn(0, x);
            let (c, _) = riccati_yn(0, x);
            let sum = s[0] * s[0] + c[0] * c[0];
            assert!(
                (sum - 1.0).abs() < 1e-13,
                "S_0² + C_0² = {sum} should be 1 at x={x}"
            );
        }
    }

    #[test]
    fn yn_zeros_first_three_of_y0_match_known() {
        // First three zeros of Y_0:
        //   0.893577_695_5
        //   3.957678_419_3
        //   7.086051_060_3
        let zeros = yn_zeros(0, 3);
        let expected = [0.893_577_695_5_f64, 3.957_678_419_3, 7.086_051_060_3];
        assert_eq!(zeros.len(), 3);
        for (got, exp) in zeros.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-4, "y0 zero {got} vs {exp}");
        }
    }

    #[test]
    fn y0_y1_zeros_wrappers_match_scipy_reference_points() {
        let (y0_z, y0_v) = y0_zeros(3);
        let expected_y0_z = [
            0.893_576_966_279_167_5_f64,
            3.957_678_419_314_857_5,
            7.086_051_060_301_744,
        ];
        let expected_y0_v = [
            -0.879_420_802_497_195_f64,
            0.402_542_671_775_024_3,
            -0.300_097_614_910_467_16,
        ];
        for ((z, v), (ez, ev)) in y0_z
            .iter()
            .zip(y0_v.iter())
            .zip(expected_y0_z.iter().zip(expected_y0_v.iter()))
        {
            assert!((z - ez).abs() < 1.0e-7, "y0 zero {z} vs {ez}");
            assert!((v - ev).abs() < 1.0e-7, "y0 companion {v} vs {ev}");
        }

        let (y1_z, y1_v) = y1_zeros(3);
        let expected_y1_z = [
            2.197_141_326_031_017_f64,
            5.429_681_040_794_132,
            8.596_005_868_330_957,
        ];
        let expected_y1_v = [
            0.520_786_412_402_267_5_f64,
            -0.340_318_045_523_441_1,
            0.271_459_877_311_590_2,
        ];
        for ((z, v), (ez, ev)) in y1_z
            .iter()
            .zip(y1_v.iter())
            .zip(expected_y1_z.iter().zip(expected_y1_v.iter()))
        {
            assert!((z - ez).abs() < 1.0e-7, "y1 zero {z} vs {ez}");
            assert!((v - ev).abs() < 1.0e-7, "y1 companion {v} vs {ev}");
        }
    }

    #[test]
    fn derivative_bessel_zeros_match_scipy_reference_points() {
        let jnp0 = jnp_zeros(0, 3);
        let expected_jnp0 = [
            3.831_705_970_207_512_5_f64,
            7.015_586_669_815_619,
            10.173_468_135_062_722,
        ];
        for (z, expected) in jnp0.iter().zip(expected_jnp0.iter()) {
            assert!(
                (z - expected).abs() < 1.0e-7,
                "jnp_zeros(0) {z} vs {expected}"
            );
        }

        let jnp1 = jnp_zeros(1, 3);
        let expected_jnp1 = [
            1.841_183_781_340_659_5_f64,
            5.331_442_773_525_032_5,
            8.536_316_366_346_286,
        ];
        for (z, expected) in jnp1.iter().zip(expected_jnp1.iter()) {
            assert!(
                (z - expected).abs() < 1.0e-7,
                "jnp_zeros(1) {z} vs {expected}"
            );
        }

        let ynp0 = ynp_zeros(0, 3);
        let expected_ynp0 = [
            2.197_141_326_031_017_f64,
            5.429_681_040_794_135,
            8.596_005_868_331_169,
        ];
        for (z, expected) in ynp0.iter().zip(expected_ynp0.iter()) {
            assert!(
                (z - expected).abs() < 1.0e-7,
                "ynp_zeros(0) {z} vs {expected}"
            );
        }

        let ynp1 = ynp_zeros(1, 3);
        let expected_ynp1 = [
            3.683_022_856_585_177_7_f64,
            6.941_499_953_654_175_5,
            10.123_404_655_436_612,
        ];
        for (z, expected) in ynp1.iter().zip(expected_ynp1.iter()) {
            assert!(
                (z - expected).abs() < 1.0e-7,
                "ynp_zeros(1) {z} vs {expected}"
            );
        }

        let (y1p_z, y1p_v) = y1p_zeros(3);
        let expected_y1p_v = [
            0.416_729_928_106_451_26_f64,
            -0.303_173_740_137_498_3,
            0.250_912_536_277_893_56,
        ];
        for ((z, v), (expected_z, expected_v)) in y1p_z
            .iter()
            .zip(y1p_v.iter())
            .zip(expected_ynp1.iter().zip(expected_y1p_v.iter()))
        {
            assert!(
                (z - expected_z).abs() < 1.0e-7,
                "y1p zero {z} vs {expected_z}"
            );
            assert!(
                (v - expected_v).abs() < 1.0e-7,
                "y1p companion {v} vs {expected_v}"
            );
        }
    }

    #[test]
    fn yn_zeros_metamorphic_zero_value() {
        for &n in &[0_u32, 1, 2] {
            let zeros = yn_zeros(n, 4);
            for z in &zeros {
                let val = yn_scalar(n as f64, *z, RuntimeMode::Strict).expect("yn");
                assert!(val.abs() < 1e-5, "Y_{n}({z}) = {val} should be ≈ 0");
            }
        }
    }

    #[test]
    fn yn_zeros_metamorphic_strictly_increasing() {
        let zeros = yn_zeros(0, 8);
        for z in &zeros {
            assert!(*z > 0.0);
        }
        for w in zeros.windows(2) {
            assert!(w[0] < w[1]);
        }
    }

    #[test]
    fn jn_small_x_large_n_matches_series() {
        // [frankenscipy-ls0q1] regression: forward recurrence
        // catastrophically cancels when x ≪ 2n; the previous code
        // returned 9.7e9 for n=10, x=0.1 when the true value is
        // ~2.7e-22. Compare against the small-x Taylor series
        // J_n(x) ≈ (x/2)^n / n! · (1 − (x/2)²/(n+1) + ...).
        let x = 0.1_f64;
        for n in 5_u32..=10 {
            let got = super::jn_nonnegative(n, x);
            // Series: (x/2)^n / n! · (1 − (x/2)² / (n+1)).
            let mut fact = 1.0_f64;
            for k in 1..=n {
                fact *= k as f64;
            }
            let half_x = x / 2.0;
            let leading = half_x.powi(n as i32) / fact;
            let series_truth = leading * (1.0 - half_x * half_x / (n as f64 + 1.0));
            let rel = ((got - series_truth) / series_truth).abs();
            assert!(
                rel < 1e-2,
                "jn({n}, 0.1) = {got}, series truth ≈ {series_truth} (rel = {rel})"
            );
            assert!(got > 0.0, "jn({n}, 0.1) = {got} should be positive");
        }
    }

    #[test]
    fn jn_zeros_first_three_of_j0_match_known() {
        // First three zeros of J_0:
        //   2.404825557695773
        //   5.520078110286311
        //   8.653727912911012
        let zeros = jn_zeros(0, 3);
        let expected = [
            2.404_825_557_695_773_f64,
            5.520_078_110_286_311,
            8.653_727_912_911_012,
        ];
        assert_eq!(zeros.len(), 3);
        for (got, exp) in zeros.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-6, "j0 zero {got} vs {exp}");
        }
    }

    #[test]
    fn jn_zeros_first_two_of_j1_match_known() {
        // First two zeros of J_1:
        //   3.831705970207512
        //   7.015586669815619
        let zeros = jn_zeros(1, 2);
        let expected = [3.831_705_970_207_512_f64, 7.015_586_669_815_619];
        for (got, exp) in zeros.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-6, "j1 zero {got} vs {exp}");
        }
    }

    #[test]
    fn jn_zeros_metamorphic_zero_value() {
        // J_n(zero) ≈ 0 at every returned zero.
        for &n in &[0_u32, 1, 2, 3] {
            let zeros = jn_zeros(n, 5);
            for z in &zeros {
                let val = jn_scalar(n as f64, *z, RuntimeMode::Strict).expect("jn");
                assert!(val.abs() < 1e-6, "J_{n}({z}) = {val} should be ≈ 0");
            }
        }
    }

    #[test]
    fn jn_zeros_large_n_first_zero_matches_olver() {
        // Regression for [frankenscipy-zppm2]: McMahon's asymptotic guess
        // diverges from the true first zero for moderate-large n. Olver
        // expansion fixes it; verify J_n(jn_zeros(n,1)[0]) ≈ 0 across the
        // failure regime (n=10, 20, 50, 100).
        for &n in &[10_u32, 20, 50, 100] {
            let zeros = jn_zeros(n, 1);
            let z = zeros[0];
            let val = jn_scalar(n as f64, z, RuntimeMode::Strict).expect("jn");
            assert!(
                val.abs() < 1e-6,
                "J_{n}({z}) = {val} should be ≈ 0 (n={n}, k=1)"
            );
        }
    }

    #[test]
    fn yn_zeros_large_n_first_zero_matches_olver() {
        for &n in &[10_u32, 20, 50, 100] {
            let zeros = yn_zeros(n, 1);
            let z = zeros[0];
            let val = yn_scalar(n as f64, z, RuntimeMode::Strict).expect("yn");
            assert!(
                val.abs() < 1e-6,
                "Y_{n}({z}) = {val} should be ≈ 0 (n={n}, k=1)"
            );
        }
    }

    #[test]
    fn jn_zeros_large_order_match_scipy_first_five() {
        // SciPy-generated goldens for k>1 guard against wrong-root-index
        // regressions that still satisfy zero-value and ordering checks.
        let cases: &[(u32, [f64; 5])] = &[
            (
                10,
                [
                    14.475_500_686_554_541,
                    18.433_463_666_966_58,
                    22.046_985_364_697_803,
                    25.509_450_554_182_83,
                    28.887_375_063_530_5,
                ],
            ),
            (
                20,
                [
                    25.417_140_814_072_52,
                    29.961_603_789_935_07,
                    33.988_702_785_496_85,
                    37.772_857_844_688_884,
                    41.413_065_513_892_63,
                ],
            ),
            (
                50,
                [
                    57.116_899_160_119_17,
                    62.807_698_764_835_36,
                    67.697_408_410_764_31,
                    72.190_366_544_956_8,
                    76.437_072_182_667_59,
                ],
            ),
        ];

        for &(n, expected) in cases {
            let zeros = jn_zeros(n, expected.len());
            for (idx, (&got, &want)) in zeros.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - want).abs() < 1.0e-8,
                    "J_{n} zero {} got {got}, expected {want}",
                    idx + 1
                );
            }
        }
    }

    #[test]
    fn yn_zeros_large_order_match_scipy_first_five() {
        // SciPy-generated goldens for k>1 guard against wrong-root-index
        // regressions that still satisfy zero-value and ordering checks.
        let cases: &[(u32, [f64; 5])] = &[
            (
                10,
                [
                    12.128_927_704_415_54,
                    16.522_284_387_521_69,
                    20.265_984_504_165_354,
                    23.791_669_720_030_3,
                    27.206_568_880_356_8,
                ],
            ),
            (
                20,
                [
                    22.625_159_281_412_943,
                    27.788_445_040_035_606,
                    32.015_532_244_516_92,
                    35.903_160_004_884_45,
                    39.607_249_902_856_74,
                ],
            ),
            (
                50,
                [
                    53.502_858_820_400_37,
                    60.112_444_427_740_58,
                    65.317_141_148_304_92,
                    69.981_432_994_306_49,
                    74.338_747_169_198_36,
                ],
            ),
        ];

        for &(n, expected) in cases {
            let zeros = yn_zeros(n, expected.len());
            for (idx, (&got, &want)) in zeros.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - want).abs() < 1.0e-8,
                    "Y_{n} zero {} got {got}, expected {want}",
                    idx + 1
                );
            }
        }
    }

    #[test]
    fn bessel_zeros_metamorphic_interlace_relation() {
        // DLMF 10.21.2: zeros of J_n and Y_n strictly interlace —
        //   y_{n,1} < j_{n,1} < y_{n,2} < j_{n,2} < ...
        // Verify across multiple n. This is a stronger metamorphic
        // property than zero-value-is-zero: it pins the *count and
        // ordering* of zeros across two different functions.
        for &n in &[0_u32, 1, 2, 5, 10, 20] {
            let k = 5;
            let jzs = jn_zeros(n, k);
            let yzs = yn_zeros(n, k);
            for i in 0..k {
                assert!(
                    yzs[i] < jzs[i],
                    "interlace fail (n={n}, i={i}): y={} should be < j={}",
                    yzs[i],
                    jzs[i]
                );
                if i + 1 < k {
                    assert!(
                        jzs[i] < yzs[i + 1],
                        "interlace fail (n={n}, i={i}): j={} should be < y_next={}",
                        jzs[i],
                        yzs[i + 1]
                    );
                }
            }
        }
    }

    #[test]
    fn bessel_zeros_metamorphic_pi_separation_for_large_k() {
        // DLMF 10.21.20: consecutive zeros of J_n satisfy
        //   j_{n,k+1} - j_{n,k} → π as k → ∞.
        // Take k=20..30 across n=0..3 and check the separation lies
        // within [π − 0.05, π + 0.05].
        for &n in &[0_u32, 1, 2, 3] {
            let zs = jn_zeros(n, 30);
            for i in 19..29 {
                let dx = zs[i + 1] - zs[i];
                let pi = std::f64::consts::PI;
                assert!(
                    (dx - pi).abs() < 0.05,
                    "n={n}, k={}: separation {dx} should be ≈ π",
                    i + 1
                );
            }
        }
    }

    #[test]
    fn iv_matches_scipy_reference_points() {
        // /testing-conformance-harnesses: pin modified Bessel I_v at
        // canonical scipy values:
        //   I_0(0)   = 1
        //   I_v(0)   = 0   for v > 0
        //   I_0(1)   ≈ 1.266_065_877_752_009
        //   I_1(1)   ≈ 0.565_159_103_992_485
        //   I_0(2)   ≈ 2.279_585_302_336_067
        // I_n(z) symmetry: I_v(-z) = (-1)^v · I_v(z) for integer v.
        assert_eq!(super::iv_scalar(0.0, 0.0), 1.0);
        assert_eq!(super::iv_scalar(1.0, 0.0), 0.0);
        assert_eq!(super::iv_scalar(2.0, 0.0), 0.0);
        assert!((super::iv_scalar(0.0, 1.0) - 1.266_065_877_752_009).abs() < 1e-9);
        assert!((super::iv_scalar(1.0, 1.0) - 0.565_159_103_992_485).abs() < 1e-9);
        assert!((super::iv_scalar(0.0, 2.0) - 2.279_585_302_336_067).abs() < 1e-9);
        // Parity (integer v): I_0(-1) = I_0(1) (even), I_1(-1) = -I_1(1).
        assert!((super::iv_scalar(0.0, -1.0) - super::iv_scalar(0.0, 1.0)).abs() < 1e-12);
        assert!((super::iv_scalar(1.0, -1.0) + super::iv_scalar(1.0, 1.0)).abs() < 1e-12);
    }

    #[test]
    fn jn_zeros_metamorphic_strictly_increasing() {
        let zeros = jn_zeros(0, 10);
        for z in &zeros {
            assert!(*z > 0.0);
        }
        for w in zeros.windows(2) {
            assert!(w[0] < w[1], "zeros must be increasing: {} < {}", w[0], w[1]);
        }
    }

    #[test]
    fn k0_matches_scipy_reference() {
        // Values verified against scipy.special.k0
        let cases = [
            (0.1, 2.4270690247020164),
            (0.5, 0.9244190712276656),
            (1.0, 0.4210244382407082),
            (2.0, 0.1138938727495334),
            (5.0, 0.0036910983340426),
        ];
        for (x, expected) in cases {
            let val = super::k0_scalar(x);
            assert!(
                (val - expected).abs() < 1e-10,
                "k0({x}) = {val}, expected {expected}"
            );
        }
    }

    #[test]
    fn k1_matches_scipy_reference() {
        // Values verified against scipy.special.k1
        let cases = [
            (0.1, 9.853_844_780_870_606),
            (0.5, 1.6564411200033007),
            (1.0, 0.6019072301972346),
            (2.0, 0.1398658818165225),
            (5.0, 0.0040446134454522),
        ];
        for (x, expected) in cases {
            let val = super::k1_scalar(x);
            assert!(
                (val - expected).abs() < 1e-10,
                "k1({x}) = {val}, expected {expected}"
            );
        }
    }

    #[test]
    fn kn_matches_scipy_reference() {
        // Values verified against scipy.special.kn
        let cases = [
            (0, 1.0, 0.4210244382407083),
            (1, 1.0, 0.6019072301972346),
            (2, 1.0, 1.6248388986351774),
            (3, 1.0, 7.101_262_824_737_944),
            (2, 2.0, 0.2537597545660559),
        ];
        for (n, x, expected) in cases {
            let val = super::kn_scalar(n, x);
            assert!(
                (val - expected).abs() < 1e-10,
                "kn({n}, {x}) = {val}, expected {expected}"
            );
        }
    }

    #[test]
    fn k0_k1_kn_at_zero_is_infinity() {
        assert!(super::k0_scalar(0.0).is_infinite());
        assert!(super::k1_scalar(0.0).is_infinite());
        assert!(super::kn_scalar(0, 0.0).is_infinite());
        assert!(super::kn_scalar(2, 0.0).is_infinite());
    }

    #[test]
    fn k0_k1_kn_negative_x_is_nan() {
        // K functions are only defined for x > 0
        assert!(super::k0_scalar(-1.0).is_nan());
        assert!(super::k1_scalar(-1.0).is_nan());
        assert!(super::kn_scalar(0, -1.0).is_nan());
    }

    #[test]
    fn i0e_matches_scipy_reference() {
        // Values verified against scipy.special.i0e
        let cases = [
            (0.0, 1.0),
            (1.0, 0.4657596075936404),
            (2.0, 0.308_508_322_553_671),
            (5.0, 0.1835408126093283),
            (10.0, 0.1278333371634286),
        ];
        for (x, expected) in cases {
            let val = super::i0e_scalar(x);
            assert!(
                (val - expected).abs() < 1e-10,
                "i0e({x}) = {val}, expected {expected}"
            );
        }
    }

    #[test]
    fn i1e_matches_scipy_reference() {
        // Values verified against scipy.special.i1e
        let cases = [
            (0.0, 0.0),
            (1.0, 0.2079104153497085),
            (2.0, 0.2152692892489377),
            (5.0, 0.1639722669445423),
            (10.0, 0.1212626813844555),
        ];
        for (x, expected) in cases {
            let val = super::i1e_scalar(x);
            assert!(
                (val - expected).abs() < 1e-10,
                "i1e({x}) = {val}, expected {expected}"
            );
        }
    }

    #[test]
    fn k0e_matches_scipy_reference() {
        // Values verified against scipy.special.k0e
        let cases = [
            (0.5, 1.5241093857739092),
            (1.0, 1.1444630798068947),
            (2.0, 0.8415682150707712),
            (5.0, 0.547_807_564_313_519),
        ];
        for (x, expected) in cases {
            let val = super::k0e_scalar(x);
            assert!(
                (val - expected).abs() < 1e-9,
                "k0e({x}) = {val}, expected {expected}"
            );
        }
    }

    #[test]
    fn k1e_matches_scipy_reference() {
        // Values verified against scipy.special.k1e
        let cases = [
            (0.5, 2.7310097082117855),
            (1.0, 1.636_153_486_263_258),
            (2.0, 1.0334768470686888),
            (5.0, 0.6002738587883125),
        ];
        for (x, expected) in cases {
            let val = super::k1e_scalar(x);
            assert!(
                (val - expected).abs() < 1e-10,
                "k1e({x}) = {val}, expected {expected}"
            );
        }
    }

    #[test]
    fn ive_kve_definition_matches_scaling() {
        // ive(v, x) = iv(v, x) * exp(-|x|)
        // kve(v, x) = kv(v, x) * exp(x)
        let x = 2.0;
        let v = 1.5;
        let iv_val = super::iv_scalar(v, x);
        let ive_val = super::ive_scalar(v, x);
        assert!(
            (ive_val - iv_val * (-x).exp()).abs() < 1e-12,
            "ive definition mismatch"
        );

        let kv_val = super::kv_scalar(v, x, RuntimeMode::Strict).unwrap();
        let kve_val = super::kve_scalar(v, x);
        assert!(
            (kve_val - kv_val * x.exp()).abs() < 1e-10,
            "kve definition mismatch"
        );
    }

    #[test]
    fn scaled_bessel_derivatives_match_scipy_reference_values() {
        // Reference: d^n/dx^n of the exponentially-scaled Bessel functions, built in
        // SciPy 1.17.1 from the Leibniz rule over scipy.special.{ivp,kvp,jvp,yvp}:
        //   d^n/dx^n[B_v(x)·e^{σx}] = e^{σx}·Σ_j C(n,j)·σ^{n-j}·B_v^{(j)}(x)
        // (Jve/Yve use σ=0 on the real axis, so they coincide with jvp/yvp.)
        use super::{BesselKind, DerivativeRule};
        let cases = [
            (BesselKind::Ive, 2.0, 3.0, 0, 0.111_782_545_296_958_19),
            (BesselKind::Ive, 2.0, 3.0, 1, 0.010_522_471_135_703_922),
            (BesselKind::Ive, 2.0, 3.0, 2, -0.012_132_149_839_202_676),
            (BesselKind::Ive, 2.0, 3.0, 3, 0.009_946_205_192_563_019),
            // Half-integer order: the n>=2 derivative recurrence shifts the order to
            // v-2 = -1.5 (< -1), which previously returned NaN via the I_v power series
            // (frankenscipy-bt1kb). Now resolved through the I_{-p} reflection identity.
            (BesselKind::Ive, 0.5, 1.5, 0, 0.309_517_616_825_399_3),
            (BesselKind::Ive, 0.5, 1.5, 1, -0.070_737_756_722_038_79),
            (BesselKind::Ive, 0.5, 1.5, 2, 0.016_679_786_355_770_48),
            (BesselKind::Ive, 0.5, 1.5, 3, 0.055_089_243_968_660_56),
            (BesselKind::Kve, 2.0, 3.0, 0, 1.235_470_584_796_376_5),
            (BesselKind::Kve, 2.0, 3.0, 1, -0.394_739_951_863_328_16),
            (BesselKind::Kve, 2.0, 3.0, 2, 0.303_021_646_180_523_35),
            (BesselKind::Kve, 2.0, 3.0, 3, -0.349_183_748_124_313),
            (BesselKind::Kve, 1.5, 0.75, 1, -4.824_008_363_721_784),
            (BesselKind::Kve, 1.5, 0.75, 2, 14.793_625_648_746_8),
            (BesselKind::Kve, 1.5, 0.75, 3, -66.464_115_233_500_12),
            (BesselKind::Jve, 2.0, 3.0, 1, 0.014_998_118_135_342_325),
            (BesselKind::Jve, 2.0, 3.0, 2, -0.275_050_073_037_275_9),
            (BesselKind::Jve, 2.0, 3.0, 3, -0.059_009_512_776_879_72),
            (BesselKind::Yve, 2.0, 3.0, 1, 0.431_608_020_448_415_95),
            (BesselKind::Yve, 2.0, 3.0, 2, -0.054_758_010_435_625_39),
            (BesselKind::Yve, 2.0, 3.0, 3, -0.126_047_074_206_702_77),
        ];
        for (kind, v, x, n, expected) in cases {
            let val = super::bessel_derivative_real_scalar(
                "scaled_deriv",
                v,
                x,
                n,
                RuntimeMode::Strict,
                kind,
                DerivativeRule::Positive,
            )
            .expect("scaled Bessel derivative should evaluate");
            assert!(
                (val - expected).abs() <= 1e-9 * (1.0 + expected.abs()),
                "{kind:?} deriv n={n} at v={v}, x={x}: got {val}, expected {expected}"
            );
        }

        // Jve/Yve on the real axis must equal the unscaled jvp/yvp derivative exactly.
        for (scaled, base) in [
            (BesselKind::Jve, BesselKind::Jv),
            (BesselKind::Yve, BesselKind::Yv),
        ] {
            for n in 0..=3 {
                let s = super::bessel_derivative_real_scalar(
                    "scaled",
                    2.0,
                    3.0,
                    n,
                    RuntimeMode::Strict,
                    scaled,
                    DerivativeRule::Alternating,
                )
                .unwrap();
                let u = super::bessel_derivative_real_scalar(
                    "unscaled",
                    2.0,
                    3.0,
                    n,
                    RuntimeMode::Strict,
                    base,
                    DerivativeRule::Alternating,
                )
                .unwrap();
                assert_eq!(
                    s.to_bits(),
                    u.to_bits(),
                    "{scaled:?} must be bit-identical to {base:?} on the real axis (n={n})"
                );
            }
        }

        // Genuinely-unsupported domains stay fail-closed in Hardened mode (K_v requires x > 0).
        assert!(
            super::bessel_derivative_real_scalar(
                "kve",
                2.0,
                -1.0,
                1,
                RuntimeMode::Hardened,
                BesselKind::Kve,
                DerivativeRule::NegativeByOrder,
            )
            .is_err(),
            "kve derivative at x<0 must remain fail-closed"
        );
    }

    #[test]
    fn iv_negative_noninteger_order_matches_scipy_via_reflection() {
        // scipy.special.iv(v, z) for negative non-integer v <= -1, where the I_v power
        // series would pass ln(v+k+1) through a non-positive argument and yield NaN.
        // Reflection identity: I_{-p}(z) = I_p(z) + (2/π) sin(pπ) K_p(z), p = |v|.
        let cases = [
            (-0.5, 0.75, 1.192_814_667_397_816_8),
            (-0.5, 1.5, 1.532_524_329_376_575_3),
            (-0.5, 3.0, 4.637_757_757_861_504),
            (-1.5, 0.75, -0.832_804_570_140_508_8),
            (-1.5, 1.5, 0.365_478_834_152_426_9),
            (-1.5, 3.0, 3.068_903_650_787_1),
            (-2.5, 0.75, 4.524_032_947_959_852),
            (-2.5, 1.5, 0.801_566_661_071_721_9),
            (-2.5, 3.0, 1.568_854_107_074_402_9),
        ];
        for (v, z, expected) in cases {
            let val = super::iv_scalar(v, z);
            assert!(val.is_finite(), "iv({v}, {z}) must be finite, got {val}");
            assert!(
                (val - expected).abs() <= 1e-12 * (1.0 + expected.abs()),
                "iv({v}, {z}) = {val}, expected {expected}"
            );
        }
    }

    #[test]
    fn j1_matches_scipy_reference_values() {
        // scipy.special.j1([0.5, 1.0, 2.0, 5.0])
        let cases = [
            (0.5, 0.24226845767487388),
            (1.0, 0.4400505857449335),
            (2.0, 0.5767248077568734),
            (5.0, -0.32757913759146523),
        ];
        for (x, expected) in cases {
            let result = super::jn_scalar(1.0, x, RuntimeMode::Strict).unwrap();
            assert!(
                (result - expected).abs() < 1e-6,
                "j1({x}) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn y1_matches_scipy_reference_values() {
        // scipy.special.y1([0.5, 1.0, 2.0, 5.0])
        let cases = [
            (0.5, -1.471_472_392_670_243),
            (1.0, -0.7812128213002887),
            (2.0, -0.10703243154093755),
            (5.0, 0.14786314339122687),
        ];
        for (x, expected) in cases {
            let result = super::yn_scalar(1.0, x, RuntimeMode::Strict).unwrap();
            assert!(
                (result - expected).abs() < 1e-6,
                "y1({x}) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn jv_matches_scipy_reference_values() {
        // scipy.special.jv([1.5, 2.5], [1.0, 2.0])
        let cases = [
            (1.5, 1.0, 0.24029783912342725),
            (2.5, 2.0, 0.223_471_788_758_165_1),
        ];
        for (v, x, expected) in cases {
            let result = super::jv_scalar(v, x);
            assert!(
                (result - expected).abs() < 1e-3,
                "jv({v}, {x}) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn yv_matches_scipy_reference_values() {
        // scipy.special.yv([1.5], [1.0])
        let cases = [(1.5, 1.0, -1.1024850657061767)];
        for (v, x, expected) in cases {
            let result = super::yv_scalar(v, x, RuntimeMode::Strict).unwrap();
            assert!(
                (result - expected).abs() < 1e-3,
                "yv({v}, {x}) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn i0_matches_scipy_reference_values() {
        // scipy.special.i0([0.5, 1.0, 2.0, 5.0])
        let cases = [
            (0.5, 1.0634833707413234),
            (1.0, 1.2660658777520082),
            (2.0, 2.279585302336067),
            (5.0, 27.239871823604442),
        ];
        for (x, expected) in cases {
            let result = super::i0_scalar(x);
            assert!(
                (result - expected).abs() < 1e-10,
                "i0({x}) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn i1_matches_scipy_reference_values() {
        // scipy.special.i1([0.5, 1.0, 2.0, 5.0])
        let cases = [
            (0.5, 0.25789430539089634),
            (1.0, 0.5651591039924851),
            (2.0, 1.590_636_854_637_329),
            (5.0, 24.33564214245053),
        ];
        for (x, expected) in cases {
            let result = super::i1_scalar(x);
            assert!(
                (result - expected).abs() < 1e-10,
                "i1({x}) = {result}, expected {expected}"
            );
        }
    }
}
