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
    F: Fn(Complex64, Complex64) -> Complex64 + Sync,
{
    match (a, b) {
        (SpecialTensor::ComplexScalar(lhs), SpecialTensor::ComplexScalar(rhs)) => {
            Ok(SpecialTensor::ComplexScalar(kernel(*lhs, *rhs)))
        }
        (SpecialTensor::ComplexVec(lhs), SpecialTensor::ComplexScalar(rhs)) => {
            let rhs = *rhs;
            par_map_indices(lhs.len(), |i| Ok(kernel(lhs[i], rhs))).map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::ComplexScalar(lhs), SpecialTensor::ComplexVec(rhs)) => {
            let lhs = *lhs;
            par_map_indices(rhs.len(), |i| Ok(kernel(lhs, rhs[i]))).map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::ComplexVec(lhs), SpecialTensor::ComplexVec(rhs)) => {
            if lhs.len() != rhs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            par_map_indices(lhs.len(), |i| Ok(kernel(lhs[i], rhs[i])))
                .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::RealScalar(lhs), SpecialTensor::ComplexScalar(rhs)) => Ok(
            SpecialTensor::ComplexScalar(kernel(Complex64::from_real(*lhs), *rhs)),
        ),
        (SpecialTensor::ComplexScalar(lhs), SpecialTensor::RealScalar(rhs)) => Ok(
            SpecialTensor::ComplexScalar(kernel(*lhs, Complex64::from_real(*rhs))),
        ),
        (SpecialTensor::RealVec(lhs), SpecialTensor::ComplexScalar(rhs)) => {
            let rhs = *rhs;
            par_map_indices(lhs.len(), |i| Ok(kernel(Complex64::from_real(lhs[i]), rhs)))
                .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::ComplexScalar(lhs), SpecialTensor::RealVec(rhs)) => {
            let lhs = *lhs;
            par_map_indices(rhs.len(), |i| Ok(kernel(lhs, Complex64::from_real(rhs[i]))))
                .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::RealScalar(lhs), SpecialTensor::ComplexVec(rhs)) => {
            let lhs = Complex64::from_real(*lhs);
            par_map_indices(rhs.len(), |i| Ok(kernel(lhs, rhs[i]))).map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::ComplexVec(lhs), SpecialTensor::RealScalar(rhs)) => {
            let rhs = Complex64::from_real(*rhs);
            par_map_indices(lhs.len(), |i| Ok(kernel(lhs[i], rhs))).map(SpecialTensor::ComplexVec)
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
            par_map_indices(lhs.len(), |i| {
                Ok(kernel(Complex64::from_real(lhs[i]), rhs[i]))
            })
            .map(SpecialTensor::ComplexVec)
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
            par_map_indices(lhs.len(), |i| {
                Ok(kernel(lhs[i], Complex64::from_real(rhs[i])))
            })
            .map(SpecialTensor::ComplexVec)
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
    // bratio's complement 1 − I_x(a, b), computed directly (frankenscipy-5pnba).
    betaincc_with_complement(a, b, x, 1.0 - x)
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

    // btdtr(a, b, x) == betainc(a, b, x), so the inverse is exactly betaincinv.
    // The dedicated inverse carries a small-x asymptotic seed
    // (x ~ (y·a·B(a,b))^{1/a}) + relative-tolerance Newton, which resolves the
    // deep a<1 tail (e.g. y=1e-8) instead of stalling at the ~1e-16 floor that
    // the generic bisection bottomed out on. frankenscipy-8urrz.
    crate::convenience::betaincinv_scalar(a, b, y)
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
    // 1 − z in closed form, dfd/(dfn·x + dfd), so a large x keeps the digits of the upper
    // tail (frankenscipy-5pnba). NaN arguments (x = inf gives inf/inf) propagate.
    let denom = dfn * x + dfd;
    let z = dfn * x / denom;
    let z1 = dfd / denom;
    if z.is_nan() || z1.is_nan() {
        return f64::NAN;
    }
    betainc_with_complement(0.5 * dfn, 0.5 * dfd, z, z1).unwrap_or(f64::NAN)
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
    // fdtrc(dfn, dfd, x) = 1 − I_z(dfn/2, dfd/2): bratio's complement, with 1 − z in closed
    // form (frankenscipy-5pnba).
    let denom = dfn * x + dfd;
    let z = dfn * x / denom;
    let z1 = dfd / denom;
    if z.is_nan() || z1.is_nan() {
        return f64::NAN;
    }
    betaincc_with_complement(0.5 * dfn, 0.5 * dfd, z, z1)
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

    // fdtr is increasing in dfd. Bracket the root two-sidedly from dfd = 1, then
    // solve with the superlinear Illinois method (~10 evals) instead of the former
    // 240-iteration bisection (a ~21× SciPy loss — each eval is a full fdtr).
    let f1 = fdtr(dfn, 1.0, x) - p;
    let (mut lo, mut flo) = (1.0_f64, f1);
    let (mut hi, mut fhi) = (1.0_f64, f1);
    while fhi < 0.0 {
        hi *= 2.0;
        if hi >= UPPER_SENTINEL {
            return UPPER_SENTINEL;
        }
        fhi = fdtr(dfn, hi, x) - p;
    }
    while flo > 0.0 {
        lo *= 0.5;
        if lo <= LOWER_SENTINEL {
            return LOWER_SENTINEL;
        }
        flo = fdtr(dfn, lo, x) - p;
    }
    // A bracket endpoint can land exactly on the root (illinois excludes endpoints).
    if flo == 0.0 {
        return lo;
    }
    if fhi == 0.0 {
        return hi;
    }

    let dfd = illinois_root(|m| fdtr(dfn, m, x) - p, lo, hi, flo, fhi);
    if dfd <= 0.0 { LOWER_SENTINEL } else { dfd }
}

/// 2^53: the first Poisson mode `j₀ = ⌊λ⌋` from which the noncentral walks cannot step.
///
/// [`ncfdtr`], [`ncfdtrc`], [`nctdtr`], [`crate::gamma::chndtr`] and
/// [`crate::gamma::chndtrc`] sum their Poisson mixture outward from `j₀`, moving the index by
/// `j ± 1.0` and stopping on a relative weight test. From 2^53 up, f64 integers are no longer
/// all representable: `j₀ + 1.0 == j₀` at `j₀ = 2^53`, and above it `j₀ − 1.0` can round back
/// to `j₀` (it does for `λ = 0.5·1.35e8²` and for `λ = 2^59`). Then `j` and the weight stop
/// moving and the downward loop never ends. That was the frankenscipy-qu5po hang. The other
/// trigger, `nc = inf` (`j₀ = ∞`), is answered before the walk with SciPy's value.
///
/// Each walk now has two exits (frankenscipy-qu5po):
///
/// 1. **`j₀ ≥ 2^53` → NaN, before any anchor is formed.** The mixture cannot be summed in
///    f64 here. SciPy 1.17.1 (Boost) is nan at the centre too: `chndtr(2^60, 3, 2^60)`,
///    `ncfdtr(3, 5, 2^60, 2^60/3)` and `nctdtr(5, 1.35e8, 1.35e8)` are all nan. In the far
///    tails SciPy instead answers 0 or 1, and this exit does not follow it, on purpose. The
///    mode anchors were then formed in log space as differences of terms of size `j₀·ln j₀`
///    (saddle-point form since frankenscipy-g9yid), so their exponent carried an absolute
///    error of that size times ε. The probe measured chndtr's
///    `t0` exponent in f64 (Python `math.lgamma` standing in for `gammaln`) against mpmath.
///    At `nc = 2^60` the error reached 4075. In 168 of 401 points within 10 sd of the mean
///    the f64 anchor underflowed to 0 while the exact one is about e^-21. A zero anchor
///    here therefore proves nothing, and returning a tail value from it would put a silent
///    0 at the centre.
/// 2. **Below 2^53, both mode anchors exactly zero → return the value of an all-zero
///    series.** The anchors are the mixture term at the mode and its recurrence increment.
///    When both are zero, the recurrence keeps every term at `w·0`, so the walk can only add
///    zeros. This exit is the walk's own result without the walk. Without it, `total` stayed
///    at 0 and the relative stop test degenerated to `w < 1e-317`, so the walk crossed the
///    whole Poisson left tail, about `38·√λ` steps. For `nctdtr(5, 1.3e8, 2)`, where
///    `λ = 8.45e15`, that is about 3.5e9 steps. SciPy 1.17.1 gives the same values:
///    `nctdtr(5, 1.3e8, 2) = 0`, `chndtr(5, 3, 1e16) = 0`, `ncfdtr(3, 5, 1e16, 2) = 0`.
///
/// The limit is a property of f64 index arithmetic, not a tolerance, so it is the same for
/// every family.
pub(crate) const POISSON_INDEX_LIMIT: f64 = 9_007_199_254_740_992.0;

/// Step cap for the upward half of the noncentral Poisson walks: `max(100_000, ⌈40·√λ⌉)`.
///
/// [`ncfdtr`], [`ncfdtrc`], [`nctdtr`], [`crate::gamma::chndtr`] and
/// [`crate::gamma::chndtrc`] walk down from `j₀` until `j` reaches 0 or the relative weight
/// test fires, so that half bounds itself. The upward half has no floor, so it also carries a
/// step cap. That cap used to be a fixed 100,000. The upward walk needs about `7–9·√λ` steps
/// before its relative test fires, so from about `λ = 1.2e8` the fixed cap cut the sum off
/// with mass still unsummed (frankenscipy-g9yid). A float transliteration of each walk was
/// measured at `x` = mean and mean ± 3 sd against SciPy 1.17.1. Where SciPy returns NaN
/// (ncfdtr and nctdtr at λ = 1e10), it was measured against a per-term sum that has no
/// recurrence and no cap:
///
/// ```text
/// λ = nc/2 (χ², F) or nc²/2 (t)     capped?   worst relative error of the capped walk
/// 1e6, 1e7, 1e8                    no        none (the cap was not reached)
/// 2e8                              yes       ≤ 1e-9; the dropped terms were below rounding
/// 1e9                              yes       6e-4 (chndtr), 0.13 (chndtrc), 8e-4 (F, t)
/// 1e10                             yes       0.16 (chndtr), 0.97 (chndtrc), 0.16 (F, t)
/// ```
///
/// For example, SciPy gives `chndtr(2.00008e10, 3, 2e10) = 0.99766`, and the capped walk gave
/// 0.84124. With this cap the walk ends on its own stop tests at every measured point: the
/// relative test fires after about `9·√λ` steps near the mean and up to `17·√λ` steps 10 sd
/// into the left tail. The absolute test `w < 1e-300` fires by about `37·√λ` steps, since
/// `w/w₀ ≈ exp(−k²/2λ)` and `w₀ ≈ 1/√(2πλ)`. So the cap is never the exit that ends a finite
/// walk. It only bounds a walk whose stop tests cannot fire, such as one with NaN weights, and
/// it stays finite because `j₀ < POISSON_INDEX_LIMIT` is checked before any walk starts.
/// Below `λ = 6.25e6` the cap is the old 100,000, so small-`λ` results are unchanged.
pub(crate) fn poisson_upward_step_cap(lam: f64) -> f64 {
    (40.0 * lam.sqrt()).ceil().max(100_000.0)
}

/// `xᵃ·yᵇ·Γ(a+b) / (Γ(a+1)·Γ(b))` for `a, b > 0` and `y = 1 − x` passed in exactly: the
/// incomplete beta increment `I_x(a,b) − I_x(a+1,b)`, and `1/a` times the front factor of
/// [`betainc_scalar`] (frankenscipy-g9yid).
///
/// From `a + b = SADDLE_POINT_MIN_SHAPE` it is `b/(a+b)` times the binomial term
/// `C(a+b, a)·xᵃyᵇ` in Loader's saddle-point form (R's `dbinom_raw` with `n = a + b`):
/// `exp(stirlerr(a+b) − stirlerr(a) − stirlerr(b) − bd0(a, (a+b)x) − bd0(b, (a+b)y))`
/// `· √(b / (2π·a·(a+b)))`. The log-space form it replaces,
/// `exp(a·ln x + b·ln y + lnΓ(a+b) − lnΓ(a+1) − lnΓ(b))`, loses `ε·a·ln a` to cancellation.
/// Worst relative error against mpmath over the ncfdtr, ncfdtrc and nctdtr mode anchors at
/// x = mean and mean ± 3 sd (dfn = 3, dfd = 50; df = 5):
///
/// ```text
/// λ        log-space   saddle point
/// 1e3      1.2e-12     4.4e-15
/// 1e6      8.0e-10     3.6e-15
/// 1e8      2.4e-7      3.6e-15
/// 1e10     4.2e-5      6.0e-15
/// 1e12     3.8e-3      4.3e-15
/// ```
///
/// The saddle-point form also tolerates an `x` and `y` that miss `x + y = 1` by a rounding:
/// a relative error `δ` in `x` moves it by `((a+b)·x − a)·δ`, which is O(δ) near the mode,
/// where the log-space form's `xᵃ` moves by `a·δ`. That matters for ncfdtrc, whose `x` and
/// `y` are rounded separately.
pub(crate) fn beta_term(a: f64, b: f64, x: f64, y: f64) -> f64 {
    if a.is_nan() || b.is_nan() || x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    let n = a + b;
    if n < gamma::SADDLE_POINT_MIN_SHAPE {
        let lg = |z: f64| gammaln_scalar(z, RuntimeMode::Strict).unwrap_or(f64::NAN);
        return (a * x.ln() + b * y.ln() + lg(n) - lg(a + 1.0) - lg(b)).exp();
    }
    if a.is_infinite() || b.is_infinite() || n.is_infinite() {
        return f64::NAN;
    }
    if x == 0.0 || y == 0.0 {
        return 0.0;
    }
    let lc = gamma::stirlerr(n)
        - gamma::stirlerr(a)
        - gamma::stirlerr(b)
        - gamma::bd0(a, n * x)
        - gamma::bd0(b, n * y);
    lc.exp() * (b / n / (std::f64::consts::TAU * a)).sqrt()
}

/// Non-central F cumulative distribution function.
///
/// Matches `scipy.special.ncfdtr(dfn, dfd, nc, f)`: the CDF at `f` of a
/// non-central F variable with `dfn`/`dfd` degrees of freedom and
/// non-centrality `nc`.
///
/// Computed as the Poisson(nc/2)-weighted mixture of central regularized
/// incomplete beta values
///
/// ```text
///   ncfdtr = Σ_{j≥0} e^{−λ} λ^j/j! · I_y(dfn/2 + j, dfd/2),
///   λ = nc/2,  y = dfn·f / (dfn·f + dfd)
/// ```
///
/// The sum is accumulated outward from the Poisson mode `j₀ = ⌊λ⌋` (mode weight
/// from `gamma::poisson_term`, saddle-point form from λ = 100) so large `nc`
/// neither underflows `e^{−λ}` nor loses precision.
#[must_use]
pub fn ncfdtr(dfn: f64, dfd: f64, nc: f64, f: f64) -> f64 {
    if dfn.is_nan() || dfd.is_nan() || nc.is_nan() || f.is_nan() {
        return f64::NAN;
    }
    if dfn <= 0.0 || dfd <= 0.0 || nc < 0.0 {
        return f64::NAN;
    }
    if nc == f64::INFINITY {
        // λ = ∞ has no Poisson mode to walk from: j₀ = ∞ and `j -= 1.0` never reaches 0, so
        // the downward loop below never ended (frankenscipy-qu5po). SciPy 1.17.1 answers
        // 1.0 at f = inf and NaN at every other f, f = 0 included:
        // ncfdtr(3, 5, inf, inf) = 1.0; ncfdtr(3, 5, inf, f) = nan for f = -inf, -2, 0, 2, 1e300.
        return if f == f64::INFINITY { 1.0 } else { f64::NAN };
    }
    // A negative f is outside the support, and SciPy 1.17.1 answers nan there, not 0
    // (frankenscipy-g9yid): ncfdtr(3, 5, 3, f) and ncfdtr(3, 5, 0, f) are nan for
    // f = -1e-300, -2 and -inf, while f = ±0 gives 0.0. ncfdtri, ncfdtrinc and the dfd/dfn
    // inverses already reject f < 0.
    if f < 0.0 {
        return f64::NAN;
    }
    if f == 0.0 {
        return 0.0;
    }
    // f = inf, or a finite f with dfn·f + dfd overflowing: the whole mass is below f, but
    // y = dfn·f / (dfn·f + dfd) was inf/inf = NaN (frankenscipy-g9yid). SciPy 1.17.1 gives
    // 1.0 for ncfdtr(3, 5, 3, inf), ncfdtr(3, 5, 3, 1e308), ncfdtr(3, 5, 0, 1e308) and
    // ncfdtr(3, inf, 3, inf). With an infinite dfn or dfd and a finite f it gives nan
    // (ncfdtr(inf, 5, 3, 2)), so the overflow case applies only to finite degrees of freedom.
    let denom = dfn * f + dfd;
    if f == f64::INFINITY || (denom == f64::INFINITY && dfn.is_finite() && dfd.is_finite()) {
        return 1.0;
    }
    let y = dfn * f / denom;
    // 1 − y in closed form, as in `ncfdtrc` (frankenscipy-g9yid). `1.0 - y` carries the
    // rounding of y as a relative error of up to ε·dfn·f/(2·dfd) in 1 − y, 4.6e-6 near the
    // mean at λ = 1e12, and ncfdtr there missed by 3.5e-6 with saddle-point anchors. With y1
    // it misses by 1.3e-9.
    let y1 = dfd / denom;
    if nc == 0.0 {
        // The central law takes the same closed-form complement (frankenscipy-5pnba): at
        // dfd = 1e-100, y rounds to 1 and `btdtr(…, y)` said 1.0, where the law is ~1e-98,
        // continuous with nc > 0 (SciPy's ncfdtr(3, 1e-100, 1, 1.5) = 1.15e-98). SciPy's own
        // cdflib `cumf` passes both, which is how its ncfdtridfd(3, ·, 0, 1.5) finds dfd = 20.
        return betainc_with_complement(0.5 * dfn, 0.5 * dfd, y, y1).unwrap_or(f64::NAN);
    }
    let lam = nc / 2.0;
    let j0 = lam.floor();
    // frankenscipy-qu5po: the walk cannot step from here (see `POISSON_INDEX_LIMIT`).
    if j0 >= POISSON_INDEX_LIMIT {
        return f64::NAN;
    }
    // Saddle-point anchors (frankenscipy-g9yid): the Poisson weight here, the incomplete beta
    // and its increment below. In log space each lost about λ·ln λ·ε (2.5e-7 at λ = 1e8,
    // 4e-3 at 1e12); see `gamma::poisson_term` and `beta_term`.
    let w0 = gamma::poisson_term(j0, lam);

    // Each Poisson term needs btdtr(0.5·dfn + j, 0.5·dfd, y) = I_y(a, b), the
    // regularized incomplete beta with a = 0.5·dfn + j, b = 0.5·dfd. Computing it
    // fresh per term is one incomplete-beta each — O(√nc) evals. Instead anchor at
    // the mode and walk the incomplete-beta first-parameter recurrence O(1)/term:
    //   I_y(a+1, b) = I_y(a, b) − u(a),  u(a) = y^a (1−y)^b · Γ(a+b)/(Γ(a+1)Γ(b)),
    //   u(a+1) = u(a)·y·(a+b)/(a+1)         (upward, a increasing);
    //   I_y(a−1, b) = I_y(a, b) + u(a−1),   u(a−1) = u(a)·a/(y·(a−1+b))  (downward).
    // One incomplete-beta at the mode + O(√nc) multiply/adds. The downward branch
    // (a decreasing, I growing toward 1 by ADDING positive u) is the stable
    // direction and carries the dominant mass; upward only adds small above-mode
    // corrections (term clamped to [0,1]). Verified vs scipy.special.ncfdtr to
    // ≤4.3e-13 rel across nc up to 4000 and tails to 1e-144.
    let b = 0.5 * dfd;
    let a0 = 0.5 * dfn + j0;
    let p0 = betainc_with_complement(a0, b, y, y1).unwrap_or(f64::NAN); // = I_y(a0, b)
    let u0 = beta_term(a0, b, y, y1); // = y^a0 (1−y)^b Γ(a0+b) / (Γ(a0+1) Γ(b))
    // frankenscipy-qu5po: the all-zero exit (see `POISSON_INDEX_LIMIT`).
    if p0 == 0.0 && u0 == 0.0 {
        return 0.0;
    }

    let mut total = 0.0_f64;
    // Upward from the mode. The cap scales with √λ (frankenscipy-g9yid; see
    // `poisson_upward_step_cap`).
    let mut w = w0;
    let mut j = j0;
    let mut a = a0;
    let mut p = p0;
    let mut u = u0;
    let cap = poisson_upward_step_cap(lam);
    let mut steps = 0.0_f64;
    while steps < cap {
        total += w * p.clamp(0.0, 1.0);
        p -= u;
        if p <= 0.0 {
            break;
        }
        j += 1.0;
        u *= y * (a + b) / (a + 1.0);
        a += 1.0;
        w *= lam / j;
        if w < 1e-300 || (w < 1e-14 * total.max(1e-300) && j > lam) {
            break;
        }
        steps += 1.0;
    }
    // Downward from the mode.
    w = w0;
    j = j0;
    a = a0;
    p = p0;
    u = u0;
    while j > 0.0 {
        w *= j / lam;
        u *= a / (y * (a - 1.0 + b));
        p += u;
        j -= 1.0;
        a -= 1.0;
        total += w * p.clamp(0.0, 1.0);
        if w < 1e-14 * total.max(1e-300) {
            break;
        }
    }
    total.clamp(0.0, 1.0)
}

/// Non-central F **survival** function: `P(X > f)`.
///
/// The complement of [`ncfdtr`], computed directly rather than as
/// `1 - ncfdtr(...)`. Matches `scipy.stats.ncf.sf(f, dfn, dfd, nc)`.
///
/// # Why this is a separate kernel and not `1.0 - ncfdtr(...)`
///
/// Once the CDF rounds to 1.0 the subtraction returns exactly zero, and it has
/// already lost most of its digits well before that. Measured against
/// `scipy.stats.ncf.sf`:
///
/// ```text
///   dfn dfd  nc  f       true sf              1 - ncfdtr
///   3   5    2   1e6     2.3303386e-14        2.3314684e-14   (3 digits left)
///   3   5    2   1e12    2.3303549e-29        0.0
///   5   200  3   1e4     4.4981732e-230       0.0
/// ```
///
/// The F tail is polynomial, so the collapse arrives later than it does for the
/// noncentral χ² (see [`crate::gamma::chndtrc`]) — but when it arrives it is just
/// as total, and the erosion before it is silent.
///
/// # Method
///
/// The same Poisson(nc/2) mixture as [`ncfdtr`], carrying the COMPLEMENTARY
/// regularized incomplete beta
///
/// ```text
///   ncfdtrc = Σ_{j≥0} e^{−λ} λ^j/j! · Q_y(dfn/2 + j, dfd/2),
///   Q_y(a, b) = 1 − I_y(a, b) = I_{1−y}(b, a),
///   λ = nc/2,  y = dfn·f / (dfn·f + dfd)
/// ```
///
/// Two things here are load-bearing, and both are places where the obvious
/// transcription of [`ncfdtr`] would silently reintroduce the very cancellation
/// this function exists to avoid:
///
/// 1. **The stable direction REVERSES.** [`ncfdtr`] walks the incomplete-beta
///    first-parameter recurrence and notes that the DOWNWARD branch is stable,
///    because `I` grows toward 1 by ADDING positive `u`. For the complement the
///    signs flip — `Q(a+1) = Q(a) + u(a)` upward, `Q(a−1) = Q(a) − u(a−1)`
///    downward — so it is the UPWARD branch that adds and is stable. That is also
///    the branch carrying the dominant mass here: `Q` increases with `a`, hence
///    with `j`. The downward branch subtracts; its terms are small in BOTH factors
///    (small Poisson weight and small `Q`), so its cancellation is subdominant, and
///    when it drives `q` to zero we stop rather than propagate a negative. The
///    resulting error is bounded by the discarded below-mode mass instead of
///    contaminating the total.
///
/// 2. **`1 − y` is formed exactly, never by subtraction.** `1 − y = dfd/(dfn·f + dfd)`
///    in closed form. For a right-tail query `y → 1`, so evaluating `1.0 - y` would
///    throw away exactly the digits the answer is made of. That exact `y1` feeds both
///    the anchor and the `(1−y)^b` factor of `u` (as `b·ln(y1)`, where [`ncfdtr`] can
///    afford `ln_1p(−y)`).
///
/// The anchor `Q_y(a₀, b)` is the incomplete beta kernel's own complement `1 − I_y(a₀, b)`,
/// which it computes directly and which underflows gracefully, so no probability is ever
/// obtained by subtracting from one.
///
/// Boundaries: `f ≤ 0 → 1`, `nc = 0` degenerates to the central `I_{1−y}(dfd/2, dfn/2)`,
/// non-positive `dfn`/`dfd` or negative `nc` → NaN.
#[must_use]
pub fn ncfdtrc(dfn: f64, dfd: f64, nc: f64, f: f64) -> f64 {
    if dfn.is_nan() || dfd.is_nan() || nc.is_nan() || f.is_nan() {
        return f64::NAN;
    }
    if dfn <= 0.0 || dfd <= 0.0 || nc < 0.0 {
        return f64::NAN;
    }
    if f <= 0.0 {
        return 1.0;
    }
    let denom = dfn * f + dfd;
    if !denom.is_finite() {
        // dfn·f overflowed; the whole mass is below f.
        return 0.0;
    }
    if nc == f64::INFINITY {
        // No Poisson mode to walk from (frankenscipy-qu5po; see `POISSON_INDEX_LIMIT`).
        // scipy.stats.ncf.sf(f, 3, 5, inf) in 1.17.1 is nan at f = 2 and f = 1e300; its
        // f = 0 → 1.0 and f = inf → 0.0 are the two returns above.
        return f64::NAN;
    }
    let y = dfn * f / denom;
    // 1 − y in closed form. NOT `1.0 - y`: for a tail query y → 1 and the
    // subtraction would discard the digits this function is built to keep.
    let y1 = dfd / denom;
    let b = 0.5 * dfd;
    if nc == 0.0 {
        // Central F survival: Q_y(dfn/2, dfd/2), bratio's complement at the exact y and y1.
        return betaincc_with_complement(0.5 * dfn, b, y, y1).clamp(0.0, 1.0);
    }
    let lam = nc / 2.0;
    let j0 = lam.floor();
    // frankenscipy-qu5po: the walk cannot step from here (see `POISSON_INDEX_LIMIT`).
    if j0 >= POISSON_INDEX_LIMIT {
        return f64::NAN;
    }
    // Saddle-point anchors, as in `ncfdtr` (frankenscipy-g9yid).
    let w0 = gamma::poisson_term(j0, lam);

    let a0 = 0.5 * dfn + j0;
    // Q_y(a0, b) = 1 − I_y(a0, b), computed DIRECTLY as bratio's complement (at the exact y and
    // y1, frankenscipy-5pnba) — never by subtracting from 1.
    let q0 = betaincc_with_complement(a0, b, y, y1);
    // u(a) = y^a (1−y)^b · Γ(a+b)/(Γ(a+1)Γ(b)), the recurrence increment shared
    // with `ncfdtr`; (1−y)^b taken from y1 so it stays exact for y → 1.
    let u0 = beta_term(a0, b, y, y1);
    // frankenscipy-qu5po: the all-zero exit (see `POISSON_INDEX_LIMIT`).
    if q0 == 0.0 && u0 == 0.0 {
        return 0.0;
    }

    let mut total = 0.0_f64;
    // Upward from the Poisson mode: Q(a+1) = Q(a) + u(a). THE STABLE DIRECTION
    // for the complement, and the one carrying the dominant mass.
    let mut w = w0;
    let mut j = j0;
    let mut a = a0;
    let mut q = q0;
    let mut u = u0;
    // The cap scales with √λ (frankenscipy-g9yid; see `poisson_upward_step_cap`).
    let cap = poisson_upward_step_cap(lam);
    let mut steps = 0.0_f64;
    while steps < cap {
        total += w * q.clamp(0.0, 1.0);
        q += u;
        if q > 1.0 {
            // Saturated: every remaining above-mode term contributes its full
            // Poisson weight. Keep summing (the weight test below terminates) —
            // breaking here would DISCARD the residual tail mass rather than the
            // negligible quantity `ncfdtr`'s mirrored `p <= 0` exit discards.
            q = 1.0;
        }
        j += 1.0;
        u *= y * (a + b) / (a + 1.0);
        a += 1.0;
        w *= lam / j;
        if w < 1e-300 || (w < 1e-14 * total.max(1e-300) && j > lam) {
            break;
        }
        steps += 1.0;
    }
    // Downward from the mode: Q(a−1) = Q(a) − u(a−1). Subtractive, hence the
    // cancellation-prone branch — but small in both factors, and stopped at zero.
    w = w0;
    j = j0;
    a = a0;
    q = q0;
    u = u0;
    while j > 0.0 {
        w *= j / lam;
        u *= a / (y * (a - 1.0 + b));
        q -= u;
        if q <= 0.0 {
            // Cancellation floor. Remaining below-mode terms are smaller still;
            // stopping bounds the error by the discarded mass instead of
            // propagating a negative Q into the sum.
            break;
        }
        j -= 1.0;
        a -= 1.0;
        total += w * q.clamp(0.0, 1.0);
        if w < 1e-14 * total.max(1e-300) {
            break;
        }
    }
    total.clamp(0.0, 1.0)
}

/// Inverse of [`ncfdtr`] in the argument `f`.
///
/// Returns `f` such that `ncfdtr(dfn, dfd, nc, f) = p`, matching
/// `scipy.special.ncfdtri(dfn, dfd, nc, p)`. `p = 0 → 0`, `p = 1 → +∞`, `p ∉ [0, 1]` → NaN.
///
/// SciPy 1.17.1 answers with Boost's quantile, which brackets the root by a geometric walk
/// from a guess and never searches a linear range. This does the same with
/// `bracket_and_solve_root` from `f = 1` (frankenscipy-g9yid), so a root many decades
/// below 1 is bracketed in O(log) CDF calls and then resolved to a relative tolerance:
/// SciPy gives `ncfdtri(3, 5, 2, 1e-20) = 6.67000004655459e-14`, where the old `[0, hi]`
/// search stopped at its first false-position step, 5.92e-20. For `p ≥ 1/2` the residual is
/// `q − ncfdtrc` with `q = 1 − p`, Boost's complement form. The answer is only as good as
/// [`ncfdtr`] at the root: where both of its mode anchors underflow it returns 0, so a `p` far
/// below `1e-300` (SciPy: `ncfdtri(3, 5, 2, 1e-300) = 1.4370079482811187e-200`) is not
/// resolved there.
///
/// SciPy is NaN from `nc ≈ 1.0293e10` for every `p`, `dfn` and `dfd` measured (Boost's series
/// hits its term limit), while [`ncfdtr`] deliberately stays finite there, and this inverse
/// stays consistent with it and answers. The owner may revisit that choice.
#[must_use]
pub fn ncfdtri(dfn: f64, dfd: f64, nc: f64, p: f64) -> f64 {
    if dfn.is_nan() || dfd.is_nan() || nc.is_nan() || p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    // nc = inf: SciPy 1.17.1 returns nan for every p, the p = 0 and p = 1 edges included
    // (frankenscipy-qu5po).
    if dfn <= 0.0 || dfd <= 0.0 || nc < 0.0 || nc == f64::INFINITY {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    // A NaN CDF (the `POISSON_INDEX_LIMIT` exit) makes the walk return NaN rather than a
    // bracket. SciPy 1.17.1: ncfdtri(3, 5, 2^60, 0.5) = nan.
    let q = 1.0 - p;
    if p < q {
        bracket_and_solve_root(|x| ncfdtr(dfn, dfd, nc, x) - p, 1.0, true)
    } else {
        bracket_and_solve_root(|x| q - ncfdtrc(dfn, dfd, nc, x), 1.0, true)
    }
}

/// Inverse of [`ncfdtr`] in the non-centrality `nc`.
///
/// Returns `nc ≥ 0` such that `ncfdtr(dfn, dfd, nc, f) = p`, matching
/// `scipy.special.ncfdtrinc(dfn, dfd, p, f)`.
///
/// SciPy 1.17.1 answers with cdflib's `cdffnc_which5` (scipy/special/cdflib.c 1855-1906),
/// which searches `nc ∈ [0, 1e4]` from `nc = 5` and returns a bound when the root is outside:
/// `0` when `p` is above the central (`nc = 0`) CDF, and `1e4` when the CDF at `nc = 1e4` is
/// still above `p`. `cdflib_invert` reproduces that search on fsci's own [`ncfdtr`]
/// (frankenscipy-g9yid). SciPy gives `ncfdtrinc(3, 5, 0.5, f) = 1e4` for `f = 1e4`, `1e6`,
/// `1e300` and `inf`, where the old unbounded doubling returned 2.6e6, NaN after about 2.4e9
/// Poisson-walk steps, and `+inf`. `p` must lie in `[0, 1 − 1e-16]`, so `p = 1` is NaN.
///
/// Inside `(0, 1e4)` the root is fsci's: cdflib sums its CDF only to a relative 1e-4
/// (`cumfnc`, cdflib.c 2952), so SciPy's interior values differ from the exact root by up
/// to about 1e-5 relative (1e-3 at `f ≈ 1e3`). Where cdflib's own CDF underflows to 0 SciPy
/// returns that artifact, e.g. `ncfdtrinc(3, 5, 1e-300, 2) = 9769.9999995115`; this does not
/// copy it.
#[must_use]
pub fn ncfdtrinc(dfn: f64, dfd: f64, p: f64, f: f64) -> f64 {
    if dfn.is_nan() || dfd.is_nan() || p.is_nan() || f.is_nan() {
        return f64::NAN;
    }
    if !cdflib_p_in_range(p) || f < 0.0 || dfn <= 0.0 || dfd <= 0.0 {
        return f64::NAN;
    }
    // cdffnc_which5: DS.small = 0, DS.big = 1e4; bounds 0 / 1e4 (cdflib.c 1862-1863, 1900).
    match cdflib_invert(|nc| ncfdtr(dfn, dfd, nc, f), p, 0.0, 1e4) {
        CdflibSearch::Root(nc) => nc,
        CdflibSearch::BelowLow => 0.0,
        CdflibSearch::AboveHigh => 1e4,
        CdflibSearch::Undefined => f64::NAN,
    }
}

/// cdflib's probability domain for its inverses: `0 ≤ p ≤ 1 − 1e-16` (for example
/// scipy/special/cdflib.c 1873), so `p = 1` and anything above it are NaN in SciPy 1.17.1.
/// `1 − 1e-16` rounds to `1 − 2⁻⁵³`, the largest double below 1, which is allowed.
fn cdflib_p_in_range(p: f64) -> bool {
    (0.0..=1.0 - 1e-16).contains(&p)
}

/// What cdflib's `dinvr` concluded about a root (frankenscipy-g9yid).
#[derive(Debug, Clone, Copy, PartialEq)]
enum CdflibSearch {
    /// A root inside `[small, big]`.
    Root(f64),
    /// No root: the answer lies below `small` (cdflib status 1).
    BelowLow,
    /// No root: the answer lies above `big` (cdflib status 2).
    AboveHigh,
    /// A CDF value was NaN. cdflib's own CDFs never are, so this is fsci's fail-closed exit.
    Undefined,
}

/// The root search of cdflib's `dinvr` (scipy/special/cdflib.c 3571-3789, SciPy 1.17.1) for
/// `cdf(x) = p` on `[small, big]`, with fsci's [`illinois_root`] where cdflib runs `dzror`.
///
/// 1. The residual `cdf(x) − p` is evaluated at both ends. `qincr = fbig > fsmall` is strict,
///    so a flat CDF counts as decreasing. If both ends have the sign that puts the root below
///    `small` the answer is [`CdflibSearch::BelowLow`]; above `big`, [`CdflibSearch::AboveHigh`]
///    (the table at cdflib.c 3645-3668).
/// 2. Otherwise the residual at the start `x = 5` is evaluated; an exact zero is the answer
///    (3682). Then cdflib steps away from 5 towards the root with step `max(0.5, 0.5·5) = 2.5`,
///    multiplied by 5 after each step that does not bracket (3677, 3690-3744).
/// 3. The bracket is solved by [`illinois_root`].
///
/// Every caller in SciPy uses the start 5, `absstp = relstp = 0.5` and `stpmul = 5`. Only the
/// range and the bound values differ, and those are the callers' own.
fn cdflib_invert(cdf: impl Fn(f64) -> f64, p: f64, small: f64, big: f64) -> CdflibSearch {
    const START: f64 = 5.0;
    const STEP_MULTIPLIER: f64 = 5.0;
    let residual = |x: f64| cdf(x) - p;
    let fsmall = residual(small);
    let fbig = residual(big);
    if fsmall.is_nan() || fbig.is_nan() {
        return CdflibSearch::Undefined;
    }
    let qincr = fbig > fsmall;
    if qincr {
        if fsmall > 0.0 {
            return CdflibSearch::BelowLow;
        }
        if fbig < 0.0 {
            return CdflibSearch::AboveHigh;
        }
    } else {
        if fsmall < 0.0 {
            return CdflibSearch::BelowLow;
        }
        if fbig > 0.0 {
            return CdflibSearch::AboveHigh;
        }
    }
    let y0 = residual(START);
    if y0.is_nan() {
        return CdflibSearch::Undefined;
    }
    if y0 == 0.0 {
        return CdflibSearch::Root(START);
    }
    // step = max(absstp, relstp·|x0|) = max(0.5, 0.5·5) (cdflib.c 3677).
    let mut step = 2.5_f64;
    let step_up = (qincr && y0 < 0.0) || (!qincr && y0 > 0.0);
    let (xlo, ylo, xhi, yhi) = if step_up {
        let (mut xlb, mut ylb) = (START, y0);
        let mut xub = (xlb + step).min(big);
        loop {
            let yub = residual(xub);
            if yub.is_nan() {
                return CdflibSearch::Undefined;
            }
            if (qincr && yub >= 0.0) || (!qincr && yub <= 0.0) {
                break (xlb, ylb, xub, yub);
            }
            if xub >= big {
                return CdflibSearch::AboveHigh;
            }
            step *= STEP_MULTIPLIER;
            xlb = xub;
            ylb = yub;
            xub = (xlb + step).min(big);
        }
    } else {
        let (mut xub, mut yub) = (START, y0);
        let mut xlb = (xub - step).max(small);
        loop {
            let ylb = residual(xlb);
            if ylb.is_nan() {
                return CdflibSearch::Undefined;
            }
            if (qincr && ylb <= 0.0) || (!qincr && ylb >= 0.0) {
                break (xlb, ylb, xub, yub);
            }
            if xlb <= small {
                return CdflibSearch::BelowLow;
            }
            step *= STEP_MULTIPLIER;
            xub = xlb;
            yub = ylb;
            xlb = (xub - step).max(small);
        }
    };
    if ylo == 0.0 {
        return CdflibSearch::Root(xlo);
    }
    if yhi == 0.0 {
        return CdflibSearch::Root(xhi);
    }
    // illinois_root solves an increasing residual with f(lo) < 0 < f(hi).
    let root = if ylo < 0.0 {
        illinois_root(residual, xlo, xhi, ylo, yhi)
    } else {
        illinois_root(|x| -residual(x), xlo, xhi, -ylo, -yhi)
    };
    if root.is_nan() {
        CdflibSearch::Undefined
    } else {
        CdflibSearch::Root(root)
    }
}

/// `boost::math::sign` for a residual: `1`, `-1`, or `0` for zero and NaN.
fn residual_sign(z: f64) -> i32 {
    if z > 0.0 {
        1
    } else if z < 0.0 {
        -1
    } else {
        0
    }
}

/// Boost's `tools::bracket_and_solve_root` (boost/math/tools/toms748_solve.hpp 516-622 at the
/// boost/math commit 5e088ffe that SciPy 1.17.1 pins), with [`illinois_root`] in place of TOMS
/// 748 for the final bracket (frankenscipy-g9yid).
///
/// `f` is monotone, increasing when `rising`. From `guess` the walk multiplies (or divides) by a
/// factor that starts at 2 and doubles after 32, 16, 8, 4, 2 and then every step, for at most
/// 400 steps; running out is NaN, where Boost raises an evaluation error that SciPy turns into
/// NaN. A downward walk whose residual never changes sign stops once `|a| < f64::MIN_POSITIVE`
/// and answers the midpoint of `[0, a]` (line 575), which is how SciPy's `chndtridf` and
/// `chndtrinc` produce values such as `2.65249474e-315` (`= 2⁻¹⁰⁴⁵`, the escape from a
/// guess of 1) when no root exists on the small side. A NaN residual, or a walk that
/// overflows to infinity (Boost's domain error), is NaN.
pub(crate) fn bracket_and_solve_root(f: impl Fn(f64) -> f64, guess: f64, rising: bool) -> f64 {
    const MAX_ITER: u32 = 400;
    let mut factor = 2.0_f64;
    let mut a = guess;
    let mut fa = f(a);
    if fa.is_nan() {
        return f64::NAN;
    }
    if fa == 0.0 {
        return a;
    }
    let mut b = a;
    let mut fb = fa;
    let mut count = MAX_ITER - 1;
    let mut step = 32_u32;
    let zero_is_right = (fa < 0.0) == if guess < 0.0 { !rising } else { rising };
    if zero_is_right {
        while residual_sign(fb) == residual_sign(fa) {
            if count == 0 {
                return f64::NAN;
            }
            if (MAX_ITER - count).is_multiple_of(step) {
                factor *= 2.0;
                if step > 1 {
                    step /= 2;
                }
            }
            a = b;
            fa = fb;
            b *= factor;
            if !b.is_finite() {
                return f64::NAN;
            }
            fb = f(b);
            if fb.is_nan() {
                return f64::NAN;
            }
            count -= 1;
        }
    } else {
        while residual_sign(fb) == residual_sign(fa) {
            if a.abs() < f64::MIN_POSITIVE {
                let (lo, hi) = if a > 0.0 { (0.0, a) } else { (a, 0.0) };
                return lo + (hi - lo) / 2.0;
            }
            if count == 0 {
                return f64::NAN;
            }
            if (MAX_ITER - count).is_multiple_of(step) {
                factor *= 2.0;
                if step > 1 {
                    step /= 2;
                }
            }
            b = a;
            fb = fa;
            a /= factor;
            fa = f(a);
            if fa.is_nan() {
                return f64::NAN;
            }
            count -= 1;
        }
    }
    if fa == 0.0 {
        return a;
    }
    if fb == 0.0 {
        return b;
    }
    let (lo, hi, flo, fhi) = if a < b {
        (a, b, fa, fb)
    } else {
        (b, a, fb, fa)
    };
    if flo < 0.0 {
        illinois_root(&f, lo, hi, flo, fhi)
    } else {
        illinois_root(|x| -f(x), lo, hi, -flo, -fhi)
    }
}

/// Midpoint for [`illinois_root`] when false position gives no usable step: the geometric
/// mean when the bracket excludes 0, so a bracket spanning many decades is halved in log
/// space, and the arithmetic mean otherwise (frankenscipy-g9yid).
fn bracket_midpoint(lo: f64, hi: f64) -> f64 {
    if lo > 0.0 {
        lo.sqrt() * hi.sqrt()
    } else if hi < 0.0 {
        -((-lo).sqrt() * (-hi).sqrt())
    } else {
        0.5 * (lo + hi)
    }
}

/// Root of a monotone-increasing `f` in the bracket `[lo, hi]` with
/// `f(lo) = flo < 0 < fhi = f(hi)`, via the Illinois modified false-position
/// method. Keeps the sign-change bracket at every step (so convergence is
/// guaranteed like bisection) but converges superlinearly, cutting the number
/// of expensive `f` evaluations from ~100 (plain bisection) to ~10-15 — the
/// dominant cost of the noncentral-t/F inverse CDFs, whose `f` is itself a
/// several-µs series. Returns the root to full `f64` precision.
///
/// Three things changed in frankenscipy-g9yid, each because the old version returned a wrong
/// root while reporting success:
///
/// 1. **A stagnating iterate is verified before it is returned.** The search stops when the
///    false-position iterate moves by less than the tolerance, or when it rounds onto an
///    endpoint. That also happens far from the root, when the residual at the stale endpoint
///    is tiny, for example on an underflowed plateau such as `1e-300 − chndtr(5, df, 2)` for
///    `df ≳ 420`. The old search returned 4.88e6 there for a root of 403.34. Now the point
///    `2·tol` further into the bracket is evaluated: a sign change there proves the root is
///    within `2·tol` and the iterate is returned; otherwise that endpoint moves to the probe
///    and a bisection step follows. On a smooth residual this costs one extra evaluation.
/// 2. **The tolerance is relative once the bracket excludes 0**, `4ε·|x|`. The old
///    `4ε·max(|x|, 1)` was absolute below 1, so a root near 1e-200 was accepted from any
///    bracket narrower than 1e-15, that is at once. Brackets that touch or straddle 0 keep
///    the old tolerance, because a root at exactly 0 has no relative scale.
/// 3. **The false-position step is `lo + (hi − lo)·flo/(flo − fhi)`**, which has no product of
///    an abscissa and a residual. The old `(lo·fhi − hi·flo)/(fhi − flo)` underflowed to 0/0
///    when both were tiny, and every step fell back to a midpoint. The midpoint is now
///    geometric when the bracket excludes 0.
///
/// A NaN residual ends the search with NaN.
pub(crate) fn illinois_root<F: Fn(f64) -> f64>(
    f: F,
    mut lo: f64,
    mut hi: f64,
    mut flo: f64,
    mut fhi: f64,
) -> f64 {
    const REL_TOL: f64 = 4.0 * f64::EPSILON;
    let mut side = 0i32;
    // `prev` holds the PREVIOUS iterate for the stagnation test; there is none yet, so
    // seed it as NaN rather than the midpoint. Otherwise, when the root sits at
    // (or within rounding of) an endpoint, the false-position candidate is
    // rejected on iteration 1, `next` falls back to the midpoint which equals the
    // seed, and `(next - prev) == 0` fires a SPURIOUS immediate return of the
    // midpoint (observed on gammainc_shape_inv/pdtrik where roots land on the
    // power-of-2 bracket endpoint).
    let mut prev = f64::NAN;
    for _ in 0..200 {
        let width = hi - lo;
        let candidate = lo + width * (flo / (flo - fhi));
        // `hug` is the endpoint the estimate claims the root sits at: 1 = hi, -1 = lo.
        let (next, mut hug) = if candidate.is_nan() {
            (bracket_midpoint(lo, hi), 0)
        } else if candidate >= hi {
            (hi, 1)
        } else if candidate <= lo {
            (lo, -1)
        } else {
            (candidate, 0)
        };
        let tol = if lo > 0.0 || hi < 0.0 {
            REL_TOL * next.abs()
        } else {
            REL_TOL * next.abs().max(1.0)
        };
        if width <= tol {
            return next;
        }
        if hug == 0 && (next - prev).abs() <= tol {
            // The iterate stopped moving next to the endpoint just replaced by `prev`.
            hug = side;
        }
        if hug != 0 {
            let probe = if hug == 1 {
                next - 2.0 * tol
            } else {
                next + 2.0 * tol
            };
            if !(probe > lo && probe < hi) {
                return next;
            }
            let fprobe = f(probe);
            if fprobe.is_nan() {
                return f64::NAN;
            }
            if fprobe == 0.0 {
                return probe;
            }
            if (hug == 1) == (fprobe < 0.0) {
                return next;
            }
            if fprobe > 0.0 {
                hi = probe;
                fhi = fprobe;
            } else {
                lo = probe;
                flo = fprobe;
            }
            let mid = bracket_midpoint(lo, hi);
            if mid > lo && mid < hi {
                let fmid = f(mid);
                if fmid.is_nan() {
                    return f64::NAN;
                }
                if fmid == 0.0 {
                    return mid;
                }
                if fmid > 0.0 {
                    hi = mid;
                    fhi = fmid;
                } else {
                    lo = mid;
                    flo = fmid;
                }
            }
            side = 0;
            prev = f64::NAN;
            continue;
        }
        prev = next;
        let fnext = f(next);
        if fnext.is_nan() {
            return f64::NAN;
        }
        if fnext == 0.0 {
            return next;
        }
        if fnext > 0.0 {
            hi = next;
            fhi = fnext;
            if side == 1 {
                flo *= 0.5; // Illinois down-weight of the stale endpoint
            }
            side = 1;
        } else {
            lo = next;
            flo = fnext;
            if side == -1 {
                fhi *= 0.5;
            }
            side = -1;
        }
    }
    lo + 0.5 * (hi - lo)
}

/// Inverse of [`nctdtr`] in the argument `t`.
///
/// Returns `t` such that `nctdtr(df, nc, t) = p`, matching
/// `scipy.special.nctdtrit(df, nc, p)`. Following SciPy, the exact boundaries `p ≤ 0`
/// and `p ≥ 1` return `+∞`.
///
/// SciPy 1.17.1 answers with Boost's quantile, which first puts its guess on the side of 0
/// where `nctdtr(df, nc, 0)` says the root lies and then brackets by a geometric walk
/// (boost/math/distributions/non_central_t.hpp 342-389). This does the same from `±1` with
/// `bracket_and_solve_root` (frankenscipy-g9yid), so the number of CDF calls grows with
/// `log|t|` only. SciPy is NaN from `|nc| ≈ 1.0145e5` for every `p` and `df` measured (Boost's
/// series hits its term limit), while [`nctdtr`] deliberately stays finite there, and this
/// inverse stays consistent with it and answers. The owner may revisit that choice.
#[must_use]
pub fn nctdtrit(df: f64, nc: f64, p: f64) -> f64 {
    if df.is_nan() || nc.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    // nc = ±inf: SciPy 1.17.1 returns nan for every p, including the p = 0 and p = 1
    // edges that otherwise return +inf below (frankenscipy-qu5po).
    if df <= 0.0 || nc.is_infinite() {
        return f64::NAN;
    }
    if p <= 0.0 || p >= 1.0 {
        return f64::INFINITY;
    }
    let at_zero = nctdtr(df, nc, 0.0);
    if at_zero.is_nan() {
        return f64::NAN;
    }
    if at_zero == p {
        return 0.0;
    }
    let guess = if at_zero < p { 1.0 } else { -1.0 };
    // A NaN CDF (the `POISSON_INDEX_LIMIT` exit) makes the walk return NaN rather than a
    // bracket. SciPy 1.17.1: nctdtrit(5, 1.35e8, 0.5) = nan.
    bracket_and_solve_root(|t| nctdtr(df, nc, t) - p, guess, true)
}

/// Inverse of [`nctdtr`] in the non-centrality `nc`.
///
/// Returns `nc` such that `nctdtr(df, nc, t) = p`, matching
/// `scipy.special.nctdtrinc(df, p, t)`.
///
/// SciPy 1.17.1 answers with cdflib's `cdftnc_which4` (scipy/special/cdflib.c 2504-2561): it
/// clamps `t` to `±f64::MAX` and `df` to at most `1e10`, searches `nc ∈ [−1e6, 1e6]` from
/// `nc = 5`, and returns `1e6` when the root is above the range and `0` (not `−1e6`, line
/// 2554) when it is below. `cdflib_invert` reproduces that search on fsci's own [`nctdtr`]
/// (frankenscipy-g9yid). SciPy gives `nctdtrinc(5, 0.5, t) = 1e6` for `t = inf`, `1e300` and
/// `3e8`, and `0.0` for `t = −inf` and `−1e9`. The old unbounded doubling returned `±inf`, or
/// NaN after about 1.2e9 Poisson-walk steps. `p` must lie in `[0, 1 − 1e-16]`, so `p = 1`
/// is NaN.
///
/// Inside the range the root is fsci's; cdflib truncates its series at a relative 1e-7
/// (`cumtnc`, cdflib.c 3362), so SciPy's interior values differ by about 1e-8 near `t = 2`
/// and by 1e-3 at `t = 1e5`. Three SciPy defects are not copied: for `3e5 ≲ |t| ≲ 1e8`
/// SciPy does not return, because `cumtnc`'s forward loop cannot meet its stop test at
/// `nc = −1e6`; at `t = 0` its C `cumnor` swaps the tail for `0 < x ≤ 0.66291`, so
/// `nctdtrinc(5, 0.5, 0) = −0.6629099965481363`; and where `cumtnc` underflows it returns
/// the artifact `nctdtrinc(5, 1e-300, 2) = 82.4999995875`.
#[must_use]
pub fn nctdtrinc(df: f64, p: f64, t: f64) -> f64 {
    if df.is_nan() || p.is_nan() || t.is_nan() {
        return f64::NAN;
    }
    if !cdflib_p_in_range(p) || df <= 0.0 {
        return f64::NAN;
    }
    // cdflib.c 2542-2543: t = fmax(fmin(t, DBL_MAX), -DBL_MAX); df = fmin(df, 1e10).
    let t = t.clamp(-f64::MAX, f64::MAX);
    let df = df.min(1e10);
    // cdftnc_which4: DS.small = -1e6, DS.big = 1e6; bounds 0 / 1e6 (2514-2515, 2554).
    match cdflib_invert(|nc| nctdtr(df, nc, t), p, -1e6, 1e6) {
        CdflibSearch::Root(nc) => nc,
        CdflibSearch::BelowLow => 0.0,
        CdflibSearch::AboveHigh => 1e6,
        CdflibSearch::Undefined => f64::NAN,
    }
}

/// cdflib's `cumfnc` reports an error (status 1, which SciPy turns into NaN) when
/// `(int)(nc/2)` overflows, that is from `nc/2 = 2³¹` on (scipy/special/cdflib.c 2966-2969).
/// It checks this only for `f > 0`: `f ≤ 0` has already returned 0 (2957-2959).
/// SciPy 1.17.1: `ncfdtridfd(3, 0.5, nc, 2)` is `1e-100` at `nc = 4294967294` and NaN at
/// `4294967296`, `1e10` and `inf`, while `ncfdtridfd(3, 0.5, 1e10, 0) = 1e-100`.
fn cumfnc_rejects_nc(nc: f64, f: f64) -> bool {
    f > 0.0 && nc / 2.0 >= 2_147_483_648.0
}

/// Inverse of [`ncfdtr`] in the denominator degrees of freedom `dfd`.
///
/// Returns `dfd` such that `ncfdtr(dfn, dfd, nc, f) = p`, matching
/// `scipy.special.ncfdtridfd(dfn, p, nc, f)`.
///
/// SciPy 1.17.1 answers with cdflib's `cdffnc_which4` (scipy/special/cdflib.c 1800-1853),
/// which searches `dfd ∈ [1e-100, 1e100]` from `dfd = 5` and returns `1e-100` below the range
/// and `1e100` above it. `cdflib_invert` reproduces the search on fsci's own [`ncfdtr`]
/// (frankenscipy-g9yid). SciPy gives `ncfdtridfd(3, 0.5, 2, inf) = 1e100` and
/// `ncfdtridfd(3, 0.5, 2, 0) = 1e-100`, where the old `[1e-8, 1e12]` search returned `1e-8`
/// and `1e12`. `p` must lie in `[0, 1 − 1e-16]`; see `cumfnc_rejects_nc` for the NaN at
/// large `nc`.
///
/// Not copied: where the target is above the `dfd → ∞` limit of the CDF, cdflib's `cumfnc`
/// rounds `1 − dfn·f/(dfn·f + dfd)` to 1 from `dfd ≈ dfn·f·2⁵⁴` on and its CDF jumps there, so
/// SciPy returns that point, `ncfdtridfd(3, 0.7, 2, 2) = 1.0808639105448624e17`. fsci's
/// [`ncfdtr`] forms `1 − y` in closed form and has no such jump; a CDF that stays at its limit
/// gives the `1e100` bound instead.
#[must_use]
pub fn ncfdtridfd(dfn: f64, p: f64, nc: f64, f: f64) -> f64 {
    if dfn.is_nan() || p.is_nan() || nc.is_nan() || f.is_nan() {
        return f64::NAN;
    }
    if !cdflib_p_in_range(p) || f < 0.0 || dfn <= 0.0 || nc < 0.0 || cumfnc_rejects_nc(nc, f) {
        return f64::NAN;
    }
    // cdffnc_which4: DS.small = 1e-100, DS.big = 1e100; bounds 1e-100 / 1e100 (1807-1808, 1846).
    match cdflib_invert(|dfd| ncfdtr(dfn, dfd, nc, f), p, 1e-100, 1e100) {
        CdflibSearch::Root(dfd) => dfd,
        CdflibSearch::BelowLow => 1e-100,
        CdflibSearch::AboveHigh => 1e100,
        CdflibSearch::Undefined => f64::NAN,
    }
}

/// Inverse of [`ncfdtr`] in the numerator degrees of freedom `dfn`.
///
/// Returns `dfn` such that `ncfdtr(dfn, dfd, nc, f) = p`, matching
/// `scipy.special.ncfdtridfn(p, dfd, nc, f)`.
///
/// SciPy 1.17.1 answers with cdflib's `cdffnc_which3` (scipy/special/cdflib.c 1745-1798): the
/// same search as [`ncfdtridfd`], over `dfn ∈ [1e-100, 1e100]` from `dfn = 5`, with the bounds
/// `1e-100` and `1e100` (frankenscipy-g9yid). SciPy gives `ncfdtridfn(p, 5, 2, 2) = 1e-100` for
/// `p = 0.2` and `1e100` for `p = 0.99`, where the old `[1e-8, 1e12]` search returned `1e-8`
/// and `1e12`, and NaN at `nc = 1e10` (see `cumfnc_rejects_nc`), where it returned 1.35e10.
/// The CDF need not be monotone in `dfn`; like cdflib, the search takes the root its steps
/// from 5 bracket first.
#[must_use]
pub fn ncfdtridfn(p: f64, dfd: f64, nc: f64, f: f64) -> f64 {
    if dfd.is_nan() || p.is_nan() || nc.is_nan() || f.is_nan() {
        return f64::NAN;
    }
    if !cdflib_p_in_range(p) || f < 0.0 || dfd <= 0.0 || nc < 0.0 || cumfnc_rejects_nc(nc, f) {
        return f64::NAN;
    }
    // cdffnc_which3: DS.small = 1e-100, DS.big = 1e100; bounds 1e-100 / 1e100 (1752-1753, 1791).
    match cdflib_invert(|dfn| ncfdtr(dfn, dfd, nc, f), p, 1e-100, 1e100) {
        CdflibSearch::Root(dfn) => dfn,
        CdflibSearch::BelowLow => 1e-100,
        CdflibSearch::AboveHigh => 1e100,
        CdflibSearch::Undefined => f64::NAN,
    }
}

/// Inverse of [`nctdtr`] in the degrees of freedom `df`.
///
/// Returns `df` such that `nctdtr(df, nc, t) = p`, matching
/// `scipy.special.nctdtridf(p, nc, t)`.
///
/// SciPy 1.17.1 answers with cdflib's `cdftnc_which3` (scipy/special/cdflib.c 2445-2502): it
/// clamps `t` to `±f64::MAX`, rejects `nc ∉ [−1e6, 1e6]` (NaN), searches `df ∈ [1e-100, 1e10]`
/// from `df = 5`, and returns `1e100` above the range and **`−1e100`** below it (line 2495).
/// `cdflib_invert` reproduces the search on fsci's own [`nctdtr`] (frankenscipy-g9yid).
/// SciPy gives `nctdtridf(0.5, 1, inf) = 1e100`, `nctdtridf(0.5, 1, −inf) = −1e100` and
/// `nctdtridf(0, 1, 1.5) = −1e100`, where the old `[1e-8, 1e12]` search returned `1e-8`,
/// `1e12` and `1e-8`. `p` must lie in `[0, 1 − 1e-16]`.
#[must_use]
pub fn nctdtridf(p: f64, nc: f64, t: f64) -> f64 {
    if p.is_nan() || nc.is_nan() || t.is_nan() {
        return f64::NAN;
    }
    if !cdflib_p_in_range(p) || !(-1e6..=1e6).contains(&nc) {
        return f64::NAN;
    }
    let t = t.clamp(-f64::MAX, f64::MAX);
    // cdftnc_which3: DS.small = 1e-100, DS.big = 1e10; bounds -1e100 / 1e100 (2455-2456, 2495).
    match cdflib_invert(|df| nctdtr(df, nc, t), p, 1e-100, 1e10) {
        CdflibSearch::Root(df) => df,
        CdflibSearch::BelowLow => -1e100,
        CdflibSearch::AboveHigh => 1e100,
        CdflibSearch::Undefined => f64::NAN,
    }
}

/// Non-central Student's t cumulative distribution function.
///
/// Matches `scipy.special.nctdtr(df, nc, t)`: the CDF at `t` of a non-central
/// t variable with `df` degrees of freedom and non-centrality `nc`.
///
/// Uses Lenth's (1989, AS 243) series. For `t ≥ 0`,
///
/// ```text
///   P(T ≤ t) = Φ(−δ) + ½ Σ_{j≥0} [ p_j·I_x(j+½, df/2) + q_j·I_x(j+1, df/2) ],
///   x = t²/(t²+df),  λ = δ²/2,
///   p_j = e^{−λ} λ^j / j!,  q_j = (δ/√2)·e^{−λ} λ^j / Γ(j+3/2)
/// ```
///
/// with `nctdtr(df, nc, t) = 1 − nctdtr(df, −nc, −t)` for `t < 0`. The series is
/// summed outward from the Poisson mode `j₀=⌊λ⌋` (mode weights from
/// `gamma::poisson_term`, with `q`'s sign carried separately) so large `nc` stays
/// stable.
#[must_use]
pub fn nctdtr(df: f64, nc: f64, t: f64) -> f64 {
    if df.is_nan() || nc.is_nan() || t.is_nan() {
        return f64::NAN;
    }
    if df <= 0.0 {
        return f64::NAN;
    }
    if nc.is_infinite() {
        // λ = nc²/2 = ∞ has no Poisson mode to walk from: j₀ = ∞ and `j -= 1.0` never
        // reaches 0 (frankenscipy-qu5po). SciPy 1.17.1 keeps only the limits at t = ±inf:
        // nctdtr(5, ±inf, inf) = 1.0, nctdtr(5, ±inf, -inf) = 0.0, and nan for every
        // finite t (-1e300, -2, 0, 2, 1e300).
        return if t == f64::INFINITY {
            1.0
        } else if t == f64::NEG_INFINITY {
            0.0
        } else {
            f64::NAN
        };
    }
    if t < 0.0 {
        // The left tail is the survival of the reflected law, taken directly: `1 − nctdtr(df,
        // −nc, −t)` cancelled to 0 (or to ~1e-16 of garbage) below about 1e-16, e.g.
        // nctdtr(5, 10, −7.40744670100678) was 0 where the law is 1e-30.
        return nctdtrc(df, -nc, -t);
    }
    let phi = crate::convenience::ndtr_scalar(-nc);
    if t == 0.0 {
        return phi;
    }
    if (t * t).is_infinite() {
        // t = inf, or t ≥ 1.3407807929942596e154 where t² overflows: x = t²/(t² + df) was
        // inf/inf = NaN (frankenscipy-g9yid). SciPy 1.17.1 answers 1.0, and so 0.0 at -t
        // through the reflection above: nctdtr(5, nc, inf) = 1.0 for nc = -3, 0, 3 and 1e4;
        // nctdtr(df, 3, 1e300) = 1.0 for df = 0.001, 0.5, 5, 1e300 and inf;
        // nctdtr(5, 3, 2e154) = nctdtr(5, 3, 1.3407807929942596e154) = 1.0; and
        // nctdtr(5, 3, -inf) = nctdtr(5, 3, -1e300) = 0.0.
        return 1.0;
    }
    let x = t * t / (t * t + df);
    let half_df = 0.5 * df;
    let lam = 0.5 * nc * nc;
    if lam == 0.0 {
        // With the closed-form 1 − x, as the λ > 0 anchors below take it (frankenscipy-5pnba):
        // at df = 1e-100, x rounds to 1 and `btdtr(0.5, df/2, x)` said 1, so nctdtr(1e-100, 0,
        // 1.2) was 1.0 where SciPy and stdtr give 0.5.
        let x1 = df / (t * t + df);
        return phi + 0.5 * betainc_with_complement(0.5, half_df, x, x1).unwrap_or(f64::NAN);
    }

    let j0 = lam.floor();
    // frankenscipy-qu5po: the walk cannot step from here (see `POISSON_INDEX_LIMIT`). This
    // also catches a finite |nc| above ~1.3e154, where nc² overflows and λ = ∞.
    if j0 >= POISSON_INDEX_LIMIT {
        return f64::NAN;
    }
    // Saddle-point anchors (frankenscipy-g9yid). In log space p0, q0 and the increments tp0,
    // tq0 each lost about λ·ln λ·ε: 2.5e-7 at λ = 1e8 and 4e-3 at λ = 1e12, and nctdtr at the
    // mean missed mpmath by 5e-8 and 5e-3 there. See `gamma::poisson_term` and `beta_term`.
    // p0 = e^{−λ} λ^{j0} / j0!, and q0 = (δ/√2)·e^{−λ} λ^{j0} / Γ(j0 + 3/2), which is
    // (δ/√(2λ))·poisson_term(j0 + ½, λ) with δ/√(2λ) = ±1 up to the rounding of λ.
    let p0 = gamma::poisson_term(j0, lam);
    let q_sign = if nc >= 0.0 { 1.0 } else { -1.0 };
    let q0 = q_sign
        * (nc.abs() / std::f64::consts::SQRT_2 / lam.sqrt())
        * gamma::poisson_term(j0 + 0.5, lam);

    // The p- and q-terms need I_x(a, df/2) along the integer-stepped chains
    // a = j+½ and a = j+1. Rather than a fresh `btdtr` continued fraction per
    // term (the old cost — 2 per j), seed ONE `btdtr` at the Poisson mode for
    // each chain and march via the incomplete-beta a-recurrence:
    //   I_x(a+1,b) = I_x(a,b) − T(a,b),  T(a,b) = xᵃ(1−x)ᵇ / (a·B(a,b)),
    //   T(a+1,b) = T(a,b)·x(a+b)/(a+1)   (downward: T(a−1)=T(a)·a/(x(a−1+b))).
    // The recurrence is absolutely stable (~1e-15); its relative precision only
    // decays where I_x→0, which coincides with negligible Poisson weight (and
    // the loop's early-exit), so the summed CDF stays exact to ~1e-14.
    //
    // 1 − x in closed form, df/(t² + df) (frankenscipy-g9yid). x rounds to within ε of 1 at
    // large t, so `1.0 - x` carried a relative error of up to ε·t²/(2·df), 6e-11 at t = 1682,
    // and with saddle-point anchors nctdtr at the mean still missed mpmath by 2e-11 at
    // λ = 1e6, 2e-9 at 1e8 and 3e-7 at 1e10. The anchors now take x1 for 1 − x: 1e-14, 1e-13
    // and 1e-12.
    let x1 = df / (t * t + df);
    let ap0 = j0 + 0.5;
    let aq0 = j0 + 1.0;
    let ip0 = betainc_with_complement(ap0, half_df, x, x1).unwrap_or(f64::NAN);
    let iq0 = betainc_with_complement(aq0, half_df, x, x1).unwrap_or(f64::NAN);
    // T(a, b) = xᵃ(1−x)ᵇ / (a·B(a, b)) = xᵃ(1−x)ᵇ·Γ(a+b) / (Γ(a+1)·Γ(b)).
    let tp0 = beta_term(ap0, half_df, x, x1);
    let tq0 = beta_term(aq0, half_df, x, x1);
    // frankenscipy-qu5po: the all-zero exit (see `POISSON_INDEX_LIMIT`). With every anchor
    // zero the series `s` is exactly 0 and the CDF is Φ(−δ).
    if ip0 == 0.0 && iq0 == 0.0 && tp0 == 0.0 && tq0 == 0.0 {
        return phi.clamp(0.0, 1.0);
    }

    let mut s = 0.0_f64;
    // Upward from the mode. The cap scales with √λ (frankenscipy-g9yid; see
    // `poisson_upward_step_cap`).
    let (mut p, mut q, mut j) = (p0, q0, j0);
    let (mut ip, mut iq, mut tp, mut tq) = (ip0, iq0, tp0, tq0);
    let (mut ap, mut aq) = (ap0, aq0);
    let cap = poisson_upward_step_cap(lam);
    let mut steps = 0.0_f64;
    while steps < cap {
        s += p * ip + q * iq;
        ip -= tp;
        tp *= x * (ap + half_df) / (ap + 1.0);
        ap += 1.0;
        iq -= tq;
        tq *= x * (aq + half_df) / (aq + 1.0);
        aq += 1.0;
        j += 1.0;
        p *= lam / j;
        q *= lam / (j + 0.5);
        let m = p.abs().max(q.abs());
        if (p.abs() < 1e-300 && q.abs() < 1e-300) || (m < 1e-17 * s.abs().max(1e-300) && j > lam) {
            break;
        }
        steps += 1.0;
    }
    // Downward from the mode.
    p = p0;
    q = q0;
    j = j0;
    ip = ip0;
    iq = iq0;
    tp = tp0;
    tq = tq0;
    ap = ap0;
    aq = aq0;
    while j > 0.0 {
        p *= j / lam;
        q *= (j + 0.5) / lam;
        j -= 1.0;
        tp *= ap / (x * (ap - 1.0 + half_df));
        ap -= 1.0;
        ip += tp;
        tq *= aq / (x * (aq - 1.0 + half_df));
        aq -= 1.0;
        iq += tq;
        s += p * ip + q * iq;
        if p.abs().max(q.abs()) < 1e-17 * s.abs().max(1e-300) {
            break;
        }
    }
    (phi + 0.5 * s).clamp(0.0, 1.0)
}

/// Non-central t survival function `P(T > t) = 1 − nctdtr(df, nc, t)`, computed without
/// forming `1 − nctdtr` where that would cancel (frankenscipy-g9yid).
///
/// Three regimes for `t > 0`:
/// - the tail on the side of the mean (`nc ≥ 0`): past Boost's crossover, the complement series
///   of [`nct_upper_series`], whose terms are all non-negative;
/// - across zero from the mean (`nc < 0`): once `1 − nctdtr` would drop below 0.1, the
///   positive-integrand quadrature of [`nct_far_tail`]. Every Poisson series there is a
///   difference of O(1) sums, and SciPy's own `nct.sf` cancels in this regime;
/// - otherwise `1 − nctdtr`, which is then at least 0.1 and loses nothing.
///
/// `t ≤ 0` is `nctdtr(df, −nc, −t)`, the positive-term `t ≥ 0` series.
#[must_use]
pub fn nctdtrc(df: f64, nc: f64, t: f64) -> f64 {
    if df.is_nan() || nc.is_nan() || t.is_nan() {
        return f64::NAN;
    }
    if df <= 0.0 {
        return f64::NAN;
    }
    if nc.is_infinite() {
        // The complement of nctdtr's limits: 1.0 at t = inf and 0.0 at t = −inf, NaN otherwise.
        return if t == f64::INFINITY {
            0.0
        } else if t == f64::NEG_INFINITY {
            1.0
        } else {
            f64::NAN
        };
    }
    if t <= 0.0 {
        return nctdtr(df, -nc, -t);
    }
    if (t * t).is_infinite() {
        return 0.0;
    }
    nctdtrc_positive_t(df, nc, t)
}

/// The share of `1 − nctdtr` below which the `nc < 0` survival leaves the subtraction for the
/// far-tail quadrature.
const NCT_FAR_TAIL_SWITCH: f64 = 0.1;

/// `P(T > t)` for `t > 0` with `t²` finite: the regime choice of [`nctdtrc`].
fn nctdtrc_positive_t(df: f64, nc: f64, t: f64) -> f64 {
    if nc >= 0.0 {
        // Boost's crossover (non_central_t_cdf): below it the CDF side is the smaller one.
        let tt = t * t;
        let x = tt / (tt + df);
        let d2 = nc * nc;
        let b = 0.5 * df;
        let c = 0.5 + b + 0.5 * d2;
        let cross = 1.0 - (b / c) * (1.0 + d2 / (2.0 * c * c));
        if x < cross {
            return 1.0 - nctdtr(df, nc, t);
        }
        return nct_upper_series(df, nc, t);
    }
    if crate::convenience::ndtr_scalar(nc) >= NCT_FAR_TAIL_SWITCH {
        let v = 1.0 - nctdtr(df, nc, t);
        if v >= NCT_FAR_TAIL_SWITCH {
            return v;
        }
    }
    nct_far_tail(df, -nc, t)
}

/// `P(T > t)` for `nc ≥ 0`, `t > 0`: `Q = ½ Σ_j [p_j·I_y(df/2, j+½) + q_j·I_y(df/2, j+1)]` with
/// `y = df/(t² + df)` in closed form. Every term is non-negative: the CDF's `Φ(−nc)` cancels
/// exactly, since `Σ p_j = 1` and `Σ q_j = erf(nc/√2)` (Boost's `non_central_t2_q`). It is
/// anchored at the Poisson mode like [`nctdtr`], each complement is taken directly with the
/// roles swapped (never `1 − I_x`), the upward walk adds (the stable direction here), and the
/// downward walk subtracts down to the cancellation floor, as [`ncfdtrc`] does.
fn nct_upper_series(df: f64, nc: f64, t: f64) -> f64 {
    let tt = t * t;
    let x = tt / (tt + df);
    let y = df / (tt + df);
    let half_df = 0.5 * df;
    let lam = 0.5 * nc * nc;
    if lam == 0.0 {
        // Student t: P(T > t) = ½·I_y(df/2, ½).
        return 0.5 * betainc_with_complement(half_df, 0.5, y, x).unwrap_or(f64::NAN);
    }
    let j0 = lam.floor();
    if j0 >= POISSON_INDEX_LIMIT {
        return f64::NAN;
    }
    let p0 = gamma::poisson_term(j0, lam);
    let q0 = (nc / std::f64::consts::SQRT_2 / lam.sqrt()) * gamma::poisson_term(j0 + 0.5, lam);
    let ap0 = j0 + 0.5;
    let aq0 = j0 + 1.0;
    let cp0 = betainc_with_complement(half_df, ap0, y, x).unwrap_or(f64::NAN);
    let cq0 = betainc_with_complement(half_df, aq0, y, x).unwrap_or(f64::NAN);
    let tp0 = beta_term(ap0, half_df, x, y);
    let tq0 = beta_term(aq0, half_df, x, y);
    // `f64::min`/`max` below would drop a NaN, so a NaN anchor ends here.
    if [p0, q0, cp0, cq0, tp0, tq0].iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    if cp0 == 0.0 && cq0 == 0.0 && tp0 == 0.0 && tq0 == 0.0 {
        return 0.0;
    }
    let mut total = 0.0_f64;
    // Upward from the mode: Ī(a+1) = Ī(a) + T(a).
    let (mut p, mut q, mut j) = (p0, q0, j0);
    let (mut cp, mut cq, mut tp, mut tq) = (cp0, cq0, tp0, tq0);
    let (mut ap, mut aq) = (ap0, aq0);
    let cap = poisson_upward_step_cap(lam);
    let mut steps = 0.0_f64;
    while steps < cap {
        total += p * cp.min(1.0) + q * cq.min(1.0);
        cp += tp;
        tp *= x * (ap + half_df) / (ap + 1.0);
        ap += 1.0;
        cq += tq;
        tq *= x * (aq + half_df) / (aq + 1.0);
        aq += 1.0;
        j += 1.0;
        p *= lam / j;
        q *= lam / (j + 0.5);
        if (p < 1e-300 && q < 1e-300) || (p.max(q) < 1e-17 * total.max(1e-300) && j > lam) {
            break;
        }
        steps += 1.0;
    }
    // Downward: Ī(a−1) = Ī(a) − T(a−1). A complement driven to ≤ 0 by rounding is the floor.
    let (mut p, mut q, mut j) = (p0, q0, j0);
    let (mut cp, mut cq, mut tp, mut tq) = (cp0, cq0, tp0, tq0);
    let (mut ap, mut aq) = (ap0, aq0);
    while j > 0.0 {
        p *= j / lam;
        q *= (j + 0.5) / lam;
        j -= 1.0;
        tp *= ap / (x * (ap - 1.0 + half_df));
        ap -= 1.0;
        cp -= tp;
        tq *= aq / (x * (aq - 1.0 + half_df));
        aq -= 1.0;
        cq -= tq;
        if cp <= 0.0 && cq <= 0.0 {
            break;
        }
        total += p * cp.max(0.0) + q * cq.max(0.0);
        if p.max(q) < 1e-17 * total.max(1e-300) {
            break;
        }
    }
    (0.5 * total).clamp(0.0, 1.0)
}

/// `W(df, d, τ) = P(T ≤ −τ | nc = d)` for `d ≥ 0`, `τ > 0`: the tail across zero from the mean,
/// as a trapezoid rule in `u` over a strictly log-concave positive integrand, so the peak is
/// unique and the stop test is a bound, not a guess. `df ≥ 2` integrates the density form and
/// `df < 2` the integration-by-parts form. The latter's incomplete-gamma factor becomes a sharp
/// step at large `df`, while the density form needs thousands of nodes at small `df`.
fn nct_far_tail(df: f64, d: f64, tau: f64) -> f64 {
    if df >= 2.0 {
        nct_far_tail_density(df, d, tau)
    } else {
        nct_far_tail_ibp(df, d, tau)
    }
}

/// `√(2/π)`.
const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;
/// `ln √(2π)`.
const NCT_LN_SQRT_2PI: f64 = 0.918_938_533_204_672_8;
/// Trapezoid node cap per direction; a log-concave integrand stops long before it.
const NCT_TRAPEZOID_MAX_NODES: usize = 100_000;

/// Density form, `s = e^u`, `z = d + τs`:
/// `W = e^{−d²/2}·∫ exp(ln df + ln pt(df/2, df·s²/2) + ln(½·erfcx(z/√2)) − dτs − (τs)²/2) du`,
/// with `pt` the Poisson term (the density of `V = df·S²` in Loader's form), and `Φ(−z)` written
/// as `½·erfcx(z/√2)·e^{−z²/2}` so that `e^{−d²/2}` comes out exactly.
fn nct_far_tail_density(df: f64, d: f64, tau: f64) -> f64 {
    use crate::convenience::erfcx_scalar;
    use std::f64::consts::FRAC_1_SQRT_2;
    let a = 0.5 * df;
    let ln_df = df.ln();
    let ln_norm = 0.5 * (std::f64::consts::TAU * a).ln();
    let mills = |z: f64| SQRT_2_OVER_PI / erfcx_scalar(z * FRAC_1_SQRT_2);
    let logf = |u: f64| {
        let s = u.exp();
        let ts = tau * s;
        let lam = 0.5 * df * s * s;
        if lam == 0.0 || ts.is_infinite() {
            return f64::NEG_INFINITY;
        }
        let z = d + ts;
        ln_df - gamma::stirlerr(a) - gamma::bd0(a, lam) - ln_norm
            + (0.5 * erfcx_scalar(z * FRAC_1_SQRT_2)).ln()
            - d * ts
            - 0.5 * ts * ts
    };
    // l′(u) = df·(1 − s²) − τs·M(d + τs), M the inverse Mills ratio; decreasing in u.
    let slope = |u: f64| {
        let s = u.exp();
        df * (1.0 - s * s) - tau * s * mills(d + tau * s)
    };
    // l′(0) < 0, and l′ > 0 at s_lo (M(z) < z + 1).
    let s_lo = 0.5_f64.min(1.5 * df / (tau * ((d + 1.0) + ((d + 1.0).powi(2) + 3.0 * df).sqrt())));
    let us = nct_peak_bisect(slope, s_lo.ln(), 0.0);
    let s = us.exp();
    let z = d + tau * s;
    let m = mills(z);
    let curvature = -2.0 * df * s * s - tau * s * m - (tau * s).powi(2) * m * (m - z);
    let (lmax, sum) = nct_trapezoid(logf, us, trapezoid_step(curvature));
    if sum == 0.0 {
        return 0.0;
    }
    nct_scale_out(d, lmax, sum)
}

/// Integration-by-parts form, conditioning on `Z = −(d + w)`, `w = e^u`, `k = df/(2τ²)`:
/// `W = e^{−d²/2}/√(2π)·∫ exp(u − dw − w²/2)·P(df/2, k·w²) du`.
fn nct_far_tail_ibp(df: f64, d: f64, tau: f64) -> f64 {
    let a = 0.5 * df;
    let k = df / (2.0 * tau * tau);
    if k == 0.0 {
        return 0.0;
    }
    let lower = |x: f64| gamma::gammainc_scalar(a, x, RuntimeMode::Strict).unwrap_or(f64::NAN);
    let logf = |u: f64| {
        let w = u.exp();
        let pv = lower(k * w * w);
        if pv <= 0.0 {
            return f64::NEG_INFINITY;
        }
        u - d * w - 0.5 * w * w + pv.ln()
    };
    // l′(u) = 1 − dw − w² + 2a·R with R = pt(a, x)/P(a, x) ∈ (0, 1].
    let parts = |u: f64| {
        let w = u.exp();
        let x = k * w * w;
        let pv = lower(x);
        let r = if pv > 0.0 {
            gamma::poisson_term(a, x) / pv
        } else {
            1.0
        };
        (1.0 - d * w - w * w + 2.0 * a * r, r, w, x)
    };
    // The roots of w² + dw = 1 and w² + dw = 1 + df bracket the peak.
    let w_lo = 2.0 / (d + (d * d + 4.0).sqrt());
    let w_hi = 2.0 * (1.0 + df) / (d + (d * d + 4.0 * (1.0 + df)).sqrt());
    let us = nct_peak_bisect(|u| parts(u).0, w_lo.ln(), w_hi.ln());
    let (_, r, w, x) = parts(us);
    let curvature = -d * w - 2.0 * w * w + 4.0 * a * r * (a * (1.0 - r) - x);
    let (lmax, sum) = nct_trapezoid(logf, us, trapezoid_step(curvature));
    if sum == 0.0 {
        return 0.0;
    }
    nct_scale_out(d, lmax - NCT_LN_SQRT_2PI, sum)
}

/// The peak of a log-concave integrand: bisection on its decreasing derivative over
/// `[lo, hi]`, where it is positive at `lo` and negative at `hi`, to a width of 1e-10.
fn nct_peak_bisect(slope: impl Fn(f64) -> f64, lo: f64, hi: f64) -> f64 {
    let (mut lo, mut hi) = (lo, hi);
    for _ in 0..200 {
        if hi - lo <= 1e-10 {
            break;
        }
        let mid = 0.5 * (lo + hi);
        if slope(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Trapezoid step from the log-integrand's curvature at the peak: `σ/2.5`, at most 0.1.
fn trapezoid_step(curvature: f64) -> f64 {
    let sigma = 1.0 / (-curvature).max(1e-300).sqrt();
    (sigma / 2.5).min(0.1)
}

/// `h·Σ_{i∈ℤ} exp(logf(peak + i·h) − lmax)` marched out from the peak. Each direction stops when
/// the geometric bound `f·r/(1 − r)` on its remainder is below 1e-17 of the running sum, where
/// `r` is the ratio of the last two nodes and log-concavity makes it non-increasing. Returns
/// `(lmax, h·sum)`.
fn nct_trapezoid(logf: impl Fn(f64) -> f64, peak: f64, h: f64) -> (f64, f64) {
    let lmax = logf(peak);
    if lmax == f64::NEG_INFINITY {
        return (f64::NEG_INFINITY, 0.0);
    }
    if lmax.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    let mut total = 1.0_f64;
    for direction in [1.0, -1.0] {
        let mut previous = lmax;
        for i in 1..=NCT_TRAPEZOID_MAX_NODES {
            let li = logf(peak + direction * i as f64 * h);
            if li == f64::NEG_INFINITY {
                break;
            }
            if li.is_nan() {
                return (f64::NAN, f64::NAN);
            }
            let fi = (li - lmax).exp();
            total += fi;
            let ratio = (li - previous).exp();
            if ratio < 1.0 && fi * ratio / (1.0 - ratio) < 1e-17 * total {
                break;
            }
            previous = li;
        }
    }
    (lmax, h * total)
}

/// `sum·exp(−d²/2 + log_rest)` with `d²` split exactly into `hi + lo`, and the exponent applied
/// in two steps below −700 so a tiny result underflows gradually instead of all at once.
fn nct_scale_out(d: f64, log_rest: f64, sum: f64) -> f64 {
    let hi = d * d;
    let lo = d.mul_add(d, -hi);
    let sum = sum * (-0.5 * lo).exp();
    let exponent = -0.5 * hi + log_rest;
    if exponent < -700.0 {
        sum * (exponent + 700.0).exp() * (-700.0_f64).exp()
    } else {
        sum * exponent.exp()
    }
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
    let half_beta = 0.5 * student_t_beta(v, t);

    if t >= 0.0 { 1.0 - half_beta } else { half_beta }
}

/// `I_x(v/2, 1/2)` at `x = v/(v + t²)`, the two-sided Student t tail `P(|T| > |t|)`, with
/// `1 − x = t²/(v + t²)` passed to the kernel in closed form (frankenscipy-5pnba). For `v ≫ t²`
/// the old `1.0 − x` was all rounding: `x` rounds to 1 from `v = 1e17` at `t = 2`, and
/// `stdtr(1e20, 2)` came out as 0.5. `t = ±∞` is `x = 0`, and `v + t²` overflowing to ∞ with a
/// finite `t` is `x = 0` as well (`I = 0`).
fn student_t_beta(v: f64, t: f64) -> f64 {
    let t2 = t * t;
    let denom = v + t2;
    let x = v / denom;
    let y = t2 / denom;
    if x.is_nan() || y.is_nan() {
        // v + t² = ∞ with t² = ∞ (t = ±∞): x = v/∞ = 0, y = ∞/∞.
        return if t2.is_infinite() && v.is_finite() {
            0.0
        } else {
            f64::NAN
        };
    }
    btdtr_pair(0.5 * v, 0.5, x, y)
}

/// `I_x(a, b)` with `y = 1 − x` passed in, and btdtr's endpoint values `I_0 = 0`, `I_1 = 1`.
fn btdtr_pair(a: f64, b: f64, x: f64, y: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if y <= 0.0 {
        return 1.0;
    }
    betainc_with_complement(a, b, x, y).unwrap_or(f64::NAN)
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
    let half_beta = 0.5 * student_t_beta(v, t);

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

    // v == 1 is the standard Cauchy distribution, whose quantile is
    // tan(π(p − 1/2)) = −cos(πp)/sin(πp). Evaluating it directly keeps full
    // precision in the tails, where the general inverse-beta path below loses
    // ~1e-6 (e.g. stdtrit(1, 1e-6) was off by 2.8e-6 vs scipy).
    if v == 1.0 {
        let pi_p = std::f64::consts::PI * p;
        return -pi_p.cos() / pi_p.sin();
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

/// Vectorized inverse Student's-t CDF `stdtrit(v, p)` over many probabilities
/// `p` for a fixed `v`. SciPy's `stdtrit` ufunc is a single-threaded per-point
/// inverse-beta root-find (~300 ns/pt); this fans the (bit-identical) fsci
/// scalar across cores via the crate's order-preserving parallel map.
#[must_use]
pub fn stdtrit_many(v: f64, p: &[f64]) -> Vec<f64> {
    par_map_indices(p.len(), |i| Ok::<f64, SpecialError>(stdtrit(v, p[i])))
        .expect("stdtrit is infallible")
}

/// Vectorized inverse noncentral-t CDF `nctdtrit(df, nc, p)` over many `p` for
/// fixed `(df, nc)`. SciPy's ufunc is very slow here (~7.8 µs/pt); with the
/// [`illinois_root`] scalar now ~8x faster, the parallel fan flips this from a
/// prior loss to a win. See [`stdtrit_many`].
#[must_use]
pub fn nctdtrit_many(df: f64, nc: f64, p: &[f64]) -> Vec<f64> {
    par_map_indices(p.len(), |i| Ok::<f64, SpecialError>(nctdtrit(df, nc, p[i])))
        .expect("nctdtrit is infallible")
}

/// Vectorized inverse negative-binomial CDF w.r.t. successes, `nbdtrik(y, n, p)`,
/// over many CDF values `y` for fixed `(n, p)`; see [`stdtrit_many`].
#[must_use]
pub fn nbdtrik_many(y: &[f64], n: f64, p: f64) -> Vec<f64> {
    par_map_indices(y.len(), |i| Ok::<f64, SpecialError>(nbdtrik(y[i], n, p)))
        .expect("nbdtrik is infallible")
}

/// Vectorized noncentral-t CDF `nctdtr(df, nc, t)` over many `t` for fixed
/// `(df, nc)`. SciPy's ufunc is single-threaded; the incomplete-beta-recurrence
/// scalar (~0.6-1.2 µs/pt, now at parity-or-faster than SciPy) fans across cores
/// order-preservingly (bit-identical to a serial map). See [`stdtrit_many`].
#[must_use]
pub fn nctdtr_many(df: f64, nc: f64, t: &[f64]) -> Vec<f64> {
    par_map_indices(t.len(), |i| Ok::<f64, SpecialError>(nctdtr(df, nc, t[i])))
        .expect("nctdtr is infallible")
}

/// Vectorized inverse noncentral-F CDF w.r.t. `f`, `ncfdtri(dfn, dfd, nc, p)`,
/// over many `p` for fixed `(dfn, dfd, nc)`. SciPy's ufunc is single-threaded
/// cdflib (~3.6 µs/pt); with the [`illinois_root`] scalar the parallel fan wins.
/// See [`stdtrit_many`].
#[must_use]
pub fn ncfdtri_many(dfn: f64, dfd: f64, nc: f64, p: &[f64]) -> Vec<f64> {
    par_map_indices(p.len(), |i| {
        Ok::<f64, SpecialError>(ncfdtri(dfn, dfd, nc, p[i]))
    })
    .expect("ncfdtri is infallible")
}

/// Vectorized inverse noncentral-F CDF w.r.t. non-centrality,
/// `ncfdtrinc(dfn, dfd, p, f)`, over many `p` for fixed `(dfn, dfd, f)`; see
/// [`ncfdtri_many`].
#[must_use]
pub fn ncfdtrinc_many(dfn: f64, dfd: f64, p: &[f64], f: f64) -> Vec<f64> {
    par_map_indices(p.len(), |i| {
        Ok::<f64, SpecialError>(ncfdtrinc(dfn, dfd, p[i], f))
    })
    .expect("ncfdtrinc is infallible")
}

/// Vectorized inverse noncentral-t CDF w.r.t. non-centrality,
/// `nctdtrinc(df, p, t)`, over many `p` for fixed `(df, t)`; see [`ncfdtri_many`].
#[must_use]
pub fn nctdtrinc_many(df: f64, p: &[f64], t: f64) -> Vec<f64> {
    par_map_indices(p.len(), |i| Ok::<f64, SpecialError>(nctdtrinc(df, p[i], t)))
        .expect("nctdtrinc is infallible")
}

/// Vectorized inverse noncentral-F CDF w.r.t. denominator dof,
/// `ncfdtridfd(dfn, p, nc, f)`, over many `p` for fixed `(dfn, nc, f)`. Each
/// solve is cdflib's bounded search (`cdflib_invert`, then Illinois) over the
/// single-threaded SciPy ufunc; the parallel fan wins. See [`stdtrit_many`].
#[must_use]
pub fn ncfdtridfd_many(dfn: f64, p: &[f64], nc: f64, f: f64) -> Vec<f64> {
    par_map_indices(p.len(), |i| {
        Ok::<f64, SpecialError>(ncfdtridfd(dfn, p[i], nc, f))
    })
    .expect("ncfdtridfd is infallible")
}

/// Vectorized inverse noncentral-F CDF w.r.t. numerator dof,
/// `ncfdtridfn(p, dfd, nc, f)`, over many `p` for fixed `(dfd, nc, f)`; see
/// [`ncfdtridfd_many`].
#[must_use]
pub fn ncfdtridfn_many(p: &[f64], dfd: f64, nc: f64, f: f64) -> Vec<f64> {
    par_map_indices(p.len(), |i| {
        Ok::<f64, SpecialError>(ncfdtridfn(p[i], dfd, nc, f))
    })
    .expect("ncfdtridfn is infallible")
}

/// Vectorized inverse noncentral-t CDF w.r.t. dof, `nctdtridf(p, nc, t)`, over
/// many `p` for fixed `(nc, t)`; see [`ncfdtridfd_many`].
#[must_use]
pub fn nctdtridf_many(p: &[f64], nc: f64, t: f64) -> Vec<f64> {
    par_map_indices(p.len(), |i| Ok::<f64, SpecialError>(nctdtridf(p[i], nc, t)))
        .expect("nctdtridf is infallible")
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

    // stdtr is increasing in df. Bracket two-sidedly from df = 1, then solve with
    // the superlinear Illinois method (~10 evals) instead of the former 240-iteration
    // bisection (each eval is a full stdtr).
    let f1 = stdtr(1.0, t) - p;
    let (mut lo, mut flo) = (1.0_f64, f1);
    let (mut hi, mut fhi) = (1.0_f64, f1);
    while fhi < 0.0 {
        hi *= 2.0;
        if hi >= UPPER_SENTINEL {
            return UPPER_SENTINEL;
        }
        fhi = stdtr(hi, t) - p;
    }
    while flo > 0.0 {
        lo *= 0.5;
        if lo <= MIN_DF_SENTINEL {
            return MIN_DF_SENTINEL;
        }
        flo = stdtr(lo, t) - p;
    }
    // A bracket endpoint can land exactly on the root (illinois excludes endpoints).
    if flo == 0.0 {
        return lo;
    }
    if fhi == 0.0 {
        return hi;
    }

    let df = illinois_root(|m| stdtr(m, t) - p, lo, hi, flo, fhi);
    if df <= 0.0 { MIN_DF_SENTINEL } else { df }
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

    // bdtr(k, n, p) = I(1-p; n-k, k+1) = 1 - betainc(k+1, n-k, p): bratio's complement at the
    // exact p, rather than I at a rounded 1 − p, which loses p's digits when p is small
    // (frankenscipy-5pnba).
    betaincc_with_complement(k + 1.0, n - k, p, 1.0 - p)
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

    // nbdtrc(k, n, p) = 1 - betainc(n, k+1, p): bratio's complement at the exact p
    // (frankenscipy-5pnba).
    betaincc_with_complement(n, k + 1.0, p, 1.0 - p)
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
    // Fast path: when the endpoints finitely bracket the root, solve with the
    // superlinear Illinois method (~12 evals) instead of DISTRIBUTION_INVERSE_ITERS
    // bisection steps — bdtrik/nbdtrik were ~13-15× SciPy losses (50µs, each iter a
    // full bdtr/nbdtr). Fall back to the robust bisection for non-finite / unbracketed
    // endpoints so all edge cases behave exactly as before.
    let (flo, fhi) = (f(lo), f(hi));
    if flo.is_finite() && fhi.is_finite() {
        let (glo, ghi) = (flo - target, fhi - target);
        if glo == 0.0 {
            return lo;
        }
        if ghi == 0.0 {
            return hi;
        }
        if glo < 0.0 && ghi > 0.0 {
            return illinois_root(|m| f(m) - target, lo, hi, glo, ghi);
        }
    }
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
    // Illinois fast path on the sign-flipped residual (f is decreasing, so
    // g = target − f is increasing); robust bisection fallback for edge cases.
    let (flo, fhi) = (f(lo), f(hi));
    if flo.is_finite() && fhi.is_finite() {
        let (glo, ghi) = (target - flo, target - fhi);
        if glo == 0.0 {
            return lo;
        }
        if ghi == 0.0 {
            return hi;
        }
        if glo < 0.0 && ghi > 0.0 {
            return illinois_root(|m| target - f(m), lo, hi, glo, ghi);
        }
    }
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

/// Evaluate `f(0..n)` into a `Vec`, parallel over index chunks for large `n`.
/// Beta-family kernels (incomplete-beta continued fractions) are expensive per element
/// and each index writes its own slot, so chunking across cores and concatenating in
/// index order is bit-identical to `(0..n).map(f).collect()` — including returning the
/// first failing index's error in index order. Used by the array arms below.
fn par_map_indices<T, G>(n: usize, f: G) -> Result<Vec<T>, SpecialError>
where
    T: Send,
    G: Fn(usize) -> Result<T, SpecialError> + Sync,
{
    let nthreads = if n < 256 {
        1
    } else {
        std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1)
            .min(n / 128)
            .max(1)
    };
    if nthreads <= 1 {
        return (0..n).map(&f).collect();
    }
    let chunk = n.div_ceil(nthreads);
    let f = &f;
    let chunk_results: Vec<Result<Vec<T>, SpecialError>> = std::thread::scope(|scope| {
        (0..nthreads)
            .filter_map(|t| {
                let i0 = t * chunk;
                if i0 >= n {
                    return None;
                }
                let i1 = (i0 + chunk).min(n);
                Some(scope.spawn(move || (i0..i1).map(f).collect::<Result<Vec<T>, _>>()))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().expect("beta array worker panicked"))
            .collect()
    });
    let mut out = Vec::with_capacity(n);
    for cr in chunk_results {
        out.extend(cr?);
    }
    Ok(out)
}

/// beta/betaln are moderate ~lgamma-cost kernels; their par_map_indices break-even
/// is ~45-60k (BlackThrush A/B 2026-06-22), far above the raw n/32 gate that
/// over-subscribes ~16 threads onto a sub-microsecond kernel (2.6-6.5x slower at n<=16k).
/// Stay serial below the shared gate. Order-preserving => byte-identical either way.
const BETA_REAL_PAR_MIN: usize = 1 << 16; // beta wins 0.91x@49152, betaln 0.91x@65536

fn par_map_indices_gated<T, G>(n: usize, f: G) -> Result<Vec<T>, SpecialError>
where
    T: Send,
    G: Fn(usize) -> Result<T, SpecialError> + Sync,
{
    if n >= BETA_REAL_PAR_MIN {
        par_map_indices(n, f)
    } else {
        (0..n).map(f).collect()
    }
}

fn map_real_binary<F>(
    function: &'static str,
    a: &SpecialTensor,
    b: &SpecialTensor,
    mode: RuntimeMode,
    kernel: F,
) -> SpecialResult
where
    F: Fn(f64, f64) -> Result<f64, SpecialError> + Sync,
{
    match (a, b) {
        (SpecialTensor::RealScalar(lhs), SpecialTensor::RealScalar(rhs)) => {
            kernel(*lhs, *rhs).map(SpecialTensor::RealScalar)
        }
        (SpecialTensor::RealVec(lhs), SpecialTensor::RealScalar(rhs)) => {
            let rhs = *rhs;
            par_map_indices_gated(lhs.len(), |i| kernel(lhs[i], rhs)).map(SpecialTensor::RealVec)
        }
        (SpecialTensor::RealScalar(lhs), SpecialTensor::RealVec(rhs)) => {
            let lhs = *lhs;
            par_map_indices_gated(rhs.len(), |i| kernel(lhs, rhs[i])).map(SpecialTensor::RealVec)
        }
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
            par_map_indices_gated(lhs.len(), |i| kernel(lhs[i], rhs[i])).map(SpecialTensor::RealVec)
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
    F: Fn(f64, f64, f64) -> Result<f64, SpecialError> + Sync,
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
        ) => {
            let (bv, cv) = (*bv, *cv);
            par_map_indices(av.len(), |i| kernel(av[i], bv, cv)).map(SpecialTensor::RealVec)
        }
        (
            SpecialTensor::RealScalar(av),
            SpecialTensor::RealVec(bv),
            SpecialTensor::RealScalar(cv),
        ) => {
            let (av, cv) = (*av, *cv);
            par_map_indices(bv.len(), |i| kernel(av, bv[i], cv)).map(SpecialTensor::RealVec)
        }
        (
            SpecialTensor::RealScalar(av),
            SpecialTensor::RealScalar(bv),
            SpecialTensor::RealVec(cv),
        ) => {
            let (av, bv) = (*av, *bv);
            par_map_indices(cv.len(), |i| kernel(av, bv, cv[i])).map(SpecialTensor::RealVec)
        }
        (SpecialTensor::ComplexScalar(_), _, _)
        | (SpecialTensor::ComplexVec(_), _, _)
        | (_, SpecialTensor::ComplexScalar(_), _)
        | (_, SpecialTensor::ComplexVec(_), _)
        | (_, _, SpecialTensor::ComplexScalar(_))
        | (_, _, SpecialTensor::ComplexVec(_)) => {
            not_yet_implemented(function, mode, "complex-valued path pending")
        }
        _ => broadcast_real_ternary(function, a, b, c, mode, kernel),
    }
}

/// The remaining real patterns of a ternary ufunc — two or three vectors, with any scalar
/// repeated — broadcast as SciPy's ufuncs do: `betainc(a, b, x)` over three arrays of one
/// length is the ordinary call. Vectors of different lengths, or an `Empty` tensor, are refused.
/// Shared by the ternary mappers of `beta` and `bessel` (frankenscipy-6ln18).
pub(crate) fn broadcast_real_ternary<F>(
    function: &'static str,
    a: &SpecialTensor,
    b: &SpecialTensor,
    c: &SpecialTensor,
    mode: RuntimeMode,
    kernel: F,
) -> SpecialResult
where
    F: Fn(f64, f64, f64) -> Result<f64, SpecialError> + Sync,
{
    let args = [a, b, c];
    let all_real = args
        .iter()
        .all(|t| matches!(t, SpecialTensor::RealScalar(_) | SpecialTensor::RealVec(_)));
    let mut lengths = args.iter().filter_map(|t| match t {
        SpecialTensor::RealVec(values) => Some(values.len()),
        _ => None,
    });
    let len = lengths.next();
    if let Some(n) = len
        && all_real
        && lengths.all(|other| other == n)
    {
        let at = |t: &SpecialTensor, i: usize| match t {
            SpecialTensor::RealVec(values) => values[i],
            SpecialTensor::RealScalar(value) => *value,
            _ => f64::NAN,
        };
        return par_map_indices(n, |i| kernel(at(a, i), at(b, i), at(c, i)))
            .map(SpecialTensor::RealVec);
    }
    let detail = if all_real && len.is_some() {
        "vector inputs must have matching lengths"
    } else {
        "unsupported broadcast pattern for ternary inputs"
    };
    record_special_trace(
        function,
        mode,
        "domain_error",
        "unsupported_broadcast_pattern",
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

/// Sign of Γ(x) on the real line. Γ > 0 for x > 0; for x < 0 it alternates on
/// each interval between consecutive negative integers. (Nonpositive-integer
/// poles are carried by the ±inf log path, so the value here is harmless there.)
fn gamma_sign(x: f64) -> f64 {
    // The two positive arms fold into one condition (frankenscipy-e2ve2); `||`
    // short-circuits exactly as the nested else-if did, so the ceil is still
    // skipped for x > 0.
    if x > 0.0 || (-x).ceil() as i64 % 2 == 0 {
        1.0
    } else {
        -1.0
    }
}

/// scipy/cephes `beta` at a nonpositive-integer argument when the OTHER argument
/// is strictly positive. The straight `Γ(a)Γ(b)/Γ(a+b)` route gives `inf − inf`
/// in the log domain (→ NaN) and the gamma-sign product picks the wrong side of
/// the pole, so handle these directly:
///   * positive integer `m` with the pole at `-k` (`k ≥ m`): the Γ(b) and Γ(a+b)
///     poles cancel, leaving the finite rational `B(m,−k) = (−1)^m (m−1)!(k−m)!/k!`
///     (computed as `(−1)^m (m−1)! / (k·(k−1)···(k−m+1))` to avoid factorial
///     overflow);
///   * otherwise `+inf` (scipy's one-signed value at the simple pole).
///
/// Returns `None` when neither arg is a nonpositive integer paired with a strictly
/// positive other arg (the both-nonpositive-integer cases are cephes-specific and
/// asymmetric — e.g. `beta(-2,-1)=-inf` but `beta(-1,-2)=+inf` — so we leave the
/// existing path to handle them rather than guess).
fn beta_nonpos_integer_special(a: f64, b: f64) -> Option<f64> {
    let is_nonpos_int = |x: f64| x <= 0.0 && x.is_finite() && x.fract() == 0.0;
    let (pole, other) = if is_nonpos_int(a) && b > 0.0 {
        (a, b)
    } else if is_nonpos_int(b) && a > 0.0 {
        (b, a)
    } else {
        return None;
    };
    let k = (-pole) as i64; // nonnegative integer magnitude of the pole
    if other.fract() == 0.0 {
        let m = other as i64; // positive integer
        if m >= 1 && m <= k {
            let mut val = if m % 2 == 0 { 1.0 } else { -1.0 };
            for i in 1..m {
                val *= i as f64; // (m-1)!
            }
            for j in 0..m {
                val /= (k - j) as f64; // k·(k-1)···(k-m+1)
            }
            return Some(val);
        }
    }
    Some(f64::INFINITY)
}

/// Largest argument for which `Γ` is finite; Cephes' `MAXGAM`. Beyond it `beta` must use
/// the logarithmic form because the direct product overflows.
const BETA_MAXGAM: f64 = 171.624_376_956_302_725;

/// Form `B(a,b)` from `Γ(a)·Γ(b)·(1/Γ(a+b))` directly for positive arguments (`true`,
/// shipping) instead of `exp(betaln(a,b))`.
///
/// NOT bit-identical to the log form and not intended to be: it is a different route, and
/// it is the incumbent's. Accuracy against the live SciPy arm is the contract.
pub static BETA_CEPHES_DIRECT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);
/// Evaluations that took the direct arm — "enabled" is not "took effect".
///
/// Accuracy contract: a diagnostic counter, not an A/B lever. It has no arms to preserve
/// anything between; incrementing it changes no numeric result.
pub static BETA_CEPHES_DIRECT_HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

pub(crate) fn beta_scalar(a: f64, b: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if let Some(v) = beta_nonpos_integer_special(a, b) {
        return Ok(v);
    }
    // Symmetry beta(a,b)=beta(b,a)
    let (a, b) = if a < b { (b, a) } else { (a, b) };

    // ── direct-Gamma fast path ───────────────────────────────────────────────────────
    //
    // WHY. `beta` went through `betaln` unconditionally — three `lgamma` calls and an
    // `exp` — where SciPy forms `Γ(a)·Γ(b)/Γ(a+b)` directly from the fast rational and only
    // falls back to logs past MAXGAM. Measured against the live SciPy arm on the identical
    // fixture, instructions per element: beta 1104.1 against 346.4, a 3.19x gap and the
    // worst cell in this crate — found only because the survey was widened to TWO-argument
    // ufuncs, which had never been measured at all.
    //
    // SCOPED TO POSITIVE ARGUMENTS ON PURPOSE. Every Γ factor is then positive, so the sign
    // bookkeeping below is unnecessary here rather than merely unused, and the Hardened
    // overflow contract stays entirely on the log path. Anything else — negative, huge, or
    // a value that does not come out finite — falls through untouched.
    if BETA_CEPHES_DIRECT.load(std::sync::atomic::Ordering::Relaxed)
        && a > 0.0
        && b > 0.0
        && a <= BETA_MAXGAM
        && b <= BETA_MAXGAM
        && a + b <= BETA_MAXGAM
    {
        let inv_sum = gamma::rgamma_value(a + b, mode);
        let ga = gamma::gamma_core(a);
        let gb = gamma::gamma_core(b);
        if inv_sum.is_finite() && ga.is_finite() && gb.is_finite() {
            // Cephes' multiply order: pair the factor whose product with 1/Γ(a+b) sits
            // closest to 1 first, which keeps the intermediate away from the exponent
            // range's edges. Reproducing it is what makes this agree to the bit.
            let value = if ((ga * inv_sum).abs() - 1.0).abs() > ((gb * inv_sum).abs() - 1.0).abs() {
                (gb * inv_sum) * ga
            } else {
                (ga * inv_sum) * gb
            };
            if value.is_finite() {
                BETA_CEPHES_DIRECT_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(value);
            }
        }
    }

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
    // Nonpositive-integer argument with a positive partner: the gammaln sum below
    // is inf − inf = NaN, so use the closed-form beta value and take its log.
    // frankenscipy-dwd3d
    if let Some(v) = beta_nonpos_integer_special(a, b) {
        return Ok(if v.is_infinite() {
            f64::INFINITY
        } else {
            v.abs().ln()
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

    betainc_with_complement(a, b, x, 1.0 - x)
}

/// `I_x(a, b)` for `a, b > 0` and `x ∈ (0, 1)`, with the complement `y = 1 − x` passed in.
///
/// [`betainc_scalar`] passes `1.0 - x`. A caller that has `1 − x` in closed form passes that
/// instead (frankenscipy-g9yid): nctdtr's `x = t²/(t²+df)` rounds to within `ε` of 1 at large
/// `t`, so `1.0 - x` carries a relative error of `ε·t²/df`, 1e-10 at `t = 1682`. nctdtr at the
/// mean then missed mpmath by 2e-11 at λ = 1e6, 2e-9 at 1e8 and 3e-7 at 1e10; with
/// `y = df/(t²+df)` it misses by 1e-14, 1e-13 and 1e-12. `x + y` must be 1 to within a few
/// roundings (`3·10⁻¹⁵`), or the result is NaN.
///
/// The kernel is TOMS 708's `bratio` (frankenscipy-5pnba; see `crate::bratio`). It replaced a
/// Numerical Recipes continued fraction with a 200-term cap, which truncated silently near the
/// mean once a shape parameter was large: `stdtr(1e15, 2)` was off by 8e-4, and
/// `betainc(2.5, 1e20, 5e-20)` = −4.9e281.
pub(crate) fn betainc_with_complement(a: f64, b: f64, x: f64, y: f64) -> Result<f64, SpecialError> {
    Ok(crate::bratio::bratio(a, b, x, y).0)
}

/// `1 − I_x(a, b)` for `a, b > 0` and `x ∈ (0, 1)`, with `y = 1 − x` passed in: `bratio`'s own
/// complement, computed directly rather than by subtracting from 1, so an upper tail keeps its
/// digits all the way down (frankenscipy-5pnba).
pub(crate) fn betaincc_with_complement(a: f64, b: f64, x: f64, y: f64) -> f64 {
    crate::bratio::bratio(a, b, x, y).1
}

/// Below this, `log_betainc_scalar` stops taking the log of `bratio`'s `I` and sums the tail in
/// log space: `I` is near the bottom of the normal range, where `bratio`'s own intermediate
/// factors start to underflow.
const LOG_BETAINC_LOG_SPACE_BELOW: f64 = 1e-290;

/// Natural log of the regularized incomplete beta function `I_x(a, b)`.
///
/// `ln I_x(a, b)` stays finite deep in the tail where `I` itself underflows to
/// 0 (so `betainc_scalar(a, b, x).ln()` would be `-inf`). Where `I` is representable it is
/// `ln` of the `bratio` kernel's `I`, or `ln1p(−(1 − I))` from its directly computed complement
/// when `I > ½`. Below [`LOG_BETAINC_LOG_SPACE_BELOW`] (the small-`I` region, `x` below
/// `(a+1)/(a+b+2)`) the tail is summed in log space (frankenscipy-5pnba): by TOMS 708's own
/// far-tail expansions in log form (`BGRAT` for `a ≥ 15, b ≤ 1`, `BRCOMP·BFRAC` for `a, b > 1`;
/// `crate::bratio::ln_bratio_lower_tail`), and otherwise as
/// `ln(xᵃ(1−x)ᵇ/B(a,b)) + ln(betacf(a,b,x)/a)` with TOMS's front factor in log form
/// (`crate::bratio::ln_brcomp`). Both replace `a·ln x + b·ln(1−x) − ln B(a,b)`, which cancels
/// to `ε·a·ln a` once the parameters are large.
///
/// For the complementary log use the reflection `ln(1 - I_x(a,b)) =
/// log_betainc_scalar(b, a, 1 - x)`.
///
/// Domain: `a > 0`, `b > 0`, `x in [0, 1]`. Returns `NaN` for invalid inputs,
/// `-inf` at `x = 0` (`I = 0`), and `0.0` at `x = 1` (`I = 1`).
#[must_use]
pub fn log_betainc_scalar(a: f64, b: f64, x: f64) -> f64 {
    if a.is_nan() || b.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if !(0.0..=1.0).contains(&x) || a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x == 1.0 {
        return 0.0;
    }

    let y = 1.0 - x;
    let (w, w1) = crate::bratio::bratio(a, b, x, y);
    if w1 < 0.5 {
        return (-w1).ln_1p();
    }
    if w >= LOG_BETAINC_LOG_SPACE_BELOW || !(x < (a + 1.0) / (a + b + 2.0)) {
        return w.ln();
    }
    if let Some(ln_i) = crate::bratio::ln_bratio_lower_tail(a, b, x, y) {
        return ln_i;
    }
    match betacf(a, b, x) {
        Some(cf) => crate::bratio::ln_brcomp(a, b, x, y) + (cf / a).ln(),
        None => f64::NAN,
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

    // bratio's own complement (frankenscipy-5pnba), not I_{1−x}(b, a) through a rounded 1 − x.
    Ok(betaincc_with_complement(a, b, x, 1.0 - x))
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

/// Lentz's continued fraction for `I_x(a, b)·a·B(a, b)/(xᵃ(1−x)ᵇ)` (Numerical Recipes'
/// `betacf`), for `x` below `(a+1)/(a+b+2)`. Only [`log_betainc_scalar`]'s deep tail uses it,
/// and only where neither TOMS far-tail expansion applies (`a ≤ 1`, or `b ≤ 1` with `a < 15`),
/// where the tail is at tiny `x` and the fraction converges in a few terms (frankenscipy-5pnba).
/// Near the mean it needs O(√a) terms, and it was once the whole betainc kernel there and
/// truncated silently at the cap; now a fraction that has not converged by the cap is `None`.
fn betacf(a: f64, b: f64, x: f64) -> Option<f64> {
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
            return Some(h);
        }
    }

    None
}

fn invert_monotone_positive(cdf: impl Fn(f64) -> f64, target: f64, increasing: bool) -> f64 {
    let mut lo = f64::MIN_POSITIVE;
    let mut hi = 1.0;
    let mut hi_value = cdf(hi);

    if !hi_value.is_finite() {
        return f64::NAN;
    }

    // cdf(lo). It is only known once the doubling loop runs (then it equals the
    // last in-bracket hi_value, which was already checked finite). NaN is the
    // "not yet computed" sentinel.
    let mut lo_value = f64::NAN;

    if increasing {
        while hi_value < target {
            lo = hi;
            lo_value = hi_value;
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
            lo_value = hi_value;
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

    // The doubling loop never ran, so cdf(lo = MIN_POSITIVE) is still unknown.
    if lo_value.is_nan() {
        lo_value = cdf(lo);
        if !lo_value.is_finite() {
            return f64::NAN;
        }
    }

    // Reduce to a monotone-INCREASING root problem g(x) = 0 with g(lo) < 0 < g(hi),
    // then solve with the superlinear Illinois method (~10-15 evals) instead of
    // ~40 plain-bisection steps. `cdf` here is btdtr's continued fraction, whose
    // cost does not vary with the probe point, so fewer evaluations is a clean
    // speed win — and the root is returned to ~4·eps (tighter than the former
    // 1e-12 bracket).
    let (glo, ghi) = if increasing {
        (lo_value - target, hi_value - target)
    } else {
        (target - lo_value, target - hi_value)
    };
    illinois_root(
        |x| {
            if increasing {
                cdf(x) - target
            } else {
                target - cdf(x)
            }
        },
        lo,
        hi,
        glo,
        ghi,
    )
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

    /// The direct-Gamma product agrees with the `exp(betaln)` route it replaces, and the arm
    /// actually fires.
    ///
    /// The two share no arithmetic — one multiplies three Γ values, the other exponentiates
    /// a sum of three log-Γ values — so agreement is real evidence. The bound is 1e-13
    /// relative because the LOG route is the less accurate side: against live SciPy the
    /// direct arm reads exactly 0 where `exp(betaln)` reads 2.006e-14.
    #[test]
    fn beta_direct_gamma_matches_the_log_route() {
        use std::sync::atomic::Ordering::Relaxed;
        static BETA_TOGGLE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = BETA_TOGGLE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let restore = BETA_CEPHES_DIRECT.load(Relaxed);

        let mut worst = 0.0_f64;
        let mut worst_at = (f64::NAN, f64::NAN);
        let mut hits_seen = 0usize;
        let mut compared = 0usize;
        for i in 0..120 {
            for j in 0..120 {
                let a = 0.25 + i as f64 * 0.12;
                let b = 0.25 + j as f64 * 0.12;
                BETA_CEPHES_DIRECT.store(true, Relaxed);
                let before = BETA_CEPHES_DIRECT_HITS.load(Relaxed);
                let direct = beta_scalar(a, b, RuntimeMode::Strict).expect("beta direct");
                if BETA_CEPHES_DIRECT_HITS.load(Relaxed) > before {
                    hits_seen += 1;
                }
                BETA_CEPHES_DIRECT.store(false, Relaxed);
                let logged = beta_scalar(a, b, RuntimeMode::Strict).expect("beta log route");
                assert!(
                    direct.is_finite() && logged.is_finite(),
                    "non-finite at a={a} b={b}: direct {direct} logged {logged}"
                );
                let rel = (direct - logged).abs() / logged.abs();
                if rel > worst {
                    worst = rel;
                    worst_at = (a, b);
                }
                compared += 1;
            }
        }
        BETA_CEPHES_DIRECT.store(restore, Relaxed);

        // MUST-HIT: the direct arm has to have run, or every comparison above was between
        // two identical evaluations.
        assert!(
            hits_seen >= compared,
            "direct arm ran {hits_seen} of {compared} times; the comparison is vacuous"
        );
        assert!(
            worst < 1.0e-13,
            "direct and log routes disagree by {worst:e} at a={} b={}",
            worst_at.0,
            worst_at.1
        );

        // The fast path is restricted to positive arguments; negatives must still reach the
        // log route and keep their sign. scipy.special.beta(-2.5, 3.0) = -1.0666666...
        let neg = beta_scalar(-2.5, 3.0, RuntimeMode::Strict).expect("negative beta");
        assert!(
            (neg + 1.066_666_666_666_666_7).abs() < 1.0e-12,
            "beta(-2.5, 3.0) should be about -1.0666667, got {neg}"
        );
    }

    #[test]
    fn stdtrit_v1_is_exact_cauchy_in_the_tails() {
        // v == 1 is the standard Cauchy. Verify via the independent round-trip
        // through the Cauchy CDF F(t) = 1/2 + atan(t)/π, which must recover p.
        // Regression: the general inverse-beta path lost ~3e-6 at p = 1e-6.
        for &p in &[
            1e-8_f64, 1e-6, 1e-3, 0.01, 0.1, 0.25, 0.75, 0.9, 0.99, 0.999_999,
        ] {
            let t = stdtrit(1.0, p);
            let cdf = 0.5 + t.atan() / std::f64::consts::PI;
            assert!(
                (cdf - p).abs() < 1e-12,
                "stdtrit(1, {p}) = {t}: CDF round-trip {cdf} != {p}"
            );
        }
    }

    type DistributionInverseCase = (fn(f64, f64, f64) -> f64, f64, f64, f64, f64);

    #[test]
    fn log_betainc_matches_betainc_where_representable() {
        // exp(log I) == I wherever I is representable (betainc_scalar is the
        // scipy-validated reference). Covers both branches and the symmetry.
        let cases = [
            (0.5, 0.5, 0.2),
            (2.0, 3.0, 0.25),
            (2.0, 3.0, 0.75),
            (5.0, 1.0, 0.5),
            (1.0, 5.0, 0.5),
            (10.0, 10.0, 0.3),
            (0.5, 2.0, 0.001),
        ];
        for (a, b, x) in cases {
            let i = betainc_scalar(a, b, x, RuntimeMode::Strict).unwrap();
            let li = log_betainc_scalar(a, b, x);
            assert!(i > 0.0, "precondition: I representable for ({a},{b},{x})");
            let rel = (li - i.ln()).abs() / i.ln().abs().max(1.0);
            assert!(
                rel <= 1e-11,
                "log_betainc({a},{b},{x})={li} vs ln(I)={}",
                i.ln()
            );
        }
    }

    #[test]
    fn log_betainc_finite_in_underflowed_tail() {
        // Deep tail: I underflows to 0 (ln I < ~-745) but log I stays finite and
        // matches the leading a*ln x - ln(a) - lnB(a,b) for tiny x.
        for (a, b, x) in [(2.0, 3.0, 1e-170), (5.0, 2.0, 1e-70), (3.0, 3.0, 1e-130)] {
            assert_eq!(
                betainc_scalar(a, b, x, RuntimeMode::Strict).unwrap(),
                0.0,
                "precondition: I underflows for ({a},{b},{x})"
            );
            let li = log_betainc_scalar(a, b, x);
            assert!(
                li.is_finite() && li < -100.0,
                "log I({a},{b},{x})={li} not finite"
            );
            // I ≈ x^a/(a·B(a,b)) for tiny x ⇒ ln I ≈ a·ln x − ln a − lnB(a,b).
            let asymp = a * x.ln() - a.ln() - betaln_scalar(a, b, RuntimeMode::Strict).unwrap();
            assert!(
                (li - asymp).abs() / asymp.abs() < 1e-2,
                "log I({a},{b},{x})={li} vs asymptotic {asymp}"
            );
        }
    }

    #[test]
    fn log_betainc_edge_and_reflection() {
        assert_eq!(log_betainc_scalar(2.0, 3.0, 0.0), f64::NEG_INFINITY);
        assert_eq!(log_betainc_scalar(2.0, 3.0, 1.0), 0.0);
        assert!(log_betainc_scalar(-1.0, 2.0, 0.5).is_nan());
        assert!(log_betainc_scalar(2.0, 3.0, 1.5).is_nan());
        // ln(1 - I_x(a,b)) == log_betainc(b, a, 1-x).
        for (a, b, x) in [(2.0, 3.0, 0.7), (5.0, 1.5, 0.9)] {
            let icc = betaincc_scalar(a, b, x, RuntimeMode::Strict).unwrap();
            let lcc = log_betainc_scalar(b, a, 1.0 - x);
            assert!((lcc - icc.ln()).abs() / icc.ln().abs().max(1.0) <= 1e-11);
        }
    }

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

    /// SciPy's ufuncs broadcast `betainc(a, b, x)` over arrays of one length, with scalars
    /// repeated. The ternary mapper used to accept one vector only and refused
    /// `betainc(a_vec, b_vec, x_vec)` as an "unsupported broadcast pattern".
    #[test]
    fn ternary_ufuncs_broadcast_several_vectors() {
        let a = vec![0.5, 2.0, 10.0, 30.0];
        let b = vec![200.0, 2.0, 0.5, 30.0];
        let x = vec![0.001, 0.5, 0.97, 0.45];
        let (ta, tb, tx) = (
            SpecialTensor::RealVec(a.clone()),
            SpecialTensor::RealVec(b.clone()),
            SpecialTensor::RealVec(x.clone()),
        );
        let expect = |i: usize, bi: f64| {
            betainc_scalar(a[i], bi, x[i], RuntimeMode::Strict).expect("scalar betainc")
        };
        let vector = |t: SpecialTensor| match t {
            SpecialTensor::RealVec(values) => values,
            _ => Vec::new(),
        };
        let all = vector(betainc(&ta, &tb, &tx, RuntimeMode::Strict).expect("three vectors"));
        assert_eq!(all.len(), 4);
        for (i, got) in all.iter().enumerate() {
            assert_eq!(got.to_bits(), expect(i, b[i]).to_bits(), "element {i}");
        }
        let b_scalar = SpecialTensor::RealScalar(3.0);
        let mixed =
            vector(betainc(&ta, &b_scalar, &tx, RuntimeMode::Strict).expect("vec, scalar, vec"));
        assert_eq!(mixed.len(), 4);
        for (i, got) in mixed.iter().enumerate() {
            assert_eq!(got.to_bits(), expect(i, 3.0).to_bits(), "element {i}");
        }
        let short = SpecialTensor::RealVec(vec![0.5, 0.5]);
        let err = betainc(&ta, &tb, &short, RuntimeMode::Strict).expect_err("mismatched lengths");
        assert_eq!(err.kind, SpecialErrorKind::DomainError);
        assert_eq!(err.detail, "vector inputs must have matching lengths");
        assert!(betainc(&ta, &tb, &SpecialTensor::Empty, RuntimeMode::Strict).is_err());
        // The same mapper serves betaincc, and bessel's wright_bessel shares the helper.
        let comp = vector(betaincc(&ta, &tb, &tx, RuntimeMode::Strict).expect("betaincc vectors"));
        assert_eq!(comp.len(), 4);
        let wb = crate::bessel::wright_bessel(&ta, &tb, &tx, RuntimeMode::Strict)
            .expect("wright_bessel vectors");
        assert!(matches!(wb, SpecialTensor::RealVec(ref v) if v.len() == 4));
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
    fn beta_nonpositive_integer_args_match_scipy() {
        // Regression (frankenscipy-dwd3d): for a nonpositive-integer argument with
        // a positive partner the Γ poles either cancel to a finite rational
        // (fsci previously got NaN from inf−inf) or give scipy's +inf (fsci
        // previously got −inf from the wrong gamma-sign side). Values from
        // scipy.special.beta / betaln 1.17.1.
        let beta_cases = [
            (2.0_f64, -1.0_f64, f64::INFINITY),
            (-1.0, 2.0, f64::INFINITY),
            (3.0, -2.0, f64::INFINITY),
            (0.5, -1.0, f64::INFINITY),
            (0.0, 2.0, f64::INFINITY),
            (2.0, -2.0, 0.5),
            (2.0, -3.0, 1.0 / 6.0),
            (3.0, -3.0, -1.0 / 3.0),
            (1.0, -1.0, -1.0),
            (1.0, -4.0, -0.25),
        ];
        for (a, b, want) in beta_cases {
            let got = beta_scalar(a, b, RuntimeMode::Strict).unwrap();
            if want.is_infinite() {
                assert!(
                    got.is_infinite() && got.is_sign_positive(),
                    "beta({a},{b}) = {got}, want +inf"
                );
            } else {
                assert!(
                    (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                    "beta({a},{b}) = {got}, want {want}"
                );
            }
        }
        // betaln tracks ln|beta|: betaln(2,-2)=ln(0.5), betaln(2,-1)=+inf.
        assert!(
            (betaln_scalar(2.0, -2.0, RuntimeMode::Strict).unwrap() - (0.5_f64).ln()).abs() < 1e-12
        );
        assert!(
            betaln_scalar(2.0, -1.0, RuntimeMode::Strict)
                .unwrap()
                .is_infinite()
        );
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
    fn nctdtr_matches_scipy_reference_values() {
        // frankenscipy: non-central t CDF was missing. Golden values from
        // scipy.special.nctdtr(df, nc, t) 1.17.1.
        let cases = [
            (10.0_f64, 2.0, 1.5, 0.3047854473760421_f64),
            (5.0, 0.0, 1.0, 0.8183912661754386), // nc=0 → central t
            (20.0, -3.0, -2.0, 0.8358989270421169), // negative nc and t (reflection)
            (8.0, 5.0, 4.0, 0.21027058165197615),
            (15.0, 1.0, 0.0, 0.15865525393145707), // t=0 → Phi(-nc)
        ];
        for (df, nc, t, want) in cases {
            let got = nctdtr(df, nc, t);
            assert!(
                (got - want).abs() <= 1e-10 * want.abs().max(1e-12),
                "nctdtr({df},{nc},{t}) = {got}, expected {want}"
            );
        }
    }

    #[test]
    fn noncentral_quantile_inverses_match_scipy() {
        // frankenscipy: golden from scipy.special.ncfdtri / nctdtrit 1.17.1.
        let nc_f = [
            (5.0_f64, 10.0, 3.0, 0.5, 1.5254092911626238_f64),
            (2.0, 4.0, 0.0, 0.7, 1.6514837167011074),
            (10.0, 20.0, 40.0, 0.3, 4.058092482878802),
        ];
        for (dfn, dfd, nc, p, want) in nc_f {
            let got = ncfdtri(dfn, dfd, nc, p);
            assert!(
                (got - want).abs() <= 1e-9 * want.abs(),
                "ncfdtri = {got}, want {want}"
            );
        }
        let nc_t = [
            (10.0_f64, 2.0, 0.3, 1.4856759815279506_f64),
            (20.0, -3.0, 0.8, -2.138962456180749),
            (5.0, 0.0, 0.5, 0.0),
        ];
        for (df, nc, p, want) in nc_t {
            let got = nctdtrit(df, nc, p);
            assert!(
                (got - want).abs() <= 1e-8 * want.abs().max(1e-6),
                "nctdtrit = {got}, want {want}"
            );
        }
        assert_eq!(ncfdtri(5.0, 10.0, 3.0, 0.0), 0.0);
        assert!(ncfdtri(5.0, 10.0, 3.0, 1.0).is_infinite());
    }

    #[test]
    fn ncfdtr_matches_scipy_reference_values() {
        // frankenscipy: non-central F CDF was missing. Golden values from
        // scipy.special.ncfdtr(dfn, dfd, nc, f) 1.17.1.
        let cases = [
            (5.0_f64, 10.0, 3.0, 2.0, 0.6391470579975839_f64),
            (2.0, 4.0, 0.0, 1.5, 0.673469387755102), // nc=0 → central F
            (10.0, 20.0, 40.0, 3.0, 0.10716411882720595), // large nc
            (3.0, 3.0, 5.0, 0.5, 0.0625950844013485),
        ];
        for (dfn, dfd, nc, f, want) in cases {
            let got = ncfdtr(dfn, dfd, nc, f);
            assert!(
                (got - want).abs() <= 1e-10 * want.abs().max(1e-12),
                "ncfdtr({dfn},{dfd},{nc},{f}) = {got}, expected {want}"
            );
        }
        assert_eq!(ncfdtr(5.0, 10.0, 3.0, 0.0), 0.0);
    }

    #[test]
    fn nc_inverse_param_matches_scipy() {
        // frankenscipy-lmffs (partial): invert the non-central F/t CDF on nc.
        // Golden from scipy.special.ncfdtrinc/nctdtrinc 1.17.1. ncfdtrinc agrees
        // only to scipy's DINVR tolerance (~1e-5); nctdtrinc to ~1e-8.
        let nc = ncfdtrinc(5.0, 10.0, 0.7, 2.0);
        assert!(
            (nc - 2.062_739_479_639_603).abs() < 1e-4,
            "ncfdtrinc = {nc}"
        );
        // Round-trip is tight regardless of scipy's tolerance.
        assert!((ncfdtr(5.0, 10.0, nc, 2.0) - 0.7).abs() < 1e-9);
        // Target above the central CDF has no nc >= 0 solution → 0.
        assert_eq!(ncfdtrinc(2.0, 5.0, 0.9, 0.8), 0.0);

        let nct = nctdtrinc(10.0, 0.7, 1.5);
        assert!(
            (nct - 0.909_707_486_336_509_2).abs() < 1e-7,
            "nctdtrinc = {nct}"
        );
        assert!((nctdtr(10.0, nct, 1.5) - 0.7).abs() < 1e-9);
    }

    #[test]
    fn ncf_nct_df_inverses_round_trip() {
        // ncfdtridfd and nctdtridf are monotone in their parameter, so inverting a CDF value
        // recovers the exact df that produced it. (These solve the inverse correctly even on the
        // inputs where scipy's cdflib DINVR fails and returns its 1e100 bound.)
        for &(dfn, dfd, nc, f) in &[
            (5.0, 10.0, 1.0, 2.0),
            (3.0, 20.0, 0.0, 1.5),
            (8.0, 6.0, 3.0, 1.2),
            (4.0, 30.0, 2.0, 0.9),
            (10.0, 15.0, 5.0, 1.8),
        ] {
            let p = ncfdtr(dfn, dfd, nc, f);
            let solved = ncfdtridfd(dfn, p, nc, f);
            assert!(
                (solved - dfd).abs() < 1e-6 * dfd,
                "ncfdtridfd recovered {solved}, expected {dfd}"
            );
            assert!((ncfdtr(dfn, solved, nc, f) - p).abs() < 1e-9);
        }
        for &(df, nc, t) in &[
            (10.0, 0.7, 1.5),
            (5.0, 0.0, 1.2),
            (20.0, 2.0, 2.5),
            (8.0, 1.5, 0.8),
        ] {
            let p = nctdtr(df, nc, t);
            let solved = nctdtridf(p, nc, t);
            assert!(
                (solved - df).abs() < 1e-5 * df,
                "nctdtridf recovered {solved}, expected {df}"
            );
            assert!((nctdtr(solved, nc, t) - p).abs() < 1e-9);
        }
        // ncfdtridfn: the noncentral-F CDF is unimodal in the numerator df, so the inverse may
        // return the larger of two roots (as scipy does) — assert it returns a *valid* root.
        for &(dfn, dfd, nc, f) in &[(5.0, 10.0, 1.0, 2.0), (8.0, 6.0, 3.0, 1.2)] {
            let p = ncfdtr(dfn, dfd, nc, f);
            let solved = ncfdtridfn(p, dfd, nc, f);
            assert!(
                (ncfdtr(solved, dfd, nc, f) - p).abs() < 1e-9,
                "ncfdtridfn root invalid"
            );
        }
        // Boundary / invalid argument handling.
        assert!(ncfdtridfd(5.0, f64::NAN, 1.0, 2.0).is_nan());
        assert!(ncfdtridfd(-1.0, 0.5, 1.0, 2.0).is_nan());
        assert!(nctdtridf(1.5, 0.0, 1.0).is_nan()); // p out of [0,1]
    }

    /// frankenscipy-g9yid. The five cdflib-backed inverses return SciPy's search bounds when
    /// the root lies outside cdflib's range, instead of searching without limit. Every expected
    /// value is SciPy 1.17.1 read live. Ranges and bounds (scipy/special/cdflib.c):
    ///
    /// ```text
    /// ncfdtrinc   cdffnc_which5  nc  ∈ [0, 1e4]        below → 0       above → 1e4
    /// nctdtrinc   cdftnc_which4  nc  ∈ [−1e6, 1e6]     below → 0       above → 1e6
    /// nctdtridf   cdftnc_which3  df  ∈ [1e-100, 1e10]  below → −1e100  above → 1e100
    /// ncfdtridfd  cdffnc_which4  dfd ∈ [1e-100, 1e100] below → 1e-100  above → 1e100
    /// ncfdtridfn  cdffnc_which3  dfn ∈ [1e-100, 1e100] below → 1e-100  above → 1e100
    /// ```
    ///
    /// The first row of each group is one the old unbounded search got wrong: 2610874.47,
    /// −inf, 1e-8, 1e-8 and 1.35e10 respectively.
    #[test]
    fn cdflib_inverses_return_scipys_search_bounds() {
        let nan = f64::NAN;
        let inf = f64::INFINITY;
        let p1 = 1.0 - f64::EPSILON / 2.0;
        let rows: Vec<(&str, f64, f64)> = vec![
            (
                "ncfdtrinc(3, 5, 0.5, 1e6)",
                ncfdtrinc(3.0, 5.0, 0.5, 1e6),
                1e4,
            ),
            (
                "ncfdtrinc(3, 5, 0.5, inf)",
                ncfdtrinc(3.0, 5.0, 0.5, inf),
                1e4,
            ),
            (
                "ncfdtrinc(3, 5, 0, 100)",
                ncfdtrinc(3.0, 5.0, 0.0, 100.0),
                1e4,
            ),
            (
                "ncfdtrinc(3, 5, 0.9, 1)",
                ncfdtrinc(3.0, 5.0, 0.9, 1.0),
                0.0,
            ),
            (
                "ncfdtrinc(3, 5, 0.5, 0)",
                ncfdtrinc(3.0, 5.0, 0.5, 0.0),
                0.0,
            ),
            (
                "ncfdtrinc(3, 5, 1 - 2^-53, 2)",
                ncfdtrinc(3.0, 5.0, p1, 2.0),
                0.0,
            ),
            ("ncfdtrinc(3, 5, 1, 2)", ncfdtrinc(3.0, 5.0, 1.0, 2.0), nan),
            ("ncfdtrinc(3, 5, 0, 0)", ncfdtrinc(3.0, 5.0, 0.0, 0.0), 5.0),
            ("nctdtrinc(5, 0.5, -inf)", nctdtrinc(5.0, 0.5, -inf), 0.0),
            ("nctdtrinc(5, 0.5, inf)", nctdtrinc(5.0, 0.5, inf), 1e6),
            ("nctdtrinc(5, 0.5, 1e300)", nctdtrinc(5.0, 0.5, 1e300), 1e6),
            ("nctdtrinc(5, 1, 2)", nctdtrinc(5.0, 1.0, 2.0), nan),
            ("nctdtridf(0, 1, 1.5)", nctdtridf(0.0, 1.0, 1.5), -1e100),
            ("nctdtridf(0.5, 1, -inf)", nctdtridf(0.5, 1.0, -inf), -1e100),
            ("nctdtridf(0.5, 1e6, 1.5)", nctdtridf(0.5, 1e6, 1.5), -1e100),
            ("nctdtridf(0.5, -5, 1.5)", nctdtridf(0.5, -5.0, 1.5), -1e100),
            ("nctdtridf(0.5, 1, inf)", nctdtridf(0.5, 1.0, inf), 1e100),
            ("nctdtridf(0.9, 1, 1.5)", nctdtridf(0.9, 1.0, 1.5), 1e100),
            (
                "nctdtridf(1 - 2^-53, 1, 1.5)",
                nctdtridf(p1, 1.0, 1.5),
                1e100,
            ),
            (
                "nctdtridf(0.5, 1, 1e300)",
                nctdtridf(0.5, 1.0, 1e300),
                1e100,
            ),
            (
                "nctdtridf(0.5, 1000000.1, 1.5)",
                nctdtridf(0.5, 1_000_000.1, 1.5),
                nan,
            ),
            ("nctdtridf(1, 1, 1.5)", nctdtridf(1.0, 1.0, 1.5), nan),
            (
                "ncfdtridfd(3, 0.5, 2, inf)",
                ncfdtridfd(3.0, 0.5, 2.0, inf),
                1e100,
            ),
            (
                "ncfdtridfd(5, 0.9, 1, 0.5)",
                ncfdtridfd(5.0, 0.9, 1.0, 0.5),
                1e100,
            ),
            (
                "ncfdtridfd(3, 0.5, 2, 0)",
                ncfdtridfd(3.0, 0.5, 2.0, 0.0),
                1e-100,
            ),
            (
                "ncfdtridfd(3, 0, 2, 2)",
                ncfdtridfd(3.0, 0.0, 2.0, 2.0),
                1e-100,
            ),
            (
                "ncfdtridfd(3, 0.5, 4294967296, 2)",
                ncfdtridfd(3.0, 0.5, 4_294_967_296.0, 2.0),
                nan,
            ),
            (
                "ncfdtridfd(3, 0.5, 1e10, 0)",
                ncfdtridfd(3.0, 0.5, 1e10, 0.0),
                1e-100,
            ),
            (
                "ncfdtridfd(3, 1, 2, 2)",
                ncfdtridfd(3.0, 1.0, 2.0, 2.0),
                nan,
            ),
            (
                "ncfdtridfn(0.5, 5, 1e10, 2)",
                ncfdtridfn(0.5, 5.0, 1e10, 2.0),
                nan,
            ),
            (
                "ncfdtridfn(0.5, 5, 2, inf)",
                ncfdtridfn(0.5, 5.0, 2.0, inf),
                1e100,
            ),
            (
                "ncfdtridfn(0.99, 5, 2, 2)",
                ncfdtridfn(0.99, 5.0, 2.0, 2.0),
                1e100,
            ),
            (
                "ncfdtridfn(0.2, 5, 2, 2)",
                ncfdtridfn(0.2, 5.0, 2.0, 2.0),
                1e-100,
            ),
            (
                "ncfdtridfn(0.5, 5, 2, 0)",
                ncfdtridfn(0.5, 5.0, 2.0, 0.0),
                1e-100,
            ),
            (
                "ncfdtridfn(0.5, 5, 4294967296, 2)",
                ncfdtridfn(0.5, 5.0, 4_294_967_296.0, 2.0),
                nan,
            ),
        ];
        for (label, got, want) in rows {
            let matches = if want.is_nan() {
                got.is_nan()
            } else {
                got == want
            };
            assert!(matches, "{label} = {got}, SciPy 1.17.1 gives {want}");
        }
    }

    /// frankenscipy-g9yid. Must not change: interior roots of the cdflib-backed inverses.
    /// SciPy 1.17.1 inverts cdflib's own CDFs, which cdflib truncates at a relative 1e-4
    /// (`cumfnc`) and 1e-7 (`cumtnc`), so these agree with SciPy only to those tolerances;
    /// against fsci's own CDF the roots are exact to rounding.
    #[test]
    fn cdflib_inverses_interior_values_do_not_move() {
        let rows = [
            (
                "ncfdtrinc(3, 5, 0.5, 2)",
                ncfdtrinc(3.0, 5.0, 0.5, 2.0),
                3.1855517561084823,
                1e-5,
            ),
            (
                "nctdtrinc(5, 0.5, 2)",
                nctdtrinc(5.0, 0.5, 2.0),
                1.8929610084247588,
                1e-7,
            ),
            (
                "nctdtrinc(5, 0.9, -2)",
                nctdtrinc(5.0, 0.9, -2.0),
                -3.4141763497275717,
                1e-7,
            ),
            (
                "nctdtridf(0.5, 1, 1.5)",
                nctdtridf(0.5, 1.0, 1.5),
                0.7061449175041883,
                1e-8,
            ),
            (
                "ncfdtridfd(3, 0.5, 2, 2)",
                ncfdtridfd(3.0, 0.5, 2.0, 2.0),
                1.8986465231034602,
                1e-5,
            ),
            (
                "ncfdtridfn(0.5, 5, 2, 2)",
                ncfdtridfn(0.5, 5.0, 2.0, 2.0),
                1.3625100439758695,
                1e-5,
            ),
        ];
        for (label, got, want, rel) in rows {
            assert!(
                (got - want).abs() <= rel * want.abs(),
                "{label} = {got}, SciPy 1.17.1 gives {want} (rel tol {rel})"
            );
        }
        // Against fsci's own CDF the roots are tight.
        let nc = ncfdtrinc(3.0, 5.0, 0.5, 2.0);
        assert!(
            (ncfdtr(3.0, 5.0, nc, 2.0) - 0.5).abs() < 1e-12,
            "ncfdtrinc round trip"
        );
        let nc = nctdtrinc(5.0, 0.5, 2.0);
        assert!(
            (nctdtr(5.0, nc, 2.0) - 0.5).abs() < 1e-12,
            "nctdtrinc round trip"
        );
        let df = nctdtridf(0.5, 1.0, 1.5);
        assert!(
            (nctdtr(df, 1.0, 1.5) - 0.5).abs() < 1e-12,
            "nctdtridf round trip"
        );
    }

    /// frankenscipy-g9yid. The cdflib-backed searches are bounded. In a float transliteration of
    /// the old unbounded doubling, each of these walked an estimated 1.2e9 to 2.4e9 Poisson
    /// terms and then answered NaN or the unbounded root 2.6e14; SciPy 1.17.1 answers all three
    /// with its cap. Run on a worker thread so a regression fails instead of hanging the suite.
    #[test]
    fn cdflib_inverse_searches_are_bounded() {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let rows: Vec<(&str, f64, f64)> = vec![
                (
                    "ncfdtrinc(3, 5, 0.5, 1e300)",
                    ncfdtrinc(3.0, 5.0, 0.5, 1e300),
                    1e4,
                ),
                (
                    "ncfdtrinc(3, 5, 0.5, 1e14)",
                    ncfdtrinc(3.0, 5.0, 0.5, 1e14),
                    1e4,
                ),
                ("nctdtrinc(5, 0.5, 3e8)", nctdtrinc(5.0, 0.5, 3e8), 1e6),
            ];
            let _ = tx.send(rows);
        });
        let rows = rx
            .recv_timeout(std::time::Duration::from_secs(60))
            .expect("a cdflib-backed inverse did not return within 60 s (frankenscipy-g9yid)");
        for (label, got, want) in rows {
            assert!(got == want, "{label} = {got}, SciPy 1.17.1 gives {want}");
        }
    }

    /// frankenscipy-g9yid. `illinois_root` returned a wrong root in three situations; each row
    /// is one of them, with the old answer.
    #[test]
    fn illinois_root_verifies_before_returning() {
        // 1. A residual stuck at a tiny positive value beyond its sign change (the underflow
        //    plateau of 1e-300 - chndtr(5, df, 2) past df ≈ 403.34, with its value at 1e-6):
        //    false position hugged the far end and the old stop returned 4882812.500000998.
        let step_at = 403.3387893865996;
        let at_lo = -0.8686981363318387;
        let step = |x: f64| if x < step_at { at_lo } else { 1e-300 };
        let root = illinois_root(step, 1e-6, 1e10, at_lo, 1e-300);
        assert!(
            (root - step_at).abs() <= 1e-14 * step_at,
            "plateau root {root}, want {step_at}"
        );
        // 2. A root near 1e-200: the old absolute tolerance accepted the whole bracket at once
        //    and returned its midpoint, 5.5e-200. The old false-position step also underflowed
        //    to 0/0 here; the new one lands on the root of this linear residual at once.
        let evaluations = std::cell::Cell::new(0_u32);
        let root = illinois_root(
            |x| {
                evaluations.set(evaluations.get() + 1);
                x - 4.7e-200
            },
            1e-200,
            1e-199,
            1e-200 - 4.7e-200,
            1e-199 - 4.7e-200,
        );
        assert!(
            (root - 4.7e-200).abs() <= 1e-15 * 4.7e-200,
            "tiny root {root}"
        );
        assert!(
            evaluations.get() <= 3,
            "tiny root took {} evaluations",
            evaluations.get()
        );
        //    A curved residual with its root at 1e-20: the old tolerance returned the first
        //    false-position estimate, 2.4785054261852173e-20.
        let curved = |x: f64| x.sqrt() - 1e-10;
        let root = illinois_root(curved, 1e-21, 1e-19, curved(1e-21), curved(1e-19));
        assert!((root - 1e-20).abs() <= 1e-14 * 1e-20, "curved root {root}");
        // 3. A flat residual near a root at 1e-10: the old stop took one stagnating step and
        //    returned 1e-30.
        let root = illinois_root(|x| x * x * x - 1e-30, -1.0, 1.0, -1.0 - 1e-30, 1.0 - 1e-30);
        assert!((root - 1e-10).abs() <= 1e-12 * 1e-10, "cube root {root}");
        // Must not change: an ordinary smooth root to full precision.
        let root = illinois_root(|x| x * x - 2.0, 1.0, 2.0, -1.0, 2.0);
        assert!(
            (root - std::f64::consts::SQRT_2).abs() <= 4.0 * f64::EPSILON,
            "sqrt 2 {root}"
        );
    }

    /// frankenscipy-g9yid. `bracket_and_solve_root` is Boost's walk: when the residual never
    /// changes sign on the way down it answers the midpoint of `[0, a]` for the first
    /// `|a| < f64::MIN_POSITIVE`, which from a guess of 1 is `2⁻¹⁰⁴⁵ = 2.65249474e-315`
    /// (SciPy 1.17.1's `chndtridf(5, 0.5, 30)`), and from 3 is `7.957484216e-315`
    /// (`chndtridf(5, 1 − 2⁻⁵³, 2)`).
    #[test]
    fn bracket_and_solve_root_escapes_like_boost() {
        assert_eq!(
            bracket_and_solve_root(|_| -1.0, 1.0, false),
            2.65249474e-315
        );
        assert_eq!(
            bracket_and_solve_root(|_| -1.0, 1.0, false),
            f64::MIN_POSITIVE * 2f64.powi(-23)
        );
        assert_eq!(
            bracket_and_solve_root(|_| -1.0, 3.0, false),
            7.957484216e-315
        );
        // An upward walk that never brackets overflows, which Boost reports as an error.
        assert!(bracket_and_solve_root(|_| 1.0, 1.0, false).is_nan());
        // A root on a bracket end is returned exactly; an interior one to full precision.
        assert_eq!(bracket_and_solve_root(|x| 2.0 - x, 1.0, false), 2.0);
        let root = bracket_and_solve_root(|x| x.ln() - 10.0, 1.0, true);
        assert!((root - 10f64.exp()).abs() <= 1e-14 * 10f64.exp(), "{root}");
    }

    /// frankenscipy-g9yid. ncfdtri and nctdtrit bracket by Boost's geometric walk, so a root far
    /// below 1 is resolved to a relative tolerance. SciPy 1.17.1 values read live; the old
    /// `[0, hi]` search returned 5.92e-20 for the first row (its absolute tolerance stopped at
    /// the first false-position step).
    #[test]
    fn noncentral_quantiles_resolve_small_roots_like_scipy() {
        let rows = [
            (
                "ncfdtri(3, 5, 2, 1e-20)",
                ncfdtri(3.0, 5.0, 2.0, 1e-20),
                6.67000004655459e-14,
            ),
            (
                "ncfdtri(3, 5, 2, 0.5)",
                ncfdtri(3.0, 5.0, 2.0, 0.5),
                1.5736032013715704,
            ),
            (
                "nctdtrit(5, 1, 0.5)",
                nctdtrit(5.0, 1.0, 0.5),
                1.0528510409473961,
            ),
        ];
        for (label, got, want) in rows {
            assert!(
                (got - want).abs() <= 1e-12 * want.abs(),
                "{label} = {got}, SciPy 1.17.1 gives {want}"
            );
        }
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

    /// `ncfdtrc` against `scipy.stats.ncf.sf`, chosen where `1 - ncfdtr` fails.
    ///
    /// The goldens span 227 orders of magnitude and are checked RELATIVELY: an
    /// absolute tolerance would be trivially satisfied at the small end, which is
    /// the only end where a survival function earns its keep. Each row carries the
    /// `1 - ncfdtr` value it is contrasted with, so what this kernel buys is
    /// visible in the source rather than only in a commit message.
    #[test]
    fn ncfdtrc_matches_scipy_survival_where_one_minus_cdf_erodes_then_collapses() {
        // dfn, dfd, nc, f, scipy.stats.ncf.sf, and what 1 - ncfdtr gives instead.
        let cases = [
            (
                3.0,
                5.0,
                2.0,
                20.0,
                9.362_115_386_065_090e-3,
                "9.362115e-03 ok",
            ),
            (
                2.0,
                20.0,
                1.0,
                60.0,
                3.851_245_644_126_540e-8,
                "3.851246e-08 ok",
            ),
            (
                10.0,
                10.0,
                40.0,
                100.0,
                1.168_395_145_190_737e-5,
                "1.168395e-05 ok",
            ),
            (
                5.0,
                10.0,
                3.0,
                500.0,
                8.759_364_975_614_790e-11,
                "8.759360e-11, 7 digits",
            ),
            (
                3.0,
                5.0,
                2.0,
                1e6,
                2.330_338_627_458_045e-14,
                "2.331468e-14, 3 digits",
            ),
            (
                3.0,
                5.0,
                2.0,
                1e12,
                2.330_354_877_710_844e-29,
                "0.0 COLLAPSED",
            ),
            (
                5.0,
                200.0,
                3.0,
                1e4,
                4.498_173_170_614_720e-230,
                "0.0 COLLAPSED",
            ),
        ];
        for (dfn, dfd, nc, f, want, note) in cases {
            let got = ncfdtrc(dfn, dfd, nc, f);
            let rel = ((got - want) / want).abs();
            assert!(
                rel < 1e-9,
                "ncfdtrc({dfn}, {dfd}, {nc}, {f:e}) = {got:e}, scipy = {want:e} \
                 (rel {rel:e}); 1 - ncfdtr gives {note}"
            );
        }

        // WHY the kernel exists, asserted rather than asserted-about: `1 - ncfdtr`
        // has NO correct digits at these arguments.
        //
        // The first version of this check asserted the subtraction returns exactly
        // 0.0, which is what SciPy's ncfdtr does. Ours does not -- it returns
        // 1.1102230246251565e-15, one ulp of 1.0, against a true 2.3304e-29. That
        // is not a milder failure than zero, it is a WORSE one: zero is visibly
        // wrong, whereas a plausible-looking 1e-15 is the double-precision noise
        // floor wearing the costume of an answer. The assertion is therefore on
        // the relative error, which catches both shapes and does not depend on
        // which side of 1.0 the cdf happens to land.
        for (dfn, dfd, nc, f, truth) in [
            (3.0, 5.0, 2.0, 1e12, 2.330_354_877_710_844e-29),
            (5.0, 200.0, 3.0, 1e4, 4.498_173_170_614_720e-230),
        ] {
            let naive = 1.0 - ncfdtr(dfn, dfd, nc, f);
            let rel = ((naive - truth) / truth).abs();
            assert!(
                rel > 0.99,
                "premise: `1 - ncfdtr` should be worthless at ({dfn}, {dfd}, {nc},                  {f:e}) -- got {naive:e} against a true {truth:e} (rel {rel:e}). If                  this now agrees, `ncfdtr` improved and this kernel's motivation,                  not its correctness, is what needs revisiting"
            );
        }

        // Complementarity, but only mid-range, where the subtraction is still
        // accurate enough for the comparison to mean anything.
        let x = 3.0;
        let (p, q) = (ncfdtr(3.0, 5.0, 2.0, x), ncfdtrc(3.0, 5.0, 2.0, x));
        assert!(
            (p + q - 1.0).abs() < 1e-12,
            "cdf {p} + sf {q} = {} should be 1 mid-range",
            p + q
        );

        // Boundaries.
        assert_eq!(ncfdtrc(3.0, 5.0, 2.0, 0.0), 1.0);
        assert_eq!(ncfdtrc(3.0, 5.0, 2.0, -1.0), 1.0);
        assert!(ncfdtrc(3.0, 5.0, 2.0, f64::NAN).is_nan());
        assert!(ncfdtrc(0.0, 5.0, 2.0, 1.0).is_nan());
        assert!(ncfdtrc(3.0, 5.0, -1.0, 1.0).is_nan());
        assert_eq!(ncfdtrc(3.0, 5.0, 2.0, f64::INFINITY), 0.0);

        // nc = 0 degenerates to the central F survival. This is a DELIBERATE
        // DIVERGENCE from the incumbent: scipy.stats.ncf.sf(1e4, 5, 200, 0)
        // returns -1.0 (its cdf saturates to 1.0 and the complement goes wrong),
        // while scipy.stats.f.sf(1e4, 5, 200) = 8.213081274649901e-238. A
        // probability of -1 is not a value worth reproducing, so this kernel
        // returns the central result and is checked against `f.sf`.
        let central = ncfdtrc(5.0, 200.0, 0.0, 1e4);
        let want_central = 8.213_081_274_649_901e-238;
        assert!(
            ((central - want_central) / want_central).abs() < 1e-9,
            "nc=0 should give the central F survival {want_central:e}, got {central:e}"
        );
        let mid = ncfdtrc(3.0, 5.0, 0.0, 2.0);
        assert!(
            (mid - 0.232_623_918_000_078_6).abs() < 1e-12,
            "nc=0 mid-range should match scipy.stats.f.sf(2, 3, 5), got {mid}"
        );
    }

    /// frankenscipy-qu5po. The noncentral χ², F and t CDFs, and the inverses built on them,
    /// hung at `nc = ±inf`, where the Poisson mode `j₀` is infinite, and once `j₀ ≥ 2^53`,
    /// where `j -= 1.0` stops moving (see `POISSON_INDEX_LIMIT`). They also walked the whole
    /// Poisson left tail when every term was zero: about 2.7e9 steps for chndtr(5, 3, 1e16)
    /// and 3.5e9 for nctdtr(5, 1.3e8, 2), both below 2^53. Every trigger runs on one worker
    /// thread and the test waits at most 10 s, so a regression fails here instead of hanging
    /// the suite.
    ///
    /// Expected values are SciPy 1.17.1, read live. chndtrc and ncfdtrc mirror
    /// scipy.stats.ncx2.sf and ncf.sf.
    ///
    /// ```text
    /// chndtr(5, 3, inf) = 0.0             chndtr(inf, 3, inf) = nan
    /// chndtr(5, 3, 1e16) = 0.0            chndtr(2^60, 3, 2^60) = nan
    /// ncx2.sf(5, 3, inf) = nan            ncx2.sf(inf, 3, inf) = 0.0
    /// ncfdtr(3, 5, inf, 2) = nan          ncfdtr(3, 5, inf, inf) = 1.0
    /// ncfdtr(3, 5, 1e16, 2) = 0.0         ncfdtr(3, 5, 2^60, 2^60/3) = nan
    /// ncf.sf(2, 3, 5, inf) = nan
    /// nctdtr(5, inf, 2) = nan             nctdtr(5, -inf, 2) = nan
    /// nctdtr(5, inf, inf) = 1.0           nctdtr(5, -inf, -inf) = 0.0
    /// nctdtr(5, 1.3e8, 2) = 0.0           nctdtr(5, -1.3e8, 2) = 1.0
    /// nctdtr(5, 1.3e8, -2) = 0.0          nctdtr(5, 1.35e8, 1.35e8) = nan
    /// chndtrix(0.5, 3, inf) = nan         chndtridf(5, 0.5, inf) = nan
    /// ncfdtri(3, 5, inf, 0.5) = nan       nctdtrit(5, inf, 0.5) = nan
    /// nctdtridf(0.5, inf, 2) = nan        ncfdtridfd(3, 0.5, inf, 2) = nan
    /// ncfdtridfn(0.5, 5, inf, 2) = nan    chndtrix(0.5, 3, 2^60) = nan
    /// ncfdtri(3, 5, 2^60, 0.5) = nan      nctdtrit(5, 1.35e8, 0.5) = nan
    /// ```
    ///
    /// Known divergence, deliberately not asserted: at `j₀ ≥ 2^53` fsci is nan in the far
    /// tails too. SciPy there gives chndtr(5, 3, 2^60) = 0.0 and nctdtr(5, ±1.35e8, 2) = 0.0
    /// and 1.0; `POISSON_INDEX_LIMIT` explains why a zero anchor there cannot be trusted.
    ///
    /// Must not change: these existing goldens run the full walk. chndtr(2, 4, 100) =
    /// 2.0596217576094693e-19 is a deep left tail whose mode anchors are about 5e-69, small
    /// but not zero, so the all-zero exit must not take it. The others are
    /// chndtr(2000, 2, 2000) = 0.49553941086177933, ncfdtr(10, 20, 40, 3) = 0.10716411882720595
    /// and nctdtr(8, 5, 4) = 0.21027058165197615.
    #[test]
    fn noncentral_families_return_on_infinite_and_huge_noncentrality() {
        const TWO_POW_60: f64 = 1_152_921_504_606_846_976.0;
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let inf = f64::INFINITY;
            let nan = f64::NAN;
            let big = TWO_POW_60;
            let rows: Vec<(&str, f64, f64)> = vec![
                ("chndtr(5, 3, inf)", gamma::chndtr(5.0, 3.0, inf), 0.0),
                ("chndtr(inf, 3, inf)", gamma::chndtr(inf, 3.0, inf), nan),
                ("chndtr(5, 3, 1e16)", gamma::chndtr(5.0, 3.0, 1e16), 0.0),
                ("chndtr(2^60, 3, 2^60)", gamma::chndtr(big, 3.0, big), nan),
                ("chndtrc(5, 3, inf)", gamma::chndtrc(5.0, 3.0, inf), nan),
                ("chndtrc(inf, 3, inf)", gamma::chndtrc(inf, 3.0, inf), 0.0),
                ("ncfdtr(3, 5, inf, 2)", ncfdtr(3.0, 5.0, inf, 2.0), nan),
                ("ncfdtr(3, 5, inf, inf)", ncfdtr(3.0, 5.0, inf, inf), 1.0),
                ("ncfdtr(3, 5, 1e16, 2)", ncfdtr(3.0, 5.0, 1e16, 2.0), 0.0),
                (
                    "ncfdtr(3, 5, 2^60, 2^60/3)",
                    ncfdtr(3.0, 5.0, big, big / 3.0),
                    nan,
                ),
                ("ncfdtrc(3, 5, inf, 2)", ncfdtrc(3.0, 5.0, inf, 2.0), nan),
                ("nctdtr(5, inf, 2)", nctdtr(5.0, inf, 2.0), nan),
                ("nctdtr(5, -inf, 2)", nctdtr(5.0, -inf, 2.0), nan),
                ("nctdtr(5, inf, inf)", nctdtr(5.0, inf, inf), 1.0),
                ("nctdtr(5, -inf, -inf)", nctdtr(5.0, -inf, -inf), 0.0),
                ("nctdtr(5, 1.3e8, 2)", nctdtr(5.0, 1.3e8, 2.0), 0.0),
                ("nctdtr(5, -1.3e8, 2)", nctdtr(5.0, -1.3e8, 2.0), 1.0),
                ("nctdtr(5, 1.3e8, -2)", nctdtr(5.0, 1.3e8, -2.0), 0.0),
                (
                    "nctdtr(5, 1.35e8, 1.35e8)",
                    nctdtr(5.0, 1.35e8, 1.35e8),
                    nan,
                ),
                ("chndtrix(0.5, 3, inf)", gamma::chndtrix(0.5, 3.0, inf), nan),
                (
                    "chndtridf(5, 0.5, inf)",
                    gamma::chndtridf(5.0, 0.5, inf),
                    nan,
                ),
                ("ncfdtri(3, 5, inf, 0.5)", ncfdtri(3.0, 5.0, inf, 0.5), nan),
                ("nctdtrit(5, inf, 0.5)", nctdtrit(5.0, inf, 0.5), nan),
                ("nctdtridf(0.5, inf, 2)", nctdtridf(0.5, inf, 2.0), nan),
                (
                    "ncfdtridfd(3, 0.5, inf, 2)",
                    ncfdtridfd(3.0, 0.5, inf, 2.0),
                    nan,
                ),
                (
                    "ncfdtridfn(0.5, 5, inf, 2)",
                    ncfdtridfn(0.5, 5.0, inf, 2.0),
                    nan,
                ),
                (
                    "chndtrix(0.5, 3, 2^60)",
                    gamma::chndtrix(0.5, 3.0, big),
                    nan,
                ),
                ("ncfdtri(3, 5, 2^60, 0.5)", ncfdtri(3.0, 5.0, big, 0.5), nan),
                ("nctdtrit(5, 1.35e8, 0.5)", nctdtrit(5.0, 1.35e8, 0.5), nan),
            ];
            let _ = tx.send(rows);
        });
        let rows = rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("a noncentral CDF or inverse did not return within 10 s (frankenscipy-qu5po)");
        for (label, got, want) in rows {
            let matches = if want.is_nan() {
                got.is_nan()
            } else {
                got == want
            };
            assert!(matches, "{label} = {got}, SciPy 1.17.1 gives {want}");
        }

        for (label, got, want) in [
            (
                "chndtr(2, 4, 100)",
                gamma::chndtr(2.0, 4.0, 100.0),
                2.059_621_757_609_469_3e-19,
            ),
            (
                "chndtr(2000, 2, 2000)",
                gamma::chndtr(2000.0, 2.0, 2000.0),
                0.495_539_410_861_779_33,
            ),
            (
                "ncfdtr(10, 20, 40, 3)",
                ncfdtr(10.0, 20.0, 40.0, 3.0),
                0.107_164_118_827_205_95,
            ),
            (
                "nctdtr(8, 5, 4)",
                nctdtr(8.0, 5.0, 4.0),
                0.210_270_581_651_976_15,
            ),
        ] {
            // The existing goldens' own bound (chndtr_matches_scipy_reference_values et al.).
            assert!(
                (got - want).abs() <= 1e-10 * want.abs().max(1e-12),
                "{label} = {got}, SciPy 1.17.1 gives {want}"
            );
        }
    }

    /// frankenscipy-g9yid. The upward half of each noncentral Poisson walk (chndtr, chndtrc,
    /// ncfdtr, ncfdtrc, nctdtr) stopped after a fixed 100,000 steps. It needs about 7–9·√λ
    /// steps, so from λ ≈ 1.2e8 the sum lost mass right of the mode and returned a wrong value
    /// with no error. The cap is now `poisson_upward_step_cap(λ) = max(100_000, ⌈40·√λ⌉)`.
    ///
    /// Expected values are SciPy 1.17.1, read live. chndtrc and ncfdtrc mirror
    /// scipy.stats.ncx2.sf and ncf.sf. The capped-walk column comes from a float
    /// transliteration of the loops before this fix:
    ///
    /// ```text
    ///                                   SciPy 1.17.1             capped walk
    /// λ = 1e10 (nc = 2e10)
    /// chndtr(2.00008e10, 3, 2e10)       0.9976608741697095       0.8412377222681212
    /// chndtr(2e10, 3, 2e10)             0.49999717905138635      0.48741686728990513
    /// ncx2.sf(2.00008e10, 3, 2e10)      0.0023391258360952       0.00011401720251455691
    /// ncx2.sf(2e10, 3, 2e10)            0.5000028209494191       0.3539348721809433
    /// λ = 1e9 (nc = 2e9; nc = 44721.36 for t)
    /// ncfdtr(3, 50, 2e9, 1.13e9)        0.9907652168580527       0.9899900470882801
    /// ncfdtr(3, 50, 2e9, 6.9e8)         0.5414581356040811       0.5410346645607019
    /// ncf.sf(6.9e8, 3, 50, 2e9)         0.4585418643965284       0.4581829329895983
    /// ncf.sf(1.13e9, 3, 50, 2e9)        0.009234783144074844     0.009227548888025352
    /// nctdtr(5, 44721.36, 53193)        0.6182192759554592       0.6177344684591549
    /// nctdtr(5, 44721.36, 120000)       0.9832721845904543       0.9825010151381184
    /// ```
    ///
    /// The tolerance is 1e-3 relative at λ = 1e10 and 1e-4 at λ = 1e9. The capped walk misses
    /// by 2.5e-2 to 0.95 and by 7.8e-4. The uncapped transliteration lands within 1e-5 and
    /// 1.5e-6. That remainder came from the mode anchors, not the walk: they were formed in
    /// log space from terms of size λ·ln λ, so they carried an error of about that size times
    /// ε, and the cap does not touch them. They are in saddle-point form now; see
    /// `noncentral_mode_anchors_hold_at_large_noncentrality`.
    ///
    /// Must not change: below λ = 6.25e6 the cap is still 100,000, so these existing goldens
    /// hold to 1e-10: chndtr(2000, 2, 2000) = 0.49553941086177933,
    /// ncfdtr(10, 20, 40, 3) = 0.10716411882720595 and nctdtr(8, 5, 4) = 0.21027058165197615.
    ///
    /// The walks run on a worker thread and the test waits at most 60 s, so a regression that
    /// stops a walk from ending fails here instead of hanging the suite.
    #[test]
    fn noncentral_upward_walk_is_not_cut_off_at_large_noncentrality() {
        let (tx, rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let rows: Vec<(&str, f64, f64, f64)> = vec![
                (
                    "chndtr(2.00008e10, 3, 2e10)",
                    gamma::chndtr(2.00008e10, 3.0, 2e10),
                    0.9976608741697095,
                    1e-3,
                ),
                (
                    "chndtr(2e10, 3, 2e10)",
                    gamma::chndtr(2e10, 3.0, 2e10),
                    0.49999717905138635,
                    1e-3,
                ),
                (
                    "chndtrc(2.00008e10, 3, 2e10)",
                    gamma::chndtrc(2.00008e10, 3.0, 2e10),
                    0.0023391258360952,
                    1e-3,
                ),
                (
                    "chndtrc(2e10, 3, 2e10)",
                    gamma::chndtrc(2e10, 3.0, 2e10),
                    0.5000028209494191,
                    1e-3,
                ),
                (
                    "ncfdtr(3, 50, 2e9, 1.13e9)",
                    ncfdtr(3.0, 50.0, 2e9, 1.13e9),
                    0.9907652168580527,
                    1e-4,
                ),
                (
                    "ncfdtr(3, 50, 2e9, 6.9e8)",
                    ncfdtr(3.0, 50.0, 2e9, 6.9e8),
                    0.5414581356040811,
                    1e-4,
                ),
                (
                    "ncfdtrc(3, 50, 2e9, 6.9e8)",
                    ncfdtrc(3.0, 50.0, 2e9, 6.9e8),
                    0.4585418643965284,
                    1e-4,
                ),
                (
                    "ncfdtrc(3, 50, 2e9, 1.13e9)",
                    ncfdtrc(3.0, 50.0, 2e9, 1.13e9),
                    0.009234783144074844,
                    1e-4,
                ),
                (
                    "nctdtr(5, 44721.36, 53193)",
                    nctdtr(5.0, 44721.36, 53193.0),
                    0.6182192759554592,
                    1e-4,
                ),
                (
                    "nctdtr(5, 44721.36, 120000)",
                    nctdtr(5.0, 44721.36, 120000.0),
                    0.9832721845904543,
                    1e-4,
                ),
            ];
            let _ = tx.send(rows);
        });
        let rows = rx
            .recv_timeout(std::time::Duration::from_secs(60))
            .expect("a noncentral CDF did not return within 60 s (frankenscipy-g9yid)");
        // The worker has sent its rows and only returns now, so this join does not wait on a
        // walk. A timeout above fails the test before reaching it.
        worker
            .join()
            .expect("the frankenscipy-g9yid worker thread panicked");
        for (label, got, want, tol) in rows {
            let rel = ((got - want) / want).abs();
            assert!(
                rel <= tol,
                "{label} = {got}, SciPy 1.17.1 gives {want} (relative error {rel:e} > {tol:e})"
            );
        }

        for (label, got, want) in [
            (
                "chndtr(2000, 2, 2000)",
                gamma::chndtr(2000.0, 2.0, 2000.0),
                0.49553941086177933,
            ),
            (
                "ncfdtr(10, 20, 40, 3)",
                ncfdtr(10.0, 20.0, 40.0, 3.0),
                0.10716411882720595,
            ),
            (
                "nctdtr(8, 5, 4)",
                nctdtr(8.0, 5.0, 4.0),
                0.21027058165197615,
            ),
        ] {
            assert!(
                (got - want).abs() <= 1e-10 * want.abs(),
                "{label} = {got}, SciPy 1.17.1 gives {want}"
            );
        }
    }

    /// frankenscipy-g9yid: the noncentral t tails away from the bulk were `1 − nctdtr` of a
    /// reflected law and cancelled to 0 (or ~1e-16 of garbage) below about 1e-16: nctdtr's left
    /// tail at t < 0 and nctdtrc's right tail at t > 0. Expected values are mpmath quadrature of
    /// E[Φ(tS − δ)] and E[Φ(δ − tS)] as two separate positive integrals (34 digits, CDF + SF = 1
    /// to 8.9e-33 on the grid; scratchpad nct_tail/refs.json). SciPy 1.17.1 is not the oracle
    /// here: it is wrong across zero from the mean, e.g. nctdtr(5, 10, −7.40744670100678) =
    /// 6.2e-17 where the law is 1e-30. Before this change fsci returned 0 at every row below.
    #[test]
    fn noncentral_t_tails_are_computed_directly() {
        let left_tail = [
            (
                1.0,
                0.5,
                -1.578_188_193_300_98e29,
                1.000_000_000_002_292_6e-30,
            ),
            (5.0, -2.0, -335_056.385_650_683, 1.000_000_000_001_415_1e-25),
            (5.0, 3.0, -347.789_610_326_029, 9.999_999_999_983_786e-17),
            (5.0, 10.0, -7.407_446_701_006_78, 9.999_999_999_990_794e-31),
            (30.0, 3.0, -17.647_597_048_407_3, 9.999_999_999_989_708e-26),
            (
                30.0,
                10.0,
                -1.728_399_704_847_42,
                1.000_000_000_000_881_6e-30,
            ),
            (200.0, 3.0, -6.760_435_488_904_36, 9.999_999_999_994_022e-21),
        ];
        for (df, nc, t, want) in left_tail {
            let got = nctdtr(df, nc, t);
            assert!(
                (got - want).abs() <= 1e-13 * want,
                "nctdtr({df}, {nc}, {t:e}) = {got:e}, mpmath {want:e}"
            );
        }
        let right_tail = [
            (5.0, 3.0, 4_405_362.281_620_58, 9.999_999_999_993_671e-31),
            (
                30.0,
                -10.0,
                1.728_399_704_847_42,
                1.000_000_000_000_881_6e-30,
            ),
            (30.0, 40.0, 615.457_028_982_028, 1.000_000_000_001_425_2e-30),
            (200.0, -2.0, 6.654_574_290_171_54, 9.999_999_999_997_589e-17),
        ];
        for (df, nc, t, want) in right_tail {
            let got = nctdtrc(df, nc, t);
            assert!(
                (got - want).abs() <= 1e-13 * want,
                "nctdtrc({df}, {nc}, {t:e}) = {got:e}, mpmath {want:e}"
            );
        }
        // The two sides agree wherever neither is tiny, and the limits hold.
        for (df, nc, t) in [
            (5.0, 3.0, 3.0),
            (5.0, -2.0, -1.0),
            (30.0, 10.0, 9.0),
            (1.0, 0.5, 0.2),
        ] {
            let closure = nctdtr(df, nc, t) + nctdtrc(df, nc, t) - 1.0;
            assert!(
                closure.abs() <= 4.0 * f64::EPSILON,
                "({df}, {nc}, {t}): {closure:e}"
            );
        }
        assert_eq!(nctdtrc(5.0, 3.0, f64::INFINITY), 0.0);
        assert_eq!(nctdtrc(5.0, 3.0, f64::NEG_INFINITY), 1.0);
        assert!(nctdtrc(5.0, f64::INFINITY, 1.0).is_nan() && nctdtrc(0.0, 1.0, 1.0).is_nan());
    }

    /// frankenscipy-g9yid, item 2. The noncentral walks start from terms at the Poisson mode:
    /// the weight `e^(−λ)λ^j₀/j₀!`, the incomplete gamma or beta there, and its increment. All
    /// were formed as the `exp` of a sum of logs of size `λ·ln λ`, so they carried an error of
    /// about `λ·ln λ·ε`, and SciPy is NaN or inaccurate at these λ. They are now in Loader's
    /// saddle-point form (`gamma::poisson_term`, `beta_term`, and the large-shape paths of the
    /// incomplete gamma and beta), and nctdtr takes `1 − x` as `df/(t² + df)`.
    ///
    /// Expected values are mpmath at 40 digits, from integrals that are not Poisson sums:
    /// chndtr for df = 3 is `∫₀ˣ ½e^(−w/2)·[Φ(√(x−w) − √nc) − Φ(−√(x−w) − √nc)] dw`, and
    /// nctdtr is `∫₀^∞ f_χ²(df)(v)·Φ(t·√(v/df) − nc) dv`. At λ = 100 and 1e4, at the mean
    /// and ±3 sd, both agree with SciPy 1.17.1 to 5e-15 wherever the CDF is above 1e-30, and
    /// at λ = 1e6 the nctdtr integral agrees with an mpmath Lenth series to 20 digits.
    /// Relative errors below are from a float transliteration of each arm.
    /// "Before" is this code with `gamma::SADDLE_POINT_MIN_SHAPE = f64::INFINITY`, which puts
    /// every anchor back in log space. The code before this change also cut the incomplete
    /// gamma series off at 2,000,000 terms and missed chndtr(2e12 + 3, 3, 2e12) by 4.3e-2.
    ///
    /// ```text
    ///                                λ       mpmath                 before    after
    /// chndtr(2e8 + 3, 3, 2e8)        1e8     0.50001410473950935    2.5e-8    1.0e-13
    /// chndtr(2e10 + 3, 3, 2e10)      1e10    0.50000141047395879    1.4e-5    1.3e-12
    /// chndtr(2e12 + 3, 3, 2e12)      1e12    0.50000014104739589    1.9e-3    1.3e-11
    /// nctdtr(5, 14142, 16821)        1.0e8   0.61822294435768955    1.1e-7    1.5e-13
    /// nctdtr(5, 141421, 168209)      1.0e10  0.61820905927730827    2.7e-5    1.5e-12
    /// nctdtr(5, 1414214, 1682089)    1.0e12  0.61820540746559938    1.3e-5    1.4e-11
    /// ```
    ///
    /// The error left after the fix grows as about 1.3e-17·√λ and is the walk's own rounding:
    /// at λ = 1e10 the walk started from mpmath-exact anchors still misses by 1.25e-12. The
    /// tolerance is 1e-16·√λ, about 8 times that and more than 20,000 times below every
    /// before-arm error.
    ///
    /// Must not change, bit-identical: with every shape below `SADDLE_POINT_MIN_SHAPE`, each
    /// anchor keeps its old log-space expression operation for operation, so
    /// chndtr(5, 3, 3) = 0.49007134573953426 as before. That literal is from the float
    /// transliteration, not from a cargo run, so it is asserted to 1e-15 rather than bit for
    /// bit.
    ///
    /// The walks run on a worker thread and the test waits at most 60 s.
    #[test]
    fn noncentral_mode_anchors_hold_at_large_noncentrality() {
        let (tx, rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let rows: Vec<(&str, f64, f64, f64)> = vec![
                (
                    "chndtr(2e8 + 3, 3, 2e8)",
                    gamma::chndtr(2e8 + 3.0, 3.0, 2e8),
                    0.500_014_104_739_509_4,
                    1e8,
                ),
                (
                    "chndtr(2e10 + 3, 3, 2e10)",
                    gamma::chndtr(2e10 + 3.0, 3.0, 2e10),
                    0.500_001_410_473_958_7,
                    1e10,
                ),
                (
                    "chndtr(2e12 + 3, 3, 2e12)",
                    gamma::chndtr(2e12 + 3.0, 3.0, 2e12),
                    0.500_000_141_047_395_9,
                    1e12,
                ),
                (
                    "nctdtr(5, 14142, 16821)",
                    nctdtr(5.0, 14142.0, 16821.0),
                    0.618_222_944_357_689_5,
                    0.5 * 14142.0 * 14142.0,
                ),
                (
                    "nctdtr(5, 141421, 168209)",
                    nctdtr(5.0, 141_421.0, 168_209.0),
                    0.618_209_059_277_308_3,
                    0.5 * 141_421.0 * 141_421.0,
                ),
                (
                    "nctdtr(5, 1414214, 1682089)",
                    nctdtr(5.0, 1_414_214.0, 1_682_089.0),
                    0.618_205_407_465_599_4,
                    0.5 * 1_414_214.0 * 1_414_214.0,
                ),
            ];
            let unchanged = gamma::chndtr(5.0, 3.0, 3.0);
            let _ = tx.send((rows, unchanged));
        });
        let received = rx.recv_timeout(std::time::Duration::from_secs(60));
        assert!(
            received.is_ok(),
            "a noncentral CDF did not return within 60 s (frankenscipy-g9yid)"
        );
        let (rows, unchanged) = received.unwrap_or_default();
        // The worker has sent its rows and only returns now, so this join does not wait on a
        // walk.
        assert!(
            worker.join().is_ok(),
            "the frankenscipy-g9yid worker thread panicked"
        );

        let mut failures = Vec::new();
        for (label, got, want, lam) in rows {
            let rel = ((got - want) / want).abs();
            let tol = 1e-16 * lam.sqrt();
            // `!(rel <= tol)` so that a NaN also fails.
            if !(rel <= tol) {
                failures.push(format!(
                    "{label} = {got}, mpmath gives {want} (relative error {rel:e} > {tol:e})"
                ));
            }
        }
        let want = 0.490_071_345_739_534_26;
        if !((unchanged - want).abs() <= 1e-15 * want) {
            failures.push(format!(
                "chndtr(5, 3, 3) = {unchanged}, it was {want} before frankenscipy-g9yid"
            ));
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// frankenscipy-g9yid. ncfdtr and nctdtr at an infinite or overflowing argument, and
    /// ncfdtr below its support, against SciPy 1.17.1 read live:
    ///
    /// ```text
    /// ncfdtr(3, 5, 3, inf) = 1.0        was nan: y = dfn·f / (dfn·f + dfd) = inf/inf
    /// ncfdtr(3, 5, 3, 1e308) = 1.0      was nan: dfn·f overflows
    /// ncfdtr(3, 5, 0, 1e308) = 1.0      was nan
    /// ncfdtr(3, inf, 3, inf) = 1.0      was nan
    /// ncfdtr(3, 5, 3, -inf) = nan       was 0
    /// ncfdtr(3, 5, 3, -2) = nan         was 0
    /// ncfdtr(3, 5, 0, -2) = nan         was 0
    /// ncfdtr(3, 5, 3, -1e-300) = nan    was 0
    /// nctdtr(5, 3, inf) = 1.0           was nan: x = t²/(t² + df) = inf/inf
    /// nctdtr(5, 3, 1e300) = 1.0         was nan: t² overflows
    /// nctdtr(5, 3, 2e154) = 1.0         was nan
    /// nctdtr(5, 0, inf) = 1.0           was nan
    /// nctdtr(0.001, 3, 1e300) = 1.0     was nan
    /// nctdtr(5, -3, inf) = 1.0          was nan
    /// nctdtr(5, 3, -inf) = 0.0          was nan
    /// nctdtr(5, 3, -1e300) = 0.0        was nan
    /// ```
    ///
    /// Must not change, SciPy 1.17.1: ncfdtr(3, 5, 3, 0) = ncfdtr(3, 5, 3, -0.0) = 0.0; a
    /// finite f with an infinite dfn stays nan, ncfdtr(inf, 5, 3, 2) = nan; the survival
    /// function already answered these, ncf.sf(inf, 3, 5, 3) = ncf.sf(1e308, 3, 5, 3) = 0.0
    /// and ncf.sf(-2, 3, 5, 3) = 1.0; and the finite goldens ncfdtr(10, 20, 40, 3) =
    /// 0.10716411882720595 and nctdtr(8, 5, 4) = 0.21027058165197615.
    #[test]
    fn ncfdtr_nctdtr_follow_scipy_at_infinite_and_negative_arguments() {
        let inf = f64::INFINITY;
        let nan = f64::NAN;
        for (label, got, want) in [
            ("ncfdtr(3, 5, 3, inf)", ncfdtr(3.0, 5.0, 3.0, inf), 1.0),
            ("ncfdtr(3, 5, 3, 1e308)", ncfdtr(3.0, 5.0, 3.0, 1e308), 1.0),
            ("ncfdtr(3, 5, 0, 1e308)", ncfdtr(3.0, 5.0, 0.0, 1e308), 1.0),
            ("ncfdtr(3, inf, 3, inf)", ncfdtr(3.0, inf, 3.0, inf), 1.0),
            ("ncfdtr(3, 5, 3, -inf)", ncfdtr(3.0, 5.0, 3.0, -inf), nan),
            ("ncfdtr(3, 5, 3, -2)", ncfdtr(3.0, 5.0, 3.0, -2.0), nan),
            ("ncfdtr(3, 5, 0, -2)", ncfdtr(3.0, 5.0, 0.0, -2.0), nan),
            (
                "ncfdtr(3, 5, 3, -1e-300)",
                ncfdtr(3.0, 5.0, 3.0, -1e-300),
                nan,
            ),
            ("nctdtr(5, 3, inf)", nctdtr(5.0, 3.0, inf), 1.0),
            ("nctdtr(5, 3, 1e300)", nctdtr(5.0, 3.0, 1e300), 1.0),
            ("nctdtr(5, 3, 2e154)", nctdtr(5.0, 3.0, 2e154), 1.0),
            ("nctdtr(5, 0, inf)", nctdtr(5.0, 0.0, inf), 1.0),
            ("nctdtr(0.001, 3, 1e300)", nctdtr(0.001, 3.0, 1e300), 1.0),
            ("nctdtr(5, -3, inf)", nctdtr(5.0, -3.0, inf), 1.0),
            ("nctdtr(5, 3, -inf)", nctdtr(5.0, 3.0, -inf), 0.0),
            ("nctdtr(5, 3, -1e300)", nctdtr(5.0, 3.0, -1e300), 0.0),
            // Must not change.
            ("ncfdtr(3, 5, 3, 0)", ncfdtr(3.0, 5.0, 3.0, 0.0), 0.0),
            ("ncfdtr(3, 5, 3, -0.0)", ncfdtr(3.0, 5.0, 3.0, -0.0), 0.0),
            ("ncfdtr(inf, 5, 3, 2)", ncfdtr(inf, 5.0, 3.0, 2.0), nan),
            ("ncfdtrc(3, 5, 3, inf)", ncfdtrc(3.0, 5.0, 3.0, inf), 0.0),
            (
                "ncfdtrc(3, 5, 3, 1e308)",
                ncfdtrc(3.0, 5.0, 3.0, 1e308),
                0.0,
            ),
            ("ncfdtrc(3, 5, 3, -2)", ncfdtrc(3.0, 5.0, 3.0, -2.0), 1.0),
        ] {
            let matches = if want.is_nan() {
                got.is_nan()
            } else {
                got == want
            };
            assert!(matches, "{label} = {got}, SciPy 1.17.1 gives {want}");
        }
        for (label, got, want) in [
            (
                "ncfdtr(10, 20, 40, 3)",
                ncfdtr(10.0, 20.0, 40.0, 3.0),
                0.10716411882720595,
            ),
            (
                "nctdtr(8, 5, 4)",
                nctdtr(8.0, 5.0, 4.0),
                0.21027058165197615,
            ),
        ] {
            assert!(
                (got - want).abs() <= 1e-10 * want.abs(),
                "{label} = {got}, SciPy 1.17.1 gives {want}"
            );
        }
    }

    // frankenscipy-5pnba: the incomplete beta kernel is TOMS 708's `bratio`.
    //
    // Expected values are mpmath 1.4.1 at 60+ digits at the EXACT double arguments (the
    // hypergeometric series where it converges geometrically, the Lentz continued fraction on
    // the small side, tanh-sinh quadrature of the log-space density near the mean; wherever
    // two apply they agree to 1e-50 or better), rounded to double, with SciPy 1.17.1 computed
    // alongside. SciPy agrees to about 1e-15 except where a row says otherwise.

    /// Relative error, 0 for an exact match (so both-zero passes).
    fn rel_err(got: f64, want: f64) -> f64 {
        if got == want {
            0.0
        } else {
            ((got - want) / want).abs()
        }
    }

    const BRATIO_TOL: f64 = 1e-13;

    /// The pre-frankenscipy-5pnba kernel, verbatim: Numerical Recipes' Lentz continued
    /// fraction with its 200-term cap, fronted by `a·beta_term` from `a + b = 100` and by
    /// `exp(a·ln x + b·ln y − ln B(a, b))` below. The before arm: see
    /// `legacy_nr_kernel_misses_the_large_parameter_rows`.
    pub(super) fn legacy_nr_betainc(a: f64, b: f64, x: f64, y: f64) -> f64 {
        fn cf(a: f64, b: f64, x: f64) -> f64 {
            const MAX_ITERS: usize = 200;
            const EPS: f64 = 3.0e-14;
            const MIN_NUM: f64 = 1.0e-300;
            let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
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
        let front = if a + b >= gamma::SADDLE_POINT_MIN_SHAPE {
            a * beta_term(a, b, x, y)
        } else {
            match betaln_scalar(a, b, RuntimeMode::Strict) {
                Ok(ln_beta) => (a * x.ln() + b * y.ln() - ln_beta).exp(),
                Err(_) => f64::NAN,
            }
        };
        if x < (a + 1.0) / (a + b + 2.0) {
            front * cf(a, b, x) / a
        } else {
            1.0 - front * cf(b, a, y) / b
        }
    }

    #[test]
    fn betainc_kernel_holds_with_one_huge_shape_parameter() {
        // stdtr(v, ±2) and stdtrc(v, 2). SciPy agrees to <= 5.3e-16.
        for (v, upper, lower) in [
            (1e10, 0.977249868038323, 0.02275013196167695),
            (1e12, 0.9772498680516858, 0.022750131948314184),
            (1e15, 0.9772498680518207, 0.022750131948179344),
            (1e20, 0.9772498680518208, 0.02275013194817921),
        ] {
            for (label, got, want) in [
                ("stdtr(v, 2)", stdtr(v, 2.0), upper),
                ("stdtr(v, -2)", stdtr(v, -2.0), lower),
                ("stdtrc(v, 2)", stdtrc(v, 2.0), lower),
            ] {
                let e = rel_err(got, want);
                assert!(
                    e <= BRATIO_TOL,
                    "{label} at v = {v:e}: {got:e} vs {want:e} ({e:.1e})"
                );
            }
        }
        // fdtr(5, dfd, 2) and fdtrc. SciPy agrees to <= 9.4e-16.
        for (dfd, cdf, sf) in [
            (1e12, 0.9247647538524961, 0.07523524614750389),
            (1e15, 0.9247647538534868, 0.07523524614651317),
            (1e20, 0.9247647538534878, 0.07523524614651218),
        ] {
            for (label, got, want) in [
                ("fdtr", fdtr(5.0, dfd, 2.0), cdf),
                ("fdtrc", fdtrc(5.0, dfd, 2.0), sf),
            ] {
                let e = rel_err(got, want);
                assert!(
                    e <= BRATIO_TOL,
                    "{label}(5, {dfd:e}, 2): {got:e} vs {want:e} ({e:.1e})"
                );
            }
        }
        // betainc(2.5, 1e20, 5e-20) and its complement through each complement entry point.
        let x = 5e-20;
        for (label, got, want) in [
            (
                "betainc",
                betainc_scalar(2.5, 1e20, x, RuntimeMode::Strict).unwrap_or(f64::NAN),
                0.9247647538534878,
            ),
            ("btdtr", btdtr(2.5, 1e20, x), 0.9247647538534878),
            (
                "betaincc",
                betaincc_scalar(2.5, 1e20, x, RuntimeMode::Strict).unwrap_or(f64::NAN),
                0.07523524614651218,
            ),
            ("btdtrc", btdtrc(2.5, 1e20, x), 0.07523524614651218),
        ] {
            let e = rel_err(got, want);
            assert!(
                e <= BRATIO_TOL,
                "{label}(2.5, 1e20, 5e-20): {got:e} vs {want:e} ({e:.1e})"
            );
        }
        // ncfdtr(5, dfd, 1, 2) → chndtr(10, 5, 1) = 0.8626668135599574 as dfd → ∞ (mpmath; SciPy
        // gives ...576). dfd = 1e15 is the Poisson sum of mpmath incomplete betas.
        for (dfd, want) in [
            (1e15, 0.8626668135599563),
            (1e20, 0.8626668135599574),
            (1e100, 0.8626668135599574),
        ] {
            let got = ncfdtr(5.0, dfd, 1.0, 2.0);
            let e = rel_err(got, want);
            assert!(
                e <= BRATIO_TOL,
                "ncfdtr(5, {dfd:e}, 1, 2): {got:e} vs {want:e} ({e:.1e})"
            );
        }
    }

    #[test]
    fn betainc_kernel_holds_with_both_shape_parameters_huge_near_the_mean() {
        // x = mean ± 1 sd rounded to a multiple of 2⁻⁵³, so 1 − x is exact: TOMS 708's BASYM.
        // SciPy misses the lower tails by up to 8.7e-12 (a = b = 1e10).
        for (a, b, x, lower, upper) in [
            (
                1e6,
                1e6,
                0.499646446697795,
                0.1586553144241098,
                0.8413446855758903,
            ),
            (
                1e6,
                1e6,
                0.500353553302205,
                0.8413446855758903,
                0.1586553144241098,
            ),
            (
                1e10,
                1e10,
                0.49999646446609414,
                0.15865525393623536,
                0.8413447460637646,
            ),
            (
                1e10,
                1e10,
                0.5000035355339059,
                0.8413447460637646,
                0.15865525393623536,
            ),
            (
                1e6,
                1e10,
                9.989001599808311e-05,
                0.1586552137265905,
                0.8413447862734095,
            ),
            (
                1e6,
                1e10,
                0.00010008998600175012,
                0.8413447862965844,
                0.15865521370341568,
            ),
            (
                1e10,
                1e6,
                0.9998999100139982,
                0.15865521370341568,
                0.8413447862965844,
            ),
            (
                1e10,
                1e6,
                0.9999001099840019,
                0.8413447862734095,
                0.1586552137265905,
            ),
        ] {
            let got = btdtr(a, b, x);
            let e = rel_err(got, lower);
            assert!(
                e <= BRATIO_TOL,
                "btdtr({a:e}, {b:e}, {x}): {got:e} vs {lower:e} ({e:.1e})"
            );
            let got = btdtrc(a, b, x);
            let e = rel_err(got, upper);
            assert!(
                e <= BRATIO_TOL,
                "btdtrc({a:e}, {b:e}, {x}): {got:e} vs {upper:e} ({e:.1e})"
            );
        }
    }

    #[test]
    fn betainc_kernel_holds_with_a_tiny_shape_parameter_and_in_the_tails() {
        for (a, b, x, lower, upper) in [
            // One parameter ≤ 1e-3 (FPSER, APSER, BPSER regimes).
            (0.001, 5.0, 0.2, 0.9997834456464276, 0.00021655435357234194),
            (5.0, 0.001, 0.9, 0.000590780689705737, 0.9994092193102943),
            (0.001, 0.001, 0.3, 0.49957696213967645, 0.5004230378603235),
            (1e-20, 0.5, 0.3, 1.0, 2.4198702426718918e-20),
            (2.5, 1e-20, 0.3, 2.525438280078752e-22, 1.0),
            (
                1e-05,
                10000.0,
                1e-05,
                0.9999817704465501,
                1.8229553449877786e-05,
            ),
            (
                300.0,
                0.0001,
                0.999,
                9.065575366839369e-05,
                0.9999093442463316,
            ),
            // x near 0 and 1: both tails, down to 3e-299.
            (2.0, 30.0, 0.9999999999, 1.0, 3.100007694563734e-299),
            (30.0, 2.0, 1e-10, 3.0999999997000033e-299, 1.0),
            (5.0, 3.0, 0.999, 0.9999999651048741, 3.489512593001509e-08),
            (0.5, 2.5, 0.9999999990686774, 1.0, 8.987298704137216e-24),
            // SciPy's betaincc here is 1.0, off by 6.4e-11.
            (0.5, 0.5, 1e-20, 6.366197723675813e-11, 0.999999999936338),
            (
                200.0,
                30.0,
                0.95,
                0.9999983407221587,
                1.6592778413140257e-06,
            ),
            (3.0, 40.0, 0.9, 1.0, 7.011999999999938e-38),
        ] {
            for (label, got, want) in [
                ("btdtr", btdtr(a, b, x), lower),
                (
                    "betaincc",
                    betaincc_scalar(a, b, x, RuntimeMode::Strict).unwrap_or(f64::NAN),
                    upper,
                ),
                ("btdtrc", btdtrc(a, b, x), upper),
            ] {
                let e = rel_err(got, want);
                assert!(
                    e <= BRATIO_TOL,
                    "{label}({a:e}, {b:e}, {x}): {got:e} vs {want:e} ({e:.1e})"
                );
            }
        }
    }

    #[test]
    fn betainc_kernel_holds_on_everyday_shape_parameters() {
        // a, b ∈ {0.5, 2, 30, 200}, x ∈ {0.1, 0.5, 0.9}: (a, b, x, I_x(a, b), 1 − I_x(a, b)).
        for (a, b, x, lower, upper) in [
            (0.5, 0.5, 0.1, 0.20483276469913345, 0.7951672353008665),
            (0.5, 0.5, 0.5, 0.5, 0.5),
            (0.5, 0.5, 0.9, 0.7951672353008665, 0.20483276469913342),
            (0.5, 2.0, 0.1, 0.458530260724415, 0.541469739275585),
            (0.5, 2.0, 0.5, 0.8838834764831844, 0.11611652351681559),
            (0.5, 2.0, 0.9, 0.9961174629530395, 0.0038825370469605085),
            (0.5, 30.0, 0.1, 0.9877175515001473, 0.012282448499852747),
            (0.5, 30.0, 0.5, 0.9999999998669794, 1.3302059355529229e-10),
            (0.5, 30.0, 0.9, 1.0, 1.0793411337245473e-31),
            (0.5, 200.0, 0.1, 0.9999999999129222, 8.707774398967445e-11),
            (0.5, 200.0, 0.5, 1.0, 3.500102489320581e-62),
            (0.5, 200.0, 0.9, 1.0, 4.201432808808219e-202),
            (2.0, 0.5, 0.1, 0.0038825370469605107, 0.9961174629530395),
            (2.0, 0.5, 0.5, 0.11611652351681559, 0.8838834764831844),
            (2.0, 0.5, 0.9, 0.5414697392755851, 0.45853026072441494),
            (2.0, 2.0, 0.1, 0.028000000000000004, 0.972),
            (2.0, 2.0, 0.5, 0.5, 0.5),
            (2.0, 2.0, 0.9, 0.972, 0.027999999999999987),
            (2.0, 30.0, 0.1, 0.8304353668991352, 0.16956463310086478),
            (2.0, 30.0, 0.5, 0.9999999850988388, 1.4901161193847656e-08),
            (2.0, 30.0, 0.9, 1.0, 2.799999999999981e-29),
            (2.0, 200.0, 0.1, 0.9999999851843339, 1.4815666128176182e-08),
            (2.0, 200.0, 0.5, 1.0, 6.285245430639753e-59),
            (2.0, 200.0, 0.9, 1.0, 1.8099999999999195e-198),
            (30.0, 0.5, 0.1, 1.0793411337245563e-31, 1.0),
            (30.0, 0.5, 0.5, 1.3302059355529229e-10, 0.9999999998669794),
            (30.0, 0.5, 0.9, 0.01228244849985276, 0.9877175515001473),
            (30.0, 2.0, 0.1, 2.8000000000000047e-29, 1.0),
            (30.0, 2.0, 0.5, 1.4901161193847656e-08, 0.9999999850988388),
            (30.0, 2.0, 0.9, 0.16956463310086492, 0.8304353668991351),
            (30.0, 30.0, 0.1, 3.1056495720556e-15, 0.9999999999999969),
            (30.0, 30.0, 0.5, 0.5, 0.5),
            (30.0, 30.0, 0.9, 0.9999999999999969, 3.105649572055577e-15),
            (30.0, 200.0, 0.1, 0.07688909815907019, 0.9231109018409298),
            (30.0, 200.0, 0.5, 1.0, 6.545600958483148e-33),
            (30.0, 200.0, 0.9, 1.0, 2.3156566951687756e-165),
            (200.0, 0.5, 0.1, 4.201432808808452e-202, 1.0),
            (200.0, 0.5, 0.5, 3.500102489320581e-62, 1.0),
            (200.0, 0.5, 0.9, 8.707774398967499e-11, 0.9999999999129222),
            (200.0, 2.0, 0.1, 1.81000000000002e-198, 1.0),
            (200.0, 2.0, 0.5, 6.285245430639753e-59, 1.0),
            (200.0, 2.0, 0.9, 1.4815666128176268e-08, 0.9999999851843339),
            (200.0, 30.0, 0.1, 2.3156566951689024e-165, 1.0),
            (200.0, 30.0, 0.5, 6.545600958483148e-33, 1.0),
            (200.0, 30.0, 0.9, 0.92311090184093, 0.07688909815906998),
            (200.0, 200.0, 0.1, 4.5332869859638815e-91, 1.0),
            (200.0, 200.0, 0.5, 0.5, 0.5),
            (200.0, 200.0, 0.9, 1.0, 4.5332869859636575e-91),
        ] {
            for (label, got, want) in [
                (
                    "betainc",
                    betainc_scalar(a, b, x, RuntimeMode::Strict).unwrap_or(f64::NAN),
                    lower,
                ),
                (
                    "betaincc",
                    betaincc_scalar(a, b, x, RuntimeMode::Strict).unwrap_or(f64::NAN),
                    upper,
                ),
            ] {
                let e = rel_err(got, want);
                assert!(
                    e <= BRATIO_TOL,
                    "{label}({a}, {b}, {x}): {got:e} vs {want:e} ({e:.1e})"
                );
            }
        }
    }

    /// The frankenscipy-g9yid item 6 row: cdflib's dfd search evaluates ncfdtr at dfd = 1e100,
    /// which the old kernel returned as 0, so the search stopped at the 1e-100 bound.
    #[test]
    fn ncfdtridfd_recovers_dfd_through_the_huge_dfd_bound() {
        let p = ncfdtr(5.0, 10.0, 1.0, 2.0);
        let got = ncfdtridfd(5.0, p, 1.0, 2.0);
        assert!(
            rel_err(got, 10.0) <= 1e-6,
            "ncfdtridfd(5, ncfdtr(5, 10, 1, 2) = {p}, 1, 2) = {got:e}, want 10"
        );
        let far = ncfdtr(5.0, 1e100, 1.0, 2.0);
        assert!(
            rel_err(far, 0.8626668135599574) <= BRATIO_TOL,
            "ncfdtr(5, 1e100, 1, 2) = {far:e}, the search's upper bound"
        );
    }

    #[test]
    fn log_betainc_holds_in_huge_parameter_and_underflowed_tails() {
        // (a, b, x, ln I_x(a, b)) from mpmath; every row but the two marked "representable" has
        // I below the smallest double.
        for (a, b, x, want) in [
            // Student t, df = 1e10: t = −40 (underflows) and t = −3 (representable).
            (5e9, 0.5, 0.9999998400000256, -803.9152306564122),
            (5e9, 0.5, 0.9999999991, -5.914578842892178),
            // a = b huge, 40 and 45 sd below the mean; and 3 sd (representable).
            (1e10, 1e10, 0.49985857864376615, -804.6084739745613),
            (1e10, 1e10, 0.4999893933982824, -6.607726222300781),
            (1e6, 1e6, 0.4840901014007768, -1017.7385112332812),
            // One huge parameter.
            (300.0, 1e15, 1e-14, -734.0965389576945),
            (1e8, 0.9, 0.999992, -800.738162266186),
            (1e12, 50.0, 0.9999999999, -18.256484964870996),
            (1e15, 3.5, 0.9999999999985, -1482.9385470738391),
            (2.5, 5e19, 1e-20, -3.285169839243992),
            // Moderate parameters, far tails.
            (30.0, 200.0, 1e-12, -742.5640374030463),
            (40.0, 0.3, 1e-8, -740.5078726811059),
            (10.0, 0.5, 1e-80, -1843.804226691833),
            (0.5, 0.5, 0.3, -0.9969312110207781),
        ] {
            let got = log_betainc_scalar(a, b, x);
            let e = rel_err(got, want);
            assert!(
                e <= BRATIO_TOL,
                "log_betainc({a:e}, {b:e}, {x}): {got} vs {want} ({e:.1e})"
            );
        }
    }

    /// The before arm, kept as a permanent control: the old kernel fails the rows the tests above
    /// pin, so those tests can tell the two kernels apart.
    #[test]
    fn legacy_nr_kernel_misses_the_large_parameter_rows() {
        // (label, legacy value, expected) — the expected values are the ones pinned above.
        let d15 = 10.0 + 1e15;
        let t20 = 1e20 + 4.0;
        for (label, legacy, want, miss) in [
            (
                "betainc(2.5, 1e20, 5e-20)",
                legacy_nr_betainc(2.5, 1e20, 5e-20, 1.0 - 5e-20),
                0.9247647538534878,
                1e-3,
            ),
            (
                "fdtr(5, 1e15, 2)",
                legacy_nr_betainc(2.5, 0.5e15, 10.0 / d15, 1e15 / d15),
                0.9247647538534868,
                1e-5,
            ),
            (
                // The old stdtr took 1 − x by subtraction; x rounds to 1 at v = 1e20.
                "stdtr(1e20, 2)",
                1.0 - 0.5 * legacy_nr_betainc(5e19, 0.5, 1e20 / t20, 1.0 - 1e20 / t20),
                0.9772498680518208,
                1e-2,
            ),
            (
                "btdtr(1e10, 1e10, mean - 1 sd)",
                legacy_nr_betainc(1e10, 1e10, 0.49999646446609414, 0.5000035355339059),
                0.15865525393623536,
                1e-12,
            ),
        ] {
            let e = rel_err(legacy, want);
            assert!(
                !(e <= miss),
                "the old kernel was expected to miss {label} by more than {miss:e}: {legacy:e} vs {want:e} ({e:.1e})"
            );
        }
    }
}
