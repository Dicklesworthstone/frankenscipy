#![forbid(unsafe_code)]

use std::f64::consts::{FRAC_2_PI, LN_2, PI};

use fsci_runtime::RuntimeMode;

use crate::gamma::gamma_scalar;
use crate::types::{
    Complex64, DispatchPlan, DispatchStep, KernelRegime, SpecialError, SpecialErrorKind,
    SpecialResult, SpecialTensor, not_yet_implemented, record_special_trace,
};

/// Zero-search candidates: (abscissa, order, index, multiplicity-ish tag).
/// Named because the tuple appears in two sibling signatures and clippy's
/// type_complexity fires on both (frankenscipy-e2ve2).
type BesselZeroCandidates = Vec<(f64, i32, i32, i32)>;

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
const BESSEL_ZERO_BRACKET_TOL: f64 = 1.0e-12;
const BESSEL_ZERO_MAX_REFINE_ITERS: usize = 80;

pub fn j0(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("j0", z, mode, 1 << 18, |x| Ok(j0_core(x)))
}

pub fn j1(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("j1", z, mode, 1 << 18, |x| Ok(j1_core(x)))
}

pub fn jn(n: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_binary("jn", n, z, mode, |order, x| jn_scalar(order, x, mode))
}

pub fn y0(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("y0", z, mode, 1 << 16, |x| y0_scalar(x, mode))
}

pub fn y1(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("y1", z, mode, 1 << 16, |x| y1_scalar(x, mode))
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
    map_real_input("i0", z, mode, 1 << 14, |x| Ok(i0_scalar(x)))
}

/// Modified Bessel function of the first kind of order 1: I_1(z).
///
/// Convenience wrapper for iv(1, z). Matches `scipy.special.i1(z)`.
pub fn i1(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("i1", z, mode, 1 << 14, |x| Ok(i1_scalar(x)))
}

/// Advance the `I_v` power series by its exact term ratio instead of carrying the term in
/// logs (`true`, shipping).
///
/// NOT bit-identical to the log form: it rounds once at the start rather than once per term,
/// and every term is positive so nothing cancels. Accuracy against the live SciPy arm is the
/// contract.
pub static IV_SERIES_TERM_RATIO: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);
/// Evaluations that took the term-ratio arm — "enabled" is not "took effect".
///
/// Accuracy contract: a diagnostic counter, not an A/B lever. It has no arms to preserve
/// anything between; incrementing it changes no numeric result.
pub static IV_SERIES_TERM_RATIO_HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Use SciPy's own Chebyshev kernels for `I0`/`I1` (`true`, shipping) instead of routing
/// them through the general order-v `iv_scalar`.
///
/// NOT bit-identical and not intended to be: it is a different approximation, and it is the
/// one the incumbent uses, so it moves our values toward SciPy. Accuracy against the live
/// SciPy arm is the contract and is checked in the same invocation by
/// `perf_special_vs_scipy`; the two forms are also cross-checked against each other by
/// `i0_i1_cephes_matches_the_iv_series`.
pub static BESSEL_I01_CEPHES: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);
/// Evaluations that took the Chebyshev arm — "enabled" is not "took effect".
///
/// Accuracy contract: a diagnostic counter, not an A/B lever. It has no arms to preserve
/// anything between; incrementing it changes no numeric result.
pub static BESSEL_I01_CEPHES_HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

// ── Cephes I0/I1 Chebyshev kernels ───────────────────────────────────────────────────
//
// WHY THESE EXIST. `i0`/`i1` delegated to `iv_scalar(0.0, x)` / `iv_scalar(1.0, x)`, the
// GENERAL order-v routine. For |x| <= 15 that takes the ascending power series, whose
// summand is formed in logs — an `exp` and a `lgamma` PER TERM — and needs tens of terms
// once `x^2/4` is large. Measured against live SciPy on the identical fixture, instructions
// per element:
//
//     i0   fsci 4175.4   scipy 339.7   12.29x
//     i1   fsci 4129.3   scipy 341.8   12.08x
//
// the two worst cells in this crate, and invisible until the harness was widened past its
// original four ops. SciPy runs a single Chebyshev expansion and one `exp`.
//
// A general-order routine is the right implementation for general order and the wrong one
// for order 0 and 1, which are the orders callers actually ask for by name. `iv_scalar`
// keeps every other order; these two names stop paying for generality they never use.
//
// Coefficients are transcribed from scipy's `xsf/cephes/{i0,i1}.h` — the code the incumbent
// runs — so this moves our values TOWARD SciPy. They are extracted mechanically rather than
// retyped, and cross-checked against the existing series by
// `i0_i1_cephes_matches_the_iv_series`.

/// Chebyshev coefficients for `exp(-x) I0(x)` on `0 <= x <= 8`, in `x/2 - 2`.
#[allow(clippy::excessive_precision)]
const I0_CHEB_A: [f64; 30] = [
    -4.41534164647933937950E-18,
    3.33079451882223809783E-17,
    -2.43127984654795469359E-16,
    1.71539128555513303061E-15,
    -1.16853328779934516808E-14,
    7.67618549860493561688E-14,
    -4.85644678311192946090E-13,
    2.95505266312963983461E-12,
    -1.72682629144155570723E-11,
    9.67580903537323691224E-11,
    -5.18979560163526290666E-10,
    2.65982372468238665035E-9,
    -1.30002500998624804212E-8,
    6.04699502254191894932E-8,
    -2.67079385394061173391E-7,
    1.11738753912010371815E-6,
    -4.41673835845875056359E-6,
    1.64484480707288970893E-5,
    -5.75419501008210370398E-5,
    1.88502885095841655729E-4,
    -5.76375574538582365885E-4,
    1.63947561694133579842E-3,
    -4.32430999505057594430E-3,
    1.05464603945949983183E-2,
    -2.37374148058994688156E-2,
    4.93052842396707084878E-2,
    -9.49010970480476444210E-2,
    1.71620901522208775349E-1,
    -3.04682672343198398683E-1,
    6.76795274409476084995E-1,
];

/// Chebyshev coefficients for `exp(-x) sqrt(x) I0(x)` on `8 <= x < inf`, in `32/x - 2`.
#[allow(clippy::excessive_precision)]
const I0_CHEB_B: [f64; 25] = [
    -7.23318048787475395456E-18,
    -4.83050448594418207126E-18,
    4.46562142029675999901E-17,
    3.46122286769746109310E-17,
    -2.82762398051658348494E-16,
    -3.42548561967721913462E-16,
    1.77256013305652638360E-15,
    3.81168066935262242075E-15,
    -9.55484669882830764870E-15,
    -4.15056934728722208663E-14,
    1.54008621752140982691E-14,
    3.85277838274214270114E-13,
    7.18012445138366623367E-13,
    -1.79417853150680611778E-12,
    -1.32158118404477131188E-11,
    -3.14991652796324136454E-11,
    1.18891471078464383424E-11,
    4.94060238822496958910E-10,
    3.39623202570838634515E-9,
    2.26666899049817806459E-8,
    2.04891858946906374183E-7,
    2.89137052083475648297E-6,
    6.88975834691682398426E-5,
    3.36911647825569408990E-3,
    8.04490411014108831608E-1,
];

/// Chebyshev coefficients for `exp(-x) I1(x) / x` on `0 <= x <= 8`, in `x/2 - 2`.
#[allow(clippy::excessive_precision)]
const I1_CHEB_A: [f64; 29] = [
    2.77791411276104639959E-18,
    -2.11142121435816608115E-17,
    1.55363195773620046921E-16,
    -1.10559694773538630805E-15,
    7.60068429473540693410E-15,
    -5.04218550472791168711E-14,
    3.22379336594557470981E-13,
    -1.98397439776494371520E-12,
    1.17361862988909016308E-11,
    -6.66348972350202774223E-11,
    3.62559028155211703701E-10,
    -1.88724975172282928790E-9,
    9.38153738649577178388E-9,
    -4.44505912879632808065E-8,
    2.00329475355213526229E-7,
    -8.56872026469545474066E-7,
    3.47025130813767847674E-6,
    -1.32731636560394358279E-5,
    4.78156510755005422638E-5,
    -1.61760815825896745588E-4,
    5.12285956168575772895E-4,
    -1.51357245063125314899E-3,
    4.15642294431288815669E-3,
    -1.05640848946261981558E-2,
    2.47264490306265168283E-2,
    -5.29459812080949914269E-2,
    1.02643658689847095384E-1,
    -1.76416518357834055153E-1,
    2.52587186443633654823E-1,
];

/// Chebyshev coefficients for `exp(-x) sqrt(x) I1(x)` on `8 <= x < inf`, in `32/x - 2`.
#[allow(clippy::excessive_precision)]
const I1_CHEB_B: [f64; 25] = [
    7.51729631084210481353E-18,
    4.41434832307170791151E-18,
    -4.65030536848935832153E-17,
    -3.20952592199342395980E-17,
    2.96262899764595013876E-16,
    3.30820231092092828324E-16,
    -1.88035477551078244854E-15,
    -3.81440307243700780478E-15,
    1.04202769841288027642E-14,
    4.27244001671195135429E-14,
    -2.10154184277266431302E-14,
    -4.08355111109219731823E-13,
    -7.19855177624590851209E-13,
    2.03562854414708950722E-12,
    1.41258074366137813316E-11,
    3.25260358301548823856E-11,
    -1.89749581235054123450E-11,
    -5.58974346219658380687E-10,
    -3.83538038596423702205E-9,
    -2.63146884688951950684E-8,
    -2.51223623787020892529E-7,
    -3.88256480887769039346E-6,
    -1.10588938762623716291E-4,
    -9.76109749136146840777E-3,
    7.78576235018280120474E-1,
];

/// Largest `x` for which `exp(x)` is finite; beyond it Cephes splits the exponential in two
/// halves because `I0(x)` stays finite to about 713.99 while `exp(x)` alone overflows.
const CEPHES_MAXLOG: f64 = 7.09782712893383996732E2;

/// `I0(x)` by SciPy's own Chebyshev kernels.
fn i0_cephes(x: f64) -> f64 {
    let x = x.abs();
    if x <= 8.0 {
        let y = (x / 2.0) - 2.0;
        return x.exp() * cephes_chbevl(y, &I0_CHEB_A);
    }
    let cheb = cephes_chbevl(32.0 / x - 2.0, &I0_CHEB_B) / x.sqrt();
    if x > CEPHES_MAXLOG {
        let e = (x / 2.0).exp();
        return e * cheb * e;
    }
    x.exp() * cheb
}

/// `I1(x)` by SciPy's own Chebyshev kernels. Odd in `x`, so the sign is restored at the end.
fn i1_cephes(x: f64) -> f64 {
    let z = x.abs();
    let out = if z <= 8.0 {
        let y = (z / 2.0) - 2.0;
        cephes_chbevl(y, &I1_CHEB_A) * z * z.exp()
    } else {
        let cheb = cephes_chbevl(32.0 / z - 2.0, &I1_CHEB_B) / z.sqrt();
        if z > CEPHES_MAXLOG {
            let e = (z / 2.0).exp();
            e * cheb * e
        } else {
            z.exp() * cheb
        }
    };
    if x < 0.0 { -out } else { out }
}

/// Scalar convenience function for I_0(x).
#[must_use]
pub fn i0_scalar(x: f64) -> f64 {
    if BESSEL_I01_CEPHES.load(std::sync::atomic::Ordering::Relaxed) {
        BESSEL_I01_CEPHES_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return i0_cephes(x);
    }
    iv_scalar(0.0, x)
}

/// Scalar convenience function for I_1(x).
#[must_use]
pub fn i1_scalar(x: f64) -> f64 {
    if BESSEL_I01_CEPHES.load(std::sync::atomic::Ordering::Relaxed) {
        BESSEL_I01_CEPHES_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return i1_cephes(x);
    }
    iv_scalar(1.0, x)
}

/// Use SciPy's own Chebyshev kernels for `K0`/`K1` (`true`, shipping) instead of routing
/// them through the general order-v `kv_scalar`.
///
/// NOT bit-identical and not intended to be: it is a different approximation, and it is the
/// one the incumbent uses, so it moves our values toward SciPy. Accuracy against the live
/// SciPy arm is the contract, checked in the same invocation by `perf_special_vs_scipy`;
/// the two forms are cross-checked against each other by `k0_k1_cephes_matches_the_kv_path`.
pub static BESSEL_K01_CEPHES: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);
/// Evaluations that took the Chebyshev arm — "enabled" is not "took effect".
///
/// Accuracy contract: a diagnostic counter, not an A/B lever. It has no arms to preserve
/// anything between; incrementing it changes no numeric result.
pub static BESSEL_K01_CEPHES_HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

// ── Cephes K0/K1 Chebyshev kernels ───────────────────────────────────────────────────
//
// WHY. `k0`/`k1` delegate to `kv_scalar(0.0, x)` / `kv_scalar(1.0, x)`, the GENERAL order-v
// routine — the same defect that made `i0`/`i1` 12x. It computes the SCALED `K_v·e^z`
// through a general-order path and then multiplies by `exp(-z)`. Measured against live
// SciPy on the identical fixture, instructions per element:
//
//     k1   fsci 808.1   scipy 330.0   2.45x
//     k0   fsci 484.8   scipy 331.7   1.46x
//
// AND THE KERNEL WAS ALREADY HERE, which is the actual finding. `K0_A`/`K0_B`/`K1_A`/`K1_B`
// and `cephes_chbevl` have been in this file all along, used by `k0e_cephes` — the exp-SCALED
// form. So `k0` was not missing a fast path; it was reaching order 0 through the general-order
// dispatch and then multiplying `k0e_cephes(x)` by `exp(-x)`, having just multiplied the
// x <= 2 branch by `exp(x)`. That branch paid TWO exponentials and a rounding round-trip to
// get back where it started. Cephes computes K0 below 2 with no exponential at all.
//
// I STARTED BY TRANSCRIBING THE COEFFICIENTS FROM `xsf/cephes/{k0,k1}.h` AND THREW THEM AWAY:
// they were byte-for-byte duplicates of tables three hundred lines up. The same check caught
// the same mistake on `y0` — GREP FOR AN EXISTING SIBLING BEFORE FETCHING ANYTHING. The only
// genuinely new tables in this file are `I0_CHEB_*`/`I1_CHEB_*`, which had no predecessor.
//
// Cross-checked against the `kv` path it replaces by `k0_k1_cephes_matches_the_kv_path`.

/// `K0(x)` for `x > 0`, UNSCALED, reusing the Chebyshev tables already in this file.
///
/// NO NEW COEFFICIENTS. `K0_A`/`K0_B` and `cephes_chbevl` were already here, used by
/// `k0e_cephes` — the exp-SCALED form. The defect was never a missing kernel: it was that
/// `kv_scalar` reached order 0 through the general-order dispatch and then multiplied
/// `k0e_cephes(x)` by `exp(-x)`, while `k0e_cephes` had just multiplied the x <= 2 branch by
/// `exp(x)`. That branch therefore paid TWO exponentials and a rounding round-trip to
/// arrive back where it started. Cephes computes `K0` below 2 with no exponential at all.
fn k0_cephes(x: f64) -> f64 {
    if x <= 2.0 {
        return cephes_chbevl(x * x - 2.0, &K0_A) - (0.5 * x).ln() * i0_scalar(x);
    }
    (-x).exp() * cephes_chbevl(8.0 / x - 2.0, &K0_B) / x.sqrt()
}

/// `K1(x)` for `x > 0`, UNSCALED. Same story as [`k0_cephes`].
fn k1_cephes(x: f64) -> f64 {
    if x <= 2.0 {
        return (0.5 * x).ln() * i1_scalar(x) + cephes_chbevl(x * x - 2.0, &K1_A) / x;
    }
    (-x).exp() * cephes_chbevl(8.0 / x - 2.0, &K1_B) / x.sqrt()
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
    map_real_input("k0", z, mode, 1 << 12, |x| k0_order_scalar(x, mode))
}

/// `K0(x)` with the domain handling `kv_scalar` performs, but the order-0 kernel.
///
/// The zero and negative-x behaviour is taken from `kv_scalar` verbatim rather than from
/// Cephes, because it is this crate's contract — `Hardened` mode returns a domain error
/// where Cephes returns NaN — and a faster kernel is not a licence to change it.
fn k0_order_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if !BESSEL_K01_CEPHES.load(std::sync::atomic::Ordering::Relaxed) {
        return kv_scalar(0.0, x, mode);
    }
    if x.is_nan() {
        return Ok(f64::NAN);
    }
    if x <= 0.0 {
        if x == 0.0 {
            return Ok(f64::INFINITY);
        }
        return domain_error_by_mode("kv", mode, format!("v=0,z={x}"), "kv requires z > 0");
    }
    BESSEL_K01_CEPHES_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(k0_cephes(x))
}

/// `K1(x)` with the domain handling `kv_scalar` performs, but the order-1 kernel.
fn k1_order_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if !BESSEL_K01_CEPHES.load(std::sync::atomic::Ordering::Relaxed) {
        return kv_scalar(1.0, x, mode);
    }
    if x.is_nan() {
        return Ok(f64::NAN);
    }
    if x <= 0.0 {
        if x == 0.0 {
            return Ok(f64::INFINITY);
        }
        return domain_error_by_mode("kv", mode, format!("v=1,z={x}"), "kv requires z > 0");
    }
    BESSEL_K01_CEPHES_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(k1_cephes(x))
}

/// Modified Bessel function of the second kind of order 1: K_1(z).
///
/// Convenience wrapper for kv(1, z). Matches `scipy.special.k1(z)`.
pub fn k1(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("k1", z, mode, 1 << 12, |x| k1_order_scalar(x, mode))
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
    map_real_input("i0e", z, mode, 1 << 14, |x| Ok(i0e_scalar(x)))
}

/// Exponentially scaled modified Bessel function I_1: i1e(x) = I_1(x) * exp(-|x|).
///
/// Matches `scipy.special.i1e(x)`.
pub fn i1e(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("i1e", z, mode, 1 << 14, |x| Ok(i1e_scalar(x)))
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
    map_real_input("k0e", z, mode, 1 << 12, |x| Ok(k0e_scalar(x)))
}

/// Exponentially scaled modified Bessel function K_1: k1e(x) = K_1(x) * exp(x).
///
/// Matches `scipy.special.k1e(x)`.
pub fn k1e(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input("k1e", z, mode, 1 << 12, |x| Ok(k1e_scalar(x)))
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

/// Scalar: ln of the exponentially-scaled modified Bessel function of the
/// first kind, `ln(ive(v, x)) = ln(I_v(x)) - |x|`.
///
/// `ive_scalar` forms `iv_scalar(v, x) * exp(-|x|)`, but `iv_scalar` overflows
/// to +inf for `|x| > ~709`, making `ive` itself `NaN` there. This routine
/// stays finite for all `x` by switching to the DLMF 10.40.1 large-argument
/// asymptotic for `|x| >= 100`:
///
/// `I_v(x) ~ e^x/sqrt(2*pi*x) * sum_{k>=0} (-1)^k a_k(v)/x^k`,
/// `a_k(v) = (4v^2-1^2)(4v^2-3^2)...(4v^2-(2k-1)^2)/(k! 8^k)`,
///
/// so `ln(ive) = -0.5*ln(2*pi*x) + ln(sum)`. Below the threshold it returns the
/// exact `ive_scalar(v, x).ln()` (where `iv` does not overflow). Verified to
/// agree with `ive_scalar(v, x).ln()` across their overlap.
///
/// Returns `NaN` for NaN inputs; for `x = 0`, `ln(ive(0,0)) = 0` and
/// `-inf` for `v > 0`.
#[must_use]
pub fn log_ive_scalar(v: f64, x: f64) -> f64 {
    if v.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    let z = x.abs();
    if z == 0.0 {
        return if v == 0.0 { 0.0 } else { f64::NEG_INFINITY };
    }

    const Z_ASYMP: f64 = 100.0;
    if z < Z_ASYMP {
        // iv does not overflow below ~709, so the scaled value is exact here…
        let ive = ive_scalar(v, x);
        if ive > 0.0 && ive.is_finite() {
            return ive.ln();
        }
        // …but I_v(x) UNDERFLOWS to 0 for large order v at moderate x (e.g.
        // I_267(12.6) ≈ 1e-320), making ive=0 and ive.ln()=-inf. Recover the
        // finite value from the ascending series in log space:
        //   ln I_v(x) = v·ln(x/2) − lnΓ(v+1) + ln Σ_m (x²/4)^m / (m!·(v+1)_m),
        // then ln(ive) = ln I_v(x) − |x|. frankenscipy.
        let half = z / 2.0;
        let x2_4 = half * half;
        let mut term = 1.0_f64;
        let mut sum = 1.0_f64;
        for m in 0..2000 {
            let mf = f64::from(m);
            term *= x2_4 / ((mf + 1.0) * (v + 1.0 + mf));
            sum += term;
            if term <= 1e-18 * sum {
                break;
            }
        }
        let ln_gamma_vp1 =
            crate::gamma::gammaln_scalar(v + 1.0, RuntimeMode::Strict).unwrap_or(f64::NAN);
        return v * half.ln() - ln_gamma_vp1 + sum.ln() - z;
    }

    // DLMF 10.40.1 asymptotic series for the scaled Bessel function.
    let mu = 4.0 * v * v;
    let mut a_k = 1.0;
    let mut series = 1.0;
    for k in 1..=6 {
        let kf = k as f64;
        a_k *= (mu - (2.0 * kf - 1.0).powi(2)) / (kf * 8.0);
        let term = a_k / z.powi(k);
        series += if k % 2 == 0 { term } else { -term };
    }
    -0.5 * (2.0 * PI * z).ln() + series.ln()
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
    let eval_real = |order: f64, x: f64| {
        bessel_derivative_real_scalar(function, order, x, derivative_order, mode, kind, rule)
    };
    let eval_complex = |order: f64, z_val: Complex64| {
        bessel_derivative_complex_scalar(function, order, z_val, derivative_order, mode, kind, rule)
    };
    match (v, z) {
        (SpecialTensor::RealScalar(order), SpecialTensor::RealScalar(x)) => {
            eval_real(*order, *x).map(SpecialTensor::RealScalar)
        }
        (SpecialTensor::RealVec(orders), SpecialTensor::RealScalar(x)) => {
            let x = *x;
            par_map_indices(orders.len(), |i| eval_real(orders[i], x)).map(SpecialTensor::RealVec)
        }
        (SpecialTensor::RealScalar(order), SpecialTensor::RealVec(xs)) => {
            let order = *order;
            par_map_indices(xs.len(), |i| eval_real(order, xs[i])).map(SpecialTensor::RealVec)
        }
        (SpecialTensor::RealVec(orders), SpecialTensor::RealVec(xs)) => {
            if orders.len() != xs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            par_map_indices(orders.len(), |i| eval_real(orders[i], xs[i]))
                .map(SpecialTensor::RealVec)
        }
        (SpecialTensor::RealScalar(order), SpecialTensor::ComplexScalar(z_val)) => {
            eval_complex(*order, *z_val).map(SpecialTensor::ComplexScalar)
        }
        (SpecialTensor::RealScalar(order), SpecialTensor::ComplexVec(zs)) => {
            let order = *order;
            par_map_indices(zs.len(), |i| eval_complex(order, zs[i])).map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::RealVec(orders), SpecialTensor::ComplexScalar(z_val)) => {
            let z_val = *z_val;
            par_map_indices(orders.len(), |i| eval_complex(orders[i], z_val))
                .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::RealVec(orders), SpecialTensor::ComplexVec(zs)) => {
            if orders.len() != zs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            par_map_indices(orders.len(), |i| eval_complex(orders[i], zs[i]))
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
    let eval_real = |order: f64, x: f64| {
        hankel_derivative_real_scalar(function, order, x, derivative_order, mode, kind)
    };
    let eval_complex = |order: f64, z_val: Complex64| {
        hankel_derivative_complex_scalar(function, order, z_val, derivative_order, mode, kind)
    };
    match (v, z) {
        (SpecialTensor::RealScalar(order), SpecialTensor::RealScalar(x)) => {
            eval_real(*order, *x).map(SpecialTensor::ComplexScalar)
        }
        (SpecialTensor::RealVec(orders), SpecialTensor::RealScalar(x)) => {
            let x = *x;
            par_map_indices(orders.len(), |i| eval_real(orders[i], x))
                .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::RealScalar(order), SpecialTensor::RealVec(xs)) => {
            let order = *order;
            par_map_indices(xs.len(), |i| eval_real(order, xs[i])).map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::RealVec(orders), SpecialTensor::RealVec(xs)) => {
            if orders.len() != xs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            par_map_indices(orders.len(), |i| eval_real(orders[i], xs[i]))
                .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::RealScalar(order), SpecialTensor::ComplexScalar(z_val)) => {
            eval_complex(*order, *z_val).map(SpecialTensor::ComplexScalar)
        }
        (SpecialTensor::RealScalar(order), SpecialTensor::ComplexVec(zs)) => {
            let order = *order;
            par_map_indices(zs.len(), |i| eval_complex(order, zs[i])).map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::RealVec(orders), SpecialTensor::ComplexScalar(z_val)) => {
            let z_val = *z_val;
            par_map_indices(orders.len(), |i| eval_complex(orders[i], z_val))
                .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::RealVec(orders), SpecialTensor::ComplexVec(zs)) => {
            if orders.len() != zs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            par_map_indices(orders.len(), |i| eval_complex(orders[i], zs[i]))
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
pub(crate) fn jv_scalar(v: f64, z: f64) -> f64 {
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
    // off). The z < 20 cutoff still left ~6-9 digits lost in z ∈ [14, 20)
    // (jv(1.5,17.2) was 6.5e-9 vs scipy ~1e-15). Crossover the series ↔
    // asymptotic/Miller at z = 14: below it the series keeps ≳10 digits, above
    // it the asymptotic/Miller path below is more accurate. Verified vs scipy
    // 1.17.1: |error| < 1e-10 for v ∈ [0.5, 50], z ∈ [0.25, 60]
    // (frankenscipy-…); was up to 6.5e-9.
    if az < 14.0 {
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
pub(crate) fn iv_scalar(v: f64, z: f64) -> f64 {
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

    // The summand peaks near k ≈ (√(v²+z²) − v)/2, which reaches a few hundred
    // for large order with z ≲ v² (iv(100,500) peaks at ~305 terms); cap well
    // past that so the tail is captured before the relative-convergence break.
    //
    // TERM RATIO, NOT A LOG PER TERM. Successive terms of this series differ by the exact
    // factor `(z²/4) / ((k+1)(v+k+1))`, so the whole sum needs ONE `exp` for the leading
    // term and nothing transcendental afterwards. The previous form carried the term in
    // logs and paid an `exp` AND THREE `ln` calls on every iteration — one of them,
    // `quarter_z2.ln()`, loop-invariant and recomputed each time. Profiled against live
    // SciPy, `exp` and `log` were 79% of `iv`'s 3715.4 instructions per element.
    //
    // Every term here is positive, so accumulating them directly cannot cancel; the
    // multiplicative form is at least as accurate as re-exponentiating a running logarithm,
    // which rounds once per term.
    if IV_SERIES_TERM_RATIO.load(std::sync::atomic::Ordering::Relaxed) {
        IV_SERIES_TERM_RATIO_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut term = log_first.exp();
        for k in 0..1000 {
            sum += term;
            if term < 1e-16 * sum && k > 10 {
                break;
            }
            let kf = k as f64;
            term *= quarter_z2 / ((kf + 1.0) * (v + kf + 1.0));
        }
    } else {
        let mut log_term = log_first;
        for k in 0..1000 {
            let term = log_term.exp();
            sum += term;

            if term < 1e-16 * sum && k > 10 {
                break;
            }

            let kf = k as f64;
            log_term += quarter_z2.ln() - (kf + 1.0).ln() - (v + kf + 1.0).ln();
        }
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
    let result = kv_scaled_value(v.abs(), z) * (-z).exp();
    if result.is_finite() {
        return Ok(result);
    }
    // The SCALED value K_v·e^z overflowed at large order (e.g. K_500(100)·e^100
    // ≈ 7e322) even though K_v itself is representable (2.7e279). Rebuild K_v
    // UNSCALED via the upward recurrence K_{ν+1} = K_{ν-1} + (2ν/z)K_ν (stable
    // for K), which grows from the tiny base orders to the finite K_v without
    // the e^z inflation. frankenscipy.
    let va = v.abs();
    let v0 = va.fract();
    let steps = va.floor() as u64;
    let base0 = kv_scaled_value(v0, z) * (-z).exp();
    if steps == 0 {
        return Ok(base0);
    }
    let base1 = kv_scaled_value(v0 + 1.0, z) * (-z).exp();
    let (mut k_prev, mut k_curr) = (base0, base1);
    for i in 1..steps {
        let order = v0 + i as f64;
        let k_next = k_prev + 2.0 * order / z * k_curr;
        k_prev = k_curr;
        k_curr = k_next;
    }
    Ok(k_curr)
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
    // Half-integer order (v = n + 1/2, the spherical-Bessel family): K_v has a
    // CLOSED FORM, so build it with an exact upward recurrence instead of the
    // ~96-point Gauss quadrature. Scaled base K_{1/2}·e^z = sqrt(π/(2z)); the
    // recurrence K_{ν+1}=K_{ν-1}+(2ν/z)K_ν holds identically for the e^z-scaled
    // values. This is analytically exact (more accurate than the integral) and
    // ~10x cheaper — fsci's jv/iv already beat SciPy ~14x, this closes kv's gap.
    if v_abs.fract() == 0.5 {
        let base0 = (std::f64::consts::PI / (2.0 * z)).sqrt(); // K_{1/2}·e^z
        let n_half = (v_abs - 0.5) as u64; // 0 → K_{1/2}, 1 → K_{3/2}, ...
        if n_half == 0 {
            return base0;
        }
        let base1 = base0 * (1.0 + 1.0 / z); // K_{3/2}·e^z
        let (mut k_prev, mut k_curr) = (base0, base1);
        for i in 1..n_half {
            let nu = 0.5 + i as f64; // generating K_{ν+1} from K_{ν-1}, K_ν
            let k_next = k_prev + 2.0 * nu / z * k_curr;
            k_prev = k_curr;
            k_curr = k_next;
        }
        return k_curr;
    }
    // Other non-integer order: Temme's method (ascending series for z≤2, CF2
    // continued fraction for z>2) then the stable upward order recurrence — the
    // same algorithm SciPy/Cephes use. Self-validating and ~machine-accurate
    // (worst 4e-15 vs mpmath over the reachable domain), and far cheaper than the
    // 96-point Gauss quadrature it replaces. Fall back to the quadrature if a
    // pathological input ever yields a non-finite value, so accuracy never regresses.
    if v_abs.fract() != 0.0 {
        let k = kv_temme_scaled(v_abs, z);
        return if k.is_finite() {
            k
        } else {
            kv_integral_scaled(v_abs, z)
        };
    }
    // Integer order: the recurrence K_{n+1} = K_{n-1} + (2n/z)K_n is identical for
    // the e^z-scaled values, so build them from the scaled K_0, K_1.
    let k0 = k0e_cephes(z);
    let n = v_abs.round() as u32;
    if n == 0 {
        return k0;
    }
    let k1 = k1e_cephes(z);
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

fn k0e_cephes(x: f64) -> f64 {
    if x <= 2.0 {
        let y = cephes_chbevl(x * x - 2.0, &K0_A) - (0.5 * x).ln() * i0_scalar(x);
        y * x.exp()
    } else {
        cephes_chbevl(8.0 / x - 2.0, &K0_B) / x.sqrt()
    }
}

fn k1e_cephes(x: f64) -> f64 {
    if x <= 2.0 {
        let y = (0.5 * x).ln() * i1_scalar(x) + cephes_chbevl(x * x - 2.0, &K1_A) / x;
        y * x.exp()
    } else {
        cephes_chbevl(8.0 / x - 2.0, &K1_B) / x.sqrt()
    }
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
/// Cached 48-node Gauss–Legendre rule on [−1, 1] (standard `gauleg` Newton iteration).
/// The scaled K_v integrand is smooth and single-peaked, so a fixed 48-point rule per
/// sub-interval resolves it to ~5e-14 — replacing the ~1500-evaluation adaptive Simpson
/// that made `kv`/`kve`/`kn` ~95x slower than SciPy (per-element quadrature). frankenscipy-8qpyn
fn gauss_legendre_48() -> &'static ([f64; 48], [f64; 48]) {
    static GL48: std::sync::OnceLock<([f64; 48], [f64; 48])> = std::sync::OnceLock::new();
    GL48.get_or_init(|| {
        const N: usize = 48;
        let mut x = [0.0f64; N];
        let mut w = [0.0f64; N];
        let m = N.div_ceil(2);
        for i in 0..m {
            let mut zr = (std::f64::consts::PI * (i as f64 + 0.75) / (N as f64 + 0.5)).cos();
            let mut pp;
            loop {
                let mut p1 = 1.0f64;
                let mut p2 = 0.0f64;
                for j in 0..N {
                    let p3 = p2;
                    p2 = p1;
                    p1 = ((2.0 * j as f64 + 1.0) * zr * p2 - j as f64 * p3) / (j as f64 + 1.0);
                }
                pp = N as f64 * (zr * p1 - p2) / (zr * zr - 1.0);
                let dz = p1 / pp;
                zr -= dz;
                if dz.abs() < 1.0e-15 {
                    break;
                }
            }
            x[i] = -zr;
            x[N - 1 - i] = zr;
            w[i] = 2.0 / ((1.0 - zr * zr) * pp * pp);
            w[N - 1 - i] = w[i];
        }
        (x, w)
    })
}

/// Fixed 48-node Gauss–Legendre quadrature of `f` over `[a, b]`.
fn gauss48_integrate(f: &impl Fn(f64) -> f64, a: f64, b: f64) -> f64 {
    let (x, w) = gauss_legendre_48();
    let c1 = 0.5 * (b - a);
    let c2 = 0.5 * (b + a);
    let mut s = 0.0;
    for k in 0..48 {
        s += w[k] * f(c1 * x[k] + c2);
    }
    c1 * s
}

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
        gauss48_integrate(&integrand, 0.0, t_star) + gauss48_integrate(&integrand, t_star, upper)
    } else {
        gauss48_integrate(&integrand, 0.0, upper)
    }
}

/// Chebyshev coefficients for Temme's Γ₁ (NR `beschb`), valid for |x| ≤ 1/2.
const BESCHB_C1: [f64; 7] = [
    -1.142_022_680_371_168e0,
    6.516_511_267_073_7e-3,
    3.087_090_173_086e-4,
    -3.470_626_964_9e-6,
    6.943_766_4e-9,
    3.677_95e-11,
    -1.356e-13,
];
/// Chebyshev coefficients for Temme's Γ₂ (NR `beschb`), valid for |x| ≤ 1/2.
const BESCHB_C2: [f64; 8] = [
    1.843_740_587_300_905e0,
    -7.685_284_084_478_67e-2,
    1.271_927_136_654_6e-3,
    -4.971_736_704_2e-6,
    -3.312_611_98e-8,
    2.423_096e-10,
    -1.702e-13,
    -1.49e-15,
];

/// Clenshaw evaluation of a Chebyshev series on `[a, b]` (NR `chebev`).
fn chebev(a: f64, b: f64, c: &[f64], x: f64) -> f64 {
    let mut d = 0.0;
    let mut dd = 0.0;
    let y = (2.0 * x - a - b) / (b - a);
    let y2 = 2.0 * y;
    for j in (1..c.len()).rev() {
        let sv = d;
        d = y2 * d - dd + c[j];
        dd = sv;
    }
    y * d - dd + 0.5 * c[0]
}

/// Temme's Γ₁, Γ₂ and the reciprocal Γ(1±x) for |x| ≤ 1/2 (NR `beschb`). The
/// Chebyshev fits stay accurate as `x → 0`, where the direct `1/Γ` difference for
/// Γ₁ = (1/Γ(1−x) − 1/Γ(1+x))/(2x) would cancel catastrophically.
fn beschb(x: f64) -> (f64, f64, f64, f64) {
    let xx = 8.0 * x * x - 1.0;
    let gam1 = chebev(-1.0, 1.0, &BESCHB_C1, xx);
    let gam2 = chebev(-1.0, 1.0, &BESCHB_C2, xx);
    let gampl = gam2 - x * gam1;
    let gammi = gam2 + x * gam1;
    (gam1, gam2, gampl, gammi)
}

/// Scaled Macdonald function `K_v(z)·e^z` for real order `v ≥ 0`, `z > 0`, via
/// Temme's method (NR `bessik` K-part): the ascending series for `z ≤ 2`, the CF2
/// continued fraction for `z > 2`, then the stable upward order recurrence. This is
/// the algorithm SciPy/Cephes use; it replaces the 96-point Gauss quadrature for
/// general non-integer / non-half-integer order (both caught earlier by closed
/// forms). Self-validating (each branch converges to `EPS`) and ~machine-accurate
/// (worst 4e-15 vs mpmath 40-dps over the full reachable `(v, z)` domain).
fn kv_temme_scaled(v: f64, z: f64) -> f64 {
    const EPS: f64 = 1e-16;
    const MAXIT: usize = 10_000;
    let nl = (v + 0.5) as i64;
    let xmu = v - nl as f64;
    let xmu2 = xmu * xmu;
    let xi = 1.0 / z;
    let xi2 = 2.0 * xi;
    let mut rkmu;
    let mut rk1;
    if z <= 2.0 {
        // Temme ascending series for K_xmu, K_{xmu+1} (|xmu| ≤ 1/2).
        let x2 = 0.5 * z;
        let pimu = std::f64::consts::PI * xmu;
        let fact = if pimu.abs() < EPS {
            1.0
        } else {
            pimu / pimu.sin()
        };
        let mut d = -x2.ln();
        let e = xmu * d;
        let fact2 = if e.abs() < EPS { 1.0 } else { e.sinh() / e };
        let (gam1, gam2, gampl, gammi) = beschb(xmu);
        let mut ff = fact * (gam1 * e.cosh() + gam2 * fact2 * d);
        let mut summ = ff;
        let ee = e.exp();
        let mut p = 0.5 * ee / gampl;
        let mut q = 0.5 / (ee * gammi);
        let mut c = 1.0;
        d = x2 * x2;
        let mut sum1 = p;
        let mut i = 1usize;
        while i <= MAXIT {
            let fi = i as f64;
            ff = (fi * ff + p + q) / (fi * fi - xmu2);
            c *= d / fi;
            p /= fi - xmu;
            q /= fi + xmu;
            let delv = c * ff;
            summ += delv;
            sum1 += c * (p - fi * ff);
            if delv.abs() < summ.abs() * EPS {
                break;
            }
            i += 1;
        }
        let ez = z.exp();
        rkmu = summ * ez;
        rk1 = sum1 * xi2 * ez;
    } else {
        // CF2 (Lentz) for the scaled K_xmu, K_{xmu+1} directly.
        let mut b = 2.0 * (1.0 + z);
        let mut d = 1.0 / b;
        let mut delh = d;
        let mut h = d;
        let mut q1 = 0.0;
        let mut q2 = 1.0;
        let a1 = 0.25 - xmu2;
        let mut q = a1;
        let mut c = a1;
        let mut a = -a1;
        let mut s = 1.0 + q * delh;
        let mut i = 2usize;
        while i <= MAXIT {
            a -= 2.0 * (i as f64 - 1.0);
            c = -a * c / i as f64;
            let qnew = (q1 - b * q2) / a;
            q1 = q2;
            q2 = qnew;
            q += c * qnew;
            b += 2.0;
            d = 1.0 / (b + a * d);
            delh *= b * d - 1.0;
            h += delh;
            let dels = q * delh;
            s += dels;
            if (dels / s).abs() < EPS {
                break;
            }
            i += 1;
        }
        h *= a1;
        rkmu = (std::f64::consts::PI / (2.0 * z)).sqrt() / s; // already e^z-scaled
        rk1 = rkmu * (xmu + z + 0.5 - h) * xi;
    }
    // Stable upward recurrence in order: K_{ν+1} = (2ν/z)K_ν + K_{ν-1}.
    for i in 1..=nl {
        let rktemp = (xmu + i as f64) * xi2 * rk1 + rkmu;
        rkmu = rk1;
        rk1 = rktemp;
    }
    rkmu
}

// Unused adaptive-Simpson quadrature: this function, adaptive_simpson_inner and
// simpson_estimate form ONE self-contained unit that nothing outside them calls.
// RETAINED rather than deleted (frankenscipy-e2ve2) -- it is a complete
// alternative quadrature, not three stray helpers, and dropping it to silence
// dead_code is the mistake recorded against frankenscipy-iit9c. If the owner
// decides to remove it, remove all three together.
#[allow(dead_code)]
fn adaptive_simpson(f: &impl Fn(f64) -> f64, a: f64, b: f64, tol: f64, depth: u32) -> f64 {
    let fa = f(a);
    let fb = f(b);
    let c = 0.5 * (a + b);
    let fc = f(c);
    let whole = simpson_estimate(a, b, fa, fb, fc);
    adaptive_simpson_inner(f, a, b, tol, whole, (fa, fb, fc), depth)
}

#[allow(dead_code)]
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

#[allow(dead_code)]
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
    // Per-element real spherical Bessel kernel; pure in (order, x) given fixed kind/mode,
    // so the real array arms below fan out across cores bit-identically via par_map_indices.
    let eval = |order: f64, x: f64| -> Result<f64, SpecialError> {
        match kind {
            SphericalKind::Jn => spherical_jn_scalar(order, x, mode),
            SphericalKind::Yn => spherical_yn_scalar(order, x, mode),
            SphericalKind::In => spherical_in_scalar(order, x, mode),
            SphericalKind::Kn => spherical_kn_scalar(order, x, mode),
        }
    };
    match (n, z) {
        // Real-real cases - use existing scalar implementations
        (SpecialTensor::RealScalar(order), SpecialTensor::RealScalar(x)) => {
            eval(*order, *x).map(SpecialTensor::RealScalar)
        }
        (SpecialTensor::RealVec(orders), SpecialTensor::RealScalar(x)) => {
            let x = *x;
            par_map_indices(orders.len(), |i| eval(orders[i], x)).map(SpecialTensor::RealVec)
        }
        (SpecialTensor::RealScalar(order), SpecialTensor::RealVec(xs)) => {
            let order = *order;
            par_map_indices(xs.len(), |i| eval(order, xs[i])).map(SpecialTensor::RealVec)
        }
        (SpecialTensor::RealVec(orders), SpecialTensor::RealVec(xs)) => {
            if orders.len() != xs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            par_map_indices(orders.len(), |i| eval(orders[i], xs[i])).map(SpecialTensor::RealVec)
        }
        // Real n, complex z - use complex implementations
        (SpecialTensor::RealScalar(order), SpecialTensor::ComplexScalar(z_val)) => {
            spherical_bessel_complex_scalar(function, *order, *z_val, mode, kind)
                .map(SpecialTensor::ComplexScalar)
        }
        (SpecialTensor::RealScalar(order), SpecialTensor::ComplexVec(zs)) => {
            let order = *order;
            par_map_indices(zs.len(), |i| {
                spherical_bessel_complex_scalar(function, order, zs[i], mode, kind)
            })
            .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::RealVec(orders), SpecialTensor::ComplexScalar(z_val)) => {
            let z_val = *z_val;
            par_map_indices(orders.len(), |i| {
                spherical_bessel_complex_scalar(function, orders[i], z_val, mode, kind)
            })
            .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::RealVec(orders), SpecialTensor::ComplexVec(zs)) => {
            if orders.len() != zs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            par_map_indices(orders.len(), |i| {
                spherical_bessel_complex_scalar(function, orders[i], zs[i], mode, kind)
            })
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

pub(crate) fn spherical_jn_scalar(
    order: f64,
    x: f64,
    mode: RuntimeMode,
) -> Result<f64, SpecialError> {
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
    if n >= 2 && ax < n_f && x > 0.0 {
        // For x < n forward recurrence cancels; the half-integer Bessel identity
        //   j_n(x) = √(π/(2x))·J_{n+½}(x)
        // is stable AND ~4× faster than the Miller downward recurrence below
        // (byte-identical, incl. the x ≪ n tail: j_9(0.1)=1.5e-18, j_50(10)=2.2e-31,
        // ≤5e-17 rel). `jv_scalar` already carries the stable small-argument path.
        return (std::f64::consts::PI / (2.0 * x)).sqrt() * jv_scalar(n_f + 0.5, x);
    }
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

    // Stream log_term across k to drop the per-term gammaln(k+1). Since
    //   log_term(k)   = k·ln(x) − ln(k!) − lnΓ(a·k+b)
    //   log_term(k−1) = (k−1)·ln(x) − ln((k−1)!) − lnΓ(a·(k−1)+b),
    // the increment is ln(x) − ln(k) + (log_rgamma_k − log_rgamma_{k−1}) — one
    // `rgamma` per term (reusing the previous) plus a cheap `ln`, versus the old
    // rgamma + gammaln (~1.8× the gamma-class work). log_term stays moderate in
    // magnitude, so no accumulation drift; a fresh gammaln is used at k=0 and
    // whenever the chain isn't finite (the negligible Γ-overflow tail).
    let mut log_term_prev = f64::NEG_INFINITY;
    let mut log_rgamma_prev = f64::NEG_INFINITY;

    for k in 0..max_terms {
        let kf = k as f64;
        let akb = a.mul_add(kf, b);
        let log_rgamma = rgamma_log_nonnegative(akb)?;
        let log_term = if k > 0 && log_term_prev.is_finite() && log_rgamma_prev.is_finite() {
            log_term_prev + ln_x - kf.ln() + (log_rgamma - log_rgamma_prev)
        } else {
            kf * ln_x - crate::gamma::gammaln_scalar(kf + 1.0, RuntimeMode::Strict)? + log_rgamma
        };
        log_term_prev = log_term;
        log_rgamma_prev = log_rgamma;

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
    real_par_min: usize,
    kernel: F,
) -> SpecialResult
where
    F: Fn(f64) -> Result<f64, SpecialError> + Sync,
{
    match input {
        SpecialTensor::RealScalar(x) => kernel(*x).map(SpecialTensor::RealScalar),
        SpecialTensor::RealVec(values) => {
            // Each element is an independent scalar evaluation written to its own output
            // slot, so large arrays fan out across cores ABOVE a per-caller work gate
            // (real_par_min — Bessel kernels span ~28ns j0 to ~845ns kv, so the n/256 gate
            // over-subscribed the cheap ones 4-16x at n<=16k; BlackThrush A/B 2026-06-22).
            // Chunks are concatenated in element order and the first failing chunk's error
            // is returned in element order, so the result (value and first error) is
            // bit-identical to the sequential `values.iter().map(kernel).collect()`.
            let n = values.len();
            let nthreads = if n < real_par_min {
                1
            } else {
                std::thread::available_parallelism()
                    .map(std::num::NonZero::get)
                    .unwrap_or(1)
                    .min(n / 256)
                    .max(1)
            };
            if nthreads <= 1 {
                return values
                    .iter()
                    .copied()
                    .map(&kernel)
                    .collect::<Result<Vec<_>, _>>()
                    .map(SpecialTensor::RealVec);
            }
            let chunk = n.div_ceil(nthreads);
            let kernel = &kernel;
            let chunk_results: Vec<Result<Vec<f64>, SpecialError>> = std::thread::scope(|scope| {
                values
                    .chunks(chunk)
                    .map(|c| {
                        scope.spawn(move || {
                            c.iter()
                                .copied()
                                .map(kernel)
                                .collect::<Result<Vec<f64>, _>>()
                        })
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .map(|h| h.join().expect("bessel array worker panicked"))
                    .collect()
            });
            let mut out = Vec::with_capacity(n);
            for cr in chunk_results {
                out.extend(cr?);
            }
            Ok(SpecialTensor::RealVec(out))
        }
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

/// Evaluate `f(0..n)` into a `Vec<f64>`, parallel over index chunks for large `n`.
/// Bessel binary/ternary kernels (jn/yn recurrences, wright_bessel series) are expensive
/// per element and each index writes its own slot, so chunking across cores and
/// concatenating in index order is bit-identical to `(0..n).map(f).collect()` — including
/// returning the first failing index's error in index order.
fn par_map_indices<T, H>(n: usize, f: H) -> Result<Vec<T>, SpecialError>
where
    T: Send,
    H: Fn(usize) -> Result<T, SpecialError> + Sync,
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
            .map(|h| {
                h.join()
                    .expect("bessel binary/ternary array worker panicked")
            })
            .collect()
    });
    let mut out = Vec::with_capacity(n);
    for cr in chunk_results {
        out.extend(cr?);
    }
    Ok(out)
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
    F: Fn(f64, f64) -> Result<f64, SpecialError> + Sync,
{
    match (lhs, rhs) {
        (SpecialTensor::RealScalar(left), SpecialTensor::RealScalar(right)) => {
            kernel(*left, *right).map(SpecialTensor::RealScalar)
        }
        (SpecialTensor::RealVec(left), SpecialTensor::RealScalar(right)) => {
            let right = *right;
            par_map_indices(left.len(), |i| kernel(left[i], right)).map(SpecialTensor::RealVec)
        }
        (SpecialTensor::RealScalar(left), SpecialTensor::RealVec(right)) => {
            let left = *left;
            par_map_indices(right.len(), |i| kernel(left, right[i])).map(SpecialTensor::RealVec)
        }
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
            par_map_indices(left.len(), |i| kernel(left[i], right[i])).map(SpecialTensor::RealVec)
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

// Horner polynomial evaluators matching Cephes polevl/p1evl (p1evl assumes an implicit leading
// 1.0). Used by the Cephes rational j0 below. frankenscipy-9l5oo.
fn cephes_polevl(x: f64, coef: &[f64]) -> f64 {
    coef.iter().fold(0.0, |acc, &c| acc * x + c)
}
fn cephes_p1evl(x: f64, coef: &[f64]) -> f64 {
    coef.iter().fold(1.0, |acc, &c| acc * x + c)
}

pub(crate) fn cephes_chbevl(x: f64, coef: &[f64]) -> f64 {
    let mut iter = coef.iter();
    let mut b0 = *iter.next().expect("non-empty Chebyshev coefficient table");
    let mut b1 = 0.0;
    let mut b2 = 0.0;
    for &c in iter {
        b2 = b1;
        b1 = b0;
        b0 = x * b1 - b2 + c;
    }
    0.5 * (b0 - b2)
}

const J0_DR1: f64 = 5.78318596294678452118E0;
const J0_DR2: f64 = 3.04712623436620863991E1;
const J0_RP: [f64; 4] = [
    -4.79443220978201773821E9,
    1.95617491946556577543E12,
    -2.49248344360967716204E14,
    9.70862251047306323952E15,
];
const J0_RQ: [f64; 8] = [
    4.99563147152651017219E2,
    1.73785401676374683123E5,
    4.84409658339962045305E7,
    1.11855537045356834862E10,
    2.11277520115489217587E12,
    3.10518229857422583814E14,
    3.18121955943204943306E16,
    1.71086294081043136091E18,
];
const J0_PP: [f64; 7] = [
    7.96936729297347051624E-4,
    8.28352392107440799803E-2,
    1.23953371646414299388E0,
    5.44725003058768775090E0,
    8.74716500199817011941E0,
    5.30324038235394892183E0,
    9.99999999999999997821E-1,
];
const J0_PQ: [f64; 7] = [
    9.24408810558863637013E-4,
    8.56288474354474431428E-2,
    1.25352743901058953537E0,
    5.47097740330417105182E0,
    8.76190883237069594232E0,
    5.30605288235394617618E0,
    1.00000000000000000218E0,
];
const J0_QP: [f64; 8] = [
    -1.13663838898469149931E-2,
    -1.28252718670509318512E0,
    -1.95539544257735972385E1,
    -9.32060152123768231369E1,
    -1.77681167980488050595E2,
    -1.47077505154951170175E2,
    -5.14105326766599330220E1,
    -6.05014350600728481186E0,
];
const J0_QQ: [f64; 7] = [
    6.43178256118178023184E1,
    8.56430025976980587198E2,
    3.88240183605401609683E3,
    7.24046774195652478189E3,
    5.93072701187316984827E3,
    2.06209331660327847417E3,
    2.42005740240291393179E2,
];
const Y0_YP: [f64; 8] = [
    1.55924367855235737965E4,
    -1.46639295903971606143E7,
    5.43526477051876500413E9,
    -9.82136065717911466409E11,
    8.75906394395366999549E13,
    -3.46628303384729719441E15,
    4.42733268572569800351E16,
    -1.84950800436986690637E16,
];
const Y0_YQ: [f64; 7] = [
    1.04128353664259848412E3,
    6.26107330137134956842E5,
    2.68919633393814121987E8,
    8.64002487103935000337E10,
    2.02979612750105546709E13,
    3.17157752842975028269E15,
    2.50596256172653059228E17,
];
const SQRT2OPI: f64 = 0.79788456080286535588; // √(2/π)
const SQRT1OPI: f64 = 0.56418958354775628695; // √(1/π)
const THREE_PI_OVER_FOUR: f64 = 2.35619449019234492885;

const J1_RP: [f64; 4] = [
    -8.99971225705559398224E8,
    4.52228297998194034323E11,
    -7.27494245221818276015E13,
    3.68295732863852883286E15,
];
const J1_RQ: [f64; 8] = [
    6.20836478118054335476E2,
    2.56987256757748830383E5,
    8.35146791431949253037E7,
    2.21511595479792499675E10,
    4.74914122079991414898E12,
    7.84369607876235854894E14,
    8.95222336184627338078E16,
    5.32278620332680085395E18,
];
const J1_PP: [f64; 7] = [
    7.62125616208173112003E-4,
    7.31397056940917570436E-2,
    1.12719608129684925192E0,
    5.11207951146807644818E0,
    8.42404590141772420927E0,
    5.21451598682361504063E0,
    1.00000000000000000254E0,
];
const J1_PQ: [f64; 7] = [
    5.71323128072548699714E-4,
    6.88455908754495404082E-2,
    1.10514232634061696926E0,
    5.07386386128601488557E0,
    8.39985554327604159757E0,
    5.20982848682361821619E0,
    9.99999999999999997461E-1,
];
const J1_QP: [f64; 8] = [
    5.10862594750176621635E-2,
    4.98213872951233449420E0,
    7.58238284132545283818E1,
    3.66779609360150777800E2,
    7.10856304998926107277E2,
    5.97489612400613639965E2,
    2.11688757100572135698E2,
    2.52070205858023719784E1,
];
const J1_QQ: [f64; 7] = [
    7.42373277035675149943E1,
    1.05644886038262816351E3,
    4.98641058337653607651E3,
    9.56231892404756170795E3,
    7.99704160447350683650E3,
    2.82619278517639096600E3,
    3.36093607810698293419E2,
];
const J1_Z1: f64 = 1.46819706421238932572E1;
const J1_Z2: f64 = 4.92184563216946036703E1;

#[allow(clippy::excessive_precision)] // coefficients verbatim from SciPy/Cephes
const K0_A: [f64; 10] = [
    1.37446543561352307156E-16,
    4.25981614279661018399E-14,
    1.03496952576338420167E-11,
    1.90451637722020886025E-9,
    2.53479107902614945675E-7,
    2.28621210311945178607E-5,
    1.26461541144692592338E-3,
    3.59799365153615016266E-2,
    3.44289899924628486886E-1,
    -5.35327393233902768720E-1,
];
#[allow(clippy::excessive_precision)] // coefficients verbatim from SciPy/Cephes
const K0_B: [f64; 25] = [
    5.30043377268626276149E-18,
    -1.64758043015242134646E-17,
    5.21039150503902756861E-17,
    -1.67823109680541210385E-16,
    5.51205597852431940784E-16,
    -1.84859337734377901440E-15,
    6.34007647740507060557E-15,
    -2.22751332699166985548E-14,
    8.03289077536357521100E-14,
    -2.98009692317273043925E-13,
    1.14034058820847496303E-12,
    -4.51459788337394416547E-12,
    1.85594911495471785253E-11,
    -7.95748924447710747776E-11,
    3.57739728140030116597E-10,
    -1.69753450938905987466E-9,
    8.57403401741422608519E-9,
    -4.66048989768794782956E-8,
    2.76681363944501510342E-7,
    -1.83175552271911948767E-6,
    1.39498137188764993662E-5,
    -1.28495495816278026384E-4,
    1.56988388573005337491E-3,
    -3.14481013119645005427E-2,
    2.44030308206595545468E0,
];
#[allow(clippy::excessive_precision)] // coefficients verbatim from SciPy/Cephes
const K1_A: [f64; 11] = [
    -7.02386347938628759343E-18,
    -2.42744985051936593393E-15,
    -6.66690169419932900609E-13,
    -1.41148839263352776110E-10,
    -2.21338763073472585583E-8,
    -2.43340614156596823496E-6,
    -1.73028895751305206302E-4,
    -6.97572385963986435018E-3,
    -1.22611180822657148235E-1,
    -3.53155960776544875667E-1,
    1.52530022733894777053E0,
];
#[allow(clippy::excessive_precision)] // coefficients verbatim from SciPy/Cephes
const K1_B: [f64; 25] = [
    -5.75674448366501715755E-18,
    1.79405087314755922667E-17,
    -5.68946255844285935196E-17,
    1.83809354436663880070E-16,
    -6.05704724837331885336E-16,
    2.03870316562433424052E-15,
    -7.01983709041831346144E-15,
    2.47715442448130437068E-14,
    -8.97670518232499435011E-14,
    3.34841966607842919884E-13,
    -1.28917396095102890680E-12,
    5.13963967348173025100E-12,
    -2.12996783842756842877E-11,
    9.21831518760500529508E-11,
    -4.19035475934189648750E-10,
    2.01504975519703286596E-9,
    -1.03457624656780970260E-8,
    5.74108412545004946722E-8,
    -3.50196060308781257119E-7,
    2.40648494783721712015E-6,
    -1.93619797416608296024E-5,
    1.95215518471351631108E-4,
    -2.85781685962277938680E-3,
    1.03923736576817238437E-1,
    2.72062619048444266945E0,
];

// Cephes rational J0 — scipy's xsf wraps these EXACT coefficients, so this matches
// scipy.special.j0 to the bit while replacing the O(~25-term) convergence-loop power series
// with a fixed rational (|x|≤5) + asymptotic modulus/phase (|x|>5). frankenscipy-9l5oo.
fn j0_core(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return 0.0;
    }
    let ax = x.abs(); // J0 is even
    if ax <= 5.0 {
        if ax < 3.0e-8 {
            return 1.0; // x²/4 below machine epsilon (covers x = 0: J0(0) = 1)
        }
        let z = ax * ax;
        if ax < 1.0e-5 {
            return 1.0 - z / 4.0;
        }
        let p = (z - J0_DR1) * (z - J0_DR2);
        return p * cephes_polevl(z, &J0_RP) / cephes_p1evl(z, &J0_RQ);
    }
    let w = 5.0 / ax;
    let qz = 25.0 / (ax * ax);
    let p = cephes_polevl(qz, &J0_PP) / cephes_polevl(qz, &J0_PQ);
    let q = cephes_polevl(qz, &J0_QP) / cephes_p1evl(qz, &J0_QQ);
    if ax < 10.0 {
        let xn = ax - std::f64::consts::FRAC_PI_4;
        (p * xn.cos() - w * q * xn.sin()) * SQRT2OPI / ax.sqrt()
    } else {
        ((p + w * q) * ax.cos() + (p - w * q) * ax.sin()) * SQRT1OPI / ax.sqrt()
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
    let ans = if ax <= 5.0 {
        let z = ax * ax;
        let r = cephes_polevl(z, &J1_RP) / cephes_p1evl(z, &J1_RQ);
        r * ax * (z - J1_Z1) * (z - J1_Z2)
    } else {
        let w = 5.0 / ax;
        let z = w * w;
        let p = cephes_polevl(z, &J1_PP) / cephes_polevl(z, &J1_PQ);
        let q = cephes_polevl(z, &J1_QP) / cephes_p1evl(z, &J1_QQ);
        let xn = ax - THREE_PI_OVER_FOUR;
        (p * xn.cos() - w * q * xn.sin()) * SQRT2OPI / ax.sqrt()
    };

    if x < 0.0 { -ans } else { ans }
}

/// Route `Y0`/`Y1` above x = 5 through SciPy's own large-argument Cephes form (`true`,
/// shipping) instead of a harmonic series and the general order-v asymptotic.
///
/// NOT bit-identical and not intended to be: it is a different approximation, and it is the
/// one the incumbent uses, so it moves our values toward SciPy. Accuracy against the live
/// SciPy arm is the contract, checked in the same invocation by `perf_special_vs_scipy`;
/// the arms are also cross-checked against each other by
/// `y0_y1_cephes_large_matches_the_series`.
pub static BESSEL_Y01_CEPHES_LARGE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);
/// Evaluations that took the large-argument Cephes arm — "enabled" is not "took effect".
///
/// Accuracy contract: a diagnostic counter, not an A/B lever. It has no arms to preserve
/// anything between; incrementing it changes no numeric result.
pub static BESSEL_Y01_CEPHES_LARGE_HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

// ── Cephes Y0/Y1 large-argument form ─────────────────────────────────────────────────
//
// WHY. Measured against live SciPy on the identical fixture (x in [0.01, 30]), instructions
// per element: y0 757.2 against 232.8 (3.25x) and y1 827.1 against 232.4 (3.56x). Above
// x = 5 we ran a convergence-controlled harmonic series out to x = 14 and then
// `yv_asymptotic`, the GENERAL order-v routine — the same shape that made `i0`/`i1` 12x.
// SciPy runs two rational functions in 25/x^2 and one sin/cos, for every x > 5.
//
// THE COEFFICIENTS ARE ALREADY HERE AND ALREADY TRUSTED. Cephes evaluates `Y0` above 5 from
// the SAME `PP/PQ/QP/QQ` tables as `J0`, differing only in how the phase is combined: J0
// takes `p·cos(xn) - w·q·sin(xn)`, Y0 takes `p·sin(xn) + w·q·cos(xn)`. Those tables are
// already in this file, already used by `j0_core`/`j1_core`, and `j0`/`j1` already measure
// at parity with SciPy (1.06x and 0.88x). So this adds NO transcribed constants and carries
// none of the risk the `i0`/`i1` tables did.
//
// SCOPED TO x > 5 DELIBERATELY. Cephes also has a dedicated small-argument branch for `Y1`
// (its own `YP`/`YQ`) which is NOT in this tree; adding it means transcribing new
// coefficients, so `Y1` below 5 keeps its power series and is left as named, measured work
// rather than half-done here.

// ── Cephes Y1 small-argument branch, 0 < x <= 5 ──────────────────────────────────────
//
// WHY. `y1` was the last cell in this family still on a power series below x = 5, and it is
// the only one of `y0`/`y1` that is not bit-identical to SciPy: max_abs 1.42e-14. `y0` has
// had Cephes' small-argument rational all along and reads exactly 0. Localising the
// difference: above 5 both functions already take the same large-argument form, so the
// residual is entirely the x <= 5 branch.
//
// WHAT THE INCUMBENT ACTUALLY RUNS, checked rather than assumed. Profiling SciPy's `y1`
// shows `xsf::cyl_bessel_y1` and `xsf::cephes::j1` — NOT `xsf::cephes::y1`. The structure
// `Y1(x) = x·P(x²)/Q(x²) + (2/π)(J1(x)·ln x − 1/x)` is the Cephes one and it calls `j1`,
// which is what the profile shows, so the tables below are the right ones. That check
// mattered: `xsf/cephes/j0.h`'s `y0` carries an extra `x >= 10` phase form that SciPy's
// `cyl_bessel_y0` does NOT use, and adopting it would have BROKEN y0's bit-identity. The
// source on GitHub main is not automatically the code the installed incumbent runs.

/// Numerator of Cephes' `Y1` small-argument rational, in `x^2`. Degree 5, six terms.
#[allow(clippy::excessive_precision)]
const Y1_SMALL_YP: [f64; 6] = [
    1.26320474790178026440E9,
    -6.47355876379160291031E11,
    1.14509511541823727583E14,
    -8.12770255501325109621E15,
    2.02439475713594898196E17,
    -7.78877196265950026825E17,
];

/// Denominator of Cephes' `Y1` small-argument rational. MONIC — the leading 1.0 is not
/// stored, so this is evaluated with `cephes_p1evl` and NOT `cephes_polevl`.
#[allow(clippy::excessive_precision)]
const Y1_SMALL_YQ: [f64; 8] = [
    5.94301592346128195359E2,
    2.35564092943068577943E5,
    7.34811944459721705660E7,
    1.87601316108706159478E10,
    3.88231277496238566008E12,
    6.20557727146953693363E14,
    6.87141087355300489866E16,
    3.97270608116560655612E18,
];

/// `Y1(x)` for `0 < x <= 5`, the Cephes small-argument form. Uses `J1`, hence the shared
/// `j1_core`.
fn y1_cephes_small(x: f64) -> f64 {
    let z = x * x;
    let w = x * (cephes_polevl(z, &Y1_SMALL_YP) / cephes_p1evl(z, &Y1_SMALL_YQ));
    w + FRAC_2_PI * (j1_core(x) * x.ln() - 1.0 / x)
}

/// `Y0(x)` for `x > 5`, the Cephes large-argument form. Shares `J0`'s tables.
fn y0_cephes_large(x: f64) -> f64 {
    let w = 5.0 / x;
    let qz = 25.0 / (x * x);
    let p = cephes_polevl(qz, &J0_PP) / cephes_polevl(qz, &J0_PQ);
    let q = cephes_polevl(qz, &J0_QP) / cephes_p1evl(qz, &J0_QQ);
    let xn = x - std::f64::consts::FRAC_PI_4;
    (p * xn.sin() + w * q * xn.cos()) * SQRT2OPI / x.sqrt()
}

/// `Y1(x)` for `x > 5`, the Cephes large-argument form. Shares `J1`'s tables.
fn y1_cephes_large(x: f64) -> f64 {
    let w = 5.0 / x;
    let z = w * w;
    let p = cephes_polevl(z, &J1_PP) / cephes_polevl(z, &J1_PQ);
    let q = cephes_polevl(z, &J1_QP) / cephes_p1evl(z, &J1_QQ);
    let xn = x - THREE_PI_OVER_FOUR;
    (p * xn.sin() + w * q * xn.cos()) * SQRT2OPI / x.sqrt()
}

fn y0_core_positive(x: f64) -> f64 {
    if x <= 5.0 {
        return y0_cephes_small(x);
    }
    if BESSEL_Y01_CEPHES_LARGE.load(std::sync::atomic::Ordering::Relaxed) {
        BESSEL_Y01_CEPHES_LARGE_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return y0_cephes_large(x);
    }
    if x < 14.0 {
        y0_series_small(x)
    } else {
        yv_asymptotic(0.0, x)
    }
}

// Cephes' fixed rational for Y0 on (0, 5]. This evaluates
// Y0(x) - 2 ln(x) J0(x) / pi without the convergence-controlled harmonic
// series below; coefficients are verbatim from scipy/xsf's cephes/j0.h.
fn y0_cephes_small(x: f64) -> f64 {
    if x < 3.0e-8 {
        return Y0_YP[7] / Y0_YQ[6] + FRAC_2_PI * x.ln();
    }
    let z = x * x;
    cephes_polevl(z, &Y0_YP) / cephes_p1evl(z, &Y0_YQ) + FRAC_2_PI * x.ln() * j0_core(x)
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

/// `J1(x)` via its power series, accurate to machine precision for |x| < 8:
/// `J1(x) = (x/2) Σ_{k≥0} (-x²/4)^k / (k! (k+1)!)`.
fn j1_series_small(x: f64) -> f64 {
    let z = x * x * 0.25;
    let mut term = 1.0;
    let mut sum = 1.0;
    for k in 1..=BESSEL_SERIES_MAX_TERMS {
        let kf = k as f64;
        term *= -z / (kf * (kf + 1.0));
        sum += term;
        if term.abs() <= f64::EPSILON * sum.abs().max(1.0) {
            break;
        }
    }
    0.5 * x * sum
}

/// `Y1(x)` via its power series (DLMF 10.8.1), accurate to machine precision for
/// small `x > 0`:
/// `Y1 = (2/π) ln(x/2) J1(x) − 2/(π x) − (1/π)(x/2) Σ_{k≥0}
///       (ψ(k+1)+ψ(k+2)) (−x²/4)^k / (k!(k+1)!)`, with `ψ(k+1) = −γ + H_k`.
fn y1_series_small(x: f64) -> f64 {
    let w = x * x * 0.25;
    let j1 = j1_series_small(x);
    let mut term = 1.0; // (-w)^k / (k!(k+1)!), k = 0
    let mut hk = 0.0; // H_0
    let mut hk1 = 1.0; // H_1
    let mut sum = 0.0;
    for k in 0..=BESSEL_SERIES_MAX_TERMS {
        if k > 0 {
            let kf = k as f64;
            term *= -w / (kf * (kf + 1.0));
            hk += 1.0 / kf; // H_k
            hk1 += 1.0 / (kf + 1.0); // H_{k+1}
        }
        let psi_sum = -2.0 * EULER_MASCHERONI + hk + hk1; // ψ(k+1)+ψ(k+2)
        let addend = term * psi_sum;
        sum += addend;
        if k > 2 && addend.abs() <= f64::EPSILON * sum.abs().max(1.0) {
            break;
        }
    }
    FRAC_2_PI * (x.ln() - LN_2) * j1 - FRAC_2_PI / x - FRAC_2_PI * 0.25 * x * sum
}

fn y1_core_positive(x: f64) -> f64 {
    if BESSEL_Y01_CEPHES_LARGE.load(std::sync::atomic::Ordering::Relaxed) {
        BESSEL_Y01_CEPHES_LARGE_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return if x > 5.0 {
            y1_cephes_large(x)
        } else {
            y1_cephes_small(x)
        };
    }
    if x < 14.0 {
        // Power series (DLMF 10.8.1) for x<14 (≳11 digits); x≥14 uses the NR fit.
        y1_series_small(x)
    } else {
        yv_asymptotic(1.0, x)
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
    Complex64::new(FRAC_2_PI, 0.0) * log_term * j1
        - Complex64::new(FRAC_2_PI, 0.0) * z.recip()
        - Complex64::new(1.0 / PI, 0.0) * (z / 2.0) * s
}

/// Complex K_v(z) for real order v.
/// K_v = π/2 * (I_{-v} - I_v) / sin(vπ) for non-integer v.
pub(crate) fn complex_kv_scalar(
    v: f64,
    z: Complex64,
    _mode: RuntimeMode,
) -> Result<Complex64, SpecialError> {
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
        (SpecialTensor::RealVec(orders), SpecialTensor::RealScalar(x)) => {
            let x = *x;
            par_map_indices(orders.len(), |i| {
                hankel_real_scalar(function, orders[i], x, mode, kind)
            })
            .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::RealScalar(order), SpecialTensor::RealVec(xs)) => {
            let order = *order;
            par_map_indices(xs.len(), |i| {
                hankel_real_scalar(function, order, xs[i], mode, kind)
            })
            .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::RealVec(orders), SpecialTensor::RealVec(xs)) => {
            if orders.len() != xs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            par_map_indices(orders.len(), |i| {
                hankel_real_scalar(function, orders[i], xs[i], mode, kind)
            })
            .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::RealScalar(order), SpecialTensor::ComplexScalar(z_val)) => {
            hankel_complex_scalar(function, *order, *z_val, mode, kind)
                .map(SpecialTensor::ComplexScalar)
        }
        (SpecialTensor::RealScalar(order), SpecialTensor::ComplexVec(zs)) => {
            let order = *order;
            par_map_indices(zs.len(), |i| {
                hankel_complex_scalar(function, order, zs[i], mode, kind)
            })
            .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::RealVec(orders), SpecialTensor::ComplexScalar(z_val)) => {
            let z_val = *z_val;
            par_map_indices(orders.len(), |i| {
                hankel_complex_scalar(function, orders[i], z_val, mode, kind)
            })
            .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::RealVec(orders), SpecialTensor::ComplexVec(zs)) => {
            if orders.len() != zs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            par_map_indices(orders.len(), |i| {
                hankel_complex_scalar(function, orders[i], zs[i], mode, kind)
            })
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
        (SpecialTensor::RealVec(orders), SpecialTensor::RealScalar(x)) => {
            let x = *x;
            par_map_indices(orders.len(), |i| {
                hankel_real_scalar(function, orders[i], x, mode, kind)
                    .map(|value| scale_hankel(value, Complex64::new(x, 0.0), scale))
            })
            .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::RealScalar(order), SpecialTensor::RealVec(xs)) => {
            let order = *order;
            par_map_indices(xs.len(), |i| {
                let x = xs[i];
                hankel_real_scalar(function, order, x, mode, kind)
                    .map(|value| scale_hankel(value, Complex64::new(x, 0.0), scale))
            })
            .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::RealVec(orders), SpecialTensor::RealVec(xs)) => {
            if orders.len() != xs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            par_map_indices(orders.len(), |i| {
                let x = xs[i];
                hankel_real_scalar(function, orders[i], x, mode, kind)
                    .map(|value| scale_hankel(value, Complex64::new(x, 0.0), scale))
            })
            .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::RealScalar(order), SpecialTensor::ComplexScalar(z_val)) => {
            hankel_complex_scalar(function, *order, *z_val, mode, kind)
                .map(|value| scale_hankel(value, *z_val, scale))
                .map(SpecialTensor::ComplexScalar)
        }
        (SpecialTensor::RealScalar(order), SpecialTensor::ComplexVec(zs)) => {
            let order = *order;
            par_map_indices(zs.len(), |i| {
                let z_val = zs[i];
                hankel_complex_scalar(function, order, z_val, mode, kind)
                    .map(|value| scale_hankel(value, z_val, scale))
            })
            .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::RealVec(orders), SpecialTensor::ComplexScalar(z_val)) => {
            let z_val = *z_val;
            par_map_indices(orders.len(), |i| {
                hankel_complex_scalar(function, orders[i], z_val, mode, kind)
                    .map(|value| scale_hankel(value, z_val, scale))
            })
            .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::RealVec(orders), SpecialTensor::ComplexVec(zs)) => {
            if orders.len() != zs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            par_map_indices(orders.len(), |i| {
                let z_val = zs[i];
                hankel_complex_scalar(function, orders[i], z_val, mode, kind)
                    .map(|value| scale_hankel(value, z_val, scale))
            })
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
    // Per-element real-order Bessel kernel (J_v/Y_v/I_v/K_v + scaled). Pure function of
    // (order, x) given the fixed `kind`/`mode`, so the array arms below fan out across cores
    // bit-identically via par_map_indices.
    let eval = |order: f64, x: f64| -> Result<f64, SpecialError> {
        match kind {
            BesselKind::Jv => Ok(jv_scalar(order, x)),
            BesselKind::Yv => yv_scalar(order, x, mode),
            BesselKind::Jve => Ok(jv_scalar(order, x)),
            BesselKind::Yve => yv_scalar(order, x, mode),
            BesselKind::Iv => Ok(iv_scalar(order, x)),
            BesselKind::Kv => kv_scalar(order, x, mode),
            BesselKind::Ive => Ok(ive_scalar(order, x)),
            BesselKind::Kve => Ok(kve_scalar(order, x)),
        }
    };
    match (v, z) {
        // Real-real: delegate to existing real scalars
        (SpecialTensor::RealScalar(order), SpecialTensor::RealScalar(x)) => {
            eval(*order, *x).map(SpecialTensor::RealScalar)
        }
        (SpecialTensor::RealVec(orders), SpecialTensor::RealScalar(x)) => {
            let x = *x;
            par_map_indices(orders.len(), |i| eval(orders[i], x)).map(SpecialTensor::RealVec)
        }
        (SpecialTensor::RealScalar(order), SpecialTensor::RealVec(xs)) => {
            let order = *order;
            par_map_indices(xs.len(), |i| eval(order, xs[i])).map(SpecialTensor::RealVec)
        }
        (SpecialTensor::RealVec(orders), SpecialTensor::RealVec(xs)) => {
            if orders.len() != xs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::ShapeMismatch,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            par_map_indices(orders.len(), |i| eval(orders[i], xs[i])).map(SpecialTensor::RealVec)
        }
        // Real order, complex z
        (SpecialTensor::RealScalar(order), SpecialTensor::ComplexScalar(z_val)) => {
            bessel_complex_scalar(function, *order, *z_val, mode, kind)
                .map(SpecialTensor::ComplexScalar)
        }
        (SpecialTensor::RealScalar(order), SpecialTensor::ComplexVec(zs)) => {
            let order = *order;
            par_map_indices(zs.len(), |i| {
                bessel_complex_scalar(function, order, zs[i], mode, kind)
            })
            .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::RealVec(orders), SpecialTensor::ComplexScalar(z_val)) => {
            let z_val = *z_val;
            par_map_indices(orders.len(), |i| {
                bessel_complex_scalar(function, orders[i], z_val, mode, kind)
            })
            .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::RealVec(orders), SpecialTensor::ComplexVec(zs)) => {
            if orders.len() != zs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::ShapeMismatch,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            par_map_indices(orders.len(), |i| {
                bessel_complex_scalar(function, orders[i], zs[i], mode, kind)
            })
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
        // kve(v, z) = kv(v, z) · exp(z) — the FULL complex exponential. Using exp(Re z)
        // keeps the magnitude but drops the exp(i·Im z) phase rotation, diverging from scipy
        // for non-real z (frankenscipy-2bmxw). ive below genuinely uses a real scale
        // exp(-|Re z|), and jve/yve use exp(-|Im z|), matching scipy's distinct definitions.
        BesselKind::Kve => complex_kv_scalar(order, z, mode).map(|k| k * z.exp()),
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
    let ax = x.abs();
    if x.is_finite() && ax >= 1.0e-15 && (n < 2 || ax >= n as f64) {
        let mut j_prev = x.sin() / x;
        s.push(x * j_prev);
        if n >= 1 {
            let mut j_curr = x.sin() / (x * x) - x.cos() / x;
            s.push(x * j_curr);
            for k in 1..n {
                let next = (2.0 * k as f64 + 1.0) / x * j_curr - j_prev;
                s.push(x * next);
                j_prev = j_curr;
                j_curr = next;
            }
        }
    } else {
        for k in 0..=n {
            s.push(x * spherical_jn_nonneg(k, x));
        }
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
    if x.is_finite() && x != 0.0 {
        let mut y_prev = -x.cos() / x;
        c.push(x * y_prev);
        if n >= 1 {
            let mut y_curr = -x.cos() / (x * x) - x.sin() / x;
            c.push(x * y_curr);
            for k in 1..n {
                let next = (2.0 * k as f64 + 1.0) / x * y_curr - y_prev;
                c.push(x * next);
                y_prev = y_curr;
                y_curr = next;
            }
        }
    } else {
        for k in 0..=n {
            c.push(x * spherical_yn_nonneg(k, x));
        }
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

fn jn_derivative_nonnegative(n: u32, x: f64) -> f64 {
    jn_derivative_from_value(n, x, jn_nonnegative(n, x))
}

fn jn_derivative_from_value(n: u32, x: f64, value: f64) -> f64 {
    if n == 0 {
        -jn_nonnegative(1, x)
    } else {
        jn_nonnegative(n - 1, x) - (n as f64 / x) * value
    }
}

fn refine_bracketed_zero(
    f_at: impl Fn(f64) -> f64,
    mut lo: f64,
    mut hi: f64,
    mut f_lo: f64,
    mut f_hi: f64,
) -> f64 {
    if f_lo == 0.0 {
        return lo;
    }
    if f_hi == 0.0 {
        return hi;
    }
    if !(lo.is_finite()
        && hi.is_finite()
        && f_lo.is_finite()
        && f_hi.is_finite()
        && lo < hi
        && f_lo.signum() != f_hi.signum())
    {
        return 0.5 * (lo + hi);
    }

    for _ in 0..BESSEL_ZERO_MAX_REFINE_ITERS {
        let width = hi - lo;
        if width < BESSEL_ZERO_BRACKET_TOL {
            break;
        }

        let midpoint = 0.5 * (lo + hi);
        let secant = if f_hi != f_lo {
            lo - f_lo * width / (f_hi - f_lo)
        } else {
            f64::NAN
        };
        let guard = width * 0.125;
        let mut probe = if secant.is_finite() && secant > lo + guard && secant < hi - guard {
            secant
        } else {
            midpoint
        };
        let mut f_probe = f_at(probe);
        if !f_probe.is_finite() && probe != midpoint {
            probe = midpoint;
            f_probe = f_at(probe);
        }
        if !f_probe.is_finite() {
            break;
        }
        if f_probe == 0.0 {
            return probe;
        }

        if f_probe.signum() == f_lo.signum() {
            lo = probe;
            f_lo = f_probe;
        } else {
            hi = probe;
            f_hi = f_probe;
        }
    }

    0.5 * (lo + hi)
}

fn refine_bracketed_zero_with_derivative(
    eval_at: impl Fn(f64) -> (f64, f64),
    mut lo: f64,
    mut hi: f64,
    mut f_lo: f64,
    mut f_hi: f64,
) -> f64 {
    if f_lo == 0.0 {
        return lo;
    }
    if f_hi == 0.0 {
        return hi;
    }
    if !(lo.is_finite()
        && hi.is_finite()
        && f_lo.is_finite()
        && f_hi.is_finite()
        && lo < hi
        && f_lo.signum() != f_hi.signum())
    {
        return 0.5 * (lo + hi);
    }

    let mut probe = 0.5 * (lo + hi);
    for _ in 0..BESSEL_ZERO_MAX_REFINE_ITERS {
        let (f_probe, df_probe) = eval_at(probe);
        if !f_probe.is_finite() {
            return refine_bracketed_zero(|x| eval_at(x).0, lo, hi, f_lo, f_hi);
        }
        if f_probe == 0.0 {
            return probe;
        }

        let correction = if df_probe.is_finite() && df_probe != 0.0 {
            f_probe / df_probe
        } else {
            f64::NAN
        };
        if correction.is_finite() && correction.abs() < 0.5 * BESSEL_ZERO_BRACKET_TOL {
            return probe;
        }

        if f_probe.signum() == f_lo.signum() {
            lo = probe;
            f_lo = f_probe;
        } else {
            hi = probe;
            f_hi = f_probe;
        }

        let width = hi - lo;
        if width < BESSEL_ZERO_BRACKET_TOL {
            break;
        }

        let midpoint = 0.5 * (lo + hi);
        let newton = probe - correction;
        let secant = if f_hi != f_lo {
            lo - f_lo * width / (f_hi - f_lo)
        } else {
            f64::NAN
        };
        let guard = width * 0.0625;
        probe = if newton.is_finite() && newton > lo + guard && newton < hi - guard {
            newton
        } else if secant.is_finite() && secant > lo + guard && secant < hi - guard {
            secant
        } else {
            midpoint
        };
    }

    0.5 * (lo + hi)
}

/// Refine a bessel-zero initial guess by bracket expansion + safeguarded interpolation.
/// Walks outward from `initial` in steps of `step`, bounded by [`floor`,
/// initial+max_outward], until a sign change of `f_at` is found, then
/// refines to ~1e-12 precision. Returns `initial` unchanged if no
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
    refine_bracketed_zero(f_at, lo, hi, f_lo, f_hi)
}

fn bracket_and_refine_jn_zero(
    n: u32,
    initial: f64,
    floor: f64,
    step: f64,
    max_outward: f64,
) -> f64 {
    let mut lo = (initial - step).max(floor);
    let mut hi = (initial + step).max(lo + step);
    let mut f_lo = jn_nonnegative(n, lo);
    let mut f_hi = jn_nonnegative(n, hi);
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
        f_lo = jn_nonnegative(n, lo);
        f_hi = jn_nonnegative(n, hi);
        walked += step;
    }
    if !f_lo.is_finite() || !f_hi.is_finite() || f_lo.signum() == f_hi.signum() {
        return initial;
    }
    refine_bracketed_zero_with_derivative(
        |x| {
            let value = jn_nonnegative(n, x);
            (value, jn_derivative_from_value(n, x, value))
        },
        lo,
        hi,
        f_lo,
        f_hi,
    )
}

fn bisect_bracketed_zero(f_at: impl Fn(f64) -> f64, lo: f64, hi: f64) -> f64 {
    let f_lo = f_at(lo);
    let f_hi = f_at(hi);
    if !lo.is_finite()
        || !hi.is_finite()
        || !f_lo.is_finite()
        || !f_hi.is_finite()
        || f_lo.signum() == f_hi.signum()
    {
        return 0.5 * (lo + hi);
    }

    refine_bracketed_zero(f_at, lo, hi, f_lo, f_hi)
}

fn refine_bracketed_jnp_zero(n: u32, lo: f64, hi: f64) -> f64 {
    let f_lo = jn_derivative_nonnegative(n, lo);
    let f_hi = jn_derivative_nonnegative(n, hi);
    if !lo.is_finite()
        || !hi.is_finite()
        || !f_lo.is_finite()
        || !f_hi.is_finite()
        || f_lo.signum() == f_hi.signum()
    {
        return 0.5 * (lo + hi);
    }

    let n_sq = (n as f64) * (n as f64);
    refine_bracketed_zero_with_derivative(
        |x| {
            let value = jn_nonnegative(n, x);
            let derivative = jn_derivative_from_value(n, x, value);
            let second_derivative = (n_sq / (x * x) - 1.0) * value - derivative / x;
            (derivative, second_derivative)
        },
        lo,
        hi,
        f_lo,
        f_hi,
    )
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
    jnp_zeros_from_function_zeros(n, nt, &function_zeros)
}

fn jnp_zeros_from_function_zeros(n: u32, nt: usize, function_zeros: &[f64]) -> Vec<f64> {
    debug_assert!(n > 0);
    debug_assert!(function_zeros.len() >= nt);

    let mut out = Vec::with_capacity(nt);
    for idx in 0..nt {
        let lo = if idx == 0 {
            1.0e-6
        } else {
            function_zeros[idx - 1] + 1.0e-6
        };
        let hi = function_zeros[idx] - 1.0e-6;
        out.push(refine_bracketed_jnp_zero(n, lo, hi));
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
        let floor = if ki >= 2 { prev_zero + 1.0e-6 } else { 1.0e-6 };
        let zero = bracket_and_refine_jn_zero(n, initial, floor, 0.5, (n_f * 0.25).max(20.0));
        out.push(zero);
        prev_zero = zero;
    }
    out
}

fn refine_jn_zero_at(n: u32, ki: usize, prev_zero: f64) -> f64 {
    let n_f = n as f64;
    let four_n_sq = 4.0 * n_f * n_f;
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
    let floor = if ki >= 2 { prev_zero + 1.0e-6 } else { 1.0e-6 };
    bracket_and_refine_jn_zero(n, initial, floor, 0.5, (n_f * 0.25).max(20.0))
}

fn jn_zeros_until(n: u32, cutoff: f64) -> (Vec<f64>, f64) {
    if !cutoff.is_finite() {
        return (Vec::new(), f64::INFINITY);
    }

    let mut out = Vec::new();
    let mut prev_zero = 0.0_f64;
    let max_serial = cutoff.ceil().max(1.0) as usize + 64;
    for ki in 1..=max_serial {
        let zero = refine_jn_zero_at(n, ki, prev_zero);
        if zero > cutoff {
            return (out, zero);
        }
        out.push(zero);
        prev_zero = zero;
    }
    (out, f64::INFINITY)
}

/// Jahnke–Emden Lambda function `Λ_vi(x) = Γ(vi+1) J_vi(x) / (x/2)^vi` and its
/// derivative for the orders `vi = v − ⌊v⌋, 1 + v − ⌊v⌋, …, v`, returned as
/// `(values, derivatives)`.
///
/// Matches `scipy.special.lmbda(v, x)` (`v ≥ 0`). The sequence is generated by
/// the stable downward recurrence `Λ_{w-1} = Λ_w − x²/(4 w (w+1)) Λ_{w+1}`
/// (Miller's algorithm), normalized to the directly evaluated lowest order
/// `Λ_{v-⌊v⌋}`, so it stays accurate where `J` underflows for large order; the
/// derivative uses `Λ_vi'(x) = −x/(2(vi+1)) · Λ_{vi+1}(x)`. At `x = 0` it
/// reproduces SciPy's specfun result: `[1, 1, …]` for non-integer `v`, and
/// `[1, 0, …]` with derivative `[0, 0.5, 0, …]` for integer `v`.
#[must_use]
pub fn lmbda(v: f64, x: f64) -> (Vec<f64>, Vec<f64>) {
    // A NEGATIVE order is out of domain. scipy raises there:
    //   scipy.special.lmbda(-1.0, 2.0) -> ValueError: argument must be > 0.
    //   scipy.special.lmbda(-0.5, 2.0) -> ValueError: argument must be > 0.
    // This used to clamp with `.max(0.0)`, so v = -1.0 silently returned the
    // v = 0.0 answer and v = -0.5 returned the v0 = 0.5 series -- a plausible
    // value for a DIFFERENT input, with no signal (frankenscipy-21uhl).
    //
    // NaN rather than an Err because that is this crate's convention on the
    // bare-named surface: 623 public functions, only 16 return Result, and
    // every one of those is a `_scalar` variant paired with a NaN-returning
    // function of the same name. A lone Result here would be the outlier.
    //
    // TAKE THE MEASURED DOMAIN, NOT THE MESSAGE: scipy says "argument must be
    // > 0." but v == 0.0 is ACCEPTED and returns 0.22389077914123562 at x = 2.
    // Only v < 0 is rejected.
    if v < 0.0 {
        return (vec![f64::NAN], vec![f64::NAN]);
    }
    let n = v.floor().max(0.0) as usize;
    let v0 = v - v.floor();
    let count = n + 1;
    let mut vl = vec![0.0_f64; count];
    let mut dl = vec![0.0_f64; count];

    if x == 0.0 {
        if v0 == 0.0 {
            vl[0] = 1.0;
            if n >= 1 {
                dl[1] = 0.5;
            }
        } else {
            vl.iter_mut().for_each(|e| *e = 1.0);
        }
        return (vl, dl);
    }

    let half = x / 2.0;
    let m_extra = 40usize.max(n + 20 + x.abs() as usize);
    let top = n + m_extra;
    // Miller downward recurrence on Λ_{v0+i}; f[top+1]=0, f[top] seeded.
    let mut f = vec![0.0_f64; top + 2];
    f[top] = 1e-30;
    for i in (1..=top).rev() {
        let w = v0 + i as f64;
        f[i - 1] = f[i] - x * x / (4.0 * w * (w + 1.0)) * f[i + 1];
    }
    let direct0 = gamma_scalar(v0 + 1.0, RuntimeMode::Strict).unwrap_or(f64::NAN)
        * jv_scalar(v0, x)
        * half.powf(-v0);
    let scale = direct0 / f[0];
    for i in 0..count {
        let vi = v0 + i as f64;
        vl[i] = f[i] * scale;
        dl[i] = -(x / (2.0 * (vi + 1.0))) * (f[i + 1] * scale);
    }
    (vl, dl)
}

/// First `nt` zeros of `J_n`, `J_n'`, `Y_n`, and `Y_n'` for a fixed integer
/// order `n`, returned as `(jn, jnp, yn, ynp)`.
///
/// Matches `scipy.special.jnyn_zeros(n, nt)` — simply the four single-function
/// zero finders evaluated for the same order.
#[must_use]
pub fn jnyn_zeros(n: u32, nt: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    (
        jn_zeros(n, nt),
        jnp_zeros(n, nt),
        yn_zeros(n, nt),
        ynp_zeros(n, nt),
    )
}

/// First `nt` zeros of the integer-order Bessel functions `J_n` and `J_n'`
/// across all orders, sorted by magnitude.
///
/// Matches `scipy.special.jnjnp_zeros(nt)`, returning `(zo, n, m, t)`:
/// `zo` are the sorted zero values, `n` the order, `m` the 1-based serial number
/// of that zero within its `(order, function)`, and `t = 0` for a zero of `J_n`
/// or `t = 1` for a zero of `J_n'`. The `x = 0` zero of `J_0'` is included first
/// as `(0.0, n=0, m=0, t=1)`.
#[must_use]
pub fn jnjnp_zeros(nt: usize) -> (Vec<f64>, Vec<i32>, Vec<i32>, Vec<i32>) {
    if nt == 0 {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }

    if nt >= 16 {
        let max_envelope = nt + 2;
        let mut cutoff = (2.0 * (nt as f64).sqrt()).max(6.0);
        for _ in 0..4 {
            let n_limit = (cutoff.ceil() as usize + 2).min(max_envelope);
            let (mut cands, serial_frontier, order_frontier) =
                jnjnp_zero_candidates_below_cutoff(cutoff, n_limit, max_envelope);
            if cands.len() >= nt {
                sort_jnjnp_candidates(&mut cands);
                let actual_cutoff = cands[nt - 1].0;
                if serial_frontier > actual_cutoff && order_frontier > actual_cutoff {
                    return collect_jnjnp_candidates(&cands, nt);
                }
                cutoff = cutoff.max(actual_cutoff);
            }
            cutoff = cutoff * 1.35 + 2.0;
        }
    }

    jnjnp_zeros_rectangular_frontier(nt)
}

fn jnjnp_zeros_rectangular_frontier(nt: usize) -> (Vec<f64>, Vec<i32>, Vec<i32>, Vec<i32>) {
    let max_envelope = nt + 2;
    let root = (nt as f64).sqrt().ceil() as usize;
    let mut per = (root.div_ceil(2) + 3).clamp(1, max_envelope);
    let mut n_max = (2 * root).clamp(1, max_envelope);

    loop {
        let (mut cands, serial_frontier, order_frontier) =
            jnjnp_zero_candidates_with_frontier(per, n_max, max_envelope);
        let mut expand_per = per < max_envelope;
        let mut expand_n_max = n_max < max_envelope;
        if cands.len() >= nt {
            sort_jnjnp_candidates(&mut cands);
            let cutoff = cands[nt - 1].0;
            if (serial_frontier > cutoff && order_frontier > cutoff)
                || (per == max_envelope && n_max == max_envelope)
            {
                return collect_jnjnp_candidates(&cands, nt);
            }
            expand_per &= serial_frontier <= cutoff;
            expand_n_max &= order_frontier <= cutoff;
        } else if per == max_envelope && n_max == max_envelope {
            sort_jnjnp_candidates(&mut cands);
            return collect_jnjnp_candidates(&cands, nt);
        }

        let next_per = if expand_per {
            per.saturating_mul(2).min(max_envelope)
        } else {
            per
        };
        let next_n_max = if expand_n_max {
            n_max.saturating_mul(2).min(max_envelope)
        } else {
            n_max
        };
        if next_per == per && next_n_max == n_max {
            sort_jnjnp_candidates(&mut cands);
            return collect_jnjnp_candidates(&cands, nt);
        }
        per = next_per;
        n_max = next_n_max;
    }
}

fn jnp_zeros_from_function_zeros_until(
    n: u32,
    cutoff: f64,
    function_zeros: &[f64],
    function_frontier: f64,
) -> (Vec<f64>, f64) {
    debug_assert!(n > 0);

    let mut out = Vec::new();
    for idx in 0..=function_zeros.len() {
        let hi_root = if idx < function_zeros.len() {
            function_zeros[idx]
        } else {
            function_frontier
        };
        if !hi_root.is_finite() {
            return (out, f64::INFINITY);
        }
        let lo = if idx == 0 {
            1.0e-6
        } else {
            function_zeros[idx - 1] + 1.0e-6
        };
        let hi = hi_root - 1.0e-6;
        if hi <= lo {
            return (out, hi_root);
        }
        let zero = refine_bracketed_jnp_zero(n, lo, hi);
        if zero > cutoff {
            return (out, zero);
        }
        out.push(zero);
        if idx == function_zeros.len() {
            return (out, hi_root);
        }
    }
    (out, f64::INFINITY)
}

fn jnjnp_zero_candidates_below_cutoff(
    cutoff: f64,
    n_limit: usize,
    max_envelope: usize,
) -> (BesselZeroCandidates, f64, f64) {
    let mut cands: Vec<(f64, i32, i32, i32)> = Vec::new();
    let mut serial_frontier = f64::INFINITY;
    cands.push((0.0, 0, 0, 1)); // J_0'(0) = 0

    let order_limit = n_limit.min(max_envelope);
    for n in 0..=order_limit as u32 {
        let (jn, jn_frontier) = jn_zeros_until(n, cutoff);
        serial_frontier = serial_frontier.min(jn_frontier);
        for (i, &x) in jn.iter().enumerate() {
            cands.push((x, n as i32, (i + 1) as i32, 0));
        }

        let (jp, jp_frontier) = if n == 0 {
            jn_zeros_until(1, cutoff)
        } else {
            jnp_zeros_from_function_zeros_until(n, cutoff, &jn, jn_frontier)
        };
        serial_frontier = serial_frontier.min(jp_frontier);
        for (i, &x) in jp.iter().enumerate() {
            cands.push((x, n as i32, (i + 1) as i32, 1));
        }
    }

    let order_frontier = if order_limit >= max_envelope {
        f64::INFINITY
    } else {
        let omitted_order = (order_limit + 1) as u32;
        let jn_first = refine_jn_zero_at(omitted_order, 1, 0.0);
        let jnp_first = refine_bracketed_jnp_zero(omitted_order, 1.0e-6, jn_first - 1.0e-6);
        jn_first.min(jnp_first)
    };

    (cands, serial_frontier, order_frontier)
}

fn jnjnp_zero_candidates_with_frontier(
    per: usize,
    n_max: usize,
    max_envelope: usize,
) -> (BesselZeroCandidates, f64, f64) {
    let mut cands: Vec<(f64, i32, i32, i32)> = Vec::new();
    let mut serial_frontier = f64::INFINITY;
    cands.push((0.0, 0, 0, 1)); // J_0'(0) = 0

    for n in 0..=n_max as u32 {
        let jn = jn_zeros(n, per);
        if let Some(&tail) = jn.last() {
            serial_frontier = serial_frontier.min(tail);
        }
        for (i, &x) in jn.iter().enumerate() {
            cands.push((x, n as i32, (i + 1) as i32, 0));
        }
        // J_0' ≡ -J_1, so its zeros are exactly the J_1 zeros; reuse them so the
        // coincident magnitudes tie bit-for-bit and the `t` tie-break orders the
        // J_n zero (t=0) before the J_n' zero (t=1), as SciPy does.
        let jp = if n == 0 {
            jn_zeros(1, per)
        } else {
            jnp_zeros_from_function_zeros(n, per, &jn)
        };
        if let Some(&tail) = jp.last() {
            serial_frontier = serial_frontier.min(tail);
        }
        for (i, &x) in jp.iter().enumerate() {
            cands.push((x, n as i32, (i + 1) as i32, 1));
        }
    }

    let order_frontier = if n_max >= max_envelope {
        f64::INFINITY
    } else {
        let omitted_order = (n_max + 1) as u32;
        let jn_first = jn_zeros(omitted_order, 1)
            .into_iter()
            .next()
            .unwrap_or(f64::INFINITY);
        let jnp_first = jnp_zeros(omitted_order, 1)
            .into_iter()
            .next()
            .unwrap_or(f64::INFINITY);
        jn_first.min(jnp_first)
    };

    (cands, serial_frontier, order_frontier)
}

fn sort_jnjnp_candidates(cands: &mut [(f64, i32, i32, i32)]) {
    cands.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .expect("Bessel zeros are finite")
            .then(a.3.cmp(&b.3))
    });
}

fn collect_jnjnp_candidates(
    cands: &[(f64, i32, i32, i32)],
    nt: usize,
) -> (Vec<f64>, Vec<i32>, Vec<i32>, Vec<i32>) {
    let kept = &cands[..nt.min(cands.len())];
    let zo = kept.iter().map(|c| c.0).collect();
    let n = kept.iter().map(|c| c.1).collect();
    let m = kept.iter().map(|c| c.2).collect();
    let t = kept.iter().map(|c| c.3).collect();
    (zo, n, m, t)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ONE lock for tests that write a Bessel perf toggle. Process-global statics under
    /// a concurrent `cargo test` otherwise produce a MASKED pass — both arms running the
    /// same code and the comparison checking a thing against itself.
    fn bessel_toggle_lock() -> std::sync::MutexGuard<'static, ()> {
        static BESSEL_TOGGLE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        BESSEL_TOGGLE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Cross-checks the direct `K0`/`K1` kernels against the general-order `kv` path they
    /// replace, and drives `BESSEL_K01_CEPHES` in both settings.
    ///
    /// The two paths evaluate the SAME Chebyshev tables — that is the point of this change,
    /// which added no coefficients — but by different routes: the direct form computes `K`
    /// below x = 2 with no exponential, while `kv_scalar` obtained the exp-SCALED value and
    /// multiplied by `exp(-x)`, so that branch made a round trip through `exp(x)·exp(-x)`.
    /// They must therefore agree to a couple of ULP, and any larger gap means the routing is
    /// wrong rather than the tables.
    #[test]
    fn k0_k1_cephes_matches_the_kv_path() {
        use std::sync::atomic::Ordering::Relaxed;
        let _guard = bessel_toggle_lock();
        let was = BESSEL_K01_CEPHES.load(Relaxed);

        // Straddle the x = 2 split: below it the log/I0 form, above it the exp/sqrt form.
        let mut points: Vec<f64> = Vec::new();
        for i in 1..=1200 {
            points.push(0.01 + (20.0 - 0.01) * (i as f64 / 1200.0));
        }
        points.extend([0.01, 0.5, 1.0, 1.999, 2.0, 2.001, 5.0, 12.0, 19.999]);

        let (mut small, mut large) = (0usize, 0usize);
        let (mut worst0, mut worst1) = (0.0_f64, 0.0_f64);
        let (mut at0, mut at1) = (f64::NAN, f64::NAN);
        let mut hits_seen = 0usize;
        for &x in &points {
            if x <= 2.0 {
                small += 1;
            } else {
                large += 1;
            }
            BESSEL_K01_CEPHES.store(true, Relaxed);
            let before = BESSEL_K01_CEPHES_HITS.load(Relaxed);
            let d0 = k0_order_scalar(x, RuntimeMode::Strict).expect("k0");
            let d1 = k1_order_scalar(x, RuntimeMode::Strict).expect("k1");
            if BESSEL_K01_CEPHES_HITS.load(Relaxed) > before {
                hits_seen += 1;
            }
            BESSEL_K01_CEPHES.store(false, Relaxed);
            let g0 = k0_order_scalar(x, RuntimeMode::Strict).expect("k0 via kv");
            let g1 = k1_order_scalar(x, RuntimeMode::Strict).expect("k1 via kv");
            assert!(
                d0.is_finite() && d1.is_finite() && g0.is_finite() && g1.is_finite(),
                "non-finite at x={x}"
            );
            let u0 = (d0 - g0).abs() / g0.abs() / f64::EPSILON;
            let u1 = (d1 - g1).abs() / g1.abs() / f64::EPSILON;
            if u0 > worst0 {
                worst0 = u0;
                at0 = x;
            }
            if u1 > worst1 {
                worst1 = u1;
                at1 = x;
            }
        }
        BESSEL_K01_CEPHES.store(was, Relaxed);

        // MUST-HIT: both branches reached, and the direct arm actually fired every time.
        assert!(
            small > 0 && large > 0,
            "grid missed a branch: x<=2 {small}, x>2 {large}"
        );
        assert!(
            hits_seen >= points.len(),
            "direct arm did not run for every point"
        );
        println!(
            "k0 worst {worst0:.1} ULP at x={at0}; k1 worst {worst1:.1} ULP at x={at1}; \
             {} points (small={small} large={large})",
            points.len()
        );
        assert!(
            worst0 < 64.0 && worst1 < 64.0,
            "direct and kv paths disagree: k0 {worst0} ULP at {at0}, k1 {worst1} ULP at {at1}"
        );
    }

    /// The term-ratio `I_v` series agrees with the log-carried form it replaces.
    ///
    /// The two share no arithmetic in the loop: one multiplies by an exact ratio, the other
    /// exponentiates a running logarithm and pays three `ln` calls per term. Agreement to a
    /// few ULP is therefore real evidence rather than a tautology. The grid spans the orders
    /// and arguments the series branch actually handles, including the small-z corner where
    /// the leading term dominates and the large-order corner where the summand peaks late.
    #[test]
    fn iv_term_ratio_series_matches_the_log_form() {
        use std::sync::atomic::Ordering::Relaxed;
        let _guard = bessel_toggle_lock();
        let restore = IV_SERIES_TERM_RATIO.load(Relaxed);

        let mut worst = 0.0_f64;
        let mut worst_at = (f64::NAN, f64::NAN);
        let mut compared = 0usize;
        let mut hits_seen = 0usize;
        for vi in 0..40 {
            for zi in 1..60 {
                let v = vi as f64 * 0.35;
                let z = zi as f64 * 0.35;
                IV_SERIES_TERM_RATIO.store(true, Relaxed);
                let before = IV_SERIES_TERM_RATIO_HITS.load(Relaxed);
                let ratio = iv_scalar(v, z);
                if IV_SERIES_TERM_RATIO_HITS.load(Relaxed) > before {
                    hits_seen += 1;
                }
                IV_SERIES_TERM_RATIO.store(false, Relaxed);
                let logged = iv_scalar(v, z);
                assert!(
                    ratio.is_finite() && logged.is_finite(),
                    "non-finite at v={v} z={z}: ratio {ratio} logged {logged}"
                );
                let rel = (ratio - logged).abs() / logged.abs().max(f64::MIN_POSITIVE);
                if rel > worst {
                    worst = rel;
                    worst_at = (v, z);
                }
                compared += 1;
            }
        }
        IV_SERIES_TERM_RATIO.store(restore, Relaxed);

        // MUST-HIT: the series branch has to have been taken, or this compared two
        // evaluations that both went somewhere else (the asymptotic arm, say).
        assert!(
            hits_seen > compared / 2,
            "series branch ran only {hits_seen} of {compared} times; the grid is off it"
        );
        println!(
            "iv term-ratio vs log form: worst rel {worst:e} at v={} z={}, {compared} points",
            worst_at.0, worst_at.1
        );
        assert!(
            worst < 1.0e-12,
            "term-ratio and log forms disagree by {worst:e} at v={} z={}",
            worst_at.0,
            worst_at.1
        );

        // Integer-order parity for negative z must survive: I_v(-z) = (-1)^v I_v(z).
        for (v, z) in [(2.0_f64, 3.0_f64), (3.0, 3.0)] {
            let pos = iv_scalar(v, z);
            let neg = iv_scalar(v, -z);
            let expect = if (v as i64) % 2 == 0 { pos } else { -pos };
            assert!(
                (neg - expect).abs() <= 1.0e-12 * pos.abs().max(1.0),
                "I_{v}(-{z}) parity broken: {neg} vs {expect}"
            );
        }
    }

    /// The `Y1` small-argument rational agrees with the power series it replaces, and both
    /// branches of `y1_core_positive` are reached.
    ///
    /// The 14 transcribed coefficients are cross-checked against an implementation that
    /// shares none of them. The bound is loose for the same reason as the `y0` reflection
    /// check: the SERIES is the less accurate side here — against live SciPy the rational is
    /// exactly 0 where the series was 1.42e-14 — so this catches a mistranscription, and the
    /// live-incumbent arm adjudicates accuracy.
    #[test]
    fn y1_cephes_small_matches_the_series() {
        use std::sync::atomic::Ordering::Relaxed;
        let _guard = bessel_toggle_lock();
        let was = BESSEL_Y01_CEPHES_LARGE.load(Relaxed);

        let mut points: Vec<f64> = Vec::new();
        for i in 1..=2000 {
            points.push(0.002 + (10.0 - 0.002) * (i as f64 / 2000.0));
        }
        points.extend([1.0e-6, 0.01, 0.5, 1.0, 2.0, 4.999, 5.0, 5.001, 9.9]);

        let (mut small, mut large) = (0usize, 0usize);
        let mut worst = 0.0_f64;
        let mut worst_at = f64::NAN;
        for &x in &points {
            if x <= 5.0 {
                small += 1;
            } else {
                large += 1;
            }
            BESSEL_Y01_CEPHES_LARGE.store(true, Relaxed);
            let cephes = y1_core_positive(x);
            BESSEL_Y01_CEPHES_LARGE.store(false, Relaxed);
            let series = y1_core_positive(x);
            assert!(
                cephes.is_finite() && series.is_finite(),
                "non-finite at x={x}: cephes {cephes} series {series}"
            );
            let rel = (cephes - series).abs() / series.abs().max(f64::MIN_POSITIVE);
            if rel > worst {
                worst = rel;
                worst_at = x;
            }
        }
        BESSEL_Y01_CEPHES_LARGE.store(was, Relaxed);

        // MUST-HIT: both branches, or this is evidence about only one of them.
        assert!(
            small > 0 && large > 0,
            "grid missed a branch: x<=5 {small}, x>5 {large}"
        );
        println!(
            "y1 cephes vs series: worst rel {worst:e} at x={worst_at}, {} points \
             (small={small} large={large})",
            points.len()
        );
        assert!(
            worst < 1.0e-6,
            "Y1 rational and series disagree by {worst:e} at x={worst_at}; \
             a transcribed coefficient is probably wrong"
        );
    }

    /// Pins `y0`/`y1` against SciPy reference values tightly enough to catch a parity
    /// defect, which the pre-existing tests did not.
    ///
    /// THE HOLE THIS CLOSES. Before the Cephes large-argument form, `y0` differed from
    /// SciPy by `max_abs = 1.27e-11` across the harness fixture and `y1` by 9.54e-12, while
    /// every test in this crate passed. Tolerances loose enough to accept 1e-11 cannot tell
    /// a correct implementation from a drifting one, so the speed work above would have
    /// landed on top of an accuracy gap nothing was watching. Verified as a real gate and
    /// not a decorative one: with `BESSEL_Y01_CEPHES_LARGE` forced off this test FAILS,
    /// reporting y0 2.13e-11 at x = 13.999 and y1 6.37e-12 at x = 11.5.
    ///
    /// JUDGED ON ABSOLUTE ERROR, NOT RELATIVE, and the distinction matters here. The same
    /// comparison reports `max_rel = 1.32e-6` for the old `y0`, which reads like six lost
    /// digits and is not: `Y0` has zeros at x ≈ 0.89, 3.96, 7.09, 10.22, 13.36 … and a
    /// relative measure divides by a value passing through zero. This crate has already
    /// been caught once quoting an inflated `max_rel` as an accuracy claim (`gammaln`, whose
    /// zeros do the same thing), so the bound below is relative but the CLAIM is absolute.
    ///
    /// The bound is 1e-13 relative: tight enough to fail the old arm by two orders of
    /// magnitude, loose enough not to be a change-detector for the last bits. Points
    /// straddle every branch boundary this file has (x = 5 for the Cephes split, x = 14 for
    /// the old series/asymptotic split) so a future re-cut is caught at the seam.
    #[test]
    fn y0_y1_match_scipy_reference_values() {
        // TAKES THE TOGGLE LOCK, and it must: this test READS a toggled kernel.
        //
        // `y0_core_positive` and `y1_core_positive` branch on `BESSEL_Y01_CEPHES_LARGE`, and
        // `y1_cephes_small_matches_the_series` temporarily stores `false` into it. Both
        // tests ran without excluding each other, so under a concurrent `cargo test` this
        // one could evaluate the OLD series path against references pinned to the Cephes
        // path and fail at ~4e-13 — observed on hz2, 1178 passed / 1 failed, from a source
        // change in an unrelated file.
        //
        // The lock already existed and every WRITER took it. That is not enough: a mutex
        // only excludes participants that take it, and a reader of shared mutable state is a
        // participant. The doc comment above even records that this test FAILS with the
        // toggle off, which is exactly the failure a racing writer induces.
        //
        // The arm is also pinned explicitly rather than inherited, so the test states which
        // path it is gating instead of depending on ambient process state, and restored
        // afterwards so it leaves nothing behind for the next test.
        use std::sync::atomic::Ordering::Relaxed;
        let _guard = bessel_toggle_lock();
        let was = BESSEL_Y01_CEPHES_LARGE.load(Relaxed);
        BESSEL_Y01_CEPHES_LARGE.store(true, Relaxed);
        // `_restore` and not `_`: a bare underscore drops the guard IMMEDIATELY, which would
        // restore the toggle before the test body ran and defeat the pin. It restores on the
        // assertion's unwind too.
        let _restore = RestoreY01Large(was);
        struct RestoreY01Large(bool);
        impl Drop for RestoreY01Large {
            fn drop(&mut self) {
                BESSEL_Y01_CEPHES_LARGE.store(self.0, std::sync::atomic::Ordering::Relaxed);
            }
        }

        // scipy 1.17.1 / numpy 2.4.3
        const REFERENCE: &[(f64, f64, f64)] = &[
            (0.1, -1.5342386513503667, -6.458951094702027),
            (0.5, -0.4445187335067066, -1.4714723926702433),
            (1.0, 0.08825696421567697, -0.7812128213002888),
            (2.5, 0.498070359615232, 0.14591813796678577),
            (4.9, -0.29205459424401364, 0.18124669204504856),
            (5.0, -0.30851762524903303, 0.14786314339122691),
            (5.1, -0.3216024491248595, 0.11373644197749971),
            (6.0, -0.28819468398157916, -0.17501034430039827),
            (7.5, 0.11731328614820863, -0.25912851048611624),
            (9.0, 0.24993669828502474, 0.10431457519671594),
            (10.0, 0.05567116728359961, 0.24901542420695388),
            (11.5, -0.22523211169118781, 0.057942547143000615),
            (13.0, -0.07820786452787612, -0.2100814084206936),
            (13.999, 0.12702585421842472, -0.16678385967336848),
            (14.0, 0.12719256858218356, -0.16664484185617212),
            (14.001, 0.12735914385018385, -0.16650566817978737),
            (17.0, -0.09263719844232356, 0.1672050360772336),
            (20.0, 0.06264059680938369, -0.1655116143625212),
            (23.5, -0.10828611769479085, 0.1216532806902693),
            (27.0, 0.1352149762078723, -0.07025123823578311),
            (30.0, -0.11729573168666398, 0.08442557066174713),
        ];

        let (mut worst0, mut worst1) = (0.0_f64, 0.0_f64);
        let (mut at0, mut at1) = (f64::NAN, f64::NAN);
        let (mut below, mut above) = (0usize, 0usize);
        for &(x, ref0, ref1) in REFERENCE {
            if x <= 5.0 {
                below += 1;
            } else {
                above += 1;
            }
            let got0 = y0_core_positive(x);
            let got1 = y1_core_positive(x);
            let rel0 = (got0 - ref0).abs() / ref0.abs();
            let rel1 = (got1 - ref1).abs() / ref1.abs();
            if rel0 > worst0 {
                worst0 = rel0;
                at0 = x;
            }
            if rel1 > worst1 {
                worst1 = rel1;
                at1 = x;
            }
        }
        // MUST-HIT: both sides of the x = 5 split have to be exercised, or this is evidence
        // about only one branch.
        assert!(
            below > 0 && above > 0,
            "reference grid missed a branch: x<=5 {below}, x>5 {above}"
        );
        assert!(
            worst0 < 1.0e-13 && worst1 < 1.0e-13,
            "y0/y1 disagree with SciPy: y0 {worst0:e} at x={at0}, y1 {worst1:e} at x={at1}"
        );
    }

    /// Cross-checks the transcribed Chebyshev tables against the independent `iv` power
    /// series over the whole domain.
    ///
    /// This is what catches a mistranscribed coefficient. 109 numbers were copied out of
    /// `xsf/cephes/{i0,i1}.h`; the series shares none of them and no code path, so agreement
    /// to a few tens of ULP is strong evidence the tables are right. A single wrong digit in
    /// any of the four tables moves its branch by far more than the bound below.
    #[test]
    fn i0_i1_cephes_matches_the_iv_series() {
        use std::sync::atomic::Ordering::Relaxed;
        let _guard = bessel_toggle_lock();
        let was = BESSEL_I01_CEPHES.load(Relaxed);

        // Span BOTH Chebyshev branches: |x| <= 8 uses the A tables, |x| > 8 the B tables.
        let mut points: Vec<f64> = Vec::new();
        for i in 0..=1500 {
            points.push(-15.0 + 30.0 * (i as f64 / 1500.0));
        }
        points.extend([
            0.0, 1e-8, -1e-8, 0.5, -0.5, 7.999, 8.0, 8.001, -8.001, 12.0, -12.0, 15.0,
        ]);

        let (mut small, mut large) = (0usize, 0usize);
        let (mut worst0, mut worst1) = (0.0_f64, 0.0_f64);
        let (mut at0, mut at1) = (f64::NAN, f64::NAN);
        let mut compared = 0usize;
        for &x in &points {
            if x.abs() <= 8.0 {
                small += 1;
            } else {
                large += 1;
            }
            let (c0, c1) = (i0_cephes(x), i1_cephes(x));
            let (s0, s1) = (iv_scalar(0.0, x), iv_scalar(1.0, x));
            assert!(
                c0.is_finite() && c1.is_finite() && s0.is_finite() && s1.is_finite(),
                "non-finite at x={x}: cephes {c0}/{c1} series {s0}/{s1}"
            );
            // I1 is odd and vanishes at 0, so guard the relative measure rather than
            // dividing by something that is legitimately zero.
            let ulp0 = (c0 - s0).abs() / f64::max(s0.abs(), f64::MIN_POSITIVE) / f64::EPSILON;
            let ulp1 = (c1 - s1).abs() / f64::max(s1.abs(), 1.0e-300) / f64::EPSILON;
            if ulp0 > worst0 {
                worst0 = ulp0;
                at0 = x;
            }
            if ulp1 > worst1 {
                worst1 = ulp1;
                at1 = x;
            }
            compared += 1;
        }
        BESSEL_I01_CEPHES.store(was, Relaxed);

        // MUST-HIT: both Chebyshev branches have to be reached, or this is evidence only
        // about the table that happened to run.
        assert!(
            small > 0 && large > 0,
            "grid missed a branch: |x|<=8 {small}, |x|>8 {large}"
        );
        assert_eq!(compared, points.len(), "comparison loop was skipped");
        println!(
            "i0 worst {worst0:.1} ULP at x={at0}; i1 worst {worst1:.1} ULP at x={at1};              {compared} points (small={small} large={large})"
        );
        // MEASURED worst is 27.5 ULP (i0, x=14.4) and 22.4 ULP (i1, x=14.88), and almost
        // all of that is the SERIES: against live SciPy the Chebyshev arm is ~1.5 ULP
        // (max_rel 3.37e-16) where the series is ~32 ULP (7.03e-15). The bound is 256 for
        // ~9x margin rather than the measured value, because it exists to catch a
        // mistranscribed coefficient — which moves a branch by orders of magnitude, not by
        // ULPs — and sitting on the measurement would make it a change-detector instead.
        assert!(
            worst0 < 256.0 && worst1 < 256.0,
            "Chebyshev and series disagree: i0 {worst0} ULP at {at0}, i1 {worst1} ULP at              {at1}; a transcribed coefficient is probably wrong"
        );
    }

    /// Drives `BESSEL_I01_CEPHES` in BOTH settings through the public batch entry, and pins
    /// that the arm actually fires — "enabled" is not "took effect".
    #[test]
    fn bessel_i01_cephes_toggle_drives_both_arms() {
        use std::sync::atomic::Ordering::Relaxed;
        let _guard = bessel_toggle_lock();
        let was = BESSEL_I01_CEPHES.load(Relaxed);

        let values: Vec<f64> = (0..512).map(|i| -12.0 + i as f64 * 0.05).collect();
        let tensor = SpecialTensor::RealVec(values.clone());
        let mut arms = Vec::new();
        for enabled in [true, false] {
            BESSEL_I01_CEPHES.store(enabled, Relaxed);
            let before = BESSEL_I01_CEPHES_HITS.load(Relaxed);
            let out = i0(&tensor, RuntimeMode::Strict).expect("i0 batch");
            let hits = BESSEL_I01_CEPHES_HITS.load(Relaxed) - before;
            let SpecialTensor::RealVec(v) = out else {
                panic!("expected RealVec")
            };
            arms.push((v, hits));
        }
        BESSEL_I01_CEPHES.store(was, Relaxed);

        // MUST-HIT, bounded rather than exact, for the same reason the gamma-family
        // mapper's counter is: `BESSEL_I01_CEPHES_HITS` is process-global, so any
        // concurrent test calling `i0`/`i1` increments it and `== values.len()` / `== 0`
        // fail intermittently on a CORRECT tree. The must-miss for this toggle is the
        // instruction A/B instead — 176.0 against 4181.2 per element — which cannot happen
        // unless the two arms really execute different code.
        assert!(
            arms[0].1 >= values.len(),
            "Chebyshev arm did not run for every element"
        );
        // The two arms are DIFFERENT approximations, so they must not be asserted equal.
        // What matters is that both produced finite, close results; accuracy against SciPy
        // is the real contract and lives in the harness.
        for (a, b) in arms[0].0.iter().zip(&arms[1].0) {
            assert!(
                (a - b).abs() <= 1.0e-9 * a.abs().max(1.0),
                "arms disagree far more than either differs from SciPy: {a} vs {b}"
            );
        }
    }

    #[test]
    fn spherical_jn_jv_route_matches_scipy_small_x() {
        // x < n now uses j_n(x) = √(π/2x)·J_{n+½}(x) (was the Miller recurrence).
        // References from scipy.special.spherical_jn (1.17.1), incl. the x≪n tail
        // (down to ~1e-31). Asserted to 1e-12 relative.
        let cases = [
            (10.0, 5.0, 0.00040734424424946),
            (30.0, 20.0, 2.1063576943611e-05),
            (5.0, 3.0, 0.016397480955999),
            (9.0, 0.1, 1.5269856934948e-18),
            (50.0, 10.0, 2.2306960232186e-31),
            (20.0, 2.0, 7.6326411008877e-20),
        ];
        for (nn, x, expected) in cases {
            let got = spherical_jn_scalar(nn, x, RuntimeMode::Strict).unwrap();
            assert!(
                (got - expected).abs() <= 1e-12 * expected.abs(),
                "spherical_jn({nn},{x}) = {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn jv_yv_scalar_match_scipy() {
        // scipy.special.jv/yv (Bessel functions of the first/second kind).
        assert!(
            (jv_scalar(0.0, 1.0) - 0.765_197_686_557_966_6).abs() < 1e-13,
            "jv(0,1)"
        );
        assert!(
            (jv_scalar(1.0, 2.0) - 0.576_724_807_756_873_6).abs() < 1e-13,
            "jv(1,2)"
        );
        assert!(
            (yv_scalar(0.0, 1.0, RuntimeMode::Strict).unwrap() - 0.088_256_964_215_677).abs()
                < 1e-13,
            "yv(0,1)"
        );
    }

    #[test]
    fn lmbda_rejects_a_negative_order_but_accepts_zero() {
        // MUST-HIT. scipy raises for v < 0; this used to clamp and answer.
        //   lmbda(-1.0, 2.0) -> ValueError: argument must be > 0.
        //   lmbda(-0.5, 2.0) -> ValueError: argument must be > 0.
        // The old clamp made v = -1.0 return the v = 0.0 answer and v = -0.5
        // return the v0 = 0.5 series -- correct for a DIFFERENT input, which is
        // worse than a NaN because it looks right (frankenscipy-21uhl).
        for v in [-1.0_f64, -0.5, -3.0, -1e-12] {
            let (vl, dl) = lmbda(v, 2.0);
            assert!(
                vl.iter().all(|z| z.is_nan()) && dl.iter().all(|z| z.is_nan()),
                "lmbda({v}, 2.0) must be NaN-signalled, got vl={vl:?} dl={dl:?}"
            );
        }

        // MUST-MISS, and it is the subtle half: scipy's message reads
        // "argument must be > 0." but v == 0.0 is ACCEPTED and returns a value.
        // A guard written from the message rather than the measurement would
        // break this case.
        //   scipy.special.lmbda(0.0, 2.0) -> ([0.22389077914123562], [-0.5767248077568736])
        let (vl, dl) = lmbda(0.0, 2.0);
        assert_eq!(vl.len(), 1, "v = 0.0 yields a single entry");
        assert!(
            (vl[0] - 0.223_890_779_141_235_62).abs() < 1e-12,
            "lmbda(0.0, 2.0) vl = {}, scipy 0.22389077914123562",
            vl[0]
        );
        assert!(
            (dl[0] - -0.576_724_807_756_873_6).abs() < 1e-12,
            "lmbda(0.0, 2.0) dl = {}, scipy -0.5767248077568736",
            dl[0]
        );

        // And a non-integer positive order is untouched.
        //   scipy.special.lmbda(0.5, 2.0) -> ([0.45464871341284085], [-0.4353977749799916])
        let (vl, dl) = lmbda(0.5, 2.0);
        assert!(
            (vl[0] - 0.454_648_713_412_840_85).abs() < 1e-12,
            "vl {}",
            vl[0]
        );
        assert!(
            (dl[0] - -0.435_397_774_979_991_6).abs() < 1e-12,
            "dl {}",
            dl[0]
        );
    }

    #[test]
    fn lmbda_matches_scipy() {
        // frankenscipy: golden (vl, dl) from scipy.special.lmbda (1.17.1).
        let close = |got: &[f64], want: &[f64], msg: &str| {
            for (i, w) in want.iter().enumerate() {
                assert!(
                    (got[i] - w).abs() < 1e-7 * w.abs().max(1.0),
                    "{msg}[{i}]: got {}, want {w}",
                    got[i]
                );
            }
        };
        let (vl, dl) = lmbda(2.0, 3.0);
        close(
            &vl,
            &[-0.2600519549, 0.2260393057, 0.4320811205],
            "lmbda(2,3) v",
        );
        close(
            &dl,
            &[-0.3390589585, -0.3240608404, -0.2747224198],
            "lmbda(2,3) d",
        );
        let (vl, dl) = lmbda(2.5, 1.0);
        close(
            &vl,
            &[0.8414709848, 0.9035060368, 0.9305257802],
            "lmbda(2.5,1) v",
        );
        close(
            &dl,
            &[-0.3011686789, -0.186105156, -0.1350987168],
            "lmbda(2.5,1) d",
        );
        // Large order: stable downward recurrence stays accurate where J underflows.
        let (vl150, _) = lmbda(150.0, 1.0);
        close(
            &vl150,
            &[0.7651976866, 0.8801011715, 0.9192278795, 0.9390409912],
            "lmbda(150,1) v",
        );
        // x = 0 edges: SciPy's specfun result (integer-v quirk reproduced).
        let (vl, dl) = lmbda(3.0, 0.0);
        assert_eq!(vl, vec![1.0, 0.0, 0.0, 0.0], "lmbda(3,0) v");
        assert_eq!(dl, vec![0.0, 0.5, 0.0, 0.0], "lmbda(3,0) d");
        let (vl, _) = lmbda(2.5, 0.0);
        assert_eq!(vl, vec![1.0, 1.0, 1.0], "lmbda(2.5,0) v");
    }

    #[test]
    fn jnyn_and_jnjnp_zeros_match_scipy() {
        // frankenscipy: golden from scipy.special.jnyn_zeros / jnjnp_zeros (1.17.1).
        let (jn, jnp, yn, ynp) = jnyn_zeros(2, 3);
        let close = |a: &[f64], b: &[f64], msg: &str| {
            assert_eq!(a.len(), b.len(), "{msg}: len");
            for (g, w) in a.iter().zip(b.iter()) {
                assert!((g - w).abs() < 1e-8, "{msg}: got {g}, want {w}");
            }
        };
        close(&jn, &[5.1356223018, 8.4172441404, 11.6198411721], "jnyn jn");
        close(
            &jnp,
            &[3.0542369282, 6.7061331942, 9.9694678231],
            "jnyn jnp",
        );
        close(&yn, &[3.3842417671, 6.7938075133, 10.0234779794], "jnyn yn");
        close(
            &ynp,
            &[5.0025829314, 8.3507247014, 11.5741954652],
            "jnyn ynp",
        );

        let (zo, n, m, t) = jnjnp_zeros(8);
        close(
            &zo,
            &[
                0.0,
                1.8411837813,
                2.4048255577,
                3.0542369282,
                3.8317059702,
                3.8317059702,
                4.2011889412,
                5.1356223018,
            ],
            "jnjnp zo",
        );
        assert_eq!(n, vec![0, 1, 0, 2, 1, 0, 3, 2], "jnjnp n");
        assert_eq!(m, vec![0, 1, 1, 1, 1, 1, 1, 1], "jnjnp m");
        assert_eq!(t, vec![1, 1, 0, 1, 0, 1, 1, 0], "jnjnp t");

        assert!(jnjnp_zeros(0).0.is_empty(), "nt=0 -> empty");
    }

    #[test]
    fn jnjnp_adaptive_envelope_matches_oversized_reference() {
        for &nt in &[1_usize, 8, 32] {
            let adaptive = jnjnp_zeros(nt);
            let (mut cands, _, _) = jnjnp_zero_candidates_with_frontier(nt + 2, nt + 2, nt + 2);
            sort_jnjnp_candidates(&mut cands);
            let reference = collect_jnjnp_candidates(&cands, nt);

            assert_eq!(
                adaptive
                    .0
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                reference
                    .0
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "zo bits for nt={nt}"
            );
            assert_eq!(adaptive.1, reference.1, "n for nt={nt}");
            assert_eq!(adaptive.2, reference.2, "m for nt={nt}");
            assert_eq!(adaptive.3, reference.3, "t for nt={nt}");
        }
    }

    #[test]
    fn jnjnp_frontier_matches_scipy_bench_cutoffs() {
        // SciPy 1.17.1 golden cutoffs for the two perf-gauntlet sizes.
        for (nt, want_zo, want_n, want_m, want_t, want_max_n, want_max_m) in [
            (64_usize, 15.268_181_461_097_873, 6, 3, 1, 13, 5),
            (128_usize, 22.046_985_364_697_804, 10, 3, 0, 19, 7),
        ] {
            let (zo, n, m, t) = jnjnp_zeros(nt);
            let last = nt - 1;
            assert!(
                (zo[last] - want_zo).abs() < 1e-10,
                "jnjnp_zeros({nt}) cutoff {} != {want_zo}",
                zo[last]
            );
            assert_eq!((n[last], m[last], t[last]), (want_n, want_m, want_t));
            assert_eq!(n.iter().copied().max(), Some(want_max_n));
            assert_eq!(m.iter().copied().max(), Some(want_max_m));
        }
    }

    #[test]
    fn log_ive_matches_ive_in_overlap_and_finite_beyond() {
        // In the range where iv does not overflow, log_ive == ln(ive) exactly /
        // to asymptotic precision.
        for &v in &[0.0, 0.5, 1.0, 2.5] {
            for &z in &[1.0, 5.0, 20.0, 60.0, 99.0, 150.0, 400.0, 650.0] {
                let li = log_ive_scalar(v, z);
                let direct = ive_scalar(v, z);
                if direct > 0.0 && direct.is_finite() {
                    let rel = (li - direct.ln()).abs() / direct.ln().abs().max(1.0);
                    assert!(
                        rel <= 1e-6,
                        "log_ive({v},{z})={li} vs ln(ive)={}",
                        direct.ln()
                    );
                }
            }
        }
        // Beyond the iv overflow threshold (z > 709): ive is NaN/0, but log_ive
        // stays finite (asymptotic ≈ −½ln(2πz)).
        for &z in &[800.0, 5000.0, 1.0e6] {
            let li = log_ive_scalar(1.0, z);
            let approx = -0.5 * (2.0 * PI * z).ln();
            assert!(li.is_finite(), "log_ive(1,{z}) not finite");
            assert!(
                (li - approx).abs() < 0.05,
                "log_ive(1,{z})={li} vs ~{approx}"
            );
        }
        assert_eq!(log_ive_scalar(0.0, 0.0), 0.0);
        assert_eq!(log_ive_scalar(1.0, 0.0), f64::NEG_INFINITY);
        assert!(log_ive_scalar(f64::NAN, 1.0).is_nan());

        // LARGE ORDER at moderate x: I_v(x) underflows to 0 (ive.ln() would be
        // −inf), but the log-space series keeps it finite. Golden values from
        // mpmath log(besseli(v,x))−x. frankenscipy regression.
        for &(v, z, want) in &[
            (267.0_f64, 12.649_f64, -748.544_985_f64),
            (500.0, 20.0, -1_479.838_351),
            (100.0, 12.649, -191.550_010),
        ] {
            let li = log_ive_scalar(v, z);
            assert!(li.is_finite(), "log_ive({v},{z}) not finite");
            assert!((li - want).abs() < 1e-3, "log_ive({v},{z})={li} vs {want}");
        }
    }

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
            assert!(
                err <= 1e-7,
                "yv({v},{re}{im:+}i) = {y:?}, scipy ({yr},{yi}), rel {err:e}"
            );
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
            (
                10.5,
                30.0,
                9.3,
                -9.37954079129976e-14,
                5.344341257316493e-14,
            ),
            (
                20.5,
                25.0,
                9.0,
                2.080674029070471e-09,
                3.6929289885929565e-09,
            ),
            (
                30.5,
                45.0,
                11.0,
                4.9377593759569705e-17,
                -4.154102966627179e-17,
            ),
            (
                50.5,
                55.0,
                18.0,
                5.0678599670482056e-17,
                9.429501885540872e-17,
            ),
            (
                5.5,
                15.0,
                5.0,
                1.5307782517384106e-07,
                1.7251233742654897e-07,
            ),
            (
                20.5,
                15.0,
                12.0,
                0.0003422838796743065,
                0.00031697965000494787,
            ),
            // Re(z) < 0 (branch-cut reflection):
            (0.5, -3.0, 2.0, -13.136389329139432, 1.7869727121447632),
            (5.5, -20.0, 8.0, -66248408.63984236, 14978308.422065388),
            (10.5, -25.0, 15.0, 1037005.5000110489, 3260618052.855438),
            (
                2.5,
                -50.0,
                -10.0,
                4.0274493691085514e20,
                -7.581988572080506e20,
            ),
            (20.5, -30.0, -12.0, -5781507165.759946, 200323296.50961077),
        ];
        for (v, re, im, kr, ki) in cases {
            let z = Complex64::new(re, im);
            let k = complex_kv_scalar(v, z, RuntimeMode::Strict).unwrap();
            let err = (k.re - kr).hypot(k.im - ki) / kr.hypot(ki);
            assert!(
                err <= 1e-7,
                "kv({v},{re}{im:+}i) = {k:?}, scipy ({kr},{ki}), rel {err:e}"
            );
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy
    fn complex_kve_matches_scipy() {
        // frankenscipy-2bmxw: exponentially-scaled K_v for complex z is
        // kve(v,z) = kv(v,z)·exp(z) — the FULL complex exponential. A prior
        // bug scaled by exp(Re z), keeping the magnitude but dropping the
        // exp(i·Im z) phase rotation, which diverged from scipy for non-real z.
        // Exercised through the public `kve` dispatcher (real order, complex z).
        // (v, re, im, kve.re, kve.im) — scipy 1.17.1.
        let cases: [(f64, f64, f64, f64, f64); 6] = [
            (
                0.5,
                3.0,
                -2.5,
                5.9634253363884471e-01,
                2.1590577564981434e-01,
            ),
            (
                2.0,
                -1.5,
                2.0,
                -1.6263736917360888e-01,
                -5.1718320252846783e-01,
            ),
            (
                0.0,
                0.2,
                4.0,
                4.6416194753566503e-01,
                -4.1575423635924635e-01,
            ),
            (
                1.0,
                5.0,
                0.0,
                6.0027385878831252e-01,
                0.0000000000000000e+00,
            ),
            (
                3.5,
                0.0,
                2.0,
                -2.4282961410487816e+00,
                1.0183177365688443e+00,
            ),
            (
                0.5,
                -4.0,
                -1.0,
                7.5415294229784310e-02,
                6.1260640081557272e-01,
            ),
        ];
        for (v, re, im, kr, ki) in cases {
            let out = kve(
                &SpecialTensor::RealScalar(v),
                &SpecialTensor::ComplexScalar(Complex64::new(re, im)),
                RuntimeMode::Strict,
            )
            .unwrap();
            let k = match out {
                SpecialTensor::ComplexScalar(k) => k,
                other => {
                    assert!(
                        matches!(other, SpecialTensor::ComplexScalar(_)),
                        "kve complex scalar input must yield a complex scalar, got {other:?}"
                    );
                    continue;
                }
            };
            let err = (k.re - kr).hypot(k.im - ki) / kr.hypot(ki);
            assert!(
                err <= 1e-9,
                "kve({v},{re}{im:+}i) = {k:?}, scipy ({kr},{ki}), rel {err:e}"
            );
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
            (
                10.5,
                30.0,
                9.3,
                -197.02815202783466,
                -425.31457727367575,
                -136745728601.30203,
                -34365056209.05175,
            ),
            (
                20.5,
                25.0,
                9.0,
                31.831636545639334,
                1.7443462385888058,
                1050761.4293616053,
                -3459922.56166499,
            ),
            (
                30.5,
                38.0,
                11.0,
                28.8899240212221,
                -89.82134451310553,
                9940418282.002474,
                30996195925.81287,
            ),
            (
                50.5,
                55.0,
                18.0,
                -900.4238940608079,
                26.820654557258933,
                19190288570304.97,
                -59319339392746.0,
            ),
            (
                20.5,
                -50.0,
                35.0,
                -10625161413727.812,
                2880710047906.2197,
                -1.3356153802331707e19,
                7.901885428551122e18,
            ),
        ];
        for (v, re, im, jr, ji, ir, ii) in cases {
            let z = Complex64::new(re, im);
            let j = complex_jv_scalar(v, z);
            let i = complex_iv_scalar(v, z);
            let je = (j.re - jr).hypot(j.im - ji) / jr.hypot(ji);
            let ie = (i.re - ir).hypot(i.im - ii) / ir.hypot(ii);
            assert!(
                je <= 1e-7,
                "jv({v},{re}{im:+}i) = {j:?}, scipy ({jr},{ji}), rel {je:e}"
            );
            assert!(
                ie <= 1e-7,
                "iv({v},{re}{im:+}i) = {i:?}, scipy ({ir},{ii}), rel {ie:e}"
            );
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
            (
                0.5,
                20.0,
                5.0,
                12.466521336498948,
                3.8267134139998005,
                7006986.415091062,
                -42048916.84546594,
                2.2684059886175998e-10,
                5.21771888934066e-10,
                -3.8273250606136826,
                12.465507499599681,
            ),
            (
                2.5,
                -15.0,
                25.0,
                4820258766.728313,
                -670823000.802485,
                -121420.56920674515,
                194474.00807032152,
                381453.9682146355,
                -610958.115067951,
                670823000.802485,
                4820258766.728313,
            ),
            (
                3.5,
                40.0,
                -10.0,
                -1109.4573598197273,
                720.8276509954086,
                -1.119458545990043e16,
                5954167327505190.0,
                -7.091076212196013e-19,
                -6.373437988023484e-19,
                720.8276552862686,
                1109.4573558385662,
            ),
            (
                0.5,
                3.0,
                80.0,
                -1.4455656554351904e33,
                -2.0033277552023882e33,
                -0.6900384488490181,
                -0.5742502926122344,
                0.004254020290227443,
                0.005526238447046366,
                2.0033277552023882e33,
                -1.4455656554351904e33,
            ),
            (
                2.5,
                50.0,
                3.0,
                0.19792798151697677,
                -1.1086923774604178,
                -2.7126068652301323e20,
                4.59770369446714e19,
                -3.6026674628426933e-23,
                -3.9101945736867964e-24,
                1.1141728356305414,
                0.19661157971907242,
            ),
        ];
        for (v, re, im, jr, ji, ir, ii, kr, ki, yr, yi) in cases {
            let z = Complex64::new(re, im);
            let checks = [
                (complex_jv_scalar(v, z), jr, ji, "jv"),
                (complex_iv_scalar(v, z), ir, ii, "iv"),
                (
                    complex_kv_scalar(v, z, RuntimeMode::Strict).unwrap(),
                    kr,
                    ki,
                    "kv",
                ),
                (
                    complex_yv_scalar(v, z, RuntimeMode::Strict).unwrap(),
                    yr,
                    yi,
                    "yv",
                ),
            ];
            for (got, wr, wi, name) in checks {
                let err = (got.re - wr).hypot(got.im - wi) / wr.hypot(wi);
                assert!(
                    err <= 1e-7,
                    "{name}({v},{re}{im:+}i) = {got:?}, scipy ({wr},{wi}), rel {err:e}"
                );
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
            (
                0,
                2.0,
                1.0,
                0.037987722915986476,
                -0.10171357546139093,
                0.800451120409994,
                0.07563855028639382,
            ),
            (
                1,
                1.5,
                2.5,
                -0.17583976180149624,
                -0.005975889337860949,
                -0.6409702450434701,
                2.4512074350972797,
            ),
            (
                2,
                3.0,
                1.0,
                0.015275965061514699,
                -0.05547174952705652,
                -0.1433635707870101,
                0.4576996364039667,
            ),
            (
                2,
                0.5,
                4.0,
                0.18495000454282312,
                0.38087077161509364,
                -3.137626774336867,
                -5.632589456759534,
            ),
            (
                5,
                8.0,
                3.0,
                -0.000460035945850506,
                0.00023640292674529355,
                1.4290304684892552,
                0.9410036389912462,
            ),
            (
                0,
                15.0,
                5.0,
                4.129869836775042e-08,
                8.634770156118545e-08,
                14.529566891182334,
                -3.4121917632978587,
            ),
            (
                1,
                5.0,
                30.0,
                0.001172778369302158,
                0.0009898467040185962,
                -157028907008.00497,
                -747235851320.0383,
            ),
            (
                3,
                2.0,
                -3.0,
                -0.06909578637591238,
                -0.1727382937957655,
                -1.0614832363273328,
                0.8158048933143148,
            ),
        ];
        for (v, re, im, kr, ki, yr, yi) in cases {
            let z = Complex64::new(re, im);
            let k = complex_kv_scalar(v as f64, z, RuntimeMode::Strict).unwrap();
            let y = complex_yv_scalar(v as f64, z, RuntimeMode::Strict).unwrap();
            let ke = (k.re - kr).hypot(k.im - ki) / kr.hypot(ki);
            let ye = (y.re - yr).hypot(y.im - yi) / yr.hypot(yi);
            assert!(
                ke <= 1e-7,
                "kv({v},{re}{im:+}i) = {k:?}, scipy ({kr},{ki}), rel {ke:e}"
            );
            assert!(
                ye <= 1e-7,
                "yv({v},{re}{im:+}i) = {y:?}, scipy ({yr},{yi}), rel {ye:e}"
            );
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
            assert!(
                err <= 1e-9,
                "wofz({re}{im:+}i) = {w:?}, scipy ({wr},{wi}), rel {err:e}"
            );
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
            assert!(
                (j - jref).abs() <= 1e-9 * jref.abs().max(1e-12),
                "jv({v},{z}) = {j:e}, scipy {jref:e}"
            );
            assert!(
                (y - yref).abs() <= 1e-9 * yref.abs().max(1e-12),
                "yv({v},{z}) = {y:e}, scipy {yref:e}"
            );
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
            assert!(
                (j - jref).abs() <= 1e-9 * jref.abs().max(1e-3),
                "jv({v},{z}) = {j}, scipy {jref}"
            );
            assert!(
                (y - yref).abs() <= 1e-9 * yref.abs().max(1e-3),
                "yv({v},{z}) = {y}, scipy {yref}"
            );
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
            let jval =
                real_value(tensor_result(jv(&scalar(v), &scalar(z), RuntimeMode::Strict)).unwrap())
                    .unwrap();
            let yval =
                real_value(tensor_result(yv(&scalar(v), &scalar(z), RuntimeMode::Strict)).unwrap())
                    .unwrap();
            assert!(
                ((jval - jref) / jref).abs() < 1e-10,
                "jv({v},{z})={jval:e} vs {jref:e}"
            );
            assert!(
                ((yval - yref) / yref).abs() < 1e-10,
                "yv({v},{z})={yval:e} vs {yref:e}"
            );
        }
    }

    #[test]
    fn kv_large_order_finite() {
        // K_v is representable (≈2.7e279) but the scaled K_v·e^z overflows f64
        // (≈7e322) at large order; the unscaled-recurrence fallback keeps kv
        // finite. Golden from scipy.special.kv. frankenscipy regression.
        let rv = |r: SpecialResult| match r {
            Ok(SpecialTensor::RealScalar(v)) => v,
            _ => f64::NAN,
        };
        let s = SpecialTensor::RealScalar;
        let m = RuntimeMode::Strict;
        for (v, z, kref) in [
            (500.0_f64, 100.0_f64, 2.731_38e279_f64),
            (700.0, 200.0, 1.221_70e280),
            (500.5, 100.0, 8.678_01e279),
        ] {
            let kval = rv(kv(&s(v), &s(z), m));
            assert!(kval.is_finite(), "kv({v},{z}) not finite");
            assert!(
                ((kval - kref) / kref).abs() < 1e-4,
                "kv({v},{z})={kval:e} vs {kref:e}"
            );
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
            assert!(
                ((kval - kref) / kref).abs() < 1e-10,
                "kv({v},{z})={kval:e} vs {kref:e}"
            );
            assert!(
                ((keval - keref) / keref).abs() < 1e-10,
                "kve({v},{z})={keval:e} vs {keref:e}"
            );
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
            assert!(
                ((kval - kref) / kref).abs() < 1e-10,
                "kv({v},{z})={kval:e} vs {kref:e}"
            );
            assert!(
                ((keval - keref) / keref).abs() < 1e-10,
                "kve({v},{z})={keval:e} vs {keref:e}"
            );
        }
        // z > 745: kv underflows to 0 (matching scipy), kve stays finite.
        for (v, z, keref) in [
            (0.0, 750.0, 0.045756939928889066),
            (2.0, 800.0, 0.04441525775942454),
            (1.0, 900.0, 0.04179453901303371),
        ] {
            assert_eq!(
                rv(kv(&s(v), &s(z), m)),
                0.0,
                "kv({v},{z}) should underflow to 0"
            );
            let keval = rv(kve(&s(v), &s(z), m));
            assert!(
                ((keval - keref) / keref).abs() < 1e-10,
                "kve({v},{z})={keval:e} vs {keref:e}"
            );
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy 1.17.1
    fn k0_k1_cephes_base_matches_scipy() {
        // Integer-order K_v builds from K0e/K1e. These base kernels now use the
        // same Cephes Chebyshev expansions as SciPy instead of two quadratures.
        let rv = |r: SpecialResult| match r {
            Ok(SpecialTensor::RealScalar(v)) => v,
            _ => f64::NAN,
        };
        let s = SpecialTensor::RealScalar;
        let m = RuntimeMode::Strict;
        for (x, k0_ref, k1_ref, k0e_ref, k1e_ref) in [
            (
                0.125,
                2.20786908674496951e0,
                7.83111829911575175e0,
                2.50184344021912874e0,
                8.87381958642633073e0,
            ),
            (
                0.5,
                9.24419071227665645e-1,
                1.65644112000330068e0,
                1.52410938577390920e0,
                2.73100970821178546e0,
            ),
            (
                1.0,
                4.21024438240708232e-1,
                6.01907230197234577e-1,
                1.14446307980689466e0,
                1.63615348626325807e0,
            ),
            (
                2.0,
                1.13893872749533401e-1,
                1.39865881816522458e-1,
                8.41568215070771175e-1,
                1.03347684706868881e0,
            ),
            (
                8.5,
                8.62575663493250739e-5,
                9.11972477500689693e-5,
                4.23935999333698021e-1,
                4.48213391563079377e-1,
            ),
            (
                20.0,
                5.74123781533652375e-10,
                5.88305796955703838e-10,
                2.78544876657182194e-1,
                2.85425496940726442e-1,
            ),
        ] {
            let got_k0 = rv(k0(&s(x), m));
            let got_k1 = rv(k1(&s(x), m));
            let got_k0e = rv(k0e(&s(x), m));
            let got_k1e = rv(k1e(&s(x), m));
            assert!(
                ((got_k0 - k0_ref) / k0_ref).abs() < 1e-13,
                "k0({x})={got_k0:e} vs {k0_ref:e}"
            );
            assert!(
                ((got_k1 - k1_ref) / k1_ref).abs() < 1e-13,
                "k1({x})={got_k1:e} vs {k1_ref:e}"
            );
            assert!(
                ((got_k0e - k0e_ref) / k0e_ref).abs() < 1e-13,
                "k0e({x})={got_k0e:e} vs {k0e_ref:e}"
            );
            assert!(
                ((got_k1e - k1e_ref) / k1e_ref).abs() < 1e-13,
                "k1e({x})={got_k1e:e} vs {k1e_ref:e}"
            );
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
            (5.0, -0.177_596_771_314_338_3, -0.308_517_625_249_033_03),
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
    fn wright_bessel_small_a_series_matches_scipy_reference_values() -> Result<(), String> {
        // Small/moderate a exercises the many-term series (where log_term is now
        // streamed instead of recomputing gammaln(k+1) each term). References from
        // scipy.special.wright_bessel (1.17.1); asserted to 1e-12 relative.
        let cases = [
            (0.5, 1.0, 2.0, 6.6906279405071434),
            (0.1, 0.5, 3.0, 16.898117554956105),
            (1.0, 1.0, 5.0, 17.05777785336906),
            (0.05, 1.5, 1.0, 3.054807083429945),
            (2.0, 0.5, 4.0, 4.2987940689384816),
            (0.3, 2.0, 8.0, 454.78294776070891),
            (0.5, 0.5, 0.5, 1.2281663095913553),
        ];
        for (a, b, x, expected) in cases {
            let actual = real_value(tensor_result(wright_bessel(
                &scalar(a),
                &scalar(b),
                &scalar(x),
                RuntimeMode::Strict,
            ))?)?;
            assert!(
                (actual - expected).abs() <= 1e-12 * expected.abs().max(1.0),
                "wright_bessel({a}, {b}, {x}) = {actual}, expected {expected}"
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
    fn riccati_jn_recurrence_reuse_matches_repeated_reference() {
        fn repeated_reference(n: u32, x: f64) -> (Vec<f64>, Vec<f64>) {
            let mut s = Vec::with_capacity(n as usize + 1);
            let mut sp = Vec::with_capacity(n as usize + 1);
            for k in 0..=n {
                s.push(x * spherical_jn_nonneg(k, x));
            }
            sp.push(x.cos());
            for k in 1..=n as usize {
                let inv_x = if x.abs() < 1.0e-300 { 0.0 } else { 1.0 / x };
                sp.push(-((k as f64) * s[k] * inv_x) + s[k - 1]);
            }
            (s, sp)
        }

        let cases = [
            (0, 0.0),
            (1, -0.0),
            (1, 1.0e-16),
            (8, 0.125),
            (16, 16.0),
            (32, -10.0),
            (64, -128.0),
            (128, 512.0),
            (4, f64::INFINITY),
            (4, f64::NEG_INFINITY),
            (4, f64::NAN),
        ];
        for (n, x) in cases {
            let actual = riccati_jn(n, x);
            let expected = repeated_reference(n, x);
            for (index, (&got, &want)) in actual.0.iter().zip(&expected.0).enumerate() {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "S[{index}] for n={n}, x={x:?}"
                );
            }
            for (index, (&got, &want)) in actual.1.iter().zip(&expected.1).enumerate() {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "S'[{index}] for n={n}, x={x:?}"
                );
            }
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
    fn riccati_yn_recurrence_reuse_matches_repeated_reference() {
        fn repeated_reference(n: u32, x: f64) -> (Vec<f64>, Vec<f64>) {
            let mut c = Vec::with_capacity(n as usize + 1);
            let mut cp = Vec::with_capacity(n as usize + 1);
            for k in 0..=n {
                c.push(x * spherical_yn_nonneg(k, x));
            }
            cp.push(x.sin());
            for k in 1..=n as usize {
                let inv_x = if x.abs() < 1.0e-300 { 0.0 } else { 1.0 / x };
                cp.push(-((k as f64) * c[k] * inv_x) + c[k - 1]);
            }
            (c, cp)
        }

        let cases = [
            (0, 0.0),
            (1, -0.0),
            (8, 0.125),
            (32, -10.0),
            (128, 512.0),
            (4, f64::INFINITY),
            (4, f64::NEG_INFINITY),
            (4, f64::NAN),
        ];
        for (n, x) in cases {
            let actual = riccati_yn(n, x);
            let expected = repeated_reference(n, x);
            for (index, (&got, &want)) in actual.0.iter().zip(&expected.0).enumerate() {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "C[{index}] for n={n}, x={x:?}"
                );
            }
            for (index, (&got, &want)) in actual.1.iter().zip(&expected.1).enumerate() {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "C'[{index}] for n={n}, x={x:?}"
                );
            }
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
        let jn1 = jn_zeros(1, 3);
        let reused_jnp1 = jnp_zeros_from_function_zeros(1, 3, &jn1);
        for (&actual, &expected) in reused_jnp1.iter().zip(jnp1.iter()) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
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
                (result - expected).abs() < 1e-12,
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
                (result - expected).abs() < 1e-12,
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
