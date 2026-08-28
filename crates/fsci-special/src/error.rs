#![forbid(unsafe_code)]

use std::f64::consts::PI;
use std::simd::num::SimdFloat;
use std::simd::{Simd, StdFloat};

use fsci_runtime::RuntimeMode;

use crate::types::{
    Complex64, DispatchPlan, DispatchStep, KernelRegime, SpecialError, SpecialErrorKind,
    SpecialResult, SpecialTensor, record_special_trace,
};

pub const ERROR_DISPATCH_PLAN: &[DispatchPlan] = &[
    DispatchPlan {
        function: "erf",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "|z| < 1",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "|z| >= 1",
            },
            DispatchStep {
                regime: KernelRegime::BackendDelegate,
                when: "complex continuation region",
            },
        ],
        notes: "Strict mode preserves endpoint parity: erf(0)=0, erf(+/-inf)=+/-1.",
    },
    DispatchPlan {
        function: "erfc",
        steps: &[DispatchStep {
            regime: KernelRegime::BackendDelegate,
            when: "use dedicated complement kernel to avoid 1-erf cancellation",
        }],
        notes: "Central requirement: erf(x)+erfc(x)=1 within tolerance on finite reals.",
    },
    DispatchPlan {
        function: "erfinv",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "|y| < 0.9",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "0.9 <= |y| < 1",
            },
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "polish via Newton/Halley refinement",
            },
        ],
        notes: "Endpoints y=+/-1 map to +/-inf with strict SciPy parity.",
    },
    DispatchPlan {
        function: "erfcinv",
        steps: &[
            DispatchStep {
                regime: KernelRegime::BackendDelegate,
                when: "map to erfinv(1-y) with tail-stable correction",
            },
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "optional refinement iteration",
            },
        ],
        notes: "Domain is [0,2] in strict mode with hardened fail-closed diagnostics for malformed inputs.",
    },
];

const INV_SQRT_PI: f64 = 0.564_189_583_547_756_3;
const TWO_INV_SQRT_PI: f64 = 2.0 * INV_SQRT_PI;

pub fn erf(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    // Same SIMD lever as erfc (2c9c639b): the |x|≤1 rational is a byte-identical SIMD
    // Horner, and the |x|>1 tail is `sign(x)·(1 − exp(−x²)·erfcx_cephes(|x|))` — the
    // exp via `simd_exp` (≤4 ulp) and the rational SIMD. Below the parallel gate, where
    // the scalar map lost ~1.5x to SciPy's SIMD ufunc.
    if let SpecialTensor::RealVec(values) = z
        && (64..(1 << 20)).contains(&values.len())
    {
        return Ok(SpecialTensor::RealVec(erf_real_vec_simd(values)));
    }
    map_unary_input_rp(
        "erf",
        z,
        mode,
        |x| Ok(erf_scalar(x)),
        |value| Ok(erf_complex_scalar(value)),
        1 << 20, // work-gated: serial small, par_map_indices for huge arrays (>=1M)
    )
}

/// 8-wide erf over a real slice. `|x| ≤ 1`: the odd rational `x·T(x²)/U(x²)`
/// (byte-identical SIMD Horner). `1 < |x| < 25`: `sign(x)·(1 − erfc(|x|))` with
/// `erfc(|x|) = exp(−x²)·erfcx_cephes(|x|)` (simd_exp × SIMD rational). `|x| ≥ 25`:
/// scalar (erf ≈ ±1). Accurate to a few ulp vs the scalar map, inside erf's 1e-13 tol.
fn erf_real_vec_simd(values: &[f64]) -> Vec<f64> {
    const LANES: usize = 8;
    let mut out = vec![0.0f64; values.len()];
    let mut i = 0;
    while i + LANES <= values.len() {
        let x = Simd::<f64, LANES>::from_slice(&values[i..i + LANES]);
        let z = x * x;
        let rat = (x * simd_horner(z, &CEPHES_ERF_T, 0.0) / simd_horner(z, &CEPHES_ERF_U, 1.0))
            .to_array();
        let ez = simd_exp(-z).to_array(); // exp(−x²)
        let cx = erfcx_cephes_real_simd(x.abs()); // erfcx_cephes(|x|)
        for j in 0..LANES {
            let xj = values[i + j];
            let xaj = xj.abs();
            out[i + j] = if xaj <= 1.0 {
                rat[j]
            } else if xaj < 25.0 {
                let erfc_abs = ez[j] * cx[j]; // erfc(|x|)
                if xj > 0.0 {
                    1.0 - erfc_abs
                } else {
                    erfc_abs - 1.0
                }
            } else {
                erf_scalar(xj)
            };
        }
        i += LANES;
    }
    while i < values.len() {
        out[i] = erf_scalar(values[i]);
        i += 1;
    }
    out
}

pub fn erfc(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    // The scalar-per-element serial map lost ~1.4x to SciPy's SIMD-vectorized ufunc
    // (broad sweep 2a032a74). erfc(x) = exp(−x²)·(P/Q or R/S) for 1 ≤ x < 25, so both
    // the Cephes rational (byte-identical SIMD Horner) and the exp (`simd_exp`, ≤4 ulp
    // vs libm, well inside the 1e-13 erfc conformance tol) vectorise 8-wide; the x<1
    // (1−erf) and x≥25 (underflow) minority stay scalar. Below the parallel gate — the
    // serial regime where the loss lived.
    if let SpecialTensor::RealVec(values) = z
        && (64..(1 << 20)).contains(&values.len())
    {
        return Ok(SpecialTensor::RealVec(erfc_real_vec_simd(values)));
    }
    map_unary_input_rp(
        "erfc",
        z,
        mode,
        |x| Ok(erfc_scalar(x)),
        |value| Ok(erfc_complex_scalar(value)),
        1 << 20, // work-gated: serial small, par_map_indices for huge arrays (>=1M)
    )
}

/// 8-wide erfc over a real slice: `exp(−x²)·erfcx_cephes(x)` for the `1 ≤ x < 25`
/// majority (SIMD exp × SIMD rational), scalar `erfc_scalar` for x < 1 / x ≥ 25 /
/// negatives. Accurate to a few ulp vs the scalar map (simd_exp ~4 ulp + the
/// pre-divided rational), far inside erfc's 1e-13 conformance tolerance.
/// 8-wide erfc over ALL real u. `|u| < 1`: `1 − erf(u)` (byte-identical to
/// `erfc_scalar`, which takes the same branch). `1 ≤ |u| < 25`: `exp(−u²)·erfcx_cephes(|u|)`
/// for u > 0, else `2 − …` (a few ulp via `simd_exp`). `|u| ≥ 25`: scalar (0 or 2).
/// Shared by erfc's and ndtr's SIMD vector paths.
pub(crate) fn erfc_full_simd_chunk(u: Simd<f64, 8>) -> [f64; 8] {
    let z = u * u;
    let erf_rat =
        (u * simd_horner(z, &CEPHES_ERF_T, 0.0) / simd_horner(z, &CEPHES_ERF_U, 1.0)).to_array();
    let ez = simd_exp(-z).to_array();
    let cx = erfcx_cephes_real_simd(u.abs()); // valid for the |u| ≥ 1 lanes
    let uarr = u.to_array();
    let mut out = [0.0f64; 8];
    for j in 0..8 {
        let uj = uarr[j];
        let uaj = uj.abs();
        out[j] = if uaj < 1.0 {
            1.0 - erf_rat[j]
        } else if uaj < 25.0 {
            let tail = ez[j] * cx[j];
            if uj > 0.0 { tail } else { 2.0 - tail }
        } else {
            erfc_scalar(uj)
        };
    }
    out
}

fn erfc_real_vec_simd(values: &[f64]) -> Vec<f64> {
    const LANES: usize = 8;
    let mut out = vec![0.0f64; values.len()];
    let mut i = 0;
    while i + LANES <= values.len() {
        let chunk = erfc_full_simd_chunk(Simd::<f64, LANES>::from_slice(&values[i..i + LANES]));
        out[i..i + LANES].copy_from_slice(&chunk);
        i += LANES;
    }
    while i < values.len() {
        out[i] = erfc_scalar(values[i]);
        i += 1;
    }
    out
}

pub fn erfinv(y: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_unary_input_rp(
        "erfinv",
        y,
        mode,
        |v| erfinv_scalar(v, mode),
        |value| erfinv_complex_scalar(value, mode),
        1 << 20, // cheap ndtri-Newton (~23ns); default-256 gate lost 18.8x@4096 (BlackThrush A/B)
    )
}

pub fn erfcinv(y: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    // The arm is read ONCE for the whole batch and passed down as a `bool`, never read per
    // element: a relaxed load inside a per-item loop is an optimisation barrier this crate
    // has already paid 13% for, and a per-element `fetch_add` costs 2 instructions more.
    let ndtri_arm = ERFCINV_NDTRI.load(std::sync::atomic::Ordering::Relaxed);
    if ndtri_arm {
        ERFCINV_NDTRI_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    map_unary_input_rp(
        "erfcinv",
        y,
        mode,
        |v| erfcinv_scalar_with_arm(v, mode, ndtri_arm),
        |value| erfcinv_complex_scalar(value, mode),
        1 << 20, // cheap ~12ns; default-256 gate lost 36.9x@4096, still loses at 262k (BlackThrush A/B)
    )
}

/// Use the crate's own `ndtri` for `erfcinv` (`true`, shipping) instead of Acklam's rational.
///
/// ACCURACY CONTRACT, and it is the entire reason this switch exists. Acklam's approximation
/// carries a published maximum relative error of 1.15e-9 and is used here with NO refinement
/// step, so `erfcinv` returned about nine correct digits while every neighbouring function in
/// this crate returns fifteen or sixteen. Measured against live SciPy over 200000 points on
/// `[0.001, 1.999]`, the Acklam arm reads `max_abs=2.28e-09`; the whole rest of the
/// one-argument sweep sits at 1e-11 or better. This arm is not a speed lever and no speed
/// claim is attached to it.
pub static ERFCINV_NDTRI: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
/// BATCHES that took the `ndtri` arm — "enabled" is not "took effect".
///
/// Counted once per batch, not once per element, so it is not a call count. A diagnostic
/// counter with no arms to preserve anything between; incrementing it changes no result.
pub static ERFCINV_NDTRI_HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Evaluate `f(0..n)` into a `Vec<T>`, parallel over index chunks for large `n`.
/// Error-function kernels (erf series / erfc continued fraction / erfinv–erfcinv Newton
/// refinement) are non-trivial per element and each index writes its own slot, so chunking
/// across cores and concatenating in index order is bit-identical to `(0..n).map(f).collect()`
/// — including returning the first failing index's error in index order. Generic over the
/// output type (f64 or Complex64).
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
            .map(|h| h.join().expect("error-fn array worker panicked"))
            .collect()
    });
    let mut out = Vec::with_capacity(n);
    for cr in chunk_results {
        out.extend(cr?);
    }
    Ok(out)
}

// Unreferenced: every entry point in this module calls map_unary_input_rp with
// an explicit parallel threshold instead. RETAINED with a note rather than
// deleted (frankenscipy-e2ve2), matching map_real_input in gamma.rs and
// map_real_or_complex in elliptic.rs; deletion is the owner's call per
// frankenscipy-iit9c.
#[allow(dead_code)]
fn map_unary_input<F, G>(
    function: &'static str,
    input: &SpecialTensor,
    mode: RuntimeMode,
    real_kernel: F,
    complex_kernel: G,
) -> SpecialResult
where
    F: Fn(f64) -> Result<f64, SpecialError> + Sync,
    G: Fn(Complex64) -> Result<Complex64, SpecialError> + Sync,
{
    map_unary_input_rp(function, input, mode, real_kernel, complex_kernel, 256)
}

/// `real_par_min` = the RealVec length below which the real arm runs SERIALLY. Cheap O(1) real
/// kernels (erf/erfc rational ~25ns/call) lose to serial under par_map_indices for short/medium
/// arrays (thread-spawn overhead >> per-call work) but WIN for huge arrays where the kernel cost
/// finally dominates: measured same-worker erf/erfc flip from a slight SciPy loss to a 2.3-2.8x win
/// at n>=2M. So they pass `1<<20` (work-gated). Heavy real kernels (erfinv/erfcinv iterative) keep
/// the default 256 so they still parallelize at lower lengths.
fn map_unary_input_rp<F, G>(
    function: &'static str,
    input: &SpecialTensor,
    mode: RuntimeMode,
    real_kernel: F,
    complex_kernel: G,
    real_par_min: usize,
) -> SpecialResult
where
    F: Fn(f64) -> Result<f64, SpecialError> + Sync,
    G: Fn(Complex64) -> Result<Complex64, SpecialError> + Sync,
{
    match input {
        SpecialTensor::RealScalar(x) => real_kernel(*x).map(SpecialTensor::RealScalar),
        SpecialTensor::RealVec(values) => {
            if values.len() < real_par_min {
                values
                    .iter()
                    .map(|&x| real_kernel(x))
                    .collect::<Result<Vec<_>, _>>()
                    .map(SpecialTensor::RealVec)
            } else {
                par_map_indices(values.len(), |i| real_kernel(values[i]))
                    .map(SpecialTensor::RealVec)
            }
        }
        SpecialTensor::ComplexScalar(value) => {
            complex_kernel(*value).map(SpecialTensor::ComplexScalar)
        }
        SpecialTensor::ComplexVec(values) => {
            par_map_indices(values.len(), |i| complex_kernel(values[i]))
                .map(SpecialTensor::ComplexVec)
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

// Cephes rational erf/erfc — scipy's `xsf` wraps these EXACT coefficients, so this matches
// `scipy.special.erf`/`erfc` to the bit while replacing the iterative Maclaurin series + Lentz
// continued fraction with O(1) polynomial evaluation (~2-6× faster kernel). frankenscipy-9l5oo.
// polevl = Horner from coef[0]; p1evl assumes an implicit leading 1.0 (Cephes convention).
fn cephes_polevl(x: f64, coef: &[f64]) -> f64 {
    coef.iter().fold(0.0, |acc, &c| acc * x + c)
}
fn cephes_p1evl(x: f64, coef: &[f64]) -> f64 {
    coef.iter().fold(1.0, |acc, &c| acc * x + c)
}
const CEPHES_ERF_T: [f64; 5] = [
    9.60497373987051638749E0,
    9.00260197203842689217E1,
    2.23200534594684319226E3,
    7.00332514112805075473E3,
    5.55923013010394962768E4,
];
const CEPHES_ERF_U: [f64; 5] = [
    3.35617141647503099647E1,
    5.21357949780152679795E2,
    4.59432382970980127987E3,
    2.26290000613890934246E4,
    4.92673942608635921086E4,
];
const CEPHES_ERFC_P: [f64; 9] = [
    2.46196981473530512524E-10,
    5.64189564831068821977E-1,
    7.46321056442269912687E0,
    4.86371970985681366614E1,
    1.96520832956077098242E2,
    5.26445194995477358631E2,
    9.34528527171957607540E2,
    1.02755188689515710272E3,
    5.57535335369399327526E2,
];
const CEPHES_ERFC_Q: [f64; 8] = [
    1.32281951154744992508E1,
    8.67072140885989742329E1,
    3.54937778887819891062E2,
    9.75708501743205489753E2,
    1.82390916687909736289E3,
    2.24633760818710981792E3,
    1.65666309194161350182E3,
    5.57535340817727675546E2,
];
const CEPHES_ERFC_R: [f64; 6] = [
    5.64189583547755073984E-1,
    1.27536670759978104416E0,
    5.01905042251180477414E0,
    6.16021097993053585195E0,
    7.40974269950448939160E0,
    2.97886665372100240670E0,
];
const CEPHES_ERFC_S: [f64; 6] = [
    2.26052863220117276590E0,
    9.39603524938001434673E0,
    1.20489539808096656605E1,
    1.70814450747565897222E1,
    9.60896809063285878198E0,
    3.36907645100081516050E0,
];
const CEPHES_MAXLOG: f64 = 7.08396418532264106224E2;

pub fn erf_scalar(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return x.signum();
    }
    if x < 0.0 {
        return -erf_scalar(-x);
    }
    // Cephes: for |x| > 1 use 1 − erfc(x); else the rational x·T(x²)/U(x²).
    if x > 1.0 {
        return 1.0 - erfc_scalar(x);
    }
    let z = x * x;
    x * cephes_polevl(z, &CEPHES_ERF_T) / cephes_p1evl(z, &CEPHES_ERF_U)
}

pub(crate) fn erf_complex_scalar(z: Complex64) -> Complex64 {
    if z.re < 0.0 {
        return -erf_complex_scalar(-z);
    }
    // Use the Maclaurin series only for small |z| (≤ 4; the 80-term sum stays
    // above the cancellation regime there — the threshold also avoids the
    // ~1e-12 jitter near z ≈ 4.5 that broke CDF monotonicity in
    // [frankenscipy-evspb]). The previous `|| z.re < 1.0` clause routed every
    // small-real-part argument to the series regardless of |Im|, but for large
    // imaginary part the series peaks past term 80 and cancels (max term
    // ~e^{|z|²}): erf(0.1+10i) was 97% off, erf(0.5-20i) ~100%. For |z| > 4 use
    // the Faddeeva relation erfc(z) = e^{-z²} w(iz): w (wofz) is accurate across
    // the whole upper half plane (iz has Im = Re(z) ≥ 0 after the reflection
    // above), with no asymptotic-floor gap near |z| ≈ 4. frankenscipy-foy2t.
    if z.abs() <= 4.0 {
        return erf_complex_series(z);
    }
    // i·z = (-Im(z)) + i·Re(z); after the Re<0 reflection Re(z) ≥ 0 so iz is in
    // wofz's native upper half plane.
    let iz = Complex64::new(-z.im, z.re);
    let w = crate::convenience::wofz_scalar(iz, RuntimeMode::Strict)
        .unwrap_or(Complex64::new(f64::NAN, f64::NAN));
    Complex64::from_real(1.0) - (-(z * z)).exp() * w
}

pub fn erfc_scalar(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x.is_sign_positive() { 0.0 } else { 2.0 };
    }
    // Cephes erfc: rational P/Q (1 ≤ |x| < 8) or R/S (|x| ≥ 8) times e^{-x²}; |x| < 1 → 1−erf.
    let xa = x.abs();
    if xa < 1.0 {
        return 1.0 - erf_scalar(x);
    }
    let z = -x * x;
    if z < -CEPHES_MAXLOG {
        return if x < 0.0 { 2.0 } else { 0.0 };
    }
    let z = z.exp();
    let (p, q) = if xa < 8.0 {
        (
            cephes_polevl(xa, &CEPHES_ERFC_P),
            cephes_p1evl(xa, &CEPHES_ERFC_Q),
        )
    } else {
        (
            cephes_polevl(xa, &CEPHES_ERFC_R),
            cephes_p1evl(xa, &CEPHES_ERFC_S),
        )
    };
    let y = z * p / q;
    if x < 0.0 { 2.0 - y } else { y }
}

/// Lentz continued fraction kernel for erfc/erfcx (x ≥ ~1):
///   1/(x + ½/(x + 1/(x + 3/2/(x + 2/(x + …))))).
/// erfc(x) = e^{-x²}/√π · h,  erfcx(x) = e^{x²}erfc(x) = h/√π.
fn erfc_cf_h(x: f64) -> f64 {
    const FPMIN: f64 = 1e-300;
    const EPS: f64 = 1e-16;
    let mut c = 1.0 / FPMIN;
    let mut d = 1.0 / x;
    let mut h = d;
    for i in 1..400 {
        let a = i as f64 / 2.0;
        d = x + a * d;
        if d == 0.0 {
            d = FPMIN;
        }
        d = 1.0 / d;
        c = x + a / c;
        if c == 0.0 {
            c = FPMIN;
        }
        let del = c * d;
        h *= del;
        if (del - 1.0).abs() <= EPS {
            break;
        }
    }
    h
}

/// Scaled complementary error function erfcx(x) = e^{x²}·erfc(x) for x ≥ ~1,
/// from the continued fraction (no overflow of the intermediate e^{x²}). Used by
/// erfcinv's deep-tail Newton iteration.
pub(crate) fn erfcx_cf_real(x: f64) -> f64 {
    erfc_cf_h(x) / PI.sqrt()
}

/// erfcx(x) = e^{x²}·erfc(x) for x ≥ 1 via the Cephes rational DIRECTLY. Since
/// `erfc_scalar` computes erfc(x) = e^{−x²}·P(x)/Q(x), the scaled form is just
/// P(x)/Q(x) — computing it here avoids the e^{x²}·e^{−x²} round-trip (two `exp`
/// calls + their cancellation error) that `e^{x²}·erfc_scalar(x)` incurred.
pub(crate) fn erfcx_cephes_real(x: f64) -> f64 {
    let xa = x.abs(); // caller guarantees x ≥ 1
    if xa < 8.0 {
        cephes_polevl(xa, &CEPHES_ERFC_P) / cephes_p1evl(xa, &CEPHES_ERFC_Q)
    } else {
        cephes_polevl(xa, &CEPHES_ERFC_R) / cephes_p1evl(xa, &CEPHES_ERFC_S)
    }
}

/// 8-wide Cephes Horner: `Σ coef[k]·x^(n-k)` starting from `init` (0.0 = polevl,
/// 1.0 = p1evl leading-1). Each lane runs the IDENTICAL fold as `cephes_polevl`/
/// `cephes_p1evl` (same op order, no FMA contraction), so it is bit-for-bit the
/// scalar result per lane.
fn simd_horner(x: Simd<f64, 8>, coef: &[f64], init: f64) -> Simd<f64, 8> {
    let mut acc = Simd::splat(init);
    for &c in coef {
        acc = acc * x + Simd::splat(c);
    }
    acc
}

// Cephes `exp` rational constants (exp.c). Not bit-identical to libm `exp`, but the
// same algorithm SciPy's own transcendental loop uses — < 1 ulp over the reduced
// range, validated against `f64::exp` in `check_simd_exp`.
const SIMD_EXP_LOG2E: f64 = std::f64::consts::LOG2_E;
const SIMD_EXP_C1: f64 = 6.931_457_519_531_25e-1; // ln2 Cody-Waite: truncated high part …
const SIMD_EXP_C2: f64 = 1.428_606_820_309_417_232_12e-6; // … + low correction (C1+C2 = ln2)
const SIMD_EXP_P: [f64; 3] = [
    1.261_771_930_748_105_908_8e-4,
    3.029_944_077_074_419_613_0e-2,
    9.999_999_999_999_999_999_1e-1,
];
const SIMD_EXP_Q: [f64; 4] = [
    3.001_985_051_386_644_550_4e-6,
    2.524_483_403_496_841_041_9e-3,
    2.272_655_482_081_550_287_7e-1,
    2.000_000_000_000_000_000_1e0,
];

/// 8-wide `exp` via the Cephes rational: Cody-Waite range reduction `x = r + n·ln2`
/// (n = round(x/ln2)), the degree-2/3 rational on the reduced `r`, and `·2^n` by
/// exponent-bit construction. ~1 ulp; the caller keeps `n` in `[-1022, 1023]` (for
/// erfc, `x = −xᵢ² ∈ [−625, −1]` on the `1 ≤ xᵢ < 25` gate → `n ∈ [−901, −1]`).
fn simd_exp(x: Simd<f64, 8>) -> Simd<f64, 8> {
    let n = (Simd::splat(SIMD_EXP_LOG2E) * x + Simd::splat(0.5)).floor();
    let xr = x - n * Simd::splat(SIMD_EXP_C1) - n * Simd::splat(SIMD_EXP_C2);
    let xx = xr * xr;
    let px = xr * simd_horner(xx, &SIMD_EXP_P, 0.0);
    let qx = simd_horner(xx, &SIMD_EXP_Q, 0.0);
    let frac = px / (qx - px);
    let mantissa = (Simd::splat(1.0) + Simd::splat(2.0) * frac).to_array();
    // ldexp(mantissa, n): scale by 2^n via the biased-exponent bits (cheap scalar
    // lane loop; the rational above — the expensive part — stayed SIMD).
    let n = n.to_array();
    let mut out = [0.0f64; 8];
    for k in 0..8 {
        let ni = n[k] as i64;
        out[k] = mantissa[k] * f64::from_bits(((ni + 1023) as u64) << 52);
    }
    Simd::from_array(out)
}

/// 8-wide [`erfcx_cephes_real`]: per lane bit-identical to the scalar kernel (same
/// P/Q vs R/S split at `xa < 8`, same Horner). Caller guarantees each lane `x ≥ 1`.
/// Both rationals are evaluated SIMD (the dominant cost); the cheap P/Q-vs-R/S
/// pick is a scalar lane loop, returned as an array.
pub(crate) fn erfcx_cephes_real_simd(x: Simd<f64, 8>) -> [f64; 8] {
    let xa = x.abs();
    let pq =
        (simd_horner(xa, &CEPHES_ERFC_P, 0.0) / simd_horner(xa, &CEPHES_ERFC_Q, 1.0)).to_array();
    let rs =
        (simd_horner(xa, &CEPHES_ERFC_R, 0.0) / simd_horner(xa, &CEPHES_ERFC_S, 1.0)).to_array();
    let xa = xa.to_array();
    let mut out = [0.0f64; 8];
    for k in 0..8 {
        out[k] = if xa[k] < 8.0 { pq[k] } else { rs[k] };
    }
    out
}

fn erfc_complex_scalar(z: Complex64) -> Complex64 {
    if z.re < 0.0 {
        return Complex64::from_real(2.0) - erfc_complex_scalar(-z);
    }
    // Threshold lowered from 4.5 to 4.0: the series form computes
    // erfc(z) = 1 − erf_series(z), which catastrophically cancels
    // when erf(z) is near 1 (z ≳ 4). Caught by [frankenscipy-evspb]
    // fuzz: erfc_scalar(4.4956) returned −1.5e-10 (negative!),
    // breaking standard_normal_cdf and PowerLognorm.cdf in the
    // deep tail. 4.0 is the tightest split that keeps the
    // asymptotic series above its precision floor (~3e-6 at
    // z=3.5) while still catching the cancellation regime.
    // See erf_complex_scalar: gate the series on |z| only (the old `|| z.re<1.0`
    // sent large-imaginary-part arguments to the truncating/cancelling series),
    // and use the Faddeeva relation erfc(z) = e^{-z²} w(iz) for |z| > 4.
    if z.abs() <= 4.0 {
        return Complex64::from_real(1.0) - erf_complex_series(z);
    }
    let iz = Complex64::new(-z.im, z.re);
    let w = crate::convenience::wofz_scalar(iz, RuntimeMode::Strict)
        .unwrap_or(Complex64::new(f64::NAN, f64::NAN));
    (-(z * z)).exp() * w
}

fn erf_complex_series(z: Complex64) -> Complex64 {
    let z2 = z * z;
    let mut term = z;
    let mut sum = term;

    for n in 0..80 {
        let numer = -z2 * ((2 * n + 1) as f64);
        let denom = ((n + 1) * (2 * n + 3)) as f64;
        term = term * numer / denom;
        sum = sum + term;
        if n >= 4 && term.abs() <= 1.0e-16 * sum.abs().max(1.0) {
            break;
        }
    }

    sum * TWO_INV_SQRT_PI
}

// Superseded by the Faddeeva (wofz) relation in erf/erfc_complex_scalar, which
// has no asymptotic-floor gap near |z| ≈ 4. Kept as a reference implementation.
#[allow(dead_code)]
fn erfc_complex_asymptotic(z: Complex64) -> Complex64 {
    let z2 = z * z;
    let mut term = Complex64::from_real(1.0);
    let mut sum = term;

    let mut last_term_abs = term.abs();
    for n in 0..60 {
        let factor = -((2 * n + 1) as f64);
        let next_term = term * factor / (z2 * 2.0);
        let next_term_abs = next_term.abs();

        if next_term_abs > last_term_abs {
            break; // Series starts to diverge
        }

        term = next_term;
        last_term_abs = next_term_abs;
        sum = sum + term;

        if n >= 4 && term.abs() <= 1.0e-16 * sum.abs().max(1.0) {
            break;
        }
    }

    ((-z2).exp() * sum / z) / PI.sqrt()
}

pub fn erfinv_scalar(y: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if y.is_nan() {
        return Ok(f64::NAN);
    }
    if y == 1.0 {
        return Ok(f64::INFINITY);
    }
    if y == -1.0 {
        return Ok(f64::NEG_INFINITY);
    }
    if y.abs() > 1.0 {
        return match mode {
            RuntimeMode::Strict => {
                record_special_trace(
                    "erfinv",
                    mode,
                    "domain_error",
                    format!("input={y}"),
                    "returned_nan",
                    "out-of-domain strict fallback",
                    false,
                );
                Ok(f64::NAN)
            }
            RuntimeMode::Hardened => {
                record_special_trace(
                    "erfinv",
                    mode,
                    "domain_error",
                    format!("input={y}"),
                    "fail_closed",
                    "erfinv domain is [-1, 1]",
                    false,
                );
                Err(SpecialError {
                    function: "erfinv",
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "erfinv domain is [-1, 1]",
                })
            }
        };
    }
    if y == 0.0 {
        return Ok(y);
    }

    let p = 0.5 * (y + 1.0);
    if p == 0.0 || p == 1.0 {
        return Ok(y.signum() * crate::convenience::erfcinv_conv(1.0 - y.abs()));
    }

    Ok(crate::convenience::ndtri_scalar(p) * std::f64::consts::FRAC_1_SQRT_2)
}

fn erfinv_complex_scalar(y: Complex64, mode: RuntimeMode) -> Result<Complex64, SpecialError> {
    if !y.re.is_finite() || !y.im.is_finite() {
        return Ok(Complex64::new(f64::NAN, f64::NAN));
    }
    if y.im == 0.0 {
        return erfinv_scalar(y.re, mode).map(Complex64::from_real);
    }
    if y.re < 0.0 || (y.re == 0.0 && y.im < 0.0) {
        return erfinv_complex_scalar(-y, mode).map(|value| -value);
    }
    if y == Complex64::new(0.0, 0.0) {
        return Ok(y);
    }

    let mut x = erfinv_complex_initial_guess(y);
    if !x.re.is_finite() || !x.im.is_finite() {
        x = y * (PI.sqrt() / 2.0);
    }

    for _ in 0..20 {
        let fx = erf_complex_scalar(x) - y;
        let dfx = (-x * x).exp() * TWO_INV_SQRT_PI;
        if dfx.abs() < 1.0e-300 {
            break;
        }
        let correction = fx / dfx;
        x = x - correction;
        if correction.abs() <= 1.0e-14 * x.abs().max(1.0) {
            break;
        }
    }

    if !x.re.is_finite() || !x.im.is_finite() {
        return match mode {
            RuntimeMode::Strict => Ok(Complex64::new(f64::NAN, f64::NAN)),
            RuntimeMode::Hardened => Err(SpecialError {
                function: "erfinv",
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "complex principal-branch iteration failed to converge",
            }),
        };
    }

    Ok(x)
}

fn erfinv_complex_initial_guess(y: Complex64) -> Complex64 {
    if y.abs() < 0.75 {
        let y3 = y * y * y;
        let y5 = y3 * y * y;
        let c1 = PI.sqrt() / 2.0;
        let c3 = PI.powf(1.5) / 24.0;
        let c5 = 7.0 * PI.powf(2.5) / 960.0;
        return y * c1 + y3 * c3 + y5 * c5;
    }

    let a = 0.147;
    let one_minus_y2 = Complex64::from_real(1.0) - y * y;
    let log_term = one_minus_y2.ln();
    let t = Complex64::from_real(2.0 / (PI * a)) + log_term * 0.5;
    complex_sqrt(complex_sqrt(t * t - log_term / a) - t)
}

fn erfcinv_scalar(y: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    erfcinv_scalar_with_arm(
        y,
        mode,
        ERFCINV_NDTRI.load(std::sync::atomic::Ordering::Relaxed),
    )
}

/// `erfcinv_scalar` with the kernel choice passed IN rather than read from the process-global
/// switch on every element. See `erfcinv` for why the read is hoisted.
fn erfcinv_scalar_with_arm(
    y: f64,
    mode: RuntimeMode,
    ndtri_arm: bool,
) -> Result<f64, SpecialError> {
    if y.is_nan() {
        return Ok(f64::NAN);
    }
    if y == 0.0 {
        return Ok(f64::INFINITY);
    }
    if y == 2.0 {
        return Ok(f64::NEG_INFINITY);
    }
    if !(0.0..=2.0).contains(&y) {
        return match mode {
            RuntimeMode::Strict => {
                record_special_trace(
                    "erfcinv",
                    mode,
                    "domain_error",
                    format!("input={y}"),
                    "returned_nan",
                    "out-of-domain strict fallback",
                    false,
                );
                Ok(f64::NAN)
            }
            RuntimeMode::Hardened => {
                record_special_trace(
                    "erfcinv",
                    mode,
                    "domain_error",
                    format!("input={y}"),
                    "fail_closed",
                    "erfcinv domain is [0, 2]",
                    false,
                );
                Err(SpecialError {
                    function: "erfcinv",
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "erfcinv domain is [0, 2]",
                })
            }
        };
    }

    if y == 1.0 {
        return Ok(0.0);
    }

    // erfcinv(y) = -Phi^-1(y/2)/sqrt(2). Both arms compute that identity; they differ only in
    // which inverse-normal kernel evaluates it. `ndtri_scalar` is the crate's Cephes-derived
    // one, already trusted by `erfinv_scalar` three functions up and by `erfcinv_conv`, and it
    // handles the deep tail by delegating there rather than by losing digits.
    if ndtri_arm {
        Ok(-crate::convenience::ndtri_scalar(0.5 * y) * std::f64::consts::FRAC_1_SQRT_2)
    } else {
        Ok(-inv_norm_cdf_scalar(0.5 * y) / 2.0_f64.sqrt())
    }
}

fn erfcinv_complex_scalar(y: Complex64, mode: RuntimeMode) -> Result<Complex64, SpecialError> {
    if y.im == 0.0 {
        return erfcinv_scalar(y.re, mode).map(Complex64::from_real);
    }
    erfinv_complex_scalar(Complex64::from_real(1.0) - y, mode)
}

fn complex_sqrt(z: Complex64) -> Complex64 {
    if z.re == 0.0 && z.im == 0.0 {
        return Complex64::from_real(0.0);
    }
    if !z.is_finite() {
        return Complex64::new(f64::NAN, f64::NAN);
    }
    let radius = z.abs();
    let real = ((radius + z.re) / 2.0).max(0.0).sqrt();
    let imag_mag = ((radius - z.re) / 2.0).max(0.0).sqrt();
    let imag = if z.im.is_sign_negative() {
        -imag_mag
    } else {
        imag_mag
    };
    Complex64::new(real, imag)
}

fn inv_norm_cdf_scalar(p: f64) -> f64 {
    if p.is_nan() {
        return f64::NAN;
    }
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }

    // Acklam's rational approximation for the inverse normal CDF.
    const P_LOW: f64 = 0.024_25;
    const P_HIGH: f64 = 1.0 - P_LOW;

    const A: [f64; 6] = [
        -3.969_683_028_665_376e+01,
        2.209_460_984_245_205e+02,
        -2.759_285_104_469_687e+02,
        1.383_577_518_672_69e+02,
        -3.066_479_806_614_716e+01,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e+01,
        1.615_858_368_580_409e+02,
        -1.556_989_798_598_866e+02,
        6.680_131_188_771_972e+01,
        -1.328_068_155_288_572e+01,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-03,
        -3.223_964_580_411_365e-01,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-03,
        3.224_671_290_700_398e-01,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_simd_matches_scalar_within_tol() {
        // erf's SIMD vector path vs the scalar kernel across every branch/sign boundary.
        let mut xs: Vec<f64> = (0..2000)
            .map(|i| -28.0 + 56.0 * (i as f64) / 2000.0)
            .collect();
        for b in [
            -1.0, 1.0, -25.0, 25.0, 0.9999999, -0.9999999, 0.0, 2.0, -2.0,
        ] {
            xs.push(b);
        }
        let simd = match erf(&SpecialTensor::RealVec(xs.clone()), RuntimeMode::Strict).unwrap() {
            SpecialTensor::RealVec(v) => v,
            _ => unreachable!(),
        };
        let mut max_abs = 0.0f64;
        for (k, &x) in xs.iter().enumerate() {
            max_abs = max_abs.max((simd[k] - erf_scalar(x)).abs());
        }
        assert!(
            max_abs < 1e-14,
            "erf simd max abs diff vs scalar = {max_abs:e}"
        );
    }

    #[test]
    fn erfc_simd_matches_scalar_within_tol() {
        // erfc's SIMD vector path vs the scalar kernel across every branch boundary.
        let mut xs: Vec<f64> = (0..2000)
            .map(|i| -1.0 + 27.0 * (i as f64) / 2000.0)
            .collect();
        for b in [1.0, 8.0, 25.0, 0.9999999, 24.9999999, -3.0, 2.0, 0.0] {
            xs.push(b);
        }
        let simd = match erfc(&SpecialTensor::RealVec(xs.clone()), RuntimeMode::Strict).unwrap() {
            SpecialTensor::RealVec(v) => v,
            _ => unreachable!(),
        };
        let mut max_rel = 0.0f64;
        for (k, &x) in xs.iter().enumerate() {
            let s = erfc_scalar(x);
            let rel = if s.abs() > 0.0 {
                (simd[k] - s).abs() / s.abs()
            } else {
                (simd[k] - s).abs()
            };
            max_rel = max_rel.max(rel);
        }
        assert!(
            max_rel < 1e-14,
            "erfc simd max rel diff vs scalar = {max_rel:e}"
        );
    }

    #[test]
    fn simd_exp_matches_libm_within_a_few_ulp() {
        // erfc's SIMD path uses simd_exp for the arg range [-625, -1]; validate the
        // whole [-708, 0] against f64::exp (both positive → bit distance == ulp).
        let mut xs: Vec<f64> = Vec::new();
        let mut t = -708.0_f64;
        while t <= 0.0 {
            xs.push(t);
            t += 0.0007;
        }
        while !xs.len().is_multiple_of(8) {
            xs.push(0.0);
        }
        let mut max_ulp = 0i64;
        // xs was padded to a multiple of 8 above, so the remainder is empty.
        let (chunks, _remainder) = xs.as_chunks::<8>();
        for chunk in chunks {
            let got = simd_exp(Simd::<f64, 8>::from_array(*chunk)).to_array();
            for (k, &xv) in chunk.iter().enumerate() {
                let want = xv.exp();
                let ulp = ((got[k].to_bits() as i64) - (want.to_bits() as i64)).abs();
                max_ulp = max_ulp.max(ulp);
            }
        }
        assert!(max_ulp <= 4, "simd_exp max ulp vs libm = {max_ulp}");
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy
    fn erf_complex_large_imaginary_matches_scipy() {
        // frankenscipy-foy2t: the `|| z.re < 1.0` series gate sent small-real /
        // large-imaginary arguments to the 80-term Maclaurin series, which
        // truncates before convergence and cancels (max term ~e^{|z|²}):
        // erf(0.1+10i) was 97% off, erf(0.5-20i) ~100%. The Faddeeva relation
        // erfc(z)=e^{-z²}w(iz) for |z|>4 fixes it. (re, im, erf.re, erf.im) from
        // scipy.special.erf 1.17.1.
        let cases: [(f64, f64, f64, f64); 6] = [
            (0.1, 10.0, 1.3784606413850375e42, -6.140976128501518e41),
            (0.5, 10.0, -5.9398727494098764e41, -1.0260784858252674e42),
            (0.1, -8.0, 4.387388758811265e26, 7.240275368450693e24),
            (-8.0, 8.0, -1.0498517541570318, 0.0011870025535653562),
            (1.0, 4.0, 456592.30438094615, 52731.820367670356),
            (0.5, -20.0, 1.036185736591006e172, -4.946816335504394e171),
        ];
        for (re, im, wr, wi) in cases {
            let g = erf_complex_scalar(Complex64::new(re, im));
            let denom = wr.hypot(wi);
            let err = (g.re - wr).hypot(g.im - wi) / denom;
            assert!(
                err <= 1e-9,
                "erf({re}{im:+}i) = {g:?}, scipy ({wr}, {wi}), rel {err:e}"
            );
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy
    fn erfc_continued_fraction_matches_scipy() {
        // frankenscipy-8nkg4: erfc via Lentz CF for x≥1 (the 1-erf series was
        // ~1e-6 off near x≈3.5 from cancellation). scipy.special.erfc 1.17.1.
        let cases = [
            (1.5, 0.03389485352468927),
            (2.0, 0.004677734981047266),
            (3.0, 2.2090496998585445e-05),
            (3.5355, 5.734457379363329e-07),
            (4.0, 1.541725790028002e-08),
            (5.0, 1.5374597944280347e-12),
            (-3.5, 1.9999992569016276),
        ];
        for (x, expected) in cases {
            let got = erfc_scalar(x);
            assert!(
                ((got - expected) / expected).abs() < 1e-13,
                "erfc({x}) = {got:e}, scipy {expected:e}"
            );
        }
    }

    fn tensor_result(result: SpecialResult) -> Result<SpecialTensor, String> {
        result.map_err(|err| format!("{err:?}"))
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

    fn scalar(value: f64) -> SpecialTensor {
        SpecialTensor::RealScalar(value)
    }

    fn complex_scalar(re: f64, im: f64) -> SpecialTensor {
        SpecialTensor::ComplexScalar(Complex64::new(re, im))
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
    fn complex_sqrt_preserves_signed_zero_branch_on_negative_real_axis() {
        let upper = complex_sqrt(Complex64::new(-1.0, 0.0));
        let lower = complex_sqrt(Complex64::new(-1.0, -0.0));
        assert!(upper.re.abs() < 1.0e-12);
        assert!(lower.re.abs() < 1.0e-12);
        assert!((upper.im - 1.0).abs() < 1.0e-12);
        assert!((lower.im + 1.0).abs() < 1.0e-12);
        assert!(upper.im.is_sign_positive());
        assert!(lower.im.is_sign_negative());
    }

    #[test]
    fn complex_erfinv_real_axis_reduces_to_scalar_path() -> Result<(), String> {
        for y in [-0.9, -0.5, 0.0, 0.5, 0.9] {
            let real_result = real_value(tensor_result(erfinv(&scalar(y), RuntimeMode::Strict))?)?;
            let complex_result = complex_value(tensor_result(erfinv(
                &complex_scalar(y, 0.0),
                RuntimeMode::Strict,
            ))?)?;
            assert!((complex_result.re - real_result).abs() < 1.0e-11);
            assert!(complex_result.im.abs() < 1.0e-11);
        }
        Ok(())
    }

    #[test]
    fn complex_erfcinv_real_axis_reduces_to_scalar_path() -> Result<(), String> {
        for y in [0.1, 0.5, 1.0, 1.5, 1.9] {
            let real_result = real_value(tensor_result(erfcinv(&scalar(y), RuntimeMode::Strict))?)?;
            let complex_result = complex_value(tensor_result(erfcinv(
                &complex_scalar(y, 0.0),
                RuntimeMode::Strict,
            ))?)?;
            assert!((complex_result.re - real_result).abs() < 1.0e-11);
            assert!(complex_result.im.abs() < 1.0e-11);
        }
        Ok(())
    }

    #[test]
    fn complex_erfinv_roundtrips_complex_erf_principal_branch() -> Result<(), String> {
        let z = Complex64::new(0.5, 0.25);
        let y = complex_value(tensor_result(erf(
            &SpecialTensor::ComplexScalar(z),
            RuntimeMode::Strict,
        ))?)?;
        let recovered = complex_value(tensor_result(erfinv(
            &SpecialTensor::ComplexScalar(y),
            RuntimeMode::Strict,
        ))?)?;
        assert_complex_close(recovered, z, 1.0e-10);
        Ok(())
    }

    #[test]
    fn complex_erfcinv_preserves_conjugation_over_vectors() -> Result<(), String> {
        let y = Complex64::new(0.7, 0.25);
        let inputs = SpecialTensor::ComplexVec(vec![y, y.conj()]);
        let outputs = tensor_result(erfcinv(&inputs, RuntimeMode::Strict))?;
        let values = match outputs {
            SpecialTensor::ComplexVec(values) => values,
            other => return Err(format!("expected ComplexVec, got {other:?}")),
        };
        assert_eq!(values.len(), 2);
        assert_complex_close(values[1], values[0].conj(), 1.0e-10);
        Ok(())
    }

    #[test]
    fn erf_metamorphic_complementary_identity() {
        // /testing-metamorphic: erf(x) + erfc(x) = 1 for all x.
        // Independent of any specific value, so it catches drift in
        // either function without hard-coded references. Verify
        // across small, moderate, and tail arguments.
        for &x in &[-3.0_f64, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.5] {
            let lhs = erf_scalar(x) + erfc_scalar(x);
            assert!(
                (lhs - 1.0).abs() < 1e-15,
                "erf({x}) + erfc({x}) = {lhs}, expected 1"
            );
        }
    }

    #[test]
    fn erf_metamorphic_odd_symmetry() {
        // erf(-x) = -erf(x). Closed-form identity from the integral.
        for &x in &[0.1_f64, 0.5, 1.0, 2.0, 3.5] {
            let lhs = super::erf_scalar(-x);
            let rhs = -super::erf_scalar(x);
            assert!(
                (lhs - rhs).abs() < 1e-15,
                "erf(-{x}) = {lhs}, expected -erf({x}) = {rhs}"
            );
        }
    }

    #[test]
    fn erf_matches_scipy_reference_values() {
        // scipy.special.erf([0.5, 1.0, 2.0])
        let cases = [
            (0.5, 0.5204998778130465),
            (1.0, 0.8427007929497149),
            (2.0, 0.9953222650189527),
        ];
        for (x, expected) in cases {
            let result = super::erf_scalar(x);
            assert!(
                (result - expected).abs() < 1e-10,
                "erf({x}) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn erfc_matches_scipy_reference_values() {
        // scipy.special.erfc([0.5, 1.0, 2.0])
        let cases = [
            (0.5, 0.4795001221869535),
            (1.0, 0.1572992070502851),
            (2.0, 0.004677734981047266),
        ];
        for (x, expected) in cases {
            let result = super::erfc_scalar(x);
            assert!(
                (result - expected).abs() < 1e-10,
                "erfc({x}) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn erfinv_matches_scipy_reference_values() {
        // scipy.special.erfinv([0.5, 0.8, 0.95])
        let cases = [
            (0.5, 0.4769362762044699),
            (0.8, 0.9061938024368232),
            (0.95, 1.3859038243496775),
        ];
        for (y, expected) in cases {
            let result = super::erfinv_scalar(y, RuntimeMode::Strict).unwrap();
            assert!(
                (result - expected).abs() < 1e-9,
                "erfinv({y}) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn erfinv_next_to_endpoint_stays_finite() {
        let y = f64::from_bits(1.0_f64.to_bits() - 1);
        let expected = 5.863_584_748_755_168;
        for (input, want) in [(y, expected), (-y, -expected)] {
            let result = super::erfinv_scalar(input, RuntimeMode::Strict).unwrap();
            assert!(
                (result - want).abs() < 5e-14,
                "erfinv({input}) = {result}, expected {want}"
            );
        }
    }

    #[test]
    fn erfcinv_matches_scipy_reference_values() {
        // scipy.special.erfcinv([0.5, 1.0, 1.5])
        let cases = [
            (0.5, 0.4769362762044699),
            (1.0, 0.0),
            (1.5, -0.4769362762044699),
        ];
        for (y, expected) in cases {
            let result =
                real_value(tensor_result(super::erfcinv(&scalar(y), RuntimeMode::Strict)).unwrap())
                    .unwrap();
            assert!(
                (result - expected).abs() < 1e-9,
                "erfcinv({y}) = {result}, expected {expected}"
            );
        }
    }
}
