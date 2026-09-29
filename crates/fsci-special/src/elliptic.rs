#![forbid(unsafe_code)]

//! Elliptic integrals and Lambert W function.
//!
//! Provides:
//! - `ellipk` — Complete elliptic integral of the first kind K(m)
//! - `ellipe` — Complete elliptic integral of the second kind E(m)
//! - `ellipkinc` — Incomplete elliptic integral of the first kind F(φ, m)
//! - `ellipeinc` — Incomplete elliptic integral of the second kind E(φ, m)
//! - `lambertw` — Lambert W function (principal branch W₀)
//! - `exp1` — Exponential integral E₁(z)
//! - `expi` — Exponential integral Ei(x)

use std::f64::consts::PI;

use fsci_runtime::RuntimeMode;

use crate::types::{
    Complex64, DispatchPlan, DispatchStep, KernelRegime, SpecialError, SpecialErrorKind,
    SpecialResult, SpecialTensor, record_special_trace,
};
use crate::{par_map_indices, par_map_indices_gated};

pub const ELLIPTIC_DISPATCH_PLAN: &[DispatchPlan] = &[
    DispatchPlan {
        function: "ellipk",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "m near 0: polynomial approximation",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "m near 1: logarithmic singularity handling",
            },
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "general m: arithmetic-geometric mean iteration",
            },
        ],
        notes: "Domain: m in [0, 1). K(m) -> inf as m -> 1.",
    },
    DispatchPlan {
        function: "ellipe",
        steps: &[DispatchStep {
            regime: KernelRegime::Recurrence,
            when: "arithmetic-geometric mean with E accumulator",
        }],
        notes: "Domain: m in [0, 1]. E(0) = π/2, E(1) = 1.",
    },
    DispatchPlan {
        function: "lambertw",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "x near 0: series expansion",
            },
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "general x: Halley iteration from initial guess",
            },
        ],
        notes: "Principal branch W₀ for x >= -1/e. W₀(0) = 0, W₀(e) = 1.",
    },
];

// ══════════════════════════════════════════════════════════════════════
// Complete Elliptic Integrals (AGM method)
// ══════════════════════════════════════════════════════════════════════

/// Complete elliptic integral of the first kind K(m).
///
/// K(m) = ∫₀^{π/2} dθ / sqrt(1 - m sin²θ)
///
/// Uses the arithmetic-geometric mean (AGM) iteration.
/// Domain: m in [0, 1). Under `errstate`, `m = 1` is SciPy's "singularity" and `m > 1` its
/// "domain error" (`ellpk`, which SciPy evaluates at `1 - m`).
pub fn ellipk(m_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    let value = map_real_or_complex_rp(
        "ellipk",
        m_tensor,
        mode,
        |m| ellipk_scalar(m, mode),
        ellipk_complex_scalar,
        1 << 20, // work-gated: serial small, par_map_indices for huge arrays (>=1M)
    )?;
    crate::sf_error_unary("ellpk", m_tensor, mode, |m| sf_ellpk(1.0 - m))?;
    Ok(value)
}

/// SciPy's `ellpk(p)` test on the complementary parameter: `p < 0` is a domain error and
/// `p = 0` a singularity.
fn sf_ellpk(p: f64) -> Option<crate::SpecialErrorCode> {
    if p < 0.0 {
        Some(crate::SpecialErrorCode::Domain)
    } else if p == 0.0 {
        Some(crate::SpecialErrorCode::Singular)
    } else {
        None
    }
}

/// Complete elliptic integral of the first kind with complementary argument.
///
/// K(1-p) where p = 1 - m is the complementary parameter.
///
/// This is numerically stable when p is small (m close to 1).
/// Matches `scipy.special.ellipkm1(p)`. Under `errstate`, `p = 0` is SciPy's "singularity"
/// and `p < 0` its "domain error" (`ellpk`).
pub fn ellipkm1(p_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    let value = map_real_or_complex_rp(
        "ellipkm1",
        p_tensor,
        mode,
        |p| ellipkm1_scalar(p, mode),
        |p| ellipkm1_complex_scalar(p, mode),
        1 << 20, // cheap Cephes ~20ns: serial until ~1M (matches ellipk); n/32 over-subscribes up to 21x
    )?;
    crate::sf_error_unary("ellpk", p_tensor, mode, sf_ellpk)?;
    Ok(value)
}

/// Complete elliptic integral of the second kind E(m).
///
/// E(m) = ∫₀^{π/2} sqrt(1 - m sin²θ) dθ
///
/// Uses the AGM method with E accumulator.
/// Domain: m in [0, 1].
pub fn ellipe(m_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_or_complex_rp(
        "ellipe",
        m_tensor,
        mode,
        |m| ellipe_scalar(m, mode),
        ellipe_complex_scalar,
        1 << 20, // work-gated: serial small, par_map_indices for huge arrays (>=1M)
    )
}

/// Incomplete elliptic integral of the first kind F(φ, m).
///
/// F(φ, m) = ∫₀^φ dθ / sqrt(1 - m sin²θ)
///
/// Uses Carlson's RF form via Gauss transformation.
pub fn ellipkinc(
    phi_tensor: &SpecialTensor,
    m_tensor: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    if let (SpecialTensor::RealScalar(phi), SpecialTensor::RealVec(m_values)) =
        (phi_tensor, m_tensor)
    {
        return ellipkinc_scalar_phi_over_m_vec(*phi, m_values, mode);
    }

    map_real_or_complex_binary(
        "ellipkinc",
        phi_tensor,
        m_tensor,
        mode,
        |phi, m| ellipkinc_scalar(phi, m, mode),
        ellipkinc_complex_scalar,
        1 << 16, // ellipkinc break-even ~45k (BlackThrush A/B: 32768 loses 1.23x, 65536 wins 0.56x)
    )
}

/// Incomplete elliptic integral of the second kind E(φ, m).
///
/// E(φ, m) = ∫₀^φ sqrt(1 - m sin²θ) dθ
pub fn ellipeinc(
    phi_tensor: &SpecialTensor,
    m_tensor: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    map_real_or_complex_binary(
        "ellipeinc",
        phi_tensor,
        m_tensor,
        mode,
        |phi, m| ellipeinc_scalar(phi, m, mode),
        ellipeinc_complex_scalar,
        1 << 15, // ellipeinc break-even ~28k (BlackThrush A/B: 32768 wins 0.76x)
    )
}

// ══════════════════════════════════════════════════════════════════════
// Lambert W Function
// ══════════════════════════════════════════════════════════════════════

/// Lambert W function, principal branch W₀(x).
///
/// Solves w * exp(w) = x for w.
/// Domain: x >= -1/e.
pub fn lambertw(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_or_complex_rp(
        "lambertw",
        x_tensor,
        mode,
        |x| lambertw_scalar(x, mode),
        |z| lambertw_complex_scalar(z, mode),
        1 << 16, // lambertw break-even ~58k (BlackThrush A/B: 32768 loses 1.70x, 65536 wins 0.87x)
    )
}

// ══════════════════════════════════════════════════════════════════════
// Exponential Integrals
// ══════════════════════════════════════════════════════════════════════

/// Exponential integral E₁(z) = ∫₁^∞ exp(-zt)/t dt for z > 0.
pub fn exp1(z_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_or_complex_rp(
        "exp1",
        z_tensor,
        mode,
        |z| exp1_scalar(z, mode),
        |z| exp1_complex_scalar(z, mode),
        1 << 15, // exp1 break-even ~18k (BlackThrush A/B: 16384 loses 1.07x, 32768 wins 0.54x)
    )
}

/// Exponential integral Ei(x) = -PV∫_{-x}^∞ exp(-t)/t dt for x > 0.
pub fn expi(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_or_complex_rp(
        "expi",
        x_tensor,
        mode,
        |x| expi_scalar(x, mode),
        |z| expi_complex_scalar(z, mode),
        1 << 16, // expi break-even ~58k (BlackThrush A/B: 32768 loses 1.61x, 65536 wins 0.84x)
    )
}

/// Generalized exponential integral E_n(x) = ∫₁^∞ exp(-xt)/t^n dt.
///
/// Matches `scipy.special.expn(n, x)`.
///
/// # Arguments
/// * `n` - Order (non-negative integer)
/// * `x` - Argument (must be > 0 for n=0,1; can be 0 for n >= 2)
pub fn expn_scalar(n: u32, x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return f64::NAN;
    }

    // Special cases
    if n == 0 {
        if x == 0.0 {
            return f64::INFINITY;
        }
        return (-x).exp() / x;
    }

    if x == 0.0 {
        if n == 1 {
            return f64::INFINITY;
        }
        // E_n(0) = 1/(n-1) for n >= 2
        return 1.0 / (n - 1) as f64;
    }

    if x == f64::INFINITY {
        return 0.0;
    }

    // Use E_1 and recurrence relation: n * E_{n+1}(x) = e^{-x} - x * E_n(x)
    // Rearranged: E_n(x) computed from E_1(x) using upward recurrence
    let e1 = exp1_scalar(x, fsci_runtime::RuntimeMode::Strict).unwrap_or(f64::NAN);
    if n == 1 {
        return e1;
    }

    // Upward recurrence from E_1 to E_n
    let exp_neg_x = (-x).exp();
    let mut e_prev = e1;
    for k in 1..n {
        // E_{k+1}(x) = (e^{-x} - x * E_k(x)) / k
        let e_next = (exp_neg_x - x * e_prev) / k as f64;
        e_prev = e_next;
    }
    e_prev
}

// ══════════════════════════════════════════════════════════════════════
// Scalar Kernels
// ══════════════════════════════════════════════════════════════════════

// Cephes complete elliptic K/E via the EXACT scipy/xsf polynomial coefficients (ellpk/ellpe) —
// O(1) rational+log instead of the ~5-6-iteration AGM, byte-matching scipy.special.ellipk/ellipe.
// frankenscipy-9l5oo. ellpk(x)=K(1−x), ellpe(x)=E(1−x); polevl = Horner from coef[0].
fn cephes_polevl(x: f64, coef: &[f64]) -> f64 {
    coef.iter().fold(0.0, |acc, &c| acc * x + c)
}
const CEPHES_MACHEP: f64 = 1.11022302462515654042E-16;
const ELLPK_C1: f64 = 1.3862943611198906188E0; // log(4)
const ELLPK_P: [f64; 11] = [
    1.37982864606273237150E-4,
    2.28025724005875567385E-3,
    7.97404013220415179367E-3,
    9.85821379021226008714E-3,
    6.87489687449949877925E-3,
    6.18901033637687613229E-3,
    8.79078273952743772254E-3,
    1.49380448916805252718E-2,
    3.08851465246711995998E-2,
    9.65735902811690126535E-2,
    1.38629436111989062502E0,
];
const ELLPK_Q: [f64; 11] = [
    2.94078955048598507511E-5,
    9.14184723865917226571E-4,
    5.94058303753167793257E-3,
    1.54850516649762399335E-2,
    2.39089602715924892727E-2,
    3.01204715227604046988E-2,
    3.73774314173823228969E-2,
    4.88280347570998239232E-2,
    7.03124996963957469739E-2,
    1.24999999999870820058E-1,
    4.99999999999999999821E-1,
];
const ELLPE_P: [f64; 11] = [
    1.53552577301013293365E-4,
    2.50888492163602060990E-3,
    8.68786816565889628429E-3,
    1.07350949056076193403E-2,
    7.77395492516787092951E-3,
    7.58395289413514708519E-3,
    1.15688436810574127319E-2,
    2.18317996015557253103E-2,
    5.68051945617860553470E-2,
    4.43147180560990850618E-1,
    1.00000000000000000299E0,
];
const ELLPE_Q: [f64; 10] = [
    3.27954898576485872656E-5,
    1.00962792679356715133E-3,
    6.50609489976927491433E-3,
    1.68862163993311317300E-2,
    2.61769742454493659583E-2,
    3.34833904888224918614E-2,
    4.27180926518931511717E-2,
    5.85936634471101055642E-2,
    9.37499997197644278445E-2,
    2.49999999999888314361E-1,
];

/// Cephes `ellpk(x) = K(1−x)`: complete elliptic K via the exact xsf polynomial.
pub(crate) fn cephes_ellpk_x(x: f64) -> f64 {
    if x < 0.0 {
        return f64::NAN;
    }
    if x > 1.0 {
        if x.is_infinite() {
            return 0.0;
        }
        return cephes_ellpk_x(1.0 / x) / x.sqrt();
    }
    if x > CEPHES_MACHEP {
        cephes_polevl(x, &ELLPK_P) - x.ln() * cephes_polevl(x, &ELLPK_Q)
    } else if x == 0.0 {
        f64::INFINITY
    } else {
        ELLPK_C1 - 0.5 * x.ln()
    }
}

/// Cephes `ellpe(x) = E(1−x)`: complete elliptic E via the exact xsf polynomial.
fn cephes_ellpe_x(x: f64) -> f64 {
    if x <= 0.0 {
        return if x == 0.0 { 1.0 } else { f64::NAN };
    }
    if x > 1.0 {
        if x.is_infinite() {
            return f64::INFINITY;
        }
        return cephes_ellpe_x(1.0 - 1.0 / x) * x.sqrt();
    }
    cephes_polevl(x, &ELLPE_P) - x.ln() * (x * cephes_polevl(x, &ELLPE_Q))
}

fn ellipk_scalar(m: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if m.is_nan() {
        return Ok(f64::NAN);
    }
    if m < 0.0 {
        if mode == RuntimeMode::Hardened {
            return domain_error("ellipk", mode, "m must be in [0, 1)");
        }
        // Reciprocal-modulus transformation K(m) = K(m/(m-1)) / sqrt(1-m), valid
        // for m < 0 where m/(m-1) lands in [0, 1). Matches scipy.special.ellipk,
        // which accepts negative parameters.
        let mt = m / (m - 1.0);
        let k = ellipk_scalar(mt, RuntimeMode::Strict)?;
        return Ok(k / (1.0 - m).sqrt());
    }
    if !(0.0..=1.0).contains(&m) {
        return domain_error("ellipk", mode, "m must be in [0, 1)");
    }
    if m >= 1.0 {
        if mode == RuntimeMode::Hardened {
            return domain_error("ellipk", mode, "m must be in [0, 1)");
        }
        return Ok(f64::INFINITY);
    }
    if m == 0.0 {
        return Ok(PI / 2.0);
    }

    // Cephes ellpk(1−m) — exact xsf polynomial (byte-matches scipy.special.ellipk), no AGM loop.
    Ok(cephes_ellpk_x(1.0 - m))
}

fn ellipkm1_scalar(p: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if p.is_nan() {
        return Ok(f64::NAN);
    }
    // scipy.special.ellipkm1(p) = K(1 - p) is defined for every p >= 0 (m = 1-p
    // <= 1): the AGM below converges for any sqrt(p), so p > 1 (negative m) is a
    // finite value (ellipkm1(2) = K(-1) = 1.3110), NOT a domain error. Only p < 0
    // (m > 1, outside the real domain) is NaN, matching scipy. The previous
    // [0, 1] guard wrongly rejected p > 1. frankenscipy-35272
    if p < 0.0 {
        return domain_error("ellipkm1", mode, "p must be >= 0 (m = 1-p must be <= 1)");
    }
    // scipy.special.ellipkm1 IS Cephes `ellpk(p)`, the kernel `ellipk` already calls. The
    // AGM this replaced kept only the leading term ln(4/√p) below p = 1e-10, dropping
    // (ln(4/√p) − 1)·p/4: 1.0e-11 relative off at p = 4.5e-11 (frankenscipy-36gsc).
    Ok(cephes_ellpk_x(p))
}

fn ellipe_scalar(m: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if m.is_nan() {
        return Ok(f64::NAN);
    }
    if m < 0.0 {
        if mode == RuntimeMode::Hardened {
            return domain_error("ellipe", mode, "m must be in [0, 1]");
        }
        // Reciprocal-modulus transformation E(m) = sqrt(1-m) * E(m/(m-1)), valid
        // for m < 0. Matches scipy.special.ellipe, which accepts negative m.
        let mt = m / (m - 1.0);
        let e = ellipe_scalar(mt, RuntimeMode::Strict)?;
        return Ok((1.0 - m).sqrt() * e);
    }
    if !(0.0..=1.0).contains(&m) {
        return domain_error("ellipe", mode, "m must be in [0, 1]");
    }
    if m == 0.0 {
        return Ok(PI / 2.0);
    }
    if m == 1.0 {
        return Ok(1.0);
    }

    // Cephes ellpe(1−m) — exact xsf polynomial (byte-matches scipy.special.ellipe), no AGM loop.
    Ok(cephes_ellpe_x(1.0 - m))
}

fn ellipkinc_scalar(phi: f64, m: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if phi.is_nan() || m.is_nan() {
        return Ok(f64::NAN);
    }
    if m > 1.0 {
        return domain_error("ellipkinc", mode, "m must be in [0, 1]");
    }
    // m < 0 is valid (scipy.special.ellipkinc accepts it): the Carlson R_F form
    // F = sinφ·R_F(cos²φ, 1-m·sin²φ, 1) stays well-defined (1-m·sin²φ > 0) and the
    // periodicity term uses K(m<0), now supported. Hardened stays conservative.
    if m < 0.0 && mode == RuntimeMode::Hardened {
        return domain_error("ellipkinc", mode, "m must be in [0, 1]");
    }
    if m >= 1.0 && mode == RuntimeMode::Hardened {
        return domain_error("ellipkinc", mode, "m must be in [0, 1)");
    }
    Ok(cephes_ellik(phi, m))
}

/// `F(φ | m)` by SciPy's own method: xsf's Cephes `ellik` (the revision SciPy 1.17.1 pins,
/// 0d0a593f), SciPy's bits.
/// - Reduce φ by multiples of π/2, carrying the K(m) terms.
/// - Past tan φ = 10, transform the amplitude.
/// - Otherwise run the descending Landen / AGM transformation to MACHEP.
/// - m < 0 goes through `cephes_ellik_neg_m`: series, asymptotic, or Carlson's R_F with its own
///   stopping rule.
/// - m = 1 is `asinh(tan φ)`.
///
/// `mod` is a C `int` in the source, so `(φ + π/2)/π` truncates toward zero.
///
/// This replaced `sinφ·R_F(cos²φ, 1 − m·sin²φ, 1)`, whose second argument cancels as m → 1.
/// It was 1.0e-10 relative off at (1.5709, 1 − 2.9e-8), where SciPy is 6e-14 (frankenscipy-zw56i).
/// A Python emulation of this code matched scipy.special.ellipkinc on 80,008 of 80,008 points:
/// m within 1e-16 of 1, negative m to −1e12, |φ| to 1e6, tiny φ, and the edges.
fn cephes_ellik(phi: f64, m: f64) -> f64 {
    const MACHEP: f64 = 1.110_223_024_625_156_540_42e-16;
    if phi.is_nan() || m.is_nan() || m > 1.0 {
        return f64::NAN;
    }
    if phi.is_infinite() || m.is_infinite() {
        return if m.is_infinite() && phi.is_finite() {
            0.0
        } else if phi.is_infinite() && m.is_finite() {
            phi
        } else {
            f64::NAN
        };
    }
    if m == 0.0 {
        return phi;
    }
    let a = 1.0 - m;
    if a == 0.0 {
        // DLMF 19.6.8 and 4.23.42.
        return if phi.abs() >= PI / 2.0 {
            f64::INFINITY
        } else {
            phi.tan().asinh()
        };
    }
    let mut npio2 = (phi / (PI / 2.0)).floor();
    if npio2.abs() % 2.0 == 1.0 {
        npio2 += 1.0;
    }
    let mut big_k = 0.0;
    let mut phi = phi;
    if npio2 != 0.0 {
        big_k = cephes_ellpk_x(a);
        phi -= npio2 * (PI / 2.0);
    }
    let negative = phi < 0.0;
    if negative {
        phi = -phi;
    }
    let mut temp = if a > 1.0 {
        cephes_ellik_neg_m(phi, m)
    } else {
        cephes_ellik_landen(phi, m, a, npio2, &mut big_k, MACHEP)
    };
    if negative {
        temp = -temp;
    }
    temp + npio2 * big_k
}

/// The 0 < m < 1 core of [`cephes_ellik`] after the π/2 reduction: the amplitude
/// transformation past tan φ = 10 (recursing once), else the descending Landen/AGM iteration.
fn cephes_ellik_landen(phi: f64, m: f64, a: f64, npio2: f64, big_k: &mut f64, machep: f64) -> f64 {
    let b = a.sqrt();
    let mut t = phi.tan();
    if t.abs() > 10.0 {
        let e = 1.0 / (b * t);
        if e.abs() < 10.0 {
            if npio2 == 0.0 {
                *big_k = cephes_ellpk_x(a);
            }
            return *big_k - cephes_ellik(e.atan(), m);
        }
    }
    let (mut aa, mut bb) = (1.0, b);
    let mut c = m.sqrt();
    // Cephes keeps `mod` and `d` as C ints. Held here as integer-valued doubles: `trunc`,
    // `floor` and doubling give the same values while |mod| < 2^31, and phi, reduced to
    // |phi| <= pi/2 on entry, only doubles per step. It keeps a saturating `as i64` and two
    // int <-> double conversions off the loop-carried chain phi -> mod -> phi.
    let mut d = 1.0_f64;
    let mut modulus = 0.0_f64;
    let mut phi = phi;
    while (c / aa).abs() > machep {
        let ratio = bb / aa;
        phi = phi + (t * ratio).atan() + modulus * PI;
        let denom = 1.0 - ratio * t * t;
        if denom.abs() > 10.0 * machep {
            t = t * (1.0 + ratio) / denom;
            modulus = ellik_quadrant(phi);
        } else {
            t = phi.tan();
            modulus = ((phi - t.atan()) / PI).floor();
        }
        c = (aa - bb) / 2.0;
        let g = (aa * bb).sqrt();
        aa = (aa + bb) / 2.0;
        bb = g;
        d += d;
    }
    (t.atan() + modulus * PI) / (d * aa)
}

/// Cephes' `mod = (phi + M_PI_2) / M_PI` for [`cephes_ellik_landen`], truncated as the C int is.
///
/// Out of line on purpose. Inline, LLVM packs this division with the loop's
/// `t * (1 + ratio) / denom` into one `vdivpd`. This one needs the iteration's `atan` result
/// and t's does not, so packing made the next `atan`'s argument wait for the current `atan`, and
/// the calls could no longer overlap.
#[inline(never)]
fn ellik_quadrant(phi: f64) -> f64 {
    ((phi + PI / 2.0) / PI).trunc()
}

/// Cephes `ellik_neg_m`: `F(φ | m)` for m < 0 and 0 < φ < π/2. A power series for small m·φ²,
/// an asymptotic form for large, and otherwise Carlson's `R_F(c − 1, c − m, c)` with
/// c = csc²φ (or the small-φ scaled form) and Cephes' own stopping rule.
fn cephes_ellik_neg_m(phi: f64, m: f64) -> f64 {
    let mpp = (m * phi) * phi;
    if -mpp < 1e-6 && phi < -m {
        return phi + (-mpp * phi * phi / 30.0 + 3.0 * mpp * mpp / 40.0 + mpp / 6.0) * phi;
    }
    if -mpp > 4e7 {
        let sm = (-m).sqrt();
        let sp = phi.sin();
        let cp = phi.cos();
        let a = (4.0 * sp * sm / (1.0 + cp)).ln();
        let b = -(1.0 + cp / sp / sp - a) / 4.0 / m;
        return (a + b) / sm;
    }
    let (scale, x, y, z) = if phi > 1e-153 && m > -1e305 {
        let s = phi.sin();
        let csc2 = 1.0 / (s * s);
        (1.0, 1.0 / (phi.tan() * phi.tan()), csc2 - m, csc2)
    } else {
        (phi, 1.0, 1.0 - m * phi * phi, 1.0)
    };
    if x == y && x == z {
        return scale / x.sqrt();
    }
    let a0 = (x + y + z) / 3.0;
    let mut a = a0;
    let (mut x1, mut y1, mut z1) = (x, y, z);
    // Carlson gives 1/pow(3*r, 1/6) for this constant; ~338.38 at r = eps.
    let mut q = 400.0 * (a0 - x).abs().max((a0 - y).abs().max((a0 - z).abs()));
    let mut n: u32 = 0;
    while q > a.abs() && n <= 100 {
        let (sx, sy, sz) = (x1.sqrt(), y1.sqrt(), z1.sqrt());
        let lam = sx * sy + sx * sz + sy * sz;
        x1 = (x1 + lam) / 4.0;
        y1 = (y1 + lam) / 4.0;
        z1 = (z1 + lam) / 4.0;
        a = (x1 + y1 + z1) / 3.0;
        n += 1;
        q /= 4.0;
    }
    // Cephes writes `(1 << 2 * n)`, an int shift that is undefined past n = 15; 4ⁿ as a double
    // is the same value wherever the C is defined.
    let pow4 = 4.0_f64.powi(n as i32);
    let xx = (a0 - x) / a / pow4;
    let yy = (a0 - y) / a / pow4;
    let zz = -(xx + yy);
    let e2 = xx * yy - zz * zz;
    let e3 = xx * yy * zz;
    scale * (1.0 - e2 / 10.0 + e3 / 14.0 + e2 * e2 / 24.0 - 3.0 * e2 * e3 / 44.0) / a.sqrt()
}

/// Carlson symmetric elliptic integral R_F(x,y,z) (Numerical Recipes §6.11):
/// R_F = ½∫₀^∞ dt/√((t+x)(t+y)(t+z)).
// No longer called: ellipkinc is Cephes `ellik` (frankenscipy-zw56i) and the public `elliprf`
// is SciPy's rf. RETAINED, as `carlson_rd` is, because `carlson_rf_rd` below claims to be
// byte-identical to this pair and cites both by name.
#[allow(dead_code)]
fn carlson_rf(mut x: f64, mut y: f64, mut z: f64) -> f64 {
    const ERRTOL: f64 = 1.3e-3; // error ~ERRTOL^6 ≈ 5e-18 ≪ machine eps; was 1e-5 (~9 iters → ~5)
    for _ in 0..1000 {
        let (sx, sy, sz) = (x.sqrt(), y.sqrt(), z.sqrt());
        let lam = sx * sy + sy * sz + sz * sx;
        x = 0.25 * (x + lam);
        y = 0.25 * (y + lam);
        z = 0.25 * (z + lam);
        let ave = (x + y + z) / 3.0;
        let dx = (ave - x) / ave;
        let dy = (ave - y) / ave;
        let dz = (ave - z) / ave;
        if dx.abs().max(dy.abs()).max(dz.abs()) < ERRTOL {
            let e2 = dx * dy - dz * dz;
            let e3 = dx * dy * dz;
            return (1.0 + (e2 / 24.0 - 0.1 - 3.0 * e3 / 44.0) * e2 + e3 / 14.0) / ave.sqrt();
        }
    }
    f64::NAN
}

/// Carlson symmetric elliptic integral R_D(x,y,z) (Numerical Recipes §6.11):
/// R_D = (3/2)∫₀^∞ dt/((t+z)√((t+x)(t+y)(t+z))).
// Superseded by the fused RF/RD duplication loop below, which computes both
// symmetric integrals in one pass at ~half the work (see the note at the
// `carlson_rf_rd` definition, frankenscipy-9l5oo). RETAINED rather than deleted
// (frankenscipy-e2ve2): it is the standalone reference form of the algorithm the
// fused loop is claimed to be equivalent to, and the doc comment below still
// cites it by name. Deleting it would remove the thing that claim refers to.
#[allow(dead_code)]
fn carlson_rd(mut x: f64, mut y: f64, mut z: f64) -> f64 {
    const ERRTOL: f64 = 1.3e-3; // error ~ERRTOL^6 ≈ 5e-18 ≪ machine eps; was 1e-5 (~9 iters → ~5)
    const C1: f64 = 3.0 / 14.0;
    const C2: f64 = 1.0 / 6.0;
    const C3: f64 = 9.0 / 22.0;
    const C4: f64 = 3.0 / 26.0;
    const C5: f64 = 0.25 * C3;
    const C6: f64 = 1.5 * C4;
    let mut s = 0.0;
    let mut fac = 1.0;
    for _ in 0..1000 {
        let (sx, sy, sz) = (x.sqrt(), y.sqrt(), z.sqrt());
        let lam = sx * sy + sy * sz + sz * sx;
        s += fac / (sz * (z + lam));
        fac *= 0.25;
        x = 0.25 * (x + lam);
        y = 0.25 * (y + lam);
        z = 0.25 * (z + lam);
        let ave = (x + y + 3.0 * z) / 5.0;
        let dx = (ave - x) / ave;
        let dy = (ave - y) / ave;
        let dz = (ave - z) / ave;
        if dx.abs().max(dy.abs()).max(dz.abs()) < ERRTOL {
            let ea = dx * dy;
            let eb = dz * dz;
            let ec = ea - eb;
            let ed = ea - 6.0 * eb;
            let ee = ed + ec + ec;
            return 3.0 * s
                + fac
                    * (1.0
                        + ed * (-C1 + C5 * ed - C6 * dz * ee)
                        + dz * (C2 * ee + dz * (-C3 * ec + dz * C4 * ea)))
                    / (ave * ave.sqrt());
        }
    }
    f64::NAN
}

/// E(φ, m) for any φ via Carlson R_F/R_D, with E(φ + nπ, m) = E(φ, m) + 2n·E(m).
/// Combined Carlson R_F(x,y,z) and R_D(x,y,z): both share the IDENTICAL duplication sequence
/// (sqrt/lam/0.25-update), differing only in their convergence `ave` and R_D's `s` accumulation.
/// Computing the sqrt-heavy sequence ONCE yields both, byte-identical to `carlson_rf(x,y,z)` +
/// `carlson_rd(x,y,z)` at ~half the work — the dominant cost of E(φ,m). frankenscipy-9l5oo.
fn carlson_rf_rd(mut x: f64, mut y: f64, mut z: f64) -> (f64, f64) {
    const ERRTOL: f64 = 1.3e-3; // error ~ERRTOL^6 ≈ 5e-18 ≪ machine eps; was 1e-5 (~9 iters → ~5)
    const C1: f64 = 3.0 / 14.0;
    const C2: f64 = 1.0 / 6.0;
    const C3: f64 = 9.0 / 22.0;
    const C4: f64 = 3.0 / 26.0;
    const C5: f64 = 0.25 * C3;
    const C6: f64 = 1.5 * C4;
    let mut s = 0.0;
    let mut fac = 1.0;
    let mut rf: Option<f64> = None;
    let mut rd: Option<f64> = None;
    for _ in 0..1000 {
        let (sx, sy, sz) = (x.sqrt(), y.sqrt(), z.sqrt());
        let lam = sx * sy + sy * sz + sz * sx;
        if rd.is_none() {
            s += fac / (sz * (z + lam));
            fac *= 0.25;
        }
        x = 0.25 * (x + lam);
        y = 0.25 * (y + lam);
        z = 0.25 * (z + lam);
        if rf.is_none() {
            let ave = (x + y + z) / 3.0;
            let dx = (ave - x) / ave;
            let dy = (ave - y) / ave;
            let dz = (ave - z) / ave;
            if dx.abs().max(dy.abs()).max(dz.abs()) < ERRTOL {
                let e2 = dx * dy - dz * dz;
                let e3 = dx * dy * dz;
                rf =
                    Some((1.0 + (e2 / 24.0 - 0.1 - 3.0 * e3 / 44.0) * e2 + e3 / 14.0) / ave.sqrt());
            }
        }
        if rd.is_none() {
            let ave = (x + y + 3.0 * z) / 5.0;
            let dx = (ave - x) / ave;
            let dy = (ave - y) / ave;
            let dz = (ave - z) / ave;
            if dx.abs().max(dy.abs()).max(dz.abs()) < ERRTOL {
                let ea = dx * dy;
                let eb = dz * dz;
                let ec = ea - eb;
                let ed = ea - 6.0 * eb;
                let ee = ed + ec + ec;
                rd = Some(
                    3.0 * s
                        + fac
                            * (1.0
                                + ed * (-C1 + C5 * ed - C6 * dz * ee)
                                + dz * (C2 * ee + dz * (-C3 * ec + dz * C4 * ea)))
                            / (ave * ave.sqrt()),
                );
            }
        }
        if let (Some(rf_value), Some(rd_value)) = (rf, rd) {
            return (rf_value, rd_value);
        }
    }
    (rf.unwrap_or(f64::NAN), rd.unwrap_or(f64::NAN))
}

fn ellipeinc_carlson(phi: f64, m: f64, mode: RuntimeMode) -> f64 {
    let n = (phi / PI).round();
    let phi_r = phi - n * PI;
    let s = phi_r.sin();
    let c = phi_r.cos();
    let cc = c * c;
    let d = 1.0 - m * s * s;
    let (rf, rd) = carlson_rf_rd(cc, d, 1.0); // one shared sqrt-sequence instead of two
    let e_r = s * rf - (m / 3.0) * s * s * s * rd;
    if n == 0.0 {
        e_r
    } else {
        2.0 * n * ellipe_scalar(m, mode).unwrap_or(f64::NAN) + e_r
    }
}

fn ellipkinc_scalar_phi_over_m_vec(phi: f64, m_values: &[f64], mode: RuntimeMode) -> SpecialResult {
    if phi.is_nan()
        || m_values.iter().any(|m| {
            m.is_nan() || !(0.0..=1.0).contains(m) || (*m >= 1.0 && mode == RuntimeMode::Hardened)
        })
    {
        return m_values
            .iter()
            .copied()
            .map(|m| ellipkinc_scalar(phi, m, mode))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec);
    }

    // Cephes `ellik` per m, each written to its own slot: chunking across cores and
    // concatenating in index order is bit-identical to the serial map (par_map_indices gates
    // n < 256 to the sequential path).
    let values = par_map_indices(m_values.len(), |i| Ok(cephes_ellik(phi, m_values[i])))?;
    Ok(SpecialTensor::RealVec(values))
}

fn ellipeinc_scalar(phi: f64, m: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if phi.is_nan() || m.is_nan() {
        return Ok(f64::NAN);
    }
    if m > 1.0 {
        return domain_error("ellipeinc", mode, "m must be in [0, 1]");
    }
    // m < 0 is valid (scipy.special.ellipeinc accepts it): the Carlson form
    // E = R_F - (m/3)·sin³φ·R_D(cos²φ, 1-m·sin²φ, 1) stays well-defined and the
    // periodicity term uses E(m<0), now supported. Hardened stays conservative.
    if m < 0.0 && mode == RuntimeMode::Hardened {
        return domain_error("ellipeinc", mode, "m must be in [0, 1]");
    }
    if phi == 0.0 {
        return Ok(0.0);
    }
    if (phi - PI / 2.0).abs() < 1e-15 {
        return ellipe_scalar(m, mode);
    }
    if m == 0.0 {
        // E(φ | 0) = φ, which Cephes' ellie returns as is. The Gauss-Legendre weight sum here,
        // 1.9999999999999998, made it φ·(1 − 2⁻⁵³), 1 ulp below SciPy.
        return Ok(phi);
    }

    // Carlson symmetric form (machine-accurate near m→1). frankenscipy-o65r0.
    Ok(ellipeinc_carlson(phi, m, mode))
}

fn ellipk_complex_scalar(m: Complex64) -> Result<Complex64, SpecialError> {
    if !m.is_finite() {
        return Ok(complex_nan());
    }
    if m.im == 0.0
        && let Ok(real_val) = ellipk_scalar(m.re, RuntimeMode::Strict)
    {
        return Ok(Complex64::from_real(real_val));
    }
    if m.re == 0.0 && m.im == 0.0 {
        return Ok(Complex64::from_real(PI / 2.0));
    }
    if m.re == 1.0 && m.im == 0.0 {
        return Ok(Complex64::new(f64::INFINITY, 0.0));
    }
    if m.im < 0.0 {
        return ellipk_complex_scalar(m.conj()).map(|res| res.conj());
    }
    Ok(complex_gauss_legendre_elliptic_f(
        Complex64::from_real(PI / 2.0),
        m,
    ))
}

fn ellipe_complex_scalar(m: Complex64) -> Result<Complex64, SpecialError> {
    if !m.is_finite() {
        return Ok(complex_nan());
    }
    if m.im == 0.0
        && let Ok(real_val) = ellipe_scalar(m.re, RuntimeMode::Strict)
    {
        return Ok(Complex64::from_real(real_val));
    }
    if m.re == 0.0 && m.im == 0.0 {
        return Ok(Complex64::from_real(PI / 2.0));
    }
    if m.re == 1.0 && m.im == 0.0 {
        return Ok(Complex64::from_real(1.0));
    }
    if m.im < 0.0 {
        return ellipe_complex_scalar(m.conj()).map(|res| res.conj());
    }
    Ok(complex_gauss_legendre_elliptic_e(
        Complex64::from_real(PI / 2.0),
        m,
    ))
}

fn ellipkm1_complex_scalar(p: Complex64, mode: RuntimeMode) -> Result<Complex64, SpecialError> {
    if !p.is_finite() {
        return Ok(complex_nan());
    }
    if p.im == 0.0
        && let Ok(real_val) = ellipkm1_scalar(p.re, mode)
    {
        return Ok(Complex64::from_real(real_val));
    }
    ellipk_complex_scalar(Complex64::from_real(1.0) - p)
}

fn ellipkinc_complex_scalar(phi: Complex64, m: Complex64) -> Result<Complex64, SpecialError> {
    if !phi.is_finite() || !m.is_finite() {
        return Ok(complex_nan());
    }
    if phi.im == 0.0
        && m.im == 0.0
        && let Ok(real_val) = ellipkinc_scalar(phi.re, m.re, RuntimeMode::Strict)
    {
        return Ok(Complex64::from_real(real_val));
    }
    if phi.re == 0.0 && phi.im == 0.0 {
        return Ok(Complex64::from_real(0.0));
    }
    if phi.im == 0.0 && (phi.re - PI / 2.0).abs() < 1.0e-15 {
        return ellipk_complex_scalar(m);
    }
    if phi.im < 0.0 || (phi.im == 0.0 && m.im < 0.0) {
        return ellipkinc_complex_scalar(phi.conj(), m.conj()).map(|res| res.conj());
    }
    Ok(complex_gauss_legendre_elliptic_f(phi, m))
}

fn ellipeinc_complex_scalar(phi: Complex64, m: Complex64) -> Result<Complex64, SpecialError> {
    if !phi.is_finite() || !m.is_finite() {
        return Ok(complex_nan());
    }
    if phi.im == 0.0
        && m.im == 0.0
        && let Ok(real_val) = ellipeinc_scalar(phi.re, m.re, RuntimeMode::Strict)
    {
        return Ok(Complex64::from_real(real_val));
    }
    if phi.re == 0.0 && phi.im == 0.0 {
        return Ok(Complex64::from_real(0.0));
    }
    if phi.im == 0.0 && (phi.re - PI / 2.0).abs() < 1.0e-15 {
        return ellipe_complex_scalar(m);
    }
    if phi.im < 0.0 || (phi.im == 0.0 && m.im < 0.0) {
        return ellipeinc_complex_scalar(phi.conj(), m.conj()).map(|res| res.conj());
    }
    Ok(complex_gauss_legendre_elliptic_e(phi, m))
}

fn lambertw_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if x.is_nan() {
        return Ok(f64::NAN);
    }
    if x == f64::INFINITY {
        return Ok(f64::INFINITY);
    }
    let min_x = -1.0 / std::f64::consts::E;
    if x < min_x - 1.0e-12 {
        return domain_error("lambertw", mode, "x must be >= -1/e for principal branch");
    }
    if (x - min_x).abs() < 1.0e-12 {
        return Ok(-1.0); // W₀(-1/e) = -1
    }
    if x == 0.0 {
        return Ok(0.0);
    }
    if (x - std::f64::consts::E).abs() < f64::EPSILON {
        return Ok(1.0);
    }

    // Initial guess
    let mut w = if x < 0.0 {
        // For x in (-1/e, 0): W is in (-1, 0). Use a quadratic approximation near -1/e.
        let p = (2.0 * (std::f64::consts::E * x + 1.0)).sqrt();
        -1.0 + p - p * p / 3.0
    } else if x < 0.5 {
        // Near 0: W(x) ≈ x - x²
        x * (1.0 - x)
    } else if x <= std::f64::consts::E {
        // Moderate x: use log-based estimate
        let lx = (1.0 + x).ln();
        lx * (1.0 - lx / (2.0 + lx))
    } else {
        // Large x: W(x) ≈ ln(x) - ln(ln(x))
        let lx = x.ln();
        lx - lx.ln()
    };

    // Halley's iteration: w_{n+1} = w_n - (w*e^w - x) / (e^w*(w+1) - (w+2)*(w*e^w - x)/(2w+2))
    for _ in 0..50 {
        let ew = w.exp();
        let wew = w * ew;
        let f = wew - x;
        if f.abs() < 1.0e-15 * (1.0 + x.abs()) {
            return Ok(w);
        }
        let denom = ew * (w + 1.0) - (w + 2.0) * f / (2.0 * w + 2.0);
        if denom.abs() < 1.0e-30 {
            break;
        }
        w -= f / denom;
    }
    Ok(w)
}

fn lambertw_complex_scalar(z: Complex64, mode: RuntimeMode) -> Result<Complex64, SpecialError> {
    if z.re.is_nan() || z.im.is_nan() {
        return Ok(complex_nan());
    }
    if !z.is_finite() {
        if z.im == 0.0 && z.re == f64::INFINITY {
            return Ok(Complex64::from_real(f64::INFINITY));
        }
        return Ok(complex_nan());
    }
    if z.re == 0.0 && z.im == 0.0 {
        return Ok(Complex64::from_real(0.0));
    }

    let min_x = -1.0 / std::f64::consts::E;
    if z.im == 0.0 && z.re >= min_x {
        return lambertw_scalar(z.re, mode).map(Complex64::from_real);
    }

    let mut w = lambertw_complex_initial_guess(z);
    let one = Complex64::from_real(1.0);
    let two = Complex64::from_real(2.0);

    for _ in 0..80 {
        let ew = w.exp();
        let wew = w * ew;
        let f = wew - z;
        if f.abs() < 1.0e-14 * (1.0 + z.abs()) {
            return Ok(w);
        }

        let w_plus_one = w + one;
        let denom = ew * w_plus_one - (w + two) * f / (w_plus_one * 2.0);
        if denom.abs() < 1.0e-30 {
            break;
        }

        let step = f / denom;
        w = w - step;
        if step.abs() < 1.0e-14 * (1.0 + w.abs()) {
            return Ok(w);
        }
    }

    Ok(w)
}

fn lambertw_complex_initial_guess(z: Complex64) -> Complex64 {
    // Principal-branch (k = 0) initial guess matching scipy's `_lambertw.pxd`
    // (Veberič 2012 / Corless 1996). The previous guess used the LARGE-|z|
    // asymptotic `log(z) - log(log(z))` for moderate |z|, which lands in the
    // basin of a non-principal branch (W·e^W = z has infinitely many roots) —
    // Halley then converged to the wrong sheet (lambertw(0.5+0.3i) gave
    // -1.97-3.68i instead of scipy's 0.372+0.152i). Three regions:
    let expn1 = Complex64::from_real(1.0 / std::f64::consts::E);
    let branch_delta = z + expn1;
    if branch_delta.abs() < 0.3 {
        // Near the branch point z = -1/e: series in p = √(2(e·z + 1)).
        let ez1 = Complex64::from_real(std::f64::consts::E) * z + Complex64::from_real(1.0);
        let p = complex_sqrt(ez1 * 2.0);
        let p2 = p * p;
        let p3 = p2 * p;
        return Complex64::from_real(-1.0) + p - p2 / 3.0 + p3 * (11.0 / 72.0);
    }

    if -1.0 < z.re && z.re < 1.5 && z.im.abs() < 1.0 && z.re > -2.5 * z.im.abs() - 0.2 {
        // Empirically good region near 0: [2/2] Padé approximant of W₀.
        let z2 = z * z;
        let num = z * (Complex64::from_real(60.0) + z * 114.0 + z2 * 17.0);
        let den = Complex64::from_real(60.0) + z * 174.0 + z2 * 101.0;
        return num / den;
    }

    z.ln()
}

fn exp1_scalar(z: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if z.is_nan() {
        return Ok(f64::NAN);
    }
    if z == 0.0 {
        return Ok(f64::INFINITY);
    }
    if z < 0.0 {
        return domain_error("exp1", mode, "z must be > 0 for real E1");
    }
    if z == f64::INFINITY {
        return Ok(0.0);
    }

    if z <= 2.0 {
        // Series: E1(z) = -γ - ln(z) + Σ_{n=1}^∞ (-1)^{n+1} * z^n / (n * n!)
        let euler_gamma = 0.577_215_664_901_532_9;
        let mut sum = -euler_gamma - z.ln();
        let mut term = 1.0_f64;
        for n in 1..300 {
            term *= z / n as f64;
            let sign = if n % 2 == 1 { 1.0 } else { -1.0 };
            let contribution = sign * term / n as f64;
            sum += contribution;
            if contribution.abs() < 1.0e-15 * sum.abs() {
                break;
            }
        }
        Ok(sum)
    } else {
        // Continued fraction (Lentz's method):
        // E1(z) = exp(-z) * (1/(z + 1/(1 + 1/(z + 2/(1 + 2/(z + ...))))))
        // Using the form: E1(z) = exp(-z) * CF where CF = 1/(z+) 1/(1+) 1/(z+) 2/(1+) 2/(z+) ...
        // Rewritten as standard CF: b_0=0, a_1=1, b_1=z, then alternating
        // a_{2k}=k, b_{2k}=1, a_{2k+1}=k, b_{2k+1}=z
        let mut d = 1.0 / z;
        let mut c = 1.0 / 1.0e-30_f64;
        let mut h = d;

        for n in 1..100 {
            let a_n = ((n + 1) / 2) as f64;
            let b_n = if n % 2 == 1 { 1.0 } else { z };
            d = 1.0 / (b_n + a_n * d);
            c = b_n + a_n / c;
            let delta = c * d;
            h *= delta;
            if (delta - 1.0).abs() < 1.0e-15 {
                break;
            }
        }
        Ok((-z).exp() * h)
    }
}

fn exp1_complex_scalar(z: Complex64, mode: RuntimeMode) -> Result<Complex64, SpecialError> {
    if z.re.is_nan() || z.im.is_nan() {
        return Ok(complex_nan());
    }
    if !z.is_finite() {
        if z.im == 0.0 && z.re == f64::INFINITY {
            return Ok(Complex64::from_real(0.0));
        }
        return Ok(complex_nan());
    }
    if z.re == 0.0 && z.im == 0.0 {
        return Ok(Complex64::new(f64::INFINITY, 0.0));
    }

    if z.im == 0.0 {
        if z.re > 0.0 {
            return exp1_scalar(z.re, mode).map(Complex64::from_real);
        }
        if z.re < 0.0 {
            let sign_pi = if z.im.is_sign_negative() { PI } else { -PI };
            return expi_scalar(-z.re, mode).map(|value| Complex64::new(-value, sign_pi));
        }
    }

    // Method selection (Zhang & Jin, *Computation of Special Functions*, E1Z): the power
    // series is an entire function (after -γ-ln z), so its only error is cancellation, which
    // grows like e^{|z|} relative to the result e^{-Re z}/|z| — negligible when Re(z) is
    // very negative (result is exp-large) but catastrophic for large positive Re(z). The
    // continued fraction is the asymptotic expansion: accurate for large |z| EXCEPT near the
    // negative-real Stokes line. So use the series for small |z| (any direction) and for the
    // deep left half-plane (|z|<20), and the CF for everything else. The old |z|<=2 cutoff
    // wrongly sent the whole |z|>2 left half-plane to the CF, diverging ~5e-3 there
    // (frankenscipy-ey069). Validated vs scipy 1.17.1 to <3.6e-7 over a wide grid.
    let a0 = z.abs();
    if a0 <= 10.0 || (z.re < 0.0 && a0 < 20.0) {
        return exp1_complex_series(z);
    }

    exp1_complex_continued_fraction(z)
}

fn exp1_complex_series(z: Complex64) -> Result<Complex64, SpecialError> {
    let euler_gamma = 0.577_215_664_901_532_9;
    let mut sum = Complex64::from_real(-euler_gamma) - z.ln();
    let mut term = Complex64::from_real(1.0);

    for n in 1..400 {
        term = term * z / n as f64;
        let sign = if n % 2 == 1 { 1.0 } else { -1.0 };
        let contribution = term * sign / n as f64;
        sum = sum + contribution;
        if contribution.abs() < 1.0e-15 * sum.abs().max(1.0) {
            break;
        }
    }

    Ok(sum)
}

fn exp1_complex_continued_fraction(z: Complex64) -> Result<Complex64, SpecialError> {
    let tiny = 1.0e-30;
    let one = Complex64::from_real(1.0);
    let mut d = one / z;
    let mut c = Complex64::from_real(1.0 / tiny);
    let mut h = d;

    for n in 1..200 {
        let a_n = ((n + 1) / 2) as f64;
        let b_n = if n % 2 == 1 { one } else { z };

        let mut d_denom = b_n + d * a_n;
        if d_denom.abs() < tiny {
            d_denom = Complex64::from_real(tiny);
        }
        d = one / d_denom;

        let mut c_term = b_n + Complex64::from_real(a_n) / c;
        if c_term.abs() < tiny {
            c_term = Complex64::from_real(tiny);
        }
        c = c_term;

        let delta = c * d;
        h = h * delta;
        if (delta - one).abs() < 1.0e-15 {
            break;
        }
    }

    Ok((-z).exp() * h)
}

fn expi_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if x.is_nan() {
        return Ok(f64::NAN);
    }
    if x == 0.0 {
        return Ok(f64::NEG_INFINITY);
    }
    if x == f64::INFINITY {
        return Ok(f64::INFINITY);
    }
    if x < 0.0 {
        // Ei(x) = -E1(-x) for x < 0. Kept here rather than delegated because
        // this branch is RuntimeMode-aware via exp1_scalar; the convenience
        // kernel takes no mode.
        return exp1_scalar(-x, mode).map(|v| -v);
    }

    // Positive x delegates to the single maintained kernel (frankenscipy-waagw).
    // This function used to carry its OWN copy of the convergent series
    // Ei(x) = γ + ln x + Σ x^n/(n·n!), capped at 100 terms with a 1e-15 stop and
    // NO asymptotic branch. That series peaks near n≈x, so at x=50 it truncates
    // mid-ascent: it returned 105856368954878360000 against scipy's
    // 105856368971316900000, a relative error of 1.6e-10 — five digits gone,
    // silently. `convenience::expi_scalar` already switches to the divergent
    // asymptotic form at x ≥ 40 and tracks scipy to ~1e-15, so the duplicate is
    // deleted rather than patched: two Ei kernels in one crate is how one of
    // them ends up stale.
    Ok(crate::convenience::expi_scalar(x))
}

fn expi_complex_scalar(z: Complex64, mode: RuntimeMode) -> Result<Complex64, SpecialError> {
    if z.re.is_nan() || z.im.is_nan() {
        return Ok(complex_nan());
    }
    if !z.is_finite() {
        if z.im == 0.0 && z.re == f64::INFINITY {
            return Ok(Complex64::from_real(f64::INFINITY));
        }
        return Ok(complex_nan());
    }
    if z.re == 0.0 && z.im == 0.0 {
        return Ok(Complex64::new(f64::NEG_INFINITY, 0.0));
    }

    if z.im == 0.0 {
        return expi_scalar(z.re, mode).map(Complex64::from_real);
    }

    if z.abs() <= 40.0 {
        return expi_complex_series(z);
    }

    expi_complex_asymptotic(z)
}

fn expi_complex_series(z: Complex64) -> Result<Complex64, SpecialError> {
    let euler_gamma = 0.577_215_664_901_532_9;
    let mut sum = Complex64::from_real(euler_gamma) + z.ln();
    let mut term = z;
    sum = sum + term;

    for n in 2..400 {
        term = term * z / n as f64;
        let contribution = term / n as f64;
        sum = sum + contribution;
        if contribution.abs() < 1.0e-15 * sum.abs().max(1.0) {
            break;
        }
    }

    Ok(sum)
}

fn expi_complex_asymptotic(z: Complex64) -> Result<Complex64, SpecialError> {
    let mut sum = Complex64::from_real(1.0);
    let mut term = Complex64::from_real(1.0);

    for k in 1..100 {
        term = term * k as f64 / z;
        sum = sum + term;
        if term.abs() < 1.0e-15 * sum.abs().max(1.0) {
            break;
        }
    }

    Ok(z.exp() / z * sum)
}

// ══════════════════════════════════════════════════════════════════════
// Helpers
// ══════════════════════════════════════════════════════════════════════

/// Default per-element threshold below which the REAL arm runs serially. Cheap real kernels
/// (O(1) polynomial/rational, e.g. ellipk/ellipe Cephes) are slower under par_map_indices than
/// serial at any practical length (thread overhead >> ~14ns/call); callers pass `usize::MAX` to
/// force serial. Heavy real kernels keep the default so they still parallelize.
// Currently unreferenced: the real and complex tensor entry points in this
// module dispatch directly rather than through this combinator. RETAINED with a
// note rather than deleted (frankenscipy-e2ve2), matching the treatment of the
// other unused dispatch helpers in this crate; deletion is the owner's call.
#[allow(dead_code)]
fn map_real_or_complex<F, G>(
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
    map_real_or_complex_rp(function, input, mode, real_kernel, complex_kernel, 256)
}

fn map_real_or_complex_rp<F, G>(
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
            par_map_indices_gated(values.len(), real_par_min, |i| real_kernel(values[i]))
                .map(SpecialTensor::RealVec)
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

fn map_real_or_complex_binary<F, G>(
    function: &'static str,
    lhs: &SpecialTensor,
    rhs: &SpecialTensor,
    mode: RuntimeMode,
    real_kernel: F,
    complex_kernel: G,
    real_par_min: usize,
) -> SpecialResult
where
    F: Fn(f64, f64) -> Result<f64, SpecialError> + Sync,
    G: Fn(Complex64, Complex64) -> Result<Complex64, SpecialError> + Sync,
{
    match (lhs, rhs) {
        (SpecialTensor::Empty, _) | (_, SpecialTensor::Empty) => {
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
        (SpecialTensor::RealScalar(left), SpecialTensor::RealScalar(right)) => {
            real_kernel(*left, *right).map(SpecialTensor::RealScalar)
        }
        (SpecialTensor::RealVec(left), SpecialTensor::RealScalar(right)) => {
            let right = *right;
            par_map_indices_gated(left.len(), real_par_min, |i| real_kernel(left[i], right))
                .map(SpecialTensor::RealVec)
        }
        (SpecialTensor::RealScalar(left), SpecialTensor::RealVec(right)) => {
            let left = *left;
            par_map_indices_gated(right.len(), real_par_min, |i| real_kernel(left, right[i]))
                .map(SpecialTensor::RealVec)
        }
        (SpecialTensor::RealVec(left), SpecialTensor::RealVec(right)) => {
            if left.len() != right.len() {
                return vector_length_error(function, mode);
            }
            par_map_indices_gated(left.len(), real_par_min, |i| real_kernel(left[i], right[i]))
                .map(SpecialTensor::RealVec)
        }
        (SpecialTensor::ComplexScalar(left), SpecialTensor::ComplexScalar(right)) => {
            complex_kernel(*left, *right).map(SpecialTensor::ComplexScalar)
        }
        (SpecialTensor::ComplexVec(left), SpecialTensor::ComplexScalar(right)) => {
            let right = *right;
            par_map_indices(left.len(), |i| complex_kernel(left[i], right))
                .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::ComplexScalar(left), SpecialTensor::ComplexVec(right)) => {
            let left = *left;
            par_map_indices(right.len(), |i| complex_kernel(left, right[i]))
                .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::ComplexVec(left), SpecialTensor::ComplexVec(right)) => {
            if left.len() != right.len() {
                return vector_length_error(function, mode);
            }
            par_map_indices(left.len(), |i| complex_kernel(left[i], right[i]))
                .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::RealScalar(left), SpecialTensor::ComplexScalar(right)) => {
            complex_kernel(Complex64::from_real(*left), *right).map(SpecialTensor::ComplexScalar)
        }
        (SpecialTensor::ComplexScalar(left), SpecialTensor::RealScalar(right)) => {
            complex_kernel(*left, Complex64::from_real(*right)).map(SpecialTensor::ComplexScalar)
        }
        (SpecialTensor::RealVec(left), SpecialTensor::ComplexScalar(right)) => {
            let right = *right;
            par_map_indices(left.len(), |i| {
                complex_kernel(Complex64::from_real(left[i]), right)
            })
            .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::ComplexScalar(left), SpecialTensor::RealVec(right)) => {
            let left = *left;
            par_map_indices(right.len(), |i| {
                complex_kernel(left, Complex64::from_real(right[i]))
            })
            .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::RealScalar(left), SpecialTensor::ComplexVec(right)) => {
            let left = Complex64::from_real(*left);
            par_map_indices(right.len(), |i| complex_kernel(left, right[i]))
                .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::ComplexVec(left), SpecialTensor::RealScalar(right)) => {
            let right = Complex64::from_real(*right);
            par_map_indices(left.len(), |i| complex_kernel(left[i], right))
                .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::RealVec(left), SpecialTensor::ComplexVec(right)) => {
            if left.len() != right.len() {
                return vector_length_error(function, mode);
            }
            par_map_indices(left.len(), |i| {
                complex_kernel(Complex64::from_real(left[i]), right[i])
            })
            .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::ComplexVec(left), SpecialTensor::RealVec(right)) => {
            if left.len() != right.len() {
                return vector_length_error(function, mode);
            }
            par_map_indices(left.len(), |i| {
                complex_kernel(left[i], Complex64::from_real(right[i]))
            })
            .map(SpecialTensor::ComplexVec)
        }
    }
}

fn vector_length_error(
    function: &'static str,
    mode: RuntimeMode,
) -> Result<SpecialTensor, SpecialError> {
    record_special_trace(
        function,
        mode,
        "domain_error",
        "lhs_len!=rhs_len",
        "fail_closed",
        "vector inputs must have matching lengths",
        false,
    );
    Err(SpecialError {
        function,
        kind: SpecialErrorKind::DomainError,
        mode,
        detail: "vector inputs must have matching lengths",
    })
}

fn complex_nan() -> Complex64 {
    Complex64::new(f64::NAN, f64::NAN)
}

fn complex_sqrt(z: Complex64) -> Complex64 {
    if z.re == 0.0 && z.im == 0.0 {
        return Complex64::from_real(0.0);
    }
    if !z.is_finite() {
        return complex_nan();
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

fn complex_gauss_legendre_elliptic_f(phi: Complex64, m: Complex64) -> Complex64 {
    complex_integrate_unit_interval(&|t| {
        let theta = phi * t;
        let sin_theta = theta.sin();
        let inside = Complex64::from_real(1.0) - m * sin_theta * sin_theta;
        phi / complex_sqrt(inside)
    })
}

fn complex_gauss_legendre_elliptic_e(phi: Complex64, m: Complex64) -> Complex64 {
    complex_integrate_unit_interval(&|t| {
        let theta = phi * t;
        let sin_theta = theta.sin();
        let inside = Complex64::from_real(1.0) - m * sin_theta * sin_theta;
        phi * complex_sqrt(inside)
    })
}

fn complex_integrate_unit_interval<F>(kernel: &F) -> Complex64
where
    F: Fn(f64) -> Complex64,
{
    let whole = complex_gauss_legendre_interval(kernel, 0.0, 1.0);
    complex_integrate_recursive(kernel, 0.0, 1.0, whole, 0)
}

fn complex_integrate_recursive<F>(
    kernel: &F,
    start: f64,
    end: f64,
    estimate: Complex64,
    depth: usize,
) -> Complex64
where
    F: Fn(f64) -> Complex64,
{
    const MAX_DEPTH: usize = 12;
    const REL_TOL: f64 = 1.0e-12;

    let mid = 0.5 * (start + end);
    let left = complex_gauss_legendre_interval(kernel, start, mid);
    let right = complex_gauss_legendre_interval(kernel, mid, end);
    let refined = left + right;

    if depth >= MAX_DEPTH || (refined - estimate).abs() <= REL_TOL * refined.abs().max(1.0) {
        return refined;
    }

    complex_integrate_recursive(kernel, start, mid, left, depth + 1)
        + complex_integrate_recursive(kernel, mid, end, right, depth + 1)
}

fn complex_gauss_legendre_interval<F>(kernel: &F, start: f64, end: f64) -> Complex64
where
    F: Fn(f64) -> Complex64,
{
    const NODES: [f64; 8] = [
        0.987_992_518_020_485_4,
        0.937_273_392_400_706,
        0.848_206_583_410_427_2,
        0.724_417_731_360_170_1,
        0.570_972_172_608_538_8,
        0.394_151_347_077_563_4,
        0.201_194_093_997_435,
        0.0,
    ];
    const WEIGHTS: [f64; 8] = [
        0.030_753_241_996_117_3,
        0.070_366_047_488_108_1,
        0.107_159_220_467_171_9,
        0.139_570_677_926_154_1,
        0.166_269_205_816_994,
        0.186_161_000_015_562_2,
        0.198_431_485_327_111_6,
        0.202_578_241_925_561_3,
    ];

    let half_width = 0.5 * (end - start);
    let midpoint = 0.5 * (start + end);
    let mut sum = Complex64::from_real(0.0);
    for idx in 0..8 {
        let offset = half_width * NODES[idx];
        let t_pos = midpoint + offset;
        let t_neg = midpoint - offset;
        let value = if idx == 7 {
            kernel(t_pos) * WEIGHTS[idx]
        } else {
            (kernel(t_pos) + kernel(t_neg)) * WEIGHTS[idx]
        };
        sum = sum + value;
    }
    sum * half_width
}

fn domain_error(
    function: &'static str,
    mode: RuntimeMode,
    detail: &'static str,
) -> Result<f64, SpecialError> {
    match mode {
        RuntimeMode::Strict => Ok(f64::NAN),
        RuntimeMode::Hardened => Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail,
        }),
    }
}

/// 15-point Gauss-Legendre quadrature for incomplete elliptic integral F(φ, m).
struct EllipkincSin2Nodes {
    pos: [f64; 8],
    neg: [f64; 8],
}

#[allow(dead_code)] // retained for reference; superseded by Carlson R_F/R_D (o65r0)
fn ellipkinc_precomputed_sin2_nodes(phi: f64) -> EllipkincSin2Nodes {
    const NODES: [f64; 8] = [
        0.987_992_518_020_485_4,
        0.937_273_392_400_706,
        0.848_206_583_410_427_2,
        0.724_417_731_360_170_1,
        0.570_972_172_608_538_8,
        0.394_151_347_077_563_4,
        0.201_194_093_997_435,
        0.0,
    ];

    let half_phi = 0.5 * phi;
    let mut pos = [0.0; 8];
    let mut neg = [0.0; 8];
    for i in 0..8 {
        let t_pos = half_phi * (1.0 + NODES[i]);
        let t_neg = half_phi * (1.0 - NODES[i]);
        pos[i] = t_pos.sin().powi(2);
        neg[i] = t_neg.sin().powi(2);
    }
    EllipkincSin2Nodes { pos, neg }
}

#[allow(dead_code)] // superseded by Carlson R_F/R_D (o65r0)
fn gauss_legendre_elliptic_f_with_sin2(phi: f64, m: f64, nodes: &EllipkincSin2Nodes) -> f64 {
    const WEIGHTS: [f64; 8] = [
        0.030_753_241_996_117_3,
        0.070_366_047_488_108_1,
        0.107_159_220_467_171_9,
        0.139_570_677_926_154_1,
        0.166_269_205_816_994,
        0.186_161_000_015_562_2,
        0.198_431_485_327_111_6,
        0.202_578_241_925_561_3,
    ];

    let half_phi = 0.5 * phi;
    let mut sum = 0.0;
    for (i, weight) in WEIGHTS.iter().enumerate() {
        let f_pos = 1.0 / (1.0 - m * nodes.pos[i]).sqrt();
        let f_neg = 1.0 / (1.0 - m * nodes.neg[i]).sqrt();
        if i == 7 {
            sum += *weight * f_pos;
        } else {
            sum += *weight * (f_pos + f_neg);
        }
    }
    half_phi * sum
}

#[allow(dead_code)] // superseded by Carlson R_F/R_D (o65r0)
fn gauss_legendre_elliptic_f(phi: f64, m: f64) -> f64 {
    // Gauss-Legendre nodes and weights for [-1, 1], n=15
    const NODES: [f64; 8] = [
        0.987_992_518_020_485_4,
        0.937_273_392_400_706,
        0.848_206_583_410_427_2,
        0.724_417_731_360_170_1,
        0.570_972_172_608_538_8,
        0.394_151_347_077_563_4,
        0.201_194_093_997_435,
        0.0,
    ];
    const WEIGHTS: [f64; 8] = [
        0.030_753_241_996_117_3,
        0.070_366_047_488_108_1,
        0.107_159_220_467_171_9,
        0.139_570_677_926_154_1,
        0.166_269_205_816_994,
        0.186_161_000_015_562_2,
        0.198_431_485_327_111_6,
        0.202_578_241_925_561_3,
    ];

    let half_phi = 0.5 * phi;
    let mut sum = 0.0;
    for i in 0..8 {
        let t_pos = half_phi * (1.0 + NODES[i]);
        let t_neg = half_phi * (1.0 - NODES[i]);
        let f_pos = 1.0 / (1.0 - m * t_pos.sin().powi(2)).sqrt();
        let f_neg = 1.0 / (1.0 - m * t_neg.sin().powi(2)).sqrt();
        if i == 7 {
            // Center node (weight only once, symmetric around 0 maps to center)
            sum += WEIGHTS[i] * f_pos;
        } else {
            sum += WEIGHTS[i] * (f_pos + f_neg);
        }
    }
    half_phi * sum
}

/// 15-point Gauss-Legendre quadrature for incomplete elliptic integral E(φ, m).
#[allow(dead_code)] // superseded by Carlson R_F/R_D (o65r0)
fn gauss_legendre_elliptic_e(phi: f64, m: f64) -> f64 {
    const NODES: [f64; 8] = [
        0.987_992_518_020_485_4,
        0.937_273_392_400_706,
        0.848_206_583_410_427_2,
        0.724_417_731_360_170_1,
        0.570_972_172_608_538_8,
        0.394_151_347_077_563_4,
        0.201_194_093_997_435,
        0.0,
    ];
    const WEIGHTS: [f64; 8] = [
        0.030_753_241_996_117_3,
        0.070_366_047_488_108_1,
        0.107_159_220_467_171_9,
        0.139_570_677_926_154_1,
        0.166_269_205_816_994,
        0.186_161_000_015_562_2,
        0.198_431_485_327_111_6,
        0.202_578_241_925_561_3,
    ];

    let half_phi = 0.5 * phi;
    let mut sum = 0.0;
    for i in 0..8 {
        let t_pos = half_phi * (1.0 + NODES[i]);
        let t_neg = half_phi * (1.0 - NODES[i]);
        let f_pos = (1.0 - m * t_pos.sin().powi(2)).sqrt();
        let f_neg = (1.0 - m * t_neg.sin().powi(2)).sqrt();
        if i == 7 {
            sum += WEIGHTS[i] * f_pos;
        } else {
            sum += WEIGHTS[i] * (f_pos + f_neg);
        }
    }
    half_phi * sum
}

// ══════════════════════════════════════════════════════════════════════
// Jacobi elliptic functions
// ══════════════════════════════════════════════════════════════════════

/// Jacobi elliptic functions sn(u, m), cn(u, m), dn(u, m), and amplitude ph(u, m).
///
/// Computed via the arithmetic-geometric mean (descending Landen transformation).
///
/// Parameters:
/// - `u`: argument (real)
/// - `m`: parameter (0 <= m <= 1)
///
/// Returns (sn, cn, dn, ph) where:
/// - sn(u, m) = sin(am(u, m))
/// - cn(u, m) = cos(am(u, m))
/// - dn(u, m) = √(1 - m·sn²)
/// - ph(u, m) = am(u, m) (the Jacobi amplitude)
///
/// Identities: sn² + cn² = 1, dn² + m·sn² = 1
pub fn ellipj(u: f64, m: f64) -> (f64, f64, f64, f64) {
    if m.is_nan() || u.is_nan() {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }

    // Special case: m = 0 => sn = sin(u), cn = cos(u), dn = 1
    if m.abs() < 1e-15 {
        return (u.sin(), u.cos(), 1.0, u);
    }

    // Special case: m = 1 => sn = tanh(u), cn = dn = sech(u)
    if (m - 1.0).abs() < 1e-15 {
        let sn = u.tanh();
        let cn = 1.0 / u.cosh();
        return (
            sn,
            cn,
            cn,
            2.0 * u.exp().atan() - std::f64::consts::FRAC_PI_2,
        );
    }

    // AGM-based computation via descending Landen transformation
    const MAX_ITER: usize = 20;
    let mut a = [0.0; MAX_ITER + 1];
    let mut b = [0.0; MAX_ITER + 1];
    let mut c = [0.0; MAX_ITER + 1];

    a[0] = 1.0;
    b[0] = (1.0 - m).sqrt();
    c[0] = m.sqrt();

    let mut n = 0;
    while c[n].abs() > 1e-16 && n < MAX_ITER {
        let a_new = (a[n] + b[n]) / 2.0;
        let b_new = (a[n] * b[n]).sqrt();
        let c_new = (a[n] - b[n]) / 2.0;
        n += 1;
        a[n] = a_new;
        b[n] = b_new;
        c[n] = c_new;
    }

    // Compute amplitude by back-substitution
    let mut phi = (1u64 << n) as f64 * a[n] * u;
    for k in (1..=n).rev() {
        phi = (phi + (c[k] / a[k] * phi.sin()).clamp(-1.0, 1.0).asin()) / 2.0;
    }

    let sn = phi.sin();
    let cn = phi.cos();
    let dn = (1.0 - m * sn * sn).sqrt();

    (sn, cn, dn, phi)
}

/// Vectorized Jacobi elliptic functions `(sn, cn, dn, ph)` over many arguments
/// `u` for a fixed parameter `m` — fsci's AGM/Landen scalar is already ~1.24x
/// faster than SciPy's cephes `ellipj` ufunc even serial (194 vs 241 ns/pt), so
/// the order-preserving parallel fan is a large win. Bit-identical to a serial
/// map; matches SciPy to ~6e-15.
#[must_use]
pub fn ellipj_many(u: &[f64], m: f64) -> Vec<(f64, f64, f64, f64)> {
    par_map_indices(u.len(), |i| {
        Ok::<(f64, f64, f64, f64), SpecialError>(ellipj(u[i], m))
    })
    .expect("ellipj is infallible")
}

/// Carlson symmetric elliptic integral of the first kind, degenerate
/// case `RC(x, y)`.
///
/// Definition (Carlson 1995, scipy.special.elliprc):
///
/// ```text
///   RC(x, y) = (1/2) ∫₀^∞ (t + x)^{-1/2} (t + y)^{-1} dt
/// ```
///
/// SciPy's bits: this is `ellint_carlson::rc` from SciPy 1.17.1's
/// `ellint_carlson_cpp_lite/_rc.hh` at the relative error bound its ufunc passes (5e-16).
/// Carlson's duplication runs until the arguments agree to that bound, then the degree-7
/// series of Carlson (1995) eq. (20) is summed with a compensated Horner scheme.
///
/// The closed forms `arccos(√(x/y))/√(y − x)` and `arccosh(√(x/y))/√(x − y)` this replaced
/// cancel as `x → y`: 1.9e-11 relative off at (9.95527, 9.955275). That is the argument pair
/// `elliprj`'s duplication hands RC once `p ≈ x`, and it made RJ 8e-5 off
/// (frankenscipy-2f8h7). Their diagonal test was an absolute `8·eps·max(x, y, 1)`, so tiny
/// arguments snapped to `1/√x` (3.3e-10 off at (1e-300, 1.000000001e-300)). `x/y`
/// overflowed at (1e300, 1e-300).
///
/// SciPy's domain: `y` zero or subnormal, or `x` negative or NaN, is NaN; an infinite
/// argument is 0; a negative `y` is the Cauchy principal value
/// `RC(x − y, −y)·√(x/(x − y))` (Carlson 1995, eq. 2.14).
#[must_use]
pub fn elliprc(x: f64, y: f64) -> f64 {
    /// SciPy's `constants::RC_C`, lowest degree first: 80080 times the series
    /// 1 + 3s²/10 + s³/7 + 3s⁴/8 + 9s⁵/22 + 159s⁶/208 + 9s⁷/8.
    const RC_C: [f64; 8] = [
        80080.0, 0.0, 24024.0, 11440.0, 30030.0, 32760.0, 61215.0, 90090.0,
    ];
    if y < 0.0 {
        return elliprc(x - y, -y) * (x / (x - y)).sqrt();
    }
    if y.is_nan() || y == 0.0 || y.is_subnormal() || x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x.is_infinite() || y.is_infinite() {
        return 0.0;
    }
    let mut am = (x + 2.0 * y) / 3.0;
    let mut fterm = (am - x).abs() / (3.0 * CARLSON_RERR).sqrt().sqrt().sqrt();
    let (mut xm, mut ym) = (x, y);
    let mut sm = y - am;
    let mut m = 0_u32;
    loop {
        // SciPy continues while `std::max(|xm − ym|, fterm) >= |Am|`; the C++ max keeps its
        // first argument unless the second is larger.
        let d = (xm - ym).abs();
        if !((if d < fterm { fterm } else { d }) >= am.abs()) {
            break;
        }
        if m > CARLSON_MAX_ITER {
            break;
        }
        let lam = 2.0 * xm.sqrt() * ym.sqrt() + ym;
        am = (am + lam) * 0.25;
        xm = (xm + lam) * 0.25;
        ym = (ym + lam) * 0.25;
        sm *= 0.25;
        fterm *= 0.25;
        m += 1;
    }
    am = (xm + ym + ym) / 3.0;
    sm /= am;
    carlson_comp_horner(sm, &RC_C) / (am.sqrt() * RC_C[0])
}

/// The relative error bound SciPy passes to every Carlson integral (`ellip_rerr` in
/// `ellint_carlson_wrap.cxx`).
const CARLSON_RERR: f64 = 5e-16;

/// SciPy's `config::max_iter` for the Carlson duplication loops.
const CARLSON_MAX_ITER: u32 = 1000;

/// Compensated Horner evaluation of `poly` (lowest degree first) at `x`: SciPy's
/// `arithmetic::dcomp_horner` (Graillat, Langlois and Louvet, Algorithm 9). The rounding
/// error of each product (by FMA) and each sum (Knuth's TwoSum) is carried in `r` and added
/// once at the end.
fn carlson_comp_horner<const N: usize>(x: f64, poly: &[f64; N]) -> f64 {
    let mut s = poly[N - 1];
    let mut r = 0.0;
    for &c in poly[..N - 1].iter().rev() {
        let prod = s * x;
        let prod_err = s.mul_add(x, -prod);
        let sum = prod + c;
        let z = sum - prod;
        let sum_err = (prod - (sum - z)) + (c - z);
        s = sum;
        r = r * x + (prod_err + sum_err);
    }
    s + r
}

/// Carlson symmetric elliptic integral of the first kind, `RF(x, y, z)`.
///
/// Definition (Carlson 1995, scipy.special.elliprf):
///
/// ```text
///   RF(x, y, z) = (1/2) ∫₀^∞ [(t + x)(t + y)(t + z)]^{-1/2} dt
/// ```
///
/// SciPy's bits: this is `ellint_carlson::rf` from SciPy 1.17.1's
/// `ellint_carlson_cpp_lite/_rf.hh` at the ufunc's relative error bound (5e-16). The
/// arguments are sorted by size and duplicated, with `λ = √x√y + √y√z + √z√x` as a compensated
/// dot product, until they agree to that bound (up to 1000 steps). Then Carlson's
/// degree-7 series in E₂, E₃ (DLMF 19.36.1) is summed with the compensated Horner scheme. A
/// zero smallest argument goes through an AGM instead (`rf0`, DLMF 19.27.3).
///
/// The kernel this replaced stopped after 32 duplications and fell back to `1/√mean`, and
/// formed `√(xz)` from the product. With z near 1e257 against x, y near 10 that returned garbage
/// (relative error 1.0 where SciPy is 2e-16), and `x·z` overflowed near 1e300
/// (frankenscipy-mo0yq).
///
/// SciPy's domain: a negative or NaN argument is NaN; any infinite argument is 0; two
/// arguments that are zero or subnormal are +∞.
#[must_use]
pub fn elliprf(x: f64, y: f64, z: f64) -> f64 {
    /// SciPy's `constants::RF_C1`, `RF_C2`, `RF_c33` and `RF_DENOM`, lowest degree first.
    const RF_C1: [f64; 4] = [0.0, -24024.0, 10010.0, -5775.0];
    const RF_C2: [f64; 3] = [17160.0, -16380.0, 15015.0];
    const RF_C33: f64 = 6930.0;
    const RF_DENOM: f64 = 240240.0;
    // `ph_good` is `x >= 0.0`, which a NaN fails.
    if !(x >= 0.0 && y >= 0.0 && z >= 0.0) {
        return f64::NAN;
    }
    if x.is_infinite() || y.is_infinite() || z.is_infinite() {
        return 0.0;
    }
    let mut sorted = [x, y, z];
    sorted.sort_by(f64::total_cmp);
    let [mut xm, mut ym, mut zm] = sorted;
    if carlson_too_small(xm) {
        if carlson_too_small(ym) {
            return f64::INFINITY;
        }
        return carlson_rf0(ym, zm, CARLSON_RERR * 0.5) - (xm / (ym * zm)).sqrt();
    }
    let mut am = carlson_sum3(xm, ym, zm) / 3.0;
    let mut xxm = am - xm;
    let mut yym = am - ym;
    let mut fterm = carlson_abs_max3(xxm, yym, am - zm) / (3.0 * CARLSON_RERR).sqrt().sqrt().sqrt();
    let mut m = 0_u32;
    loop {
        let aam = am.abs();
        if !(aam <= fterm || aam <= carlson_abs_max3(xxm, yym, am - zm)) {
            break;
        }
        if m > CARLSON_MAX_ITER {
            break;
        }
        let (sx, sy, sz) = (xm.sqrt(), ym.sqrt(), zm.sqrt());
        let lam = carlson_dot3([sx, sy, sz], [sy, sz, sx]);
        am = (am + lam) * 0.25;
        xm = (xm + lam) * 0.25;
        ym = (ym + lam) * 0.25;
        zm = (zm + lam) * 0.25;
        xxm *= 0.25;
        yym *= 0.25;
        fterm *= 0.25;
        m += 1;
    }
    am = carlson_sum3(xm, ym, zm) / 3.0;
    xxm /= am;
    yym /= am;
    let zzm = -(xxm + yym);
    let e2 = xxm * yym - zzm * zzm;
    let e3 = xxm * (yym * zzm);
    let mut s = carlson_comp_horner(e2, &RF_C1);
    s += e3 * (carlson_comp_horner(e2, &RF_C2) + e3 * RF_C33);
    s /= RF_DENOM;
    s += 1.0;
    s / am.sqrt()
}

/// SciPy's `argcheck::too_small`: zero or subnormal.
fn carlson_too_small(v: f64) -> bool {
    v == 0.0 || v.is_subnormal()
}

/// `|std::max({a, b, c}, abscmp)|`: the largest magnitude of the three.
fn carlson_abs_max3(a: f64, b: f64, c: f64) -> f64 {
    let mut best = a;
    for v in [b, c] {
        if best.abs() < v.abs() {
            best = v;
        }
    }
    best.abs()
}

/// Knuth's TwoSum: `x + y` and its exact rounding error (SciPy's `arithmetic::eft_sum`).
fn carlson_two_sum(x: f64, y: f64) -> (f64, f64) {
    let s = x + y;
    let z = s - x;
    (s, (x - (s - z)) + (y - z))
}

/// SciPy's `arithmetic::sum2` of three terms: compensated, in order.
fn carlson_sum3(a: f64, b: f64, c: f64) -> f64 {
    let (mut p, mut s) = (0.0, 0.0);
    for v in [a, b, c] {
        let (t, e) = carlson_two_sum(v, p);
        p = t;
        s += e;
    }
    p + s
}

/// SciPy's `arithmetic::dot2` of three pairs: each product's error by FMA, each sum's by TwoSum.
fn carlson_dot3(x: [f64; 3], y: [f64; 3]) -> f64 {
    let (mut p, mut s) = (0.0, 0.0);
    for (a, b) in x.into_iter().zip(y) {
        let h = a * b;
        let r = a.mul_add(b, -h);
        let (t, q) = carlson_two_sum(p, h);
        p = t;
        s += q + r;
    }
    p + s
}

/// SciPy's `rf0`: `RF(0, x, y)` by the arithmetic-geometric mean, to `2√rerr`.
fn carlson_rf0(x: f64, y: f64, rerr: f64) -> f64 {
    let rsq = 2.0 * rerr.sqrt();
    let (mut xm, mut ym) = (x.sqrt(), y.sqrt());
    let mut m = 0_u32;
    while (xm - ym).abs() >= rsq * xm.abs().min(ym.abs()) {
        if m > CARLSON_MAX_ITER {
            break;
        }
        (xm, ym) = ((xm + ym) * 0.5, (xm * ym).sqrt());
        m += 1;
    }
    PI / (xm + ym)
}

/// Carlson symmetric elliptic integral of the second kind, `RD(x, y, z)`.
///
/// Definition (Carlson 1995, scipy.special.elliprd):
///
/// ```text
///   RD(x, y, z) = (3/2) ∫₀^∞ (t + x)^{-1/2} (t + y)^{-1/2}
///                          (t + z)^{-3/2} dt
/// ```
///
/// Computed via Carlson's duplication algorithm; the third argument
/// `z` is special (the integrand is t^{-3/2} in z), so the iterative
/// substitution accumulates a running `sum_term` that contributes to
/// the final result. RD is symmetric only in `(x, y)` — `z` plays a
/// distinguished role.
///
/// Returns NaN for negative or NaN inputs and for the case where two
/// or more arguments are zero (non-integrable singularity).
///
/// Resolves [frankenscipy-xdi1c].
#[must_use]
pub fn elliprd(x: f64, y: f64, z: f64) -> f64 {
    if x.is_nan() || y.is_nan() || z.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 || y < 0.0 || z <= 0.0 {
        // z must be strictly positive; x, y can be 0 (but not both).
        return f64::NAN;
    }
    if x == 0.0 && y == 0.0 {
        return f64::INFINITY;
    }

    let mut xn = x;
    let mut yn = y;
    let mut zn = z;
    let mut sum_term = 0.0_f64;
    let mut factor = 1.0_f64;
    const TOL: f64 = 1e-4;

    for _ in 0..32 {
        let mu = (xn + yn + 3.0 * zn) / 5.0;
        let ex = 1.0 - xn / mu;
        let ey = 1.0 - yn / mu;
        let ez = 1.0 - zn / mu;
        let max_e = ex.abs().max(ey.abs()).max(ez.abs());
        if max_e < TOL {
            // 5th-order Taylor correction (Carlson 1995 eq. 2.14).
            let ea = ex * ey;
            let eb = ez * ez;
            let ec = ea - eb;
            let ed = ea - 6.0 * eb;
            let ef = ed + ec + ec;
            let s1 = ed * (-3.0 / 14.0 + 9.0 / 88.0 * ed - 4.5 / 26.0 * ez * ef);
            let s2 = ez * (ef / 6.0 + ez * (-9.0 / 22.0 * ec + ez * 3.0 / 26.0 * ea));
            return 3.0 * sum_term + factor * (1.0 + s1 + s2) / (mu * mu.sqrt());
        }
        let lam = (xn * yn).sqrt() + (yn * zn).sqrt() + (xn * zn).sqrt();
        sum_term += factor / ((zn + lam) * zn.sqrt());
        factor /= 4.0;
        xn = 0.25 * (xn + lam);
        yn = 0.25 * (yn + lam);
        zn = 0.25 * (zn + lam);
    }
    f64::NAN
}

/// Carlson symmetric elliptic integral of the second kind, `RG(x, y, z)`.
///
/// Definition (Carlson 1995, scipy.special.elliprg):
///
/// ```text
///   RG(x, y, z) = (1/(4π)) ∫∫_{S²} √(x·u² + y·v² + z·w²) dω
/// ```
///
/// the average of √(x·u² + y·v² + z·w²) over the unit sphere. This
/// connects the Carlson family to the standard complete elliptic
/// integrals: RG(0, 1, 1) = E(0)/2 = π/4, and 2·RG(0, 1−k², 1) =
/// E(k) (the complete elliptic integral of the second kind).
///
/// Implemented via Carlson's stable identity:
///
/// ```text
///   RG(x, y, z) = (z · RF(x, y, z) + √(x·y / z)) / 2
///                  + (z − x) · (z − y) · RD(x, y, z) / (−6)
/// ```
///
/// for `z > 0`. When any argument is `0` we permute to place a
/// positive value last (RG is symmetric in all three arguments).
///
/// Returns NaN for negative or NaN inputs.
///
/// Resolves [frankenscipy-781n7].
#[must_use]
pub fn elliprg(x: f64, y: f64, z: f64) -> f64 {
    if x.is_nan() || y.is_nan() || z.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 || y < 0.0 || z < 0.0 {
        return f64::NAN;
    }

    // RG is symmetric in all three arguments; permute so that the last
    // argument is positive (the Carlson formula requires z > 0).
    let (a, b, c) = if z > 0.0 {
        (x, y, z)
    } else if y > 0.0 {
        (x, z, y)
    } else if x > 0.0 {
        (y, z, x)
    } else {
        // All zero — RG(0, 0, 0) = 0.
        return 0.0;
    };

    // Carlson identity (a, b, c) with c > 0:
    //   RG = (c · RF(a, b, c) + √(a·b/c)) / 2
    //        + (c − a)·(c − b) · RD(a, b, c) / (−6)
    let rf = elliprf(a, b, c);
    let term_rf = (c * rf + (a * b / c).sqrt()) / 2.0;
    let term_rd = if a == 0.0 && b == 0.0 {
        // (c − 0)(c − 0)·RD(0, 0, c)/(−6) — RD(0, 0, c) is +∞,
        // but the (c − a)(c − b) factor makes the limit well-defined.
        // Skip: when a = b = 0, RG(0, 0, c) = √c / 2.
        return c.sqrt() / 2.0;
    } else {
        let rd = elliprd(a, b, c);
        (c - a) * (c - b) * rd / (-6.0)
    };
    term_rf + term_rd
}

/// Carlson symmetric elliptic integral of the third kind, `RJ(x, y, z, p)`.
///
/// Definition (Carlson 1995, scipy.special.elliprj):
///
/// ```text
///   RJ(x, y, z, p) = (3/2) ∫₀^∞ [(t + x)(t + y)(t + z)]^{-1/2} (t + p)^{-1} dt
/// ```
///
/// Computed via Carlson's duplication algorithm with an auxiliary
/// running RC sum that absorbs the (t + p) pole. At each step:
///
/// * `λ = √(xy) + √(yz) + √(xz)`, the standard substitution.
/// * `α = (p (√x + √y + √z) + √(xyz))²`, `β = p (p + λ)²`,
///   `sum += factor · RC(α, β)`.
/// * `(x, y, z, p) → ((x + λ)/4, (y + λ)/4, (z + λ)/4, (p + λ)/4)`,
///   `factor /= 4`.
///
/// When the relative residuals are small, close with a 5th-order
/// Taylor series in (Eₓ, E_y, E_z, E_p). Returns
/// `RJ = 3·sum + factor · (1 + Taylor) · μ^{-3/2}`.
///
/// Constraints (matches scipy's real-valued path):
///   * `x, y, z ≥ 0` with at most one zero,
///   * `p > 0` (Cauchy-PV path for `p < 0` deferred — returns NaN),
///   * all finite.
///
/// Closed-form anchors:
///   * `RJ(x, x, x, x) = x^{-3/2}` (the integrand collapses).
///   * `RJ(x, y, z, z) = RD(x, y, z)` (the same integral).
///   * `RJ(x, y, z, p)` is symmetric in `(x, y, z)`.
///
/// Resolves [frankenscipy-ewuqd]; completes the Carlson family
/// (RC, RF, RD, RG, RJ).
#[must_use]
pub fn elliprj(x: f64, y: f64, z: f64, p: f64) -> f64 {
    if x.is_nan() || y.is_nan() || z.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 || y < 0.0 || z < 0.0 {
        return f64::NAN;
    }
    // p == 0 is a genuine pole (RC(·, 0) = ∞ enters with unit weight); scipy
    // returns NaN there, so keep failing closed only on the exact zero.
    if p == 0.0 {
        return f64::NAN;
    }
    let zero_count = (x == 0.0) as usize + (y == 0.0) as usize + (z == 0.0) as usize;
    if zero_count >= 2 {
        // RJ diverges when two of (x, y, z) are zero.
        return f64::INFINITY;
    }
    if p < 0.0 {
        // Cauchy principal value for p < 0 (Carlson 1995, eq. 2.20; Boost's
        // ellint_rj reduction). Sort x ≤ y ≤ z so y is the middle argument,
        // then express the PV through a positive-p RJ plus RF and RC:
        //   q   = −p,  pmy = (z − y)(y − x)/(y + q),  pn = pmy + y  (> 0)
        //   RJ  = [ pmy·RJ(x, y, z, pn) − 3·RF(x, y, z)
        //           + 3·√(xyz / (xz + pn·q))·RC(xz + pn·q, pn·q) ] / (y + q)
        // Every recursive argument is non-negative, so the positive-p branch
        // and RC's diagonal handle the corners (e.g. a single zero argument).
        let mut xs = [x, y, z];
        xs.sort_unstable_by(|a, b| a.total_cmp(b));
        let [xt, yt, zt] = xs;
        let q = -p;
        let pmy = (zt - yt) * (yt - xt) / (yt + q);
        let pn = pmy + yt;
        let rc_arg = xt * zt + pn * q;
        let value = pmy * elliprj(xt, yt, zt, pn) - 3.0 * elliprf(xt, yt, zt)
            + 3.0 * (xt * yt * zt / rc_arg).sqrt() * elliprc(rc_arg, pn * q);
        return value / (yt + q);
    }

    let mut xn = x;
    let mut yn = y;
    let mut zn = z;
    let mut pn = p;
    let mut sum = 0.0_f64;
    let mut factor = 1.0_f64;
    const TOL: f64 = 5e-4;

    for _ in 0..100 {
        let mu = 0.2 * (xn + yn + zn + 2.0 * pn);
        let ex = 1.0 - xn / mu;
        let ey = 1.0 - yn / mu;
        let ez = 1.0 - zn / mu;
        let ep = 1.0 - pn / mu;
        let max_e = ex.abs().max(ey.abs()).max(ez.abs()).max(ep.abs());
        if max_e < TOL {
            // 5th-order Taylor closing (Carlson 1995, Numerical Recipes 6.11).
            // Note: the eb term's leading coefficient is C7 = C2/2 = 1/6,
            // not C2 itself — that distinction is the difference between
            // a correct RJ and one that fails the RJ(x, y, z, z) = RD test.
            const C1: f64 = 3.0 / 14.0;
            const C2: f64 = 1.0 / 3.0;
            const C3: f64 = 3.0 / 22.0;
            const C4: f64 = 3.0 / 26.0;
            const C5: f64 = 0.75 * C3;
            const C6: f64 = 1.5 * C4;
            const C7: f64 = 0.5 * C2;
            const C8: f64 = C3 + C3;
            let ea = ex * (ey + ez) + ey * ez;
            let eb = ex * ey * ez;
            let ec = ep * ep;
            let ed = ea - 3.0 * ec;
            let ee = eb + 2.0 * ep * (ea - ec);
            let series = 1.0
                + ed * (-C1 + C5 * ed - C6 * ee)
                + eb * (C7 + ep * (-C8 + ep * C4))
                + ep * ea * (C2 - ep * C3)
                - C2 * ep * ec;
            return 3.0 * sum + factor * series / (mu * mu.sqrt());
        }

        let sqx = xn.sqrt();
        let sqy = yn.sqrt();
        let sqz = zn.sqrt();
        let lambda = sqx * sqy + sqy * sqz + sqx * sqz;
        let alpha_root = pn * (sqx + sqy + sqz) + sqx * sqy * sqz;
        let alpha = alpha_root * alpha_root;
        let beta = pn * (pn + lambda) * (pn + lambda);
        sum += factor * elliprc(alpha, beta);

        factor *= 0.25;
        xn = 0.25 * (xn + lambda);
        yn = 0.25 * (yn + lambda);
        zn = 0.25 * (zn + lambda);
        pn = 0.25 * (pn + lambda);
    }
    f64::NAN
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ellipk_ellipe_complete_match_scipy() {
        // scipy.special.ellipk/ellipe (complete elliptic integrals).
        let m = RuntimeMode::Strict;
        assert!(
            (ellipk_scalar(0.5, m).unwrap() - 1.854_074_677_301_371_9).abs() < 1e-13,
            "ellipk(0.5)"
        );
        assert!(
            (ellipk_scalar(0.0, m).unwrap() - std::f64::consts::FRAC_PI_2).abs() < 1e-14,
            "ellipk(0)=pi/2"
        );
        assert!(
            (ellipe_scalar(0.5, m).unwrap() - 1.350_643_881_047_675_5).abs() < 1e-13,
            "ellipe(0.5)"
        );
        assert!(
            (ellipe_scalar(1.0, m).unwrap() - 1.0).abs() < 1e-14,
            "ellipe(1)=1"
        );
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    // golden constants verbatim from scipy
    // φ test inputs 1.5707/1.5707963 are deliberate angles near (but not equal
    // to) π/2 with their own scipy reference values — not FRAC_PI_2.
    #[allow(clippy::approx_constant)]
    fn ellipkinc_ellipeinc_match_scipy() {
        // frankenscipy-o65r0: Carlson R_F/R_D replace fixed Gauss-Legendre, which
        // was 0.9% off at the m→1, φ→π/2 corner. scipy.special 1.17.1.
        let m = RuntimeMode::Strict;
        let s = |v: f64| SpecialTensor::RealScalar(v);
        let g = |r: SpecialResult| match r {
            Ok(SpecialTensor::RealScalar(v)) => v,
            _ => f64::NAN,
        };
        let cases = [
            (1.0, 0.5, 1.0832167728451687, 0.92732988362444),
            (1.5, 0.99, 3.0360140973397103, 1.0083662457039582),
            (1.5707, 0.9999, 5.9819568099632745, 1.0002736191478188),
            (1.5707963, 0.999999, 8.29402466870448, 1.0000038969993772),
            (0.5, 1.0, 0.5222381032784403, 0.479425538604203),
            (2.5, 0.5, 3.0444084774872615, 2.0805595497588447),
            (-2.0, 0.3, -2.220590552128474, -1.8089647253633316),
            (5.0, 0.9, 8.558511085027696, 3.4153983161223085),
        ];
        for (phi, mm, f_ref, e_ref) in cases {
            let f = g(ellipkinc(&s(phi), &s(mm), m));
            let e = g(ellipeinc(&s(phi), &s(mm), m));
            // ~3e-12 at the most extreme m→1, φ→π/2 corner (Carlson ERRTOL
            // floor); the prior fixed-quadrature error there was ~0.9%.
            assert!(
                ((f - f_ref) / f_ref).abs() < 1e-11,
                "ellipkinc({phi},{mm}) = {f}, scipy {f_ref}"
            );
            assert!(
                ((e - e_ref) / e_ref).abs() < 1e-11,
                "ellipeinc({phi},{mm}) = {e}, scipy {e_ref}"
            );
        }
    }

    fn assert_close(actual: f64, expected: f64, tol: f64, msg: &str) {
        assert!(
            (actual - expected).abs() < tol,
            "{msg}: got {actual}, expected {expected} (diff={})",
            (actual - expected).abs()
        );
    }

    fn assert_complex_close(actual: Complex64, expected: Complex64, tol: f64, msg: &str) {
        let delta = (actual - expected).abs();
        assert!(
            delta <= tol,
            "{msg}: got {}+{}i, expected {}+{}i (|delta|={delta})",
            actual.re,
            actual.im,
            expected.re,
            expected.im,
        );
    }

    fn eval_scalar(result: SpecialResult) -> f64 {
        match result.expect("should succeed") {
            SpecialTensor::RealScalar(v) => v,
            other => {
                assert!(
                    matches!(&other, SpecialTensor::RealScalar(_)),
                    "expected RealScalar, got {other:?}"
                );
                f64::NAN
            }
        }
    }

    fn eval_complex_scalar(result: SpecialResult) -> Complex64 {
        match result.expect("should succeed") {
            SpecialTensor::ComplexScalar(v) => v,
            other => {
                assert!(
                    matches!(&other, SpecialTensor::ComplexScalar(_)),
                    "expected ComplexScalar, got {other:?}"
                );
                Complex64::new(f64::NAN, f64::NAN)
            }
        }
    }

    // ── Complete elliptic integrals ───────────────────────────────

    #[test]
    fn ellipk_at_zero() {
        let m = SpecialTensor::RealScalar(0.0);
        let result = eval_scalar(ellipk(&m, RuntimeMode::Strict));
        assert_close(result, PI / 2.0, 1e-12, "K(0) = π/2");
    }

    #[test]
    fn ellipk_at_half() {
        // K(0.5) ≈ 1.854_074_677
        let m = SpecialTensor::RealScalar(0.5);
        let result = eval_scalar(ellipk(&m, RuntimeMode::Strict));
        assert_close(result, 1.854_074_677, 1e-8, "K(0.5)");
    }

    #[test]
    fn ellipk_at_one_is_infinity() {
        let m = SpecialTensor::RealScalar(1.0);
        let result = eval_scalar(ellipk(&m, RuntimeMode::Strict));
        assert!(result.is_infinite(), "K(1) should be infinite");
    }

    #[test]
    fn hardened_ellipk_rejects_singular_endpoint() {
        let m = SpecialTensor::RealScalar(1.0);
        let err = ellipk(&m, RuntimeMode::Hardened).expect_err("hardened rejects K(1)");
        assert_eq!(err.kind, SpecialErrorKind::DomainError);
    }

    #[test]
    fn ellipe_at_zero() {
        let m = SpecialTensor::RealScalar(0.0);
        let result = eval_scalar(ellipe(&m, RuntimeMode::Strict));
        assert_close(result, PI / 2.0, 1e-12, "E(0) = π/2");
    }

    #[test]
    fn ellipe_at_one() {
        let m = SpecialTensor::RealScalar(1.0);
        let result = eval_scalar(ellipe(&m, RuntimeMode::Strict));
        assert_close(result, 1.0, 1e-12, "E(1) = 1");
    }

    #[test]
    fn ellipe_at_half() {
        // E(0.5) ≈ 1.350_643_881
        let m = SpecialTensor::RealScalar(0.5);
        let result = eval_scalar(ellipe(&m, RuntimeMode::Strict));
        assert_close(result, 1.350_643_881, 1e-8, "E(0.5)");
    }

    #[test]
    fn ellipk_ellipe_legendre_relation() {
        // Legendre's relation: K(m)*E(1-m) + E(m)*K(1-m) - K(m)*K(1-m) = π/2
        let m = 0.3;
        let m1 = 1.0 - m;
        let km = eval_scalar(ellipk(&SpecialTensor::RealScalar(m), RuntimeMode::Strict));
        let em = eval_scalar(ellipe(&SpecialTensor::RealScalar(m), RuntimeMode::Strict));
        let km1 = eval_scalar(ellipk(&SpecialTensor::RealScalar(m1), RuntimeMode::Strict));
        let em1 = eval_scalar(ellipe(&SpecialTensor::RealScalar(m1), RuntimeMode::Strict));
        let legendre = km * em1 + em * km1 - km * km1;
        assert_close(legendre, PI / 2.0, 1e-6, "Legendre relation");
    }

    // ── Incomplete elliptic integrals ─────────────────────────────

    #[test]
    fn ellipkinc_is_scipys_cephes_ellik_bit_for_bit() -> Result<(), SpecialError> {
        // frankenscipy-zw56i: ellipkinc is xsf's Cephes ellik. SciPy 1.17.1's bits through the
        // Landen iteration, m -> 1 (the old R_F form was 1.0e-10 off at the first point), m = 1,
        // the amplitude transformation, |phi| past pi/2, m < 0 (series, Carlson and asymptotic
        // branches) and m = 0, which is phi exactly. (phi, m, scipy.special.ellipkinc).
        let cases: [(f64, f64, f64); 14] = [
            (1.5708642874071383, 0.9999999709353826, 10.451933845280488),
            (1.0, 0.5, 1.0832167728451687),
            (1.5707963267948966, 0.5, 1.8540746773013719),
            (2.0, 0.3, 2.220590552128474),
            (-2.5, 0.7, -3.4768751906448916),
            (1.4, 0.999999999, 2.45799558245953),
            (1.4, 1.0, 2.4579955903729784),
            (1.0, 0.0, 1.0),
            (20.0, 0.9, 32.39944907100371),
            (0.5, -3.0, 0.45396297924155427),
            (1.2, -1e9, 0.0003594981570130712),
            (1e-200, -5.0, 1e-200),
            (1.5, -1e5, 0.02236326885456018),
            (0.3, 0.99999999999, 0.3046039744016562),
        ];
        for (phi, m, want) in cases {
            let got = ellipkinc_scalar(
                std::hint::black_box(phi),
                std::hint::black_box(m),
                RuntimeMode::Strict,
            )?;
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "ellipkinc({phi:?}, {m:?}) = {got:?}, SciPy {want:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn ellipkinc_at_pi_half_equals_complete() {
        let phi = SpecialTensor::RealScalar(PI / 2.0);
        let m = SpecialTensor::RealScalar(0.5);
        let incomplete = eval_scalar(ellipkinc(&phi, &m, RuntimeMode::Strict));
        let complete = eval_scalar(ellipk(&m, RuntimeMode::Strict));
        assert_close(incomplete, complete, 1e-6, "F(π/2, m) = K(m)");
    }

    #[test]
    fn ellipeinc_at_pi_half_equals_complete() {
        let phi = SpecialTensor::RealScalar(PI / 2.0);
        let m = SpecialTensor::RealScalar(0.5);
        let incomplete = eval_scalar(ellipeinc(&phi, &m, RuntimeMode::Strict));
        let complete = eval_scalar(ellipe(&m, RuntimeMode::Strict));
        assert_close(incomplete, complete, 1e-6, "E(π/2, m) = E(m)");
    }

    #[test]
    fn ellipkinc_at_zero_phi() {
        let phi = SpecialTensor::RealScalar(0.0);
        let m = SpecialTensor::RealScalar(0.5);
        let result = eval_scalar(ellipkinc(&phi, &m, RuntimeMode::Strict));
        assert_close(result, 0.0, 1e-12, "F(0, m) = 0");
    }

    #[test]
    fn incomplete_elliptic_m_zero_is_phi() {
        // F(phi | 0) = E(phi | 0) = phi, which Cephes' ellik and ellie return as is. This test
        // used to pin 0.5*phi*1.9999999999999998, the Gauss-Legendre weight sum, which was
        // 1 ulp below SciPy (frankenscipy-zw56i).
        for phi in [PI / 6.0, PI / 4.0, PI / 3.0, PI / 2.0 - 0.1] {
            let expected = phi;
            let phi_tensor = SpecialTensor::RealScalar(phi);
            let m = SpecialTensor::RealScalar(0.0);
            let kinc = eval_scalar(ellipkinc(&phi_tensor, &m, RuntimeMode::Strict));
            let einc = eval_scalar(ellipeinc(&phi_tensor, &m, RuntimeMode::Strict));
            assert_eq!(kinc.to_bits(), expected.to_bits(), "F(phi, 0) bits");
            assert_eq!(einc.to_bits(), expected.to_bits(), "E(phi, 0) bits");
        }

        let phi = SpecialTensor::RealScalar(PI / 2.0);
        let m = SpecialTensor::RealScalar(0.0);
        let kinc = eval_scalar(ellipkinc(&phi, &m, RuntimeMode::Strict));
        let einc = eval_scalar(ellipeinc(&phi, &m, RuntimeMode::Strict));
        assert_eq!(kinc.to_bits(), (PI / 2.0).to_bits());
        assert_eq!(einc.to_bits(), (PI / 2.0).to_bits());
    }

    #[test]
    fn hardened_ellipkinc_rejects_unit_parameter() {
        let phi = SpecialTensor::RealScalar(PI / 4.0);
        let m = SpecialTensor::RealScalar(1.0);
        let err = ellipkinc(&phi, &m, RuntimeMode::Hardened)
            .expect_err("hardened rejects singular incomplete K parameter");
        assert_eq!(err.kind, SpecialErrorKind::DomainError);
    }

    #[test]
    fn ellipkinc_broadcasts_scalar_phi_over_m_vector() {
        let phi = SpecialTensor::RealScalar(PI / 2.0);
        let m = SpecialTensor::RealVec(vec![0.0, 0.5]);
        let result =
            ellipkinc(&phi, &m, RuntimeMode::Strict).expect("scalar phi should broadcast over m");
        match result {
            SpecialTensor::RealVec(values) => {
                assert_eq!(values.len(), 2);
                assert_close(values[0], PI / 2.0, 1e-12, "F(π/2, 0) = K(0)");
                assert_close(values[1], 1.854_074_677, 1e-6, "F(π/2, 0.5) = K(0.5)");
            }
            other => assert!(
                matches!(&other, SpecialTensor::RealVec(_)),
                "expected RealVec, got {other:?}"
            ),
        }
    }

    #[test]
    fn ellipkinc_broadcasts_vector_phi_over_scalar_m() {
        let phi = SpecialTensor::RealVec(vec![0.0, PI / 2.0]);
        let m = SpecialTensor::RealScalar(0.5);
        let result = ellipkinc(&phi, &m, RuntimeMode::Strict)
            .expect("vector phi should broadcast over scalar m");
        match result {
            SpecialTensor::RealVec(values) => {
                assert_eq!(values.len(), 2);
                assert_close(values[0], 0.0, 1e-12, "F(0, m) = 0");
                assert_close(values[1], 1.854_074_677, 1e-6, "F(π/2, 0.5) = K(0.5)");
            }
            other => assert!(
                matches!(&other, SpecialTensor::RealVec(_)),
                "expected RealVec, got {other:?}"
            ),
        }
    }

    #[test]
    fn ellipeinc_supports_pairwise_vector_inputs() {
        let phi = SpecialTensor::RealVec(vec![0.0, PI / 2.0]);
        let m = SpecialTensor::RealVec(vec![0.0, 0.5]);
        let result = ellipeinc(&phi, &m, RuntimeMode::Strict).expect("pairwise vector ellipeinc");
        match result {
            SpecialTensor::RealVec(values) => {
                assert_eq!(values.len(), 2);
                assert_close(values[0], 0.0, 1e-12, "E(0, 0) = 0");
                assert_close(values[1], 1.350_643_881, 1e-6, "E(π/2, 0.5) = E(0.5)");
            }
            other => assert!(
                matches!(&other, SpecialTensor::RealVec(_)),
                "expected RealVec, got {other:?}"
            ),
        }
    }

    #[test]
    fn ellipkinc_rejects_mismatched_vector_lengths() {
        let phi = SpecialTensor::RealVec(vec![0.0, PI / 2.0]);
        let m = SpecialTensor::RealVec(vec![0.5]);
        let err = ellipkinc(&phi, &m, RuntimeMode::Strict)
            .expect_err("mismatched vector lengths should error");
        assert_eq!(err.kind, SpecialErrorKind::DomainError);
    }

    #[test]
    fn ellipk_complex_reference_value() {
        let m = SpecialTensor::ComplexScalar(Complex64::new(0.5, 0.25));
        let result = eval_complex_scalar(ellipk(&m, RuntimeMode::Strict));
        assert_complex_close(
            result,
            Complex64::new(1.802_606_224_637_205_2, 0.194_528_590_856_802_02),
            1e-10,
            "complex ellipk reference",
        );
    }

    #[test]
    fn ellipe_complex_reference_value() {
        let m = SpecialTensor::ComplexScalar(Complex64::new(0.5, 0.25));
        let result = eval_complex_scalar(ellipe(&m, RuntimeMode::Strict));
        assert_complex_close(
            result,
            Complex64::new(1.360_868_682_163_129_7, -0.123_873_344_256_178_68),
            1e-10,
            "complex ellipe reference",
        );
    }

    #[test]
    fn ellipkinc_complex_reference_value() {
        let phi = SpecialTensor::ComplexScalar(Complex64::new(0.3, 0.4));
        let m = SpecialTensor::ComplexScalar(Complex64::new(0.5, 0.25));
        let result = eval_complex_scalar(ellipkinc(&phi, &m, RuntimeMode::Strict));
        assert_complex_close(
            result,
            Complex64::new(0.288_721_883_612_193_9, 0.398_838_834_080_766),
            1e-10,
            "complex ellipkinc reference",
        );
    }

    #[test]
    fn ellipeinc_complex_reference_value() {
        let phi = SpecialTensor::ComplexScalar(Complex64::new(0.3, 0.4));
        let m = SpecialTensor::ComplexScalar(Complex64::new(0.5, 0.25));
        let result = eval_complex_scalar(ellipeinc(&phi, &m, RuntimeMode::Strict));
        assert_complex_close(
            result,
            Complex64::new(0.311_621_477_340_673_9, 0.400_835_393_840_843_63),
            1e-10,
            "complex ellipeinc reference",
        );
    }

    #[test]
    fn elliptic_complex_real_axis_reduces_to_real_kernels() {
        let real_m = 0.2;
        let real_phi = 0.7;

        let k_real = eval_scalar(ellipk(
            &SpecialTensor::RealScalar(real_m),
            RuntimeMode::Strict,
        ));
        let k_complex = eval_complex_scalar(ellipk(
            &SpecialTensor::ComplexScalar(Complex64::from_real(real_m)),
            RuntimeMode::Strict,
        ));
        assert_complex_close(
            k_complex,
            Complex64::from_real(k_real),
            1e-12,
            "ellipk real axis",
        );

        let e_real = eval_scalar(ellipe(
            &SpecialTensor::RealScalar(real_m),
            RuntimeMode::Strict,
        ));
        let e_complex = eval_complex_scalar(ellipe(
            &SpecialTensor::ComplexScalar(Complex64::from_real(real_m)),
            RuntimeMode::Strict,
        ));
        assert_complex_close(
            e_complex,
            Complex64::from_real(e_real),
            1e-12,
            "ellipe real axis",
        );

        let f_real = eval_scalar(ellipkinc(
            &SpecialTensor::RealScalar(real_phi),
            &SpecialTensor::RealScalar(real_m),
            RuntimeMode::Strict,
        ));
        let f_complex = eval_complex_scalar(ellipkinc(
            &SpecialTensor::ComplexScalar(Complex64::from_real(real_phi)),
            &SpecialTensor::ComplexScalar(Complex64::from_real(real_m)),
            RuntimeMode::Strict,
        ));
        assert_complex_close(
            f_complex,
            Complex64::from_real(f_real),
            1e-12,
            "ellipkinc real axis",
        );

        let ei_real = eval_scalar(ellipeinc(
            &SpecialTensor::RealScalar(real_phi),
            &SpecialTensor::RealScalar(real_m),
            RuntimeMode::Strict,
        ));
        let ei_complex = eval_complex_scalar(ellipeinc(
            &SpecialTensor::ComplexScalar(Complex64::from_real(real_phi)),
            &SpecialTensor::ComplexScalar(Complex64::from_real(real_m)),
            RuntimeMode::Strict,
        ));
        assert_complex_close(
            ei_complex,
            Complex64::from_real(ei_real),
            1e-12,
            "ellipeinc real axis",
        );
    }

    #[test]
    fn ellipkinc_complex_broadcasts_scalar_phi_over_vector_m() {
        let phi = SpecialTensor::ComplexScalar(Complex64::new(0.3, 0.4));
        let z = Complex64::new(0.5, 0.25);
        let m = SpecialTensor::ComplexVec(vec![z, z.conj()]);
        let result = ellipkinc(&phi, &m, RuntimeMode::Strict).expect("complex broadcast");
        match result {
            SpecialTensor::ComplexVec(values) => {
                assert_eq!(values.len(), 2);
                assert_complex_close(
                    values[0],
                    Complex64::new(0.288_721_883_612_193_9, 0.398_838_834_080_766),
                    1e-10,
                    "broadcast first element",
                );
                assert_complex_close(
                    values[1],
                    Complex64::new(0.291_734_331_966_050_5, 0.408_643_099_789_330_1),
                    1e-10,
                    "broadcast second element",
                );
            }
            other => assert!(
                matches!(&other, SpecialTensor::ComplexVec(_)),
                "expected ComplexVec, got {other:?}"
            ),
        }
    }

    #[test]
    fn ellipkm1_complex_matches_ellipk_of_complement() {
        let p = SpecialTensor::ComplexScalar(Complex64::new(0.4, -0.2));
        let result = eval_complex_scalar(ellipkm1(&p, RuntimeMode::Strict));
        let expected = eval_complex_scalar(ellipk(
            &SpecialTensor::ComplexScalar(Complex64::new(0.6, 0.2)),
            RuntimeMode::Strict,
        ));
        assert_complex_close(result, expected, 1e-12, "ellipkm1 complex complement");
    }

    #[test]
    fn ellipkm1_complex_real_axis_reduces_to_real_kernel() {
        let p = 0.3;
        let real_result = eval_scalar(ellipkm1(&SpecialTensor::RealScalar(p), RuntimeMode::Strict));
        let complex_result = eval_complex_scalar(ellipkm1(
            &SpecialTensor::ComplexScalar(Complex64::from_real(p)),
            RuntimeMode::Strict,
        ));
        assert_complex_close(
            complex_result,
            Complex64::from_real(real_result),
            1e-12,
            "ellipkm1 real-axis reduction",
        );
    }

    #[test]
    fn ellipkm1_complex_broadcasts_vector_inputs() {
        let values = vec![Complex64::from_real(0.2), Complex64::new(0.4, -0.2)];
        let result = ellipkm1(
            &SpecialTensor::ComplexVec(values.clone()),
            RuntimeMode::Strict,
        )
        .expect("complex ellipkm1 vector broadcast");
        match result {
            SpecialTensor::ComplexVec(items) => {
                assert_eq!(items.len(), values.len());
                for (value, actual) in values.iter().zip(items.iter()) {
                    let expected = eval_complex_scalar(ellipkm1(
                        &SpecialTensor::ComplexScalar(*value),
                        RuntimeMode::Strict,
                    ));
                    assert_complex_close(*actual, expected, 1e-12, "ellipkm1 vector lane");
                }
            }
            other => assert!(
                matches!(&other, SpecialTensor::ComplexVec(_)),
                "expected ComplexVec, got {other:?}"
            ),
        }
    }

    // ── Lambert W function ────────────────────────────────────────

    #[test]
    fn lambertw_at_zero() {
        let x = SpecialTensor::RealScalar(0.0);
        let result = eval_scalar(lambertw(&x, RuntimeMode::Strict));
        assert_close(result, 0.0, 1e-12, "W(0) = 0");
    }

    #[test]
    fn lambertw_at_e() {
        let x = SpecialTensor::RealScalar(std::f64::consts::E);
        let result = eval_scalar(lambertw(&x, RuntimeMode::Strict));
        assert_close(result, 1.0, 1e-10, "W(e) = 1");
    }

    #[test]
    fn lambertw_at_positive_infinity() {
        let x = SpecialTensor::RealScalar(f64::INFINITY);
        let result = eval_scalar(lambertw(&x, RuntimeMode::Strict));
        assert_eq!(result, f64::INFINITY);
    }

    #[test]
    fn lambertw_at_neg_inv_e() {
        let x = SpecialTensor::RealScalar(-1.0 / std::f64::consts::E);
        let result = eval_scalar(lambertw(&x, RuntimeMode::Strict));
        assert_close(result, -1.0, 1e-10, "W(-1/e) = -1");
    }

    #[test]
    fn lambertw_identity() {
        // W(x) * exp(W(x)) = x
        for &x in &[0.5, 1.0, 2.0, 10.0, 100.0] {
            let w = eval_scalar(lambertw(&SpecialTensor::RealScalar(x), RuntimeMode::Strict));
            let check = w * w.exp();
            assert_close(check, x, 1e-10, &format!("W({x})*exp(W({x})) = {x}"));
        }
    }

    #[test]
    fn lambertw_domain_error_hardened() {
        let x = SpecialTensor::RealScalar(-1.0); // < -1/e
        let result = lambertw(&x, RuntimeMode::Hardened);
        assert!(result.is_err(), "should reject x < -1/e in hardened mode");
    }

    #[test]
    fn lambertw_complex_real_axis_reduces_to_real_kernel() {
        for x in [-1.0 / std::f64::consts::E, -0.2, 0.5, 3.0] {
            let real_result =
                eval_scalar(lambertw(&SpecialTensor::RealScalar(x), RuntimeMode::Strict));
            let complex_result = eval_complex_scalar(lambertw(
                &SpecialTensor::ComplexScalar(Complex64::from_real(x)),
                RuntimeMode::Strict,
            ));
            assert_complex_close(
                complex_result,
                Complex64::from_real(real_result),
                1e-12,
                "lambertw real-axis reduction",
            );
        }
    }

    #[test]
    fn lambertw_complex_identity_principal_branch() {
        let z = Complex64::new(0.5, 0.75);
        let w = eval_complex_scalar(lambertw(
            &SpecialTensor::ComplexScalar(z),
            RuntimeMode::Strict,
        ));
        assert_complex_close(w * w.exp(), z, 1e-10, "lambertw complex identity");
    }

    #[test]
    fn lambertw_complex_negative_real_uses_principal_branch() {
        let z = Complex64::from_real(-1.0);
        let w = eval_complex_scalar(lambertw(
            &SpecialTensor::ComplexScalar(z),
            RuntimeMode::Strict,
        ));
        assert!(
            w.im > 0.0,
            "principal branch should choose the upper-half-plane value on the cut",
        );
        assert_complex_close(w * w.exp(), z, 1e-10, "lambertw branch-cut identity");
    }

    #[test]
    fn lambertw_complex_matches_scipy_principal_branch() {
        // Regression: the W·e^W = z identity holds on EVERY branch, so the
        // metamorphic tests above passed while the principal-branch guess
        // (large-|z| asymptotic) converged to the wrong sheet — e.g.
        // lambertw(0.5+0.3i) returned -1.97-3.68i instead of scipy's
        // 0.372+0.152i. These reference values are scipy.special.lambertw(z, 0)
        // (1.17.1); they pin the branch, not just the defining equation.
        let cases: &[(f64, f64, f64, f64)] = &[
            (0.5, 0.3, 0.372_030_602_393_5, 0.152_166_010_103_015_42),
            (1.3, 0.7, 0.701_900_647_738_376_3, 0.207_068_190_438_305_32),
            (-1.5, 2.0, 0.640_576_739_989_03, 1.151_256_393_356_353_5),
            (2.0, -2.5, 1.035_286_384_300_418_5, -0.469_942_289_784_801_2),
            (-3.0, -1.0, 0.608_235_777_667_986_4, -1.610_213_135_169_645),
            (0.2, 4.0, 1.073_172_875_178_164, 0.850_615_061_208_774_6),
            (-0.4, 0.0, -0.944_089_738_264_935_8, 0.407_267_964_032_857_8),
            (0.0, 1.5, 0.545_153_223_345_237_3, 0.677_542_092_022_338_5),
            (
                10.0,
                -3.0,
                1.769_471_344_173_386_3,
                -0.186_465_225_817_075_56,
            ),
        ];
        for &(re, im, wre, wim) in cases {
            let w = eval_complex_scalar(lambertw(
                &SpecialTensor::ComplexScalar(Complex64::new(re, im)),
                RuntimeMode::Strict,
            ));
            let denom = wre.hypot(wim).max(1.0);
            let err = (w.re - wre).hypot(w.im - wim) / denom;
            assert!(
                err < 1e-12,
                "lambertw({re}+{im}i) = {}+{}i, scipy {wre}+{wim}i (relerr {err:e})",
                w.re,
                w.im
            );
        }
    }

    #[test]
    fn lambertw_complex_vector_preserves_shape_and_conjugation() {
        let z = Complex64::new(0.4, 0.7);
        let result = lambertw(
            &SpecialTensor::ComplexVec(vec![z, z.conj()]),
            RuntimeMode::Strict,
        )
        .expect("complex vector lambertw");
        match result {
            SpecialTensor::ComplexVec(values) => {
                assert_eq!(values.len(), 2);
                assert_complex_close(values[1], values[0].conj(), 1e-12, "lambertw conjugation");
                let scalar = eval_complex_scalar(lambertw(
                    &SpecialTensor::ComplexScalar(z),
                    RuntimeMode::Strict,
                ));
                assert_complex_close(
                    values[0],
                    scalar,
                    1e-12,
                    "lambertw vector lane matches scalar",
                );
            }
            other => assert!(
                matches!(&other, SpecialTensor::ComplexVec(_)),
                "expected ComplexVec, got {other:?}"
            ),
        }
    }

    // ── Exponential integrals ─────────────────────────────────────

    #[test]
    fn exp1_known_value() {
        // E1(1) ≈ 0.219_383_934_4
        let z = SpecialTensor::RealScalar(1.0);
        let result = eval_scalar(exp1(&z, RuntimeMode::Strict));
        assert_close(result, 0.219_383_934_4, 1e-8, "E1(1)");
    }

    #[test]
    fn exp1_large_z() {
        // E1(20) ≈ 9.8355e-11 (verified against Wolfram Alpha)
        let z = 20.0;
        let result = eval_scalar(exp1(&SpecialTensor::RealScalar(z), RuntimeMode::Strict));
        // The asymptotic series exp(-z)/z is only approximate; use known reference
        assert_close(result, 9.835_525_290_7e-11, 1e-15, "E1(20)");
    }

    #[test]
    fn exp1_real_zero_is_positive_infinity() {
        let result = eval_scalar(exp1(&SpecialTensor::RealScalar(0.0), RuntimeMode::Strict));
        assert!(
            result.is_infinite() && result.is_sign_positive(),
            "E1(0) should match scipy.special.exp1(0) = +inf"
        );
    }

    #[test]
    fn expi_known_value() {
        // Ei(1) ≈ 1.895_117_816_4
        let x = SpecialTensor::RealScalar(1.0);
        let result = eval_scalar(expi(&x, RuntimeMode::Strict));
        assert_close(result, 1.895_117_816_4, 1e-6, "Ei(1)");
    }

    /// Exact `scipy.special.expi` goldens across the real line (frankenscipy-waagw).
    ///
    /// The pre-existing coverage was one point (x=1) at 1e-6 plus internal
    /// consistency checks — real-axis reduction, the Si/Ci identity, conjugation
    /// — which pin fsci against ITSELF and would all survive a systematically
    /// wrong Ei. This is the first test that pins the value against SciPy.
    ///
    /// Compared RELATIVELY. `assert_close` above is an absolute bound, which is
    /// meaningless over a range that spans |Ei| from 9.8e-11 at x=-20 to 1.1e20
    /// at x=50: a 1e-13 absolute bound is unsatisfiable at the top and
    /// vacuously true at the bottom.
    ///
    /// Reference values from scipy.special.expi, SciPy 1.17.1.
    #[test]
    fn expi_matches_scipy_reference_values() {
        // (x, scipy.special.expi(x))
        let cases = [
            (-20.0, -9.835_525_290_649_882e-11),
            (-10.0, -4.156_968_929_685_325e-6),
            (-2.0, -0.048_900_510_708_061_125),
            (-0.5, -0.559_773_594_776_160_8),
            (-0.1, -1.822_923_958_419_390_6),
            (-0.01, -4.037_929_576_538_113),
            (0.01, -4.017_929_465_426_669),
            (0.1, -1.622_812_813_969_276_6),
            (0.5, 0.454_219_904_863_173_54),
            (1.0, 1.895_117_816_355_937),
            (2.0, 4.954_234_356_001_891),
            (5.0, 40.185_275_355_803_17),
            (10.0, 2_492.228_976_241_877_3),
            (20.0, 25_615_652.664_056_595),
            (50.0, 1.058_563_689_713_169e20),
        ];
        const REL_TOL: f64 = 1.0e-13;
        for (x, expected) in cases {
            let got = eval_scalar(expi(&SpecialTensor::RealScalar(x), RuntimeMode::Strict));
            let rel = (got - expected).abs() / expected.abs();
            assert!(
                rel <= REL_TOL,
                "Ei({x}): got {got}, scipy {expected} (rel {rel:.3e} > {REL_TOL:.0e})"
            );
        }

        // The real root of Ei, where SciPy itself returns 6.1e-16 rather than 0.
        // A relative check is meaningless here, so bound it absolutely: the
        // cancellation near the root must not leave a macroscopic residue.
        let root = 0.372_507_410_781_366_8;
        let at_root = eval_scalar(expi(&SpecialTensor::RealScalar(root), RuntimeMode::Strict));
        assert!(
            at_root.abs() < 1.0e-14,
            "Ei(root={root}) should vanish, got {at_root}"
        );
    }

    #[test]
    fn expi_at_zero_is_neg_inf() {
        let x = SpecialTensor::RealScalar(0.0);
        let result = eval_scalar(expi(&x, RuntimeMode::Strict));
        assert!(
            result.is_infinite() && result.is_sign_negative(),
            "Ei(0) = -inf"
        );
    }

    #[test]
    fn expi_complex_real_axis_reduces_to_real_kernel() {
        for x in [-2.0, -0.5, 1.0, 3.0] {
            let real_result = eval_scalar(expi(&SpecialTensor::RealScalar(x), RuntimeMode::Strict));
            let complex_result = eval_complex_scalar(expi(
                &SpecialTensor::ComplexScalar(Complex64::from_real(x)),
                RuntimeMode::Strict,
            ));
            assert_complex_close(
                complex_result,
                Complex64::from_real(real_result),
                1e-12,
                "expi real-axis reduction",
            );
        }
    }

    #[test]
    fn expi_complex_imaginary_axis_matches_sici_identity() {
        let (si, ci) = crate::convenience::sici(1.0);
        let result = eval_complex_scalar(expi(
            &SpecialTensor::ComplexScalar(Complex64::new(0.0, 1.0)),
            RuntimeMode::Strict,
        ));
        assert_complex_close(
            result,
            Complex64::new(ci, si + std::f64::consts::FRAC_PI_2),
            1e-10,
            "Ei(i) matches Ci(1) + i*(Si(1)+π/2)",
        );
    }

    #[test]
    fn expi_complex_vector_preserves_shape_and_conjugation() {
        let z = Complex64::new(1.0, 1.0);
        let result = expi(
            &SpecialTensor::ComplexVec(vec![z, z.conj()]),
            RuntimeMode::Strict,
        )
        .expect("complex vector expi");
        match result {
            SpecialTensor::ComplexVec(values) => {
                assert_eq!(values.len(), 2);
                assert_complex_close(values[1], values[0].conj(), 1e-12, "expi conjugation");
                let scalar = eval_complex_scalar(expi(
                    &SpecialTensor::ComplexScalar(z),
                    RuntimeMode::Strict,
                ));
                assert_complex_close(values[0], scalar, 1e-12, "expi vector lane matches scalar");
            }
            other => assert!(
                matches!(&other, SpecialTensor::ComplexVec(_)),
                "expected ComplexVec, got {other:?}"
            ),
        }
    }

    #[test]
    fn exp1_domain_error_hardened() {
        let z = SpecialTensor::RealScalar(-1.0);
        let result = exp1(&z, RuntimeMode::Hardened);
        assert!(result.is_err(), "should reject z <= 0 in hardened mode");
    }

    #[test]
    fn exp1_complex_positive_real_axis_reduces_to_real_kernel() {
        let x = 0.7;
        let real_result = eval_scalar(exp1(&SpecialTensor::RealScalar(x), RuntimeMode::Strict));
        let complex_result = eval_complex_scalar(exp1(
            &SpecialTensor::ComplexScalar(Complex64::from_real(x)),
            RuntimeMode::Strict,
        ));
        assert_complex_close(
            complex_result,
            Complex64::from_real(real_result),
            1e-12,
            "exp1 real-axis reduction",
        );
    }

    #[test]
    fn exp1_complex_negative_real_uses_principal_branch() {
        let x = 3.0;
        let result = eval_complex_scalar(exp1(
            &SpecialTensor::ComplexScalar(Complex64::from_real(-x)),
            RuntimeMode::Strict,
        ));
        let expected_real = -expi_scalar(x, RuntimeMode::Strict).expect("Ei(x) should succeed");
        assert_complex_close(
            result,
            Complex64::new(expected_real, -PI),
            1e-12,
            "exp1 principal branch on negative real axis",
        );
    }

    #[test]
    fn exp1_complex_negative_real_signed_zero_selects_lower_branch() {
        let x = 3.0;
        let result = eval_complex_scalar(exp1(
            &SpecialTensor::ComplexScalar(Complex64::new(-x, -0.0)),
            RuntimeMode::Strict,
        ));
        let expected_real = -expi_scalar(x, RuntimeMode::Strict).expect("Ei(x) should succeed");
        assert_complex_close(
            result,
            Complex64::new(expected_real, PI),
            1e-12,
            "exp1 lower branch on negative real axis",
        );
        assert!(result.im.is_sign_positive());
    }

    #[test]
    fn exp1_complex_vector_preserves_shape_and_conjugation() {
        let z = Complex64::new(1.0, 1.0);
        let result = exp1(
            &SpecialTensor::ComplexVec(vec![z, z.conj()]),
            RuntimeMode::Strict,
        )
        .expect("complex vector exp1");
        match result {
            SpecialTensor::ComplexVec(values) => {
                assert_eq!(values.len(), 2);
                assert_complex_close(values[1], values[0].conj(), 1e-12, "exp1 conjugation");
                let scalar = eval_complex_scalar(exp1(
                    &SpecialTensor::ComplexScalar(z),
                    RuntimeMode::Strict,
                ));
                assert_complex_close(values[0], scalar, 1e-12, "exp1 vector lane matches scalar");
            }
            other => assert!(
                matches!(&other, SpecialTensor::ComplexVec(_)),
                "expected ComplexVec, got {other:?}"
            ),
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy
    fn exp1_complex_off_axis_matches_scipy() {
        // frankenscipy-ey069: the old |z|<=2 series/CF split sent the whole |z|>2 left
        // half-plane to the continued fraction, which is the asymptotic expansion and
        // diverges (~5e-3) away from large |z| — e.g. exp1(-3-1j) was off by ~8.6e-6.
        // The Zhang-Jin region rule (series for |z|<=10 or deep left half |z|<20) fixes it.
        // (z_re, z_im, exp1.re, exp1.im) — scipy 1.17.1.
        let cases: [(f64, f64, f64, f64); 8] = [
            (-3.0, -1.0, -7.8231346760015779e+00, -2.9559271304025110e+00),
            (-5.0, 0.5, -3.7262468961367944e+01, 1.1283268496460263e+01),
            (-3.0, 2.0, -2.8074890821669083e+00, 5.9603353047969971e+00),
            (-8.0, 1.0, -2.8803230551285594e+02, 3.2292700501048904e+02),
            (-1.0, -2.0, -1.0421677081649352e+00, -5.5990877234808067e-01),
            (-15.0, 3.0, 2.1516189796411578e+05, 7.9849102715220142e+04),
            (3.0, 2.0, -9.0959208747944942e-03, -6.9001792622122027e-03),
            (0.5, 0.3, 4.2221132422501800e-01, -3.0537113617426670e-01),
        ];
        for (zr, zi, er, ei) in cases {
            let c = eval_complex_scalar(exp1(
                &SpecialTensor::ComplexScalar(Complex64::new(zr, zi)),
                RuntimeMode::Strict,
            ));
            let rel = (c.re - er).hypot(c.im - ei) / er.hypot(ei);
            assert!(
                rel <= 1e-9,
                "exp1({zr}{zi:+}i) = {c:?}, scipy ({er},{ei}), rel {rel:e}"
            );
        }
    }

    // ── Vector inputs ─────────────────────────────────────────────

    #[test]
    fn ellipk_vector_input() {
        let m = SpecialTensor::RealVec(vec![0.0, 0.5]);
        let result = ellipk(&m, RuntimeMode::Strict).expect("should succeed");
        match result {
            SpecialTensor::RealVec(values) => {
                assert_eq!(values.len(), 2);
                assert_close(values[0], PI / 2.0, 1e-12, "K(0)");
                assert_close(values[1], 1.854_074_677, 1e-8, "K(0.5)");
            }
            other => assert!(
                matches!(&other, SpecialTensor::RealVec(_)),
                "expected RealVec, got {other:?}"
            ),
        }
    }

    #[test]
    fn lambertw_vector_input() {
        let x = SpecialTensor::RealVec(vec![0.0, std::f64::consts::E]);
        let result = lambertw(&x, RuntimeMode::Strict).expect("should succeed");
        match result {
            SpecialTensor::RealVec(values) => {
                assert_eq!(values.len(), 2);
                assert_close(values[0], 0.0, 1e-12, "W(0)");
                assert_close(values[1], 1.0, 1e-10, "W(e)");
            }
            other => assert!(
                matches!(&other, SpecialTensor::RealVec(_)),
                "expected RealVec, got {other:?}"
            ),
        }
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

    // ── Jacobi elliptic functions ────────────────────────────────────

    #[test]
    fn ellipj_at_zero() {
        // sn(0, m) = 0, cn(0, m) = 1, dn(0, m) = 1
        let (sn, cn, dn, ph) = ellipj(0.0, 0.5);
        assert_close(sn, 0.0, 1e-12, "sn(0) = 0");
        assert_close(cn, 1.0, 1e-12, "cn(0) = 1");
        assert_close(dn, 1.0, 1e-12, "dn(0) = 1");
        assert_close(ph, 0.0, 1e-12, "am(0) = 0");
    }

    #[test]
    fn ellipj_m_zero_is_trig() {
        // m=0: sn = sin, cn = cos, dn = 1
        let u = 1.0;
        let (sn, cn, dn, _) = ellipj(u, 0.0);
        assert_close(sn, u.sin(), 1e-12, "sn(u,0) = sin(u)");
        assert_close(cn, u.cos(), 1e-12, "cn(u,0) = cos(u)");
        assert_close(dn, 1.0, 1e-12, "dn(u,0) = 1");
    }

    #[test]
    fn ellipj_m_one_is_hyp() {
        // m=1: sn = tanh, cn = dn = sech
        let u = 1.0;
        let (sn, cn, dn, _) = ellipj(u, 1.0);
        assert_close(sn, u.tanh(), 1e-12, "sn(u,1) = tanh(u)");
        assert_close(cn, 1.0 / u.cosh(), 1e-12, "cn(u,1) = sech(u)");
        assert_close(dn, 1.0 / u.cosh(), 1e-12, "dn(u,1) = sech(u)");
    }

    #[test]
    fn ellipj_pythagorean_identity() {
        // sn² + cn² = 1
        for m in [0.1, 0.3, 0.5, 0.7, 0.9] {
            for u in [0.5, 1.0, 2.0, 3.0] {
                let (sn, cn, _, _) = ellipj(u, m);
                let sum = sn * sn + cn * cn;
                assert!(
                    (sum - 1.0).abs() < 1e-10,
                    "sn²+cn²=1 failed: u={u}, m={m}, sum={sum}"
                );
            }
        }
    }

    #[test]
    fn ellipj_dn_identity() {
        // dn² + m*sn² = 1
        for m in [0.1, 0.3, 0.5, 0.7, 0.9] {
            for u in [0.5, 1.0, 2.0, 3.0] {
                let (sn, _, dn, _) = ellipj(u, m);
                let sum = dn * dn + m * sn * sn;
                assert!(
                    (sum - 1.0).abs() < 1e-10,
                    "dn²+m·sn²=1 failed: u={u}, m={m}, sum={sum}"
                );
            }
        }
    }

    #[test]
    fn ellipj_nan_passthrough() {
        let (sn, cn, dn, ph) = ellipj(f64::NAN, 0.5);
        assert!(sn.is_nan());
        assert!(cn.is_nan());
        assert!(dn.is_nan());
        assert!(ph.is_nan());
    }

    // ── Generalized exponential integral E_n ──────────────────────────

    #[test]
    fn expn_special_cases() {
        // E_0(x) = exp(-x)/x
        assert_close(expn_scalar(0, 1.0), (-1.0_f64).exp(), 1e-12, "E_0(1)");
        assert_close(expn_scalar(0, 2.0), (-2.0_f64).exp() / 2.0, 1e-12, "E_0(2)");

        // E_1(1) ≈ 0.219_383_934_4 (matches exp1)
        assert_close(expn_scalar(1, 1.0), 0.219_383_934_4, 1e-8, "E_1(1)");

        // E_n(0) = 1/(n-1) for n >= 2
        assert_close(expn_scalar(2, 0.0), 1.0, 1e-12, "E_2(0) = 1");
        assert_close(expn_scalar(3, 0.0), 0.5, 1e-12, "E_3(0) = 0.5");
        assert_close(expn_scalar(4, 0.0), 1.0 / 3.0, 1e-12, "E_4(0) = 1/3");
    }

    #[test]
    fn expn_known_values() {
        // Values verified against SciPy/Wolfram Alpha
        // E_2(1) ≈ 0.148_495_506_8
        assert_close(expn_scalar(2, 1.0), 0.148_495_506_8, 1e-6, "E_2(1)");

        // E_3(1) ≈ 0.109_691_632_2
        assert_close(expn_scalar(3, 1.0), 0.109_691_632_2, 1e-6, "E_3(1)");

        // E_2(0.5) ≈ 0.326_643_070_9
        assert_close(expn_scalar(2, 0.5), 0.326_643_070_9, 1e-6, "E_2(0.5)");
    }

    #[test]
    fn expn_large_x() {
        // For large x, E_n(x) ≈ exp(-x)/x
        let x = 10.0_f64;
        let asymp = (-x).exp() / x;
        // E_1(10) should be close to but slightly larger than asymptotic
        let e1 = expn_scalar(1, x);
        assert!(
            e1 > 0.9 * asymp && e1 < 1.5 * asymp,
            "E_1(10) near asymptotic"
        );
    }

    #[test]
    fn expn_recurrence() {
        // Recurrence: E_{n+1}(x) = (exp(-x) - x*E_n(x)) / n for n >= 1
        let x = 2.0;
        for n in 1..5 {
            let en = expn_scalar(n, x);
            let en1 = expn_scalar(n + 1, x);
            let computed = ((-x).exp() - x * en) / n as f64;
            assert_close(en1, computed, 1e-6, &format!("E_{} recurrence", n + 1));
        }
    }

    #[test]
    fn ellipkm1_basic() {
        use super::*;

        // ellipkm1(p) = ellipk(1-p)
        // ellipkm1(1) = ellipk(0) = π/2
        let result = ellipkm1_scalar(1.0, RuntimeMode::Strict).unwrap();
        assert_close(result, std::f64::consts::PI / 2.0, 1e-10, "ellipkm1(1)");

        // ellipkm1(0.5) = ellipk(0.5) ≈ 1.854
        let result = ellipkm1_scalar(0.5, RuntimeMode::Strict).unwrap();
        assert_close(result, 1.854, 0.001, "ellipkm1(0.5)");

        // ellipkm1(0) = ellipk(1) = infinity
        let result = ellipkm1_scalar(0.0, RuntimeMode::Strict).unwrap();
        assert!(result.is_infinite() && result.is_sign_positive());
    }

    #[test]
    fn ellipkm1_accepts_p_greater_than_one_match_scipy() {
        // Regression (frankenscipy-35272): ellipkm1(p) = K(1-p) is defined for
        // every p >= 0, so p > 1 (negative m) is finite, not a domain error. The
        // old [0,1] guard returned NaN. Values from scipy.special.ellipkm1 1.17.1.
        let cases = [
            (1.5_f64, 1.415_737_208_425_956_f64),
            (2.0, 1.311_028_777_146_059_8),
            (5.0, 1.009_452_909_989_211_3),
            (10.0, 0.815_264_309_589_721_3),
            (100.0, 0.369_563_736_298_987_5),
        ];
        for (p, want) in cases {
            let got = ellipkm1_scalar(p, RuntimeMode::Strict).unwrap();
            assert!(
                (got - want).abs() <= 1e-9 * want.abs().max(1.0),
                "ellipkm1({p}) = {got}, want {want}"
            );
            // identity ellipkm1(p) == ellipk(1-p)
            let k = ellipk_scalar(1.0 - p, RuntimeMode::Strict).unwrap();
            assert!(
                (got - k).abs() <= 1e-12 * k.abs().max(1.0),
                "ellipkm1({p}) != ellipk(1-{p})"
            );
        }
        // p < 0 (m > 1) is outside the real domain -> NaN, matching scipy.
        assert!(ellipkm1_scalar(-1.0, RuntimeMode::Strict).unwrap().is_nan());
    }

    #[test]
    fn ellipkm1_is_scipys_ellpk_bit_for_bit() -> Result<(), SpecialError> {
        // frankenscipy-36gsc: ellipkm1 is Cephes ellpk(p), SciPy's kernel. The old AGM kept
        // only ln(4/sqrt p) below p = 1e-10 (1e-11 off at 4.46e-11) and ran its own AGM above.
        // (p, scipy.special.ellipkm1(p)).
        let cases: [(f64, f64); 14] = [
            (1e-300, 346.77405831022674),
            (1e-100, 116.51554901082218),
            (1e-17, 20.95826765156928),
            (4.462405696195516e-11, 13.302668365535762),
            (1e-10, 12.8992198263876),
            (3e-10, 12.349913682607308),
            (1e-05, 7.142772450581779),
            (0.1, 2.5780921133481733),
            (0.5, 1.8540746773013719),
            (1.0, 1.5707963267948966),
            (2.0, 1.3110287771460598),
            (10000000000.0, 0.000128992198263876),
            (f64::INFINITY, 0.0),
            (0.0, f64::INFINITY),
        ];
        for (p, want) in cases {
            let got = ellipkm1_scalar(std::hint::black_box(p), RuntimeMode::Strict)?;
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "ellipkm1({p:?}) = {got:?}, SciPy {want:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn ellipkm1_consistency_with_ellipk() {
        use super::*;

        // For moderate values, ellipkm1(p) should equal ellipk(1-p)
        for &p in &[0.1, 0.3, 0.5, 0.7, 0.9] {
            let k1 = ellipkm1_scalar(p, RuntimeMode::Strict).unwrap();
            let k2 = ellipk_scalar(1.0 - p, RuntimeMode::Strict).unwrap();
            assert_close(k1, k2, 1e-10, &format!("ellipkm1({p}) vs ellipk(1-{p})"));
        }
    }

    #[test]
    fn elliprc_x_equals_y_is_one_over_sqrt_x() {
        // /porting-to-rust + /testing-golden-artifacts for
        // [frankenscipy-mxxij]: closed form RC(x, x) = 1/√x.
        for &x in &[0.5_f64, 1.0, 2.0, 4.0, 9.0, 25.0] {
            let actual = elliprc(x, x);
            let expected = 1.0 / x.sqrt();
            assert_close(
                actual,
                expected,
                1e-12,
                &format!("RC({x}, {x}) = {actual}, expected {expected}"),
            );
        }
    }

    #[test]
    fn elliprc_x_zero_y_positive_is_pi_over_2_sqrt_y() {
        // RC(0, y) = arccos(0) / √y = (π/2) / √y for y > 0.
        for &y in &[0.5_f64, 1.0, 4.0, 16.0] {
            let expected = std::f64::consts::FRAC_PI_2 / y.sqrt();
            let actual = elliprc(0.0, y);
            assert_close(
                actual,
                expected,
                1e-12,
                &format!("RC(0, {y}) = {actual}, expected π/(2√{y}) = {expected}"),
            );
        }
    }

    #[test]
    fn elliprc_x_greater_than_y_uses_acosh_branch() {
        // For x > y > 0: RC(x, y) = arccosh(√(x/y)) / √(x − y).
        // Pin RC(2, 1) = arccosh(√2) / 1 = arccosh(√2).
        let r = elliprc(2.0, 1.0);
        let expected = (2.0_f64.sqrt()).acosh();
        assert_close(r, expected, 1e-12, "RC(2, 1) = arccosh(√2)");

        // RC(4, 1) = arccosh(2) / √3.
        let r = elliprc(4.0, 1.0);
        let expected = 2.0_f64.acosh() / 3.0_f64.sqrt();
        assert_close(r, expected, 1e-12, "RC(4, 1)");
    }

    #[test]
    fn elliprc_negative_x_or_nan_returns_nan() {
        assert!(elliprc(-1.0, 1.0).is_nan());
        assert!(elliprc(f64::NAN, 1.0).is_nan());
        assert!(elliprc(1.0, f64::NAN).is_nan());
        // y=0 is outside scipy.special.elliprc's domain → NaN (frankenscipy-rmrmx).
        assert!(elliprc(1.0, 0.0).is_nan());
        assert!(elliprc(0.0, 0.0).is_nan());
    }

    #[test]
    fn elliprf_is_scipys_carlson_duplication_bit_for_bit() {
        // frankenscipy-mo0yq: RF is SciPy 1.17.1's ellint_carlson::rf. A Python emulation of
        // _rf.hh matched scipy.special.elliprf on 90,008 of 90,008 points. The old kernel
        // stopped after 32 duplications, so the 1e257 and 1e300 spreads came back as garbage.
        // The zero and subnormal smallest arguments take the AGM route. The last two points
        // are ones where an uncompensated dot/sum gives other bits. (x, y, z, SciPy).
        let cases: [(f64, f64, f64, f64); 15] = [
            (1.0, 2.0, 3.0, 0.7269459354689082),
            (0.5, 1.5, 4.0, 0.7763737038752115),
            (1.0, 1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0, 1.5707963267948966),
            (0.0, 2.0, 4.0, 0.9270373386506858),
            (
                8.960983117050649,
                5.910412268763193,
                7.609714370453767e257,
                3.3999741139146905e-127,
            ),
            (1.0, 2.0, 1e300, 3.458926847232072e-148),
            (1e300, 1e300, 1e300, 1e-150),
            (1e-300, 1e-300, 1e-300, 1e150),
            (5e-324, 1.0, 2.0, 1.3110287771460598),
            (1e-200, 3.0, 7.0, 0.7256311852272992),
            (1e-08, 0.0001, 10000.0, 0.10586684426237014),
            (3.0, 5.0, 7.0, 0.454895415591073),
            (
                8.069528945079265,
                8.09861381839129,
                5.201723054317205,
                0.37691120417664714,
            ),
            (
                2.787370884848005,
                8.808546616015729,
                0.735722929390691,
                0.5593580230850238,
            ),
        ];
        for (x, y, z, want) in cases {
            let got = elliprf(
                std::hint::black_box(x),
                std::hint::black_box(y),
                std::hint::black_box(z),
            );
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "elliprf({x:?}, {y:?}, {z:?}) = {got:?}, SciPy {want:?}"
            );
        }
        // SciPy's domain edges: two zeros are +inf, an infinite argument 0, a negative one NaN.
        assert_eq!(elliprf(0.0, 0.0, 1.0), f64::INFINITY);
        assert_eq!(
            elliprf(f64::INFINITY, 1.0, 1.0).to_bits(),
            0.0_f64.to_bits()
        );
        assert!(elliprf(-1.0, 1.0, 1.0).is_nan());
    }

    #[test]
    fn elliprc_is_scipys_carlson_duplication_bit_for_bit() {
        // frankenscipy-2f8h7: RC is SciPy 1.17.1's ellint_carlson::rc, so its bits are pinned.
        // A Python emulation of _rc.hh matched scipy.special.elliprc on 110,013 of 110,013
        // points. The closed forms it replaced miss most of these: near the diagonal, where
        // elliprj's duplication calls RC; at tiny arguments, where an absolute diagonal test
        // snapped to 1/sqrt(x); and at (1e300, 1e-300), where x/y overflowed. The last point
        // is the one in that sweep where an uncompensated Horner gives different bits.
        // (x, y, scipy.special.elliprc(x, y)).
        let cases: [(f64, f64, f64); 17] = [
            (1.0, 2.0, 0.7853981633974482),
            (2.0, 1.0, 0.881373587019543),
            (0.0, 1.0, 1.5707963267948963),
            (0.0, 0.25, 3.1415926535897927),
            (4.0, 4.000000000003638, 0.49999999999984845),
            (4.0, 3.999999999996362, 0.5000000000001517),
            (9.955270134189597, 9.955274904175287, 0.31693733816370123),
            (1e-300, 1.0000000010000002e-300, 9.999999996666667e149),
            (1e300, 1e-300, 6.914686750787736e-148),
            (1e-300, 1e300, 1.5707963267948966e-150),
            (1.0, -1.0, 0.6232252401402306),
            (0.0, -1.0, 0.0),
            (3.0, -1.0, 0.6584789484624084),
            (0.5, -1e-12, 20.028211473896718),
            (7.0, 7.0, 0.37796447300922725),
            (2.5e-8, 31000000.0, 0.00028212334359609825),
            (
                169768.46417187434,
                2557356.0496944487,
                0.0008479294238313679,
            ),
        ];
        for (x, y, want) in cases {
            let got = elliprc(std::hint::black_box(x), std::hint::black_box(y));
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "elliprc({x:?}, {y:?}) = {got:?}, SciPy {want:?}"
            );
        }
        // SciPy's domain edges: a subnormal y is NaN like y = 0; an infinite argument is 0,
        // including y = -inf through the principal value.
        assert!(elliprc(1.0, 5e-324).is_nan());
        assert!(elliprc(-0.5, -1.0).is_nan());
        for (x, y) in [
            (f64::INFINITY, 1.0),
            (1.0, f64::INFINITY),
            (1.0, f64::NEG_INFINITY),
        ] {
            let got = elliprc(std::hint::black_box(x), std::hint::black_box(y));
            assert_eq!(
                got.to_bits(),
                0.0_f64.to_bits(),
                "elliprc({x}, {y}) = {got:?}"
            );
        }
    }

    #[test]
    fn elliprc_negative_y_uses_cauchy_pv() {
        // Closed form for y < 0 via the identity:
        //   RC(x, y) = √(x / (x − y)) · RC(x − y, −y).
        // RC(x − y, −y) lands in the x > y > 0 branch, where
        // RC(a, b) = arccosh(√(a/b)) / √(a − b). Pin a few cases.

        // RC(1, -1): xm=2, yp=1. RC(2, 1) = arccosh(√2). Prefactor √(1/2).
        let actual = elliprc(1.0, -1.0);
        assert!(actual.is_finite(), "RC(1, -1) must be real and finite");
        let expected = (1.0_f64 / 2.0).sqrt() * 2.0_f64.sqrt().acosh();
        assert_close(actual, expected, 1e-12, "RC(1, -1) Cauchy PV");

        // RC(0, -1): xm=1, yp=1. RC(1, 1) = 1. Prefactor √(0/1) = 0.
        // So RC(0, -1) = 0.
        let actual = elliprc(0.0, -1.0);
        assert_close(actual, 0.0, 1e-15, "RC(0, -1) Cauchy PV degenerate");

        // RC(3, -1): xm=4, yp=1. RC(4, 1) = arccosh(2) / √3. Prefactor √(3/4).
        let actual = elliprc(3.0, -1.0);
        let expected = (3.0_f64 / 4.0).sqrt() * 2.0_f64.acosh() / 3.0_f64.sqrt();
        assert_close(actual, expected, 1e-12, "RC(3, -1) Cauchy PV");
    }

    #[test]
    fn elliprf_diagonal_is_one_over_sqrt_x() {
        // /porting-to-rust + /testing-golden-artifacts for
        // [frankenscipy-1ww0j]: RF(x, x, x) = 1/√x for x > 0.
        // Closed form, since the integrand collapses to (t+x)^{-3/2}.
        for &x in &[0.5_f64, 1.0, 2.0, 4.0, 9.0, 25.0] {
            let actual = elliprf(x, x, x);
            let expected = 1.0 / x.sqrt();
            assert_close(
                actual,
                expected,
                1e-12,
                &format!("RF({x}, {x}, {x}) = {actual}, expected {expected}"),
            );
        }
    }

    #[test]
    fn elliprf_zero_argument_with_equal_others() {
        // RF(0, y, y) = π / (2√y) (the complete elliptic integral
        // of the first kind degenerates here).
        for &y in &[0.5_f64, 1.0, 4.0, 16.0] {
            let expected = std::f64::consts::FRAC_PI_2 / y.sqrt();
            let actual = elliprf(0.0, y, y);
            assert_close(
                actual,
                expected,
                1e-9,
                &format!("RF(0, {y}, {y}) = {actual}, expected π/(2√{y}) = {expected}"),
            );
        }
    }

    #[test]
    fn elliprf_scipy_reference_value() {
        // scipy.special.elliprf(1.0, 2.0, 4.0) ≈ 0.6850858166334364
        // (Carlson 1995, table of reference values).
        let actual = elliprf(1.0, 2.0, 4.0);
        let expected = 0.685_085_816_633_436_4_f64;
        assert_close(actual, expected, 1e-9, "RF(1, 2, 4) scipy reference");
    }

    #[test]
    fn elliprf_symmetric_in_arguments() {
        // RF(x, y, z) is symmetric in all three arguments.
        for &(x, y, z) in &[(1.0_f64, 2.0, 3.0), (0.5, 1.5, 4.0)] {
            let xyz = elliprf(x, y, z);
            let yxz = elliprf(y, x, z);
            let zyx = elliprf(z, y, x);
            assert_close(xyz, yxz, 1e-12, "RF symmetry x↔y");
            assert_close(xyz, zyx, 1e-12, "RF symmetry x↔z");
        }
    }

    #[test]
    fn elliprf_negative_or_nan_returns_nan() {
        assert!(elliprf(-1.0, 1.0, 1.0).is_nan());
        assert!(elliprf(f64::NAN, 1.0, 1.0).is_nan());
        // Two zeros → divergent → +∞.
        assert!(elliprf(0.0, 0.0, 1.0).is_infinite());
    }

    #[test]
    fn elliprd_diagonal_is_x_to_minus_three_halves() {
        // /porting-to-rust + /testing-golden-artifacts for
        // [frankenscipy-xdi1c]: RD(x, x, x) = x^{-3/2} for x > 0.
        // Closed form: the integrand becomes (t + x)^{-5/2}.
        for &x in &[0.5_f64, 1.0, 2.0, 4.0, 9.0, 25.0] {
            let actual = elliprd(x, x, x);
            let expected = x.powf(-1.5);
            assert_close(
                actual,
                expected,
                1e-9,
                &format!("RD({x}, {x}, {x}) = {actual}, expected x^{{-3/2}} = {expected}"),
            );
        }
    }

    #[test]
    fn elliprd_symmetric_in_first_two_args() {
        // RD(x, y, z) = RD(y, x, z) but NOT RD(z, y, x). Symmetry is
        // only in the first two arguments.
        for &(x, y, z) in &[(1.0_f64, 2.0, 3.0), (0.5, 1.5, 4.0)] {
            let xyz = elliprd(x, y, z);
            let yxz = elliprd(y, x, z);
            assert_close(xyz, yxz, 1e-10, "RD(x, y, z) = RD(y, x, z)");
        }
    }

    #[test]
    fn elliprd_scipy_reference_value() {
        // scipy.special.elliprd(0, 2, 1) ≈ 1.7972103521033898 — Carlson
        // 1995 reference value (and a known scipy-doc test case).
        let actual = elliprd(0.0, 2.0, 1.0);
        let expected = 1.797_210_352_103_389_8_f64;
        assert_close(actual, expected, 1e-9, "RD(0, 2, 1) scipy reference");
    }

    #[test]
    fn elliprd_negative_or_nan_returns_nan() {
        assert!(elliprd(-1.0, 1.0, 1.0).is_nan());
        assert!(elliprd(1.0, -1.0, 1.0).is_nan());
        assert!(elliprd(1.0, 1.0, 0.0).is_nan()); // z must be > 0
        assert!(elliprd(1.0, 1.0, -1.0).is_nan());
        assert!(elliprd(f64::NAN, 1.0, 1.0).is_nan());
        // x = y = 0 → divergent → +∞.
        assert!(elliprd(0.0, 0.0, 1.0).is_infinite());
    }

    #[test]
    fn elliprg_diagonal_is_sqrt_x() {
        // /porting-to-rust + /testing-golden-artifacts for
        // [frankenscipy-781n7]: RG(x, x, x) = √x.
        // Surface average of √(x·1) over the unit sphere = √x.
        for &x in &[0.5_f64, 1.0, 2.0, 4.0, 9.0, 25.0] {
            let actual = elliprg(x, x, x);
            let expected = x.sqrt();
            assert_close(
                actual,
                expected,
                1e-9,
                &format!("RG({x}, {x}, {x}) = {actual}, expected √{x} = {expected}"),
            );
        }
    }

    #[test]
    fn elliprg_zero_one_one_is_pi_over_four() {
        // RG(0, 1, 1) = π/4 (= half of E(0) = π/2).
        let actual = elliprg(0.0, 1.0, 1.0);
        let expected = std::f64::consts::PI / 4.0;
        assert_close(actual, expected, 1e-9, "RG(0, 1, 1) = π/4");
    }

    #[test]
    fn elliprg_two_zeros_is_half_sqrt_remaining() {
        // RG(0, 0, c) = √c / 2 for c > 0 (degenerate case).
        for &c in &[1.0_f64, 4.0, 9.0, 16.0] {
            let actual = elliprg(0.0, 0.0, c);
            let expected = c.sqrt() / 2.0;
            assert_close(
                actual,
                expected,
                1e-12,
                &format!("RG(0, 0, {c}) = {actual}, expected √c/2 = {expected}"),
            );
        }
        // All zero → 0.
        assert_eq!(elliprg(0.0, 0.0, 0.0), 0.0);
    }

    #[test]
    fn elliprg_symmetric_in_all_three_arguments() {
        // RG is fully symmetric (unlike RD).
        let (x, y, z) = (1.0_f64, 2.0, 3.0);
        let xyz = elliprg(x, y, z);
        let yxz = elliprg(y, x, z);
        let zyx = elliprg(z, y, x);
        let yzx = elliprg(y, z, x);
        assert_close(xyz, yxz, 1e-10, "RG x↔y");
        assert_close(xyz, zyx, 1e-10, "RG x↔z");
        assert_close(xyz, yzx, 1e-10, "RG cyclic");
    }

    #[test]
    fn elliprg_negative_or_nan_returns_nan() {
        assert!(elliprg(-1.0, 1.0, 1.0).is_nan());
        assert!(elliprg(1.0, -1.0, 1.0).is_nan());
        assert!(elliprg(1.0, 1.0, -1.0).is_nan());
        assert!(elliprg(f64::NAN, 1.0, 1.0).is_nan());
    }

    // ─── elliprj: Carlson elliptic integral of the third kind ───────────

    #[test]
    fn elliprj_diagonal_is_x_pow_neg_three_halves() {
        // /porting-to-rust + /testing-golden-artifacts for
        // [frankenscipy-ewuqd]: RJ(x, x, x, x) = x^{-3/2} closed form
        // (the integrand collapses to (t + x)^{-5/2}).
        for &x in &[0.5_f64, 1.0, 2.0, 4.0, 9.0, 25.0] {
            let actual = elliprj(x, x, x, x);
            let expected = x.powf(-1.5);
            assert_close(
                actual,
                expected,
                1e-10,
                &format!("RJ({x}, {x}, {x}, {x}) = {actual}, expected {expected}"),
            );
        }
    }

    #[test]
    fn elliprj_p_equals_z_matches_elliprd() {
        // RJ(x, y, z, z) = RD(x, y, z) — same integrand under p → z.
        for &(x, y, z) in &[
            (1.0_f64, 2.0, 3.0),
            (0.5, 1.5, 4.0),
            (0.25, 0.5, 1.0),
            (0.0, 1.0, 1.0),
            (3.0, 5.0, 7.0),
        ] {
            let rj = elliprj(x, y, z, z);
            let rd = elliprd(x, y, z);
            assert_close(rj, rd, 1e-9, &format!("RJ({x}, {y}, {z}, {z}) vs RD"));
        }
    }

    #[test]
    fn elliprj_scipy_reference_value() {
        // scipy.special.elliprj(1.0, 2.0, 3.0, 4.0) ≈ 0.239848099749568
        let actual = elliprj(1.0, 2.0, 3.0, 4.0);
        assert_close(actual, 0.239_848_099_749_568, 1e-9, "RJ SciPy reference");
    }

    #[test]
    fn elliprj_symmetric_in_xyz() {
        // RJ is symmetric in (x, y, z) but not p.
        let p = 4.0_f64;
        for &(x, y, z) in &[(1.0_f64, 2.0, 3.0), (0.5, 1.5, 4.0)] {
            let xyz = elliprj(x, y, z, p);
            let yxz = elliprj(y, x, z, p);
            let zyx = elliprj(z, y, x, p);
            let yzx = elliprj(y, z, x, p);
            assert_close(xyz, yxz, 1e-10, "RJ x↔y");
            assert_close(xyz, zyx, 1e-10, "RJ x↔z");
            assert_close(xyz, yzx, 1e-10, "RJ cyclic");
        }
    }

    #[test]
    fn elliprj_input_contract() {
        assert!(elliprj(-1.0, 1.0, 1.0, 1.0).is_nan());
        assert!(elliprj(1.0, -1.0, 1.0, 1.0).is_nan());
        assert!(elliprj(1.0, 1.0, -1.0, 1.0).is_nan());
        assert!(elliprj(f64::NAN, 1.0, 1.0, 1.0).is_nan());
        assert!(elliprj(1.0, 1.0, 1.0, f64::NAN).is_nan());
        // p == 0 is a genuine pole → NaN; p < 0 is the finite Cauchy-PV branch
        // (frankenscipy-vrikc), matching scipy.special.elliprj(1,1,1,-1).
        assert!(elliprj(1.0, 1.0, 1.0, 0.0).is_nan());
        assert!((elliprj(1.0, 1.0, 1.0, -1.0) - -0.5651621397896541).abs() < 1e-6);
        // Two zeros among (x, y, z) → divergence.
        assert!(elliprj(0.0, 0.0, 1.0, 1.0).is_infinite());
    }

    #[test]
    fn elliprj_homogeneity_scaling_law() {
        // RJ(λx, λy, λz, λp) = λ^{-3/2} · RJ(x, y, z, p) — same
        // homogeneity degree as RD; exercises the duplication path
        // at very different scales for the same predicted ratio.
        let bases: &[(f64, f64, f64, f64)] = &[
            (1.0, 2.0, 3.0, 4.0),
            (0.5, 1.5, 4.0, 2.0),
            (0.25, 0.5, 1.0, 0.75),
        ];
        for &(x, y, z, p) in bases {
            let base = elliprj(x, y, z, p);
            for &lam in &[0.25_f64, 1.0, 4.0, 16.0] {
                let scaled = elliprj(lam * x, lam * y, lam * z, lam * p);
                let predicted = base * lam.powf(-1.5);
                assert_close(
                    scaled,
                    predicted,
                    1e-9 * predicted.abs().max(1.0),
                    &format!("RJ homogeneity at ({x}, {y}, {z}, {p}) λ={lam}"),
                );
            }
        }
    }

    #[test]
    fn ellipk_matches_scipy_reference_values() {
        // scipy.special.ellipk([0, 0.25, 0.5, 0.75])
        // -> [1.5707963267948966, 1.6857503548125961, 1.8540746773013719, 2.1565156474996432]
        let cases = [
            (0.0, std::f64::consts::FRAC_PI_2), // K(0) = π/2
            (0.25, 1.685_750_354_812_596),
            (0.5, 1.8540746773013719),
            (0.75, 2.1565156474996432),
        ];
        for (m, expected) in cases {
            let got = ellipk_scalar(m, RuntimeMode::Strict).expect("ellipk");
            assert_close(got, expected, 1e-10, &format!("ellipk({m})"));
        }
    }

    #[test]
    fn ellipe_matches_scipy_reference_values() {
        // scipy.special.ellipe([0, 0.25, 0.5, 0.75])
        // -> [1.5707963267948966, 1.4674622093394272, 1.3506438810476755, 1.2110560275684594]
        let cases = [
            (0.0, std::f64::consts::FRAC_PI_2), // E(0) = π/2
            (0.25, 1.4674622093394272),
            (0.5, 1.3506438810476755),
            (0.75, 1.2110560275684594),
        ];
        for (m, expected) in cases {
            let got = ellipe_scalar(m, RuntimeMode::Strict).expect("ellipe");
            assert_close(got, expected, 1e-10, &format!("ellipe({m})"));
        }
    }

    #[test]
    fn ellipk_ellipe_negative_m_match_scipy() {
        // scipy.special.ellipk / ellipe accept negative m (reciprocal-modulus
        // transformation); we previously fail-closed (NaN) outside [0, 1].
        let ellipk_cases = [
            (-0.5, 1.415737208425956_f64),
            (-1.0, 1.3110287771460598),
            (-3.0, 1.0782578237498215),
            (-10.0, 0.7908718902387385),
        ];
        for (m, expected) in ellipk_cases {
            let got = ellipk_scalar(m, RuntimeMode::Strict).expect("ellipk(neg m)");
            assert_close(got, expected, 1e-12, &format!("ellipk({m})"));
        }
        let ellipe_cases = [
            (-0.5, 1.7517712756948174_f64),
            (-1.0, 1.9100988945138562),
            (-3.0, 2.422112055136919),
            (-10.0, 3.639138038417769),
        ];
        for (m, expected) in ellipe_cases {
            let got = ellipe_scalar(m, RuntimeMode::Strict).expect("ellipe(neg m)");
            assert_close(got, expected, 1e-12, &format!("ellipe({m})"));
        }
    }

    #[test]
    fn ellipkinc_ellipeinc_negative_m_match_scipy() {
        // Incomplete elliptic integrals accept negative m in scipy; we previously
        // fail-closed. The Carlson forms already handle m<0; the periodicity term
        // uses the now-supported K(m<0)/E(m<0). Includes phi>π/2 (periodicity).
        let kinc = [
            (0.7, -0.5, 0.6763102040712793_f64),
            (1.2, -2.0, 0.9540256933864918),
            (2.5, -3.0, 1.5990904596280129),
            (0.7, -8.0, 0.5142051636884055),
        ];
        for (phi, m, expected) in kinc {
            let got = ellipkinc_scalar(phi, m, RuntimeMode::Strict).expect("ellipkinc(neg m)");
            assert_close(got, expected, 1e-11, &format!("ellipkinc({phi},{m})"));
        }
        let einc = [
            (0.7, -0.5, 0.7251349471673342_f64),
            (1.2, -2.0, 1.551875519436463),
            (2.5, -3.0, 4.095897935543082),
            (0.7, -8.0, 1.0069768195814057),
        ];
        for (phi, m, expected) in einc {
            let got = ellipeinc_scalar(phi, m, RuntimeMode::Strict).expect("ellipeinc(neg m)");
            assert_close(got, expected, 1e-11, &format!("ellipeinc({phi},{m})"));
        }
    }

    #[test]
    fn ellipj_matches_scipy_reference_values() {
        // scipy.special.ellipj(0.5, 0.25) returns (sn, cn, dn, ph)
        // sn = 0.4706, cn = 0.8823, dn = 0.9406
        let (sn, cn, dn, _ph) = ellipj(0.5, 0.25);
        assert_close(sn, 0.4750829360, 1e-4, "ellipj sn(0.5, 0.25)");
        assert_close(cn, 0.8799410230, 1e-4, "ellipj cn(0.5, 0.25)");
        assert_close(dn, 0.9713773988, 1e-4, "ellipj dn(0.5, 0.25)");
    }

    #[test]
    fn ellipj_zero_matches_scipy_reference_values() {
        // scipy.special.ellipj(0, m) = (0, 1, 1, 0) for any m
        let (sn, cn, dn, ph) = ellipj(0.0, 0.5);
        assert_close(sn, 0.0, 1e-10, "ellipj sn(0, 0.5)");
        assert_close(cn, 1.0, 1e-10, "ellipj cn(0, 0.5)");
        assert_close(dn, 1.0, 1e-10, "ellipj dn(0, 0.5)");
        assert_close(ph, 0.0, 1e-10, "ellipj ph(0, 0.5)");
    }

    #[test]
    fn expn_matches_scipy_reference_values() {
        // scipy.special.expn(1, 1) ≈ 0.2193839344
        // scipy.special.expn(2, 1) ≈ 0.1484955068
        let e1_1 = expn_scalar(1, 1.0);
        assert_close(e1_1, 0.2193839344, 1e-6, "expn(1, 1)");

        let e2_1 = expn_scalar(2, 1.0);
        assert_close(e2_1, 0.1484955068, 1e-6, "expn(2, 1)");
    }

    #[test]
    fn elliprc_matches_scipy_reference_values() {
        // scipy.special.elliprc(1, 2) ≈ 0.7853981634
        let rc = elliprc(1.0, 2.0);
        assert_close(rc, std::f64::consts::FRAC_PI_4, 1e-6, "elliprc(1, 2)");
    }

    #[test]
    fn elliprf_matches_scipy_reference_values() {
        // scipy.special.elliprf(0, 1, 2) ≈ 1.3110287771
        let rf = elliprf(0.0, 1.0, 2.0);
        assert_close(rf, 1.3110287771, 1e-6, "elliprf(0, 1, 2)");
    }

    #[test]
    fn elliprd_matches_scipy_reference_values() {
        // scipy.special.elliprd(0, 1, 2) ≈ 1.0679379896673962
        let rd = elliprd(0.0, 1.0, 2.0);
        assert_close(rd, 1.0679379896673962, 1e-6, "elliprd(0, 1, 2)");
    }

    #[test]
    fn elliprg_matches_scipy_reference_values() {
        // scipy.special.elliprg(0, 1, 2) ≈ 0.9550494472569279
        let rg = elliprg(0.0, 1.0, 2.0);
        assert_close(rg, 0.9550494472569279, 1e-6, "elliprg(0, 1, 2)");
    }

    #[test]
    fn elliprj_matches_scipy_reference_values() {
        // scipy.special.elliprj(0, 1, 2, 3) ≈ 0.7768862377858233
        let rj = elliprj(0.0, 1.0, 2.0, 3.0);
        assert_close(rj, 0.7768862377858233, 1e-6, "elliprj(0, 1, 2, 3)");
    }

    #[test]
    fn elliprj_negative_p_cauchy_principal_value() {
        // frankenscipy-vrikc: p < 0 is the Cauchy-PV branch. scipy.special 1.17.1
        // computes a finite value; fsci previously failed closed to NaN.
        assert_close(
            elliprj(1.0, 2.0, 3.0, -1.0),
            -0.0932404524386764,
            1e-6,
            "rj(1,2,3,-1)",
        );
        assert_close(
            elliprj(0.5, 1.0, 2.0, -0.5),
            -0.16068487318451444,
            1e-6,
            "rj(.5,1,2,-.5)",
        );
        assert_close(
            elliprj(2.0, 3.0, 4.0, -1.0),
            0.05098889152113835,
            1e-6,
            "rj(2,3,4,-1)",
        );
        assert_close(
            elliprj(0.1, 0.5, 1.0, -0.3),
            -1.9602030387320362,
            1e-6,
            "rj(.1,.5,1,-.3)",
        );
        // a single zero argument resolves through RC's diagonal / positive branch
        assert_close(
            elliprj(0.0, 2.0, 3.0, -1.0),
            -0.8732889802533521,
            1e-6,
            "rj(0,2,3,-1)",
        );
        assert_close(
            elliprj(1.0, 0.0, 3.0, -1.0),
            -1.3022045412166559,
            1e-6,
            "rj(1,0,3,-1)",
        );
        // p == 0 is a genuine pole (scipy → NaN); two zeros diverge (→ inf)
        assert!(elliprj(1.0, 2.0, 3.0, 0.0).is_nan(), "rj(1,2,3,0) is NaN");
        assert!(
            elliprj(0.0, 0.0, 3.0, -1.0).is_infinite(),
            "rj(0,0,3,-1) is inf"
        );
    }
}
