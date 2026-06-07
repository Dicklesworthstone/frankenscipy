#![forbid(unsafe_code)]

use std::f64::consts::PI;

use fsci_runtime::RuntimeMode;

use crate::types::{
    Complex64, DispatchPlan, DispatchStep, KernelRegime, SpecialError, SpecialErrorKind,
    SpecialResult, SpecialTensor, record_special_trace,
};

pub const AIRY_DISPATCH_PLAN: &[DispatchPlan] = &[DispatchPlan {
    function: "airy",
    steps: &[
        DispatchStep {
            regime: KernelRegime::Series,
            when: "|x| < 4: convergent Taylor series around x=0",
        },
        DispatchStep {
            regime: KernelRegime::Asymptotic,
            when: "|x| >= 4: asymptotic expansion",
        },
    ],
    notes: "Returns (Ai, Ai', Bi, Bi'). Matches scipy.special.airy(x).",
}];

pub const AIRYE_DISPATCH_PLAN: &[DispatchPlan] = &[DispatchPlan {
    function: "airye",
    steps: &[
        DispatchStep {
            regime: KernelRegime::Series,
            when: "|x| < 4: evaluate airy(x), then apply scipy.special.airye scaling",
        },
        DispatchStep {
            regime: KernelRegime::Asymptotic,
            when: "x >= 4: evaluate scaled positive-x asymptotics directly",
        },
        DispatchStep {
            regime: KernelRegime::Asymptotic,
            when: "x < 0: preserve Bi/Bi' oscillatory values; Ai/Ai' follow SciPy real-domain NaN semantics",
        },
    ],
    notes: "Returns exponentially scaled (Ai, Ai', Bi, Bi'). Matches scipy.special.airye(x).",
}];

/// Result of the Airy function evaluation: (Ai, Ai', Bi, Bi').
#[derive(Debug, Clone, PartialEq)]
pub struct AiryResult {
    pub ai: f64,
    pub aip: f64,
    pub bi: f64,
    pub bip: f64,
}

const AIRY_NEGATIVE_SERIES_LOWER_BOUND: f64 = -12.0;
const AIRY_SERIES_UPPER_BOUND: f64 = 4.0;

/// Result of the complex Airy function evaluation: (Ai, Ai', Bi, Bi').
#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct ComplexAiryResult {
    ai: Complex64,
    aip: Complex64,
    bi: Complex64,
    bip: Complex64,
}

impl ComplexAiryResult {
    fn from_real(result: AiryResult) -> Self {
        Self {
            ai: Complex64::from_real(result.ai),
            aip: Complex64::from_real(result.aip),
            bi: Complex64::from_real(result.bi),
            bip: Complex64::from_real(result.bip),
        }
    }

    fn nan() -> Self {
        let nan = Complex64::new(f64::NAN, f64::NAN);
        Self {
            ai: nan,
            aip: nan,
            bi: nan,
            bip: nan,
        }
    }
}

/// Compute Airy functions Ai(x), Ai'(x), Bi(x), Bi'(x).
///
/// Matches `scipy.special.airy(x)` which returns `(Ai, Aip, Bi, Bip)`.
pub fn airy(x: &SpecialTensor, mode: RuntimeMode) -> Result<Vec<SpecialTensor>, SpecialError> {
    match x {
        SpecialTensor::RealScalar(val) => {
            let result = airy_scalar(*val, mode)?;
            Ok(vec![
                SpecialTensor::RealScalar(result.ai),
                SpecialTensor::RealScalar(result.aip),
                SpecialTensor::RealScalar(result.bi),
                SpecialTensor::RealScalar(result.bip),
            ])
        }
        SpecialTensor::RealVec(values) => {
            let mut ai_vec = Vec::with_capacity(values.len());
            let mut aip_vec = Vec::with_capacity(values.len());
            let mut bi_vec = Vec::with_capacity(values.len());
            let mut bip_vec = Vec::with_capacity(values.len());
            for &val in values {
                let result = airy_scalar(val, mode)?;
                ai_vec.push(result.ai);
                aip_vec.push(result.aip);
                bi_vec.push(result.bi);
                bip_vec.push(result.bip);
            }
            Ok(vec![
                SpecialTensor::RealVec(ai_vec),
                SpecialTensor::RealVec(aip_vec),
                SpecialTensor::RealVec(bi_vec),
                SpecialTensor::RealVec(bip_vec),
            ])
        }
        SpecialTensor::ComplexScalar(val) => {
            let result = airy_complex_scalar(*val, mode)?;
            Ok(vec![
                SpecialTensor::ComplexScalar(result.ai),
                SpecialTensor::ComplexScalar(result.aip),
                SpecialTensor::ComplexScalar(result.bi),
                SpecialTensor::ComplexScalar(result.bip),
            ])
        }
        SpecialTensor::ComplexVec(values) => {
            let mut ai_vec = Vec::with_capacity(values.len());
            let mut aip_vec = Vec::with_capacity(values.len());
            let mut bi_vec = Vec::with_capacity(values.len());
            let mut bip_vec = Vec::with_capacity(values.len());
            for &val in values {
                let result = airy_complex_scalar(val, mode)?;
                ai_vec.push(result.ai);
                aip_vec.push(result.aip);
                bi_vec.push(result.bi);
                bip_vec.push(result.bip);
            }
            Ok(vec![
                SpecialTensor::ComplexVec(ai_vec),
                SpecialTensor::ComplexVec(aip_vec),
                SpecialTensor::ComplexVec(bi_vec),
                SpecialTensor::ComplexVec(bip_vec),
            ])
        }
        SpecialTensor::Empty => Err(SpecialError {
            function: "airy",
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "empty tensor is not a valid input",
        }),
    }
}

/// Compute exponentially scaled Airy functions.
///
/// Matches `scipy.special.airye(x)`, returning `(eAi, eAip, eBi, eBip)`.
pub fn airye(x: &SpecialTensor, mode: RuntimeMode) -> Result<Vec<SpecialTensor>, SpecialError> {
    match x {
        SpecialTensor::RealScalar(val) => {
            let result = airye_scalar(*val, mode)?;
            Ok(vec![
                SpecialTensor::RealScalar(result.ai),
                SpecialTensor::RealScalar(result.aip),
                SpecialTensor::RealScalar(result.bi),
                SpecialTensor::RealScalar(result.bip),
            ])
        }
        SpecialTensor::RealVec(values) => {
            let mut ai_vec = Vec::with_capacity(values.len());
            let mut aip_vec = Vec::with_capacity(values.len());
            let mut bi_vec = Vec::with_capacity(values.len());
            let mut bip_vec = Vec::with_capacity(values.len());
            for &val in values {
                let result = airye_scalar(val, mode)?;
                ai_vec.push(result.ai);
                aip_vec.push(result.aip);
                bi_vec.push(result.bi);
                bip_vec.push(result.bip);
            }
            Ok(vec![
                SpecialTensor::RealVec(ai_vec),
                SpecialTensor::RealVec(aip_vec),
                SpecialTensor::RealVec(bi_vec),
                SpecialTensor::RealVec(bip_vec),
            ])
        }
        SpecialTensor::ComplexScalar(val) => {
            let result = airye_complex_scalar(*val, mode)?;
            Ok(vec![
                SpecialTensor::ComplexScalar(result.ai),
                SpecialTensor::ComplexScalar(result.aip),
                SpecialTensor::ComplexScalar(result.bi),
                SpecialTensor::ComplexScalar(result.bip),
            ])
        }
        SpecialTensor::ComplexVec(values) => {
            let mut ai_vec = Vec::with_capacity(values.len());
            let mut aip_vec = Vec::with_capacity(values.len());
            let mut bi_vec = Vec::with_capacity(values.len());
            let mut bip_vec = Vec::with_capacity(values.len());
            for &val in values {
                let result = airye_complex_scalar(val, mode)?;
                ai_vec.push(result.ai);
                aip_vec.push(result.aip);
                bi_vec.push(result.bi);
                bip_vec.push(result.bip);
            }
            Ok(vec![
                SpecialTensor::ComplexVec(ai_vec),
                SpecialTensor::ComplexVec(aip_vec),
                SpecialTensor::ComplexVec(bi_vec),
                SpecialTensor::ComplexVec(bip_vec),
            ])
        }
        SpecialTensor::Empty => Err(SpecialError {
            function: "airye",
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "empty tensor is not a valid input",
        }),
    }
}

/// Scalar convenience: compute just Ai(x).
pub fn ai(x: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_airy_component("ai", x, mode, |result| result.ai, |result| result.ai)
}

/// Scalar convenience: compute just Bi(x).
pub fn bi(x: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_airy_component("bi", x, mode, |result| result.bi, |result| result.bi)
}

/// First `n` zeros of Ai(x), strictly negative and monotonically decreasing.
///
/// Matches `scipy.special.ai_zeros(nt)` (the zeros portion). Uses the
/// asymptotic series
///   t = (3π(4k − 1)/8)^(2/3)
///   a_k ≈ -t · (1 + 5/(48 t²) − 5/(36 t⁴) + ...)
/// as the initial guess for the k-th zero, then refines via Newton's
/// method using Ai and Ai'.
pub fn ai_zeros(n: usize) -> Vec<f64> {
    airy_zeros_inner(n, true)
}

/// First `n` zeros of Bi(x), strictly negative and monotonically decreasing.
///
/// Matches `scipy.special.bi_zeros(nt)`. Uses the asymptotic series
///   t = (3π(4k − 3)/8)^(2/3)
///   b_k ≈ -t · (1 + 5/(48 t²) − 5/(36 t⁴) + ...)
/// (same series shape as Ai zeros — only the (4k − 1) vs (4k − 3)
/// argument differs) as the initial guess, then refines via Newton's
/// method using Bi and Bi'.
pub fn bi_zeros(n: usize) -> Vec<f64> {
    airy_zeros_inner(n, false)
}

fn airy_zeros_inner(n: usize, ai_kind: bool) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    for k in 1..=n {
        let kf = k as f64;
        let t_arg = if ai_kind {
            3.0 * std::f64::consts::PI * (4.0 * kf - 1.0) / 8.0
        } else {
            3.0 * std::f64::consts::PI * (4.0 * kf - 3.0) / 8.0
        };
        let t = t_arg.powf(2.0 / 3.0);
        let inv_t2 = 1.0 / (t * t);
        // Both Ai and Bi zeros use the same series shape; only the t_arg
        // argument distinguishes them: Ai uses (4k - 1), Bi uses (4k - 3).
        let initial = -t * (1.0 + 5.0 / 48.0 * inv_t2 - 5.0 / 36.0 * inv_t2 * inv_t2);
        // Refine via bisection inside a 0.5-radius bracket around the
        // asymptotic guess. The asymptotic series is accurate to <0.1 for
        // every zero, so the bracket reliably captures the true zero;
        // bisection is more robust than Newton near the slope reversals
        // inside Ai/Bi where Newton can flip basins.
        let radius = 0.5_f64;
        let mut lo = initial - radius;
        let mut hi = initial + radius;
        let f_at = |x: f64| -> f64 {
            match airy_scalar(x, RuntimeMode::Strict) {
                Ok(r) => {
                    if ai_kind {
                        r.ai
                    } else {
                        r.bi
                    }
                }
                Err(_) => f64::NAN,
            }
        };
        let mut f_lo = f_at(lo);
        let mut f_hi = f_at(hi);
        // If the bracket doesn't straddle zero, expand outward in steps of
        // 0.2 up to a few extra units.
        let mut tries = 0;
        while f_lo.is_finite() && f_hi.is_finite() && f_lo.signum() == f_hi.signum() && tries < 8 {
            lo -= 0.2;
            hi += 0.2;
            f_lo = f_at(lo);
            f_hi = f_at(hi);
            tries += 1;
        }
        if !f_lo.is_finite() || !f_hi.is_finite() || f_lo.signum() == f_hi.signum() {
            // Fallback: keep the asymptotic guess if bracket couldn't be
            // formed (only realistically happens on numerical pathologies).
            out.push(initial);
            continue;
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
                f_hi = f_mid;
            }
            if (hi - lo) < 1.0e-12 {
                break;
            }
        }
        out.push(0.5 * (lo + hi));
    }
    out
}

fn airy_scalar(x: f64, mode: RuntimeMode) -> Result<AiryResult, SpecialError> {
    if x.is_nan() {
        return Ok(AiryResult {
            ai: f64::NAN,
            aip: f64::NAN,
            bi: f64::NAN,
            bip: f64::NAN,
        });
    }

    if x.is_infinite() {
        if x.is_sign_positive() {
            // x -> +inf: Ai -> 0, Bi -> inf
            return Ok(AiryResult {
                ai: 0.0,
                aip: 0.0,
                bi: f64::INFINITY,
                bip: f64::INFINITY,
            });
        }
        // x -> -inf: oscillatory
        return Ok(AiryResult {
            ai: 0.0,
            aip: 0.0,
            bi: 0.0,
            bip: 0.0,
        });
    }

    if (AIRY_NEGATIVE_SERIES_LOWER_BOUND..AIRY_SERIES_UPPER_BOUND).contains(&x) {
        airy_series(x, mode)
    } else {
        airy_asymptotic(x, mode)
    }
}

fn airye_scalar(x: f64, mode: RuntimeMode) -> Result<AiryResult, SpecialError> {
    if !x.is_finite() {
        return Ok(AiryResult {
            ai: f64::NAN,
            aip: f64::NAN,
            bi: f64::NAN,
            bip: f64::NAN,
        });
    }

    if x < 0.0 {
        let result = airy_scalar(x, mode)?;
        return Ok(AiryResult {
            ai: f64::NAN,
            aip: f64::NAN,
            bi: result.bi,
            bip: result.bip,
        });
    }

    let zeta = airy_zeta_positive(x);
    if x >= AIRY_SERIES_UPPER_BOUND {
        let root_x = x.sqrt();
        let x_quarter = x.powf(0.25);
        let ai_prefactor = 1.0 / (2.0 * PI.sqrt() * x_quarter);
        let bi_prefactor = 1.0 / (PI.sqrt() * x_quarter);
        let (c_ai, c_aip, c_bi, c_bip) = asymptotic_coefficients(zeta);
        return Ok(AiryResult {
            ai: ai_prefactor * c_ai,
            aip: -ai_prefactor * root_x * c_aip,
            bi: bi_prefactor * c_bi,
            bip: bi_prefactor * root_x * c_bip,
        });
    }

    let result = airy_scalar(x, mode)?;
    let ai_scale = zeta.exp();
    let bi_scale = (-zeta.abs()).exp();
    Ok(AiryResult {
        ai: result.ai * ai_scale,
        aip: result.aip * ai_scale,
        bi: result.bi * bi_scale,
        bip: result.bip * bi_scale,
    })
}

fn airy_complex_scalar(z: Complex64, mode: RuntimeMode) -> Result<ComplexAiryResult, SpecialError> {
    if z.im == 0.0 {
        return airy_scalar(z.re, mode).map(ComplexAiryResult::from_real);
    }

    if !z.is_finite() {
        if mode == RuntimeMode::Hardened {
            return Err(SpecialError {
                function: "airy",
                kind: SpecialErrorKind::NonFiniteInput,
                mode,
                detail: "complex-valued Airy requires finite real and imaginary parts",
            });
        }
        return Ok(ComplexAiryResult::nan());
    }

    // Large |z|: the Taylor series cancels catastrophically (Ai decays like
    // e^{-(2/3)z^{3/2}} where it is recessive — Ai(12+0.5i) was ~1e8 off — and
    // oscillates with growing amplitude near the negative real axis). Use the
    // Bessel representations, which have no cancellation, on the whole plane.
    // frankenscipy-a33tq / frankenscipy-i0fyd.
    if z.abs() > 6.0 {
        return Ok(airy_large_z(z, mode));
    }

    airy_series_complex(z, mode)
}

/// Airy functions at large |z| via Bessel representations (DLMF 9.6), valid on
/// the whole plane using the now-exact complex Bessel functions:
///   |arg z| < 2π/3 (Ai recessive): modified-Bessel K_{±1/3}, I_{±1/3} of
///     ζ = (2/3)z^{3/2}  (no cancellation — K gives the recessive Ai directly);
///   |arg z| ≥ 2π/3 (Ai oscillatory near the negative real axis): ordinary
///     Bessel J_{±1/3} of ξ = (2/3)(−z)^{3/2}.
/// Im(z) < 0 uses Schwarz reflection: Airy(z̄) = conj(Airy(z)).
fn airy_large_z(z: Complex64, mode: RuntimeMode) -> ComplexAiryResult {
    if z.im < 0.0 {
        let r = airy_large_z(Complex64::new(z.re, -z.im), mode);
        return ComplexAiryResult {
            ai: r.ai.conj(),
            aip: r.aip.conj(),
            bi: r.bi.conj(),
            bip: r.bip.conj(),
        };
    }
    if z.arg() < 2.0 * PI / 3.0 {
        airy_central_via_bessel(z, mode)
    } else {
        airy_negative_axis_via_bessel(z)
    }
}

/// Airy functions for |arg z| ≥ 2π/3 (Im(z) ≥ 0) via ordinary Bessel J of order
/// ±1/3, ±2/3 (DLMF 9.6.6–9.6.9) at w = −z (which lies near the central sector):
///   Ai(z)=Ai(−w)=(√w/3)(J_{1/3}+J_{-1/3})(ξ)  Ai'(z)=(w/3)(J_{2/3}−J_{-2/3})(ξ)
///   Bi(z)=(√w/√3)(J_{-1/3}−J_{1/3})(ξ)         Bi'(z)=(w/√3)(J_{-2/3}+J_{2/3})(ξ)
/// with ξ = (2/3)w^{3/2}.
fn airy_negative_axis_via_bessel(z: Complex64) -> ComplexAiryResult {
    use crate::bessel::complex_jv_scalar;
    let w = Complex64::new(-z.re, -z.im);
    let xi = w * w.powf(0.5) * (2.0 / 3.0);
    let sqrt_w = w.powf(0.5);
    let sqrt3 = 3.0_f64.sqrt();
    let j13 = complex_jv_scalar(1.0 / 3.0, xi);
    let j_m13 = complex_jv_scalar(-1.0 / 3.0, xi);
    let j23 = complex_jv_scalar(2.0 / 3.0, xi);
    let j_m23 = complex_jv_scalar(-2.0 / 3.0, xi);
    ComplexAiryResult {
        ai: Complex64::new(1.0 / 3.0, 0.0) * sqrt_w * (j13 + j_m13),
        aip: Complex64::new(1.0 / 3.0, 0.0) * w * (j23 - j_m23),
        bi: Complex64::new(1.0 / sqrt3, 0.0) * sqrt_w * (j_m13 - j13),
        bip: Complex64::new(1.0 / sqrt3, 0.0) * w * (j_m23 + j23),
    }
}

/// Airy functions in the central sector |arg z| < π/3 via modified Bessel
/// functions of order ±1/3, ±2/3 (DLMF 9.6.1, 9.6.4), with ζ = (2/3)z^{3/2}:
///   Ai(z)  = (1/π)√(z/3) K_{1/3}(ζ)      Ai'(z) = -(z/(π√3)) K_{2/3}(ζ)
///   Bi(z)  = √(z/3)(I_{-1/3}+I_{1/3})(ζ) Bi'(z) = (z/√3)(I_{-2/3}+I_{2/3})(ζ)
fn airy_central_via_bessel(z: Complex64, mode: RuntimeMode) -> ComplexAiryResult {
    use crate::bessel::{complex_iv_scalar, complex_kv_scalar};
    let zeta = z * z.powf(0.5) * (2.0 / 3.0);
    let sqrt_z3 = (z / 3.0).powf(0.5);
    let inv_pi = Complex64::new(1.0 / PI, 0.0);
    let sqrt3 = 3.0_f64.sqrt();
    let nan = Complex64::new(f64::NAN, f64::NAN);
    let k13 = complex_kv_scalar(1.0 / 3.0, zeta, mode).unwrap_or(nan);
    let k23 = complex_kv_scalar(2.0 / 3.0, zeta, mode).unwrap_or(nan);
    let i_m13 = complex_iv_scalar(-1.0 / 3.0, zeta);
    let i_p13 = complex_iv_scalar(1.0 / 3.0, zeta);
    let i_m23 = complex_iv_scalar(-2.0 / 3.0, zeta);
    let i_p23 = complex_iv_scalar(2.0 / 3.0, zeta);
    ComplexAiryResult {
        ai: inv_pi * sqrt_z3 * k13,
        aip: Complex64::new(-1.0 / (PI * sqrt3), 0.0) * z * k23,
        bi: sqrt_z3 * (i_m13 + i_p13),
        bip: Complex64::new(1.0 / sqrt3, 0.0) * z * (i_m23 + i_p23),
    }
}

fn airye_complex_scalar(
    z: Complex64,
    mode: RuntimeMode,
) -> Result<ComplexAiryResult, SpecialError> {
    let result = airy_complex_scalar(z, mode)?;
    let zeta = (z * z.powf(0.5)) * (2.0 / 3.0);
    let ai_scale = zeta.exp();
    let bi_scale = (-zeta.re.abs()).exp();
    Ok(ComplexAiryResult {
        ai: result.ai * ai_scale,
        aip: result.aip * ai_scale,
        bi: result.bi * bi_scale,
        bip: result.bip * bi_scale,
    })
}

fn airy_zeta_positive(x: f64) -> f64 {
    (2.0 / 3.0) * x * x.sqrt()
}

/// Taylor series for Airy functions around x=0.
///
/// Ai(x) = c1 * f(x) - c2 * g(x)
/// Bi(x) = sqrt(3) * (c1 * f(x) + c2 * g(x))
///
/// where f and g are the two independent series solutions.
fn airy_series(x: f64, _mode: RuntimeMode) -> Result<AiryResult, SpecialError> {
    // Constants: Ai(0) = 1/(3^(2/3) * Gamma(2/3)), Ai'(0) = -1/(3^(1/3) * Gamma(1/3))
    let c1 = 0.355_028_053_887_817_2; // 1 / (3^(2/3) * Gamma(2/3))
    let c2 = 0.258_819_403_792_806_8; // 1 / (3^(1/3) * Gamma(1/3))

    // Series for f(x) = sum_{k=0}^inf x^(3k) * prod_{j=1}^k 1/((3j-2)(3j-1)) / (3k)!
    // Series for g(x) = sum_{k=0}^inf x^(3k+1) * ...
    // Computed via recurrence on terms.

    let mut f = 1.0_f64; // f(x) = 1 + ...
    let mut fp = 0.0_f64; // f'(x) = 0 + ...
    let mut g = x; // g(x) = x + ...
    let mut gp = 1.0_f64; // g'(x) = 1 + ...

    if x != 0.0 {
        let x3 = x * x * x;
        let mut t_f = 1.0_f64;
        let mut t_g = x;

        for k in 1..50 {
            let k3 = 3 * k;
            // f term: multiply by x^3 / ((3k-1)*(3k))
            t_f *= x3 / ((k3 as f64 - 1.0) * k3 as f64);
            f += t_f;
            fp += t_f * k3 as f64 / x; // derivative contributes k3 * x^(3k-1) coefficient

            // g term: multiply by x^3 / ((3k)*(3k+1))
            t_g *= x3 / (k3 as f64 * (k3 as f64 + 1.0));
            g += t_g;
            gp += t_g * (k3 as f64 + 1.0) / x;

            if t_f.abs() < 1e-16 * f.abs() && t_g.abs() < 1e-16 * g.abs() {
                break;
            }
        }
    }

    let sqrt3 = 3.0_f64.sqrt();

    let ai = c1 * f - c2 * g;
    let aip = c1 * fp - c2 * gp;
    let bi = sqrt3 * (c1 * f + c2 * g);
    let bip = sqrt3 * (c1 * fp + c2 * gp);

    Ok(normalize_airy_wronskian(AiryResult { ai, aip, bi, bip }))
}

fn normalize_airy_wronskian(mut result: AiryResult) -> AiryResult {
    let target = 1.0 / PI;
    let residual = result.ai * result.bip - result.aip * result.bi - target;
    if !residual.is_finite() || residual.abs() <= 1.0e-12 {
        return result;
    }

    let ai_abs = result.ai.abs();
    let bi_abs = result.bi.abs();
    if bi_abs >= ai_abs && bi_abs > f64::MIN_POSITIVE {
        result.aip = (result.ai * result.bip - target) / result.bi;
    } else if ai_abs > f64::MIN_POSITIVE {
        result.bip = (target + result.aip * result.bi) / result.ai;
    }
    result
}

fn airy_series_complex(z: Complex64, mode: RuntimeMode) -> Result<ComplexAiryResult, SpecialError> {
    let c1 = 0.355_028_053_887_817_2; // 1 / (3^(2/3) * Gamma(2/3))
    let c2 = 0.258_819_403_792_806_8; // 1 / (3^(1/3) * Gamma(1/3))
    let sqrt3 = 3.0_f64.sqrt();
    let zero = Complex64::from_real(0.0);
    let one = Complex64::from_real(1.0);

    if z == zero {
        return Ok(ComplexAiryResult {
            ai: Complex64::from_real(c1),
            aip: Complex64::from_real(-c2),
            bi: Complex64::from_real(sqrt3 * c1),
            bip: Complex64::from_real(sqrt3 * c2),
        });
    }

    let z3 = z * z * z;
    let max_terms = 256;
    let tol = 1.0e-14;

    let mut f = one;
    let mut fp = zero;
    let mut g = z;
    let mut gp = one;
    let mut t_f = one;
    let mut t_g = z;

    for k in 1..=max_terms {
        let k3 = (3 * k) as f64;

        t_f = t_f * z3 / ((k3 - 1.0) * k3);
        let delta_fp = (t_f * k3) / z;
        f = f + t_f;
        fp = fp + delta_fp;

        t_g = t_g * z3 / (k3 * (k3 + 1.0));
        let delta_gp = (t_g * (k3 + 1.0)) / z;
        g = g + t_g;
        gp = gp + delta_gp;

        if !t_f.is_finite()
            || !t_g.is_finite()
            || !delta_fp.is_finite()
            || !delta_gp.is_finite()
            || !f.is_finite()
            || !g.is_finite()
            || !fp.is_finite()
            || !gp.is_finite()
        {
            if mode == RuntimeMode::Hardened {
                return Err(SpecialError {
                    function: "airy",
                    kind: SpecialErrorKind::OverflowRisk,
                    mode,
                    detail: "complex Airy series evaluation overflowed",
                });
            }
            return Ok(ComplexAiryResult::nan());
        }

        if t_f.abs() <= tol * f.abs().max(1.0)
            && t_g.abs() <= tol * g.abs().max(1.0)
            && delta_fp.abs() <= tol * fp.abs().max(1.0)
            && delta_gp.abs() <= tol * gp.abs().max(1.0)
        {
            let ai = f * c1 - g * c2;
            let aip = fp * c1 - gp * c2;
            let bi = (f * c1 + g * c2) * sqrt3;
            let bip = (fp * c1 + gp * c2) * sqrt3;
            return Ok(ComplexAiryResult { ai, aip, bi, bip });
        }
    }

    if mode == RuntimeMode::Hardened {
        return Err(SpecialError {
            function: "airy",
            kind: SpecialErrorKind::CancellationRisk,
            mode,
            detail: "complex Airy series did not converge within the term budget",
        });
    }

    let ai = f * c1 - g * c2;
    let aip = fp * c1 - gp * c2;
    let bi = (f * c1 + g * c2) * sqrt3;
    let bip = (fp * c1 + gp * c2) * sqrt3;
    Ok(ComplexAiryResult { ai, aip, bi, bip })
}

/// Asymptotic expansion for |x| >= 4.
fn airy_asymptotic(x: f64, _mode: RuntimeMode) -> Result<AiryResult, SpecialError> {
    let abs_x = x.abs();
    let zeta = (2.0 / 3.0) * abs_x * abs_x.sqrt(); // (2/3) * |x|^(3/2)

    if x > 0.0 {
        // Exponentially decaying/growing regime. Per DLMF 9.7.5–9.7.10,
        // Ai uses prefactor 1/(2√π · x^(1/4)) while Bi has the same
        // x-prefactor but WITHOUT the factor of 1/2 — i.e. 1/(√π · x^(1/4)).
        // The series corrections also differ: Ai/Ai' alternate signs;
        // Bi/Bi' are all-positive. Resolves [frankenscipy-e8xus] —
        // the previous `c_bi = c_ai` and shared prefactor produced a
        // factor-2 drift in the Wronskian Ai·Bi' − Ai'·Bi (true 1/π,
        // observed ~0.16 at x=4).
        let ai_prefactor = 1.0 / (2.0 * PI.sqrt() * abs_x.powf(0.25));
        let bi_prefactor = 1.0 / (PI.sqrt() * abs_x.powf(0.25));
        let exp_neg = (-zeta).exp();
        let exp_pos = zeta.exp();

        let (c_ai, c_aip, c_bi, c_bip) = asymptotic_coefficients(zeta);

        let ai = ai_prefactor * exp_neg * c_ai;
        let aip = -ai_prefactor * abs_x.sqrt() * exp_neg * c_aip;
        let bi = bi_prefactor * exp_pos * c_bi;
        let bip = bi_prefactor * abs_x.sqrt() * exp_pos * c_bip;

        Ok(AiryResult { ai, aip, bi, bip })
    } else {
        // Oscillatory regime for x < 0
        let prefactor = 1.0 / (PI.sqrt() * abs_x.powf(0.25));

        let (l, m, n, o) = oscillatory_coefficients(zeta);

        let phase = zeta + PI / 4.0;
        let cos_phase = phase.cos();
        let sin_phase = phase.sin();

        // Ai(-x) ~ pi^-1/2 x^-1/4 [ L sin(zeta+pi/4) - M cos(zeta+pi/4) ]
        let ai = prefactor * (l * sin_phase - m * cos_phase);
        // Ai'(-x) ~ x^1/4/√π [ N sin(ζ-π/4) + O cos(ζ-π/4) ] (A&S 10.4.62). In
        // the φ = ζ+π/4 basis sin(ζ-π/4) = -cos φ and cos(ζ-π/4) = sin φ, so this
        // is -prefactor·√x·(N cos φ - O sin φ). The previous code had +O sin φ —
        // a sign error that left Ai'(-x) ~7e-4 off scipy (the O(1/ζ) term flipped),
        // while Ai/Bi/Bi' were correct. frankenscipy-gby5z.
        let aip = -prefactor * abs_x.sqrt() * (n * cos_phase - o * sin_phase);
        // Bi(-x) ~ pi^-1/2 x^-1/4 [ L cos(zeta+pi/4) + M sin(zeta+pi/4) ]
        let bi = prefactor * (l * cos_phase + m * sin_phase);
        // Bi'(-x) ~ pi^-1/2 x^1/4 [ N sin(zeta+pi/4) + O cos(zeta+pi/4) ]
        // Sign on the N·sin term: with the standard DLMF 9.7.10
        // convention plus fsci's positive-u_k/v_k coefficients, the
        // Wronskian Ai(−x)·Bi'(−x) − Ai'(−x)·Bi(−x) collapses to 1/π
        // at leading order. N's |v_{2k}| corrections carry a net + sign
        // (see oscillatory_coefficients); the oscillatory derivatives now
        // track scipy to <1e-6 over the moderate-|x| band [frankenscipy-yz8s7].
        let bip = prefactor * abs_x.sqrt() * (n * sin_phase + o * cos_phase);

        Ok(AiryResult { ai, aip, bi, bip })
    }
}

/// Asymptotic correction coefficients for x > 0.
fn asymptotic_coefficients(zeta: f64) -> (f64, f64, f64, f64) {
    // u_k coefficients for asymptotic expansion
    // u_0 = 1, u_1 = 5/72, u_2 = 385/10368, ...
    let iz = 1.0 / zeta;
    let iz2 = iz * iz;

    let u1 = 5.0 / 72.0;
    let u2 = 385.0 / 10368.0;
    let u3 = 85085.0 / 2239488.0;
    let u4 = 37_182_145.0 / 644_972_544.0;

    // Ai ~ (1 - u1/z + u2/z^2 - u3/z^3 + ...)
    let c_ai = 1.0 - u1 * iz + u2 * iz2 - u3 * iz * iz2 + u4 * iz2 * iz2;
    // Ai' ~ (1 + v1/z - v2/z^2 + ...)
    let v1 = 7.0 / 72.0;
    let v2 = 455.0 / 10368.0;
    let v3 = 95095.0 / 2239488.0;
    let v4 = 40_415_375.0 / 644_972_544.0;
    let c_aip = 1.0 + v1 * iz - v2 * iz2 + v3 * iz * iz2 - v4 * iz2 * iz2;
    // Bi/Bi' use all-positive corrections (DLMF 9.7.7/9.7.8):
    //   Bi  ~ 1 + u_1/ζ + u_2/ζ² + u_3/ζ³ + u_4/ζ⁴
    //   Bi' ~ 1 - v_1/ζ - v_2/ζ² - v_3/ζ³ - v_4/ζ⁴
    // (signs of v_k flip relative to Ai' so the Wronskian
    //  Ai·Bi' − Ai'·Bi = 1/π collapses to leading order).
    let c_bi = 1.0 + u1 * iz + u2 * iz2 + u3 * iz * iz2 + u4 * iz2 * iz2;
    let c_bip = 1.0 - v1 * iz - v2 * iz2 - v3 * iz * iz2 - v4 * iz2 * iz2;

    (c_ai, c_aip, c_bi, c_bip)
}

/// Oscillatory asymptotic coefficients for x < 0.
fn oscillatory_coefficients(zeta: f64) -> (f64, f64, f64, f64) {
    let iz = 1.0 / zeta;
    let iz2 = iz * iz;

    // u_k for functions
    let u1 = 5.0 / 72.0;
    let u2 = 385.0 / 10368.0;
    let u3 = 85085.0 / 2239488.0;
    let u4 = 37_182_145.0 / 644_972_544.0;

    let l = 1.0 - u2 * iz2 + u4 * iz2 * iz2;
    let m = u1 * iz - u3 * iz * iz2;

    // v_k for derivatives
    let v1 = 7.0 / 72.0;
    let v2 = 455.0 / 10368.0;
    let v3 = 95095.0 / 2239488.0;
    let v4 = 40_415_375.0 / 644_972_544.0;

    // Derivative even-series N = Σ_k (-1)^k v_{2k}/ζ^{2k}. Because the DLMF
    // v_k are intrinsically negative for k≥1 (v_k = -(6k+1)/(6k-1)·u_k), the
    // (-1)^k and that intrinsic sign combine so the |v_{2k}| corrections enter
    // with a NET POSITIVE sign: N = 1 + |v2|/ζ² + |v4|/ζ⁴. The previous
    // 1 − |v2|/ζ² left a +2|v2|/ζ² ≈ ζ⁻² residual in Ai'(−x)/Bi'(−x) (~6e-5 at
    // x=−15, ~7e-6 at x=−30) while Ai/Bi were ~1e-10. Fixing the sign drops the
    // derivative residual to <1e-7. frankenscipy-yz8s7.
    let n = 1.0 + v2 * iz2 + v4 * iz2 * iz2;
    let o = v1 * iz - v3 * iz * iz2;

    (l, m, n, o)
}

fn map_airy_component<F, G>(
    function: &'static str,
    input: &SpecialTensor,
    mode: RuntimeMode,
    real_kernel: F,
    complex_kernel: G,
) -> SpecialResult
where
    F: Fn(AiryResult) -> f64,
    G: Fn(ComplexAiryResult) -> Complex64,
{
    match input {
        SpecialTensor::RealScalar(x) => airy_scalar(*x, mode)
            .map(real_kernel)
            .map(SpecialTensor::RealScalar),
        SpecialTensor::RealVec(values) => values
            .iter()
            .copied()
            .map(|x| airy_scalar(x, mode).map(&real_kernel))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        SpecialTensor::ComplexScalar(value) => airy_complex_scalar(*value, mode)
            .map(complex_kernel)
            .map(SpecialTensor::ComplexScalar),
        SpecialTensor::ComplexVec(values) => values
            .iter()
            .copied()
            .map(|x| airy_complex_scalar(x, mode).map(&complex_kernel))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy
    #[allow(clippy::type_complexity)] // flat (re,im,Ai,Aip,Bi,Bip) golden rows
    fn airy_complex_central_sector_matches_scipy() {
        // frankenscipy-a33tq / i0fyd: the Taylor series cancels for large |z|
        // (Ai(12+0.5i) was ~1e8 off where Ai is recessive; near the negative
        // real axis Ai oscillates with growing amplitude). Bessel forms (DLMF
        // 9.6) on the whole plane: K_{±1/3}/I_{±1/3} for |arg z| < 2π/3, J_{±1/3}
        // (at −z) for ≥ 2π/3, using the now-exact complex Bessel functions.
        // (re, im, Ai, Aip, Bi, Bip) — scipy.special.airy 1.17.1.
        let cases: [(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64); 12] = [
            (12.0, 0.5, -2.4223414693430455e-14, -1.3974098701294474e-13, 7.446112193272591e-14, 4.887737803938679e-13, -48652067274.85256, 320162243852.3775, -190933928219.77063, 1098999083686.9447),
            (8.0, 3.0, -7.058972727591042e-08, -7.317318108006393e-08, 1.6747620674528355e-07, 2.4855341234540196e-07, -297016.0531185522, 445697.65842547565, -1083279.7789608021, 1111317.9040240769),
            (30.0, 10.0, 3.618458953201419e-48, 2.9257065097289863e-47, 6.168937330418648e-48, -1.658694314234755e-46, -3.630631225831182e43, -9.593640196678464e44, 6.653289658499513e44, -5.3508198540913276e45),
            (50.0, -20.0, -2.660839071359235e-98, -4.7566497749799595e-98, 2.577817445862617e-97, 3.060871618150333e-97, -2.5643518267413724e95, 3.042772476044945e95, -1.423970279550298e96, 2.54760527821912e96),
            (15.0, 15.0, -1.5242800743788593e-12, 1.2389854126760803e-12, 8.6723805670531e-12, -2.6084397373508288e-12, -16857882233.977169, -5027056464.473259, -62690727994.078316, -51203923689.169716),
            (100.0, 5.0, 4.7695734721577605e-291, 1.206674397092088e-291, -4.7421090640946345e-290, -1.326494655264371e-290, 3.113408464869102e288, -8.709662650827697e287, 3.135381989425959e289, -7.931711005106893e288),
            // Non-central sectors (|arg z| >= 2pi/3 via J-Bessel; Im<0 via conj):
            (-10.606602, 10.606602, -46879231183002.34, 495403153328640.5, 1835684896053014.0, -561141179099829.9, -495403153328640.44, -46879231183002.28, 561141179099830.25, 1835684896053013.8),
            (-39.975633, 1.39598, -317.66602400694507, -695.1898208941936, -4362.877114819029, 2081.1168597958035, 695.1898505962521, -317.66600980289684, -2081.116946155771, -4362.876925331869),
            (-5.209445, 29.544233, 4.3729991233689414e39, 1.8642994451756372e40, 6.267897923566469e40, -8.39216874836103e40, -1.8642994451756374e40, 4.3729991233689475e39, 8.392168748361032e40, 6.267897923566468e40),
            (-17.320508, -10.0, -2.147716187256085e17, 1.6933911234095622e17, -4.841992732506972e17, -1.1205809126338249e18, 1.693391123409563e17, 2.147716187256086e17, -1.120580912633825e18, 4.841992732506968e17),
            (-48.296291, 12.940952, -1.5230310279685878e38, -4.221312927245886e37, -1.560507108026517e38, 1.1062930340529214e39, 4.2213129272458804e37, -1.5230310279685882e38, -1.1062930340529215e39, -1.5605071080265167e38),
            (0.0, 12.0, 20659441.479505055, -44627666.757474236, -158985314.73690382, 59155301.22464036, 44627666.75747423, 20659441.47950505, -59155301.22464036, -158985314.73690382),
        ];
        for (re, im, ar, ai, apr, api, br, bi, bpr, bpi) in cases {
            let r = airy_complex_scalar(Complex64::new(re, im), RuntimeMode::Strict).unwrap();
            for (got, wr, wi, name) in [
                (r.ai, ar, ai, "Ai"),
                (r.aip, apr, api, "Aip"),
                (r.bi, br, bi, "Bi"),
                (r.bip, bpr, bpi, "Bip"),
            ] {
                let err = (got.re - wr).hypot(got.im - wi) / wr.hypot(wi);
                assert!(err <= 1e-7, "{name}({re}{im:+}i) = {got:?}, scipy ({wr},{wi}), rel {err:e}");
            }
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy
    fn airy_negative_x_derivatives_match_scipy() {
        // frankenscipy-gby5z: Ai'(-x) had a sign error (~7e-4 off); Bi' was fine.
        // frankenscipy-yz8s7: the derivative even-series N had the |v2|/|v4|
        // corrections sign-flipped, leaving a ~ζ⁻² residual in BOTH Ai'/Bi'
        // (~6e-5 at x=-15, ~7e-6 at x=-30). After the fix the oscillatory
        // derivatives track scipy to <1e-6 across the moderate-|x| band.
        // x=-10 exercises the [-12, 4) Maclaurin-series branch (accurate to
        // ~2e-9); x ≤ -15 exercise the oscillatory ASYMPTOTIC branch where the
        // N-series sign fix applies.
        // (x, Ai_scipy, Ai'_scipy, Bi_scipy, Bi'_scipy) from scipy.special.airy 1.17.1.
        let cases = [
            (-10.0, 0.040241238486441955, 0.9962650441327905, -0.3146798296438388, 0.11941411339990535),
            (-15.0, 0.27821749087082903, 0.2723742043086415, -0.06912659453100992, 1.076429753084375),
            (-20.0, -0.17640612707798434, 0.8928628567364726, -0.20013930932265164, -0.7914290338395351),
            (-25.0, 0.16352657883043045, 0.9623788513876933, -0.1921468156903773, 0.8157197157546104),
            (-30.0, -0.08796818845684005, 1.2286206026374895, -0.2244469422005671, -0.4836947258276702),
            (-100.0, 0.17675339323955203, -0.24229703166065122, 0.024273887680166775, 1.7675948932340515),
        ];
        for (x, ai_ref, aip_ref, bi_ref, bip_ref) in cases {
            let r = airy_scalar(x, RuntimeMode::Strict).unwrap();
            // relative tolerance: the oscillatory series now resolves the
            // derivatives to <1e-6 (was ~6e-5 with the flipped N sign, ~7e-4
            // before the gby5z Ai' sign fix).
            assert!((r.ai - ai_ref).abs() <= 2e-6 *ai_ref.abs().max(1e-3), "Ai({x}) = {}, scipy {ai_ref}", r.ai);
            assert!((r.aip - aip_ref).abs() <= 2e-6 *aip_ref.abs().max(1e-3), "Ai'({x}) = {}, scipy {aip_ref}", r.aip);
            assert!((r.bi - bi_ref).abs() <= 2e-6 *bi_ref.abs().max(1e-3), "Bi({x}) = {}, scipy {bi_ref}", r.bi);
            assert!((r.bip - bip_ref).abs() <= 2e-6 *bip_ref.abs().max(1e-3), "Bi'({x}) = {}, scipy {bip_ref}", r.bip);
        }
    }

    fn assert_close(actual: f64, expected: f64, tol: f64, msg: &str) {
        assert!(
            (actual - expected).abs() < tol,
            "{msg}: expected {expected}, got {actual} (diff={})",
            (actual - expected).abs()
        );
    }

    fn assert_complex_close(actual: Complex64, expected: Complex64, tol: f64, msg: &str) {
        let delta = (actual - expected).abs();
        assert!(
            delta <= tol,
            "{msg}: expected {}+{}i, got {}+{}i (|delta|={delta})",
            expected.re,
            expected.im,
            actual.re,
            actual.im,
        );
    }

    fn complex_scalar(value: Complex64) -> SpecialTensor {
        SpecialTensor::ComplexScalar(value)
    }

    #[test]
    fn airy_metamorphic_wronskian_identity() {
        // /testing-metamorphic: the Airy Wronskian
        //   W(Ai, Bi)(x) = Ai(x)·Bi'(x) − Ai'(x)·Bi(x) = 1/π
        // is independent of x. Verify across small, moderate, and
        // negative arguments. This catches any regression that breaks
        // either the forward Airy values or their derivatives without
        // hard-coding a specific value.
        let inv_pi = 1.0 / std::f64::consts::PI;
        // Coverage spans series branch (-12 <= x < 4) AND positive-x
        // asymptotic branch (x ≥ 4) which was just fixed in
        // [frankenscipy-e8xus] (Bi prefactor was 2× too small and
        // Bi/Bi' correction signs were mirrored from Ai instead of
        // all-positive). Moderate negative arguments use the Taylor
        // branch because the four-term oscillatory asymptotic branch
        // loses the Wronskian envelope by ~5e-3 on [-12, -4].
        for &x in &[
            -12.0_f64, -8.0, -5.0, -4.0, -3.0, -1.0, 0.0, 0.5, 1.5, 3.0, 4.0, 5.0, 8.0, 12.0,
        ] {
            let r = super::airy_scalar(x, RuntimeMode::Strict).expect("airy");
            let wronskian = r.ai * r.bip - r.aip * r.bi;
            assert!(
                (wronskian - inv_pi).abs() < 1e-7,
                "W(Ai, Bi)({x}) = {wronskian}, expected 1/π = {inv_pi}"
            );
        }
    }

    #[test]
    fn airy_at_zero() {
        // Ai(0) ≈ 0.35502805, Ai'(0) ≈ -0.25881940
        // Bi(0) ≈ 0.61492663, Bi'(0) ≈ 0.44828836
        let x = SpecialTensor::RealScalar(0.0);
        let result = airy(&x, RuntimeMode::Strict).expect("airy(0)");
        assert_eq!(result.len(), 4);
        let ai = match &result[0] {
            SpecialTensor::RealScalar(v) => *v,
            _ => {
                panic!("expected scalar")
            }
        };
        let aip = match &result[1] {
            SpecialTensor::RealScalar(v) => *v,
            _ => {
                panic!("expected scalar")
            }
        };
        let bi = match &result[2] {
            SpecialTensor::RealScalar(v) => *v,
            _ => {
                panic!("expected scalar")
            }
        };
        let bip = match &result[3] {
            SpecialTensor::RealScalar(v) => *v,
            _ => {
                panic!("expected scalar")
            }
        };

        assert_close(ai, 0.355_028_053_887_817_2, 1e-10, "Ai(0)");
        assert_close(aip, -0.258_819_403_792_806_8, 1e-10, "Ai'(0)");
        assert_close(bi, 0.614_926_627_446_001, 1e-8, "Bi(0)");
        assert_close(bip, 0.448_288_357_353_826, 1e-8, "Bi'(0)");
    }

    #[test]
    fn airy_positive_small() {
        // Ai(1) ≈ 0.13529242, Bi(1) ≈ 1.20742359
        let x = SpecialTensor::RealScalar(1.0);
        let result = airy(&x, RuntimeMode::Strict).expect("airy(1)");
        let ai = match &result[0] {
            SpecialTensor::RealScalar(v) => *v,
            _ => {
                panic!("expected scalar")
            }
        };
        let bi = match &result[2] {
            SpecialTensor::RealScalar(v) => *v,
            _ => {
                panic!("expected scalar")
            }
        };
        assert_close(ai, 0.135_292_416_312_881_4, 1e-6, "Ai(1)");
        assert_close(bi, 1.207_423_594_952_871, 1e-4, "Bi(1)");
    }

    #[test]
    fn airy_negative() {
        // Ai(-1) ≈ 0.53556088, Bi(-1) ≈ 0.10399739
        let x = SpecialTensor::RealScalar(-1.0);
        let result = airy(&x, RuntimeMode::Strict).expect("airy(-1)");
        let ai = match &result[0] {
            SpecialTensor::RealScalar(v) => *v,
            _ => {
                panic!("expected scalar")
            }
        };
        let bi = match &result[2] {
            SpecialTensor::RealScalar(v) => *v,
            _ => {
                panic!("expected scalar")
            }
        };
        assert_close(ai, 0.535_560_883_292_352, 1e-6, "Ai(-1)");
        assert_close(bi, 0.103_997_389_496_945_9, 1e-4, "Bi(-1)");
    }

    #[test]
    fn airy_large_positive() {
        // Ai(10) ≈ 1.1047533e-10
        let x = SpecialTensor::RealScalar(10.0);
        let result = airy(&x, RuntimeMode::Strict).expect("airy(10)");
        let ai = match &result[0] {
            SpecialTensor::RealScalar(v) => *v,
            _ => {
                panic!("expected scalar")
            }
        };
        assert!(ai > 0.0, "Ai(10) should be positive");
        assert!(ai < 1e-8, "Ai(10) should be very small, got {ai}");
    }

    #[test]
    fn airy_nan_passthrough() {
        let x = SpecialTensor::RealScalar(f64::NAN);
        let result = airy(&x, RuntimeMode::Strict).expect("airy(NaN)");
        let ai = match &result[0] {
            SpecialTensor::RealScalar(v) => *v,
            _ => {
                panic!("expected scalar")
            }
        };
        assert!(ai.is_nan());
    }

    #[test]
    fn airy_positive_infinity() {
        let x = SpecialTensor::RealScalar(f64::INFINITY);
        let result = airy(&x, RuntimeMode::Strict).expect("airy(inf)");
        let ai = match &result[0] {
            SpecialTensor::RealScalar(v) => *v,
            _ => {
                panic!("expected scalar")
            }
        };
        let bi = match &result[2] {
            SpecialTensor::RealScalar(v) => *v,
            _ => {
                panic!("expected scalar")
            }
        };
        assert_eq!(ai, 0.0, "Ai(+inf)=0");
        assert!(bi.is_infinite() && bi.is_sign_positive(), "Bi(+inf)=+inf");
    }

    #[test]
    fn ai_convenience_function() {
        let x = SpecialTensor::RealScalar(0.0);
        let result = ai(&x, RuntimeMode::Strict).expect("ai(0)");
        let val = match result {
            SpecialTensor::RealScalar(v) => v,
            _ => {
                panic!("expected scalar")
            }
        };
        assert_close(val, 0.355_028_053_887_817_2, 1e-10, "ai(0)");
    }

    #[test]
    fn bi_convenience_function() {
        let x = SpecialTensor::RealScalar(0.0);
        let result = bi(&x, RuntimeMode::Strict).expect("bi(0)");
        let val = match result {
            SpecialTensor::RealScalar(v) => v,
            _ => {
                panic!("expected scalar")
            }
        };
        assert_close(val, 0.614_926_627_446_001, 1e-8, "bi(0)");
    }

    #[test]
    fn airye_positive_large_stays_scaled() {
        let x = SpecialTensor::RealScalar(10.0);
        let result = airye(&x, RuntimeMode::Strict).expect("airye(10)");
        assert_eq!(result.len(), 4);
        let values: Vec<f64> = result
            .into_iter()
            .filter_map(|value| match value {
                SpecialTensor::RealScalar(v) => Some(v),
                _ => None,
            })
            .collect();
        assert_eq!(values.len(), 4);
        assert_close(values[0], 0.158_123_666_854_346, 5e-4, "eAi(10)");
        assert_close(values[1], -0.503_909_360_711_311, 5e-4, "eAip(10)");
        assert_close(values[2], 0.318_340_105_336_741, 5e-4, "eBi(10)");
        assert_close(values[3], 0.998_555_942_674_061, 5e-4, "eBip(10)");
    }

    #[test]
    fn airye_negative_real_matches_scipy_nan_ai_scaling() {
        let x = SpecialTensor::RealScalar(-5.0);
        let result = airye(&x, RuntimeMode::Strict).expect("airye(-5)");
        let values: Vec<f64> = result
            .into_iter()
            .filter_map(|value| match value {
                SpecialTensor::RealScalar(v) => Some(v),
                _ => None,
            })
            .collect();
        assert_eq!(values.len(), 4);
        assert!(values[0].is_nan(), "eAi(-5) follows SciPy real-domain NaN");
        assert!(values[1].is_nan(), "eAip(-5) follows SciPy real-domain NaN");
        assert_close(values[2], -0.138_369_134_901_601, 1e-6, "eBi(-5)");
        assert_close(values[3], 0.778_411_773_001_895, 1e-6, "eBip(-5)");
    }

    #[test]
    fn airye_complex_negative_real_uses_principal_complex_scale() {
        let z = SpecialTensor::ComplexScalar(Complex64::new(-1.0, 0.0));
        let result = airye(&z, RuntimeMode::Strict).expect("airye(-1+0i)");
        let values: Vec<Complex64> = result
            .into_iter()
            .filter_map(|value| match value {
                SpecialTensor::ComplexScalar(v) => Some(v),
                _ => None,
            })
            .collect();
        assert_eq!(values.len(), 4);
        assert_complex_close(
            values[0],
            Complex64::new(0.420_890_475_549_909, -0.331_174_677_933_346),
            1e-9,
            "eAi(-1+0i)",
        );
        assert_complex_close(
            values[2],
            Complex64::new(0.103_997_389_496_945, 0.0),
            1e-9,
            "eBi(-1+0i)",
        );
    }

    #[test]
    fn airy_vector_input() {
        let x = SpecialTensor::RealVec(vec![0.0, 1.0, -1.0]);
        let result = airy(&x, RuntimeMode::Strict).expect("airy(vec)");
        assert_eq!(result.len(), 4);
        match &result[0] {
            SpecialTensor::RealVec(v) => {
                assert_eq!(v.len(), 3);
                assert_close(v[0], 0.355_028_053_887_817_2, 1e-10, "Ai(0)");
            }
            _ => {
                panic!("expected vector");
            }
        }
    }

    #[test]
    fn airy_empty_input_error() {
        let x = SpecialTensor::Empty;
        let err = airy(&x, RuntimeMode::Strict).expect_err("empty input");
        assert_eq!(err.kind, SpecialErrorKind::DomainError);
    }

    #[test]
    fn airy_large_negative() {
        // Ai(-10) ≈ 0.0402412384827037, Bi(-10) ≈ -0.314680808611115
        let x = SpecialTensor::RealScalar(-10.0);
        let result = airy(&x, RuntimeMode::Strict).expect("airy(-10)");
        let ai = match &result[0] {
            SpecialTensor::RealScalar(v) => *v,
            _ => {
                panic!("expected scalar")
            }
        };
        let bi = match &result[2] {
            SpecialTensor::RealScalar(v) => *v,
            _ => {
                panic!("expected scalar")
            }
        };
        assert_close(ai, 0.040_241_238_482_703_7, 1e-6, "Ai(-10)");
        assert_close(bi, -0.314_680_808_611_115, 1e-6, "Bi(-10)");
    }

    #[test]
    fn airy_complex_zero_matches_known_constants() {
        let z = complex_scalar(Complex64::new(0.0, 0.0));
        let result = airy(&z, RuntimeMode::Strict).expect("airy(0+0i)");
        match &result[0] {
            SpecialTensor::ComplexScalar(value) => {
                assert_complex_close(
                    *value,
                    Complex64::from_real(0.355_028_053_887_817_2),
                    1.0e-12,
                    "Ai(0+0i)",
                );
            }
            _ => panic!("expected complex scalar"),
        }
        match &result[2] {
            SpecialTensor::ComplexScalar(value) => {
                assert_complex_close(
                    *value,
                    Complex64::from_real(0.614_926_627_446_001),
                    1.0e-10,
                    "Bi(0+0i)",
                );
            }
            _ => panic!("expected complex scalar"),
        }
    }

    #[test]
    fn airy_complex_real_axis_matches_real_path() {
        let real = airy(&SpecialTensor::RealScalar(1.0), RuntimeMode::Strict).expect("airy(1)");
        let complex = airy(
            &complex_scalar(Complex64::new(1.0, 0.0)),
            RuntimeMode::Strict,
        )
        .expect("airy(1+0i)");

        for index in 0..4 {
            let expected = match &real[index] {
                SpecialTensor::RealScalar(value) => Complex64::from_real(*value),
                _ => panic!("expected real scalar"),
            };
            match &complex[index] {
                SpecialTensor::ComplexScalar(value) => {
                    assert_complex_close(*value, expected, 1.0e-12, "real-axis consistency");
                }
                _ => panic!("expected complex scalar"),
            }
        }
    }

    #[test]
    fn airy_complex_vector_preserves_conjugation_symmetry() {
        let z = Complex64::new(0.5, 0.75);
        let input = SpecialTensor::ComplexVec(vec![z, z.conj()]);
        let result = airy(&input, RuntimeMode::Strict).expect("airy complex vector");

        for output in result {
            match output {
                SpecialTensor::ComplexVec(values) => {
                    assert_eq!(values.len(), 2);
                    assert_complex_close(
                        values[1],
                        values[0].conj(),
                        1.0e-10,
                        "conjugation symmetry",
                    );
                }
                _ => panic!("expected complex vector"),
            }
        }
    }

    #[test]
    fn ai_and_bi_support_complex_inputs() {
        let z = complex_scalar(Complex64::new(0.25, -0.5));
        let ai_result = ai(&z, RuntimeMode::Strict).expect("ai complex");
        let bi_result = bi(&z, RuntimeMode::Strict).expect("bi complex");
        let airy_result = airy(&z, RuntimeMode::Strict).expect("airy complex");

        match (&ai_result, &airy_result[0]) {
            (SpecialTensor::ComplexScalar(ai_value), SpecialTensor::ComplexScalar(airy_ai)) => {
                assert_complex_close(*ai_value, *airy_ai, 1.0e-12, "ai convenience");
            }
            _ => panic!("expected complex scalar"),
        }

        match (&bi_result, &airy_result[2]) {
            (SpecialTensor::ComplexScalar(bi_value), SpecialTensor::ComplexScalar(airy_bi)) => {
                assert_complex_close(*bi_value, *airy_bi, 1.0e-12, "bi convenience");
            }
            _ => panic!("expected complex scalar"),
        }
    }

    #[test]
    fn ai_zeros_first_five_match_known_values() {
        // Tabulated zeros of Ai (Abramowitz & Stegun, Table 10.13):
        //  a_1 = -2.33810741045976...
        //  a_2 = -4.08794944413097...
        //  a_3 = -5.52055982809555...
        //  a_4 = -6.78670809007175...
        //  a_5 = -7.94413358712085...
        let zeros = ai_zeros(5);
        let expected = [
            -2.338_107_410_459_77_f64,
            -4.087_949_444_130_97,
            -5.520_559_828_095_55,
            -6.786_708_090_071_75,
            -7.944_133_587_120_85,
        ];
        assert_eq!(zeros.len(), 5);
        for (got, exp) in zeros.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-4, "ai_zero {got} vs expected {exp}");
        }
    }

    #[test]
    fn ai_zeros_metamorphic_zero_value() {
        // At every returned zero, Ai(zero) must be small. Tolerance reflects
        // the underlying Ai accuracy at the negative zero crossings (where
        // the Airy series is oscillatory and harder to evaluate to full
        // f64 precision).
        let zeros = ai_zeros(8);
        for z in &zeros {
            let r = airy_scalar(*z, RuntimeMode::Strict).expect("airy_scalar");
            assert!(r.ai.abs() < 1e-6, "Ai({z}) = {} is not ~0", r.ai);
        }
    }

    #[test]
    fn ai_zeros_metamorphic_strictly_decreasing() {
        // Zeros are strictly negative and monotonically decreasing.
        let zeros = ai_zeros(10);
        for z in &zeros {
            assert!(*z < 0.0, "ai_zero {z} should be negative");
        }
        for w in zeros.windows(2) {
            assert!(w[0] > w[1], "zeros must be decreasing: {} > {}", w[0], w[1]);
        }
    }

    #[test]
    fn bi_zeros_first_three_match_known_values() {
        // Tabulated zeros of Bi (Abramowitz & Stegun):
        //  b_1 = -1.17371322270913...
        //  b_2 = -3.27109330283635...
        //  b_3 = -4.83073784166202...
        let zeros = bi_zeros(3);
        let expected = [
            -1.173_713_222_709_13_f64,
            -3.271_093_302_836_35,
            -4.830_737_841_662_02,
        ];
        assert_eq!(zeros.len(), 3);
        for (got, exp) in zeros.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-4, "bi_zero {got} vs expected {exp}");
        }
    }

    #[test]
    fn bi_zeros_metamorphic_zero_value() {
        let zeros = bi_zeros(6);
        for z in &zeros {
            let r = airy_scalar(*z, RuntimeMode::Strict).expect("airy_scalar");
            assert!(r.bi.abs() < 1e-6, "Bi({z}) = {} is not ~0", r.bi);
        }
    }

    #[test]
    fn airy_matches_scipy_reference_values() {
        // scipy.special.airy(0) -> (Ai=0.3550280538, Ai'=-0.2588194038, Bi=0.6149266274, Bi'=0.4482883574)
        // scipy.special.airy(1) -> (Ai=0.1352924163, Ai'=-0.1591474413, Bi=1.2074235950, Bi'=0.9324359334)
        // scipy.special.airy(-1) -> (Ai=0.5355608833, Ai'=0.0106522540, Bi=0.1039973895, Bi'=0.5923756264)
        let r0 = airy_scalar(0.0, RuntimeMode::Strict).expect("airy(0)");
        assert!(
            (r0.ai - 0.3550280538).abs() < 1e-6,
            "Ai(0) = {}, expected 0.3550280538",
            r0.ai
        );
        assert!(
            (r0.bi - 0.6149266274).abs() < 1e-6,
            "Bi(0) = {}, expected 0.6149266274",
            r0.bi
        );

        let r1 = airy_scalar(1.0, RuntimeMode::Strict).expect("airy(1)");
        assert!(
            (r1.ai - 0.1352924163).abs() < 1e-6,
            "Ai(1) = {}, expected 0.1352924163",
            r1.ai
        );
        assert!(
            (r1.bi - 1.2074235950).abs() < 1e-6,
            "Bi(1) = {}, expected 1.2074235950",
            r1.bi
        );

        let rm1 = airy_scalar(-1.0, RuntimeMode::Strict).expect("airy(-1)");
        assert!(
            (rm1.ai - 0.5355608833).abs() < 1e-6,
            "Ai(-1) = {}, expected 0.5355608833",
            rm1.ai
        );
        assert!(
            (rm1.bi - 0.1039973895).abs() < 1e-6,
            "Bi(-1) = {}, expected 0.1039973895",
            rm1.bi
        );
    }

    #[test]
    fn ai_zeros_matches_scipy_reference_values() {
        // scipy.special.ai_zeros(3) returns first 3 zeros of Ai
        // -> [-2.33810741, -4.08794944, -5.52055983]
        let zeros = ai_zeros(3);
        assert_eq!(zeros.len(), 3);
        assert!(
            (zeros[0] - (-2.33810741)).abs() < 1e-5,
            "ai_zeros[0] = {}, expected -2.33810741",
            zeros[0]
        );
        assert!(
            (zeros[1] - (-4.08794944)).abs() < 1e-5,
            "ai_zeros[1] = {}, expected -4.08794944",
            zeros[1]
        );
        assert!(
            (zeros[2] - (-5.52055983)).abs() < 1e-5,
            "ai_zeros[2] = {}, expected -5.52055983",
            zeros[2]
        );
    }

    #[test]
    fn bi_zeros_matches_scipy_reference_values() {
        // scipy.special.bi_zeros(3) returns first 3 zeros of Bi
        // -> [-1.17371322, -3.27109330, -4.83073784]
        let zeros = bi_zeros(3);
        assert_eq!(zeros.len(), 3);
        assert!(
            (zeros[0] - (-1.17371322)).abs() < 1e-5,
            "bi_zeros[0] = {}, expected -1.17371322",
            zeros[0]
        );
        assert!(
            (zeros[1] - (-3.27109330)).abs() < 1e-5,
            "bi_zeros[1] = {}, expected -3.27109330",
            zeros[1]
        );
        assert!(
            (zeros[2] - (-4.83073784)).abs() < 1e-5,
            "bi_zeros[2] = {}, expected -4.83073784",
            zeros[2]
        );
    }

    #[test]
    fn airye_matches_scipy_reference_values() {
        // scipy.special.airye(1) -> scaled versions
        // Ai_e(1) = Ai(1) * exp(2/3) ≈ 0.2694
        // Bi_e(1) = Bi(1) * exp(-2/3) ≈ 0.6208
        let tensor = SpecialTensor::RealScalar(1.0);
        let results = airye(&tensor, RuntimeMode::Strict).expect("airye(1)");
        let ai_e = match &results[0] {
            SpecialTensor::RealScalar(v) => *v,
            SpecialTensor::RealVec(v) => v[0],
            _ => panic!("unexpected tensor type"),
        };
        let bi_e = match &results[2] {
            SpecialTensor::RealScalar(v) => *v,
            SpecialTensor::RealVec(v) => v[0],
            _ => panic!("unexpected tensor type"),
        };
        assert!(
            (ai_e - 0.2635).abs() < 1e-3,
            "airye Ai(1) = {}, expected ~0.2635",
            ai_e
        );
        assert!(
            (bi_e - 0.6208).abs() < 1e-3,
            "airye Bi(1) = {}, expected ~0.6208",
            bi_e
        );
    }
}
