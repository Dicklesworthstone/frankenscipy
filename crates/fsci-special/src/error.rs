#![forbid(unsafe_code)]

use std::f64::consts::PI;

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
    map_unary_input(
        "erf",
        z,
        mode,
        |x| Ok(erf_scalar(x)),
        |value| Ok(erf_complex_scalar(value)),
    )
}

pub fn erfc(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_unary_input(
        "erfc",
        z,
        mode,
        |x| Ok(erfc_scalar(x)),
        |value| Ok(erfc_complex_scalar(value)),
    )
}

pub fn erfinv(y: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_unary_input(
        "erfinv",
        y,
        mode,
        |v| erfinv_scalar(v, mode),
        |value| erfinv_complex_scalar(value, mode),
    )
}

pub fn erfcinv(y: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_unary_input(
        "erfcinv",
        y,
        mode,
        |v| erfcinv_scalar(v, mode),
        |value| erfcinv_complex_scalar(value, mode),
    )
}

fn map_unary_input<F, G>(
    function: &'static str,
    input: &SpecialTensor,
    mode: RuntimeMode,
    real_kernel: F,
    complex_kernel: G,
) -> SpecialResult
where
    F: Fn(f64) -> Result<f64, SpecialError>,
    G: Fn(Complex64) -> Result<Complex64, SpecialError>,
{
    match input {
        SpecialTensor::RealScalar(x) => real_kernel(*x).map(SpecialTensor::RealScalar),
        SpecialTensor::RealVec(values) => values
            .iter()
            .copied()
            .map(real_kernel)
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        SpecialTensor::ComplexScalar(value) => {
            complex_kernel(*value).map(SpecialTensor::ComplexScalar)
        }
        SpecialTensor::ComplexVec(values) => values
            .iter()
            .copied()
            .map(complex_kernel)
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
    if x < 1.0 {
        return erf_series_real(x);
    }
    // For x ≥ 1 the erf series itself loses ~4 digits to cancellation (its
    // alternating intermediate terms reach ~e^{x²}); since erfc is now machine-
    // accurate via the continued fraction and is small here, erf = 1 - erfc is
    // both machine-accurate and consistent with erfc. frankenscipy-8nkg4.
    1.0 - erfc_cf_real(x)
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
    if x < 0.0 {
        return 2.0 - erfc_scalar(-x);
    }
    if x < 1.0 {
        // erfc = 1 - erf; below 1 the subtraction is well-conditioned
        // (erf < 0.85) and the erf series is accurate.
        return 1.0 - erf_series_real(x);
    }
    // For x ≥ 1 the 1 - erf form catastrophically cancels (erf → 1, the
    // alternating erf series has intermediate terms ~e^{x²}), leaving erfc only
    // ~1e-6 accurate near x ≈ 3.5. Use the Lentz continued fraction instead,
    // which has no cancellation and is machine-accurate. frankenscipy-8nkg4.
    erfc_cf_real(x)
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

/// erfc(x) for x ≥ 1 via the continued fraction (no 1-erf cancellation).
fn erfc_cf_real(x: f64) -> f64 {
    (-x * x).exp() * erfc_cf_h(x) / PI.sqrt()
}

/// Scaled complementary error function erfcx(x) = e^{x²}·erfc(x) for x ≥ ~1,
/// from the continued fraction (no overflow of the intermediate e^{x²}). Used by
/// erfcinv's deep-tail Newton iteration.
pub(crate) fn erfcx_cf_real(x: f64) -> f64 {
    erfc_cf_h(x) / PI.sqrt()
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

fn erf_series_real(x: f64) -> f64 {
    let x2 = x * x;
    let mut term = x;
    let mut sum = term;

    for n in 0..80 {
        let numer = -x2 * ((2 * n + 1) as f64);
        let denom = ((n + 1) * (2 * n + 3)) as f64;
        term = term * numer / denom;
        sum += term;
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

    let mut x = inv_norm_cdf_scalar(0.5 * (y + 1.0)) / 2.0_f64.sqrt();

    // Newton-Raphson refinement for double precision accuracy.
    if y.abs() < 0.7 {
        // Use erf for central values
        for _ in 0..2 {
            let fx = erf_scalar(x) - y;
            let dfx = TWO_INV_SQRT_PI * (-x * x).exp();
            if dfx.abs() < 1e-300 {
                break;
            }
            x -= fx / dfx;
        }
    } else {
        // Use erfc for tail values to avoid 1 - small precision loss
        let yc = if y > 0.0 { 1.0 - y } else { 1.0 + y };
        let sign = y.signum();
        x = x.abs();
        for _ in 0..2 {
            let fx = erfc_scalar(x) - yc;
            let dfx = -TWO_INV_SQRT_PI * (-x * x).exp();
            if dfx.abs() < 1e-300 {
                break;
            }
            x -= fx / dfx;
        }
        x *= sign;
    }

    Ok(x)
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

    Ok(-inv_norm_cdf_scalar(0.5 * y) / 2.0_f64.sqrt())
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
            assert!(err <= 1e-9, "erf({re}{im:+}i) = {g:?}, scipy ({wr}, {wi}), rel {err:e}");
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
            assert!(((got - expected) / expected).abs() < 1e-13, "erfc({x}) = {got:e}, scipy {expected:e}");
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
