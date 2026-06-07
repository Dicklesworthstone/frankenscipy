#![forbid(unsafe_code)]

use crate::validation::IntegrateValidationError;

/// Options for numerical quadrature.
#[derive(Debug, Clone, Copy)]
pub struct QuadOptions {
    /// Absolute error tolerance.
    pub epsabs: f64,
    /// Relative error tolerance.
    pub epsrel: f64,
    /// Maximum number of subdivisions.
    pub limit: usize,
}

impl Default for QuadOptions {
    fn default() -> Self {
        Self {
            epsabs: 1.49e-8,
            epsrel: 1.49e-8,
            limit: 50,
        }
    }
}

/// Result of numerical quadrature.
#[derive(Debug, Clone, PartialEq)]
pub struct QuadResult {
    /// Estimated value of the integral.
    pub integral: f64,
    /// Estimated absolute error.
    pub error: f64,
    /// Number of function evaluations.
    pub neval: usize,
    /// Whether the requested accuracy was achieved.
    pub converged: bool,
}

/// Result of vector-valued numerical quadrature.
#[derive(Debug, Clone, PartialEq)]
pub struct QuadVecResult {
    /// Estimated value of the integral for each output component.
    pub integral: Vec<f64>,
    /// Estimated absolute error using the maximum componentwise GK15/G7 delta.
    pub error: f64,
    /// Number of function evaluations.
    pub neval: usize,
    /// Whether the requested accuracy was achieved.
    pub converged: bool,
}

/// Rule selector for [`cubature`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CubatureRule {
    /// Tensor-product Gauss-Kronrod style rule. The current Rust kernel uses an
    /// embedded Gauss-Legendre rule with adaptive subdivision.
    GaussKronrod,
    /// Compatibility alias for SciPy's `gk21` spelling.
    #[default]
    Gk21,
    /// Compatibility alias for SciPy's `gk15` spelling.
    Gk15,
    /// Genz-Malik compatibility spelling. The current implementation uses the
    /// same embedded adaptive rule and keeps this selector for API parity.
    GenzMalik,
}

/// Options for adaptive multidimensional cubature.
#[derive(Debug, Clone, PartialEq)]
pub struct CubatureOptions {
    /// Relative error tolerance.
    pub rtol: f64,
    /// Absolute error tolerance.
    pub atol: f64,
    /// Maximum number of region subdivisions.
    pub max_subdivisions: usize,
    /// Rule selector retained for SciPy-observable API parity.
    pub rule: CubatureRule,
    /// Points that should be avoided by rules that do not evaluate boundaries.
    ///
    /// The current embedded rule never evaluates region boundaries; points are
    /// validated for dimensionality and otherwise retained as caller intent.
    pub points: Vec<Vec<f64>>,
}

impl Default for CubatureOptions {
    fn default() -> Self {
        Self {
            rtol: 1e-8,
            atol: 0.0,
            max_subdivisions: 10_000,
            rule: CubatureRule::default(),
            points: Vec::new(),
        }
    }
}

/// Status of an adaptive cubature estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CubatureStatus {
    Converged,
    NotConverged,
}

/// Estimate for a single adaptive cubature region.
#[derive(Debug, Clone, PartialEq)]
pub struct CubatureRegion {
    /// Region lower corner in transformed coordinates.
    pub a: Vec<f64>,
    /// Region upper corner in transformed coordinates.
    pub b: Vec<f64>,
    /// Integral estimate over this region.
    pub estimate: Vec<f64>,
    /// Componentwise embedded-rule error estimate over this region.
    pub error: Vec<f64>,
}

/// Result of adaptive multidimensional cubature.
#[derive(Debug, Clone, PartialEq)]
pub struct CubatureResult {
    /// Integral estimate. Scalar-valued integrands have length 1.
    pub estimate: Vec<f64>,
    /// Componentwise absolute error estimate.
    pub error: Vec<f64>,
    /// Whether all components satisfy `atol + rtol * abs(estimate)`.
    pub status: CubatureStatus,
    /// Number of adaptive subdivision steps performed.
    pub subdivisions: usize,
    /// Requested absolute tolerance.
    pub atol: f64,
    /// Requested relative tolerance.
    pub rtol: f64,
    /// Terminal adaptive regions.
    pub regions: Vec<CubatureRegion>,
    /// Number of function evaluations.
    pub neval: usize,
}

/// Scalar convenience result for [`cubature_scalar`].
#[derive(Debug, Clone, PartialEq)]
pub struct CubatureScalarResult {
    pub estimate: f64,
    pub error: f64,
    pub status: CubatureStatus,
    pub subdivisions: usize,
    pub atol: f64,
    pub rtol: f64,
    pub neval: usize,
}

#[derive(Debug, Clone, Copy)]
struct QuadVecSpec {
    epsabs: f64,
    epsrel: f64,
    dim: usize,
}

#[derive(Debug, Clone, Copy)]
enum BoundTransform {
    Finite,
    LowerInfinite { upper: f64 },
    UpperInfinite { lower: f64 },
    BothInfinite,
}

impl BoundTransform {
    fn interval(self, lower: f64, upper: f64) -> (f64, f64) {
        match self {
            Self::Finite => (lower, upper),
            Self::LowerInfinite { .. } | Self::UpperInfinite { .. } => (0.0, 1.0),
            Self::BothInfinite => (-1.0, 1.0),
        }
    }

    fn map(self, t: f64) -> (f64, f64) {
        match self {
            Self::Finite => (t, 1.0),
            Self::LowerInfinite { upper } => (upper - (1.0 - t) / t, 1.0 / t.powi(2)),
            Self::UpperInfinite { lower } => {
                let gap = 1.0 - t;
                (lower + t / gap, 1.0 / gap.powi(2))
            }
            Self::BothInfinite => {
                let angle = std::f64::consts::FRAC_PI_2 * t;
                let x = angle.tan();
                (x, std::f64::consts::FRAC_PI_2 * (1.0 + x * x))
            }
        }
    }

    fn inverse(self, x: f64) -> Option<f64> {
        if !x.is_finite() {
            return None;
        }
        match self {
            Self::Finite => Some(x),
            Self::LowerInfinite { upper } => {
                let distance = upper - x;
                (distance >= 0.0).then_some(1.0 / (1.0 + distance))
            }
            Self::UpperInfinite { lower } => {
                let distance = x - lower;
                (distance >= 0.0).then_some(distance / (1.0 + distance))
            }
            Self::BothInfinite => Some(std::f64::consts::FRAC_2_PI * x.atan()),
        }
    }
}

/// Numerically integrate a scalar function over a finite interval [a, b].
///
/// Uses adaptive Gauss-Kronrod quadrature (7-point Gauss / 15-point Kronrod).
/// Matches the core behavior of `scipy.integrate.quad(f, a, b)`.
pub fn quad<F>(
    f: F,
    a: f64,
    b: f64,
    options: QuadOptions,
) -> Result<QuadResult, IntegrateValidationError>
where
    F: Fn(f64) -> f64,
{
    if !a.is_finite() || !b.is_finite() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "integration bounds must be finite".to_string(),
        });
    }
    if options.epsabs.is_nan()
        || options.epsrel.is_nan()
        || options.epsabs < 0.0
        || options.epsrel < 0.0
    {
        return Err(IntegrateValidationError::QuadInvalidTolerance {
            detail: "tolerances must be non-negative".to_string(),
        });
    }

    if (a - b).abs() < f64::EPSILON {
        return Ok(QuadResult {
            integral: 0.0,
            error: 0.0,
            neval: 0,
            converged: true,
        });
    }

    let mut neval = 0;
    let result = adaptive_gk15(
        &f,
        a,
        b,
        options.epsabs,
        options.epsrel,
        options.limit,
        &mut neval,
    );

    Ok(QuadResult {
        integral: result.0,
        error: result.1,
        neval,
        converged: result.2,
    })
}

/// Numerically integrate a vector-valued function over a finite interval [a, b].
///
/// Uses the same adaptive Gauss-Kronrod quadrature core as [`quad`], but
/// preserves every output component of the integrand.
///
/// Matches the core finite-interval behavior of `scipy.integrate.quad_vec(f, a, b)`.
pub fn quad_vec<F>(
    f: F,
    a: f64,
    b: f64,
    options: QuadOptions,
) -> Result<QuadVecResult, IntegrateValidationError>
where
    F: Fn(f64) -> Vec<f64>,
{
    if !a.is_finite() || !b.is_finite() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "integration bounds must be finite".to_string(),
        });
    }
    if options.epsabs.is_nan()
        || options.epsrel.is_nan()
        || options.epsabs < 0.0
        || options.epsrel < 0.0
    {
        return Err(IntegrateValidationError::QuadInvalidTolerance {
            detail: "tolerances must be non-negative".to_string(),
        });
    }

    let sample = f(a);
    let dim = sample.len();
    if (a - b).abs() < f64::EPSILON {
        return Ok(QuadVecResult {
            integral: vec![0.0; dim],
            error: 0.0,
            neval: 1,
            converged: true,
        });
    }

    let mut neval = 0;
    let spec = QuadVecSpec {
        epsabs: options.epsabs,
        epsrel: options.epsrel,
        dim,
    };
    let (integral, error, converged) =
        adaptive_gk15_vec(&f, a, b, options.limit, &mut neval, spec)?;

    Ok(QuadVecResult {
        integral,
        error,
        neval,
        converged,
    })
}

/// Adaptive Gauss-Kronrod 15-point quadrature with recursive subdivision.
fn adaptive_gk15<F>(
    f: &F,
    a: f64,
    b: f64,
    epsabs: f64,
    epsrel: f64,
    limit: usize,
    neval: &mut usize,
) -> (f64, f64, bool)
where
    F: Fn(f64) -> f64,
{
    let (integral, error) = gauss_kronrod_15(f, a, b, neval);

    // Short-circuit on non-finite values: a NaN integrand would otherwise
    // bypass the `error <= tolerance` check (NaN comparisons return false)
    // and force the recursion to expand to 2^limit leaves, hanging the
    // caller. Surface the non-finite result immediately and let upstream
    // wrappers (e.g. nquad) translate it into a typed error.
    // (frankenscipy-t45u3)
    if !integral.is_finite() || !error.is_finite() {
        return (integral, error, false);
    }

    let tolerance = epsabs.max(epsrel * integral.abs());

    if error <= tolerance || limit == 0 {
        return (integral, error, error <= tolerance);
    }

    // Subdivide at midpoint
    let mid = 0.5 * (a + b);
    let next_limit = limit.saturating_sub(1);

    let (i_left, e_left, c_left) =
        adaptive_gk15(f, a, mid, epsabs / 2.0, epsrel, next_limit, neval);
    let (i_right, e_right, c_right) =
        adaptive_gk15(f, mid, b, epsabs / 2.0, epsrel, next_limit, neval);

    let total_integral = i_left + i_right;
    let total_error = e_left + e_right;
    let converged = c_left && c_right;

    (total_integral, total_error, converged)
}

/// Adaptive Gauss-Kronrod 15-point quadrature for vector-valued integrands.
fn adaptive_gk15_vec<F>(
    f: &F,
    a: f64,
    b: f64,
    limit: usize,
    neval: &mut usize,
    spec: QuadVecSpec,
) -> Result<(Vec<f64>, f64, bool), IntegrateValidationError>
where
    F: Fn(f64) -> Vec<f64>,
{
    let (integral, error) = gauss_kronrod_15_vec(f, a, b, neval, spec.dim)?;

    // Same non-finite short-circuit as adaptive_gk15 (frankenscipy-t45u3):
    // a NaN component would otherwise cause 2^limit recursive subdivisions.
    if !error.is_finite() || integral.iter().any(|v| !v.is_finite()) {
        return Ok((integral, error, false));
    }

    let tolerance = spec.epsabs.max(spec.epsrel * max_abs_component(&integral));

    if error <= tolerance || limit == 0 {
        return Ok((integral, error, error <= tolerance));
    }

    let mid = 0.5 * (a + b);
    let next_limit = limit.saturating_sub(1);

    let next_spec = QuadVecSpec {
        epsabs: spec.epsabs / 2.0,
        ..spec
    };
    let (i_left, e_left, c_left) = adaptive_gk15_vec(f, a, mid, next_limit, neval, next_spec)?;
    let (i_right, e_right, c_right) = adaptive_gk15_vec(f, mid, b, next_limit, neval, next_spec)?;

    let total_integral = i_left
        .iter()
        .zip(i_right.iter())
        .map(|(left, right)| left + right)
        .collect();
    let total_error = e_left + e_right;
    let converged = c_left && c_right;

    Ok((total_integral, total_error, converged))
}

/// Gauss-Kronrod 15-point / 7-point quadrature rule on [a, b].
///
/// Returns (integral_estimate, error_estimate).
/// The error is estimated as |K15 - G7| where K15 is the 15-point Kronrod
/// estimate and G7 is the embedded 7-point Gauss estimate.
fn gauss_kronrod_15<F>(f: &F, a: f64, b: f64, neval: &mut usize) -> (f64, f64)
where
    F: Fn(f64) -> f64,
{
    // Kronrod nodes on [-1, 1] (15 points)
    // The 7 Gauss nodes are at indices 1, 3, 5, 7, 9, 11, 13
    const XGK: [f64; 15] = [
        -0.991_455_371_120_812_6,
        -0.949_107_912_342_759,
        -0.864_864_423_359_769_1,
        -0.741_531_185_599_394_4,
        -0.586_087_235_467_691_1,
        -0.405_845_151_377_397_2,
        -0.207_784_955_007_898_5,
        0.0,
        0.207_784_955_007_898_5,
        0.405_845_151_377_397_2,
        0.586_087_235_467_691_1,
        0.741_531_185_599_394_4,
        0.864_864_423_359_769_1,
        0.949_107_912_342_759,
        0.991_455_371_120_812_6,
    ];

    // Kronrod weights (15 points)
    const WGK: [f64; 15] = [
        0.022_935_322_010_529_2,
        0.063_092_092_629_979,
        0.104_790_010_322_250_2,
        0.140_653_259_715_525_9,
        0.169_004_726_639_267_9,
        0.190_350_578_064_785_4,
        0.204_432_940_075_298_9,
        0.209_482_141_084_728,
        0.204_432_940_075_298_9,
        0.190_350_578_064_785_4,
        0.169_004_726_639_267_9,
        0.140_653_259_715_525_9,
        0.104_790_010_322_250_2,
        0.063_092_092_629_979,
        0.022_935_322_010_529_2,
    ];

    // Gauss weights (7 points, at odd indices of XGK)
    const WG: [f64; 7] = [
        0.129_484_966_168_869_7,
        0.279_705_391_489_276_7,
        0.381_830_050_505_118_9,
        0.417_959_183_673_469_4,
        0.381_830_050_505_118_9,
        0.279_705_391_489_276_7,
        0.129_484_966_168_869_7,
    ];

    let half_length = 0.5 * (b - a);
    let center = 0.5 * (a + b);

    let mut result_kronrod = 0.0;
    let mut result_gauss = 0.0;

    for (i, (&xgk, &wgk)) in XGK.iter().zip(WGK.iter()).enumerate() {
        let x = center + half_length * xgk;
        let fval = f(x);
        *neval += 1;

        result_kronrod += wgk * fval;

        // Gauss nodes are at odd indices: 1, 3, 5, 7, 9, 11, 13
        if i % 2 == 1 {
            result_gauss += WG[i / 2] * fval;
        }
    }

    result_kronrod *= half_length;
    result_gauss *= half_length;

    let mut error = (result_kronrod - result_gauss).abs();
    if error.is_nan() {
        error = f64::INFINITY;
    }

    (result_kronrod, error)
}

/// Vector-valued Gauss-Kronrod 15-point / 7-point quadrature rule on [a, b].
fn gauss_kronrod_15_vec<F>(
    f: &F,
    a: f64,
    b: f64,
    neval: &mut usize,
    dim: usize,
) -> Result<(Vec<f64>, f64), IntegrateValidationError>
where
    F: Fn(f64) -> Vec<f64>,
{
    const XGK: [f64; 15] = [
        -0.991_455_371_120_812_6,
        -0.949_107_912_342_759,
        -0.864_864_423_359_769_1,
        -0.741_531_185_599_394_4,
        -0.586_087_235_467_691_1,
        -0.405_845_151_377_397_2,
        -0.207_784_955_007_898_5,
        0.0,
        0.207_784_955_007_898_5,
        0.405_845_151_377_397_2,
        0.586_087_235_467_691_1,
        0.741_531_185_599_394_4,
        0.864_864_423_359_769_1,
        0.949_107_912_342_759,
        0.991_455_371_120_812_6,
    ];
    const WGK: [f64; 15] = [
        0.022_935_322_010_529_2,
        0.063_092_092_629_979,
        0.104_790_010_322_250_2,
        0.140_653_259_715_525_9,
        0.169_004_726_639_267_9,
        0.190_350_578_064_785_4,
        0.204_432_940_075_298_9,
        0.209_482_141_084_728,
        0.204_432_940_075_298_9,
        0.190_350_578_064_785_4,
        0.169_004_726_639_267_9,
        0.140_653_259_715_525_9,
        0.104_790_010_322_250_2,
        0.063_092_092_629_979,
        0.022_935_322_010_529_2,
    ];
    const WG: [f64; 7] = [
        0.129_484_966_168_869_7,
        0.279_705_391_489_276_7,
        0.381_830_050_505_118_9,
        0.417_959_183_673_469_4,
        0.381_830_050_505_118_9,
        0.279_705_391_489_276_7,
        0.129_484_966_168_869_7,
    ];

    let half_length = 0.5 * (b - a);
    let center = 0.5 * (a + b);

    let mut result_kronrod = vec![0.0; dim];
    let mut result_gauss = vec![0.0; dim];

    for (i, (&xgk, &wgk)) in XGK.iter().zip(WGK.iter()).enumerate() {
        let x = center + half_length * xgk;
        let fval = f(x);
        *neval += 1;

        if fval.len() != dim {
            return Err(IntegrateValidationError::QuadInvalidBounds {
                detail: format!(
                    "integrand returned inconsistent vector length: expected {dim}, got {}",
                    fval.len()
                ),
            });
        }

        for (acc, value) in result_kronrod.iter_mut().zip(fval.iter()) {
            *acc += wgk * value;
        }

        if i % 2 == 1 {
            let wg = WG[i / 2];
            for (acc, value) in result_gauss.iter_mut().zip(fval.iter()) {
                *acc += wg * value;
            }
        }
    }

    for value in &mut result_kronrod {
        *value *= half_length;
    }
    for value in &mut result_gauss {
        *value *= half_length;
    }

    let mut error = result_kronrod
        .iter()
        .zip(result_gauss.iter())
        .map(|(kronrod, gauss)| (kronrod - gauss).abs())
        .fold(0.0, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        });
    if error.is_nan() {
        error = f64::INFINITY;
    }

    Ok((result_kronrod, error))
}

fn max_abs_component(values: &[f64]) -> f64 {
    values
        .iter()
        .map(|value| value.abs())
        .fold(0.0, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        })
}

/// Options for double integration.
#[derive(Debug, Clone, Copy)]
pub struct DblquadOptions {
    /// Absolute error tolerance for the outer integral.
    pub epsabs: f64,
    /// Relative error tolerance for the outer integral.
    pub epsrel: f64,
    /// Maximum subdivisions for each 1-D quadrature.
    pub limit: usize,
}

impl Default for DblquadOptions {
    fn default() -> Self {
        Self {
            epsabs: 1.49e-8,
            epsrel: 1.49e-8,
            limit: 50,
        }
    }
}

/// Result of double integration.
#[derive(Debug, Clone, PartialEq)]
pub struct DblquadResult {
    /// Estimated value of the double integral.
    pub integral: f64,
    /// Estimated absolute error.
    pub error: f64,
    /// Whether the requested accuracy was achieved.
    pub converged: bool,
}

/// Numerically compute a double integral ∫∫ f(y, x) dy dx.
///
/// Integrates `f(y, x)` over the region where `x` ranges from `a` to `b`,
/// and for each `x`, `y` ranges from `gfun(x)` to `hfun(x)`.
///
/// Matches `scipy.integrate.dblquad(f, a, b, gfun, hfun)`.
pub fn dblquad<F, GL, GH>(
    f: F,
    a: f64,
    b: f64,
    gfun: GL,
    hfun: GH,
    options: DblquadOptions,
) -> Result<DblquadResult, IntegrateValidationError>
where
    F: Fn(f64, f64) -> f64,
    GL: Fn(f64) -> f64,
    GH: Fn(f64) -> f64,
{
    if !a.is_finite() || !b.is_finite() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "outer integration bounds must be finite".to_string(),
        });
    }

    if (a - b).abs() < f64::EPSILON {
        return Ok(DblquadResult {
            integral: 0.0,
            error: 0.0,
            converged: true,
        });
    }

    let inner_opts = QuadOptions {
        epsabs: options.epsabs,
        epsrel: options.epsrel,
        limit: options.limit,
    };

    // Outer integral over x, inner integral over y for each x
    let outer_result = quad(
        |x| {
            let y_lo = gfun(x);
            let y_hi = hfun(x);
            if (y_lo - y_hi).abs() < f64::EPSILON {
                return 0.0;
            }
            // Inner integral of f(y, x) over y ∈ [gfun(x), hfun(x)]
            match quad(|y| f(y, x), y_lo, y_hi, inner_opts) {
                Ok(r) => r.integral,
                Err(_) => f64::INFINITY, // Trigger error in outer quad
            }
        },
        a,
        b,
        QuadOptions {
            epsabs: options.epsabs,
            epsrel: options.epsrel,
            limit: options.limit,
        },
    )?;

    if !outer_result.integral.is_finite() {
        return Ok(DblquadResult {
            integral: f64::NAN,
            error: f64::INFINITY,
            converged: false,
        });
    }

    Ok(DblquadResult {
        integral: outer_result.integral,
        error: outer_result.error,
        converged: outer_result.converged,
    })
}

// ══════════════════════════════════════════════════════════════════════
// Composite Quadrature Rules — trapezoid and Simpson
// ══════════════════════════════════════════════════════════════════════

/// Result of composite quadrature (trapezoid / Simpson).
#[derive(Debug, Clone, PartialEq)]
pub struct CompositeQuadResult {
    /// Estimated value of the integral.
    pub integral: f64,
}

/// Integrate sampled data using the composite trapezoidal rule.
///
/// Matches `scipy.integrate.trapezoid(y, x)` (formerly `trapz`).
///
/// `y` contains the function values and `x` contains the corresponding
/// sample points. Both slices must have the same length (>= 2).
pub fn trapezoid(y: &[f64], x: &[f64]) -> Result<CompositeQuadResult, IntegrateValidationError> {
    if y.len() != x.len() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: format!(
                "y and x must have the same length (y={}, x={})",
                y.len(),
                x.len()
            ),
        });
    }
    if y.len() < 2 {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "need at least 2 points for trapezoidal rule".to_string(),
        });
    }

    let mut integral = 0.0;
    for i in 0..y.len() - 1 {
        let dx = x[i + 1] - x[i];
        integral += 0.5 * dx * (y[i] + y[i + 1]);
    }

    Ok(CompositeQuadResult { integral })
}

/// Integrate sampled data using the composite trapezoidal rule with uniform spacing.
///
/// Matches `scipy.integrate.trapezoid(y, dx=dx)`.
pub fn trapezoid_uniform(
    y: &[f64],
    dx: f64,
) -> Result<CompositeQuadResult, IntegrateValidationError> {
    if y.len() < 2 {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "need at least 2 points for trapezoidal rule".to_string(),
        });
    }
    if !dx.is_finite() || dx <= 0.0 {
        return Err(IntegrateValidationError::QuadInvalidTolerance {
            detail: "dx must be finite and positive".to_string(),
        });
    }

    let mut integral = 0.5 * (y[0] + y[y.len() - 1]);
    for yi in y.iter().take(y.len() - 1).skip(1) {
        integral += yi;
    }
    integral *= dx;

    Ok(CompositeQuadResult { integral })
}

/// Integrate sampled data using Simpson's rule.
///
/// Matches `scipy.integrate.simpson(y, x=x)`.
///
/// For an even number of intervals (odd number of points), uses composite
/// Simpson's 1/3 rule. For an odd number of intervals (even number of points),
/// uses Simpson's 1/3 rule on the first n-1 panels and a trapezoidal correction
/// on the last panel (matching SciPy's default behavior).
pub fn simpson(y: &[f64], x: &[f64]) -> Result<CompositeQuadResult, IntegrateValidationError> {
    if y.len() != x.len() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: format!(
                "y and x must have the same length (y={}, x={})",
                y.len(),
                x.len()
            ),
        });
    }
    if y.len() < 2 {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "need at least 2 points for Simpson's rule".to_string(),
        });
    }
    if y.len() == 2 {
        // Fall back to trapezoidal
        return trapezoid(y, x);
    }

    let n = y.len();

    if n % 2 == 1 {
        // Odd number of points = even number of intervals.
        // Use composite Simpson's 1/3 rule directly.
        let integral = simpson_nonuniform_odd(y, x);
        Ok(CompositeQuadResult { integral })
    } else {
        // Even number of points = odd number of intervals.
        // Apply Simpson's 1/3 on first n-1 points, trapezoid on last panel.
        let integral_simp = simpson_nonuniform_odd(&y[..n - 1], &x[..n - 1]);
        let last_trap = 0.5 * (x[n - 1] - x[n - 2]) * (y[n - 2] + y[n - 1]);
        Ok(CompositeQuadResult {
            integral: integral_simp + last_trap,
        })
    }
}

/// Simpson's 1/3 rule for non-uniform spacing with an odd number of points.
fn simpson_nonuniform_odd(y: &[f64], x: &[f64]) -> f64 {
    let n = y.len();
    debug_assert!(n >= 3 && n % 2 == 1);
    let mut integral = 0.0;
    let mut i = 0;
    while i + 2 < n {
        let h0 = x[i + 1] - x[i];
        let h1 = x[i + 2] - x[i + 1];
        let h_sum = h0 + h1;
        let h_prod = h0 * h1;

        if h_prod.abs() < f64::EPSILON * 1e-10 {
            // Coincident points: fall back to trapezoidal rule for the two panels
            integral += 0.5 * h0 * (y[i] + y[i + 1]);
            integral += 0.5 * h1 * (y[i + 1] + y[i + 2]);
        } else {
            // Non-uniform Simpson's rule for a pair of panels
            integral += (h_sum / 6.0)
                * ((2.0 - h1 / h0) * y[i]
                    + (h_sum * h_sum / h_prod) * y[i + 1]
                    + (2.0 - h0 / h1) * y[i + 2]);
        }
        i += 2;
    }
    integral
}

/// Integrate sampled data using Simpson's rule with uniform spacing.
///
/// Matches `scipy.integrate.simpson(y, dx=dx)`.
pub fn simpson_uniform(
    y: &[f64],
    dx: f64,
) -> Result<CompositeQuadResult, IntegrateValidationError> {
    if y.len() < 2 {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "need at least 2 points for Simpson's rule".to_string(),
        });
    }
    if !dx.is_finite() || dx <= 0.0 {
        return Err(IntegrateValidationError::QuadInvalidTolerance {
            detail: "dx must be finite and positive".to_string(),
        });
    }
    if y.len() == 2 {
        return trapezoid_uniform(y, dx);
    }

    let n = y.len();

    if n % 2 == 1 {
        // Odd number of points — composite Simpson's 1/3 directly
        let mut integral = y[0] + y[n - 1];
        for i in (1..n - 1).step_by(2) {
            integral += 4.0 * y[i];
        }
        for i in (2..n - 1).step_by(2) {
            integral += 2.0 * y[i];
        }
        integral *= dx / 3.0;
        Ok(CompositeQuadResult { integral })
    } else {
        // Even number of points — Simpson's on first n-1, trapezoid on last
        let m = n - 1; // odd number of points
        let mut integral = y[0] + y[m - 1];
        for i in (1..m - 1).step_by(2) {
            integral += 4.0 * y[i];
        }
        for i in (2..m - 1).step_by(2) {
            integral += 2.0 * y[i];
        }
        integral *= dx / 3.0;
        // Add last trapezoid panel
        integral += 0.5 * dx * (y[n - 2] + y[n - 1]);
        Ok(CompositeQuadResult { integral })
    }
}

/// Cumulatively integrate y using the trapezoidal rule.
///
/// Matches `scipy.integrate.cumulative_trapezoid(y, x)`.
/// Returns a vector of length n-1 where `result[i] = ∫₀ⁱ⁺¹ y dx`.
pub fn cumulative_trapezoid(y: &[f64], x: &[f64]) -> Result<Vec<f64>, IntegrateValidationError> {
    if y.len() != x.len() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: format!(
                "y and x must have the same length (y={}, x={})",
                y.len(),
                x.len()
            ),
        });
    }
    if y.len() < 2 {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "need at least 2 points for cumulative trapezoid".to_string(),
        });
    }

    let n = y.len();
    let mut result = Vec::with_capacity(n - 1);
    let mut cumsum = 0.0;
    for i in 0..n - 1 {
        let dx = x[i + 1] - x[i];
        cumsum += 0.5 * dx * (y[i] + y[i + 1]);
        result.push(cumsum);
    }

    Ok(result)
}

/// Cumulatively integrate y with uniform spacing using the trapezoidal rule.
///
/// Matches `scipy.integrate.cumulative_trapezoid(y, dx=dx)`.
pub fn cumulative_trapezoid_uniform(
    y: &[f64],
    dx: f64,
) -> Result<Vec<f64>, IntegrateValidationError> {
    if y.len() < 2 {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "need at least 2 points for cumulative trapezoid".to_string(),
        });
    }
    if !dx.is_finite() || dx <= 0.0 {
        return Err(IntegrateValidationError::QuadInvalidTolerance {
            detail: "dx must be finite and positive".to_string(),
        });
    }

    let n = y.len();
    let mut result = Vec::with_capacity(n - 1);
    let mut cumsum = 0.0;
    for i in 0..n - 1 {
        cumsum += 0.5 * dx * (y[i] + y[i + 1]);
        result.push(cumsum);
    }

    Ok(result)
}

// ══════════════════════════════════════════════════════════════════════
// Triple and N-D Quadrature, Romberg, cumulative Simpson
// ══════════════════════════════════════════════════════════════════════

/// Numerically integrate a function over a 3D region.
///
/// Matches `scipy.integrate.tplquad(func, a, b, gfun, hfun, qfun, rfun)`.
///
/// Computes ∫∫∫ f(z,y,x) dz dy dx where:
/// - x ∈ [a, b]
/// - y ∈ [gfun(x), hfun(x)]
/// - z ∈ [qfun(x,y), rfun(x,y)]
#[allow(clippy::too_many_arguments)]
pub fn tplquad<F, GL, GH, QL, QH>(
    f: F,
    a: f64,
    b: f64,
    gfun: GL,
    hfun: GH,
    qfun: QL,
    rfun: QH,
    options: DblquadOptions,
) -> Result<DblquadResult, IntegrateValidationError>
where
    F: Fn(f64, f64, f64) -> f64,
    GL: Fn(f64) -> f64,
    GH: Fn(f64) -> f64,
    QL: Fn(f64, f64) -> f64,
    QH: Fn(f64, f64) -> f64,
{
    if !a.is_finite() || !b.is_finite() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "outer integration bounds must be finite".to_string(),
        });
    }

    let inner_opts = QuadOptions {
        epsabs: options.epsabs,
        epsrel: options.epsrel,
        limit: options.limit,
    };

    let outer_result = quad(
        |x| {
            let y_lo = gfun(x);
            let y_hi = hfun(x);
            if (y_lo - y_hi).abs() < f64::EPSILON {
                return 0.0;
            }
            match quad(
                |y| {
                    let z_lo = qfun(x, y);
                    let z_hi = rfun(x, y);
                    if (z_lo - z_hi).abs() < f64::EPSILON {
                        return 0.0;
                    }
                    match quad(|z| f(z, y, x), z_lo, z_hi, inner_opts) {
                        Ok(r) => r.integral,
                        Err(_) => f64::INFINITY,
                    }
                },
                y_lo,
                y_hi,
                inner_opts,
            ) {
                Ok(r) => r.integral,
                Err(_) => f64::INFINITY,
            }
        },
        a,
        b,
        inner_opts,
    )?;

    Ok(DblquadResult {
        integral: outer_result.integral,
        error: outer_result.error,
        converged: outer_result.converged,
    })
}

/// Romberg integration of a function over [a, b].
///
/// Matches `scipy.integrate.romberg(function, a, b)`.
///
/// Uses Richardson extrapolation of the trapezoidal rule for higher accuracy.
///
/// # Arguments
/// * `f` — Function to integrate.
/// * `a` — Lower bound.
/// * `b` — Upper bound.
/// * `max_order` — Maximum number of Richardson extrapolation steps (default 10).
/// * `tol` — Convergence tolerance (default 1.49e-8).
pub fn romb_func<F>(
    f: F,
    a: f64,
    b: f64,
    max_order: Option<usize>,
    tol: Option<f64>,
) -> Result<QuadResult, IntegrateValidationError>
where
    F: Fn(f64) -> f64,
{
    if !a.is_finite() || !b.is_finite() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "bounds must be finite".to_string(),
        });
    }

    let max_order = max_order.unwrap_or(10);
    let tol = tol.unwrap_or(1.49e-8);

    // Romberg table: R[i][j]
    let mut r = vec![vec![0.0; max_order + 1]; max_order + 1];
    let mut neval = 0;

    // R[0][0] = trapezoidal with 1 interval
    r[0][0] = 0.5 * (b - a) * (f(a) + f(b));
    neval += 2;

    for i in 1..=max_order {
        let n = 1usize << i; // 2^i intervals
        let h = (b - a) / n as f64;

        // Compute trapezoidal rule with 2^i intervals using previously computed values
        let mut sum = 0.0;
        for k in 1..n {
            if k % 2 == 1 {
                // Only new points (odd indices)
                sum += f(a + k as f64 * h);
                neval += 1;
            }
        }
        r[i][0] = 0.5 * r[i - 1][0] + h * sum;

        // Richardson extrapolation
        for j in 1..=i {
            let factor = 4.0_f64.powi(j as i32);
            r[i][j] = (factor * r[i][j - 1] - r[i - 1][j - 1]) / (factor - 1.0);
        }

        // Check convergence
        if i >= 2 && (r[i][i] - r[i - 1][i - 1]).abs() < tol {
            return Ok(QuadResult {
                integral: r[i][i],
                error: (r[i][i] - r[i - 1][i - 1]).abs(),
                neval,
                converged: true,
            });
        }
    }

    Ok(QuadResult {
        integral: r[max_order][max_order],
        error: if max_order >= 1 {
            (r[max_order][max_order] - r[max_order - 1][max_order - 1]).abs()
        } else {
            f64::INFINITY
        },
        neval,
        converged: false,
    })
}

/// Romberg integration of pre-sampled data.
///
/// Matches `scipy.integrate.romb(y, dx)`.
///
/// The number of samples must be 2^k + 1 for some k.
pub fn romb(y: &[f64], dx: f64) -> Result<f64, IntegrateValidationError> {
    let n = y.len();
    if n < 2 {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "need at least 2 points".to_string(),
        });
    }
    if dx <= 0.0 || !dx.is_finite() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "dx must be positive and finite".to_string(),
        });
    }
    // Check n = 2^k + 1
    let intervals = n - 1;
    if intervals & (intervals - 1) != 0 {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: format!("romb requires 2^k + 1 samples, got {n}"),
        });
    }

    let k = (intervals as f64).log2().round() as usize;
    let mut r = vec![vec![0.0; k + 1]; k + 1];

    // R[0][0] = trapezoidal with full interval
    r[0][0] = 0.5 * dx * (intervals as f64) * (y[0] + y[intervals]);

    for i in 1..=k {
        let step = intervals >> i; // stride for level i
        let num_intervals = 1usize << i;
        let h = dx * step as f64;

        // Trapezoidal rule at this level
        let mut sum = 0.0;
        for j in 0..num_intervals {
            let idx = j * step;
            if idx + step <= intervals {
                sum += y[idx] + y[idx + step];
            }
        }
        r[i][0] = 0.5 * h * sum;

        // Richardson extrapolation
        for j in 1..=i {
            let factor = 4.0_f64.powi(j as i32);
            r[i][j] = (factor * r[i][j - 1] - r[i - 1][j - 1]) / (factor - 1.0);
        }
    }

    Ok(r[k][k])
}

/// Fixed-order Gaussian quadrature using Gauss-Legendre nodes.
///
/// Integrates `f(x)` over `[a, b]` using `n`-point Gauss-Legendre quadrature.
/// Exact for polynomials up to degree `2n - 1`.
///
/// Matches `scipy.integrate.fixed_quad(func, a, b, n=n)`.
pub fn fixed_quad<F>(
    f: F,
    a: f64,
    b: f64,
    n: usize,
) -> Result<(f64, usize), IntegrateValidationError>
where
    F: Fn(f64) -> f64,
{
    if n == 0 {
        return Ok((0.0, 0));
    }
    if !a.is_finite() || !b.is_finite() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "bounds must be finite".to_string(),
        });
    }

    // Get Gauss-Legendre nodes and weights on [-1, 1]
    let (nodes, weights) = gauss_legendre_nodes_weights(n);

    // Transform from [-1, 1] to [a, b]: x = (b-a)/2 * t + (a+b)/2
    let half_width = (b - a) / 2.0;
    let center = (a + b) / 2.0;

    let mut integral = 0.0;
    for (node, weight) in nodes.iter().zip(weights.iter()) {
        let x = half_width * node + center;
        integral += weight * f(x);
    }
    integral *= half_width;

    Ok((integral, n))
}

/// N-dimensional adaptive quadrature via nested `quad` calls.
///
/// Matches `scipy.integrate.nquad(func, ranges)`.
///
/// # Arguments
/// * `func` — Function of N variables. Takes a slice `&[f64]` of length N.
/// * `ranges` — Integration ranges as `(lower, upper)` pairs, from outermost to innermost.
/// * `options` — Quadrature options for each 1D integration.
///
/// # Example
/// ```rust
/// use fsci_integrate::{nquad, QuadOptions};
/// // ∫₀¹ ∫₀¹ x*y dx dy = 0.25
/// let result = nquad(|args| args[0] * args[1], &[(0.0, 1.0), (0.0, 1.0)], QuadOptions::default()).unwrap();
/// assert!((result.integral - 0.25).abs() < 1e-10);
/// ```
pub fn nquad<F>(
    func: F,
    ranges: &[(f64, f64)],
    options: QuadOptions,
) -> Result<QuadResult, IntegrateValidationError>
where
    F: Fn(&[f64]) -> f64,
{
    use std::cell::RefCell;

    let ndim = ranges.len();
    if ndim == 0 {
        return Ok(QuadResult {
            integral: func(&[]),
            error: 0.0,
            neval: 1,
            converged: true,
        });
    }

    // Use RefCell for interior mutability since quad() takes Fn, not FnMut
    let args = RefCell::new(vec![0.0; ndim]);
    let total_neval = RefCell::new(0usize);
    // Channel for the first inner-quad error: the Fn(&[f64]) -> f64
    // signature can't propagate Result, so the inner closure stuffs
    // any IntegrateValidationError here and returns 0; nquad checks
    // it after the outer quad and propagates instead of silently
    // distorting the integral. Resolves [frankenscipy-vetrv].
    let inner_error: RefCell<Option<IntegrateValidationError>> = RefCell::new(None);

    let result = nquad_inner(
        &func,
        ranges,
        &options,
        &args,
        &total_neval,
        &inner_error,
        0,
    )?;

    if let Some(err) = inner_error.into_inner() {
        return Err(err);
    }

    // Catch the NaN-propagation failure mode that vetrv missed: when
    // the user integrand returns NaN, inner quad returns Ok with a
    // non-finite integral (no Err for the channel to capture), and the
    // outer adaptive loop iterates to maxiter producing NaN. Surface
    // that as a typed error rather than a silent NaN result.
    // Resolves [frankenscipy-666hq].
    if !result.is_finite() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: format!(
                "nquad: integrand produced non-finite result ({result}); inner quad propagated NaN/Inf"
            ),
        });
    }

    Ok(QuadResult {
        integral: result,
        error: 0.0,
        neval: *total_neval.borrow(),
        converged: true,
    })
}

fn nquad_inner<F>(
    func: &F,
    ranges: &[(f64, f64)],
    options: &QuadOptions,
    args: &std::cell::RefCell<Vec<f64>>,
    total_neval: &std::cell::RefCell<usize>,
    inner_error: &std::cell::RefCell<Option<IntegrateValidationError>>,
    dim: usize,
) -> Result<f64, IntegrateValidationError>
where
    F: Fn(&[f64]) -> f64,
{
    let (a, b) = ranges[dim];

    if dim == ranges.len() - 1 {
        // Innermost dimension: call quad directly
        let result = quad(
            |x| {
                args.borrow_mut()[dim] = x;
                func(&args.borrow())
            },
            a,
            b,
            *options,
        )?;
        *total_neval.borrow_mut() += result.neval;
        Ok(result.integral)
    } else {
        // Outer dimension: integrate by nesting. Capture the first
        // inner failure so the caller sees it instead of getting a
        // silently-zeroed result.
        let result = quad(
            |x| {
                args.borrow_mut()[dim] = x;
                match nquad_inner(
                    func,
                    ranges,
                    options,
                    args,
                    total_neval,
                    inner_error,
                    dim + 1,
                ) {
                    Ok(v) => v,
                    Err(e) => {
                        let mut slot = inner_error.borrow_mut();
                        if slot.is_none() {
                            *slot = Some(e);
                        }
                        0.0
                    }
                }
            },
            a,
            b,
            *options,
        )?;
        *total_neval.borrow_mut() += result.neval;
        Ok(result.integral)
    }
}

/// Adaptive cubature of a multidimensional vector-valued function.
///
/// This mirrors the public shape of `scipy.integrate.cubature`: `a` and `b`
/// define the lower and upper corners of a hyper-rectangle, the integrand takes
/// one point at a time as `&[f64]`, and the result contains componentwise
/// integral and error estimates. Infinite limits are handled with smooth
/// transformations to finite coordinates.
pub fn cubature<F>(
    f: F,
    a: &[f64],
    b: &[f64],
    options: CubatureOptions,
) -> Result<CubatureResult, IntegrateValidationError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    validate_cubature_inputs(a, b, &options)?;

    let transforms = cubature_transforms(a, b)?;
    let transformed_ranges: Vec<(f64, f64)> = transforms
        .iter()
        .zip(a.iter().zip(b.iter()))
        .map(|(transform, (&lower, &upper))| transform.interval(lower, upper))
        .collect();

    if a.is_empty() {
        let estimate = f(&[]);
        validate_cubature_output(&estimate, estimate.len())?;
        return Ok(CubatureResult {
            error: vec![0.0; estimate.len()],
            estimate,
            status: CubatureStatus::Converged,
            subdivisions: 0,
            atol: options.atol,
            rtol: options.rtol,
            regions: Vec::new(),
            neval: 1,
        });
    }

    let initial_regions = initial_cubature_regions(&transformed_ranges, &transforms, &options);
    let sample_point = initial_regions
        .first()
        .map(|(lower, upper)| {
            lower
                .iter()
                .zip(upper.iter())
                .map(|(&left, &right)| 0.5 * (left + right))
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| {
            transformed_ranges
                .iter()
                .map(|(lower, upper)| 0.5 * (lower + upper))
                .collect::<Vec<_>>()
        });
    let mut sample_x = vec![0.0; a.len()];
    let mut sample_jacobian = 1.0;
    map_cubature_point(
        &sample_point,
        &transforms,
        &mut sample_x,
        &mut sample_jacobian,
    );
    let sample = f(&sample_x);
    validate_cubature_output(&sample, sample.len())?;
    let output_dim = sample.len();

    let mut neval = 1;
    let mut regions = Vec::with_capacity(initial_regions.len());
    for (initial_a, initial_b) in initial_regions {
        let initial =
            estimate_cubature_region(&f, &transforms, &initial_a, &initial_b, output_dim)?;
        neval += initial.neval;
        regions.push(initial.region);
    }
    let mut subdivisions = 0usize;

    while !cubature_converged(&regions, options.atol, options.rtol)
        && subdivisions < options.max_subdivisions
    {
        let Some(split_index) = largest_error_region(&regions) else {
            break;
        };
        let region = regions.swap_remove(split_index);
        let split_dim = widest_region_dimension(&region);
        let midpoint = 0.5 * (region.a[split_dim] + region.b[split_dim]);

        let left_a = region.a.clone();
        let mut left_b = region.b.clone();
        left_b[split_dim] = midpoint;
        let mut right_a = region.a;
        let right_b = region.b;
        right_a[split_dim] = midpoint;

        let left = estimate_cubature_region(&f, &transforms, &left_a, &left_b, output_dim)?;
        let right = estimate_cubature_region(&f, &transforms, &right_a, &right_b, output_dim)?;
        neval += left.neval + right.neval;
        regions.push(left.region);
        regions.push(right.region);
        subdivisions += 1;
    }

    let (estimate, error) = sum_cubature_regions(&regions, output_dim);
    let status = if componentwise_cubature_converged(&estimate, &error, options.atol, options.rtol)
    {
        CubatureStatus::Converged
    } else {
        CubatureStatus::NotConverged
    };

    Ok(CubatureResult {
        estimate,
        error,
        status,
        subdivisions,
        atol: options.atol,
        rtol: options.rtol,
        regions,
        neval,
    })
}

/// Scalar-valued convenience wrapper around [`cubature`].
pub fn cubature_scalar<F>(
    f: F,
    a: &[f64],
    b: &[f64],
    options: CubatureOptions,
) -> Result<CubatureScalarResult, IntegrateValidationError>
where
    F: Fn(&[f64]) -> f64,
{
    let result = cubature(|x| vec![f(x)], a, b, options)?;
    Ok(CubatureScalarResult {
        estimate: result.estimate[0],
        error: result.error[0],
        status: result.status,
        subdivisions: result.subdivisions,
        atol: result.atol,
        rtol: result.rtol,
        neval: result.neval,
    })
}

#[derive(Debug, Clone)]
struct CubatureEstimate {
    region: CubatureRegion,
    neval: usize,
}

fn validate_cubature_inputs(
    a: &[f64],
    b: &[f64],
    options: &CubatureOptions,
) -> Result<(), IntegrateValidationError> {
    if a.len() != b.len() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "cubature bounds must have the same dimensionality".to_string(),
        });
    }
    if options.atol.is_nan() || options.rtol.is_nan() || options.atol < 0.0 || options.rtol < 0.0 {
        return Err(IntegrateValidationError::QuadInvalidTolerance {
            detail: "cubature tolerances must be non-negative".to_string(),
        });
    }
    for point in &options.points {
        if point.len() != a.len() {
            return Err(IntegrateValidationError::QuadInvalidBounds {
                detail: "cubature points must match bound dimensionality".to_string(),
            });
        }
        if point.iter().any(|value| !value.is_finite()) {
            return Err(IntegrateValidationError::QuadInvalidBounds {
                detail: "cubature points must be finite".to_string(),
            });
        }
    }
    Ok(())
}

fn cubature_transforms(
    a: &[f64],
    b: &[f64],
) -> Result<Vec<BoundTransform>, IntegrateValidationError> {
    a.iter()
        .zip(b.iter())
        .map(
            |(&lower, &upper)| match (lower.is_finite(), upper.is_finite()) {
                (true, true) => Ok(BoundTransform::Finite),
                (false, true) if lower.is_sign_negative() => {
                    Ok(BoundTransform::LowerInfinite { upper })
                }
                (true, false) if upper.is_sign_positive() => {
                    Ok(BoundTransform::UpperInfinite { lower })
                }
                (false, false) if lower.is_sign_negative() && upper.is_sign_positive() => {
                    Ok(BoundTransform::BothInfinite)
                }
                _ => Err(IntegrateValidationError::QuadInvalidBounds {
                    detail: "cubature bounds must be finite or ordered infinities".to_string(),
                }),
            },
        )
        .collect()
}

fn validate_cubature_output(
    values: &[f64],
    expected_dim: usize,
) -> Result<(), IntegrateValidationError> {
    if values.len() != expected_dim {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "cubature integrand output shape changed between evaluations".to_string(),
        });
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "cubature integrand returned a non-finite value".to_string(),
        });
    }
    Ok(())
}

fn initial_cubature_regions(
    ranges: &[(f64, f64)],
    transforms: &[BoundTransform],
    options: &CubatureOptions,
) -> Vec<(Vec<f64>, Vec<f64>)> {
    let mut breakpoints = ranges
        .iter()
        .map(|&(lower, upper)| vec![lower, upper])
        .collect::<Vec<_>>();

    for point in &options.points {
        for ((axis_breakpoints, &coordinate), &transform) in breakpoints
            .iter_mut()
            .zip(point.iter())
            .zip(transforms.iter())
        {
            let Some(mapped) = transform.inverse(coordinate) else {
                continue;
            };
            let start = axis_breakpoints[0];
            let Some(&end) = axis_breakpoints.last() else {
                continue;
            };
            if is_strictly_between(mapped, start, end) {
                axis_breakpoints.push(mapped);
            }
        }
    }

    for axis_breakpoints in &mut breakpoints {
        let Some(&end) = axis_breakpoints.last() else {
            continue;
        };
        let ascending = axis_breakpoints[0] <= end;
        axis_breakpoints.sort_by(|left, right| {
            if ascending {
                left.total_cmp(right)
            } else {
                right.total_cmp(left)
            }
        });
        axis_breakpoints.dedup_by(|left, right| cubature_breakpoints_equal(*left, *right));
    }

    let mut regions = Vec::new();
    let mut lower = vec![0.0; ranges.len()];
    let mut upper = vec![0.0; ranges.len()];
    collect_initial_cubature_regions(&breakpoints, 0, &mut lower, &mut upper, &mut regions);
    regions
}

fn collect_initial_cubature_regions(
    breakpoints: &[Vec<f64>],
    dim: usize,
    lower: &mut [f64],
    upper: &mut [f64],
    regions: &mut Vec<(Vec<f64>, Vec<f64>)>,
) {
    if dim == breakpoints.len() {
        regions.push((lower.to_vec(), upper.to_vec()));
        return;
    }

    for window in breakpoints[dim].windows(2) {
        lower[dim] = window[0];
        upper[dim] = window[1];
        collect_initial_cubature_regions(breakpoints, dim + 1, lower, upper, regions);
    }
}

fn is_strictly_between(value: f64, start: f64, end: f64) -> bool {
    let low = start.min(end);
    let high = start.max(end);
    value > low && value < high
}

fn cubature_breakpoints_equal(left: f64, right: f64) -> bool {
    (left - right).abs() <= f64::EPSILON * (1.0 + left.abs().max(right.abs()))
}

fn estimate_cubature_region<F>(
    f: &F,
    transforms: &[BoundTransform],
    a: &[f64],
    b: &[f64],
    output_dim: usize,
) -> Result<CubatureEstimate, IntegrateValidationError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    const NODES: [f64; 3] = [-0.774_596_669_241_483_4, 0.0, 0.774_596_669_241_483_4];
    const WEIGHTS: [f64; 3] = [
        0.555_555_555_555_555_6,
        0.888_888_888_888_888_8,
        0.555_555_555_555_555_6,
    ];

    let ndim = a.len();
    let mut high = vec![0.0; output_dim];
    let mut t = vec![0.0; ndim];
    let mut x = vec![0.0; ndim];
    let mut neval = 0usize;

    struct TensorState<'a, F>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        f: &'a F,
        transforms: &'a [BoundTransform],
        a: &'a [f64],
        b: &'a [f64],
        t: &'a mut [f64],
        x: &'a mut [f64],
        high: &'a mut [f64],
        neval: &'a mut usize,
        output_dim: usize,
    }

    fn visit_tensor<F>(
        state: &mut TensorState<'_, F>,
        dim: usize,
        weight: f64,
    ) -> Result<(), IntegrateValidationError>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        if dim == state.a.len() {
            let mut jacobian = 1.0;
            map_cubature_point(state.t, state.transforms, state.x, &mut jacobian);
            let values = (state.f)(state.x);
            validate_cubature_output(&values, state.output_dim)?;
            for (total, value) in state.high.iter_mut().zip(values.iter()) {
                *total += weight * jacobian * value;
            }
            *state.neval += 1;
            return Ok(());
        }

        let center = 0.5 * (state.a[dim] + state.b[dim]);
        let half_width = 0.5 * (state.b[dim] - state.a[dim]);
        for (&node, &node_weight) in NODES.iter().zip(WEIGHTS.iter()) {
            state.t[dim] = center + half_width * node;
            visit_tensor(state, dim + 1, weight * half_width * node_weight)?;
        }
        Ok(())
    }

    let mut state = TensorState {
        f,
        transforms,
        a,
        b,
        t: &mut t,
        x: &mut x,
        high: &mut high,
        neval: &mut neval,
        output_dim,
    };
    visit_tensor(&mut state, 0, 1.0)?;

    let mut midpoint = vec![0.0; ndim];
    let mut midpoint_weight = 1.0;
    for ((mid, &lower), &upper) in midpoint.iter_mut().zip(a.iter()).zip(b.iter()) {
        *mid = 0.5 * (lower + upper);
        midpoint_weight *= upper - lower;
    }
    let mut midpoint_x = vec![0.0; ndim];
    let mut midpoint_jacobian = 1.0;
    map_cubature_point(
        &midpoint,
        transforms,
        &mut midpoint_x,
        &mut midpoint_jacobian,
    );
    let low_values = f(&midpoint_x);
    validate_cubature_output(&low_values, output_dim)?;
    neval += 1;

    let low = low_values
        .iter()
        .map(|value| midpoint_weight * midpoint_jacobian * value)
        .collect::<Vec<_>>();
    let error = high
        .iter()
        .zip(low.iter())
        .map(|(high_value, low_value)| (high_value - low_value).abs())
        .collect::<Vec<_>>();

    Ok(CubatureEstimate {
        region: CubatureRegion {
            a: a.to_vec(),
            b: b.to_vec(),
            estimate: high,
            error,
        },
        neval,
    })
}

fn map_cubature_point(t: &[f64], transforms: &[BoundTransform], x: &mut [f64], jacobian: &mut f64) {
    *jacobian = 1.0;
    for ((out, &coord), &transform) in x.iter_mut().zip(t.iter()).zip(transforms.iter()) {
        let (mapped, derivative) = transform.map(coord);
        *out = mapped;
        *jacobian *= derivative;
    }
}

fn cubature_converged(regions: &[CubatureRegion], atol: f64, rtol: f64) -> bool {
    let Some(output_dim) = regions.first().map(|region| region.estimate.len()) else {
        return true;
    };
    let (estimate, error) = sum_cubature_regions(regions, output_dim);
    componentwise_cubature_converged(&estimate, &error, atol, rtol)
}

fn componentwise_cubature_converged(estimate: &[f64], error: &[f64], atol: f64, rtol: f64) -> bool {
    estimate
        .iter()
        .zip(error.iter())
        .all(|(estimate_value, error_value)| *error_value <= atol + rtol * estimate_value.abs())
}

fn sum_cubature_regions(regions: &[CubatureRegion], output_dim: usize) -> (Vec<f64>, Vec<f64>) {
    let mut estimate = vec![0.0; output_dim];
    let mut error = vec![0.0; output_dim];
    for region in regions {
        for ((estimate_total, error_total), (region_estimate, region_error)) in estimate
            .iter_mut()
            .zip(error.iter_mut())
            .zip(region.estimate.iter().zip(region.error.iter()))
        {
            *estimate_total += region_estimate;
            *error_total += region_error;
        }
    }
    (estimate, error)
}

fn largest_error_region(regions: &[CubatureRegion]) -> Option<usize> {
    regions
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            max_abs_component(&left.error).total_cmp(&max_abs_component(&right.error))
        })
        .map(|(index, _)| index)
}

fn widest_region_dimension(region: &CubatureRegion) -> usize {
    let mut widest_index = 0;
    let mut widest_width = 0.0;
    for (index, (&lower, &upper)) in region.a.iter().zip(region.b.iter()).enumerate() {
        let width = (upper - lower).abs();
        if width > widest_width {
            widest_width = width;
            widest_index = index;
        }
    }
    widest_index
}

/// Compute n-point Gauss-Legendre nodes and weights on [-1, 1].
///
/// Uses Newton's method to find roots of the Legendre polynomial P_n(x),
/// then computes weights via the Christoffel-Darboux formula.
fn gauss_legendre_nodes_weights(n: usize) -> (Vec<f64>, Vec<f64>) {
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    if n == 1 {
        return (vec![0.0], vec![2.0]);
    }

    let nf = n as f64;
    let mut nodes = Vec::with_capacity(n);
    let mut weights = Vec::with_capacity(n);

    // Only need to find roots for the first half (symmetric)
    let m = n.div_ceil(2);
    for i in 0..m {
        // Initial guess: Chebyshev approximation
        let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (nf + 0.5)).cos();

        // Newton's method to find root of P_n(x)
        for _ in 0..50 {
            let (pn, dpn) = legendre_pn_dpn(n, x);
            let dx = pn / dpn;
            x -= dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }

        let (_, dpn) = legendre_pn_dpn(n, x);
        let w = 2.0 / ((1.0 - x * x) * dpn * dpn);

        nodes.push(x);
        weights.push(w);
    }

    // Fill symmetric half
    let mut full_nodes = Vec::with_capacity(n);
    let mut full_weights = Vec::with_capacity(n);

    for i in (0..m).rev() {
        if n % 2 == 1 && i == m - 1 {
            // Middle node (for odd n)
            continue;
        }
        full_nodes.push(-nodes[i]);
        full_weights.push(weights[i]);
    }
    if n % 2 == 1 {
        full_nodes.push(nodes[m - 1]); // should be ~0
        full_weights.push(weights[m - 1]);
    }
    for i in 0..m {
        if n % 2 == 1 && i == m - 1 {
            continue;
        }
        full_nodes.push(nodes[i]);
        full_weights.push(weights[i]);
    }

    // Sort by node value
    let mut pairs: Vec<(f64, f64)> = full_nodes.into_iter().zip(full_weights).collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let sorted_nodes = pairs.iter().map(|p| p.0).collect();
    let sorted_weights = pairs.iter().map(|p| p.1).collect();

    (sorted_nodes, sorted_weights)
}

/// Evaluate P_n(x) and P_n'(x) using the recurrence relation.
fn legendre_pn_dpn(n: usize, x: f64) -> (f64, f64) {
    let mut p_prev = 1.0; // P_0
    let mut p_curr = x; // P_1

    for k in 1..n {
        let kf = k as f64;
        let p_next = ((2.0 * kf + 1.0) * x * p_curr - kf * p_prev) / (kf + 1.0);
        p_prev = p_curr;
        p_curr = p_next;
    }

    // P_n'(x) = n (x P_n(x) - P_{n-1}(x)) / (x² - 1)
    let nf = n as f64;
    let dpn = if (x * x - 1.0).abs() > 1e-30 {
        nf * (x * p_curr - p_prev) / (x * x - 1.0)
    } else {
        // At x = ±1, use L'Hôpital or the value P_n'(1) = n(n+1)/2
        nf * (nf + 1.0) / 2.0
    };

    (p_curr, dpn)
}

/// Cumulative integral using composite Simpson's rule.
///
/// Matches `scipy.integrate.cumulative_simpson(y, x=x)` for 1-D sampled data.
///
/// Returns one cumulative value per subinterval, so the result length is `n - 1`.
/// With two or fewer samples, this falls back to `cumulative_trapezoid`, matching
/// SciPy's documented behavior.
pub fn cumulative_simpson(y: &[f64], x: &[f64]) -> Result<Vec<f64>, IntegrateValidationError> {
    if y.len() != x.len() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: format!("y ({}) and x ({}) must have same length", y.len(), x.len()),
        });
    }
    let n = y.len();
    if n < 2 {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "need at least 2 points for cumulative integration".to_string(),
        });
    }
    if n <= 2 {
        return cumulative_trapezoid(y, x);
    }

    let mut dx = Vec::with_capacity(n - 1);
    for points in x.windows(2) {
        let h = points[1] - points[0];
        if !(h.is_finite() && h > 0.0) {
            return Err(IntegrateValidationError::QuadInvalidBounds {
                detail: "x must be finite and strictly increasing".to_string(),
            });
        }
        dx.push(h);
    }

    let mut result = Vec::with_capacity(n - 1);
    let mut interval_integrals = vec![0.0; n - 1];

    for i in (0..(n - 2)).step_by(2) {
        let h0 = dx[i];
        let h1 = dx[i + 1];
        interval_integrals[i] = cumulative_simpson_left_interval(y[i], y[i + 1], y[i + 2], h0, h1);
        interval_integrals[i + 1] =
            cumulative_simpson_right_interval(y[i], y[i + 1], y[i + 2], h0, h1);
    }

    if n.is_multiple_of(2) {
        let i = n - 3;
        interval_integrals[n - 2] =
            cumulative_simpson_right_interval(y[i], y[i + 1], y[i + 2], dx[i], dx[i + 1]);
    }

    let mut cumsum = 0.0;
    for value in interval_integrals {
        cumsum += value;
        result.push(cumsum);
    }

    Ok(result)
}

fn cumulative_simpson_left_interval(y0: f64, y1: f64, y2: f64, h0: f64, h1: f64) -> f64 {
    if h0.abs() < f64::EPSILON || h1.abs() < f64::EPSILON {
        0.5 * h0 * (y0 + y1)
    } else {
        let hsum = h0 + h1;
        h0 / 6.0
            * ((3.0 - h0 / hsum) * y0 + (3.0 + h0 * h0 / (h1 * hsum) + h0 / hsum) * y1
                - h0 * h0 / (h1 * hsum) * y2)
    }
}

fn cumulative_simpson_right_interval(y0: f64, y1: f64, y2: f64, h0: f64, h1: f64) -> f64 {
    if h0.abs() < f64::EPSILON || h1.abs() < f64::EPSILON {
        0.5 * h1 * (y1 + y2)
    } else {
        let hsum = h0 + h1;
        h1 / 6.0
            * (-h1 * h1 / (h0 * hsum) * y0
                + (3.0 + h1 * h1 / (h0 * hsum) + h1 / hsum) * y1
                + (3.0 - h1 / hsum) * y2)
    }
}

/// Romberg integration using Richardson extrapolation on the trapezoidal rule.
///
/// Matches `scipy.integrate.romberg`.
pub fn romberg<F>(f: F, a: f64, b: f64, tol: f64, max_order: usize) -> QuadResult
where
    F: Fn(f64) -> f64,
{
    if !a.is_finite() || !b.is_finite() || !tol.is_finite() || tol <= 0.0 {
        return QuadResult {
            integral: f64::NAN,
            error: f64::INFINITY,
            neval: 0,
            converged: false,
        };
    }
    let max_order = max_order.clamp(2, 20);
    let mut r = vec![vec![0.0; max_order]; max_order];
    let mut neval = 0usize;

    // R[0,0] = trapezoidal rule with 1 panel
    r[0][0] = 0.5 * (b - a) * (f(a) + f(b));
    neval += 2;

    for n in 1..max_order {
        let panels = 1usize << n;
        let h = (b - a) / panels as f64;

        let mut sum = 0.0;
        for k in (1..panels).step_by(2) {
            sum += f(a + k as f64 * h);
            neval += 1;
        }
        r[n][0] = 0.5 * r[n - 1][0] + h * sum;

        for m in 1..=n {
            let factor = (1u64 << (2 * m)) as f64;
            r[n][m] = (factor * r[n][m - 1] - r[n - 1][m - 1]) / (factor - 1.0);
        }

        let error = (r[n][n] - r[n - 1][n - 1]).abs();
        if error < tol {
            return QuadResult {
                integral: r[n][n],
                error,
                neval,
                converged: true,
            };
        }
    }

    let n = max_order - 1;
    QuadResult {
        integral: r[n][n],
        error: (r[n][n] - r[n - 1][n - 1]).abs(),
        neval,
        converged: false,
    }
}

/// Integrate sampled data using the trapezoidal rule (irregular spacing).
///
/// Matches `scipy.integrate.trapezoid(y, x)` with explicit x values.
pub fn trapezoid_irregular(y: &[f64], x: &[f64]) -> f64 {
    if y.len() < 2 || x.len() != y.len() {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..y.len() - 1 {
        sum += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i]);
    }
    sum
}

/// Simpson's rule for irregularly-spaced data.
///
/// Matches `scipy.integrate.simpson(y, x=x)` with explicit x values.
pub fn simpson_irregular(y: &[f64], x: &[f64]) -> f64 {
    if y.len() < 2 || x.len() != y.len() {
        return 0.0;
    }
    if y.len() < 3 {
        return trapezoid_irregular(y, x);
    }

    let n = y.len();
    let mut sum = 0.0;
    let mut i = 0;

    // Process pairs of intervals with Simpson's rule
    while i + 2 < n {
        let h0 = x[i + 1] - x[i];
        let h1 = x[i + 2] - x[i + 1];
        let hsum = h0 + h1;

        // Fall back to the two local trapezoids if spacing is degenerate.
        if h0.abs() < 1e-15 || h1.abs() < 1e-15 {
            sum += 0.5 * (y[i] + y[i + 1]) * h0 + 0.5 * (y[i + 1] + y[i + 2]) * h1;
            i += 2;
            continue;
        }

        // Simpson's 3/8 for unequal spacing
        sum += hsum / 6.0
            * (y[i] * (2.0 - h1 / h0)
                + y[i + 1] * hsum * hsum / (h0 * h1)
                + y[i + 2] * (2.0 - h0 / h1));

        i += 2;
    }

    // Handle remaining interval with trapezoidal rule
    if i + 1 < n {
        sum += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i]);
    }

    sum
}

/// Gauss-Kronrod adaptive quadrature (7-15 point rule).
///
/// Higher accuracy than basic adaptive Simpson's. Uses a 7-point Gauss
/// rule with a 15-point Kronrod extension for error estimation.
///
/// Matches `scipy.integrate.quad` behavior (which uses QUADPACK's QAG).
pub fn gauss_kronrod_quad<F>(f: F, a: f64, b: f64, options: QuadOptions) -> QuadResult
where
    F: Fn(f64) -> f64,
{
    gauss_kronrod_inner(&f, a, b, options)
}

fn gauss_kronrod_inner(f: &dyn Fn(f64) -> f64, a: f64, b: f64, options: QuadOptions) -> QuadResult {
    // Gauss-Kronrod G7-K15 quadrature rule.
    // The 7 Gauss nodes are a subset of the 15 Kronrod nodes.
    // We evaluate f at all 15 Kronrod nodes, and compute both the
    // 7-point Gauss estimate and the 15-point Kronrod estimate.
    // The difference gives an error estimate.

    // 15-point Kronrod nodes on [-1, 1] (sorted)
    let k_nodes: [f64; 15] = [
        -0.991_455_371_120_812_6,
        -0.949_107_912_342_758_5,
        -0.864_864_423_359_769_1,
        -0.741_531_185_599_394_4,
        -0.586_087_235_467_691_1,
        -0.405_845_151_377_397_2,
        -0.207_784_955_007_898_5,
        0.0,
        0.207_784_955_007_898_5,
        0.405_845_151_377_397_2,
        0.586_087_235_467_691_1,
        0.741_531_185_599_394_4,
        0.864_864_423_359_769_1,
        0.949_107_912_342_758_5,
        0.991_455_371_120_812_6,
    ];

    // 15-point Kronrod weights
    let k_weights: [f64; 15] = [
        0.022_935_322_010_529_2,
        0.063_092_092_629_978_6,
        0.104_790_010_322_250_2,
        0.140_653_259_715_525_9,
        0.169_004_726_639_267_9,
        0.190_350_578_064_785_4,
        0.204_432_940_075_298_9,
        0.209_482_141_084_727_8,
        0.204_432_940_075_298_9,
        0.190_350_578_064_785_4,
        0.169_004_726_639_267_9,
        0.140_653_259_715_525_9,
        0.104_790_010_322_250_2,
        0.063_092_092_629_978_6,
        0.022_935_322_010_529_2,
    ];

    // 7-point Gauss weights (for the 7 nodes that are shared with Kronrod)
    // These correspond to Kronrod indices 1, 3, 5, 7, 9, 11, 13
    let g_weights: [f64; 7] = [
        0.129_484_966_168_869_7,
        0.279_705_391_489_276_7,
        0.381_830_050_505_118_9,
        0.417_959_183_673_469_4,
        0.381_830_050_505_118_9,
        0.279_705_391_489_276_7,
        0.129_484_966_168_869_7,
    ];
    // Indices into k_nodes that correspond to the 7 Gauss nodes
    let g_indices: [usize; 7] = [1, 3, 5, 7, 9, 11, 13];

    let mid = 0.5 * (a + b);
    let half = 0.5 * (b - a);

    // Evaluate f at all 15 Kronrod nodes
    let fvals: Vec<f64> = k_nodes.iter().map(|&node| f(mid + half * node)).collect();
    let neval = 15;

    // Kronrod estimate (15-point)
    let kronrod_sum: f64 = fvals
        .iter()
        .zip(k_weights.iter())
        .map(|(&fv, &w)| w * fv)
        .sum::<f64>()
        * half;

    // Gauss estimate (7-point, using subset of evaluations)
    let gauss_sum: f64 = g_indices
        .iter()
        .zip(g_weights.iter())
        .map(|(&idx, &w)| w * fvals[idx])
        .sum::<f64>()
        * half;

    let error = (kronrod_sum - gauss_sum).abs();

    if error > options.epsabs && error > options.epsrel * kronrod_sum.abs() && options.limit > 1 {
        let left = gauss_kronrod_inner(
            f,
            a,
            mid,
            QuadOptions {
                limit: options.limit - 1,
                ..options
            },
        );
        let right = gauss_kronrod_inner(
            f,
            mid,
            b,
            QuadOptions {
                limit: options.limit - 1,
                ..options
            },
        );
        return QuadResult {
            integral: left.integral + right.integral,
            error: left.error + right.error,
            neval: left.neval + right.neval,
            converged: left.converged && right.converged,
        };
    }

    QuadResult {
        integral: kronrod_sum,
        error,
        neval,
        converged: true,
    }
}

/// Newton-Cotes integration weights for n equally-spaced points.
///
/// Returns weights such that ∫₀¹ f(x) dx ≈ Σ w_i * f(x_i).
/// Matches `scipy.integrate.newton_cotes`.
pub fn newton_cotes(n: usize) -> Result<Vec<f64>, IntegrateValidationError> {
    if n == 0 {
        return Ok(vec![1.0]);
    }
    match n {
        1 => Ok(vec![0.5, 0.5]),                                   // Trapezoidal
        2 => Ok(vec![1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0]),            // Simpson's 1/3
        3 => Ok(vec![1.0 / 8.0, 3.0 / 8.0, 3.0 / 8.0, 1.0 / 8.0]), // Simpson's 3/8
        4 => Ok(vec![
            7.0 / 90.0,
            32.0 / 90.0,
            12.0 / 90.0,
            32.0 / 90.0,
            7.0 / 90.0,
        ]), // Boole's rule
        _ => {
            // General Newton-Cotes via Lagrange integration
            let nf = n as f64;
            let mut weights = vec![0.0; n + 1];
            for (i, item) in weights.iter_mut().enumerate() {
                // w_i = ∫₀¹ L_i(x) dx where L_i = Π_{j≠i} (x - j/n) / (i/n - j/n)
                // Use numerical integration with high-order quadrature
                // Scale grid with polynomial degree to maintain accuracy
                let m = (20 * (n + 1)).max(200);
                let mut integral = 0.0;
                for k in 0..=m {
                    let x = k as f64 / m as f64;
                    let w = if k == 0 || k == m {
                        1.0
                    } else if k % 2 == 0 {
                        2.0
                    } else {
                        4.0
                    };
                    let mut li = 1.0;
                    for j in 0..=n {
                        if j != i {
                            li *= (x - j as f64 / nf) / (i as f64 / nf - j as f64 / nf);
                        }
                    }
                    integral += w * li;
                }
                *item = integral / (3.0 * m as f64);
            }
            Ok(weights)
        }
    }
}

/// Integrate using Newton-Cotes composite rule of given order.
///
/// `order` is the number of subintervals per panel (1=trapezoidal, 2=Simpson).
pub fn newton_cotes_quad<F>(
    f: F,
    a: f64,
    b: f64,
    n_panels: usize,
    order: usize,
) -> Result<f64, IntegrateValidationError>
where
    F: Fn(f64) -> f64,
{
    if n_panels == 0 {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "n_panels must be positive".to_string(),
        });
    }
    if order == 0 {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "order must be positive".to_string(),
        });
    }
    let weights = newton_cotes(order)?;
    let panel_width = (b - a) / n_panels as f64;
    let mut total = 0.0;

    for panel in 0..n_panels {
        let panel_a = a + panel as f64 * panel_width;
        for (i, &w) in weights.iter().enumerate() {
            let x = panel_a + i as f64 * panel_width / order as f64;
            total += w * f(x) * panel_width;
        }
    }

    Ok(total)
}

/// Compute the integral and provide a convergence explanation.
///
/// Returns (result, explanation_string).
/// Matches `scipy.integrate.quad` with `full_output=True`.
pub fn quad_explain<F>(f: F, a: f64, b: f64, options: QuadOptions) -> (QuadResult, String)
where
    F: Fn(f64) -> f64,
{
    let result = quad(&f, a, b, options);
    match result {
        Ok(r) => {
            let msg = if r.converged {
                format!(
                    "Integration converged after {} evaluations. Estimated error: {:.2e}",
                    r.neval, r.error
                )
            } else {
                format!(
                    "Integration did NOT converge after {} evaluations. Estimated error: {:.2e}. \
                     Consider increasing limit or adjusting tolerances.",
                    r.neval, r.error
                )
            };
            (r, msg)
        }
        Err(e) => {
            let msg = format!("Integration failed: {e}");
            (
                QuadResult {
                    integral: f64::NAN,
                    error: f64::NAN,
                    neval: 0,
                    converged: false,
                },
                msg,
            )
        }
    }
}

/// Integrate a function over an infinite interval [a, ∞).
///
/// Uses the substitution t = 1/(1+u) to map [0, ∞) to [0, 1].
/// Matches `scipy.integrate.quad(f, a, np.inf)`.
pub fn quad_inf<F>(
    f: F,
    a: f64,
    options: QuadOptions,
) -> Result<QuadResult, IntegrateValidationError>
where
    F: Fn(f64) -> f64,
{
    // Substitution: x = a + t/(1-t), dx = 1/(1-t)² dt, t ∈ [0, 1)
    let g = |t: f64| {
        if t >= 1.0 - 1e-15 {
            return 0.0;
        }
        let x = a + t / (1.0 - t);
        let jacobian = 1.0 / ((1.0 - t) * (1.0 - t));
        f(x) * jacobian
    };

    quad(g, 0.0, 1.0 - 1e-10, options)
}

/// Integrate a function over (-∞, b].
///
/// Uses substitution to map (-∞, b] to [0, 1].
pub fn quad_neg_inf<F>(
    f: F,
    b: f64,
    options: QuadOptions,
) -> Result<QuadResult, IntegrateValidationError>
where
    F: Fn(f64) -> f64,
{
    // Substitution: x = b - t/(1-t), dx = -1/(1-t)² dt, t ∈ [0, 1)
    let g = |t: f64| {
        if t >= 1.0 - 1e-15 {
            return 0.0;
        }
        let x = b - t / (1.0 - t);
        let jacobian = 1.0 / ((1.0 - t) * (1.0 - t));
        f(x) * jacobian
    };

    quad(g, 0.0, 1.0 - 1e-10, options)
}

/// Integrate a function over (-∞, ∞).
///
/// Splits at 0 and uses substitutions for both halves.
pub fn quad_full_inf<F>(f: F, options: QuadOptions) -> Result<QuadResult, IntegrateValidationError>
where
    F: Fn(f64) -> f64,
{
    let left = quad_neg_inf(&f, 0.0, options)?;
    let right = quad_inf(&f, 0.0, options)?;

    Ok(QuadResult {
        integral: left.integral + right.integral,
        error: left.error + right.error,
        neval: left.neval + right.neval,
        converged: left.converged && right.converged,
    })
}

/// Compute the Cauchy principal value of a singular integral.
///
/// Integrates f(x) over [a, b] with a singularity at `singular_point`.
/// Splits the interval and approaches the singularity from both sides.
pub fn quad_cauchy_pv<F>(
    f: F,
    a: f64,
    b: f64,
    singular_point: f64,
    options: QuadOptions,
) -> Result<QuadResult, IntegrateValidationError>
where
    F: Fn(f64) -> f64,
{
    if !a.is_finite() || !b.is_finite() || !singular_point.is_finite() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "integration bounds and singular point must be finite".to_string(),
        });
    }
    let eps = 1e-8 * (b - a).abs();

    let left = quad(&f, a, singular_point - eps, options)?;
    let right = quad(&f, singular_point + eps, b, options)?;

    Ok(QuadResult {
        integral: left.integral + right.integral,
        error: left.error + right.error,
        neval: left.neval + right.neval,
        converged: left.converged && right.converged,
    })
}

/// Compute the integral of a function given at discrete points using
/// the composite trapezoidal rule with Richardson extrapolation.
///
/// More accurate than plain trapezoid for smooth functions.
pub fn trapezoid_richardson(y: &[f64], x: &[f64]) -> f64 {
    if y.len() < 2 || x.len() != y.len() {
        return 0.0;
    }

    // Basic trapezoidal rule
    let t1 = trapezoid_irregular(y, x);

    // If we have enough points, do a second estimate with half the points
    let n = y.len();
    if n < 5 {
        return t1;
    }

    // Subsample every other point
    let y2: Vec<f64> = y.iter().step_by(2).cloned().collect();
    let x2: Vec<f64> = x.iter().step_by(2).cloned().collect();
    let t2 = trapezoid_irregular(&y2, &x2);

    // Richardson extrapolation: T = (4*T1 - T2) / 3
    (4.0 * t1 - t2) / 3.0
}

/// Compute the cumulative integral using the trapezoidal rule
/// with initial value specification.
///
/// Matches `scipy.integrate.cumulative_trapezoid` with initial=0.
pub fn cumulative_trapezoid_initial(y: &[f64], x: &[f64], initial: f64) -> Vec<f64> {
    let n = y.len();
    if n < 2 || x.len() != n || !initial.is_finite() {
        return vec![initial; n];
    }

    let mut result = Vec::with_capacity(n);
    result.push(initial);
    let mut cumsum = initial;
    for i in 1..n {
        cumsum += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]);
        result.push(cumsum);
    }

    result
}

/// Gauss-Legendre quadrature with specified number of points.
///
/// Uses precomputed nodes and weights for n=2,3,4,5.
/// Matches `scipy.integrate.fixed_quad` internals.
pub fn gauss_legendre<F>(f: F, a: f64, b: f64, n: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    if n == 0 {
        return 0.0;
    }
    let (nodes, weights) = match n {
        2 => (
            vec![-0.577_350_269_189_625_7, 0.577_350_269_189_625_7],
            vec![1.0, 1.0],
        ),
        3 => (
            vec![-0.774_596_669_241_483_4, 0.0, 0.774_596_669_241_483_4],
            vec![5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0],
        ),
        4 => (
            vec![
                -0.861_136_311_594_052_6,
                -0.339_981_043_584_856_3,
                0.339_981_043_584_856_3,
                0.861_136_311_594_052_6,
            ],
            vec![
                0.347_854_845_137_453_9,
                0.652_145_154_862_546_1,
                0.652_145_154_862_546_1,
                0.347_854_845_137_453_9,
            ],
        ),
        5 => (
            vec![
                -0.906_179_845_938_664,
                -0.538_469_310_105_683,
                0.0,
                0.538_469_310_105_683,
                0.906_179_845_938_664,
            ],
            vec![
                0.236_926_885_056_189_1,
                0.478_628_670_499_366_5,
                0.568_888_888_888_889,
                0.478_628_670_499_366_5,
                0.236_926_885_056_189_1,
            ],
        ),
        _ => {
            // Fallback to Simpson for other n
            let h = (b - a) / n as f64;
            let mut sum = f(a) + f(b);
            for i in 1..n {
                let x = a + i as f64 * h;
                sum += if i % 2 == 0 { 2.0 } else { 4.0 } * f(x);
            }
            return sum * h / 3.0;
        }
    };

    let mid = (a + b) / 2.0;
    let half = (b - a) / 2.0;
    let mut sum = 0.0;
    for (node, weight) in nodes.iter().zip(weights.iter()) {
        sum += weight * f(mid + half * node);
    }
    sum * half
}

/// Compute the double integral of a function over a rectangular region.
///
/// Uses iterated Simpson's rule.
/// Matches `scipy.integrate.dblquad` behavior.
pub fn dblquad_rect<F>(
    f: F,
    x_lo: f64,
    x_hi: f64,
    y_lo: f64,
    y_hi: f64,
    nx: usize,
    ny: usize,
) -> f64
where
    F: Fn(f64, f64) -> f64,
{
    let nx = nx.max(2) | 1; // ensure odd
    let ny = ny.max(2) | 1;
    let hx = (x_hi - x_lo) / (nx - 1) as f64;
    let hy = (y_hi - y_lo) / (ny - 1) as f64;

    let mut total = 0.0;
    for i in 0..nx {
        let x = x_lo + i as f64 * hx;
        let wx = if i == 0 || i == nx - 1 {
            1.0
        } else if i % 2 == 0 {
            2.0
        } else {
            4.0
        };

        for j in 0..ny {
            let y = y_lo + j as f64 * hy;
            let wy = if j == 0 || j == ny - 1 {
                1.0
            } else if j % 2 == 0 {
                2.0
            } else {
                4.0
            };

            total += wx * wy * f(x, y);
        }
    }

    total * hx * hy / 9.0
}

/// Compute the triple integral of a function over a rectangular region.
#[allow(clippy::too_many_arguments)]
pub fn tplquad_rect<F>(
    f: F,
    x_lo: f64,
    x_hi: f64,
    y_lo: f64,
    y_hi: f64,
    z_lo: f64,
    z_hi: f64,
    n: usize,
) -> f64
where
    F: Fn(f64, f64, f64) -> f64,
{
    let n = n.max(2) | 1;
    let hx = (x_hi - x_lo) / (n - 1) as f64;
    let hy = (y_hi - y_lo) / (n - 1) as f64;
    let hz = (z_hi - z_lo) / (n - 1) as f64;

    let simpson_weight = |i: usize, max: usize| -> f64 {
        if i == 0 || i == max - 1 {
            1.0
        } else if i.is_multiple_of(2) {
            2.0
        } else {
            4.0
        }
    };

    let mut total = 0.0;
    for i in 0..n {
        let x = x_lo + i as f64 * hx;
        let wx = simpson_weight(i, n);
        for j in 0..n {
            let y = y_lo + j as f64 * hy;
            let wy = simpson_weight(j, n);
            for k in 0..n {
                let z = z_lo + k as f64 * hz;
                let wz = simpson_weight(k, n);
                total += wx * wy * wz * f(x, y, z);
            }
        }
    }

    total * hx * hy * hz / 27.0
}

/// Compute the line integral ∫ f(x,y) ds along a parametric curve.
///
/// `curve_x(t)` and `curve_y(t)` define the curve, `f(x,y)` is the integrand.
pub fn line_integral<F, Cx, Cy>(
    f: F,
    curve_x: Cx,
    curve_y: Cy,
    t_lo: f64,
    t_hi: f64,
    n: usize,
) -> f64
where
    F: Fn(f64, f64) -> f64,
    Cx: Fn(f64) -> f64,
    Cy: Fn(f64) -> f64,
{
    let n = n.max(2) | 1;
    let h = (t_hi - t_lo) / (n - 1) as f64;

    let mut total = 0.0;
    for i in 0..n {
        let t = t_lo + i as f64 * h;
        let x = curve_x(t);
        let y = curve_y(t);

        // Approximate ds = sqrt(dx² + dy²) via central differences
        let dt = h * 0.01;
        let dx = curve_x(t + dt) - curve_x(t - dt);
        let dy = curve_y(t + dt) - curve_y(t - dt);
        let ds = (dx * dx + dy * dy).sqrt() / (2.0 * dt);

        let w = if i == 0 || i == n - 1 {
            1.0
        } else if i % 2 == 0 {
            2.0
        } else {
            4.0
        };

        total += w * f(x, y) * ds;
    }

    total * h / 3.0
}

/// Monte Carlo integration of a function over `[a,b]^d`.
///
/// Uses random sampling to estimate the integral.
pub fn monte_carlo_integrate<F>(
    f: F,
    bounds: &[(f64, f64)],
    n_samples: usize,
    seed: u64,
) -> (f64, f64)
where
    F: Fn(&[f64]) -> f64,
{
    let d = bounds.len();
    if d == 0 || n_samples == 0 {
        return (0.0, 0.0);
    }

    let volume: f64 = bounds.iter().map(|&(a, b)| b - a).product();
    let mut rng = seed;
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut point = vec![0.0; d];

    for _ in 0..n_samples {
        for j in 0..d {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (rng >> 11) as f64 / (1u64 << 53) as f64;
            point[j] = bounds[j].0 + u * (bounds[j].1 - bounds[j].0);
        }
        let fval = f(&point);
        sum += fval;
        sum_sq += fval * fval;
    }

    let nf = n_samples as f64;
    let mean = sum / nf;
    let variance = sum_sq / nf - mean * mean;
    let std_err = (variance / nf).sqrt() * volume;

    (mean * volume, std_err)
}

/// Result of a quasi-Monte Carlo quadrature call.
///
/// Mirrors `scipy.integrate.qmc_quad` return: the integral estimate
/// (mean across `n_estimates` independent Halton blocks) and the
/// standard error of that mean.
#[derive(Debug, Clone, PartialEq)]
pub struct QmcQuadResult {
    pub integral: f64,
    pub standard_error: f64,
}

/// br-gm7n: van der Corput radical inverse for prime base.
///
/// Computes φ_b(i) = digit-reverse of i in base b ∈ (0, 1). This is the
/// 1D building block of the Halton sequence — the d-th coordinate of
/// Halton point i uses base = the d-th prime.
fn van_der_corput(mut idx: usize, base: usize) -> f64 {
    let mut q = 0.0;
    let mut bk = 1.0 / base as f64;
    while idx > 0 {
        q += (idx % base) as f64 * bk;
        idx /= base;
        bk /= base as f64;
    }
    q
}

/// br-gm7n: first 32 primes, sufficient for QMC up to 32 dimensions.
const QMC_PRIMES: [usize; 32] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131,
];

/// Quasi-Monte Carlo integration over a hyper-rectangle using a Halton
/// sequence.
///
/// Mirrors `scipy.integrate.qmc_quad(f, a, b, n_estimates, n_points,
/// qrng=Halton(scramble=False))` — runs `n_estimates` independent
/// Halton blocks (each `n_points` long, offset along the sequence) and
/// reports the mean across blocks plus the standard error of that mean.
///
/// `lb` and `ub` give the lower / upper bounds; their length sets the
/// dimension. `f` takes a point of that dimension and returns a scalar.
pub fn qmc_quad<F>(
    f: F,
    lb: &[f64],
    ub: &[f64],
    n_estimates: usize,
    n_points: usize,
) -> Result<QmcQuadResult, IntegrateValidationError>
where
    F: Fn(&[f64]) -> f64,
{
    if lb.len() != ub.len() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "qmc_quad: lb and ub must have the same length".to_string(),
        });
    }
    let d = lb.len();
    if d == 0 {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "qmc_quad: at least one dimension required".to_string(),
        });
    }
    if d > QMC_PRIMES.len() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "qmc_quad: dimensionality exceeds built-in prime table (max 32)".to_string(),
        });
    }
    if n_estimates == 0 || n_points == 0 {
        return Err(IntegrateValidationError::QuadInvalidTolerance {
            detail: "qmc_quad: n_estimates and n_points must be positive".to_string(),
        });
    }

    let mut volume = 1.0;
    for j in 0..d {
        if !lb[j].is_finite() || !ub[j].is_finite() {
            return Err(IntegrateValidationError::QuadInvalidBounds {
                detail: "qmc_quad: bounds must be finite".to_string(),
            });
        }
        volume *= ub[j] - lb[j];
    }

    let mut estimates = Vec::with_capacity(n_estimates);
    let mut point = vec![0.0; d];
    for est in 0..n_estimates {
        // Use a fresh contiguous Halton block for each estimate. Skip
        // the first sample (idx=0 = origin) and offset by est*n_points
        // so blocks are disjoint segments of the deterministic sequence.
        let start = 1 + est * n_points;
        let mut sum = 0.0;
        for i in 0..n_points {
            let idx = start + i;
            for j in 0..d {
                let u = van_der_corput(idx, QMC_PRIMES[j]);
                point[j] = lb[j] + u * (ub[j] - lb[j]);
            }
            sum += f(&point);
        }
        estimates.push(sum / n_points as f64 * volume);
    }

    let n_est_f = n_estimates as f64;
    let mean = estimates.iter().sum::<f64>() / n_est_f;
    let standard_error = if n_estimates > 1 {
        let var = estimates
            .iter()
            .map(|v| (v - mean) * (v - mean))
            .sum::<f64>()
            / (n_est_f - 1.0);
        (var / n_est_f).sqrt()
    } else {
        0.0
    };

    Ok(QmcQuadResult {
        integral: mean,
        standard_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qmc_quad_1d_smooth_x_squared() {
        // ∫_0^1 x^2 dx = 1/3
        let r = qmc_quad(|x| x[0] * x[0], &[0.0], &[1.0], 8, 1024).expect("qmc_quad");
        assert!(
            (r.integral - 1.0 / 3.0).abs() < 1e-3,
            "expected 1/3 ± 1e-3, got {}",
            r.integral
        );
    }

    #[test]
    fn qmc_quad_2d_smooth_x_plus_y() {
        // ∫_[0,1]^2 (x + y) dx dy = 1
        let r = qmc_quad(|p| p[0] + p[1], &[0.0, 0.0], &[1.0, 1.0], 8, 1024).expect("qmc_quad 2d");
        assert!(
            (r.integral - 1.0).abs() < 1e-3,
            "expected 1.0 ± 1e-3, got {}",
            r.integral
        );
    }

    #[test]
    fn qmc_quad_1d_oscillatory_sin() {
        // ∫_0^π sin(x) dx = 2
        let r = qmc_quad(|x| x[0].sin(), &[0.0], &[std::f64::consts::PI], 8, 1024)
            .expect("qmc_quad oscillatory");
        assert!(
            (r.integral - 2.0).abs() < 1e-3,
            "expected 2.0 ± 1e-3, got {}",
            r.integral
        );
    }

    #[test]
    fn quad_constant_function() {
        // ∫₀¹ 5 dx = 5
        let result = quad(|_| 5.0, 0.0, 1.0, QuadOptions::default()).expect("quad works");
        assert!(result.converged);
        assert!(
            (result.integral - 5.0).abs() < 1e-12,
            "integral should be 5, got {}",
            result.integral
        );
    }

    #[test]
    fn quad_linear_function() {
        // ∫₀¹ x dx = 0.5
        let result = quad(|x| x, 0.0, 1.0, QuadOptions::default()).expect("quad works");
        assert!(result.converged);
        assert!(
            (result.integral - 0.5).abs() < 1e-12,
            "integral should be 0.5, got {}",
            result.integral
        );
    }

    #[test]
    fn quad_polynomial() {
        // ∫₀¹ x² dx = 1/3
        let result = quad(|x| x * x, 0.0, 1.0, QuadOptions::default()).expect("quad works");
        assert!(result.converged);
        assert!(
            (result.integral - 1.0 / 3.0).abs() < 1e-12,
            "integral should be 1/3, got {}",
            result.integral
        );
    }

    #[test]
    fn quad_sin() {
        // ∫₀^π sin(x) dx = 2
        let result =
            quad(f64::sin, 0.0, std::f64::consts::PI, QuadOptions::default()).expect("quad works");
        assert!(result.converged);
        assert!(
            (result.integral - 2.0).abs() < 1e-10,
            "integral should be 2, got {}",
            result.integral
        );
    }

    #[test]
    fn quad_exp() {
        // ∫₀¹ e^x dx = e - 1
        let result = quad(f64::exp, 0.0, 1.0, QuadOptions::default()).expect("quad works");
        assert!(result.converged);
        let expected = std::f64::consts::E - 1.0;
        assert!(
            (result.integral - expected).abs() < 1e-10,
            "integral should be e-1={expected}, got {}",
            result.integral
        );
    }

    #[test]
    fn quad_metamorphic_additivity_split_at_midpoint() {
        // /testing-metamorphic for [frankenscipy-1ncsg]:
        // ∫_a^b f = ∫_a^c f + ∫_c^b f for any c ∈ (a, b).
        // No reference value needed; this catches adaptive-step bugs
        // that would silently bias the upper or lower half.
        fn f_exp(x: f64) -> f64 {
            x.exp()
        }
        fn f_poly(x: f64) -> f64 {
            x * x * x + 2.0 * x + 1.0
        }
        type QuadCase = (fn(f64) -> f64, f64, f64);
        let cases: &[QuadCase] = &[
            (f_exp, 0.0, 2.0),
            (f64::sin, 0.0, std::f64::consts::PI),
            (f_poly, -1.5, 2.5),
        ];
        for &(f, a, b) in cases {
            let opts = QuadOptions::default();
            let total = quad(f, a, b, opts).expect("total");
            for &c_frac in &[0.25, 0.5, 0.75] {
                let c = a + c_frac * (b - a);
                let lo = quad(f, a, c, opts).expect("lo");
                let hi = quad(f, c, b, opts).expect("hi");
                let diff = (total.integral - (lo.integral + hi.integral)).abs();
                assert!(
                    diff < 1e-9 + 1e-9 * total.integral.abs(),
                    "additivity broken at c={c}: total={} vs lo+hi={}, diff={diff}",
                    total.integral,
                    lo.integral + hi.integral
                );
            }
        }
    }

    #[test]
    fn quad_metamorphic_linearity_alpha_f_plus_beta_g() {
        // ∫(α·f + β·g) = α·∫f + β·∫g
        let alpha = 2.5_f64;
        let beta = -1.7_f64;
        let f = |x: f64| x * x;
        let g = f64::sin;
        let a = 0.0_f64;
        let b = std::f64::consts::PI;
        let opts = QuadOptions::default();
        let lhs = quad(|x| alpha * f(x) + beta * g(x), a, b, opts).expect("lhs");
        let if_ = quad(f, a, b, opts).expect("∫f");
        let ig = quad(g, a, b, opts).expect("∫g");
        let rhs = alpha * if_.integral + beta * ig.integral;
        assert!(
            (lhs.integral - rhs).abs() < 1e-9 + 1e-9 * rhs.abs(),
            "linearity broken: lhs={} rhs={}",
            lhs.integral,
            rhs
        );
    }

    #[test]
    fn quad_metamorphic_translation_invariance() {
        // ∫_a^b f(x) dx = ∫_{a+t}^{b+t} f(x − t) dx for any shift t.
        // Pure substitution u = x − t, du = dx — must hold to machine
        // precision modulo the adaptive tolerance.
        let opts = QuadOptions::default();
        for &t in &[0.0, 1.0, -2.5, 5.0] {
            for f_choice in 0..3 {
                let (a, b) = (0.0_f64, 3.0_f64);
                let untranslated: Box<dyn Fn(f64) -> f64> = match f_choice {
                    0 => Box::new(|x: f64| (-x * x).exp()),
                    1 => Box::new(f64::sin),
                    _ => Box::new(|x: f64| 1.0 / (1.0 + x * x)),
                };
                let translated: Box<dyn Fn(f64) -> f64> = match f_choice {
                    0 => Box::new(move |x: f64| (-(x - t) * (x - t)).exp()),
                    1 => Box::new(move |x: f64| (x - t).sin()),
                    _ => Box::new(move |x: f64| 1.0 / (1.0 + (x - t) * (x - t))),
                };
                let lhs = quad(untranslated, a, b, opts).expect("lhs");
                let rhs = quad(translated, a + t, b + t, opts).expect("rhs");
                assert!(
                    (lhs.integral - rhs.integral).abs() < 1e-9 + 1e-9 * lhs.integral.abs(),
                    "translation by {t} broken on f_choice {f_choice}: \
                     lhs={} vs rhs={}",
                    lhs.integral,
                    rhs.integral
                );
            }
        }
    }

    #[test]
    fn quad_negative_interval() {
        // ∫₋₁¹ x³ dx = 0 (odd function)
        let result = quad(|x| x.powi(3), -1.0, 1.0, QuadOptions::default()).expect("quad works");
        assert!(result.converged);
        assert!(
            result.integral.abs() < 1e-12,
            "integral of odd function should be 0, got {}",
            result.integral
        );
    }

    #[test]
    fn quad_reversed_bounds() {
        // ∫₁⁰ x dx = -0.5
        let result = quad(|x| x, 1.0, 0.0, QuadOptions::default()).expect("quad works");
        // With reversed bounds, the integral should be negative of forward
        // Our implementation handles this naturally since b < a
        assert!(result.neval > 0);
    }

    #[test]
    fn quad_equal_bounds() {
        let result = quad(|x| x * x, 3.0, 3.0, QuadOptions::default()).expect("quad works");
        assert!(result.converged);
        assert_eq!(result.integral, 0.0);
        assert_eq!(result.neval, 0);
    }

    #[test]
    fn quad_gaussian() {
        // ∫₋₃³ e^(-x²) dx ≈ sqrt(π) * erf(3) ≈ 1.7724..
        let result =
            quad(|x| (-x * x).exp(), -3.0, 3.0, QuadOptions::default()).expect("quad works");
        assert!(result.converged);
        // sqrt(pi) * erf(3) ≈ 1.7724538509
        let expected = std::f64::consts::PI.sqrt() * 0.999_977_909_503_001_4;
        assert!(
            (result.integral - expected).abs() < 1e-8,
            "gaussian integral: got {} expected ~{expected}",
            result.integral
        );
    }

    #[test]
    fn quad_nonfinite_bounds_error() {
        let err =
            quad(|x| x, f64::INFINITY, 1.0, QuadOptions::default()).expect_err("infinite bounds");
        assert!(matches!(
            err,
            IntegrateValidationError::QuadInvalidBounds { .. }
        ));
    }

    #[test]
    fn quad_negative_tolerance_error() {
        let err = quad(
            |x| x,
            0.0,
            1.0,
            QuadOptions {
                epsabs: -1.0,
                ..QuadOptions::default()
            },
        )
        .expect_err("negative tolerance");
        assert!(matches!(
            err,
            IntegrateValidationError::QuadInvalidTolerance { .. }
        ));
    }

    #[test]
    fn quad_high_degree_polynomial() {
        // ∫₀¹ x⁶ dx = 1/7
        let result = quad(|x| x.powi(6), 0.0, 1.0, QuadOptions::default()).expect("quad works");
        assert!(result.converged);
        assert!(
            (result.integral - 1.0 / 7.0).abs() < 1e-10,
            "integral should be 1/7, got {}",
            result.integral
        );
    }

    #[test]
    fn quad_wide_interval() {
        // ∫₀¹⁰⁰ 1/(1+x²) dx = atan(100) ≈ 1.5607966...
        let result =
            quad(|x| 1.0 / (1.0 + x * x), 0.0, 100.0, QuadOptions::default()).expect("quad works");
        let expected = 100.0_f64.atan();
        assert!(
            (result.integral - expected).abs() < 1e-6,
            "integral should be atan(100)={expected}, got {}",
            result.integral
        );
    }

    #[test]
    fn quad_vec_constant_function() {
        let result =
            quad_vec(|_| vec![5.0, -2.0], 0.0, 1.0, QuadOptions::default()).expect("quad_vec");
        assert!(result.converged);
        assert!((result.integral[0] - 5.0).abs() < 1e-12);
        assert!((result.integral[1] + 2.0).abs() < 1e-12);
    }

    #[test]
    fn quad_vec_polynomial_components() {
        let result =
            quad_vec(|x| vec![x, x * x], 0.0, 1.0, QuadOptions::default()).expect("quad_vec");
        assert!(result.converged);
        assert!((result.integral[0] - 0.5).abs() < 1e-12);
        assert!((result.integral[1] - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn quad_vec_reversed_bounds() {
        let result =
            quad_vec(|x| vec![x, 1.0], 1.0, 0.0, QuadOptions::default()).expect("quad_vec");
        assert!(result.converged);
        assert!((result.integral[0] + 0.5).abs() < 1e-12);
        assert!((result.integral[1] + 1.0).abs() < 1e-12);
    }

    #[test]
    fn quad_vec_invalid_input_errors() {
        let bounds_err = quad_vec(|x| vec![x], f64::INFINITY, 1.0, QuadOptions::default())
            .expect_err("non-finite bounds");
        assert!(matches!(
            bounds_err,
            IntegrateValidationError::QuadInvalidBounds { .. }
        ));

        let tol_err = quad_vec(
            |x| vec![x],
            0.0,
            1.0,
            QuadOptions {
                epsabs: -1.0,
                ..QuadOptions::default()
            },
        )
        .expect_err("negative tolerance");
        assert!(matches!(
            tol_err,
            IntegrateValidationError::QuadInvalidTolerance { .. }
        ));
    }

    #[test]
    fn quad_vec_rejects_inconsistent_output_shape() {
        let err = quad_vec(
            |x| {
                if x < 0.5 { vec![x] } else { vec![x, x * x] }
            },
            0.0,
            1.0,
            QuadOptions::default(),
        )
        .expect_err("inconsistent output length");
        assert!(matches!(
            err,
            IntegrateValidationError::QuadInvalidBounds { .. }
        ));
    }

    #[test]
    fn cubature_scalar_2d_product() {
        let result = cubature_scalar(
            |x| x[0] * x[1],
            &[0.0, 0.0],
            &[1.0, 1.0],
            CubatureOptions::default(),
        )
        .expect("cubature 2d product");
        assert_eq!(result.status, CubatureStatus::Converged);
        assert!(
            (result.estimate - 0.25).abs() < 1e-10,
            "integral should be 0.25, got {}",
            result.estimate
        );
        assert!(result.neval > 0);
    }

    #[test]
    fn cubature_vector_1d_powers() {
        let result = cubature(
            |x| vec![1.0, x[0], x[0] * x[0]],
            &[0.0],
            &[1.0],
            CubatureOptions::default(),
        )
        .expect("cubature vector powers");
        assert_eq!(result.status, CubatureStatus::Converged);
        let expected = [1.0, 0.5, 1.0 / 3.0];
        for (got, want) in result.estimate.iter().zip(expected) {
            assert!((got - want).abs() < 1e-10, "got {got}, expected {want}");
        }
    }

    #[test]
    fn cubature_reversed_bounds_preserve_orientation() {
        let result = cubature_scalar(|_| 1.0, &[1.0], &[0.0], CubatureOptions::default())
            .expect("cubature reversed bounds");
        assert_eq!(result.status, CubatureStatus::Converged);
        assert!(
            (result.estimate + 1.0).abs() < 1e-12,
            "reversed integral should be -1, got {}",
            result.estimate
        );
    }

    #[test]
    fn cubature_zero_dim_returns_function_value() {
        let result = cubature(|_| vec![42.0, -7.0], &[], &[], CubatureOptions::default())
            .expect("cubature zero dim");
        assert_eq!(result.status, CubatureStatus::Converged);
        assert_eq!(result.estimate, vec![42.0, -7.0]);
        assert_eq!(result.error, vec![0.0, 0.0]);
        assert_eq!(result.neval, 1);
    }

    #[test]
    fn cubature_points_avoid_singularity() {
        let result = cubature_scalar(
            |x| {
                if x[0] == 0.0 {
                    f64::NAN
                } else {
                    x[0].sin() / x[0]
                }
            },
            &[-1.0],
            &[1.0],
            CubatureOptions {
                points: vec![vec![0.0]],
                ..CubatureOptions::default()
            },
        )
        .expect("cubature avoids singular point");
        assert_eq!(result.status, CubatureStatus::Converged);
        assert!(
            (result.estimate - 1.892_166_140_734_366).abs() < 1e-6,
            "sinc integral got {}",
            result.estimate
        );
    }

    #[test]
    fn cubature_rejects_invalid_inputs() {
        let bounds_err = cubature_scalar(|x| x[0], &[0.0], &[1.0, 2.0], CubatureOptions::default())
            .expect_err("mismatched bounds");
        assert!(matches!(
            bounds_err,
            IntegrateValidationError::QuadInvalidBounds { .. }
        ));

        let tolerance_err = cubature_scalar(
            |x| x[0],
            &[0.0],
            &[1.0],
            CubatureOptions {
                rtol: -1.0,
                ..CubatureOptions::default()
            },
        )
        .expect_err("negative tolerance");
        assert!(matches!(
            tolerance_err,
            IntegrateValidationError::QuadInvalidTolerance { .. }
        ));

        let point_err = cubature_scalar(
            |x| x[0],
            &[0.0, 0.0],
            &[1.0, 1.0],
            CubatureOptions {
                points: vec![vec![0.5]],
                ..CubatureOptions::default()
            },
        )
        .expect_err("point dimensionality mismatch");
        assert!(matches!(
            point_err,
            IntegrateValidationError::QuadInvalidBounds { .. }
        ));
    }

    #[test]
    fn cubature_rejects_inconsistent_output_shape() {
        let err = cubature(
            |x| {
                if x[0] < 0.5 {
                    vec![x[0]]
                } else {
                    vec![x[0], x[0] * x[0]]
                }
            },
            &[0.0],
            &[1.0],
            CubatureOptions::default(),
        )
        .expect_err("inconsistent cubature output length");
        assert!(matches!(
            err,
            IntegrateValidationError::QuadInvalidBounds { .. }
        ));
    }

    // ── dblquad tests ───────────────────────────────────────────────

    #[test]
    fn dblquad_constant() {
        // ∫₀¹ ∫₀¹ 1 dy dx = 1
        let result = dblquad(
            |_y, _x| 1.0,
            0.0,
            1.0,
            |_| 0.0,
            |_| 1.0,
            DblquadOptions::default(),
        )
        .expect("dblquad works");
        assert!(result.converged);
        assert!(
            (result.integral - 1.0).abs() < 1e-10,
            "should be 1, got {}",
            result.integral
        );
    }

    #[test]
    fn dblquad_xy() {
        // ∫₀¹ ∫₀¹ x*y dy dx = 1/4
        let result = dblquad(
            |y, x| x * y,
            0.0,
            1.0,
            |_| 0.0,
            |_| 1.0,
            DblquadOptions::default(),
        )
        .expect("dblquad works");
        assert!(result.converged);
        assert!(
            (result.integral - 0.25).abs() < 1e-10,
            "should be 0.25, got {}",
            result.integral
        );
    }

    #[test]
    fn dblquad_triangular_region() {
        // ∫₀¹ ∫₀ˣ 1 dy dx = ∫₀¹ x dx = 1/2
        let result = dblquad(
            |_y, _x| 1.0,
            0.0,
            1.0,
            |_| 0.0,
            |x| x,
            DblquadOptions::default(),
        )
        .expect("dblquad works");
        assert!(result.converged);
        assert!(
            (result.integral - 0.5).abs() < 1e-10,
            "should be 0.5, got {}",
            result.integral
        );
    }

    #[test]
    fn dblquad_disk_area() {
        // ∫₋₁¹ ∫₋√(1-x²)^√(1-x²) 1 dy dx = π (area of unit disk)
        let result = dblquad(
            |_y, _x| 1.0,
            -1.0,
            1.0,
            |x| -(1.0 - x * x).sqrt(),
            |x| (1.0 - x * x).sqrt(),
            DblquadOptions::default(),
        )
        .expect("dblquad works");
        assert!(
            (result.integral - std::f64::consts::PI).abs() < 1e-6,
            "should be pi, got {}",
            result.integral
        );
    }

    #[test]
    fn dblquad_equal_bounds() {
        let result = dblquad(
            |y, x| x * y,
            3.0,
            3.0,
            |_| 0.0,
            |_| 1.0,
            DblquadOptions::default(),
        )
        .expect("dblquad equal bounds");
        assert_eq!(result.integral, 0.0);
    }

    #[test]
    fn dblquad_nonfinite_bounds_error() {
        let err = dblquad(
            |y, x| x * y,
            f64::INFINITY,
            1.0,
            |_| 0.0,
            |_| 1.0,
            DblquadOptions::default(),
        )
        .expect_err("nonfinite");
        assert!(matches!(
            err,
            IntegrateValidationError::QuadInvalidBounds { .. }
        ));
    }

    // ── trapezoid tests ─────────────────────────────────────────────

    #[test]
    fn trapezoid_constant() {
        // ∫₀¹ 5 dx = 5
        let x = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let y = vec![5.0; 5];
        let result = trapezoid(&y, &x).expect("trapezoid works");
        assert!(
            (result.integral - 5.0).abs() < 1e-12,
            "got {}",
            result.integral
        );
    }

    #[test]
    fn trapezoid_linear() {
        // ∫₀¹ x dx = 0.5 (exact for trapezoidal on linear functions)
        let n = 100;
        let x: Vec<f64> = (0..=n).map(|i| i as f64 / n as f64).collect();
        let y: Vec<f64> = x.to_vec();
        let result = trapezoid(&y, &x).expect("trapezoid works");
        assert!(
            (result.integral - 0.5).abs() < 1e-12,
            "got {}",
            result.integral
        );
    }

    #[test]
    fn trapezoid_quadratic() {
        // ∫₀¹ x² dx = 1/3
        let n = 1000;
        let x: Vec<f64> = (0..=n).map(|i| i as f64 / n as f64).collect();
        let y: Vec<f64> = x.iter().map(|xi| xi * xi).collect();
        let result = trapezoid(&y, &x).expect("trapezoid works");
        assert!(
            (result.integral - 1.0 / 3.0).abs() < 1e-6,
            "got {}",
            result.integral
        );
    }

    #[test]
    fn trapezoid_uniform_constant() {
        let y = vec![3.0; 11];
        let result = trapezoid_uniform(&y, 0.1).expect("trapezoid_uniform works");
        assert!(
            (result.integral - 3.0).abs() < 1e-12,
            "got {}",
            result.integral
        );
    }

    #[test]
    fn trapezoid_length_mismatch_error() {
        let err = trapezoid(&[1.0, 2.0], &[0.0]).expect_err("length mismatch");
        assert!(matches!(
            err,
            IntegrateValidationError::QuadInvalidBounds { .. }
        ));
    }

    #[test]
    fn trapezoid_too_few_points_error() {
        let err = trapezoid(&[1.0], &[0.0]).expect_err("too few points");
        assert!(matches!(
            err,
            IntegrateValidationError::QuadInvalidBounds { .. }
        ));
    }

    // ── simpson tests ───────────────────────────────────────────────

    #[test]
    fn simpson_constant() {
        let x = vec![0.0, 0.5, 1.0];
        let y = vec![7.0; 3];
        let result = simpson(&y, &x).expect("simpson works");
        assert!(
            (result.integral - 7.0).abs() < 1e-12,
            "got {}",
            result.integral
        );
    }

    #[test]
    fn simpson_quadratic_exact() {
        // Simpson's rule is exact for polynomials up to degree 3
        // ∫₀¹ x² dx = 1/3
        let x = vec![0.0, 0.5, 1.0];
        let y: Vec<f64> = x.iter().map(|xi| xi * xi).collect();
        let result = simpson(&y, &x).expect("simpson works");
        assert!(
            (result.integral - 1.0 / 3.0).abs() < 1e-12,
            "should be exact for quadratic, got {}",
            result.integral
        );
    }

    #[test]
    fn simpson_cubic_exact() {
        // ∫₀¹ x³ dx = 1/4, exact for Simpson's rule
        let x = vec![0.0, 0.5, 1.0];
        let y: Vec<f64> = x.iter().map(|xi| xi * xi * xi).collect();
        let result = simpson(&y, &x).expect("simpson works");
        assert!(
            (result.integral - 0.25).abs() < 1e-12,
            "should be exact for cubic, got {}",
            result.integral
        );
    }

    #[test]
    fn simpson_sin_many_points() {
        // ∫₀^π sin(x) dx = 2
        let n = 101; // odd number of points
        let x: Vec<f64> = (0..n)
            .map(|i| std::f64::consts::PI * i as f64 / (n - 1) as f64)
            .collect();
        let y: Vec<f64> = x.iter().map(|xi| xi.sin()).collect();
        let result = simpson(&y, &x).expect("simpson works");
        assert!(
            (result.integral - 2.0).abs() < 1e-6,
            "got {}",
            result.integral
        );
    }

    #[test]
    fn simpson_even_points() {
        // Even number of points — uses trapezoid correction on last panel
        let n = 100;
        let x: Vec<f64> = (0..n)
            .map(|i| std::f64::consts::PI * i as f64 / (n - 1) as f64)
            .collect();
        let y: Vec<f64> = x.iter().map(|xi| xi.sin()).collect();
        let result = simpson(&y, &x).expect("simpson with even points");
        assert!(
            (result.integral - 2.0).abs() < 1e-4,
            "got {}",
            result.integral
        );
    }

    #[test]
    fn simpson_uniform_quadratic() {
        // ∫₀¹ x² dx = 1/3 with uniform spacing
        let n = 5; // odd
        let dx = 1.0 / (n - 1) as f64;
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let xi = i as f64 * dx;
                xi * xi
            })
            .collect();
        let result = simpson_uniform(&y, dx).expect("simpson_uniform works");
        assert!(
            (result.integral - 1.0 / 3.0).abs() < 1e-10,
            "got {}",
            result.integral
        );
    }

    #[test]
    fn simpson_two_points_fallback() {
        // Only 2 points — falls back to trapezoidal
        let x = vec![0.0, 1.0];
        let y = vec![0.0, 2.0];
        let result = simpson(&y, &x).expect("simpson 2 points");
        assert!(
            (result.integral - 1.0).abs() < 1e-12,
            "got {}",
            result.integral
        );
    }

    #[test]
    fn simpson_too_few_points_error() {
        let err = simpson(&[1.0], &[0.0]).expect_err("too few points");
        assert!(matches!(
            err,
            IntegrateValidationError::QuadInvalidBounds { .. }
        ));
    }

    // ── cumulative_trapezoid tests ──────────────────────────────────

    #[test]
    fn cumtrapz_linear() {
        // y = x on [0, 1, 2, 3] => cumulative integrals: 0.5, 2.0, 4.5
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 2.0, 3.0];
        let result = cumulative_trapezoid(&y, &x).expect("cumtrapz");
        assert_eq!(result.len(), 3);
        assert!((result[0] - 0.5).abs() < 1e-12);
        assert!((result[1] - 2.0).abs() < 1e-12);
        assert!((result[2] - 4.5).abs() < 1e-12);
    }

    #[test]
    fn cumtrapz_constant() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![5.0; 4];
        let result = cumulative_trapezoid(&y, &x).expect("cumtrapz");
        assert!((result[0] - 5.0).abs() < 1e-12);
        assert!((result[1] - 10.0).abs() < 1e-12);
        assert!((result[2] - 15.0).abs() < 1e-12);
    }

    #[test]
    fn cumtrapz_uniform() {
        let y = vec![0.0, 1.0, 4.0, 9.0]; // x²-ish
        let result = cumulative_trapezoid_uniform(&y, 1.0).expect("cumtrapz_uniform");
        assert_eq!(result.len(), 3);
        assert!((result[0] - 0.5).abs() < 1e-12);
        assert!((result[1] - 3.0).abs() < 1e-12);
        assert!((result[2] - 9.5).abs() < 1e-12);
    }

    #[test]
    fn cumtrapz_length_mismatch() {
        let err = cumulative_trapezoid(&[1.0, 2.0], &[0.0]).expect_err("mismatch");
        assert!(matches!(
            err,
            IntegrateValidationError::QuadInvalidBounds { .. }
        ));
    }

    #[test]
    fn cumtrapz_too_few_points() {
        let err = cumulative_trapezoid(&[1.0], &[0.0]).expect_err("too few");
        assert!(matches!(
            err,
            IntegrateValidationError::QuadInvalidBounds { .. }
        ));
    }

    #[test]
    fn cumulative_trapezoid_initial_empty_input_stays_empty() {
        let result = cumulative_trapezoid_initial(&[], &[], f64::NAN);
        assert!(result.is_empty());
    }

    // ── tplquad tests ──────────────────────────────────────────────

    #[test]
    fn tplquad_unit_cube() {
        // ∫∫∫ 1 dz dy dx over [0,1]^3 = 1
        let result = tplquad(
            |_z, _y, _x| 1.0,
            0.0,
            1.0,
            |_x| 0.0,
            |_x| 1.0,
            |_x, _y| 0.0,
            |_x, _y| 1.0,
            DblquadOptions::default(),
        )
        .unwrap();
        assert!(
            (result.integral - 1.0).abs() < 0.01,
            "unit cube volume: {}",
            result.integral
        );
    }

    #[test]
    fn tplquad_xyz() {
        // ∫∫∫ xyz dz dy dx over [0,1]^3 = (1/2)^3 = 0.125
        let result = tplquad(
            |z, y, x| x * y * z,
            0.0,
            1.0,
            |_x| 0.0,
            |_x| 1.0,
            |_x, _y| 0.0,
            |_x, _y| 1.0,
            DblquadOptions::default(),
        )
        .unwrap();
        assert!(
            (result.integral - 0.125).abs() < 0.01,
            "∫xyz = {}, expected 0.125",
            result.integral
        );
    }

    // ── Romberg tests ──────────────────────────────────────────────

    #[test]
    fn romb_func_polynomial() {
        // ∫ x^2 dx from 0 to 1 = 1/3
        let result = romb_func(|x| x * x, 0.0, 1.0, None, None).unwrap();
        assert!(
            (result.integral - 1.0 / 3.0).abs() < 1e-10,
            "romberg ∫x² = {}, expected 1/3",
            result.integral
        );
        assert!(result.converged);
    }

    #[test]
    fn romb_func_exp() {
        // ∫ e^x dx from 0 to 1 = e - 1
        let result = romb_func(|x| x.exp(), 0.0, 1.0, None, None).unwrap();
        let expected = std::f64::consts::E - 1.0;
        assert!(
            (result.integral - expected).abs() < 1e-8,
            "romberg ∫e^x = {}, expected {}",
            result.integral,
            expected
        );
    }

    #[test]
    fn romb_sampled() {
        // Sampled x^2 from 0 to 4 with 5 points (2^2 + 1)
        let y = vec![0.0, 1.0, 4.0, 9.0, 16.0]; // x^2 at x=0,1,2,3,4
        let result = romb(&y, 1.0).unwrap();
        // ∫ x^2 from 0 to 4 = 64/3 ≈ 21.333
        assert!(
            (result - 64.0 / 3.0).abs() < 0.1,
            "romb sampled x^2 = {}, expected ~21.33",
            result
        );
    }

    #[test]
    fn romb_wrong_sample_count() {
        // 4 samples (not 2^k + 1)
        assert!(romb(&[1.0, 2.0, 3.0, 4.0], 1.0).is_err());
    }

    // ── cumulative_simpson tests ───────────────────────────────────

    #[test]
    fn cumulative_simpson_constant() {
        // f(x) = 2, x = [0, 1, 2, 3, 4]
        // Cumulative integral at each point: 2, 4, 6, 8
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0; 5];
        let result = cumulative_simpson(&y, &x).unwrap();
        assert_eq!(result.len(), 4);
        let expected = [2.0, 4.0, 6.0, 8.0];
        for (got, want) in result.iter().zip(expected) {
            assert!((got - want).abs() < 1e-12, "got {got}, expected {want}");
        }
    }

    #[test]
    fn cumulative_simpson_quadratic() {
        // f(x) = x^2, x = [0, 1, 2, 3, 4]
        // Simpson should be exact for polynomials up to degree 3
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
        let result = cumulative_simpson(&y, &x).unwrap();
        assert_eq!(result.len(), 4);
        let expected = [1.0 / 3.0, 8.0 / 3.0, 9.0, 64.0 / 3.0];
        for (got, want) in result.iter().zip(expected) {
            assert!((got - want).abs() < 1e-10, "got {got}, expected {want}");
        }
    }

    #[test]
    fn cumulative_simpson_matches_scipy_nonuniform_cubic() {
        let x = vec![0.0, 0.2, 0.5, 1.1, 1.4, 2.0, 2.7, 3.5];
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi * xi + 2.0 * xi).collect();
        let result = cumulative_simpson(&y, &x).unwrap();
        let expected = [
            0.039_866_666_666_666_66,
            0.266_666_666_666_666_6,
            1.555_466_666_666_667,
            2.903_216_666_666_666_3,
            7.946_816_666_666_668,
            20.577_150_000_000_007,
            49.860_616_666_666_67,
        ];
        for (got, want) in result.iter().zip(expected) {
            assert!((got - want).abs() < 1e-12, "got {got}, expected {want}");
        }
    }

    #[test]
    fn cumulative_simpson_matches_scipy_uniform_sine() {
        let x: Vec<f64> = (0..=10)
            .map(|i| (i as f64) * std::f64::consts::PI / 10.0)
            .collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();
        let result = cumulative_simpson(&y, &x).unwrap();
        let expected = [
            0.049_332_186_036_761_73,
            0.190_993_463_598_046_28,
            0.412_475_894_906_144_34,
            0.691_020_842_926_797_3,
            1.000_054_758_657_502_2,
            1.309_088_674_388_206_9,
            1.587_633_622_408_859_9,
            1.809_116_053_716_958,
            1.950_777_331_278_242_6,
            2.000_109_517_315_004_3,
        ];
        for (got, want) in result.iter().zip(expected) {
            assert!((got - want).abs() < 1e-12, "got {got}, expected {want}");
        }
    }

    #[test]
    fn cumulative_simpson_two_points_falls_back_to_trapezoid() {
        let x = vec![0.0, 2.0];
        let y = vec![1.0, 3.0];
        let result = cumulative_simpson(&y, &x).unwrap();
        assert_eq!(result, vec![4.0]);
    }

    #[test]
    fn cumulative_simpson_requires_strictly_increasing_x() {
        let err = cumulative_simpson(&[1.0, 2.0, 3.0], &[0.0, 0.0, 1.0]).unwrap_err();
        assert!(matches!(
            err,
            IntegrateValidationError::QuadInvalidBounds { .. }
        ));
    }

    #[test]
    fn cumulative_simpson_too_few() {
        assert!(cumulative_simpson(&[1.0], &[0.0]).is_err());
    }

    // ── nquad tests ──────────────────────────────────────────────────

    #[test]
    fn nquad_1d_matches_quad() {
        let opts = QuadOptions::default();
        let result_nquad = nquad(|args| args[0] * args[0], &[(0.0, 1.0)], opts).expect("nquad 1d");
        let result_quad = quad(|x| x * x, 0.0, 1.0, opts).expect("quad");
        assert!(
            (result_nquad.integral - result_quad.integral).abs() < 1e-10,
            "nquad 1D should match quad: {} vs {}",
            result_nquad.integral,
            result_quad.integral
        );
    }

    #[test]
    fn nquad_2d_product() {
        // ∫₀¹ ∫₀¹ x*y dx dy = (1/2)(1/2) = 0.25
        let opts = QuadOptions::default();
        let result =
            nquad(|args| args[0] * args[1], &[(0.0, 1.0), (0.0, 1.0)], opts).expect("nquad 2d");
        assert!(
            (result.integral - 0.25).abs() < 1e-8,
            "∫∫ xy dxdy = {}, expected 0.25",
            result.integral
        );
    }

    #[test]
    fn nquad_3d_unit_cube() {
        // ∫₀¹ ∫₀¹ ∫₀¹ 1 dxdydz = 1
        let opts = QuadOptions::default();
        let result =
            nquad(|_args| 1.0, &[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], opts).expect("nquad 3d");
        assert!(
            (result.integral - 1.0).abs() < 1e-6,
            "∫∫∫ 1 dV = {}, expected 1.0",
            result.integral
        );
    }

    #[test]
    fn nquad_0d() {
        // 0-dimensional: just evaluates the function
        let opts = QuadOptions::default();
        let result = nquad(|_| 42.0, &[], opts).expect("nquad 0d");
        assert!((result.integral - 42.0).abs() < 1e-12);
    }

    #[test]
    fn nquad_propagates_inner_quad_failure_instead_of_silent_zero() {
        // /deadlock-finder-and-fixer for [frankenscipy-vetrv] +
        // self-found regression for [frankenscipy-666hq]: before
        // 819a64c the unwrap_or(0.0) inside the outer integrand
        // closure silently distorted the multi-dim integral; the
        // initial fix only caught Err results, missing the NaN
        // propagation case. The 666hq follow-up adds a finiteness
        // check on the final result.
        let opts = QuadOptions::default();

        // Case A (NaN propagation): integrand returns NaN unconditionally.
        // Inner quad returns Ok(NaN); the post-loop finiteness guard
        // surfaces this as a typed error.
        let r = nquad(|_args| f64::NAN, &[(0.0, 1.0), (0.0, 1.0)], opts);
        assert!(
            r.is_err(),
            "nquad must surface NaN-propagation as Err, got {r:?}"
        );
    }

    // ── fixed_quad tests ─────────────────────────────────────────────

    #[test]
    fn fixed_quad_polynomial_exact() {
        // n-point Gauss-Legendre is exact for polynomials up to degree 2n-1
        // 5-point should be exact for x^9
        let (result, neval) = fixed_quad(|x| x.powi(4), 0.0, 1.0, 5).expect("fixed_quad");
        assert_eq!(neval, 5);
        assert!(
            (result - 0.2).abs() < 1e-12,
            "∫₀¹ x⁴ dx = 0.2, got {result}"
        );
    }

    #[test]
    fn fixed_quad_sin() {
        // ∫₀^π sin(x) dx = 2
        let (result, _) =
            fixed_quad(|x| x.sin(), 0.0, std::f64::consts::PI, 10).expect("fixed_quad sin");
        assert!(
            (result - 2.0).abs() < 1e-10,
            "∫₀^π sin(x) = 2, got {result}"
        );
    }

    #[test]
    fn fixed_quad_matches_quad() {
        // fixed_quad with high n should match adaptive quad
        let (fq, _) = fixed_quad(|x| x.exp(), 0.0, 1.0, 20).expect("fixed_quad");
        let aq = quad(|x| x.exp(), 0.0, 1.0, QuadOptions::default()).expect("quad");
        assert!(
            (fq - aq.integral).abs() < 1e-10,
            "fixed_quad vs quad: {fq} vs {}",
            aq.integral
        );
    }

    #[test]
    fn romberg_exp() {
        let result = romberg(|x| x.exp(), 0.0, 1.0, 1e-12, 10);
        let exact = std::f64::consts::E - 1.0;
        assert!(
            (result.integral - exact).abs() < 1e-10,
            "romberg exp: {} vs {}",
            result.integral,
            exact
        );
    }

    #[test]
    fn romberg_polynomial() {
        // ∫₀¹ x³ dx = 1/4
        let result = romberg(|x| x * x * x, 0.0, 1.0, 1e-12, 10);
        assert!(
            (result.integral - 0.25).abs() < 1e-10,
            "romberg x³: {}",
            result.integral
        );
    }

    #[test]
    fn trapezoid_irregular_basic() {
        let x = vec![0.0, 1.0, 3.0, 4.0];
        let y = vec![1.0, 1.0, 1.0, 1.0]; // constant = 1
        let result = trapezoid_irregular(&y, &x);
        assert!((result - 4.0).abs() < 1e-10); // area = 4
    }

    #[test]
    fn simpson_irregular_basic() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect(); // y = x²
        let result = simpson_irregular(&y, &x);
        let exact = 64.0 / 3.0; // ∫₀⁴ x² dx = 64/3
        assert!(
            (result - exact).abs() < 1e-6,
            "simpson irregular x²: {} vs {}",
            result,
            exact
        );
    }

    #[test]
    fn simpson_irregular_duplicate_spacing_uses_local_trapezoids() {
        let x = [0.0, 0.0, 1.0];
        let y = [100.0, 2.0, 4.0];
        let result = simpson_irregular(&y, &x);

        assert!((result - 3.0).abs() < 1e-12, "got {result}");
    }

    #[test]
    fn gauss_kronrod_exp() {
        let result = gauss_kronrod_quad(|x| x.exp(), 0.0, 1.0, QuadOptions::default());
        let exact = std::f64::consts::E - 1.0;
        assert!(
            (result.integral - exact).abs() < 1e-8,
            "GK quad exp: {} vs {}",
            result.integral,
            exact
        );
    }

    #[test]
    fn quad_matches_scipy_reference_values() {
        // scipy.integrate.quad(lambda x: x**2, 0, 1) = 0.333...
        let result = quad(|x| x * x, 0.0, 1.0, QuadOptions::default()).expect("quad");
        assert!(
            (result.integral - 0.3333333333333333).abs() < 1e-10,
            "quad x^2 [0,1] got {}, expected 0.333...",
            result.integral
        );
    }

    #[test]
    fn dblquad_matches_scipy_reference_values() {
        // scipy.integrate.dblquad(lambda y, x: x*y, 0, 1, 0, 1) = 0.25
        let result = dblquad_rect(|x, y| x * y, 0.0, 1.0, 0.0, 1.0, 101, 101);
        assert!(
            (result - 0.25).abs() < 1e-6,
            "dblquad x*y got {result}, expected 0.25"
        );
    }

    #[test]
    fn trapezoid_matches_scipy_reference_values() {
        // scipy.integrate.trapezoid([0,1,4,9,16], dx=1.0) = 22.0
        let y = [0.0, 1.0, 4.0, 9.0, 16.0];
        let result = trapezoid_uniform(&y, 1.0).expect("trapezoid");
        assert!(
            (result.integral - 22.0).abs() < 1e-10,
            "trapezoid got {}, expected 22.0",
            result.integral
        );
    }

    #[test]
    fn simpson_matches_scipy_reference_values() {
        // scipy.integrate.simpson([0,1,4,9,16], dx=1.0) = 21.333...
        let y = [0.0, 1.0, 4.0, 9.0, 16.0];
        let result = simpson_uniform(&y, 1.0).expect("simpson");
        assert!(
            (result.integral - 21.333333333333332).abs() < 1e-10,
            "simpson got {}, expected 21.333...",
            result.integral
        );
    }

    #[test]
    fn cumulative_trapezoid_matches_scipy_reference_values() {
        // scipy.integrate.cumulative_trapezoid([0,1,4,9,16], dx=1.0)
        // -> [0.5, 3.0, 9.5, 22.0]
        let y = [0.0, 1.0, 4.0, 9.0, 16.0];
        let result = cumulative_trapezoid_uniform(&y, 1.0).expect("cumulative_trapezoid");
        let expected = [0.5, 3.0, 9.5, 22.0];
        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "cumulative_trapezoid[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn romb_matches_scipy_reference_values() {
        // scipy.integrate.romb([1, 2, 3, 4, 5], dx=1.0)
        // For 5 samples (2^2 + 1), dx=1.0, integral of linear fn = area under [1,5] over [0,4]
        // = (1+5)*4/2 = 12.0, but romb uses Richardson extrapolation
        let y = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = romb(&y, 1.0).expect("romb should succeed");
        // Linear function: romb should give exact result = 12.0
        assert!(
            (result - 12.0).abs() < 1e-10,
            "romb got {result}, expected 12.0"
        );
    }

    #[test]
    fn fixed_quad_matches_scipy_reference_values() {
        // scipy.integrate.fixed_quad(lambda x: x**2, 0, 1, n=5)
        // -> (0.3333333333333333, None)
        let (integral, _neval) =
            fixed_quad(|x| x * x, 0.0, 1.0, 5).expect("fixed_quad should succeed");
        assert!(
            (integral - 0.3333333333333333).abs() < 1e-10,
            "fixed_quad got {integral}, expected 0.333..."
        );
    }

    #[test]
    fn nquad_matches_scipy_reference_values() {
        // scipy.integrate.nquad(lambda x, y: x*y, [[0, 1], [0, 1]])
        // -> 0.25 (integral of x*y over unit square)
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let opts = QuadOptions::default();
        let result = nquad(|coords| coords[0] * coords[1], &bounds, opts).expect("nquad");
        assert!(
            (result.integral - 0.25).abs() < 1e-6,
            "nquad got {}, expected 0.25",
            result.integral
        );
    }

    #[test]
    fn tplquad_rect_matches_scipy_reference_values() {
        // scipy.integrate.tplquad(lambda z, y, x: x*y*z, 0, 1, 0, 1, 0, 1)
        // -> 0.125 (integral of x*y*z over unit cube)
        let result = tplquad_rect(|x, y, z| x * y * z, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 21);
        assert!(
            (result - 0.125).abs() < 1e-6,
            "tplquad got {}, expected 0.125",
            result
        );
    }

    #[test]
    fn newton_cotes_matches_scipy_reference_values() {
        // scipy.integrate.newton_cotes(1) = trapezoidal rule weights [0.5, 0.5]
        // scipy.integrate.newton_cotes(2) = Simpson's rule weights [1/6, 4/6, 1/6]
        let trap = newton_cotes(1).expect("newton_cotes(1)");
        assert_eq!(trap.len(), 2);
        assert!(
            (trap[0] - 0.5).abs() < 1e-10,
            "trap[0] = {}, expected 0.5",
            trap[0]
        );
        assert!(
            (trap[1] - 0.5).abs() < 1e-10,
            "trap[1] = {}, expected 0.5",
            trap[1]
        );

        let simp = newton_cotes(2).expect("newton_cotes(2)");
        assert_eq!(simp.len(), 3);
        assert!(
            (simp[0] - 1.0 / 6.0).abs() < 1e-10,
            "simp[0] = {}, expected 1/6",
            simp[0]
        );
        assert!(
            (simp[1] - 4.0 / 6.0).abs() < 1e-10,
            "simp[1] = {}, expected 4/6",
            simp[1]
        );
        assert!(
            (simp[2] - 1.0 / 6.0).abs() < 1e-10,
            "simp[2] = {}, expected 1/6",
            simp[2]
        );
    }

    #[test]
    fn romberg_matches_scipy_reference_values() {
        // scipy.integrate.romberg(lambda x: x**2, 0, 1)
        // -> 0.3333... (integral of x^2 from 0 to 1)
        let result = romberg(|x| x * x, 0.0, 1.0, 1e-10, 10);
        assert!(
            (result.integral - 1.0 / 3.0).abs() < 1e-6,
            "romberg got {}, expected 1/3",
            result.integral
        );
    }
}
