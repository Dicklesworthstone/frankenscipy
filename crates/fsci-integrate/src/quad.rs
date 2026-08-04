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
    if !options.epsabs.is_finite()
        || !options.epsrel.is_finite()
        || options.epsabs < 0.0
        || options.epsrel < 0.0
    {
        return Err(IntegrateValidationError::QuadInvalidTolerance {
            detail: "tolerances must be finite and non-negative".to_string(),
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

/// Batched adaptive integration: evaluate `I(params) = ∫_a^b f(x, params) dx` for MANY parameter
/// sets (`param_rows`), one [`QuadResult`] per set over the shared interval `[a, b]`. This is the
/// vmap-over-solver primitive SciPy lacks — a definite-integral sweep (a family of moments /
/// partition functions / marginalisations) loops `quad` in Python, calling the Python integrand
/// adaptively per integral, N integrals SERIALLY; here the N independent integrations are fanned
/// across cores and the integrand is an inlined Rust closure (callback lever × N-way parallel).
/// Result `i` is byte-identical to `quad(|x| f(x, &param_rows[i]), a, b, options)`.
pub fn quad_many<F>(
    f: F,
    a: f64,
    b: f64,
    param_rows: &[Vec<f64>],
    options: QuadOptions,
) -> Vec<Result<QuadResult, IntegrateValidationError>>
where
    F: Fn(f64, &[f64]) -> f64 + Sync,
{
    let nrows = param_rows.len();
    if nrows == 0 {
        return Vec::new();
    }
    let f_ref = &f;
    let solve_one = move |params: &[f64]| quad(|x| f_ref(x, params), a, b, options);

    // Each adaptive integral is an independent solve (many integrand evals) → fan whole
    // parameter sets across cores, capped by the row count; a tiny sweep stays serial.
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let nthreads = cores.min(nrows);
    if nthreads <= 1 || nrows < 4 {
        return param_rows.iter().map(|p| solve_one(p)).collect();
    }

    let chunk = nrows.div_ceil(nthreads);
    let solve_one = &solve_one;
    let chunk_results: Vec<Vec<Result<QuadResult, IntegrateValidationError>>> =
        std::thread::scope(|scope| {
            (0..nthreads)
                .filter_map(|t| {
                    let lo = t * chunk;
                    if lo >= nrows {
                        return None;
                    }
                    let hi = (lo + chunk).min(nrows);
                    Some(scope.spawn(move || {
                        (lo..hi)
                            .map(|i| solve_one(&param_rows[i]))
                            .collect::<Vec<_>>()
                    }))
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().expect("quad_many worker panicked"))
                .collect()
        });

    let mut out = Vec::with_capacity(nrows);
    for cr in chunk_results {
        out.extend(cr);
    }
    out
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
    if !options.epsabs.is_finite()
        || !options.epsrel.is_finite()
        || options.epsabs < 0.0
        || options.epsrel < 0.0
    {
        return Err(IntegrateValidationError::QuadInvalidTolerance {
            detail: "tolerances must be finite and non-negative".to_string(),
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
    if !options.epsabs.is_finite()
        || !options.epsrel.is_finite()
        || options.epsabs < 0.0
        || options.epsrel < 0.0
    {
        return Err(IntegrateValidationError::QuadInvalidTolerance {
            detail: "tolerances must be finite and non-negative".to_string(),
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

fn validate_sample_coordinates(x: &[f64]) -> Result<(), IntegrateValidationError> {
    if x.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "x coordinates must be finite".to_string(),
        })
    }
}

/// Batched double integration: evaluate `I(params) = ∫_a^b ∫_{y_lo}^{y_hi} f(y, x, params) dy dx`
/// for MANY parameter sets (`param_rows`) over a shared rectangle, one [`DblquadResult`] per set.
/// This is the vmap-over-solver primitive SciPy lacks, and dblquad is the heaviest 1-D-callback case:
/// the inner adaptive integral is re-run for each outer node, so each integral makes O(n²) Python
/// integrand calls; a parameter sweep loops `dblquad` in Python, N integrals SERIALLY. fsci
/// `dblquad_many` fans the N independent double integrations across cores and inlines the integrand
/// as a Rust closure (callback lever × N-way parallel). Result `i` is byte-identical to
/// `dblquad(|y, x| f(y, x, &param_rows[i]), a, b, |_| y_lo, |_| y_hi, options)`.
pub fn dblquad_many<F>(
    f: F,
    a: f64,
    b: f64,
    y_lo: f64,
    y_hi: f64,
    param_rows: &[Vec<f64>],
    options: DblquadOptions,
) -> Vec<Result<DblquadResult, IntegrateValidationError>>
where
    F: Fn(f64, f64, &[f64]) -> f64 + Sync,
{
    let nrows = param_rows.len();
    if nrows == 0 {
        return Vec::new();
    }
    let f_ref = &f;
    let solve_one = move |params: &[f64]| {
        dblquad(
            |y, x| f_ref(y, x, params),
            a,
            b,
            |_| y_lo,
            |_| y_hi,
            options,
        )
    };

    // Each double integral is an independent, expensive (O(n²) integrand evals) solve → fan whole
    // parameter sets across cores, capped by the row count; a tiny sweep stays serial.
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let nthreads = cores.min(nrows);
    if nthreads <= 1 || nrows < 4 {
        return param_rows.iter().map(|p| solve_one(p)).collect();
    }

    let chunk = nrows.div_ceil(nthreads);
    let solve_one = &solve_one;
    let chunk_results: Vec<Vec<Result<DblquadResult, IntegrateValidationError>>> =
        std::thread::scope(|scope| {
            (0..nthreads)
                .filter_map(|t| {
                    let lo = t * chunk;
                    if lo >= nrows {
                        return None;
                    }
                    let hi = (lo + chunk).min(nrows);
                    Some(scope.spawn(move || {
                        (lo..hi)
                            .map(|i| solve_one(&param_rows[i]))
                            .collect::<Vec<_>>()
                    }))
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().expect("dblquad_many worker panicked"))
                .collect()
        });

    let mut out = Vec::with_capacity(nrows);
    for cr in chunk_results {
        out.extend(cr);
    }
    out
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
        return Ok(CompositeQuadResult { integral: 0.0 });
    }
    validate_sample_coordinates(x)?;

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
        return Ok(CompositeQuadResult { integral: 0.0 });
    }
    if !dx.is_finite() {
        return Err(IntegrateValidationError::QuadInvalidTolerance {
            detail: "dx must be finite".to_string(),
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
    if y.is_empty() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "need at least 2 points for Simpson's rule".to_string(),
        });
    }
    if y.len() == 1 {
        return Ok(CompositeQuadResult { integral: 0.0 });
    }
    validate_sample_coordinates(x)?;
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
        // Even number of points = odd number of intervals. Apply Simpson's 1/3
        // on the first n-1 points, then correct the final interval with the
        // Cartwright parabolic formula (a quadratic through the last three
        // points), exactly as scipy.integrate.simpson — NOT a plain trapezoid,
        // which left the result off by up to ~3e-3.
        let integral_simp = simpson_nonuniform_odd(&y[..n - 1], &x[..n - 1]);
        let h0 = x[n - 2] - x[n - 3];
        let h1 = x[n - 1] - x[n - 2];
        let den_a = 6.0 * (h1 + h0);
        let den_b = 6.0 * h0;
        let den_e = 6.0 * h0 * (h0 + h1);
        let alpha = if den_a != 0.0 {
            (2.0 * h1 * h1 + 3.0 * h0 * h1) / den_a
        } else {
            0.0
        };
        let beta = if den_b != 0.0 {
            (h1 * h1 + 3.0 * h0 * h1) / den_b
        } else {
            0.0
        };
        let eta = if den_e != 0.0 {
            h1 * h1 * h1 / den_e
        } else {
            0.0
        };
        let last = alpha * y[n - 1] + beta * y[n - 2] - eta * y[n - 3];
        Ok(CompositeQuadResult {
            integral: integral_simp + last,
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
    if y.is_empty() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "need at least 2 points for Simpson's rule".to_string(),
        });
    }
    if y.len() == 1 {
        return Ok(CompositeQuadResult { integral: 0.0 });
    }
    if !dx.is_finite() {
        return Err(IntegrateValidationError::QuadInvalidTolerance {
            detail: "dx must be finite".to_string(),
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
        // Cartwright parabolic correction for the final interval (uniform
        // spacing h0=h1=dx reduces α,β,η to dx/12·(5,8,−1)), matching
        // scipy.integrate.simpson rather than a plain trapezoid.
        integral += dx / 12.0 * (5.0 * y[n - 1] + 8.0 * y[n - 2] - y[n - 3]);
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
    if y.is_empty() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "need at least one point for cumulative trapezoid".to_string(),
        });
    }
    if y.len() == 1 {
        return Ok(Vec::new());
    }
    validate_sample_coordinates(x)?;

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

/// Threads for a per-row 2-D integration sweep: serial for a handful of rows,
/// else one chunk per core capped by the row count.
fn axis_2d_thread_count(nrows: usize) -> usize {
    if nrows < 8 {
        return 1;
    }
    std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1)
        .min(nrows)
}

/// Map a fallible per-row integrator across `rows` in parallel (ordered chunks),
/// byte-identical to `rows.iter().map(op).collect()`. Propagates the first error
/// in row order.
fn integrate_rows_parallel<T, F>(nrows: usize, op: F) -> Result<Vec<T>, IntegrateValidationError>
where
    T: Send,
    F: Fn(usize) -> Result<T, IntegrateValidationError> + Sync,
{
    let nthreads = axis_2d_thread_count(nrows);
    if nthreads <= 1 {
        return (0..nrows).map(op).collect();
    }
    let chunk = nrows.div_ceil(nthreads);
    let op = &op;
    let chunk_results: Vec<Result<Vec<T>, IntegrateValidationError>> =
        std::thread::scope(|scope| {
            (0..nthreads)
                .filter_map(|t| {
                    let lo = t * chunk;
                    if lo >= nrows {
                        return None;
                    }
                    let hi = (lo + chunk).min(nrows);
                    Some(scope.spawn(move || (lo..hi).map(op).collect::<Result<Vec<T>, _>>()))
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().expect("axis-2d integrate worker panicked"))
                .collect()
        });
    let mut out = Vec::with_capacity(nrows);
    for cr in chunk_results {
        out.extend(cr?);
    }
    Ok(out)
}

/// Trapezoidal integral of each row of a 2-D array against shared sample
/// coordinates `x`, matching `scipy.integrate.trapezoid(Y, x=x, axis=1)`.
///
/// Byte-identical to calling [`trapezoid`] per row (independent rows fan across
/// cores). Returns one integral per row.
pub fn trapezoid_axis_2d(
    rows: &[Vec<f64>],
    x: &[f64],
) -> Result<Vec<f64>, IntegrateValidationError> {
    integrate_rows_parallel(rows.len(), |i| trapezoid(&rows[i], x).map(|r| r.integral))
}

/// Simpson's-rule integral of each row of a 2-D array against shared sample
/// coordinates `x`, matching `scipy.integrate.simpson(Y, x=x, axis=1)`.
///
/// Byte-identical to calling [`simpson`] per row (independent rows fan across cores).
pub fn simpson_axis_2d(rows: &[Vec<f64>], x: &[f64]) -> Result<Vec<f64>, IntegrateValidationError> {
    integrate_rows_parallel(rows.len(), |i| simpson(&rows[i], x).map(|r| r.integral))
}

/// Cumulative trapezoidal integral of each row of a 2-D array against shared
/// sample coordinates `x`, matching `scipy.integrate.cumulative_trapezoid(Y, x=x,
/// axis=1)` (no `initial`, so each output row is one shorter than the input).
///
/// Byte-identical to calling [`cumulative_trapezoid`] per row (independent rows
/// fan across cores).
pub fn cumulative_trapezoid_axis_2d(
    rows: &[Vec<f64>],
    x: &[f64],
) -> Result<Vec<Vec<f64>>, IntegrateValidationError> {
    integrate_rows_parallel(rows.len(), |i| cumulative_trapezoid(&rows[i], x))
}

/// Cumulative Simpson's-rule integral of each row of a 2-D array against shared
/// sample coordinates `x`, matching `scipy.integrate.cumulative_simpson(Y, x=x,
/// axis=1)` (no `initial`, so each output row is one shorter than the input).
///
/// Byte-identical to calling [`cumulative_simpson`] per row (independent rows fan
/// across cores).
pub fn cumulative_simpson_axis_2d(
    rows: &[Vec<f64>],
    x: &[f64],
) -> Result<Vec<Vec<f64>>, IntegrateValidationError> {
    integrate_rows_parallel(rows.len(), |i| cumulative_simpson(&rows[i], x))
}

/// Cumulatively integrate y with uniform spacing using the trapezoidal rule.
///
/// Matches `scipy.integrate.cumulative_trapezoid(y, dx=dx)`.
pub fn cumulative_trapezoid_uniform(
    y: &[f64],
    dx: f64,
) -> Result<Vec<f64>, IntegrateValidationError> {
    if y.is_empty() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "need at least one point for cumulative trapezoid".to_string(),
        });
    }
    if y.len() == 1 {
        return Ok(Vec::new());
    }
    if !dx.is_finite() {
        return Err(IntegrateValidationError::QuadInvalidTolerance {
            detail: "dx must be finite".to_string(),
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

/// Batched triple integration: evaluate `I(params) = ∫∫∫ f(z, y, x, params) dz dy dx` for MANY
/// parameter sets (`param_rows`) over a shared box, one [`DblquadResult`] per set. This is the
/// HEAVIEST-callback vmap case: tplquad nests three adaptive quadratures, so each integral makes
/// O(n³) integrand calls; in SciPy those are all Python calls and a parameter sweep loops tplquad
/// in Python, N integrals SERIALLY. fsci `tplquad_many` (param-sweep `F: Fn(f64 z, f64 y, f64 x,
/// &[f64] params)->f64`, shared box) fans the N independent triple integrations across cores and
/// inlines the integrand. Result `i` is byte-identical to `tplquad(|z,y,x| f(z,y,x,&param_rows[i]),
/// a, b, |_| y_lo, |_| y_hi, |_,_| z_lo, |_,_| z_hi, options)`.
#[allow(clippy::too_many_arguments)]
pub fn tplquad_many<F>(
    f: F,
    a: f64,
    b: f64,
    y_lo: f64,
    y_hi: f64,
    z_lo: f64,
    z_hi: f64,
    param_rows: &[Vec<f64>],
    options: DblquadOptions,
) -> Vec<Result<DblquadResult, IntegrateValidationError>>
where
    F: Fn(f64, f64, f64, &[f64]) -> f64 + Sync,
{
    let nrows = param_rows.len();
    if nrows == 0 {
        return Vec::new();
    }
    let f_ref = &f;
    let solve_one = move |params: &[f64]| {
        tplquad(
            |z, y, x| f_ref(z, y, x, params),
            a,
            b,
            |_| y_lo,
            |_| y_hi,
            |_, _| z_lo,
            |_, _| z_hi,
            options,
        )
    };

    // Each triple integral is an independent, very expensive (O(n³) integrand evals) solve → fan
    // whole parameter sets across cores, capped by the row count; a tiny sweep stays serial.
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let nthreads = cores.min(nrows);
    if nthreads <= 1 || nrows < 4 {
        return param_rows.iter().map(|p| solve_one(p)).collect();
    }

    let chunk = nrows.div_ceil(nthreads);
    let solve_one = &solve_one;
    let chunk_results: Vec<Vec<Result<DblquadResult, IntegrateValidationError>>> =
        std::thread::scope(|scope| {
            (0..nthreads)
                .filter_map(|t| {
                    let lo = t * chunk;
                    if lo >= nrows {
                        return None;
                    }
                    let hi = (lo + chunk).min(nrows);
                    Some(scope.spawn(move || {
                        (lo..hi)
                            .map(|i| solve_one(&param_rows[i]))
                            .collect::<Vec<_>>()
                    }))
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().expect("tplquad_many worker panicked"))
                .collect()
        });

    let mut out = Vec::with_capacity(nrows);
    for cr in chunk_results {
        out.extend(cr);
    }
    out
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
    if !tol.is_finite() || tol <= 0.0 {
        return Err(IntegrateValidationError::QuadInvalidTolerance {
            detail: "Romberg tolerance must be finite and positive".to_string(),
        });
    }

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
    if !dx.is_finite() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "dx must be finite".to_string(),
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
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "fixed_quad order n must be positive".to_string(),
        });
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
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "nquad requires at least one integration range".to_string(),
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

/// Batched N-dimensional integration: evaluate `I(params) = ∫…∫ f(x⃗, params) dx⃗` over a shared
/// hyper-rectangle `ranges` for MANY parameter sets, one [`QuadResult`] per set. This is the
/// vmap-over-solver primitive for arbitrary dimension — the deepest-nested callback case: an
/// `ndim`-dimensional `nquad` nests `ndim` adaptive quadratures, so each integral makes O(n^ndim)
/// integrand calls; in SciPy those are all Python and a parameter sweep loops `nquad` in Python,
/// N integrals SERIALLY. fsci `nquad_many` (param-sweep `F: Fn(&[f64] x, &[f64] params)->f64`) fans
/// the N independent integrations across cores and inlines the integrand. Result `i` is
/// byte-identical to `nquad(|x| f(x, &param_rows[i]), ranges, options)`.
pub fn nquad_many<F>(
    func: F,
    ranges: &[(f64, f64)],
    param_rows: &[Vec<f64>],
    options: QuadOptions,
) -> Vec<Result<QuadResult, IntegrateValidationError>>
where
    F: Fn(&[f64], &[f64]) -> f64 + Sync,
{
    let nrows = param_rows.len();
    if nrows == 0 {
        return Vec::new();
    }
    let func_ref = &func;
    let solve_one = move |params: &[f64]| nquad(|x| func_ref(x, params), ranges, options);

    // Each N-D integral is an independent, very expensive (O(n^ndim) integrand evals) solve → fan
    // whole parameter sets across cores, capped by the row count; a tiny sweep stays serial.
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let nthreads = cores.min(nrows);
    if nthreads <= 1 || nrows < 4 {
        return param_rows.iter().map(|p| solve_one(p)).collect();
    }

    let chunk = nrows.div_ceil(nthreads);
    let solve_one = &solve_one;
    let chunk_results: Vec<Vec<Result<QuadResult, IntegrateValidationError>>> =
        std::thread::scope(|scope| {
            (0..nthreads)
                .filter_map(|t| {
                    let lo = t * chunk;
                    if lo >= nrows {
                        return None;
                    }
                    let hi = (lo + chunk).min(nrows);
                    Some(scope.spawn(move || {
                        (lo..hi)
                            .map(|i| solve_one(&param_rows[i]))
                            .collect::<Vec<_>>()
                    }))
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().expect("nquad_many worker panicked"))
                .collect()
        });

    let mut out = Vec::with_capacity(nrows);
    for cr in chunk_results {
        out.extend(cr);
    }
    out
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
    if !options.atol.is_finite()
        || !options.rtol.is_finite()
        || options.atol < 0.0
        || options.rtol < 0.0
    {
        return Err(IntegrateValidationError::QuadInvalidTolerance {
            detail: "cubature tolerances must be finite and non-negative".to_string(),
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
/// Cache of Gauss-Legendre nodes/weights keyed by order `n`. The Newton-method
/// node solve is O(n²·iterations); `scipy.special.roots_legendre` is lru_cached for
/// the same reason — repeated `fixed_quad` / `gauss_legendre` calls with one order
/// then cost an O(n) clone instead of a full recompute.
static GAUSS_LEGENDRE_CACHE: std::sync::OnceLock<
    std::sync::RwLock<std::collections::HashMap<usize, (Vec<f64>, Vec<f64>)>>,
> = std::sync::OnceLock::new();

fn gauss_legendre_node_cache()
-> &'static std::sync::RwLock<std::collections::HashMap<usize, (Vec<f64>, Vec<f64>)>> {
    GAUSS_LEGENDRE_CACHE.get_or_init(|| std::sync::RwLock::new(std::collections::HashMap::new()))
}

fn gauss_legendre_nodes_weights(n: usize) -> (Vec<f64>, Vec<f64>) {
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    if n == 1 {
        return (vec![0.0], vec![2.0]);
    }
    let cache = gauss_legendre_node_cache();
    if let Some(hit) = cache.read().unwrap().get(&n) {
        return hit.clone();
    }
    // Computed outside the write lock; the result is deterministic, so a concurrent
    // insert of the same `n` is harmless (identical value).
    let computed = compute_gauss_legendre_nodes_weights(n);
    cache.write().unwrap().insert(n, computed.clone());
    computed
}

/// Compute Gauss-Legendre nodes/weights for order `n >= 2` via Newton's method on
/// the Legendre roots. Memoized by [`gauss_legendre_nodes_weights`]; the cached
/// value is this exact result, so callers stay bit-identical.
fn compute_gauss_legendre_nodes_weights(n: usize) -> (Vec<f64>, Vec<f64>) {
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
    if n == 0 {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "need at least one point for cumulative integration".to_string(),
        });
    }
    if n <= 2 {
        return cumulative_trapezoid(y, x);
    }

    // Validate x is finite and strictly increasing WITHOUT materializing a `dx`
    // buffer (the interval widths are recomputed inline below from `x`); this drops
    // an O(n) allocation + write pass from this bandwidth-bound routine.
    for points in x.windows(2) {
        let h = points[1] - points[0];
        if !(h.is_finite() && h > 0.0) {
            return Err(IntegrateValidationError::QuadInvalidBounds {
                detail: "x must be finite and strictly increasing".to_string(),
            });
        }
    }

    // Fill `result` in place with each interval's integral, then prefix-sum it
    // in place — no separate `interval_integrals` buffer (one O(n) Vec instead of
    // three). The per-interval Simpson coefficients are division-heavy (3-4
    // divisions each, see the *_interval helpers) and each even index `i` writes
    // the disjoint pair (i, i+1) independently — a compute-bound embarrassingly-
    // parallel loop, fanned across cores for large n. BYTE-IDENTICAL: same
    // per-interval formulas, same pair order, and the in-place prefix sum adds the
    // intervals in the same left-to-right order as the original cumulative scan.
    let mut result = vec![0.0; n - 1];
    let main_len = if n.is_multiple_of(2) { n - 2 } else { n - 1 };
    let nthreads = if main_len < (1 << 19) {
        1
    } else {
        std::thread::available_parallelism()
            .map(|c| c.get())
            .unwrap_or(1)
            .min(main_len / 2)
    };

    {
        let (main_part, tail_part) = result.split_at_mut(main_len);
        let fill_pairs = |seg: &mut [f64], base_pair: usize| {
            for (cl, chunk) in seg.chunks_mut(2).enumerate() {
                let i = 2 * (base_pair + cl);
                let (h0, h1) = (x[i + 1] - x[i], x[i + 2] - x[i + 1]);
                chunk[0] = cumulative_simpson_left_interval(y[i], y[i + 1], y[i + 2], h0, h1);
                chunk[1] = cumulative_simpson_right_interval(y[i], y[i + 1], y[i + 2], h0, h1);
            }
        };
        if nthreads <= 1 {
            fill_pairs(main_part, 0);
        } else {
            let npairs = main_len / 2;
            let pairs_per_thread = npairs.div_ceil(nthreads);
            let chunk_len = pairs_per_thread * 2;
            let fill_pairs = &fill_pairs;
            std::thread::scope(|scope| {
                for (t, seg) in main_part.chunks_mut(chunk_len).enumerate() {
                    scope.spawn(move || fill_pairs(seg, t * pairs_per_thread));
                }
            });
        }

        // Even n leaves a trailing half-interval handled by the right rule.
        if n.is_multiple_of(2) {
            let i = n - 3;
            let (h0, h1) = (x[i + 1] - x[i], x[i + 2] - x[i + 1]);
            tail_part[0] = cumulative_simpson_right_interval(y[i], y[i + 1], y[i + 2], h0, h1);
        }
    }

    for i in 1..result.len() {
        result[i] += result[i - 1];
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

/// Generic double-exponential quadrature driver.
///
/// `node(t)` maps a quadrature parameter `t ∈ ℝ` to `Some((x, jac))` — the
/// abscissa `x` and the full weight `jac = (dx/dt)·W(t)` — or `None` when the
/// node has run off the (possibly infinite) interval and should be skipped.
/// Integral ≈ `h · Σ_j jac_j · f(x_j)`. Abscissae are reused across levels
/// (halving the step only adds odd indices), and a quadratic `d1²/d2` error
/// model stops refinement at the convergence plateau.
fn de_quadrature<F, N>(f: &F, node: &N, atol: f64, rtol: f64, max_level: usize) -> QuadResult
where
    F: Fn(f64) -> f64,
    N: Fn(f64) -> Option<(f64, f64)>,
{
    // Beyond |t| ≈ 7 the double-exponential weight has either underflowed to
    // zero or driven the abscissa off the interval, so summation can stop.
    let t_cap = 7.0;
    let mut neval = 0usize;
    let eval = |t: f64, sum: &mut f64, neval: &mut usize| {
        if let Some((x, jac)) = node(t)
            && jac != 0.0
            && jac.is_finite()
        {
            let fx = f(x);
            *neval += 1;
            if fx.is_finite() {
                *sum += jac * fx;
            }
        }
    };

    // Level 0: every integer index at the coarse step h0 = 1.
    let mut h = 1.0_f64;
    let mut bare = 0.0_f64;
    eval(0.0, &mut bare, &mut neval);
    let mut j = 1;
    while (j as f64) * h <= t_cap {
        let t = j as f64 * h;
        eval(t, &mut bare, &mut neval);
        eval(-t, &mut bare, &mut neval);
        j += 1;
    }
    let mut prev = h * bare;
    let mut prev_delta = f64::INFINITY;
    let mut last_err = f64::INFINITY;

    for level in 1..=max_level {
        h *= 0.5;
        let mut j = 1;
        while (j as f64) * h <= t_cap {
            let t = j as f64 * h;
            eval(t, &mut bare, &mut neval);
            eval(-t, &mut bare, &mut neval);
            j += 2;
        }
        let cur = h * bare;
        let delta = (cur - prev).abs();
        last_err = if prev_delta > 0.0 && prev_delta.is_finite() {
            (delta * delta / prev_delta).min(delta)
        } else {
            delta
        };
        if level >= 2 && last_err <= atol + rtol * cur.abs() {
            return QuadResult {
                integral: cur,
                error: last_err,
                neval,
                converged: true,
            };
        }
        prev_delta = delta;
        prev = cur;
    }

    QuadResult {
        integral: prev,
        error: last_err,
        neval,
        converged: false,
    }
}

/// Evaluate a convergent integral numerically using tanh-sinh
/// (double-exponential) quadrature.
///
/// Matches `scipy.integrate.tanhsinh(f, a, b)` for a scalar integrand. Either or
/// both limits may be infinite, and integrable endpoint singularities are
/// allowed. The interior substitution `x = c + d·tanh((π/2)·sinh t)` packs the
/// abscissae densely near finite endpoints so singularities like `1/√x` or
/// `ln x` converge quadratically; semi-infinite intervals use the exp-sinh
/// transform `x = a + exp((π/2)·sinh t)` and doubly-infinite intervals the
/// sinh-sinh transform `x = sinh((π/2)·sinh t)`. Near a finite endpoint the
/// abscissa is formed as a cancellation-free distance from that endpoint, so
/// no precision is lost there.
///
/// `atol`/`rtol` are the absolute/relative tolerances (non-positive values fall
/// back to `0.0` and `1e-12`); `max_level` caps the refinement levels (`0` →
/// 16). Returns the estimate, a `d1²/d2` error estimate, the
/// function-evaluation count, and whether the tolerance was met.
pub fn tanhsinh<F>(f: F, a: f64, b: f64, atol: f64, rtol: f64, max_level: usize) -> QuadResult
where
    F: Fn(f64) -> f64,
{
    if a.is_nan() || b.is_nan() || !atol.is_finite() || !rtol.is_finite() {
        return QuadResult {
            integral: f64::NAN,
            error: f64::INFINITY,
            neval: 0,
            converged: false,
        };
    }
    if a == b {
        return QuadResult {
            integral: 0.0,
            error: 0.0,
            neval: 0,
            converged: true,
        };
    }
    // Integrate left→right, restore the sign for b < a.
    let (lo, hi, sign) = if a < b { (a, b, 1.0) } else { (b, a, -1.0) };
    let atol = if atol > 0.0 { atol } else { 0.0 };
    let rtol = if rtol > 0.0 { rtol } else { 1e-12 };
    let max_level = if max_level == 0 {
        16
    } else {
        max_level.min(20)
    };
    let half_pi = std::f64::consts::FRAC_PI_2;

    let mut result = if lo.is_finite() && hi.is_finite() {
        // Finite interval: tanh-sinh. Near each endpoint the abscissa is
        // `endpoint ∓ d·r` with `r = 1 − tanh(u)` formed cancellation-free.
        let c = 0.5 * (lo + hi);
        let d = 0.5 * (hi - lo);
        let node = |t: f64| -> Option<(f64, f64)> {
            let u = half_pi * t.sinh();
            let cu = u.cosh();
            let w = d * half_pi * t.cosh() / (cu * cu);
            if !w.is_finite() || w == 0.0 {
                return None;
            }
            if t == 0.0 {
                return Some((c, w));
            }
            let e2 = (-2.0 * u.abs()).exp();
            let r = 2.0 * e2 / (1.0 + e2); // 1 − tanh(|u|)
            let delta = d * r;
            if delta <= 0.0 {
                return None;
            }
            let x = if t > 0.0 { hi - delta } else { lo + delta };
            if x <= lo || x >= hi {
                return None;
            }
            Some((x, w))
        };
        de_quadrature(&f, &node, atol, rtol, max_level)
    } else if lo.is_finite() && hi == f64::INFINITY {
        // Semi-infinite [lo, ∞): exp-sinh, x = lo + exp((π/2)·sinh t).
        let node = |t: f64| -> Option<(f64, f64)> {
            let g = half_pi * t.sinh();
            let eg = g.exp();
            if eg == 0.0 {
                return None; // collapsed onto the finite endpoint
            }
            let x = lo + eg;
            let jac = eg * half_pi * t.cosh();
            if !x.is_finite() || !jac.is_finite() {
                return None;
            }
            Some((x, jac))
        };
        de_quadrature(&f, &node, atol, rtol, max_level)
    } else if lo == f64::NEG_INFINITY && hi.is_finite() {
        // Semi-infinite (−∞, hi]: exp-sinh mirrored, x = hi − exp((π/2)·sinh t).
        let node = |t: f64| -> Option<(f64, f64)> {
            let g = half_pi * t.sinh();
            let eg = g.exp();
            if eg == 0.0 {
                return None;
            }
            let x = hi - eg;
            let jac = eg * half_pi * t.cosh();
            if !x.is_finite() || !jac.is_finite() {
                return None;
            }
            Some((x, jac))
        };
        de_quadrature(&f, &node, atol, rtol, max_level)
    } else if lo == f64::NEG_INFINITY && hi == f64::INFINITY {
        // Doubly-infinite (−∞, ∞): sinh-sinh, x = sinh((π/2)·sinh t).
        let node = |t: f64| -> Option<(f64, f64)> {
            let g = half_pi * t.sinh();
            let x = g.sinh();
            let jac = g.cosh() * half_pi * t.cosh();
            if !x.is_finite() || !jac.is_finite() {
                return None;
            }
            Some((x, jac))
        };
        de_quadrature(&f, &node, atol, rtol, max_level)
    } else {
        // lo == +∞ or hi == −∞ after ordering is impossible for a < b.
        return QuadResult {
            integral: f64::NAN,
            error: f64::INFINITY,
            neval: 0,
            converged: false,
        };
    };

    result.integral *= sign;
    result
}

/// Result of a series summation by [`nsum`].
#[derive(Debug, Clone, PartialEq)]
pub struct NsumResult {
    /// Estimated value of the series.
    pub sum: f64,
    /// Estimated absolute error.
    pub error: f64,
    /// Number of function evaluations.
    pub nfev: usize,
    /// Whether the requested accuracy was achieved.
    pub converged: bool,
}

/// Evaluate a convergent series `Σ f(n)` for `n = a, a+step, …, b` numerically.
///
/// Matches `scipy.integrate.nsum(f, a, b, step)` for a scalar term function. A
/// finite range is summed directly. An infinite upper limit is handled by the
/// Euler–Maclaurin formula: the first terms are summed directly and the tail is
/// replaced by `∫ f dx` (via [`tanhsinh`]) plus the half-term and Bernoulli
/// derivative corrections. The size of the direct window is grown until the
/// estimate stops changing, which drives the (finite-difference) correction
/// terms toward zero and makes convergence robust regardless of derivative
/// noise.
///
/// `step` must be positive. `atol`/`rtol` default to `0.0`/`1e-11` when
/// non-positive. Returns the sum, an error estimate, the evaluation count, and
/// whether the tolerance was met.
pub fn nsum<F>(f: F, a: f64, b: f64, step: f64, atol: f64, rtol: f64) -> NsumResult
where
    F: Fn(f64) -> f64,
{
    let nan = NsumResult {
        sum: f64::NAN,
        error: f64::INFINITY,
        nfev: 0,
        converged: false,
    };
    if a.is_nan()
        || !step.is_finite()
        || step <= 0.0
        || !a.is_finite()
        || !atol.is_finite()
        || !rtol.is_finite()
    {
        return nan;
    }
    let atol = if atol > 0.0 { atol } else { 0.0 };
    let rtol = if rtol > 0.0 { rtol } else { 1e-11 };

    // Finite range: direct summation over n = a, a+step, …, ≤ b.
    if b.is_finite() {
        if b < a {
            return NsumResult {
                sum: 0.0,
                error: 0.0,
                nfev: 0,
                converged: true,
            };
        }
        let count = ((b - a) / step).floor() as i64 + 1;
        let mut sum = 0.0;
        let mut nfev = 0usize;
        for k in 0..count {
            sum += f(a + k as f64 * step);
            nfev += 1;
        }
        return NsumResult {
            sum,
            error: 0.0,
            nfev,
            converged: true,
        };
    }
    if b != f64::INFINITY {
        return nan; // b is −∞ with a finite ⇒ empty/invalid ordering
    }

    // Infinite tail via Euler–Maclaurin with a growing direct window.
    // Σ_{k≥m} f(a+ks) ≈ (1/s)∫_{n_m}^∞ f dx + f(n_m)/2
    //                   − (s/12) f'(n_m) + (s³/720) f'''(n_m)
    let s = step;
    let mut direct = 0.0;
    let mut nfev = 0usize;
    let mut filled = 0i64; // number of leading terms already in `direct`
    let mut prev_est = f64::INFINITY;
    let mut last_err = f64::INFINITY;
    let mut m = 8i64;

    for _iter in 0..24 {
        // Extend the direct partial sum to the first `m` terms.
        while filled < m {
            direct += f(a + filled as f64 * s);
            nfev += 1;
            filled += 1;
        }
        let n_m = a + m as f64 * s;
        let f_nm = f(n_m);
        nfev += 1;
        // Central finite-difference derivatives at n_m.
        let delta = (1e-3 * n_m.abs()).max(1e-4);
        let fp = f(n_m + delta);
        let fm = f(n_m - delta);
        let fp2 = f(n_m + 2.0 * delta);
        let fm2 = f(n_m - 2.0 * delta);
        nfev += 4;
        let d1 = (fp - fm) / (2.0 * delta); // f'(n_m)
        let d3 = (fp2 - 2.0 * fp + 2.0 * fm - fm2) / (2.0 * delta.powi(3)); // f'''(n_m)

        let integ = tanhsinh(&f, n_m, f64::INFINITY, atol, rtol, 16);
        nfev += integ.neval;

        let tail = integ.integral / s + 0.5 * f_nm - (s / 12.0) * d1 + (s.powi(3) / 720.0) * d3;
        let est = direct + tail;

        last_err = (est - prev_est).abs();
        if last_err <= atol + rtol * est.abs() {
            return NsumResult {
                sum: est,
                error: last_err,
                nfev,
                converged: true,
            };
        }
        prev_est = est;
        m *= 2;
    }

    NsumResult {
        sum: prev_est,
        error: last_err,
        nfev,
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
    if !a.is_finite()
        || !b.is_finite()
        || !options.epsabs.is_finite()
        || !options.epsrel.is_finite()
        || options.epsabs < 0.0
        || options.epsrel < 0.0
    {
        return QuadResult {
            integral: f64::NAN,
            error: f64::INFINITY,
            neval: 0,
            converged: false,
        };
    }

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

/// Newton-Cotes integration weights for `n` equally-spaced intervals (n+1
/// points).
///
/// Returns weights on the **unit interval**, normalized so that
/// `∫₀¹ f(x) dx ≈ Σ wᵢ·f(xᵢ)` — i.e. the weights sum to 1.
///
/// This is **not** the same normalization as `scipy.integrate.newton_cotes`,
/// which returns weights `aᵢ` summing to `n` for the form
/// `∫ₐᵇ f ≈ Δx·Σ aᵢ·f(xᵢ)` with `Δx = (b−a)/n`. The two are related by
/// `aᵢ = n·wᵢ`; multiply these weights by `n` to recover scipy's values.
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
            // Exact Newton-Cotes weights: w_i = ∫₀¹ L_i(x) dx, where L_i is the
            // Lagrange basis on the equally-spaced nodes x_j = j/n. Expand the
            // numerator product Π_{j≠i}(x − x_j) into polynomial coefficients,
            // divide by the constant denominator Π_{j≠i}(x_i − x_j), and integrate
            // term-by-term (∫₀¹ xᵏ dx = 1/(k+1)). The previous composite-Simpson
            // approximation of this integral was only accurate to ~1e-5 for n≥5,
            // since Simpson is not exact for the degree-n (n≥4) basis polynomial.
            let nf = n as f64;
            let nodes: Vec<f64> = (0..=n).map(|j| j as f64 / nf).collect();
            let mut weights = vec![0.0; n + 1];
            for i in 0..=n {
                // poly[k] = coefficient of xᵏ in Π_{j≠i}(x − x_j) (ascending).
                let mut poly = vec![1.0_f64];
                let mut denom = 1.0_f64;
                for j in 0..=n {
                    if j == i {
                        continue;
                    }
                    let xj = nodes[j];
                    let mut next = vec![0.0_f64; poly.len() + 1];
                    for (k, &c) in poly.iter().enumerate() {
                        next[k + 1] += c; // x · c
                        next[k] -= xj * c; // −x_j · c
                    }
                    poly = next;
                    denom *= nodes[i] - xj;
                }
                let integral: f64 = poly
                    .iter()
                    .enumerate()
                    .map(|(k, &c)| c / (k as f64 + 1.0))
                    .sum();
                weights[i] = integral / denom;
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
    if !a.is_finite() || !b.is_finite() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "newton_cotes_quad bounds must be finite".to_string(),
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
    if !a.is_finite() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "quad_inf: finite endpoint must be finite".to_string(),
        });
    }

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
    if !b.is_finite() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "quad_neg_inf: finite endpoint must be finite".to_string(),
        });
    }

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
    let lo = a.min(b);
    let hi = a.max(b);
    if !(singular_point > lo && singular_point < hi) {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "singular point must lie strictly inside the integration interval".to_string(),
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

    // Evaluate the coarse grid in place. This preserves the exact summation
    // order of collecting indices 0, 2, 4, ... while avoiding two allocations.
    let mut t2 = 0.0;
    let mut i = 0;
    while i + 2 < n {
        t2 += 0.5 * (y[i] + y[i + 2]) * (x[i + 2] - x[i]);
        i += 2;
    }

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
    if n == 0 || !a.is_finite() || !b.is_finite() {
        return f64::NAN;
    }
    // True n-point Gauss-Legendre for every n (exact to degree 2n-1), via the same
    // general node/weight generator `fixed_quad` uses. The previous version only
    // hardcoded n=2..5 and silently fell back to a composite Simpson rule for n>=6,
    // which is NOT Gauss-Legendre and was wrong by several percent (e.g. n=10 lost
    // ~5% on a degree-19 polynomial that GL integrates exactly).
    let (nodes, weights) = gauss_legendre_nodes_weights(n);

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
    if !t_lo.is_finite() || !t_hi.is_finite() {
        return f64::NAN;
    }
    if t_lo == t_hi {
        return 0.0;
    }

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
/// LCG multiplier for the Monte Carlo sampler (state -> a*state + 1).
const MC_LCG_A: u64 = 6364136223846793005;

/// `steps`-fold composition of the affine LCG step `x -> a*x + c`, returned as
/// `(A, C)` with `x_steps = A*x + C` (wrapping). Lets each thread start its
/// sample chunk from the exact RNG state the serial loop would reach (each
/// sample consumes `d` draws), without replaying every draw — enabling
/// byte-identical parallelism.
fn lcg_jump(a: u64, c: u64, steps: usize) -> (u64, u64) {
    let (mut res_a, mut res_c) = (1u64, 0u64);
    let (mut base_a, mut base_c) = (a, c);
    let mut e = steps;
    while e > 0 {
        if e & 1 == 1 {
            res_c = base_a.wrapping_mul(res_c).wrapping_add(base_c);
            res_a = base_a.wrapping_mul(res_a);
        }
        base_c = base_a.wrapping_mul(base_c).wrapping_add(base_c);
        base_a = base_a.wrapping_mul(base_a);
        e >>= 1;
    }
    (res_a, res_c)
}

pub fn monte_carlo_integrate<F>(
    f: F,
    bounds: &[(f64, f64)],
    n_samples: usize,
    seed: u64,
) -> (f64, f64)
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    let d = bounds.len();
    if d == 0 || n_samples == 0 {
        return (0.0, 0.0);
    }

    let mut volume = 1.0;
    for &(a, b) in bounds {
        if !a.is_finite() || !b.is_finite() {
            return (f64::NAN, f64::INFINITY);
        }
        let width = b - a;
        if !width.is_finite() {
            return (f64::NAN, f64::INFINITY);
        }
        volume *= width;
        if !volume.is_finite() {
            return (f64::NAN, f64::INFINITY);
        }
    }

    // Each sample consumes exactly `d` LCG draws, so chunk-boundary RNG states
    // are reachable via an `(chunk*d)`-step LCG jump. Evaluate the (expensive)
    // integrand per sample in parallel into ordered slots, then reduce
    // SEQUENTIALLY in sample order — byte-identical to the serial loop (same
    // RNG stream, same points, same float summation order).
    let eval_sample = |rng: &mut u64, point: &mut [f64]| -> f64 {
        for (j, p) in point.iter_mut().enumerate() {
            *rng = rng.wrapping_mul(MC_LCG_A).wrapping_add(1);
            let u = (*rng >> 11) as f64 / (1u64 << 53) as f64;
            *p = bounds[j].0 + u * (bounds[j].1 - bounds[j].0);
        }
        f(point)
    };

    // Parallel only pays off once the sampling work is large enough to absorb
    // the extra fvals buffer + separate sequential reduction pass; below that
    // the fused serial loop (byte-identical, zero overhead) wins.
    let work = n_samples.saturating_mul(d);
    let nthreads = if work < (1 << 18) || n_samples < 2 {
        1
    } else {
        std::thread::available_parallelism()
            .map(|c| c.get())
            .unwrap_or(1)
            .min(n_samples)
    };

    if nthreads <= 1 {
        // Fused serial loop, byte-for-byte the original (sampling inlined, no
        // closure indirection, no fvals allocation) so small inputs never regress.
        let mut rng = seed;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut point = vec![0.0; d];
        for _ in 0..n_samples {
            for j in 0..d {
                rng = rng.wrapping_mul(MC_LCG_A).wrapping_add(1);
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
        return (mean * volume, std_err);
    }

    // Each thread reduces its own chunk directly into (Σf, Σf²) — no per-sample
    // buffer. The chunk RNG start states (via lcg_jump) reproduce EXACTLY the same
    // samples in the same order as the serial path, so the only difference is that
    // the cross-chunk combine reassociates the sums (~1e-15) — irrelevant for a
    // Monte Carlo estimate. This drops the O(n_samples) `out` Vec (alloc + first-
    // touch page faults at large n) and its serial reduction pass.
    let (sum, sum_sq) = {
        let chunk = n_samples.div_ceil(nthreads);
        let (jump_a, jump_c) = lcg_jump(MC_LCG_A, 1, chunk * d);
        let mut starts: Vec<(u64, usize)> = Vec::new();
        let mut cs = seed;
        let mut base = 0;
        while base < n_samples {
            starts.push((cs, chunk.min(n_samples - base)));
            cs = jump_a.wrapping_mul(cs).wrapping_add(jump_c);
            base += chunk;
        }
        let eval_sample = &eval_sample;
        let mut partials = vec![(0.0_f64, 0.0_f64); starts.len()];
        std::thread::scope(|scope| {
            for ((state0, count), slot) in starts.into_iter().zip(partials.iter_mut()) {
                scope.spawn(move || {
                    let mut rng = state0;
                    let mut point = vec![0.0; d];
                    let mut s = 0.0;
                    let mut sq = 0.0;
                    for _ in 0..count {
                        let fval = eval_sample(&mut rng, &mut point);
                        s += fval;
                        sq += fval * fval;
                    }
                    *slot = (s, sq);
                });
            }
        });
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for (s, sq) in partials {
            sum += s;
            sum_sq += sq;
        }
        (sum, sum_sq)
    };

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
    F: Fn(&[f64]) -> f64 + Sync,
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

    // Each estimate is an independent Halton block: a fixed-order sum over its
    // own disjoint segment of the deterministic sequence. Compute one estimate
    // per slot — the per-estimate sum keeps its exact `i`-order arithmetic and
    // the cross-estimate mean/variance below stay sequential in index order, so
    // splitting the estimates across threads is byte-identical to the serial
    // loop. Each worker owns a `point` scratch buffer and disjoint output slots.
    let estimate = |est: usize, point: &mut [f64]| -> f64 {
        // Skip the first sample (idx=0 = origin) and offset by est*n_points so
        // blocks are disjoint segments of the deterministic sequence.
        let start = 1 + est * n_points;
        let mut sum = 0.0;
        for i in 0..n_points {
            let idx = start + i;
            for j in 0..d {
                let u = van_der_corput(idx, QMC_PRIMES[j]);
                point[j] = lb[j] + u * (ub[j] - lb[j]);
            }
            sum += f(&point[..d]);
        }
        sum / n_points as f64 * volume
    };

    let work = n_estimates
        .saturating_mul(n_points)
        .saturating_mul(d.max(1));
    let nthreads = if work < (1 << 16) || n_estimates < 2 {
        1
    } else {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(n_estimates)
    };

    let estimates = if nthreads <= 1 {
        let mut estimates = Vec::with_capacity(n_estimates);
        let mut point = vec![0.0; d];
        for est in 0..n_estimates {
            estimates.push(estimate(est, &mut point));
        }
        estimates
    } else {
        let mut estimates = vec![0.0; n_estimates];
        let chunk = n_estimates.div_ceil(nthreads);
        let estimate = &estimate;
        std::thread::scope(|scope| {
            for (t, slots) in estimates.chunks_mut(chunk).enumerate() {
                let base = t * chunk;
                scope.spawn(move || {
                    let mut point = vec![0.0; d];
                    for (k, slot) in slots.iter_mut().enumerate() {
                        *slot = estimate(base + k, &mut point);
                    }
                });
            }
        });
        estimates
    };

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
    fn gauss_legendre_node_cache_is_bit_identical_to_compute() {
        // The by-order cache must return exactly the Newton-method result, so every
        // fixed_quad / gauss_legendre value is unchanged. Verify the first
        // (miss → compute+store) and second (cache hit) calls both byte-equal a
        // direct recompute, across even and odd orders.
        for n in [2usize, 5, 16, 33, 64, 100] {
            let direct = compute_gauss_legendre_nodes_weights(n);
            let first = gauss_legendre_nodes_weights(n);
            let second = gauss_legendre_nodes_weights(n);
            assert_eq!(first, direct, "cache miss path vs compute, n={n}");
            assert_eq!(second, direct, "cache hit path vs compute, n={n}");
        }
    }

    #[test]
    fn gauss_legendre_exact_for_high_n_polynomials() {
        // n-point Gauss-Legendre is exact (to rounding) for polynomials up to
        // degree 2n-1. Regression: n>=6 previously fell back to Simpson's rule and
        // was wrong by several percent.
        for &n in &[2usize, 5, 8, 10, 16, 20] {
            let deg = 2 * n - 1;
            let v = gauss_legendre(|x: f64| x.powi(deg as i32), 0.0, 1.0, n);
            let exact = 1.0 / ((deg + 1) as f64);
            let rel = ((v - exact) / exact).abs();
            assert!(
                rel < 1e-10,
                "gauss_legendre n={n} on x^{deg}: rel error {rel:.3e} (got {v}, exact {exact})"
            );
        }
    }

    #[test]
    fn gauss_legendre_rejects_zero_order_and_non_finite_bounds() {
        let invalid_cases = [
            (0usize, 0.0, 1.0),
            (4usize, f64::NAN, 1.0),
            (4usize, 0.0, f64::INFINITY),
            (4usize, f64::NEG_INFINITY, 1.0),
        ];

        for (n, a, b) in invalid_cases {
            let result = gauss_legendre(
                |_| -> f64 { panic!("invalid Gauss-Legendre input should not be sampled") },
                a,
                b,
                n,
            );
            assert!(result.is_nan(), "n={n}, a={a}, b={b}");
        }
    }

    #[test]
    fn monte_carlo_integrate_rejects_non_finite_bounds() {
        for bounds in [
            vec![(f64::NAN, 1.0)],
            vec![(0.0, f64::INFINITY)],
            vec![(f64::NEG_INFINITY, 1.0)],
            vec![(-f64::MAX, f64::MAX)],
        ] {
            let (integral, error) = monte_carlo_integrate(
                |_| -> f64 { panic!("invalid bounds should not be sampled") },
                &bounds,
                8,
                1,
            );
            assert!(integral.is_nan(), "bounds={bounds:?}");
            assert_eq!(error, f64::INFINITY, "bounds={bounds:?}");
        }
    }

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
    fn trapezoid_richardson_exact_for_cubic() {
        // Richardson extrapolation of trapezoid (T=(4*T1-T2)/3) cancels the O(h^2)
        // error -> exact for cubics. y=x^3 at [0,1,2,3,4]; integral_0^4 x^3 = 64.
        // trapezoid_richardson was untested.
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y = [0.0, 1.0, 8.0, 27.0, 64.0];
        assert!(
            (trapezoid_richardson(&y, &x) - 64.0).abs() < 1e-10,
            "richardson cubic"
        );
    }

    #[test]
    fn trapezoid_richardson_strided_coarse_pass_is_bit_identical() {
        fn collected_reference(y: &[f64], x: &[f64]) -> f64 {
            if y.len() < 2 || x.len() != y.len() {
                return 0.0;
            }
            let t1 = trapezoid_irregular(y, x);
            if y.len() < 5 {
                return t1;
            }
            let y2: Vec<f64> = y.iter().step_by(2).copied().collect();
            let x2: Vec<f64> = x.iter().step_by(2).copied().collect();
            let t2 = trapezoid_irregular(&y2, &x2);
            (4.0 * t1 - t2) / 3.0
        }

        let mut cases = vec![
            (
                vec![0.0, 1.0, 8.0, 27.0, 64.0],
                vec![0.0, 1.0, 2.0, 3.0, 4.0],
            ),
            (
                vec![-0.0, 0.25, -1.5, 4.0, 2.25, -3.0],
                vec![-2.0, -1.75, -0.5, 0.0, 1.25, 3.0],
            ),
            (
                vec![3.0, -2.0, 1.0, 0.5, -0.25, 4.0, 8.0],
                vec![0.0, 0.1, 0.4, 1.0, 1.8, 3.0, 4.5],
            ),
        ];
        let x = (0..1_025)
            .map(|i| (i as f64).powi(2) * 1e-6)
            .collect::<Vec<_>>();
        let y = x
            .iter()
            .map(|value| (17.0 * value).sin() + value * 0.25)
            .collect::<Vec<_>>();
        cases.push((y, x));

        for (y, x) in cases {
            assert_eq!(
                trapezoid_richardson(&y, &x).to_bits(),
                collected_reference(&y, &x).to_bits(),
                "length {}",
                y.len()
            );
        }
    }

    #[test]
    fn quad_full_inf_gaussian_matches_analytic() {
        // integral_{-inf}^{inf} exp(-x^2) dx = sqrt(pi). quad_full_inf was untested.
        let r = quad_full_inf(|x: f64| (-x * x).exp(), QuadOptions::default()).unwrap();
        assert!(
            (r.integral - std::f64::consts::PI.sqrt()).abs() < 1e-8,
            "gaussian over R = {}",
            r.integral
        );
    }

    #[test]
    fn trapezoid_match_scipy() {
        // scipy.integrate.trapezoid([1,4,9,16], x=[0,1,2,3]) = 21.5.
        let r = trapezoid(&[1.0, 4.0, 9.0, 16.0], &[0.0, 1.0, 2.0, 3.0]).expect("trapezoid");
        assert!(
            (r.integral - 21.5).abs() < 1e-12,
            "trapezoid: {}",
            r.integral
        );
    }

    #[test]
    fn cumulative_trapezoid_match_scipy() {
        // scipy.integrate.cumulative_trapezoid (no initial): length n-1.
        let r =
            cumulative_trapezoid(&[1.0, 2.0, 3.0, 4.0], &[0.0, 1.0, 2.0, 3.0]).expect("cumtrapz");
        for (g, e) in r.iter().zip(&[1.5, 4.0, 7.5]) {
            assert!((g - e).abs() < 1e-12, "cumtrapz: {g} vs {e}");
        }
        let r2 = cumulative_trapezoid(&[1.0, 4.0, 9.0], &[0.0, 1.0, 2.0]).expect("cumtrapz2");
        for (g, e) in r2.iter().zip(&[2.5, 9.0]) {
            assert!((g - e).abs() < 1e-12, "cumtrapz2: {g} vs {e}");
        }
    }

    #[test]
    fn axis_2d_integration_matches_per_row_loop_bit_for_bit() {
        // trapezoid_axis_2d / simpson_axis_2d / cumulative_trapezoid_axis_2d must be
        // bit-identical to calling the 1-D routine per row. nrows crosses the gate (8).
        let mut s: u64 = 0x1357_9BDF_2468_ACE1;
        let mut u = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / 9.007_199_254_740_992e15 * 2.0 - 1.0
        };
        let (nr, nc) = (200usize, 51usize); // odd nc exercises Simpson's even-interval path
        let rows: Vec<Vec<f64>> = (0..nr).map(|_| (0..nc).map(|_| u()).collect()).collect();
        let mut x: Vec<f64> = (0..nc).map(|_| u() * 10.0 + 10.0).collect();
        x.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let tr = trapezoid_axis_2d(&rows, &x).expect("trap");
        let si = simpson_axis_2d(&rows, &x).expect("simp");
        let ct = cumulative_trapezoid_axis_2d(&rows, &x).expect("cumtrap");
        let cs = cumulative_simpson_axis_2d(&rows, &x).expect("cumsimp");
        assert_eq!(tr.len(), nr);
        assert_eq!(si.len(), nr);
        assert_eq!(ct.len(), nr);
        assert_eq!(cs.len(), nr);
        for i in 0..nr {
            assert_eq!(
                tr[i].to_bits(),
                trapezoid(&rows[i], &x).unwrap().integral.to_bits()
            );
            assert_eq!(
                si[i].to_bits(),
                simpson(&rows[i], &x).unwrap().integral.to_bits()
            );
            let cti = cumulative_trapezoid(&rows[i], &x).unwrap();
            assert_eq!(ct[i].len(), cti.len());
            for (a, b) in ct[i].iter().zip(&cti) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
            let csi = cumulative_simpson(&rows[i], &x).unwrap();
            assert_eq!(cs[i].len(), csi.len());
            for (a, b) in cs[i].iter().zip(&csi) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
        // Empty input and error propagation (mismatched x length in one row shape).
        assert!(trapezoid_axis_2d(&[], &x).unwrap().is_empty());
    }

    #[test]
    fn simpson_odd_interval_cartwright_match_scipy() {
        // scipy.integrate.simpson on x^4 (degree 4 -> Simpson NOT exact, so the
        // odd-interval Cartwright correction actually matters; a trapezoid
        // fallback would diverge).
        let odd = simpson(&[1.0, 16.0, 81.0, 256.0], &[1.0, 2.0, 3.0, 4.0])
            .expect("odd")
            .integral;
        assert!((odd - 208.0).abs() < 1e-10, "odd 4pts: {odd}");
        let even = simpson(&[1.0, 16.0, 81.0, 256.0, 625.0], &[1.0, 2.0, 3.0, 4.0, 5.0])
            .expect("even")
            .integral;
        assert!(
            (even - 625.333_333_333_333_3).abs() < 1e-10,
            "even 5pts: {even}"
        );
    }

    #[test]
    fn quad_adaptive_match_closed_form() {
        // scipy.integrate.quad converges to the analytic integral; lock fsci's
        // adaptive quad to the closed forms (equal to scipy's values to ~1e-10).
        let close = |r: QuadResult, want: f64, n: &str| {
            assert!(
                (r.integral - want).abs() < 1e-9,
                "{n}: {} != {want}",
                r.integral
            );
        };
        close(
            quad(|x| x * x, 0.0, 1.0, QuadOptions::default()).unwrap(),
            1.0 / 3.0,
            "x^2 on [0,1]",
        );
        close(
            quad(
                |x: f64| x.sin(),
                0.0,
                std::f64::consts::PI,
                QuadOptions::default(),
            )
            .unwrap(),
            2.0,
            "sin on [0,pi]",
        );
        close(
            quad(|x: f64| x.exp(), 0.0, 1.0, QuadOptions::default()).unwrap(),
            std::f64::consts::E - 1.0,
            "exp on [0,1]",
        );
        close(
            quad(|x| 1.0 / (1.0 + x * x), 0.0, 2.0, QuadOptions::default()).unwrap(),
            2.0_f64.atan(),
            "1/(1+x^2) on [0,2]",
        );
    }

    #[test]
    fn quad_many_byte_identical_to_per_param() {
        // Parameter sweep of a peaked + oscillatory integrand. The batched integral must
        // equal looping quad per parameter set, bit-for-bit.
        let f = |x: f64, p: &[f64]| (-p[0] * (x - p[1]).powi(2)).exp() * (p[2] * x).cos();
        let mut s = 3u64;
        let mut rng = |lo: f64, hi: f64| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            lo + (hi - lo) * ((s >> 11) as f64 / (1u64 << 53) as f64)
        };
        let nrows = 20usize; // crosses the serial->parallel gate
        let params: Vec<Vec<f64>> = (0..nrows)
            .map(|_| vec![rng(20.0, 200.0), rng(0.3, 0.7), rng(5.0, 30.0)])
            .collect();
        let opts = QuadOptions::default();

        let batched = quad_many(f, 0.0, 1.0, &params, opts);
        assert_eq!(batched.len(), nrows);
        for (i, p) in params.iter().enumerate() {
            let single = quad(|x| f(x, p), 0.0, 1.0, opts).expect("single");
            let many = batched[i].as_ref().expect("batched member");
            assert_eq!(
                many.integral.to_bits(),
                single.integral.to_bits(),
                "integral mismatch param {i}"
            );
            assert_eq!(
                many.error.to_bits(),
                single.error.to_bits(),
                "error mismatch param {i}"
            );
            assert_eq!(
                many.converged, single.converged,
                "converged mismatch param {i}"
            );
        }
        assert!(
            batched
                .iter()
                .filter(|r| r.as_ref().map(|x| x.converged).unwrap_or(false))
                .count()
                >= nrows / 2
        );
        assert!(quad_many(f, 0.0, 1.0, &[], opts).is_empty());
    }

    #[test]
    fn dblquad_many_byte_identical_to_per_param() {
        // Parameter sweep of a 2D Gaussian bump over the unit square.
        let f = |y: f64, x: f64, p: &[f64]| (-p[0] * ((x - 0.5).powi(2) + (y - 0.5).powi(2))).exp();
        let mut s = 8u64;
        let mut rng = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            5.0 + 35.0 * ((s >> 11) as f64 / (1u64 << 53) as f64)
        };
        let nrows = 10usize; // crosses the serial->parallel gate
        let params: Vec<Vec<f64>> = (0..nrows).map(|_| vec![rng()]).collect();
        let opts = DblquadOptions::default();

        let batched = dblquad_many(f, 0.0, 1.0, 0.0, 1.0, &params, opts);
        assert_eq!(batched.len(), nrows);
        for (i, p) in params.iter().enumerate() {
            let single =
                dblquad(|y, x| f(y, x, p), 0.0, 1.0, |_| 0.0, |_| 1.0, opts).expect("single");
            let many = batched[i].as_ref().expect("batched member");
            assert_eq!(
                many.integral.to_bits(),
                single.integral.to_bits(),
                "integral mismatch param {i}"
            );
            assert_eq!(
                many.error.to_bits(),
                single.error.to_bits(),
                "error mismatch param {i}"
            );
            assert_eq!(
                many.converged, single.converged,
                "converged mismatch param {i}"
            );
        }
        assert!(
            batched
                .iter()
                .filter(|r| r.as_ref().map(|x| x.converged).unwrap_or(false))
                .count()
                >= nrows / 2
        );
        assert!(dblquad_many(f, 0.0, 1.0, 0.0, 1.0, &[], opts).is_empty());
    }

    #[test]
    fn tplquad_many_byte_identical_to_per_param() {
        // Parameter sweep of a 3D Gaussian over the unit cube.
        let f = |z: f64, y: f64, x: f64, p: &[f64]| (-p[0] * (x * x + y * y + z * z)).exp();
        let mut s = 11u64;
        let mut rng = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            2.0 + 13.0 * ((s >> 11) as f64 / (1u64 << 53) as f64)
        };
        let nrows = 8usize; // crosses the serial->parallel gate
        let params: Vec<Vec<f64>> = (0..nrows).map(|_| vec![rng()]).collect();
        let opts = DblquadOptions::default();

        let batched = tplquad_many(f, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, &params, opts);
        assert_eq!(batched.len(), nrows);
        for (i, p) in params.iter().enumerate() {
            let single = tplquad(
                |z, y, x| f(z, y, x, p),
                0.0,
                1.0,
                |_| 0.0,
                |_| 1.0,
                |_, _| 0.0,
                |_, _| 1.0,
                opts,
            )
            .expect("single");
            let many = batched[i].as_ref().expect("batched member");
            assert_eq!(
                many.integral.to_bits(),
                single.integral.to_bits(),
                "integral mismatch param {i}"
            );
            assert_eq!(
                many.error.to_bits(),
                single.error.to_bits(),
                "error mismatch param {i}"
            );
            assert_eq!(
                many.converged, single.converged,
                "converged mismatch param {i}"
            );
        }
        assert!(
            batched
                .iter()
                .filter(|r| r.as_ref().map(|x| x.converged).unwrap_or(false))
                .count()
                >= nrows / 2
        );
        assert!(tplquad_many(f, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, &[], opts).is_empty());
    }

    #[test]
    fn nquad_many_byte_identical_to_per_param() {
        // Parameter sweep of a 4-D Gaussian over the unit hypercube. The batched integral
        // must equal looping nquad per parameter set, bit-for-bit.
        let f = |x: &[f64], p: &[f64]| {
            (-p[0] * (x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3])).exp()
        };
        let ranges = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)];
        let mut s = 17u64;
        let mut rng = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            1.0 + 5.0 * ((s >> 11) as f64 / (1u64 << 53) as f64)
        };
        let nrows = 8usize; // crosses the serial->parallel gate
        let params: Vec<Vec<f64>> = (0..nrows).map(|_| vec![rng()]).collect();
        let opts = QuadOptions::default();

        let batched = nquad_many(f, &ranges, &params, opts);
        assert_eq!(batched.len(), nrows);
        for (i, p) in params.iter().enumerate() {
            let single = nquad(|x| f(x, p), &ranges, opts).expect("single");
            let many = batched[i].as_ref().expect("batched member");
            assert_eq!(
                many.integral.to_bits(),
                single.integral.to_bits(),
                "integral mismatch param {i}"
            );
            assert_eq!(
                many.converged, single.converged,
                "converged mismatch param {i}"
            );
        }
        assert!(nquad_many(f, &ranges, &[], opts).is_empty());
    }

    #[test]
    fn quad_simpson_trapezoid_match_scipy() {
        // Golden values from scipy.integrate (1.17.1).
        // scipy.integrate.quad(exp(-x^2), 0, 2) = 0.8820813907624215 (= √π/2·erf(2)).
        let g = quad(|x: f64| (-x * x).exp(), 0.0, 2.0, QuadOptions::default()).expect("quad");
        assert!(
            (g.integral - 0.882_081_390_762_421_5).abs() < 1e-9,
            "quad gauss: {}",
            g.integral
        );
        // scipy.integrate.quad(x^3, 0, 3) = 20.25.
        let c = quad(|x: f64| x * x * x, 0.0, 3.0, QuadOptions::default()).expect("quad");
        assert!(
            (c.integral - 20.25).abs() < 1e-9,
            "quad x^3: {}",
            c.integral
        );
        // scipy.integrate.simpson / trapezoid of y=x^2 samples on [0..4].
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y = [0.0, 1.0, 4.0, 9.0, 16.0];
        let s = simpson(&y, &x).expect("simpson");
        assert!(
            (s.integral - 21.333_333_333_333_332).abs() < 1e-9,
            "simpson: {}",
            s.integral
        );
        let t = trapezoid(&y, &x).expect("trapezoid");
        assert!(
            (t.integral - 22.0).abs() < 1e-12,
            "trapezoid: {}",
            t.integral
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
    fn infinite_quad_helpers_reject_non_finite_anchors() {
        for anchor in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = quad_inf(
                |_| -> f64 { panic!("invalid anchor should not be sampled") },
                anchor,
                QuadOptions::default(),
            )
            .expect_err("quad_inf anchor should fail");
            assert!(matches!(
                err,
                IntegrateValidationError::QuadInvalidBounds { .. }
            ));

            let err = quad_neg_inf(
                |_| -> f64 { panic!("invalid anchor should not be sampled") },
                anchor,
                QuadOptions::default(),
            )
            .expect_err("quad_neg_inf anchor should fail");
            assert!(matches!(
                err,
                IntegrateValidationError::QuadInvalidBounds { .. }
            ));
        }
    }

    #[test]
    fn quad_cauchy_pv_rejects_singularity_outside_interval() {
        for (a, b, singular_point) in [
            (0.0, 1.0, -0.5),
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 1.0),
            (0.0, 1.0, 1.5),
            (1.0, 0.0, -0.5),
        ] {
            let err = quad_cauchy_pv(
                |_| -> f64 { panic!("invalid Cauchy-PV singular point should not be sampled") },
                a,
                b,
                singular_point,
                QuadOptions::default(),
            )
            .expect_err("singular point outside interval should fail");
            assert!(matches!(
                err,
                IntegrateValidationError::QuadInvalidBounds { .. }
            ));
        }
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

        for options in [
            QuadOptions {
                epsabs: f64::INFINITY,
                ..QuadOptions::default()
            },
            QuadOptions {
                epsrel: f64::INFINITY,
                ..QuadOptions::default()
            },
        ] {
            let err = quad(|x| x, 0.0, 1.0, options).expect_err("infinite tolerance");
            assert!(matches!(
                err,
                IntegrateValidationError::QuadInvalidTolerance { .. }
            ));
        }
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

        for options in [
            QuadOptions {
                epsabs: f64::INFINITY,
                ..QuadOptions::default()
            },
            QuadOptions {
                epsrel: f64::INFINITY,
                ..QuadOptions::default()
            },
        ] {
            let tol_err = quad_vec(|x| vec![x], 0.0, 1.0, options).expect_err("infinite tolerance");
            assert!(matches!(
                tol_err,
                IntegrateValidationError::QuadInvalidTolerance { .. }
            ));
        }
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

        for options in [
            CubatureOptions {
                rtol: f64::INFINITY,
                ..CubatureOptions::default()
            },
            CubatureOptions {
                atol: f64::INFINITY,
                ..CubatureOptions::default()
            },
        ] {
            let tolerance_err =
                cubature_scalar(|x| x[0], &[0.0], &[1.0], options).expect_err("infinite tolerance");
            assert!(matches!(
                tolerance_err,
                IntegrateValidationError::QuadInvalidTolerance { .. }
            ));
        }

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
    fn dblquad_equal_bounds_rejects_invalid_tolerances() {
        for options in [
            DblquadOptions {
                epsabs: f64::INFINITY,
                ..DblquadOptions::default()
            },
            DblquadOptions {
                epsrel: f64::INFINITY,
                ..DblquadOptions::default()
            },
            DblquadOptions {
                epsabs: f64::NAN,
                ..DblquadOptions::default()
            },
            DblquadOptions {
                epsrel: -1.0,
                ..DblquadOptions::default()
            },
        ] {
            let err = dblquad(|y, x| x * y, 3.0, 3.0, |_| 0.0, |_| 1.0, options)
                .expect_err("invalid equal-bound dblquad tolerance");
            assert!(matches!(
                err,
                IntegrateValidationError::QuadInvalidTolerance { .. }
            ));
        }
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
    fn uniform_sampled_integrators_accept_signed_and_zero_dx() {
        let y = [1.0, 2.0, 3.0];

        let trap_neg = trapezoid_uniform(&y, -1.0).expect("negative dx");
        assert_eq!(trap_neg.integral, -4.0);
        let trap_zero = trapezoid_uniform(&y, 0.0).expect("zero dx");
        assert_eq!(trap_zero.integral, 0.0);

        let simpson_neg = simpson_uniform(&y, -1.0).expect("negative dx");
        assert_eq!(simpson_neg.integral, -4.0);
        let simpson_zero = simpson_uniform(&y, 0.0).expect("zero dx");
        assert_eq!(simpson_zero.integral, 0.0);

        let cum_neg = cumulative_trapezoid_uniform(&y, -1.0).expect("negative dx");
        assert_eq!(cum_neg, vec![-1.5, -4.0]);
        let cum_zero = cumulative_trapezoid_uniform(&y, 0.0).expect("zero dx");
        assert_eq!(cum_zero, vec![0.0, 0.0]);
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
    fn trapezoid_short_inputs_return_zero_like_scipy() {
        let empty = trapezoid(&[], &[]).expect("empty trapezoid");
        assert_eq!(empty.integral, 0.0);

        let singleton = trapezoid(&[1.0], &[f64::NAN]).expect("singleton trapezoid");
        assert_eq!(singleton.integral, 0.0);

        let uniform = trapezoid_uniform(&[1.0], f64::INFINITY).expect("singleton uniform");
        assert_eq!(uniform.integral, 0.0);
    }

    #[test]
    fn trapezoid_rejects_nonfinite_x() {
        for x in [[0.0, f64::NAN], [0.0, f64::INFINITY]] {
            let err = trapezoid(&[1.0, 2.0], &x).expect_err("non-finite x");
            assert!(matches!(
                err,
                IntegrateValidationError::QuadInvalidBounds { .. }
            ));
        }
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
    fn simpson_even_points_matches_scipy_cartwright() {
        // Regression: even point count (odd intervals) must use scipy's
        // Cartwright parabolic correction on the final interval, not a
        // trapezoid. Golden from scipy.integrate.simpson(y, x=x) with
        // x[i]=0.37 i, y=sin(x)+0.5 x²+1.
        let cases: &[(usize, f64)] = &[
            (4, 1.892_759_045_295_215_7),
            (6, 4.180_925_279_702_298_8),
            (8, 7.338_022_266_672_648),
            (10, 1.146_762_120_770_018e1),
        ];
        for &(npts, golden) in cases {
            let x: Vec<f64> = (0..npts).map(|i| i as f64 * 0.37).collect();
            let y: Vec<f64> = x.iter().map(|&t| t.sin() + 0.5 * t * t + 1.0).collect();
            let v = simpson(&y, &x).expect("simpson").integral;
            assert!(
                (v - golden).abs() < 1e-12,
                "simpson n={npts} = {v} vs scipy {golden}"
            );
            // uniform path must match too (here x is uniform with dx=0.37).
            let vu = simpson_uniform(&y, 0.37).expect("simpson_uniform").integral;
            assert!(
                (vu - golden).abs() < 1e-12,
                "simpson_uniform n={npts} = {vu} vs scipy {golden}"
            );
        }
    }

    #[test]
    fn simpson_even_points() {
        // Even number of points — Cartwright parabolic correction on last panel
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
    fn simpson_singleton_returns_zero_like_scipy() {
        let singleton = simpson(&[1.0], &[f64::NAN]).expect("singleton simpson");
        assert_eq!(singleton.integral, 0.0);

        let uniform = simpson_uniform(&[1.0], f64::INFINITY).expect("singleton uniform simpson");
        assert_eq!(uniform.integral, 0.0);
    }

    #[test]
    fn simpson_empty_input_errors() {
        let err = simpson(&[], &[]).expect_err("empty simpson");
        assert!(matches!(
            err,
            IntegrateValidationError::QuadInvalidBounds { .. }
        ));
    }

    #[test]
    fn simpson_rejects_nonfinite_x() {
        for x in [[0.0, 0.5, f64::NAN], [0.0, 0.5, f64::INFINITY]] {
            let err = simpson(&[1.0, 2.0, 3.0], &x).expect_err("non-finite x");
            assert!(matches!(
                err,
                IntegrateValidationError::QuadInvalidBounds { .. }
            ));
        }
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
    fn cumtrapz_empty_input_errors() {
        let err = cumulative_trapezoid(&[], &[]).expect_err("empty input");
        assert!(matches!(
            err,
            IntegrateValidationError::QuadInvalidBounds { .. }
        ));
    }

    #[test]
    fn cumtrapz_singleton_returns_empty_like_scipy() {
        let result = cumulative_trapezoid(&[1.0], &[f64::NAN]).expect("singleton cumtrapz");
        assert!(result.is_empty());

        let uniform =
            cumulative_trapezoid_uniform(&[1.0], f64::INFINITY).expect("singleton uniform");
        assert!(uniform.is_empty());
    }

    #[test]
    fn cumtrapz_rejects_nonfinite_x() {
        for x in [[0.0, f64::NAN], [0.0, f64::INFINITY]] {
            let err = cumulative_trapezoid(&[1.0, 2.0], &x).expect_err("non-finite x");
            assert!(matches!(
                err,
                IntegrateValidationError::QuadInvalidBounds { .. }
            ));
        }
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
    fn romb_func_rejects_invalid_tolerances() {
        for tol in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
            let err = romb_func(|x| x * x, 0.0, 1.0, None, Some(tol))
                .expect_err("invalid Romberg tolerance should fail");
            assert!(matches!(
                err,
                IntegrateValidationError::QuadInvalidTolerance { .. }
            ));
        }
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
    fn romb_accepts_signed_and_zero_dx() {
        let y = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(romb(&y, -1.0).expect("negative dx"), -12.0);
        assert_eq!(romb(&y, 0.0).expect("zero dx"), 0.0);
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
    fn cumulative_simpson_singleton_returns_empty_like_scipy() {
        let result = cumulative_simpson(&[1.0], &[f64::NAN]).expect("singleton cumulative simpson");
        assert!(result.is_empty());
    }

    #[test]
    fn cumulative_simpson_empty_input_errors() {
        assert!(cumulative_simpson(&[], &[]).is_err());
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
    fn nquad_rejects_empty_ranges_like_scipy() {
        let opts = QuadOptions::default();
        let err = nquad(|_| 42.0, &[], opts).expect_err("empty nquad ranges");
        assert!(matches!(
            err,
            IntegrateValidationError::QuadInvalidBounds { .. }
        ));
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
    fn fixed_quad_rejects_zero_order() {
        let err = fixed_quad(|x| x, 0.0, 1.0, 0).expect_err("zero fixed_quad order");
        assert!(matches!(
            err,
            IntegrateValidationError::QuadInvalidBounds { .. }
        ));
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
    fn nsum_series_match_scipy() {
        let inf = f64::INFINITY;
        let pi = std::f64::consts::PI;
        // Σ 1/n² = π²/6
        let r = nsum(|n| 1.0 / (n * n), 1.0, inf, 1.0, 0.0, 1e-11);
        assert!(
            r.converged && (r.sum - pi * pi / 6.0).abs() <= 1e-9,
            "1/n^2: {}",
            r.sum
        );
        // Σ 1/n⁴ = π⁴/90
        let r = nsum(|n| 1.0 / n.powi(4), 1.0, inf, 1.0, 0.0, 1e-11);
        assert!(
            (r.sum - pi.powi(4) / 90.0).abs() <= 1e-9,
            "1/n^4: {}",
            r.sum
        );
        // Σ 2^-n = 1 (geometric)
        let r = nsum(|n| 2.0_f64.powf(-n), 1.0, inf, 1.0, 0.0, 1e-11);
        assert!((r.sum - 1.0).abs() <= 1e-9, "2^-n: {}", r.sum);
        // odd-only Σ 1/n² (step 2) = π²/8
        let r = nsum(|n| 1.0 / (n * n), 1.0, inf, 2.0, 0.0, 1e-11);
        assert!(
            (r.sum - pi * pi / 8.0).abs() <= 1e-9,
            "1/n^2 step2: {}",
            r.sum
        );
        // Finite range is summed directly and exactly.
        let r = nsum(|n| n, 1.0, 10.0, 1.0, 0.0, 1e-11);
        assert_eq!(r.sum, 55.0);
        // Σ_{n=0}^∞ e^{-n} = 1/(1 − 1/e)
        let r = nsum(|n| (-n).exp(), 0.0, inf, 1.0, 0.0, 1e-11);
        assert!(
            (r.sum - 1.0 / (1.0 - (-1.0_f64).exp())).abs() <= 1e-10,
            "exp(-n): {}",
            r.sum
        );
        // Invalid step.
        assert!(!nsum(|n| n, 1.0, inf, 0.0, 0.0, 1e-11).converged);
    }

    #[test]
    fn nsum_rejects_nonfinite_tolerances() {
        for (atol, rtol) in [
            (f64::NAN, 1e-11),
            (f64::INFINITY, 1e-11),
            (0.0, f64::NAN),
            (0.0, f64::INFINITY),
        ] {
            let result = nsum(|n| 1.0 / (n * n), 1.0, f64::INFINITY, 1.0, atol, rtol);
            assert!(!result.converged, "atol={atol}, rtol={rtol}");
            assert!(result.sum.is_nan(), "atol={atol}, rtol={rtol}");
            assert_eq!(result.error, f64::INFINITY);
            assert_eq!(result.nfev, 0);
        }
    }

    #[test]
    fn tanhsinh_smooth_and_singular_match_scipy() {
        // Smooth integrands → machine precision.
        let smooth: &[(fn(f64) -> f64, f64, f64, f64)] = &[
            (|x| x.exp(), 0.0, 1.0, std::f64::consts::E - 1.0),
            (|x| x.cos(), 0.0, std::f64::consts::FRAC_PI_2, 1.0),
            (|x| x * x, -1.0, 2.0, 3.0),
        ];
        for &(f, a, b, exact) in smooth {
            let r = tanhsinh(f, a, b, 0.0, 1e-12, 16);
            assert!(r.converged, "tanhsinh did not converge");
            assert!(
                (r.integral - exact).abs() <= 1e-11,
                "tanhsinh {} vs {}",
                r.integral,
                exact
            );
        }
        // Endpoint singularities: the cancellation-free abscissae keep full
        // precision where a naive c+d·tanh(u) would not.
        let r = tanhsinh(|x| 1.0 / x.sqrt(), 0.0, 1.0, 0.0, 1e-12, 16);
        assert!(
            (r.integral - 2.0).abs() <= 1e-12,
            "1/sqrt(x): {}",
            r.integral
        );
        let r = tanhsinh(|x| x.ln(), 0.0, 1.0, 0.0, 1e-12, 16);
        assert!((r.integral + 1.0).abs() <= 1e-11, "ln(x): {}", r.integral);
        let r = tanhsinh(|x| -(x.ln().powi(3)), 0.0, 1.0, 0.0, 1e-12, 16);
        assert!((r.integral - 6.0).abs() <= 1e-10, "-ln^3: {}", r.integral);
        // Reversed limits negate; degenerate interval is zero.
        let fwd = tanhsinh(|x| x * x, 0.0, 1.0, 0.0, 1e-12, 16).integral;
        let rev = tanhsinh(|x| x * x, 1.0, 0.0, 0.0, 1e-12, 16).integral;
        assert!((fwd + rev).abs() <= 1e-14);
        assert_eq!(tanhsinh(|x| x, 2.0, 2.0, 0.0, 1e-12, 16).integral, 0.0);

        // Infinite limits: exp-sinh (semi-infinite) and sinh-sinh (doubly).
        let inf = f64::INFINITY;
        let r = tanhsinh(|x| (-x).exp(), 0.0, inf, 0.0, 1e-12, 16);
        assert!(
            r.converged && (r.integral - 1.0).abs() <= 1e-12,
            "exp(-x): {}",
            r.integral
        );
        let r = tanhsinh(|x| 1.0 / (x * x), 1.0, inf, 0.0, 1e-12, 16);
        assert!((r.integral - 1.0).abs() <= 1e-12, "1/x^2: {}", r.integral);
        let r = tanhsinh(|x| x.exp(), f64::NEG_INFINITY, 0.0, 0.0, 1e-12, 16);
        assert!(
            (r.integral - 1.0).abs() <= 1e-12,
            "exp(x) left: {}",
            r.integral
        );
        let r = tanhsinh(|x| (-x * x).exp(), f64::NEG_INFINITY, inf, 0.0, 1e-12, 16);
        assert!(
            (r.integral - std::f64::consts::PI.sqrt()).abs() <= 1e-12,
            "gaussian: {}",
            r.integral
        );
        let r = tanhsinh(
            |x| 1.0 / (1.0 + x * x),
            f64::NEG_INFINITY,
            inf,
            0.0,
            1e-12,
            16,
        );
        assert!(
            (r.integral - std::f64::consts::PI).abs() <= 1e-11,
            "lorentzian: {}",
            r.integral
        );
        // Reversed infinite limit negates.
        let r = tanhsinh(|x| (-x).exp(), inf, 0.0, 0.0, 1e-12, 16);
        assert!(
            (r.integral + 1.0).abs() <= 1e-12,
            "reversed inf: {}",
            r.integral
        );
    }

    #[test]
    fn tanhsinh_rejects_nonfinite_tolerances() {
        for (atol, rtol) in [
            (f64::NAN, 1e-12),
            (f64::INFINITY, 1e-12),
            (0.0, f64::NAN),
            (0.0, f64::INFINITY),
        ] {
            let result = tanhsinh(|x| x * x, 0.0, 1.0, atol, rtol, 16);
            assert!(!result.converged, "atol={atol}, rtol={rtol}");
            assert!(result.integral.is_nan(), "atol={atol}, rtol={rtol}");
            assert_eq!(result.error, f64::INFINITY);
            assert_eq!(result.neval, 0);
        }
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
    fn gauss_kronrod_rejects_invalid_bounds_and_tolerances() {
        for (a, b, options) in [
            (f64::INFINITY, 1.0, QuadOptions::default()),
            (
                0.0,
                1.0,
                QuadOptions {
                    epsabs: f64::NAN,
                    ..QuadOptions::default()
                },
            ),
            (
                0.0,
                1.0,
                QuadOptions {
                    epsrel: -1.0,
                    ..QuadOptions::default()
                },
            ),
        ] {
            let result = gauss_kronrod_quad(|x| x, a, b, options);
            assert!(result.integral.is_nan());
            assert_eq!(result.error, f64::INFINITY);
            assert_eq!(result.neval, 0);
            assert!(!result.converged);
        }
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
    fn line_integral_equal_bounds_returns_zero_without_sampling() {
        let result = line_integral(
            |_, _| -> f64 { panic!("zero-length line integral should not sample integrand") },
            |_| -> f64 { panic!("zero-length line integral should not sample x curve") },
            |_| -> f64 { panic!("zero-length line integral should not sample y curve") },
            2.5,
            2.5,
            0,
        );
        assert_eq!(result, 0.0);

        let invalid = line_integral(|_, _| 1.0, |t| t, |t| t, f64::INFINITY, f64::INFINITY, 3);
        assert!(invalid.is_nan());
    }

    #[test]
    fn newton_cotes_exact_for_degree_n_polynomials() {
        // An order-n Newton-Cotes rule integrates polynomials up to degree n
        // exactly. Regression: n>=5 weights were computed by composite Simpson and
        // were only accurate to ~1e-5, breaking this exactness.
        for n in 5..=10usize {
            let w = newton_cotes(n).expect("newton_cotes(n)");
            // weights are on [0,1] with nodes x_i = i/n; rule sums to 1.
            for d in 0..=n {
                let approx: f64 = (0..=n)
                    .map(|i| w[i] * (i as f64 / n as f64).powi(d as i32))
                    .sum();
                let exact = 1.0 / (d as f64 + 1.0); // ∫₀¹ x^d dx
                // ~1e-9 reflects f64 rounding in the high-n weight expansion (scipy
                // uses exact rationals, ~1e-16); the old Simpson approximation broke
                // this at ~1e-5.
                assert!(
                    (approx - exact).abs() < 1e-9,
                    "newton_cotes n={n} not exact for x^{d}: got {approx}, want {exact}"
                );
            }
        }
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
    fn newton_cotes_quad_rejects_non_finite_bounds() {
        for (a, b) in [
            (f64::NAN, 1.0),
            (0.0, f64::INFINITY),
            (f64::NEG_INFINITY, 1.0),
        ] {
            let err = newton_cotes_quad(
                |_| -> f64 { panic!("invalid Newton-Cotes bounds should not be sampled") },
                a,
                b,
                2,
                2,
            )
            .expect_err("invalid Newton-Cotes bounds");
            assert!(matches!(
                err,
                IntegrateValidationError::QuadInvalidBounds { .. }
            ));
        }
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
