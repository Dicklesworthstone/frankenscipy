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
    /// Estimated absolute error in the maximum-component norm, including roundoff.
    pub error: f64,
    /// Number of function evaluations.
    pub neval: usize,
    /// Whether the requested accuracy was achieved.
    pub converged: bool,
}

/// Rule selector for [`cubature`], SciPy's `rule=` argument.
///
/// Each rule yields a higher-order estimate and an embedded lower-order one; a region's error
/// is their difference. Every rule avoids the region boundary, which is what `points` relies
/// on.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CubatureRule {
    /// SciPy's `"gauss-kronrod"`: the same rule as [`CubatureRule::Gk21`].
    GaussKronrod,
    /// `"gk21"` (SciPy's default): the tensor product of the 21-point Kronrod rule in every
    /// dimension (21^ndim evaluations per region), error against the embedded 10-point Gauss
    /// product.
    #[default]
    Gk21,
    /// `"gk15"`: the tensor product of the 15-point Kronrod rule, error against the embedded
    /// 7-point Gauss product.
    Gk15,
    /// `"genz-malik"`: the degree-7 Genz–Malik rule with its embedded degree-5 rule,
    /// 2^ndim + 2·ndim² + 2·ndim + 1 evaluations per region. Defined for ndim ≥ 2 only.
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
    /// The cubature rule applied on each region.
    pub rule: CubatureRule,
    /// Points where the integrand must not be evaluated (e.g. singularities). The initial
    /// region is split so that each point lies on subregion boundaries, which no rule
    /// evaluates. Points on an infinite axis are mapped through the same transform.
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

#[derive(Debug)]
struct QuadVecPanel {
    a: f64,
    b: f64,
    integral: Vec<f64>,
    error: f64,
    roundoff: f64,
}

impl PartialEq for QuadVecPanel {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for QuadVecPanel {}

impl PartialOrd for QuadVecPanel {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QuadVecPanel {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.error
            .total_cmp(&other.error)
            .then_with(|| other.a.total_cmp(&self.a))
            .then_with(|| other.b.total_cmp(&self.b))
    }
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

/// QUADPACK's diagnostics for one integration: `scipy.integrate.quad(..., full_output=1)`'s
/// `ier`, message and `infodict`.
#[derive(Debug, Clone, PartialEq)]
pub struct QuadInfo {
    /// QUADPACK's `ier`: 0 success, 1 `limit` subintervals reached, 2 roundoff prevents the
    /// tolerance, 3 extremely bad integrand behaviour, 4 roundoff in the extrapolation table,
    /// 5 the integral is probably divergent or slowly convergent.
    pub ier: u8,
    /// SciPy's message for `ier` (empty on success).
    pub message: String,
    /// Number of subintervals produced (`infodict["last"]`).
    pub last: usize,
    /// Left ends, right ends, integrals and error estimates of the subintervals. For an
    /// infinite range they are on the transformed variable `t ∈ (0, 1]`, as in SciPy.
    pub alist: Vec<f64>,
    pub blist: Vec<f64>,
    pub rlist: Vec<f64>,
    pub elist: Vec<f64>,
    /// Subinterval indices (0-based) in decreasing order of error estimate.
    pub iord: Vec<usize>,
}

/// `scipy.integrate.quad(f, a, b, epsabs, epsrel, limit)`: adaptive QUADPACK integration.
///
/// Finite `[a, b]` runs QAGSE: 21-point Gauss–Kronrod panels, the worst panel bisected first,
/// at most `limit` subintervals, and Wynn's epsilon algorithm extrapolating the sum (so
/// integrable endpoint singularities such as `x^-0.9` converge). An infinite bound runs QAGIE
/// on `x = bound ± (1 − t)/t`. `b < a` integrates over `[b, a]` and negates, as SciPy does.
/// `converged` is QUADPACK's `ier == 0`; [`quad_full_output`] reports `ier` and the intervals.
///
/// This used to be a recursive GK15 bisection that treated `limit` as a recursion depth (up to
/// 2^limit panels), refined every unconverged panel, had no extrapolation, rejected infinite
/// bounds, and truncated [`quad_inf`]'s mapped range at `t = 1 − 1e-10` (frankenscipy-1ksfv.8).
///
/// When QUADPACK stops short (`ier` 1–5) this raises SciPy's `IntegrationWarning` with
/// SciPy's message, as `quad` without `full_output` does; [`quad_full_output`] returns the
/// message instead.
///
/// # Errors
/// NaN bounds, negative or non-finite tolerances, and QUADPACK's invalid input
/// (`epsabs <= 0` with `epsrel < max(50·eps, 5e-29)`, SciPy's ValueError).
pub fn quad<F>(
    f: F,
    a: f64,
    b: f64,
    options: QuadOptions,
) -> Result<QuadResult, IntegrateValidationError>
where
    F: Fn(f64) -> f64,
{
    quad_full_output(f, a, b, &[], options).map(warn_unless_converged)
}

/// SciPy's `quad` without `full_output` turns QUADPACK's `ier` 1–5 into an `IntegrationWarning`
/// carrying the message `full_output` would have returned.
fn warn_unless_converged((result, info): (QuadResult, QuadInfo)) -> QuadResult {
    if info.ier != 0 {
        fsci_runtime::warn(
            fsci_runtime::WarningCategory::IntegrationWarning,
            info.message,
        );
    }
    result
}

/// [`quad`] with breakpoints: `scipy.integrate.quad(f, a, b, points=points)`. The points
/// strictly inside `(min(a, b), max(a, b))` become initial panel edges (QUADPACK QAGPE), so a
/// kink or singularity there is never sampled across; the rest are ignored, as in SciPy.
///
/// # Errors
/// As [`quad`], and infinite bounds (SciPy: "Infinity inputs cannot be used with break
/// points").
pub fn quad_points<F>(
    f: F,
    a: f64,
    b: f64,
    points: &[f64],
    options: QuadOptions,
) -> Result<QuadResult, IntegrateValidationError>
where
    F: Fn(f64) -> f64,
{
    quad_full_output(f, a, b, points, options).map(warn_unless_converged)
}

/// [`quad`] / [`quad_points`] with SciPy's `full_output=1` diagnostics.
///
/// # Errors
/// As [`quad_points`].
pub fn quad_full_output<F>(
    f: F,
    a: f64,
    b: f64,
    points: &[f64],
    options: QuadOptions,
) -> Result<(QuadResult, QuadInfo), IntegrateValidationError>
where
    F: Fn(f64) -> f64,
{
    if a.is_nan() || b.is_nan() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "integration bounds must not be NaN".to_string(),
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
    let limit = options.limit.max(1);

    // Exact equality, as SciPy: a tiny but nonzero interval can still carry a large integral.
    if a == b {
        let result = QuadResult {
            integral: 0.0,
            error: 0.0,
            neval: 0,
            // status: a == b exactly, the integral is 0
            converged: true,
        };
        let info = QuadInfo {
            ier: 0,
            message: String::new(),
            last: 0,
            alist: Vec::new(),
            blist: Vec::new(),
            rlist: Vec::new(),
            elist: Vec::new(),
            iord: Vec::new(),
        };
        return Ok((result, info));
    }

    let flip = b < a;
    let (lo, hi) = if flip { (b, a) } else { (a, b) };
    let mut g = |x: f64| f(x);
    let (epsabs, epsrel) = (options.epsabs, options.epsrel);
    let run = if lo.is_finite() && hi.is_finite() {
        let mut inside: Vec<f64> = points
            .iter()
            .copied()
            .filter(|&p| lo < p && p < hi)
            .collect();
        inside.sort_by(f64::total_cmp);
        inside.dedup();
        if inside.is_empty() {
            crate::quadpack::qagse(&mut g, lo, hi, epsabs, epsrel, limit)
        } else {
            crate::quadpack::qagpe(&mut g, lo, hi, &inside, epsabs, epsrel, limit)
        }
    } else {
        if !points.is_empty() {
            return Err(IntegrateValidationError::QuadInvalidBounds {
                detail: "Infinity inputs cannot be used with break points.".to_string(),
            });
        }
        match (lo.is_finite(), hi.is_finite()) {
            (true, false) => crate::quadpack::qagie(&mut g, lo, 1, epsabs, epsrel, limit),
            (false, true) => crate::quadpack::qagie(&mut g, hi, -1, epsabs, epsrel, limit),
            _ => crate::quadpack::qagie(&mut g, 0.0, 2, epsabs, epsrel, limit),
        }
    };
    if run.ier == 6 {
        return Err(IntegrateValidationError::QuadInvalidTolerance {
            detail: "If 'epsabs'<=0, 'epsrel' must be greater than both 5e-29 and \
                     50*(machine epsilon)."
                .to_string(),
        });
    }
    let result = QuadResult {
        integral: if flip { -run.result } else { run.result },
        error: run.abserr,
        neval: run.neval,
        // status: QUADPACK ier == 0 (abserr <= max(epsabs, epsrel*|result|) was reached)
        converged: run.ier == 0,
    };
    let info = QuadInfo {
        ier: run.ier,
        message: quadpack_message(run.ier, limit),
        last: run.last,
        alist: run.alist,
        blist: run.blist,
        rlist: run.rlist,
        elist: run.elist,
        iord: run.iord,
    };
    Ok((result, info))
}

/// The weight functions of `scipy.integrate.quad(f, a, b, weight=..., wvar=...)`: the
/// integral is of `f(x)·w(x)`, with `f` the smooth part.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuadWeight {
    /// `weight='cos'`, `wvar=omega`: `w(x) = cos(omega·x)` (QAWOE; QAWFE on `[a, ∞)`).
    Cos(f64),
    /// `weight='sin'`, `wvar=omega`: `w(x) = sin(omega·x)`.
    Sin(f64),
    /// `weight='alg'`, `wvar=(alpha, beta)`: `w(x) = (x−a)^alpha·(b−x)^beta` (QAWSE),
    /// `alpha, beta > −1`.
    Alg { alpha: f64, beta: f64 },
    /// `weight='alg-loga'`: the `Alg` weight times `log(x−a)`.
    AlgLogA { alpha: f64, beta: f64 },
    /// `weight='alg-logb'`: the `Alg` weight times `log(b−x)`.
    AlgLogB { alpha: f64, beta: f64 },
    /// `weight='alg-log'`: the `Alg` weight times `log(x−a)·log(b−x)`.
    AlgLog { alpha: f64, beta: f64 },
    /// `weight='cauchy'`, `wvar=c`: `w(x) = 1/(x−c)`, the Cauchy principal value when `c` is
    /// inside the interval (QAWCE); `c` must not be an endpoint.
    Cauchy(f64),
}

/// The controls only the cos/sin weights use: SciPy's `maxp1` (Chebyshev moment levels kept)
/// and `limlst` (cycles on an infinite range). SciPy's defaults are 50 and 50.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuadWeightOptions {
    pub maxp1: usize,
    pub limlst: usize,
}

impl Default for QuadWeightOptions {
    fn default() -> Self {
        Self {
            maxp1: 50,
            limlst: 50,
        }
    }
}

/// `scipy.integrate.quad(f, a, b, weight=..., wvar=...)`: `∫_a^b f(x)·w(x) dx` for a
/// [`QuadWeight`], by QUADPACK's weighted integrators (QAWOE/QAWFE for cos/sin, QAWSE for the
/// algebraic–logarithmic weights, QAWCE for the Cauchy principal value). `b < a` integrates
/// over `[b, a]` and negates, as SciPy does.
///
/// # Errors
/// As [`quad`]; an infinite range with an `Alg*`/`Cauchy` weight or with both ends infinite;
/// and QUADPACK's invalid input (e.g. `alpha <= -1`, `c` on an endpoint, `epsabs <= 0` on an
/// infinite cos/sin range), SciPy's ValueError.
pub fn quad_weighted<F>(
    f: F,
    a: f64,
    b: f64,
    weight: QuadWeight,
    options: QuadOptions,
    weight_options: QuadWeightOptions,
) -> Result<QuadResult, IntegrateValidationError>
where
    F: Fn(f64) -> f64,
{
    quad_weighted_full_output(f, a, b, weight, options, weight_options).map(|(result, _)| result)
}

/// [`quad_weighted`] with SciPy's `full_output=1` diagnostics. On `[a, ∞)` with a cos/sin
/// weight the subinterval lists are per CYCLE (SciPy's `rslst`/`erlst`), `last` is the number
/// of cycles, and `ier` follows QAWFE (1: `limlst` cycles, 4: the extrapolation over cycles did
/// not converge, 7: a cycle did not converge).
///
/// One divergence from SciPy 1.17.1: with `omega == 0` on `[a, ∞)`, SciPy (as netlib
/// QUADPACK's `dqawfe`) integrates from 0 instead of `a`; this integrates from `a`.
///
/// # Errors
/// As [`quad_weighted`].
pub fn quad_weighted_full_output<F>(
    f: F,
    a: f64,
    b: f64,
    weight: QuadWeight,
    options: QuadOptions,
    weight_options: QuadWeightOptions,
) -> Result<(QuadResult, QuadInfo), IntegrateValidationError>
where
    F: Fn(f64) -> f64,
{
    use crate::quadpack::{FourierMoments, qawce, qawfe, qawoe, qawse};
    if a.is_nan() || b.is_nan() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "integration bounds must not be NaN".to_string(),
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
    let limit = options.limit.max(1);
    let (epsabs, epsrel) = (options.epsabs, options.epsrel);
    let empty_info = || QuadInfo {
        ier: 0,
        message: String::new(),
        last: 0,
        alist: Vec::new(),
        blist: Vec::new(),
        rlist: Vec::new(),
        elist: Vec::new(),
        iord: Vec::new(),
    };
    if a == b {
        let result = QuadResult {
            integral: 0.0,
            error: 0.0,
            neval: 0,
            // status: a == b exactly, the integral is 0
            converged: true,
        };
        return Ok((result, empty_info()));
    }
    let flip = b < a;
    let (lo, hi) = if flip { (b, a) } else { (a, b) };
    let invalid = |detail: &str| IntegrateValidationError::QuadInvalidBounds {
        detail: detail.to_string(),
    };
    let finite_only = |run: crate::quadpack::Qags| (run, None::<crate::quadpack::Qawfe>);
    let mut g = |x: f64| f(x);
    let (run, cycles) = match weight {
        QuadWeight::Cos(omega) | QuadWeight::Sin(omega) => {
            let integr = if matches!(weight, QuadWeight::Cos(_)) {
                1
            } else {
                2
            };
            match (lo.is_finite(), hi.is_finite()) {
                (true, true) => {
                    let mut moments = FourierMoments::new(weight_options.maxp1.max(1));
                    finite_only(qawoe(
                        &mut g,
                        lo,
                        hi,
                        omega,
                        integr,
                        epsabs,
                        epsrel,
                        limit,
                        1,
                        weight_options.maxp1,
                        &mut moments,
                    ))
                }
                (true, false) | (false, true) => {
                    // (-∞, b] is SciPy's remap: ∫_{-b}^∞ f(-y)·w(y), negated for sin.
                    let negative = !lo.is_finite();
                    let start = if negative { -hi } else { lo };
                    let mut h = |x: f64| {
                        if !negative {
                            f(x)
                        } else if integr == 1 {
                            f(-x)
                        } else {
                            -f(-x)
                        }
                    };
                    let fourier = qawfe(
                        &mut h,
                        start,
                        omega,
                        integr,
                        epsabs,
                        weight_options.limlst,
                        limit,
                        weight_options.maxp1.max(1),
                    );
                    if fourier.ier == 6 {
                        return Err(IntegrateValidationError::QuadInvalidTolerance {
                            detail: "Sine or cosine weighted integrals with infinite domain \
                                     must have 'epsabs'>0 and limlst >= 3."
                                .to_string(),
                        });
                    }
                    let run = crate::quadpack::Qags {
                        result: fourier.result,
                        abserr: fourier.abserr,
                        neval: fourier.neval,
                        ier: fourier.ier,
                        last: fourier.lst,
                        alist: Vec::new(),
                        blist: Vec::new(),
                        rlist: fourier.rslst.clone(),
                        elist: fourier.erlst.clone(),
                        iord: Vec::new(),
                    };
                    (run, Some(fourier))
                }
                (false, false) => {
                    return Err(invalid(
                        "Cannot integrate with this weight from -Inf to +Inf.",
                    ));
                }
            }
        }
        QuadWeight::Alg { alpha, beta }
        | QuadWeight::AlgLogA { alpha, beta }
        | QuadWeight::AlgLogB { alpha, beta }
        | QuadWeight::AlgLog { alpha, beta } => {
            if !lo.is_finite() || !hi.is_finite() {
                return Err(invalid(
                    "Cannot integrate with this weight over an infinite interval.",
                ));
            }
            let integr = match weight {
                QuadWeight::Alg { .. } => 1,
                QuadWeight::AlgLogA { .. } => 2,
                QuadWeight::AlgLogB { .. } => 3,
                _ => 4,
            };
            if !(alpha > -1.0 && beta > -1.0) {
                return Err(invalid("Alpha and beta must be greater than -1."));
            }
            finite_only(qawse(
                &mut g, lo, hi, alpha, beta, integr, epsabs, epsrel, limit,
            ))
        }
        QuadWeight::Cauchy(c) => {
            if !lo.is_finite() || !hi.is_finite() {
                return Err(invalid(
                    "Cannot integrate with this weight over an infinite interval.",
                ));
            }
            // QUADPACK rejects only c on an endpoint; outside the interval there is no pole.
            if c.is_nan() || c == lo || c == hi {
                return Err(invalid(
                    "The Cauchy weight's wvar must not be NaN or an endpoint of the interval.",
                ));
            }
            finite_only(qawce(&mut g, lo, hi, c, epsabs, epsrel, limit))
        }
    };
    if run.ier == 6 {
        return Err(IntegrateValidationError::QuadInvalidTolerance {
            detail: "The input is invalid (QUADPACK ier=6): if 'epsabs'<=0, 'epsrel' must be \
                     greater than both 5e-29 and 50*(machine epsilon); the weighted \
                     integrators need limit >= 2 (alg) and maxp1 >= 1 (cos/sin)."
                .to_string(),
        });
    }
    let message = match &cycles {
        Some(fourier) => qawfe_message(fourier.ier),
        None => quadpack_message(run.ier, limit),
    };
    let result = QuadResult {
        integral: if flip { -run.result } else { run.result },
        error: run.abserr,
        neval: run.neval,
        // status: QUADPACK ier == 0 for the weighted integrator that ran
        converged: run.ier == 0,
    };
    let info = QuadInfo {
        ier: run.ier,
        message,
        last: run.last,
        alist: run.alist,
        blist: run.blist,
        rlist: run.rlist,
        elist: run.elist,
        iord: run.iord,
    };
    Ok((result, info))
}

/// SciPy's `quad` messages for QAWFE's cycle-level `ier` (cos/sin weight on an infinite range).
fn qawfe_message(ier: u8) -> String {
    match ier {
        0 => String::new(),
        1 => "The maximum number of cycles allowed has been achieved. One can allow more \
              cycles by increasing the value of limlst."
            .to_string(),
        4 => "The extrapolation table constructed for convergence acceleration of the series \
              formed by the integral contributions over the cycles, does not converge to \
              within the requested accuracy."
            .to_string(),
        7 => "Bad integrand behavior occurs within one or more of the cycles.".to_string(),
        _ => "Unknown error.".to_string(),
    }
}

/// SciPy's `quad` message for a QUADPACK `ier`.
fn quadpack_message(ier: u8, limit: usize) -> String {
    match ier {
        0 => String::new(),
        1 => format!(
            "The maximum number of subdivisions ({limit}) has been achieved. If increasing the \
             limit yields no improvement it is advised to analyze the integrand in order to \
             determine the difficulties. If the position of a local difficulty can be \
             determined (singularity, discontinuity) one will probably gain from splitting up \
             the interval and calling the integrator on the subranges. Perhaps a \
             special-purpose integrator should be used."
        ),
        2 => "The occurrence of roundoff error is detected, which prevents the requested \
              tolerance from being achieved. The error may be underestimated."
            .to_string(),
        3 => "Extremely bad integrand behavior occurs at some points of the integration \
              interval."
            .to_string(),
        4 => "The algorithm does not converge. Roundoff error is detected in the extrapolation \
              table. It is assumed that the requested tolerance cannot be achieved, and that \
              the returned result (if full_output = 1) is the best which can be obtained."
            .to_string(),
        5 => "The integral is probably divergent, or slowly convergent.".to_string(),
        _ => "Unknown error.".to_string(),
    }
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

/// Numerically integrate a vector-valued function over [a, b].
///
/// Adaptive GK15 quadrature refines the interval with the largest error until the
/// summed error meets the requested tolerance in the maximum-component norm.
/// `limit` bounds the number of active subintervals, rather than recursion depth.
/// Semi-infinite and infinite intervals use a change of variables to a finite
/// interval. Reversing the bounds negates the result. Nonzero intervals are sampled
/// only at interior nodes, so integrable endpoint singularities are supported.
///
/// This corresponds to `scipy.integrate.quad_vec` with `norm="max"` and
/// `quadrature="gk15"`; it does not use scalar QUADPACK's epsilon extrapolation.
///
/// # Errors
/// Rejects NaN bounds, identical infinite bounds, invalid tolerances, changing
/// output shapes, and nonzero intervals without a representable interior node.
pub fn quad_vec<F>(
    f: F,
    a: f64,
    b: f64,
    options: QuadOptions,
) -> Result<QuadVecResult, IntegrateValidationError>
where
    F: Fn(f64) -> Vec<f64>,
{
    if a.is_nan() || b.is_nan() || (a == b && !a.is_finite()) {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "integration bounds must not be NaN or the same infinity".to_string(),
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

    if a == b {
        let dim = f(a).len();
        return Ok(QuadVecResult {
            integral: vec![0.0; dim],
            error: 0.0,
            neval: 1,
            // status: a == b exactly, every component integral is 0
            converged: true,
        });
    }

    let (lo, hi, sign) = if a < b { (a, b, 1.0) } else { (b, a, -1.0) };
    let transform = match (lo.is_finite(), hi.is_finite()) {
        (true, true) => BoundTransform::Finite,
        (false, true) => BoundTransform::LowerInfinite { upper: hi },
        (true, false) => BoundTransform::UpperInfinite { lower: lo },
        (false, false) => BoundTransform::BothInfinite,
    };
    let (left, right) = transform.interval(lo, hi);
    let mapped = |t| {
        let (x, jacobian) = transform.map(t);
        f(x).into_iter().map(|value| value * jacobian).collect()
    };
    let mut neval = 0;
    let (mut integral, error, converged) =
        adaptive_gk15_vec(&mapped, left, right, &mut neval, options)?;
    for value in &mut integral {
        *value *= sign;
    }

    Ok(QuadVecResult {
        integral,
        error,
        neval,
        converged,
    })
}

/// Global error control with at most `limit` active panels and no recursive calls.
fn adaptive_gk15_vec<F>(
    f: &F,
    a: f64,
    b: f64,
    neval: &mut usize,
    options: QuadOptions,
) -> Result<(Vec<f64>, f64, bool), IntegrateValidationError>
where
    F: Fn(f64) -> Vec<f64>,
{
    let first = gauss_kronrod_15_vec(f, a, b, neval, None)?;
    let mut integral = first.integral.clone();
    let mut error = first.error;
    let mut roundoff = first.roundoff;
    let mut panels = std::collections::BinaryHeap::from([first]);

    loop {
        let total_error = error + roundoff;
        if !total_error.is_finite() || integral.iter().any(|v| !v.is_finite()) {
            return Ok((integral, f64::INFINITY, false));
        }
        let tolerance = options
            .epsabs
            .max(options.epsrel * max_abs_component(&integral));
        // Leave headroom for the accumulated roundoff estimate, as quad_vec does.
        if error <= tolerance / 8.0 && total_error <= tolerance {
            return Ok((integral, total_error, true));
        }
        if panels.len() >= options.limit.max(1) || error <= roundoff {
            return Ok((integral, total_error, false));
        }
        let worst = panels.pop().expect("at least one quadrature panel");
        let mid = worst.a.midpoint(worst.b);
        if worst.a.next_up() >= mid || mid.next_up() >= worst.b {
            return Ok((integral, total_error, false));
        }
        let dim = Some(integral.len());
        let left = gauss_kronrod_15_vec(f, worst.a, mid, neval, dim)?;
        let right = gauss_kronrod_15_vec(f, mid, worst.b, neval, dim)?;
        for (i, value) in integral.iter_mut().enumerate() {
            *value += (left.integral[i] - worst.integral[i]) + right.integral[i];
        }
        error = (error - worst.error).max(0.0) + left.error + right.error;
        roundoff += left.roundoff + right.roundoff;
        panels.push(left);
        panels.push(right);
    }
}

/// Vector-valued Gauss-Kronrod 15-point / 7-point quadrature rule on [a, b].
fn gauss_kronrod_15_vec<F>(
    f: &F,
    a: f64,
    b: f64,
    neval: &mut usize,
    dim: Option<usize>,
) -> Result<QuadVecPanel, IntegrateValidationError>
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

    if a.next_up() >= b {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "interval has no representable interior quadrature node".to_string(),
        });
    }
    let width = b - a;
    let half_length = if width.is_finite() {
        0.5 * width
    } else {
        0.5 * b - 0.5 * a
    };
    let center = a.midpoint(b);
    let mut center_value = f(center);
    *neval += 1;
    let dim = dim.unwrap_or(center_value.len());

    let mut result_kronrod = vec![0.0; dim];
    let mut result_gauss = vec![0.0; dim];
    let mut absolute = vec![0.0; dim];
    let mut samples = Vec::with_capacity(XGK.len());

    for (i, (&xgk, &wgk)) in XGK.iter().zip(WGK.iter()).enumerate() {
        // Rounding can otherwise move an interior Kronrod node onto a singular
        // endpoint after repeated refinement. Keep representable nodes inside.
        let mut x = center + half_length * xgk;
        if x <= a && a.next_up() < b {
            x = a.next_up();
        } else if x >= b && b.next_down() > a {
            x = b.next_down();
        }
        let fval = if i == 7 {
            std::mem::take(&mut center_value)
        } else {
            *neval += 1;
            f(x)
        };

        if fval.len() != dim {
            return Err(IntegrateValidationError::QuadInvalidBounds {
                detail: format!(
                    "integrand returned inconsistent vector length: expected {dim}, got {}",
                    fval.len()
                ),
            });
        }

        for j in 0..dim {
            result_kronrod[j] += wgk * fval[j];
            absolute[j] += wgk * fval[j].abs();
        }

        if i % 2 == 1 {
            let wg = WG[i / 2];
            for (acc, value) in result_gauss.iter_mut().zip(fval.iter()) {
                *acc += wg * value;
            }
        }
        samples.push(fval);
    }

    let mut deviation = vec![0.0; dim];
    for (sample, weight) in samples.iter().zip(WGK) {
        for j in 0..dim {
            deviation[j] += weight * (sample[j] - 0.5 * result_kronrod[j]).abs();
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
    let deviation = max_abs_component(&deviation) * half_length.abs();
    if deviation != 0.0 && error != 0.0 {
        error = deviation * (200.0 * (error / deviation)).powf(1.5).min(1.0);
    }
    let roundoff = 50.0 * f64::EPSILON * (max_abs_component(&absolute) * half_length.abs());
    if error.is_nan() || roundoff.is_nan() {
        error = f64::INFINITY;
    } else {
        error = error.max(roundoff);
    }

    Ok(QuadVecPanel {
        a,
        b,
        integral: result_kronrod,
        error,
        roundoff,
    })
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
    /// Absolute error tolerance for every nested 1-D integral.
    pub epsabs: f64,
    /// Relative error tolerance for every nested 1-D integral.
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
    /// Largest absolute-error estimate from any nested 1-D integral, as in SciPy.
    /// This is a diagnostic for the nested solves, not a rigorous bound on the total error.
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
/// The returned error includes every inner solve. Invalid callable bounds return
/// their validation error; a failed inner solve makes the result unconverged.
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
    // Infinite limits are integrated by `quad` (QUADPACK QAGIE), as in SciPy; they used to be
    // rejected (frankenscipy-1ksfv.8).
    if a.is_nan() || b.is_nan() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "outer integration bounds must not be NaN".to_string(),
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

    if a == b {
        return Ok(DblquadResult {
            integral: 0.0,
            error: 0.0,
            // status: outer a == b exactly, the integral is 0
            converged: true,
        });
    }

    let inner_opts = QuadOptions {
        epsabs: options.epsabs,
        epsrel: options.epsrel,
        limit: options.limit,
    };
    // An inner integral that misses its tolerance makes the whole result unconverged (SciPy
    // raises an IntegrationWarning from the inner quad); the outer flag alone cannot see it.
    let inner_converged = std::cell::Cell::new(true);
    let inner_abserr = std::cell::Cell::new(0.0_f64);
    let inner_error = std::cell::RefCell::new(None);

    // Outer integral over x, inner integral over y for each x
    let outer_result = quad(
        |x| {
            if inner_error.borrow().is_some() {
                return 0.0;
            }
            let y_lo = gfun(x);
            let y_hi = hfun(x);
            if y_lo == y_hi {
                return 0.0;
            }
            // Inner integral of f(y, x) over y ∈ [gfun(x), hfun(x)]
            match quad(|y| f(y, x), y_lo, y_hi, inner_opts) {
                Ok(r) => {
                    if !r.converged {
                        inner_converged.set(false);
                    }
                    inner_abserr.set(if r.error.is_finite() {
                        inner_abserr.get().max(r.error)
                    } else {
                        f64::INFINITY
                    });
                    r.integral
                }
                Err(error) => {
                    *inner_error.borrow_mut() = Some(error);
                    0.0
                }
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

    if let Some(error) = inner_error.into_inner() {
        return Err(error);
    }
    if !outer_result.integral.is_finite() {
        return Ok(DblquadResult {
            integral: f64::NAN,
            error: f64::INFINITY,
            converged: false,
        });
    }

    let error = if outer_result.error.is_finite() {
        outer_result.error.max(inner_abserr.get())
    } else {
        f64::INFINITY
    };
    Ok(DblquadResult {
        integral: outer_result.integral,
        error,
        converged: outer_result.converged && inner_converged.get() && error.is_finite(),
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
/// uses Simpson's 1/3 rule on the first n-2 intervals and the Cartwright
/// parabolic correction on the last interval, matching SciPy's default.
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

    Ok(CompositeQuadResult {
        integral: simpson_nonuniform(y, x),
    })
}

/// Shared sampled-data kernel, including the final parabolic interval.
fn simpson_nonuniform(y: &[f64], x: &[f64]) -> f64 {
    let n = y.len();
    let odd_len = if n % 2 == 1 { n } else { n - 1 };
    let integral = simpson_nonuniform_odd(&y[..odd_len], &x[..odd_len]);
    if n % 2 == 1 {
        return integral;
    }
    let h0 = x[n - 2] - x[n - 3];
    let h1 = x[n - 1] - x[n - 2];
    // Preserve the established zero-denominator convention for degenerate
    // final panels. Cumulative Simpson only accepts strictly increasing x.
    let last = if h1 == 0.0 {
        0.0
    } else if h0 == 0.0 {
        de_weighted_term(h1, 1.0 / 3.0, y[n - 1])
    } else if h0 == -h1 {
        de_weighted_term(h1, 1.0 / 3.0, y[n - 2])
    } else {
        cumulative_simpson_right_interval(y[n - 3], y[n - 2], y[n - 1], h0, h1)
    };
    integral + last
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
        // Exactly zero, not a floor. This branch exists for COINCIDENT points,
        // where the Simpson weights below divide by h0, h1 and their product;
        // that is undefined only when a spacing is zero. The test used to be
        // `h_prod.abs() < f64::EPSILON * 1e-10` (2.22e-26), and h_prod = h0·h1
        // carries units of x², so it asked how big the integration interval was:
        // an ordinary grid on [0, 1e-12] had every panel swapped for the
        // TRAPEZOID rule, silently and with no error (frankenscipy-bshyq).
        // Measured, identical grid, relative error against the exact integral of
        // x³: 5.451e-10 at x-scale 1, and 2.771e-5 at 1e-12 — which is the
        // trapezoid value, i.e. the rule itself had been substituted. SciPy
        // holds 5.451e-10 across that whole range and guards the same
        // denominators with an exact-zero test,
        // `np.true_divide(..., where=den != 0)` in _quadrature.py.
        if h0 == 0.0 || h1 == 0.0 {
            // Coincident points: fall back to trapezoidal rule for the two panels
            integral += 0.5 * h0 * (y[i] + y[i + 1]);
            integral += 0.5 * h1 * (y[i + 1] + y[i + 2]);
        } else {
            // Ratios of normalized spacings carry no coordinate units. In
            // particular, h0*h1 and (h0+h1)^2 must not underflow/overflow just
            // because the caller changes from seconds to another unit.
            let scale = h0.abs().max(h1.abs());
            let u = h0 / scale;
            let v = h1 / scale;
            // On a retracing grid, subtract the original spacings before
            // scaling: subtracting separately rounded u and v can erase a
            // substantial fraction of the small net span. Same-sign widths
            // are added after scaling so their sum cannot overflow.
            let sum = if h0.is_sign_negative() == h1.is_sign_negative() {
                u + v
            } else {
                (h0 + h1) / scale
            };
            let weights = [
                (sum / 6.0) * (2.0 - v / u),
                (sum / 6.0) * (sum / u) * (sum / v),
                (sum / 6.0) * (2.0 - u / v),
            ];
            integral += sampled_simpson_weighted_sum(scale, weights, [y[i], y[i + 1], y[i + 2]]);
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
///
/// The reported error is the largest estimate across all three levels, as in
/// SciPy. Invalid callable bounds propagate their validation error immediately
/// after the active outer quadrature returns, without retrying the failed solve.
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
    // Infinite limits are integrated by `quad` (QUADPACK QAGIE), as in SciPy.
    if a.is_nan() || b.is_nan() {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "outer integration bounds must not be NaN".to_string(),
        });
    }

    let inner_opts = QuadOptions {
        epsabs: options.epsabs,
        epsrel: options.epsrel,
        limit: options.limit,
    };

    // Any nested integral that misses its tolerance makes the result unconverged (see dblquad).
    let inner_converged = std::cell::Cell::new(true);
    let inner_abserr = std::cell::Cell::new(0.0_f64);
    let inner_error = std::cell::RefCell::new(None);
    let outer_result = quad(
        |x| {
            if inner_error.borrow().is_some() {
                return 0.0;
            }
            let y_lo = gfun(x);
            let y_hi = hfun(x);
            if y_lo == y_hi {
                return 0.0;
            }
            match quad(
                |y| {
                    if inner_error.borrow().is_some() {
                        return 0.0;
                    }
                    let z_lo = qfun(x, y);
                    let z_hi = rfun(x, y);
                    if z_lo == z_hi {
                        return 0.0;
                    }
                    match quad(|z| f(z, y, x), z_lo, z_hi, inner_opts) {
                        Ok(r) => {
                            if !r.converged {
                                inner_converged.set(false);
                            }
                            inner_abserr.set(if r.error.is_finite() {
                                inner_abserr.get().max(r.error)
                            } else {
                                f64::INFINITY
                            });
                            r.integral
                        }
                        Err(error) => {
                            *inner_error.borrow_mut() = Some(error);
                            0.0
                        }
                    }
                },
                y_lo,
                y_hi,
                inner_opts,
            ) {
                Ok(r) => {
                    if !r.converged {
                        inner_converged.set(false);
                    }
                    inner_abserr.set(if r.error.is_finite() {
                        inner_abserr.get().max(r.error)
                    } else {
                        f64::INFINITY
                    });
                    r.integral
                }
                Err(error) => {
                    *inner_error.borrow_mut() = Some(error);
                    0.0
                }
            }
        },
        a,
        b,
        inner_opts,
    )?;

    if let Some(error) = inner_error.into_inner() {
        return Err(error);
    }
    if !outer_result.integral.is_finite() {
        return Ok(DblquadResult {
            integral: f64::NAN,
            error: f64::INFINITY,
            converged: false,
        });
    }
    let error = if outer_result.error.is_finite() {
        outer_result.error.max(inner_abserr.get())
    } else {
        f64::INFINITY
    };
    Ok(DblquadResult {
        integral: outer_result.integral,
        error,
        converged: outer_result.converged && inner_converged.get() && error.is_finite(),
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
                // status: |R[i][i] - R[i-1][i-1]| < tol with i >= 2 (checked just above)
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

/// One integration interval for [`nquad_with_ranges`].
///
/// Ranges use SciPy's variable order: the first variable is integrated innermost.
/// The callback for variable `i` receives only the already fixed outer coordinates,
/// `args[i + 1..]`, in the same order as the integrand's arguments. Captured data
/// can supply additional parameters without allocating a new range for every call.
#[derive(Clone, Copy)]
pub enum NquadRange<'a> {
    Fixed(f64, f64),
    Dynamic(&'a dyn Fn(&[f64]) -> (f64, f64)),
}

impl std::fmt::Debug for NquadRange<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fixed(lower, upper) => formatter
                .debug_tuple("Fixed")
                .field(lower)
                .field(upper)
                .finish(),
            Self::Dynamic(_) => formatter.write_str("Dynamic(<function>)"),
        }
    }
}

/// Adaptive integration over nested, possibly nonrectangular domains.
///
/// As in `scipy.integrate.nquad`, `ranges[0]` bounds `args[0]` and is integrated
/// innermost. A dynamic range sees the outer coordinates; for example the range
/// for `args[0]` in three dimensions receives `[args[1], args[2]]`.
/// Infinite and reversed intervals are supported by the underlying [`quad`].
///
/// `options` may be empty (defaults), contain one common set of options, or have
/// one entry per variable, in the same innermost-first order as `ranges`.
/// `neval` counts evaluations of the user's integrand, and `error` is the largest
/// error estimate from any nested integration. A failed inner integration marks
/// the whole result unconverged; an invalid range or tolerance returns an error.
///
/// ```rust
/// use fsci_integrate::{nquad_with_ranges, NquadRange};
/// let inner = |outer: &[f64]| (0.0, outer[0]);
/// let result = nquad_with_ranges(
///     |_| 1.0,
///     &[NquadRange::Dynamic(&inner), NquadRange::Fixed(0.0, 1.0)],
///     &[],
/// ).unwrap();
/// assert!((result.integral - 0.5).abs() < 1e-12);
/// ```
pub fn nquad_with_ranges<F>(
    func: F,
    ranges: &[NquadRange<'_>],
    options: &[QuadOptions],
) -> Result<QuadResult, IntegrateValidationError>
where
    F: Fn(&[f64]) -> f64,
{
    if options.len() > 1 && options.len() != ranges.len() {
        return Err(IntegrateValidationError::QuadInvalidTolerance {
            detail: "nquad options must be empty, common to all variables, or one per variable"
                .to_string(),
        });
    }
    for range in ranges {
        if let NquadRange::Fixed(lower, upper) = range
            && (lower.is_nan() || upper.is_nan())
        {
            return Err(IntegrateValidationError::QuadInvalidBounds {
                detail: "nquad integration bounds must not be NaN".to_string(),
            });
        }
    }
    let default_options = [QuadOptions::default()];
    let options = if options.is_empty() {
        &default_options[..]
    } else {
        options
    };
    // Validate every level before any user callback, including levels beneath
    // an empty outer interval. Include QUADPACK's relative-tolerance floor;
    // a zero-width dummy quad would return before checking that floor.
    for &option in options {
        if !option.epsabs.is_finite()
            || !option.epsrel.is_finite()
            || option.epsabs < 0.0
            || option.epsrel < 0.0
            || (option.epsabs <= 0.0 && option.epsrel < (50.0 * f64::EPSILON).max(5e-29))
        {
            return Err(IntegrateValidationError::QuadInvalidTolerance {
                detail: "nquad tolerances must be finite and non-negative; with zero epsabs, epsrel must meet QUADPACK's tolerance floor".to_string(),
            });
        }
    }
    nquad_run(
        &func,
        ranges.len(),
        &|dim, args: &[f64]| match ranges[dim] {
            NquadRange::Fixed(lower, upper) => (lower, upper),
            NquadRange::Dynamic(bounds) => bounds(&args[dim + 1..]),
        },
        options,
        true,
    )
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
/// For coordinate-dependent bounds and different options per variable, use
/// [`nquad_with_ranges`]. `neval` counts only calls to `func`.
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
    let ndim = ranges.len();
    // scipy's nquad with an empty ranges list evaluates the function once at
    // the (empty) point vector and returns that value with neval = 1 — the
    // 0-D degenerate case. The earlier blanket rejection broke the
    // diff_integrate_romberg_nquad live harness (frankenscipy-0y8z8
    // follow-up, ia47s).
    if ndim == 0 {
        let value = func(&[]);
        let converged = value.is_finite();
        return Ok(QuadResult {
            integral: value,
            error: 0.0,
            neval: 1,
            converged,
        });
    }
    // All nquad ranges are static, so reject malformed nested bounds before
    // starting the outer adaptive loop.  Deferring this to an inner `quad`
    // call leaves the outer callback to discover the same error repeatedly.
    // Besides wasting work, that was the source of long-running interrupted
    // nquad tests that could outlive their cargo parent.
    // Infinite limits are integrated by `quad` (QUADPACK QAGIE), as in SciPy's nquad; only NaN
    // is malformed.
    if ranges
        .iter()
        .any(|&(lower, upper)| lower.is_nan() || upper.is_nan())
    {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "nquad integration bounds must not be NaN".to_string(),
        });
    }

    nquad_run(
        &func,
        ndim,
        &|dim, _args: &[f64]| ranges[dim],
        &[options],
        false,
    )
}

fn nquad_run<F, R>(
    func: &F,
    ndim: usize,
    range: &R,
    options: &[QuadOptions],
    innermost_first: bool,
) -> Result<QuadResult, IntegrateValidationError>
where
    F: Fn(&[f64]) -> f64,
    R: Fn(usize, &[f64]) -> (f64, f64),
{
    use std::cell::RefCell;

    if ndim == 0 {
        let integral = func(&[]);
        return Ok(QuadResult {
            integral,
            error: 0.0,
            neval: 1,
            converged: integral.is_finite(),
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
    // Cleared by ANY nested quad that misses its tolerance, at any depth.
    let all_converged = std::cell::Cell::new(true);
    // SciPy's `_NQuad` keeps `self.abserr = max(self.abserr, abserr)` over EVERY nested quad
    // call, so an unresolved inner integral shows in the reported error even when the outer
    // integrand is smooth (frankenscipy-szq1n.7; the error used to be a literal 0.0).
    let max_abserr = std::cell::Cell::new(0.0_f64);

    let result = nquad_inner(
        func,
        range,
        options,
        &NquadState {
            args: &args,
            total_neval: &total_neval,
            inner_error: &inner_error,
            all_converged: &all_converged,
            max_abserr: &max_abserr,
        },
        0,
        ndim,
        innermost_first,
    )?;
    let error = max_abserr.get();

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
        error,
        neval: *total_neval.borrow(),
        converged: all_converged.get(),
    })
}

/// Shared interior state of one `nquad` evaluation (the callbacks are `Fn`, so it is borrowed).
struct NquadState<'s> {
    args: &'s std::cell::RefCell<Vec<f64>>,
    total_neval: &'s std::cell::RefCell<usize>,
    inner_error: &'s std::cell::RefCell<Option<IntegrateValidationError>>,
    all_converged: &'s std::cell::Cell<bool>,
    max_abserr: &'s std::cell::Cell<f64>,
}

impl NquadState<'_> {
    /// Fold one nested quad's verdict into the shared state.
    fn record(&self, result: &QuadResult) {
        if !result.converged {
            self.all_converged.set(false);
        }
        self.max_abserr.set(self.max_abserr.get().max(result.error));
    }
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

/// Returns this level's integral; its evaluation count, convergence and abserr go into `state`.
fn nquad_inner<F, R>(
    func: &F,
    range: &R,
    options: &[QuadOptions],
    state: &NquadState<'_>,
    depth: usize,
    ndim: usize,
    innermost_first: bool,
) -> Result<f64, IntegrateValidationError>
where
    F: Fn(&[f64]) -> f64,
    R: Fn(usize, &[f64]) -> (f64, f64),
{
    let NquadState {
        args, inner_error, ..
    } = *state;
    let dim = if innermost_first {
        ndim - 1 - depth
    } else {
        depth
    };
    let (a, b) = range(dim, &args.borrow());
    let level_options = options[if options.len() == 1 { 0 } else { dim }];

    if depth == ndim - 1 {
        // Innermost dimension: call quad directly
        let result = quad(
            |x| {
                args.borrow_mut()[dim] = x;
                *state.total_neval.borrow_mut() += 1;
                func(&args.borrow())
            },
            a,
            b,
            level_options,
        )?;
        state.record(&result);
        Ok(result.integral)
    } else {
        // Outer dimension: integrate by nesting. Capture the first
        // inner failure so the caller sees it instead of getting a
        // silently-zeroed result.
        let result = quad(
            |x| {
                // An inner failure is terminal for this nquad invocation.
                // `quad` may ask its callback for additional samples before
                // returning, but retrying the already-failed nested solve at
                // every sample can turn one malformed range into a long-lived
                // hot test process. Preserve the first typed error and make
                // the remaining outer samples cheap until it is propagated.
                if inner_error.borrow().is_some() {
                    return 0.0;
                }
                args.borrow_mut()[dim] = x;
                match nquad_inner(
                    func,
                    range,
                    options,
                    state,
                    depth + 1,
                    ndim,
                    innermost_first,
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
            level_options,
        )?;
        state.record(&result);
        Ok(result.integral)
    }
}

/// Adaptive cubature of a multidimensional vector-valued function.
///
/// This mirrors the public shape of `scipy.integrate.cubature`: `a` and `b`
/// define the lower and upper corners of a hyper-rectangle, the integrand takes
/// one point at a time as `&[f64]`, and the result contains componentwise
/// integral and error estimates. Infinite limits are handled with smooth
/// transformations to finite coordinates. Scaled accumulation preserves finite
/// integrals across extreme axis lengths and integrand magnitudes.
///
/// Nonzero regions need at least one representable interior point on every axis.
/// An unsplittable region or an unrepresentable panel estimate returns
/// `NotConverged`; non-finite totals never certify convergence. In particular,
/// individually underflowed panel integrals remain unconverged even if their
/// mathematical sum would be representable.
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

    let initial_regions = initial_cubature_regions(&transformed_ranges, &transforms, &options)?;
    // Fail before any callback when the interior-only rule has no legal node.
    for (lower, upper) in &initial_regions {
        for (&lo, &hi) in lower.iter().zip(upper) {
            if lo.min(hi).next_up() >= lo.max(hi) {
                return Err(IntegrateValidationError::QuadInvalidBounds {
                    detail: "nonzero cubature interval has no representable interior node"
                        .to_string(),
                });
            }
        }
    }
    let sample_point = initial_regions
        .first()
        .map(|(lower, upper)| {
            lower
                .iter()
                .zip(upper.iter())
                .map(|(&left, &right)| left.midpoint(right))
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| {
            transformed_ranges
                .iter()
                .map(|(lower, upper)| lower.midpoint(*upper))
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
    if sample_x.iter().any(|value| !value.is_finite()) {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "cubature sample point is not representable".to_string(),
        });
    }
    let sample = f(&sample_x);
    validate_cubature_output(&sample, sample.len())?;
    let output_dim = sample.len();

    let mut neval = 1;
    let mut regions = Vec::with_capacity(initial_regions.len());
    let mut arithmetic_representable = true;
    for (initial_a, initial_b) in initial_regions {
        let initial = estimate_cubature_region(
            &f,
            &transforms,
            &initial_a,
            &initial_b,
            output_dim,
            options.rule,
        )?;
        neval += initial.neval;
        arithmetic_representable &= initial.representable;
        regions.push(initial.region);
    }
    let mut subdivisions = 0usize;

    // SciPy's loop: take the region with the largest error and split it at its midpoint in
    // EVERY dimension (2^ndim subregions, `_split_subregion`).
    while arithmetic_representable && subdivisions < options.max_subdivisions {
        let (estimate, error) = sum_cubature_regions(&regions, output_dim);
        if componentwise_cubature_converged(&estimate, &error, options.atol, options.rtol)
            || estimate.iter().chain(&error).any(|value| !value.is_finite())
        {
            break;
        }
        let Some(split_index) = largest_error_region(&regions) else {
            break;
        };
        let region = &regions[split_index];
        let midpoint: Vec<f64> = region
            .a
            .iter()
            .zip(region.b.iter())
            .map(|(&lower, &upper)| lower.midpoint(upper))
            .collect();
        // Keep the parent if either child would have no interior node. Without
        // this check refinement repeats unchanged regions or samples a boundary.
        if region
            .a
            .iter()
            .zip(&region.b)
            .zip(&midpoint)
            .any(|((&lo, &hi), &mid)| {
                let (left, right) = (lo.min(hi), lo.max(hi));
                left.next_up() >= mid || mid.next_up() >= right
            })
        {
            break;
        }
        let region = regions.swap_remove(split_index);
        for corner in 0..(1_usize << region.a.len()) {
            let (sub_a, sub_b): (Vec<f64>, Vec<f64>) = (0..region.a.len())
                .map(|k| {
                    if (corner >> k) & 1 == 0 {
                        (region.a[k], midpoint[k])
                    } else {
                        (midpoint[k], region.b[k])
                    }
                })
                .unzip();
            let sub = estimate_cubature_region(
                &f,
                &transforms,
                &sub_a,
                &sub_b,
                output_dim,
                options.rule,
            )?;
            neval += sub.neval;
            arithmetic_representable &= sub.representable;
            regions.push(sub.region);
        }
        subdivisions += 1;
    }

    let (estimate, error) = sum_cubature_regions(&regions, output_dim);
    let status = if arithmetic_representable
        && componentwise_cubature_converged(&estimate, &error, options.atol, options.rtol)
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
    representable: bool,
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
    if a.iter().chain(b).any(|value| value.is_nan()) {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "cubature bounds must not contain NaN".to_string(),
        });
    }
    // Both tensor rules and all-axis subdivision have exponential work counts.
    // Check arithmetic before shifts, allocations, or user callbacks.
    let dimensions = u32::try_from(a.len()).ok();
    let corners = dimensions.and_then(|n| 1_usize.checked_shl(n));
    let nodes = dimensions.and_then(|n| match options.rule {
        CubatureRule::GaussKronrod | CubatureRule::Gk21 => 21_usize.checked_pow(n),
        CubatureRule::Gk15 => 15_usize.checked_pow(n),
        CubatureRule::GenzMalik => corners.and_then(|count| {
            count.checked_add(2 * a.len() * a.len() + 2 * a.len() + 1)
        }),
    });
    if corners.is_none() || nodes.is_none() {
        return Err(IntegrateValidationError::ResourceExhausted {
            message: "cubature rule or subdivision cardinality exceeds usize".to_string(),
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
    if options.rule == CubatureRule::GenzMalik && a.len() < 2 {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: "Genz-Malik cubature is only defined for ndim >= 2".to_string(),
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
) -> Result<Vec<(Vec<f64>, Vec<f64>)>, IntegrateValidationError> {
    let mut breakpoints = ranges
        .iter()
        .map(|&(lower, upper)| vec![lower, upper])
        .collect::<Vec<_>>();

    for point in &options.points {
        for (((axis_breakpoints, &coordinate), &transform), &(start, end)) in breakpoints
            .iter_mut()
            .zip(point.iter())
            .zip(transforms.iter())
            .zip(ranges)
        {
            let Some(mapped) = transform.inverse(coordinate) else {
                continue;
            };
            // Appended points are not domain endpoints. Always test against
            // the original range, independently of the caller's point order.
            if is_strictly_between(mapped, start, end) {
                axis_breakpoints.push(mapped);
            }
        }
    }

    for (axis_breakpoints, &(start, end)) in breakpoints.iter_mut().zip(ranges) {
        let ascending = start <= end;
        axis_breakpoints.sort_by(|left, right| {
            if ascending {
                left.total_cmp(right)
            } else {
                right.total_cmp(left)
            }
        });
        // Every distinct representable boundary carries an interval, however
        // small. Tolerance-based deduplication can erase the entire integral.
        axis_breakpoints.dedup_by(|left, right| *left == *right);
    }

    let region_count = breakpoints.iter().try_fold(1_usize, |count, points| {
        count.checked_mul(points.len().saturating_sub(1))
    });
    let Some(region_count) = region_count else {
        return Err(IntegrateValidationError::ResourceExhausted {
            message: "cubature initial partition cardinality exceeds usize".to_string(),
        });
    };
    let mut regions = Vec::new();
    regions.try_reserve(region_count).map_err(|_| {
        IntegrateValidationError::ResourceExhausted {
            message: "cubature initial partition exceeds available memory".to_string(),
        }
    })?;
    let mut lower = vec![0.0; ranges.len()];
    let mut upper = vec![0.0; ranges.len()];
    collect_initial_cubature_regions(&breakpoints, 0, &mut lower, &mut upper, &mut regions);
    Ok(regions)
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

/// A 1-D Kronrod rule on [-1, 1] with its embedded Gauss rule: `gauss[i]` is the Gauss weight
/// of `nodes[i]`, 0 where the node is a Kronrod-only node. The tables are SciPy's
/// `GaussKronrodQuadrature(21 | 15)` (QUADPACK `qk21`/`qk15`) rounded to f64.
struct NestedRule1d {
    nodes: &'static [f64],
    kronrod: &'static [f64],
    gauss: &'static [f64],
}

const GK21_RULE: NestedRule1d = NestedRule1d {
    nodes: &[
        0.995_657_163_025_808_1,
        0.973_906_528_517_171_7,
        0.930_157_491_355_708_2,
        0.865_063_366_688_984_5,
        0.780_817_726_586_416_9,
        0.679_409_568_299_024_4,
        0.562_757_134_668_604_7,
        0.433_395_394_129_247_2,
        0.294_392_862_701_460_2,
        0.148_874_338_981_631_22,
        0.0,
        -0.148_874_338_981_631_22,
        -0.294_392_862_701_460_2,
        -0.433_395_394_129_247_2,
        -0.562_757_134_668_604_7,
        -0.679_409_568_299_024_4,
        -0.780_817_726_586_416_9,
        -0.865_063_366_688_984_5,
        -0.930_157_491_355_708_2,
        -0.973_906_528_517_171_7,
        -0.995_657_163_025_808_1,
    ],
    kronrod: &[
        0.011_694_638_867_371_874,
        0.032_558_162_307_964_725,
        0.054_755_896_574_351_995,
        0.075_039_674_810_919_96,
        0.093_125_454_583_697_6,
        0.109_387_158_802_297_64,
        0.123_491_976_262_065_84,
        0.134_709_217_311_473_34,
        0.142_775_938_577_060_09,
        0.147_739_104_901_338_49,
        0.149_445_554_002_916_9,
        0.147_739_104_901_338_49,
        0.142_775_938_577_060_09,
        0.134_709_217_311_473_34,
        0.123_491_976_262_065_84,
        0.109_387_158_802_297_64,
        0.093_125_454_583_697_6,
        0.075_039_674_810_919_96,
        0.054_755_896_574_351_995,
        0.032_558_162_307_964_725,
        0.011_694_638_867_371_874,
    ],
    gauss: &[
        0.0,
        0.066_671_344_308_688_14,
        0.0,
        0.149_451_349_150_580_6,
        0.0,
        0.219_086_362_515_982_04,
        0.0,
        0.269_266_719_309_996_35,
        0.0,
        0.295_524_224_714_752_87,
        0.0,
        0.295_524_224_714_752_87,
        0.0,
        0.269_266_719_309_996_35,
        0.0,
        0.219_086_362_515_982_04,
        0.0,
        0.149_451_349_150_580_6,
        0.0,
        0.066_671_344_308_688_14,
        0.0,
    ],
};

const GK15_RULE: NestedRule1d = NestedRule1d {
    nodes: &[
        0.991_455_371_120_812_6,
        0.949_107_912_342_758_5,
        0.864_864_423_359_769_1,
        0.741_531_185_599_394_5,
        0.586_087_235_467_691_1,
        0.405_845_151_377_397_2,
        0.207_784_955_007_898_48,
        0.0,
        -0.207_784_955_007_898_48,
        -0.405_845_151_377_397_2,
        -0.586_087_235_467_691_1,
        -0.741_531_185_599_394_5,
        -0.864_864_423_359_769_1,
        -0.949_107_912_342_758_5,
        -0.991_455_371_120_812_6,
    ],
    kronrod: &[
        0.022_935_322_010_529_224,
        0.063_092_092_629_978_56,
        0.104_790_010_322_250_19,
        0.140_653_259_715_525_92,
        0.169_004_726_639_267_9,
        0.190_350_578_064_785_42,
        0.204_432_940_075_298_89,
        0.209_482_141_084_727_82,
        0.204_432_940_075_298_89,
        0.190_350_578_064_785_42,
        0.169_004_726_639_267_9,
        0.140_653_259_715_525_92,
        0.104_790_010_322_250_19,
        0.063_092_092_629_978_56,
        0.022_935_322_010_529_224,
    ],
    gauss: &[
        0.0,
        0.129_484_966_168_869_7,
        0.0,
        0.279_705_391_489_276_64,
        0.0,
        0.381_830_050_505_118_9,
        0.0,
        0.417_959_183_673_469_4,
        0.0,
        0.381_830_050_505_118_9,
        0.0,
        0.279_705_391_489_276_64,
        0.0,
        0.129_484_966_168_869_7,
        0.0,
    ],
};

/// Calls `visit(node, higher_weight, lower_weight)` for every node of the tensor-product rule
/// on [-1, 1]^ndim (SciPy's `ProductNestedFixed`); the lower weight is the embedded Gauss
/// product, 0 at every node with a Kronrod-only coordinate.
fn for_each_product_node(
    rule: &NestedRule1d,
    ndim: usize,
    mut visit: impl FnMut(&[f64], f64, f64) -> Result<(), IntegrateValidationError>,
) -> Result<(), IntegrateValidationError> {
    let mut index = vec![0_usize; ndim];
    let mut node = vec![0.0; ndim];
    loop {
        let (mut higher, mut lower) = (1.0, 1.0);
        for (coord, &i) in node.iter_mut().zip(index.iter()) {
            *coord = rule.nodes[i];
            higher *= rule.kronrod[i];
            lower *= rule.gauss[i];
        }
        visit(&node, higher, lower)?;
        let mut dim = ndim;
        loop {
            if dim == 0 {
                return Ok(());
            }
            dim -= 1;
            index[dim] += 1;
            if index[dim] < rule.nodes.len() {
                break;
            }
            index[dim] = 0;
        }
    }
}

/// Calls `visit(node, higher_weight, lower_weight)` for every node of the degree-7 Genz–Malik
/// rule on [-1, 1]^ndim with its embedded degree-5 rule (SciPy's `GenzMalikCubature`; the
/// 2^ndim corner nodes carry no degree-5 weight).
fn for_each_genz_malik_node(
    ndim: usize,
    mut visit: impl FnMut(&[f64], f64, f64) -> Result<(), IntegrateValidationError>,
) -> Result<(), IntegrateValidationError> {
    let d = ndim as f64;
    let scale = 2.0_f64.powi(ndim as i32);
    let (l2, l3, l4, l5) = (
        (9.0_f64 / 70.0).sqrt(),
        (9.0_f64 / 10.0).sqrt(),
        (9.0_f64 / 10.0).sqrt(),
        (9.0_f64 / 19.0).sqrt(),
    );
    let w1 = scale * (12824.0 - 9120.0 * d + 400.0 * d * d) / 19683.0;
    let w2 = scale * 980.0 / 6561.0;
    let w3 = scale * (1820.0 - 400.0 * d) / 19683.0;
    let w4 = scale * 200.0 / 19683.0;
    let w5 = 6859.0 / 19683.0;
    let v1 = scale * (729.0 - 950.0 * d + 50.0 * d * d) / 729.0;
    let v2 = scale * 245.0 / 486.0;
    let v3 = scale * (265.0 - 100.0 * d) / 1458.0;
    let v4 = scale * 25.0 / 729.0;

    let mut node = vec![0.0; ndim];
    visit(&node, w1, v1)?;
    for i in 0..ndim {
        for (value, higher, lower) in [(l2, w2, v2), (-l2, w2, v2), (l3, w3, v3), (-l3, w3, v3)] {
            node[i] = value;
            visit(&node, higher, lower)?;
        }
        node[i] = 0.0;
    }
    for i in 0..ndim {
        for j in (i + 1)..ndim {
            for (si, sj) in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
                node[i] = si * l4;
                node[j] = sj * l4;
                visit(&node, w4, v4)?;
            }
            node[i] = 0.0;
            node[j] = 0.0;
        }
    }
    for corner in 0..(1_usize << ndim) {
        for (k, coord) in node.iter_mut().enumerate() {
            *coord = if (corner >> k) & 1 == 0 { l5 } else { -l5 };
        }
        visit(&node, w5, 0.0)?;
    }
    Ok(())
}

/// A finite value represented without restricting intermediate products to
/// f64's exponent range. Cubature weights include both very small and very
/// large axis lengths; multiplying those lengths first loses valid integrals.
#[derive(Clone, Copy)]
struct CubatureProduct {
    mantissa: f64,
    exponent: i32,
}

impl CubatureProduct {
    fn one() -> Self {
        Self {
            mantissa: 1.0,
            exponent: 0,
        }
    }

    fn multiply(&mut self, value: f64) {
        if value == 0.0 || self.mantissa == 0.0 {
            self.mantissa *= value;
            self.exponent = 0;
            return;
        }
        let (mantissa, exponent) = cubature_frexp(value);
        self.mantissa *= mantissa;
        self.exponent += exponent;
        if self.mantissa.abs() >= 2.0 {
            self.mantissa *= 0.5;
            self.exponent += 1;
        }
    }
}

/// Decompose a finite, nonzero f64 as mantissa * 2^exponent with
/// 1 <= abs(mantissa) < 2, including subnormal input.
fn cubature_frexp(value: f64) -> (f64, i32) {
    let (value, adjustment) = if value.abs() < f64::MIN_POSITIVE {
        (value * 18_014_398_509_481_984.0, -54)
    } else {
        (value, 0)
    };
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32 - 1023 + adjustment;
    let mantissa = f64::from_bits((bits & 0x800f_ffff_ffff_ffff) | (1023_u64 << 52));
    (mantissa, exponent)
}

/// Exact nonpositive power of two, or zero when it is unrepresentable.
fn cubature_downscale(exponent: i32) -> f64 {
    if exponent < -1074 {
        0.0
    } else if exponent < -1022 {
        f64::from_bits(1_u64 << (exponent + 1074))
    } else {
        f64::from_bits(((exponent + 1023) as u64) << 52)
    }
}

/// Neumaier summation with a shared exponent. Converting each weighted term
/// first would erase a subnormal integral even when the final sum is nonzero.
#[derive(Default)]
struct CubatureSum {
    value: f64,
    correction: f64,
    exponent: i32,
}

impl CubatureSum {
    fn add(&mut self, term: CubatureProduct) {
        if term.mantissa == 0.0 {
            return;
        }
        if self.value == 0.0 && self.correction == 0.0 {
            self.value = term.mantissa;
            self.exponent = term.exponent;
            return;
        }
        if term.exponent > self.exponent {
            let factor = cubature_downscale(self.exponent - term.exponent);
            self.value *= factor;
            self.correction *= factor;
            self.exponent = term.exponent;
        }
        let value = term.mantissa * cubature_downscale(term.exponent - self.exponent);
        let next = self.value + value;
        self.correction += if self.value.abs() >= value.abs() {
            (self.value - next) + value
        } else {
            (value - next) + self.value
        };
        self.value = next;
    }

    fn total(&self) -> f64 {
        let value = self.value + self.correction;
        if value == 0.0 {
            return value;
        }
        let (mantissa, adjustment) = cubature_frexp(value);
        let exponent = self.exponent + adjustment;
        if exponent > 1023 {
            f64::INFINITY.copysign(mantissa)
        } else if exponent < -1075 {
            0.0_f64.copysign(mantissa)
        } else if exponent < -1022 {
            // Do the final rounding to subnormal precision only once.
            let factor = f64::from_bits(((exponent + 1074 + 1023) as u64) << 52);
            (mantissa * factor) * f64::from_bits(1)
        } else {
            mantissa * f64::from_bits(((exponent + 1023) as u64) << 52)
        }
    }
}

/// One region's estimate under `rule`: the higher-order rule's value and, as its error, the
/// absolute difference from the embedded lower-order rule (SciPy `NestedFixedRule`). Nodes
/// map from [-1, 1]^ndim onto the region, then through the infinite-limit transforms.
fn estimate_cubature_region<F>(
    f: &F,
    transforms: &[BoundTransform],
    a: &[f64],
    b: &[f64],
    output_dim: usize,
    rule: CubatureRule,
) -> Result<CubatureEstimate, IntegrateValidationError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let ndim = a.len();
    let center: Vec<f64> = a.iter().zip(b).map(|(&lo, &hi)| lo.midpoint(hi)).collect();
    let mut half = Vec::with_capacity(ndim);
    let mut volume = CubatureProduct::one();
    for (&lo, &hi) in a.iter().zip(b) {
        let width = hi - lo;
        if width.is_finite() {
            half.push(0.5 * width);
            // Retain an exact factor of two even if width is subnormal.
            volume.multiply(width);
            volume.exponent -= 1;
        } else {
            let half_width = 0.5 * hi - 0.5 * lo;
            half.push(half_width);
            volume.multiply(half_width);
        }
    }
    let mut higher: Vec<CubatureSum> = (0..output_dim).map(|_| CubatureSum::default()).collect();
    let mut lower: Vec<CubatureSum> = (0..output_dim).map(|_| CubatureSum::default()).collect();
    let mut x = vec![0.0; ndim];
    let mut neval = 0_usize;

    let visit =
        |node: &[f64], w_higher: f64, w_lower: f64| -> Result<(), IntegrateValidationError> {
            let mut weight = volume;
            for k in 0..ndim {
                // Interior rules must not round onto an excluded endpoint.
                let left = a[k].min(b[k]).next_up();
                let right = a[k].max(b[k]).next_down();
                let t = (center[k] + half[k] * node[k]).clamp(left, right);
                let (mapped, derivative) = transforms[k].map(t);
                if !mapped.is_finite() || !derivative.is_finite() {
                    return Err(IntegrateValidationError::QuadInvalidBounds {
                        detail: "cubature transformed node or weight is not representable"
                            .to_string(),
                    });
                }
                x[k] = mapped;
                weight.multiply(derivative);
            }
            let values = f(&x);
            validate_cubature_output(&values, output_dim)?;
            neval += 1;
            let mut higher_weight = weight;
            higher_weight.multiply(w_higher);
            let mut lower_weight = weight;
            lower_weight.multiply(w_lower);
            for ((h, l), &value) in higher.iter_mut().zip(lower.iter_mut()).zip(&values) {
                let mut high_term = higher_weight;
                high_term.multiply(value);
                h.add(high_term);
                let mut low_term = lower_weight;
                low_term.multiply(value);
                l.add(low_term);
            }
            Ok(())
        };
    match rule {
        CubatureRule::GaussKronrod | CubatureRule::Gk21 => {
            for_each_product_node(&GK21_RULE, ndim, visit)?;
        }
        CubatureRule::Gk15 => for_each_product_node(&GK15_RULE, ndim, visit)?,
        CubatureRule::GenzMalik => for_each_genz_malik_node(ndim, visit)?,
    }

    let estimate: Vec<f64> = higher.iter().map(CubatureSum::total).collect();
    let representable = higher.iter().chain(&lower).all(|sum| {
        let total = sum.total();
        total.is_finite() && (total != 0.0 || sum.value + sum.correction == 0.0)
    });
    let mut error: Vec<f64> = estimate
        .iter()
        .zip(&lower)
        .map(|(&high, low)| {
            let low = low.total();
            if high.is_finite() && low.is_finite() {
                (high - low).abs()
            } else {
                f64::INFINITY
            }
        })
        .collect();
    if !representable {
        for value in &mut error {
            if *value == 0.0 {
                *value = f64::from_bits(1);
            }
        }
    }

    Ok(CubatureEstimate {
        region: CubatureRegion {
            a: a.to_vec(),
            b: b.to_vec(),
            estimate,
            error,
        },
        neval,
        representable,
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

fn componentwise_cubature_converged(estimate: &[f64], error: &[f64], atol: f64, rtol: f64) -> bool {
    estimate
        .iter()
        .zip(error.iter())
        .all(|(estimate_value, error_value)| {
            estimate_value.is_finite()
                && error_value.is_finite()
                && *error_value >= 0.0
                && *error_value <= atol + rtol * estimate_value.abs()
        })
}

fn sum_cubature_regions(regions: &[CubatureRegion], output_dim: usize) -> (Vec<f64>, Vec<f64>) {
    let total = |component: usize, errors: bool| {
        let values = regions.iter().map(|region| {
            if errors {
                region.error[component]
            } else {
                region.estimate[component]
            }
        });
        if values.clone().any(|value| !value.is_finite()) {
            return values.sum();
        }
        let mut sum = CubatureSum::default();
        for value in values {
            let mut term = CubatureProduct::one();
            term.multiply(value);
            sum.add(term);
        }
        sum.total()
    };
    let estimate = (0..output_dim).map(|i| total(i, false)).collect();
    let error = (0..output_dim).map(|i| total(i, true)).collect();
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

/// Compute n-point Gauss-Legendre nodes and weights on [-1, 1].
///
/// Uses Newton's method to find roots of the Legendre polynomial P_n(x),
/// then computes weights via the Christoffel-Darboux formula.
/// Cache of Gauss-Legendre nodes/weights keyed by order `n`. The Newton-method
/// node solve is O(n²·iterations); `scipy.special.roots_legendre` is lru_cached for
/// the same reason — repeated `fixed_quad` / `gauss_legendre` calls with one order
/// then cost an O(n) clone instead of a full recompute.
/// `(nodes, weights)` for one Gauss-Legendre order.
type GaussLegendreNodesWeights = (Vec<f64>, Vec<f64>);

/// The order-keyed node/weight cache behind [`gauss_legendre_node_cache`].
///
/// Named rather than spelled inline so the `static` and the accessor share one
/// definition; writing it out twice tripped `clippy::type_complexity`
/// (frankenscipy-3qjah). Purely a naming change -- same type, same layout.
type GaussLegendreCache =
    std::sync::RwLock<std::collections::HashMap<usize, GaussLegendreNodesWeights>>;

static GAUSS_LEGENDRE_CACHE: std::sync::OnceLock<GaussLegendreCache> = std::sync::OnceLock::new();

fn gauss_legendre_node_cache() -> &'static GaussLegendreCache {
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
    if h0 == 0.0 || h1 == 0.0 {
        0.5 * h0 * (y0 + y1)
    } else {
        let scale = h0.abs().max(h1.abs());
        let u = h0 / scale;
        let v = h1 / scale;
        let sum = if h0.is_sign_negative() == h1.is_sign_negative() {
            u + v
        } else {
            (h0 + h1) / scale
        };
        let fraction = u / sum;
        let ratio = (u / v) * fraction;
        sampled_simpson_weighted_sum(
            h0,
            [
                (3.0 - fraction) / 6.0,
                (3.0 + ratio + fraction) / 6.0,
                -ratio / 6.0,
            ],
            [y0, y1, y2],
        )
    }
}

fn cumulative_simpson_right_interval(y0: f64, y1: f64, y2: f64, h0: f64, h1: f64) -> f64 {
    if h0 == 0.0 || h1 == 0.0 {
        0.5 * h1 * (y1 + y2)
    } else {
        let scale = h0.abs().max(h1.abs());
        let u = h0 / scale;
        let v = h1 / scale;
        let sum = if h0.is_sign_negative() == h1.is_sign_negative() {
            u + v
        } else {
            (h0 + h1) / scale
        };
        let fraction = v / sum;
        let ratio = (v / u) * fraction;
        sampled_simpson_weighted_sum(
            h1,
            [
                -ratio / 6.0,
                (3.0 + ratio + fraction) / 6.0,
                (3.0 - fraction) / 6.0,
            ],
            [y0, y1, y2],
        )
    }
}

fn sampled_simpson_weighted_sum(width: f64, weights: [f64; 3], y: [f64; 3]) -> f64 {
    // Weight each ordinate before summing. A large ordinate and a tiny
    // interval can have a finite integral even if an unweighted sum does not.
    de_weighted_term(width, weights[0], y[0])
        + de_weighted_term(width, weights[1], y[1])
        + de_weighted_term(width, weights[2], y[2])
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
                // status: |R[n][n] - R[n-1][n-1]| < tol (checked just above)
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

/// Two distinct samples closest to a finite endpoint. Infer local power-law
/// decay from actual endpoint distances, not from transformed t coordinates:
/// rounding of x can make the last transformed terms look artificially small.
#[derive(Default)]
struct DeEndpointTail {
    nearest: Option<(f64, f64)>,
    next: Option<(f64, f64)>,
}

impl DeEndpointTail {
    fn record(&mut self, distance: f64, magnitude: f64) {
        if !distance.is_finite() || distance <= 0.0 {
            return;
        }
        match self.nearest {
            None => self.nearest = Some((distance, magnitude)),
            Some((nearest, _)) if distance < nearest => {
                self.next = self.nearest;
                self.nearest = Some((distance, magnitude));
            }
            Some((nearest, _))
                if distance > nearest && self.next.is_none_or(|(next, _)| distance < next) =>
            {
                self.next = Some((distance, magnitude));
            }
            _ => {}
        }
    }

    fn error(&self) -> f64 {
        let (Some((distance, value)), Some((other_distance, other_value))) =
            (self.nearest, self.next)
        else {
            return 0.0;
        };
        if value == 0.0 {
            return 0.0;
        }
        if other_value == 0.0 {
            return f64::INFINITY;
        }
        // For f(x) ~ C*d^p, the unresolved integral is f(x)*d/(p+1).
        // This component matters when p approaches -1: the outer transformed
        // term alone need not bound even a slowly convergent elementary tail.
        let power = (value.ln() - other_value.ln()) / (distance.ln() - other_distance.ln());
        let integrability = 1.0 + power;
        if !integrability.is_finite() || integrability <= 0.0 {
            return f64::INFINITY;
        }
        de_weighted_term(1.0 / integrability, distance, value)
    }
}

/// The two outermost transformed samples on one side of the integration
/// interval. A local exponential tail estimate complements the finite-endpoint
/// power estimate: logarithmic singularities can decay arbitrarily slowly in
/// transformed coordinates even though their local power appears integrable.
#[derive(Default)]
struct DeTransformedTail {
    outer: Option<(f64, f64)>,
    next: Option<(f64, f64)>,
}

impl DeTransformedTail {
    fn record(&mut self, distance: f64, magnitude: f64) {
        match self.outer {
            None => self.outer = Some((distance, magnitude)),
            Some((outer, _)) if distance > outer => {
                self.next = self.outer;
                self.outer = Some((distance, magnitude));
            }
            Some((outer, _))
                if distance < outer && self.next.is_none_or(|(next, _)| distance > next) =>
            {
                self.next = Some((distance, magnitude));
            }
            _ => {}
        }
    }

    fn error(&self) -> f64 {
        let (Some((distance, value)), Some((other_distance, other_value))) =
            (self.outer, self.next)
        else {
            return 0.0;
        };
        if value == 0.0 {
            return 0.0;
        }
        if other_value == 0.0 {
            return f64::INFINITY;
        }
        let decay = (other_value.ln() - value.ln()) / (distance - other_distance);
        if !decay.is_finite() || decay <= 0.0 {
            return f64::INFINITY;
        }
        // For g(t) ~ C*exp(-decay*|t|), the unresolved transformed integral
        // is g(t)/decay. Keep the physical endpoint estimate as well: rounded
        // abscissae can make this observed decay artificially steep.
        value / decay
    }
}

/// A compensated sum of the *weighted* quadrature terms. Keeping the step
/// inside each term avoids an intermediate sum that grows by 2 at every level
/// even when the integral itself is representable.
struct DeSum {
    value: f64,
    correction: f64,
    absolute: f64,
    outer_t: [f64; 2],
    outer_terms: [f64; 2],
    endpoints: [DeEndpointTail; 2],
    transformed_tails: [DeTransformedTail; 2],
}

impl Default for DeSum {
    fn default() -> Self {
        Self {
            value: 0.0,
            correction: 0.0,
            absolute: 0.0,
            outer_t: [f64::INFINITY, f64::NEG_INFINITY],
            outer_terms: [0.0; 2],
            endpoints: std::array::from_fn(|_| DeEndpointTail::default()),
            transformed_tails: std::array::from_fn(|_| DeTransformedTail::default()),
        }
    }
}

impl DeSum {
    fn add(&mut self, term: f64) {
        let next = self.value + term;
        if self.value.abs() >= term.abs() {
            self.correction += (self.value - next) + term;
        } else {
            self.correction += (term - next) + self.value;
        }
        self.value = next;
        self.absolute += term.abs();
    }

    fn halve(&mut self) {
        self.value *= 0.5;
        self.correction *= 0.5;
        self.absolute *= 0.5;
    }

    fn total(&self) -> f64 {
        self.value + self.correction
    }
}

/// Multiply a positive step and Jacobian by a finite integrand value without
/// losing a representable contribution to an underflowed intermediate weight.
fn de_weighted_term(h: f64, jac: f64, fx: f64) -> f64 {
    let weight = h * jac;
    let term = weight * fx;
    if weight.is_normal() && term.is_finite() {
        return term;
    }
    let mut factors = [h, jac, fx];
    factors.sort_by(|a, b| a.abs().total_cmp(&b.abs()));
    (factors[0] * factors[2]) * factors[1]
}

/// Generic double-exponential quadrature driver.
///
/// `node(t)` maps a quadrature parameter `t ∈ ℝ` to `Some((x, jac))` — the
/// abscissa `x` and the full weight `jac = (dx/dt)·W(t)` — or `None` when the
/// node has run off the (possibly infinite) interval and should be skipped.
/// `endpoints` names finite endpoints in transformed-tail order (`t → -∞`,
/// `t → +∞`), not physical left-to-right order. In particular, mirrored
/// exp-sinh approaches its finite upper endpoint as `t → -∞`.
/// Integral ≈ `h · Σ_j jac_j · f(x_j)`. Abscissae are reused across levels
/// (halving the step only adds odd indices). The inter-level error estimate is
/// bounded below by roundoff and the outermost retained terms, so a stable
/// truncated sum does not certify an unresolved endpoint tail.
fn de_quadrature<F, N>(
    f: &F,
    node: &N,
    endpoints: [Option<f64>; 2],
    atol: f64,
    rtol: f64,
    max_level: usize,
) -> QuadResult
where
    F: Fn(f64) -> f64,
    N: Fn(f64) -> Option<(f64, f64)>,
{
    // Beyond |t| ≈ 7 the double-exponential weight has either underflowed to
    // zero or driven the abscissa off the interval, so summation can stop.
    let t_cap = 7.0;
    let mut neval = 0usize;
    let eval = |t: f64, h: f64, sum: &mut DeSum, neval: &mut usize| {
        // Geometry may omit rounded endpoint nodes, but a non-finite value at
        // an interior node invalidates the integral. Do not turn it into zero.
        if !sum.total().is_finite() {
            return;
        }
        if let Some((x, jac)) = node(t)
            && jac != 0.0
        {
            if !x.is_finite() || !jac.is_finite() {
                sum.value = f64::NAN;
                return;
            }
            let fx = f(x);
            *neval += 1;
            if !fx.is_finite() {
                sum.value = f64::NAN;
                return;
            }
            sum.add(de_weighted_term(h, jac, fx));
            for (endpoint, tail) in endpoints.iter().zip(&mut sum.endpoints) {
                if let Some(endpoint) = endpoint {
                    tail.record((x - endpoint).abs(), fx.abs());
                }
            }
            // Do not multiply this tail indicator by h: refining the same
            // representability-limited endpoint must not make its omitted
            // contribution look smaller. This is the outer-term component
            // of double-exponential error estimation, independent of the
            // difference between two successive truncated sums.
            let side = usize::from(t >= 0.0);
            let outer_term = de_weighted_term(1.0, jac, fx).abs();
            sum.transformed_tails[side].record(t.abs(), outer_term);
            if (side == 0 && t < sum.outer_t[0]) || (side == 1 && t > sum.outer_t[1]) {
                sum.outer_t[side] = t;
                sum.outer_terms[side] = outer_term;
            }
        }
    };

    // Level 0: every integer index at the coarse step h0 = 1.
    let mut h = 1.0_f64;
    let mut sum = DeSum::default();
    eval(0.0, h, &mut sum, &mut neval);
    let mut j = 1;
    while (j as f64) * h <= t_cap {
        let t = j as f64 * h;
        eval(t, h, &mut sum, &mut neval);
        eval(-t, h, &mut sum, &mut neval);
        j += 1;
    }
    if !sum.total().is_finite() || neval == 0 {
        return QuadResult {
            integral: f64::NAN,
            error: f64::INFINITY,
            neval,
            converged: false,
        };
    }
    let mut prev = sum.total();
    let mut prev_delta = f64::INFINITY;
    let mut last_err = f64::INFINITY;

    for level in 1..=max_level {
        h *= 0.5;
        sum.halve();
        let mut j = 1;
        while (j as f64) * h <= t_cap {
            let t = j as f64 * h;
            eval(t, h, &mut sum, &mut neval);
            eval(-t, h, &mut sum, &mut neval);
            j += 2;
        }
        if !sum.total().is_finite() {
            return QuadResult {
                integral: f64::NAN,
                error: f64::INFINITY,
                neval,
                converged: false,
            };
        }
        let cur = sum.total();
        let delta = (cur - prev).abs();
        last_err = if prev_delta > 0.0 && prev_delta.is_finite() {
            // Form the ratio before multiplying: delta squared can underflow
            // to zero while the error is still larger than a relative tolerance.
            (delta * (delta / prev_delta)).min(delta)
        } else {
            delta
        };
        let roundoff =
            (f64::EPSILON * sum.absolute).max(if cur == 0.0 { 0.0 } else { f64::from_bits(1) });
        // The three indicators on one side estimate the same omitted tail,
        // so keep their maximum. The two sides are disjoint contributions:
        // their absolute errors add, even when the retained sums have stopped
        // changing. Taking one maximum across both can certify half of the
        // actual missing mass for two same-sign endpoint singularities.
        let tail_error: f64 = (0..2)
            .map(|side| {
                sum.outer_terms[side]
                    .max(sum.endpoints[side].error())
                    .max(sum.transformed_tails[side].error())
            })
            .sum();
        last_err = last_err.max(roundoff).max(tail_error);
        if level >= 2 && last_err.is_finite() && last_err <= atol + rtol * cur.abs() {
            return QuadResult {
                integral: cur,
                error: last_err,
                neval,
                // status: inter-level, roundoff and endpoint-tail estimates all meet tolerance
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
/// 16). Returns the estimate, an error estimate accounting for successive
/// refinement, floating-point roundoff, and the outermost retained terms, the
/// function-evaluation count, and whether the tolerance was met.
/// Non-finite interior values or accumulated quadrature sums return an
/// unconverged result with a non-finite integral and infinite error. Rounded
/// endpoint nodes are omitted so integrable endpoint singularities remain supported.
/// If an endpoint is too coarsely represented to meet the requested tolerance,
/// the result remains unconverged even if the truncated sums stop changing.
/// Error estimates use observed local decay and are practical diagnostics,
/// not rigorous bounds for an arbitrary callable.
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
            // status: a == b exactly, the integral is 0
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
        if lo.next_up() >= hi {
            return QuadResult {
                integral: f64::NAN,
                error: f64::INFINITY,
                neval: 0,
                converged: false,
            };
        }
        let c = lo.midpoint(hi);
        let width = hi - lo;
        let d = if width.is_finite() {
            0.5 * width
        } else {
            0.5 * hi - 0.5 * lo
        };
        let node = |t: f64| -> Option<(f64, f64)> {
            let u = half_pi * t.sinh();
            let cu = u.cosh();
            // Form the dimensionless weight first: d * half_pi * cosh(t)
            // can overflow even when the final weight is representable.
            let w = d * (half_pi * t.cosh() / (cu * cu));
            if w == 0.0 {
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
        de_quadrature(&f, &node, [Some(lo), Some(hi)], atol, rtol, max_level)
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
            if !x.is_finite() || !jac.is_finite() || x <= lo {
                return None;
            }
            Some((x, jac))
        };
        de_quadrature(&f, &node, [Some(lo), None], atol, rtol, max_level)
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
            if !x.is_finite() || !jac.is_finite() || x >= hi {
                return None;
            }
            Some((x, jac))
        };
        de_quadrature(&f, &node, [Some(hi), None], atol, rtol, max_level)
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
        de_quadrature(&f, &node, [None, None], atol, rtol, max_level)
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
/// finite range is summed directly. A left-infinite range includes the finite
/// upper endpoint and steps down from it. A bilateral range uses integer
/// multiples of `step` on either side of zero. Each infinite tail uses the
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
        || b.is_nan()
        || a == f64::INFINITY
        || b == f64::NEG_INFINITY
        || !step.is_finite()
        || step <= 0.0
        || !atol.is_finite()
        || !rtol.is_finite()
    {
        return nan;
    }
    let atol = if atol > 0.0 { atol } else { 0.0 };
    let rtol = if rtol > 0.0 { rtol } else { 1e-11 };

    if a == f64::NEG_INFINITY {
        if b.is_finite() {
            return nsum_right(|x| f(-x), -b, step, atol, rtol);
        }

        // Sum the two tails independently. Pairing f(x) + f(-x) term by
        // term could mask failure of either tail through cancellation.
        let right = nsum_right(&f, 0.0, step, 0.5 * atol, rtol);
        if !right.sum.is_finite() {
            return right;
        }
        let left = nsum_right(|x| f(-x), step, step, 0.5 * atol, rtol);
        let sum = right.sum + left.sum;
        let error = if sum.is_finite() {
            right.error + left.error + sum_residual(right.sum, left.sum, sum).abs()
        } else {
            f64::INFINITY
        };
        return NsumResult {
            sum,
            error,
            nfev: right.nfev + left.nfev,
            converged: right.converged
                && left.converged
                && error.is_finite()
                && error <= atol + rtol * sum.abs(),
        };
    }

    // Finite range: direct summation over n = a, a+step, …, ≤ b.
    if b.is_finite() {
        if b < a {
            return NsumResult {
                sum: 0.0,
                error: 0.0,
                nfev: 0,
                // status: finite b < a is an empty range, the sum is exactly 0
                converged: true,
            };
        }
        let width = b - a;
        let intervals = if width.is_finite() {
            (width / step).floor()
        } else {
            // Opposite-sign endpoints can overflow the span while defining
            // just a handful of representable terms.
            ((0.5 * b - 0.5 * a) / (0.5 * step)).floor()
        };
        // A saturating float-to-integer cast followed by +1 can panic, wrap
        // into an empty loop, or schedule an effectively unbounded loop.
        // Beyond 2^53, f64 cannot distinguish consecutive grid indices either.
        // Refuse an unrepresentable grid before evaluating the caller's code.
        if !intervals.is_finite()
            || intervals >= (1_u64 << 53) as f64
            || intervals >= usize::MAX as f64
            || (intervals >= 1.0 && a + step == a)
        {
            return nan;
        }
        let count = intervals as usize + 1;
        let mut direct = CompensatedSum::default();
        let mut error = 0.0;
        let mut previous = f64::NEG_INFINITY;
        for k in 0..count {
            let index = k as f64;
            let mut x = a + index * step;
            if !x.is_finite() || x > b {
                x = index.mul_add(step, a);
            }
            if !x.is_finite() || x <= previous || x > b {
                return NsumResult { nfev: k, ..nan };
            }
            previous = x;
            error += direct.add(f(x));
            let sum = direct.total();
            if !sum.is_finite() {
                return NsumResult {
                    sum,
                    error: f64::INFINITY,
                    nfev: k + 1,
                    converged: false,
                };
            }
        }
        let sum = direct.total();
        error += sum_residual(direct.value, direct.correction, sum).abs();
        return NsumResult {
            sum,
            error,
            nfev: count,
            converged: error.is_finite() && error <= atol + rtol * sum.abs(),
        };
    }
    nsum_right(f, a, step, atol, rtol)
}

fn nsum_right<F>(f: F, a: f64, s: f64, atol: f64, rtol: f64) -> NsumResult
where
    F: Fn(f64) -> f64,
{
    // Infinite tail via Euler–Maclaurin with a growing direct window.
    // Σ_{k≥m} f(a+ks) ≈ (1/s)∫_{n_m}^∞ f dx + f(n_m)/2
    //                   − (s/12) f'(n_m) + (s³/720) f'''(n_m)
    let mut direct = CompensatedSum::default();
    let mut direct_error = 0.0;
    let mut nfev = 0usize;
    let mut filled = 0i64; // number of leading terms already in `direct`
    let mut prev_est = f64::INFINITY;
    let mut last_err = f64::INFINITY;
    let mut m = 8i64;

    for _iter in 0..24 {
        let n_m = a + m as f64 * s;
        if !n_m.is_finite() || a + s == a {
            return NsumResult {
                sum: f64::NAN,
                error: f64::INFINITY,
                nfev,
                converged: false,
            };
        }
        // Extend the direct partial sum to the first `m` terms.
        while filled < m {
            direct_error += direct.add(f(a + filled as f64 * s));
            nfev += 1;
            filled += 1;
            if !direct.total().is_finite() {
                return NsumResult {
                    sum: direct.total(),
                    error: f64::INFINITY,
                    nfev,
                    converged: false,
                };
            }
        }
        let f_nm = f(n_m);
        nfev += 1;
        // Central finite-difference derivatives at n_m.
        // Keep derivative probes inside the summand's domain even when
        // the first grid point is far from the origin.
        let span = n_m - a;
        let delta = (1e-3 * n_m.abs().min(span)).max(1e-4).min(0.25 * span);
        if delta == 0.0 || !(n_m + 2.0 * delta).is_finite() {
            return NsumResult {
                sum: f64::NAN,
                error: f64::INFINITY,
                nfev,
                converged: false,
            };
        }
        let fp = f(n_m + delta);
        let fm = f(n_m - delta);
        let fp2 = f(n_m + 2.0 * delta);
        let fm2 = f(n_m - 2.0 * delta);
        nfev += 4;
        if [f_nm, fp, fm, fp2, fm2]
            .iter()
            .any(|value| !value.is_finite())
        {
            return NsumResult {
                sum: f64::NAN,
                error: f64::INFINITY,
                nfev,
                converged: false,
            };
        }
        let d1 = (fp - fm) / (2.0 * delta); // f'(n_m)
        let d3 = (fp2 - 2.0 * fp + 2.0 * fm - fm2) / (2.0 * delta.powi(3)); // f'''(n_m)

        let integral_atol = (atol * s).min(f64::MAX);
        // The lattice term f(n_m) has already been sampled and is finite.
        // Integrating in the offset u keeps the exp-sinh weight for tiny u
        // even when n_m + u rounds back to n_m. Otherwise a harmless large
        // translation can force every refinement level at the lower bound.
        let integ = if n_m <= 0.5 * f64::MAX {
            tanhsinh(|u| f(n_m + u), 0.0, f64::INFINITY, integral_atol, rtol, 16)
        } else {
            // Keep the native transform's overflow check for an upper-range
            // origin where adding a positive offset could produce infinity.
            tanhsinh(&f, n_m, f64::INFINITY, integral_atol, rtol, 16)
        };
        nfev += integ.neval;
        if !integ.integral.is_finite() {
            return NsumResult {
                sum: f64::NAN,
                error: f64::INFINITY,
                nfev,
                converged: false,
            };
        }

        let tail = integ.integral / s + 0.5 * f_nm - (s / 12.0) * d1 + (s.powi(3) / 720.0) * d3;
        // Keep the direct sum's compensation when adding the final tail.
        let mut estimate = direct;
        let tail_roundoff = estimate.add(tail);
        let est = estimate.total();
        if !est.is_finite() {
            return NsumResult {
                sum: est,
                error: f64::INFINITY,
                nfev,
                converged: false,
            };
        }

        let roundoff = sum_residual(estimate.value, estimate.correction, est).abs();
        last_err =
            (est - prev_est).abs() + integ.error / s + direct_error + tail_roundoff + roundoff;
        // Agreement between windows cannot remove a quadrature error they
        // share. The integral contributes in series units, divided by step.
        if integ.converged && last_err.is_finite() && last_err <= atol + rtol * est.abs() {
            return NsumResult {
                sum: est,
                error: last_err,
                nfev,
                // status: quadrature converged and the combined error meets tolerance
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

    simpson_nonuniform(y, x)
}

/// Gauss-Kronrod adaptive quadrature (7-15 point rule).
///
/// Refines the subinterval with the largest estimated error until the sum of
/// all subinterval errors meets `max(epsabs, epsrel * abs(integral))`.
/// `limit` bounds the number of active subintervals: at most
/// `15 + 30 * (limit - 1)` integrand evaluations are performed. The error
/// estimate includes a roundoff floor, so cancellation or an exhausted budget
/// cannot certify convergence from the individual subintervals alone.
///
/// Finite reversed bounds negate the integral. Equal bounds return zero
/// without evaluating `f`. Invalid controls, non-finite samples, or intervals
/// with no representable interior node return `converged=false`. For infinite
/// intervals, break points, or QUADPACK extrapolation use [`quad`].
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
        || options.limit == 0
    {
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
            // status: the empty interval has an exactly zero integral.
            converged: true,
        };
    }

    let (lower, upper, sign) = if a < b { (a, b, 1.0) } else { (b, a, -1.0) };
    let mut neval = 0;
    let first = gauss_kronrod_panel(&f, lower, upper, &mut neval);
    let mut integral = CompensatedSum::default();
    let mut error = CompensatedSum::default();
    let mut roundoff = CompensatedSum::default();
    integral.add(first.integral);
    error.add(first.error);
    roundoff.add(first.roundoff);
    let mut panels = std::collections::BinaryHeap::from([first]);

    loop {
        let value = integral.total();
        let summed_error = error.total();
        let roundoff_error = roundoff.total();
        let abs_error = summed_error.max(roundoff_error);
        let tolerance = options.epsabs.max(options.epsrel * value.abs());
        let finite = value.is_finite() && summed_error.is_finite() && roundoff_error.is_finite();
        let converged = finite && abs_error <= tolerance;
        if converged || !finite || panels.len() >= options.limit || abs_error <= roundoff_error {
            return QuadResult {
                integral: sign * value,
                error: if finite { abs_error } else { f64::INFINITY },
                neval,
                converged,
            };
        }

        let worst = panels.pop().expect("at least one quadrature panel");
        let mid = worst.a.midpoint(worst.b);
        if worst.a.next_up() >= mid || mid.next_up() >= worst.b {
            return QuadResult {
                integral: sign * value,
                error: abs_error,
                neval,
                converged: false,
            };
        }
        let left = gauss_kronrod_panel(&f, worst.a, mid, &mut neval);
        let right = gauss_kronrod_panel(&f, mid, worst.b, &mut neval);

        // Keep the children separate from the removed parent in the error
        // sums: a replacement delta can round away small child errors before
        // cancellation restores their scale. Integral replacements also need
        // to avoid overflowing when a large negative panel is removed.
        gauss_kronrod_replace_panel(&mut integral, worst.integral, left.integral, right.integral);
        error.add(-worst.error);
        error.add(left.error);
        error.add(right.error);
        roundoff.add(-worst.roundoff);
        roundoff.add(left.roundoff);
        roundoff.add(right.roundoff);
        panels.push(left);
        panels.push(right);
    }
}

#[derive(Debug)]
struct GaussKronrodPanel {
    a: f64,
    b: f64,
    integral: f64,
    error: f64,
    roundoff: f64,
}

impl PartialEq for GaussKronrodPanel {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for GaussKronrodPanel {}

impl PartialOrd for GaussKronrodPanel {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GaussKronrodPanel {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.error
            .total_cmp(&other.error)
            .then_with(|| other.a.total_cmp(&self.a))
            .then_with(|| other.b.total_cmp(&self.b))
    }
}

#[derive(Clone, Copy, Default)]
struct CompensatedSum {
    value: f64,
    correction: f64,
}

impl CompensatedSum {
    /// Neumaier summation. Return the roundoff lost while accumulating the
    /// correction so callers that report summation error can account for it.
    fn add(&mut self, term: f64) -> f64 {
        let next = self.value + term;
        let loss = if next.is_finite() {
            let residual = sum_residual(self.value, term, next);
            let correction = self.correction + residual;
            let loss = sum_residual(self.correction, residual, correction).abs();
            self.correction = correction;
            loss
        } else {
            self.correction = 0.0;
            f64::INFINITY
        };
        self.value = next;
        loss
    }

    fn total(&self) -> f64 {
        self.value + self.correction
    }
}

// FastTwoSum with the larger operand first. For finite inputs and sum,
// this is the signed roundoff discarded by the floating-point addition.
fn sum_residual(a: f64, b: f64, sum: f64) -> f64 {
    if a.abs() >= b.abs() {
        (a - sum) + b
    } else {
        (b - sum) + a
    }
}

fn gauss_kronrod_replace_panel(sum: &mut CompensatedSum, parent: f64, left: f64, right: f64) {
    // Add the three terms separately so that removing a large coarse-panel
    // estimate cannot erase tiny children. Prefer cancellation at each step:
    // removing a negative parent first can overflow a positive total even
    // when adding its negative children makes the refined total finite.
    let mut terms = [-parent, left, right];
    for i in 0..terms.len() {
        let mut next = i;
        for j in i + 1..terms.len() {
            let candidate_cancels = terms[j].is_sign_negative() != sum.value.is_sign_negative();
            let next_cancels = terms[next].is_sign_negative() != sum.value.is_sign_negative();
            if (candidate_cancels && !next_cancels)
                || (candidate_cancels == next_cancels && terms[j].abs() > terms[next].abs())
            {
                next = j;
            }
        }
        terms.swap(i, next);
        sum.add(terms[i]);
    }
}

fn gauss_kronrod_panel(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    neval: &mut usize,
) -> GaussKronrodPanel {
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

    let failed = || GaussKronrodPanel {
        a,
        b,
        integral: f64::NAN,
        error: f64::INFINITY,
        roundoff: f64::INFINITY,
    };
    if a.next_up() >= b {
        return failed();
    }
    let mid = a.midpoint(b);
    let width = b - a;
    let half = if width.is_finite() {
        0.5 * width
    } else {
        0.5 * b - 0.5 * a
    };

    // Evaluate f at all 15 Kronrod nodes
    let fvals: [f64; 15] = std::array::from_fn(|i| {
        let x = (mid + half * k_nodes[i]).clamp(a.next_up(), b.next_down());
        *neval += 1;
        f(x)
    });
    if fvals.iter().any(|value| !value.is_finite()) {
        return failed();
    }

    // Work in the integrand's local units, then apply the interval width.
    // An unscaled Kronrod sum is twice a constant integrand and can overflow
    // even on [0, 1] when the integral is representable. Normalized sums also
    // keep the embedded-rule difference and deviation estimate in range.
    let scale = fvals.iter().fold(0.0_f64, |s, value| s.max(value.abs()));
    if scale == 0.0 {
        return GaussKronrodPanel {
            a,
            b,
            integral: 0.0,
            error: 0.0,
            roundoff: 0.0,
        };
    }
    let mut kronrod = CompensatedSum::default();
    let mut gauss = CompensatedSum::default();
    let mut absolute = CompensatedSum::default();
    for (&value, &weight) in fvals.iter().zip(&k_weights) {
        let normalized = value / scale;
        kronrod.add(weight * normalized);
        absolute.add(weight * normalized.abs());
    }
    for (&idx, &weight) in g_indices.iter().zip(&g_weights) {
        gauss.add(weight * (fvals[idx] / scale));
    }
    let mean = 0.5 * kronrod.total();
    let mut deviation = CompensatedSum::default();
    for (&value, &weight) in fvals.iter().zip(&k_weights) {
        deviation.add(weight * (value / scale - mean).abs());
    }
    let deviation = deviation.total();
    let mut error = (kronrod.total() - gauss.total()).abs();
    // QUADPACK's resasc correction and roundoff floor; an embedded-rule
    // coincidence alone is not an accuracy certificate under cancellation.
    if deviation > 0.0 && error > 0.0 {
        error = deviation * (200.0 * (error / deviation)).powf(1.5).min(1.0);
    }
    let roundoff = 50.0 * f64::EPSILON * absolute.total();
    error = error.max(roundoff);
    let scaled_bound = |value: f64| {
        let bound = de_weighted_term(half, scale, value);
        if bound == 0.0 && value > 0.0 {
            f64::from_bits(1)
        } else {
            bound
        }
    };

    GaussKronrodPanel {
        a,
        b,
        integral: de_weighted_term(half, scale, kronrod.total()),
        error: scaled_bound(error),
        roundoff: scaled_bound(roundoff),
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
    // `full_output` returns the message rather than warning it.
    let result = quad_full_output(&f, a, b, &[], options).map(|(result, _)| result);
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

/// `scipy.integrate.quad(f, a, np.inf)`: [`quad`] over `[a, ∞)` (QUADPACK QAGIE). This used
/// to map to `t ∈ [0, 1)` and stop at `t = 1 − 1e-10`, losing the tail it cut off.
///
/// # Errors
/// A non-finite `a`, and as [`quad`].
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
    quad(f, a, f64::INFINITY, options)
}

/// `scipy.integrate.quad(f, -np.inf, b)`: [`quad`] over `(-∞, b]` (QUADPACK QAGIE).
///
/// # Errors
/// A non-finite `b`, and as [`quad`].
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
    quad(f, f64::NEG_INFINITY, b, options)
}

/// `scipy.integrate.quad(f, -np.inf, np.inf)`: [`quad`] over the real line (QUADPACK QAGIE
/// with both tails folded onto one transformed interval).
///
/// # Errors
/// As [`quad`].
pub fn quad_full_inf<F>(f: F, options: QuadOptions) -> Result<QuadResult, IntegrateValidationError>
where
    F: Fn(f64) -> f64,
{
    quad(f, f64::NEG_INFINITY, f64::INFINITY, options)
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

/// Compute the cumulative integral using the trapezoidal rule, with SciPy's `initial` argument:
/// one entry per sample, `initial` first.
///
/// Matches `scipy.integrate.cumulative_trapezoid(y, x, initial=initial)`. Since SciPy 1.12 the
/// only accepted value is 0 (`initial=None` is [`cumulative_trapezoid`]); any other value, NaN
/// included, raises `ValueError: initial must be None or 0`, and so does this function. A signed
/// zero is kept as given, as SciPy's `np.full` keeps it. The inputs are validated exactly as
/// [`cumulative_trapezoid`] validates them.
pub fn cumulative_trapezoid_initial(
    y: &[f64],
    x: &[f64],
    initial: f64,
) -> Result<Vec<f64>, IntegrateValidationError> {
    if initial != 0.0 {
        return Err(IntegrateValidationError::QuadInvalidBounds {
            detail: format!("`initial` must be None or 0, got {initial}"),
        });
    }
    let integral = cumulative_trapezoid(y, x)?;
    let mut result = Vec::with_capacity(integral.len() + 1);
    result.push(initial);
    result.extend(integral);
    Ok(result)
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

/// Worker count for a `monte_carlo_integrate` call over `n_samples` points in
/// `d` dimensions.
///
/// Parallel only pays off once the sampling work is large enough to absorb the
/// chunk setup + cross-chunk combine; below that the fused serial loop
/// (byte-identical, zero overhead) wins. Returning `1` selects that serial
/// loop, so this is also the predicate that decides whether a given call is
/// bit-identical to the serial reference or only agrees with it to within
/// cross-chunk reassociation.
///
/// Named rather than left inline so the gate itself is testable — the numeric
/// tests below would otherwise pass without ever reaching the parallel
/// reduction, which is exactly how this lever went uncovered
/// (frankenscipy-6x365).
fn monte_carlo_thread_count(n_samples: usize, d: usize) -> usize {
    let work = n_samples.saturating_mul(d);
    if work < (1 << 18) || n_samples < 2 {
        1
    } else {
        std::thread::available_parallelism()
            .map(|c| c.get())
            .unwrap_or(1)
            .min(n_samples)
    }
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

    let nthreads = monte_carlo_thread_count(n_samples, d);

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

/// Randomized QMC integration, retained here for compatibility with the quadrature module.
pub use crate::qmc::{QmcQuadOptions, QmcQuadResult, qmc_quad, qmc_quad_with_options};

#[cfg(test)]
mod tests {
    use super::*;

    /// One quadrature case: `(integrand, a, b, exact)`. Named to keep
    /// `clippy::type_complexity` quiet without reshaping the tuple
    /// (frankenscipy-3qjah); same type, same order.
    type QuadCase = (fn(f64) -> f64, f64, f64, f64);

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

    /// Serial reference for [`monte_carlo_integrate`]: the exact LCG stream,
    /// sample points, and left-to-right summation order the pre-parallel
    /// implementation used. Written out here rather than reusing any part of
    /// the production path so the comparison below is an independent oracle.
    fn monte_carlo_serial_reference(
        f: impl Fn(&[f64]) -> f64,
        bounds: &[(f64, f64)],
        n_samples: usize,
        seed: u64,
    ) -> (f64, f64) {
        let d = bounds.len();
        let mut volume = 1.0;
        for &(a, b) in bounds {
            volume *= b - a;
        }
        let mut rng = seed;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut point = vec![0.0; d];
        for _ in 0..n_samples {
            for (j, p) in point.iter_mut().enumerate() {
                rng = rng.wrapping_mul(MC_LCG_A).wrapping_add(1);
                let u = (rng >> 11) as f64 / (1u64 << 53) as f64;
                *p = bounds[j].0 + u * (bounds[j].1 - bounds[j].0);
            }
            let fval = f(&point);
            sum += fval;
            sum_sq += fval * fval;
        }
        let nf = n_samples as f64;
        let mean = sum / nf;
        let variance = sum_sq / nf - mean * mean;
        ((mean * volume), (variance / nf).sqrt() * volume)
    }

    /// Two-arm control for the parallel-reduction gate: one shape that MUST
    /// take the serial loop and one that MUST take the parallel reduction on
    /// any multi-core host. Without this, the numeric test below would report a
    /// clean pass while never once reaching the parallel path.
    #[test]
    fn monte_carlo_thread_count_gates_on_sampling_work() {
        // Must-miss arm: work = 4096*1 and 2*64 are both under the 1<<18 gate,
        // and a single sample can never be split.
        assert_eq!(monte_carlo_thread_count(4096, 1), 1, "work below gate");
        assert_eq!(monte_carlo_thread_count(2, 64), 1, "work below gate");
        assert_eq!(monte_carlo_thread_count(1, 1 << 20), 1, "single sample");

        // Must-hit arm: work = 262_144*2 clears the gate, so the only thing
        // that can hold this at 1 is a genuinely single-core host.
        let cores = std::thread::available_parallelism()
            .map(|c| c.get())
            .unwrap_or(1);
        let hit = monte_carlo_thread_count(1 << 18, 2);
        if cores > 1 {
            assert!(
                hit > 1,
                "work above the gate must fan out on a {cores}-core host, got {hit}"
            );
        } else {
            assert_eq!(hit, 1, "single-core host has nothing to fan out to");
        }
        assert!(
            hit <= cores,
            "worker count {hit} must not exceed {cores} cores"
        );
        // Which arm the above-gate case actually took is a property of the
        // host, so print it: a reviewer running with `--nocapture` can see the
        // parallel reduction was reached rather than assume it (frankenscipy-yq1k8).
        println!("monte_carlo gate: cores={cores}, workers_above_gate={hit}");
    }

    #[test]
    fn monte_carlo_parallel_reduction_agrees_with_serial_reference() {
        let f = |x: &[f64]| (x[0] * x[1]).sin() + x[0] * x[0];
        let bounds = [(0.0_f64, 1.0), (-0.5, 2.0)];

        // Serial arm: the gate returns 1, so this must be BIT-identical.
        let n_serial = 4096;
        assert_eq!(
            monte_carlo_thread_count(n_serial, bounds.len()),
            1,
            "this arm is only meaningful while it stays under the gate"
        );
        let (got, got_se) = monte_carlo_integrate(f, &bounds, n_serial, 20_260_815);
        let (want, want_se) = monte_carlo_serial_reference(f, &bounds, n_serial, 20_260_815);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "serial arm must be bit-exact"
        );
        assert_eq!(
            got_se.to_bits(),
            want_se.to_bits(),
            "serial arm std-err must be bit-exact"
        );

        // Parallel arm: same RNG stream and same sample points via lcg_jump, so
        // the ONLY admissible difference is cross-chunk reassociation of the
        // two sums. That is documented as non-bit-identical, so this asserts a
        // tight relative agreement instead of equal bits.
        let n_par = 1 << 18;
        let (got, got_se) = monte_carlo_integrate(f, &bounds, n_par, 20_260_815);
        let (want, want_se) = monte_carlo_serial_reference(f, &bounds, n_par, 20_260_815);
        assert!(
            (got - want).abs() <= 1e-12 * want.abs(),
            "parallel reduction drifted beyond reassociation: got {got}, serial {want}"
        );
        assert!(
            (got_se - want_se).abs() <= 1e-9 * want_se.abs(),
            "parallel std-err drifted beyond reassociation: got {got_se}, serial {want_se}"
        );
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

    /// NaN bounds are rejected; infinite bounds are integrated (QAGIE), with `b < a` negated,
    /// as SciPy does. Infinite bounds used to be rejected.
    #[test]
    fn quad_rejects_nan_bounds_and_integrates_infinite_ones() {
        for (a, b) in [(f64::NAN, 1.0), (0.0, f64::NAN)] {
            let err = quad(|x| x, a, b, QuadOptions::default()).expect_err("NaN bound");
            assert!(matches!(
                err,
                IntegrateValidationError::QuadInvalidBounds { .. }
            ));
        }
        let tail = (-1.0_f64).exp();
        let forward = quad(|x| (-x).exp(), 1.0, f64::INFINITY, QuadOptions::default()).unwrap();
        let reversed = quad(|x| (-x).exp(), f64::INFINITY, 1.0, QuadOptions::default()).unwrap();
        assert!(forward.converged && (forward.integral - tail).abs() < 1e-14);
        assert_eq!(reversed.integral, -forward.integral);
        let err = quad_points(|x| x, 0.0, f64::INFINITY, &[1.0], QuadOptions::default())
            .expect_err("points with an infinite range");
        assert!(matches!(
            err,
            IntegrateValidationError::QuadInvalidBounds { .. }
        ));
    }

    /// frankenscipy-1ksfv.8: `quad` is QUADPACK. Each case is SciPy 1.17.1's
    /// `quad(f, a, b, full_output=1)` (points where given): the value, and the SAME
    /// neval / last / ier, because the algorithm is the same statement by statement. The old
    /// recursive GK15 differed in all of them, e.g. it could not extrapolate x^-0.9 and it
    /// truncated [0, ∞) at t = 1 − 1e-10.
    #[test]
    fn quad_matches_scipy_quadpack_full_output() {
        type Integrand = fn(f64) -> f64;
        // (name, f, a, b, points, SciPy value, SciPy abserr, neval, last, ier)
        let cases: [(
            &str,
            Integrand,
            f64,
            f64,
            &[f64],
            f64,
            f64,
            usize,
            usize,
            u8,
        ); 12] = [
            (
                "x^-0.5",
                |x| x.powf(-0.5),
                0.0,
                1.0,
                &[],
                1.999_999_999_999_999_1,
                3.774_758_283_725_532e-15,
                231,
                6,
                0,
            ),
            (
                "log x",
                f64::ln,
                0.0,
                1.0,
                &[],
                -0.999_999_999_999_999_9,
                1.110_223_024_625_156_3e-15,
                231,
                6,
                0,
            ),
            (
                "x^-0.9",
                |x| x.powf(-0.9),
                0.0,
                1.0,
                &[],
                9.999_999_999_999_792,
                6.803_446_694_902_959e-13,
                231,
                6,
                0,
            ),
            (
                "1/(1+x^2)",
                |x| 1.0 / (1.0 + x * x),
                0.0,
                f64::INFINITY,
                &[],
                1.570_796_326_794_896_6,
                2.577_791_520_551_927_4e-10,
                45,
                2,
                0,
            ),
            (
                "exp(-x^2)",
                |x| (-x * x).exp(),
                f64::NEG_INFINITY,
                f64::INFINITY,
                &[],
                1.772_453_850_905_515_9,
                1.420_263_678_094_492_3e-8,
                270,
                5,
                0,
            ),
            (
                "exp",
                f64::exp,
                f64::NEG_INFINITY,
                1.0,
                &[],
                std::f64::consts::E,
                1.588_185_429_179_666_6e-10,
                135,
                5,
                0,
            ),
            (
                "sin(1e4 x)",
                |x| (1e4 * x).sin(),
                0.0,
                1.0,
                &[],
                -0.004_014_431_846_301_359_4,
                0.158_410_487_996_306_91,
                2079,
                50,
                1,
            ),
            (
                "|x-1|",
                |x| (x - 1.0).abs(),
                0.0,
                3.0,
                &[],
                2.5,
                2.775_557_561_562_891_4e-15,
                189,
                5,
                0,
            ),
            (
                "|x-1| points",
                |x| (x - 1.0).abs(),
                0.0,
                3.0,
                &[1.0],
                2.5,
                2.775_557_561_562_891_4e-14,
                42,
                2,
                0,
            ),
            (
                "|x-0.3|^-0.5 points",
                |x| (x - 0.3).abs().powf(-0.5),
                0.0,
                1.0,
                &[0.3],
                2.768_765_168_078_440_6,
                2.176_037_128_265_306_8e-13,
                462,
                12,
                0,
            ),
            (
                "cos",
                f64::cos,
                0.0,
                std::f64::consts::FRAC_PI_2,
                &[],
                0.999_999_999_999_999_9,
                1.110_223_024_625_156_4e-14,
                21,
                1,
                0,
            ),
            (
                "x^2 reversed",
                |x| x * x,
                2.0,
                0.0,
                &[],
                -2.666_666_666_666_667,
                2.960_594_732_333_751e-14,
                21,
                1,
                0,
            ),
        ];
        for (name, f, a, b, points, value, abserr, neval, last, ier) in cases {
            let (result, info) =
                quad_full_output(f, a, b, points, QuadOptions::default()).expect(name);
            assert!(
                (result.integral - value).abs() <= 1e-14 * value.abs(),
                "{name}: {} vs SciPy {value}",
                result.integral
            );
            assert!(
                (result.error - abserr).abs() <= 1e-3 * abserr,
                "{name}: abserr {} vs SciPy {abserr}",
                result.error
            );
            assert_eq!(
                (result.neval, info.last, info.ier),
                (neval, last, ier),
                "{name}: (neval, last, ier)"
            );
            assert_eq!(result.converged, ier == 0, "{name}");
            assert_eq!(info.message.is_empty(), ier == 0, "{name}");
        }
    }

    /// A NaN integrand must not make `quad` run away (frankenscipy-t45u3): QUADPACK's work is
    /// bounded by `limit` subintervals. It reports non-convergence.
    #[test]
    fn quad_nan_integrand_is_bounded_and_not_converged() {
        let r = quad(|_| f64::NAN, 0.0, 1.0, QuadOptions::default()).expect("quad");
        assert!(!r.converged);
        assert!(r.neval <= 21 * (2 * 50 - 1), "neval {}", r.neval);
        // QUADPACK's invalid-input case is SciPy's ValueError.
        let err = quad(
            |x| x,
            0.0,
            1.0,
            QuadOptions {
                epsabs: 0.0,
                epsrel: 1e-20,
                ..QuadOptions::default()
            },
        )
        .expect_err("epsabs 0 with a tiny epsrel");
        assert!(matches!(
            err,
            IntegrateValidationError::QuadInvalidTolerance { .. }
        ));
    }

    #[test]
    fn quad_warns_integration_warning_where_scipy_does() {
        // SciPy: quad(lambda x: sin(1/x) if x else 0, 0, 1, limit=5) warns IntegrationWarning
        // ("The maximum number of subdivisions (5) has been achieved. ...") and returns
        // 0.5057893582392152; with full_output=1 the message is returned, not warned.
        let f = |x: f64| if x == 0.0 { 0.0 } else { (1.0 / x).sin() };
        let options = QuadOptions {
            limit: 5,
            ..QuadOptions::default()
        };
        let (r, warnings) = fsci_runtime::catch_warnings(|| quad(f, 0.0, 1.0, options));
        let r = r.expect("quad");
        assert!(!r.converged);
        assert!(
            (r.integral - 0.505_789_358_239_215_2).abs() < 1e-12,
            "{}",
            r.integral
        );
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert_eq!(
            warnings[0].category,
            fsci_runtime::WarningCategory::IntegrationWarning
        );
        assert!(
            warnings[0]
                .message
                .starts_with("The maximum number of subdivisions (5) has been achieved."),
            "{}",
            warnings[0].message
        );

        let (full, warnings) =
            fsci_runtime::catch_warnings(|| quad_full_output(f, 0.0, 1.0, &[], options));
        assert_eq!(full.expect("full output").1.ier, 1);
        assert!(
            warnings.is_empty(),
            "full_output must not warn: {warnings:?}"
        );

        let (_, warnings) =
            fsci_runtime::catch_warnings(|| quad(|x| x * x, 0.0, 1.0, QuadOptions::default()));
        assert!(warnings.is_empty(), "{warnings:?}");
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

    /// frankenscipy-1ksfv.8: `quad(weight=...)` is QUADPACK's QAWCE / QAWSE / QAWOE / QAWFE.
    /// Each case is SciPy 1.17.1's `quad(f, a, b, weight=..., wvar=..., full_output=1)`: the
    /// value and the same (neval, last). The Cauchy weight replaces `quad_cauchy_pv`, which
    /// excised ±1e-8·(b−a) around the pole.
    #[test]
    fn quad_weighted_matches_scipy_quadpack() {
        type Integrand = fn(f64) -> f64;
        let w = QuadWeightOptions::default();
        let inf = f64::INFINITY;
        // (name, f, a, b, weight, SciPy value, neval, last)
        let cases: [(&str, Integrand, f64, f64, QuadWeight, f64, usize, usize); 10] = [
            (
                "cauchy 1",
                |_| 1.0,
                -1.0,
                2.0,
                QuadWeight::Cauchy(0.0),
                std::f64::consts::LN_2,
                25,
                1,
            ),
            (
                "cauchy lorentz",
                |x| 1.0 / (1.0 + x * x),
                0.0,
                5.0,
                QuadWeight::Cauchy(2.0),
                -0.794_076_938_958_521_6,
                145,
                4,
            ),
            (
                "alg chebyshev",
                |_| 1.0,
                -1.0,
                1.0,
                QuadWeight::Alg {
                    alpha: -0.5,
                    beta: -0.5,
                },
                3.141_592_653_589_792_7,
                50,
                2,
            ),
            (
                "alg-loga",
                f64::cos,
                0.0,
                1.0,
                QuadWeight::AlgLogA {
                    alpha: 0.5,
                    beta: 0.25,
                },
                -0.360_298_430_050_768,
                50,
                2,
            ),
            (
                "alg-logb",
                f64::exp,
                0.0,
                1.0,
                QuadWeight::AlgLogB {
                    alpha: 0.0,
                    beta: -0.5,
                },
                -9.850_406_137_810_443,
                40,
                2,
            ),
            (
                "alg-log",
                |_| 1.0,
                0.0,
                1.0,
                QuadWeight::AlgLog {
                    alpha: -0.3,
                    beta: -0.3,
                },
                0.602_161_690_066_764_7,
                50,
                2,
            ),
            (
                "sin 50",
                |x| 1.0 / (1.0 + x * x),
                0.0,
                3.0,
                QuadWeight::Sin(50.0),
                0.018_635_017_300_020_58,
                75,
                2,
            ),
            (
                "cos 200",
                |x| (-x * x).exp(),
                -3.0,
                3.0,
                QuadWeight::Cos(200.0),
                9.143_821_840_874_086e-8,
                75,
                2,
            ),
            (
                "sin -30",
                f64::exp,
                0.0,
                2.0,
                QuadWeight::Sin(-30.0),
                -0.265_117_342_364_312_56,
                25,
                0,
            ),
            (
                "sin 2 on [0, inf)",
                |x| 1.0 / (1.0 + x * x),
                0.0,
                inf,
                QuadWeight::Sin(2.0),
                0.515_905_663_307_661_9,
                410,
                12,
            ),
        ];
        for (name, f, a, b, weight, value, neval, last) in cases {
            let (result, info) =
                quad_weighted_full_output(f, a, b, weight, QuadOptions::default(), w).expect(name);
            assert!(
                (result.integral - value).abs() <= 1e-14 * value.abs(),
                "{name}: {} vs SciPy {value}",
                result.integral
            );
            assert_eq!(
                (result.neval, info.last, info.ier),
                (neval, last, 0),
                "{name}"
            );
            assert!(result.converged, "{name}");
        }

        // cos(0·x) on [1, ∞): SciPy (netlib dqawfe) integrates from 0 and returns 1.0; the
        // integral is e^-1.
        let r = quad_weighted(
            |x| (-x).exp(),
            1.0,
            inf,
            QuadWeight::Cos(0.0),
            QuadOptions::default(),
            w,
        )
        .expect("omega 0");
        assert!(
            (r.integral - (-1.0_f64).exp()).abs() < 1e-12,
            "{}",
            r.integral
        );

        // (-∞, 0] remaps to [0, ∞) as in SciPy: ∫ e^x cos x = 1/2, ∫ e^x sin x = -1/2.
        for (weight, value) in [(QuadWeight::Cos(1.0), 0.5), (QuadWeight::Sin(1.0), -0.5)] {
            let r = quad_weighted(f64::exp, -inf, 0.0, weight, QuadOptions::default(), w)
                .expect("negative half line");
            assert!(
                (r.integral - value).abs() < 1e-12,
                "{weight:?}: {}",
                r.integral
            );
        }

        // What SciPy rejects.
        let rejected = [
            (0.0, inf, QuadWeight::Cauchy(1.0)),
            (0.0, 1.0, QuadWeight::Cauchy(0.0)),
            (
                0.0,
                1.0,
                QuadWeight::Alg {
                    alpha: -1.0,
                    beta: 0.0,
                },
            ),
            (-inf, inf, QuadWeight::Cos(1.0)),
        ];
        for (a, b, weight) in rejected {
            assert!(
                quad_weighted(|x| x, a, b, weight, QuadOptions::default(), w).is_err(),
                "{weight:?} on [{a}, {b}] must be rejected"
            );
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
    fn quad_vec_limit_bounds_work_and_counts_every_evaluation() {
        for limit in [1, 2, 4, 8] {
            let calls = std::cell::Cell::new(0);
            let result = quad_vec(
                |x| {
                    calls.set(calls.get() + 1);
                    vec![(37.0 * x).sin(), x.exp()]
                },
                0.0,
                1.0,
                QuadOptions {
                    epsabs: 0.0,
                    epsrel: 0.0,
                    limit,
                },
            )
            .unwrap();
            assert!(!result.converged, "roundoff prevents exact quadrature");
            assert_eq!(result.neval, calls.get());
            assert!(
                calls.get() <= 15 + 30 * (limit - 1),
                "limit={limit} permits at most {limit} panels, got {} calls",
                calls.get()
            );
        }
    }

    #[test]
    fn quad_vec_avoids_endpoints_and_refines_integrable_singularities() {
        let result = quad_vec(
            |x| {
                assert!(x > 0.0 && x < 1.0, "quadrature must use interior nodes");
                vec![x.sqrt().recip(), (1.0 - x).sqrt().recip()]
            },
            0.0,
            1.0,
            QuadOptions {
                epsabs: 1e-7,
                epsrel: 1e-7,
                limit: 128,
            },
        )
        .unwrap();
        assert!(result.converged, "{result:?}");
        for value in result.integral {
            assert!((value - 2.0).abs() < 2e-7, "{value}");
        }
    }

    #[test]
    fn quad_vec_integrates_infinite_bounds_in_both_directions() {
        for (a, b, expected) in [
            (0.0, f64::INFINITY, 1.0),
            (f64::INFINITY, 0.0, -1.0),
            (f64::NEG_INFINITY, 0.0, 1.0),
            (0.0, f64::NEG_INFINITY, -1.0),
            (f64::NEG_INFINITY, f64::INFINITY, 2.0),
            (f64::INFINITY, f64::NEG_INFINITY, -2.0),
        ] {
            let calls = std::cell::Cell::new(0);
            let result = quad_vec(
                |x| {
                    assert!(x.is_finite());
                    calls.set(calls.get() + 1);
                    let decay = (-x.abs()).exp();
                    vec![decay, x.abs() * decay]
                },
                a,
                b,
                QuadOptions {
                    epsabs: 1e-10,
                    epsrel: 1e-10,
                    limit: 128,
                },
            )
            .unwrap();
            assert!(result.converged, "{a}..{b}: {result:?}");
            assert_eq!(result.neval, calls.get());
            for value in result.integral {
                assert!((value - expected).abs() < 1e-9, "{a}..{b}: {value}");
            }
        }
    }

    #[test]
    fn quad_vec_respects_roundoff_and_nonfinite_integrands() {
        let exact = quad_vec(
            |_| vec![1.0, -2.0],
            0.0,
            1.0,
            QuadOptions {
                epsabs: 0.0,
                epsrel: 0.0,
                limit: 4,
            },
        )
        .unwrap();
        assert!(!exact.converged);
        assert!(exact.error > 0.0);
        assert!(
            exact.neval <= 45,
            "roundoff must stop unproductive refinement"
        );

        for value in [f64::NAN, f64::INFINITY] {
            let result = quad_vec(|_| vec![value, 1.0], 0.0, 1.0, QuadOptions::default()).unwrap();
            assert!(!result.converged);
            assert!(!result.error.is_finite());
            assert_eq!(result.neval, 15);
        }
    }

    #[test]
    fn quad_vec_invalid_input_errors() {
        let bounds_err =
            quad_vec(|x| vec![x], f64::NAN, 1.0, QuadOptions::default()).expect_err("NaN bound");
        assert!(matches!(
            bounds_err,
            IntegrateValidationError::QuadInvalidBounds { .. }
        ));

        let adjacent_err = quad_vec(
            |_| panic!("no interior point exists, so the callback must not run"),
            1.0,
            1.0_f64.next_up(),
            QuadOptions::default(),
        )
        .expect_err("no representable interior");
        assert!(matches!(
            adjacent_err,
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
    fn cubature_preserves_small_nonzero_domains() {
        for width in [1e-20, 1e-100, 1e-200] {
            for reverse in [false, true] {
                let (a, b, expected) = if reverse {
                    (width, 0.0, -1.0)
                } else {
                    (0.0, width, 1.0)
                };
                let result = cubature_scalar(
                    |_| 1.0 / width,
                    &[a],
                    &[b],
                    CubatureOptions {
                        max_subdivisions: 0,
                        ..CubatureOptions::default()
                    },
                )
                .expect("a small nonzero interval still has a nonzero integral");
                assert_eq!(result.status, CubatureStatus::Converged);
                assert!(
                    (result.estimate - expected).abs() < 5e-14,
                    "width={width}, reverse={reverse}: {result:?}"
                );
            }
        }
    }

    #[test]
    fn cubature_singularity_points_are_order_independent() {
        let boundaries = [0.0, 0.25, 0.5, 0.625, 1.0];
        for reverse in [false, true] {
            for points in [
                vec![vec![0.25], vec![0.625], vec![0.5]],
                vec![vec![0.625], vec![0.5], vec![0.25], vec![0.5]],
                vec![vec![0.5], vec![0.25], vec![0.625]],
            ] {
                let (a, b) = if reverse { (1.0, 0.0) } else { (0.0, 1.0) };
                let result = cubature(
                    |x| {
                        assert!(
                            !boundaries.contains(&x[0]),
                            "sampled an excluded point or region boundary: {x:?}"
                        );
                        vec![2.0, x[0]]
                    },
                    &[a],
                    &[b],
                    CubatureOptions {
                        points,
                        max_subdivisions: 0,
                        ..CubatureOptions::default()
                    },
                )
                .expect("all singularity points must split the initial domain");
                let sign = if reverse { -1.0 } else { 1.0 };
                assert_eq!(result.status, CubatureStatus::Converged);
                assert!((result.estimate[0] - 2.0 * sign).abs() < 1e-13);
                assert!((result.estimate[1] - 0.5 * sign).abs() < 1e-13);
                assert_eq!(result.regions.len(), boundaries.len() - 1);
                for (i, region) in result.regions.iter().enumerate() {
                    let (lo, hi) = if reverse {
                        (boundaries[4 - i], boundaries[3 - i])
                    } else {
                        (boundaries[i], boundaries[i + 1])
                    };
                    assert_eq!(region.a, vec![lo]);
                    assert_eq!(region.b, vec![hi]);
                }
            }
        }
    }

    #[test]
    fn cubature_keeps_distinct_nearby_breakpoints() {
        let middle = 0.5_f64;
        let next = middle.next_up();
        let ranges = [(0.0, 1.0), (1.0, 0.0)];
        let regions = initial_cubature_regions(
            &ranges,
            &[BoundTransform::Finite; 2],
            &CubatureOptions {
                points: vec![vec![middle, middle], vec![next, next]],
                ..CubatureOptions::default()
            },
        )
        .expect("distinct representable boundaries are retained");
        assert_eq!(regions.len(), 9);
        assert!(
            regions
                .iter()
                .any(|(a, b)| a == &[middle, next] && b == &[next, middle])
        );
        let volume: f64 = regions
            .iter()
            .map(|(a, b)| (b[0] - a[0]) * (b[1] - a[1]))
            .sum();
        assert!((volume + 1.0).abs() < 1e-15);
    }

    #[test]
    fn cubature_range_safe_geometry_and_weights() {
        let cases = [
            (vec![1e308], vec![1.2e308], 1.0 / (1.2e308 - 1e308), 1.0),
            (vec![-1e308], vec![1e308], 1e-308, 2.0),
            (vec![0.0, 0.0], vec![1e-200, 1e-200], 1e300, 1e-100),
            (vec![0.0, 0.0], vec![1e200, 1e200], 1e-300, 1e100),
            (vec![0.0], vec![1.0], 1e308, 1e308),
            (
                vec![0.0],
                vec![3.0 * f64::from_bits(1)],
                1e308,
                (3.0 * f64::from_bits(1)) * 1e308,
            ),
        ];
        for (a, b, density, expected) in cases {
            for rule in [
                CubatureRule::Gk21,
                CubatureRule::Gk15,
                CubatureRule::GenzMalik,
            ] {
                if rule == CubatureRule::GenzMalik && a.len() < 2 {
                    continue;
                }
                let result = cubature_scalar(
                    |x| {
                        for ((&coordinate, &lo), &hi) in x.iter().zip(&a).zip(&b) {
                            assert!(coordinate.is_finite() && coordinate > lo && coordinate < hi);
                        }
                        density
                    },
                    &a,
                    &b,
                    CubatureOptions {
                        rule,
                        max_subdivisions: 0,
                        ..CubatureOptions::default()
                    },
                )
                .expect("finite integral despite extreme intermediate scales");
                assert_eq!(result.status, CubatureStatus::Converged, "{rule:?}: {result:?}");
                assert!(
                    (result.estimate / expected - 1.0).abs() < 5e-13,
                    "{rule:?}: {result:?}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn cubature_mixed_axis_scales_preserve_volume_and_orientation() {
        for widths in [
            [1e-200, 1e-200, 1e200, 1e200],
            [1e200, 1e200, 1e-200, 1e-200],
            [1e-200, 1e200, 1e-200, 1e200],
        ] {
            for reverse in [false, true] {
                let mut a = [0.0; 4];
                let mut b = widths;
                if reverse {
                    a[0] = widths[0];
                    b[0] = 0.0;
                }
                let result = cubature_scalar(
                    |_| 1.0,
                    &a,
                    &b,
                    CubatureOptions {
                        rule: CubatureRule::GenzMalik,
                        max_subdivisions: 0,
                        ..CubatureOptions::default()
                    },
                )
                .expect("mixed scale volume remains one");
                let expected = if reverse { -1.0 } else { 1.0 };
                assert_eq!(result.status, CubatureStatus::Converged);
                assert!((result.estimate - expected).abs() < 5e-13, "{result:?}");
            }
        }
    }

    #[test]
    fn cubature_preserves_subnormal_totals_and_reports_panel_underflow() {
        let tiny = f64::from_bits(1);
        for rule in [CubatureRule::Gk21, CubatureRule::Gk15] {
            let result = cubature_scalar(
                |_| tiny,
                &[0.0],
                &[1.0],
                CubatureOptions {
                    rule,
                    ..CubatureOptions::default()
                },
            )
            .expect("weighted terms must be accumulated before subnormal rounding");
            assert_eq!(result.status, CubatureStatus::Converged);
            assert_eq!(result.estimate.to_bits(), tiny.to_bits());

            // Each half-panel rounds below the available result precision.
            // Report that limit instead of calling an erased integral converged.
            let split = cubature_scalar(
                |_| tiny,
                &[0.0],
                &[1.0],
                CubatureOptions {
                    rule,
                    points: vec![vec![0.5]],
                    ..CubatureOptions::default()
                },
            )
            .expect("panel underflow returns diagnostics");
            assert_eq!(split.status, CubatureStatus::NotConverged);
            assert_eq!(split.subdivisions, 0);
            assert!(split.error > 0.0);
        }
    }

    #[test]
    fn cubature_nonfinite_estimates_never_converge() {
        assert!(!componentwise_cubature_converged(
            &[f64::INFINITY],
            &[f64::INFINITY],
            0.0,
            1e-8,
        ));
        for points in [Vec::new(), vec![vec![1.0], vec![2.0], vec![3.0]]] {
            let result = cubature_scalar(
                |_| 1e308,
                &[0.0],
                &[4.0],
                CubatureOptions {
                    points,
                    ..CubatureOptions::default()
                },
            )
            .expect("unrepresentable integral returns nonconvergence diagnostics");
            assert_eq!(result.status, CubatureStatus::NotConverged);
            assert!(!result.estimate.is_finite());
            assert_eq!(result.subdivisions, 0);
        }
    }

    #[test]
    fn cubature_unsplittable_region_keeps_its_estimate() {
        let a = 1.0_f64;
        let b = f64::from_bits(a.to_bits() + 2);
        let result = cubature(
            |x| {
                assert!(x[0] > a && x[0] < b);
                vec![(10.0 * x[1]).exp() / (b - a)]
            },
            &[a, 0.0],
            &[b, 1.0],
            CubatureOptions {
                rule: CubatureRule::GenzMalik,
                rtol: 1e-15,
                ..CubatureOptions::default()
            },
        )
        .expect("an unsplittable region retains its best estimate");
        assert_eq!(result.status, CubatureStatus::NotConverged);
        assert_eq!(result.subdivisions, 0);
        assert_eq!(result.regions.len(), 1);
        assert!(result.estimate[0].is_finite() && result.estimate[0] > 0.0);
        assert_eq!(result.estimate, result.regions[0].estimate);
        assert_eq!(result.error, result.regions[0].error);
    }

    #[test]
    fn cubature_rejects_unrepresentable_geometry_before_callbacks() {
        let adjacent = cubature_scalar(
            |_| panic!("an interval without an interior node must fail before evaluation"),
            &[1.0],
            &[1.0_f64.next_up()],
            CubatureOptions::default(),
        );
        assert!(matches!(
            adjacent,
            Err(IntegrateValidationError::QuadInvalidBounds { .. })
        ));
        for rule in [CubatureRule::Gk21, CubatureRule::GenzMalik] {
            let dimensions = usize::BITS as usize;
            let excessive = cubature_scalar(
                |_| panic!("unrepresentable node counts must fail before evaluation"),
                &vec![0.0; dimensions],
                &vec![1.0; dimensions],
                CubatureOptions {
                    rule,
                    ..CubatureOptions::default()
                },
            );
            assert!(matches!(
                excessive,
                Err(IntegrateValidationError::ResourceExhausted { .. })
            ));
        }
        for nan in [f64::NAN, -f64::NAN] {
            for (a, b) in [(nan, 1.0), (0.0, nan)] {
                assert!(
                    cubature_scalar(
                        |_| panic!("NaN must not select an infinite-bound transform"),
                        &[a],
                        &[b],
                        CubatureOptions::default(),
                    )
                    .is_err()
                );
            }
        }
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

    /// frankenscipy-szq1n.12: `rule` used to be accepted and ignored (every rule ran one
    /// embedded 3-point Gauss rule). With no subdivision each rule's single-region estimate,
    /// error and node count must be SciPy's: `ProductNestedFixed([GaussKronrodQuadrature(21 |
    /// 15)] * 2)` and `GenzMalikCubature(2)`, SciPy 1.17.1.
    #[test]
    fn cubature_rule_selects_the_scipy_rule() {
        let f = |x: &[f64]| (1.3 * (x[0] + x[1])).exp() * (1.0 + x[0] * x[0]);
        let single_region = |rule| {
            cubature_scalar(
                f,
                &[0.0, 0.0],
                &[1.0, 1.5],
                CubatureOptions {
                    rule,
                    max_subdivisions: 0,
                    ..CubatureOptions::default()
                },
            )
            .expect("cubature")
        };
        // (rule, SciPy estimate, SciPy error, rule nodes); neval adds the one sample call.
        let cases = [
            (CubatureRule::Gk21, 13.742_741_172_032_046, 1.4e-14, 441),
            (
                CubatureRule::GaussKronrod,
                13.742_741_172_032_046,
                1.4e-14,
                441,
            ),
            (CubatureRule::Gk15, 13.742_741_172_032_044, 1.4e-14, 225),
            (
                CubatureRule::GenzMalik,
                13.742_792_387_031_008,
                0.020_992_239_217_037_377,
                17,
            ),
        ];
        for (rule, estimate, error, nodes) in cases {
            let result = single_region(rule);
            assert!(
                (result.estimate - estimate).abs() <= 1e-13 * estimate,
                "{rule:?}: estimate {} vs SciPy {estimate}",
                result.estimate
            );
            if error > 1e-6 {
                assert!(
                    (result.error - error).abs() <= 1e-9 * error,
                    "{rule:?}: error {} vs SciPy {error}",
                    result.error
                );
            } else {
                assert!(result.error < 1e-12, "{rule:?}: error {}", result.error);
            }
            assert_eq!(result.neval, nodes + 1, "{rule:?}");
        }

        // Adaptive runs converge with every rule; the exact value is 13.74274117203205.
        for rule in [
            CubatureRule::Gk21,
            CubatureRule::Gk15,
            CubatureRule::GenzMalik,
        ] {
            let result = cubature_scalar(
                f,
                &[0.0, 0.0],
                &[1.0, 1.5],
                CubatureOptions {
                    rule,
                    rtol: 1e-10,
                    ..CubatureOptions::default()
                },
            )
            .expect("cubature");
            assert_eq!(result.status, CubatureStatus::Converged, "{rule:?}");
            assert!(
                (result.estimate - 13.742_741_172_032_05).abs() < 1e-9,
                "{rule:?}: {}",
                result.estimate
            );
        }

        // SciPy: "Genz-Malik cubature is only defined for ndim >= 2".
        let err = cubature_scalar(
            |x| x[0],
            &[0.0],
            &[1.0],
            CubatureOptions {
                rule: CubatureRule::GenzMalik,
                ..CubatureOptions::default()
            },
        )
        .expect_err("1-D Genz-Malik");
        assert!(matches!(
            err,
            IntegrateValidationError::QuadInvalidBounds { .. }
        ));
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
    fn dblquad_nan_bounds_error() {
        let err = dblquad(
            |y, x| x * y,
            f64::NAN,
            1.0,
            |_| 0.0,
            |_| 1.0,
            DblquadOptions::default(),
        )
        .expect_err("NaN bound");
        assert!(matches!(
            err,
            IntegrateValidationError::QuadInvalidBounds { .. }
        ));
    }

    /// Infinite limits, now that `quad` is QUADPACK (they used to be rejected). SciPy 1.17.1:
    /// dblquad(exp(-x-y), 0, ∞, 0, ∞) = 0.9999999999976233; dblquad and nquad of
    /// exp(-x²-y²) over R² = 3.141592653589777; tplquad(exp(-x-y-z), [0, ∞)³) =
    /// 0.9999999999603953.
    #[test]
    fn multi_dimensional_quad_integrates_infinite_limits() {
        let inf = f64::INFINITY;
        let d = dblquad(
            |y, x| (-x - y).exp(),
            0.0,
            inf,
            |_| 0.0,
            |_| inf,
            DblquadOptions::default(),
        )
        .expect("dblquad");
        assert!(d.converged && (d.integral - 1.0).abs() < 1e-8, "{d:?}");
        let g = dblquad(
            |y, x| (-x * x - y * y).exp(),
            -inf,
            inf,
            |_| -inf,
            |_| inf,
            DblquadOptions::default(),
        )
        .expect("dblquad R^2");
        assert!((g.integral - std::f64::consts::PI).abs() < 1e-8, "{g:?}");
        let t = tplquad(
            |z, y, x| (-x - y - z).exp(),
            0.0,
            inf,
            |_| 0.0,
            |_| inf,
            |_, _| 0.0,
            |_, _| inf,
            DblquadOptions::default(),
        )
        .expect("tplquad");
        assert!((t.integral - 1.0).abs() < 1e-8, "{t:?}");
        let n = nquad(
            |v| (-v[0] * v[0] - v[1] * v[1]).exp(),
            &[(-inf, inf), (-inf, inf)],
            QuadOptions::default(),
        )
        .expect("nquad R^2");
        assert!(
            n.converged && (n.integral - std::f64::consts::PI).abs() < 1e-8,
            "{n:?}"
        );
        let nan = nquad(|v| v[0], &[(f64::NAN, 1.0)], QuadOptions::default());
        assert!(matches!(
            nan,
            Err(IntegrateValidationError::QuadInvalidBounds { .. })
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
            // Shortest literal that round-trips to the same f64 as the original
            // 4.180_925_279_702_298_8 (both are 0x4010b94479fd5d36) --
            // clippy::excessive_precision, frankenscipy-3qjah. Value unchanged.
            (6, 4.180_925_279_702_299),
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

    /// SciPy 1.17.1 `cumulative_trapezoid(y, x, initial=v)` raises `ValueError: initial must be
    /// None or 0` for every v other than 0, and `At least one point is required` for an empty y.
    /// With initial=0 it prepends that zero (sign kept) to `cumulative_trapezoid(y, x)`.
    #[test]
    fn cumulative_trapezoid_initial_accepts_only_zero_like_scipy() {
        let y = [1.0, 2.0, 4.0, 7.0];
        let x = [0.0, 1.0, 1.5, 3.0];
        for initial in [5.0, -3.0, 1e-300, f64::NAN, f64::INFINITY] {
            let err = cumulative_trapezoid_initial(&y, &x, initial)
                .expect_err("SciPy raises for a nonzero initial");
            assert!(matches!(
                err,
                IntegrateValidationError::QuadInvalidBounds { .. }
            ));
        }
        // SciPy: [0.0, 1.5, 3.0, 11.25].
        let tail = cumulative_trapezoid(&y, &x).expect("valid samples");
        for initial in [0.0, -0.0] {
            let got = cumulative_trapezoid_initial(&y, &x, initial).expect("initial = 0");
            assert_eq!(got.len(), y.len());
            assert_eq!(got[0].to_bits(), initial.to_bits());
            assert_eq!(&got[1..], tail.as_slice());
        }
        assert_eq!(
            cumulative_trapezoid_initial(&y, &x, 0.0).expect("initial = 0"),
            vec![0.0, 1.5, 3.0, 11.25]
        );
        assert_eq!(
            cumulative_trapezoid_initial(&[2.0], &[0.0], 0.0).expect("one sample"),
            vec![0.0]
        );
        assert!(cumulative_trapezoid_initial(&[], &[], 0.0).is_err());
        assert!(cumulative_trapezoid_initial(&[1.0, 2.0], &[0.0], 0.0).is_err());
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
    fn nquad_callable_ranges_use_scipy_variable_order() {
        let inner = |outer: &[f64]| {
            assert_eq!(outer.len(), 1);
            (0.0, 2.0 * outer[0])
        };
        let result = nquad_with_ranges(
            |args| args[0] + 10.0 * args[1],
            &[NquadRange::Dynamic(&inner), NquadRange::Fixed(0.0, 3.0)],
            &[],
        )
        .unwrap();
        assert!(result.converged);
        assert!((result.integral - 198.0).abs() < 1e-11);

        let simplex_inner = |outer: &[f64]| {
            assert_eq!(outer.len(), 2);
            (0.0, 1.0 - outer[0] - outer[1])
        };
        let simplex_middle = |outer: &[f64]| (0.0, 1.0 - outer[0]);
        let result = nquad_with_ranges(
            |_| 1.0,
            &[
                NquadRange::Dynamic(&simplex_inner),
                NquadRange::Dynamic(&simplex_middle),
                NquadRange::Fixed(0.0, 1.0),
            ],
            &[QuadOptions::default()],
        )
        .unwrap();
        assert!(result.converged);
        assert!((result.integral - 1.0 / 6.0).abs() < 1e-13);
    }

    #[test]
    fn nquad_callable_ranges_support_infinite_and_reversed_intervals() {
        for sign in [-1.0, 1.0] {
            let inner = |outer: &[f64]| {
                if sign > 0.0 {
                    (0.0, outer[0])
                } else {
                    (outer[0], 0.0)
                }
            };
            let result = nquad_with_ranges(
                |args| (-args[0] - args[1]).exp(),
                &[
                    NquadRange::Dynamic(&inner),
                    NquadRange::Fixed(0.0, f64::INFINITY),
                ],
                &[QuadOptions {
                    epsabs: 1e-10,
                    epsrel: 1e-10,
                    limit: 100,
                }],
            )
            .unwrap();
            assert!(result.converged);
            assert!((result.integral - sign * 0.5).abs() < 1e-9);
        }
    }

    #[test]
    fn nquad_counts_only_user_integrand_evaluations() {
        let samples = std::cell::Cell::new(0);
        let func = |_: &[f64]| {
            samples.set(samples.get() + 1);
            1.0
        };
        let result = nquad(func, &[(0.0, 1.0); 3], QuadOptions::default()).unwrap();
        assert_eq!(result.neval, samples.get());
        assert_eq!(result.neval, 21usize.pow(3));
        samples.set(0);
        let result = nquad_with_ranges(func, &[NquadRange::Fixed(0.0, 1.0); 3], &[]).unwrap();
        assert_eq!(result.neval, samples.get());
        assert_eq!(result.neval, 21usize.pow(3));
    }

    #[test]
    fn nquad_per_dimension_options_preserve_inner_failure() {
        let inner_options = QuadOptions {
            limit: 3,
            ..QuadOptions::default()
        };
        let singular = |x: f64| (x - 1.0 / 3.0).abs().powf(-0.9);
        let inner = quad(singular, 0.0, 1.0, inner_options).unwrap();
        let result = nquad_with_ranges(
            |args| singular(args[0]),
            &[NquadRange::Fixed(0.0, 1.0); 2],
            &[inner_options, QuadOptions::default()],
        )
        .unwrap();
        assert!(!result.converged);
        assert!((result.integral - inner.integral).abs() < 1e-12);
        assert_eq!(result.error, inner.error);
        assert_eq!(result.neval, 21 * inner.neval);
    }

    #[test]
    fn nquad_invalid_dynamic_range_fails_once_without_sampling_integrand() {
        let bound_calls = std::cell::Cell::new(0);
        let invalid = |_: &[f64]| {
            bound_calls.set(bound_calls.get() + 1);
            (0.0, f64::NAN)
        };
        let result = nquad_with_ranges(
            |_| panic!("invalid inner bounds must not reach the integrand"),
            &[NquadRange::Dynamic(&invalid), NquadRange::Fixed(0.0, 1.0)],
            &[],
        );
        assert!(matches!(
            result,
            Err(IntegrateValidationError::QuadInvalidBounds { .. })
        ));
        assert_eq!(bound_calls.get(), 1);
    }

    #[test]
    fn nquad_rejects_invalid_options_before_any_bound_callback() {
        let bounds = |_: &[f64]| panic!("options must be validated first");
        let ranges = [NquadRange::Dynamic(&bounds), NquadRange::Fixed(0.0, 0.0)];
        assert!(nquad_with_ranges(|_| 1.0, &ranges, &[QuadOptions::default(); 3]).is_err());
        let bad = QuadOptions {
            epsabs: f64::NAN,
            ..QuadOptions::default()
        };
        assert!(nquad_with_ranges(|_| 1.0, &ranges, &[bad, QuadOptions::default()]).is_err());
        let empty = nquad_with_ranges(
            |args| {
                assert!(args.is_empty());
                42.0
            },
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(empty.integral, 42.0);
        assert_eq!(empty.neval, 1);
    }

    #[test]
    fn nquad_rejects_unattainable_tolerances_beneath_empty_ranges() {
        let bounds = |_: &[f64]| panic!("invalid tolerances must not evaluate bounds");
        let ranges = [NquadRange::Dynamic(&bounds), NquadRange::Fixed(0.0, 0.0)];
        for epsrel in [0.0, 49.0 * f64::EPSILON] {
            let options = [
                QuadOptions {
                    epsabs: 0.0,
                    epsrel,
                    limit: 50,
                },
                QuadOptions::default(),
            ];
            assert!(matches!(
                nquad_with_ranges(
                    |_| panic!("invalid options must not evaluate a model"),
                    &ranges,
                    &options
                ),
                Err(IntegrateValidationError::QuadInvalidTolerance { .. })
            ));
        }
    }

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
    fn nquad_empty_ranges_evaluates_0d_like_scipy() {
        let opts = QuadOptions::default();
        let res = nquad(|_| 42.0, &[], opts).expect("empty nquad ranges evaluates 0-D");
        assert_eq!(res.integral, 42.0);
        assert_eq!(res.neval, 1);
        assert!(res.converged);
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

    #[test]
    fn nquad_stops_after_a_malformed_inner_range() {
        let samples = std::cell::Cell::new(0usize);
        let result = nquad(
            |_args| {
                samples.set(samples.get() + 1);
                1.0
            },
            &[(0.0, 1.0), (0.0, f64::NAN)],
            QuadOptions::default(),
        );
        assert!(
            result.is_err(),
            "an invalid nested range must be reported without retrying it"
        );
        assert_eq!(
            samples.get(),
            0,
            "invalid nested bounds must fail before the outer adaptive loop samples the integrand"
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
    fn nsum_left_infinite_uses_the_finite_endpoint_and_step() {
        let end = 100_000.25;
        let calls = std::cell::Cell::new(0);
        let result = nsum(
            |x| {
                calls.set(calls.get() + 1);
                assert!(x.is_finite() && x <= end);
                (x - end).exp()
            },
            f64::NEG_INFINITY,
            end,
            0.5,
            1e-12,
            1e-10,
        );
        let exact = 1.0 / -(-0.5_f64).exp_m1();
        assert!(
            result.converged && (result.sum - exact).abs() < 1e-9,
            "{result:?}"
        );
        assert_eq!(result.nfev, calls.get());
        assert!(
            result.nfev < 5000,
            "shifted endpoint exhausted refinement: {result:?}"
        );
    }

    #[test]
    fn nsum_bilateral_geometric_tails_count_zero_once() {
        let calls = std::cell::Cell::new(0);
        let result = nsum(
            |x| {
                calls.set(calls.get() + 1);
                if x < 0.0 {
                    (2.0 * x).exp()
                } else {
                    (-x).exp()
                }
            },
            f64::NEG_INFINITY,
            f64::INFINITY,
            0.5,
            1e-12,
            1e-10,
        );
        let exact = 1.0 / -(-0.5_f64).exp_m1() + 1.0 / 1.0_f64.exp_m1();
        assert!(
            result.converged && (result.sum - exact).abs() < 1e-9,
            "{result:?}"
        );
        assert!(result.error <= 1e-12 + 1e-10 * result.sum.abs());
        assert_eq!(result.nfev, calls.get());
    }

    #[test]
    fn nsum_bilateral_tests_error_against_the_combined_sum() {
        let cancelled = nsum(
            |x| {
                if x == 0.0 {
                    0.0
                } else {
                    x.signum() / (1.0 + x * x)
                }
            },
            f64::NEG_INFINITY,
            f64::INFINITY,
            1.0,
            0.0,
            1e-6,
        );
        // Both one-sided series converge. Their exact sum is zero, so
        // agreement to a relative tolerance on each tail is insufficient.
        assert!(cancelled.sum.is_finite() && cancelled.error.is_finite());
        assert!(!cancelled.converged && cancelled.error > 1e-6 * cancelled.sum.abs());

        let large = (1_u64 << 53) as f64;
        let rounded = nsum(
            |x| {
                if x == 0.0 {
                    large
                } else if x == -1.0 {
                    1.0
                } else {
                    0.0
                }
            },
            f64::NEG_INFINITY,
            f64::INFINITY,
            1.0,
            0.0,
            1e-20,
        );
        assert!(!rounded.converged);
        assert_eq!((rounded.sum, rounded.error), (large, 1.0));
    }

    #[test]
    fn nsum_bilateral_propagates_a_nonfinite_tail() {
        for invalid_left in [false, true] {
            let calls = std::cell::Cell::new(0);
            let result = nsum(
                |x| {
                    calls.set(calls.get() + 1);
                    if (x < 0.0) == invalid_left {
                        f64::NAN
                    } else {
                        0.0
                    }
                },
                f64::NEG_INFINITY,
                f64::INFINITY,
                1.0,
                1e-12,
                1e-10,
            );
            assert!(!result.converged && result.sum.is_nan());
            assert_eq!(result.error, f64::INFINITY);
            assert_eq!(result.nfev, calls.get());
        }
    }

    #[test]
    fn nsum_retains_cancellation_and_reports_unrepresentable_low_bits() {
        let terms = [1e16, 1.0, -1e16];
        let exact = nsum(|x| terms[x as usize], 0.0, 2.0, 1.0, 0.0, 1e-11);
        assert!(exact.converged);
        assert_eq!((exact.sum, exact.error), (1.0, 0.0));

        // The mathematical sum of 2^53 and 1 lies between adjacent floats.
        let large = (1_u64 << 53) as f64;
        let rounded = nsum(
            |x| if x == 0.0 { large } else { 1.0 },
            0.0,
            1.0,
            1.0,
            0.0,
            1e-20,
        );
        assert!(!rounded.converged);
        assert_eq!(rounded.sum, large);
        assert_eq!(rounded.error, 1.0);
    }

    #[test]
    fn nsum_handles_wide_spans_and_detects_collapsed_interior_nodes() {
        let wide = nsum(
            |x| (x / f64::MAX).powi(2),
            -f64::MAX,
            f64::MAX,
            f64::MAX,
            0.0,
            1e-11,
        );
        assert!(wide.converged);
        assert_eq!((wide.sum, wide.nfev), (2.0, 3));

        let boundary = (1_u64 << 53) as f64;
        let aliased = nsum(|_| 1.0, boundary - 1.0, boundary + 4.0, 1.0, 0.0, 1e-11);
        assert!(!aliased.converged && aliased.sum.is_nan());
        assert_eq!(aliased.nfev, 2);
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
        let smooth: &[QuadCase] = &[
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
    fn tanhsinh_nonfinite_interior_values_never_converge() {
        for (a, b) in [
            (0.0, 1.0),
            (0.0, f64::INFINITY),
            (f64::NEG_INFINITY, 0.0),
            (f64::NEG_INFINITY, f64::INFINITY),
        ] {
            for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let calls = std::cell::Cell::new(0usize);
                let result = tanhsinh(
                    |_| {
                        calls.set(calls.get() + 1);
                        bad
                    },
                    a,
                    b,
                    0.0,
                    1e-12,
                    16,
                );
                assert!(!result.converged, "{a}..{b}, value={bad}: {result:?}");
                assert!(result.integral.is_nan());
                assert_eq!(result.error, f64::INFINITY);
                assert_eq!(result.neval, 1);
                assert_eq!(calls.get(), 1);
            }
        }
    }

    #[test]
    fn tanhsinh_stops_when_refinement_finds_nonfinite_values() {
        let calls = std::cell::Cell::new(0usize);
        let rejected = std::cell::Cell::new(false);
        let result = tanhsinh(
            |x| {
                assert!(!rejected.get(), "evaluated again after an invalid sample");
                calls.set(calls.get() + 1);
                if (0.30..0.34).contains(&x) {
                    rejected.set(true);
                    f64::NAN
                } else {
                    1.0
                }
            },
            0.0,
            1.0,
            0.0,
            1e-12,
            16,
        );
        assert!(rejected.get(), "must exercise a new refinement node");
        assert!(!result.converged);
        assert!(result.integral.is_nan());
        assert_eq!(result.error, f64::INFINITY);
        assert_eq!(result.neval, calls.get());
        assert!(result.neval > 1 && result.neval < 100);
    }

    #[test]
    fn tanhsinh_large_finite_intervals_keep_finite_nodes_and_weights() {
        for (a, b, exact) in [
            (-1e308_f64, 1e308_f64, 2.0),
            (1e308, -1e308, -2.0),
            (1e308, 1.7e308, 0.7),
        ] {
            let result = tanhsinh(
                |x| {
                    assert!(x.is_finite() && x > a.min(b) && x < a.max(b));
                    1e-308
                },
                a,
                b,
                0.0,
                1e-12,
                16,
            );
            assert!(result.converged, "{a}..{b}: {result:?}");
            assert!((result.integral - exact).abs() <= 1e-12);
            assert!(result.neval > 0 && result.neval < 1000);
        }
        let result = tanhsinh(|x| 1e-308 * (x / 1e308), 1e308, 1.7e308, 0.0, 1e-12, 16);
        assert!(result.converged);
        assert!((result.integral - 0.945).abs() <= 1e-12);
    }

    #[test]
    fn tanhsinh_error_estimate_preserves_relative_accuracy_at_small_scales() {
        for scale in [1e-200, 1e-300, -1e-200, 1e200] {
            let result = tanhsinh(|x| scale * x.exp(), 0.0, 1.0, 0.0, 1e-12, 16);
            assert!(result.converged, "scale={scale}: {result:?}");
            assert!((result.integral / scale - (std::f64::consts::E - 1.0)).abs() <= 1e-12);
            assert!(result.error.is_finite());
        }
    }

    #[test]
    fn tanhsinh_weighted_accumulation_preserves_large_finite_integrals() {
        // The unweighted sum overflowed during refinement although every
        // sample and the final integral are finite. Scaling each level inside
        // the accumulator must make the answer continuous across this range.
        for scale in [1e300, 1e307, 5e307, -5e307] {
            let result = tanhsinh(|x| scale * x.exp(), 0.0, 1.0, 0.0, 1e-12, 12);
            assert!(result.converged, "scale={scale}: {result:?}");
            assert!(result.integral.is_finite());
            assert!(result.error.is_finite());
            assert!(
                (result.integral / scale - (std::f64::consts::E - 1.0)).abs() < 1e-12,
                "scale={scale}: {result:?}"
            );
            assert!(result.neval < 1000);
        }
        // A subnormal h*jac must not discard a contribution that a large
        // finite integrand brings back into the normal range.
        let term = de_weighted_term(0.125, f64::from_bits(1), 1e300);
        let expected = (f64::from_bits(1) * 1e300) * 0.125;
        assert_eq!(term, expected);
        assert!(term.is_normal());
    }

    #[test]
    fn tanhsinh_endpoint_tail_limits_are_not_false_convergence() {
        let exact = std::f64::consts::PI.sqrt();
        for (lo, hi, direction) in [(1.0, f64::INFINITY, 1.0), (f64::NEG_INFINITY, -1.0, -1.0)] {
            let result = tanhsinh(
                |x| {
                    let distance = direction * x - 1.0;
                    (-distance).exp() / distance.sqrt()
                },
                lo,
                hi,
                0.0,
                1e-12,
                10,
            );
            let actual_error = (result.integral - exact).abs();
            assert!(actual_error > 1e-9, "must exercise endpoint rounding");
            assert!(!result.converged, "{result:?}");
            assert!(
                result.error >= actual_error,
                "{result:?}, error={actual_error}"
            );
        }
        for exponent in [-0.5, -0.75, -0.9] {
            let result = tanhsinh(|x| (1.0 - x).powf(exponent), 0.0, 1.0, 0.0, 1e-12, 10);
            let actual_error = (result.integral - 1.0 / (exponent + 1.0)).abs();
            assert!(!result.converged, "exponent={exponent}: {result:?}");
            assert!(
                result.error >= actual_error,
                "exponent={exponent}: {result:?}, error={actual_error}"
            );
        }
    }

    #[test]
    fn tanhsinh_error_includes_roundoff_after_cancellation() {
        let result = tanhsinh(|x| x - 0.5, 0.0, 1.0, 0.0, 1e-12, 8);
        assert!(result.integral.abs() < 1e-15);
        assert!(result.error > 0.0, "{result:?}");
        assert!(
            !result.converged,
            "a zero-relative-only tolerance is unresolved"
        );
        let with_absolute_tolerance = tanhsinh(|x| x - 0.5, 0.0, 1.0, 1e-12, 1e-12, 8);
        assert!(with_absolute_tolerance.converged);
    }

    #[test]
    fn tanhsinh_slow_endpoint_decay_does_not_hide_a_large_omitted_tail() {
        // The last transformed term is only about 13 or 19 here, but the
        // omitted integral is about 69 or 963. Even atol=20 is unresolved.
        for exponent in [-0.99, -0.999] {
            let exact = 1.0 / (1.0 + exponent);
            let result = tanhsinh(|x| (1.0 - x).powf(exponent), 0.0, 1.0, 20.0, 1e-12, 10);
            let actual_error = (result.integral - exact).abs();
            assert!(actual_error > 20.0, "exercise unresolved mass: {result:?}");
            assert!(!result.converged, "exponent={exponent}: {result:?}");
            assert!(
                result.error >= actual_error,
                "{result:?}, error={actual_error}"
            );
        }
    }

    #[test]
    fn tanhsinh_two_unresolved_endpoint_tails_are_additive() {
        // Each endpoint loses about 69.7 for exponent -0.99, or 964.6 for
        // -0.999. The old maximum certified atol=100 / 1000 even though
        // the total missing mass was about 138.6 / 1928.0, respectively.
        for (exponent, atol) in [(-0.99_f64, 100.0), (-0.999, 1000.0)] {
            let exact = 2.0 / (1.0 + exponent);
            for (a, b, sign) in [(1.0, 2.0, 1.0), (2.0, 1.0, -1.0)] {
                let result = tanhsinh(
                    |x| (x - 1.0).powf(exponent) + (2.0 - x).powf(exponent),
                    a,
                    b,
                    atol,
                    1e-12,
                    10,
                );
                let actual_error = (result.integral - sign * exact).abs();
                assert!(actual_error > atol, "must exercise both unresolved tails");
                assert!(!result.converged, "exponent={exponent}: {result:?}");
                assert!(result.integral.is_finite() && result.error.is_finite());
                assert!(
                    result.error >= actual_error,
                    "exponent={exponent}: {result:?}, actual_error={actual_error}"
                );
            }
        }
    }

    #[test]
    fn tanhsinh_mirrored_infinite_endpoints_share_error_estimates() {
        // Mirrored exp-sinh has its finite physical upper endpoint on the
        // negative transformed tail. Combining tail indicators must preserve
        // reflection symmetry instead of counting that endpoint twice.
        for exponent in [-0.5, -0.99] {
            let integrand = |distance: f64| (-distance).exp() * distance.powf(exponent);
            let upper = tanhsinh(|x| integrand(x - 1.0), 1.0, f64::INFINITY, 0.0, 1e-12, 10);
            let lower = tanhsinh(
                |x| integrand(-x - 1.0),
                f64::NEG_INFINITY,
                -1.0,
                0.0,
                1e-12,
                10,
            );
            assert_eq!(upper, lower, "exponent={exponent}");
        }
    }

    #[test]
    fn tanhsinh_logarithmic_endpoint_tails_do_not_claim_convergence() {
        // Substitution u=1-ln(x) gives an exact integral 1/(q-1). These
        // logarithmic tails decay slowly even in transformed coordinates;
        // a local endpoint power alone misses most of their unresolved mass.
        for q in [1.1, 1.01] {
            let exact = 1.0 / (q - 1.0);
            let result = tanhsinh(
                |x| 1.0 / (x * (1.0 - x.ln()).powf(q)),
                0.0,
                1.0,
                1.0,
                1e-12,
                10,
            );
            let actual_error = (result.integral - exact).abs();
            assert!(result.integral.is_finite() && result.error.is_finite());
            assert!(actual_error > 1.0, "exercise unresolved mass: {result:?}");
            assert!(!result.converged, "q={q}: {result:?}");
            assert!(
                result.error >= actual_error,
                "q={q}: {result:?}, error={actual_error}"
            );
        }
    }

    #[test]
    fn tanhsinh_unresolved_endpoint_tails_do_not_claim_convergence() {
        // Both integrals are exactly 2, but their singular endpoint distances
        // cannot be represented below roughly one ulp. Refinement alone used
        // to certify 1e-12 despite an actual error of about 2e-8.
        let cases: &[QuadCase] = &[
            (|x| 1.0 / (1.0 - x).sqrt(), 0.0, 1.0, 2.0),
            (|x| 1.0 / (x - 1.0).sqrt(), 1.0, 2.0, 2.0),
            (|x| 1.0 / (1.0 - x).sqrt(), 1.0, 0.0, -2.0),
        ];
        for &(f, a, b, exact) in cases {
            let result = tanhsinh(f, a, b, 0.0, 1e-12, 16);
            let actual_error = (result.integral - exact).abs();
            assert!(result.integral.is_finite() && result.error.is_finite());
            assert!(!result.converged, "unresolved tail: {result:?}");
            assert!(actual_error > 1e-10, "exercise the representability limit");
            assert!(result.error >= actual_error, "{result:?}, exact={exact}");
            assert!(result.error > 1e-12 * result.integral.abs());
        }
    }

    #[test]
    fn tanhsinh_resolvable_endpoint_tolerance_still_converges() {
        let calls = std::cell::Cell::new(0usize);
        let result = tanhsinh(
            |x| {
                assert!(x > 0.0 && x < 1.0, "must not evaluate an endpoint");
                calls.set(calls.get() + 1);
                1.0 / (1.0 - x).sqrt()
            },
            0.0,
            1.0,
            0.0,
            1e-6,
            16,
        );
        assert!(result.converged, "{result:?}");
        assert!((result.integral - 2.0).abs() <= result.error);
        assert!(result.error <= 1e-6 * result.integral.abs());
        assert_eq!(result.neval, calls.get());
        assert!(
            result.neval < 1000,
            "resolvable tolerances should stop early"
        );
    }

    #[test]
    fn tanhsinh_roundoff_does_not_masquerade_as_exact_convergence() {
        // A relative tolerance alone cannot certify a cancellation-dominated
        // integral whose true value is zero, even if two estimates coincide.
        let cancelled = tanhsinh(|x| x - 0.5, 0.0, 1.0, 0.0, 1e-12, 8);
        assert!(!cancelled.converged, "{cancelled:?}");
        assert!(cancelled.error >= 0.24 * f64::EPSILON);
        assert!(cancelled.integral.abs() <= cancelled.error);

        let too_tight = tanhsinh(|_| 1.0, 0.0, 1.0, 0.0, 1e-20, 8);
        assert!(!too_tight.converged, "{too_tight:?}");
        assert!(too_tight.error >= f64::EPSILON * too_tight.integral.abs());

        let absolute = tanhsinh(|x| x - 0.5, 0.0, 1.0, 1e-12, 1e-12, 8);
        assert!(
            absolute.converged,
            "an absolute tolerance resolves cancellation"
        );
        assert!(absolute.integral.abs() <= 1e-12);
    }

    #[test]
    fn tanhsinh_shifted_infinite_intervals_do_not_sample_singular_endpoints() {
        let upper = tanhsinh(
            |x| {
                assert!(x > 1.0, "rounded onto the singular lower endpoint");
                (-(x - 1.0)).exp() / (x - 1.0).sqrt()
            },
            1.0,
            f64::INFINITY,
            0.0,
            1e-6,
            16,
        );
        let lower = tanhsinh(
            |x| {
                assert!(x < -1.0, "rounded onto the singular upper endpoint");
                (x + 1.0).exp() / (-1.0 - x).sqrt()
            },
            f64::NEG_INFINITY,
            -1.0,
            0.0,
            1e-6,
            16,
        );
        for result in [upper, lower] {
            assert!(result.converged, "{result:?}");
            assert!((result.integral - std::f64::consts::PI.sqrt()).abs() <= 1e-6);
        }
    }

    #[test]
    fn tanhsinh_unrepresentable_nodes_or_accumulation_do_not_report_success() {
        let adjacent = tanhsinh(
            |_| panic!("adjacent bounds have no interior node"),
            1.0,
            1.0_f64.next_up(),
            0.0,
            1e-12,
            16,
        );
        assert!(!adjacent.converged);
        assert_eq!(adjacent.neval, 0);
        assert_eq!(adjacent.error, f64::INFINITY);
        let empty_nodes = tanhsinh(
            |_| panic!("no representable finite interior point"),
            f64::MAX,
            f64::INFINITY,
            0.0,
            1e-12,
            16,
        );
        assert!(!empty_nodes.converged);
        assert_eq!(empty_nodes.neval, 0);
        assert_eq!(empty_nodes.error, f64::INFINITY);
        let overflow = tanhsinh(|_| f64::MAX, 0.0, 2.0, 0.0, 1e-12, 16);
        assert!(!overflow.converged);
        assert!(overflow.integral.is_nan());
        assert_eq!(overflow.error, f64::INFINITY);
        assert!(overflow.neval < 100);
    }

    #[test]
    fn nsum_nonfinite_terms_derivatives_and_tails_stop_with_bounded_work() {
        let cases: &[fn(f64) -> f64] = &[
            |_| f64::NAN,
            |x| if x == 9.0 { f64::NAN } else { 1.0 / (x * x) },
            |x| {
                if (10.0..10.5).contains(&x) {
                    f64::NAN
                } else {
                    1.0 / (x * x)
                }
            },
        ];
        for &f in cases {
            let calls = std::cell::Cell::new(0usize);
            let result = nsum(
                |x| {
                    calls.set(calls.get() + 1);
                    assert!(calls.get() <= 1000, "invalid series kept expanding");
                    f(x)
                },
                1.0,
                f64::INFINITY,
                1.0,
                0.0,
                1e-11,
            );
            assert!(!result.converged);
            assert!(!result.sum.is_finite());
            assert_eq!(result.error, f64::INFINITY);
            assert_eq!(result.nfev, calls.get());
        }
        let finite = nsum(|_| f64::INFINITY, 1.0, 100.0, 1.0, 0.0, 1e-11);
        assert!(!finite.converged);
        assert_eq!(finite.nfev, 1);
        assert_eq!(finite.error, f64::INFINITY);
    }

    #[test]
    fn nsum_rejects_unrepresentable_finite_grids_before_evaluation() {
        for (a, b, step) in [
            (-f64::MAX, f64::MAX, 1.0),
            (0.0, 1.0, f64::from_bits(1)),
            (0.0, (1_u64 << 53) as f64, 1.0),
            (1e100, 1e100_f64.next_up(), 1.0),
            (1.0, 1.0_f64.next_up(), f64::EPSILON / 4.0),
        ] {
            let result = nsum(
                |_| panic!("invalid grid must be rejected before evaluation"),
                a,
                b,
                step,
                0.0,
                1e-11,
            );
            assert!(!result.converged, "{a}..{b} step={step}: {result:?}");
            assert!(result.sum.is_nan());
            assert_eq!(result.error, f64::INFINITY);
            assert_eq!(result.nfev, 0);
        }
        let single = nsum(|_| 7.0, f64::MAX, f64::MAX, 1.0, 0.0, 1e-11);
        assert!(single.converged);
        assert_eq!(single.sum, 7.0);
        assert_eq!(single.nfev, 1);
        let finite = nsum(|x| x * x, -2.0, 2.0, 0.5, 0.0, 1e-11);
        assert!(finite.converged);
        assert_eq!(finite.sum, 15.0);
        assert_eq!(finite.nfev, 9);
    }

    #[test]
    fn gauss_kronrod_nonfinite_samples_never_converge() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let result = gauss_kronrod_quad(|_| bad, 0.0, 1.0, QuadOptions::default());
            assert!(!result.converged, "{bad}: {result:?}");
            assert!(result.integral.is_nan());
            assert_eq!(result.error, f64::INFINITY);
            assert_eq!(result.neval, 15);
        }
    }

    #[test]
    fn gauss_kronrod_refinement_counts_every_function_evaluation() {
        let calls = std::cell::Cell::new(0usize);
        let result = gauss_kronrod_quad(
            |x| {
                calls.set(calls.get() + 1);
                (x - 0.5).abs()
            },
            0.0,
            1.0,
            QuadOptions {
                epsabs: 1e-12,
                epsrel: 1e-12,
                ..QuadOptions::default()
            },
        );
        assert!(result.converged, "{result:?}");
        assert!((result.integral - 0.25).abs() <= 1e-12);
        assert!(result.neval > 15, "the cusp must require refinement");
        assert_eq!(result.neval, calls.get());
    }

    #[test]
    fn gauss_kronrod_limit_is_a_global_subinterval_budget() {
        for limit in [1, 2, 3, 6] {
            let calls = std::cell::Cell::new(0usize);
            let result = gauss_kronrod_quad(
                |x| {
                    calls.set(calls.get() + 1);
                    (10_000.0 * x).sin()
                },
                0.0,
                1.0,
                QuadOptions {
                    epsabs: 1e-12,
                    epsrel: 0.0,
                    limit,
                },
            );
            assert!(!result.converged, "limit={limit}: {result:?}");
            assert!(result.error > 1e-12);
            assert_eq!(result.neval, calls.get());
            assert!(
                result.neval <= 15 + 30 * (limit - 1),
                "limit={limit} exceeded its panel budget: {result:?}"
            );
        }
    }

    #[test]
    fn gauss_kronrod_certifies_the_sum_of_all_subinterval_errors() {
        // The recursive implementation accepted each cusp at epsabs=1e-6
        // and returned success with estimated error 1.5723e-6 and actual
        // error 1.5814e-6. Both errors must obey the single global budget.
        let options = QuadOptions {
            epsabs: 1e-6,
            epsrel: 0.0,
            limit: 128,
        };
        let expected = 4.0 / 3.0 * ((1.0_f64 / 3.0).powf(1.5) + (2.0_f64 / 3.0).powf(1.5));
        for (a, b, sign) in [(0.0, 1.0, 1.0), (1.0, 0.0, -1.0)] {
            let result = gauss_kronrod_quad(
                |x| (x - 1.0 / 3.0).abs().sqrt() + (x - 2.0 / 3.0).abs().sqrt(),
                a,
                b,
                options,
            );
            assert!(result.converged, "{result:?}");
            assert!(result.error <= options.epsabs, "{result:?}");
            assert!((result.integral - sign * expected).abs() <= options.epsabs);
            assert!(result.neval <= 15 + 30 * (options.limit - 1));
        }
    }

    #[test]
    fn gauss_kronrod_cancellation_cannot_certify_a_relative_only_tolerance() {
        let options = QuadOptions {
            epsabs: 0.0,
            epsrel: 1e-10,
            limit: 64,
        };
        let result = gauss_kronrod_quad(f64::sin, 0.0, std::f64::consts::TAU, options);
        assert!(!result.converged, "{result:?}");
        assert!(result.error > options.epsrel * result.integral.abs());
        assert!(result.integral.abs() < 1e-12);
        assert!(result.error.is_finite());
        assert!(result.neval <= 15 + 30 * (options.limit - 1));
    }

    #[test]
    fn gauss_kronrod_preserves_representable_integrals_at_extreme_scales() {
        let options = QuadOptions {
            epsabs: 0.0,
            epsrel: 1e-12,
            ..QuadOptions::default()
        };
        for scale in [1e-300, 1e-200, 1e200, 5e307, 1e308, -1e308] {
            let result = gauss_kronrod_quad(|_| scale, 0.0, 1.0, options);
            assert!(result.converged, "scale={scale}: {result:?}");
            assert!((result.integral / scale - 1.0).abs() <= 4.0 * f64::EPSILON);
            assert!(result.error.is_finite() && result.error > 0.0);
            assert_eq!(result.neval, 15);
        }

        let result = gauss_kronrod_quad(
            |x| {
                assert!(x.is_finite() && x > -1e308 && x < 1e308);
                1e-308
            },
            -1e308,
            1e308,
            options,
        );
        assert!(result.converged, "{result:?}");
        assert!((result.integral - 2.0).abs() < 1e-14);

        let width = 2.0 * f64::from_bits(1);
        let result = gauss_kronrod_quad(|_| 1e300, 0.0, width, options);
        assert!(result.converged, "{result:?}");
        assert!((result.integral / (width * 1e300) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn gauss_kronrod_retains_small_integrals_after_refining_large_coarse_estimates() {
        for sign in [-1.0, 1.0] {
            // Changing a function at one point does not change its integral.
            // The coarse rule samples that point, while both refined panels
            // see the small constant. A single replacement delta loses it.
            let result = gauss_kronrod_quad(
                |x| sign * if x == 0.5 { 1e300 } else { 1e-300 },
                0.0,
                1.0,
                QuadOptions {
                    epsabs: 0.0,
                    epsrel: 1e-8,
                    limit: 2,
                },
            );
            assert!(result.converged, "{result:?}");
            assert_eq!(result.neval, 45);
            assert!((result.integral / (sign * 1e-300) - 1.0).abs() <= 4.0 * f64::EPSILON);
            assert!(result.error > 0.0);
            assert!(result.error <= 1e-8 * result.integral.abs());
        }
    }

    #[test]
    fn gauss_kronrod_handles_empty_and_unsampleable_intervals_before_callbacks() {
        let empty = gauss_kronrod_quad(
            |_| panic!("an empty interval needs no evaluation"),
            3.0,
            3.0,
            QuadOptions::default(),
        );
        assert!(empty.converged);
        assert_eq!(empty.integral, 0.0);
        assert_eq!(empty.error, 0.0);
        assert_eq!(empty.neval, 0);

        for (a, b) in [(1.0, 1.0_f64.next_up()), (1.0_f64.next_up(), 1.0)] {
            let result = gauss_kronrod_quad(
                |_| panic!("no representable interior point"),
                a,
                b,
                QuadOptions::default(),
            );
            assert!(!result.converged);
            assert!(result.integral.is_nan());
            assert_eq!(result.error, f64::INFINITY);
            assert_eq!(result.neval, 0);
        }
        let no_budget = gauss_kronrod_quad(
            |_| panic!("a zero panel budget must be rejected before evaluation"),
            0.0,
            1.0,
            QuadOptions {
                limit: 0,
                ..QuadOptions::default()
            },
        );
        assert!(!no_budget.converged);
        assert_eq!(no_budget.neval, 0);
    }

    #[test]
    fn gauss_kronrod_refines_endpoint_singularities_without_sampling_bounds() {
        let options = QuadOptions {
            epsabs: 1e-8,
            epsrel: 0.0,
            limit: 256,
        };
        let result = gauss_kronrod_quad(
            |x| {
                assert!(x > 0.0 && x < 1.0);
                1.0 / x.sqrt()
            },
            0.0,
            1.0,
            options,
        );
        assert!(result.converged, "{result:?}");
        assert!(result.error <= options.epsabs);
        assert!((result.integral - 2.0).abs() <= options.epsabs);
        assert!(result.neval <= 15 + 30 * (options.limit - 1));
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
    fn simpson_sampled_quadratics_are_invariant_to_coordinate_units() {
        let grid = [0.0_f64, 0.25, 0.75, 1.5, 2.0, 3.0];
        for scale in [1e-200_f64, 1e-20, 1.0, 1e20, 1e200] {
            for n in 3..=grid.len() {
                let x: Vec<_> = grid[..n].iter().map(|t| scale * t).collect();
                // Normalize y inversely so every integral remains O(1), even
                // when squared grid spacings underflow or overflow.
                let y: Vec<_> = grid[..n].iter().map(|t| t * t / scale).collect();
                let expected = grid[n - 1].powi(3) / 3.0;
                for (name, actual) in [
                    ("simpson", simpson(&y, &x).unwrap().integral),
                    ("simpson_irregular", simpson_irregular(&y, &x)),
                ] {
                    assert!(
                        (actual - expected).abs() <= 2e-13 * expected,
                        "{name}, scale={scale:e}, n={n}: {actual} != {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn simpson_nearly_retracing_grids_preserve_small_spans() {
        let delta = 2.0_f64.powi(-50);
        let grid = [0.0, 3.0, delta];
        for scale in [2.0_f64.powi(-500), 1.0, 2.0_f64.powi(500)] {
            for degree in 0..=2 {
                for sign in [-1.0, 1.0] {
                    for reversed in [false, true] {
                        let mut x: Vec<_> = grid.iter().map(|t| sign * scale * t).collect();
                        let mut y: Vec<_> = grid.iter().map(|t| t.powi(degree) / scale).collect();
                        let mut expected = sign * delta.powi(degree + 1) / f64::from(degree + 1);
                        if reversed {
                            x.reverse();
                            y.reverse();
                            expected = -expected;
                        }
                        // Simpson is exact for these polynomials. The small
                        // integral must retain relative accuracy even though
                        // the path almost returns to its starting point.
                        for (name, actual) in [
                            ("simpson", simpson(&y, &x).unwrap().integral),
                            ("simpson_irregular", simpson_irregular(&y, &x)),
                        ] {
                            assert!(
                                (actual / expected - 1.0).abs() <= 16.0 * f64::EPSILON,
                                "{name}, degree={degree}, scale={scale:e}, sign={sign}, \
                                 reversed={reversed}: {actual:e} != {expected:e}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn simpson_cartwright_handles_nearly_retracing_final_interval() {
        let delta = 2.0_f64.powi(-50);
        for scale in [2.0_f64.powi(-500), 1.0, 2.0_f64.powi(500)] {
            for sign in [-1.0, 1.0] {
                let x = [-sign * scale, 0.0, sign * scale * 3.0, sign * scale * delta];
                let y = [0.0, 0.0, 0.0, delta / scale];
                // The first panel contributes zero. Integrating the last
                // ordinate's Lagrange basis on the final interval gives this
                // expression without subtracting large nearly equal terms.
                let expected = sign * (-1.5 - delta / 2.0 + delta * delta / 3.0);
                for (name, actual) in [
                    ("simpson", simpson(&y, &x).unwrap().integral),
                    ("simpson_irregular", simpson_irregular(&y, &x)),
                ] {
                    assert!(
                        (actual / expected - 1.0).abs() <= 16.0 * f64::EPSILON,
                        "{name}, scale={scale:e}, sign={sign}: {actual:e} != {expected:e}"
                    );
                }
            }
        }
    }

    #[test]
    fn cumulative_simpson_sampled_quadratics_are_invariant_to_coordinate_units() {
        let grid = [0.0_f64, 0.25, 0.75, 1.5, 2.0, 3.0];
        for scale in [1e-200_f64, 1e-20, 1.0, 1e20, 1e200] {
            for n in 3..=grid.len() {
                let x: Vec<_> = grid[..n].iter().map(|t| scale * t).collect();
                let y: Vec<_> = grid[..n].iter().map(|t| t * t / scale).collect();
                let actual = cumulative_simpson(&y, &x).unwrap();
                assert_eq!(actual.len(), n - 1);
                for (i, value) in actual.iter().enumerate() {
                    let expected = grid[i + 1].powi(3) / 3.0;
                    assert!(
                        (value - expected).abs() <= 2e-13 * expected,
                        "cumulative_simpson, scale={scale:e}, n={n}, i={i}: {value} != {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn simpson_irregular_even_samples_use_the_cartwright_correction() {
        for grid in [
            vec![0.0_f64, 1.0, 2.0, 3.0],
            vec![0.0, 0.2, 0.9, 1.7, 2.1, 3.0],
            vec![3.0, 2.0, 1.0, 0.0],
        ] {
            let y: Vec<_> = grid.iter().map(|x| x * x).collect();
            let expected = (grid[grid.len() - 1].powi(3) - grid[0].powi(3)) / 3.0;
            let actual = simpson_irregular(&y, &grid);
            assert!(
                (actual - expected).abs() < 1e-13,
                "grid={grid:?}: {actual} != {expected}"
            );
        }
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

    // br-szq1n.7: `converged` must be false when the panel budget stops refinement
    // while the error estimate is still above tolerance (it used to be `true`),
    // and true when the tolerance test passes.
    #[test]
    fn gauss_kronrod_converged_flag_reflects_the_tolerance_test() {
        let singular = |x: f64| (x - 1.0 / 3.0).abs().powf(-0.9);
        let shallow = gauss_kronrod_quad(
            singular,
            0.0,
            1.0,
            QuadOptions {
                limit: 3,
                ..QuadOptions::default()
            },
        );
        assert!(
            !shallow.converged,
            "a budget of 3 panels cannot resolve |x-1/3|^-0.9 (error {:e})",
            shallow.error
        );
        let smooth = gauss_kronrod_quad(|x| x.exp(), 0.0, 1.0, QuadOptions::default());
        assert!(smooth.converged, "exp on [0,1] meets the default tolerance");
    }

    // br-szq1n.7: every leaf of ∫₀^{2π} sin passes its own epsrel test (each half is ±2), but the
    // total is ~1e-16 and its error cannot meet epsrel·|total|. SciPy flags this (ier != 0,
    // roundoff); the flag used to be the AND of the leaves, i.e. true.
    #[test]
    fn quad_converged_is_decided_on_the_total_not_the_leaves() {
        let rel_only = QuadOptions {
            epsabs: 0.0,
            epsrel: 1e-10,
            ..QuadOptions::default()
        };
        let cancelling = quad(f64::sin, 0.0, 2.0 * std::f64::consts::PI, rel_only).unwrap();
        let bound = rel_only.epsrel * cancelling.integral.abs();
        println!(
            "sin over [0,2pi]: I={:e} err={:e} bound={bound:e}",
            cancelling.integral, cancelling.error
        );
        assert!(
            !cancelling.converged,
            "error {:e} exceeds epsrel*|I| = {bound:e}",
            cancelling.error
        );
        let one_sign = quad(f64::sin, 0.0, std::f64::consts::PI, rel_only).unwrap();
        assert!(one_sign.converged, "sin over [0,pi] = 2 meets epsrel 1e-10");
        let vec = quad_vec(|x| vec![x.sin()], 0.0, 2.0 * std::f64::consts::PI, rel_only).unwrap();
        assert!(!vec.converged, "quad_vec uses the same total test");
    }

    // br-szq1n.7: a zero-width test of |a-b| < EPSILON returned 0 for ∫₀^{1e-17} 1e20 dx = 1000.
    // SciPy only short-circuits a == b exactly.
    #[test]
    fn quad_integrates_a_tiny_but_nonzero_interval() {
        let tiny = quad(|_| 1e20, 0.0, 1e-17, QuadOptions::default()).unwrap();
        assert!(
            (tiny.integral - 1000.0).abs() < 1e-9,
            "got {} (SciPy 1000.0000000000001)",
            tiny.integral
        );
        let empty = quad(|_| 1e20, 0.5, 0.5, QuadOptions::default()).unwrap();
        assert_eq!(empty.integral, 0.0);
        assert!(empty.converged);
    }

    // br-szq1n.7: nquad used to return error 0.0 and converged true whatever the nested quads did.
    // SciPy: nquad(|x-1/3|^-0.9, [(0,1),(0,1)], opts={'limit':3}) -> (8.2256, abserr 4.70), the
    // abserr being the MAX over every nested quad call. The singular variable here is the
    // innermost one (SciPy's first argument), so the outer integrand is constant and only the
    // inner calls carry the error.
    #[test]
    fn nquad_reports_the_largest_nested_error_and_any_unconverged_inner_quad() {
        let shallow = QuadOptions {
            limit: 3,
            ..QuadOptions::default()
        };
        let singular = nquad(
            |args| (args[1] - 1.0 / 3.0).abs().powf(-0.9),
            &[(0.0, 1.0), (0.0, 1.0)],
            shallow,
        )
        .unwrap();
        println!(
            "nquad singular: I={} err={:e} converged={}",
            singular.integral, singular.error, singular.converged
        );
        assert!(!singular.converged, "inner quad cannot converge at limit 3");
        assert!(singular.error > 1e-3, "outer abserr is reported, not 0.0");
        let smooth = nquad(
            |args| args[0] * args[1],
            &[(0.0, 1.0), (0.0, 1.0)],
            QuadOptions::default(),
        )
        .unwrap();
        assert!(smooth.converged);
        assert!(smooth.error < 1e-12, "smooth abserr {:e}", smooth.error);

        // dblquad and tplquad AND the inner flags in the same way.
        let d = dblquad(
            |y, _x| (y - 1.0 / 3.0).abs().powf(-0.9),
            0.0,
            1.0,
            |_| 0.0,
            |_| 1.0,
            DblquadOptions {
                limit: 3,
                ..DblquadOptions::default()
            },
        )
        .unwrap();
        assert!(
            !d.converged,
            "dblquad inner quad cannot converge at limit 3"
        );
        assert!(d.error >= singular.error, "dblquad dropped the inner error");
        let t = tplquad(
            |z, _y, _x| (z - 1.0 / 3.0).abs().powf(-0.9),
            0.0,
            1.0,
            |_| 0.0,
            |_| 1.0,
            |_, _| 0.0,
            |_, _| 1.0,
            DblquadOptions {
                limit: 3,
                ..DblquadOptions::default()
            },
        )
        .unwrap();
        assert!(
            !t.converged,
            "tplquad innermost quad cannot converge at limit 3"
        );
        assert!(t.error >= singular.error, "tplquad dropped the inner error");
    }

    #[test]
    fn nested_quad_preserves_inner_errors_even_when_every_level_converges() {
        let options = DblquadOptions {
            epsabs: 1e-8,
            epsrel: 1e-8,
            limit: 50,
        };
        let one = quad(
            |x| (100.0 * x).sin(),
            0.0,
            1.0,
            QuadOptions {
                epsabs: options.epsabs,
                epsrel: options.epsrel,
                limit: options.limit,
            },
        )
        .unwrap();
        // The oscillatory integral's error is about 9.46e-11. Integrating
        // that constant over an outer unit interval reports only 1.53e-17.
        // Every level converges, so AND-ing convergence flags cannot detect
        // the six orders of magnitude lost by reporting just the outer error.
        assert!(one.converged && one.error > 1e-12);
        let exact = (1.0 - 100.0_f64.cos()) / 100.0;
        for (a, b, sign) in [(0.0, 1.0, 1.0), (1.0, 0.0, -1.0)] {
            let double =
                dblquad(|y, _| (100.0 * y).sin(), a, b, |_| 0.0, |_| 1.0, options).unwrap();
            assert!(double.converged);
            assert!((double.integral - sign * exact).abs() < 1e-12);
            assert_eq!(double.error.to_bits(), one.error.to_bits());
            for hard_middle in [false, true] {
                let triple = tplquad(
                    |z, y, _| (100.0 * if hard_middle { y } else { z }).sin(),
                    a,
                    b,
                    |_| 0.0,
                    |_| 1.0,
                    |_, _| 0.0,
                    |_, _| 1.0,
                    options,
                )
                .unwrap();
                assert!(triple.converged);
                assert!((triple.integral - sign * exact).abs() < 1e-12);
                assert!(triple.error > 1e-12);
                assert!((triple.error - one.error).abs() < 1e-13);
            }
        }
    }

    #[test]
    fn nested_quad_propagates_invalid_callable_bounds_without_retrying() {
        let bound_calls = std::cell::Cell::new(0);
        let integrand_calls = std::cell::Cell::new(0);
        let double = dblquad(
            |_, _| {
                integrand_calls.set(integrand_calls.get() + 1);
                1.0
            },
            0.0,
            1.0,
            |_| {
                bound_calls.set(bound_calls.get() + 1);
                f64::NAN
            },
            |_| 1.0,
            DblquadOptions::default(),
        );
        assert!(matches!(
            double,
            Err(IntegrateValidationError::QuadInvalidBounds { .. })
        ));
        assert_eq!(bound_calls.get(), 1);
        assert_eq!(integrand_calls.get(), 0);

        for bad_middle in [false, true] {
            bound_calls.set(0);
            let triple = tplquad(
                |_, _, _| {
                    integrand_calls.set(integrand_calls.get() + 1);
                    1.0
                },
                0.0,
                1.0,
                |_| {
                    if bad_middle {
                        bound_calls.set(bound_calls.get() + 1);
                        f64::NAN
                    } else {
                        0.0
                    }
                },
                |_| 1.0,
                |_, _| {
                    bound_calls.set(bound_calls.get() + 1);
                    f64::NAN
                },
                |_, _| 1.0,
                DblquadOptions::default(),
            );
            assert!(matches!(
                triple,
                Err(IntegrateValidationError::QuadInvalidBounds { .. })
            ));
            assert_eq!(bound_calls.get(), 1);
            assert_eq!(integrand_calls.get(), 0);
        }
    }

    // br-szq1n.7: a non-finite finite-range partial sum is SciPy's status -3, not converged.
    #[test]
    fn nsum_nonfinite_finite_range_is_not_converged() {
        let bad = nsum(
            |n| if n == 3.0 { f64::NAN } else { n },
            1.0,
            5.0,
            1.0,
            0.0,
            0.0,
        );
        assert!(bad.sum.is_nan());
        assert!(!bad.converged);
        let good = nsum(|n| n, 1.0, 5.0, 1.0, 0.0, 0.0);
        assert_eq!(good.sum, 15.0);
        assert!(good.converged);
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

    /// A smoothly non-uniform grid on `[0, x_scale]`. The integrand is x³
    /// because Simpson's rule is EXACT for cubics up to grid effects while the
    /// trapezoid rule is not, so a silent substitution of one rule for the other
    /// shows up as five orders of magnitude rather than as a rounding argument.
    fn bshyq_nonuniform_grid(x_scale: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1) as f64;
                (t + 0.3 * (3.0 * std::f64::consts::PI * t).sin() / (3.0 * std::f64::consts::PI))
                    * x_scale
            })
            .collect()
    }

    /// frankenscipy-bshyq. `simpson_nonuniform_odd` guarded the Simpson weights
    /// with `h_prod.abs() < f64::EPSILON * 1e-10` (2.22e-26) and substituted the
    /// TRAPEZOID rule when it fired. But `h_prod = h0·h1` is a product of two
    /// grid spacings, so it carries units of x²: the test asks how big the
    /// integration interval is, not whether two points are coincident, and a
    /// legitimately fine grid quietly gets a much worse answer with no error.
    ///
    /// Measured live on scipy 1.17.1 / numpy 2.4.3 (harness
    /// `scripts/scipy_scale_probe.py`, section "simpson on a fine grid"), on
    /// this fixture: `scipy.integrate.simpson` holds a relative error of
    /// 5.451e-10 at every x-scale from 1e-0 down to 1e-14, where h0·h1 is
    /// 1.225e-33, while the trapezoid answer is 2.771e-05 — about fifty
    /// thousand times worse. SciPy's guard is an exact-zero denominator test,
    /// `np.true_divide(..., where=den != 0)`, not a floor.
    #[test]
    fn simpson_keeps_using_simpsons_rule_on_a_finely_spaced_grid() {
        // Each routine is measured against ITS OWN accuracy on the same grid at
        // scale 1, not against an absolute bar. Rescaling x rescales the exact
        // answer and every panel identically, so a routine that keeps applying
        // the same rule keeps the same RELATIVE error — which is precisely what
        // SciPy does here (flat 5.451e-10 from 1e-0 to 1e-14). This phrasing
        // cannot be confused by one routine being inherently less accurate than
        // the other; it only fails if the rule CHANGES with the scale.
        let reference = |x_scale: f64| {
            let x = bshyq_nonuniform_grid(x_scale, 201);
            let y: Vec<f64> = x.iter().map(|value| value.powi(3)).collect();
            let exact = (x[x.len() - 1].powi(4) - x[0].powi(4)) / 4.0;
            let simpson_error =
                (simpson(&y, &x).expect("simpson").integral - exact).abs() / exact.abs();
            let irregular_error = (simpson_irregular(&y, &x) - exact).abs() / exact.abs();
            (simpson_error, irregular_error)
        };

        let (simpson_baseline, irregular_baseline) = reference(1.0);
        for exponent in [5_i32, 10, 12, 14] {
            let (simpson_error, irregular_error) = reference(10.0_f64.powi(-exponent));
            // BOTH public entries, because they carry the same defect with two
            // different constants and fixing one would leave the other wrong —
            // the lesson frankenscipy-kwi99 paid for with eig/eigvals.
            // `simpson` guards h0·h1 against 2.22e-26 and degrades from x-scale
            // 1e-11; `simpson_irregular` guards each spacing directly against
            // 1e-15 and degrades from 1e-13.
            for (name, error, baseline) in [
                ("simpson", simpson_error, simpson_baseline),
                ("simpson_irregular", irregular_error, irregular_baseline),
            ] {
                assert!(
                    error <= baseline * 10.0 + 1e-15,
                    "{name} at x-scale 1e-{exponent}: relative error {error:.3e} against \
                     {baseline:.3e} for the identical grid at scale 1. Scaling x does not \
                     change the geometry, so this is the quadrature rule being silently \
                     substituted — SciPy holds 5.451e-10 across this whole range"
                );
            }
        }
    }

    /// Negative case for frankenscipy-bshyq: the branch exists for COINCIDENT
    /// points, where the Simpson weights divide by zero, and it must still fire
    /// there. A repeated abscissa is the only input that makes `h0·h1` zero.
    #[test]
    fn simpson_still_falls_back_when_two_points_coincide() {
        // x[1] == x[2], so the second panel has zero width.
        let x = [0.0, 0.5, 0.5, 1.5, 2.5];
        let y = [0.0, 1.0, 1.0, 3.0, 5.0];
        let result = simpson(&y, &x).expect("simpson").integral;
        assert!(
            result.is_finite(),
            "coincident points must take the trapezoid fallback, got {result}"
        );

        // And the same at a tiny scale, where the old absolute floor would have
        // caught it by accident of magnitude rather than by coincidence.
        let scale = 1.0e-12;
        let x: Vec<f64> = x.iter().map(|value| value * scale).collect();
        let y: Vec<f64> = y.iter().map(|value| value * scale).collect();
        let scaled = simpson(&y, &x).expect("simpson").integral;
        assert!(
            scaled.is_finite(),
            "coincident points at 1e-12 must still be finite, got {scaled}"
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
