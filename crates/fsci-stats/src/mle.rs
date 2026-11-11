//! Fallible maximum-likelihood fitting with explicit parameter constraints.
//!
//! [`fit_mle`] and [`fit_mle_censored`] minimize the negative log likelihood
//! using Nelder–Mead. The
//! model factory defines the parameterization, including any location and scale
//! parameters; [`MleParameter::Fixed`] holds a parameter exactly constant. This
//! also supports distributions whose concrete type has fixed location or scale.
//! No method-of-moments estimate is returned as a successful MLE.

use fsci_opt::{MinimizeMethodOptions, MinimizeOptions, OptimizeMethod, minimize};

use crate::{
    CensoredData, ContinuousDistribution, FitError, RuntimeMode, active_stats_mode,
    hardened_max_sample_len,
};

/// Borrow both representations so censoring never mutates the caller's data or
/// allocates an observation vector on every likelihood evaluation.
#[derive(Clone, Copy)]
enum MleSample<'a> {
    Exact(&'a [f64]),
    Censored(&'a CensoredData),
}

impl MleSample<'_> {
    fn len(self) -> usize {
        match self {
            Self::Exact(data) => data.len(),
            Self::Censored(data) => data.len(),
        }
    }

    fn validate(self) -> Result<(), FitError> {
        match self {
            Self::Exact(data) => crate::validate_finite_fit_data(data, 1, "maximum likelihood"),
            Self::Censored(data) => {
                if data.is_empty() {
                    return Err(FitError::InsufficientData {
                        required: 1,
                        actual: 0,
                    });
                }
                for (name, values) in [
                    ("uncensored", &data.uncensored),
                    ("left-censored", &data.left),
                    ("right-censored", &data.right),
                ] {
                    if values.iter().any(|value| !value.is_finite()) {
                        return Err(FitError::UnsupportedData(format!(
                            "{name} MLE observations must be finite"
                        )));
                    }
                }
                for (index, &(lower, upper)) in data.interval.iter().enumerate() {
                    if lower.is_nan()
                        || upper.is_nan()
                        || lower > upper
                        || (!lower.is_finite() && !upper.is_finite())
                    {
                        return Err(FitError::UnsupportedData(format!(
                            "MLE interval {index} must have ordered bounds and at least one finite endpoint"
                        )));
                    }
                }
                Ok(())
            }
        }
    }

    fn has_exact(self) -> bool {
        match self {
            Self::Exact(data) => !data.is_empty(),
            Self::Censored(data) => {
                !data.uncensored.is_empty()
                    || data.interval.iter().any(|&(lower, upper)| lower == upper)
            }
        }
    }

    fn log_terms<'a, D: ContinuousDistribution>(
        &'a self,
        distribution: &'a D,
    ) -> impl Iterator<Item = f64> + 'a {
        let (exact, left, right, interval): (&[f64], &[f64], &[f64], &[(f64, f64)]) =
            match self {
                Self::Exact(data) => (*data, &[], &[], &[]),
                Self::Censored(data) => {
                    (&data.uncensored, &data.left, &data.right, &data.interval)
                }
            };
        exact
            .iter()
            .map(move |&x| distribution.logpdf(x))
            .chain(
                left.iter()
                    .map(move |&x| checked_log_probability(distribution.logcdf(x))),
            )
            .chain(
                right
                    .iter()
                    .map(move |&x| checked_log_probability(distribution.logsf(x))),
            )
            .chain(interval.iter().map(move |&(lower, upper)| {
                if lower == upper {
                    distribution.logpdf(lower)
                } else if lower == f64::NEG_INFINITY {
                    checked_log_probability(distribution.logcdf(upper))
                } else if upper == f64::INFINITY {
                    checked_log_probability(distribution.logsf(lower))
                } else {
                    interval_log_probability(distribution, lower, upper)
                }
            }))
    }
}

fn checked_log_probability(value: f64) -> f64 {
    if value > 0.0 { f64::NAN } else { value }
}

/// log(exp(larger) - exp(smaller)), without subtracting rounded probabilities.
fn log_probability_difference(larger: f64, smaller: f64) -> f64 {
    if larger.is_nan() || smaller.is_nan() || larger > 0.0 || smaller > larger {
        return f64::NAN;
    }
    if larger == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    larger + (-(smaller - larger).exp_m1()).ln()
}

fn interval_log_probability<D: ContinuousDistribution>(
    distribution: &D,
    lower: f64,
    upper: f64,
) -> f64 {
    // Use CDF differences in the left tail and SF differences in the right
    // tail. For a normal interval [40, 41], both CDFs round to one and both
    // SFs underflow to zero, but the log survival probabilities remain useful.
    let log_cdf_upper = checked_log_probability(distribution.logcdf(upper));
    let log_sf_lower = checked_log_probability(distribution.logsf(lower));
    if log_cdf_upper.is_nan() || log_sf_lower.is_nan() {
        return f64::NAN;
    }
    if log_cdf_upper < log_sf_lower {
        log_probability_difference(log_cdf_upper, distribution.logcdf(lower))
    } else {
        log_probability_difference(log_sf_lower, distribution.logsf(upper))
    }
}

pub(crate) fn validate_positive_sample(data: &[f64], family: &str) -> Result<(), FitError> {
    crate::validate_finite_fit_data(data, 2, family)?;
    if data.iter().any(|&x| x <= 0.0) {
        return Err(FitError::UnsupportedData(format!(
            "{family} MLE with fixed location zero requires positive observations"
        )));
    }
    if data.iter().all(|&x| x == data[0]) {
        return Err(FitError::NonConvergent(format!(
            "{family} MLE requires a nonconstant sample"
        )));
    }
    Ok(())
}

fn validate_shape_sample(
    data: &[f64],
    minimum_sample: usize,
    family: &str,
) -> Result<(), FitError> {
    if active_stats_mode() == RuntimeMode::Hardened && data.len() > hardened_max_sample_len() {
        return Err(FitError::ResourceExhausted {
            detail: "Hardened mode fit sample length exceeds HARDENED_MAX_SAMPLE_LEN",
        });
    }
    crate::validate_finite_fit_data(data, minimum_sample, family)
}

/// With location zero and scale one, the chi/chi-squared likelihood has a
/// unique maximum at psi(df / 2) = mean(log(X^power)) - log(2). Computing its
/// sufficient statistic in log space also avoids squaring large chi samples.
pub(crate) fn chi_shape_mle(
    data: &[f64],
    power: f64,
    minimum_sample: usize,
    family: &str,
) -> Result<f64, FitError> {
    validate_shape_sample(data, minimum_sample, family)?;
    if data.iter().any(|&x| x <= 0.0) {
        return Err(FitError::UnsupportedData(format!(
            "{family} MLE with fixed location zero requires strictly positive observations"
        )));
    }
    let mut mean_log = 0.0;
    let mut correction = 0.0;
    let n = data.len() as f64;
    for &x in data {
        let term = x.ln() / n - correction;
        let next = mean_log + term;
        correction = (next - mean_log) - term;
        mean_log = next;
    }
    let target = power * mean_log - std::f64::consts::LN_2;
    // psi is strictly increasing on the positive axis. Start above the
    // asymptotic inverse and retain a bracket instead of trusting an
    // unconstrained Newton step near df=0.
    let mut upper = (target.exp() + 1.0).max(1.0);
    if !upper.is_finite() {
        return Err(FitError::NonConvergent(format!(
            "{family} MLE degrees of freedom exceed the finite parameter range"
        )));
    }
    while fsci_special::digamma_scalar(upper) < target {
        upper *= 2.0;
        if !upper.is_finite() {
            return Err(FitError::NonConvergent(format!(
                "{family} MLE degrees of freedom exceed the finite parameter range"
            )));
        }
    }
    let mut lower = 0.0;
    for _ in 0..100 {
        let middle = lower + 0.5 * (upper - lower);
        if middle == lower || middle == upper {
            break;
        }
        if fsci_special::digamma_scalar(middle) < target {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    let df = lower + upper;
    if df.is_finite() && df > 0.0 {
        Ok(df)
    } else {
        Err(FitError::NonConvergent(format!(
            "{family} MLE degrees of freedom exceed the finite parameter range"
        )))
    }
}

/// psi(a + 1/2) - psi(a), evaluated without subtracting nearly equal
/// digammas for concentrated symmetric beta distributions. Shift to a >= 16
/// by recurrence, then use the Bernoulli expansion through a^-12.
fn half_digamma_difference(mut a: f64) -> f64 {
    let mut correction = 0.0;
    while a < 16.0 {
        correction += (0.5 / a) / (a + 0.5);
        a += 1.0;
    }
    let inverse = a.recip();
    let square = inverse * inverse;
    correction
        + 0.5 * inverse
        + square
            * (0.125
                + square
                    * (-1.0 / 64.0
                        + square
                            * (1.0 / 128.0
                                + square
                                    * (-17.0 / 2048.0
                                        + square * (31.0 / 2048.0 - square * 691.0 / 16384.0)))))
}

pub(crate) fn log_one_minus_square(x: f64) -> f64 {
    if x.abs() < 0.5 {
        // Separate logarithms cancel their leading ±x terms near zero.
        (-x * x).ln_1p()
    } else {
        // Here 1-x*x would instead lose relative accuracy near |x|=1.
        (-x.abs()).ln_1p() + x.abs().ln_1p()
    }
}

/// Shape-only R-distribution MLE. The score is strictly decreasing, and
/// 1/(2a) < psi(a+1/2)-psi(a) < 1/a supplies a finite root bracket.
pub(crate) fn rdist_shape_mle(data: &[f64]) -> Result<f64, FitError> {
    validate_shape_sample(data, 2, "RDist")?;
    if data.iter().any(|&x| x.abs() >= 1.0) {
        return Err(FitError::UnsupportedData(
            "RDist MLE requires observations strictly inside (-1, 1)".into(),
        ));
    }
    let mut target = 0.0;
    let mut correction = 0.0;
    let n = data.len() as f64;
    for &x in data {
        let term = -log_one_minus_square(x) / n - correction;
        let next = target + term;
        correction = (next - target) - term;
        target = next;
    }
    let mut upper = target.recip();
    if !upper.is_finite() || target <= 0.0 {
        return Err(FitError::NonConvergent(
            "RDist likelihood has no finite representable shape maximum".into(),
        ));
    }
    let mut lower = 0.5 * upper;
    for _ in 0..100 {
        let middle = lower + 0.5 * (upper - lower);
        if middle == lower || middle == upper {
            break;
        }
        if half_digamma_difference(middle) > target {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    let shape = lower + upper;
    if shape.is_finite() && shape > 0.0 {
        Ok(shape)
    } else {
        Err(FitError::NonConvergent(
            "RDist MLE shape exceeds the finite parameter range".into(),
        ))
    }
}

/// One parameter of a likelihood model, in the order expected by its factory.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MleParameter {
    /// An unconstrained finite parameter, starting at this value.
    Free(f64),
    /// A strictly positive parameter, optimized in logarithmic coordinates.
    Positive(f64),
    /// A parameter strictly greater than `lower`, optimized as `ln(x - lower)`.
    Above { initial: f64, lower: f64 },
    /// A parameter in a closed interval. Infinite endpoints are allowed.
    Bounded {
        initial: f64,
        lower: f64,
        upper: f64,
    },
    /// A parameter held at exactly this finite value, without optimization.
    Fixed(f64),
}

impl MleParameter {
    fn initial(self) -> f64 {
        match self {
            Self::Free(x) | Self::Positive(x) | Self::Fixed(x) => x,
            Self::Above { initial, .. } | Self::Bounded { initial, .. } => initial,
        }
    }

    fn validate(self) -> bool {
        if !self.initial().is_finite() {
            return false;
        }
        match self {
            Self::Free(_) | Self::Fixed(_) => true,
            Self::Positive(x) => x > 0.0,
            Self::Above { initial, lower } => {
                lower.is_finite() && initial > lower && (initial - lower).is_finite()
            }
            Self::Bounded {
                initial,
                lower,
                upper,
            } => lower < upper && initial >= lower && initial <= upper,
        }
    }

    fn encode(self) -> f64 {
        match self {
            Self::Positive(x) => x.ln(),
            Self::Above { initial, lower } => (initial - lower).ln(),
            _ => self.initial(),
        }
    }

    fn decode(self, value: f64) -> Option<f64> {
        let decoded = match self {
            Self::Positive(_) => value.exp(),
            Self::Above { lower, .. } => lower + value.exp(),
            Self::Fixed(x) => x,
            _ => value,
        };
        let valid = decoded.is_finite()
            && match self {
                Self::Positive(_) => decoded > 0.0,
                Self::Above { lower, .. } => decoded > lower,
                Self::Bounded { lower, upper, .. } => decoded >= lower && decoded <= upper,
                _ => true,
            };
        valid.then_some(decoded)
    }
}

/// Budgets and stopping tolerances for [`fit_mle`] and [`fit_mle_censored`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MleOptions {
    pub max_iterations: usize,
    pub max_evaluations: usize,
    /// Absolute simplex tolerance in optimizer coordinates. Positive parameters
    /// use log coordinates, so this is approximately a relative tolerance there.
    pub parameter_tolerance: f64,
    /// Absolute simplex tolerance on the *mean* negative log likelihood.
    pub likelihood_tolerance: f64,
}

impl Default for MleOptions {
    fn default() -> Self {
        Self {
            max_iterations: 2_000,
            max_evaluations: 8_000,
            parameter_tolerance: 1.0e-8,
            likelihood_tolerance: 1.0e-10,
        }
    }
}

/// A converged maximum-likelihood fit and its unpenalized likelihood.
///
/// Like SciPy's local fitting optimizer, this does not certify a global maximum.
/// `parameters` includes fixed values in the original factory order.
#[derive(Debug)]
pub struct MleResult<D> {
    pub distribution: D,
    pub parameters: Vec<f64>,
    pub negative_log_likelihood: f64,
    pub iterations: usize,
    pub evaluations: usize,
}

/// Fit a continuous likelihood model without panicking on invalid data or a
/// failed optimization.
///
/// `make_distribution` receives parameters in the supplied order and must return
/// `None` for combinations outside the model's domain. It must not call a
/// panicking constructor on invalid combinations. This allows coupled domains
/// as well as the scalar bounds in [`MleParameter`]. A large finite penalty guides
/// the search past trial distributions with observations outside their support;
/// a successful result must have a finite, unpenalized log density at *every*
/// observation. Budget exhaustion returns [`FitError::NonConvergent`].
///
/// Location and scale are never introduced or fitted implicitly. For example,
/// `GammaDist` below has location zero and fits its shape while fixing scale:
///
/// ```
/// use fsci_stats::{GammaDist, MleOptions, MleParameter, fit_mle};
/// let data = [0.2, 0.4, 1.1, 1.8, 3.0, 4.2];
/// let result = fit_mle(
///     &data,
///     &[MleParameter::Positive(1.0), MleParameter::Fixed(2.0)],
///     |p| Some(GammaDist { a: p[0], scale: p[1] }),
///     MleOptions::default(),
/// ).unwrap();
/// assert_eq!(result.distribution.scale, 2.0);
/// ```
pub fn fit_mle<D, F>(
    data: &[f64],
    parameters: &[MleParameter],
    make_distribution: F,
    options: MleOptions,
) -> Result<MleResult<D>, FitError>
where
    D: ContinuousDistribution,
    F: Fn(&[f64]) -> Option<D>,
{
    fit_mle_sample(MleSample::Exact(data), parameters, make_distribution, options)
}

/// Fit a continuous distribution to exact, left-, right-, or interval-censored
/// observations using maximum likelihood.
///
/// The likelihood contributes `logpdf(x)` for exact observations, `logcdf(x)`
/// for left censoring, `logsf(x)` for right censoring, and
/// `log(cdf(upper) - cdf(lower))` for interval censoring. Interval differences
/// use logarithms of the smaller tail to avoid cancellation and underflow.
///
/// All explicit exact/left/right observations must be finite. An interval may
/// have a lower bound of negative infinity or an upper bound of positive
/// infinity, but at least one endpoint must be finite. Equal finite endpoints
/// denote an exact observation, matching SciPy's `CensoredData` convention.
/// The input is validated without modifying its vectors.
///
/// Parameters, fixed values, parameter bounds, budgets, and runtime modes have
/// the same meanings as in [`fit_mle`]. A successful fit has a finite,
/// unpenalized likelihood for every observation. Data whose censored events all
/// have probability one at the fitted parameters are rejected as uninformative
/// when no exact observations are present. As with any local MLE optimizer,
/// convergence does not prove identifiability or a global maximum.
///
/// ```
/// use fsci_stats::{CensoredData, GammaDist, MleOptions, MleParameter};
/// use fsci_stats::mle::fit_mle_censored;
///
/// let data = CensoredData::right_censored(
///     &[1.0, 2.0, 3.0, 4.0],
///     &[false, false, true, true],
/// ).unwrap();
/// // Gamma(shape=1) is exponential. Its censored scale MLE is total
/// // exposure / observed event count, here 10 / 2 = 5.
/// let result = fit_mle_censored(
///     &data,
///     &[MleParameter::Fixed(1.0), MleParameter::Positive(1.0)],
///     |p| Some(GammaDist { a: p[0], scale: p[1] }),
///     MleOptions::default(),
/// ).unwrap();
/// assert!((result.distribution.scale - 5.0).abs() < 1.0e-5);
/// ```
pub fn fit_mle_censored<D, F>(
    data: &CensoredData,
    parameters: &[MleParameter],
    make_distribution: F,
    options: MleOptions,
) -> Result<MleResult<D>, FitError>
where
    D: ContinuousDistribution,
    F: Fn(&[f64]) -> Option<D>,
{
    fit_mle_sample(
        MleSample::Censored(data),
        parameters,
        make_distribution,
        options,
    )
}

fn fit_mle_sample<D, F>(
    data: MleSample<'_>,
    parameters: &[MleParameter],
    make_distribution: F,
    options: MleOptions,
) -> Result<MleResult<D>, FitError>
where
    D: ContinuousDistribution,
    F: Fn(&[f64]) -> Option<D>,
{
    if active_stats_mode() == RuntimeMode::Hardened && data.len() > hardened_max_sample_len() {
        return Err(FitError::ResourceExhausted {
            detail: "Hardened mode fit sample length exceeds HARDENED_MAX_SAMPLE_LEN",
        });
    }
    data.validate()?;
    if options.max_iterations == 0
        || options.max_evaluations == 0
        || !options.parameter_tolerance.is_finite()
        || options.parameter_tolerance <= 0.0
        || !options.likelihood_tolerance.is_finite()
        || options.likelihood_tolerance <= 0.0
    {
        return Err(FitError::UnsupportedData(
            "MLE budgets and tolerances must be positive and finite".into(),
        ));
    }
    for (index, &parameter) in parameters.iter().enumerate() {
        if !parameter.validate() {
            return Err(FitError::UnsupportedData(format!(
                "invalid MLE parameter specification at index {index}"
            )));
        }
    }
    let initial: Vec<_> = parameters.iter().map(|p| p.initial()).collect();
    if make_distribution(&initial).is_none() {
        return Err(FitError::UnsupportedData(
            "initial or fixed MLE parameters are outside the model's domain".into(),
        ));
    }
    let free: Vec<_> = parameters
        .iter()
        .enumerate()
        .filter_map(|(i, p)| (!matches!(p, MleParameter::Fixed(_))).then_some(i))
        .collect();
    let reconstruct = |point: &[f64]| -> Option<Vec<f64>> {
        let mut values = initial.clone();
        for (&index, &value) in free.iter().zip(point) {
            values[index] = parameters[index].decode(value)?;
        }
        Some(values)
    };
    let finish = |values: Vec<f64>, iterations, evaluations| {
        let distribution = make_distribution(&values).ok_or_else(|| {
            FitError::NonConvergent("MLE produced parameters outside the model's domain".into())
        })?;
        let mut nll = 0.0;
        let mut correction = 0.0;
        let mut all_zero = true;
        for log_density in data.log_terms(&distribution) {
            if !log_density.is_finite() {
                return Err(FitError::NonConvergent(
                    "MLE has non-finite log density at an observation".into(),
                ));
            }
            all_zero &= log_density == 0.0;
            // Compensated summation keeps the reported LL useful for AIC and for
            // comparisons between models with close likelihoods.
            let term = -log_density - correction;
            let next = nll + term;
            correction = (next - nll) - term;
            nll = next;
        }
        if !nll.is_finite() {
            return Err(FitError::NonConvergent(
                "MLE negative log likelihood is not finite".into(),
            ));
        }
        if !data.has_exact() && all_zero {
            return Err(FitError::NonConvergent(
                "censored observations have probability one and provide no fitting information".into(),
            ));
        }
        Ok(MleResult {
            distribution,
            parameters: values,
            negative_log_likelihood: nll,
            iterations,
            evaluations,
        })
    };
    if free.is_empty() {
        return finish(initial, 0, 1);
    }

    let start: Vec<_> = free.iter().map(|&i| parameters[i].encode()).collect();
    let bounds: Vec<_> = free
        .iter()
        .map(|&i| match parameters[i] {
            MleParameter::Bounded { lower, upper, .. } => (
                lower.is_finite().then_some(lower),
                upper.is_finite().then_some(upper),
            ),
            _ => (None, None),
        })
        .collect();
    // A nonzero step in log space is needed at shape=1 (encoded as zero).
    // Reflect an initial vertex at a bound instead of producing a flat simplex.
    let mut simplex = vec![start.clone(); free.len() + 1];
    for (i, &(lower, upper)) in bounds.iter().enumerate() {
        let mut step = 0.05 * start[i].abs().max(1.0);
        if let (Some(lo), Some(hi)) = (lower, upper) {
            step = step.min(0.05 * (hi - lo));
        }
        let mut value = start[i] + step;
        if !value.is_finite() || upper.is_some_and(|hi| value > hi) {
            value = start[i] - step;
        }
        if let Some(lo) = lower {
            value = value.max(lo);
        }
        if let Some(hi) = upper {
            value = value.min(hi);
        }
        if !value.is_finite() || value == start[i] {
            return Err(FitError::UnsupportedData(format!(
                "MLE parameter {} has no representable initial simplex step",
                free[i]
            )));
        }
        simplex[i + 1][i] = value;
    }

    // SciPy's finite support penalty: 100 * ln(max_float) per bad observation.
    // A final penalized value is never accepted by `finish`.
    let penalty = 100.0 * f64::MAX.ln();
    let objective = |point: &[f64]| {
        let Some(values) = reconstruct(point) else {
            return f64::MAX / 16.0;
        };
        let Some(distribution) = make_distribution(&values) else {
            return f64::MAX / 16.0;
        };
        let mut mean_nll = 0.0;
        let n = data.len() as f64;
        for log_density in data.log_terms(&distribution) {
            mean_nll += if log_density.is_finite() {
                -log_density / n
            } else {
                penalty / n
            };
        }
        if mean_nll.is_finite() {
            mean_nll
        } else {
            f64::MAX / 16.0
        }
    };
    let result = minimize(
        objective,
        &start,
        MinimizeOptions {
            method: Some(OptimizeMethod::NelderMead),
            maxiter: Some(options.max_iterations),
            maxfev: Some(options.max_evaluations),
            bounds: Some(&bounds),
            gradient_available: false,
            mode: active_stats_mode(),
            method_options: MinimizeMethodOptions {
                xatol: Some(options.parameter_tolerance),
                fatol: Some(options.likelihood_tolerance),
                initial_simplex: Some(&simplex),
                ..MinimizeMethodOptions::default()
            },
            ..MinimizeOptions::default()
        },
    )
    .map_err(|error| FitError::NonConvergent(error.to_string()))?;
    if !result.success {
        return Err(FitError::NonConvergent(format!(
            "{} ({} iterations, {} evaluations)",
            result.message, result.nit, result.nfev
        )));
    }
    let values = reconstruct(&result.x).ok_or_else(|| {
        FitError::NonConvergent("MLE parameters are not finite or are outside their domains".into())
    })?;
    finish(values, result.nit, result.nfev)
}
