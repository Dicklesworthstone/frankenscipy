//! Fallible maximum-likelihood fitting with explicit parameter constraints.
//!
//! [`fit_mle`] minimizes the negative log likelihood using Nelder–Mead. The
//! model factory defines the parameterization, including any location and scale
//! parameters; [`MleParameter::Fixed`] holds a parameter exactly constant. This
//! also supports distributions whose concrete type has fixed location or scale.
//! No method-of-moments estimate is returned as a successful MLE.

use fsci_opt::{MinimizeMethodOptions, MinimizeOptions, OptimizeMethod, minimize};

use crate::{
    ContinuousDistribution, FitError, RuntimeMode, active_stats_mode, hardened_max_sample_len,
};

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

/// Budgets and stopping tolerances for [`fit_mle`].
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
    if active_stats_mode() == RuntimeMode::Hardened && data.len() > hardened_max_sample_len() {
        return Err(FitError::ResourceExhausted {
            detail: "Hardened mode fit sample length exceeds HARDENED_MAX_SAMPLE_LEN",
        });
    }
    crate::validate_finite_fit_data(data, 1, "maximum likelihood")?;
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
        for &x in data {
            let log_density = distribution.logpdf(x);
            if !log_density.is_finite() {
                return Err(FitError::NonConvergent(
                    "MLE has non-finite log density at an observation".into(),
                ));
            }
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
        for &x in data {
            let log_density = distribution.logpdf(x);
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
