//! Fallible maximum-likelihood fitting with explicit parameter constraints.
//!
//! [`fit_mle`] and [`fit_mle_censored`] minimize the negative log likelihood
//! using Nelder–Mead. The
//! model factory defines the parameterization, including any location and scale
//! parameters; [`MleParameter::Fixed`] holds a parameter exactly constant. This
//! also supports distributions whose concrete type has fixed location or scale.
//! No method-of-moments estimate is returned as a successful MLE.
//!
//! [`LocationScale`] supplies the affine distribution `loc + scale * Z` for any
//! continuous base distribution. [`fit_mle_location_scale`] and
//! [`fit_mle_location_scale_censored`] fit its shapes, location, and scale with
//! the same explicit bounds and fixed-parameter masks.
//!
//! [`fit_mle_discrete`] fits signed integer observations with finite parameter
//! bounds using differential evolution. Integer shape parameters are enforced
//! throughout the search, and equal bounds hold parameters exactly fixed.

pub use fsci_opt::DifferentialEvolutionOptions;
use fsci_opt::{
    MinimizeMethodOptions, MinimizeOptions, OptimizeMethod, differential_evolution, minimize,
};
use rand::Rng;

use crate::{
    CensoredData, ContinuousDistribution, DiscreteDistribution, FitError, RuntimeMode, StatsError,
    active_stats_mode, hardened_max_sample_len,
};

/// The continuous distribution of `X = loc + scale * Z` for a base `Z`.
///
/// Location must be finite and scale must be positive and finite. Parameters
/// are immutable so evaluation cannot invalidate these constraints. The base
/// distribution retains responsibility for validating its own parameters.
/// Log densities, log tails, and inverse survival probabilities delegate to the
/// corresponding base methods without first rounding ordinary probabilities.
///
/// Use [`fit_mle_location_scale`] or [`fit_mle_location_scale_censored`] to fit
/// shapes together with location and scale. A generic adapter cannot infer the
/// base family's shape parameterization, so its trait-level `try_fit` retains
/// the default [`FitError::NotImplemented`] behavior.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocationScale<D> {
    base: D,
    loc: f64,
    scale: f64,
    log_scale: f64,
}

impl<D> LocationScale<D> {
    /// Construct an affine distribution, rejecting a nonfinite location or a
    /// nonpositive/nonfinite scale without panicking.
    pub fn new(base: D, loc: f64, scale: f64) -> Result<Self, StatsError> {
        if !loc.is_finite() || !scale.is_finite() || scale <= 0.0 {
            return Err(StatsError::InvalidArgument(
                "location must be finite and scale must be positive and finite".into(),
            ));
        }
        Ok(Self {
            base,
            loc,
            scale,
            log_scale: scale.ln(),
        })
    }

    /// The base distribution, before the affine transformation.
    #[must_use]
    pub fn base(&self) -> &D {
        &self.base
    }

    /// The additive location parameter.
    #[must_use]
    pub fn loc(&self) -> f64 {
        self.loc
    }

    /// The positive multiplicative scale parameter.
    #[must_use]
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Recover the base distribution and affine parameters.
    #[must_use]
    pub fn into_parts(self) -> (D, f64, f64) {
        (self.base, self.loc, self.scale)
    }

    fn standardize(&self, x: f64) -> f64 {
        let difference = x - self.loc;
        if difference.is_infinite() && x.is_finite() {
            // Opposite-sign finite inputs can overflow subtraction even when
            // their standardized difference is representable. Dividing first
            // is safe here because the two terms cannot cancel.
            x / self.scale - self.loc / self.scale
        } else {
            difference / self.scale
        }
    }

    fn transform(&self, z: f64) -> f64 {
        // A fused operation also handles cancellation when scale*z alone
        // would overflow but adding location gives a finite quantile or mean.
        self.scale.mul_add(z, self.loc)
    }
}

impl<D: ContinuousDistribution> ContinuousDistribution for LocationScale<D> {
    fn pdf(&self, x: f64) -> f64 {
        let z = self.standardize(x);
        let base_density = self.base.pdf(z);
        let density = base_density / self.scale;
        if base_density.is_subnormal() || density == 0.0 || density.is_infinite() {
            // A small scale can rescue a base density that underflows, and a
            // large scale can rescue one that overflows. Subnormal base
            // densities also lose relative precision before the rescaling.
            (self.base.logpdf(z) - self.log_scale).exp()
        } else {
            density
        }
    }

    fn logpdf(&self, x: f64) -> f64 {
        self.base.logpdf(self.standardize(x)) - self.log_scale
    }

    fn cdf(&self, x: f64) -> f64 {
        self.base.cdf(self.standardize(x))
    }

    fn logcdf(&self, x: f64) -> f64 {
        self.base.logcdf(self.standardize(x))
    }

    fn sf(&self, x: f64) -> f64 {
        self.base.sf(self.standardize(x))
    }

    fn logsf(&self, x: f64) -> f64 {
        self.base.logsf(self.standardize(x))
    }

    fn ppf(&self, q: f64) -> f64 {
        self.transform(self.base.ppf(q))
    }

    fn isf(&self, q: f64) -> f64 {
        self.transform(self.base.isf(q))
    }

    fn cdf_sf_is_cheap(&self) -> bool {
        self.base.cdf_sf_is_cheap()
    }

    fn cdf_sf_par_min(&self) -> usize {
        self.base.cdf_sf_par_min()
    }

    fn ppf_isf_is_cheap(&self) -> bool {
        self.base.ppf_isf_is_cheap()
    }

    fn ppf_isf_par_min(&self) -> usize {
        self.base.ppf_isf_par_min()
    }

    fn mean(&self) -> f64 {
        self.transform(self.base.mean())
    }

    fn var(&self) -> f64 {
        // Multiplying the variance first avoids a spurious infinity or zero
        // from scale*scale when the final scaled variance is representable.
        self.scale * (self.scale * self.base.var())
    }

    fn std(&self) -> f64 {
        self.scale * self.base.std()
    }

    fn entropy(&self) -> f64 {
        self.base.entropy() + self.log_scale
    }

    fn try_entropy(&self) -> Result<f64, StatsError> {
        self.base.try_entropy().map(|value| value + self.log_scale)
    }

    fn median(&self) -> f64 {
        self.transform(self.base.median())
    }

    fn mode(&self) -> f64 {
        self.transform(self.base.mode())
    }

    fn skewness(&self) -> f64 {
        self.base.skewness()
    }

    fn try_skewness(&self) -> Result<f64, StatsError> {
        self.base.try_skewness()
    }

    fn kurtosis(&self) -> f64 {
        self.base.kurtosis()
    }

    fn try_kurtosis(&self) -> Result<f64, StatsError> {
        self.base.try_kurtosis()
    }

    fn interval(&self, alpha: f64) -> (f64, f64) {
        let (lower, upper) = self.base.interval(alpha);
        (self.transform(lower), self.transform(upper))
    }

    fn rvs(&self, n: usize, rng: &mut impl Rng) -> Vec<f64> {
        let mut values = self.base.rvs(n, rng);
        for value in &mut values {
            *value = self.transform(*value);
        }
        values
    }
}

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
        let (exact, left, right, interval): (&[f64], &[f64], &[f64], &[(f64, f64)]) = match self {
            Self::Exact(data) => (*data, &[], &[], &[]),
            Self::Censored(data) => (&data.uncensored, &data.left, &data.right, &data.interval),
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
    let (larger, smaller) = if log_cdf_upper < log_sf_lower {
        (log_cdf_upper, distribution.logcdf(lower))
    } else {
        (log_sf_lower, distribution.logsf(upper))
    };
    let difference = log_probability_difference(larger, smaller);
    if difference.is_nan() {
        // An invalid or decreasing tail must not be repaired by a density
        // integral: the supplied probability model is inconsistent.
        return difference;
    }
    // expm1 preserves a small difference between two *distinct* logarithms,
    // but cannot recover digits already lost in those logarithms. In the
    // centre of a normal distribution, for example, [0, 1e-20] has positive
    // mass although both log-CDFs and both log-SFs round to the same value.
    // Quadrature is only needed where this subtraction is ill-conditioned.
    let cancellation_scale = 1.0_f64.max(larger.abs()).max(smaller.abs());
    if (difference == f64::NEG_INFINITY || larger - smaller < 1.0e-4 * cancellation_scale)
        && let Some(integral) = density_interval_log_probability(distribution, lower, upper)
    {
        return integral;
    }
    difference
}

/// Integrate in unit-interval coordinates and scale by a log density so neither
/// a subnormal interval width nor an underflowing tail density erases its mass.
/// Keep the tail result when the density is singular or the bounded adaptive
/// quadrature cannot establish the requested relative accuracy.
fn density_interval_log_probability<D: ContinuousDistribution>(
    distribution: &D,
    lower: f64,
    upper: f64,
) -> Option<f64> {
    let width = upper - lower;
    if !width.is_finite() || width <= 0.0 {
        return None;
    }
    let midpoint = width.mul_add(0.5, lower);
    if midpoint <= lower || midpoint >= upper {
        // With no representable interior point, evaluating a support endpoint
        // can fabricate mass in an interval that lies entirely outside support.
        return None;
    }
    let log_scale = distribution.logpdf(midpoint);
    if !log_scale.is_finite() {
        return None;
    }
    let relative_tolerance = 1.0e-11;
    let (result, _) = fsci_integrate::quad_full_output(
        |t| (distribution.logpdf(width.mul_add(t, lower)) - log_scale).exp(),
        0.0,
        1.0,
        &[],
        fsci_integrate::QuadOptions {
            epsabs: 0.0,
            epsrel: relative_tolerance,
            limit: 32,
        },
    )
    .ok()?;
    if !result.converged
        || !result.integral.is_finite()
        || result.integral <= 0.0
        || !result.error.is_finite()
        || result.error > relative_tolerance * result.integral
    {
        return None;
    }
    let probability = width.ln() + log_scale + result.integral.ln();
    (probability.is_finite() && probability <= 0.0).then_some(probability)
}

/// Solve the two beta shape scores from logarithmic sufficient statistics.
/// Beta-prime fitting uses the same likelihood after a parameter-independent
/// change of variables, but must retain these logs before its transformed
/// observations round to zero or one.
pub(crate) fn beta_shapes_mle(
    log_x: f64,
    log_one_minus_x: f64,
    initial: (f64, f64),
    family: &str,
) -> Result<(f64, f64), FitError> {
    let (mut a, mut b) = initial;
    if !a.is_finite() || !b.is_finite() || a <= 0.0 || b <= 0.0 {
        // With very dispersed beta-prime data the moment transform can round
        // to a Bernoulli sample, giving a zero concentration. The geometric
        // means still retain a valid interior starting point. Evaluate their
        // Jensen gap without subtracting a rounded value close to one.
        let gap = if log_x > log_one_minus_x {
            -log_x.exp_m1() - log_one_minus_x.exp()
        } else {
            -log_one_minus_x.exp_m1() - log_x.exp()
        };
        if gap.is_finite() && gap > 0.0 {
            a = -log_one_minus_x.exp_m1() / (2.0 * gap);
            b = -log_x.exp_m1() / (2.0 * gap);
        }
    }
    if !a.is_finite() || !b.is_finite() || a <= 0.0 || b <= 0.0 {
        return Err(FitError::NonConvergent(format!(
            "{family} MLE has no finite initial shapes for a degenerate sample"
        )));
    }
    let objective = |a: f64, b: f64| {
        crate::unwrap_special_scalar(fsci_special::betaln_scalar(a, b, active_stats_mode()))
            - (a - 1.0) * log_x
            - (b - 1.0) * log_one_minus_x
    };
    let mut value = objective(a, b);

    // The shape log-likelihood is strictly concave. Backtracking keeps the
    // Newton iterates positive and decreases the negative log-likelihood,
    // without imposing artificial upper or lower limits on either shape.
    for _ in 0..200 {
        let psi_a = crate::digamma(a);
        let psi_b = crate::digamma(b);
        let psi_sum = crate::digamma(a + b);
        let ga = psi_a - psi_sum - log_x;
        let gb = psi_b - psi_sum - log_one_minus_x;
        let tri_sum = fsci_special::trigamma(a + b);
        let haa = fsci_special::trigamma(a) - tri_sum;
        let hbb = fsci_special::trigamma(b) - tri_sum;
        let determinant = haa * hbb - tri_sum * tri_sum;
        if !determinant.is_finite() || determinant <= 0.0 {
            break;
        }
        let da = (hbb * ga + tri_sum * gb) / determinant;
        let db = (tri_sum * ga + haa * gb) / determinant;
        if !da.is_finite() || !db.is_finite() {
            break;
        }
        if (da / a).abs().max((db / b).abs()) <= 1e-8 {
            return Ok((a - da, b - db));
        }
        // At large concentration, subtracting digammas can no longer resolve
        // the score although the ill-conditioned Hessian amplifies roundoff
        // into a visible relative step. Preserve the last resolved estimate.
        let score_roundoff_a = 2.0 * f64::EPSILON * (psi_a.abs() + psi_sum.abs() + log_x.abs());
        let score_roundoff_b =
            2.0 * f64::EPSILON * (psi_b.abs() + psi_sum.abs() + log_one_minus_x.abs());
        if ga.abs() <= score_roundoff_a && gb.abs() <= score_roundoff_b {
            return Ok((a, b));
        }
        let mut fraction = 1.0;
        let mut accepted = false;
        for _ in 0..60 {
            let next_a = a - fraction * da;
            let next_b = b - fraction * db;
            if next_a > 0.0 && next_b > 0.0 && next_a.is_finite() && next_b.is_finite() {
                let next_value = objective(next_a, next_b);
                let roundoff = 8.0 * f64::EPSILON * (1.0 + value.abs());
                if next_value.is_finite() && next_value <= value + roundoff {
                    a = next_a;
                    b = next_b;
                    value = next_value;
                    accepted = true;
                    break;
                }
            }
            fraction *= 0.5;
        }
        if !accepted {
            break;
        }
    }
    Err(FitError::NonConvergent(format!(
        "{family} MLE score equations did not converge"
    )))
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

/// Fit Cauchy location and scale. Quartiles provide only the initial guess;
/// unlike the former estimator, the returned parameters maximize the actual
/// likelihood. Work in interquartile-range units so an optimizer's absolute
/// coordinate tolerance does not depend on the sample's physical units.
pub(crate) fn cauchy_mle(data: &[f64]) -> Result<crate::Cauchy, FitError> {
    validate_shape_sample(data, 2, "Cauchy")?;
    let mut sorted = data.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    let mut maximum_count = 1;
    let mut count = 1;
    let mut unique = 1;
    for pair in sorted.windows(2) {
        if pair[0] == pair[1] {
            count += 1;
            maximum_count = maximum_count.max(count);
        } else {
            count = 1;
            unique += 1;
        }
    }
    if maximum_count > data.len() - maximum_count {
        // At a point repeated k times the likelihood grows as scale^(n-2k).
        // A majority repeated value therefore gives an unbounded likelihood,
        // not a valid MLE with an arbitrarily tiny positive scale.
        return Err(FitError::NonConvergent(
            "Cauchy likelihood is unbounded at a majority repeated observation".into(),
        ));
    }
    let origin = crate::quantile_sorted(&sorted, 0.5);
    let q25 = crate::quantile_sorted(&sorted, 0.25);
    let q75 = crate::quantile_sorted(&sorted, 0.75);
    let difference = q75 - q25;
    let spread = if difference.is_finite() {
        0.5 * difference
    } else {
        0.5 * q75 - 0.5 * q25
    };
    if !spread.is_finite() || spread <= 0.0 {
        return Err(FitError::NonConvergent(
            "Cauchy sample has no positive finite fitting scale".into(),
        ));
    }
    if maximum_count == data.len() - maximum_count {
        if unique == 2 {
            // Equal mass at two distinct points has a continuum of maxima.
            // The midpoint and half-distance select a deterministic member,
            // including the ordinary two-observation case.
            let lower = sorted[0];
            let upper = sorted[sorted.len() - 1];
            let difference = upper - lower;
            let scale = if difference.is_finite() {
                0.5 * difference
            } else {
                0.5 * upper - 0.5 * lower
            };
            if scale.is_finite() && scale > 0.0 {
                return Ok(crate::Cauchy { loc: origin, scale });
            }
        }
        // The joint score equation averages points on a unit circle to zero.
        // Half the mass at one point requires the other half at a single
        // opposite point. With more than two distinct values there is no
        // stationary point at positive scale: only a boundary supremum.
        return Err(FitError::NonConvergent(
            "Cauchy likelihood has no finite positive-scale maximum for this repeated sample"
                .into(),
        ));
    }
    let standardized: Vec<_> = data
        .iter()
        .map(|&x| {
            let difference = x - origin;
            if difference.is_finite() {
                difference / spread
            } else {
                x / spread - origin / spread
            }
        })
        .collect();
    if standardized.iter().any(|x| !x.is_finite()) {
        return Err(FitError::NonConvergent(
            "Cauchy sample range is not representable in fitting coordinates".into(),
        ));
    }
    let fitted = fit_mle(
        &standardized,
        &[MleParameter::Free(0.0), MleParameter::Positive(1.0)],
        |p| {
            Some(crate::Cauchy {
                loc: p[0],
                scale: p[1],
            })
        },
        MleOptions::default(),
    )?
    .distribution;
    let distribution = crate::Cauchy {
        loc: spread.mul_add(fitted.loc, origin),
        scale: spread * fitted.scale,
    };
    if !distribution.loc.is_finite()
        || !distribution.scale.is_finite()
        || distribution.scale <= 0.0
        || data.iter().any(|&x| !distribution.logpdf(x).is_finite())
    {
        return Err(FitError::NonConvergent(
            "Cauchy MLE is not representable with finite parameters and log densities".into(),
        ));
    }
    Ok(distribution)
}

/// The Landau law has no moments; quantile matching supplies a scale-equivariant
/// start without letting a large right-tail observation dominate the search.
pub(crate) fn landau_mle(data: &[f64]) -> Result<crate::Landau, FitError> {
    validate_shape_sample(data, 2, "Landau")?;
    let quantiles = crate::quantile(data, &[0.0, 0.25, 0.5, 0.75, 1.0]);
    let minimum = quantiles[0];
    let minimum_count = data.iter().filter(|&&x| x == minimum).count();
    if minimum_count > data.len() - minimum_count {
        // At the minimum, each repeated density grows like 1/scale; at every
        // larger observation the Landau right tail decays like scale. More
        // than half the sample at the minimum therefore gives an unbounded
        // likelihood, not a valid small-scale MLE.
        return Err(FitError::NonConvergent(
            "Landau likelihood is unbounded as scale tends to zero".into(),
        ));
    }
    let standard_iqr = crate::landau::landau_ppf(0.75) - crate::landau::landau_ppf(0.25);
    let iqr = quantiles[3] - quantiles[1];
    let mut spread = if iqr.is_finite() {
        iqr / standard_iqr
    } else {
        quantiles[3] / standard_iqr - quantiles[1] / standard_iqr
    };
    if spread == 0.0 {
        // A nonconstant, heavily rounded sample can have zero IQR. Its range
        // still supplies a valid start unless the MLE scale is unrepresentable.
        spread = quantiles[4] / standard_iqr - minimum / standard_iqr;
    }
    if !spread.is_finite() || spread <= 0.0 {
        return Err(FitError::NonConvergent(
            "Landau sample has no positive finite fitting scale".into(),
        ));
    }
    let origin = quantiles[2];
    let standardized: Vec<_> = data
        .iter()
        .map(|&x| {
            let difference = x - origin;
            if difference.is_finite() {
                difference / spread
            } else {
                x / spread - origin / spread
            }
        })
        .collect();
    if standardized.iter().any(|x| !x.is_finite()) {
        return Err(FitError::NonConvergent(
            "Landau sample range is not representable in fitting coordinates".into(),
        ));
    }
    let fitted = fit_mle(
        &standardized,
        &[
            MleParameter::Free(-crate::landau::landau_ppf(0.5)),
            MleParameter::Positive(1.0),
        ],
        |p| {
            Some(crate::Landau {
                loc: p[0],
                scale: p[1],
            })
        },
        MleOptions::default(),
    )?
    .distribution;
    let distribution = crate::Landau {
        loc: spread.mul_add(fitted.loc, origin),
        scale: spread * fitted.scale,
    };
    if !distribution.loc.is_finite()
        || !distribution.scale.is_finite()
        || distribution.scale <= 0.0
        || data.iter().any(|&x| !distribution.logpdf(x).is_finite())
    {
        return Err(FitError::NonConvergent(
            "Landau MLE is not representable with finite parameters and log densities".into(),
        ));
    }
    Ok(distribution)
}

/// Work in the sample's unit range. Nonnegative margins keep every observation
/// inside the trial support while c <= d preserves the ordered plateau. Multiple
/// starts prevent a collapsed uniform/triangular simplex from hiding an interior
/// maximum; all candidates still use the same unpenalized likelihood driver.
pub(crate) fn trapezoid_mle(data: &[f64]) -> Result<crate::Trapezoid, FitError> {
    validate_shape_sample(data, 2, "Trapezoid")?;
    let minimum = data.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let width = maximum - minimum;
    if !width.is_finite() || width <= 0.0 {
        return Err(FitError::NonConvergent(
            "Trapezoid MLE requires a positive, finite sample range".into(),
        ));
    }
    let standardized: Vec<_> = data.iter().map(|&x| (x - minimum) / width).collect();
    let mut best: Option<MleResult<crate::Trapezoid>> = None;
    let mut failure = FitError::NonConvergent("Trapezoid MLE did not converge".into());
    for [c, d, left_margin, right_margin] in [
        [0.2, 0.8, 0.05, 0.05],
        [0.1, 0.3, 0.05, 0.05],
        [0.7, 0.9, 0.05, 0.05],
        [0.5, 0.5, 0.05, 0.05],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.4, 0.0, 0.05],
        [0.6, 1.0, 0.05, 0.0],
    ] {
        let bounded = |initial, upper| MleParameter::Bounded {
            initial,
            lower: 0.0,
            upper,
        };
        let result = fit_mle(
            &standardized,
            &[
                bounded(c, 1.0),
                bounded(d, 1.0),
                bounded(left_margin, f64::INFINITY),
                bounded(right_margin, f64::INFINITY),
            ],
            |p| {
                let scale = 1.0 + p[2] + p[3];
                (p[0] <= p[1] && scale.is_finite()).then_some(crate::Trapezoid {
                    c: p[0],
                    d: p[1],
                    loc: -p[2],
                    scale,
                })
            },
            MleOptions::default(),
        );
        match result {
            Ok(candidate) => {
                if best.as_ref().is_none_or(|previous| {
                    candidate.negative_log_likelihood < previous.negative_log_likelihood
                }) {
                    best = Some(candidate);
                }
            }
            Err(error) => failure = error,
        }
    }
    let fitted = best.ok_or(failure)?.distribution;
    let distribution = crate::Trapezoid {
        c: fitted.c,
        d: fitted.d,
        loc: width.mul_add(fitted.loc, minimum),
        scale: width * fitted.scale,
    };
    if !distribution.loc.is_finite()
        || !distribution.scale.is_finite()
        || distribution.scale <= 0.0
        || data.iter().any(|&x| !distribution.logpdf(x).is_finite())
    {
        return Err(FitError::NonConvergent(
            "Trapezoid MLE is not representable with finite parameters and log densities".into(),
        ));
    }
    Ok(distribution)
}

fn compensated_mean(values: impl Iterator<Item = f64>, count: usize) -> f64 {
    let mut mean = 0.0;
    let mut correction = 0.0;
    for value in values {
        if !value.is_finite() {
            return value;
        }
        let term = value / count as f64 - correction;
        let next = mean + term;
        correction = (next - mean) - term;
        mean = next;
    }
    mean
}

/// Standardized Rice MLE. The likelihood is concave in b^2: its boundary
/// maximum is b=0 when mean(x^2)<=2; otherwise its unique positive score root
/// lies below mean(x), because I1(z)/I0(z)<1.
pub(crate) fn rice_shape_mle(data: &[f64]) -> Result<f64, FitError> {
    validate_shape_sample(data, 2, "Rice")?;
    if data.iter().any(|&x| x <= 0.0) {
        return Err(FitError::UnsupportedData(
            "Rice MLE with fixed location zero requires strictly positive observations".into(),
        ));
    }
    if compensated_mean(data.iter().map(|x| x * x), data.len()) <= 2.0 {
        return Ok(0.0);
    }
    let score = |b: f64| {
        compensated_mean(
            data.iter().map(|&x| {
                let z = b * x;
                if z < 1.0e-3 {
                    // I1(z)/(z*I0(z)); dividing the score by b avoids the
                    // spurious zero root and cancellation near the boundary.
                    let square = z * z;
                    x * x
                        * (0.5
                            + square
                                * (-1.0 / 16.0 + square * (1.0 / 96.0 - square * (11.0 / 6144.0))))
                } else if z.is_infinite() {
                    // For an overflowing finite product, the Bessel ratio
                    // differs from one by much less than one f64 ulp.
                    x / b
                } else {
                    (x / b) * (fsci_special::i1e_scalar(z) / fsci_special::i0e_scalar(z))
                }
            }),
            data.len(),
        ) - 1.0
    };
    let mut lower = 0.0;
    let mut upper = compensated_mean(data.iter().copied(), data.len());
    if score(upper) == 0.0 {
        return Ok(upper);
    }
    for _ in 0..128 {
        let middle = lower + 0.5 * (upper - lower);
        if middle == lower || middle == upper {
            return Ok(middle);
        }
        let value = score(middle);
        if value.is_nan() {
            return Err(FitError::NonConvergent(
                "Rice likelihood score could not be evaluated".into(),
            ));
        }
        if value > 0.0 {
            lower = middle;
        } else if value < 0.0 {
            upper = middle;
        } else {
            return Ok(middle);
        }
        if upper - lower <= 8.0 * f64::EPSILON * middle {
            return Ok(lower + 0.5 * (upper - lower));
        }
    }
    Err(FitError::NonConvergent(
        "Rice likelihood score did not converge".into(),
    ))
}

/// 1 - K0(z)/K1(z), retaining the small positive difference when the two
/// exponentially scaled Bessel functions round to the same value.
fn k01_ratio_complement(z: f64) -> f64 {
    if z >= 1.0e4 {
        let inverse = z.recip();
        inverse
            * (0.5
                + inverse
                    * (-0.375 + inverse * (0.375 + inverse * (-0.492_187_5 + inverse * 0.843_75))))
    } else {
        1.0 - fsci_special::bessel::k0e_scalar(z) / fsci_special::bessel::k1e_scalar(z)
    }
}

/// Profile out NIG skew exactly: b=a*mean(x)/hypot(1,mean(x)). The remaining
/// shape score is q-mean(r*K0(a*r)/K1(a*r)), q=hypot(1,mean(x)), r=hypot(1,x).
/// K0/K1 increases monotonically, so a bracketed scalar solve finds the unique
/// finite maximum without returning a moment estimate as an MLE.
pub(crate) fn norminvgauss_shapes_mle(data: &[f64]) -> Result<(f64, f64), FitError> {
    validate_shape_sample(data, 2, "NormInvGauss")?;
    if data.iter().all(|&x| x == data[0]) {
        return Err(FitError::NonConvergent(
            "NormInvGauss MLE is unbounded for a constant sample".into(),
        ));
    }
    let mean = compensated_mean(data.iter().copied(), data.len());
    let q = mean.hypot(1.0);
    let skew = mean / q;
    if !mean.is_finite() || skew.abs() >= 1.0 {
        return Err(FitError::NonConvergent(
            "NormInvGauss MLE skew is outside the representable parameter domain".into(),
        ));
    }
    let radii: Vec<_> = data.iter().map(|x| x.hypot(1.0)).collect();
    // Rationalize r - (1/q + skew*x) before averaging. Its expectation is
    // the Jensen gap mean(r)-q, which direct subtraction loses completely
    // for concentrated samples. Using x-mean also makes constant data exact.
    let gap = compensated_mean(
        data.iter().zip(&radii).map(|(&x, &radius)| {
            let projection = skew.mul_add(x, q.recip());
            if projection <= 0.0 {
                radius - projection
            } else {
                let difference = x - mean;
                let orthogonal = if difference.is_finite() {
                    difference / q
                } else {
                    x / q - skew
                };
                (orthogonal / radius) * orthogonal / (1.0 + projection / radius)
            }
        }),
        data.len(),
    );
    if !gap.is_finite() || gap <= 0.0 {
        return Err(FitError::NonConvergent(
            "NormInvGauss likelihood gap is outside the finite parameter range".into(),
        ));
    }
    let score = |a: f64| {
        if gap < 0.5 * q {
            // The complement form preserves a score of order 1/a even when
            // q and mean(r*K0/K1) agree to every available significant bit.
            compensated_mean(
                radii.iter().map(|&radius| {
                    let z = a * radius;
                    if z.is_infinite() {
                        0.5 / a
                    } else {
                        radius * k01_ratio_complement(z)
                    }
                }),
                data.len(),
            ) - gap
        } else {
            q - compensated_mean(
                radii.iter().map(|&radius| {
                    let z = a * radius;
                    if z < 1.0e-10 {
                        // K1(z) can overflow near zero although its ratio,
                        // multiplied by radius, is finite and informative.
                        (radius * z) * (-z.ln() + std::f64::consts::LN_2 - crate::EULER_MASCHERONI)
                    } else if z.is_infinite() {
                        radius
                    } else {
                        radius
                            * (fsci_special::bessel::k0e_scalar(z)
                                / fsci_special::bessel::k1e_scalar(z))
                    }
                }),
                data.len(),
            )
        }
    };
    // 0 < 1-K0(z)/K1(z) < 1/(2z) gives an upper shape bracket.
    let upper = 0.5 / gap;
    let smallest = f64::from_bits(1);
    if !upper.is_finite() || upper <= 0.0 || score(smallest) <= 0.0 {
        return Err(FitError::NonConvergent(
            "NormInvGauss MLE shape is outside the finite parameter range".into(),
        ));
    }
    let mut log_lower = smallest.ln();
    let mut log_upper = upper.ln();
    for _ in 0..128 {
        let log_middle = log_lower + 0.5 * (log_upper - log_lower);
        let a = log_middle.exp();
        let value = score(a);
        if value.is_nan() {
            return Err(FitError::NonConvergent(
                "NormInvGauss likelihood score could not be evaluated".into(),
            ));
        }
        if value > 0.0 {
            log_lower = log_middle;
        } else if value < 0.0 {
            log_upper = log_middle;
        }
        if value == 0.0 || log_upper - log_lower <= 8.0 * f64::EPSILON * (1.0 + log_middle.abs()) {
            let b = a * skew;
            if a.is_finite() && a > 0.0 && b.is_finite() && b.abs() < a {
                return Ok((a, b));
            }
            break;
        }
    }
    Err(FitError::NonConvergent(
        "NormInvGauss likelihood score did not converge to representable shapes".into(),
    ))
}

/// log(a) - psi(a), preserving the small positive gap at large shape.
fn log_minus_digamma(a: f64) -> f64 {
    if a < 16.0 {
        a.ln() - crate::digamma(a)
    } else {
        // Bernoulli expansion from the Gamma MLE: subtracting the two
        // logarithmic terms directly loses the score for concentrated data.
        let inverse = a.recip();
        let square = inverse * inverse;
        0.5 * inverse
            + square
                * (1.0 / 12.0
                    + square
                        * (-1.0 / 120.0
                            + square
                                * (1.0 / 252.0
                                    + square
                                        * (-1.0 / 240.0
                                            + square
                                                * (1.0 / 132.0 - square * (691.0 / 32760.0))))))
    }
}

/// Shared shape score for Gamma and standardized Nakagami likelihoods.
/// The strict bounds 1/(2a) < log(a)-psi(a) < 1/a bracket the unique root
/// without an arbitrary upper shape limit or an unconstrained Newton step.
pub(crate) fn gamma_shape_from_log_gap(target: f64, family: &str) -> Result<f64, FitError> {
    if !target.is_finite() || target <= 0.0 {
        return Err(FitError::NonConvergent(format!(
            "{family} MLE requires a finite positive likelihood gap"
        )));
    }
    let mut lower = 0.5 / target;
    let mut upper = 1.0 / target;
    if !upper.is_finite() || upper <= 0.0 {
        return Err(FitError::NonConvergent(format!(
            "{family} MLE shape exceeds the finite parameter range"
        )));
    }
    for _ in 0..100 {
        let shape = lower + 0.5 * (upper - lower);
        if log_minus_digamma(shape) > target {
            lower = shape;
        } else {
            upper = shape;
        }
        if upper - lower <= 4.0 * f64::EPSILON * shape {
            break;
        }
    }
    Ok(lower + 0.5 * (upper - lower))
}

/// x^2 - 1 - 2*log(x), nonnegative for positive x. Centering at one avoids
/// cancellation of all significant bits for tightly concentrated samples.
pub(crate) fn nakagami_shape_statistic(x: f64) -> f64 {
    let offset = x - 1.0;
    if offset.abs() <= 0.5 {
        offset * offset - 2.0 * fsci_special::log1pmx_scalar(offset)
    } else {
        x * x - 1.0 - 2.0 * x.ln()
    }
}

pub(crate) fn nakagami_shape_mle(data: &[f64]) -> Result<f64, FitError> {
    validate_shape_sample(data, 2, "Nakagami")?;
    if data.iter().any(|&x| x <= 0.0) {
        return Err(FitError::UnsupportedData(
            "Nakagami MLE with fixed location zero requires strictly positive observations".into(),
        ));
    }
    // Location and scale are fixed at zero and one in the concrete type.
    // The log-likelihood score is log(nu)-psi(nu)-mean(x^2-1-2*log(x)).
    // Constant samples away from one have a finite maximum; only all-ones
    // data have an unbounded concentration parameter.
    let mut target = 0.0;
    let mut correction = 0.0;
    let n = data.len() as f64;
    for &x in data {
        let term = nakagami_shape_statistic(x) / n - correction;
        let next = target + term;
        correction = (next - target) - term;
        target = next;
    }
    gamma_shape_from_log_gap(target, "Nakagami")
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

/// Inverse-gamma shape MLE at location zero and scale one. Its sufficient
/// statistic is -mean(log(x)); unlike a moment estimate, the score remains
/// defined for shapes below one and for samples whose arithmetic sum overflows.
pub(crate) fn inverse_gamma_shape_mle(data: &[f64]) -> Result<f64, FitError> {
    validate_shape_sample(data, 2, "InverseGamma")?;
    let mut target = 0.0;
    let mut correction = 0.0;
    let n = data.len() as f64;
    for &x in data {
        if x <= 0.0 {
            return Err(FitError::UnsupportedData(
                "InverseGamma MLE with fixed location zero requires positive observations".into(),
            ));
        }
        let term = -x.ln() / n - correction;
        let next = target + term;
        correction = (next - target) - term;
        target = next;
    }

    shape_from_digamma(target, "InverseGamma")
}

/// Solve a positive shape score `psi(a) = target`. Several fixed-scale
/// gamma-family likelihoods share this strictly monotone equation.
pub(crate) fn shape_from_digamma(target: f64, family: &str) -> Result<f64, FitError> {
    // psi(a) increases strictly from -infinity to infinity. The asymptotic
    // inverse gives a starting upper bound; expand it if rounding requires it.
    let mut upper = (target.exp() + 1.0).max(1.0);
    if !upper.is_finite() {
        return Err(FitError::NonConvergent(format!(
            "{family} MLE shape exceeds the finite parameter range"
        )));
    }
    while fsci_special::digamma_scalar(upper) < target {
        let next = (2.0 * upper).min(f64::MAX);
        if next == upper {
            return Err(FitError::NonConvergent(format!(
                "{family} MLE shape exceeds the finite parameter range"
            )));
        }
        upper = next;
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
    Ok(lower + 0.5 * (upper - lower))
}

/// Both standardized log-logistic and inverse-Weibull likelihoods are strictly
/// concave in the positive shape. Writing t=log(x)/max|log(x)| and q=c*max|log(x)|
/// gives the monotone score q*mean(t*response(q*t)) = 1. The normalization avoids
/// arbitrary shape cutoffs and preserves the root for observations close to one.
fn unit_scale_log_shape_mle(
    data: &[f64],
    family: &str,
    response: fn(f64) -> f64,
) -> Result<f64, FitError> {
    validate_shape_sample(data, 2, family)?;
    if data.iter().any(|&x| x <= 0.0) {
        return Err(FitError::UnsupportedData(format!(
            "{family} MLE with fixed location zero requires positive observations"
        )));
    }
    let mut logs: Vec<f64> = data.iter().map(|x| x.ln()).collect();
    let scale = logs.iter().fold(0.0_f64, |largest, x| largest.max(x.abs()));
    if scale == 0.0 {
        return Err(FitError::NonConvergent(format!(
            "{family} likelihood is unbounded for observations all equal to one"
        )));
    }
    for value in &mut logs {
        *value /= scale;
    }
    let n = data.len() as f64;
    let score = |q: f64| {
        let mut sum = 0.0;
        let mut correction = 0.0;
        for &t in &logs {
            // Every term is nonnegative for both response functions. An
            // overflowing exponential is therefore a valid upper-bracket
            // signal, never a cancellation or a spurious missing root.
            let contribution = t * response(q * t) / n;
            if contribution.is_infinite() {
                return f64::INFINITY;
            }
            let term = contribution - correction;
            let next = sum + term;
            if next.is_infinite() {
                return f64::INFINITY;
            }
            correction = (next - sum) - term;
            sum = next;
        }
        q * sum
    };
    let mut lower = 0.0;
    let mut upper = 1.0;
    while score(upper) < 1.0 {
        lower = upper;
        upper *= 2.0;
        if !upper.is_finite() {
            return Err(FitError::NonConvergent(format!(
                "{family} MLE shape exceeds the finite parameter range"
            )));
        }
    }
    for _ in 0..100 {
        let middle = lower + 0.5 * (upper - lower);
        if middle == lower || middle == upper {
            break;
        }
        if score(middle) < 1.0 {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    let shape = (lower + 0.5 * (upper - lower)) / scale;
    if shape.is_finite() && shape > 0.0 {
        Ok(shape)
    } else {
        Err(FitError::NonConvergent(format!(
            "{family} MLE shape exceeds the finite parameter range"
        )))
    }
}

pub(crate) fn fisk_shape_mle(data: &[f64]) -> Result<f64, FitError> {
    unit_scale_log_shape_mle(data, "Fisk", |z| (0.5 * z).tanh())
}

pub(crate) fn invweibull_shape_mle(data: &[f64]) -> Result<f64, FitError> {
    unit_scale_log_shape_mle(data, "InvWeibull", |z| -(-z).exp_m1())
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

/// Budgets and stopping tolerances for the continuous MLE drivers.
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
/// `parameters` includes fixed values in the supplied order. The location/scale
/// helpers append location and scale after the base family's shapes.
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
    fit_mle_sample(
        MleSample::Exact(data),
        parameters,
        make_distribution,
        options,
    )
}

/// Fit a discrete likelihood model with bounded real and integer parameters.
///
/// This supplies the bounded maximum-likelihood workflow of `scipy.stats.fit`
/// for any [`DiscreteDistribution`], including families whose trait-level
/// `try_fit` is unavailable. Observations remain signed integers throughout.
/// `bounds` gives a finite closed interval for each parameter in the order
/// expected by `make_distribution`; equal bounds fix that parameter exactly.
/// Mark integer shape parameters in [`DifferentialEvolutionOptions::integrality`].
/// An empty mask means all parameters are real. The optimizer applies the
/// integer restrictions before calling the factory, including at initialization.
///
/// The factory defines the model's parameterization and must return `None` for
/// invalid parameter combinations instead of calling a panicking constructor.
/// No location or scale parameter is introduced implicitly. A model factory
/// that includes a discrete location must mark it as integral too. Parameter
/// domains cannot be inferred from a generic distribution type: callers supply
/// their intersection with the requested search bounds.
///
/// Differential evolution searches globally within those bounds, with the
/// seed, population, generation budget and objective-spread tolerance given by
/// `options`. It is stochastic and does not certify the global optimum. This
/// driver does not perform a continuous polishing step that could violate the
/// integer constraints. Exhausting the generation budget returns
/// [`FitError::NonConvergent`], even when an approximate candidate is available.
///
/// A finite support penalty guides trial distributions past out-of-support
/// observations. A successful result must have finite, nonpositive `logpmf`
/// at every observation; a penalized candidate is never reported as a fit.
/// Repeated observations are counted once and evaluated with their exact
/// multiplicities, without expanding or mutating the input. The returned
/// likelihood is the total unpenalized negative log likelihood, suitable for
/// model comparison. Hardened mode enforces the usual fit sample-size cap
/// before allocating the frequency table or invoking user code.
///
/// ```
/// use fsci_stats::mle::DifferentialEvolutionOptions;
/// use fsci_stats::{Binomial, fit_mle_discrete};
///
/// let fitted = fit_mle_discrete(
///     &[1, 3, 4, 2, 0, 5, 2, 3],
///     &[(10.0, 10.0), (0.0, 1.0)],
///     |p| Some(Binomial::new(p[0] as u64, p[1])),
///     DifferentialEvolutionOptions {
///         integrality: vec![true, false],
///         seed: Some(42),
///         tol: 1e-12,
///         ..Default::default()
///     },
/// ).unwrap();
/// assert_eq!(fitted.distribution.n, 10);
/// assert!((fitted.distribution.p - 0.25).abs() < 1e-5);
/// ```
pub fn fit_mle_discrete<D, F>(
    data: &[i64],
    bounds: &[(f64, f64)],
    make_distribution: F,
    options: DifferentialEvolutionOptions,
) -> Result<MleResult<D>, FitError>
where
    D: DiscreteDistribution,
    F: Fn(&[f64]) -> Option<D>,
{
    if active_stats_mode() == RuntimeMode::Hardened && data.len() > hardened_max_sample_len() {
        return Err(FitError::ResourceExhausted {
            detail: "Hardened mode fit sample length exceeds HARDENED_MAX_SAMPLE_LEN",
        });
    }
    if data.is_empty() {
        return Err(FitError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }
    let mut frequencies = std::collections::BTreeMap::<i64, usize>::new();
    for &observation in data {
        *frequencies.entry(observation).or_default() += 1;
    }
    let penalty = 100.0 * f64::MAX.ln();
    let n = data.len() as f64;
    let objective = |parameters: &[f64]| {
        let Some(distribution) = make_distribution(parameters) else {
            return f64::MAX / 16.0;
        };
        let mut mean_nll = 0.0;
        let mut correction = 0.0;
        for (&observation, &count) in &frequencies {
            let log_mass = distribution.logpmf(observation);
            let contribution = if log_mass.is_finite() && log_mass <= 0.0 {
                -log_mass
            } else {
                penalty
            };
            let term = (count as f64 / n) * contribution - correction;
            let next = mean_nll + term;
            correction = (next - mean_nll) - term;
            mean_nll = next;
        }
        if mean_nll.is_finite() {
            mean_nll
        } else {
            f64::MAX / 16.0
        }
    };
    let result = differential_evolution(objective, bounds, options).map_err(|error| {
        FitError::UnsupportedData(format!(
            "invalid discrete MLE search specification: {error}"
        ))
    })?;
    if !result.success {
        return Err(FitError::NonConvergent(format!(
            "{} ({} generations, {} evaluations)",
            result.message, result.nit, result.nfev
        )));
    }
    let distribution = make_distribution(&result.x).ok_or_else(|| {
        FitError::NonConvergent("discrete MLE parameters are outside the model's domain".into())
    })?;
    let mut nll = 0.0;
    let mut correction = 0.0;
    for (&observation, &count) in &frequencies {
        let log_mass = distribution.logpmf(observation);
        if !log_mass.is_finite() || log_mass > 0.0 {
            return Err(FitError::NonConvergent(format!(
                "discrete MLE has invalid or zero probability at observation {observation}"
            )));
        }
        let term = -(count as f64) * log_mass - correction;
        let next = nll + term;
        correction = (next - nll) - term;
        nll = next;
    }
    if !nll.is_finite() {
        return Err(FitError::NonConvergent(
            "discrete MLE negative log likelihood is not finite".into(),
        ));
    }
    Ok(MleResult {
        distribution,
        parameters: result.x,
        negative_log_likelihood: nll,
        iterations: result.nit,
        evaluations: result.nfev,
    })
}

/// Fit a continuous distribution to exact, left-, right-, or interval-censored
/// observations using maximum likelihood.
///
/// The likelihood contributes `logpdf(x)` for exact observations, `logcdf(x)`
/// for left censoring, `logsf(x)` for right censoring, and
/// `log(cdf(upper) - cdf(lower))` for interval censoring. Interval differences
/// use logarithms of the smaller tail to avoid cancellation and underflow.
/// When the tail logarithms themselves lose the interval's separation, a
/// bounded, error-checked quadrature of the scaled density can recover its
/// probability without exponentiating the small final result.
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

/// Fit a continuous family's shapes, location, and scale to exact observations.
///
/// `make_standard` receives only `shape_parameters`, in their supplied order,
/// and must return `None` outside the shape domain. It should construct the
/// family's standard form with any intrinsic location and scale held fixed
/// (usually zero and one). Fitting an intrinsic location/scale as a shape as
/// well as the outer location/scale generally makes the model unidentifiable.
/// The returned distribution is `location + scale * make_standard(shapes)`.
///
/// Every shape, location, and scale can be fixed with [`MleParameter::Fixed`]
/// or constrained by the other [`MleParameter`] variants. Use
/// [`MleParameter::Positive`] for a free scale; every trial scale is also
/// required to be positive and finite by [`LocationScale`]. Result parameters
/// are ordered `[shapes..., location, scale]`, including fixed values exactly.
/// Budgets, likelihood validation, runtime limits, and local convergence have
/// the same meanings as in [`fit_mle`].
///
/// ```
/// use fsci_stats::{ContinuousDistribution, GammaDist, MleOptions, MleParameter};
/// use fsci_stats::fit_mle_location_scale;
///
/// let result = fit_mle_location_scale(
///     &[3.2, 3.8, 4.5, 5.0, 6.1, 8.0],
///     &[MleParameter::Fixed(2.0)],
///     MleParameter::Fixed(3.0),
///     MleParameter::Positive(1.0),
///     |shapes| Some(GammaDist { a: shapes[0], scale: 1.0 }),
///     MleOptions::default(),
/// ).unwrap();
/// assert_eq!(result.distribution.loc(), 3.0);
/// assert_eq!(result.distribution.base().a, 2.0);
/// assert!(result.distribution.cdf(5.0) > 0.0);
/// ```
pub fn fit_mle_location_scale<D, F>(
    data: &[f64],
    shape_parameters: &[MleParameter],
    location: MleParameter,
    scale: MleParameter,
    make_standard: F,
    options: MleOptions,
) -> Result<MleResult<LocationScale<D>>, FitError>
where
    D: ContinuousDistribution,
    F: Fn(&[f64]) -> Option<D>,
{
    fit_mle_location_scale_sample(
        MleSample::Exact(data),
        shape_parameters,
        location,
        scale,
        make_standard,
        options,
    )
}

/// Fit shapes, location, and scale to exact and censored observations.
///
/// The parameterization and result ordering match [`fit_mle_location_scale`].
/// Exact observations contribute the affine density, including its `1/scale`
/// Jacobian. Censored observations contribute probabilities, so they receive
/// no density Jacobian. Logarithmic tails are evaluated in standardized
/// coordinates; finite intervals use the stable differences described in
/// [`fit_mle_censored`]. The input observations are never transformed in place.
pub fn fit_mle_location_scale_censored<D, F>(
    data: &CensoredData,
    shape_parameters: &[MleParameter],
    location: MleParameter,
    scale: MleParameter,
    make_standard: F,
    options: MleOptions,
) -> Result<MleResult<LocationScale<D>>, FitError>
where
    D: ContinuousDistribution,
    F: Fn(&[f64]) -> Option<D>,
{
    fit_mle_location_scale_sample(
        MleSample::Censored(data),
        shape_parameters,
        location,
        scale,
        make_standard,
        options,
    )
}

fn fit_mle_location_scale_sample<D, F>(
    data: MleSample<'_>,
    shape_parameters: &[MleParameter],
    location: MleParameter,
    scale: MleParameter,
    make_standard: F,
    options: MleOptions,
) -> Result<MleResult<LocationScale<D>>, FitError>
where
    D: ContinuousDistribution,
    F: Fn(&[f64]) -> Option<D>,
{
    // Check before constructing the parameter vector or calling user code,
    // matching the generic MLE driver's fail-closed sample bound.
    if active_stats_mode() == RuntimeMode::Hardened && data.len() > hardened_max_sample_len() {
        return Err(FitError::ResourceExhausted {
            detail: "Hardened mode fit sample length exceeds HARDENED_MAX_SAMPLE_LEN",
        });
    }
    let n_shapes = shape_parameters.len();
    let mut parameters = shape_parameters.to_vec();
    parameters.extend([location, scale]);
    fit_mle_sample(
        data,
        &parameters,
        |values| {
            let base = make_standard(&values[..n_shapes])?;
            LocationScale::new(base, values[n_shapes], values[n_shapes + 1]).ok()
        },
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
                "censored observations have probability one and provide no fitting information"
                    .into(),
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

#[cfg(test)]
mod interval_likelihood_tests {
    use super::*;
    use crate::{GammaDist, Normal, with_stats_runtime_mode};

    #[test]
    fn normal_intervals_keep_mass_when_both_tail_logs_round_equal() {
        let normal = Normal {
            loc: 0.0,
            scale: 1.0,
        };
        for width in [1.0e-20, 1.0e-300, 1.0e-310] {
            assert_eq!(normal.logcdf(0.0), normal.logcdf(width));
            assert_eq!(normal.logsf(0.0), normal.logsf(width));
            let actual = interval_log_probability(&normal, 0.0, width);
            // The density differs from phi(0) by O(width^2), below any f64
            // correction in these intervals. The probability itself can
            // underflow, but its logarithm is still representable.
            let expected = width.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln();
            assert!(actual.is_finite(), "width={width:e}");
            assert!(
                (actual - expected).abs() < 2.0e-12,
                "{actual} vs {expected}"
            );
        }
    }

    #[test]
    fn narrow_normal_intervals_fit_in_both_tails_and_at_extreme_scale() {
        for mode in [RuntimeMode::Strict, RuntimeMode::Hardened] {
            for (lower, upper, scale) in [
                (-40.0_f64, -40.0 + 1.0e-12, 1.0),
                (-8.0, -8.0 + 1.0e-13, 1.0),
                (8.0, 8.0 + 1.0e-13, 1.0),
                (40.0, 40.0 + 1.0e-12, 1.0),
                (0.0, 1.0, 1.0e300),
            ] {
                let normal = Normal { loc: 0.0, scale };
                let width = upper - lower;
                // Midpoint integration error is O((width/scale)^2),
                // negligible here even for the far-tail observations.
                let expected = -(width.ln() + normal.logpdf(width.mul_add(0.5, lower)));
                let data = CensoredData::interval_censored(&[lower], &[upper]).unwrap();
                let fitted = with_stats_runtime_mode(mode, || {
                    fit_mle_censored(
                        &data,
                        &[MleParameter::Fixed(0.0), MleParameter::Fixed(scale)],
                        |p| {
                            Some(Normal {
                                loc: p[0],
                                scale: p[1],
                            })
                        },
                        MleOptions::default(),
                    )
                })
                .unwrap();
                assert!(
                    (fitted.negative_log_likelihood - expected).abs() < 2.0e-10,
                    "[{lower}, {upper}], scale={scale}: {} vs {expected}",
                    fitted.negative_log_likelihood
                );
            }
        }
    }

    #[test]
    fn narrow_interval_fit_recovers_normal_location_and_scale() {
        let lower = [-2.0, -1.0, 0.5, 1.0, 3.0];
        let upper = lower.map(|x| x + 1.0e-10);
        let data = CensoredData::interval_censored(&lower, &upper).unwrap();
        let fitted = fit_mle_censored(
            &data,
            &[MleParameter::Free(0.0), MleParameter::Positive(1.0)],
            |p| {
                Some(Normal {
                    loc: p[0],
                    scale: p[1],
                })
            },
            MleOptions::default(),
        )
        .unwrap();
        let expected_loc = 0.3 + 0.5e-10;
        let expected_scale = (lower.iter().map(|x| (x - 0.3).powi(2)).sum::<f64>() / 5.0).sqrt();
        assert!((fitted.distribution.loc - expected_loc).abs() < 2.0e-7);
        assert!((fitted.distribution.scale - expected_scale).abs() < 2.0e-7);
        assert!(fitted.iterations > 0);
        let reference = Normal {
            loc: expected_loc,
            scale: expected_scale,
        };
        let expected_nll = -lower
            .iter()
            .zip(&upper)
            .map(|(&lo, &hi)| {
                let width = hi - lo;
                reference.logpdf(width.mul_add(0.5, lo)) + width.ln()
            })
            .sum::<f64>();
        assert!((fitted.negative_log_likelihood - expected_nll).abs() < 1.0e-9);
    }

    #[test]
    fn exponential_intervals_preserve_support_and_analytic_probability() {
        let exponential = GammaDist { a: 1.0, scale: 2.0 };
        for (lower, upper) in [(8.0_f64, 8.0 + 1.0e-13), (2000.0, 2000.0 + 1.0e-11)] {
            let expected = -lower / 2.0 + (-(-(upper - lower) / 2.0).exp_m1()).ln();
            let actual = interval_log_probability(&exponential, lower, upper);
            assert!(
                (actual - expected).abs() < 2.0e-10,
                "{actual} vs {expected}"
            );
        }
        for lower in [-1.0e-20, -f64::from_bits(1)] {
            assert_eq!(
                interval_log_probability(&exponential, lower, 0.0),
                f64::NEG_INFINITY
            );
        }
        let at_boundary = interval_log_probability(&exponential, 0.0, 1.0e-300);
        assert!((at_boundary - (0.5e-300_f64).ln()).abs() < 2.0e-12);
    }

    struct InvalidModel {
        log_density: f64,
        decreasing_cdf: bool,
    }

    impl ContinuousDistribution for InvalidModel {
        fn pdf(&self, _: f64) -> f64 {
            self.log_density.exp()
        }
        fn logpdf(&self, x: f64) -> f64 {
            if self.log_density == f64::INFINITY && x <= 0.75 {
                // The midpoint is valid, but quadrature must reject a model
                // that fails at other interior evaluation points.
                0.0
            } else {
                self.log_density
            }
        }
        fn cdf(&self, x: f64) -> f64 {
            self.logcdf(x).exp()
        }
        fn logcdf(&self, x: f64) -> f64 {
            if self.decreasing_cdf {
                -2.0 - x
            } else {
                -std::f64::consts::LN_2
            }
        }
        fn logsf(&self, _: f64) -> f64 {
            -std::f64::consts::LN_2
        }
        fn ppf(&self, q: f64) -> f64 {
            q
        }
        fn mean(&self) -> f64 {
            0.5
        }
        fn var(&self) -> f64 {
            1.0 / 12.0
        }
    }

    #[test]
    fn density_fallback_does_not_repair_invalid_models() {
        let decreasing = InvalidModel {
            log_density: 0.0,
            decreasing_cdf: true,
        };
        assert!(interval_log_probability(&decreasing, 0.0, 1.0).is_nan());
        for log_density in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1000.0] {
            let invalid = InvalidModel {
                log_density,
                decreasing_cdf: false,
            };
            assert!(density_interval_log_probability(&invalid, 0.0, 1.0).is_none());
            assert_eq!(
                interval_log_probability(&invalid, 0.0, 1.0),
                f64::NEG_INFINITY
            );
        }
    }
}

#[cfg(test)]
mod location_scale_fit_tests {
    use super::*;
    use crate::{Landau, Trapezoid};

    #[test]
    fn landau_mle_rejects_undefined_and_unbounded_samples() {
        assert!(matches!(
            Landau::try_fit(&[]),
            Err(FitError::InsufficientData { .. })
        ));
        assert!(matches!(
            Landau::try_fit(&[0.0, f64::NAN]),
            Err(FitError::UnsupportedData(_))
        ));
        for sample in [&[2.0, 2.0][..], &[0.0, 0.0, 1.0][..]] {
            assert!(matches!(
                Landau::try_fit(sample),
                Err(FitError::NonConvergent(_))
            ));
        }
    }

    #[test]
    fn trapezoid_mle_includes_uniform_support_boundary() {
        let fitted = Trapezoid::try_fit(&[0.0, 0.0, 1.0, 1.0]).unwrap();
        assert_eq!(fitted.c, 0.0);
        assert_eq!(fitted.d, 1.0);
        assert_eq!(fitted.loc, 0.0);
        assert_eq!(fitted.scale, 1.0);
        assert_eq!(fitted.logpdf(0.0), 0.0);
        assert_eq!(fitted.logpdf(1.0), 0.0);
    }

    #[test]
    fn trapezoid_log_likelihood_survives_density_overflow_and_underflow() {
        let tiny = Trapezoid::new(0.25, 0.75, 0.0, 1.0e-310);
        assert!(tiny.pdf(0.5e-310).is_infinite());
        let expected = (4.0_f64 / 3.0).ln() - tiny.scale.ln();
        assert!((tiny.logpdf(0.5e-310) - expected).abs() < 1.0e-12);

        let huge = Trapezoid::new(0.5, 0.5, 0.0, 1.0e308);
        assert_eq!(huge.pdf(1.0e-100), 0.0);
        let expected = 4.0_f64.ln() + 1.0e-100_f64.ln() - 2.0 * huge.scale.ln();
        assert!((huge.logpdf(1.0e-100) - expected).abs() < 1.0e-12);
        let uniform = Trapezoid::new(0.0, 1.0, 0.0, 1.0e308);
        assert_eq!(uniform.logpdf(-1.0e-100), f64::NEG_INFINITY);
        assert!(uniform.logpdf(f64::NAN).is_nan());
    }
}
