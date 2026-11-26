//! Continuous distributions defined by a user's density and support.

use crate::{ContinuousDistribution, StatsError};
use fsci_integrate::{QuadOptions, quad_full_output};

/// Specification for a scalar continuous distribution. Store the bound shape
/// parameters in the implementing type. The density must integrate to one.
/// Optional analytic callbacks avoid numerical integration for the indicated tail.
pub trait ContinuousDistributionSpec {
    /// Ordered endpoints; either endpoint may be infinite.
    fn support(&self) -> (f64, f64);
    /// Normalized density on the support, including its finite endpoints.
    fn pdf(&self, x: f64) -> f64;
    /// Optional analytic cumulative probability inside the support.
    fn cdf(&self, _x: f64) -> Option<f64> {
        None
    }
    /// Optional analytic survival probability inside the support.
    fn sf(&self, _x: f64) -> Option<f64> {
        None
    }
}

/// A user-defined distribution constructed by [`make_distribution`].
#[derive(Clone, Debug)]
pub struct CustomDistribution<S> {
    spec: S,
    lower: f64,
    upper: f64,
}

/// Construct a scalar continuous distribution from its support and density.
///
/// Like SciPy's custom continuous distribution protocol, a density alone
/// suffices. The resulting model implements [`ContinuousDistribution`],
/// including quantiles, random sampling and batch evaluation. The Rust
/// specification holds the shape parameters of one instantiated distribution.
///
/// The caller supplies a normalized nonnegative PDF. Adaptive quadrature is
/// used for probabilities without formulas and for the first two moments.
/// Failed integrals return NaN. Analytic CDF/SF callbacks are useful for
/// narrow peaks that adaptive sampling could miss.
/// Shape fitting, entropy, skewness, kurtosis and mode use the trait's
/// unsupported defaults.
///
/// # Errors
/// Returns [`StatsError::InvalidArgument`] on unordered or NaN endpoints.
///
/// # Examples
/// ```
/// use fsci_stats::{ContinuousDistribution, ContinuousDistributionSpec, make_distribution};
///
/// struct Triangle { width: f64 }
/// impl ContinuousDistributionSpec for Triangle {
///     fn support(&self) -> (f64, f64) { (0.0, self.width) }
///     fn pdf(&self, x: f64) -> f64 { 2.0 * x / self.width.powi(2) }
/// }
/// let model = make_distribution(Triangle { width: 2.0 }).unwrap();
/// assert!((model.cdf(1.0) - 0.25).abs() < 1e-10);
/// ```
pub fn make_distribution<S: ContinuousDistributionSpec>(
    spec: S,
) -> Result<CustomDistribution<S>, StatsError> {
    let (lower, upper) = spec.support();
    if lower.is_nan() || upper.is_nan() || lower >= upper {
        return Err(StatsError::InvalidArgument(
            "custom distribution support must have increasing non-NaN endpoints".into(),
        ));
    }
    Ok(CustomDistribution { spec, lower, upper })
}

impl<S: ContinuousDistributionSpec> CustomDistribution<S> {
    #[must_use]
    pub fn support(&self) -> (f64, f64) {
        (self.lower, self.upper)
    }

    fn integral(&self, a: f64, b: f64, f: impl Fn(f64) -> f64) -> f64 {
        if a == b {
            return 0.0;
        }
        let opts = QuadOptions {
            epsabs: 0.0,
            epsrel: 1e-11,
            limit: 200,
        };
        match quad_full_output(f, a, b, &[], opts) {
            Ok((result, _)) if result.converged => result.integral,
            _ => f64::NAN,
        }
    }

    fn tail(&self, x: f64, upper: bool) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x <= self.lower {
            return if upper { 1.0 } else { 0.0 };
        }
        if x >= self.upper {
            return if upper { 0.0 } else { 1.0 };
        }

        let formula = if upper {
            self.spec.sf(x)
        } else {
            self.spec.cdf(x)
        };
        if let Some(p) = formula {
            return if (0.0..=1.0).contains(&p) {
                p
            } else {
                f64::NAN
            };
        }

        let area = |right| {
            let (a, b) = if right {
                (x, self.upper)
            } else {
                (self.lower, x)
            };
            self.integral(a, b, |t| self.pdf(t))
        };
        // Prefer the nearby endpoint on finite intervals. On a half-line,
        // integrate towards infinity for large |x|: a very long finite
        // interval can hide the whole density near its distant endpoint.
        let split = match (self.lower.is_finite(), self.upper.is_finite()) {
            (true, true) => self.lower.midpoint(self.upper),
            (false, false) => 0.0,
            (true, false) => self.lower + 1.0,
            (false, true) => self.upper - 1.0,
        };
        let right = x > split;
        let near = area(right);
        let p = if right == upper {
            near
        } else if (0.0..=0.5).contains(&near) {
            1.0 - near
        } else {
            // Integrate a small requested probability directly, so subtracting
            // from one never erases a representable tail.
            area(upper)
        };
        if (-1e-11..=1.0 + 1e-11).contains(&p) {
            p.clamp(0.0, 1.0)
        } else {
            f64::NAN
        }
    }

    fn inverse(&self, q: f64, upper: bool) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }
        if q == 0.0 {
            return if upper { self.upper } else { self.lower };
        }
        if q == 1.0 {
            return if upper { self.lower } else { self.upper };
        }
        let (target, right) = if q > 0.5 {
            (1.0 - q, !upper)
        } else {
            (q, upper)
        };
        // Bisect the ordered float encodings. This needs at most 64 evaluations
        // even with infinite endpoints and resolves subnormal or translated
        // supports without relying on a finite mean or a fixed x tolerance.
        let sign = 1u64 << 63;
        let value = |k: u64| f64::from_bits(if k & sign == 0 { !k } else { k ^ sign });
        let mut lo = crate::argsort_float_key(self.lower);
        let mut hi = crate::argsort_float_key(self.upper);
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            let p = self.tail(value(mid), right);
            if !p.is_finite() {
                return f64::NAN;
            }
            if (right && p <= target) || (!right && p >= target) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        value(hi)
    }
}

impl<S: ContinuousDistributionSpec> ContinuousDistribution for CustomDistribution<S> {
    fn pdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if !x.is_finite() || x < self.lower || x > self.upper {
            return 0.0;
        }
        let density = self.spec.pdf(x);
        if density >= 0.0 { density } else { f64::NAN }
    }

    fn cdf(&self, x: f64) -> f64 {
        self.tail(x, false)
    }

    fn sf(&self, x: f64) -> f64 {
        self.tail(x, true)
    }

    fn logcdf(&self, x: f64) -> f64 {
        let p = self.cdf(x);
        if p > 0.5 {
            (-self.sf(x)).ln_1p()
        } else {
            p.ln()
        }
    }

    fn logsf(&self, x: f64) -> f64 {
        let p = self.sf(x);
        if p > 0.5 {
            (-self.cdf(x)).ln_1p()
        } else {
            p.ln()
        }
    }

    fn ppf(&self, q: f64) -> f64 {
        self.inverse(q, false)
    }

    fn isf(&self, q: f64) -> f64 {
        self.inverse(q, true)
    }

    fn interval(&self, alpha: f64) -> (f64, f64) {
        if !(0.0..=1.0).contains(&alpha) {
            return (f64::NAN, f64::NAN);
        }
        let tail = (1.0 - alpha) * 0.5;
        (self.ppf(tail), self.isf(tail))
    }

    fn rvs(&self, n: usize, rng: &mut impl rand::Rng) -> Vec<f64> {
        use rand::distr::{Distribution, Open01};
        (0..n).map(|_| self.ppf(Open01.sample(rng))).collect()
    }

    fn mean(&self) -> f64 {
        // Integrate positive and negative contributions separately. Symmetric
        // cancellation across infinity cannot define a missing first moment.
        let zero = 0.0_f64.clamp(self.lower, self.upper);
        let first = |x| x * self.pdf(x);
        self.integral(self.lower, zero, first) + self.integral(zero, self.upper, first)
    }

    fn var(&self) -> f64 {
        let mean = self.mean();
        if !mean.is_finite() {
            return f64::NAN;
        }
        let centered = |x| {
            let dx = x - mean;
            dx * self.pdf(x) * dx
        };
        let split = mean.clamp(self.lower, self.upper);
        self.integral(self.lower, split, centered) + self.integral(split, self.upper, centered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct LogUniform(f64, f64);

    impl ContinuousDistributionSpec for LogUniform {
        fn support(&self) -> (f64, f64) {
            (self.0, self.1)
        }
        fn pdf(&self, x: f64) -> f64 {
            assert!((self.0..=self.1).contains(&x));
            1.0 / (x * (self.1 / self.0).ln())
        }
    }

    #[test]
    fn custom_density_supplies_distribution_operations() {
        for (a, b) in [(1.0, 3.0), (2.0, 17.0)] {
            let d = make_distribution(LogUniform(a, b)).unwrap();
            let span = (b / a).ln();
            let mean = (b - a) / span;
            assert_eq!(d.support(), (a, b));
            assert_eq!(d.pdf(a * 0.5), 0.0);
            assert_eq!(d.cdf(b + 1.0), 1.0);
            assert_eq!(d.sf(a), 1.0);
            assert!(d.cdf(f64::NAN).is_nan());
            assert!(d.isf(-0.1).is_nan());
            assert_eq!(d.ppf(0.0), a);
            assert_eq!(d.isf(0.0), b);
            assert!((d.mean() - mean).abs() < 1e-12);
            assert!((d.var() - ((b * b - a * a) / (2.0 * span) - mean * mean)).abs() < 1e-12);
            for p in [0.01, 0.3, 0.5, 0.95] {
                let x = a * (p * span).exp();
                assert!((d.ppf(p) - x).abs() < 1e-12);
                assert!((d.isf(1.0 - p) - x).abs() < 1e-12);
                assert!((d.cdf(x) - p).abs() < 1e-12);
            }
        }
        assert!(make_distribution(LogUniform(1.0, 1.0)).is_err());
        assert!(make_distribution(LogUniform(f64::NAN, 1.0)).is_err());
    }

    struct Exp;

    impl ContinuousDistributionSpec for Exp {
        fn support(&self) -> (f64, f64) {
            (0.0, f64::INFINITY)
        }
        fn pdf(&self, x: f64) -> f64 {
            (-x).exp()
        }
    }

    #[test]
    fn semi_infinite_support_preserves_survival_and_inverse_survival() {
        let d = make_distribution(Exp).unwrap();
        assert!(d.ppf(1.0).is_infinite());
        assert!((d.mean() - 1.0).abs() < 1e-10);
        assert!((d.var() - 1.0).abs() < 1e-10);
        assert!((d.sf(80.0) / (-80.0_f64).exp() - 1.0).abs() < 1e-10);
        assert!((d.isf(1e-40) + 1e-40_f64.ln()).abs() < 1e-9);
        assert_eq!(d.cdf(1e8), 1.0);
        assert!((d.logcdf(80.0) / -(-80.0_f64).exp() - 1.0).abs() < 1e-10);
        let alpha = 1.0_f64.next_down();
        let (_, upper) = d.interval(alpha);
        assert!((upper + ((1.0 - alpha) * 0.5).ln()).abs() < 1e-10);
    }

    struct NegativeUniform;

    impl ContinuousDistributionSpec for NegativeUniform {
        fn support(&self) -> (f64, f64) {
            (-7.0, -3.0)
        }
        fn pdf(&self, _x: f64) -> f64 {
            0.25
        }
    }

    #[test]
    fn negative_support_moments_quantiles_and_samples() {
        use rand::{SeedableRng, rngs::StdRng};
        let d = make_distribution(NegativeUniform).unwrap();
        assert!((d.mean() + 5.0).abs() < 1e-12);
        assert!((d.var() - 4.0 / 3.0).abs() < 1e-12);
        assert!((d.median() + 5.0).abs() < 1e-12);
        assert_eq!(d.interval(1.0), (-7.0, -3.0));
        let mut rng = StdRng::seed_from_u64(33);
        let draws = d.rvs(8, &mut rng);
        assert_eq!(draws.len(), 8);
        assert!(draws.iter().all(|x| (-7.0..-3.0).contains(x)));
        assert!(draws.windows(2).any(|xs| xs[0] != xs[1]));
    }

    struct Formula;
    impl ContinuousDistributionSpec for Formula {
        fn support(&self) -> (f64, f64) {
            (-2.0, 2.0)
        }
        fn pdf(&self, _x: f64) -> f64 {
            panic!("analytic probabilities need no PDF")
        }
        fn cdf(&self, x: f64) -> Option<f64> {
            Some((x + 2.0) * 0.25)
        }
        fn sf(&self, x: f64) -> Option<f64> {
            Some((2.0 - x) * 0.25)
        }
    }

    #[test]
    fn analytic_callbacks_apply_only_inside_support() {
        let d = make_distribution(Formula).unwrap();
        assert_eq!(d.cdf_many(&[-3.0, -1.0, 1.0, 3.0]), [0.0, 0.25, 0.75, 1.0]);
        assert_eq!(d.sf(1.5), 0.125);
        assert!((d.ppf(0.25) + 1.0).abs() < 1e-15);
    }
}
