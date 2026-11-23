//! Regression coverage for interval-censored likelihoods whose endpoint tail
//! probabilities become indistinguishable in floating-point arithmetic.

use fsci_runtime::RuntimeMode;
use fsci_stats::{
    CensoredData, ContinuousDistribution, FitError, GammaDist, MleOptions, MleParameter, Normal,
    fit_mle_censored, with_stats_runtime_mode,
};

fn log_mass<D: ContinuousDistribution + Clone>(
    distribution: D,
    lower: f64,
    upper: f64,
) -> Result<f64, FitError> {
    let sample = CensoredData::interval_censored(&[lower], &[upper]).unwrap();
    fit_mle_censored(
        &sample,
        &[],
        |_| Some(distribution.clone()),
        MleOptions::default(),
    )
    .map(|fit| -fit.negative_log_likelihood)
}

#[test]
fn narrow_normal_intervals_recover_mass_after_tail_log_rounding() {
    let normal = Normal {
        loc: 0.0,
        scale: 1.0,
    };
    for width in [1.0e-20, 1.0e-300, 1.0e-310] {
        assert_eq!(normal.logcdf(0.0), normal.logcdf(width));
        assert_eq!(normal.logsf(0.0), normal.logsf(width));
        let actual = log_mass(normal, 0.0, width).unwrap();
        let expected = width.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln();
        assert!(
            (actual - expected).abs() < 2.0e-12,
            "{actual} vs {expected}"
        );
    }
}

#[test]
fn narrow_normal_intervals_remain_finite_in_both_tails_and_at_huge_scale() {
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
            let expected = width.ln() + normal.logpdf(width.mul_add(0.5, lower));
            let actual = with_stats_runtime_mode(mode, || log_mass(normal, lower, upper)).unwrap();
            assert!(
                (actual - expected).abs() < 2.0e-10,
                "{actual} vs {expected}"
            );
        }
    }
}

#[test]
fn narrow_interval_fit_recovers_normal_location_and_scale() {
    let lower = [-2.0, -1.0, 0.5, 1.0, 3.0];
    let upper = lower.map(|x| x + 1.0e-10);
    let sample = CensoredData::interval_censored(&lower, &upper).unwrap();
    let fit = fit_mle_censored(
        &sample,
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
    assert!((fit.distribution.loc - expected_loc).abs() < 2.0e-7);
    assert!((fit.distribution.scale - expected_scale).abs() < 2.0e-7);
    assert!(fit.iterations > 0);
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
    assert!((fit.negative_log_likelihood - expected_nll).abs() < 1.0e-9);
}

#[test]
fn exponential_intervals_preserve_support_and_analytic_probability() {
    let exponential = GammaDist { a: 1.0, scale: 2.0 };
    for (lower, upper) in [(8.0_f64, 8.0 + 1.0e-13), (2000.0, 2000.0 + 1.0e-11)] {
        let expected = -lower / 2.0 + (-(-(upper - lower) / 2.0).exp_m1()).ln();
        let actual = log_mass(exponential, lower, upper).unwrap();
        assert!(
            (actual - expected).abs() < 2.0e-10,
            "{actual} vs {expected}"
        );
    }
    for lower in [-1.0e-20, -f64::from_bits(1)] {
        assert!(matches!(
            log_mass(exponential, lower, 0.0),
            Err(FitError::NonConvergent(_))
        ));
    }
    let at_boundary = log_mass(exponential, 0.0, 1.0e-300).unwrap();
    assert!((at_boundary - (0.5e-300_f64).ln()).abs() < 2.0e-12);
}

#[derive(Clone)]
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
fn invalid_tail_and_density_models_are_not_accepted_as_fits() {
    assert!(matches!(
        log_mass(
            InvalidModel {
                log_density: 0.0,
                decreasing_cdf: true
            },
            0.0,
            1.0
        ),
        Err(FitError::NonConvergent(_))
    ));
    for log_density in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1000.0] {
        assert!(matches!(
            log_mass(
                InvalidModel {
                    log_density,
                    decreasing_cdf: false
                },
                0.0,
                1.0
            ),
            Err(FitError::NonConvergent(_))
        ));
    }
}
