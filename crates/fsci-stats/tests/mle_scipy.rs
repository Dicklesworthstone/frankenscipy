//! Public MLE regressions and live likelihood comparisons for newly fitted families.
//!
//! `FSCI_REQUIRE_SCIPY_ORACLE=1 cargo test -p fsci-stats --test mle_scipy -- --nocapture`
//! requires SciPy 1.17.1 and NumPy 2.4.3; otherwise a missing Python/SciPy installation
//! is an explicit skip. `FSCI_PYTHON` selects the interpreter, and the actual versions
//! are printed. The public regression tests do not require Python.

use std::process::Command;

use fsci_runtime::RuntimeMode;
use fsci_stats::{
    Alpha, Argus, BetaDist, Burr3, Burr12, CensoredData, Chi, ChiSquared, ContinuousDistribution,
    CrystalBall, FDistribution, FitError, FoldedCauchy, GammaDist, Kappa3, Lognormal, MleOptions,
    MleParameter, Normal, RDist, SkewCauchy, Weibull, fit_mle, fit_mle_censored,
    with_hardened_stats_caps, with_stats_runtime_mode,
};

const ORACLE: &str = r#"
import os
import numpy as np
import scipy
from scipy import stats

print('versions', scipy.__version__, np.__version__)
if os.environ.get('FSCI_REQUIRE_SCIPY_ORACLE') == '1':
    assert (scipy.__version__, np.__version__) == ('1.17.1', '2.4.3'), \
        f'required oracle is SciPy 1.17.1 / NumPy 2.4.3, got {scipy.__version__} / {np.__version__}'
cases = [
    ('burr', stats.burr, [(0.8, 1.4), (2.0, 3.0), (4.0, 0.7)]),
    ('burr12', stats.burr12, [(0.8, 1.2), (2.0, 3.0), (5.0, 0.7)]),
    ('kappa3', stats.kappa3, [(0.8,), (2.0,), (5.0,)]),
    ('foldcauchy', stats.foldcauchy, [(0.0,), (1.5,), (5.0,)]),
    ('crystalball', stats.crystalball, [(0.8, 2.0), (1.5, 3.0), (2.5, 5.0)]),
    ('gamma', stats.gamma, [(0.3,), (0.8,), (5.0,)]),
    ('beta', stats.beta, [(0.3, 0.6), (2.0, 5.0), (8.0, 0.7)]),
    ('weibull_min', stats.weibull_min, [(0.4,), (1.5,), (6.0,)]),
    ('lognorm', stats.lognorm, [(0.3,), (0.9,), (1.5,)]),
    ('skewcauchy', stats.skewcauchy, [(-0.7,), (0.0,), (0.6,)]),
    ('alpha', stats.alpha, [(0.2,), (2.0,), (6.0,)]),
    ('argus', stats.argus, [(0.8,), (2.0,), (5.0,)]),
    ('f', stats.f, [(1.0, 1.5), (5.0, 10.0), (30.0, 8.0)]),
    ('chi2', stats.chi2, [(0.3,), (4.0,), (50.0,)]),
    ('chi', stats.chi, [(0.3,), (4.0,), (50.0,)]),
    ('rdist', stats.rdist, [(1.5,), (5.0,), (30.0,)]),
]
for family_index, (name, dist, shapes) in enumerate(cases):
    for case_index, shape in enumerate(shapes):
        scale = 2.0 if name in ('gamma', 'weibull_min', 'lognorm') else 1.0
        data = dist.rvs(*shape, scale=scale, size=2000,
                        random_state=np.random.default_rng(12700 + 17*family_index + case_index))
        kwargs = {'floc': 0}
        if name not in ('gamma', 'weibull_min', 'lognorm'):
            kwargs['fscale'] = 1
        fitted = dist.fit(data, **kwargs)
        nll = -np.sum(dist.logpdf(data, *fitted[:-2], loc=fitted[-2], scale=fitted[-1]))
        parameters = fitted[:-2]
        if name in ('gamma', 'weibull_min', 'lognorm'):
            parameters = (*parameters, fitted[-1])
        print(name, case_index, len(parameters), *(format(x, '.17g') for x in parameters),
              format(nll, '.17g'))
        print(' '.join(format(x, '.17g') for x in data))
"#;

fn likelihood<D: ContinuousDistribution>(distribution: &D, data: &[f64]) -> f64 {
    -data.iter().map(|&x| distribution.logpdf(x)).sum::<f64>()
}

#[test]
fn shape_mle_maximizes_likelihood_instead_of_matching_moments() {
    // The roots were independently evaluated at 60 digits from the likelihood
    // scores; the likelihoods also agree with SciPy 1.17's fixed-loc/scale fit.
    let chi2_data = [2.0, 4.0, 6.0];
    let chi2 = ChiSquared::try_fit(&chi2_data).unwrap();
    assert!((chi2.df - 4.589_741_071_010_053).abs() < 1.0e-12);
    assert!((likelihood(&chi2, &chi2_data) - 6.212_701_145_212_076).abs() < 1.0e-12);
    assert!(likelihood(&chi2, &chi2_data) + 0.07 < likelihood(&ChiSquared::new(4.0), &chi2_data));
    assert_eq!(ChiSquared::fit(&chi2_data), chi2);

    let chi_data = [0.5_f64, 1.0, 1.5, 2.0, 2.5, 3.0, 1.7, 1.3];
    let chi = Chi::try_fit(&chi_data).unwrap();
    let moment_df = chi_data.iter().map(|x| x * x).sum::<f64>() / chi_data.len() as f64;
    assert!((chi.df - 3.163_339_845_788_104_4).abs() < 1.0e-12);
    assert!((likelihood(&chi, &chi_data) - 9.020_813_538_035_092).abs() < 1.0e-12);
    assert!(likelihood(&chi, &chi_data) + 0.05 < likelihood(&Chi::new(moment_df), &chi_data));

    let point = 0.2_f64.sqrt();
    let rdist_data = [-point, point];
    let rdist = RDist::try_fit(&rdist_data).unwrap();
    assert!((rdist.c - 4.927_460_070_166_763).abs() < 1.0e-12);
    assert!((likelihood(&rdist, &rdist_data) - 0.997_103_731_389_696_2).abs() < 1.0e-12);
    assert!(likelihood(&rdist, &rdist_data) + 0.02 < likelihood(&RDist::new(4.0), &rdist_data));
}

#[test]
fn shape_mle_preserves_chi_square_transformation_and_identifiable_repeated_samples() {
    let data = [0.1_f64, 0.3, 0.7, 2.0, 4.0, 8.0];
    let squared = data.map(|x| x * x);
    let chi = Chi::try_fit(&data).unwrap();
    let chi2 = ChiSquared::try_fit(&squared).unwrap();
    assert!((chi.df - chi2.df).abs() < 1.0e-13);

    // A constant sample can still identify a shape when location and scale
    // are fixed. RDist's previous variance inversion incorrectly rejected it.
    let rdist = RDist::try_fit(&[0.5; 8]).unwrap();
    let nll = likelihood(&rdist, &[0.5; 8]);
    for shape in [rdist.c * 0.99, rdist.c * 1.01] {
        assert!(nll < likelihood(&RDist::new(shape), &[0.5; 8]));
    }
    assert!(Chi::try_fit(&[2.0; 8]).unwrap().df.is_finite());
    assert!(ChiSquared::try_fit(&[2.0; 8]).unwrap().df.is_finite());
}

#[test]
fn shape_mle_handles_extreme_sufficient_statistics() {
    let data = [1.0e-300_f64, 1.0e-200, 1.0e-100];
    for (power, df) in [
        (1.0, ChiSquared::try_fit(&data).unwrap().df),
        (2.0, Chi::try_fit(&data).unwrap().df),
    ] {
        let target =
            power * data.iter().map(|x| x.ln()).sum::<f64>() / 3.0 - std::f64::consts::LN_2;
        assert!((fsci_special::digamma_scalar(df / 2.0) - target).abs() < 1.0e-10);
    }
    // Subtracting two digammas, or taking log(1-x*x), loses this score.
    for x in [1.0e-8_f64, 1.0e-100] {
        let fitted = RDist::try_fit(&[-x, x]).unwrap();
        assert!(fitted.c.is_finite());
        assert!((fitted.c * x * x - 1.0).abs() < 5.0e-15);
    }
    // 80-digit evaluation at the exact f64 input. Forming 1-x*x first
    // introduces a 2.75e-7 likelihood error at this support boundary.
    let concentrated = RDist::new(1000.0);
    assert_eq!(concentrated.pdf(0.999_999_99), 0.0);
    assert!((concentrated.logpdf(0.999_999_99) + 8_843.504_559_014_134).abs() < 2.0e-11);
    // A logarithmic sufficient statistic remains valid even when summing
    // the sample, or its squares, overflows.
    assert!(ChiSquared::try_fit(&[1.0e307; 20]).unwrap().df.is_finite());
    assert!(Chi::try_fit(&[1.0e153; 20]).unwrap().df.is_finite());
    assert!(matches!(
        Chi::try_fit(&[1.0e200; 2]),
        Err(FitError::NonConvergent(_))
    ));
}

#[test]
fn shape_mle_rejects_invalid_samples_and_unbounded_likelihoods() {
    for data in [
        &[0.0, 1.0][..],
        &[-1.0, 1.0],
        &[f64::NAN, 1.0],
        &[f64::INFINITY, 1.0],
    ] {
        assert!(matches!(
            Chi::try_fit(data),
            Err(FitError::UnsupportedData(_))
        ));
        assert!(matches!(
            ChiSquared::try_fit(data),
            Err(FitError::UnsupportedData(_))
        ));
        assert!(ChiSquared::fit(data).df.is_nan());
    }
    assert!(matches!(
        ChiSquared::try_fit(&[]),
        Err(FitError::InsufficientData { .. })
    ));
    for data in [&[-1.0, 0.0][..], &[0.0, 1.0], &[0.0, f64::NAN]] {
        assert!(matches!(
            RDist::try_fit(data),
            Err(FitError::UnsupportedData(_))
        ));
    }
    assert!(matches!(
        RDist::try_fit(&[0.0; 8]),
        Err(FitError::NonConvergent(_))
    ));
    with_stats_runtime_mode(RuntimeMode::Hardened, || {
        with_hardened_stats_caps(1, 1, || {
            assert!(matches!(
                Chi::try_fit(&[1.0, 2.0]),
                Err(FitError::ResourceExhausted { .. })
            ));
            assert!(matches!(
                ChiSquared::try_fit(&[1.0, 2.0]),
                Err(FitError::ResourceExhausted { .. })
            ));
            assert!(matches!(
                RDist::try_fit(&[0.1, 0.2]),
                Err(FitError::ResourceExhausted { .. })
            ));
        });
    });
}

#[test]
fn f_distribution_mle_fits_heavy_tails_without_finite_sample_moments() {
    // SciPy F fitting uses likelihood, not the existence of the family's
    // first two moments. These F MLEs have infinite mean/variance or variance.
    // References: scipy.stats.f.fit(data, floc=0, fscale=1), SciPy 1.17.0.
    let cases = [
        (
            [
                0.02, 0.05, 0.1, 0.15, 0.3, 0.5, 0.8, 1.5, 3.0, 8.0, 25.0, 100.0,
            ],
            [1.246_044_954_941_134_5, 1.209_589_814_253_467_3],
            26.569_364_618_839_074,
        ),
        (
            [0.2, 0.35, 0.5, 0.7, 0.9, 1.0, 1.2, 1.4, 1.8, 2.5, 4.0, 8.0],
            [7.199_934_851_824_194, 3.839_962_291_413_743_7],
            18.568_068_233_416_327,
        ),
    ];
    for (data, expected, expected_nll) in cases {
        let fitted = FDistribution::try_fit(&data).expect("finite heavy-tail F MLE");
        assert!(
            fitted.dfd < 4.0,
            "MLE must allow infinite-variance families"
        );
        for (actual, reference) in [fitted.dfn, fitted.dfd].into_iter().zip(expected) {
            assert!(
                (actual - reference).abs() < 2.0e-4,
                "{actual} != {reference}"
            );
        }
        let nll = likelihood(&fitted, &data);
        assert!(nll.is_finite() && nll <= expected_nll + 1.0e-9);
        let infallible = FDistribution::fit(&data);
        assert_eq!(infallible, fitted);
    }
}

#[test]
fn f_distribution_mle_rejects_invalid_samples() {
    for data in [&[][..], &[1.0], &[0.0, 1.0], &[-1.0, 2.0], &[1.0, f64::NAN]] {
        assert!(FDistribution::try_fit(data).is_err());
        let fitted = FDistribution::fit(data);
        assert!(fitted.dfn.is_nan() && fitted.dfd.is_nan());
    }
    assert!(matches!(
        FDistribution::try_fit(&[1.0, 1.0, 1.0]),
        Err(FitError::NonConvergent(_))
    ));
}

#[test]
fn f_distribution_likelihood_survives_extreme_observations() {
    let distribution = FDistribution::new(5.0, 10.0);
    // 100-digit Decimal evaluation with B(5/2,5) = 24 / product(5/2 .. 13/2).
    // The direct dfn*x expression overflows at 1e308 although logpdf is finite.
    for (x, expected) in [
        (1.0e308, -4_247.639_888_414_259),
        (1.0e-300, -1_033.824_532_262_782_4),
    ] {
        let actual = distribution.logpdf(x);
        assert!(actual.is_finite());
        assert!(
            (actual - expected).abs() < 1.0e-11,
            "{actual} != {expected}"
        );
    }
    assert_eq!(distribution.pdf(f64::INFINITY), 0.0);
    assert_eq!(distribution.logpdf(f64::INFINITY), f64::NEG_INFINITY);
    assert!(distribution.pdf(f64::NAN).is_nan());
    assert!(distribution.logpdf(f64::NAN).is_nan());
}

#[test]
fn new_shape_fits_reject_invalid_samples_and_nonexistent_interior_maxima() {
    assert!(matches!(
        Alpha::try_fit(&[0.5]),
        Err(FitError::InsufficientData { .. })
    ));
    assert!(matches!(
        Argus::try_fit(&[0.5]),
        Err(FitError::InsufficientData { .. })
    ));
    assert!(matches!(
        SkewCauchy::try_fit(&[0.5]),
        Err(FitError::InsufficientData { .. })
    ));
    for data in [&[0.0, 0.5][..], &[-1.0, 0.5], &[f64::NAN, 0.5]] {
        assert!(matches!(
            Alpha::try_fit(data),
            Err(FitError::UnsupportedData(_))
        ));
        assert!(matches!(
            Argus::try_fit(data),
            Err(FitError::UnsupportedData(_))
        ));
    }
    assert!(matches!(
        Argus::try_fit(&[0.5, 1.0]),
        Err(FitError::UnsupportedData(_))
    ));
    assert!(matches!(
        SkewCauchy::try_fit(&[-1.0, f64::INFINITY]),
        Err(FitError::UnsupportedData(_))
    ));
    assert!(matches!(
        Alpha::try_fit(&[3.0, 4.0, 5.0]),
        Err(FitError::NonConvergent(_))
    ));
    assert!(matches!(
        Argus::try_fit(&[0.1, 0.2, 0.3]),
        Err(FitError::NonConvergent(_))
    ));
    for data in [&[0.0, 0.0][..], &[1.0, 2.0], &[-2.0, -1.0]] {
        assert!(matches!(
            SkewCauchy::try_fit(data),
            Err(FitError::NonConvergent(_))
        ));
    }
}

#[test]
fn shape_fits_allow_repeated_data_when_the_shape_is_identifiable() {
    let alpha = Alpha::try_fit(&[0.5; 8]).unwrap();
    let normal = Normal::standard();
    let score = 2.0 - alpha.a - normal.pdf(alpha.a) / normal.cdf(alpha.a);
    assert!(score.abs() < 1.0e-6, "Alpha score at MLE: {score}");

    let argus = Argus::try_fit(&[0.9; 8]).unwrap();
    let fitted_likelihood = likelihood(&argus, &[0.9; 8]);
    for chi in [argus.chi * 0.99, argus.chi * 1.01] {
        assert!(fitted_likelihood < likelihood(&Argus::new(chi), &[0.9; 8]));
    }

    let skew = SkewCauchy::try_fit(&[-2.0, -0.5, 0.0, 0.5, 2.0]).unwrap();
    assert!(skew.a.abs() < 1.0e-6);
}

#[test]
fn argus_small_shape_preserves_density_normalization_and_probability_tails() {
    // At χ→0 the normalized limiting density is 3x sqrt(1-x²),
    // with survival probability (1-x²)^(3/2).
    for chi in [1.0e-150, 1.0e-8] {
        let distribution = Argus::new(chi);
        for x in [0.1, 0.5, 0.9, 1.0 - 1.0e-12] {
            let t = (1.0_f64 - x) * (1.0 + x);
            let expected_pdf = 3.0 * x * t.sqrt();
            let expected_sf = t.powf(1.5);
            assert!((distribution.pdf(x) / expected_pdf - 1.0).abs() < 1.0e-13);
            assert!((distribution.sf(x) / expected_sf - 1.0).abs() < 1.0e-13);
            assert!((distribution.logsf(x) - expected_sf.ln()).abs() < 1.0e-13);
            assert!((distribution.cdf(x) + distribution.sf(x) - 1.0).abs() < 1.0e-15);
            assert!((distribution.logcdf(x).exp() - distribution.cdf(x)).abs() < 1.0e-15);
        }
        assert_eq!(distribution.cdf(0.0), 0.0);
        assert_eq!(distribution.cdf(1.0), 1.0);
        assert_eq!(distribution.sf(0.0), 1.0);
        assert_eq!(distribution.sf(1.0), 0.0);
        assert!(distribution.logpdf(f64::NAN).is_nan());
        assert!(distribution.logsf(f64::NAN).is_nan());
    }
}

#[test]
fn new_shape_likelihoods_remain_finite_when_pdf_underflows() {
    let skew = SkewCauchy::new(0.3);
    for x in [-1.0e200_f64, 1.0e200] {
        let side_scale = 1.0 + 0.3 * x.signum();
        let expected = -std::f64::consts::PI.ln() - 2.0 * (x.abs() / side_scale).ln();
        assert!(skew.logpdf(x).is_finite());
        assert!((skew.logpdf(x) - expected).abs() < 1.0e-12);
    }
    let argus = Argus::new(100.0);
    assert_eq!(argus.pdf(0.5), 0.0);
    assert!(argus.logpdf(0.5).is_finite());
    assert!(Alpha::new(2.0).logpdf(1.0e200).is_finite());

    // χ² overflows, but χ²(1-x²) remains representable near the endpoint.
    let x = f64::from_bits(1.0_f64.to_bits() - 1);
    let value = Argus::new(1.0e160).logpdf(x);
    assert!(value.is_finite());
    assert!((value / -1.110_223_024_625_156_4e304 - 1.0).abs() < 1.0e-14);
}

#[test]
fn mle_log_densities_remain_finite_when_density_powers_overflow() {
    // The power-law references were evaluated with 80-digit Decimal
    // arithmetic. SciPy 1.17.0 itself overflows several of these tails.
    let rows = [
        (
            Burr3::new(10.0, 4.0).logpdf(1.0e-100),
            -8_976.392_983_222_664,
        ),
        (
            Burr3::new(10.0, 4.0).logpdf(1.0e100),
            -2_529.154_722_839_336_5,
        ),
        (
            Burr12::new(10.0, 4.0).logpdf(1.0e-100),
            -2_068.637_704_240_527,
        ),
        (
            Burr12::new(10.0, 4.0).logpdf(1.0e100),
            -9_436.910_001_821_474,
        ),
        (Kappa3::new(10.0).logpdf(1.0e100), -2_530.541_017_200_456_3),
        (
            FoldedCauchy::new(2.0).logpdf(1.0e200),
            -921.485_619_902_907_7,
        ),
        // Independent 100-digit normalization and tail integral at
        // beta=.5, m=1000. Direct large-power formulas lose precision.
        (
            CrystalBall::new(0.5, 1000.0).logpdf(0.0),
            -1.252_762_946_212_599_3,
        ),
        (
            CrystalBall::new(0.5, 1000.0).logpdf(-10.0),
            -6.116_517_293_386_037,
        ),
    ];
    for (actual, expected) in rows {
        assert!(actual.is_finite());
        assert!(
            (actual - expected).abs() <= 2.0e-15 * expected.abs().max(1.0),
            "logpdf {actual} != {expected}"
        );
    }
}

#[test]
fn newly_supported_families_return_typed_fit_errors() {
    for data in [&[][..], &[f64::NAN, 1.0][..], &[f64::INFINITY, 2.0][..]] {
        assert!(Burr3::try_fit(data).is_err());
        assert!(Burr12::try_fit(data).is_err());
        assert!(Kappa3::try_fit(data).is_err());
        assert!(FoldedCauchy::try_fit(data).is_err());
        assert!(CrystalBall::try_fit(data).is_err());
    }
    assert!(Burr3::try_fit(&[-1.0, 2.0]).is_err());
    assert!(Burr12::try_fit(&[0.0, 2.0]).is_err());
    assert!(Kappa3::try_fit(&[-1.0, 2.0]).is_err());
    assert!(FoldedCauchy::try_fit(&[-1.0, 2.0]).is_err());
}

#[test]
fn mle_normal_matches_closed_form_and_holds_location_exactly() {
    let data = [-2.5, -0.3, 0.2, 1.7, 2.9, 3.2, 5.4];
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    let fitted = fit_mle(
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
    assert!((fitted.distribution.loc - mean).abs() < 2.0e-7);
    assert!((fitted.distribution.scale - variance.sqrt()).abs() < 2.0e-7);
    assert!(fitted.evaluations > 0);

    let fixed_loc = -0.0_f64;
    let fixed = fit_mle(
        &data,
        &[MleParameter::Fixed(fixed_loc), MleParameter::Positive(1.0)],
        |p| {
            Some(Normal {
                loc: p[0],
                scale: p[1],
            })
        },
        MleOptions::default(),
    )
    .unwrap();
    let scale = (data.iter().map(|x| x * x).sum::<f64>() / data.len() as f64).sqrt();
    assert_eq!(fixed.parameters[0].to_bits(), fixed_loc.to_bits());
    assert_eq!(fixed.distribution.loc.to_bits(), fixed_loc.to_bits());
    assert!((fixed.distribution.scale - scale).abs() < 2.0e-7);
}

#[test]
fn mle_fixed_gamma_shape_has_analytic_scale_mle() {
    let data = [0.001, 0.01, 0.02, 0.1, 0.3, 0.6, 1.7, 4.0];
    let shape = 0.3;
    let fitted = fit_mle(
        &data,
        &[MleParameter::Fixed(shape), MleParameter::Positive(1.0)],
        |p| {
            Some(GammaDist {
                a: p[0],
                scale: p[1],
            })
        },
        MleOptions::default(),
    )
    .unwrap();
    let expected = data.iter().sum::<f64>() / (data.len() as f64 * shape);
    assert_eq!(fitted.distribution.a, shape);
    assert!((fitted.distribution.scale - expected).abs() < 2.0e-7);
    let expected_nll = -data
        .iter()
        .map(|&x| fitted.distribution.logpdf(x))
        .sum::<f64>();
    assert!((fitted.negative_log_likelihood - expected_nll).abs() < 1.0e-12);
}

#[test]
fn mle_bounded_parameter_can_attain_boundary() {
    let fitted = fit_mle(
        &[3.0, 4.0, 5.0],
        &[
            MleParameter::Bounded {
                initial: 0.0,
                lower: -1.0,
                upper: 1.0,
            },
            MleParameter::Fixed(2.0),
        ],
        |p| {
            Some(Normal {
                loc: p[0],
                scale: p[1],
            })
        },
        MleOptions::default(),
    )
    .unwrap();
    assert_eq!(fitted.distribution.loc, 1.0);
    assert_eq!(fitted.distribution.scale, 2.0);
}

#[test]
fn mle_rejects_invalid_data_domains_and_budget_exhaustion() {
    let params = [MleParameter::Positive(1.0), MleParameter::Positive(1.0)];
    let factory = |p: &[f64]| {
        Some(GammaDist {
            a: p[0],
            scale: p[1],
        })
    };
    assert!(matches!(
        fit_mle(&[], &params, factory, MleOptions::default()),
        Err(FitError::InsufficientData { .. })
    ));
    for data in [&[f64::NAN][..], &[f64::INFINITY][..], &[-1.0, -2.0][..]] {
        assert!(fit_mle(data, &params, factory, MleOptions::default()).is_err());
    }
    let invalid = [MleParameter::Positive(0.0), MleParameter::Positive(1.0)];
    assert!(fit_mle(&[0.1, 0.5, 2.0], &invalid, factory, MleOptions::default()).is_err());
    assert!(matches!(
        fit_mle(
            &[0.1, 0.5, 2.0],
            &params,
            factory,
            MleOptions {
                max_evaluations: 1,
                ..MleOptions::default()
            },
        ),
        Err(FitError::NonConvergent(_))
    ));
}

#[test]
fn mle_all_fixed_evaluates_without_optimization_and_enforces_support() {
    let factory = |p: &[f64]| {
        Some(GammaDist {
            a: p[0],
            scale: p[1],
        })
    };
    let params = [MleParameter::Fixed(2.0), MleParameter::Fixed(3.0)];
    let fit = fit_mle(&[0.2, 1.0, 5.0], &params, factory, MleOptions::default()).unwrap();
    assert_eq!(fit.parameters, [2.0, 3.0]);
    assert_eq!(fit.iterations, 0);
    assert_eq!(fit.evaluations, 1);
    assert!(fit_mle(&[-1.0], &params, factory, MleOptions::default()).is_err());
}

#[test]
fn mle_hardened_enforces_sample_cap() {
    with_stats_runtime_mode(RuntimeMode::Hardened, || {
        with_hardened_stats_caps(2, 2, || {
            let result = fit_mle(
                &[0.0, 1.0, 2.0],
                &[MleParameter::Free(0.0), MleParameter::Positive(1.0)],
                |p| {
                    Some(Normal {
                        loc: p[0],
                        scale: p[1],
                    })
                },
                MleOptions::default(),
            );
            assert!(matches!(result, Err(FitError::ResourceExhausted { .. })));
        });
    });
}

#[test]
fn kappa3_endpoint_small_shapes_and_repeated_sample_mle() {
    // At x=0 the exact density is a^(-1/a), including the endpoint in
    // SciPy's support. It is not zero, and zero observations are valid inputs.
    for (a, density) in [
        (0.5, 4.0),
        (1.0, 1.0),
        (2.0, std::f64::consts::FRAC_1_SQRT_2),
    ] {
        let distribution = Kappa3::new(a);
        assert!((distribution.pdf(0.0) - density).abs() < 2.0e-15);
        assert!((distribution.logpdf(0.0) - density.ln()).abs() < 2.0e-15);
        assert_eq!(distribution.pdf(-1.0), 0.0);
        assert_eq!(distribution.logpdf(-1.0), f64::NEG_INFINITY);
        assert!(distribution.pdf(f64::NAN).is_nan());
        assert!(distribution.logpdf(f64::NAN).is_nan());
    }
    // Independent 400-digit Decimal evaluations of
    // ln(a) - (1+a)*ln(1+a)/a. Factoring this through log(a)+softplus(...)
    // loses approximately one nat for a=1e-20; forming 1/a overflows for
    // the final subnormal shape even though its log density is finite.
    for (a, log_density) in [
        (1.0e-12, -28.631_021_115_929_047),
        (1.0e-20, -47.051_701_859_880_914),
        (1.0e-310, -714.801_378_828_154_2),
    ] {
        let distribution = Kappa3::new(a);
        let actual = distribution.logpdf(1.0);
        assert!(actual.is_finite());
        assert!(
            (actual - log_density).abs() <= 3.0e-13,
            "a={a}: {actual} != {log_density}"
        );
        assert!((distribution.pdf(1.0) / log_density.exp() - 1.0).abs() <= 5.0e-13);
    }

    // SciPy 1.17.1 / NumPy 2.4.3, floc=0, fscale=1. A constant sample
    // need not be degenerate when only a shape parameter is fitted.
    for (data, expected_shape, expected_nll) in [
        (
            [2.0, 2.0, 2.0],
            2.017_773_437_500_002_5,
            5.983_348_312_569_074,
        ),
        ([0.0, 1.0, 2.0], 2.857_421_875_000_004, 3.212_509_245_717_98),
    ] {
        let fitted = Kappa3::try_fit(&data).expect("valid shape-only Kappa3 MLE");
        let nll = likelihood(&fitted, &data);
        assert!(fitted.a.is_finite() && fitted.a > 0.0);
        assert!(
            (fitted.a - expected_shape).abs() < 1.0e-4,
            "shape {} != {expected_shape}",
            fitted.a
        );
        assert!(
            nll.is_finite() && nll <= expected_nll + 1.0e-9,
            "Kappa3 likelihood {nll} worse than SciPy {expected_nll}"
        );
    }
}

#[test]
fn crystal_ball_large_shape_density_and_probabilities_are_usable() {
    let distribution = CrystalBall::new(0.5, 1000.0);
    // Gaussian-arm anchors match SciPy 1.17.1 / NumPy 2.4.3. The power-tail
    // anchors use 100-digit evaluation of its analytic density and integral:
    // SciPy's CDF itself overflows on that arm at these shapes.
    for (x, density, cdf) in [
        (-10.0, 0.002_206_125_888_462_988, 0.004_437_647_620_486_86),
        (-0.5, 0.252_141_977_785_448_6, 0.504_788_744_315_212_4),
        (0.0, 0.285_714_292_080_791_2, 0.641_910_238_502_054_4),
        (1.0, 0.173_294_478_065_090_32, 0.886_374_355_918_577_2),
    ] {
        let actual_pdf = distribution.pdf(x);
        let actual_cdf = distribution.cdf(x);
        let actual_sf = distribution.sf(x);
        assert!(actual_pdf.is_finite() && actual_pdf > 0.0);
        assert!(
            (actual_pdf / density - 1.0).abs() < 1.0e-12,
            "pdf({x})={actual_pdf}"
        );
        assert!((actual_cdf - cdf).abs() < 2.0e-14, "cdf({x})={actual_cdf}");
        assert!((actual_cdf + actual_sf - 1.0).abs() < 2.0e-15);
    }
    let small_right_tail = distribution.sf(10.0);
    assert!(small_right_tail > 0.0);
    assert!((small_right_tail / 5.457_182_704_142_045e-24 - 1.0).abs() < 1.0e-12);
    assert_eq!(distribution.cdf(f64::NEG_INFINITY), 0.0);
    assert_eq!(distribution.cdf(f64::INFINITY), 1.0);
    assert_eq!(distribution.sf(f64::NEG_INFINITY), 1.0);
    assert_eq!(distribution.sf(f64::INFINITY), 0.0);

    // A large transition parameter must retain the small Gaussian CDF on
    // the negative half-line instead of computing it as 1 - sf.
    let near_normal = CrystalBall::new(20.0, 1000.0);
    assert!((near_normal.cdf(-10.0) / 7.619_853_024_160_47e-24 - 1.0).abs() < 1.0e-12);
}

#[test]
fn mle_matches_live_scipy_on_seeded_fits() {
    let python = std::env::var("FSCI_PYTHON").unwrap_or_else(|_| "python3".into());
    let required = std::env::var("FSCI_REQUIRE_SCIPY_ORACLE").as_deref() == Ok("1");
    let available = Command::new(&python)
        .args(["-c", "import numpy; import scipy"])
        .output()
        .is_ok_and(|output| output.status.success());
    if !available {
        assert!(
            !required,
            "required SciPy oracle is unavailable via {python}"
        );
        eprintln!("SKIP mle_scipy: SciPy oracle unavailable via {python}; compared cases = 0");
        return;
    }
    let output = Command::new(&python).args(["-c", ORACLE]).output().unwrap();
    assert!(
        output.status.success(),
        "SciPy MLE oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let mut lines = stdout.lines();
    let versions = lines.next().expect("oracle version metadata");
    assert!(versions.starts_with("versions "));
    eprintln!("MLE oracle {versions}");
    if required {
        assert_eq!(
            versions, "versions 1.17.1 2.4.3",
            "required oracle must match both pinned versions"
        );
    }
    let mut compared = 0;
    while let Some(header) = lines.next() {
        let mut fields = header.split_whitespace();
        let family = fields.next().unwrap();
        let case = fields.next().unwrap();
        let nparams: usize = fields.next().unwrap().parse().unwrap();
        let expected: Vec<f64> = fields
            .by_ref()
            .take(nparams)
            .map(|value| value.parse().unwrap())
            .collect();
        let expected_nll: f64 = fields.next().unwrap().parse().unwrap();
        assert!(
            expected_nll.is_finite(),
            "{family}/{case}: nonfinite oracle likelihood"
        );
        let data: Vec<f64> = lines
            .next()
            .expect("oracle sample data")
            .split_whitespace()
            .map(|value| value.parse().unwrap())
            .collect();
        assert_eq!(data.len(), 2000);
        let two_positive = [MleParameter::Positive(1.0), MleParameter::Positive(1.0)];
        let options = MleOptions::default();
        let (actual, nll) = match family {
            "chi2" => {
                let fitted = ChiSquared::try_fit(&data).unwrap();
                (vec![fitted.df], likelihood(&fitted, &data))
            }
            "chi" => {
                let fitted = Chi::try_fit(&data).unwrap();
                (vec![fitted.df], likelihood(&fitted, &data))
            }
            "rdist" => {
                let fitted = RDist::try_fit(&data).unwrap();
                (vec![fitted.c], likelihood(&fitted, &data))
            }
            "f" => {
                let fitted = FDistribution::try_fit(&data).unwrap();
                (vec![fitted.dfn, fitted.dfd], likelihood(&fitted, &data))
            }
            "skewcauchy" => {
                let fitted = SkewCauchy::try_fit(&data).unwrap();
                (vec![fitted.a], likelihood(&fitted, &data))
            }
            "alpha" => {
                let fitted = Alpha::try_fit(&data).unwrap();
                (vec![fitted.a], likelihood(&fitted, &data))
            }
            "argus" => {
                let fitted = Argus::try_fit(&data).unwrap();
                (vec![fitted.chi], likelihood(&fitted, &data))
            }
            "burr" => {
                let fitted = Burr3::try_fit(&data).unwrap();
                (vec![fitted.c, fitted.d], likelihood(&fitted, &data))
            }
            "burr12" => {
                let fitted = Burr12::try_fit(&data).unwrap();
                (vec![fitted.c, fitted.d], likelihood(&fitted, &data))
            }
            "kappa3" => {
                let fitted = Kappa3::try_fit(&data).unwrap();
                (vec![fitted.a], likelihood(&fitted, &data))
            }
            "foldcauchy" => {
                let fitted = FoldedCauchy::try_fit(&data).unwrap();
                (vec![fitted.c], likelihood(&fitted, &data))
            }
            "crystalball" => {
                let fitted = CrystalBall::try_fit(&data).unwrap();
                (
                    vec![fitted.beta_param, fitted.m],
                    likelihood(&fitted, &data),
                )
            }
            "gamma" => {
                let fitted = fit_mle(
                    &data,
                    &two_positive,
                    |p| {
                        Some(GammaDist {
                            a: p[0],
                            scale: p[1],
                        })
                    },
                    options,
                )
                .unwrap();
                (fitted.parameters, fitted.negative_log_likelihood)
            }
            "beta" => {
                let fitted = fit_mle(
                    &data,
                    &two_positive,
                    |p| Some(BetaDist { a: p[0], b: p[1] }),
                    options,
                )
                .unwrap();
                (fitted.parameters, fitted.negative_log_likelihood)
            }
            "weibull_min" => {
                let fitted = fit_mle(
                    &data,
                    &two_positive,
                    |p| {
                        Some(Weibull {
                            c: p[0],
                            scale: p[1],
                        })
                    },
                    options,
                )
                .unwrap();
                (fitted.parameters, fitted.negative_log_likelihood)
            }
            "lognorm" => {
                let fitted = fit_mle(
                    &data,
                    &two_positive,
                    |p| {
                        Some(Lognormal {
                            s: p[0],
                            scale: p[1],
                        })
                    },
                    options,
                )
                .unwrap();
                (fitted.parameters, fitted.negative_log_likelihood)
            }
            _ => panic!("unexpected family from oracle: {family}"),
        };
        assert!(nll.is_finite());
        assert_eq!(actual.len(), expected.len());
        assert!(
            nll <= expected_nll + 1.0e-6 * expected_nll.abs().max(1.0),
            "{family}/{case}: fsci NLL {nll} worse than SciPy {expected_nll}; \
             fsci parameters {actual:?}, SciPy {expected:?}"
        );
        if matches!(family, "gamma" | "beta" | "weibull_min" | "lognorm") {
            for (&value, &reference) in actual.iter().zip(&expected) {
                assert!(
                    (value - reference).abs() <= 1.0e-4 * reference.abs().max(1.0e-8),
                    "{family}/{case}: parameter {value} != {reference}"
                );
            }
        }
        eprintln!(
            "{family}/{case}: MLE fsci={actual:?} scipy={expected:?}; \
             NLL fsci={nll:.12e} scipy={expected_nll:.12e}"
        );
        compared += 1;
    }
    assert_eq!(
        compared, 48,
        "every seeded case must produce a real comparison"
    );
    eprintln!("MLE compared cases = {compared}; families = 16; samples per case = 2000");
}

#[test]
fn censored_mle_uses_survival_for_unobserved_failures() {
    // Exponential censored MLE: scale = total exposure / observed failures.
    // Treating the censoring times as events would incorrectly return 30 / 7.
    let data = CensoredData::new(
        vec![1.0, 2.0, 4.0, 7.0],
        vec![],
        vec![3.0, 5.0, 8.0],
        vec![],
    );
    let original = data.clone();
    for mode in [RuntimeMode::Strict, RuntimeMode::Hardened] {
        let fitted = with_stats_runtime_mode(mode, || {
            fit_mle_censored(
                &data,
                &[MleParameter::Fixed(1.0), MleParameter::Positive(1.0)],
                |p| {
                    Some(GammaDist {
                        a: p[0],
                        scale: p[1],
                    })
                },
                MleOptions::default(),
            )
        })
        .expect("right-censored exponential MLE");
        let expected_scale = 30.0_f64 / 4.0;
        let expected_nll = 4.0 * expected_scale.ln() + 30.0 / expected_scale;
        assert_eq!(fitted.parameters[0], 1.0);
        assert!((fitted.parameters[1] - expected_scale).abs() < 2.0e-6);
        assert!((fitted.negative_log_likelihood - expected_nll).abs() < 1.0e-10);
    }
    assert_eq!(data, original, "fitting must not rewrite censoring metadata");
}

#[test]
fn censored_mle_exact_observations_preserve_the_uncensored_fit() {
    let values = [-2.5, -0.3, 0.2, 1.7, 2.9, 3.2, 5.4];
    let parameters = [MleParameter::Free(0.0), MleParameter::Positive(1.0)];
    let factory = |p: &[f64]| {
        Some(Normal {
            loc: p[0],
            scale: p[1],
        })
    };
    let original = fit_mle(&values, &parameters, factory, MleOptions::default()).unwrap();
    let samples = [
        CensoredData::new(values.to_vec(), vec![], vec![], vec![]),
        CensoredData::interval_censored(&values, &values).unwrap(),
    ];
    for data in samples {
        let fitted =
            fit_mle_censored(&data, &parameters, factory, MleOptions::default()).unwrap();
        assert_eq!(
            fitted
                .parameters
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            original
                .parameters
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            fitted.negative_log_likelihood.to_bits(),
            original.negative_log_likelihood.to_bits()
        );
        assert_eq!(fitted.iterations, original.iterations);
        assert_eq!(fitted.evaluations, original.evaluations);
    }
}

#[test]
fn censored_mle_mixed_likelihood_and_interval_normalization() {
    let data = CensoredData::new(
        vec![0.2, 0.7],
        vec![-0.8],
        vec![1.3],
        vec![
            (-0.3, 0.4),
            (f64::NEG_INFINITY, -1.1),
            (1.8, f64::INFINITY),
            (0.5, 0.5),
        ],
    );
    let parameters = [MleParameter::Fixed(-0.0), MleParameter::Fixed(1.2)];
    let fitted = fit_mle_censored(
        &data,
        &parameters,
        |p| {
            Some(Normal {
                loc: p[0],
                scale: p[1],
            })
        },
        MleOptions::default(),
    )
    .unwrap();
    let normal = Normal {
        loc: 0.0,
        scale: 1.2,
    };
    let expected = -(normal.logpdf(0.2)
        + normal.logpdf(0.7)
        + normal.logcdf(-0.8)
        + normal.logsf(1.3)
        + (normal.cdf(0.4) - normal.cdf(-0.3)).ln()
        + normal.logcdf(-1.1)
        + normal.logsf(1.8)
        + normal.logpdf(0.5));
    assert!((fitted.negative_log_likelihood - expected).abs() < 1.0e-12);
    assert_eq!(fitted.parameters[0].to_bits(), (-0.0_f64).to_bits());
    assert_eq!(fitted.iterations, 0);
    assert_eq!(fitted.evaluations, 1);
}

#[test]
fn censored_mle_interval_likelihood_survives_probability_underflow() {
    let gamma = GammaDist { a: 1.0, scale: 2.0 };
    assert_eq!(gamma.logsf(-1.0), 0.0);
    assert_eq!(gamma.logsf(0.0), 0.0);
    assert_eq!(gamma.logsf(f64::INFINITY), f64::NEG_INFINITY);
    assert!(gamma.logsf(f64::NAN).is_nan());
    assert!((gamma.logsf(2000.0) + 1000.0).abs() < 1.0e-10);
    assert_eq!(gamma.cdf(2000.0), 1.0);
    assert_eq!(gamma.cdf(2002.0), 1.0);
    assert_eq!(gamma.sf(2000.0), 0.0);
    assert_eq!(gamma.sf(2002.0), 0.0);
    let data = CensoredData::interval_censored(&[2000.0], &[2002.0]).unwrap();
    let fitted = fit_mle_censored(
        &data,
        &[MleParameter::Fixed(1.0), MleParameter::Fixed(2.0)],
        |p| {
            Some(GammaDist {
                a: p[0],
                scale: p[1],
            })
        },
        MleOptions::default(),
    )
    .unwrap();
    // P(2000 < Exp(scale=2) < 2002) = exp(-1000) * (1 - exp(-1)).
    let expected = 1000.0 - (-(-1.0_f64).exp()).ln_1p();
    assert!(fitted.negative_log_likelihood.is_finite());
    assert!((fitted.negative_log_likelihood - expected).abs() < 1.0e-10);

    let mut likelihoods = Vec::new();
    for (lower, upper) in [(-41.0, -40.0), (40.0, 41.0)] {
        let data = CensoredData::interval_censored(&[lower], &[upper]).unwrap();
        let fitted = fit_mle_censored(
            &data,
            &[MleParameter::Fixed(0.0), MleParameter::Fixed(1.0)],
            |p| {
                Some(Normal {
                    loc: p[0],
                    scale: p[1],
                })
            },
            MleOptions::default(),
        )
        .unwrap();
        assert!(fitted.negative_log_likelihood.is_finite());
        likelihoods.push(fitted.negative_log_likelihood);
    }
    assert!((likelihoods[0] - likelihoods[1]).abs() < 1.0e-11);
    // Normal tail reference independently agrees with the Mills-ratio expansion.
    assert!((likelihoods[0] - 804.608_442_013_753_8).abs() < 1.0e-10);
}

#[test]
fn censored_mle_rejects_invalid_impossible_and_uninformative_data() {
    let factory = |p: &[f64]| {
        Some(Normal {
            loc: p[0],
            scale: p[1],
        })
    };
    let parameters = [MleParameter::Fixed(0.0), MleParameter::Fixed(1.0)];
    let options = MleOptions::default();
    assert!(matches!(
        fit_mle_censored(&CensoredData::default(), &parameters, factory, options),
        Err(FitError::InsufficientData { .. })
    ));
    for data in [
        CensoredData::new(vec![f64::NAN], vec![], vec![], vec![]),
        CensoredData::new(vec![], vec![f64::INFINITY], vec![], vec![]),
        CensoredData::new(vec![], vec![], vec![f64::NEG_INFINITY], vec![]),
        CensoredData::new(vec![], vec![], vec![], vec![(2.0, 1.0)]),
        CensoredData::new(vec![], vec![], vec![], vec![(f64::NAN, 1.0)]),
        CensoredData::new(vec![], vec![], vec![], vec![(0.0, f64::NAN)]),
        CensoredData::new(
            vec![],
            vec![],
            vec![],
            vec![(f64::NEG_INFINITY, f64::INFINITY)],
        ),
        CensoredData::new(
            vec![],
            vec![],
            vec![],
            vec![(f64::INFINITY, f64::INFINITY)],
        ),
        CensoredData::new(
            vec![],
            vec![],
            vec![],
            vec![(f64::NEG_INFINITY, f64::NEG_INFINITY)],
        ),
    ] {
        assert!(matches!(
            fit_mle_censored(&data, &parameters, factory, options),
            Err(FitError::UnsupportedData(_))
        ));
    }

    let gamma = |p: &[f64]| {
        Some(GammaDist {
            a: p[0],
            scale: p[1],
        })
    };
    let parameters = [MleParameter::Fixed(1.0), MleParameter::Fixed(1.0)];
    for data in [
        // Impossible event for a distribution supported on nonnegative values.
        CensoredData::new(vec![], vec![-1.0], vec![], vec![]),
        // Probability-one events alone cannot identify model parameters.
        CensoredData::new(vec![], vec![], vec![-1.0, -2.0], vec![]),
    ] {
        assert!(matches!(
            fit_mle_censored(&data, &parameters, gamma, options),
            Err(FitError::NonConvergent(_))
        ));
    }
    let data = CensoredData::new(vec![0.2, 0.7], vec![-0.8], vec![1.3], vec![]);
    assert!(matches!(
        fit_mle_censored(
            &data,
            &[MleParameter::Free(0.0), MleParameter::Positive(1.0)],
            factory,
            MleOptions {
                max_evaluations: 1,
                ..options
            },
        ),
        Err(FitError::NonConvergent(_))
    ));
}

#[test]
fn censored_mle_hardened_counts_every_censoring_kind_toward_sample_cap() {
    with_stats_runtime_mode(RuntimeMode::Hardened, || {
        with_hardened_stats_caps(3, 3, || {
            let data = CensoredData::new(
                vec![0.0],
                vec![-1.0],
                vec![1.0],
                vec![(-0.2, 0.2)],
            );
            let result = fit_mle_censored(
                &data,
                &[MleParameter::Free(0.0), MleParameter::Positive(1.0)],
                |p| {
                    Some(Normal {
                        loc: p[0],
                        scale: p[1],
                    })
                },
                MleOptions::default(),
            );
            assert!(matches!(result, Err(FitError::ResourceExhausted { .. })));
        });
    });
}

const CENSORED_ORACLE: &str = r#"
import os
import numpy as np
import scipy
from scipy import stats, optimize

print('versions', scipy.__version__, np.__version__)
if os.environ.get('FSCI_REQUIRE_SCIPY_ORACLE') == '1':
    assert (scipy.__version__, np.__version__) == ('1.17.1', '2.4.3')

def optimizer(fun, start, args=(), disp=0):
    result = optimize.minimize(fun, start, args=args, method='Nelder-Mead',
                              options={'xatol': 1e-10, 'fatol': 1e-10,
                                       'maxiter': 10000, 'maxfev': 40000})
    assert result.success, result.message
    return result.x

def censor(values, lower, upper, width):
    exact, left, right, intervals = [], [], [], []
    for i, x in enumerate(values):
        if x <= lower:
            left.append(lower)
        elif x >= upper:
            right.append(upper)
        elif i % 3 == 0:
            lo = width * np.floor(x / width)
            intervals.append((lo, lo + width))
        else:
            exact.append(x)
    return exact, left, right, intervals

def logmass(dist, lo, hi):
    lc, ls = dist.logcdf(hi), dist.logsf(lo)
    a, b = (lc, dist.logcdf(lo)) if lc < ls else (ls, dist.logsf(hi))
    return a + np.log(-np.expm1(b - a))

def emit(case, family, policy, params, frozen, parts):
    exact, left, right, intervals = parts
    nll = -(sum(frozen.logpdf(exact)) + sum(frozen.logcdf(left))
            + sum(frozen.logsf(right))
            + sum(logmass(frozen, lo, hi) for lo, hi in intervals))
    assert np.isfinite(nll)
    print(case, family, policy, *(format(x, '.17g') for x in params),
          format(nll, '.17g'))
    for label, values in [('u', exact), ('l', left), ('r', right),
                          ('i', [x for pair in intervals for x in pair])]:
        print(label, *(format(x, '.17g') for x in values))

specs = [
    ('normal', stats.norm, (), 1.2, 2.5, -1.5, 3.5, 0.5),
    ('gamma', stats.gamma, (0.7,), 0.0, 2.4, 0.1, 4.0, 0.25),
    ('weibull', stats.weibull_min, (1.6,), 0.0, 3.2, 0.4, 4.5, 0.4),
]
for index, (family, dist, shapes, loc, scale, lower, upper, width) in enumerate(specs):
    values = dist.rvs(*shapes, loc=loc, scale=scale, size=400,
                      random_state=np.random.default_rng(91100 + index))
    parts = censor(values, lower, upper, width)
    data = stats.CensoredData(uncensored=parts[0], left=parts[1],
                              right=parts[2], interval=parts[3])
    kwargs = {'optimizer': optimizer}
    if family != 'normal':
        kwargs['floc'] = 0.0
    fitted = dist.fit(data, **kwargs)
    params = fitted if family == 'normal' else (fitted[0], fitted[-1])
    frozen = dist(*fitted[:-2], loc=fitted[-2], scale=fitted[-1])
    emit(family + '_mixed', family, 'free', params, frozen, parts)

    if family == 'normal':
        kwargs['floc'] = loc
        policy = 'fixed0'
    elif family == 'gamma':
        kwargs['fscale'] = scale
        policy = 'fixed1'
    else:
        kwargs['f0'] = shapes[0]
        policy = 'fixed0'
    fitted = dist.fit(data, **kwargs)
    params = fitted if family == 'normal' else (fitted[0], fitted[-1])
    frozen = dist(*fitted[:-2], loc=fitted[-2], scale=fitted[-1])
    emit(family + '_fixed', family, policy, params, frozen, parts)

    if family == 'normal':
        intervals = [(width * np.floor(x / width),
                      width * np.floor(x / width) + width) for x in values]
        interval_data = stats.CensoredData(interval=intervals)
        fitted = stats.norm.fit(interval_data, optimizer=optimizer)
        emit('normal_intervals', 'normal', 'free', fitted,
             stats.norm(*fitted), ([], [], [], intervals))

parts = ([1.0, 2.0, 4.0, 7.0], [], [3.0, 5.0, 8.0], [])
data = stats.CensoredData(uncensored=parts[0], right=parts[2])
fitted = stats.expon.fit(data, floc=0.0, optimizer=optimizer)
emit('exponential_right', 'gamma', 'fixed0', (1.0, fitted[-1]),
     stats.expon(*fitted), parts)
emit('normal_tails', 'normal', 'fixed_both', (0.0, 1.0), stats.norm(),
     ([], [], [], [(-41.0, -40.0), (40.0, 41.0)]))
"#;

fn censor_values(line: &str, label: &str) -> Vec<f64> {
    let mut values = line.split_whitespace();
    assert_eq!(values.next(), Some(label), "censored oracle line label");
    values.map(|value| value.parse().unwrap()).collect()
}

#[test]
fn censored_mle_matches_live_scipy() {
    let python = std::env::var("FSCI_PYTHON").unwrap_or_else(|_| "python3".into());
    let required = std::env::var("FSCI_REQUIRE_SCIPY_ORACLE").as_deref() == Ok("1");
    let available = Command::new(&python)
        .args(["-c", "import numpy; import scipy"])
        .output()
        .is_ok_and(|output| output.status.success());
    if !available {
        assert!(!required, "required SciPy oracle is unavailable via {python}");
        eprintln!("SKIP censored MLE: SciPy oracle unavailable via {python}; compared cases = 0");
        return;
    }
    let output = Command::new(&python)
        .args(["-c", CENSORED_ORACLE])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "SciPy censored MLE oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let mut lines = stdout.lines();
    let versions = lines.next().expect("censored oracle version metadata");
    assert!(versions.starts_with("versions "));
    eprintln!("Censored MLE oracle {versions}");
    if required {
        assert_eq!(versions, "versions 1.17.1 2.4.3");
    }

    let mut compared = 0;
    while let Some(header) = lines.next() {
        let fields: Vec<_> = header.split_whitespace().collect();
        assert_eq!(fields.len(), 6);
        let (case, family, policy) = (fields[0], fields[1], fields[2]);
        let expected = [
            fields[3].parse::<f64>().unwrap(),
            fields[4].parse::<f64>().unwrap(),
        ];
        let expected_nll = fields[5].parse::<f64>().unwrap();
        assert!(expected.iter().all(|value| value.is_finite()));
        assert!(expected_nll.is_finite());
        let exact = censor_values(lines.next().unwrap(), "u");
        let left = censor_values(lines.next().unwrap(), "l");
        let right = censor_values(lines.next().unwrap(), "r");
        let intervals = censor_values(lines.next().unwrap(), "i");
        assert_eq!(intervals.len() % 2, 0);
        let data = CensoredData::new(
            exact,
            left,
            right,
            intervals
                .chunks_exact(2)
                .map(|pair| (pair[0], pair[1]))
                .collect(),
        );
        let mut parameters = [
            if family == "normal" {
                MleParameter::Free(0.0)
            } else {
                MleParameter::Positive(1.0)
            },
            MleParameter::Positive(1.0),
        ];
        if policy == "fixed0" || policy == "fixed_both" {
            parameters[0] = MleParameter::Fixed(expected[0]);
        }
        if policy == "fixed1" || policy == "fixed_both" {
            parameters[1] = MleParameter::Fixed(expected[1]);
        }
        let options = MleOptions::default();
        let (actual, nll) = match family {
            "normal" => {
                let fit = fit_mle_censored(
                    &data,
                    &parameters,
                    |p| {
                        Some(Normal {
                            loc: p[0],
                            scale: p[1],
                        })
                    },
                    options,
                )
                .unwrap();
                (fit.parameters, fit.negative_log_likelihood)
            }
            "gamma" => {
                let fit = fit_mle_censored(
                    &data,
                    &parameters,
                    |p| {
                        Some(GammaDist {
                            a: p[0],
                            scale: p[1],
                        })
                    },
                    options,
                )
                .unwrap();
                (fit.parameters, fit.negative_log_likelihood)
            }
            "weibull" => {
                let fit = fit_mle_censored(
                    &data,
                    &parameters,
                    |p| {
                        Some(Weibull {
                            c: p[0],
                            scale: p[1],
                        })
                    },
                    options,
                )
                .unwrap();
                (fit.parameters, fit.negative_log_likelihood)
            }
            _ => panic!("unexpected censored oracle family {family}"),
        };
        assert!(nll.is_finite(), "{case}: nonfinite fitted likelihood");
        assert_eq!(actual.len(), expected.len(), "{case}: parameter count");
        assert!(
            (nll - expected_nll).abs() <= 1.0e-7 * expected_nll.abs().max(1.0),
            "{case}: fsci NLL {nll} != SciPy {expected_nll}"
        );
        for (i, (&value, &reference)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (value - reference).abs() <= 2.0e-5 * reference.abs().max(1.0),
                "{case}: parameter {i}: {value} != {reference}"
            );
            if matches!(parameters[i], MleParameter::Fixed(_)) {
                assert_eq!(value.to_bits(), reference.to_bits());
            }
        }
        eprintln!(
            "{case}: censored MLE fsci={actual:?} scipy={expected:?}; \
             NLL fsci={nll:.12e} scipy={expected_nll:.12e}; observations={}",
            data.len()
        );
        compared += 1;
    }
    assert_eq!(compared, 9, "every censored oracle case must be compared");
    eprintln!("Censored MLE compared cases = {compared}");
}
