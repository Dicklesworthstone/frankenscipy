//! Public MLE regressions and live likelihood comparisons for newly fitted families.
//!
//! `FSCI_REQUIRE_SCIPY_ORACLE=1 cargo test -p fsci-stats --test mle_scipy -- --nocapture`
//! requires SciPy 1.17.1 and NumPy 2.4.3; otherwise a missing Python/SciPy installation
//! is an explicit skip. `FSCI_PYTHON` selects the interpreter, and the actual versions
//! are printed. The public regression tests do not require Python.

use std::process::Command;

use fsci_runtime::RuntimeMode;
use fsci_stats::{
    BetaDist, Burr3, Burr12, ContinuousDistribution, CrystalBall, FitError, FoldedCauchy,
    GammaDist, Kappa3, Lognormal, MleOptions, MleParameter, Normal, Weibull, fit_mle,
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
fn mle_matches_live_scipy_on_twenty_seven_seeded_fits() {
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
        compared, 27,
        "every seeded case must produce a real comparison"
    );
    eprintln!("MLE compared cases = {compared}; families = 9; samples per case = 2000");
}
