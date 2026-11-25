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
    CrystalBall, DiscreteDistribution, ExponWeibull, FDistribution, FitError, FoldedCauchy,
    GammaDist, JohnsonSU, Kappa3, Landau, LocationScale, LogSeries, Lognormal, Mielke, MleOptions,
    MleParameter, MleResult, Nakagami, NormInvGauss, Normal, RDist, RandInt, Rice, SkewCauchy,
    StatsError, StudentT, Trapezoid, TruncNormal, TruncPareto, TruncWeibullMin, Weibull, Zipfian,
    fit_mle, fit_mle_censored, fit_mle_location_scale, fit_mle_location_scale_censored,
    with_hardened_stats_caps, with_stats_runtime_mode,
};
use fsci_stats::{
    Cauchy, DoubleGamma, Erlang, ExponNorm, FoldedNormal, GenGamma, GenNorm, HalfGenNorm,
    LaplaceAsymmetric, NoncentralChiSquared, SkewNorm,
};
use rand::{SeedableRng, rngs::StdRng};

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
    ('nakagami', stats.nakagami, [(0.25,), (2.0,), (1000.0,)]),
    ('gompertz', stats.gompertz, [(0.1,), (1.0,), (10.0,)]),
    ('genlogistic', stats.genlogistic, [(0.2,), (1.0,), (5.0,)]),
    ('mielke', stats.mielke, [(0.2, 1.4), (2.0, 4.0), (40.0, 20.0)]),
    ('exponweib', stats.exponweib, [(0.2, 1.4), (2.0, 1.5), (40.0, 20.0)]),
    ('rice', stats.rice, [(0.0,), (2.0,), (25.0,)]),
    ('norminvgauss', stats.norminvgauss, [(1.5, 0.5), (5.0, -2.0), (50.0, 45.0)]),
    ('laplace_asymmetric', stats.laplace_asymmetric, [(0.2,), (1.0,), (5.0,)]),
    ('dgamma', stats.dgamma, [(0.4,), (2.0,), (10.0,)]),
    ('foldnorm', stats.foldnorm, [(0.0,), (0.8,), (8.0,)]),
    ('skewnorm', stats.skewnorm, [(-5.0,), (0.0,), (4.0,)]),
    ('erlang', stats.erlang, [(1,), (4,), (30,)]),
    ('ncx2', stats.ncx2, [(1.0, 0.5), (5.0, 2.0), (100.0, 200.0)]),
    ('gennorm', stats.gennorm, [(0.3,), (2.0,), (20.0,)]),
    ('halfgennorm', stats.halfgennorm, [(0.3,), (2.0,), (20.0,)]),
    ('gengamma', stats.gengamma, [(0.5, 0.5), (2.0, 1.5), (4.0, 3.0),
                                (0.5, -0.5), (2.0, -1.5), (4.0, -3.0)]),
]
for family_index, (name, dist, shapes) in enumerate(cases):
    for case_index, shape in enumerate(shapes):
        scale = 2.0 if name in ('gamma', 'weibull_min', 'lognorm', 'erlang') else 1.0
        data = dist.rvs(*shape, scale=scale, size=2000,
                        random_state=np.random.default_rng(12700 + 17*family_index + case_index))
        kwargs = {'floc': 0}
        if name not in ('gamma', 'weibull_min', 'lognorm'):
            kwargs['fscale'] = 1
        if name == 'erlang':
            # SciPy's unconstrained erlang.fit allows fractional shape.
            # The Rust family has an integer shape, so profile SciPy's
            # likelihood over the neighboring integers and fit each scale.
            continuous = stats.gamma.fit(data, floc=0)[0]
            candidates = sorted({1, max(1, int(np.floor(continuous))),
                                 max(1, int(np.ceil(continuous)))})
            fits = [dist.fit(data, f0=k, floc=0) for k in candidates]
            fitted = min(fits, key=lambda p: -np.sum(dist.logpdf(
                data, p[0], loc=p[-2], scale=p[-1])))
        elif name == 'gengamma':
            # Both signs of c are valid and separated by the excluded c=0
            # boundary. SciPy's default positive start can miss a much
            # better negative-c optimum, so compare both regions live.
            fits = [dist.fit(data, a0, c0, **kwargs)
                    for a0, c0 in ((1.0, 1.0), (2.0, 1.5), (0.5, 0.8),
                                   (1.0, -1.0), (2.0, -1.5), (0.5, -0.8))]
            fitted = min(fits, key=lambda p: -np.sum(dist.logpdf(
                data, *p[:-2], loc=p[-2], scale=p[-1])))
        else:
            fitted = dist.fit(data, **kwargs)
        nll = -np.sum(dist.logpdf(data, *fitted[:-2], loc=fitted[-2], scale=fitted[-1]))
        parameters = fitted[:-2]
        if name == 'erlang':
            parameters = (*parameters, 1/fitted[-1])
        elif name in ('gamma', 'weibull_min', 'lognorm'):
            parameters = (*parameters, fitted[-1])
        print(name, case_index, len(parameters), *(format(x, '.17g') for x in parameters),
              format(nll, '.17g'))
        print(' '.join(format(x, '.17g') for x in data))
"#;

fn likelihood<D: ContinuousDistribution>(distribution: &D, data: &[f64]) -> f64 {
    -data.iter().map(|&x| distribution.logpdf(x)).sum::<f64>()
}

fn discrete_likelihood<D: DiscreteDistribution>(distribution: &D, data: &[i64]) -> f64 {
    -data.iter().map(|&x| distribution.logpmf(x)).sum::<f64>()
}

#[test]
fn noncentral_chi_squared_mle_fits_interior_and_central_boundaries() {
    for mode in [RuntimeMode::Strict, RuntimeMode::Hardened] {
        with_stats_runtime_mode(mode, || {
            // Underdispersion gives a negative moment estimate for nc.
            // Its constrained likelihood instead has an attained central fit.
            let central_data = [9.8, 9.9, 10.0, 10.1, 10.2];
            let fitted = NoncentralChiSquared::try_fit(&central_data).unwrap();
            assert_eq!(fitted.nc, 0.0);
            assert_eq!(fitted.df, ChiSquared::try_fit(&central_data).unwrap().df);
            assert!((likelihood(&fitted, &central_data) - 12.044_664_477_8).abs() < 1e-9);
            assert_eq!(fitted, NoncentralChiSquared::fit(&central_data));

            // Variance > 4*mean made the old moment fit reject this valid
            // positive sample. Independent SciPy multi-start likelihood fit.
            let data = [0.1_f64, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8];
            let mean = data.iter().sum::<f64>() / data.len() as f64;
            let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
            assert!(variance > 4.0 * mean);
            let fitted = NoncentralChiSquared::try_fit(&data).unwrap();
            assert!(fitted.df > 0.0 && fitted.nc > 0.0);
            let nll = likelihood(&fitted, &data);
            assert!((nll - 17.005_520_165_5).abs() < 1e-8);
            let central = NoncentralChiSquared::new(ChiSquared::try_fit(&data).unwrap().df, 0.0);
            assert!(nll < likelihood(&central, &data) - 0.1);
            for factor in [0.95, 1.05] {
                assert!(
                    nll < likelihood(
                        &NoncentralChiSquared::new(fitted.df * factor, fitted.nc),
                        &data
                    )
                );
                assert!(
                    nll < likelihood(
                        &NoncentralChiSquared::new(fitted.df, fitted.nc * factor),
                        &data
                    )
                );
            }

            let small = [0.001, 0.01, 0.1, 0.5, 1.0, 3.0];
            let fitted = NoncentralChiSquared::try_fit(&small).unwrap();
            assert!(fitted.df > 0.0 && fitted.nc > 0.0);
            assert!((likelihood(&fitted, &small) - 0.636_474_065).abs() < 1e-8);
        });
    }
}

#[test]
fn noncentral_chi_squared_density_keeps_large_order_and_tail_likelihoods() {
    // SciPy 1.17.1 reference log densities. The former unconditional six-term
    // Bessel expansion gave positive log densities for the first two cases,
    // and the former pdf reconstructed exp(sqrt(nc*x)) and overflowed.
    for (df, nc, x, expected) in [
        (100.0, 200.0, 300.0, -4.373_951_302_723_176),
        (500.0, 1000.0, 1500.0, -5.177_761_867_296_926),
        (4.0, 10_000.0, 10_000.0, -6.217_293_401_627_874),
        (2.0, 2000.0, 1e-10, -1000.693_147_130_610),
        (10.0, 10_000.0, 1.0, -4922.914_488_648_738),
    ] {
        let distribution = NoncentralChiSquared::new(df, nc);
        let actual = distribution.logpdf(x);
        assert!(
            (actual - expected).abs() < 1e-10,
            "{df}/{nc}/{x}: {actual} != {expected}"
        );
        assert_eq!(distribution.pdf(x), actual.exp());
    }
    // 80-digit positive Poisson/Gamma sums. At tiny df, SciPy's Bessel
    // order rounds to -1 and drops the k=0 component; its logpdf is not
    // a valid reference for these cases.
    for (df, nc, x, expected) in [
        (1e-8, 4.0, 1e-12, 6.517_393_032_683_641),
        (1e-16, 4.0, 1e-20, 6.517_393_171_418_901),
        (1e-10, 1e-5, 1e-10, -0.693_147_181_779_534_4),
    ] {
        let actual = NoncentralChiSquared::new(df, nc).logpdf(x);
        assert!(
            (actual - expected).abs() < 1e-12,
            "{df}/{nc}/{x}: {actual} != {expected}"
        );
    }
    let endpoint = NoncentralChiSquared::new(2.0, 4.0);
    assert_eq!(endpoint.pdf(0.0), 0.0); // Pinned SciPy PDF endpoint convention.
    assert_eq!(endpoint.logpdf(0.0), -std::f64::consts::LN_2 - 2.0);
    assert_eq!(endpoint.pdf(f64::INFINITY), 0.0);
    assert_eq!(endpoint.logpdf(f64::INFINITY), f64::NEG_INFINITY);
    assert_eq!(endpoint.logpdf(-1.0), f64::NEG_INFINITY);
    assert!(endpoint.logpdf(f64::NAN).is_nan());
    for df in [0.5, 2.0, 10.0] {
        let noncentral = NoncentralChiSquared::new(df, 0.0);
        let central = ChiSquared::new(df);
        for x in [0.0, 0.01, 1.0, 10.0] {
            assert_eq!(noncentral.logpdf(x), central.logpdf(x));
        }
    }
}

#[test]
fn noncentral_chi_squared_mle_rejects_singular_and_unattained_fits() {
    // This sample's supremum is at df -> 0, nc approximately 8.34926.
    // The positive-df family has no attained MLE there.
    let sample = [1.0, 2.0, 8.0, 30.0];
    let Err(FitError::NonConvergent(message)) = NoncentralChiSquared::try_fit(&sample) else {
        panic!("excluded zero-df maximum must not be reported as a finite fit");
    };
    assert!(message.contains("df=0"), "{message}");
    for data in [&[0.0, 0.0][..], &[0.0, 1.0][..]] {
        assert!(matches!(
            NoncentralChiSquared::try_fit(data),
            Err(FitError::NonConvergent(_))
        ));
    }
    assert!(matches!(
        NoncentralChiSquared::try_fit(&[]),
        Err(FitError::InsufficientData { .. })
    ));
    assert!(matches!(
        NoncentralChiSquared::try_fit(&[1.0]),
        Err(FitError::InsufficientData { .. })
    ));
    for bad in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            NoncentralChiSquared::try_fit(&[0.5, bad, 2.0]),
            Err(FitError::UnsupportedData(_))
        ));
    }
    let panic = std::panic::catch_unwind(|| NoncentralChiSquared::fit(&[0.0, 1.0])).unwrap_err();
    assert!(matches!(
        panic.downcast_ref::<FitError>(),
        Some(FitError::NonConvergent(_))
    ));
    with_stats_runtime_mode(RuntimeMode::Hardened, || {
        with_hardened_stats_caps(2, 2, || {
            assert!(matches!(
                NoncentralChiSquared::try_fit(&[0.5, 1.0, 2.0]),
                Err(FitError::ResourceExhausted { .. })
            ));
        });
    });
}

#[test]
fn skew_normal_mle_uses_signed_score_and_handles_large_shapes() {
    for mode in [RuntimeMode::Strict, RuntimeMode::Hardened] {
        with_stats_runtime_mode(mode, || {
            // The sample skewness is exactly zero, so the former moment fit
            // returned a=0 even though the fixed-location likelihood increases.
            let data = [-1.0_f64, 0.0, 1.0, 2.0, 3.0];
            let fitted = SkewNorm::try_fit(&data).unwrap();
            assert!(fitted.a > 0.5 && fitted.a < 0.7);
            assert!(likelihood(&SkewNorm::new(0.0), &data) - likelihood(&fitted, &data) > 0.5);
            let score: f64 = data
                .iter()
                .map(|&x| {
                    let z = fitted.a * x;
                    x * (-0.5 * z * z
                        - 0.5 * (2.0 * std::f64::consts::PI).ln()
                        - fsci_special::log_ndtr_scalar(z))
                    .exp()
                })
                .sum();
            assert!(score.abs() < 1e-11, "skew-normal likelihood score={score}");
            let reflected = data.map(|x| -x);
            assert!((fitted.a + SkewNorm::try_fit(&reflected).unwrap().a).abs() < 1e-12);
            for offset in [-0.1, 0.1] {
                assert!(
                    likelihood(&fitted, &data)
                        < likelihood(&SkewNorm::new(fitted.a + offset), &data)
                );
            }
            assert_eq!(fitted.a, SkewNorm::fit(&data).a);

            // Arbitrary sample skewness need not lie in the model's population
            // skewness range for the likelihood to have a finite maximum.
            let dispersed = [-0.1, 0.2, 0.3, 1.0, 8.0];
            let fitted = SkewNorm::try_fit(&dispersed).unwrap();
            assert!(fitted.a > 0.0);
            assert!(likelihood(&fitted, &dispersed).is_finite());

            let small = [-0.0001, 0.001, 0.002, 0.003];
            let fitted = SkewNorm::try_fit(&small).unwrap();
            assert!(fitted.a > 50.0);
            let scale = 1e-180;
            let scaled = small.map(|x| x * scale);
            let scaled_fit = SkewNorm::try_fit(&scaled).unwrap();
            assert!((scaled_fit.a * scale / fitted.a - 1.0).abs() < 2e-11);
            assert!(likelihood(&scaled_fit, &scaled).is_finite());
            assert!(scaled_fit.mean() > 0.79 && scaled_fit.mean() < 0.80);
            assert!(SkewNorm::try_fit(&[-1.0, 2.0]).unwrap().a > 0.0);
        });
    }
}

#[test]
fn erlang_mle_profiles_rate_and_selects_integer_shape() {
    for mode in [RuntimeMode::Strict, RuntimeMode::Hardened] {
        with_stats_runtime_mode(mode, || {
            let data = [0.5_f64, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
            let fitted = Erlang::try_fit(&data).unwrap();
            let mean = data.iter().sum::<f64>() / data.len() as f64;
            let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
            let old = Erlang::new((mean * mean / variance).round() as usize, mean / variance);
            assert!(likelihood(&fitted, &data) < likelihood(&old, &data) - 0.05);
            assert!((fitted.mean() - mean).abs() < 1e-13);
            for k in 1..=16 {
                let candidate = Erlang::new(k, k as f64 / mean);
                assert!(likelihood(&fitted, &data) <= likelihood(&candidate, &data) + 1e-12);
            }
            assert_eq!(fitted, Erlang::fit(&data));
            // Rounding the continuous Gamma MLE would choose one here.
            // The integer likelihood instead favors k=2.
            let neighbors = [1.0, 6.3];
            let gamma = GammaDist::try_fit(&neighbors).unwrap();
            assert!(gamma.a > 1.0 && gamma.a < 1.5);
            let fitted = Erlang::try_fit(&neighbors).unwrap();
            assert_eq!(fitted.k, 2);
            assert!(
                likelihood(&fitted, &neighbors)
                    < likelihood(&Erlang::new(1, 1.0 / 3.65), &neighbors)
            );

            let zeros = [0.0, 0.0, 2.0];
            let fitted = Erlang::try_fit(&zeros).unwrap();
            assert_eq!(fitted.k, 1);
            assert_eq!(fitted.rate, 1.5);
            assert_eq!(fitted.pdf(0.0), 1.5);
            assert_eq!(fitted.logpdf(0.0), 1.5_f64.ln());
            assert!(likelihood(&fitted, &zeros).is_finite());

            // A valid concentrated fit exceeds the factorial/power range of
            // the former density implementation.
            let concentrated = [9.8, 9.9, 10.0, 10.1, 10.2];
            let fitted = Erlang::try_fit(&concentrated).unwrap();
            assert!(fitted.k > 1000);
            let points = [-1.0, 0.0, 9.8, 10.0, 10.2, f64::INFINITY];
            let pdf = fitted.pdf_many(&points);
            let logpdf = fitted.logpdf_many(&points);
            for (index, &x) in points.iter().enumerate() {
                assert_eq!(pdf[index], fitted.pdf(x));
                assert_eq!(logpdf[index], fitted.logpdf(x));
            }
            assert!(fitted.pdf(10.0).is_finite() && fitted.pdf(10.0) > 0.0);
            assert!(likelihood(&fitted, &concentrated).is_finite());
            let scaled = data.map(|x| x * 1e150);
            let scaled_fit = Erlang::try_fit(&scaled).unwrap();
            let original = Erlang::try_fit(&data).unwrap();
            assert_eq!(scaled_fit.k, original.k);
            assert!((scaled_fit.rate * 1e150 / original.rate - 1.0).abs() < 1e-12);
        });
    }
}

#[test]
fn skew_normal_and_erlang_mles_reject_unidentifiable_or_invalid_samples() {
    for data in [&[][..], &[1.0][..]] {
        assert!(matches!(
            SkewNorm::try_fit(data),
            Err(FitError::InsufficientData { .. })
        ));
        assert!(matches!(
            Erlang::try_fit(data),
            Err(FitError::InsufficientData { .. })
        ));
    }
    for data in [[0.0, 1.0, 2.0], [-2.0, -1.0, 0.0], [0.0, 0.0, 0.0]] {
        assert!(matches!(
            SkewNorm::try_fit(&data),
            Err(FitError::NonConvergent(_))
        ));
    }
    for data in [[0.0, 0.0], [2.0, 2.0]] {
        assert!(matches!(
            Erlang::try_fit(&data),
            Err(FitError::NonConvergent(_))
        ));
    }
    assert!(matches!(
        Erlang::try_fit(&[-0.1, 1.0]),
        Err(FitError::UnsupportedData(_))
    ));
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            SkewNorm::try_fit(&[-1.0, bad, 2.0]),
            Err(FitError::UnsupportedData(_))
        ));
        assert!(matches!(
            Erlang::try_fit(&[1.0, bad, 2.0]),
            Err(FitError::UnsupportedData(_))
        ));
    }
    with_stats_runtime_mode(RuntimeMode::Hardened, || {
        with_hardened_stats_caps(2, 2, || {
            assert!(matches!(
                SkewNorm::try_fit(&[-1.0, 0.0, 2.0]),
                Err(FitError::ResourceExhausted { .. })
            ));
            assert!(matches!(
                Erlang::try_fit(&[1.0, 2.0, 3.0]),
                Err(FitError::ResourceExhausted { .. })
            ));
        });
    });
}

#[test]
fn standardized_shape_mles_improve_likelihood_over_moment_estimates() {
    for mode in [RuntimeMode::Strict, RuntimeMode::Hardened] {
        with_stats_runtime_mode(mode, || {
            let data = [-10.0_f64, -0.5, 0.2, 1.0, 4.0];
            let fitted = LaplaceAsymmetric::try_fit(&data).unwrap();
            let mean = data.iter().sum::<f64>() / data.len() as f64;
            let moment_shape = 0.5 * ((mean * mean + 4.0).sqrt() - mean);
            let moment_fit = LaplaceAsymmetric::new(moment_shape);
            assert!(likelihood(&moment_fit, &data) - likelihood(&fitted, &data) > 0.5);
            let positive = data.iter().map(|&x| x.max(0.0)).sum::<f64>() / data.len() as f64;
            let negative = data.iter().map(|&x| (-x).max(0.0)).sum::<f64>() / data.len() as f64;
            let k = fitted.kappa;
            let score = (1.0 - k * k) / (1.0 + k * k) - k * positive + negative / k;
            assert!(
                score.abs() < 1e-13,
                "asymmetric Laplace likelihood score={score}"
            );
            let reflected: Vec<_> = data.iter().map(|&x| -x).collect();
            let reflected_fit = LaplaceAsymmetric::try_fit(&reflected).unwrap();
            assert!((fitted.kappa * reflected_fit.kappa - 1.0).abs() < 1e-13);
            assert_eq!(fitted.kappa, LaplaceAsymmetric::fit(&data).kappa);

            let data = [-1.5_f64, -0.5, 0.5, 1.0, 2.0, -2.5, 0.7, -0.3];
            let fitted = DoubleGamma::try_fit(&data).unwrap();
            let second_moment = data.iter().map(|x| x * x).sum::<f64>() / data.len() as f64;
            let moment_fit = DoubleGamma::new(0.5 * ((1.0 + 4.0 * second_moment).sqrt() - 1.0));
            assert!(likelihood(&moment_fit, &data) - likelihood(&fitted, &data) > 0.01);
            let target = data.iter().map(|x| x.abs().ln()).sum::<f64>() / data.len() as f64;
            assert!((fsci_special::digamma_scalar(fitted.a) - target).abs() < 1e-12);
            let reflected: Vec<_> = data.iter().map(|&x| -x).collect();
            assert_eq!(fitted.a, DoubleGamma::try_fit(&reflected).unwrap().a);
            assert_eq!(fitted.a, DoubleGamma::fit(&data).a);

            // The mean is below the half-normal mean, so the former moment
            // fit returned c=0. The likelihood has a positive interior maximum
            // because the second moment exceeds one.
            let data = [0.0_f64, 0.0, 0.0, 3.0];
            let fitted = FoldedNormal::try_fit(&data).unwrap();
            assert!(fitted.c > 0.7 && fitted.c < 0.8);
            assert!(likelihood(&FoldedNormal::new(0.0), &data) - likelihood(&fitted, &data) > 0.4);
            let score = data.iter().map(|&x| x * (fitted.c * x).tanh()).sum::<f64>()
                / data.len() as f64
                - fitted.c;
            assert!(
                score.abs() < 1e-13,
                "folded-normal likelihood score={score}"
            );
            for factor in [0.9, 1.1] {
                assert!(
                    likelihood(&fitted, &data)
                        < likelihood(&FoldedNormal::new(factor * fitted.c), &data)
                );
            }
            assert_eq!(fitted.c, FoldedNormal::fit(&data).c);
        });
    }
}

#[test]
fn standardized_shape_mles_validate_samples_and_honor_hardened_caps() {
    assert!(matches!(
        LaplaceAsymmetric::try_fit(&[]),
        Err(FitError::InsufficientData { .. })
    ));
    assert!(matches!(
        FoldedNormal::try_fit(&[]),
        Err(FitError::InsufficientData { .. })
    ));
    for data in [&[][..], &[1.0][..]] {
        assert!(matches!(
            DoubleGamma::try_fit(data),
            Err(FitError::InsufficientData { .. })
        ));
    }
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            LaplaceAsymmetric::try_fit(&[1.0, bad]),
            Err(FitError::UnsupportedData(_))
        ));
        assert!(matches!(
            DoubleGamma::try_fit(&[1.0, bad]),
            Err(FitError::UnsupportedData(_))
        ));
        assert!(matches!(
            FoldedNormal::try_fit(&[1.0, bad]),
            Err(FitError::UnsupportedData(_))
        ));
    }
    assert!(matches!(
        DoubleGamma::try_fit(&[0.0, 1.0]),
        Err(FitError::NonConvergent(_))
    ));
    assert!(matches!(
        FoldedNormal::try_fit(&[-0.1, 1.0]),
        Err(FitError::UnsupportedData(_))
    ));
    assert_eq!(LaplaceAsymmetric::try_fit(&[0.0, 0.0]).unwrap().kappa, 1.0);
    assert_eq!(FoldedNormal::try_fit(&[0.0, 0.0]).unwrap().c, 0.0);
    assert_eq!(FoldedNormal::try_fit(&[0.2, 0.6, 1.0]).unwrap().c, 0.0);
    with_stats_runtime_mode(RuntimeMode::Hardened, || {
        with_hardened_stats_caps(2, 2, || {
            assert!(matches!(
                LaplaceAsymmetric::try_fit(&[-1.0, 0.0, 1.0]),
                Err(FitError::ResourceExhausted { .. })
            ));
            assert!(matches!(
                DoubleGamma::try_fit(&[-1.0, 0.5, 1.0]),
                Err(FitError::ResourceExhausted { .. })
            ));
            assert!(matches!(
                FoldedNormal::try_fit(&[0.0, 1.0, 2.0]),
                Err(FitError::ResourceExhausted { .. })
            ));
        });
    });
}

#[test]
fn fitted_asymmetric_laplace_extreme_shapes_preserve_probability_functions() {
    for magnitude in [1e200, 1e308] {
        for sign in [-1.0, 1.0] {
            let data = [sign * magnitude; 3];
            let fitted = LaplaceAsymmetric::try_fit(&data).unwrap();
            let reflected = LaplaceAsymmetric::try_fit(&[-sign * magnitude; 3]).unwrap();
            assert!(fitted.logcdf(0.0).is_finite());
            assert!(fitted.logsf(0.0).is_finite());
            for q in [0.25_f64, 0.5, 0.75] {
                let x = fitted.ppf(q);
                assert!(x.is_finite(), "kappa={}, q={q}", fitted.kappa);
                assert!((fitted.cdf(x) - q).abs() < 1e-12);
                assert!((fitted.sf(x) - (1.0 - q)).abs() < 1e-12);
                assert!((fitted.logcdf(x) - q.ln()).abs() < 1e-12);
                assert!((fitted.logsf(x) - (-q).ln_1p()).abs() < 1e-12);
                assert!(fitted.pdf(x) > 0.0);
                assert!((fitted.pdf(x) / fitted.logpdf(x).exp() - 1.0).abs() < 1e-12);
                let inverse_survival = fitted.isf(q);
                assert!(inverse_survival.is_finite());
                assert!((fitted.sf(inverse_survival) - q).abs() < 1e-12);
                assert!((reflected.cdf(-x) - fitted.sf(x)).abs() < 1e-12);
                assert!((reflected.pdf(-x) / fitted.pdf(x) - 1.0).abs() < 1e-12);
                assert!((reflected.ppf(1.0 - q) / -x - 1.0).abs() < 1e-12);
            }
            assert_eq!(fitted.ppf(0.0), f64::NEG_INFINITY);
            assert_eq!(fitted.ppf(1.0), f64::INFINITY);
            assert_eq!(fitted.isf(0.0), f64::INFINITY);
            assert_eq!(fitted.isf(1.0), f64::NEG_INFINITY);
            for q in [-0.1, 1.1, f64::NAN] {
                assert!(fitted.ppf(q).is_nan());
                assert!(fitted.isf(q).is_nan());
            }
        }
    }
    // The ordinary probabilities underflow, but their logarithms must
    // remain finite even if x*kappa or x/kappa also rounds to zero.
    let tiny = LaplaceAsymmetric::new(f64::from_bits(1));
    assert!(tiny.logcdf(0.0).is_finite());
    assert!(tiny.logcdf(0.1).is_finite());
    let huge = LaplaceAsymmetric::new(f64::MAX);
    assert!(huge.logsf(0.0).is_finite());
    assert!(huge.logsf(-f64::from_bits(1)).is_finite());
}

#[test]
fn standardized_shape_mles_keep_extreme_and_near_boundary_solutions() {
    for sign in [-1.0_f64, 1.0] {
        let data = [sign * 1e300, sign * 2e300];
        let fitted = LaplaceAsymmetric::try_fit(&data).unwrap();
        let expected = if sign > 0.0 { 1.0 / 1.5e300 } else { 1.5e300 };
        assert!((fitted.kappa / expected - 1.0).abs() < 1e-12);
        assert!(likelihood(&fitted, &data).is_finite());
    }
    let repeated = [f64::MAX; 3];
    let fitted = LaplaceAsymmetric::try_fit(&repeated).unwrap();
    assert!(fitted.kappa > 0.0 && likelihood(&fitted, &repeated).is_finite());
    assert!(
        LaplaceAsymmetric::new(f64::from_bits(1))
            .logpdf(0.0)
            .is_finite()
    );

    let tiny = [1e-250_f64, -2e-250];
    let fitted = DoubleGamma::try_fit(&tiny).unwrap();
    let target = 0.5 * (tiny[0].ln() + tiny[1].abs().ln());
    assert!((fsci_special::digamma_scalar(fitted.a) - target).abs() < 1e-10);
    assert!(likelihood(&fitted, &tiny).is_finite());

    // Unit scale is fixed: even a tiny relative shape error here would
    // displace the mean by many standard deviations. The saturated score
    // should return the representable sample mean exactly.
    let large = [1e150; 3];
    let fitted = FoldedNormal::try_fit(&large).unwrap();
    assert_eq!(fitted.c, 1e150);
    assert!(likelihood(&fitted, &large).is_finite());
    let separated = [1.00000000000005e150, 1.0000000000001e150];
    let fitted = FoldedNormal::try_fit(&separated).unwrap();
    let mean = separated[1] * (0.5 * (separated[0] / separated[1]) + 0.5);
    assert_eq!(fitted.c, mean);
    let near_boundary = [1.0 + f64::EPSILON];
    let fitted = FoldedNormal::try_fit(&near_boundary).unwrap();
    let expected = (6.0 * f64::EPSILON).sqrt();
    assert!((fitted.c / expected - 1.0).abs() < 1e-12);
}

#[test]
fn student_t_mle_solves_the_standardized_likelihood() {
    // Score roots evaluated independently at 70 decimal digits (mpmath).
    // The old quantile-ratio estimate could not return any df below one,
    // and rejected a constant sample even though loc and scale are fixed.
    for (data, expected_df) in [
        (vec![7.0, 7.0, 7.0], 0.426_146_305_754_734_5),
        (vec![-20.0, -2.0, 0.0, 1.0, 100.0], 0.396_291_473_268_043_1),
        (
            vec![-2.0, -0.3, 0.0, 0.5, 3.0, 10.0],
            0.898_808_574_240_352_6,
        ),
    ] {
        let fitted = StudentT::try_fit(&data).unwrap();
        assert!(
            (fitted.df / expected_df - 1.0).abs() < 2.0e-9,
            "df={} vs {expected_df}",
            fitted.df
        );
        assert_eq!(fitted, StudentT::fit(&data));
        let nll = likelihood(&fitted, &data);
        for factor in [0.5, 2.0] {
            assert!(likelihood(&StudentT::new(factor * fitted.df), &data) > nll + 1.0e-4);
        }
        let step = 1.0e-5_f64;
        let left = StudentT::new(fitted.df * (-step).exp());
        let right = StudentT::new(fitted.df * step.exp());
        let score = (likelihood(&right, &data) - likelihood(&left, &data)) / (2.0 * step);
        assert!(score.abs() < 1.0e-7, "log-df likelihood score={score}");
    }
    // Repeated observations and a single observation are identifiable for
    // this one-parameter standardized model.
    let single = StudentT::try_fit(&[7.0]).unwrap();
    assert!((single.df - 0.426_146_305_754_734_5).abs() < 1.0e-9);
}

#[test]
fn student_t_mle_rejects_gaussian_boundary_and_invalid_data() {
    assert!(matches!(
        StudentT::try_fit(&[]),
        Err(FitError::InsufficientData { .. })
    ));
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            StudentT::try_fit(&[7.0, bad]),
            Err(FitError::UnsupportedData(_))
        ));
    }
    for data in [vec![0.0; 3], vec![-1.0, 0.0, 1.0], vec![0.1, 0.3, 0.7]] {
        assert!(matches!(
            StudentT::try_fit(&data),
            Err(FitError::NonConvergent(_))
        ));
        assert!(StudentT::fit(&data).df.is_nan());
    }
    with_stats_runtime_mode(RuntimeMode::Hardened, || {
        with_hardened_stats_caps(2, 2, || {
            assert!(matches!(
                StudentT::try_fit(&[7.0; 3]),
                Err(FitError::ResourceExhausted { .. })
            ));
            assert!(StudentT::try_fit(&[7.0; 2]).is_ok());
        });
    });
}

#[test]
fn student_t_mle_retains_extreme_tails_and_large_finite_degrees_of_freedom() {
    let data = [-1.0e200, 0.0, 1.0e200];
    let fitted = StudentT::try_fit(&data).unwrap();
    let expected = 0.002_693_896_956_872_523_3;
    assert!((fitted.df / expected - 1.0).abs() < 2.0e-9);
    assert!(likelihood(&fitted, &data).is_finite());
    assert!((StudentT::try_fit(&[1.0e200, 0.0, 1.0e200]).unwrap().df - fitted.df).abs() < 1.0e-12);

    // 100-digit score roots for the exact binary64 observations. These have
    // finite MLEs close to, but distinct from, the Gaussian boundary.
    for (x, expected_df) in [
        (1.5538, 31_058.007_225_042_71),
        (1.553774, 31_124_588.279_881_88),
        (1.553773975, 833_335_612.053_296_3),
    ] {
        let fitted = StudentT::try_fit(&[x]).unwrap();
        assert!(
            (fitted.df / expected_df - 1.0).abs() < 2.0e-6,
            "x={x}: df={} vs {expected_df}",
            fitted.df
        );
    }
}

#[test]
fn student_t_density_keeps_finite_power_law_tails_and_the_gaussian_limit() {
    // 70-digit references, with 450 digits for df=1e300 and the least
    // positive subnormal df to resolve the log-gamma difference.
    for (df, x, expected) in [
        (0.5, 1.0e200, -692.605_921_209_545_2),
        (1.0e10, 1.0, -1.418_938_533_254_672_7),
        (1.0e300, 2.0, -2.918_938_533_204_672_7),
        (1.0e-300, 1.0e-160, -346.080_911_129_666_8),
        (f64::from_bits(1), 1.0, -745.133_219_101_941_2),
        (f64::from_bits(1), 0.0, -372.913_183_141_250_6),
    ] {
        let distribution = StudentT::new(df);
        let actual = distribution.logpdf(x);
        assert!(
            (actual - expected).abs() < 2.0e-12,
            "df={df:e}, x={x:e}: {actual} vs {expected}"
        );
        assert_eq!(actual, distribution.logpdf(-x));
        let density = distribution.pdf(x);
        let reference = expected.exp();
        assert!(
            (density - reference).abs() <= 2.0e-12 * reference + f64::from_bits(1),
            "df={df:e}, x={x:e}: density={density:e} vs {reference:e}"
        );
        let points = [-1.0e200, -2.0, 0.0, 1.0, 1.0e200];
        let densities = distribution.pdf_many(&points);
        let log_densities = distribution.logpdf_many(&points);
        for (index, &point) in points.iter().enumerate() {
            assert_eq!(
                densities[index].to_bits(),
                distribution.pdf(point).to_bits()
            );
            assert_eq!(
                log_densities[index].to_bits(),
                distribution.logpdf(point).to_bits()
            );
        }
        assert_eq!(distribution.logpdf(f64::INFINITY), f64::NEG_INFINITY);
        assert_eq!(distribution.logpdf(f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert!(distribution.logpdf(f64::NAN).is_nan());
    }
    let normal = Normal::new(0.0, 1.0);
    let gaussian_limit = StudentT::new(f64::INFINITY);
    for x in [-4.0, 0.0, 1.0, 10.0] {
        assert!((gaussian_limit.logpdf(x) - normal.logpdf(x)).abs() < 1.0e-13);
    }
}

#[test]
fn student_t_mle_matches_live_scipy_with_fixed_location_and_scale() {
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
        eprintln!("SKIP StudentT MLE: SciPy oracle unavailable; compared cases = 0");
        return;
    }
    let script = r#"
import os
import numpy as np
import scipy
from scipy import stats, optimize
print('versions', scipy.__version__, np.__version__)
if os.environ.get('FSCI_REQUIRE_SCIPY_ORACLE') == '1':
    assert (scipy.__version__, np.__version__) == ('1.17.1', '2.4.3')
def accurate_fit(fun, start, args=(), disp=0):
    return optimize.fmin(fun, start, args=args, disp=disp,
                         xtol=1e-10, ftol=1e-10, maxiter=2000, maxfun=8000)
datasets = [stats.t.rvs(df, size=2000,
                       random_state=np.random.default_rng(85400 + case))
            for case, df in enumerate([.15, .5, 1., 2., 8., 30.])]
datasets.extend([np.array([7., 7., 7.]), np.array([-20., -2., 0., 1., 100.]),
                 np.array([-2., -.3, 0., .5, 3., 10.])])
for case, data in enumerate(datasets):
    candidates = [stats.t.fit(data, start, floc=0, fscale=1,
                              optimizer=accurate_fit)
                  for start in [.1, 1., 10.]]
    fitted = min(candidates, key=lambda p: -stats.t.logpdf(data, *p).sum())
    assert fitted[1:] == (0, 1)
    nll = -stats.t.logpdf(data, *fitted).sum()
    assert np.isfinite(nll) and fitted[0] > 0
    print(case, format(fitted[0], '.17g'), format(nll, '.17g'))
    print(' '.join(format(x, '.17g') for x in data))
"#;
    let output = Command::new(&python).args(["-c", script]).output().unwrap();
    assert!(
        output.status.success(),
        "SciPy StudentT MLE oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let mut lines = stdout.lines();
    let versions = lines.next().expect("oracle version metadata");
    eprintln!("StudentT MLE oracle {versions}");
    if required {
        assert_eq!(versions, "versions 1.17.1 2.4.3");
    }
    let mut compared = 0;
    while let Some(header) = lines.next() {
        let fields: Vec<_> = header.split_whitespace().collect();
        assert_eq!(fields.len(), 3, "malformed oracle header");
        let case: usize = fields[0].parse().unwrap();
        let expected_df: f64 = fields[1].parse().unwrap();
        let expected_nll: f64 = fields[2].parse().unwrap();
        let data: Vec<f64> = lines
            .next()
            .expect("oracle observations")
            .split_whitespace()
            .map(|value| value.parse().unwrap())
            .collect();
        assert_eq!(
            data.len(),
            if case < 6 { 2000 } else { [3, 5, 6][case - 6] }
        );
        let fitted = StudentT::try_fit(&data).unwrap();
        let nll = likelihood(&fitted, &data);
        assert!(
            (fitted.df / expected_df - 1.0).abs() < 1.0e-5,
            "case {case}: df {} vs {expected_df}",
            fitted.df
        );
        assert!(
            (nll - expected_nll).abs() <= 1.0e-10 * expected_nll.abs().max(1.0),
            "case {case}: NLL {nll} vs {expected_nll}"
        );
        eprintln!(
            "StudentT/{case}: df fsci={:.12e} SciPy={expected_df:.12e}; NLL fsci={nll:.12e} SciPy={expected_nll:.12e}",
            fitted.df
        );
        compared += 1;
    }
    assert_eq!(
        compared, 9,
        "all standardized StudentT fits must be compared"
    );
    eprintln!("StudentT MLE compared cases = {compared}");
}

#[test]
fn exponnorm_fit_optimizes_the_standardized_likelihood() {
    let data = [-2.0, 0.3, 0.4, 1.0, 3.0, 9.0];
    let fitted = ExponNorm::try_fit(&data).unwrap();
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    assert!(likelihood(&ExponNorm::new(mean), &data) - likelihood(&fitted, &data) > 0.03);
    assert_eq!(fitted, ExponNorm::fit(&data));
    let step = 1e-5_f64;
    let left = ExponNorm::new(fitted.k * (-step).exp());
    let right = ExponNorm::new(fitted.k * step.exp());
    let score = (likelihood(&right, &data) - likelihood(&left, &data)) / (2.0 * step);
    assert!(score.abs() < 2e-6, "log-shape likelihood score={score}");

    // A nonpositive mean does not rule out an interior shape maximum.
    let negative_mean = [-10.0, 9.0];
    let fitted = ExponNorm::try_fit(&negative_mean).unwrap();
    assert!((fitted.k - 4.435_271_660_447_556).abs() < 2e-6);
    assert!(
        likelihood(&Normal::new(0.0, 1.0), &negative_mean) - likelihood(&fitted, &negative_mean)
            > 30.0
    );

    // The normal-boundary likelihood can overflow while a finite interior
    // exponential-tail likelihood remains well conditioned.
    let huge = [1e155, 2e155];
    let fitted = ExponNorm::try_fit(&huge).unwrap();
    assert!((fitted.k / 1.5e155 - 1.0).abs() < 2e-6);
    assert!(likelihood(&fitted, &huge).is_finite());
}

#[test]
fn exponnorm_fit_selects_between_separated_likelihood_maxima() {
    // Independently solved likelihood scores at 60 decimal digits (mpmath).
    // The larger mode wins in the first sample; the smaller wins in the
    // second. A sole mean start, or always choosing the largest root, fails.
    for (zeros, outlier, expected, inferior) in [
        (
            1000,
            20.0,
            0.207_897_708_759_834_3,
            0.013_782_264_693_285_957,
        ),
        (
            500,
            10.0,
            0.011_241_494_870_528_716,
            0.157_453_776_582_290_7,
        ),
    ] {
        let mut data = vec![0.0; zeros];
        data.push(outlier);
        let fitted = ExponNorm::try_fit(&data).unwrap();
        assert!(
            (fitted.k - expected).abs() < 3e-6 * expected,
            "zeros={zeros}, outlier={outlier}: {} vs {expected}",
            fitted.k
        );
        assert!(likelihood(&ExponNorm::new(inferior), &data) - likelihood(&fitted, &data) > 2.0);
    }
}

#[test]
fn exponnorm_fit_handles_boundary_invalid_data_and_resource_caps() {
    for data in [&[][..], &[1.0][..]] {
        assert!(matches!(
            ExponNorm::try_fit(data),
            Err(FitError::InsufficientData { .. })
        ));
    }
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            ExponNorm::try_fit(&[1.0, invalid]),
            Err(FitError::UnsupportedData(_))
        ));
    }
    for data in [vec![0.0, 0.0], vec![-1.0, -2.0], vec![-0.1, 0.1]] {
        assert!(matches!(
            ExponNorm::try_fit(&data),
            Err(FitError::NonConvergent(_))
        ));
    }
    // MAX/3 rounds upward: summing the three prescaled observations can
    // overflow. The profile must reject infinity instead of halving forever.
    assert!(matches!(
        ExponNorm::try_fit(&[f64::MAX; 3]),
        Err(FitError::NonConvergent(_))
    ));
    with_stats_runtime_mode(RuntimeMode::Hardened, || {
        with_hardened_stats_caps(2, 2, || {
            assert!(matches!(
                ExponNorm::try_fit(&[0.0, 1.0, 2.0]),
                Err(FitError::ResourceExhausted { .. })
            ));
            assert!(ExponNorm::try_fit(&[1.0, 1.0]).is_ok());
        });
    });
}

#[test]
fn exponnorm_density_and_probabilities_retain_the_normal_limit() {
    let normal = Normal::new(0.0, 1.0);
    for shape in [1e-50, 1e-200, f64::from_bits(1)] {
        let distribution = ExponNorm::new(shape);
        for x in [-20.0, -2.0, 0.0, 2.0, 20.0] {
            assert!((distribution.logpdf(x) - normal.logpdf(x)).abs() < 1e-13);
            for (actual, expected) in [
                (distribution.pdf(x), normal.pdf(x)),
                (distribution.cdf(x), normal.cdf(x)),
                (distribution.sf(x), normal.sf(x)),
            ] {
                assert!((actual - expected).abs() <= 2e-13 * expected);
            }
        }
        assert_eq!(distribution.logpdf(f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert_eq!(distribution.logpdf(f64::INFINITY), f64::NEG_INFINITY);
        assert_eq!(distribution.cdf(f64::NEG_INFINITY), 0.0);
        assert_eq!(distribution.cdf(f64::INFINITY), 1.0);
        assert_eq!(distribution.sf(f64::NEG_INFINITY), 1.0);
        assert_eq!(distribution.sf(f64::INFINITY), 0.0);
    }
    // 80-digit evaluation of the defining erfc expression. These values
    // exercise the small-shape cancellation and both sides of a=32.
    for (shape, x, expected) in [
        (0.0001, -2.0, -2.919_138_523_203_34),
        (0.0001, 0.0, -0.918_938_543_204_672_5),
        (0.0001, 2.0, -2.918_738_523_206_007),
        (0.01, 0.0, -0.919_038_508_216_997_3),
        (0.03125, -2.0, -2.980_426_344_071_958_5),
        (0.03125, 0.0, -0.919_912_722_925_647_6),
        (0.03125, 2.0, -2.855_508_053_543_470_6),
    ] {
        let actual = ExponNorm::new(shape).logpdf(x);
        assert!(
            (actual - expected).abs() < 2e-14,
            "k={shape}, x={x}: {actual}"
        );
    }
    let distribution = ExponNorm::new(2.0);
    assert!((distribution.logpdf(0.5) + 0.125 + 4.0_f64.ln()).abs() < 1e-15);
    for shape in [1e100, 1e200, 1e308] {
        let distribution = ExponNorm::new(shape);
        for x in [-2.0, 0.0, 2.0] {
            assert!((0.0..=1.0).contains(&distribution.cdf(x)));
            assert!((0.0..=1.0).contains(&distribution.sf(x)));
            assert!((distribution.cdf(x) + distribution.sf(x) - 1.0).abs() < 1e-15);
        }
    }
}

#[test]
fn exponnorm_mle_matches_live_scipy_with_fixed_location_and_scale() {
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
        eprintln!("SKIP ExponNorm MLE: SciPy oracle unavailable; compared cases = 0");
        return;
    }
    let script = r#"
import os
import numpy as np
import scipy
from scipy import stats, optimize
print('versions', scipy.__version__, np.__version__)
if os.environ.get('FSCI_REQUIRE_SCIPY_ORACLE') == '1':
    assert (scipy.__version__, np.__version__) == ('1.17.1', '2.4.3')
def accurate_fit(fun, start, args=(), disp=0):
    return optimize.fmin(fun, start, args=args, disp=disp,
                         xtol=1e-10, ftol=1e-10, maxiter=2000, maxfun=8000)
datasets = [stats.exponnorm.rvs(k, size=2000,
                              random_state=np.random.default_rng(78300 + case))
            for case, k in enumerate([.15, .7, 3., 20.])]
datasets.extend([np.array([-10., 9.]), np.r_[np.zeros(1000), 20.],
                 np.r_[np.zeros(500), 10.]])
for case, data in enumerate(datasets):
    # The shape likelihood can have separated interior maxima. Give the
    # independent SciPy optimizer starts on both sides of those basins.
    candidates = [stats.exponnorm.fit(data, start, floc=0, fscale=1,
                                      optimizer=accurate_fit)
                  for start in [.01, .1, 1., 10.]]
    fitted = min(candidates, key=lambda p: -stats.exponnorm.logpdf(data, *p).sum())
    assert fitted[1:] == (0, 1)
    nll = -stats.exponnorm.logpdf(data, *fitted).sum()
    print(case, format(fitted[0], '.17g'), format(nll, '.17g'))
    print(' '.join(format(x, '.17g') for x in data))
"#;
    let output = Command::new(&python).args(["-c", script]).output().unwrap();
    assert!(
        output.status.success(),
        "SciPy ExponNorm MLE oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let mut lines = stdout.lines();
    let versions = lines.next().expect("oracle version metadata");
    eprintln!("ExponNorm MLE oracle {versions}");
    if required {
        assert_eq!(versions, "versions 1.17.1 2.4.3");
    }
    let mut compared = 0;
    while let Some(header) = lines.next() {
        let fields: Vec<_> = header.split_whitespace().collect();
        let case: usize = fields[0].parse().unwrap();
        let expected_shape: f64 = fields[1].parse().unwrap();
        let expected_nll: f64 = fields[2].parse().unwrap();
        let data: Vec<f64> = lines
            .next()
            .expect("oracle observations")
            .split_whitespace()
            .map(|value| value.parse().unwrap())
            .collect();
        assert_eq!(
            data.len(),
            if case < 4 {
                2000
            } else {
                [2, 1001, 501][case - 4]
            }
        );
        let fitted = ExponNorm::try_fit(&data).unwrap();
        let nll = likelihood(&fitted, &data);
        assert!(
            (fitted.k - expected_shape).abs() <= 3e-6 * expected_shape,
            "case {case}: shape {} vs {expected_shape}",
            fitted.k
        );
        assert!(
            (nll - expected_nll).abs() <= 1e-10 * expected_nll.abs().max(1.0),
            "case {case}: NLL {nll} vs {expected_nll}"
        );
        eprintln!(
            "ExponNorm/{case}: shape fsci={:.12e} SciPy={expected_shape:.12e}; NLL fsci={nll:.12e} SciPy={expected_nll:.12e}",
            fitted.k
        );
        compared += 1;
    }
    assert_eq!(
        compared, 7,
        "all standardized ExponNorm fits must be compared"
    );
    eprintln!("ExponNorm MLE compared cases = {compared}");
}

#[test]
fn cauchy_fit_maximizes_likelihood_instead_of_returning_quartiles() {
    let data = [-4.0, -1.0, 0.2, 0.4, 0.7, 2.0, 7.0];
    let fitted = Cauchy::try_fit(&data).unwrap();
    let quartiles = fsci_stats::quantile(&data, &[0.25, 0.5, 0.75]);
    let initial = Cauchy::new(quartiles[1], 0.5 * (quartiles[2] - quartiles[0]));
    assert!(likelihood(&initial, &data) - likelihood(&fitted, &data) > 0.005);
    assert_eq!(fitted, Cauchy::fit(&data));

    // Both likelihood scores must vanish. This fails for median/half-IQR
    // even though that estimator can look plausible on a symmetric sample.
    let mut location_score = 0.0;
    let mut scale_score = 0.0;
    for &x in &data {
        let z = (x - fitted.loc) / fitted.scale;
        location_score += 2.0 * z / (1.0 + z * z);
        scale_score += (z * z - 1.0) / (1.0 + z * z);
    }
    assert!(
        location_score.abs() < 2.0e-6,
        "location score={location_score}"
    );
    assert!(scale_score.abs() < 2.0e-6, "scale score={scale_score}");

    for factor in [1.0e-150_f64, 1.0e150] {
        let offset = -3.0 * factor;
        let transformed: Vec<_> = data.iter().map(|&x| factor.mul_add(x, offset)).collect();
        let actual = Cauchy::try_fit(&transformed).unwrap();
        assert!((actual.loc / factor + 3.0 - fitted.loc).abs() < 2.0e-6);
        assert!((actual.scale / factor - fitted.scale).abs() < 2.0e-6);
        let jacobian = data.len() as f64 * factor.ln();
        assert!(
            (likelihood(&actual, &transformed) - likelihood(&fitted, &data) - jacobian).abs()
                < 1.0e-9
        );
    }
}

#[test]
fn cauchy_fit_rejects_undefined_maxima_and_honors_sample_caps() {
    for data in [&[][..], &[1.0][..]] {
        assert!(matches!(
            Cauchy::try_fit(data),
            Err(FitError::InsufficientData { .. })
        ));
        assert!(Cauchy::fit(data).loc.is_nan());
    }
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            Cauchy::try_fit(&[1.0, invalid]),
            Err(FitError::UnsupportedData(_))
        ));
    }
    for data in [
        vec![1.0, 1.0],
        vec![-1.0, 0.0, 0.0, 0.0, 2.0],
        vec![0.0, 0.0, 1.0, 2.0],
        vec![-2.0, -1.0, 0.0, 0.0],
    ] {
        assert!(matches!(
            Cauchy::try_fit(&data),
            Err(FitError::NonConvergent(_))
        ));
    }
    // Equal mass on two distinct observations has nonunique finite maxima.
    // Choose the midpoint / half-distance deterministically.
    for data in [vec![1.0, 3.0], vec![1.0, 3.0, 1.0, 3.0]] {
        assert_eq!(Cauchy::try_fit(&data).unwrap(), Cauchy::new(2.0, 1.0));
    }
    assert_eq!(
        Cauchy::try_fit(&[-f64::MAX, f64::MAX]).unwrap(),
        Cauchy::new(0.0, f64::MAX)
    );
    with_stats_runtime_mode(RuntimeMode::Hardened, || {
        with_hardened_stats_caps(2, 2, || {
            assert!(matches!(
                Cauchy::try_fit(&[-1.0, 0.0, 1.0]),
                Err(FitError::ResourceExhausted { .. })
            ));
            assert_eq!(Cauchy::try_fit(&[1.0, 3.0]).unwrap(), Cauchy::new(2.0, 1.0));
        });
    });
}

#[test]
fn cauchy_log_likelihood_retains_extreme_finite_observations() {
    let log_pi = std::f64::consts::PI.ln();
    for (distribution, x, expected) in [
        (Cauchy::default(), 1.0e300, -log_pi - 2.0 * 1.0e300_f64.ln()),
        (
            Cauchy::new(-1.0e308, 1.0),
            1.0e308,
            -log_pi - 2.0 * (1.0e308_f64.ln() + 2.0_f64.ln()),
        ),
        (
            Cauchy::new(0.0, 1.0e-300),
            1.0e300,
            1.0e-300_f64.ln() - log_pi - 2.0 * 1.0e300_f64.ln(),
        ),
        (Cauchy::new(0.0, 1.0e300), 0.0, -log_pi - 1.0e300_f64.ln()),
    ] {
        let actual = distribution.logpdf(x);
        assert!(actual.is_finite());
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "{distribution:?}, x={x}"
        );
    }
}

#[test]
fn cauchy_mle_matches_live_scipy_with_free_and_fixed_parameters() {
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
        eprintln!("SKIP Cauchy MLE: SciPy oracle unavailable; compared cases = 0");
        return;
    }
    let script = r#"
import os
import numpy as np
import scipy
from scipy import stats, optimize
print('versions', scipy.__version__, np.__version__)
if os.environ.get('FSCI_REQUIRE_SCIPY_ORACLE') == '1':
    assert (scipy.__version__, np.__version__) == ('1.17.1', '2.4.3')
def accurate_fit(fun, start, args=(), disp=0):
    return optimize.fmin(fun, start, args=args, disp=disp,
                         xtol=1e-10, ftol=1e-10, maxiter=2000, maxfun=8000)
for case, (loc, scale) in enumerate([(0., 1.), (3., .2), (-5., 4.)]):
    data = stats.cauchy.rvs(loc=loc, scale=scale, size=2000,
                            random_state=np.random.default_rng(71800 + case))
    for kind, kwargs in [('free', {}), ('floc', {'floc': loc}),
                         ('fscale', {'fscale': scale})]:
        fitted = stats.cauchy.fit(data, optimizer=accurate_fit, **kwargs)
        nll = -stats.cauchy.logpdf(data, *fitted).sum()
        print(case, kind, *(format(x, '.17g') for x in fitted), format(nll, '.17g'))
        print(' '.join(format(x, '.17g') for x in data))
"#;
    let output = Command::new(&python).args(["-c", script]).output().unwrap();
    assert!(
        output.status.success(),
        "SciPy Cauchy MLE oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let mut lines = stdout.lines();
    let versions = lines.next().expect("oracle version metadata");
    eprintln!("Cauchy MLE oracle {versions}");
    if required {
        assert_eq!(versions, "versions 1.17.1 2.4.3");
    }
    let mut compared = 0;
    while let Some(header) = lines.next() {
        let fields: Vec<_> = header.split_whitespace().collect();
        let case: usize = fields[0].parse().unwrap();
        let kind = fields[1];
        let expected_loc: f64 = fields[2].parse().unwrap();
        let expected_scale: f64 = fields[3].parse().unwrap();
        let expected_nll: f64 = fields[4].parse().unwrap();
        let data: Vec<f64> = lines
            .next()
            .expect("oracle observations")
            .split_whitespace()
            .map(|value| value.parse().unwrap())
            .collect();
        assert_eq!(data.len(), 2000);
        let fitted = if kind == "free" {
            Cauchy::try_fit(&data).unwrap()
        } else {
            let quartiles = fsci_stats::quantile(&data, &[0.25, 0.5, 0.75]);
            let parameters = [
                if kind == "floc" {
                    MleParameter::Fixed(expected_loc)
                } else {
                    MleParameter::Free(quartiles[1])
                },
                if kind == "fscale" {
                    MleParameter::Fixed(expected_scale)
                } else {
                    MleParameter::Positive(0.5 * (quartiles[2] - quartiles[0]))
                },
            ];
            fit_mle(
                &data,
                &parameters,
                |p| {
                    Some(Cauchy {
                        loc: p[0],
                        scale: p[1],
                    })
                },
                MleOptions::default(),
            )
            .unwrap()
            .distribution
        };
        let nll = likelihood(&fitted, &data);
        assert!(nll <= expected_nll + 1.0e-8 * expected_nll.abs().max(1.0));
        assert!(
            (fitted.loc - expected_loc).abs() <= 2.0e-6 * expected_loc.abs().max(expected_scale),
            "{case}/{kind}: location {} != {expected_loc}",
            fitted.loc
        );
        assert!(
            (fitted.scale - expected_scale).abs() <= 2.0e-6 * expected_scale,
            "{case}/{kind}: scale {} != {expected_scale}",
            fitted.scale
        );
        if kind == "floc" {
            assert_eq!(fitted.loc.to_bits(), expected_loc.to_bits());
        } else if kind == "fscale" {
            assert_eq!(fitted.scale.to_bits(), expected_scale.to_bits());
        }
        eprintln!(
            "Cauchy/{case}/{kind}: fsci=({:.12e},{:.12e}) SciPy=({expected_loc:.12e},{expected_scale:.12e}) NLL fsci={nll:.12e} SciPy={expected_nll:.12e}",
            fitted.loc, fitted.scale
        );
        compared += 1;
    }
    assert_eq!(compared, 9, "all free/fixed Cauchy fits must be compared");
    eprintln!("Cauchy MLE compared cases = {compared}; samples per case = 2000");
}

#[test]
fn landau_and_trapezoid_mle_preserve_location_scale_equivariance() {
    let landau = Landau::new(2.0, 1.7);
    let trapezoid = Trapezoid::new(0.15, 0.65, -2.0, 4.0);
    let landau_data: Vec<_> = (0..256)
        .map(|i| landau.ppf((i as f64 + 0.5) / 256.0))
        .collect();
    let trapezoid_data: Vec<_> = (0..256)
        .map(|i| trapezoid.ppf((i as f64 + 0.5) / 256.0))
        .collect();
    let landau_fit = Landau::try_fit(&landau_data).unwrap();
    let trapezoid_fit = Trapezoid::try_fit(&trapezoid_data).unwrap();
    assert!(likelihood(&landau_fit, &landau_data) <= likelihood(&landau, &landau_data));
    assert!(likelihood(&trapezoid_fit, &trapezoid_data) <= likelihood(&trapezoid, &trapezoid_data));
    assert!((landau_fit.loc - 2.0).abs() < 0.1);
    assert!((landau_fit.scale - 1.7).abs() < 0.1);
    assert!((trapezoid_fit.c - 0.15).abs() < 0.1);
    assert!((trapezoid_fit.d - 0.65).abs() < 0.1);

    for factor in [1.0e-150_f64, 1.0e150] {
        let offset = -4.0 * factor;
        let scaled_landau: Vec<_> = landau_data
            .iter()
            .map(|&x| factor.mul_add(x, offset))
            .collect();
        let scaled_trapezoid: Vec<_> = trapezoid_data
            .iter()
            .map(|&x| factor.mul_add(x, offset))
            .collect();
        let fitted_landau = Landau::try_fit(&scaled_landau).unwrap();
        let fitted_trapezoid = Trapezoid::try_fit(&scaled_trapezoid).unwrap();
        for (actual, expected) in [
            (fitted_landau.loc / factor + 4.0, landau_fit.loc),
            (fitted_landau.scale / factor, landau_fit.scale),
            (fitted_trapezoid.loc / factor + 4.0, trapezoid_fit.loc),
            (fitted_trapezoid.scale / factor, trapezoid_fit.scale),
            (fitted_trapezoid.c, trapezoid_fit.c),
            (fitted_trapezoid.d, trapezoid_fit.d),
        ] {
            assert!(
                (actual - expected).abs() <= 2.0e-6 * expected.abs().max(1.0),
                "factor={factor}: fitted parameter {actual} != {expected}"
            );
        }
        let jacobian = landau_data.len() as f64 * factor.ln();
        assert!(
            (likelihood(&fitted_landau, &scaled_landau)
                - likelihood(&landau_fit, &landau_data)
                - jacobian)
                .abs()
                < 1.0e-8
        );
        assert!(
            (likelihood(&fitted_trapezoid, &scaled_trapezoid)
                - likelihood(&trapezoid_fit, &trapezoid_data)
                - jacobian)
                .abs()
                < 1.0e-8
        );
    }
}

#[test]
fn landau_and_trapezoid_mle_reject_invalid_samples_and_honor_caps() {
    for data in [&[][..], &[1.0][..]] {
        assert!(matches!(
            Landau::try_fit(data),
            Err(FitError::InsufficientData { .. })
        ));
        assert!(matches!(
            Trapezoid::try_fit(data),
            Err(FitError::InsufficientData { .. })
        ));
    }
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            Landau::try_fit(&[0.0, invalid]),
            Err(FitError::UnsupportedData(_))
        ));
        assert!(matches!(
            Trapezoid::try_fit(&[0.0, invalid]),
            Err(FitError::UnsupportedData(_))
        ));
    }
    for data in [[0.0, 0.0], [1.0e300, 1.0e300]] {
        assert!(matches!(
            Landau::try_fit(&data),
            Err(FitError::NonConvergent(_))
        ));
        assert!(matches!(
            Trapezoid::try_fit(&data),
            Err(FitError::NonConvergent(_))
        ));
    }
    // Landau's infinite right-tail variance is not a failure. A repeated
    // minimum is different: its likelihood is unbounded when its multiplicity
    // is more than half the sample, so a shrinking scale is not a valid fit.
    assert!(matches!(
        Landau::try_fit(&[0.0, 0.0, 1.0]),
        Err(FitError::NonConvergent(_))
    ));
    assert!(matches!(
        Trapezoid::try_fit(&[-f64::MAX, f64::MAX]),
        Err(FitError::NonConvergent(_))
    ));
    with_stats_runtime_mode(RuntimeMode::Hardened, || {
        with_hardened_stats_caps(2, 2, || {
            assert!(matches!(
                Landau::try_fit(&[0.0, 1.0, 2.0]),
                Err(FitError::ResourceExhausted { .. })
            ));
            assert!(matches!(
                Trapezoid::try_fit(&[0.0, 1.0, 2.0]),
                Err(FitError::ResourceExhausted { .. })
            ));
        });
    });
}

#[test]
fn landau_and_trapezoid_mle_reuse_exact_parameter_masks() {
    let landau = Landau::new(-3.0, 2.0);
    let data: Vec<_> = (0..256)
        .map(|i| landau.ppf((i as f64 + 0.5) / 256.0))
        .collect();
    let result = fit_mle(
        &data,
        &[MleParameter::Fixed(-3.0), MleParameter::Positive(1.0)],
        |p| {
            Some(Landau {
                loc: p[0],
                scale: p[1],
            })
        },
        MleOptions::default(),
    )
    .unwrap();
    assert_eq!(result.parameters[0], -3.0);
    assert!((result.parameters[1] - 2.0).abs() < 0.1);
    for multiplier in [0.9, 1.1] {
        let perturbed = Landau::new(-3.0, result.distribution.scale * multiplier);
        assert!(result.negative_log_likelihood < likelihood(&perturbed, &data));
    }

    let trapezoid = Trapezoid::new(0.2, 0.8, -2.0, 4.0);
    let data: Vec<_> = (0..256)
        .map(|i| trapezoid.ppf((i as f64 + 0.5) / 256.0))
        .collect();
    let result = fit_mle(
        &data,
        &[
            MleParameter::Fixed(0.2),
            MleParameter::Fixed(0.8),
            MleParameter::Fixed(-2.0),
            MleParameter::Positive(4.5),
        ],
        |p| {
            Some(Trapezoid {
                c: p[0],
                d: p[1],
                loc: p[2],
                scale: p[3],
            })
        },
        MleOptions::default(),
    )
    .unwrap();
    assert_eq!(&result.parameters[..3], &[0.2, 0.8, -2.0]);
    assert!((result.parameters[3] - 4.0).abs() < 0.1);
    for multiplier in [0.99, 1.01] {
        let perturbed = Trapezoid::new(0.2, 0.8, -2.0, result.distribution.scale * multiplier);
        assert!(result.negative_log_likelihood < likelihood(&perturbed, &data));
    }
}

const AFFINE_FIT_ORACLE: &str = r#"
import os
import numpy as np
import scipy
from scipy import stats

print('versions', scipy.__version__, np.__version__)
if os.environ.get('FSCI_REQUIRE_SCIPY_ORACLE') == '1':
    assert (scipy.__version__, np.__version__) == ('1.17.1', '2.4.3')

cases = [
    ('landau', stats.landau, [
        (0.0, 1.0), (-20.0, 0.03), (1e6, 25.0), (3.0, 1e-8)]),
    ('trapezoid', stats.trapezoid, [
        (0.2, 0.8, 3.0, 2.0), (0.1, 0.3, -5.0, 0.25),
        (0.7, 0.9, 12.0, 7.0), (0.0, 1.0, 0.0, 1.0),
        (0.5, 0.5, 0.0, 1.0), (0.0, 0.5, 0.0, 1.0), (0.5, 1.0, 0.0, 1.0)]),
]
for name, dist, parameters in cases:
    for case_index, p in enumerate(parameters):
        seed = (42000 if name == 'landau' else 41000) + case_index
        data = dist.rvs(*p[:-2], loc=p[-2], scale=p[-1], size=2000,
                        random_state=np.random.default_rng(seed))
        fitted = dist.fit(data)
        nll = -np.sum(dist.logpdf(data, *fitted[:-2], loc=fitted[-2], scale=fitted[-1]))
        true_nll = -np.sum(dist.logpdf(data, *p[:-2], loc=p[-2], scale=p[-1]))
        assert np.isfinite(nll) and np.isfinite(true_nll), (name, case_index)
        print(name, case_index, len(fitted), *(format(x, '.17g') for x in fitted),
              format(nll, '.17g'), format(true_nll, '.17g'))
        print(' '.join(format(x, '.17g') for x in data))
"#;

#[test]
fn landau_and_trapezoid_mle_match_live_scipy() {
    let python = std::env::var("FSCI_PYTHON").unwrap_or_else(|_| "python3".into());
    let required = std::env::var("FSCI_REQUIRE_SCIPY_ORACLE").as_deref() == Ok("1");
    let available = Command::new(&python)
        .args(["-c", "import numpy; import scipy"])
        .output()
        .is_ok_and(|output| output.status.success());
    if !available {
        assert!(
            !required,
            "required Landau/Trapezoid SciPy oracle is unavailable"
        );
        eprintln!("SKIP Landau/Trapezoid MLE: SciPy unavailable; compared cases = 0");
        return;
    }
    let output = Command::new(&python)
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("OMP_NUM_THREADS", "1")
        .args(["-c", AFFINE_FIT_ORACLE])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "affine-fit oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let mut lines = stdout.lines();
    let versions = lines.next().expect("oracle version metadata");
    assert!(versions.starts_with("versions "));
    if required {
        assert_eq!(versions, "versions 1.17.1 2.4.3");
    }
    eprintln!("Landau/Trapezoid MLE oracle {versions}");
    let mut compared = [0; 2];
    while let Some(header) = lines.next() {
        let mut fields = header.split_whitespace();
        let family = fields.next().unwrap();
        let case = fields.next().unwrap();
        let parameter_count: usize = fields.next().unwrap().parse().unwrap();
        let expected: Vec<f64> = fields
            .by_ref()
            .take(parameter_count)
            .map(|value| value.parse().unwrap())
            .collect();
        let expected_nll: f64 = fields.next().unwrap().parse().unwrap();
        let true_nll: f64 = fields.next().unwrap().parse().unwrap();
        assert!(fields.next().is_none());
        assert!(expected_nll.is_finite() && true_nll.is_finite());
        let data: Vec<f64> = lines
            .next()
            .expect("oracle observation row")
            .split_whitespace()
            .map(|value| value.parse().unwrap())
            .collect();
        assert_eq!(data.len(), 2000);
        let (actual, nll) = match family {
            "landau" => {
                let fitted = Landau::try_fit(&data).expect("Landau MLE");
                compared[0] += 1;
                (vec![fitted.loc, fitted.scale], likelihood(&fitted, &data))
            }
            "trapezoid" => {
                let fitted = Trapezoid::try_fit(&data).expect("Trapezoid MLE");
                assert!(fitted.c >= 0.0 && fitted.c <= fitted.d && fitted.d <= 1.0);
                compared[1] += 1;
                (
                    vec![fitted.c, fitted.d, fitted.loc, fitted.scale],
                    likelihood(&fitted, &data),
                )
            }
            _ => panic!("unexpected affine-fit family: {family}"),
        };
        assert_eq!(actual.len(), expected.len());
        assert!(nll.is_finite());
        for reference in [expected_nll, true_nll] {
            assert!(
                nll <= reference + 1.0e-6 * reference.abs().max(1.0),
                "{family}/{case}: NLL {nll} > {reference}; fsci={actual:?}, SciPy={expected:?}"
            );
        }
        eprintln!(
            "{family}/{case}: fsci={actual:?} scipy={expected:?}; NLL fsci={nll:.12e} scipy={expected_nll:.12e}"
        );
    }
    assert_eq!(
        compared,
        [4, 7],
        "all eleven seeded affine fits must be compared"
    );
}

#[test]
fn discrete_mle_preserves_likelihood_near_parameter_boundaries() {
    for mode in [RuntimeMode::Strict, RuntimeMode::Hardened] {
        with_stats_runtime_mode(mode, || {
            for (data, expected_p, expected_nll) in [
                (vec![2], 0.715_331_862_959_161_6, 1.591_439_908_596_979),
                (
                    vec![1, 1, 1, 1, 1, 2],
                    0.259_750_921_820_885_4,
                    2.920_886_545_434_61,
                ),
                (
                    vec![10_000_000, 20_000_000],
                    0.999_999_996_580_082_9,
                    38.972_113_490_036_82,
                ),
            ] {
                // Independent 65-digit score roots, not method-of-moments
                // reconstruction of the fitted distribution's own draws.
                let fitted = LogSeries::try_fit(&data).unwrap();
                assert!((fitted.p - expected_p).abs() <= 2.0 * f64::EPSILON);
                let nll = discrete_likelihood(&fitted, &data);
                assert!((nll - expected_nll).abs() < 2.0e-12, "{mode:?}: {nll}");
                for factor in [0.9, 1.1] {
                    let p = fitted.p * factor;
                    if p < 1.0 {
                        assert!(nll < discrete_likelihood(&LogSeries::new(p), &data));
                    }
                }
            }
            let mut rare = vec![1; 100_000];
            rare.push(2);
            let fitted = LogSeries::try_fit(&rare).unwrap();
            assert!((fitted.p / 1.999_946_668_044_41e-5 - 1.0).abs() < 2.0e-12);
            assert!((fitted.mean() - (1.0 + 1.0 / rare.len() as f64)).abs() < 3.0e-15);

            let mut concentrated = vec![1; 999];
            concentrated.push(2);
            let fitted = Zipfian::try_fit_with_n(&concentrated, 20).unwrap();
            assert!((fitted.a - 10.006_520_963_236_412).abs() < 3.0e-13);
            let nll = discrete_likelihood(&fitted, &concentrated);
            assert!((nll - 7.925_542_267_572_123).abs() < 3.0e-11);
            for factor in [0.9, 1.1] {
                let perturbed = Zipfian::new(factor * fitted.a, fitted.n);
                assert!(nll < discrete_likelihood(&perturbed, &concentrated));
            }
        });
    }
}

const DISCRETE_FIT_ORACLE: &str = r#"
import os
import numpy as np
import scipy
from scipy import optimize, stats

print('versions', scipy.__version__, np.__version__)
if os.environ.get('FSCI_REQUIRE_SCIPY_ORACLE') == '1':
    assert (scipy.__version__, np.__version__) == ('1.17.1', '2.4.3')

def optimizer(fun, bounds, **kwargs):
    return optimize.differential_evolution(
        fun, bounds, rng=np.random.default_rng(27501),
        tol=1e-11, atol=1e-11, polish=True, **kwargs)

cases = [
    ('logser', stats.logser, [(0.2,), (0.7,), (0.95,)]),
    ('randint', stats.randint, [(-6, 4), (5, 15), (-3, -2)]),
    ('zipfian', stats.zipfian, [(0.2, 10), (1.2, 25), (3.0, 40)]),
    ('zipfian_fixed', stats.zipfian, [(0.2, 10), (1.2, 25), (3.0, 40)]),
]
for family_index, (name, dist, shapes) in enumerate(cases):
    for case_index, shape in enumerate(shapes):
        data = dist.rvs(*shape, size=2000,
                        random_state=np.random.default_rng(27400 + 17*family_index + case_index))
        if name == 'logser':
            bounds = {'p': (1e-12, 1-1e-12)}
        elif name == 'randint':
            bounds = {'low': (int(data.min())-5, int(data.min())),
                      'high': (int(data.max())+1, int(data.max())+6)}
        elif name == 'zipfian':
            bounds = {'a': (0, 20), 'n': (int(data.max()), int(data.max())+10)}
        else:
            bounds = {'a': (0, 20), 'n': (shape[1], shape[1])}
        result = stats.fit(dist, data, bounds=bounds, optimizer=optimizer)
        assert result.success, (name, case_index, result.message)
        if name == 'logser':
            parameters = (result.params.p, 0, 0)
        elif name == 'randint':
            parameters = (0, result.params.low, result.params.high)
        else:
            parameters = (result.params.a, result.params.n, 0)
        print(name, case_index, *(format(float(p), '.17g') for p in parameters),
              format(result.nllf(), '.17g'))
        print(' '.join(str(int(x)) for x in data))
"#;

#[test]
fn discrete_mle_matches_live_scipy_on_seeded_fits() {
    let python = std::env::var("FSCI_PYTHON").unwrap_or_else(|_| "python3".into());
    let required = std::env::var("FSCI_REQUIRE_SCIPY_ORACLE").as_deref() == Ok("1");
    let available = Command::new(&python)
        .args(["-c", "import numpy; import scipy"])
        .output()
        .is_ok_and(|output| output.status.success());
    if !available {
        assert!(
            !required,
            "required discrete-fit SciPy oracle is unavailable"
        );
        eprintln!("SKIP discrete MLE: SciPy oracle unavailable; compared cases = 0");
        return;
    }
    let output = Command::new(&python)
        .args(["-c", DISCRETE_FIT_ORACLE])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "discrete-fit SciPy oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let mut lines = stdout.lines();
    let versions = lines.next().expect("oracle version metadata");
    assert!(versions.starts_with("versions "));
    eprintln!("discrete MLE oracle {versions}");
    if required {
        assert_eq!(versions, "versions 1.17.1 2.4.3");
    }
    let mut compared = 0;
    while let Some(header) = lines.next() {
        let fields: Vec<&str> = header.split_whitespace().collect();
        assert_eq!(fields.len(), 6);
        let family = fields[0];
        let expected: Vec<f64> = fields[2..].iter().map(|s| s.parse().unwrap()).collect();
        let data: Vec<i64> = lines
            .next()
            .expect("oracle observation row")
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        assert_eq!(data.len(), 2000);
        let nll = match family {
            "logser" => {
                let fitted = LogSeries::try_fit(&data).unwrap();
                assert!((fitted.p - expected[0]).abs() < 2.0e-6);
                discrete_likelihood(&fitted, &data)
            }
            "randint" => {
                let fitted = RandInt::try_fit(&data).unwrap();
                assert_eq!(fitted.low as f64, expected[1]);
                assert_eq!(fitted.high as f64, expected[2]);
                discrete_likelihood(&fitted, &data)
            }
            "zipfian" | "zipfian_fixed" => {
                let fitted = if family == "zipfian_fixed" {
                    Zipfian::try_fit_with_n(&data, expected[1] as u32).unwrap()
                } else {
                    Zipfian::try_fit(&data).unwrap()
                };
                assert_eq!(f64::from(fitted.n), expected[1]);
                assert!((fitted.a - expected[0]).abs() < 2.0e-5);
                discrete_likelihood(&fitted, &data)
            }
            other => panic!("unexpected discrete-fit oracle family {other}"),
        };
        let scipy_nll = expected[3];
        assert!(nll.is_finite() && scipy_nll.is_finite());
        assert!(
            (nll - scipy_nll).abs() <= 1.0e-8 * (1.0 + scipy_nll.abs()),
            "{family}/{}: fsci NLL={nll}, SciPy NLL={scipy_nll}",
            fields[1]
        );
        eprintln!(
            "{family}/{}: fsci NLL={nll:.14}, SciPy NLL={scipy_nll:.14}",
            fields[1]
        );
        compared += 1;
    }
    assert_eq!(
        compared, 12,
        "all discrete-fit oracle cases must be compared"
    );
    eprintln!("discrete MLE compared cases = {compared}");
}

#[test]
fn rice_mle_finds_interior_and_boundary_likelihood_maxima() {
    // Independent 65-digit Bessel score roots and likelihoods at the exact
    // f64 observations. A valid low-power sample maximizes the likelihood
    // at b=0; a dispersed sample is not fitted by its second moment.
    for mode in [RuntimeMode::Strict, RuntimeMode::Hardened] {
        with_stats_runtime_mode(mode, || {
            for (data, expected_shape, expected_nll) in [
                (vec![0.1, 0.3, 0.8], 0.0, 4.099_701_448_634_192),
                (
                    vec![1.5, 2.0, 2.5, 1.8, 2.2, 1.7],
                    1.592_952_485_197_401_5,
                    5.338_886_178_802_261,
                ),
                (
                    vec![1.0, 2.0, 5.0, 10.0],
                    4.382_179_946_147_269,
                    28.798_957_737_546_45,
                ),
                (
                    vec![1000.0, 1001.0, 999.0, 1002.0],
                    1_000.499_500_249_500_6,
                    6.175_754_382_568_785,
                ),
            ] {
                let fitted = Rice::try_fit(&data).expect("finite Rice MLE");
                assert!(
                    (fitted.b - expected_shape).abs() <= 2.0e-12 * (1.0 + expected_shape),
                    "{mode:?}: b={} expected {expected_shape}",
                    fitted.b
                );
                let nll = likelihood(&fitted, &data);
                assert!((nll - expected_nll).abs() < 2.0e-11, "{mode:?}: NLL={nll}");
                assert_eq!(Rice::fit(&data), fitted);
            }
        });
    }
}

#[test]
fn normal_inverse_gaussian_mle_preserves_concentrated_and_skewed_samples() {
    // Independent 65-digit roots after profiling the skew. Concentrated
    // cases exercise the rationalized Jensen gap and complementary Bessel
    // ratio, where a direct score subtraction rounds to zero prematurely.
    for mode in [RuntimeMode::Strict, RuntimeMode::Hardened] {
        with_stats_runtime_mode(mode, || {
            for (data, a, b, expected_nll) in [
                (
                    vec![-2.0, -1.0, 0.0, 1.0, 2.0],
                    0.356_555_689_185_252_15,
                    0.0,
                    9.781_298_084_427_22,
                ),
                (
                    vec![-1.0, 0.5, 2.0, -0.3, 1.5, 0.8, -2.1, 3.0, 1.1, 0.2],
                    0.511_973_017_685_176_8,
                    0.253_530_596_808_591_3,
                    18.929_628_467_574_305,
                ),
                (
                    vec![0.9, 1.0, 1.1],
                    422.940_513_793_993,
                    299.064_105_342_255,
                    -3.258_191_925_375_759,
                ),
                (
                    vec![99.0, 100.0, 101.0],
                    1_500_075.009_375_531_2,
                    1_500_000.011_249_875,
                    3.648_617_941_202_396_2,
                ),
                (
                    vec![-1.0e-5, 0.0, 1.0e-5],
                    14_999_999_999.624_998,
                    0.0,
                    -30.890_158_457_421_414,
                ),
            ] {
                let fitted = NormInvGauss::try_fit(&data).expect("finite NIG MLE");
                assert!(fitted.a.is_finite() && fitted.a > 0.0);
                assert!(fitted.b.is_finite() && fitted.b.abs() < fitted.a);
                assert!(
                    (fitted.a - a).abs() <= 2.0e-10 * (1.0 + a),
                    "{mode:?}: a={} expected {a}",
                    fitted.a
                );
                assert!(
                    (fitted.b - b).abs() <= 2.0e-10 * (1.0 + b.abs()),
                    "{mode:?}: b={} expected {b}",
                    fitted.b
                );
                let nll = likelihood(&fitted, &data);
                assert!(nll.is_finite());
                assert!(
                    (nll - expected_nll).abs() < 1.0e-8 * (1.0 + expected_nll.abs()),
                    "{mode:?}: NLL={nll} expected {expected_nll}"
                );

                let mean = data.iter().sum::<f64>() / data.len() as f64;
                let q = mean.hypot(1.0);
                assert!((fitted.b / fitted.a - mean / q).abs() < 2.0e-15);
                let radial_score = data
                    .iter()
                    .map(|x| {
                        let radius = x.hypot(1.0);
                        let z = fitted.a * radius;
                        radius * fsci_special::bessel::k0e_scalar(z)
                            / fsci_special::bessel::k1e_scalar(z)
                    })
                    .sum::<f64>()
                    / data.len() as f64;
                assert!((q - radial_score).abs() < 2.0e-11 * q);
                for factor in [0.9, 1.1] {
                    let perturbed = NormInvGauss::new(factor * fitted.a, factor * fitted.b);
                    assert!(nll < likelihood(&perturbed, &data));
                }
                let infallible = NormInvGauss::fit(&data);
                assert_eq!(infallible.a, fitted.a);
                assert_eq!(infallible.b, fitted.b);
            }
        });
    }
}

#[test]
fn bessel_likelihoods_do_not_overflow_at_finite_high_shapes() {
    let log_two_pi = (2.0 * std::f64::consts::PI).ln();
    let rice = Rice::new(1.0e200);
    let rice_logpdf = rice.logpdf(1.0e200);
    assert!(rice_logpdf.is_finite());
    assert!((rice_logpdf + 0.5 * log_two_pi).abs() < 2.0e-12);
    assert!((rice.pdf(1.0e200) * (2.0 * std::f64::consts::PI).sqrt() - 1.0).abs() < 2.0e-12);
    assert_eq!(rice.logpdf(0.0), f64::NEG_INFINITY);
    assert_eq!(rice.logpdf(f64::INFINITY), f64::NEG_INFINITY);
    assert!(rice.logpdf(f64::NAN).is_nan());

    for a in [1.0e200, f64::MAX] {
        let distribution = NormInvGauss::new(a, 0.0);
        let actual = distribution.logpdf(0.0);
        let expected = 0.5 * (a.ln() - log_two_pi);
        assert!(actual.is_finite());
        assert!((actual - expected).abs() < 2.0e-12);
        let density = distribution.pdf(0.0);
        assert!(density.is_finite() && density > 0.0);
        assert!((density.ln() - actual).abs() < 2.0e-12);
        assert!(distribution.var().is_finite() && distribution.var() > 0.0);
        assert_eq!(distribution.logpdf(f64::INFINITY), f64::NEG_INFINITY);
        assert_eq!(distribution.logpdf(f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert!(distribution.logpdf(f64::NAN).is_nan());
    }
    let cauchy_limit = NormInvGauss::new(1.0e-310, 0.0);
    assert!((cauchy_limit.logpdf(0.0) + std::f64::consts::PI.ln()).abs() < 2.0e-12);
    assert!((cauchy_limit.pdf(0.0) * std::f64::consts::PI - 1.0).abs() < 2.0e-12);
}

#[test]
fn bessel_mle_rejects_undefined_data_and_honors_hardened_caps() {
    for data in [vec![], vec![1.0]] {
        assert!(matches!(
            Rice::try_fit(&data),
            Err(FitError::InsufficientData { .. })
        ));
        assert!(matches!(
            NormInvGauss::try_fit(&data),
            Err(FitError::InsufficientData { .. })
        ));
    }
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            Rice::try_fit(&[0.5, invalid, 2.0]),
            Err(FitError::UnsupportedData(_))
        ));
        assert!(matches!(
            NormInvGauss::try_fit(&[-1.0, invalid, 2.0]),
            Err(FitError::UnsupportedData(_))
        ));
    }
    for invalid in [0.0, -1.0] {
        assert!(matches!(
            Rice::try_fit(&[0.5, invalid, 2.0]),
            Err(FitError::UnsupportedData(_))
        ));
    }
    for constant in [-1.0, 0.0, 1.0] {
        assert!(matches!(
            NormInvGauss::try_fit(&[constant, constant]),
            Err(FitError::NonConvergent(_))
        ));
    }
    // Constant positive data still identify a fixed-scale Rice shape.
    let repeated = Rice::try_fit(&[5.0_f64.sqrt(), 5.0_f64.sqrt()]).unwrap();
    assert!((repeated.b - 1.960_636_775_996_733).abs() < 2.0e-12);

    with_stats_runtime_mode(RuntimeMode::Hardened, || {
        with_hardened_stats_caps(2, 2, || {
            assert!(matches!(
                Rice::try_fit(&[0.5, 1.0, 2.0]),
                Err(FitError::ResourceExhausted { .. })
            ));
            assert!(matches!(
                NormInvGauss::try_fit(&[-1.0, 0.0, 1.0]),
                Err(FitError::ResourceExhausted { .. })
            ));
            assert!(Rice::try_fit(&[0.5, 2.0]).is_ok());
            assert!(NormInvGauss::try_fit(&[-1.0, 1.0]).is_ok());
        });
    });
}

#[test]
fn nakagami_mle_fits_the_fixed_scale_likelihood() {
    let ordinary = vec![0.5, 0.7, 1.0, 1.2, 1.5, 0.8, 1.1];
    let scaled: Vec<_> = ordinary.iter().map(|x| 3.5 * x).collect();
    // Independent 100-digit roots and likelihoods at the exact f64 samples.
    // SciPy 1.17.1 fits agree to its default optimizer tolerance. The old
    // moment ratio gave identical shapes before/after rescaling, even though
    // the concrete distribution fixes scale=1. Constant data away from one
    // also have a finite shape-only MLE and must not be rejected as degenerate.
    for mode in [RuntimeMode::Strict, RuntimeMode::Hardened] {
        with_stats_runtime_mode(mode, || {
            for (data, expected_shape, expected_nll) in [
                (
                    ordinary.clone(),
                    2.552_027_326_371_895,
                    1.664_305_741_143_636_8,
                ),
                (
                    scaled.clone(),
                    0.087_759_956_055_839_38,
                    27.932_952_000_990_9,
                ),
                (
                    vec![2.0, 2.0],
                    0.406_692_729_164_263_7,
                    4.417_143_924_415_318,
                ),
                (
                    vec![0.5, 0.5],
                    0.916_687_435_099_460_8,
                    0.494_922_538_002_487_1,
                ),
            ] {
                let fitted = Nakagami::try_fit(&data).expect("finite fixed-scale Nakagami MLE");
                assert!(
                    (fitted.nu / expected_shape - 1.0).abs() < 2.0e-12,
                    "{mode:?}: shape {} != {expected_shape}",
                    fitted.nu
                );
                let nll = likelihood(&fitted, &data);
                assert!(
                    (nll - expected_nll).abs() < 2.0e-11,
                    "{mode:?}: likelihood {nll} != {expected_nll}"
                );
            }
        });
    }
}

#[test]
fn nakagami_mle_preserves_concentrated_samples_without_a_shape_ceiling() {
    // The second sample's likelihood gap is about 1.33e-20. Directly
    // subtracting x^2-1-2*log(x), or log(nu)-digamma(nu), loses it. The
    // references use 100-digit arithmetic on the exact f64 observations.
    for (data, expected_shape, expected_nll) in [
        (
            [0.99, 1.0, 1.01],
            3_750.072_905_352_076,
            -10.166_888_456_629_632,
        ),
        (
            [1.0 - 1.0e-10, 1.0, 1.0 + 1.0e-10],
            3.749_999_379_447_294e19,
            -65.428_934_604_148_5,
        ),
    ] {
        let fitted = Nakagami::try_fit(&data).expect("concentrated sample must fit");
        assert!((fitted.nu / expected_shape - 1.0).abs() < 3.0e-14);
        let nll = likelihood(&fitted, &data);
        assert!((nll - expected_nll).abs() < 1.0e-11, "NLL={nll}");
        assert!(fitted.mean().is_finite() && fitted.mean() > 0.0 && fitted.mean() <= 1.0);
        assert!(fitted.var().is_finite() && fitted.var() > 0.0);
        assert!(fitted.pdf(1.0).is_finite() && fitted.pdf(1.0) > 0.0);
    }
}

#[test]
fn nakagami_small_shapes_endpoints_and_large_shape_densities_remain_usable() {
    let small = Nakagami::new(0.25);
    assert_eq!(small.mode(), 0.0);
    assert_eq!(small.pdf(0.0), f64::INFINITY);
    assert_eq!(small.logpdf(0.0), f64::INFINITY);
    assert!((small.pdf(1.0) - 0.303_780_786_595_024_94).abs() < 1.0e-14);
    assert!((small.mean() - 0.675_978_240_067_284_8).abs() < 1.0e-14);
    let half = Nakagami::new(0.5);
    assert_eq!(half.mode(), 0.0);
    assert!((half.pdf(0.0) - (2.0 / std::f64::consts::PI).sqrt()).abs() < 1.0e-15);
    assert_eq!(Nakagami::new(1.0).pdf(0.0), 0.0);

    for shape in [1.0e-300, 1.0e-310] {
        let distribution = Nakagami::new(shape);
        // As nu tends to zero, f(1) / (2*nu) tends to one and
        // E[X] / sqrt(pi*nu) tends to one. Corrections at these shapes
        // are far below double-precision resolution.
        assert!((distribution.pdf(1.0) / (2.0 * shape) - 1.0).abs() < 2.0e-13);
        assert!(
            (distribution.mean() / (std::f64::consts::PI.sqrt() * shape.sqrt()) - 1.0).abs()
                < 1.0e-14
        );
        assert!(distribution.logpdf(1.0).is_finite());
        assert_eq!(distribution.var(), 1.0);
    }
    let diffuse_fit = Nakagami::try_fit(&[1.0e150, 1.0e150]).unwrap();
    assert!((diffuse_fit.nu / 1.0e-300 - 1.0).abs() < 2.0e-14);
    assert!(diffuse_fit.logpdf(1.0e150).is_finite());

    // Direct 100-digit gamma-ratio and density evaluations. At nu=1000
    // the old gamma exponentials were inf/inf. At nu=1e30 even subtracting
    // log-gammas (also used by SciPy's density) loses the finite answer.
    for (shape, density, mean, variance) in [
        (
            16.0,
            3.174_961_011_908_986,
            0.992_219_198_457_234_9,
            0.015_501_062_212_882_232,
        ),
        (
            1000.0,
            25.229_222_697_443,
            0.999_875_007_817_382_2,
            0.000_249_968_742_189_944_2,
        ),
        (
            1.0e10,
            79_788.456_079_621_63,
            0.999_999_999_987_5,
            2.499_999_999_968_75e-11,
        ),
        (1.0e30, 797_884_560_802_865.4, 1.0, 2.5e-31),
        (1.0e100, 7.978_845_608_028_654e49, 1.0, 2.5e-101),
    ] {
        let distribution = Nakagami::new(shape);
        assert!((distribution.pdf(1.0) / density - 1.0).abs() < 3.0e-14);
        assert!((distribution.logpdf(1.0) - density.ln()).abs() < 3.0e-14);
        assert!((distribution.mean() - mean).abs() < 3.0e-15);
        assert!((distribution.var() / variance - 1.0).abs() < 3.0e-14);
    }

    let points = [
        f64::NEG_INFINITY,
        -1.0,
        -0.0,
        0.0,
        0.3,
        1.0,
        100.0,
        f64::INFINITY,
        f64::NAN,
    ];
    for shape in [0.25, 0.5, 2.0, 16.0, 1000.0, 1.0e30, f64::MAX] {
        let distribution = Nakagami::new(shape);
        assert_eq!(distribution.pdf(f64::NEG_INFINITY), 0.0);
        assert_eq!(distribution.pdf(f64::INFINITY), 0.0);
        assert_eq!(distribution.logpdf(f64::INFINITY), f64::NEG_INFINITY);
        assert!(distribution.pdf(f64::NAN).is_nan());
        assert!(distribution.logpdf(f64::NAN).is_nan());
        for ((&x, density), log_density) in points
            .iter()
            .zip(distribution.pdf_many(&points))
            .zip(distribution.logpdf_many(&points))
        {
            assert_eq!(density.to_bits(), distribution.pdf(x).to_bits());
            assert_eq!(log_density.to_bits(), distribution.logpdf(x).to_bits());
        }
    }
}

#[test]
fn nakagami_mle_rejects_undefined_samples_and_honors_hardened_caps() {
    for data in [vec![], vec![1.0]] {
        assert!(matches!(
            Nakagami::try_fit(&data),
            Err(FitError::InsufficientData { .. })
        ));
    }
    for data in [
        [0.0, 1.0],
        [-1.0, 1.0],
        [f64::NAN, 1.0],
        [f64::INFINITY, 1.0],
        [f64::NEG_INFINITY, 1.0],
    ] {
        assert!(matches!(
            Nakagami::try_fit(&data),
            Err(FitError::UnsupportedData(_))
        ));
    }
    assert!(matches!(
        Nakagami::try_fit(&[1.0, 1.0]),
        Err(FitError::NonConvergent(_))
    ));
    assert!(matches!(
        Nakagami::try_fit(&[f64::MAX, f64::MAX]),
        Err(FitError::NonConvergent(_))
    ));
    with_stats_runtime_mode(RuntimeMode::Hardened, || {
        with_hardened_stats_caps(2, 2, || {
            assert!(matches!(
                Nakagami::try_fit(&[0.9, 1.0, 1.1]),
                Err(FitError::ResourceExhausted { .. })
            ));
            assert!(Nakagami::try_fit(&[0.9, 1.1]).is_ok());
        });
    });
}

#[test]
fn truncated_normal_fit_uses_exact_sample_endpoints_including_far_tails() {
    for data in [
        vec![-1.7, -0.3, 0.2, 1.1, 1.4, 2.2],
        vec![40.0, 40.005, 40.01, 40.03],
        vec![1.0e-200, 1.2e-200, 1.8e-200, 2.0e-200],
    ] {
        let fitted = TruncNormal::try_fit(&data).expect("finite nonconstant sample");
        assert_eq!(fitted.a, data.iter().copied().fold(f64::INFINITY, f64::min));
        assert_eq!(
            fitted.b,
            data.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        );
        assert!(likelihood(&fitted, &data).is_finite());
        let width = fitted.b - fitted.a;
        let wider = TruncNormal::new(fitted.a - width, fitted.b + width);
        assert!(likelihood(&fitted, &data) < likelihood(&wider, &data));
        // The generic fit surface must now reach this implementation, too.
        let generic = fsci_stats::fit::<TruncNormal>(&data);
        assert_eq!(generic.a, fitted.a);
        assert_eq!(generic.b, fitted.b);
    }
}

#[test]
fn truncated_normal_fit_respects_runtime_guards_and_explicit_projection() {
    use fsci_stats::{StatsOptions, active_stats_mode, hardened_max_sample_len, sync_audit_ledger};

    assert!(matches!(
        TruncNormal::try_fit(&[-f64::MAX, f64::MAX]),
        Err(FitError::NonConvergent(_))
    ));
    let old_cap = hardened_max_sample_len();
    with_hardened_stats_caps(2, 2, || {
        let sample = [-1.0, 0.0, 1.0];
        assert!(matches!(
            TruncNormal::try_fit_with_options(
                &sample,
                StatsOptions::new(RuntimeMode::Hardened),
                None,
            ),
            Err(FitError::ResourceExhausted { .. })
        ));
        assert!(
            TruncNormal::try_fit_with_options(
                &sample,
                StatsOptions::new(RuntimeMode::Strict),
                None,
            )
            .is_ok()
        );
        let ledger = sync_audit_ledger();
        let result = with_stats_runtime_mode(RuntimeMode::Hardened, || {
            TruncNormal::try_fit_with_audit(&sample, &ledger)
        });
        assert!(matches!(result, Err(FitError::ResourceExhausted { .. })));
        assert_eq!(ledger.lock().unwrap().len(), 1);
    });
    assert_eq!(hardened_max_sample_len(), old_cap);
    assert_eq!(active_stats_mode(), RuntimeMode::Strict);
    let ledger = sync_audit_ledger();
    let recovered = TruncNormal::try_fit_with_options(
        &[-1.0, f64::NAN, 2.0],
        StatsOptions::new(RuntimeMode::Hardened).with_non_finite_projection(true),
        Some(&ledger),
    )
    .unwrap();
    assert_eq!((recovered.a, recovered.b), (-1.0, 2.0));
    assert_eq!(ledger.lock().unwrap().len(), 1);
    assert_eq!(active_stats_mode(), RuntimeMode::Strict);
}

#[test]
fn truncated_pareto_fit_solves_both_signs_of_the_shape_score() {
    for b in [-2.0, 0.5, 2.0] {
        let original = TruncPareto::new(b, 6.0);
        let data: Vec<_> = (0..400)
            .map(|i| original.ppf((f64::from(i) + 0.5) / 400.0))
            .collect();
        let fitted = TruncPareto::try_fit(&data).unwrap();
        assert_eq!(fitted.c, data.iter().copied().fold(1.0, f64::max));
        assert_eq!(fitted.b.signum(), b.signum());
        assert!((fitted.b - b).abs() < 0.05);
        let width = fitted.c.ln();
        let empirical = data.iter().map(|x| x.ln()).sum::<f64>() / data.len() as f64;
        let score = fitted.b.recip() - width / (fitted.b * width).exp_m1() - empirical;
        assert!(score.abs() < 2.0e-13, "b={b}, score={score}");
        assert!(likelihood(&fitted, &data) <= likelihood(&original, &data));
        assert!(fitted.logpdf(fitted.c).is_finite());
    }
    // Small nonzero b has a finite log-uniform limit. The old power
    // difference lost significant digits in its normalizing denominator.
    let reference = 1.0 / (2.0 * 10.0_f64.ln());
    for b in [-1.0e-14, 1.0e-14] {
        let distribution = TruncPareto::new(b, 10.0);
        assert!((distribution.pdf(2.0) / reference - 1.0).abs() < 3.0e-14);
    }
}

#[test]
fn truncated_weibull_fit_profiles_bounds_and_identifies_narrow_support() {
    let original = TruncWeibullMin::new(2.0, 0.5, 3.0);
    let data: Vec<_> = (0..400)
        .map(|i| original.ppf((f64::from(i) + 0.5) / 400.0))
        .collect();
    let fitted = TruncWeibullMin::try_fit(&data).unwrap();
    assert_eq!(fitted.a, data[0]);
    assert_eq!(fitted.b, data[399]);
    assert!((fitted.c - 2.0).abs() < 0.1);
    assert!(fitted.logpdf(fitted.a).is_finite());
    assert!(likelihood(&fitted, &data) <= likelihood(&original, &data));

    // Shapes of ordinary magnitude are indistinguishable from c=0 on
    // this log-support scale, but a finite shape near 1e8 improves the
    // likelihood. The fitter must reach it instead of declaring a limit.
    let mut concentrated = vec![1.0; 12];
    concentrated[0] = 1.0 - 1.0e-8;
    concentrated[11] = 1.0 + 1.0e-8;
    let fitted = TruncWeibullMin::try_fit(&concentrated).unwrap();
    let log_width = ((fitted.b - fitted.a) / fitted.a).ln_1p();
    let boundary_nll = concentrated.iter().map(|x| x.ln()).sum::<f64>()
        + concentrated.len() as f64 * log_width.ln();
    assert!(fitted.c > 1.0e6);
    assert!(likelihood(&fitted, &concentrated) < boundary_nll - 0.5);
}

#[test]
fn truncated_weibull_density_retains_closed_boundary_and_extreme_normalization() {
    // For c=1 this is a truncated exponential, with a nonzero density
    // at its lower endpoint. Both endpoints follow SciPy's closed support.
    let exponential = TruncWeibullMin::new(1.0, 0.5, 2.0);
    let expected = 1.0 / -(-1.5_f64).exp_m1();
    assert!((exponential.pdf(0.5) - expected).abs() < 1.0e-14);
    assert_eq!(exponential.pdf(0.49), 0.0);
    assert!(exponential.pdf(2.0) > 0.0);
    assert!(TruncWeibullMin::new(0.5, 0.0, 2.0).pdf(0.0).is_infinite());
    assert_eq!(TruncWeibullMin::new(2.0, 0.0, 2.0).pdf(0.0), 0.0);
    assert!(TruncWeibullMin::new(1.0, 0.0, 2.0).pdf(0.0).is_finite());
    let far_tail = TruncWeibullMin::new(10.0, 10.0, 11.0);
    assert!((far_tail.pdf(10.0) / 1.0e10 - 1.0).abs() < 1.0e-13);
    let narrow = TruncWeibullMin::new(0.3, 1.0, 1.0 + 1.0e-12);
    assert!((narrow.pdf(1.0) * (narrow.b - narrow.a) - 1.0).abs() < 1.0e-10);
}

#[test]
fn truncated_fits_reject_invalid_data_and_unattained_maxima_without_panicking() {
    for data in [
        vec![],
        vec![1.0],
        vec![1.0, f64::NAN, 2.0],
        vec![1.0, f64::INFINITY, 2.0],
        vec![1.0, f64::NEG_INFINITY, 2.0],
        vec![2.0, 2.0, 2.0],
    ] {
        assert!(TruncNormal::try_fit(&data).is_err());
        assert!(TruncPareto::try_fit(&data).is_err());
        assert!(TruncWeibullMin::try_fit(&data).is_err());
    }
    assert!(matches!(
        TruncPareto::try_fit(&[0.5, 1.0, 2.0]),
        Err(FitError::UnsupportedData(_))
    ));
    assert!(matches!(
        TruncPareto::try_fit(&[1.0, std::f64::consts::E]),
        Err(FitError::NonConvergent(_))
    ));
    assert!(matches!(
        TruncPareto::try_fit(&[1.0, 1.0, 1.0]),
        Err(FitError::NonConvergent(_))
    ));
    assert!(matches!(
        TruncWeibullMin::try_fit(&[0.0, 1.0, 2.0]),
        Err(FitError::UnsupportedData(_))
    ));
    assert!(matches!(
        TruncWeibullMin::try_fit(&[-1.0, 1.0, 2.0]),
        Err(FitError::UnsupportedData(_))
    ));
    // The conditional Weibull density on [1, 2] tilts toward the lower
    // endpoint. These observations favor its excluded c -> 0 limit, so
    // an arbitrarily small positive optimizer result is not a finite MLE.
    assert!(matches!(
        TruncWeibullMin::try_fit(&[1.0, 2.0, 2.0]),
        Err(FitError::NonConvergent(_))
    ));
}

const TRUNCATED_FIT_ORACLE: &str = r#"
import numpy as np
import scipy
from scipy import stats

print('versions', scipy.__version__, np.__version__)
cases = [
    ('truncnorm', stats.truncnorm, [(-1., 2.), (5., 6.), (-6., -4.)]),
    ('truncpareto', stats.truncpareto, [(0.5, 5.), (2., 20.), (-2., 5.)]),
    ('truncweibull_min', stats.truncweibull_min, [(0.7, 0., 4.), (2., 0.5, 3.), (4., 1., 2.)]),
]
for family_index, (name, dist, shapes) in enumerate(cases):
    for case_index, shape in enumerate(shapes):
        data = dist.rvs(*shape, size=2000,
                        random_state=np.random.default_rng(30100 + 17*family_index + case_index))
        fitted = dist.fit(data, *shape, floc=0, fscale=1)
        nll = -np.sum(dist.logpdf(data, *fitted[:-2], loc=0, scale=1))
        assert np.all(np.isfinite(fitted)) and np.isfinite(nll)
        print(name, case_index, len(fitted)-2, *(format(x, '.17g') for x in fitted[:-2]),
              format(nll, '.17g'))
        print(' '.join(format(x, '.17g') for x in data))
"#;

#[test]
fn truncated_mle_matches_live_scipy_on_seeded_fits() {
    let python = std::env::var("FSCI_PYTHON").unwrap_or_else(|_| "python3".into());
    let required = std::env::var("FSCI_REQUIRE_SCIPY_ORACLE").as_deref() == Ok("1");
    let available = Command::new(&python)
        .args(["-c", "import scipy, numpy"])
        .output()
        .is_ok_and(|output| output.status.success());
    if !available {
        assert!(
            !required,
            "required truncated-fit SciPy oracle is unavailable"
        );
        eprintln!("SKIP: truncated-fit SciPy oracle unavailable");
        return;
    }
    let output = Command::new(&python)
        .args(["-c", TRUNCATED_FIT_ORACLE])
        .output()
        .expect("run truncated-fit oracle");
    assert!(
        output.status.success(),
        "truncated-fit oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let mut lines = stdout.lines();
    let versions = lines.next().expect("truncated-fit oracle provenance");
    assert!(versions.starts_with("versions "));
    if required {
        assert_eq!(versions, "versions 1.17.1 2.4.3");
    }
    eprintln!("Truncated MLE oracle {versions}");
    let mut compared = 0;
    for _ in 0..9 {
        let header: Vec<_> = lines
            .next()
            .expect("fit metadata")
            .split_whitespace()
            .collect();
        let family = header[0];
        let count: usize = header[2].parse().unwrap();
        assert_eq!(header.len(), count + 4);
        let expected: Vec<f64> = header[3..3 + count]
            .iter()
            .map(|value| value.parse().unwrap())
            .collect();
        let expected_nll: f64 = header[3 + count].parse().unwrap();
        let data: Vec<f64> = lines
            .next()
            .expect("shared seeded sample")
            .split_whitespace()
            .map(|value| value.parse().unwrap())
            .collect();
        assert_eq!(data.len(), 2000);
        let (parameters, nll) = match family {
            "truncnorm" => {
                let fitted = TruncNormal::try_fit(&data).unwrap();
                (vec![fitted.a, fitted.b], likelihood(&fitted, &data))
            }
            "truncpareto" => {
                let fitted = TruncPareto::try_fit(&data).unwrap();
                (vec![fitted.b, fitted.c], likelihood(&fitted, &data))
            }
            "truncweibull_min" => {
                let fitted = TruncWeibullMin::try_fit(&data).unwrap();
                (
                    vec![fitted.c, fitted.a, fitted.b],
                    likelihood(&fitted, &data),
                )
            }
            unknown => panic!("unexpected truncated-fit family {unknown}"),
        };
        assert_eq!(parameters.len(), expected.len());
        assert!(parameters.iter().all(|value| value.is_finite()));
        assert!(nll.is_finite());
        assert!(
            nll <= expected_nll + 1.0e-6 * expected_nll.abs().max(1.0),
            "{family}/{}: Rust NLL={nll}, SciPy NLL={expected_nll}; Rust={parameters:?}, SciPy={expected:?}",
            header[1]
        );
        eprintln!(
            "{family}/{}: Rust={parameters:?}, SciPy={expected:?}, NLL={nll}, SciPy NLL={expected_nll}",
            header[1]
        );
        compared += 1;
    }
    assert_eq!(compared, 9);
    assert!(lines.next().is_none(), "every oracle row must be consumed");
    eprintln!("Truncated MLE compared cases=9; families=3; samples per case=2000");
}

#[test]
fn location_scale_distribution_transforms_support_moments_and_probabilities() {
    let distribution = LocationScale::new(BetaDist::new(2.0, 3.0), -3.0, 5.0).unwrap();
    // Independent Beta(2,3) formulas: F(1/2)=11/16, f(1/2)=3/2,
    // mean=2/5, variance=1/25, skewness=2/7, excess kurtosis=-9/14.
    assert!((distribution.pdf(-0.5) - 0.3).abs() < 1.0e-15);
    assert!((distribution.logpdf(-0.5) - 0.3_f64.ln()).abs() < 1.0e-14);
    assert!((distribution.cdf(-0.5) - 0.6875).abs() < 1.0e-14);
    assert!((distribution.sf(-0.5) - 0.3125).abs() < 1.0e-14);
    assert!((distribution.logcdf(-0.5) - 0.6875_f64.ln()).abs() < 1.0e-14);
    assert!((distribution.logsf(-0.5) - 0.3125_f64.ln()).abs() < 1.0e-14);
    assert!((distribution.ppf(0.6875) + 0.5).abs() < 1.0e-12);
    assert!((distribution.isf(0.3125) + 0.5).abs() < 1.0e-12);
    assert_eq!(distribution.ppf(0.0), -3.0);
    assert_eq!(distribution.ppf(1.0), 2.0);
    assert_eq!(distribution.cdf(-4.0), 0.0);
    assert_eq!(distribution.sf(3.0), 0.0);
    assert_eq!(distribution.pdf(-4.0), 0.0);
    assert_eq!(distribution.logpdf(3.0), f64::NEG_INFINITY);
    assert!((distribution.mean() + 1.0).abs() < 1.0e-15);
    assert!((distribution.var() - 1.0).abs() < 1.0e-15);
    assert!((distribution.std() - 1.0).abs() < 1.0e-15);
    assert!((distribution.mode() + 4.0 / 3.0).abs() < 1.0e-15);
    assert!((distribution.skewness() - 2.0 / 7.0).abs() < 1.0e-15);
    assert!((distribution.kurtosis() + 9.0 / 14.0).abs() < 1.0e-15);
    assert!((distribution.entropy() - 1.374_531_262_646_099_5).abs() < 1.0e-13);
    assert_eq!(distribution.try_entropy().unwrap(), distribution.entropy());
    assert_eq!(
        distribution.try_skewness().unwrap(),
        distribution.skewness()
    );
    assert_eq!(
        distribution.try_kurtosis().unwrap(),
        distribution.kurtosis()
    );
    assert!((distribution.median() + 1.071_362_159_338_052_2).abs() < 1.0e-12);
    let (lower, upper) = distribution.interval(0.8);
    assert!((distribution.cdf(lower) - 0.1).abs() < 1.0e-12);
    assert!((distribution.sf(upper) - 0.1).abs() < 1.0e-12);
    assert_eq!(distribution.base().a, 2.0);
    assert_eq!(distribution.loc(), -3.0);
    assert_eq!(distribution.scale(), 5.0);
    assert_eq!(
        distribution.into_parts(),
        (BetaDist::new(2.0, 3.0), -3.0, 5.0)
    );
}

#[test]
fn location_scale_preserves_tail_accuracy_and_avoids_intermediate_overflow() {
    let distribution = LocationScale::new(Normal::standard(), 3.0, 2.0).unwrap();
    assert!(distribution.sf(23.0) > 0.0);
    assert!((distribution.logsf(83.0) + 804.608_442_013_753_8).abs() < 1.0e-11);
    assert!((distribution.logcdf(-77.0) + 804.608_442_013_753_8).abs() < 1.0e-11);
    let quantile = distribution.isf(1.0e-100);
    assert!(quantile.is_finite());
    assert!((distribution.logsf(quantile) - 1.0e-100_f64.ln()).abs() < 1.0e-10);

    let wide = LocationScale::new(Normal::standard(), -1.0e308, 1.0e308).unwrap();
    assert!((wide.cdf(1.0e308) - 0.977_249_868_051_820_8).abs() < 1.0e-15);
    assert!(wide.logpdf(1.0e308).is_finite());
    let cancelled = LocationScale::new(Normal::new(2.0, 1.0), -1.0e308, 1.0e308).unwrap();
    assert_eq!(cancelled.mean(), 1.0e308);

    let narrow = LocationScale::new(Normal::standard(), 0.0, 1.0e-200).unwrap();
    assert_eq!(Normal::standard().pdf(40.0), 0.0);
    let density = narrow.pdf(4.0e-199);
    assert!(density > 0.0 && density.is_finite());
    assert!((density.ln() - narrow.logpdf(4.0e-199)).abs() < 1.0e-12);
    // 100-digit Decimal evaluation at the exact f64 x and scale. Rescaling
    // the nonzero subnormal base PDF would lose about 1.2e-11 relatively.
    let reference = 1.097_221_052_007_641_3e-114;
    assert!((narrow.pdf(3.8e-199) / reference - 1.0).abs() < 3.0e-13);
    let rescued = LocationScale::new(Normal::new(0.0, 1.0e-310), 0.0, 1.0e100).unwrap();
    assert!(rescued.pdf(0.0).is_finite() && rescued.pdf(0.0) > 0.0);
    assert!((rescued.pdf(0.0).ln() - rescued.logpdf(0.0)).abs() < 1.0e-12);
    for (base_scale, outer_scale, variance) in
        [(1.0e-150, 1.0e200, 1.0e100), (1.0e150, 1.0e-200, 1.0e-100)]
    {
        let scaled = LocationScale::new(Normal::new(0.0, base_scale), 0.0, outer_scale).unwrap();
        assert!((scaled.var() / variance - 1.0).abs() < 1.0e-14);
        assert!((scaled.std() / variance.sqrt() - 1.0).abs() < 1.0e-14);
    }
}

#[test]
fn location_scale_validates_parameters_and_preserves_the_base_sampler() {
    for (loc, scale) in [
        (f64::NAN, 1.0),
        (f64::INFINITY, 1.0),
        (f64::NEG_INFINITY, 1.0),
        (0.0, 0.0),
        (0.0, -1.0),
        (0.0, f64::NAN),
        (0.0, f64::INFINITY),
    ] {
        assert!(matches!(
            LocationScale::new(Normal::standard(), loc, scale),
            Err(StatsError::InvalidArgument(_))
        ));
    }
    let base = GammaDist::new(2.0, 1.0);
    let distribution = LocationScale::new(base, -4.0, 3.0).unwrap();
    let mut base_rng = StdRng::seed_from_u64(4721);
    let mut affine_rng = StdRng::seed_from_u64(4721);
    let expected: Vec<_> = base
        .rvs(100, &mut base_rng)
        .into_iter()
        .map(|z| 3.0_f64.mul_add(z, -4.0))
        .collect();
    assert_eq!(distribution.rvs(100, &mut affine_rng), expected);
    assert!(matches!(
        LocationScale::<GammaDist>::try_fit(&[1.0, 2.0, 3.0]),
        Err(FitError::NotImplemented { .. })
    ));
}

#[test]
fn location_scale_mle_fits_free_location_scale_and_holds_shapes_exactly() {
    let data = [-3.0, 0.5, 1.2, 3.5, 8.0];
    let expected = Normal::try_fit(&data).unwrap();
    let fitted = fit_mle_location_scale(
        &data,
        &[],
        MleParameter::Free(0.0),
        MleParameter::Positive(1.0),
        |shapes| {
            assert!(shapes.is_empty());
            Some(Normal::standard())
        },
        MleOptions::default(),
    )
    .unwrap();
    assert!((fitted.distribution.loc() - expected.loc).abs() < 1.0e-6);
    assert!((fitted.distribution.scale() - expected.scale).abs() < 1.0e-6);
    assert!((fitted.negative_log_likelihood - likelihood(&expected, &data)).abs() < 1.0e-10);
    assert_eq!(fitted.parameters.len(), 2);

    let shifted = [3.2, 3.8, 4.5, 5.0, 6.1, 8.0];
    let fitted = fit_mle_location_scale(
        &shifted,
        &[MleParameter::Fixed(2.0)],
        MleParameter::Fixed(3.0),
        MleParameter::Positive(1.0),
        |shapes| Some(GammaDist::new(shapes[0], 1.0)),
        MleOptions::default(),
    )
    .unwrap();
    assert_eq!(&fitted.parameters[..2], &[2.0, 3.0]);
    assert_eq!(fitted.distribution.base().a, 2.0);
    assert_eq!(fitted.distribution.base().scale, 1.0);
    // For a known shape and location, scale MLE is mean(X-loc)/shape.
    assert!((fitted.distribution.scale() - 1.05).abs() < 1.0e-7);
}

#[test]
fn location_scale_censored_likelihood_adds_the_jacobian_only_for_exact_values() {
    let original = CensoredData::new(
        vec![0.0, 1.0],
        vec![-40.0],
        vec![40.0],
        vec![(40.0, 41.0), (-41.0, -40.0)],
    );
    let shifted = CensoredData::new(
        vec![3.0, 5.0],
        vec![-77.0],
        vec![83.0],
        vec![(83.0, 85.0), (-79.0, -77.0)],
    );
    let original_fit = fit_mle_censored(
        &original,
        &[],
        |_| Some(Normal::standard()),
        MleOptions::default(),
    )
    .unwrap();
    let fitted = fit_mle_location_scale_censored(
        &shifted,
        &[],
        MleParameter::Fixed(3.0),
        MleParameter::Fixed(2.0),
        |_| Some(Normal::standard()),
        MleOptions::default(),
    )
    .unwrap();
    assert_eq!(fitted.parameters, vec![3.0, 2.0]);
    assert_eq!((fitted.iterations, fitted.evaluations), (0, 1));
    let expected = original_fit.negative_log_likelihood + 2.0 * 2.0_f64.ln();
    assert!((fitted.negative_log_likelihood - expected).abs() < 1.0e-11);
    assert_eq!(shifted.interval, vec![(83.0, 85.0), (-79.0, -77.0)]);
}

#[test]
fn location_scale_mle_rejects_invalid_support_parameters_and_exhausted_budgets() {
    let shapes = [MleParameter::Fixed(2.0), MleParameter::Fixed(3.0)];
    let factory = |p: &[f64]| Some(BetaDist::new(p[0], p[1]));
    for data in [&[][..], &[4.9], &[9.1], &[f64::NAN]] {
        assert!(
            fit_mle_location_scale(
                data,
                &shapes,
                MleParameter::Fixed(5.0),
                MleParameter::Fixed(4.0),
                factory,
                MleOptions::default(),
            )
            .is_err()
        );
    }
    for scale in [0.0, -1.0, f64::INFINITY] {
        assert!(matches!(
            fit_mle_location_scale(
                &[6.0, 7.0],
                &shapes,
                MleParameter::Fixed(5.0),
                MleParameter::Fixed(scale),
                factory,
                MleOptions::default(),
            ),
            Err(FitError::UnsupportedData(_))
        ));
    }
    assert!(matches!(
        fit_mle_location_scale(
            &[0.0, 1.0, 3.0],
            &[],
            MleParameter::Free(0.0),
            MleParameter::Positive(1.0),
            |_| Some(Normal::standard()),
            MleOptions {
                max_evaluations: 1,
                ..MleOptions::default()
            },
        ),
        Err(FitError::NonConvergent(_))
    ));
    with_stats_runtime_mode(RuntimeMode::Hardened, || {
        with_hardened_stats_caps(1, 1, || {
            let factory = |_: &[f64]| -> Option<Normal> {
                panic!("sample cap must be checked before constructing a distribution")
            };
            assert!(matches!(
                fit_mle_location_scale(
                    &[0.0, 1.0],
                    &[],
                    MleParameter::Free(0.0),
                    MleParameter::Positive(1.0),
                    factory,
                    MleOptions::default(),
                ),
                Err(FitError::ResourceExhausted { .. })
            ));
            assert!(matches!(
                fit_mle_location_scale_censored(
                    &CensoredData::new(vec![], vec![0.0], vec![1.0], vec![]),
                    &[],
                    MleParameter::Free(0.0),
                    MleParameter::Positive(1.0),
                    factory,
                    MleOptions::default(),
                ),
                Err(FitError::ResourceExhausted { .. })
            ));
        });
    });
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
            "gengamma" => {
                let fitted = GenGamma::try_fit(&data).unwrap();
                (vec![fitted.a, fitted.c], likelihood(&fitted, &data))
            }
            "gennorm" => {
                let fitted = GenNorm::try_fit(&data).unwrap();
                (vec![fitted.beta], likelihood(&fitted, &data))
            }
            "halfgennorm" => {
                let fitted = HalfGenNorm::try_fit(&data).unwrap();
                (vec![fitted.beta], likelihood(&fitted, &data))
            }
            "ncx2" => {
                let fitted = NoncentralChiSquared::try_fit(&data).unwrap();
                (vec![fitted.df, fitted.nc], likelihood(&fitted, &data))
            }
            "skewnorm" => {
                let fitted = SkewNorm::try_fit(&data).unwrap();
                (vec![fitted.a], likelihood(&fitted, &data))
            }
            "erlang" => {
                let fitted = Erlang::try_fit(&data).unwrap();
                (
                    vec![fitted.k as f64, fitted.rate],
                    likelihood(&fitted, &data),
                )
            }
            "laplace_asymmetric" => {
                let fitted = LaplaceAsymmetric::try_fit(&data).unwrap();
                (vec![fitted.kappa], likelihood(&fitted, &data))
            }
            "dgamma" => {
                let fitted = DoubleGamma::try_fit(&data).unwrap();
                (vec![fitted.a], likelihood(&fitted, &data))
            }
            "foldnorm" => {
                let fitted = FoldedNormal::try_fit(&data).unwrap();
                (vec![fitted.c], likelihood(&fitted, &data))
            }
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
            "nakagami" => {
                let fitted = Nakagami::try_fit(&data).unwrap();
                (vec![fitted.nu], likelihood(&fitted, &data))
            }
            "gompertz" => {
                let fitted = fsci_stats::Gompertz::try_fit(&data).unwrap();
                (vec![fitted.c], likelihood(&fitted, &data))
            }
            "genlogistic" => {
                let fitted = fsci_stats::GenLogistic::try_fit(&data).unwrap();
                (vec![fitted.c], likelihood(&fitted, &data))
            }
            "mielke" => {
                let fitted = Mielke::try_fit(&data).unwrap();
                (vec![fitted.k, fitted.s], likelihood(&fitted, &data))
            }
            "exponweib" => {
                let fitted = ExponWeibull::try_fit(&data).unwrap();
                (vec![fitted.a, fitted.c], likelihood(&fitted, &data))
            }
            "rice" => {
                let fitted = Rice::try_fit(&data).unwrap();
                (vec![fitted.b], likelihood(&fitted, &data))
            }
            "norminvgauss" => {
                let fitted = NormInvGauss::try_fit(&data).unwrap();
                (vec![fitted.a, fitted.b], likelihood(&fitted, &data))
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
        if matches!(
            family,
            "gamma" | "beta" | "weibull_min" | "lognorm" | "mielke" | "exponweib" | "gengamma"
        ) {
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
        compared, 99,
        "every seeded case must produce a real comparison"
    );
    eprintln!("MLE compared cases = {compared}; families = 32; samples per case = 2000");
}

#[test]
fn mielke_profile_fit_uses_the_likelihood_and_has_no_shape_grid_cap() {
    let data = [0.2, 0.5, 0.8, 1.1, 1.9, 4.0, 8.0];
    for mode in [RuntimeMode::Strict, RuntimeMode::Hardened] {
        with_stats_runtime_mode(mode, || {
            let fitted = Mielke::try_fit(&data).unwrap();
            assert!((fitted.k - 1.699_724_340_636_757_5).abs() < 2e-6);
            assert!((fitted.s - 1.360_158_431_045_784_4).abs() < 2e-6);
            assert!((likelihood(&fitted, &data) - 12.846_688_114_889_446).abs() < 1e-11);

            // The analytic k-score must vanish; changing either fitted shape
            // independently must increase the actual sample NLL.
            let score_k = data.len() as f64 / fitted.k
                - data
                    .iter()
                    .map(|x| x.powf(-fitted.s).ln_1p() / fitted.s)
                    .sum::<f64>();
            assert!(score_k.abs() < 1e-12);
            let score_s = data
                .iter()
                .map(|x| {
                    let y = x.ln();
                    let t = fitted.s * y;
                    fitted.k / fitted.s.powi(2) * t.exp().ln_1p()
                        - (1.0 + fitted.k / fitted.s) * y / (1.0 + (-t).exp())
                })
                .sum::<f64>();
            assert!(score_s.abs() < 2e-6, "Mielke s-score = {score_s}");
            for factor in [0.9, 1.1] {
                assert!(
                    likelihood(&fitted, &data)
                        < likelihood(&Mielke::new(factor * fitted.k, fitted.s), &data)
                );
                assert!(
                    likelihood(&fitted, &data)
                        < likelihood(&Mielke::new(fitted.k, factor * fitted.s), &data)
                );
            }

            let source = Mielke::new(40.0, 20.0);
            let samples: Vec<_> = (0..2000)
                .map(|i| source.ppf((i as f64 + 0.5) / 2000.0))
                .collect();
            let fitted = Mielke::try_fit(&samples).unwrap();
            assert!(fitted.k > 30.0 && fitted.s > 15.0);
            assert!(likelihood(&fitted, &samples) <= likelihood(&source, &samples));
        });
    }
}

#[test]
fn mielke_profile_fit_rejects_limiting_samples_and_honors_hardened_caps() {
    for data in [vec![], vec![2.0]] {
        assert!(matches!(
            Mielke::try_fit(&data),
            Err(FitError::InsufficientData { .. })
        ));
    }
    for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            Mielke::try_fit(&[0.5, invalid, 2.0]),
            Err(FitError::UnsupportedData(_))
        ));
    }
    for data in [
        vec![0.2, 0.5, 0.8, 1.0],
        vec![2.0, 2.0],
        // A finite optimum exists mathematically, but its k is larger than
        // f64::MAX. The finite representability edge is not an MLE.
        vec![2.0, 2.0001],
        // Here rounding can even make both finite endpoint likelihoods
        // slightly worse than the middle: only the score detects the wall.
        vec![2.0, 2.0002],
        vec![1.1, 1.1001],
        // These lie immediately beyond the finite k boundary; a small
        // scaled score alone cannot certify a representable maximum.
        vec![2.0, 2.004_741_612_574_921_6],
        vec![2.0, 2.004_742_039_362_732],
        vec![2.0, 2.004_742_082_041_513],
        vec![2.0, 2.004_742_086_309_391],
    ] {
        assert!(matches!(
            Mielke::try_fit(&data),
            Err(FitError::NonConvergent(_))
        ));
    }
    with_stats_runtime_mode(RuntimeMode::Hardened, || {
        with_hardened_stats_caps(2, 2, || {
            assert!(matches!(
                Mielke::try_fit(&[0.5, 1.0, 2.0]),
                Err(FitError::ResourceExhausted { .. })
            ));
            assert!(Mielke::try_fit(&[0.5, 2.0]).is_ok());
        });
    });
}

#[test]
fn mielke_log_density_preserves_finite_likelihood_when_powers_overflow() {
    let distribution = Mielke::new(2000.0, 1000.0);
    for x in [0.25, 0.5, 2.0, 4.0] {
        let actual = distribution.logpdf(x);
        assert!(actual.is_finite());
        let lx = x.ln();
        let expected = if x > 1.0 {
            2000.0_f64.ln() - 1001.0 * lx
        } else {
            2000.0_f64.ln() + 1999.0 * lx
        };
        assert!((actual - expected).abs() < 1e-12);
        assert_eq!(distribution.pdf(x), actual.exp());
    }
    assert_eq!(distribution.logpdf(f64::INFINITY), f64::NEG_INFINITY);
    assert_eq!(Mielke::new(0.5, 2.0).logpdf(0.0), f64::INFINITY);
    assert_eq!(Mielke::new(1.0, 2.0).logpdf(0.0), 0.0);
    assert_eq!(distribution.logpdf(0.0), f64::NEG_INFINITY);
    assert!(distribution.logpdf(f64::NAN).is_nan());
    let finite_after_rescaling = Mielke::new(1e308, 0.5).logpdf(1e308);
    assert!(finite_after_rescaling.is_finite());
    assert!((finite_after_rescaling / -2e154 - 1.0).abs() < 1e-12);
    // s*ln(x) overflows, although the limiting density is exactly one.
    assert_eq!(Mielke::new(1.0, 1e308).logpdf(0.01), 0.0);
}

#[test]
fn mielke_fitted_high_shapes_keep_probabilities_and_quantiles_usable() {
    for data in [[2.0, 2.1, 2.2, 2.3], [2.0, 2.02, 2.04, 2.06]] {
        let fitted = Mielke::try_fit(&data).unwrap();
        assert!(fitted.k > 1e6);
        for x in data {
            let cdf = fitted.cdf(x);
            let sf = fitted.sf(x);
            assert!(cdf.is_finite() && cdf > 0.0 && cdf < 1.0);
            assert!(sf.is_finite() && sf > 0.0 && sf < 1.0);
            assert!((cdf + sf - 1.0).abs() < 2e-15);
        }
        for q in [0.1, 0.5, 0.9] {
            let x = fitted.ppf(q);
            assert!(x.is_finite());
            assert!((fitted.cdf(x) - q).abs() < 1e-12);
        }
    }
    // Independent analytical profile-score root for this finite sample.
    let fitted = Mielke::try_fit(&[2.0, 2.02, 2.04, 2.06]).unwrap();
    assert!((fitted.ppf(0.5) - 2.026_066_324_557_367_3).abs() < 2e-6);
    // Just inside the representability boundary, k is about 1.7964e308.
    // The score certificate must not introduce another large-shape cap.
    let near_limit = Mielke::try_fit(&[2.0, 2.004_742_091_525_687]).unwrap();
    assert!(near_limit.k.is_finite() && near_limit.k > 1e308);
    assert!(near_limit.ppf(0.5).is_finite());

    // s/k is huge and -(s/k)*ln(q) overflows; the limiting quantile
    // q^(1/k) is nevertheless finite.
    let limiting = Mielke::new(1.0, 1e308);
    assert!((limiting.ppf(0.01) - 0.01).abs() < 1e-16);
    assert!((limiting.cdf(0.01) - 0.01).abs() < 1e-16);
    assert_eq!(limiting.ppf(0.0), 0.0);
    assert_eq!(limiting.ppf(1.0), f64::INFINITY);
    assert!(limiting.ppf(f64::NAN).is_nan());
}

#[test]
fn exponweibull_profile_fit_handles_a_unit_median_and_maximizes_likelihood() {
    let data = [0.5, 0.8, 1.0, 1.2, 2.0];
    // Independent 65-digit likelihood-score solution at these observations.
    // The old quartile matcher had no candidate when the sample median was 1.
    for mode in [RuntimeMode::Strict, RuntimeMode::Hardened] {
        with_stats_runtime_mode(mode, || {
            let fitted = ExponWeibull::try_fit(&data).unwrap();
            assert!((fitted.a - 1.736_613_448_391_161).abs() < 2e-6);
            assert!((fitted.c - 1.794_628_798_993_685).abs() < 2e-6);
            assert!((likelihood(&fitted, &data) - 3.284_193_592_822_329_4).abs() < 1e-11);
            let score_a = data.len() as f64 / fitted.a
                + data
                    .iter()
                    .map(|x| (-(-x.powf(fitted.c)).exp_m1()).ln())
                    .sum::<f64>();
            let score_c = data.len() as f64 / fitted.c
                + data
                    .iter()
                    .map(|x| {
                        let w = x.powf(fitted.c);
                        x.ln() * ((fitted.a - 1.0) * w / w.exp_m1() + 1.0 - w)
                    })
                    .sum::<f64>();
            assert!(score_a.abs() < 1e-12);
            assert!(score_c.abs() < 2e-6, "ExponWeibull c-score={score_c}");
            for factor in [0.9, 1.1] {
                assert!(
                    likelihood(&fitted, &data)
                        < likelihood(&ExponWeibull::new(factor * fitted.a, fitted.c), &data)
                );
                assert!(
                    likelihood(&fitted, &data)
                        < likelihood(&ExponWeibull::new(fitted.a, factor * fitted.c), &data)
                );
            }
        });
    }
}

#[test]
fn exponweibull_profile_fit_rejects_limiting_samples_and_honors_hardened_caps() {
    for data in [vec![], vec![2.0]] {
        assert!(matches!(
            ExponWeibull::try_fit(&data),
            Err(FitError::InsufficientData { .. })
        ));
    }
    for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            ExponWeibull::try_fit(&[0.5, invalid, 2.0]),
            Err(FitError::UnsupportedData(_))
        ));
    }
    for data in [
        vec![0.2, 0.5, 0.8, 1.0],
        vec![2.0, 2.0],
        // Independent log(a) optima exceed 4008, 2165, and 3125
        // respectively. An apparent minimum at a=f64::MAX is not an MLE.
        vec![2.0, 2.0001],
        vec![2.0, 2.0002],
        vec![1.1, 1.10001],
        // Much closer to the a=f64::MAX boundary: objective roundoff can
        // fake a strict minimum and the scaled score is already tiny.
        vec![2.0, 2.000_713_380_004_696],
        vec![2.0, 2.000_713_444_215_317],
        vec![2.0, 2.000_713_450_636_379_5],
    ] {
        assert!(matches!(
            ExponWeibull::try_fit(&data),
            Err(FitError::NonConvergent(_))
        ));
    }
    with_stats_runtime_mode(RuntimeMode::Hardened, || {
        with_hardened_stats_caps(2, 2, || {
            assert!(matches!(
                ExponWeibull::try_fit(&[0.5, 1.0, 2.0]),
                Err(FitError::ResourceExhausted { .. })
            ));
            assert!(ExponWeibull::try_fit(&[0.5, 2.0]).is_ok());
        });
    });
}

#[test]
fn exponweibull_fitted_high_shapes_preserve_probabilities_and_quantiles() {
    for data in [vec![2.0, 2.01, 2.02, 2.03], vec![2.0, 2.001, 2.002, 2.003]] {
        let fitted = ExponWeibull::try_fit(&data).unwrap();
        assert!(fitted.a > 1e15, "a={} was artificially capped", fitted.a);
        for &x in &data {
            assert!(fitted.logpdf(x).is_finite());
            let cdf = fitted.cdf(x);
            let sf = fitted.sf(x);
            assert!(cdf.is_finite() && cdf > 0.0 && cdf < 1.0);
            assert!(sf.is_finite() && sf > 0.0 && sf < 1.0);
            assert!((cdf + sf - 1.0).abs() < 2e-15);
        }
        for q in [0.1, 0.5, 0.9] {
            let x = fitted.ppf(q);
            assert!(x.is_finite());
            assert!((fitted.cdf(x) - q).abs() < 2e-12);
        }
        for factor in [0.9, 1.1] {
            assert!(
                likelihood(&fitted, &data)
                    < likelihood(&ExponWeibull::new(factor * fitted.a, fitted.c), &data)
            );
        }
    }
    // Independent 90-digit likelihood-score solution. Default SciPy fit
    // stops well short of this concentrated sample's global maximum.
    let data = [2.0, 2.001, 2.002, 2.003];
    let fitted = ExponWeibull::try_fit(&data).unwrap();
    assert!((likelihood(&fitted, &data) - -21.417_963_830_126_58).abs() < 1e-9);
    assert!((fitted.ppf(0.5) - 2.001_310_892_956_218_4).abs() < 2e-7);
    // This close sample has a finite MLE, unlike the tighter sample above.
    let fitted = ExponWeibull::try_fit(&[1.1, 1.1001]).unwrap();
    assert!(fitted.a > 1e170 && fitted.a < 1e190);
    assert!((fitted.c - 63.272_996_726_439_76).abs() < 2e-5);
    assert!(fitted.ppf(0.5).is_finite());
    // The adjacent sample on the finite side of the boundary remains
    // fit-able even though its a is about 1.7966e308.
    let near_limit = ExponWeibull::try_fit(&[2.0, 2.000_713_452_063_282]).unwrap();
    assert!(near_limit.a.is_finite() && near_limit.a > 1e308);
    assert!(near_limit.ppf(0.5).is_finite());
}

#[test]
fn exponweibull_density_and_probability_tails_remain_finite() {
    for (a, c) in [(1.0, 2000.0), (0.5, 2000.0)] {
        let distribution = ExponWeibull::new(a, c);
        let expected = (a * c).ln() + (a * c - 1.0) * 0.5_f64.ln();
        assert!((distribution.logpdf(0.5) - expected).abs() < 1e-12);
        assert_eq!(distribution.pdf(0.5), distribution.logpdf(0.5).exp());
        assert_eq!(distribution.logpdf(f64::INFINITY), f64::NEG_INFINITY);
        assert!(distribution.logpdf(f64::NAN).is_nan());
    }
    let left = ExponWeibull::new(0.01, 1.0);
    assert!((left.cdf(1e-100) - 0.1).abs() < 1e-15);
    assert!((left.sf(1e-100) - 0.9).abs() < 1e-15);
    assert!((left.ppf(0.1).ln() - 1e-100_f64.ln()).abs() < 1e-12);
    // w is reconstructed from exp(c*ln(x)); allow its binary64 rounding.
    assert!(
        (ExponWeibull::new(1.0, 1.0).logsf(1000.0) + 1000.0).abs() < 4.0 * f64::EPSILON * 1000.0
    );
    let right = ExponWeibull::new(1e308, 1.0);
    assert!((right.logpdf(1000.0) - (1e308_f64.ln() - 1000.0)).abs() < 1e-11);
    assert!((right.logsf(1000.0) - (1e308_f64.ln() - 1000.0)).abs() < 1e-11);

    // c*ln(x), and t=-ln(q)/a in the inverse, overflow separately.
    // Their finite power-law limits must survive the parameter rescaling.
    let limiting = ExponWeibull::new(1e-308, 1e308);
    assert!(limiting.logpdf(0.01).abs() < 1e-12);
    assert!((limiting.cdf(0.01) - 0.01).abs() < 1e-15);
    assert!((limiting.ppf(0.01) - 0.01).abs() < 1e-15);
    assert_eq!(limiting.ppf(0.0), 0.0);
    assert_eq!(limiting.ppf(1.0), f64::INFINITY);
    assert!(limiting.ppf(f64::NAN).is_nan());
    assert_eq!(ExponWeibull::new(0.5, 0.5).logpdf(0.0), f64::INFINITY);
    assert_eq!(ExponWeibull::new(1.0, 1.0).logpdf(0.0), 0.0);
    assert_eq!(ExponWeibull::new(2.0, 2.0).logpdf(0.0), f64::NEG_INFINITY);
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
    assert_eq!(
        data, original,
        "fitting must not rewrite censoring metadata"
    );
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
        let fitted = fit_mle_censored(&data, &parameters, factory, MleOptions::default()).unwrap();
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
        CensoredData::new(vec![], vec![], vec![], vec![(f64::INFINITY, f64::INFINITY)]),
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
            let data = CensoredData::new(vec![0.0], vec![-1.0], vec![1.0], vec![(-0.2, 0.2)]);
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
        assert!(
            !required,
            "required SciPy oracle is unavailable via {python}"
        );
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
                .as_chunks::<2>()
                .0
                .iter()
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

const LOCATION_SCALE_ORACLE: &str = r#"
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

def logmass(dist, lo, hi):
    lc, ls = dist.logcdf(hi), dist.logsf(lo)
    a, b = (lc, dist.logcdf(lo)) if lc < ls else (ls, dist.logsf(hi))
    return a + np.log(-np.expm1(b - a))

# Masks are in [shapes..., loc, scale] order. Starts are generated from the
# declared model, never from the fitted reference parameters.
specs = [
    ('gamma', stats.gamma, (3.5,), -2.0, 1.7, '000'),
    ('gamma', stats.gamma, (5.0,), 3.0, 0.8, '101'),
    ('beta', stats.beta, (2.5, 4.0), -3.0, 8.0, '0011'),
    ('beta', stats.beta, (4.0, 2.0), 2.0, 3.0, '1010'),
    ('lognorm', stats.lognorm, (0.45,), -4.0, 2.5, '010'),
    ('lognorm', stats.lognorm, (0.8,), 1.5, 0.7, '100'),
    ('johnsonsu', stats.johnsonsu, (-0.8, 0.9), 2.0, 2.7, '0000'),
    ('johnsonsu', stats.johnsonsu, (1.2, 2.0), -3.0, 1.1, '0101'),
]
for index, (family, dist, shapes, loc, scale, mask) in enumerate(specs):
    values = dist.rvs(*shapes, loc=loc, scale=scale, size=800,
                      random_state=np.random.default_rng(61020 + index))
    truth = (*shapes, loc, scale)
    initial = [value + 0.15 if family == 'johnsonsu' and i == 0 else 0.9*value
               for i, value in enumerate(shapes)] + [loc - 0.15*scale, 1.1*scale]
    for i, fixed in enumerate(mask):
        if fixed == '1':
            initial[i] = truth[i]
    lower, upper = dist.ppf([0.15, 0.8], *shapes, loc=loc, scale=scale)
    width = 0.15*scale
    exact, left, right, intervals = [], [], [], []
    for i, x in enumerate(values):
        if x <= lower:
            left.append(lower)
        elif x >= upper:
            right.append(upper)
        elif i % 4 == 0:
            lo = loc + width*np.floor((x-loc)/width)
            intervals.append((lo, lo+width))
        else:
            exact.append(x)
    mixed = stats.CensoredData(uncensored=exact, left=left, right=right,
                               interval=intervals)
    for kind, data, parts in [
        ('exact', values, (values, [], [], [])),
        ('censored', mixed, (exact, left, right, intervals)),
    ]:
        kwargs = {'optimizer': optimizer, 'loc': initial[-2], 'scale': initial[-1]}
        for i, fixed in enumerate(mask):
            if fixed == '1':
                key = ('f'+str(i) if i < len(shapes) else
                       'floc' if i == len(shapes) else 'fscale')
                kwargs[key] = truth[i]
        fitted = dist.fit(data, *initial[:-2], **kwargs)
        frozen = dist(*fitted[:-2], loc=fitted[-2], scale=fitted[-1])
        u, l, r, intervals_used = parts
        nll = -(sum(frozen.logpdf(u)) + sum(frozen.logcdf(l))
                + sum(frozen.logsf(r))
                + sum(logmass(frozen, lo, hi) for lo, hi in intervals_used))
        assert np.isfinite(nll) and np.all(np.isfinite(fitted))
        print(family, index, kind, mask, len(fitted),
              *(format(x, '.17g') for x in fitted), format(nll, '.17g'))
        for label, numbers in [('start', initial), ('u', u), ('l', l), ('r', r),
                               ('i', [x for pair in intervals_used for x in pair])]:
            print(label, *(format(x, '.17g') for x in numbers))
"#;

fn fit_location_scale_oracle<D, F>(
    data: &CensoredData,
    parameters: &[MleParameter],
    censored: bool,
    factory: F,
) -> MleResult<LocationScale<D>>
where
    D: ContinuousDistribution,
    F: Fn(&[f64]) -> Option<D>,
{
    let n_shapes = parameters.len() - 2;
    let shapes = &parameters[..n_shapes];
    let location = parameters[n_shapes];
    let scale = parameters[n_shapes + 1];
    let options = MleOptions::default();
    if censored {
        fit_mle_location_scale_censored(data, shapes, location, scale, factory, options).unwrap()
    } else {
        fit_mle_location_scale(&data.uncensored, shapes, location, scale, factory, options).unwrap()
    }
}

#[test]
fn location_scale_mle_matches_live_scipy() {
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
        eprintln!("SKIP location/scale MLE: SciPy unavailable via {python}; compared cases = 0");
        return;
    }
    let output = Command::new(&python)
        .args(["-c", LOCATION_SCALE_ORACLE])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "SciPy location/scale MLE oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let mut lines = stdout.lines();
    let versions = lines
        .next()
        .expect("location/scale oracle version metadata");
    assert!(versions.starts_with("versions "));
    eprintln!("Location/scale MLE oracle {versions}");
    if required {
        assert_eq!(versions, "versions 1.17.1 2.4.3");
    }
    let mut compared = 0;
    while let Some(header) = lines.next() {
        let fields: Vec<_> = header.split_whitespace().collect();
        let (family, case, kind, mask) = (fields[0], fields[1], fields[2], fields[3]);
        let n_parameters: usize = fields[4].parse().unwrap();
        assert_eq!(fields.len(), n_parameters + 6);
        assert_eq!(mask.len(), n_parameters);
        let expected: Vec<f64> = fields[5..5 + n_parameters]
            .iter()
            .map(|value| value.parse().unwrap())
            .collect();
        let expected_nll: f64 = fields[5 + n_parameters].parse().unwrap();
        let initial = censor_values(lines.next().unwrap(), "start");
        assert_eq!(initial.len(), n_parameters);
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
                .as_chunks::<2>()
                .0
                .iter()
                .map(|p| (p[0], p[1]))
                .collect(),
        );
        assert_eq!(data.len(), 800);
        let parameters: Vec<_> = initial
            .iter()
            .zip(mask.bytes())
            .enumerate()
            .map(|(i, (&value, fixed))| {
                if fixed == b'1' {
                    MleParameter::Fixed(value)
                } else if i == n_parameters - 2 || (family == "johnsonsu" && i == 0) {
                    MleParameter::Free(value)
                } else {
                    MleParameter::Positive(value)
                }
            })
            .collect();
        let censored = match kind {
            "exact" => false,
            "censored" => true,
            _ => panic!("unexpected observation kind: {kind}"),
        };
        let (actual, nll) = match family {
            "gamma" => {
                let fit = fit_location_scale_oracle(&data, &parameters, censored, |p| {
                    Some(GammaDist {
                        a: p[0],
                        scale: 1.0,
                    })
                });
                (fit.parameters, fit.negative_log_likelihood)
            }
            "beta" => {
                let fit = fit_location_scale_oracle(&data, &parameters, censored, |p| {
                    Some(BetaDist { a: p[0], b: p[1] })
                });
                (fit.parameters, fit.negative_log_likelihood)
            }
            "lognorm" => {
                let fit = fit_location_scale_oracle(&data, &parameters, censored, |p| {
                    Some(Lognormal {
                        s: p[0],
                        scale: 1.0,
                    })
                });
                (fit.parameters, fit.negative_log_likelihood)
            }
            "johnsonsu" => {
                let fit = fit_location_scale_oracle(&data, &parameters, censored, |p| {
                    Some(JohnsonSU { a: p[0], b: p[1] })
                });
                (fit.parameters, fit.negative_log_likelihood)
            }
            _ => panic!("unexpected location/scale oracle family {family}"),
        };
        assert!(nll.is_finite() && expected_nll.is_finite());
        assert_eq!(actual.len(), expected.len());
        assert!(
            (nll - expected_nll).abs() <= 1.0e-7 * expected_nll.abs().max(1.0),
            "{family}/{case}/{kind}: fsci NLL {nll} != SciPy {expected_nll}"
        );
        for (i, (&value, &reference)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (value - reference).abs() <= 1.0e-4 * reference.abs().max(1.0),
                "{family}/{case}/{kind}: parameter {i}: {value} != {reference}"
            );
            if let MleParameter::Fixed(fixed) = parameters[i] {
                assert_eq!(value.to_bits(), fixed.to_bits());
                assert_eq!(value.to_bits(), reference.to_bits());
            }
        }
        eprintln!(
            "{family}/{case}/{kind}: location/scale MLE fsci={actual:?} scipy={expected:?}; \
             NLL fsci={nll:.12e} scipy={expected_nll:.12e}"
        );
        compared += 1;
    }
    assert_eq!(
        compared, 16,
        "every location/scale oracle case must be compared"
    );
    eprintln!(
        "Location/scale MLE compared cases = {compared}; families = 4; samples per case = 800"
    );
}

#[test]
fn closed_form_shape_fits_maximize_the_actual_likelihood() {
    use fsci_stats::{GenLogistic, Gompertz};

    // expm1(log(2)), expm1(log(3)), expm1(log(4)) are 1, 2, 3:
    // n/sum(expm1(x)) = 1/2, independent of any sample moment.
    let gompertz_data = [2.0_f64.ln(), 3.0_f64.ln(), 4.0_f64.ln()];
    let gompertz = Gompertz::try_fit(&gompertz_data).unwrap();
    assert!((gompertz.c - 0.5).abs() < 2.0e-15);
    assert!((likelihood(&gompertz, &gompertz_data) - 1.901_387_711_331_890_4).abs() < 1.0e-13);

    // The likelihood statistic is log(4) + log(2) + log(4/3).
    // A symmetric sample has mean zero, but its shape MLE is not one.
    let logistic_data = [-3.0_f64.ln(), 0.0, 3.0_f64.ln()];
    let logistic = GenLogistic::try_fit(&logistic_data).unwrap();
    let expected = 3.0 / (32.0_f64 / 3.0).ln();
    assert!((logistic.c / expected - 1.0).abs() < 2.0e-15);
    assert!((likelihood(&logistic, &logistic_data) - 4.656_313_407_535_147).abs() < 1.0e-13);
    assert!(
        likelihood(&logistic, &logistic_data) < likelihood(&GenLogistic::new(1.0), &logistic_data)
    );

    for factor in [0.5, 2.0] {
        assert!(
            likelihood(&gompertz, &gompertz_data)
                < likelihood(&Gompertz::new(factor * gompertz.c), &gompertz_data)
        );
        assert!(
            likelihood(&logistic, &logistic_data)
                < likelihood(&GenLogistic::new(factor * logistic.c), &logistic_data)
        );
    }
}

#[test]
fn closed_form_shape_fits_preserve_extreme_finite_parameters() {
    use fsci_stats::{GenLogistic, Gompertz};

    for mode in [RuntimeMode::Strict, RuntimeMode::Hardened] {
        with_stats_runtime_mode(mode, || {
            let narrow = Gompertz::try_fit(&[1.0e-300, 2.0e-300]).unwrap();
            assert!((narrow.c / (2.0e300 / 3.0) - 1.0).abs() < 3.0e-15);
            assert!(narrow.logpdf(1.0e-300).is_finite());
            assert!((narrow.mean() * narrow.c - 1.0).abs() < 3.0e-15);
            let expected_cdf = -(-1.0_f64).exp_m1();
            assert!((narrow.cdf(narrow.c.recip()) - expected_cdf).abs() < 3.0e-15);

            // exp(710) overflows, but exp(-710) is a positive f64.
            let wide = Gompertz::try_fit(&[710.0, 710.0]).unwrap();
            assert!((wide.c / (-710.0_f64).exp() - 1.0).abs() < 3.0e-13);
            assert!((wide.logpdf(710.0) + 1.0).abs() < 3.0e-13);
            assert!((wide.cdf(710.0) - expected_cdf).abs() < 3.0e-13);
            assert!((wide.logsf(710.0) + 1.0).abs() < 3.0e-13);
            assert!((wide.ppf(expected_cdf) - 710.0).abs() < 3.0e-13);

            let left = GenLogistic::try_fit(&[-1.0e308, -1.0e308]).unwrap();
            assert!((left.c / 1.0e-308 - 1.0).abs() < 3.0e-15);
            assert!((left.logpdf(-1.0e308) + 1.0 + 1.0e308_f64.ln()).abs() < 3.0e-13);
            assert!((left.cdf(-1.0e308) - (-1.0_f64).exp()).abs() < 3.0e-15);
            assert!(left.pdf(-1.0e308) > 0.0);
            assert!((left.ppf((-1.0_f64).exp()) / -1.0e308 - 1.0).abs() < 3.0e-15);

            let right = GenLogistic::try_fit(&[700.0, 700.0]).unwrap();
            assert!((right.c / 700.0_f64.exp() - 1.0).abs() < 3.0e-15);
            assert!((right.logpdf(700.0) + 1.0).abs() < 3.0e-13);
        });
    }
}

#[test]
fn closed_form_shape_fits_reject_undefined_limits_and_honor_caps() {
    use fsci_stats::{GenLogistic, Gompertz};

    for data in [vec![], vec![1.0]] {
        assert!(matches!(
            Gompertz::try_fit(&data),
            Err(FitError::InsufficientData { .. })
        ));
        assert!(matches!(
            GenLogistic::try_fit(&data),
            Err(FitError::InsufficientData { .. })
        ));
    }
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            Gompertz::try_fit(&[1.0, invalid]),
            Err(FitError::UnsupportedData(_))
        ));
        assert!(matches!(
            GenLogistic::try_fit(&[1.0, invalid]),
            Err(FitError::UnsupportedData(_))
        ));
    }
    assert!(matches!(
        Gompertz::try_fit(&[-0.1, 1.0]),
        Err(FitError::UnsupportedData(_))
    ));
    for data in [[0.0, 0.0], [1000.0, 1000.0]] {
        assert!(matches!(
            Gompertz::try_fit(&data),
            Err(FitError::NonConvergent(_))
        ));
    }
    assert!(matches!(
        GenLogistic::try_fit(&[1000.0, 1000.0]),
        Err(FitError::NonConvergent(_))
    ));
    with_stats_runtime_mode(RuntimeMode::Hardened, || {
        with_hardened_stats_caps(2, 2, || {
            assert!(matches!(
                Gompertz::try_fit(&[0.1, 0.2, 0.3]),
                Err(FitError::ResourceExhausted { .. })
            ));
            assert!(matches!(
                GenLogistic::try_fit(&[0.1, 0.2, 0.3]),
                Err(FitError::ResourceExhausted { .. })
            ));
            assert!(Gompertz::try_fit(&[0.1, 0.2]).is_ok());
            assert!(GenLogistic::try_fit(&[0.1, 0.2]).is_ok());
        });
    });
}

fn generalized_normal_likelihood_score(data: &[f64], beta: f64) -> f64 {
    // Differentiate the published density, evaluating powers directly rather
    // than using the fitter's log-space likelihood or optimizer coordinates.
    let weighted_log: f64 = data
        .iter()
        .filter(|&&x| x != 0.0)
        .map(|x| x.abs().powf(beta) * x.abs().ln())
        .sum();
    fsci_special::digamma_scalar(1.0 + beta.recip())
        - beta * beta * weighted_log / data.len() as f64
}

#[test]
fn generalized_normal_mle_uses_likelihood_for_both_families() {
    let data = [-0.5_f64, 0.0, 0.5, -1.0, 1.0, 0.3, -0.7, 0.8, -0.4];
    let folded = data.map(f64::abs);
    for mode in [RuntimeMode::Strict, RuntimeMode::Hardened] {
        with_stats_runtime_mode(mode, || {
            let fitted = GenNorm::try_fit(&data).unwrap();
            let half = HalfGenNorm::try_fit(&folded).unwrap();
            assert_eq!(fitted.beta.to_bits(), half.beta.to_bits());
            assert_eq!(fitted, GenNorm::fit(&data));
            assert_eq!(half, HalfGenNorm::fit(&folded));
            assert!(generalized_normal_likelihood_score(&data, fitted.beta).abs() < 1.0e-10);
            // The former estimators matched the second moment and folded
            // mean, respectively; both have appreciably worse likelihood.
            let old_full = GenNorm::new(2.362_139_243_050_265);
            let old_half = HalfGenNorm::new(1.891_807_245_086_320_1);
            assert!(likelihood(&old_full, &data) - likelihood(&fitted, &data) > 0.7);
            assert!(likelihood(&old_half, &folded) - likelihood(&half, &folded) > 1.0);
            for factor in [0.9, 1.1] {
                assert!(
                    likelihood(&GenNorm::new(factor * fitted.beta), &data)
                        > likelihood(&fitted, &data)
                );
            }
        });
    }
}

#[test]
fn generalized_normal_mle_handles_shape_extremes_and_zero_samples() {
    for mode in [RuntimeMode::Strict, RuntimeMode::Hardened] {
        with_stats_runtime_mode(mode, || {
            // At zero, the shape remains identifiable with scale fixed.
            // The score is digamma(1+1/beta)=0, with beta between 2 and 2.3.
            for data in [vec![0.0, 0.0], vec![0.0, 0.0, 1.0, 1.0]] {
                let fitted = GenNorm::try_fit(&data).unwrap();
                let half = HalfGenNorm::try_fit(&data).unwrap();
                assert!(fitted.beta > 2.0 && fitted.beta < 2.3);
                assert_eq!(fitted.beta.to_bits(), half.beta.to_bits());
                assert!(generalized_normal_likelihood_score(&data, fitted.beta).abs() < 1.0e-12);
            }

            // These data have a poor local optimum near beta=2.18 and a
            // much better one near 19026. A single ordinary starting value
            // does not suffice, and the old beta<=50 bracket excludes it.
            let near_unit = [0.999, 0.9991, 0.9992, 0.9993, 0.9994];
            let fitted = GenNorm::try_fit(&near_unit).unwrap();
            assert!(fitted.beta > 18_000.0 && fitted.beta < 20_000.0);
            assert!(generalized_normal_likelihood_score(&near_unit, fitted.beta).abs() < 1.0e-9);
            assert!(
                likelihood(&fitted, &near_unit) < near_unit.len() as f64 * std::f64::consts::LN_2
            );
            let half = HalfGenNorm::try_fit(&near_unit).unwrap();
            assert_eq!(fitted.beta.to_bits(), half.beta.to_bits());
            assert!(likelihood(&half, &near_unit) < 0.0);

            // Large finite data need shapes below the old beta>=0.1 cutoff.
            // The still larger sample below overflows empirical moments.
            let large = [1.0e20, 1.0e22, 1.0e24, 1.0e26];
            let fitted = GenNorm::try_fit(&large).unwrap();
            assert!(fitted.beta > 0.05 && fitted.beta < 0.06);
            assert!(likelihood(&fitted, &large).is_finite());
            assert!(generalized_normal_likelihood_score(&large, fitted.beta).abs() < 1.0e-9);
            let extreme = [1.0e300, 1.0e308];
            let fitted = GenNorm::try_fit(&extreme).unwrap();
            assert!(fitted.beta > 0.0 && fitted.beta < 0.01);
            assert!(likelihood(&fitted, &extreme).is_finite());
            assert!(generalized_normal_likelihood_score(&extreme, fitted.beta).abs() < 1.0e-8);

            // Even 1+1/beta rounds to 1 here. The likelihood's finite-shape
            // advantage survives only when its log-Gamma term is retained.
            let adjacent = [
                f64::from_bits(1.0_f64.to_bits() - 1),
                f64::from_bits(1.0_f64.to_bits() - 2),
            ];
            let fitted = GenNorm::try_fit(&adjacent).unwrap();
            assert!(fitted.beta > 1.0e17 && fitted.beta < 1.0e18);
            assert!(generalized_normal_likelihood_score(&adjacent, fitted.beta).abs() < 1.0e-8);
        });
    }
}

#[test]
fn generalized_normal_mle_rejects_invalid_data_and_honors_caps() {
    for data in [vec![], vec![1.0]] {
        assert!(matches!(
            GenNorm::try_fit(&data),
            Err(FitError::InsufficientData { .. })
        ));
        assert!(matches!(
            HalfGenNorm::try_fit(&data),
            Err(FitError::InsufficientData { .. })
        ));
    }
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            GenNorm::try_fit(&[0.5, invalid]),
            Err(FitError::UnsupportedData(_))
        ));
        assert!(matches!(
            HalfGenNorm::try_fit(&[0.5, invalid]),
            Err(FitError::UnsupportedData(_))
        ));
    }
    assert!(matches!(
        HalfGenNorm::try_fit(&[0.5, -0.1]),
        Err(FitError::UnsupportedData(_))
    ));
    assert!(GenNorm::try_fit(&[0.5, -0.1]).is_ok());
    with_stats_runtime_mode(RuntimeMode::Hardened, || {
        with_hardened_stats_caps(2, 2, || {
            for data in [[0.1, 0.5, 1.0], [0.0, 0.0, 0.0]] {
                assert!(matches!(
                    GenNorm::try_fit(&data),
                    Err(FitError::ResourceExhausted { .. })
                ));
                assert!(matches!(
                    HalfGenNorm::try_fit(&data),
                    Err(FitError::ResourceExhausted { .. })
                ));
            }
            assert!(GenNorm::try_fit(&[0.0, 0.0]).is_ok());
            assert!(HalfGenNorm::try_fit(&[0.0, 0.0]).is_ok());
        });
    });
}

#[test]
fn normal_fit_range_regressions_preserve_public_workflows() {
    for mode in [RuntimeMode::Strict, RuntimeMode::Hardened] {
        with_stats_runtime_mode(mode, || {
            for magnitude in [1.0e200, 1.0e-200, 1.0e-310] {
                let data = [magnitude, 2.0 * magnitude];
                let fitted = Normal::try_fit(&data).unwrap();
                assert!((fitted.loc / magnitude - 1.5).abs() < 1.0e-12);
                assert!((fitted.scale / magnitude - 0.5).abs() < 1.0e-12);
                assert_eq!(Normal::fit(&data), fitted);
                assert_eq!(fitted.std(), fitted.scale);
                assert!(data.iter().all(|&x| fitted.logpdf(x).is_finite()));

                let gof = Normal::try_fit_for_goodness_of_fit(&data).unwrap();
                assert_eq!(gof.loc, fitted.loc);
                assert!((gof.scale / magnitude - std::f64::consts::FRAC_1_SQRT_2).abs() < 1.0e-12);
            }
            let data = [f64::MAX, f64::MAX, -f64::MAX];
            let fitted = Normal::try_fit(&data).unwrap();
            assert!((fitted.loc / f64::MAX - 1.0 / 3.0).abs() < 1.0e-15);
            assert!((fitted.scale / f64::MAX - (8.0_f64 / 9.0).sqrt()).abs() < 1.0e-15);
            assert!(data.iter().all(|&x| fitted.logpdf(x).is_finite()));
        });
    }
}

#[test]
fn normal_fit_matches_live_scipy_across_representable_scales() {
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
        eprintln!("SKIP Normal range MLE: SciPy oracle unavailable; compared cases = 0");
        return;
    }
    let script = r#"
import os
import numpy as np
import scipy
from scipy import stats
print('versions', scipy.__version__, np.__version__)
if os.environ.get('FSCI_REQUIRE_SCIPY_ORACLE') == '1':
    assert (scipy.__version__, np.__version__) == ('1.17.1', '2.4.3')
samples = [[-2., -1., 0., 1., 4.], [1., 2., 3., 5., 8.],
           [-8., -5., -3., -2., -1.], [0., 1., 1., 2., 3., 10.]]
for sample in samples:
    for factor in [1., 1e200, 1e-200, 1e-310, 1e307]:
        data = np.array(sample) * factor
        # Normalize the actual rounded observations. Direct NumPy variance
        # also over/underflows here, so use normal-model affine invariance as
        # the oracle instead of accepting matching infinities or zeros.
        normalized = data / factor
        loc, scale = stats.norm.fit(normalized)
        sample_scale = np.std(normalized, ddof=1)
        nll = -np.sum(stats.norm.logpdf(normalized, loc=loc, scale=scale))
        nll += len(data) * np.log(factor)
        print(*(format(x, '.17g') for x in [factor, loc, scale, sample_scale, nll]))
        print(' '.join(format(x, '.17g') for x in data))
"#;
    let output = Command::new(&python).args(["-c", script]).output().unwrap();
    assert!(
        output.status.success(),
        "SciPy Normal range MLE oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let mut lines = stdout.lines();
    let versions = lines.next().expect("oracle version metadata");
    eprintln!("Normal range MLE oracle {versions}");
    if required {
        assert_eq!(versions, "versions 1.17.1 2.4.3");
    }
    let mut compared = 0;
    while let Some(header) = lines.next() {
        let expected: Vec<f64> = header
            .split_whitespace()
            .map(|field| field.parse().unwrap())
            .collect();
        assert_eq!(expected.len(), 5);
        let data: Vec<f64> = lines
            .next()
            .expect("oracle observations")
            .split_whitespace()
            .map(|field| field.parse().unwrap())
            .collect();
        let factor = expected[0];
        for mode in [RuntimeMode::Strict, RuntimeMode::Hardened] {
            with_stats_runtime_mode(mode, || {
                let fitted = Normal::try_fit(&data).unwrap();
                let gof = Normal::try_fit_for_goodness_of_fit(&data).unwrap();
                assert!(
                    (fitted.loc / factor - expected[1]).abs() < 1.0e-12,
                    "normal location: factor={factor:e}, mode={mode:?}, fitted={fitted:?}"
                );
                assert!(
                    (fitted.scale / factor - expected[2]).abs() < 1.0e-12,
                    "normal MLE scale: factor={factor:e}, mode={mode:?}, fitted={fitted:?}"
                );
                assert!(
                    (gof.scale / factor - expected[3]).abs() < 1.0e-12,
                    "normal GoF scale: factor={factor:e}, mode={mode:?}, fitted={gof:?}"
                );
                let nll = likelihood(&fitted, &data);
                assert!(nll.is_finite());
                assert!(
                    (nll - expected[4]).abs() < 2.0e-9,
                    "normal likelihood: factor={factor:e}, actual={nll}, scipy={}",
                    expected[4]
                );
            });
            compared += 1;
        }
    }
    assert_eq!(compared, 40, "normal range MLE cases must all be compared");
    eprintln!("Normal range MLE compared cases = {compared}");
}
