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
    CrystalBall, FDistribution, FitError, FoldedCauchy, GammaDist, JohnsonSU, Kappa3,
    LocationScale, Lognormal, Mielke, MleOptions, MleParameter, MleResult, Nakagami, Normal, RDist,
    SkewCauchy, StatsError, TruncNormal, TruncPareto, TruncWeibullMin, Weibull, fit_mle,
    fit_mle_censored, fit_mle_location_scale, fit_mle_location_scale_censored,
    with_hardened_stats_caps, with_stats_runtime_mode,
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
            3.749_999_379_447_294_5e19,
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
            "gamma" | "beta" | "weibull_min" | "lognorm" | "mielke"
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
        compared, 60,
        "every seeded case must produce a real comparison"
    );
    eprintln!("MLE compared cases = {compared}; families = 20; samples per case = 2000");
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
            intervals.chunks_exact(2).map(|p| (p[0], p[1])).collect(),
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
