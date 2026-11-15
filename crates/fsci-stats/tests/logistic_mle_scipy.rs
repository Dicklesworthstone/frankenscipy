//! Logistic likelihood regression and live SciPy comparisons (frankenscipy-1ksfv.14).
//!
//! Set `FSCI_REQUIRE_SCIPY_ORACLE=1` and `SCIPY_PYTHON` to require the pinned
//! SciPy 1.17.1 / NumPy 2.4.3 oracle. Pure Rust regressions always run.

use std::process::Command;

use fsci_runtime::RuntimeMode;
use fsci_stats::{
    ContinuousDistribution, FitError, Logistic, with_hardened_stats_caps, with_stats_runtime_mode,
};

fn nll(distribution: Logistic, data: &[f64]) -> f64 {
    -data.iter().map(|&x| distribution.logpdf(x)).sum::<f64>()
}

#[test]
fn logistic_mle_improves_likelihood_and_solves_score() {
    let data = [-3.0, -2.0, -1.0, 0.0, 0.25, 0.5, 1.0, 3.0, 20.0];
    let fitted = Logistic::try_fit(&data).expect("logistic MLE");
    // scipy.stats.logistic.fit(data), SciPy 1.17.1. The old moment estimator
    // had NLL 28.792063073921028 and did not solve either likelihood score.
    assert!((fitted.loc - 0.581_070_123_760_128_7).abs() < 1.0e-8);
    assert!((fitted.scale - 2.690_419_497_422_026).abs() < 1.0e-8);
    assert!((nll(fitted, &data) - 28.154_765_680_762_882).abs() < 1.0e-11);
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    let moments = Logistic::new(mean, (3.0 * variance).sqrt() / std::f64::consts::PI);
    assert!(nll(moments, &data) - nll(fitted, &data) > 0.6);
    let location_score: f64 = data
        .iter()
        .map(|&x| ((x - fitted.loc) / (2.0 * fitted.scale)).tanh())
        .sum();
    let scale_score: f64 = data
        .iter()
        .map(|&x| {
            let z = (x - fitted.loc) / fitted.scale;
            z * (0.5 * z).tanh() - 1.0
        })
        .sum();
    assert!(location_score.abs() < 1.0e-9, "{location_score}");
    assert!(scale_score.abs() < 1.0e-9, "{scale_score}");
    assert_eq!(Logistic::fit(&data), fitted);
}

#[test]
fn logistic_mle_is_affine_equivariant_at_extreme_scales() {
    let sample = [-3.0, -2.0, -1.0, 0.0, 0.25, 0.5, 1.0, 3.0, 20.0];
    let reference = Logistic::try_fit(&sample).unwrap();
    for scale in [1.0e-200, 1.0, 1.0e200] {
        let offset = 7.0 * scale;
        let data: Vec<f64> = sample.iter().map(|&x| x * scale + offset).collect();
        let fitted = Logistic::try_fit(&data).unwrap();
        assert!(((fitted.loc - offset) / scale - reference.loc).abs() < 1.0e-10);
        assert!((fitted.scale / scale - reference.scale).abs() < 1.0e-10);
        assert!(nll(fitted, &data).is_finite());
    }
    let data = [-f64::MAX, 0.0, f64::MAX];
    let fitted = Logistic::try_fit(&data).unwrap();
    assert!(fitted.loc.abs() < 1.0e-10 * f64::MAX);
    assert!(fitted.scale.is_finite() && fitted.scale > 0.0);
    assert!(nll(fitted, &data).is_finite());
}

#[test]
fn logistic_mle_resolves_a_concentrated_sample_with_an_outlier() {
    // The inverse-scale optimum is far from the moment initialization. It
    // remains finite because the outlier prevents scale from collapsing.
    let mut data = vec![0.0; 2000];
    data[1999] = 1.0;
    let fitted = Logistic::try_fit(&data).unwrap();
    assert!((fitted.loc - 5.002_501_667_9e-7).abs() < 1.0e-14);
    assert!((fitted.scale - 5.0e-4).abs() < 1.0e-13);
    assert!(nll(fitted, &data).is_finite());
}

#[test]
fn logistic_density_handles_both_tails_without_cancellation() {
    let distribution = Logistic::new(0.0, 1.0);
    for x in [100.0, 400.0, 1000.0, f64::MAX, f64::INFINITY] {
        assert_eq!(distribution.pdf(-x), distribution.pdf(x));
        assert_eq!(distribution.logpdf(-x), distribution.logpdf(x));
        assert!(!distribution.pdf(-x).is_nan());
        assert!(!distribution.logpdf(-x).is_nan());
    }
    assert_eq!(distribution.logpdf(-f64::MAX), -f64::MAX);
    assert_eq!(distribution.logpdf(f64::NEG_INFINITY), f64::NEG_INFINITY);
    let tiny = Logistic::new(0.0, 1.0e-300);
    for z in [-740.0, -1000.0] {
        let x = z * tiny.scale;
        assert!(tiny.pdf(x) > 0.0);
        assert!((tiny.pdf(x).ln() - tiny.logpdf(x)).abs() < 1.0e-12);
    }
    let wide = Logistic::new(-f64::MAX, f64::MAX);
    assert!((wide.logpdf(f64::MAX) + f64::MAX.ln() - distribution.logpdf(2.0)).abs() < 1.0e-12);
    assert!(distribution.pdf(f64::NAN).is_nan());
    assert!(distribution.logpdf(f64::NAN).is_nan());
}

#[test]
fn logistic_mle_rejects_degenerate_samples_and_honors_caps() {
    for data in [vec![], vec![1.0]] {
        assert!(matches!(
            Logistic::try_fit(&data),
            Err(FitError::InsufficientData { .. })
        ));
    }
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            Logistic::try_fit(&[0.0, invalid]),
            Err(FitError::UnsupportedData(_))
        ));
    }
    for data in [[1.0; 3], [f64::MAX; 3]] {
        assert!(matches!(
            Logistic::try_fit(&data),
            Err(FitError::NonConvergent(_))
        ));
        assert!(Logistic::fit(&data).scale.is_nan());
    }
    with_stats_runtime_mode(RuntimeMode::Hardened, || {
        with_hardened_stats_caps(2, 2, || {
            assert!(matches!(
                Logistic::try_fit(&[0.0, 1.0, 2.0]),
                Err(FitError::ResourceExhausted { .. })
            ));
            assert!(Logistic::try_fit(&[0.0, 1.0]).is_ok());
        });
    });
}

#[test]
fn logistic_mle_matches_live_scipy() {
    let python = std::env::var("SCIPY_PYTHON")
        .or_else(|_| std::env::var("FSCI_PYTHON"))
        .unwrap_or_else(|_| "python3".into());
    let required = std::env::var("FSCI_REQUIRE_SCIPY_ORACLE").as_deref() == Ok("1");
    let available = Command::new(&python)
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("OMP_NUM_THREADS", "1")
        .args(["-c", "import scipy, numpy"])
        .status()
        .is_ok_and(|status| status.success());
    if !available {
        assert!(!required, "required SciPy oracle is unavailable");
        eprintln!("SKIP Logistic MLE oracle: SciPy unavailable");
        return;
    }
    let output = Command::new(&python)
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("OMP_NUM_THREADS", "1")
        .args([
            "-c",
            r#"
import os
import numpy as np
import scipy
from scipy import stats
print('versions', scipy.__version__, np.__version__)
if os.environ.get('FSCI_REQUIRE_SCIPY_ORACLE') == '1':
    assert (scipy.__version__, np.__version__) == ('1.17.1', '2.4.3')
for index, (loc, scale) in enumerate([(-4.0, 0.2), (0.0, 1.0), (3.5, 6.0)]):
    data = stats.logistic.rvs(loc=loc, scale=scale, size=2000,
                              random_state=np.random.default_rng(17420 + index))
    fitted = stats.logistic.fit(data)
    nll = -stats.logistic.logpdf(data, *fitted).sum()
    print(index, *(format(float(x), '.17g') for x in (*fitted, nll)))
    print(' '.join(format(x, '.17g') for x in data))
"#,
        ])
        .output()
        .expect("run SciPy oracle");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let mut lines = stdout.lines();
    eprintln!("{}", lines.next().expect("oracle versions"));
    let mut compared = 0;
    while let Some(row) = lines.next() {
        let fields: Vec<f64> = row.split_whitespace().map(|v| v.parse().unwrap()).collect();
        let data: Vec<f64> = lines
            .next()
            .expect("oracle data")
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        assert_eq!(fields.len(), 4);
        assert_eq!(data.len(), 2000);
        for mode in [RuntimeMode::Strict, RuntimeMode::Hardened] {
            with_stats_runtime_mode(mode, || {
                let fitted = Logistic::try_fit(&data).expect("logistic MLE");
                let actual = nll(fitted, &data);
                assert!(actual <= fields[3] + 1.0e-6 * fields[3].abs().max(1.0));
                assert!((fitted.loc - fields[1]).abs() < 1.0e-6 * fields[2]);
                assert!((fitted.scale - fields[2]).abs() < 1.0e-6 * fields[2]);
                eprintln!(
                    "logistic case={} mode={mode:?} fit={fitted:?} NLL={actual:.14} SciPy={:.14}",
                    fields[0], fields[3]
                );
            });
        }
        compared += 1;
    }
    assert_eq!(compared, 3, "every Logistic fixture must be compared");
}
