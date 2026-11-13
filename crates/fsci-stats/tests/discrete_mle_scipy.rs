//! Bounded discrete MLE: integer parameters, signed observations and live SciPy.
//!
//! Set FSCI_REQUIRE_SCIPY_ORACLE=1 and FSCI_PYTHON to require the pinned
//! SciPy 1.17.1 / NumPy 2.4.3 oracle. All other tests are self-contained.

use fsci_opt::DifferentialEvolutionOptions;
use fsci_runtime::RuntimeMode;
use fsci_stats::{
    Binomial, DiscreteDistribution, DiscreteLaplace, FitError, MleResult, NegBinomial, Poisson,
    RandInt, fit_mle_discrete, with_hardened_stats_caps, with_stats_runtime_mode,
};

fn options(integrality: Vec<bool>) -> DifferentialEvolutionOptions {
    DifferentialEvolutionOptions {
        integrality,
        seed: Some(9107),
        tol: 1e-12,
        popsize: 20,
        ..Default::default()
    }
}

#[test]
fn discrete_mle_fits_integer_shapes_and_fixed_parameters() {
    // These are the exact probabilities of Binomial(6, 0.5), as 64 observations.
    // Its likelihood maximum can be checked independently by profiling every n
    // against the closed-form conditional p MLE.
    let mut data = Vec::new();
    for (k, count) in [1, 6, 15, 20, 15, 6, 1].into_iter().enumerate() {
        data.extend(std::iter::repeat_n(k as i64, count));
    }
    let factory = |p: &[f64]| {
        assert_eq!(p[0].fract(), 0.0, "factory must never see a relaxed n");
        Some(Binomial::new(p[0] as u64, p[1]))
    };
    let fitted = fit_mle_discrete(
        &data,
        &[(1.2, 12.8), (0.0, 1.0)],
        factory,
        options(vec![true, false]),
    )
    .unwrap();
    assert_eq!(fitted.parameters[0], 6.0);
    assert!((fitted.parameters[1] - 0.5).abs() < 2e-6);
    for n in 6..=12 {
        let candidate = Binomial::new(n, 3.0 / n as f64);
        let profiled: f64 = -data.iter().map(|&k| candidate.logpmf(k)).sum::<f64>();
        assert!(fitted.negative_log_likelihood <= profiled + 1e-9);
    }

    let fixed = fit_mle_discrete(
        &data,
        &[(10.0, 10.0), (0.0, 1.0)],
        factory,
        options(vec![true, false]),
    )
    .unwrap();
    assert_eq!(fixed.parameters[0].to_bits(), 10.0_f64.to_bits());
    assert!((fixed.distribution.p - 0.3).abs() < 2e-6);
    assert!(fixed.negative_log_likelihood > fitted.negative_log_likelihood);
    let again = fit_mle_discrete(
        &data,
        &[(10.0, 10.0), (0.0, 1.0)],
        factory,
        options(vec![true, false]),
    )
    .unwrap();
    assert_eq!(fixed.parameters, again.parameters);
    assert_eq!(fixed.evaluations, again.evaluations);
    assert_eq!(fixed.iterations, again.iterations);
    assert_eq!(fixed.negative_log_likelihood, again.negative_log_likelihood);
}

#[test]
fn discrete_mle_preserves_signed_samples_and_coupled_domains() {
    let data = [-5, -2, 0, 4, 2, -5, 4];
    let original = data;
    for mode in [RuntimeMode::Strict, RuntimeMode::Hardened] {
        with_stats_runtime_mode(mode, || {
            let fitted = fit_mle_discrete(
                &data,
                &[(-10.0, 3.0), (-3.0, 10.0)],
                |p| (p[0] < p[1]).then(|| RandInt::new(p[0] as i64, p[1] as i64)),
                options(vec![true, true]),
            )
            .unwrap();
            assert_eq!(fitted.parameters, [-5.0, 5.0]);
            assert!((fitted.negative_log_likelihood - 7.0 * 10.0_f64.ln()).abs() < 1e-13);

            let laplace = fit_mle_discrete(
                &data,
                &[(0.01, 5.0)],
                |p| Some(DiscreteLaplace::new(p[0])),
                options(vec![]),
            )
            .unwrap();
            let expected = (7.0_f64 / 22.0).asinh();
            assert!((laplace.parameters[0] - expected).abs() < 2e-6);
        });
    }
    assert_eq!(data, original);
}

#[test]
fn discrete_mle_uses_log_mass_and_rejects_false_successes() {
    let tail = fit_mle_discrete(
        &[1_000, 1_200],
        &[(2.0, 2.0)],
        |p| Some(Poisson::new(p[0])),
        options(vec![]),
    )
    .unwrap();
    assert_eq!(tail.distribution.pmf(1_000), 0.0);
    assert!(tail.negative_log_likelihood.is_finite());
    assert!(tail.negative_log_likelihood > 1_000.0);

    let invalid_support = fit_mle_discrete(
        &[-1, 2, 3],
        &[(0.1, 10.0)],
        |p| Some(Poisson::new(p[0])),
        options(vec![]),
    );
    assert!(matches!(invalid_support, Err(FitError::NonConvergent(_))));
    let invalid_model =
        fit_mle_discrete::<Poisson, _>(&[1, 2], &[(0.0, 1.0)], |_| None, options(vec![]));
    assert!(matches!(invalid_model, Err(FitError::NonConvergent(_))));

    let exhausted = fit_mle_discrete(
        &[0, 1, 2, 2, 3, 4, 5],
        &[(0.1, 20.0)],
        |p| Some(Poisson::new(p[0])),
        DifferentialEvolutionOptions {
            maxiter: 1,
            ..options(vec![])
        },
    );
    assert!(matches!(exhausted, Err(FitError::NonConvergent(_))));
}

#[test]
fn discrete_mle_validates_inputs_before_calling_the_factory() {
    let never = |_: &[f64]| -> Option<Poisson> { panic!("invalid request reached factory") };
    assert!(matches!(
        fit_mle_discrete(&[], &[(0.0, 1.0)], never, options(vec![])),
        Err(FitError::InsufficientData { .. })
    ));
    for (bounds, mask) in [
        (vec![(0.2, 0.8)], vec![true]),
        (vec![(2.0, 1.0)], vec![]),
        (vec![(0.0, f64::INFINITY)], vec![]),
        (vec![(0.0, 1.0)], vec![true, false]),
    ] {
        assert!(matches!(
            fit_mle_discrete(&[1, 2], &bounds, never, options(mask)),
            Err(FitError::UnsupportedData(_))
        ));
    }
    with_stats_runtime_mode(RuntimeMode::Hardened, || {
        with_hardened_stats_caps(2, 2, || {
            assert!(matches!(
                fit_mle_discrete(&[1, 2, 3], &[(0.0, 5.0)], never, options(vec![])),
                Err(FitError::ResourceExhausted { .. })
            ));
        });
    });
}

const ORACLE: &str = r#"
import os
import numpy as np
import scipy
from scipy import stats, optimize
print('versions', scipy.__version__, np.__version__)
if os.environ.get('FSCI_REQUIRE_SCIPY_ORACLE') == '1':
    assert (scipy.__version__, np.__version__) == ('1.17.1', '2.4.3')
cases = [
    ('binom', stats.binom, [(8, .25), (15, .6), (30, .8)], [(1, 60), (0, 1)]),
    ('nbinom', stats.nbinom, [(2, .25), (7, .6), (14, .8)], [(1, 30), (.01, .99)]),
    ('poisson', stats.poisson, [(.6,), (4,), (15,)], [(.001, 30)]),
    ('dlaplace', stats.dlaplace, [(.3,), (1,), (2,)], [(.05, 4)]),
    ('randint', stats.randint, [(-8, 5), (-3, 12), (4, 16)], [(-12, 6), (1, 22)]),
]
for fi, (name, dist, parameters, bounds) in enumerate(cases):
    for ci, shapes in enumerate(parameters):
        data = dist.rvs(*shapes, size=2000,
                        random_state=np.random.default_rng(61000 + 17*fi + ci))
        def solver(fun, bounds, **kwargs):
            return optimize.differential_evolution(
                fun, bounds, **kwargs, seed=9100 + 17*fi + ci,
                tol=1e-12, popsize=20, polish=True)
        fitted = stats.fit(dist, data, bounds, optimizer=solver)
        assert fitted.success, (name, ci, fitted.message)
        parameters = tuple(fitted.params)[:-1]  # loc is explicitly fixed at zero
        print(name, ci, len(parameters), *(format(x, '.17g') for x in parameters),
              format(fitted.nllf(), '.17g'))
        print(' '.join(str(int(x)) for x in data))
"#;

fn parts<D>(result: Result<MleResult<D>, FitError>) -> (Vec<f64>, f64) {
    let result = result.expect("bounded discrete MLE must converge");
    (result.parameters, result.negative_log_likelihood)
}

#[test]
fn bounded_discrete_fits_match_live_scipy_likelihoods() {
    let required = std::env::var("FSCI_REQUIRE_SCIPY_ORACLE").as_deref() == Ok("1");
    let python = std::env::var("FSCI_PYTHON").unwrap_or_else(|_| "python3".into());
    let available = std::process::Command::new(&python)
        .args(["-c", "import scipy, numpy"])
        .output()
        .is_ok_and(|output| output.status.success());
    if !available {
        assert!(!required, "required SciPy oracle unavailable: {python}");
        eprintln!("SKIP bounded_discrete_fits_match_live_scipy_likelihoods: no SciPy");
        return;
    }
    let output = std::process::Command::new(&python)
        .args(["-c", ORACLE])
        .output()
        .expect("start SciPy oracle");
    assert!(
        output.status.success(),
        "SciPy oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let mut lines = stdout.lines();
    println!("{}", lines.next().expect("oracle version line"));
    let mut compared = 0;
    while let Some(header) = lines.next() {
        let columns: Vec<_> = header.split_whitespace().collect();
        let family = columns[0];
        let case: usize = columns[1].parse().unwrap();
        let shape_count: usize = columns[2].parse().unwrap();
        let expected: Vec<f64> = columns[3..3 + shape_count]
            .iter()
            .map(|x| x.parse().unwrap())
            .collect();
        let expected_nll: f64 = columns[3 + shape_count].parse().unwrap();
        let data: Vec<i64> = lines
            .next()
            .expect("shared oracle sample")
            .split_whitespace()
            .map(|x| x.parse().unwrap())
            .collect();
        assert_eq!(data.len(), 2_000);
        let (actual, nll) = match family {
            "binom" => parts(fit_mle_discrete(
                &data,
                &[(1.0, 60.0), (0.0, 1.0)],
                |p| Some(Binomial::new(p[0] as u64, p[1])),
                options(vec![true, false]),
            )),
            "nbinom" => parts(fit_mle_discrete(
                &data,
                &[(1.0, 30.0), (0.01, 0.99)],
                |p| Some(NegBinomial::new(p[0], p[1])),
                options(vec![true, false]),
            )),
            "poisson" => parts(fit_mle_discrete(
                &data,
                &[(0.001, 30.0)],
                |p| Some(Poisson::new(p[0])),
                options(vec![]),
            )),
            "dlaplace" => parts(fit_mle_discrete(
                &data,
                &[(0.05, 4.0)],
                |p| Some(DiscreteLaplace::new(p[0])),
                options(vec![]),
            )),
            "randint" => parts(fit_mle_discrete(
                &data,
                &[(-12.0, 6.0), (1.0, 22.0)],
                |p| (p[0] < p[1]).then(|| RandInt::new(p[0] as i64, p[1] as i64)),
                options(vec![true, true]),
            )),
            _ => panic!("unexpected oracle family {family}"),
        };
        println!(
            "discrete MLE {family}[{case}] n={} params={actual:?} scipy={expected:?} nll={nll:.15e} scipy_nll={expected_nll:.15e}",
            data.len()
        );
        assert_eq!(actual.len(), expected.len());
        for (i, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
            if family == "randint" || (i == 0 && matches!(family, "binom" | "nbinom")) {
                assert_eq!(got, want, "{family}[{case}] integral parameter {i}");
            } else {
                assert!((got - want).abs() <= 2e-5 * (1.0 + want.abs()));
            }
        }
        assert!(
            (nll - expected_nll).abs() <= 1e-8 * expected_nll.abs().max(1.0),
            "{family}[{case}] likelihood disagreement"
        );
        compared += 1;
    }
    assert_eq!(compared, 15, "every family and parameter set must compare");
    println!("compared_discrete_fits={compared}");
}
