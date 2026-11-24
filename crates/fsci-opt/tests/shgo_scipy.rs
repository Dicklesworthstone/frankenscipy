#![forbid(unsafe_code)]

//! Simplicial global-search regressions and required live-SciPy comparisons.
//! Set FSCI_REQUIRE_SCIPY_ORACLE=1 and FSCI_PYTHON to require the pinned oracle.

use std::cell::Cell;
use std::process::Command;

use fsci_opt::{Constraint, ConvergenceStatus, OptError, ShgoOptions, shgo, shgo_with_options};
use serde::Deserialize;

fn himmelblau(x: &[f64]) -> f64 {
    (x[0] * x[0] + x[1] - 11.0).powi(2) + (x[0] + x[1] * x[1] - 7.0).powi(2)
}

#[test]
fn shgo_retains_every_sampled_basin_instead_of_eight_ranked_restarts() {
    let frequency = 12.0 * std::f64::consts::PI;
    let result = shgo_with_options(
        |x| ((frequency * x[0]).sin() / frequency).powi(2),
        &[(0.0, 1.0)],
        ShgoOptions {
            initial_subdivisions: 48,
            local_tolerance: 1e-7,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(result.optimum.success, "{result:?}");
    assert_eq!(result.xl.len(), 13, "{result:?}");
    assert_eq!(result.local_minimizations, 13);
    for well in 0..=12 {
        let expected = f64::from(well) / 12.0;
        assert!(
            result
                .xl
                .iter()
                .any(|point| (point[0] - expected).abs() < 1e-7),
            "missing well at {expected}: {:?}",
            result.xl
        );
    }
    assert!(result.funl.iter().all(|value| *value < 1e-16));
    assert!(result.funl.windows(2).all(|pair| pair[0] <= pair[1]));
}

#[test]
fn shgo_refinement_reuses_vertices_and_collapses_flat_components() {
    let calls = Cell::new(0usize);
    let result = shgo_with_options(
        |_| {
            calls.set(calls.get() + 1);
            7.0
        },
        &[(-1.0, 1.0); 2],
        ShgoOptions {
            initial_subdivisions: 2,
            iterations: 3,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(result.optimum.success, "{result:?}");
    assert_eq!(result.optimum.nit, 3);
    // The 3x3 and 5x5 vertices occur again in the final 9x9 grid.
    assert_eq!(result.samples, 81);
    assert_eq!(result.sampling_nfev, 81);
    assert_eq!(result.local_minimizations, 3);
    assert_eq!(result.xl.len(), 1);
    assert_eq!(result.funl, vec![7.0]);
    assert_eq!(result.optimum.nfev, calls.get());
    assert_eq!(result.optimum.nfev, result.sampling_nfev + result.nlfev);
}

#[test]
fn shgo_checks_exponential_sampling_and_edge_budgets_before_callbacks() {
    for options in [
        ShgoOptions {
            initial_subdivisions: usize::MAX,
            ..Default::default()
        },
        ShgoOptions {
            max_samples: 24,
            ..Default::default()
        },
        ShgoOptions {
            max_complex_edges: 1,
            ..Default::default()
        },
    ] {
        let constraints = [Constraint::ineq(|_| {
            panic!("preflight must precede constraints")
        })];
        assert!(matches!(
            shgo_with_options(
                |_| panic!("preflight must precede objective evaluation"),
                &[(0.0, 1.0); 2],
                ShgoOptions {
                    constraints: &constraints,
                    ..options
                },
            ),
            Err(OptError::InvalidArgument { .. })
        ));
    }
    // The previous implementation attempted to allocate 3^40 vertices.
    assert!(matches!(
        shgo(
            |_| panic!("high-dimensional grid must be refused"),
            &[(0.0, 1.0); 40]
        ),
        Err(OptError::InvalidArgument { .. })
    ));
    for options in [
        ShgoOptions {
            iterations: 0,
            ..Default::default()
        },
        ShgoOptions {
            local_tolerance: f64::NAN,
            ..Default::default()
        },
        ShgoOptions {
            minimizer_tolerance: -1.0,
            ..Default::default()
        },
    ] {
        assert!(shgo_with_options(|_| panic!("invalid options"), &[(0.0, 1.0)], options).is_err());
    }
}

#[test]
fn shgo_keeps_best_point_and_reports_unfinished_refinement() {
    let result = shgo_with_options(
        |x| (x[0] - 0.37).powi(2),
        &[(0.0, 1.0)],
        ShgoOptions {
            iterations: 2,
            max_samples: 5,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!result.optimum.success);
    assert_eq!(result.optimum.status, ConvergenceStatus::MaxEvaluations);
    assert_eq!(result.optimum.nit, 1);
    assert!((result.optimum.x[0] - 0.37).abs() < 1e-6);
    assert_eq!(result.sampling_nfev, 5);
    assert_eq!(result.xl.len(), 1);
}

#[test]
fn shgo_objective_budgets_count_actual_callbacks_and_do_not_claim_success() {
    for budget in [1, 5, 8] {
        let calls = Cell::new(0usize);
        let result = shgo_with_options(
            |x| {
                calls.set(calls.get() + 1);
                (x[0] - 0.37).powi(2)
            },
            &[(0.0, 1.0)],
            ShgoOptions {
                max_evaluations: budget,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(!result.optimum.success, "budget {budget}: {result:?}");
        assert_eq!(result.optimum.status, ConvergenceStatus::MaxEvaluations);
        assert_eq!(calls.get(), budget);
        assert_eq!(result.optimum.nfev, budget);
        assert_eq!(result.sampling_nfev + result.nlfev, budget);
        assert!(result.optimum.fun.unwrap().is_finite());
        assert!((0.0..=1.0).contains(&result.optimum.x[0]));
    }
    let limited = shgo_with_options(
        |x| (x[0] - 0.37).powi(2),
        &[(0.0, 1.0)],
        ShgoOptions {
            local_max_evaluations: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!limited.optimum.success, "{limited:?}");
    assert_eq!(limited.optimum.status, ConvergenceStatus::MaxEvaluations);
    assert_eq!(limited.failed_local_minimizations, 1);
    assert!(limited.xl.is_empty());
    assert_eq!(limited.nlfev, 1);

    let stalled = shgo_with_options(
        |x| 100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2),
        &[(-1.5, 1.5); 2],
        ShgoOptions {
            local_max_iterations: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!stalled.optimum.success, "{stalled:?}");
    assert!(stalled.failed_local_minimizations > 0);
    assert_eq!(stalled.optimum.status, ConvergenceStatus::MaxIterations);
}

#[test]
fn shgo_finite_sampling_geometry_covers_extreme_intervals() {
    let result = shgo_with_options(
        |x| {
            assert!(x[0].is_finite());
            3.0
        },
        &[(-1e308, 1e308)],
        ShgoOptions {
            local_max_iterations: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!result.optimum.success);
    assert_eq!(result.sampling_nfev, 5);
    assert_eq!(result.optimum.fun, Some(3.0));
    assert_eq!(result.optimum.status, ConvergenceStatus::MaxIterations);

    let missing = shgo(|_| f64::NAN, &[(0.0, 1.0)]).unwrap();
    assert!(!missing.success);
    assert_eq!(missing.status, ConvergenceStatus::NanEncountered);
    assert!(missing.fun.is_none());
    assert!(missing.x.is_empty());
}

#[test]
fn shgo_local_domain_failures_preserve_the_best_finite_sample() {
    let result = shgo_with_options(
        |x| {
            if (4.0 * x[0]).fract() == 0.0 {
                (x[0] - 0.37).powi(2)
            } else {
                // The grid is valid but every numerical-gradient stencil
                // leaves the objective's domain. A finite seed is not success.
                f64::NAN
            }
        },
        &[(0.0, 1.0)],
        ShgoOptions::default(),
    )
    .unwrap();
    assert!(!result.optimum.success, "{result:?}");
    assert_eq!(result.failed_local_minimizations, 1);
    assert_eq!(result.optimum.x, vec![0.25]);
    assert_eq!(result.optimum.fun, Some((0.25_f64 - 0.37).powi(2)));
    assert!(result.xl.is_empty());
    assert_eq!(result.optimum.nfev, result.sampling_nfev + result.nlfev);

    let finite_component = shgo(
        |x| {
            if x[0] < 0.0 {
                f64::NAN
            } else {
                (x[0] - 0.3).powi(2)
            }
        },
        &[(-1.0, 1.0)],
    )
    .unwrap();
    assert!(finite_component.success, "{finite_component:?}");
    assert!((finite_component.x[0] - 0.3).abs() < 1e-6);
}

#[test]
fn shgo_constraints_filter_sampling_and_are_verified_after_local_solves() {
    let constraints = [Constraint::ineq(|x| vec![x[0] - 0.5])];
    let sampled = shgo_with_options(
        |x| {
            assert!(x[0] >= 0.5);
            x[0]
        },
        &[(0.0, 1.0)],
        ShgoOptions {
            constraints: &constraints,
            local_max_iterations: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(sampled.samples, 5);
    assert_eq!(sampled.sampling_nfev, 3);
    assert_eq!(sampled.optimum.x, vec![0.5]);

    let equality = [Constraint::eq(|x| vec![x[0] + x[1] - 1.0])];
    let fit = shgo_with_options(
        |x| x[0] * x[0] + x[1] * x[1],
        &[(0.0, 1.0); 2],
        ShgoOptions {
            constraints: &equality,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(fit.optimum.success, "{fit:?}");
    assert!(
        fit.optimum
            .x
            .iter()
            .all(|value| (*value - 0.5).abs() < 1e-7)
    );
    assert!(fit.optimum.maxcv.unwrap() < 1e-7);

    let impossible = [Constraint::ineq(|_| vec![-1.0])];
    let no_solution = shgo_with_options(
        |_| panic!("infeasible samples must not evaluate the objective"),
        &[(0.0, 1.0)],
        ShgoOptions {
            constraints: &impossible,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!no_solution.optimum.success);
    assert_eq!(no_solution.optimum.status, ConvergenceStatus::Infeasible);
    assert_eq!(no_solution.optimum.nfev, 0);

    let malformed = [Constraint::ineq(|_| vec![f64::NAN])];
    assert!(matches!(
        shgo_with_options(
            |_| panic!("malformed constraints must be refused first"),
            &[(0.0, 1.0)],
            ShgoOptions {
                constraints: &malformed,
                ..Default::default()
            },
        ),
        Err(OptError::NonFiniteInput { .. })
    ));
    let changing = [Constraint::ineq(|x| {
        if x[0] == 0.0 {
            vec![1.0]
        } else {
            vec![1.0, 1.0]
        }
    })];
    assert!(matches!(
        shgo_with_options(
            |_| 0.0,
            &[(0.0, 1.0)],
            ShgoOptions {
                constraints: &changing,
                ..Default::default()
            },
        ),
        Err(OptError::InvalidArgument { .. })
    ));
}

#[derive(Debug, Deserialize)]
struct OracleRow {
    name: String,
    x: Vec<f64>,
    fun: f64,
    xl: Vec<Vec<f64>>,
}

#[test]
fn shgo_global_and_local_minima_match_live_scipy() {
    let required = std::env::var("FSCI_REQUIRE_SCIPY_ORACLE").as_deref() == Ok("1");
    let python = std::env::var("FSCI_PYTHON").unwrap_or_else(|_| "python3".into());
    let available = Command::new(&python)
        .args(["-c", "import scipy; import numpy"])
        .output()
        .is_ok_and(|output| output.status.success());
    if !available {
        assert!(!required, "required SciPy oracle unavailable via {python}");
        eprintln!("SKIP shgo: missing SciPy; compared cases = 0");
        return;
    }
    let script = r#"
import json, os
import numpy as np
import scipy
from scipy.optimize import shgo
if os.environ.get('FSCI_REQUIRE_SCIPY_ORACLE') == '1':
    assert (scipy.__version__, np.__version__) == ('1.17.1', '2.4.3')
cases = [
    ('shifted', lambda x: (x[0]-.37)**2+2*(x[1]+.28)**2, [(-2,2)]*2, None),
    ('quartic', lambda x: x[0]**4-4*x[0]**2+x[0], [(-3,3)], None),
    ('himmelblau', lambda x: (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2, [(-6,6)]*2, None),
    ('boundary', lambda x: (x[0]+2)**2+(x[1]-.3)**2, [(0,1)]*2, None),
    ('inequality', lambda x: x[0]**2+x[1]**2, [(0,2)]*2,
     {'type':'ineq','fun':lambda x: x[0]+x[1]-1}),
    ('equality', lambda x: x[0]**2+x[1]**2, [(0,1)]*2,
     {'type':'eq','fun':lambda x: x[0]+x[1]-1}),
]
rows=[]
for name, fun, bounds, constraints in cases:
    result=shgo(fun,bounds,n=128,iters=3,constraints=constraints,
                minimizer_kwargs={'method':'SLSQP','options':{'ftol':1e-12,'maxiter':1000}})
    assert result.success, (name,result)
    rows.append(dict(name=name,x=result.x.tolist(),fun=float(result.fun),xl=result.xl.tolist()))
print(json.dumps(rows,allow_nan=False))
"#;
    let output = Command::new(&python).args(["-c", script]).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let rows: Vec<OracleRow> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(rows.len(), 6);
    let mut compared = 0usize;
    for row in rows {
        let equality = [Constraint::eq(|x| vec![x[0] + x[1] - 1.0])];
        let inequality = [Constraint::ineq(|x| vec![x[0] + x[1] - 1.0])];
        let options = ShgoOptions {
            initial_subdivisions: 8,
            iterations: 2,
            local_tolerance: 1e-9,
            local_max_iterations: 1000,
            constraints: match row.name.as_str() {
                "inequality" => &inequality,
                "equality" => &equality,
                _ => &[],
            },
            ..Default::default()
        };
        let result = match row.name.as_str() {
            "shifted" => shgo_with_options(
                |x| (x[0] - 0.37).powi(2) + 2.0 * (x[1] + 0.28).powi(2),
                &[(-2.0, 2.0); 2],
                options,
            ),
            "quartic" => shgo_with_options(
                |x| x[0].powi(4) - 4.0 * x[0] * x[0] + x[0],
                &[(-3.0, 3.0)],
                options,
            ),
            "himmelblau" => shgo_with_options(himmelblau, &[(-6.0, 6.0); 2], options),
            "boundary" => shgo_with_options(
                |x| (x[0] + 2.0).powi(2) + (x[1] - 0.3).powi(2),
                &[(0.0, 1.0); 2],
                options,
            ),
            "inequality" => {
                shgo_with_options(|x| x[0] * x[0] + x[1] * x[1], &[(0.0, 2.0); 2], options)
            }
            "equality" => {
                shgo_with_options(|x| x[0] * x[0] + x[1] * x[1], &[(0.0, 1.0); 2], options)
            }
            name => panic!("unexpected oracle case {name}"),
        }
        .unwrap();
        assert!(result.optimum.success, "{}: {result:?}", row.name);
        let error = (result.optimum.fun.unwrap() - row.fun).abs();
        assert!(
            error <= 1e-8 * (1.0 + row.fun.abs()),
            "{}: {result:?} vs {row:?}",
            row.name
        );
        if row.name != "himmelblau" {
            assert!(
                result
                    .optimum
                    .x
                    .iter()
                    .zip(&row.x)
                    .all(|(a, b)| (a - b).abs() < 2e-5),
                "{}: {result:?} vs {row:?}",
                row.name
            );
        } else {
            // Four analytically known basins; compare the entire minima set,
            // since any one of them may be selected as the global best.
            let expected = [
                [3.0, 2.0],
                [-2.80511808695, 3.13131251825],
                [-3.77931025338, -3.28318599129],
                [3.58442834033, -1.84812652696],
            ];
            assert_eq!(result.xl.len(), 4, "{result:?}");
            for minimum in expected {
                assert!(
                    result
                        .xl
                        .iter()
                        .any(|point| point.iter().zip(minimum).all(|(a, b)| (a - b).abs() < 2e-5)),
                    "{result:?}"
                );
                assert!(
                    row.xl
                        .iter()
                        .any(|point| point.iter().zip(minimum).all(|(a, b)| (a - b).abs() < 2e-5)),
                    "{row:?}"
                );
            }
        }
        assert_eq!(result.optimum.nfev, result.sampling_nfev + result.nlfev);
        compared += 1;
        eprintln!(
            "shgo {}: value error {error:.3e}, {} local minima, {} objective calls",
            row.name,
            result.xl.len(),
            result.optimum.nfev
        );
    }
    assert_eq!(compared, 6);
    eprintln!("shgo live SciPy compared cases = {compared}");
}
