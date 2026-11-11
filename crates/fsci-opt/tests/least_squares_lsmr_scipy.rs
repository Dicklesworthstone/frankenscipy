#![forbid(unsafe_code)]

//! The iterative nonlinear least-squares backend against live SciPy.
//! Every generated case must be compared. Set FSCI_REQUIRE_SCIPY_ORACLE=1 to
//! require the repository's pinned SciPy 1.17.1 / NumPy 2.4.3 installation.

use std::process::Command;

use fsci_opt::{
    FiniteDiffMethod, JacobianSparsity, LeastSquaresJacobian, LeastSquaresMethod,
    LeastSquaresOptions, LossKind, OptError, TrustRegionLsmrOptions, TrustRegionSolver, XScale,
    least_squares, least_squares_bounded,
};
use fsci_sparse::{CsrMatrix, Shape2D};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Case {
    problem: String,
    method: String,
    loss: String,
    bounded: bool,
    scale: String,
    regularize: bool,
    jacobian_kind: String,
}

fn sparse_jacobian(problem: &str, x: &[f64]) -> CsrMatrix {
    let dense = jacobian(problem, x);
    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = vec![0];
    for row in &dense {
        for (column, &value) in row.iter().enumerate() {
            if value != 0.0 {
                data.push(value);
                indices.push(column);
            }
        }
        indptr.push(data.len());
    }
    CsrMatrix::from_components(
        Shape2D::new(dense.len(), x.len()),
        data,
        indices,
        indptr,
        true,
    )
    .unwrap()
}

fn pattern(problem: &str, n: usize) -> JacobianSparsity {
    let rows = match problem {
        "rosenbrock" => vec![vec![0, 1], vec![0]],
        "exponential" => vec![vec![0, 1]; 40],
        "scaled" => vec![vec![0, 1], vec![0, 1], vec![0]],
        "coupled" => (0..n)
            .map(|i| if i == 0 { vec![0] } else { vec![i - 1, i] })
            .collect(),
        _ => panic!("unknown problem"),
    };
    JacobianSparsity::from_rows(rows, n).unwrap()
}

fn residuals(problem: &str, x: &[f64]) -> Vec<f64> {
    match problem {
        "rosenbrock" => vec![10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]],
        "exponential" => (0..40)
            .map(|i| {
                let time = 4.0 * i as f64 / 39.0;
                let outlier = match i {
                    5 => 3.0,
                    20 => -2.5,
                    _ => 0.0,
                };
                x[0] * (-x[1] * time).exp() - 2.0 * (-0.3 * time).exp() - outlier
            })
            .collect(),
        "scaled" => vec![
            x[0] + 1.0e6 * x[1] - 4.0,
            2.0 * x[0] - 1.0e6 * x[1] - 2.0,
            x[0] - 2.0,
        ],
        "coupled" => (0..x.len())
            .map(|i| {
                let value = x[i] - (1.0 + i as f64 / x.len() as f64);
                let previous = if i == 0 {
                    0.0
                } else {
                    x[i - 1] - (1.0 + (i - 1) as f64 / x.len() as f64)
                };
                2.0 * value + 0.25 * value * value - previous
            })
            .collect(),
        _ => panic!("unknown test problem {problem}"),
    }
}

fn jacobian(problem: &str, x: &[f64]) -> Vec<Vec<f64>> {
    match problem {
        "rosenbrock" => vec![vec![-20.0 * x[0], 10.0], vec![-1.0, 0.0]],
        "exponential" => (0..40)
            .map(|i| {
                let time = 4.0 * i as f64 / 39.0;
                let exponential = (-x[1] * time).exp();
                vec![exponential, -time * x[0] * exponential]
            })
            .collect(),
        "scaled" => vec![vec![1.0, 1.0e6], vec![2.0, -1.0e6], vec![1.0, 0.0]],
        "coupled" => (0..x.len())
            .map(|i| {
                let mut row = vec![0.0; x.len()];
                row[i] = 2.0 + 0.5 * (x[i] - (1.0 + i as f64 / x.len() as f64));
                if i > 0 {
                    row[i - 1] = -1.0;
                }
                row
            })
            .collect(),
        _ => panic!("unknown test problem {problem}"),
    }
}

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for method in ["trf", "dogbox"] {
        for (problem, bounded, loss, scale) in [
            ("rosenbrock", false, "linear", "unit"),
            ("rosenbrock", true, "linear", "unit"),
            ("exponential", true, "linear", "unit"),
            ("exponential", false, "soft_l1", "unit"),
            ("exponential", false, "huber", "unit"),
            ("exponential", false, "cauchy", "unit"),
            ("exponential", false, "arctan", "unit"),
            ("exponential", true, "soft_l1", "jac"),
            ("scaled", false, "linear", "jac"),
            ("scaled", false, "linear", "fixed"),
            ("coupled", false, "linear", "unit"),
        ] {
            cases.push(Case {
                problem: problem.into(),
                method: method.into(),
                loss: loss.into(),
                bounded,
                scale: scale.into(),
                regularize: true,
                jacobian_kind: "dense".into(),
            });
        }
    }
    for problem in ["rosenbrock", "coupled"] {
        cases.push(Case {
            problem: problem.into(),
            method: "trf".into(),
            loss: "linear".into(),
            bounded: false,
            scale: "unit".into(),
            regularize: false,
            jacobian_kind: "dense".into(),
        });
    }
    let dense_cases = cases.clone();
    for kind in ["sparse", "two-point", "three-point"] {
        for mut case in dense_cases.clone() {
            case.jacobian_kind = kind.into();
            cases.push(case);
        }
    }
    cases
}

const ORACLE: &str = r#"
import json, os, sys
import numpy as np
import scipy
from scipy.optimize import least_squares
from scipy.sparse import csr_array
if os.environ.get("FSCI_REQUIRE_SCIPY_ORACLE") == "1":
    assert (scipy.__version__, np.__version__) == ("1.17.1", "2.4.3")
cases = json.loads(sys.argv[1])
rows = []
for case in cases:
    problem = case["problem"]
    if problem == "rosenbrock":
        fun = lambda x: np.array([10*(x[1]-x[0]**2), 1-x[0]])
        jac = lambda x: np.array([[-20*x[0], 10], [-1, 0]])
        start = [2., 2.] if case["bounded"] else [-1.2, 1.]
        bounds = ([-np.inf, 1.5], [np.inf, np.inf]) if case["bounded"] else (-np.inf, np.inf)
    elif problem == "exponential":
        t = 4*np.arange(40)/39
        y = 2*np.exp(-0.3*t)
        y[5] += 3
        y[20] -= 2.5
        fun = lambda x: x[0]*np.exp(-x[1]*t)-y
        jac = lambda x: np.column_stack((np.exp(-x[1]*t), -t*x[0]*np.exp(-x[1]*t)))
        start = [1., 1.]
        bounds = ([0., 0.5], [10., 5.]) if case["bounded"] else (-np.inf, np.inf)
    elif problem == "scaled":
        a = np.array([[1., 1e6], [2., -1e6], [1., 0.]])
        fun = lambda x: a@x - [4., 2., 2.]
        jac = lambda x: a
        start = [1., 1e-6]
        bounds = (-np.inf, np.inf)
    elif problem == "coupled":
        target = 1 + np.arange(40)/40
        def fun(x):
            d = x-target
            return 2*d + 0.25*d*d - np.r_[0., d[:-1]]
        def jac(x):
            return np.diag(2 + 0.5*(x-target)) + np.diag(-np.ones(39), -1)
        start = np.zeros(40)
        bounds = (-np.inf, np.inf)
    else:
        raise AssertionError(problem)
    scaling = "jac" if case["scale"] == "jac" else ([1., 1e-6] if case["scale"] == "fixed" else 1.)
    options = dict(atol=1e-10, btol=1e-10, conlim=1e12, maxiter=200)
    if case["method"] == "trf":
        options["regularize"] = case["regularize"]
    structure = None
    if case["jacobian_kind"] != "dense":
        dense_jac = jac
        if case["jacobian_kind"] == "sparse":
            jac = lambda x, dense_jac=dense_jac: csr_array(dense_jac(x))
        else:
            structure = csr_array(np.ones_like(dense_jac(start)))
            if problem == "coupled":
                structure = csr_array(np.eye(40) + np.diag(np.ones(39), -1))
            jac = "3-point" if case["jacobian_kind"] == "three-point" else "2-point"
    result = least_squares(fun, start, jac=jac, method=case["method"], bounds=bounds,
                           jac_sparsity=structure,
                           tr_solver="lsmr", tr_options=options, loss=case["loss"],
                           f_scale=0.5 if case["loss"] in ("huber", "cauchy", "arctan") else 0.1,
                           x_scale=scaling, max_nfev=2000,
                           ftol=1e-12, xtol=1e-12, gtol=1e-12)
    assert result.success, (case, result.message)
    assert np.isin(result.active_mask, [-1, 0, 1]).all(), (case, result.active_mask)
    rows.append(dict(x=result.x.tolist(), cost=result.cost,
                     active_mask=result.active_mask.astype(int).tolist(), status=int(result.status)))
print(json.dumps(dict(scipy=scipy.__version__, numpy=np.__version__, rows=rows), allow_nan=False))
"#;

#[derive(Deserialize)]
struct OracleRow {
    x: Vec<f64>,
    cost: f64,
    active_mask: Vec<i8>,
    status: i32,
}

#[derive(Deserialize)]
struct Oracle {
    scipy: String,
    numpy: String,
    rows: Vec<OracleRow>,
}

#[test]
fn iterative_least_squares_matches_live_scipy() {
    let required = std::env::var("FSCI_REQUIRE_SCIPY_ORACLE").as_deref() == Ok("1");
    let python = std::env::var("FSCI_PYTHON").unwrap_or_else(|_| "python3".into());
    let available = Command::new(&python)
        .args(["-c", "import scipy; import numpy"])
        .output()
        .is_ok_and(|output| output.status.success());
    if !available {
        assert!(!required, "required SciPy oracle unavailable via {python}");
        eprintln!("SKIP least-squares LSMR: missing SciPy; compared cases = 0");
        return;
    }
    let cases = cases();
    let output = Command::new(&python)
        .args(["-c", ORACLE, &serde_json::to_string(&cases).unwrap()])
        .output()
        .expect("launch SciPy");
    assert!(
        output.status.success(),
        "SciPy LSMR oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let oracle: Oracle = serde_json::from_slice(&output.stdout).expect("SciPy result JSON");
    if required {
        assert_eq!((&*oracle.scipy, &*oracle.numpy), ("1.17.1", "2.4.3"));
    }
    assert_eq!(oracle.rows.len(), cases.len());
    let mut compared = 0;
    for (case, reference) in cases.iter().zip(&oracle.rows) {
        let (x0, lower, upper) = match case.problem.as_str() {
            "rosenbrock" if case.bounded => (
                vec![2.0, 2.0],
                vec![f64::NEG_INFINITY, 1.5],
                vec![f64::INFINITY; 2],
            ),
            "rosenbrock" => (vec![-1.2, 1.0], vec![], vec![]),
            "exponential" => (vec![1.0, 1.0], vec![0.0, 0.5], vec![10.0, 5.0]),
            "scaled" => (vec![1.0, 1.0e-6], vec![], vec![]),
            "coupled" => (vec![0.0; 40], vec![], vec![]),
            _ => unreachable!(),
        };
        let name = case.problem.clone();
        let options = LeastSquaresOptions {
            method: Some(if case.method == "dogbox" {
                LeastSquaresMethod::Dogbox
            } else {
                LeastSquaresMethod::Trf
            }),
            tr_solver: TrustRegionSolver::Lsmr(TrustRegionLsmrOptions {
                atol: 1.0e-10,
                btol: 1.0e-10,
                conlim: 1.0e12,
                max_iter: Some(200),
                regularize: case.regularize,
                ..TrustRegionLsmrOptions::default()
            }),
            jac: match case.jacobian_kind.as_str() {
                "dense" => Some(LeastSquaresJacobian::new(move |x| jacobian(&name, x))),
                "sparse" => Some(LeastSquaresJacobian::new_sparse(move |x| {
                    sparse_jacobian(&name, x)
                })),
                _ => None,
            },
            jac_sparsity: matches!(case.jacobian_kind.as_str(), "two-point" | "three-point")
                .then(|| pattern(&case.problem, x0.len())),
            diff_method: if case.jacobian_kind == "three-point" {
                FiniteDiffMethod::ThreePoint
            } else {
                FiniteDiffMethod::TwoPoint
            },
            loss: match case.loss.as_str() {
                "soft_l1" => LossKind::SoftL1,
                "huber" => LossKind::Huber,
                "cauchy" => LossKind::Cauchy,
                "arctan" => LossKind::Arctan,
                _ => LossKind::Linear,
            },
            f_scale: if matches!(case.loss.as_str(), "huber" | "cauchy" | "arctan") {
                0.5
            } else {
                0.1
            },
            x_scale: match case.scale.as_str() {
                "jac" => Some(XScale::Jac),
                "fixed" => Some(XScale::Fixed(vec![1.0, 1.0e-6])),
                _ => None,
            },
            max_nfev: Some(2000),
            ftol: 1.0e-12,
            xtol: 1.0e-12,
            gtol: 1.0e-12,
            ..LeastSquaresOptions::default()
        };
        let fun = |x: &[f64]| residuals(&case.problem, x);
        let result = if case.bounded {
            least_squares_bounded(fun, &x0, &lower, &upper, options)
        } else {
            least_squares(fun, &x0, options)
        }
        .unwrap_or_else(|error| panic!("{case:?}: {error}"));
        assert!(
            result.success && result.status > 0 && reference.status > 0,
            "{case:?}"
        );
        assert_eq!(result.x.len(), reference.x.len());
        assert_eq!(result.active_mask, reference.active_mask, "{case:?}");
        assert!(result.optimality.is_finite());
        assert_eq!(result.jac.is_sparse(), case.jacobian_kind != "dense");
        for (actual, expected) in result.x.iter().zip(&reference.x) {
            assert!(
                (actual - expected).abs() <= 1.0e-6 * expected.abs().max(1.0),
                "{case:?}: x {actual} != {expected}"
            );
        }
        assert!(
            (result.cost - reference.cost).abs() <= 1.0e-8 * reference.cost.abs().max(1.0),
            "{case:?}: cost {} != {}",
            result.cost,
            reference.cost
        );
        eprintln!(
            "{case:?}: cost={} SciPy={} nfev={} status={} optimality={}",
            result.cost, reference.cost, result.nfev, result.status, result.optimality
        );
        compared += 1;
    }
    assert_eq!(compared, 96);
    eprintln!("Least-squares LSMR compared cases = {compared}");
}

#[test]
fn lsmr_handles_scalar_rank_deficient_and_numerical_jacobians() {
    for method in [LeastSquaresMethod::Trf, LeastSquaresMethod::Dogbox] {
        let options = LeastSquaresOptions {
            method: Some(method),
            tr_solver: TrustRegionSolver::Lsmr(TrustRegionLsmrOptions::default()),
            ..LeastSquaresOptions::default()
        };
        let scalar = least_squares(|x| vec![x[0] - 3.0], &[0.0], options.clone()).unwrap();
        assert!(scalar.success);
        assert!((scalar.x[0] - 3.0).abs() < 1.0e-7);
        let deficient = least_squares(
            |x| vec![x[0] + x[1] - 3.0, 2.0 * (x[0] + x[1] - 3.0)],
            &[0.0, 0.0],
            options.clone(),
        )
        .unwrap();
        assert!(deficient.success);
        assert!(deficient.cost < 1.0e-14);
        assert!((deficient.x[0] - deficient.x[1]).abs() < 1.0e-7);
        let limited = least_squares(
            |x| vec![10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]],
            &[-1.2, 1.0],
            LeastSquaresOptions {
                max_nfev: Some(1),
                ..options
            },
        )
        .unwrap();
        assert_eq!(limited.nfev, 1);
        assert_eq!(limited.status, 0);
        assert!(!limited.success);
    }
}

#[test]
fn lsmr_rejects_invalid_controls_and_lm_selection() {
    for field in 0..4 {
        let mut controls = TrustRegionLsmrOptions::default();
        match field {
            0 => controls.atol = -1.0,
            1 => controls.btol = f64::NAN,
            2 => controls.conlim = f64::INFINITY,
            _ => controls.damp = f64::NAN,
        }
        let result = least_squares(
            |x| vec![x[0] - 1.0],
            &[0.0],
            LeastSquaresOptions {
                tr_solver: TrustRegionSolver::Lsmr(controls),
                ..LeastSquaresOptions::default()
            },
        );
        assert!(matches!(result, Err(OptError::InvalidArgument { .. })));
    }
    let result = least_squares(
        |x| vec![x[0] - 1.0],
        &[0.0],
        LeastSquaresOptions {
            method: Some(LeastSquaresMethod::Lm),
            tr_solver: TrustRegionSolver::Lsmr(TrustRegionLsmrOptions::default()),
            ..LeastSquaresOptions::default()
        },
    );
    assert!(matches!(result, Err(OptError::InvalidArgument { .. })));
}

#[test]
fn grouped_differences_reduce_calls_and_keep_mixed_bound_stencils() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    // The final parameter is structurally unused. Interior, lower-bound and
    // upper-bound coordinates occur in the same color, so a group cannot use
    // one common central/one-sided stencil.
    let m = 257;
    let n = m + 1;
    let rows = (0..m)
        .map(|i| if i == 0 { vec![0] } else { vec![i, i - 1, i] })
        .collect();
    let sparsity = JacobianSparsity::from_rows(rows, n).unwrap();
    assert_eq!(sparsity.color_count(), 2);
    assert_eq!(sparsity.nnz(), 2 * m - 1);
    let x0: Vec<_> = (0..n).map(|i| [0.0, 0.5, 1.0][i % 3]).collect();
    let lower = vec![0.0; n];
    let upper = vec![1.0; n];
    for method in [LeastSquaresMethod::Trf, LeastSquaresMethod::Dogbox] {
        for diff_method in [FiniteDiffMethod::TwoPoint, FiniteDiffMethod::ThreePoint] {
            let calls = AtomicUsize::new(0);
            let fun = |x: &[f64]| {
                calls.fetch_add(1, Ordering::Relaxed);
                assert!(x.iter().all(|&v| (0.0..=1.0).contains(&v)));
                (0..m)
                    .map(|i| x[i].sin() + if i == 0 { 0.0 } else { x[i - 1].powi(2) })
                    .collect()
            };
            let options = LeastSquaresOptions {
                method: Some(method),
                diff_method,
                max_nfev: Some(1),
                ..LeastSquaresOptions::default()
            };
            let sparse = least_squares_bounded(
                fun,
                &x0,
                &lower,
                &upper,
                LeastSquaresOptions {
                    jac_sparsity: Some(sparsity.clone()),
                    ..options.clone()
                },
            )
            .unwrap();
            let stencil_calls = if diff_method == FiniteDiffMethod::ThreePoint {
                2
            } else {
                1
            };
            assert_eq!(calls.load(Ordering::Relaxed), 1 + 2 * stencil_calls);
            let csr = sparse
                .jac
                .as_sparse()
                .expect("grouped Jacobian stays compressed");
            assert_eq!(csr.shape(), Shape2D::new(m, n));
            assert_eq!(csr.nnz(), 2 * m - 1);
            let dense_values = sparse.jac.to_dense();
            for i in 0..m {
                let tolerance = if stencil_calls == 2 { 2e-9 } else { 5e-8 };
                assert!((dense_values[i][i] - sparse.x[i].cos()).abs() < tolerance);
                if i > 0 {
                    assert!((dense_values[i][i - 1] - 2.0 * sparse.x[i - 1]).abs() < tolerance);
                }
                assert_eq!(dense_values[i][n - 1], 0.0);
            }
            calls.store(0, Ordering::Relaxed);
            let dense = least_squares_bounded(fun, &x0, &lower, &upper, options).unwrap();
            assert_eq!(calls.load(Ordering::Relaxed), 1 + n * stencil_calls);
            for (i, row) in dense.jac.as_dense().unwrap().iter().enumerate() {
                for (j, &value) in row.iter().enumerate() {
                    if j == i || (i > 0 && j == i - 1) {
                        assert_eq!(value, dense_values[i][j], "dependency ({i}, {j})");
                    } else {
                        // A one-sided three-point difference of a constant can
                        // leave cancellation roundoff. Structural zeros skip
                        // that arithmetic; require the dense control to meet
                        // the same derivative accuracy as the analytic checks.
                        let tolerance = if stencil_calls == 2 { 2e-9 } else { 5e-8 };
                        assert_eq!(dense_values[i][j], 0.0);
                        assert!(
                            value.abs() < tolerance,
                            "structural zero ({i}, {j}): {value}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn automatic_sparse_solver_handles_ten_thousand_parameters_without_dense_storage() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let n = 10_000;
    let sparsity = JacobianSparsity::from_rows((0..n).map(|i| vec![i]).collect(), n).unwrap();
    assert_eq!(sparsity.color_count(), 1);
    let calls = AtomicUsize::new(0);
    let result = least_squares(
        |x| {
            calls.fetch_add(1, Ordering::Relaxed);
            x.iter()
                .enumerate()
                .map(|(i, &v)| v - 1.0 - i as f64 / n as f64)
                .collect()
        },
        &vec![0.0; n],
        LeastSquaresOptions {
            jac_sparsity: Some(sparsity),
            x_scale: Some(XScale::Jac),
            gtol: 1e-12,
            xtol: 1e-12,
            ftol: 1e-12,
            ..LeastSquaresOptions::default()
        },
    )
    .unwrap();
    assert!(result.success, "{}", result.message);
    assert!(result.cost < 1e-15, "cost={}", result.cost);
    assert_eq!(result.jac.as_sparse().unwrap().nnz(), n);
    assert_eq!(calls.load(Ordering::Relaxed), result.nfev + result.njev);
    eprintln!(
        "10,000-parameter fit: {} residual calls, {} Jacobians, {} stored entries",
        calls.load(Ordering::Relaxed),
        result.njev,
        n
    );
}

#[test]
fn sparse_jacobian_contracts_fail_explicitly_and_callables_override_hints() {
    let dense_lm = least_squares(
        |x| vec![x[0] - 2.0],
        &[0.0],
        LeastSquaresOptions {
            method: Some(LeastSquaresMethod::Lm),
            jac: Some(LeastSquaresJacobian::new(|_| vec![vec![1.0]])),
            jac_sparsity: Some(JacobianSparsity::from_rows(vec![], 42).unwrap()),
            ..LeastSquaresOptions::default()
        },
    )
    .unwrap();
    assert!(dense_lm.success);
    assert!((dense_lm.x[0] - 2.0).abs() < 1e-6);
    let diagonal = JacobianSparsity::from_rows(vec![vec![0], vec![1]], 2).unwrap();
    assert!(JacobianSparsity::from_rows(vec![vec![2]], 2).is_err());
    for options in [
        LeastSquaresOptions {
            jac_sparsity: Some(JacobianSparsity::from_rows(vec![vec![0]], 1).unwrap()),
            ..LeastSquaresOptions::default()
        },
        LeastSquaresOptions {
            jac_sparsity: Some(diagonal.clone()),
            tr_solver: TrustRegionSolver::Exact,
            ..LeastSquaresOptions::default()
        },
        LeastSquaresOptions {
            jac_sparsity: Some(diagonal),
            method: Some(LeastSquaresMethod::Lm),
            ..LeastSquaresOptions::default()
        },
    ] {
        assert!(matches!(
            least_squares(|x| x.to_vec(), &[1.0, 1.0], options),
            Err(OptError::InvalidArgument { .. })
        ));
    }
    let callback = LeastSquaresJacobian::new_sparse(|_| {
        CsrMatrix::from_components(Shape2D::new(1, 1), vec![1.0], vec![0], vec![0, 1], true)
            .unwrap()
    });
    for method in [LeastSquaresMethod::Trf, LeastSquaresMethod::Dogbox] {
        let result = least_squares(
            |x| vec![x[0] - 2.0],
            &[0.0],
            LeastSquaresOptions {
                method: Some(method),
                jac: Some(callback.clone()),
                jac_sparsity: Some(JacobianSparsity::from_rows(vec![], 42).unwrap()),
                ..LeastSquaresOptions::default()
            },
        )
        .unwrap();
        assert!(result.success);
        assert!((result.x[0] - 2.0).abs() < 1e-8);
        assert!(result.jac.is_sparse());
    }
    for (method, tr_solver) in [
        (LeastSquaresMethod::Lm, TrustRegionSolver::Auto),
        (LeastSquaresMethod::Trf, TrustRegionSolver::Exact),
    ] {
        assert!(matches!(
            least_squares(
                |x| vec![x[0] - 2.0],
                &[0.0],
                LeastSquaresOptions {
                    method: Some(method),
                    tr_solver,
                    jac: Some(callback.clone()),
                    ..LeastSquaresOptions::default()
                },
            ),
            Err(OptError::InvalidArgument { .. })
        ));
    }
    let changing = least_squares(
        |x| if x[0] == 0.0 { vec![1.0] } else { vec![] },
        &[0.0],
        LeastSquaresOptions {
            jac_sparsity: Some(JacobianSparsity::from_rows(vec![vec![0]], 1).unwrap()),
            ..LeastSquaresOptions::default()
        },
    );
    assert!(matches!(changing, Err(OptError::InvalidArgument { .. })));
    for method in [LeastSquaresMethod::Trf, LeastSquaresMethod::Dogbox] {
        for residual_count in [0, 2] {
            let result = least_squares(
                move |_| vec![2.0; residual_count],
                &[1.0, 2.0, 3.0],
                LeastSquaresOptions {
                    method: Some(method),
                    x_scale: Some(XScale::Jac),
                    jac_sparsity: Some(
                        JacobianSparsity::from_rows(vec![vec![]; residual_count], 3).unwrap(),
                    ),
                    ..LeastSquaresOptions::default()
                },
            )
            .unwrap();
            assert!(result.success);
            assert_eq!(result.x, vec![1.0, 2.0, 3.0]);
            assert_eq!(result.grad, vec![0.0; 3]);
            assert_eq!(
                result.jac.as_sparse().unwrap().shape(),
                Shape2D::new(residual_count, 3)
            );
        }
        let result = least_squares(
            |x| vec![x[0] - 1.0],
            &[0.0, 2.0, 3.0],
            LeastSquaresOptions {
                method: Some(method),
                jac_sparsity: Some(JacobianSparsity::from_rows(vec![vec![0]], 3).unwrap()),
                ..LeastSquaresOptions::default()
            },
        )
        .unwrap();
        assert!(result.success);
        assert!((result.x[0] - 1.0).abs() < 1e-8);
        assert_eq!(&result.x[1..], &[2.0, 3.0]);
    }
}
