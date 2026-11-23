#![forbid(unsafe_code)]
//! Live SciPy differential checks for LSQR controls, diagnostics, and variance.
//! Set FSCI_REQUIRE_SCIPY_ORACLE=1 to make a missing oracle an error and
//! SCIPY_PYTHON to the pinned SciPy 1.17.1 / NumPy 2.4.3 interpreter.

use std::io::Write;
use std::process::{Command, Stdio};

use fsci_sparse::{
    CooMatrix, CsrMatrix, DenseLinearOperator, FormatConvertible, FunctionOperator, LinearOperator,
    LsqrOptions, LsqrResult, Shape2D, lsqr_with_options,
};
use serde_json::{Value, json};

// Residual estimates near Krylov exhaustion can differ by roundoff; the
// absolute allowance is far below the requested solution tolerances.
const VALUE_RTOL: f64 = 5e-10;
const VALUE_ATOL: f64 = 2e-12;

#[derive(Clone)]
struct Case {
    id: &'static str,
    a: Vec<Vec<f64>>,
    b: Vec<f64>,
    atol: f64,
    btol: f64,
    conlim: f64,
    max_iter: Option<usize>,
    damp: f64,
    x0: Option<Vec<f64>>,
    calc_var: bool,
}

impl Case {
    fn new(id: &'static str, a: &[&[f64]], b: &[f64]) -> Self {
        Self {
            id,
            a: a.iter().map(|row| row.to_vec()).collect(),
            b: b.to_vec(),
            atol: 1e-10,
            btol: 1e-10,
            conlim: 1e8,
            max_iter: None,
            damp: 0.0,
            x0: None,
            calc_var: false,
        }
    }

    fn options(&self) -> LsqrOptions<'_> {
        LsqrOptions {
            atol: self.atol,
            btol: self.btol,
            conlim: self.conlim,
            max_iter: self.max_iter,
            damp: self.damp,
            x0: self.x0.as_deref(),
            calc_var: self.calc_var,
            ..LsqrOptions::default()
        }
    }

    fn query(&self) -> Value {
        json!({
            "id": self.id, "a": self.a, "b": self.b,
            "atol": self.atol, "btol": self.btol, "conlim": self.conlim,
            "iter_lim": self.max_iter, "damp": self.damp.abs(),
            "x0": self.x0, "calc_var": self.calc_var,
        })
    }
}

fn cases() -> Vec<Case> {
    let diagonal: &[&[f64]] = &[&[1.0, 0.0, 0.0], &[0.0, 0.1, 0.0], &[0.0, 0.0, 0.01]];
    let identity: &[&[f64]] = &[&[1.0, 0.0], &[0.0, 1.0]];
    let mut cases = vec![
        Case::new(
            "square_consistent",
            &[&[4.0, 1.0], &[1.0, 3.0]],
            &[1.0, 2.0],
        ),
        Case::new(
            "tall_consistent",
            &[&[3.0, 1.0], &[1.0, 2.0], &[2.0, -1.0]],
            &[5.0, 5.0, 0.0],
        ),
        Case::new(
            "tall_inconsistent",
            &[&[1.0, 0.0], &[0.0, 1.0], &[1.0, 1.0]],
            &[1.0, 2.0, 4.0],
        ),
        Case::new(
            "wide_minimum_norm",
            &[&[1.0, 0.0, 1.0], &[0.0, 1.0, 1.0]],
            &[1.0, 3.0],
        ),
        Case::new(
            "rank_deficient",
            &[&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0], &[1.0, 0.0, 1.0]],
            &[1.0, -1.0, 2.0],
        ),
        Case::new("zero_operator", &[&[0.0, 0.0]], &[1.0]),
        Case::new("zero_rhs", identity, &[0.0, 0.0]),
        Case::new("exact_breakdown", &[&[1.0], &[0.0]], &[1.0, 1.0]),
    ];
    let mut exact = Case::new("stationary_warm_start", identity, &[1.0, 2.0]);
    exact.x0 = Some(vec![1.0, 2.0]);
    exact.damp = 0.5;
    exact.calc_var = true;
    cases.push(exact);
    let mut nullspace = Case::new("nullspace_warm_start", &[&[1.0, 0.0]], &[0.0]);
    nullspace.x0 = Some(vec![0.0, 5.0]);
    cases.push(nullspace);
    for (id, damp) in [
        ("warm_start_variance", 0.0),
        ("damped_variance", 0.5),
        ("negative_damping", -0.5),
    ] {
        let mut case = Case::new(
            id,
            &[&[1.0, 0.0, 0.0], &[0.0, 2.0, 0.0], &[0.0, 0.0, 3.0]],
            &[1.0, 2.0, 4.0],
        );
        case.damp = damp;
        case.x0 = Some(vec![0.2, -0.1, 0.5]);
        case.calc_var = true;
        cases.push(case);
    }
    let mut condition = Case::new("condition_limit", diagonal, &[1.0; 3]);
    condition.atol = 0.0;
    condition.btol = 0.0;
    condition.conlim = 1.1;
    condition.max_iter = Some(20);
    cases.push(condition);
    let mut budget = Case::new("iteration_limit", diagonal, &[1.0; 3]);
    budget.atol = 0.0;
    budget.btol = 0.0;
    budget.conlim = 0.0;
    budget.max_iter = Some(1);
    cases.push(budget);
    let mut zero_budget = Case::new("zero_iteration_budget", diagonal, &[1.0; 3]);
    zero_budget.max_iter = Some(0);
    cases.push(zero_budget);
    let mut atol_only = Case::new("atol_only", diagonal, &[1.0; 3]);
    atol_only.atol = 0.2;
    atol_only.btol = 0.0;
    cases.push(atol_only);
    let mut btol_only = Case::new("btol_overrides_iteration_limit", diagonal, &[1.0; 3]);
    btol_only.atol = 0.0;
    btol_only.btol = 0.9;
    btol_only.max_iter = Some(1);
    cases.push(btol_only);
    cases
}

fn oracle(cases: &[Case]) -> Option<Value> {
    let script = r#"
import json
import sys
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
assert scipy.__version__ == "1.17.1", scipy.__version__
assert np.__version__ == "2.4.3", np.__version__
results = []
for c in json.load(sys.stdin):
    out = lsqr(csr_matrix(c["a"]), np.asarray(c["b"], dtype=float),
               atol=c["atol"], btol=c["btol"], conlim=c["conlim"],
               iter_lim=c["iter_lim"], damp=c["damp"], calc_var=c["calc_var"],
               x0=None if c["x0"] is None else np.asarray(c["x0"], dtype=float))
    assert np.isfinite(out[0]).all() and np.isfinite(out[3:9]).all(), c["id"]
    assert np.isfinite(out[9]).all(), c["id"]
    results.append({"id": c["id"], "solution": out[0].tolist(),
                    "istop": int(out[1]), "iterations": int(out[2]),
                    "diagnostics": [float(v) for v in out[3:9]], "var": out[9].tolist()})
print(json.dumps(results, allow_nan=False))
"#;
    let required = std::env::var_os("FSCI_REQUIRE_SCIPY_ORACLE").is_some();
    let python = std::env::var("SCIPY_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let spawned = Command::new(python)
        .args(["-c", script])
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("OMP_NUM_THREADS", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let mut child = match spawned {
        Ok(child) => child,
        Err(error) => {
            assert!(
                !required,
                "required LSQR SciPy oracle could not start: {error}"
            );
            eprintln!("skipping optional LSQR SciPy oracle: {error}");
            return None;
        }
    };
    let query: Vec<_> = cases.iter().map(Case::query).collect();
    let encoded = serde_json::to_vec(&query).expect("encode LSQR cases");
    let write_result = child
        .stdin
        .take()
        .expect("oracle stdin")
        .write_all(&encoded);
    let output = child.wait_with_output().expect("wait for LSQR oracle");
    if !output.status.success() || write_result.is_err() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !required,
            "required LSQR oracle failed: {stderr}; input={write_result:?}"
        );
        eprintln!("skipping optional LSQR oracle: {stderr}");
        return None;
    }
    Some(serde_json::from_slice(&output.stdout).expect("valid LSQR oracle output"))
}

fn sparse_matrix(matrix: &[Vec<f64>]) -> CsrMatrix {
    let mut data = Vec::new();
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    for (i, row) in matrix.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            if value != 0.0 {
                data.push(value);
                rows.push(i);
                cols.push(j);
            }
        }
    }
    CooMatrix::from_triplets(
        Shape2D::new(matrix.len(), matrix[0].len()),
        data,
        rows,
        cols,
        false,
    )
    .expect("valid COO")
    .to_csr()
    .expect("valid CSR")
}

fn close(actual: f64, expected: f64) -> bool {
    actual.is_finite()
        && expected.is_finite()
        && (actual - expected).abs() <= VALUE_ATOL + VALUE_RTOL * expected.abs()
}

fn compare(result: &LsqrResult, expected: &Value) -> bool {
    let solution = expected["solution"].as_array().expect("oracle solution");
    let variance = expected["var"].as_array().expect("oracle variance");
    let diagnostics = expected["diagnostics"]
        .as_array()
        .expect("oracle diagnostics");
    assert_eq!(diagnostics.len(), 6);
    result.istop == expected["istop"].as_i64().expect("oracle stop code") as i32
        && result.iterations == expected["iterations"].as_u64().expect("oracle iterations") as usize
        && result.solution.len() == solution.len()
        && result.var.len() == variance.len()
        && result
            .solution
            .iter()
            .zip(solution)
            .all(|(&a, b)| close(a, b.as_f64().expect("finite solution")))
        && result
            .var
            .iter()
            .zip(variance)
            .all(|(&a, b)| close(a, b.as_f64().expect("finite variance")))
        && [
            result.r1norm,
            result.r2norm,
            result.anorm,
            result.acond,
            result.arnorm,
            result.xnorm,
        ]
        .iter()
        .zip(diagnostics)
        .all(|(&a, b)| close(a, b.as_f64().expect("finite diagnostic")))
}

#[test]
fn lsqr_options_match_live_scipy() {
    let cases = cases();
    let Some(expected) = oracle(&cases) else {
        return;
    };
    let expected = expected.as_array().expect("oracle array");
    assert_eq!(expected.len(), cases.len());
    let mut compared = 0;
    for (case, reference) in cases.iter().zip(expected) {
        assert_eq!(reference["id"].as_str(), Some(case.id));
        let dense = DenseLinearOperator::from_rows(&case.a).expect("dense operator");
        let sparse = sparse_matrix(&case.a);
        let function = FunctionOperator::new(sparse.shape(), |x| dense.matvec(x))
            .with_rmatvec(|x| dense.rmatvec(x));
        for (kind, operator) in [
            ("csr", &sparse as &dyn LinearOperator),
            ("dense", &dense as &dyn LinearOperator),
            ("matrix_free", &function as &dyn LinearOperator),
        ] {
            let result =
                lsqr_with_options(operator, &case.b, case.options()).unwrap_or_else(|error| {
                    panic!(
                        "{} {kind}: Rust error vs finite SciPy result: {error}; oracle={reference}",
                        case.id
                    )
                });
            assert!(
                compare(&result, reference),
                "{} {kind}: Rust={result:?}; SciPy={reference}",
                case.id
            );
            compared += 1;
            println!(
                "{} {kind}: istop={} iterations={} acond={:.6e} compared={compared}",
                case.id, result.istop, result.iterations, result.acond
            );
        }
    }
    assert_eq!(compared, 3 * cases.len());
    assert!(compared >= 54, "the LSQR oracle corpus must actually run");
    println!("LSQR SciPy differential comparisons: {compared}");
}

#[test]
fn comparator_rejects_incorrect_diagnostics_and_missing_variance() {
    let expected = json!({
        "istop": 1, "iterations": 1, "solution": [2.0],
        "diagnostics": [0.0, 0.0, 1.0, 1.0, 0.0, 2.0], "var": [1.0],
    });
    let reference = LsqrResult {
        solution: vec![2.0],
        istop: 1,
        iterations: 1,
        r1norm: 0.0,
        r2norm: 0.0,
        anorm: 1.0,
        acond: 1.0,
        arnorm: 0.0,
        xnorm: 2.0,
        var: vec![1.0],
    };
    assert!(compare(&reference, &expected));
    let mut wrong = reference.clone();
    wrong.istop = 7;
    assert!(!compare(&wrong, &expected));
    wrong = reference.clone();
    wrong.acond = 0.0;
    assert!(!compare(&wrong, &expected));
    wrong = reference.clone();
    wrong.var = vec![0.0];
    assert!(!compare(&wrong, &expected));
    wrong = reference;
    wrong.solution[0] = f64::NAN;
    assert!(!compare(&wrong, &expected));
}

mod lsqr_options_tests {
    use fsci_runtime::RuntimeMode;
    use fsci_sparse::{
        DenseLinearOperator, FunctionOperator, LsqrOptions, Shape2D, SparseError, lsqr_with_options,
    };

    fn dense(rows: &[&[f64]]) -> DenseLinearOperator {
        DenseLinearOperator::from_rows(&rows.iter().map(|row| row.to_vec()).collect::<Vec<_>>())
            .expect("rectangular matrix")
    }

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 2e-12 * expected.abs().max(1.0),
            "actual={actual:.17e}, expected={expected:.17e}"
        );
    }

    #[test]
    fn independent_tolerances_and_stop_reasons() {
        let a = dense(&[&[1.0, 0.0, 0.0], &[0.0, 0.1, 0.0], &[0.0, 0.0, 0.01]]);
        let base = LsqrOptions {
            atol: 0.0,
            btol: 0.0,
            conlim: 0.0,
            max_iter: Some(20),
            ..LsqrOptions::default()
        };
        let condition = lsqr_with_options(
            &a,
            &[1.0; 3],
            LsqrOptions {
                conlim: 1.1,
                ..base
            },
        )
        .expect("condition-limited iterate");
        assert_eq!(condition.istop, 3);
        assert_eq!(condition.iterations, 2);
        assert!(condition.acond > 1.1);
        let budget = lsqr_with_options(
            &a,
            &[1.0; 3],
            LsqrOptions {
                max_iter: Some(1),
                ..base
            },
        )
        .expect("budget-limited iterate");
        assert_eq!(budget.istop, 7);
        assert_eq!(budget.iterations, 1);
        let operator_tolerance =
            lsqr_with_options(&a, &[1.0; 3], LsqrOptions { atol: 0.2, ..base })
                .expect("operator tolerance");
        assert_eq!(operator_tolerance.istop, 2);
        let rhs_tolerance = lsqr_with_options(
            &a,
            &[1.0; 3],
            LsqrOptions {
                btol: 0.9,
                max_iter: Some(1),
                ..base
            },
        )
        .expect("rhs tolerance overrides final-iteration limit");
        assert_eq!(rhs_tolerance.istop, 1);
        assert_eq!(rhs_tolerance.iterations, 1);
        for (left, right) in operator_tolerance
            .solution
            .iter()
            .zip(&rhs_tolerance.solution)
        {
            assert_eq!(left.to_bits(), right.to_bits());
        }
        assert_eq!(rhs_tolerance.var, vec![0.0; 3]);
    }

    #[test]
    fn damped_warm_start_variance_and_residual_estimates() {
        let a = dense(&[&[1.0, 0.0, 0.0], &[0.0, 2.0, 0.0], &[0.0, 0.0, 3.0]]);
        let b = [1.0, 2.0, 4.0];
        let start = [0.2, -0.1, 0.5];
        for damp in [0.0, 0.5, -0.5] {
            let result = lsqr_with_options(
                &a,
                &b,
                LsqrOptions {
                    atol: 1e-12,
                    btol: 1e-12,
                    damp,
                    x0: Some(&start),
                    calc_var: true,
                    ..LsqrOptions::default()
                },
            )
            .expect("regularized least squares with covariance estimates");
            assert!(matches!(result.istop, 1 | 2), "{result:?}");
            assert_eq!(result.iterations, 3);
            let mut residual = 0.0_f64;
            let mut correction = 0.0_f64;
            for index in 0..3 {
                let diagonal = (index + 1) as f64;
                let denominator = diagonal * diagonal + damp * damp;
                close(
                    result.solution[index],
                    (diagonal * b[index] + damp * damp * start[index]) / denominator,
                );
                close(result.var[index], 1.0 / denominator);
                residual = residual.hypot(b[index] - diagonal * result.solution[index]);
                correction = correction.hypot(result.solution[index] - start[index]);
            }
            close(result.r1norm, residual);
            close(result.r2norm, residual.hypot(damp * correction));
            close(result.xnorm, correction);
            close(result.anorm, (14.0 + 3.0 * damp * damp).sqrt());
            assert!(result.arnorm < 1e-12, "{result:?}");
        }
    }

    #[test]
    fn stationary_points_and_zero_budget_keep_scipy_initialization() {
        let identity = dense(&[&[1.0, 0.0], &[0.0, 1.0]]);
        let start = [1.0, -2.0];
        let exact = lsqr_with_options(
            &identity,
            &start,
            LsqrOptions {
                x0: Some(&start),
                damp: 2.0,
                calc_var: true,
                ..LsqrOptions::default()
            },
        )
        .expect("exact warm start");
        assert_eq!(exact.solution, start);
        assert_eq!(exact.istop, 0);
        assert_eq!(exact.iterations, 0);
        assert_eq!(exact.r1norm, 0.0);
        assert_eq!(exact.xnorm, 0.0);
        assert_eq!(exact.var, vec![0.0; 2]);
        let zero = dense(&[&[0.0, 0.0]]);
        let stationary =
            lsqr_with_options(&zero, &[3.0], LsqrOptions::default()).expect("zero operator");
        assert_eq!(stationary.istop, 0);
        assert_eq!(stationary.iterations, 0);
        assert_eq!(stationary.r1norm, 3.0);
        assert_eq!(stationary.anorm, 0.0);
        assert_eq!(stationary.acond, 0.0);
        let budget = lsqr_with_options(
            &identity,
            &start,
            LsqrOptions {
                max_iter: Some(0),
                ..LsqrOptions::default()
            },
        )
        .expect("zero budget");
        assert_eq!(budget.iterations, 0);
        assert_eq!(budget.istop, 0);
        assert_eq!(budget.solution, vec![0.0; 2]);
        close(budget.arnorm, 5.0_f64.sqrt());
    }

    #[test]
    fn zero_rhs_damping_penalizes_the_correction_from_the_warm_start() {
        let a = dense(&[&[1.0, 0.0], &[0.0, 2.0]]);
        let start = [2.0, -3.0];
        let result = lsqr_with_options(
            &a,
            &[0.0, 0.0],
            LsqrOptions {
                atol: 1e-12,
                btol: 1e-12,
                damp: 2.0,
                x0: Some(&start),
                ..LsqrOptions::default()
            },
        )
        .expect("zero rhs does not erase a regularized warm start");
        assert_eq!(result.istop, 2);
        close(result.solution[0], 1.6);
        close(result.solution[1], -1.5);
        close(result.r1norm, 1.6_f64.hypot(3.0));
        close(result.xnorm, 0.4_f64.hypot(1.5));
        assert!(result.arnorm < 1e-12);
    }

    #[test]
    fn hardened_diagnostics_survive_extreme_operator_scales() {
        let start = [0.25, 0.75];
        for scale in [1e-200, 1e-160, 1.0, 1e160, 1e200] {
            let a = dense(&[&[scale, 0.0], &[0.0, 2.0 * scale]]);
            for damping in [0.0, 1.0] {
                let result = lsqr_with_options(
                    &a,
                    &[scale, 2.0 * scale],
                    LsqrOptions {
                        mode: RuntimeMode::Hardened,
                        atol: 1e-12,
                        btol: 1e-12,
                        damp: damping * scale,
                        x0: Some(&start),
                        calc_var: true,
                        max_iter: Some(20),
                        ..LsqrOptions::default()
                    },
                )
                .expect("scaled damped problem");
                assert!(matches!(result.istop, 1 | 2), "scale={scale}, {result:?}");
                let expected = if damping == 0.0 {
                    [1.0, 1.0]
                } else {
                    [0.625, 0.95]
                };
                close(result.solution[0], expected[0]);
                close(result.solution[1], expected[1]);
                assert!(
                    result.acond.is_finite() && result.acond < 10.0,
                    "{result:?}"
                );
                for (index, &variance) in result.var.iter().enumerate() {
                    if scale <= 1e-160 {
                        assert_eq!(variance, f64::INFINITY);
                    } else if scale >= 1e200 {
                        assert_eq!(variance, 0.0);
                    } else if scale == 1e160 {
                        let diagonal = (index + 1) as f64;
                        let expected = 1e-320 / (diagonal * diagonal + damping * damping);
                        assert!(variance.is_subnormal());
                        assert!((variance - expected).abs() <= f64::from_bits(2));
                    }
                }
            }
        }
        let identity = dense(&[&[1.0]]);
        let smallest = f64::from_bits(1);
        let result = lsqr_with_options(&identity, &[smallest], LsqrOptions::default())
            .expect("subnormal rhs remains a nonzero problem");
        assert_eq!(result.solution, vec![smallest]);
        assert_eq!(result.istop, 1);
        assert_eq!(result.iterations, 1);
        let small_identity = dense(&[
            &[1e-308, 0.0, 0.0, 0.0],
            &[0.0, 1e-308, 0.0, 0.0],
            &[0.0, 0.0, 1e-308, 0.0],
            &[0.0, 0.0, 0.0, 1e-308],
        ]);
        let large_solution = lsqr_with_options(
            &small_identity,
            &[1.0; 4],
            LsqrOptions {
                mode: RuntimeMode::Hardened,
                ..LsqrOptions::default()
            },
        )
        .expect("an overflowing aggregate diagnostic must not reject finite solution components");
        assert_eq!(large_solution.istop, 1);
        assert_eq!(large_solution.xnorm, f64::INFINITY);
        for entry in large_solution.solution {
            close(entry / 1e308, 1.0);
        }
    }

    #[test]
    fn rejects_invalid_controls_nonfinite_products_and_unrepresentable_updates() {
        let identity = dense(&[&[1.0]]);
        for controls in [
            LsqrOptions {
                atol: -1.0,
                ..LsqrOptions::default()
            },
            LsqrOptions {
                btol: f64::NAN,
                ..LsqrOptions::default()
            },
            LsqrOptions {
                conlim: f64::NAN,
                ..LsqrOptions::default()
            },
            LsqrOptions {
                damp: f64::INFINITY,
                ..LsqrOptions::default()
            },
        ] {
            assert!(matches!(
                lsqr_with_options(&identity, &[1.0], controls),
                Err(SparseError::InvalidArgument { .. })
            ));
        }
        assert!(matches!(
            lsqr_with_options(
                &identity,
                &[1.0],
                LsqrOptions {
                    x0: Some(&[0.0, 1.0]),
                    ..LsqrOptions::default()
                }
            ),
            Err(SparseError::IncompatibleShape { .. })
        ));
        let bad = FunctionOperator::new(Shape2D::new(1, 1), |_| Ok(vec![f64::NAN]))
            .with_rmatvec(|x| Ok(x.to_vec()));
        assert!(matches!(
            lsqr_with_options(&bad, &[1.0], LsqrOptions::default()),
            Err(SparseError::NonFiniteInput { .. })
        ));
        let incomplete = FunctionOperator::new(Shape2D::new(1, 1), |_| Ok(Vec::new()))
            .with_rmatvec(|x| Ok(x.to_vec()));
        assert!(matches!(
            lsqr_with_options(&incomplete, &[1.0], LsqrOptions::default()),
            Err(SparseError::InvalidShape { .. })
        ));
        let bad_transpose = FunctionOperator::new(Shape2D::new(1, 1), |x| Ok(x.to_vec()))
            .with_rmatvec(|_| Ok(Vec::new()));
        assert!(matches!(
            lsqr_with_options(&bad_transpose, &[1.0], LsqrOptions::default()),
            Err(SparseError::InvalidShape { .. })
        ));
        assert!(matches!(
            lsqr_with_options(
                &identity,
                &[0.0],
                LsqrOptions {
                    x0: Some(&[f64::NAN]),
                    ..LsqrOptions::default()
                }
            ),
            Err(SparseError::NonFiniteInput { .. })
        ));
        let huge = dense(&[&[1e200]]);
        assert!(matches!(
            lsqr_with_options(&huge, &[1e-200], LsqrOptions::default()),
            Err(SparseError::SingularMatrix { .. })
        ));
    }
}
