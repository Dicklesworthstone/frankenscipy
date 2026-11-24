#![forbid(unsafe_code)]
//! Live SciPy comparisons for caller-supplied CG inverse preconditioners.
//!
//! Every case compares both CSR and matrix-free operators. Range-limit
//! regressions use analytical references in the library tests because SciPy's
//! own unscaled dot products underflow or overflow on those inputs.

use std::io::Write;
use std::process::{Command, Stdio};

use fsci_sparse::{
    CooMatrix, FormatConvertible, FunctionOperator, IterativeSolveOptions, LinearOperator, Shape2D,
    cg_preconditioned,
};
use serde_json::{Value, json};

const SOLUTION_RTOL: f64 = 2.0e-9;

struct Case {
    id: &'static str,
    matrix: Vec<Vec<f64>>,
    rhs: Vec<f64>,
    inverse_diagonal: Vec<f64>,
    initial: Option<Vec<f64>>,
    max_iter: usize,
}

fn product(matrix: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| row.iter().zip(x).map(|(a, x)| a * x).sum())
        .collect()
}

fn case(id: &'static str, n: usize, spread: bool, rhs_scale: f64) -> Case {
    let scaling: Vec<f64> = (0..n)
        .map(|i| {
            if spread {
                10.0_f64.powf(6.0 * i as f64 / (n - 1) as f64 - 3.0)
            } else {
                1.0
            }
        })
        .collect();
    let mut matrix = vec![vec![0.0; n]; n];
    for i in 0..n {
        matrix[i][i] = 4.0 * scaling[i] * scaling[i];
        if i + 1 < n {
            let entry = -scaling[i] * scaling[i + 1];
            matrix[i][i + 1] = entry;
            matrix[i + 1][i] = entry;
        }
    }
    let solution: Vec<f64> = (0..n)
        .map(|i| (1.0 + (0.3 * i as f64).sin()) * rhs_scale)
        .collect();
    let rhs = product(&matrix, &solution);
    let inverse_diagonal = (0..n).map(|i| 1.0 / matrix[i][i]).collect();
    Case {
        id,
        matrix,
        rhs,
        inverse_diagonal,
        initial: None,
        max_iter: n * 10,
    }
}

fn cases() -> Vec<Case> {
    let mut cases = vec![
        case("small_spd", 3, false, 1.0),
        case("scaled_spd", 15, true, 1.0),
        case("small_rhs", 9, false, 1e-100),
        case("large_rhs", 9, false, 1e100),
    ];
    let mut warm = case("warm_start", 15, true, 1.0);
    warm.initial = Some((0..15).map(|i| 0.01 * (i % 5) as f64).collect());
    cases.push(warm);
    let mut budget = case("iteration_budget", 9, false, 1.0);
    budget.max_iter = 1;
    cases.push(budget);
    let mut exact = case("exact_initial", 6, false, 1.0);
    exact.initial = Some((0..6).map(|i| i as f64 + 1.0).collect());
    exact.rhs = product(&exact.matrix, exact.initial.as_ref().unwrap());
    cases.push(exact);
    let mut diagonal = case("exact_diagonal_preconditioner", 12, true, 1.0);
    for (i, row) in diagonal.matrix.iter_mut().enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            if i != j {
                *value = 0.0;
            }
        }
    }
    diagonal.rhs = (0..12).map(|i| diagonal.matrix[i][i]).collect();
    cases.push(diagonal);
    cases
}

fn oracle(cases: &[Case]) -> Option<Value> {
    let script = r#"
import json
import sys
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, cg
assert scipy.__version__ == "1.17.1", scipy.__version__
assert np.__version__ == "2.4.3", np.__version__
out = []
for c in json.load(sys.stdin):
    a = csr_matrix(c["a"])
    b = np.asarray(c["b"], dtype=float)
    diagonal = np.asarray(c["inverse_diagonal"], dtype=float)
    m = LinearOperator(a.shape, matvec=lambda r: diagonal * r, dtype=float)
    x0 = None if c["x0"] is None else np.asarray(c["x0"], dtype=float)
    steps = []
    x, info = cg(a, b, x0=x0, M=m, rtol=1e-10, atol=0.0,
                 maxiter=c["maxiter"], callback=lambda x: steps.append(1))
    assert np.isfinite(x).all(), c["id"]
    out.append({"id": c["id"], "x": x.tolist(), "info": int(info),
                "iterations": len(steps)})
print(json.dumps(out, allow_nan=False))
"#;
    let required = std::env::var_os("FSCI_REQUIRE_SCIPY_ORACLE").is_some();
    let python = std::env::var("SCIPY_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let spawned = Command::new(python)
        .arg("-c")
        .arg(script)
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("OMP_NUM_THREADS", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let mut child = match spawned {
        Ok(child) => child,
        Err(error) => {
            assert!(!required, "required SciPy oracle could not start: {error}");
            eprintln!("skipping optional SciPy CG oracle: {error}");
            return None;
        }
    };
    let query: Vec<Value> = cases
        .iter()
        .map(|case| {
            json!({"id": case.id, "a": case.matrix, "b": case.rhs,
                   "inverse_diagonal": case.inverse_diagonal,
                   "x0": case.initial, "maxiter": case.max_iter})
        })
        .collect();
    let input = serde_json::to_vec(&query).expect("serialize CG oracle inputs");
    let write_result = child.stdin.take().unwrap().write_all(&input);
    let output = child.wait_with_output().expect("wait for CG oracle");
    if !output.status.success() || write_result.is_err() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !required,
            "required SciPy CG oracle failed: {stderr}; input={write_result:?}"
        );
        eprintln!("skipping optional SciPy CG oracle: {stderr}");
        return None;
    }
    Some(serde_json::from_slice(&output.stdout).expect("valid SciPy CG results"))
}

#[test]
fn cg_general_preconditioning_matches_scipy() {
    let cases = cases();
    let Some(oracle) = oracle(&cases) else {
        return;
    };
    let oracle = oracle.as_array().expect("one result per case");
    assert_eq!(oracle.len(), cases.len());
    let mut compared = 0;
    for (case, expected) in cases.iter().zip(oracle) {
        assert_eq!(expected["id"].as_str(), Some(case.id));
        let mut values = Vec::new();
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        for (i, row) in case.matrix.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                if value != 0.0 {
                    values.push(value);
                    rows.push(i);
                    cols.push(j);
                }
            }
        }
        let n = case.rhs.len();
        let csr = CooMatrix::from_triplets(Shape2D::new(n, n), values, rows, cols, true)
            .unwrap()
            .to_csr()
            .unwrap();
        let operator =
            FunctionOperator::new(Shape2D::new(n, n), |x: &[f64]| Ok(product(&case.matrix, x)));
        let expected_x: Vec<f64> = expected["x"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap())
            .collect();
        assert_eq!(expected_x.len(), n);
        let scale = expected_x.iter().map(|x| x.abs()).fold(0.0, f64::max);
        assert!(scale > 0.0);
        for (storage, a) in [
            ("csr", &csr as &dyn LinearOperator),
            ("matrix_free", &operator as &dyn LinearOperator),
        ] {
            let result = cg_preconditioned(
                a,
                &case.rhs,
                |residual| {
                    Ok(residual
                        .iter()
                        .zip(&case.inverse_diagonal)
                        .map(|(r, m)| r * m)
                        .collect())
                },
                case.initial.as_deref(),
                IterativeSolveOptions {
                    tol: 1e-10,
                    max_iter: Some(case.max_iter),
                    ..IterativeSolveOptions::default()
                },
            )
            .unwrap();
            assert_eq!(
                result.converged,
                expected["info"].as_i64() == Some(0),
                "{} {storage}: {result:?}",
                case.id
            );
            let relative_error = result
                .solution
                .iter()
                .zip(&expected_x)
                .map(|(actual, expected)| ((actual - expected) / scale).abs())
                .fold(0.0, f64::max);
            assert!(
                relative_error <= SOLUTION_RTOL,
                "{} {storage}: relative solution error {relative_error:e}",
                case.id
            );
            if case.id == "exact_diagonal_preconditioner" {
                assert_eq!(result.iterations, 1);
            }
            if case.id == "exact_initial" {
                assert_eq!(result.iterations, 0);
            }
            eprintln!(
                "{} {storage}: rust iterations {}, scipy {}, residual {:e}, solution error {:e}",
                case.id,
                result.iterations,
                expected["iterations"],
                result.residual_norm,
                relative_error
            );
            compared += 1;
        }
    }
    assert_eq!(compared, cases.len() * 2);
    assert_eq!(compared, 16);
    eprintln!("CG compared {compared} cases against live SciPy");
}
