#![forbid(unsafe_code)]
//! Differential coverage of LSMR's stopping controls and complete diagnostics.
//!
//! The oracle is SciPy 1.17.1 / NumPy 2.4.3. Every case compares dense, CSR,
//! and matrix-free Rust operators. Counts are asserted exactly; an oracle
//! failure cannot silently turn an individual case into a pass. Set
//! FSCI_REQUIRE_SCIPY_ORACLE=1 to require the oracle and SCIPY_PYTHON to select it.

use std::io::Write;
use std::process::{Command, Stdio};

use fsci_sparse::{
    CooMatrix, DenseLinearOperator, FormatConvertible, FunctionOperator, LinearOperator,
    LsmrOptions, LsmrResult, Shape2D, lsmr_with_options,
};
use serde_json::{Value, json};

// Small conditioned systems should agree to much better than the solver's
// default 1e-6 tolerances. The absolute allowance covers residual estimates
// rounding to different numbers close to zero after Krylov exhaustion.
const VALUE_RTOL: f64 = 5.0e-10;
const VALUE_ATOL: f64 = 2.0e-12;

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
}

impl Case {
    fn new(id: &'static str, a: &[&[f64]], b: &[f64]) -> Self {
        Self {
            id,
            a: a.iter().map(|row| row.to_vec()).collect(),
            b: b.to_vec(),
            atol: 1.0e-10,
            btol: 1.0e-10,
            conlim: 1.0e8,
            max_iter: None,
            damp: 0.0,
            x0: None,
        }
    }

    fn options(&self) -> LsmrOptions<'_> {
        LsmrOptions {
            atol: self.atol,
            btol: self.btol,
            conlim: self.conlim,
            max_iter: self.max_iter,
            damp: self.damp,
            x0: self.x0.as_deref(),
            ..LsmrOptions::default()
        }
    }

    fn query(&self) -> Value {
        json!({
            "id": self.id,
            "a": self.a,
            "b": self.b,
            "atol": self.atol,
            "btol": self.btol,
            "conlim": self.conlim,
            "maxiter": self.max_iter,
            "damp": self.damp,
            "x0": self.x0,
        })
    }
}

fn cases() -> Vec<Case> {
    let diagonal: &[&[f64]] = &[&[1.0, 0.0, 0.0], &[0.0, 0.1, 0.0], &[0.0, 0.0, 0.01]];
    let identity: &[&[f64]] = &[&[1.0, 0.0], &[0.0, 1.0]];
    let mut cases = vec![
        Case::new(
            "rectangular_consistent",
            &[&[3.0, 1.0], &[1.0, 2.0], &[2.0, -1.0]],
            &[5.0, 5.0, 0.0],
        ),
        Case::new(
            "rectangular_inconsistent",
            &[&[1.0, 0.0], &[0.0, 1.0], &[1.0, 1.0]],
            &[1.0, 2.0, 4.0],
        ),
        Case::new(
            "wide_minimum_norm",
            &[&[1.0, 0.0, 1.0], &[0.0, 1.0, 1.0]],
            &[1.0, 3.0],
        ),
        Case::new(
            "rank_deficient_inconsistent",
            &[&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0], &[1.0, 0.0, 1.0]],
            &[1.0, -1.0, 2.0],
        ),
    ];
    let mut damped = Case::new("damped_warm_start", identity, &[3.0, -2.0]);
    damped.damp = 2.0;
    damped.x0 = Some(vec![1.0, 4.0]);
    cases.push(damped);
    let mut negative_damp = Case::new("negative_damping", identity, &[3.0, -2.0]);
    negative_damp.damp = -2.0;
    cases.push(negative_damp);
    let mut stationary = Case::new("stationary_warm_start", identity, &[1.0, 2.0]);
    stationary.x0 = Some(vec![1.0, 2.0]);
    cases.push(stationary);
    let mut nullspace = Case::new("nullspace_zero_rhs", &[&[1.0, 0.0]], &[0.0]);
    nullspace.x0 = Some(vec![0.0, 5.0]);
    cases.push(nullspace);
    let mut zero_rhs = Case::new("zero_rhs_nonstationary_start", identity, &[0.0, 0.0]);
    zero_rhs.x0 = Some(vec![2.0, -3.0]);
    cases.push(zero_rhs);
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
    let mut no_budget = Case::new("zero_iteration_budget", diagonal, &[1.0; 3]);
    no_budget.max_iter = Some(0);
    cases.push(no_budget);
    let mut atol_only = Case::new("atol_only_inconsistent", &[&[1.0], &[0.0]], &[1.0, 3.0]);
    atol_only.btol = 0.0;
    cases.push(atol_only);
    let mut btol_only = Case::new("btol_overrides_iteration_limit", &[&[1.0], &[0.0]], &[1.0, 3.0]);
    btol_only.atol = 0.0;
    btol_only.btol = 0.99;
    cases.push(btol_only);
    cases.push(Case::new("zero_operator", &[&[0.0, 0.0]], &[1.0]));
    cases
}

fn oracle(cases: &[Case]) -> Option<Value> {
    let script = r#"
import json
import sys
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsmr
assert scipy.__version__ == "1.17.1", scipy.__version__
assert np.__version__ == "2.4.3", np.__version__
results = []
for c in json.load(sys.stdin):
    out = lsmr(csr_matrix(c["a"]), np.asarray(c["b"], dtype=float),
               atol=c["atol"], btol=c["btol"], conlim=c["conlim"],
               maxiter=c["maxiter"], damp=c["damp"],
               x0=None if c["x0"] is None else np.asarray(c["x0"], dtype=float))
    x, istop, itn, normr, normar, norma, conda, normx = out
    assert np.isfinite(x).all() and np.isfinite(out[3:]).all(), c["id"]
    results.append({"id": c["id"], "x": x.tolist(), "istop": int(istop),
                    "iterations": int(itn),
                    "diagnostics": [float(v) for v in out[3:]]})
print(json.dumps(results, allow_nan=False))
"#;
    let required = std::env::var_os("FSCI_REQUIRE_SCIPY_ORACLE").is_some();
    let python = std::env::var("SCIPY_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let mut command = Command::new(python);
    let spawned = command
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
            eprintln!("skipping optional SciPy LSMR oracle: {error}");
            return None;
        }
    };
    let query: Vec<Value> = cases.iter().map(Case::query).collect();
    let encoded = serde_json::to_vec(&query).expect("serialize oracle input");
    let write_result = child.stdin.take().expect("oracle stdin").write_all(&encoded);
    let output = child.wait_with_output().expect("wait for LSMR oracle");
    if !output.status.success() || write_result.is_err() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !required,
            "required SciPy LSMR oracle failed: {stderr}; input={write_result:?}"
        );
        eprintln!("skipping optional SciPy LSMR oracle: {stderr}");
        return None;
    }
    Some(serde_json::from_slice(&output.stdout).expect("valid SciPy LSMR results"))
}

fn close(actual: f64, expected: f64) -> bool {
    actual.is_finite()
        && expected.is_finite()
        && (actual - expected).abs() <= VALUE_ATOL + VALUE_RTOL * expected.abs()
}

fn compare(result: &LsmrResult, expected: &Value) -> bool {
    let x = expected["x"].as_array().expect("oracle solution");
    let diagnostics = expected["diagnostics"]
        .as_array()
        .expect("oracle diagnostics");
    assert_eq!(diagnostics.len(), 5);
    result.istop == expected["istop"].as_i64().expect("oracle stop code") as i32
        && result.iterations == expected["iterations"].as_u64().expect("oracle iterations") as usize
        && result.solution.len() == x.len()
        && result
            .solution
            .iter()
            .zip(x)
            .all(|(&a, b)| close(a, b.as_f64().expect("finite solution")))
        && [
            result.normr,
            result.normar,
            result.norma,
            result.conda,
            result.normx,
        ]
            .iter()
            .zip(diagnostics)
            .all(|(&a, b)| close(a, b.as_f64().expect("finite diagnostic")))
}

#[test]
fn lsmr_options_scipy() {
    let cases = cases();
    let Some(expected) = oracle(&cases) else {
        return;
    };
    let expected = expected.as_array().expect("oracle result array");
    assert_eq!(expected.len(), cases.len());
    let mut compared = 0;
    let mut failures = Vec::new();
    for (case, expected) in cases.iter().zip(expected) {
        assert_eq!(expected["id"].as_str(), Some(case.id));
        let dense = DenseLinearOperator::from_rows(&case.a).expect("dense operator");
        let mut data = Vec::new();
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        for (i, row) in case.a.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                if value != 0.0 {
                    data.push(value);
                    rows.push(i);
                    cols.push(j);
                }
            }
        }
        let sparse = CooMatrix::from_triplets(
            Shape2D::new(case.a.len(), case.a[0].len()),
            data,
            rows,
            cols,
            false,
        )
        .expect("COO matrix")
        .to_csr()
        .expect("CSR matrix");
        let matrix_free = FunctionOperator::new(dense.shape(), |x| dense.matvec(x))
            .with_rmatvec(|x| dense.rmatvec(x));
        for (kind, operator) in [
            ("dense", &dense as &dyn LinearOperator),
            ("csr", &sparse as &dyn LinearOperator),
            ("matrix_free", &matrix_free as &dyn LinearOperator),
        ] {
            let result = lsmr_with_options(operator, &case.b, case.options())
                .unwrap_or_else(|error| panic!("{} / {kind}: {error}", case.id));
            let pass = compare(&result, expected);
            compared += 1;
            eprintln!(
                "{} / {kind}: compared=1 istop={} iterations={} pass={pass}",
                case.id, result.istop, result.iterations
            );
            if !pass {
                failures.push(format!(
                    "{} / {kind}: Rust={result:?}, SciPy={expected}",
                    case.id
                ));
            }
        }
    }
    assert_eq!(compared, 3 * cases.len());
    assert!(compared > 0, "a zero-comparison run cannot pass");
    assert!(
        failures.is_empty(),
        "LSMR differential failures: {failures:#?}"
    );
    eprintln!("LSMR compared {compared} operator/case pairs against SciPy 1.17.1 / NumPy 2.4.3");
}
