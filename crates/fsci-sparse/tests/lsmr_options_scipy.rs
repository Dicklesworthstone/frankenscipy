#![forbid(unsafe_code)]
//! Differential coverage of LSMR's stopping controls and complete diagnostics.
//!
//! The oracle is SciPy 1.17.1 / NumPy 2.4.3. Every case compares dense, CSR,
//! and matrix-free Rust operators. Counts are asserted exactly; an oracle
//! failure cannot silently turn an individual case into a pass. Set
//! FSCI_REQUIRE_SCIPY_ORACLE=1 to require the oracle and SCIPY_PYTHON to select it.

use std::io::Write;
use std::process::{Command, Stdio};

use fsci_runtime::RuntimeMode;
use fsci_sparse::{
    CooMatrix, CsrMatrix, DenseLinearOperator, FormatConvertible, FunctionOperator,
    IterativeSolveOptions, LinearOperator, LsmrOptions, LsmrResult, Shape2D, lsmr_regularized,
    lsmr_with_options, lsqr_regularized,
};
use serde_json::{Value, json};

// Small conditioned systems should agree to much better than the solver's
// default 1e-6 tolerances. The absolute allowance covers residual estimates
// rounding to different numbers close to zero after Krylov exhaustion.
const VALUE_RTOL: f64 = 5.0e-10;
const VALUE_ATOL: f64 = 2.0e-12;

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
    let mut btol_only = Case::new(
        "btol_overrides_iteration_limit",
        &[&[1.0], &[0.0]],
        &[1.0, 3.0],
    );
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
    run_oracle(script, cases)
}

fn run_oracle(script: &str, cases: &[Case]) -> Option<Value> {
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
            eprintln!("skipping optional SciPy least-squares oracle: {error}");
            return None;
        }
    };
    let query: Vec<Value> = cases.iter().map(Case::query).collect();
    let encoded = serde_json::to_vec(&query).expect("serialize oracle input");
    let write_result = child
        .stdin
        .take()
        .expect("oracle stdin")
        .write_all(&encoded);
    let output = child
        .wait_with_output()
        .expect("wait for least-squares oracle");
    if !output.status.success() || write_result.is_err() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !required,
            "required SciPy least-squares oracle failed: {stderr}; input={write_result:?}"
        );
        eprintln!("skipping optional SciPy least-squares oracle: {stderr}");
        return None;
    }
    Some(serde_json::from_slice(&output.stdout).expect("valid SciPy least-squares results"))
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
    .expect("COO matrix")
    .to_csr()
    .expect("CSR matrix")
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
        let sparse = sparse_matrix(&case.a);
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

#[test]
fn signed_regularization_and_stationary_warm_starts_match_their_objective() {
    let identity: &[&[f64]] = &[&[1.0, 0.0], &[0.0, 1.0]];
    let rank_deficient: &[&[f64]] = &[&[1.0, 2.0], &[2.0, 4.0], &[0.0, 0.0]];
    let cold = Case::new("identity_cold", identity, &[3.0, -2.0]);
    let mut warm = Case::new("identity_warm", identity, &[3.0, -2.0]);
    warm.x0 = Some(vec![1.0, 4.0]);
    let inconsistent = Case::new("rank_deficient_cold", rank_deficient, &[1.0, 3.0, 5.0]);
    let mut inconsistent_warm = Case::new("rank_deficient_warm", rank_deficient, &[1.0, 3.0, 5.0]);
    inconsistent_warm.x0 = Some(vec![1.0, -1.0]);
    let mut cases = Vec::new();
    let mut analytical = Vec::new();
    for (base, solution) in [
        (cold, vec![0.6, -0.4]),
        (warm, vec![1.4, 2.8]),
        (inconsistent, vec![7.0 / 29.0, 14.0 / 29.0]),
        (inconsistent_warm, vec![41.0 / 29.0, -5.0 / 29.0]),
    ] {
        for damping in [-2.0, 2.0] {
            let mut case = base.clone();
            case.damp = damping;
            case.max_iter = Some(20);
            cases.push(case);
            analytical.push(solution.clone());
        }
    }
    for damping in [-2.0, 0.0, 2.0] {
        let mut case = Case::new("nullspace_stationary_warm", &[&[1.0, 0.0]], &[0.0]);
        case.x0 = Some(vec![0.0, 5.0]);
        case.damp = damping;
        case.max_iter = Some(20);
        cases.push(case);
        analytical.push(vec![0.0, 5.0]);
    }
    let mut zero = Case::new("zero_rhs_cold", &[&[1.0, 0.0]], &[0.0]);
    zero.damp = 2.0;
    zero.max_iter = Some(20);
    cases.push(zero);
    analytical.push(vec![0.0, 0.0]);

    // SciPy LSQR ignores a negative damp in its rotation, despite damping
    // representing a squared penalty. FrankenSciPy explicitly accepts either
    // sign with the same objective, so LSQR's oracle uses abs(damp). This is a
    // deliberate documented-contract difference, not raw negative-damp parity.
    let script = r#"
import json
import sys
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr, lsmr
assert scipy.__version__ == "1.17.1", scipy.__version__
assert np.__version__ == "2.4.3", np.__version__
results = []
for c in json.load(sys.stdin):
    a, b = csr_matrix(c["a"]), np.asarray(c["b"], dtype=float)
    x0 = None if c["x0"] is None else np.asarray(c["x0"], dtype=float)
    qr = lsqr(a, b, damp=abs(c["damp"]), x0=x0,
              atol=1e-12, btol=1e-12, iter_lim=c["maxiter"])
    mr = lsmr(a, b, damp=c["damp"], x0=x0,
              atol=1e-12, btol=1e-12, maxiter=c["maxiter"])
    assert np.isfinite(qr[0]).all() and np.isfinite(mr[0]).all(), c["id"]
    assert qr[1] in (0, 1, 2, 4, 5) and mr[1] in (0, 1, 2, 4, 5), c["id"]
    results.append({"id": c["id"], "lsqr": qr[0].tolist(), "lsmr": mr[0].tolist()})
print(json.dumps(results, allow_nan=False))
"#;
    let Some(expected) = run_oracle(script, &cases) else {
        return;
    };
    let expected = expected.as_array().expect("regularized oracle results");
    assert_eq!(cases.len(), 12);
    assert_eq!(expected.len(), cases.len());
    assert_eq!(analytical.len(), cases.len());
    let controls = IterativeSolveOptions {
        tol: 1e-12,
        max_iter: Some(20),
        ..IterativeSolveOptions::default()
    };
    let mut compared = 0;
    for ((case, expected), analytical) in cases.iter().zip(expected).zip(analytical) {
        assert_eq!(expected["id"].as_str(), Some(case.id));
        let dense = DenseLinearOperator::from_rows(&case.a).expect("dense operator");
        let sparse = sparse_matrix(&case.a);
        let matrix_free = FunctionOperator::new(dense.shape(), |x| dense.matvec(x))
            .with_rmatvec(|x| dense.rmatvec(x));
        for (kind, operator) in [
            ("dense", &dense as &dyn LinearOperator),
            ("csr", &sparse as &dyn LinearOperator),
            ("matrix_free", &matrix_free as &dyn LinearOperator),
        ] {
            for (name, result) in [
                (
                    "lsqr",
                    lsqr_regularized(operator, &case.b, case.damp, case.x0.as_deref(), controls),
                ),
                (
                    "lsmr",
                    lsmr_regularized(operator, &case.b, case.damp, case.x0.as_deref(), controls),
                ),
            ] {
                let result = result.unwrap_or_else(|error| {
                    panic!(
                        "{} / damp={} / {kind} / {name}: {error}",
                        case.id, case.damp
                    )
                });
                assert!(
                    result.converged,
                    "{} / {kind} / {name}: {result:?}",
                    case.id
                );
                let reference = expected[name].as_array().expect("oracle solution");
                assert_eq!(result.solution.len(), analytical.len());
                assert_eq!(reference.len(), analytical.len());
                for ((&actual, oracle), &exact) in
                    result.solution.iter().zip(reference).zip(&analytical)
                {
                    let oracle = oracle.as_f64().expect("finite oracle component");
                    assert!(
                        close(actual, exact) && close(actual, oracle) && close(oracle, exact),
                        "{} / damp={} / {kind} / {name}: actual={actual:e}, oracle={oracle:e}, analytical={exact:e}",
                        case.id,
                        case.damp
                    );
                }
                if case.b.iter().all(|&value| value == 0.0) {
                    assert_eq!(result.iterations, 0);
                    assert_eq!(result.residual_norm, 0.0);
                } else {
                    let residual =
                        case.a
                            .iter()
                            .zip(&case.b)
                            .fold(0.0_f64, |norm, (row, &target)| {
                                let image: f64 =
                                    row.iter().zip(&result.solution).map(|(a, x)| a * x).sum();
                                norm.hypot(image - target)
                            });
                    let rhs_norm = case
                        .b
                        .iter()
                        .fold(0.0_f64, |norm, &value| norm.hypot(value));
                    assert!(close(result.residual_norm, residual / rhs_norm));
                }
                compared += 1;
                eprintln!(
                    "{} / damp={} / {kind} / {name}: compared=1 analytical_and_scipy=true",
                    case.id, case.damp
                );
            }
        }
    }
    assert_eq!(
        compared, 72,
        "all operator, sign, warm-start and solver cases must run"
    );
    eprintln!(
        "Regularized LSQR/LSMR compared {compared} cases against analytical solutions and SciPy"
    );
}

#[test]
fn lsmr_condition_limit_preserves_strict_parity_and_hardened_scale_invariance() {
    // SciPy seeds the largest QR diagonal with an unscaled 1.0. Strict mode
    // preserves its externally visible condition estimate and stopping point.
    // Hardened mode excludes that placeholder: rescaling A and b together
    // must not turn the same condition limit into an earlier termination.
    let scales = [1e-4, 1.0, 1e4];
    let mut cases = Vec::new();
    for (id, scale) in [
        ("condition_scaled_small", scales[0]),
        ("condition_scaled_unit", scales[1]),
        ("condition_scaled_large", scales[2]),
    ] {
        let mut case = Case::new(
            id,
            &[
                &[scale, 0.0, 0.0],
                &[0.0, 0.1 * scale, 0.0],
                &[0.0, 0.0, 0.01 * scale],
            ],
            &[scale; 3],
        );
        case.atol = 0.0;
        case.btol = 0.0;
        case.conlim = 2.0;
        case.max_iter = Some(20);
        cases.push(case);
    }
    let Some(expected) = oracle(&cases) else {
        return;
    };
    let expected = expected.as_array().expect("condition-limit oracle results");
    assert_eq!(expected.len(), 3);
    let mut hardened_reference: Option<(f64, LsmrResult)> = None;
    let mut compared = 0;
    let mut scale_sensitive_controls = 0;
    for ((case, expected), scale) in cases.iter().zip(expected).zip(scales) {
        let dense = DenseLinearOperator::from_rows(&case.a).expect("dense operator");
        let sparse = sparse_matrix(&case.a);
        let matrix_free = FunctionOperator::new(dense.shape(), |x| dense.matvec(x))
            .with_rmatvec(|x| dense.rmatvec(x));
        for (kind, operator) in [
            ("dense", &dense as &dyn LinearOperator),
            ("csr", &sparse as &dyn LinearOperator),
            ("matrix_free", &matrix_free as &dyn LinearOperator),
        ] {
            let strict = lsmr_with_options(operator, &case.b, case.options())
                .unwrap_or_else(|error| panic!("{} / {kind} / strict: {error}", case.id));
            assert!(
                compare(&strict, expected),
                "{} / {kind}: strict={strict:?}, SciPy={expected}",
                case.id
            );
            compared += 1;
            let hardened = lsmr_with_options(
                operator,
                &case.b,
                LsmrOptions {
                    mode: RuntimeMode::Hardened,
                    ..case.options()
                },
            )
            .unwrap_or_else(|error| panic!("{} / {kind} / hardened: {error}", case.id));
            assert_eq!(hardened.istop, 3, "condition limit, not convergence");
            assert_eq!(hardened.iterations, 2);
            assert!(hardened.conda > 2.0 && hardened.conda < 11.0);

            // Recompute both residuals directly from the unscaled analytical
            // diagonal, independently of the bidiagonalization estimates.
            for result in [&strict, &hardened] {
                let mut residual_norm = 0.0_f64;
                let mut normal_residual_norm = 0.0_f64;
                for (&diagonal, &x) in [1.0, 0.1, 0.01].iter().zip(&result.solution) {
                    let residual = diagonal * x - 1.0;
                    residual_norm = residual_norm.hypot(residual);
                    normal_residual_norm = normal_residual_norm.hypot(diagonal * residual);
                }
                assert!(close(result.normr / scale, residual_norm));
                assert!(close(result.normar / scale / scale, normal_residual_norm));
                assert!(residual_norm > 0.9, "a limited solve is still unfinished");
            }
            if let Some((reference_scale, reference)) = &hardened_reference {
                assert_eq!(hardened.istop, reference.istop);
                assert_eq!(hardened.iterations, reference.iterations);
                assert!(close(hardened.conda, reference.conda));
                assert!(close(
                    hardened.norma / scale,
                    reference.norma / reference_scale
                ));
                for (&actual, &expected) in hardened.solution.iter().zip(&reference.solution) {
                    assert!(
                        close(actual, expected),
                        "scale changed the returned iterate"
                    );
                }
            } else {
                hardened_reference = Some((scale, hardened.clone()));
            }
            compared += 1;
            if scale == scales[0] {
                // This control proves the fixture sees the seed, rather than
                // checking two paths on which the seed is already irrelevant.
                assert_eq!(strict.iterations, 1);
                assert!(strict.conda > 100.0 * hardened.conda);
                assert!((strict.solution[1] - hardened.solution[1]).abs() > 9.0);
                scale_sensitive_controls += 1;
            }
            if scale == scales[2] {
                assert!(close(strict.conda, hardened.conda));
            }
            eprintln!(
                "{} / {kind}: strict_conda={:e} strict_iterations={} hardened_conda={:e} hardened_iterations={}",
                case.id, strict.conda, strict.iterations, hardened.conda, hardened.iterations
            );
        }
    }
    assert_eq!(compared, 18);
    assert_eq!(scale_sensitive_controls, 3);
    eprintln!("LSMR condition policy compared 9 strict SciPy cases and 9 hardened invariants");
}
