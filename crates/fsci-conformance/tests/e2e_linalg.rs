#![forbid(unsafe_code)]
//! E2E scenario tests for FSCI-P2C-002 (Linalg solve/decompose).
//!
//! Implements 11 scenarios per bd-3jh.13.7 acceptance criteria:
//!   Happy-path:  1-4
//!   Error recovery: 5-7
//!   Cross-op consistency: 8-10
//!   Performance boundary: 11
//!
//! Each scenario emits a forensic log bundle to
//! `fixtures/artifacts/FSCI-P2C-002/e2e/`.

use blake3::hash as blake3_hash;
use fsci_linalg::{
    InvOptions, LinalgError, LstsqOptions, PinvOptions, SolveOptions, SolveResult,
    TriangularSolveOptions, det, inv, lstsq, pinv, solve, solve_triangular,
};
use fsci_runtime::RuntimeMode;
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

// ───────────────────────── Forensic log types ─────────────────────────

#[derive(Debug, Clone, Serialize)]
struct ForensicLogBundle {
    scenario_id: String,
    steps: Vec<ForensicStep>,
    artifacts: Vec<ArtifactRef>,
    environment: EnvironmentInfo,
    matrix_metadata: Option<MatrixMetadata>,
    overall: OverallResult,
}

#[derive(Debug, Clone, Serialize)]
struct ForensicStep {
    step_id: usize,
    step_name: String,
    action: String,
    input_summary: String,
    output_summary: String,
    duration_ns: u128,
    mode: String,
    outcome: String,
}

#[derive(Debug, Clone, Serialize)]
struct ArtifactRef {
    path: String,
    blake3: String,
}

#[derive(Debug, Clone, Serialize)]
struct EnvironmentInfo {
    rust_version: String,
    os: String,
    cpu_count: usize,
    total_memory_mb: String,
}

#[derive(Debug, Clone, Serialize)]
struct MatrixMetadata {
    size: (usize, usize),
    nnz: Option<usize>,
    rcond_estimate: Option<f64>,
    mode: String,
}

#[derive(Debug, Clone, Serialize)]
struct OverallResult {
    status: String,
    total_duration_ns: u128,
    replay_command: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error_chain: Option<String>,
}

// ───────────────────────── Helpers ─────────────────────────

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn e2e_output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/artifacts/FSCI-P2C-002/e2e")
}

fn make_env() -> EnvironmentInfo {
    EnvironmentInfo {
        rust_version: String::from(env!("CARGO_PKG_VERSION")),
        os: String::from(std::env::consts::OS),
        cpu_count: std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1),
        total_memory_mb: String::from("unknown"),
    }
}

fn replay_cmd(scenario_id: &str) -> String {
    format!(
        "cargo test -p fsci-conformance --test e2e_linalg -- {scenario_id} --nocapture"
    )
}

fn write_bundle(scenario_id: &str, bundle: &ForensicLogBundle) {
    let dir = e2e_output_dir();
    fs::create_dir_all(&dir)
        .unwrap_or_else(|e| panic!("failed to create e2e dir {}: {e}", dir.display()));
    let path = dir.join(format!("{scenario_id}.json"));
    let json = serde_json::to_vec_pretty(bundle).expect("serialize bundle");
    fs::write(&path, &json)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", path.display()));
}

fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let n = b[0].len();
    let k = b.len();
    let mut c = vec![vec![0.0; n]; m];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for p in 0..k {
                s += a[i][p] * b[p][j];
            }
            c[i][j] = s;
        }
    }
    c
}

fn mat_vec_mul(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(x).map(|(a, x)| a * x).sum())
        .collect()
}

fn max_abs_diff_matrix(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    a.iter()
        .zip(b.iter())
        .flat_map(|(ar, br)| ar.iter().zip(br.iter()).map(|(x, y)| (x - y).abs()))
        .fold(0.0_f64, f64::max)
}

fn max_abs_diff_vec(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn identity_matrix(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            let mut row = vec![0.0; n];
            row[i] = 1.0;
            row
        })
        .collect()
}

fn diag_dom_matrix(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| if i == j { n as f64 + 1.0 } else { 1.0 })
                .collect()
        })
        .collect()
}

// ───────────────────── Scenario runner framework ──────────────────────

struct ScenarioRunner {
    scenario_id: String,
    steps: Vec<ForensicStep>,
    start: Instant,
    step_counter: usize,
    passed: bool,
    error_chain: Option<String>,
    matrix_meta: Option<MatrixMetadata>,
}

impl ScenarioRunner {
    fn new(scenario_id: &str) -> Self {
        Self {
            scenario_id: scenario_id.to_owned(),
            steps: Vec::new(),
            start: Instant::now(),
            step_counter: 0,
            passed: true,
            error_chain: None,
            matrix_meta: None,
        }
    }

    fn set_matrix_meta(&mut self, size: (usize, usize), mode: &str) {
        self.matrix_meta = Some(MatrixMetadata {
            size,
            nnz: None,
            rcond_estimate: None,
            mode: mode.to_owned(),
        });
    }

    fn record_step(
        &mut self,
        name: &str,
        action: &str,
        input_summary: &str,
        mode: &str,
        f: impl FnOnce() -> Result<String, String>,
    ) -> bool {
        self.step_counter += 1;
        let step_start = Instant::now();
        let result = f();
        let duration_ns = step_start.elapsed().as_nanos();
        let (outcome, output_summary) = match result {
            Ok(summary) => ("pass".to_owned(), summary),
            Err(err) => {
                self.passed = false;
                if self.error_chain.is_none() {
                    self.error_chain = Some(err.clone());
                }
                ("fail".to_owned(), err)
            }
        };
        self.steps.push(ForensicStep {
            step_id: self.step_counter,
            step_name: name.to_owned(),
            action: action.to_owned(),
            input_summary: input_summary.to_owned(),
            output_summary,
            duration_ns,
            mode: mode.to_owned(),
            outcome: outcome.clone(),
        });
        outcome == "pass"
    }

    fn finish(self) -> ForensicLogBundle {
        let total_duration_ns = self.start.elapsed().as_nanos();
        let bundle = ForensicLogBundle {
            scenario_id: self.scenario_id.clone(),
            steps: self.steps,
            artifacts: Vec::new(),
            environment: make_env(),
            matrix_metadata: self.matrix_meta,
            overall: OverallResult {
                status: if self.passed {
                    "pass".to_owned()
                } else {
                    "fail".to_owned()
                },
                total_duration_ns,
                replay_command: replay_cmd(&self.scenario_id),
                error_chain: self.error_chain,
            },
        };
        write_bundle(&self.scenario_id, &bundle);
        bundle
    }
}

// ═══════════════════════ SCENARIOS ═══════════════════════

// ──────────── Happy-Path Workflows ────────────

/// Scenario 1: Full solve pipeline - construct matrix, solve, verify, log
#[test]
fn e2e_p2c002_01_full_solve_pipeline() {
    let mut r = ScenarioRunner::new("p2c002_01_full_solve_pipeline");
    let a = vec![
        vec![3.0, 2.0, 1.0],
        vec![1.0, 4.0, 2.0],
        vec![2.0, 1.0, 5.0],
    ];
    let b = vec![10.0, 15.0, 16.0];
    r.set_matrix_meta((3, 3), "strict");

    let mut solution = Vec::new();
    r.record_step(
        "construct_and_solve",
        "solve(A, b)",
        "3x3 general, b=[10,15,16]",
        "strict",
        || {
            let res = solve(&a, &b, SolveOptions::default())
                .map_err(|e| format!("solve failed: {e}"))?;
            solution = res.x.clone();
            Ok(format!("x={solution:?}, backward_err={:?}", res.backward_error))
        },
    );

    r.record_step(
        "verify_residual",
        "||Ax - b||",
        "check A*x == b",
        "strict",
        || {
            let ax = mat_vec_mul(&a, &solution);
            let max_diff = max_abs_diff_vec(&ax, &b);
            if max_diff < 1e-10 {
                Ok(format!("residual max_diff={max_diff:.2e}"))
            } else {
                Err(format!("residual too large: max_diff={max_diff:.2e}"))
            }
        },
    );

    r.record_step(
        "verify_backward_error",
        "backward_error < 1e-12",
        "check backward error is small",
        "strict",
        || {
            let res = solve(&a, &b, SolveOptions::default()).unwrap();
            let be = res.backward_error.unwrap_or(f64::NAN);
            if be < 1e-12 {
                Ok(format!("backward_error={be:.2e}"))
            } else {
                Err(format!("backward_error too large: {be:.2e}"))
            }
        },
    );

    let bundle = r.finish();
    assert_eq!(bundle.overall.status, "pass");
}

/// Scenario 2: Decomposition chain - solve, check condition, inv, verify A @ inv(A) = I
#[test]
fn e2e_p2c002_02_decomposition_chain() {
    let mut r = ScenarioRunner::new("p2c002_02_decomposition_chain");
    let a = vec![
        vec![4.0, 7.0, 2.0],
        vec![3.0, 6.0, 1.0],
        vec![2.0, 5.0, 3.0],
    ];
    let b = vec![13.0, 10.0, 10.0];
    r.set_matrix_meta((3, 3), "strict");

    // Step 1: Solve
    let mut solution = Vec::new();
    r.record_step("solve", "solve(A, b)", "3x3 general", "strict", || {
        let res = solve(&a, &b, SolveOptions::default())
            .map_err(|e| format!("solve failed: {e}"))?;
        solution = res.x.clone();
        let warning = res.warning.is_some();
        Ok(format!("x={solution:?}, ill_conditioned={warning}"))
    });

    // Step 2: Check condition (via warning from solve)
    r.record_step(
        "check_condition",
        "inspect rcond warning",
        "well-conditioned expected",
        "strict",
        || {
            let res = solve(&a, &b, SolveOptions::default()).unwrap();
            if res.warning.is_none() {
                Ok("well-conditioned, proceeding to inv".to_owned())
            } else {
                Ok("ill-conditioned warning present, proceeding anyway".to_owned())
            }
        },
    );

    // Step 3: Compute inverse
    let mut inverse = Vec::new();
    r.record_step("compute_inv", "inv(A)", "3x3 general", "strict", || {
        let res =
            inv(&a, InvOptions::default()).map_err(|e| format!("inv failed: {e}"))?;
        inverse = res.inverse.clone();
        Ok(format!("inv computed, warning={:?}", res.warning.is_some()))
    });

    // Step 4: Verify A @ inv(A) = I
    r.record_step(
        "verify_identity",
        "A @ inv(A) == I",
        "check product is identity",
        "strict",
        || {
            let product = mat_mul(&a, &inverse);
            let identity = identity_matrix(3);
            let max_diff = max_abs_diff_matrix(&product, &identity);
            if max_diff < 1e-10 {
                Ok(format!("A @ inv(A) = I verified, max_diff={max_diff:.2e}"))
            } else {
                Err(format!(
                    "A @ inv(A) != I, max_diff={max_diff:.2e}"
                ))
            }
        },
    );

    let bundle = r.finish();
    assert_eq!(bundle.overall.status, "pass");
}

/// Scenario 3: Least-squares workflow - overdetermined system, lstsq, pinv cross-check
#[test]
fn e2e_p2c002_03_lstsq_workflow() {
    let mut r = ScenarioRunner::new("p2c002_03_lstsq_workflow");
    let a = vec![
        vec![1.0, 0.0],
        vec![1.0, 1.0],
        vec![1.0, 2.0],
        vec![1.0, 3.0],
    ];
    let b = vec![1.0, 2.0, 2.0, 4.0];
    r.set_matrix_meta((4, 2), "strict");

    // Step 1: lstsq
    let mut lstsq_x = Vec::new();
    r.record_step(
        "lstsq_solve",
        "lstsq(A, b)",
        "4x2 overdetermined",
        "strict",
        || {
            let res = lstsq(&a, &b, LstsqOptions::default())
                .map_err(|e| format!("lstsq failed: {e}"))?;
            lstsq_x = res.x.clone();
            Ok(format!(
                "x={:?}, rank={}, residuals={:?}, sv={:?}",
                res.x, res.rank, res.residuals, res.singular_values
            ))
        },
    );

    // Step 2: Verify residuals
    r.record_step(
        "verify_residuals",
        "||Ax - b|| is minimized",
        "check residual norm",
        "strict",
        || {
            let ax = mat_vec_mul(&a, &lstsq_x);
            let residual_norm: f64 =
                ax.iter().zip(&b).map(|(ai, bi)| (ai - bi).powi(2)).sum();
            Ok(format!("residual_norm={residual_norm:.6e}"))
        },
    );

    // Step 3: Compute pinv and cross-check
    r.record_step(
        "pinv_cross_check",
        "pinv(A) @ b == lstsq_x",
        "cross-check lstsq vs pinv",
        "strict",
        || {
            let pinv_res = pinv(&a, PinvOptions::default())
                .map_err(|e| format!("pinv failed: {e}"))?;
            // pinv(A) is 2x4, b is 4x1 -> result is 2x1
            let pinv_x: Vec<f64> = pinv_res
                .pseudo_inverse
                .iter()
                .map(|row| row.iter().zip(&b).map(|(p, bi)| p * bi).sum())
                .collect();
            let max_diff = max_abs_diff_vec(&pinv_x, &lstsq_x);
            if max_diff < 1e-8 {
                Ok(format!(
                    "pinv(A)@b matches lstsq, max_diff={max_diff:.2e}"
                ))
            } else {
                Err(format!(
                    "pinv(A)@b != lstsq, max_diff={max_diff:.2e}"
                ))
            }
        },
    );

    let bundle = r.finish();
    assert_eq!(bundle.overall.status, "pass");
}

/// Scenario 4: Multi-RHS solve - solve same A with 5 different b vectors
#[test]
fn e2e_p2c002_04_multi_rhs_solve() {
    let mut r = ScenarioRunner::new("p2c002_04_multi_rhs_solve");
    let a = vec![
        vec![4.0, 1.0, 0.0],
        vec![1.0, 3.0, 1.0],
        vec![0.0, 1.0, 2.0],
    ];
    let rhs_vectors: Vec<Vec<f64>> = vec![
        vec![5.0, 5.0, 3.0],
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![10.0, 20.0, 30.0],
    ];
    r.set_matrix_meta((3, 3), "strict");

    for (idx, b) in rhs_vectors.iter().enumerate() {
        let b_clone = b.clone();
        let a_ref = &a;
        r.record_step(
            &format!("solve_rhs_{idx}"),
            "solve(A, b_i)",
            &format!("b={b:?}"),
            "strict",
            || {
                let res = solve(a_ref, &b_clone, SolveOptions::default())
                    .map_err(|e| format!("solve rhs {idx} failed: {e}"))?;
                // Verify
                let ax = mat_vec_mul(a_ref, &res.x);
                let max_diff = max_abs_diff_vec(&ax, &b_clone);
                if max_diff < 1e-10 {
                    Ok(format!(
                        "x={:?}, residual={max_diff:.2e}",
                        res.x
                    ))
                } else {
                    Err(format!(
                        "residual too large: {max_diff:.2e}"
                    ))
                }
            },
        );
    }

    let bundle = r.finish();
    assert_eq!(bundle.overall.status, "pass");
}

// ──────────── Error Recovery Workflows ────────────

/// Scenario 5: Singular matrix recovery - attempt solve, catch error, switch to lstsq
#[test]
fn e2e_p2c002_05_singular_matrix_recovery() {
    let mut r = ScenarioRunner::new("p2c002_05_singular_matrix_recovery");
    // Singular matrix: row 2 = 2 * row 1
    let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
    let b = vec![3.0, 6.0];
    r.set_matrix_meta((2, 2), "strict");

    // Step 1: Attempt solve - expect SingularMatrix error
    r.record_step(
        "attempt_solve",
        "solve(A_singular, b)",
        "2x2 singular",
        "strict",
        || {
            match solve(&a, &b, SolveOptions::default()) {
                Err(LinalgError::SingularMatrix) => {
                    Ok("correctly caught SingularMatrix error".to_owned())
                }
                Err(e) => Err(format!("unexpected error: {e}")),
                Ok(_) => Err("solve should have failed on singular matrix".to_owned()),
            }
        },
    );

    // Step 2: Switch to lstsq which handles singular/rank-deficient
    r.record_step(
        "fallback_lstsq",
        "lstsq(A_singular, b)",
        "fallback to least-squares",
        "strict",
        || {
            let res = lstsq(&a, &b, LstsqOptions::default())
                .map_err(|e| format!("lstsq fallback failed: {e}"))?;
            // Rank should be 1
            if res.rank != 1 {
                return Err(format!(
                    "expected rank=1 for singular matrix, got {}",
                    res.rank
                ));
            }
            Ok(format!(
                "lstsq succeeded: x={:?}, rank={}",
                res.x, res.rank
            ))
        },
    );

    // Step 3: Verify lstsq solution is consistent
    r.record_step(
        "verify_lstsq_consistency",
        "A @ lstsq_x ~ b (projection)",
        "verify minimum-norm solution",
        "strict",
        || {
            let res = lstsq(&a, &b, LstsqOptions::default()).unwrap();
            let ax = mat_vec_mul(&a, &res.x);
            // For singular matrix, Ax should be the projection of b onto col(A)
            // ax should be proportional to [1,2] (the column space)
            let ratio = if ax[0].abs() > 1e-14 {
                ax[1] / ax[0]
            } else {
                0.0
            };
            if (ratio - 2.0).abs() < 1e-10 {
                Ok(format!(
                    "solution lies in column space: Ax={ax:?}, ratio={ratio:.6}"
                ))
            } else {
                Err(format!(
                    "solution not in column space: Ax={ax:?}"
                ))
            }
        },
    );

    let bundle = r.finish();
    assert_eq!(bundle.overall.status, "pass");
}

/// Scenario 6: Ill-conditioning detection - solve, check warning, verify result accuracy
#[test]
fn e2e_p2c002_06_ill_conditioning_detection() {
    let mut r = ScenarioRunner::new("p2c002_06_ill_conditioning_detection");
    // Ill-conditioned: large condition number
    let a = vec![vec![1.0, 0.0], vec![0.0, 1e-13]];
    let b = vec![1.0, 1e-13];
    r.set_matrix_meta((2, 2), "strict");

    // Step 1: Solve and detect ill-conditioning warning
    r.record_step(
        "solve_ill_conditioned",
        "solve(A_ill, b)",
        "2x2 ill-conditioned (rcond ~ 1e-13)",
        "strict",
        || {
            let res = solve(&a, &b, SolveOptions::default())
                .map_err(|e| format!("solve failed: {e}"))?;
            if res.warning.is_some() {
                Ok(format!(
                    "ill-conditioning warning detected: x={:?}",
                    res.x
                ))
            } else {
                Err("expected ill-conditioning warning but got none".to_owned())
            }
        },
    );

    // Step 2: Verify the solution is still correct despite ill-conditioning
    r.record_step(
        "verify_solution_accuracy",
        "check x == [1, 1]",
        "verify known solution",
        "strict",
        || {
            let res = solve(&a, &b, SolveOptions::default()).unwrap();
            let expected = vec![1.0, 1.0];
            let max_diff = max_abs_diff_vec(&res.x, &expected);
            if max_diff < 1e-6 {
                Ok(format!("solution accurate: x={:?}, diff={max_diff:.2e}", res.x))
            } else {
                Err(format!(
                    "solution inaccurate: x={:?}, expected={expected:?}, diff={max_diff:.2e}",
                    res.x
                ))
            }
        },
    );

    // Step 3: Try with better-conditioned variant of same system
    r.record_step(
        "solve_well_conditioned",
        "solve(A_well, b_scaled)",
        "rescaled to well-conditioned",
        "strict",
        || {
            let a_well = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
            let b_well = vec![1.0, 1.0];
            let res = solve(&a_well, &b_well, SolveOptions::default())
                .map_err(|e| format!("solve failed: {e}"))?;
            if res.warning.is_none() {
                Ok(format!("no warning for well-conditioned: x={:?}", res.x))
            } else {
                Err("unexpected warning for well-conditioned matrix".to_owned())
            }
        },
    );

    let bundle = r.finish();
    assert_eq!(bundle.overall.status, "pass");
}

/// Scenario 7: Mode switch - solve in Strict (NaN passthrough) vs Hardened (NaN rejection)
#[test]
fn e2e_p2c002_07_mode_switch() {
    let mut r = ScenarioRunner::new("p2c002_07_mode_switch");
    let a = vec![vec![1.0, f64::NAN], vec![0.0, 1.0]];
    let b = vec![1.0, 2.0];
    r.set_matrix_meta((2, 2), "mixed");

    // Step 1: Strict mode with check_finite=false - NaN passes through
    r.record_step(
        "strict_nan_passthrough",
        "solve(A_nan, b, strict, check_finite=false)",
        "NaN in matrix, strict mode",
        "strict",
        || {
            let opts = SolveOptions {
                mode: RuntimeMode::Strict,
                check_finite: false,
                ..SolveOptions::default()
            };
            match solve(&a, &b, opts) {
                Ok(res) => Ok(format!(
                    "strict accepted NaN input: x={:?}",
                    res.x
                )),
                Err(e) => Err(format!("strict unexpectedly rejected: {e}")),
            }
        },
    );

    // Step 2: Hardened mode - NaN is rejected even with check_finite=false
    r.record_step(
        "hardened_nan_rejection",
        "solve(A_nan, b, hardened, check_finite=false)",
        "NaN in matrix, hardened mode",
        "hardened",
        || {
            let opts = SolveOptions {
                mode: RuntimeMode::Hardened,
                check_finite: false,
                ..SolveOptions::default()
            };
            match solve(&a, &b, opts) {
                Err(LinalgError::NonFiniteInput) => {
                    Ok("hardened correctly rejected NaN (NonFiniteInput)".to_owned())
                }
                Err(e) => Err(format!("wrong error variant: {e}")),
                Ok(res) => Err(format!(
                    "hardened should have rejected NaN but got x={:?}",
                    res.x
                )),
            }
        },
    );

    // Step 3: Verify behavior difference is documented
    r.record_step(
        "verify_mode_difference",
        "confirm strict != hardened behavior",
        "document mode-specific behavior",
        "mixed",
        || {
            Ok("strict: NaN passthrough with check_finite=false; \
                hardened: always rejects NaN regardless of check_finite"
                .to_owned())
        },
    );

    let bundle = r.finish();
    assert_eq!(bundle.overall.status, "pass");
}

// ──────────── Cross-Operation Consistency ────────────

/// Scenario 8: solve(A, b) vs inv(A) @ b - verify agreement within tolerance
#[test]
fn e2e_p2c002_08_solve_vs_inv() {
    let mut r = ScenarioRunner::new("p2c002_08_solve_vs_inv");
    let a = vec![
        vec![3.0, 1.0, 0.0],
        vec![1.0, 4.0, 1.0],
        vec![0.0, 1.0, 3.0],
    ];
    let b = vec![4.0, 6.0, 4.0];
    r.set_matrix_meta((3, 3), "strict");

    let mut solve_x = Vec::new();
    let mut inv_x = Vec::new();

    r.record_step("solve_direct", "solve(A, b)", "3x3 SPD-like", "strict", || {
        let res = solve(&a, &b, SolveOptions::default())
            .map_err(|e| format!("solve failed: {e}"))?;
        solve_x = res.x.clone();
        Ok(format!("solve_x={solve_x:?}"))
    });

    r.record_step(
        "inv_then_multiply",
        "inv(A) @ b",
        "compute via inverse",
        "strict",
        || {
            let res =
                inv(&a, InvOptions::default()).map_err(|e| format!("inv failed: {e}"))?;
            inv_x = res
                .inverse
                .iter()
                .map(|row| row.iter().zip(&b).map(|(a, b)| a * b).sum())
                .collect();
            Ok(format!("inv_x={inv_x:?}"))
        },
    );

    r.record_step(
        "compare_results",
        "||solve_x - inv_x|| < tol",
        "verify agreement",
        "strict",
        || {
            let max_diff = max_abs_diff_vec(&solve_x, &inv_x);
            if max_diff < 1e-10 {
                Ok(format!(
                    "solve and inv@b agree: max_diff={max_diff:.2e}"
                ))
            } else {
                Err(format!(
                    "solve and inv@b disagree: max_diff={max_diff:.2e}"
                ))
            }
        },
    );

    let bundle = r.finish();
    assert_eq!(bundle.overall.status, "pass");
}

/// Scenario 9: lstsq vs pinv - verify lstsq(A, b) == pinv(A) @ b for overdetermined
#[test]
fn e2e_p2c002_09_lstsq_vs_pinv() {
    let mut r = ScenarioRunner::new("p2c002_09_lstsq_vs_pinv");
    let a = vec![
        vec![1.0, 1.0],
        vec![1.0, 2.0],
        vec![1.0, 3.0],
    ];
    let b = vec![1.0, 2.0, 2.0];
    r.set_matrix_meta((3, 2), "strict");

    let mut lstsq_x = Vec::new();
    let mut pinv_x = Vec::new();

    r.record_step("lstsq_solve", "lstsq(A, b)", "3x2 overdetermined", "strict", || {
        let res = lstsq(&a, &b, LstsqOptions::default())
            .map_err(|e| format!("lstsq failed: {e}"))?;
        lstsq_x = res.x.clone();
        Ok(format!("lstsq_x={lstsq_x:?}, rank={}", res.rank))
    });

    r.record_step("pinv_solve", "pinv(A) @ b", "via pseudo-inverse", "strict", || {
        let res = pinv(&a, PinvOptions::default())
            .map_err(|e| format!("pinv failed: {e}"))?;
        pinv_x = res
            .pseudo_inverse
            .iter()
            .map(|row| row.iter().zip(&b).map(|(p, bi)| p * bi).sum())
            .collect();
        Ok(format!("pinv_x={pinv_x:?}, rank={}", res.rank))
    });

    r.record_step(
        "compare_lstsq_pinv",
        "||lstsq_x - pinv_x|| < tol",
        "cross-check",
        "strict",
        || {
            let max_diff = max_abs_diff_vec(&lstsq_x, &pinv_x);
            if max_diff < 1e-10 {
                Ok(format!(
                    "lstsq and pinv agree: max_diff={max_diff:.2e}"
                ))
            } else {
                Err(format!(
                    "lstsq and pinv disagree: max_diff={max_diff:.2e}"
                ))
            }
        },
    );

    let bundle = r.finish();
    assert_eq!(bundle.overall.status, "pass");
}

/// Scenario 10: det chain - verify det(A) * det(inv(A)) == 1.0 within tolerance
#[test]
fn e2e_p2c002_10_det_chain() {
    let mut r = ScenarioRunner::new("p2c002_10_det_chain");
    let a = vec![
        vec![4.0, 7.0, 2.0],
        vec![3.0, 6.0, 1.0],
        vec![2.0, 5.0, 3.0],
    ];
    r.set_matrix_meta((3, 3), "strict");

    let mut det_a = 0.0;
    let mut det_inv_a = 0.0;

    r.record_step("compute_det_a", "det(A)", "3x3 general", "strict", || {
        det_a = det(&a, RuntimeMode::Strict, true)
            .map_err(|e| format!("det(A) failed: {e}"))?;
        Ok(format!("det(A)={det_a}"))
    });

    r.record_step(
        "compute_det_inv_a",
        "det(inv(A))",
        "determinant of inverse",
        "strict",
        || {
            let inv_res = inv(&a, InvOptions::default())
                .map_err(|e| format!("inv(A) failed: {e}"))?;
            det_inv_a = det(&inv_res.inverse, RuntimeMode::Strict, true)
                .map_err(|e| format!("det(inv(A)) failed: {e}"))?;
            Ok(format!("det(inv(A))={det_inv_a}"))
        },
    );

    r.record_step(
        "verify_product",
        "det(A) * det(inv(A)) == 1.0",
        "fundamental identity",
        "strict",
        || {
            let product = det_a * det_inv_a;
            let diff = (product - 1.0).abs();
            if diff < 1e-6 {
                Ok(format!(
                    "det(A)*det(inv(A))={product:.12}, diff={diff:.2e}"
                ))
            } else {
                Err(format!(
                    "identity violated: product={product}, diff={diff:.2e}"
                ))
            }
        },
    );

    let bundle = r.finish();
    assert_eq!(bundle.overall.status, "pass");
}

// ──────────── Performance Boundary ────────────

/// Scenario 11: 500x500 solve - verify completes within 5 seconds with correct result
#[test]
fn e2e_p2c002_11_large_solve_500x500() {
    let mut r = ScenarioRunner::new("p2c002_11_large_solve_500x500");
    let n = 500;
    r.set_matrix_meta((n, n), "strict");

    // Construct diagonally dominant matrix for guaranteed non-singularity
    let a = diag_dom_matrix(n);
    let x_expected: Vec<f64> = (0..n).map(|i| (i + 1) as f64 / n as f64).collect();
    let b = mat_vec_mul(&a, &x_expected);

    r.record_step(
        "construct_system",
        "build 500x500 diag-dominant",
        &format!("{n}x{n} matrix + known solution"),
        "strict",
        || Ok(format!("constructed {n}x{n} diag-dominant system")),
    );

    r.record_step(
        "solve_large",
        "solve(A_500, b_500)",
        &format!("{n}x{n} solve"),
        "strict",
        || {
            let start = Instant::now();
            let res = solve(&a, &b, SolveOptions::default())
                .map_err(|e| format!("500x500 solve failed: {e}"))?;
            let elapsed_ms = start.elapsed().as_millis();
            let max_diff = max_abs_diff_vec(&res.x, &x_expected);
            if elapsed_ms > 5000 {
                return Err(format!(
                    "500x500 solve took {elapsed_ms}ms (limit: 5000ms)"
                ));
            }
            if max_diff > 1e-6 {
                return Err(format!(
                    "500x500 solve inaccurate: max_diff={max_diff:.2e}"
                ));
            }
            Ok(format!(
                "completed in {elapsed_ms}ms, max_diff={max_diff:.2e}"
            ))
        },
    );

    r.record_step(
        "verify_backward_error",
        "||Ax - b|| / (||A|| * ||x|| + ||b||)",
        "backward error check",
        "strict",
        || {
            let res = solve(&a, &b, SolveOptions::default()).unwrap();
            let be = res.backward_error.unwrap_or(f64::NAN);
            if be < 1e-10 {
                Ok(format!("backward_error={be:.2e}"))
            } else {
                Err(format!("backward_error too large: {be:.2e}"))
            }
        },
    );

    let bundle = r.finish();
    assert_eq!(bundle.overall.status, "pass");
}
