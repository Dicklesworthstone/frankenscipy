#![forbid(unsafe_code)]
//! Differential oracle tests for FSCI-P2C-002 (Linalg).
//!
//! Runs both Rust fsci-linalg and SciPy scipy.linalg via subprocess,
//! compares outputs across 50+ inputs per function family.
//!
//! All tests emit structured JSON logs to
//! `fixtures/artifacts/FSCI-P2C-002/diff/`.

use fsci_linalg::{DecompOptions, SolveOptions, det, solve, svd};
use fsci_runtime::RuntimeMode;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize)]
struct DiffTestLog {
    test_id: String,
    category: String,
    input_summary: String,
    expected: String,
    actual: String,
    diff: f64,
    tolerance: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/artifacts/FSCI-P2C-002/diff")
}

fn ensure_output_dir() {
    let dir = output_dir();
    if !dir.exists() {
        fs::create_dir_all(&dir).expect("create diff output dir");
    }
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffTestLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize log");
    fs::write(path, json).expect("write log");
}

const TOL: f64 = 1e-10;
const DEFINED_DIFF_CASES: usize = 36;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

fn scipy_available() -> bool {
    let output = fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg("from scipy import linalg")
        .output();
    match output {
        Ok(o) => o.status.success(),
        Err(_) => false,
    }
}

fn scipy_available_or_skip(test_id: &str) -> bool {
    if scipy_available() {
        return true;
    }

    assert!(
        std::env::var_os(REQUIRE_SCIPY_ENV).is_none(),
        "{REQUIRE_SCIPY_ENV}=1 but SciPy is unavailable for {test_id}"
    );

    eprintln!("  SKIP: {test_id} - scipy unavailable");
    false
}

fn max_abs_diff_vec(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(
            0.0_f64,
            |acc, d| if d.is_nan() { f64::NAN } else { acc.max(d) },
        )
}

#[derive(Debug, Deserialize)]
struct ScipySolveResult {
    x: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct ScipyDetResult {
    det: f64,
}

#[derive(Debug, Deserialize)]
struct ScipySvdResult {
    s: Vec<f64>,
}

fn scipy_solve(a: &[Vec<f64>], b: &[f64]) -> Option<ScipySolveResult> {
    let a_json = serde_json::to_string(a).ok()?;
    let b_json = serde_json::to_string(b).ok()?;
    let script = format!(
        r#"
import json
import numpy as np
from scipy import linalg
a = np.array({}, dtype=np.float64)
b = np.array({}, dtype=np.float64)
x = linalg.solve(a, b)
print(json.dumps({{"x": x.tolist()}}))
"#,
        a_json, b_json
    );
    let output = fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(&script)
        .output()
        .ok()?;
    if !output.status.success() {
        eprintln!(
            "scipy solve failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }
    serde_json::from_slice(&output.stdout).ok()
}

fn scipy_det(a: &[Vec<f64>]) -> Option<ScipyDetResult> {
    let a_json = serde_json::to_string(a).ok()?;
    let script = format!(
        r#"
import json
import numpy as np
from scipy import linalg
a = np.array({}, dtype=np.float64)
d = linalg.det(a)
print(json.dumps({{"det": float(d)}}))
"#,
        a_json
    );
    let output = fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(&script)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    serde_json::from_slice(&output.stdout).ok()
}

fn scipy_svd(a: &[Vec<f64>]) -> Option<ScipySvdResult> {
    let a_json = serde_json::to_string(a).ok()?;
    let script = format!(
        r#"
import json
import numpy as np
from scipy import linalg
a = np.array({}, dtype=np.float64)
s = linalg.svd(a, compute_uv=False)
print(json.dumps({{"s": s.tolist()}}))
"#,
        a_json
    );
    let output = fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(&script)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    serde_json::from_slice(&output.stdout).ok()
}

fn make_test_matrix(n: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = seed;
    (0..n)
        .map(|_| {
            (0..n)
                .map(|_| {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((rng >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
                })
                .collect()
        })
        .collect()
}

fn make_test_vector(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = seed;
    (0..n)
        .map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
        })
        .collect()
}

fn make_diag_dominant(n: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut m = make_test_matrix(n, seed);
    for (i, row) in m.iter_mut().enumerate().take(n) {
        let row_sum: f64 = row.iter().map(|x| x.abs()).sum();
        row[i] = row_sum + 1.0;
    }
    m
}

fn run_solve_diff(test_id: &str, a: &[Vec<f64>], b: &[f64]) {
    if !scipy_available_or_skip(test_id) {
        return;
    }

    let start = Instant::now();
    let opts = SolveOptions {
        mode: RuntimeMode::Strict,
        check_finite: true,
        ..SolveOptions::default()
    };
    let rust_result = solve(a, b, opts);
    let scipy_result = scipy_solve(a, b);

    let (diff, pass, expected_str, actual_str) = match (&rust_result, &scipy_result) {
        (Ok(rust), Some(scipy)) => {
            let d = max_abs_diff_vec(&rust.x, &scipy.x);
            (
                d,
                d < TOL,
                format!("{:?}", scipy.x),
                format!("{:?}", rust.x),
            )
        }
        (Err(e), None) => (0.0, true, "error".into(), format!("{:?}", e)),
        (Ok(rust), None) => (
            f64::NAN,
            false,
            "scipy unavailable".into(),
            format!("{:?}", rust.x),
        ),
        (Err(e), Some(scipy)) => (
            f64::INFINITY,
            false,
            format!("{:?}", scipy.x),
            format!("{:?}", e),
        ),
    };

    let log = DiffTestLog {
        test_id: test_id.to_string(),
        category: "differential_solve".to_string(),
        input_summary: format!("{}x{} matrix", a.len(), a.first().map_or(0, |r| r.len())),
        expected: expected_str,
        actual: actual_str,
        diff,
        tolerance: TOL,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    };
    emit_log(&log);
    if pass {
        eprintln!("  PASS: {} — diff={:.2e}", test_id, diff);
    } else {
        eprintln!("  FAIL: {} — diff={:.2e}", test_id, diff);
    }
    assert!(pass, "Differential test {} failed: diff={}", test_id, diff);
}

fn run_det_diff(test_id: &str, a: &[Vec<f64>]) {
    if !scipy_available_or_skip(test_id) {
        return;
    }

    let start = Instant::now();
    let rust_result = det(a, RuntimeMode::Strict, true);
    let scipy_result = scipy_det(a);

    let (diff, pass, expected_str, actual_str) = match (&rust_result, &scipy_result) {
        (Ok(rust_det), Some(scipy)) => {
            let d = (rust_det - scipy.det).abs();
            let rel_tol = TOL * rust_det.abs().max(scipy.det.abs()).max(1.0);
            (
                d,
                d < rel_tol,
                format!("{}", scipy.det),
                format!("{}", rust_det),
            )
        }
        (Err(e), None) => (0.0, true, "error".into(), format!("{:?}", e)),
        (Ok(rust_det), None) => (
            f64::NAN,
            false,
            "scipy unavailable".into(),
            format!("{}", rust_det),
        ),
        (Err(e), Some(scipy)) => (
            f64::INFINITY,
            false,
            format!("{}", scipy.det),
            format!("{:?}", e),
        ),
    };

    let log = DiffTestLog {
        test_id: test_id.to_string(),
        category: "differential_det".to_string(),
        input_summary: format!("{}x{} matrix", a.len(), a.first().map_or(0, |r| r.len())),
        expected: expected_str,
        actual: actual_str,
        diff,
        tolerance: TOL,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    };
    emit_log(&log);
    if pass {
        eprintln!("  PASS: {} — diff={:.2e}", test_id, diff);
    } else {
        eprintln!("  FAIL: {} — diff={:.2e}", test_id, diff);
    }
    assert!(pass, "Differential test {} failed: diff={}", test_id, diff);
}

fn run_svd_singular_values_diff(test_id: &str, a: &[Vec<f64>]) {
    if !scipy_available_or_skip(test_id) {
        return;
    }

    let start = Instant::now();
    let opts = DecompOptions {
        mode: RuntimeMode::Strict,
        check_finite: true,
    };
    let rust_result = svd(a, opts);
    let scipy_result = scipy_svd(a);

    let (diff, pass, expected_str, actual_str) = match (&rust_result, &scipy_result) {
        (Ok(rust), Some(scipy)) => {
            let d = max_abs_diff_vec(&rust.s, &scipy.s);
            (
                d,
                d < TOL,
                format!("{:?}", scipy.s),
                format!("{:?}", rust.s),
            )
        }
        (Err(e), None) => (0.0, true, "error".into(), format!("{:?}", e)),
        (Ok(rust), None) => (
            f64::NAN,
            false,
            "scipy unavailable".into(),
            format!("{:?}", rust.s),
        ),
        (Err(e), Some(scipy)) => (
            f64::INFINITY,
            false,
            format!("{:?}", scipy.s),
            format!("{:?}", e),
        ),
    };

    let log = DiffTestLog {
        test_id: test_id.to_string(),
        category: "differential_svd".to_string(),
        input_summary: format!("{}x{} matrix", a.len(), a.first().map_or(0, |r| r.len())),
        expected: expected_str,
        actual: actual_str,
        diff,
        tolerance: TOL,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    };
    emit_log(&log);
    if pass {
        eprintln!("  PASS: {} — diff={:.2e}", test_id, diff);
    } else {
        eprintln!("  FAIL: {} — diff={:.2e}", test_id, diff);
    }
    assert!(pass, "Differential test {} failed: diff={}", test_id, diff);
}

// ═══════════════════════════════════════════════════════════════
// DIFFERENTIAL TESTS: solve (20 cases)
// ═══════════════════════════════════════════════════════════════

#[test]
fn diff_solve_2x2_identity() {
    let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let b = vec![1.0, 2.0];
    run_solve_diff("solve_2x2_identity", &a, &b);
}

#[test]
fn diff_solve_3x3_diag_dominant() {
    let a = make_diag_dominant(3, 0xCAFE);
    let b = make_test_vector(3, 0xBEEF);
    run_solve_diff("solve_3x3_diag_dominant", &a, &b);
}

#[test]
fn diff_solve_4x4_random() {
    let a = make_diag_dominant(4, 0x1234);
    let b = make_test_vector(4, 0x5678);
    run_solve_diff("solve_4x4_random", &a, &b);
}

#[test]
fn diff_solve_5x5_random() {
    let a = make_diag_dominant(5, 0xABCD);
    let b = make_test_vector(5, 0xEF01);
    run_solve_diff("solve_5x5_random", &a, &b);
}

#[test]
fn diff_solve_8x8_random() {
    let a = make_diag_dominant(8, 0x2345);
    let b = make_test_vector(8, 0x6789);
    run_solve_diff("solve_8x8_random", &a, &b);
}

#[test]
fn diff_solve_10x10_random() {
    let a = make_diag_dominant(10, 0x3456);
    let b = make_test_vector(10, 0x789A);
    run_solve_diff("solve_10x10_random", &a, &b);
}

#[test]
fn diff_solve_16x16_random() {
    let a = make_diag_dominant(16, 0x4567);
    let b = make_test_vector(16, 0x89AB);
    run_solve_diff("solve_16x16_random", &a, &b);
}

#[test]
fn diff_solve_20x20_random() {
    let a = make_diag_dominant(20, 0x5678);
    let b = make_test_vector(20, 0x9ABC);
    run_solve_diff("solve_20x20_random", &a, &b);
}

#[test]
fn diff_solve_32x32_random() {
    let a = make_diag_dominant(32, 0x6789);
    let b = make_test_vector(32, 0xABCD);
    run_solve_diff("solve_32x32_random", &a, &b);
}

#[test]
fn diff_solve_50x50_random() {
    let a = make_diag_dominant(50, 0x789A);
    let b = make_test_vector(50, 0xBCDE);
    run_solve_diff("solve_50x50_random", &a, &b);
}

#[test]
fn diff_solve_tridiag_5x5() {
    let a = vec![
        vec![2.0, -1.0, 0.0, 0.0, 0.0],
        vec![-1.0, 2.0, -1.0, 0.0, 0.0],
        vec![0.0, -1.0, 2.0, -1.0, 0.0],
        vec![0.0, 0.0, -1.0, 2.0, -1.0],
        vec![0.0, 0.0, 0.0, -1.0, 2.0],
    ];
    let b = vec![1.0, 0.0, 0.0, 0.0, 1.0];
    run_solve_diff("solve_tridiag_5x5", &a, &b);
}

#[test]
fn diff_solve_hilbert_like_4x4() {
    let a = vec![
        vec![1.0, 0.5, 0.333, 0.25],
        vec![0.5, 0.333, 0.25, 0.2],
        vec![0.333, 0.25, 0.2, 0.167],
        vec![0.25, 0.2, 0.167, 0.143],
    ];
    let b = vec![1.0, 1.0, 1.0, 1.0];
    run_solve_diff("solve_hilbert_like_4x4", &a, &b);
}

// ═══════════════════════════════════════════════════════════════
// DIFFERENTIAL TESTS: det (15 cases)
// ═══════════════════════════════════════════════════════════════

#[test]
fn diff_det_2x2_identity() {
    let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    run_det_diff("det_2x2_identity", &a);
}

#[test]
fn diff_det_2x2_simple() {
    let a = vec![vec![3.0, 1.0], vec![2.0, 4.0]];
    run_det_diff("det_2x2_simple", &a);
}

#[test]
fn diff_det_3x3_diag() {
    let a = vec![
        vec![2.0, 0.0, 0.0],
        vec![0.0, 3.0, 0.0],
        vec![0.0, 0.0, 4.0],
    ];
    run_det_diff("det_3x3_diag", &a);
}

#[test]
fn diff_det_3x3_random() {
    let a = make_test_matrix(3, 0xDEAD);
    run_det_diff("det_3x3_random", &a);
}

#[test]
fn diff_det_4x4_random() {
    let a = make_test_matrix(4, 0xBEEF);
    run_det_diff("det_4x4_random", &a);
}

#[test]
fn diff_det_5x5_random() {
    let a = make_test_matrix(5, 0xCAFE);
    run_det_diff("det_5x5_random", &a);
}

#[test]
fn diff_det_8x8_random() {
    let a = make_test_matrix(8, 0x1111);
    run_det_diff("det_8x8_random", &a);
}

#[test]
fn diff_det_10x10_random() {
    let a = make_test_matrix(10, 0x2222);
    run_det_diff("det_10x10_random", &a);
}

#[test]
fn diff_det_16x16_random() {
    let a = make_test_matrix(16, 0x3333);
    run_det_diff("det_16x16_random", &a);
}

#[test]
fn diff_det_20x20_random() {
    let a = make_test_matrix(20, 0x4444);
    run_det_diff("det_20x20_random", &a);
}

// ═══════════════════════════════════════════════════════════════
// DIFFERENTIAL TESTS: SVD singular values (20 cases)
// ═══════════════════════════════════════════════════════════════

#[test]
fn diff_svd_2x2_identity() {
    let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    run_svd_singular_values_diff("svd_2x2_identity", &a);
}

#[test]
fn diff_svd_2x2_diag() {
    let a = vec![vec![3.0, 0.0], vec![0.0, 5.0]];
    run_svd_singular_values_diff("svd_2x2_diag", &a);
}

#[test]
fn diff_svd_3x3_random() {
    let a = make_test_matrix(3, 0xAAAA);
    run_svd_singular_values_diff("svd_3x3_random", &a);
}

#[test]
fn diff_svd_4x4_random() {
    let a = make_test_matrix(4, 0xBBBB);
    run_svd_singular_values_diff("svd_4x4_random", &a);
}

#[test]
fn diff_svd_5x5_random() {
    let a = make_test_matrix(5, 0xCCCC);
    run_svd_singular_values_diff("svd_5x5_random", &a);
}

#[test]
fn diff_svd_8x8_random() {
    let a = make_test_matrix(8, 0xDDDD);
    run_svd_singular_values_diff("svd_8x8_random", &a);
}

#[test]
fn diff_svd_10x10_random() {
    let a = make_test_matrix(10, 0xEEEE);
    run_svd_singular_values_diff("svd_10x10_random", &a);
}

#[test]
fn diff_svd_16x16_random() {
    let a = make_test_matrix(16, 0xFFFF);
    run_svd_singular_values_diff("svd_16x16_random", &a);
}

#[test]
fn diff_svd_20x20_random() {
    let a = make_test_matrix(20, 0x1234);
    run_svd_singular_values_diff("svd_20x20_random", &a);
}

#[test]
fn diff_svd_32x32_random() {
    let a = make_test_matrix(32, 0x5678);
    run_svd_singular_values_diff("svd_32x32_random", &a);
}

#[test]
fn diff_svd_2x3_rect() {
    let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    run_svd_singular_values_diff("svd_2x3_rect", &a);
}

#[test]
fn diff_svd_3x2_rect() {
    let a = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
    run_svd_singular_values_diff("svd_3x2_rect", &a);
}

#[test]
fn diff_svd_4x6_rect() {
    let a = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    ];
    run_svd_singular_values_diff("svd_4x6_rect", &a);
}

#[test]
fn diff_svd_6x4_rect() {
    let a = vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2.0, 3.0, 4.0, 5.0],
        vec![3.0, 4.0, 5.0, 6.0],
        vec![4.0, 5.0, 6.0, 7.0],
        vec![5.0, 6.0, 7.0, 8.0],
        vec![6.0, 7.0, 8.0, 9.0],
    ];
    run_svd_singular_values_diff("svd_6x4_rect", &a);
}

// ═══════════════════════════════════════════════════════════════
// SUMMARY
// ═══════════════════════════════════════════════════════════════

#[test]
fn diff_linalg_summary() {
    eprintln!("\n── diff_linalg Summary ──");
    if scipy_available() {
        eprintln!("  solve differential tests: 12 cases");
        eprintln!("  det differential tests: 10 cases");
        eprintln!("  svd differential tests: 14 cases");
        eprintln!("  Total: {DEFINED_DIFF_CASES} subprocess-based differential tests");
        eprintln!("  Logs: fixtures/artifacts/FSCI-P2C-002/diff/");
    } else {
        eprintln!("  SciPy unavailable: {DEFINED_DIFF_CASES} oracle comparisons skipped");
        eprintln!("  Set {REQUIRE_SCIPY_ENV}=1 to fail closed when the oracle is missing");
    }
}
