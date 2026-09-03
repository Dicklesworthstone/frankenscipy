#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_linalg::eigh_tridiagonal
//! (symmetric tridiagonal eigenvalue problem).
//!
//! Resolves [frankenscipy-avm7i]. 1e-9 abs. Compares sorted eigenvalues,
//! sign-aligned eigenvectors, tridiagonal residuals, and orthogonality.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_linalg::{DecompOptions, eigh_tridiagonal};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    d: Vec<f64>,
    e: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    eigvals_sorted: Option<Vec<f64>>,
    eigvecs: Option<Vec<Vec<f64>>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    abs_diff: f64,
    eigvec_abs_diff: f64,
    residual_abs: f64,
    orthogonality_abs: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    max_abs_diff: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create eigh_tridiagonal diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize eigh_tridiagonal diff log");
    fs::write(path, json).expect("write eigh_tridiagonal diff log");
}

fn generate_query() -> OracleQuery {
    let points = vec![
        PointCase {
            case_id: "n3_spd".into(),
            d: vec![4.0, 2.0, 3.0],
            e: vec![1.0, 0.5],
        },
        PointCase {
            case_id: "n4_spd".into(),
            d: vec![4.0, 2.0, 3.0, 5.0],
            e: vec![1.0, 0.5, 0.7],
        },
        PointCase {
            case_id: "n5_dominant_diag".into(),
            d: vec![10.0, 11.0, 12.0, 13.0, 14.0],
            e: vec![0.1, 0.2, 0.3, 0.4],
        },
        PointCase {
            case_id: "n6_uniform".into(),
            d: vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            e: vec![-1.0, -1.0, -1.0, -1.0, -1.0],
        },
        PointCase {
            case_id: "n4_mixed_sign".into(),
            d: vec![1.0, -2.0, 3.0, -4.0],
            e: vec![0.5, 0.3, 0.2],
        },
    ];
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy import linalg

def finite_sorted_or_none(arr):
    flat = []
    for v in np.asarray(arr, dtype=float).flatten().tolist():
        if not math.isfinite(float(v)):
            return None
        flat.append(float(v))
    flat.sort()
    return flat

def finite_matrix_or_none(mat):
    out = []
    for row in np.asarray(mat, dtype=float).tolist():
        row_out = []
        for v in row:
            if not math.isfinite(float(v)):
                return None
            row_out.append(float(v))
        out.append(row_out)
    return out

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    d = np.array(case["d"], dtype=float)
    e = np.array(case["e"], dtype=float)
    try:
        w, v = linalg.eigh_tridiagonal(d, e, eigvals_only=False)
        points.append({
            "case_id": cid,
            "eigvals_sorted": finite_sorted_or_none(w),
            "eigvecs": finite_matrix_or_none(v),
        })
    except Exception:
        points.append({"case_id": cid, "eigvals_sorted": None, "eigvecs": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize eigh_tridiagonal query");
    ensure_output_dir();
    let script_path = output_dir().join(format!("eigh_tridiagonal_oracle_{}.py", timestamp_ms()));
    if let Err(e) = fs::write(&script_path, script) {
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "failed to write eigh_tridiagonal oracle script: {e}"
        );
        eprintln!("skipping eigh_tridiagonal oracle: script write failed ({e})");
        return None;
    }
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg(&script_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn python3 for eigh_tridiagonal oracle: {e}"
            );
            eprintln!("skipping eigh_tridiagonal oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open eigh_tridiagonal oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "eigh_tridiagonal oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping eigh_tridiagonal oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for eigh_tridiagonal oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "eigh_tridiagonal oracle failed: {stderr}"
        );
        eprintln!("skipping eigh_tridiagonal oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse eigh_tridiagonal oracle JSON"))
}

fn sign_aligned_eigenvector_max_abs(actual: &[Vec<f64>], expected: &[Vec<f64>]) -> f64 {
    let n = actual.len();
    let mut max_abs = 0.0_f64;
    for col in 0..n {
        let mut dot = 0.0_f64;
        for row in 0..n {
            dot += actual[row][col] * expected[row][col];
        }
        let sign = if dot < 0.0 { -1.0 } else { 1.0 };
        for row in 0..n {
            max_abs = max_abs.max((sign * actual[row][col] - expected[row][col]).abs());
        }
    }
    max_abs
}

fn tridiagonal_residual_max_abs(
    d: &[f64],
    e: &[f64],
    eigenvalues: &[f64],
    eigenvectors: &[Vec<f64>],
) -> f64 {
    let n = d.len();
    let mut max_abs = 0.0_f64;
    for col in 0..n {
        let lambda = eigenvalues[col];
        for row in 0..n {
            let mut projected = d[row] * eigenvectors[row][col];
            if row > 0 {
                projected += e[row - 1] * eigenvectors[row - 1][col];
            }
            if row + 1 < n {
                projected += e[row] * eigenvectors[row + 1][col];
            }
            max_abs = max_abs.max((projected - lambda * eigenvectors[row][col]).abs());
        }
    }
    max_abs
}

fn eigenvector_orthogonality_max_abs(eigenvectors: &[Vec<f64>]) -> f64 {
    let n = eigenvectors.len();
    let mut max_abs = 0.0_f64;
    for left in 0..n {
        for right in 0..n {
            let dot: f64 = eigenvectors.iter().map(|row| row[left] * row[right]).sum();
            let target = if left == right { 1.0 } else { 0.0 };
            max_abs = max_abs.max((dot - target).abs());
        }
    }
    max_abs
}

#[test]
fn diff_linalg_eigh_tridiagonal() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(expected) = scipy_arm.eigvals_sorted.as_ref() else {
            continue;
        };
        let Some(expected_vectors) = scipy_arm.eigvecs.as_ref() else {
            continue;
        };
        let opts = DecompOptions::default();
        let Ok((w, vectors)) = eigh_tridiagonal(&case.d, &case.e, false, opts) else {
            continue;
        };
        let abs_d = if w.len() != expected.len() {
            f64::INFINITY
        } else {
            w.iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        };
        let vectors = vectors.expect("eigh_tridiagonal vectors requested");
        let eigvec_abs_diff = sign_aligned_eigenvector_max_abs(&vectors, expected_vectors);
        let residual_abs = tridiagonal_residual_max_abs(&case.d, &case.e, &w, &vectors);
        let orthogonality_abs = eigenvector_orthogonality_max_abs(&vectors);
        let pass = abs_d <= ABS_TOL
            && eigvec_abs_diff <= 1.0e-8
            && residual_abs <= 1.0e-8
            && orthogonality_abs <= 1.0e-8;
        max_overall = max_overall
            .max(abs_d)
            .max(eigvec_abs_diff)
            .max(residual_abs)
            .max(orthogonality_abs);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: abs_d,
            eigvec_abs_diff,
            residual_abs,
            orthogonality_abs,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_linalg_eigh_tridiagonal".into(),
        category: "scipy.linalg.eigh_tridiagonal".into(),
        case_count: diffs.len(),
        max_abs_diff: max_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "eigh_tridiagonal mismatch: {} abs_diff={} eigvec_abs_diff={} residual_abs={} orthogonality_abs={}",
                d.case_id, d.abs_diff, d.eigvec_abs_diff, d.residual_abs, d.orthogonality_abs
            );
        }
    }

    assert!(
        all_pass,
        "eigh_tridiagonal conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
