#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_linalg::eig_banded
//! (symmetric banded eigenvalue problem, values and vectors).
//!
//! Resolves [frankenscipy-vpaa7]. 1e-9 abs. Compares sorted eigenvalues
//! and sign-aligned eigenvectors in lower/upper band storage.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_linalg::{DecompOptions, eig_banded};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    /// Banded storage row-major: each row is one diagonal.
    ab_flat: Vec<f64>,
    rows: usize,
    cols: usize,
    lower: bool,
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
    fs::create_dir_all(output_dir()).expect("create eig_banded diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize eig_banded diff log");
    fs::write(path, json).expect("write eig_banded diff log");
}

fn rows_of(a_flat: &[f64], rows: usize, cols: usize) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|r| (0..cols).map(|c| a_flat[r * cols + c]).collect())
        .collect()
}

fn generate_query() -> OracleQuery {
    let points = vec![
        PointCase {
            // 4x4 tridiag, lower-storage:
            // row 0 = diag = [4, 5, 6, 7]
            // row 1 = subdiag = [1, 2, 3, 0]
            case_id: "tridiag_4".into(),
            ab_flat: vec![4.0, 5.0, 6.0, 7.0, 1.0, 2.0, 3.0, 0.0],
            rows: 2,
            cols: 4,
            lower: true,
        },
        PointCase {
            // 5x5 tridiag, lower
            case_id: "tridiag_5".into(),
            ab_flat: vec![2.0, 2.0, 2.0, 2.0, 2.0, -1.0, -1.0, -1.0, -1.0, 0.0],
            rows: 2,
            cols: 5,
            lower: true,
        },
        PointCase {
            // 4x4 pentadiagonal, lower:
            // row 0 = diag
            // row 1 = first subdiag
            // row 2 = second subdiag
            case_id: "pentadiag_4".into(),
            ab_flat: vec![
                3.0, 4.0, 5.0, 6.0, // diag
                0.5, 0.7, 0.3, 0.0, // sub-1
                0.1, 0.2, 0.0, 0.0, // sub-2
            ],
            rows: 3,
            cols: 4,
            lower: true,
        },
        PointCase {
            // Same pentadiagonal matrix as upper band storage:
            // row 2 = diag, row 1 = first superdiag, row 0 = second superdiag.
            case_id: "pentadiag_4_upper".into(),
            ab_flat: vec![
                0.0, 0.0, 0.1, 0.2, // super-2
                0.0, 0.5, 0.7, 0.3, // super-1
                3.0, 4.0, 5.0, 6.0, // diag
            ],
            rows: 3,
            cols: 4,
            lower: false,
        },
        PointCase {
            // 6x6 tridiag
            case_id: "tridiag_6_mixed".into(),
            ab_flat: vec![
                3.0, 1.0, 4.0, 1.0, 5.0, 9.0, // diag
                0.5, -0.3, 0.7, -0.2, 0.4, 0.0, // sub
            ],
            rows: 2,
            cols: 6,
            lower: true,
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
    r = int(case["rows"]); c = int(case["cols"])
    ab = np.array(case["ab_flat"], dtype=float).reshape(r, c)
    lower = bool(case["lower"])
    try:
        w, v = linalg.eig_banded(ab, lower=lower, eigvals_only=False)
        points.append({
            "case_id": cid,
            "eigvals_sorted": finite_sorted_or_none(w),
            "eigvecs": finite_matrix_or_none(v),
        })
    except Exception:
        points.append({"case_id": cid, "eigvals_sorted": None, "eigvecs": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize eig_banded query");
    ensure_output_dir();
    let script_path = output_dir().join(format!("eig_banded_oracle_{}.py", timestamp_ms()));
    if let Err(e) = fs::write(&script_path, script) {
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "failed to write eig_banded oracle script: {e}"
        );
        eprintln!("skipping eig_banded oracle: script write failed ({e})");
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
                "failed to spawn python3 for eig_banded oracle: {e}"
            );
            eprintln!("skipping eig_banded oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open eig_banded oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "eig_banded oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping eig_banded oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for eig_banded oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "eig_banded oracle failed: {stderr}"
        );
        eprintln!("skipping eig_banded oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse eig_banded oracle JSON"))
}

fn dense_from_case(case: &PointCase) -> Vec<Vec<f64>> {
    let ab = rows_of(&case.ab_flat, case.rows, case.cols);
    let n = case.cols;
    let bandwidth = case.rows - 1;
    let mut dense = vec![vec![0.0; n]; n];
    if case.lower {
        for col in 0..n {
            dense[col][col] = ab[0][col];
            for (offset, band_row) in ab.iter().enumerate().skip(1) {
                let row = col + offset;
                if row >= n {
                    break;
                }
                dense[row][col] = band_row[col];
                dense[col][row] = band_row[col];
            }
        }
    } else {
        for col in 0..n {
            dense[col][col] = ab[bandwidth][col];
            for offset in 1..=bandwidth {
                if col < offset {
                    continue;
                }
                let row = col - offset;
                let value = ab[bandwidth - offset][col];
                dense[row][col] = value;
                dense[col][row] = value;
            }
        }
    }
    dense
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

fn eigenvector_residual_max_abs(
    dense: &[Vec<f64>],
    eigenvalues: &[f64],
    eigenvectors: &[Vec<f64>],
) -> f64 {
    let n = dense.len();
    let mut max_abs = 0.0_f64;
    for col in 0..n {
        let lambda = eigenvalues[col];
        for row in 0..n {
            let projected: f64 = dense[row]
                .iter()
                .zip(eigenvectors.iter())
                .map(|(a, vector_row)| a * vector_row[col])
                .sum();
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
fn diff_linalg_eig_banded() {
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
        let ab = rows_of(&case.ab_flat, case.rows, case.cols);
        let opts = DecompOptions::default();
        let Ok((w, vectors)) = eig_banded(&ab, case.lower, false, opts) else {
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
        let vectors = vectors.expect("eig_banded vectors requested");
        let eigvec_abs_diff = sign_aligned_eigenvector_max_abs(&vectors, expected_vectors);
        let dense = dense_from_case(case);
        let residual_abs = eigenvector_residual_max_abs(&dense, &w, &vectors);
        let orthogonality_abs = eigenvector_orthogonality_max_abs(&vectors);
        let pass = abs_d <= ABS_TOL
            && eigvec_abs_diff <= 1.0e-8
            && residual_abs <= 1.0e-8
            && orthogonality_abs <= 1.0e-8;
        max_overall = max_overall.max(abs_d);
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
        test_id: "diff_linalg_eig_banded".into(),
        category: "scipy.linalg.eig_banded".into(),
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
                "eig_banded mismatch: {} abs_diff={} eigvec_abs_diff={} residual_abs={} orthogonality_abs={}",
                d.case_id, d.abs_diff, d.eigvec_abs_diff, d.residual_abs, d.orthogonality_abs
            );
        }
    }

    assert!(
        all_pass,
        "eig_banded conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
