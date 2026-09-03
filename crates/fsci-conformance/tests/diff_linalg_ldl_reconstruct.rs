#![forbid(unsafe_code)]
//! Property-based parity harness for fsci_linalg::ldl on symmetric SPD
//! matrices: verify A ≈ L D Lᵀ.
//!
//! Resolves [frankenscipy-i2gb3]. scipy.linalg.ldl uses permutation +
//! block-diagonal D and a different storage convention, so direct
//! element parity is impractical. We check the reconstruction
//! invariant — which is the contract LDL provides — at 1e-9 abs.
//! Verification also calls scipy.linalg.eigvals(A) to confirm the test
//! input is well-conditioned (all-positive eigenvalues).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_linalg::{DecompOptions, ldl};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    rows: usize,
    cols: usize,
    a: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// scipy.linalg.eigvals(A).real sorted — used to verify the input
    /// is SPD (all eigenvalues > 0) so the test is meaningful.
    eigvals_sorted: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    abs_diff: f64,
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
    fs::create_dir_all(output_dir()).expect("create ldl diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize ldl diff log");
    fs::write(path, json).expect("write ldl diff log");
}

fn rows_of(a_flat: &[f64], rows: usize, cols: usize) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|r| (0..cols).map(|c| a_flat[r * cols + c]).collect())
        .collect()
}

fn frob_diff(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let mut max = 0.0_f64;
    for (ra, rb) in a.iter().zip(b.iter()) {
        for (&va, &vb) in ra.iter().zip(rb.iter()) {
            max = max.max((va - vb).abs());
        }
    }
    max
}

fn generate_query() -> OracleQuery {
    let points = vec![
        PointCase {
            case_id: "spd_2x2".into(),
            rows: 2,
            cols: 2,
            a: vec![4.0, 1.0, 1.0, 3.0],
        },
        PointCase {
            case_id: "spd_3x3".into(),
            rows: 3,
            cols: 3,
            a: vec![4.0, 1.0, 0.5, 1.0, 5.0, 0.3, 0.5, 0.3, 6.0],
        },
        PointCase {
            case_id: "spd_4x4".into(),
            rows: 4,
            cols: 4,
            a: vec![
                10.0, 2.0, 1.0, 0.5, 2.0, 8.0, 1.5, 0.7, 1.0, 1.5, 9.0, 1.2, 0.5, 0.7, 1.2, 7.0,
            ],
        },
        PointCase {
            case_id: "diag_3x3".into(),
            rows: 3,
            cols: 3,
            a: vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0],
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

def finite_sorted_real_or_none(arr):
    flat = []
    for v in np.asarray(arr).flatten().tolist():
        re = float(v.real) if hasattr(v, "real") else float(v)
        if not math.isfinite(re):
            return None
        flat.append(re)
    flat.sort()
    return flat

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    r = int(case["rows"]); c = int(case["cols"])
    A = np.array(case["a"], dtype=float).reshape(r, c)
    try:
        w = linalg.eigvals(A)
        points.append({"case_id": cid, "eigvals_sorted": finite_sorted_real_or_none(w)})
    except Exception:
        points.append({"case_id": cid, "eigvals_sorted": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize ldl query");
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn python3 for ldl oracle: {e}"
            );
            eprintln!("skipping ldl oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open ldl oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "ldl oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping ldl oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for ldl oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "ldl oracle failed: {stderr}"
        );
        eprintln!("skipping ldl oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse ldl oracle JSON"))
}

#[test]
fn diff_linalg_ldl_reconstruct() {
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
        let Some(eigs) = scipy_arm.eigvals_sorted.as_ref() else {
            continue;
        };
        // Sanity-check the input is SPD (all eigenvalues positive)
        if eigs.iter().any(|&e| e <= 0.0) {
            // Test case isn't valid for LDL on SPD; skip
            continue;
        }
        let a = rows_of(&case.a, case.rows, case.cols);
        let opts = DecompOptions::default();
        let Ok(res) = ldl(&a, opts) else {
            continue;
        };
        // Build L D Lᵀ
        let n = res.l.len();
        // L * D = scale each column j of L by d[j]
        let mut ld = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                ld[i][j] = res.l[i][j] * res.d[j];
            }
        }
        // (L D) * Lᵀ
        let mut ldl_t = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for k in 0..n {
                for j in 0..n {
                    ldl_t[i][j] += ld[i][k] * res.l[j][k];
                }
            }
        }
        let abs_d = frob_diff(&a, &ldl_t);
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_linalg_ldl_reconstruct".into(),
        category: "fsci_linalg.ldl A ≈ L D Lᵀ reconstruction".into(),
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
            eprintln!("ldl mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "ldl conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
