#![forbid(unsafe_code)]
//! Live numerical reference checks for the cross-covariance
//! matrix function `cross_cov(X, Y)` → Cov(X_i, Y_j).
//!
//! Resolves [frankenscipy-tjodj]. fsci's cross_cov uses
//! ddof=1 sample covariance. The oracle reproduces this in
//! numpy via `np.cov(stacked, ddof=1)[:dx, dx:dx+dy]`.
//!
//! 3 (X, Y) fixtures = 3 matrix cases via subprocess. Each
//! case compares the cross-covariance matrix element-wise
//! (max-abs aggregation). Tol 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::cross_cov;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    x: Vec<Vec<f64>>,
    y: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    matrix: Option<Vec<Vec<f64>>>,
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
    fs::create_dir_all(output_dir()).expect("create cross_cov diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize cross_cov diff log");
    fs::write(path, json).expect("write cross_cov diff log");
}

fn generate_query() -> OracleQuery {
    let fixtures: Vec<(&str, Vec<Vec<f64>>, Vec<Vec<f64>>)> = vec![
        // X: 2D, Y: 2D, n=8 obs
        (
            "two_two",
            (1..=8)
                .map(|i| vec![i as f64, (i as f64).powi(2)])
                .collect(),
            (1..=8)
                .map(|i| vec![(i as f64).sqrt(), -(i as f64)])
                .collect(),
        ),
        // X: 3D, Y: 1D
        (
            "three_one",
            (1..=10)
                .map(|i| vec![i as f64, (i as f64) / 2.0, ((i + 1) as f64).powi(2)])
                .collect(),
            (1..=10).map(|i| vec![i as f64 * 0.5]).collect(),
        ),
        // X: 2D, Y: 3D
        (
            "two_three",
            (1..=12).map(|i| vec![i as f64, (i + 5) as f64]).collect(),
            (1..=12)
                .map(|i| vec![i as f64, (i as f64).sqrt(), (i as f64).powi(2) / 4.0])
                .collect(),
        ),
    ];

    let points = fixtures
        .into_iter()
        .map(|(name, x, y)| PointCase {
            case_id: name.into(),
            x,
            y,
        })
        .collect();
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np

def mat_or_none(arr):
    out = []
    for row in arr:
        row_out = []
        for v in row:
            try:
                v = float(v)
            except Exception:
                return None
            if not math.isfinite(v):
                return None
            row_out.append(v)
        out.append(row_out)
    return out

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    x = np.array(case["x"], dtype=float)
    y = np.array(case["y"], dtype=float)
    val = None
    try:
        # Stack x and y column-wise, transpose so rows are variables.
        stacked = np.hstack([x, y]).T  # shape (dx + dy, n)
        full_cov = np.cov(stacked, ddof=1)
        dx = x.shape[1]; dy = y.shape[1]
        # Cross-covariance is the upper-right dx × dy block.
        block = full_cov[:dx, dx:dx + dy]
        val = mat_or_none(block.tolist())
    except Exception:
        val = None
    points.append({"case_id": cid, "matrix": val})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize cross_cov query");
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
                "failed to spawn python3 for cross_cov oracle: {e}"
            );
            eprintln!("skipping cross_cov oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open cross_cov oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "cross_cov oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping cross_cov oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for cross_cov oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "cross_cov oracle failed: {stderr}"
        );
        eprintln!("skipping cross_cov oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse cross_cov oracle JSON"))
}

#[test]
fn diff_stats_cross_cov() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(scipy_mat) = &scipy_arm.matrix else {
            continue;
        };
        let rust_mat = cross_cov(&case.x, &case.y);
        if rust_mat.len() != scipy_mat.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let mut max_local = 0.0_f64;
        let mut shape_ok = true;
        for (rrow, srow) in rust_mat.iter().zip(scipy_mat.iter()) {
            if rrow.len() != srow.len() {
                shape_ok = false;
                break;
            }
            for (a, b) in rrow.iter().zip(srow.iter()) {
                if a.is_finite() {
                    max_local = max_local.max((a - b).abs());
                }
            }
        }
        if !shape_ok {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        max_overall = max_overall.max(max_local);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: max_local,
            pass: max_local <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_cross_cov".into(),
        category: "cross_cov (numpy reference)".into(),
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
            eprintln!("cross_cov mismatch: {} abs={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "cross_cov conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
