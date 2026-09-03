#![forbid(unsafe_code)]
//! Live SciPy differential coverage for
//! `scipy.optimize.linear_sum_assignment(cost_matrix)` — the
//! Hungarian/Kuhn–Munkres algorithm.
//!
//! Resolves [frankenscipy-dtc34]. fsci_opt::linear_sum_assignment is
//! exposed via lib.rs and matches scipy on probe inputs.
//!
//! Note: when multiple assignments achieve the optimal total cost,
//! the chosen (row, col) tuples can differ between implementations.
//! We compare on the total assignment cost (always unique) rather
//! than the column indices.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_opt::linear_sum_assignment;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-015";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    cost: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    total_cost: Option<f64>,
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
    fs::create_dir_all(output_dir()).expect("create lsap diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize lsap diff log");
    fs::write(path, json).expect("write lsap diff log");
}

fn generate_query() -> OracleQuery {
    let matrices: &[(&str, Vec<Vec<f64>>)] = &[
        (
            "3x3_textbook",
            vec![
                vec![4.0, 1.0, 3.0],
                vec![2.0, 0.0, 5.0],
                vec![3.0, 2.0, 2.0],
            ],
        ),
        (
            "4x4_workers",
            vec![
                vec![10.0, 19.0, 8.0, 15.0],
                vec![10.0, 18.0, 7.0, 17.0],
                vec![13.0, 16.0, 9.0, 14.0],
                vec![12.0, 19.0, 8.0, 18.0],
            ],
        ),
        ("2x2_trivial", vec![vec![1.0, 5.0], vec![3.0, 2.0]]),
        (
            "5x5_identity_min",
            // Diagonal is the unique minimum; expect col[i] = i.
            (0..5)
                .map(|i| (0..5).map(|j| if i == j { 1.0 } else { 10.0 }).collect())
                .collect(),
        ),
        (
            "3x5_rectangular_wider",
            // 3 rows × 5 cols — scipy returns the 3 best column assignments.
            vec![
                vec![5.0, 1.0, 4.0, 9.0, 7.0],
                vec![2.0, 8.0, 3.0, 6.0, 0.0],
                vec![4.0, 2.0, 1.0, 5.0, 3.0],
            ],
        ),
        (
            "4x3_rectangular_taller",
            vec![
                vec![5.0, 1.0, 4.0],
                vec![2.0, 8.0, 3.0],
                vec![6.0, 0.0, 7.0],
                vec![4.0, 2.0, 1.0],
            ],
        ),
        ("1x1", vec![vec![42.0]]),
        (
            "negative_costs",
            vec![
                vec![-5.0, -1.0, -3.0],
                vec![-2.0, -8.0, -5.0],
                vec![-3.0, -2.0, -7.0],
            ],
        ),
    ];
    let points = matrices
        .iter()
        .map(|(name, m)| PointCase {
            case_id: (*name).to_string(),
            cost: m.clone(),
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
from scipy.optimize import linear_sum_assignment

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    try:
        cost = np.array(case["cost"], dtype=float)
        rows, cols = linear_sum_assignment(cost)
        total = float(cost[rows, cols].sum())
        points.append({"case_id": cid, "total_cost": fnone(total)})
    except Exception:
        points.append({"case_id": cid, "total_cost": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize lsap query");
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
                "failed to spawn python3 for lsap oracle: {e}"
            );
            eprintln!("skipping lsap oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open lsap oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "lsap oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping lsap oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for lsap oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "lsap oracle failed: {stderr}"
        );
        eprintln!("skipping lsap oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse lsap oracle JSON"))
}

fn fsci_total_cost(cost: &[Vec<f64>]) -> Option<f64> {
    let (rows, cols) = linear_sum_assignment(cost).ok()?;
    let mut total = 0.0_f64;
    for (&r, &c) in rows.iter().zip(cols.iter()) {
        total += cost[r][c];
    }
    Some(total)
}

#[test]
fn diff_opt_linear_sum_assignment() {
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
        let Some(fsci_total) = fsci_total_cost(&case.cost) else {
            continue;
        };
        if let Some(scipy_total) = scipy_arm.total_cost {
            let abs_d = (fsci_total - scipy_total).abs();
            max_overall = max_overall.max(abs_d);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                abs_diff: abs_d,
                pass: abs_d <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_linear_sum_assignment".into(),
        category: "scipy.optimize.linear_sum_assignment".into(),
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
                "lsap total-cost mismatch: {} abs_diff={}",
                d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.optimize.linear_sum_assignment conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
