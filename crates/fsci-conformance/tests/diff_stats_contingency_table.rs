#![forbid(unsafe_code)]
//! Live numerical reference checks for fsci's
//! `contingency_table(x, y) → (table, row_labels, col_labels)`.
//!
//! Resolves [frankenscipy-asjrv]. The oracle reproduces the
//! same result in numpy: `np.unique` for the row/col labels
//! and a count loop for the cell counts.
//!
//! 4 (x, y) categorical fixtures × 3 arms (table cells +
//! row_labels + col_labels) = 12 cases via subprocess. Tol
//! 1e-12 abs (integer counts).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::contingency_table;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    x: Vec<u64>,
    y: Vec<u64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    table: Option<Vec<Vec<i64>>>,
    row_labels: Option<Vec<i64>>,
    col_labels: Option<Vec<i64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    arm: String,
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
    fs::create_dir_all(output_dir()).expect("create contingency_table diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize contingency_table diff log");
    fs::write(path, json).expect("write contingency_table diff log");
}

fn generate_query() -> OracleQuery {
    let fixtures: Vec<(&str, Vec<u64>, Vec<u64>)> = vec![
        // 2 row classes × 2 col classes
        (
            "small_2x2",
            vec![0, 0, 1, 1, 0, 1, 1, 0, 0, 1],
            vec![0, 1, 0, 1, 1, 0, 0, 1, 0, 1],
        ),
        // 3 row × 3 col
        (
            "medium_3x3",
            vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
            vec![0, 1, 2, 1, 0, 2, 0, 2, 1, 1, 1, 0],
        ),
        // Sparse: only some (x, y) combinations appear
        ("sparse", vec![0, 0, 0, 2, 2, 5], vec![1, 1, 3, 1, 3, 1]),
        // 4 row × 2 col
        (
            "tall_4x2",
            vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
            vec![0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1],
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

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    x = np.array(case["x"], dtype=int)
    y = np.array(case["y"], dtype=int)
    out = {"case_id": cid, "table": None, "row_labels": None, "col_labels": None}
    try:
        rows = sorted(set(x.tolist()))
        cols = sorted(set(y.tolist()))
        nr = len(rows); nc = len(cols)
        row_idx = {v: i for i, v in enumerate(rows)}
        col_idx = {v: i for i, v in enumerate(cols)}
        table = [[0] * nc for _ in range(nr)]
        for xi, yi in zip(x.tolist(), y.tolist()):
            table[row_idx[xi]][col_idx[yi]] += 1
        out["table"] = table
        out["row_labels"] = rows
        out["col_labels"] = cols
    except Exception:
        pass
    points.append(out)
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize contingency_table query");
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
                "failed to spawn python3 for contingency_table oracle: {e}"
            );
            eprintln!("skipping contingency_table oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open contingency_table oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "contingency_table oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping contingency_table oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for contingency_table oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "contingency_table oracle failed: {stderr}"
        );
        eprintln!("skipping contingency_table oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse contingency_table oracle JSON"))
}

#[test]
fn diff_stats_contingency_table() {
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
        let x_usize: Vec<usize> = case.x.iter().map(|&v| v as usize).collect();
        let y_usize: Vec<usize> = case.y.iter().map(|&v| v as usize).collect();
        let (rust_table, rust_rows, rust_cols) = contingency_table(&x_usize, &y_usize);

        // table
        if let Some(scipy_table) = &scipy_arm.table {
            let mut max_local = 0.0_f64;
            let mut shape_ok = rust_table.len() == scipy_table.len();
            if shape_ok {
                for (rrow, srow) in rust_table.iter().zip(scipy_table.iter()) {
                    if rrow.len() != srow.len() {
                        shape_ok = false;
                        break;
                    }
                    for (a, b) in rrow.iter().zip(srow.iter()) {
                        let abs = (*a as i64 - b).unsigned_abs() as f64;
                        max_local = max_local.max(abs);
                    }
                }
            }
            max_overall = max_overall.max(max_local);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "table".into(),
                abs_diff: if shape_ok { max_local } else { f64::INFINITY },
                pass: shape_ok && max_local <= ABS_TOL,
            });
        }

        // row_labels
        if let Some(scipy_rows) = &scipy_arm.row_labels {
            let mut max_local = 0.0_f64;
            let shape_ok = rust_rows.len() == scipy_rows.len();
            if shape_ok {
                for (a, b) in rust_rows.iter().zip(scipy_rows.iter()) {
                    let abs = (*a as i64 - b).unsigned_abs() as f64;
                    max_local = max_local.max(abs);
                }
            }
            max_overall = max_overall.max(max_local);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "row_labels".into(),
                abs_diff: if shape_ok { max_local } else { f64::INFINITY },
                pass: shape_ok && max_local <= ABS_TOL,
            });
        }

        // col_labels
        if let Some(scipy_cols) = &scipy_arm.col_labels {
            let mut max_local = 0.0_f64;
            let shape_ok = rust_cols.len() == scipy_cols.len();
            if shape_ok {
                for (a, b) in rust_cols.iter().zip(scipy_cols.iter()) {
                    let abs = (*a as i64 - b).unsigned_abs() as f64;
                    max_local = max_local.max(abs);
                }
            }
            max_overall = max_overall.max(max_local);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "col_labels".into(),
                abs_diff: if shape_ok { max_local } else { f64::INFINITY },
                pass: shape_ok && max_local <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_contingency_table".into(),
        category: "contingency_table (numpy reference)".into(),
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
                "contingency_table mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "contingency_table conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
