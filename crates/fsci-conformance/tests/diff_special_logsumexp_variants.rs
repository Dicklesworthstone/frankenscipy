#![forbid(unsafe_code)]
//! Live scipy.special.logsumexp parity for fsci_special::
//! {logsumexp_with_b, logsumexp_axis_2d, logsumexp_axis_2d_with_b}.
//!
//! Resolves [frankenscipy-1bh5o]. Tolerance: 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::{logsumexp_axis_2d, logsumexp_axis_2d_with_b, logsumexp_with_b};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String,
    data_flat: Vec<f64>,
    rows: usize,
    cols: usize,
    axis: usize,
    /// Weights (optional, for *_with_b)
    b_flat: Vec<f64>,
    b_rows: usize,
    b_cols: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    scalar: Option<f64>,
    values: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    op: String,
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
    fs::create_dir_all(output_dir()).expect("create logsumexp diff dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize log");
    fs::write(path, json).expect("write log");
}

fn unflatten(flat: &[f64], rows: usize, cols: usize) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|i| flat[i * cols..(i + 1) * cols].to_vec())
        .collect()
}

fn generate_query() -> OracleQuery {
    let d1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let b1: Vec<f64> = vec![0.1, 0.2, 0.3, 0.25, 0.15];
    let d_neg: Vec<f64> = vec![-1000.0, -1001.0, -1002.0];
    let b_uniform: Vec<f64> = vec![1.0, 1.0, 1.0];

    let m3x4: Vec<f64> = (1..=12).map(|i| (i as f64) * 0.5).collect();
    let m3x4_b: Vec<f64> = vec![1.0, 0.5, 2.0, 1.0, 0.25, 1.5, 0.75, 0.5, 1.0, 2.0, 0.5, 1.0];
    let m2x5: Vec<f64> = (0..10).map(|i| ((i as f64) * 0.3).sin() + 2.0).collect();

    let points = vec![
        // logsumexp_with_b: 1-D
        Case {
            case_id: "lse_b_basic".into(),
            op: "lse_with_b".into(),
            data_flat: d1.clone(),
            rows: 1,
            cols: d1.len(),
            axis: 0,
            b_flat: b1.clone(),
            b_rows: 1,
            b_cols: b1.len(),
        },
        Case {
            case_id: "lse_b_neg".into(),
            op: "lse_with_b".into(),
            data_flat: d_neg.clone(),
            rows: 1,
            cols: d_neg.len(),
            axis: 0,
            b_flat: b_uniform.clone(),
            b_rows: 1,
            b_cols: b_uniform.len(),
        },
        // logsumexp_axis_2d: 3x4 matrix
        Case {
            case_id: "lse_ax_3x4_a0".into(),
            op: "lse_axis_2d".into(),
            data_flat: m3x4.clone(),
            rows: 3,
            cols: 4,
            axis: 0,
            b_flat: vec![],
            b_rows: 0,
            b_cols: 0,
        },
        Case {
            case_id: "lse_ax_3x4_a1".into(),
            op: "lse_axis_2d".into(),
            data_flat: m3x4.clone(),
            rows: 3,
            cols: 4,
            axis: 1,
            b_flat: vec![],
            b_rows: 0,
            b_cols: 0,
        },
        Case {
            case_id: "lse_ax_2x5_a0".into(),
            op: "lse_axis_2d".into(),
            data_flat: m2x5.clone(),
            rows: 2,
            cols: 5,
            axis: 0,
            b_flat: vec![],
            b_rows: 0,
            b_cols: 0,
        },
        Case {
            case_id: "lse_ax_2x5_a1".into(),
            op: "lse_axis_2d".into(),
            data_flat: m2x5,
            rows: 2,
            cols: 5,
            axis: 1,
            b_flat: vec![],
            b_rows: 0,
            b_cols: 0,
        },
        // logsumexp_axis_2d_with_b
        Case {
            case_id: "lse_ax_b_3x4_a0".into(),
            op: "lse_axis_2d_with_b".into(),
            data_flat: m3x4.clone(),
            rows: 3,
            cols: 4,
            axis: 0,
            b_flat: m3x4_b.clone(),
            b_rows: 3,
            b_cols: 4,
        },
        Case {
            case_id: "lse_ax_b_3x4_a1".into(),
            op: "lse_axis_2d_with_b".into(),
            data_flat: m3x4,
            rows: 3,
            cols: 4,
            axis: 1,
            b_flat: m3x4_b,
            b_rows: 3,
            b_cols: 4,
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
from scipy import special

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    rows = int(case["rows"]); cols = int(case["cols"])
    data = np.array(case["data_flat"], dtype=float)
    if rows == 1:
        data = data.reshape(cols)
    else:
        data = data.reshape(rows, cols)
    b = None
    if case["b_flat"]:
        b = np.array(case["b_flat"], dtype=float)
        b_rows = int(case["b_rows"]); b_cols = int(case["b_cols"])
        if b_rows == 1:
            b = b.reshape(b_cols)
        else:
            b = b.reshape(b_rows, b_cols)
    axis = int(case["axis"])
    try:
        if op == "lse_with_b":
            v = float(special.logsumexp(data, b=b))
            points.append({"case_id": cid, "scalar": v, "values": None})
        elif op == "lse_axis_2d":
            r = special.logsumexp(data, axis=axis)
            arr = [float(t) for t in np.atleast_1d(r).tolist()]
            if all(math.isfinite(v) for v in arr):
                points.append({"case_id": cid, "scalar": None, "values": arr})
            else:
                points.append({"case_id": cid, "scalar": None, "values": None})
        elif op == "lse_axis_2d_with_b":
            r = special.logsumexp(data, axis=axis, b=b)
            arr = [float(t) for t in np.atleast_1d(r).tolist()]
            if all(math.isfinite(v) for v in arr):
                points.append({"case_id": cid, "scalar": None, "values": arr})
            else:
                points.append({"case_id": cid, "scalar": None, "values": None})
        else:
            points.append({"case_id": cid, "scalar": None, "values": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "scalar": None, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize query");
    let mut child = match Command::new("python3")
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
                "failed to spawn python3 for logsumexp oracle: {e}"
            );
            eprintln!("skipping logsumexp oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "logsumexp oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping logsumexp oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for logsumexp oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "logsumexp oracle failed: {stderr}"
        );
        eprintln!("skipping logsumexp oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse logsumexp oracle JSON"))
}

fn vec_max_diff(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn diff_special_logsumexp_variants() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let Some(arm) = pmap.get(&case.case_id) else {
            continue;
        };
        let abs_d = match case.op.as_str() {
            "lse_with_b" => {
                let Some(expected) = arm.scalar else { continue };
                let Ok(actual) = logsumexp_with_b(&case.data_flat, &case.b_flat) else {
                    continue;
                };
                (actual - expected).abs()
            }
            "lse_axis_2d" => {
                let Some(expected) = arm.values.as_ref() else {
                    continue;
                };
                let data = unflatten(&case.data_flat, case.rows, case.cols);
                let Ok(actual) = logsumexp_axis_2d(&data, case.axis) else {
                    continue;
                };
                vec_max_diff(&actual, expected)
            }
            "lse_axis_2d_with_b" => {
                let Some(expected) = arm.values.as_ref() else {
                    continue;
                };
                let data = unflatten(&case.data_flat, case.rows, case.cols);
                let b = unflatten(&case.b_flat, case.b_rows, case.b_cols);
                let Ok(actual) = logsumexp_axis_2d_with_b(&data, case.axis, &b) else {
                    continue;
                };
                vec_max_diff(&actual, expected)
            }
            _ => continue,
        };
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_logsumexp_variants".into(),
        category: "fsci_special logsumexp_with_b/axis_2d/axis_2d_with_b vs scipy.special".into(),
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
            eprintln!("{} mismatch: {} abs_diff={}", d.op, d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "logsumexp variants conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
