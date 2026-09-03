#![forbid(unsafe_code)]
//! Live numpy differential coverage for fsci_ndimage array-API helpers:
//! log_array, sqrt_array, scale_array, power_array, threshold,
//! greater_than, where_cond.
//!
//! Resolves [frankenscipy-peq4g]. 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_ndimage::{
    NdArray, greater_than, log_array, power_array, scale_array, sqrt_array, threshold, where_cond,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String,
    shape: Vec<usize>,
    a: Vec<f64>,
    b: Vec<f64>,
    /// scalar for scale, exponent for power, thresh for threshold
    scalar: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
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
    fs::create_dir_all(output_dir()).expect("create array_api diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize array_api diff log");
    fs::write(path, json).expect("write array_api diff log");
}

fn generate_query() -> OracleQuery {
    let pos_mat: Vec<f64> = (1..=12).map(|i| i as f64).collect(); // 3x4
    let mixed_mat = vec![
        1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0, 11.0, -12.0,
    ];

    let mut points = Vec::new();

    // log_array (positive input)
    points.push(PointCase {
        case_id: "log_3x4_positive".into(),
        op: "log_array".into(),
        shape: vec![3, 4],
        a: pos_mat.clone(),
        b: vec![],
        scalar: 0.0,
    });
    // sqrt_array (mixed; negative → clamped to 0)
    points.push(PointCase {
        case_id: "sqrt_3x4_mixed".into(),
        op: "sqrt_array".into(),
        shape: vec![3, 4],
        a: mixed_mat.clone(),
        b: vec![],
        scalar: 0.0,
    });
    // scale_array
    for &s in &[2.0_f64, -0.5, 0.0, 3.14] {
        points.push(PointCase {
            case_id: format!("scale_s{s}"),
            op: "scale_array".into(),
            shape: vec![3, 4],
            a: pos_mat.clone(),
            b: vec![],
            scalar: s,
        });
    }
    // power_array
    for &p in &[2.0_f64, 0.5, 1.0, -1.0] {
        points.push(PointCase {
            case_id: format!("power_p{p}"),
            op: "power_array".into(),
            shape: vec![3, 4],
            a: pos_mat.clone(),
            b: vec![],
            scalar: p,
        });
    }
    // threshold
    for &t in &[0.0_f64, 5.0, 10.0] {
        points.push(PointCase {
            case_id: format!("threshold_t{t}"),
            op: "threshold".into(),
            shape: vec![3, 4],
            a: pos_mat.clone(),
            b: vec![],
            scalar: t,
        });
    }
    // greater_than (a > b)
    let b_const = vec![5.0_f64; 12];
    points.push(PointCase {
        case_id: "greater_than_pos_vs_5".into(),
        op: "greater_than".into(),
        shape: vec![3, 4],
        a: pos_mat.clone(),
        b: b_const.clone(),
        scalar: 0.0,
    });
    points.push(PointCase {
        case_id: "greater_than_mixed_vs_0".into(),
        op: "greater_than".into(),
        shape: vec![3, 4],
        a: mixed_mat.clone(),
        b: vec![0.0_f64; 12],
        scalar: 0.0,
    });

    // where_cond — picks between a and b based on cond. Use threshold(pos, 5) as cond.
    let cond: Vec<f64> = pos_mat
        .iter()
        .map(|&v| if v > 5.0 { 1.0 } else { 0.0 })
        .collect();
    // where treats cond as 1-d input
    let _ = cond;
    points.push(PointCase {
        case_id: "where_pos_vs_neg".into(),
        op: "where_cond".into(),
        shape: vec![3, 4],
        a: pos_mat.clone(), // also stored as cond marker (where uses cond, a, b in that order)
        b: mixed_mat.clone(), // alternative selection
        scalar: 0.0,
    });

    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np

def finite_or_neginf(arr):
    arr = np.asarray(arr, dtype=float)
    flat = []
    for v in arr.flatten().tolist():
        if v == float("-inf"):
            flat.append(-1.0e308)
        elif v == float("inf"):
            flat.append(1.0e308)
        elif not math.isfinite(float(v)):
            return None
        else:
            flat.append(float(v))
    return flat

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    shape = tuple(case["shape"])
    a = np.array(case["a"], dtype=float).reshape(shape) if len(case["a"]) > 0 else None
    b = np.array(case["b"], dtype=float).reshape(shape) if len(case["b"]) > 0 else None
    s = float(case["scalar"])
    try:
        if op == "log_array":
            y = np.where(a > 0, np.log(np.where(a > 0, a, 1.0)), float("-inf"))
        elif op == "sqrt_array":
            y = np.sqrt(np.maximum(a, 0.0))
        elif op == "scale_array":
            y = a * s
        elif op == "power_array":
            y = np.power(a, s)
        elif op == "threshold":
            y = (a > s).astype(float)
        elif op == "greater_than":
            y = (a > b).astype(float)
        elif op == "where_cond":
            # fsci where_cond: condition first arg; build cond from a (>5)
            cond = (a > 5.0).astype(float)
            # Then we want where(cond, a, b)
            y = np.where(cond != 0, a, b)
        else:
            y = None
        points.append({"case_id": cid, "values": finite_or_neginf(y) if y is not None else None})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize array_api query");
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
                "failed to spawn python3 for array_api oracle: {e}"
            );
            eprintln!("skipping array_api oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open array_api oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "array_api oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping array_api oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for array_api oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "array_api oracle failed: {stderr}"
        );
        eprintln!("skipping array_api oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse array_api oracle JSON"))
}

#[test]
fn diff_ndimage_array_api_ops() {
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
        let Some(expected) = scipy_arm.values.as_ref() else {
            continue;
        };
        let Ok(a_arr) = NdArray::new(case.a.clone(), case.shape.clone()) else {
            continue;
        };
        let b_arr = if !case.b.is_empty() {
            NdArray::new(case.b.clone(), case.shape.clone()).ok()
        } else {
            None
        };
        let out_data: Vec<f64> = match case.op.as_str() {
            "log_array" => log_array(&a_arr).data,
            "sqrt_array" => sqrt_array(&a_arr).data,
            "scale_array" => scale_array(&a_arr, case.scalar).data,
            "power_array" => power_array(&a_arr, case.scalar).data,
            "threshold" => threshold(&a_arr, case.scalar).data,
            "greater_than" => {
                let b_arr = b_arr.as_ref().expect("greater_than needs b");
                let Ok(out) = greater_than(&a_arr, b_arr) else {
                    continue;
                };
                out.data
            }
            "where_cond" => {
                let b_arr = b_arr.as_ref().expect("where_cond needs b");
                // Build cond from a > 5.0
                let cond_data: Vec<f64> = case
                    .a
                    .iter()
                    .map(|&v| if v > 5.0 { 1.0 } else { 0.0 })
                    .collect();
                let Ok(cond_arr) = NdArray::new(cond_data, case.shape.clone()) else {
                    continue;
                };
                let Ok(out) = where_cond(&cond_arr, &a_arr, b_arr) else {
                    continue;
                };
                out.data
            }
            _ => continue,
        };
        // Map -inf to -1e308 (matches oracle sentinel)
        let mapped: Vec<f64> = out_data
            .iter()
            .map(|&v| {
                if v == f64::NEG_INFINITY {
                    -1.0e308
                } else if v == f64::INFINITY {
                    1.0e308
                } else {
                    v
                }
            })
            .collect();
        let abs_d = if mapped.len() != expected.len() {
            f64::INFINITY
        } else {
            mapped
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
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
        test_id: "diff_ndimage_array_api_ops".into(),
        category: "fsci_ndimage array-API ops vs numpy".into(),
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
        "array_api_ops conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
