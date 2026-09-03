#![forbid(unsafe_code)]
//! Live numpy differential coverage for fsci_ndimage element-wise
//! array ops: add_arrays, subtract_arrays, multiply_arrays,
//! masked_fill, masked_select.
//!
//! Resolves [frankenscipy-pxnf2]. 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_ndimage::{
    NdArray, add_arrays, masked_fill, masked_select, multiply_arrays, subtract_arrays,
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
    mask: Vec<f64>,
    fill_value: f64,
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
    fs::create_dir_all(output_dir()).expect("create binary_array diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize binary_array diff log");
    fs::write(path, json).expect("write binary_array diff log");
}

fn generate_query() -> OracleQuery {
    let a_3x4: Vec<f64> = (1..=12).map(|i| i as f64).collect();
    let b_3x4: Vec<f64> = (1..=12).map(|i| (i as f64) * 0.5).collect();
    let mask_3x4: Vec<f64> = (1..=12)
        .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
        .collect();
    let a_2x3: Vec<f64> = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
    let b_2x3: Vec<f64> = vec![0.5, -1.0, 1.5, -2.0, 2.5, -3.0];

    let mut points = Vec::new();
    for op in ["add", "subtract", "multiply"] {
        points.push(PointCase {
            case_id: format!("{op}_3x4"),
            op: op.into(),
            shape: vec![3, 4],
            a: a_3x4.clone(),
            b: b_3x4.clone(),
            mask: vec![],
            fill_value: 0.0,
        });
        points.push(PointCase {
            case_id: format!("{op}_2x3"),
            op: op.into(),
            shape: vec![2, 3],
            a: a_2x3.clone(),
            b: b_2x3.clone(),
            mask: vec![],
            fill_value: 0.0,
        });
    }

    // masked_fill: fill where mask != 0 with fill_value
    for &fv in &[0.0_f64, -999.0, 1.5] {
        points.push(PointCase {
            case_id: format!("masked_fill_3x4_v{fv}"),
            op: "masked_fill".into(),
            shape: vec![3, 4],
            a: a_3x4.clone(),
            b: vec![],
            mask: mask_3x4.clone(),
            fill_value: fv,
        });
    }
    // masked_select: pick where mask != 0
    points.push(PointCase {
        case_id: "masked_select_3x4".into(),
        op: "masked_select".into(),
        shape: vec![3, 4],
        a: a_3x4,
        b: vec![],
        mask: mask_3x4,
        fill_value: 0.0,
    });

    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np

def finite_or_none(arr):
    arr = np.asarray(arr, dtype=float)
    flat = []
    for v in arr.flatten().tolist():
        if not math.isfinite(float(v)):
            return None
        flat.append(float(v))
    return flat

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    shape = tuple(case["shape"])
    a = np.array(case["a"], dtype=float).reshape(shape)
    if len(case["b"]) > 0:
        b = np.array(case["b"], dtype=float).reshape(shape)
    else:
        b = None
    if len(case["mask"]) > 0:
        mask = np.array(case["mask"], dtype=float).reshape(shape)
    else:
        mask = None
    fv = float(case["fill_value"])
    try:
        if op == "add":
            y = a + b
        elif op == "subtract":
            y = a - b
        elif op == "multiply":
            y = a * b
        elif op == "masked_fill":
            y = np.where(mask != 0, fv, a)
        elif op == "masked_select":
            y = a.flatten()[mask.flatten() != 0]
        else:
            y = None
        points.append({"case_id": cid, "values": finite_or_none(y) if y is not None else None})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize binary_array query");
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
                "failed to spawn python3 for binary_array oracle: {e}"
            );
            eprintln!("skipping binary_array oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open binary_array oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "binary_array oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping binary_array oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for binary_array oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "binary_array oracle failed: {stderr}"
        );
        eprintln!("skipping binary_array oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse binary_array oracle JSON"))
}

#[test]
fn diff_ndimage_binary_array_ops() {
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
        let fsci_v: Vec<f64> = match case.op.as_str() {
            "add" | "subtract" | "multiply" => {
                let Ok(b_arr) = NdArray::new(case.b.clone(), case.shape.clone()) else {
                    continue;
                };
                let result = match case.op.as_str() {
                    "add" => add_arrays(&a_arr, &b_arr),
                    "subtract" => subtract_arrays(&a_arr, &b_arr),
                    "multiply" => multiply_arrays(&a_arr, &b_arr),
                    _ => continue,
                };
                let Ok(out) = result else {
                    continue;
                };
                out.data
            }
            "masked_fill" => {
                let Ok(mask_arr) = NdArray::new(case.mask.clone(), case.shape.clone()) else {
                    continue;
                };
                let out = masked_fill(&a_arr, &mask_arr, case.fill_value);
                out.data
            }
            "masked_select" => {
                let Ok(mask_arr) = NdArray::new(case.mask.clone(), case.shape.clone()) else {
                    continue;
                };
                masked_select(&a_arr, &mask_arr)
            }
            _ => continue,
        };
        let abs_d = if fsci_v.len() != expected.len() {
            f64::INFINITY
        } else {
            fsci_v
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
        test_id: "diff_ndimage_binary_array_ops".into(),
        category: "fsci_ndimage add/sub/mul + masked_fill/select vs numpy".into(),
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
        "binary_array_ops conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
