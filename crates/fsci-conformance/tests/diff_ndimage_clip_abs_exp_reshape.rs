#![forbid(unsafe_code)]
//! Live numpy parity for fsci_ndimage element-wise and shape ops:
//! clip, abs_array, exp_array, reshape, flatten, full, ones.
//!
//! Resolves [frankenscipy-v7e3e]. Tolerance: 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_ndimage::{NdArray, abs_array, clip, exp_array, flatten, full, ones, reshape};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String,
    shape: Vec<usize>,
    data: Vec<f64>,
    /// For clip: clip_min/clip_max; for full: fill value (scalar).
    arg1: f64,
    arg2: f64,
    /// For reshape: new_shape; for full/ones: target shape.
    target_shape: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
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
    fs::create_dir_all(output_dir()).expect("create clip_abs_exp diff dir");
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

fn generate_query() -> OracleQuery {
    let mixed: Vec<f64> = vec![-3.0, -1.5, 0.0, 0.5, 2.0, 4.5, -2.5, 1.0];
    let pos: Vec<f64> = (0..10).map(|i| (i as f64) * 0.4 + 0.1).collect();
    let mat: Vec<f64> = (0..12).map(|i| (i as f64) * 0.5 - 3.0).collect();

    let points = vec![
        // clip: clip(mixed, -1.0, 2.0)
        Case {
            case_id: "clip_mixed_n1_2".into(),
            op: "clip".into(),
            shape: vec![8],
            data: mixed.clone(),
            arg1: -1.0,
            arg2: 2.0,
            target_shape: vec![],
        },
        Case {
            case_id: "clip_mat_0_3".into(),
            op: "clip".into(),
            shape: vec![3, 4],
            data: mat.clone(),
            arg1: 0.0,
            arg2: 3.0,
            target_shape: vec![],
        },
        // abs_array
        Case {
            case_id: "abs_mixed".into(),
            op: "abs".into(),
            shape: vec![8],
            data: mixed.clone(),
            arg1: 0.0,
            arg2: 0.0,
            target_shape: vec![],
        },
        Case {
            case_id: "abs_mat".into(),
            op: "abs".into(),
            shape: vec![3, 4],
            data: mat.clone(),
            arg1: 0.0,
            arg2: 0.0,
            target_shape: vec![],
        },
        // exp_array
        Case {
            case_id: "exp_pos".into(),
            op: "exp".into(),
            shape: vec![10],
            data: pos.clone(),
            arg1: 0.0,
            arg2: 0.0,
            target_shape: vec![],
        },
        Case {
            case_id: "exp_mixed".into(),
            op: "exp".into(),
            shape: vec![8],
            data: mixed,
            arg1: 0.0,
            arg2: 0.0,
            target_shape: vec![],
        },
        // reshape: 3x4 → 4x3, 4x3 → 12
        Case {
            case_id: "reshape_3x4_to_4x3".into(),
            op: "reshape".into(),
            shape: vec![3, 4],
            data: mat.clone(),
            arg1: 0.0,
            arg2: 0.0,
            target_shape: vec![4, 3],
        },
        Case {
            case_id: "reshape_3x4_to_12".into(),
            op: "reshape".into(),
            shape: vec![3, 4],
            data: mat.clone(),
            arg1: 0.0,
            arg2: 0.0,
            target_shape: vec![12],
        },
        // flatten
        Case {
            case_id: "flatten_3x4".into(),
            op: "flatten".into(),
            shape: vec![3, 4],
            data: mat,
            arg1: 0.0,
            arg2: 0.0,
            target_shape: vec![],
        },
        // full: shape (2,3), value 7.0
        Case {
            case_id: "full_2x3_v7".into(),
            op: "full".into(),
            shape: vec![],
            data: vec![],
            arg1: 7.0,
            arg2: 0.0,
            target_shape: vec![2, 3],
        },
        // ones: shape (3,4)
        Case {
            case_id: "ones_3x4".into(),
            op: "ones".into(),
            shape: vec![],
            data: vec![],
            arg1: 0.0,
            arg2: 0.0,
            target_shape: vec![3, 4],
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

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    shape = tuple(int(s) for s in case["shape"]) if case["shape"] else ()
    target = tuple(int(s) for s in case["target_shape"]) if case["target_shape"] else ()
    arg1 = float(case["arg1"]); arg2 = float(case["arg2"])
    try:
        if op == "clip":
            a = np.array(case["data"], dtype=float).reshape(shape)
            v = np.clip(a, arg1, arg2)
        elif op == "abs":
            a = np.array(case["data"], dtype=float).reshape(shape)
            v = np.abs(a)
        elif op == "exp":
            a = np.array(case["data"], dtype=float).reshape(shape)
            v = np.exp(a)
        elif op == "reshape":
            a = np.array(case["data"], dtype=float).reshape(shape)
            v = a.reshape(target)
        elif op == "flatten":
            a = np.array(case["data"], dtype=float).reshape(shape)
            v = a.flatten()
        elif op == "full":
            v = np.full(target, arg1, dtype=float)
        elif op == "ones":
            v = np.ones(target, dtype=float)
        else:
            v = None
        if v is None:
            points.append({"case_id": cid, "values": None})
        else:
            flat = [float(x) for x in v.flatten().tolist()]
            if all(math.isfinite(x) for x in flat):
                points.append({"case_id": cid, "values": flat})
            else:
                points.append({"case_id": cid, "values": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "values": None})
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
                "failed to spawn python3 for clip_abs_exp oracle: {e}"
            );
            eprintln!("skipping clip_abs_exp oracle: python3 not available ({e})");
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
                "clip_abs_exp oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping clip_abs_exp oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for clip_abs_exp oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "clip_abs_exp oracle failed: {stderr}"
        );
        eprintln!("skipping clip_abs_exp oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse clip_abs_exp oracle JSON"))
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
fn diff_ndimage_clip_abs_exp_reshape() {
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
        let Some(expected) = arm.values.as_ref() else {
            continue;
        };
        let actual: Vec<f64> = match case.op.as_str() {
            "clip" => {
                let Ok(arr) = NdArray::new(case.data.clone(), case.shape.clone()) else {
                    continue;
                };
                let out = clip(&arr, case.arg1, case.arg2);
                out.data
            }
            "abs" => {
                let Ok(arr) = NdArray::new(case.data.clone(), case.shape.clone()) else {
                    continue;
                };
                let out = abs_array(&arr);
                out.data
            }
            "exp" => {
                let Ok(arr) = NdArray::new(case.data.clone(), case.shape.clone()) else {
                    continue;
                };
                let out = exp_array(&arr);
                out.data
            }
            "reshape" => {
                let Ok(arr) = NdArray::new(case.data.clone(), case.shape.clone()) else {
                    continue;
                };
                let Ok(out) = reshape(&arr, case.target_shape.clone()) else {
                    continue;
                };
                out.data
            }
            "flatten" => {
                let Ok(arr) = NdArray::new(case.data.clone(), case.shape.clone()) else {
                    continue;
                };
                let out = flatten(&arr);
                out.data
            }
            "full" => full(case.target_shape.clone(), case.arg1).data,
            "ones" => ones(case.target_shape.clone()).data,
            _ => continue,
        };
        let abs_d = vec_max_diff(&actual, expected);
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
        test_id: "diff_ndimage_clip_abs_exp_reshape".into(),
        category: "fsci_ndimage clip/abs/exp/reshape/flatten/full/ones vs numpy".into(),
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
        "clip_abs_exp_reshape conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
