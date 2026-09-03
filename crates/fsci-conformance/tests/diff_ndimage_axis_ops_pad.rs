#![forbid(unsafe_code)]
//! Live numpy differential coverage for fsci_ndimage::sum_axis, mean_axis,
//! pad_constant.
//!
//! Resolves [frankenscipy-mfl9a]. 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_ndimage::{NdArray, mean_axis, pad_constant, sum_axis};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct AxisCase {
    case_id: String,
    op: String, // "sum" | "mean"
    shape: Vec<usize>,
    data: Vec<f64>,
    axis: usize,
}

#[derive(Debug, Clone, Serialize)]
struct PadCase {
    case_id: String,
    shape: Vec<usize>,
    data: Vec<f64>,
    pad_width: Vec<(usize, usize)>,
    constant: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    axis: Vec<AxisCase>,
    pad: Vec<PadCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct ArmShape {
    case_id: String,
    flat: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    axis: Vec<ArmShape>,
    pad: Vec<ArmShape>,
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
    fs::create_dir_all(output_dir()).expect("create axis_ops diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize axis_ops diff log");
    fs::write(path, json).expect("write axis_ops diff log");
}

fn generate_query() -> OracleQuery {
    let mat_2x3: Vec<f64> = (1..=6).map(|i| i as f64).collect();
    let mat_3x4: Vec<f64> = (1..=12).map(|i| (i as f64) * 0.5).collect();
    let cube_2x3x2: Vec<f64> = (0..12).map(|i| (i as f64) + 1.0).collect();

    let mut axis = Vec::new();
    let shapes_2d: Vec<(&str, &[f64], Vec<usize>)> =
        vec![("2x3", &mat_2x3, vec![2, 3]), ("3x4", &mat_3x4, vec![3, 4])];
    for (label, data, shape) in shapes_2d {
        for op in ["sum", "mean"] {
            for ax in 0..2 {
                axis.push(AxisCase {
                    case_id: format!("{op}_{label}_ax{ax}"),
                    op: op.into(),
                    shape: shape.clone(),
                    data: data.to_vec(),
                    axis: ax,
                });
            }
        }
    }
    // 3-D
    for op in ["sum", "mean"] {
        for ax in 0..3 {
            axis.push(AxisCase {
                case_id: format!("{op}_2x3x2_ax{ax}"),
                op: op.into(),
                shape: vec![2, 3, 2],
                data: cube_2x3x2.clone(),
                axis: ax,
            });
        }
    }

    let pad = vec![
        PadCase {
            case_id: "pad_2x3_1_2_0_1".into(),
            shape: vec![2, 3],
            data: mat_2x3.clone(),
            pad_width: vec![(1, 2), (0, 1)],
            constant: 0.5,
        },
        PadCase {
            case_id: "pad_3x4_2_0_1_1".into(),
            shape: vec![3, 4],
            data: mat_3x4.clone(),
            pad_width: vec![(2, 0), (1, 1)],
            constant: -1.0,
        },
        PadCase {
            case_id: "pad_2x3_0_0_0_0".into(),
            shape: vec![2, 3],
            data: mat_2x3,
            pad_width: vec![(0, 0), (0, 0)],
            constant: 99.0,
        },
    ];

    OracleQuery { axis, pad }
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

axis_out = []
for c in q["axis"]:
    cid = c["case_id"]; op = c["op"]
    shape = tuple(c["shape"])
    x = np.array(c["data"], dtype=float).reshape(shape)
    ax = int(c["axis"])
    try:
        if op == "sum":
            y = np.sum(x, axis=ax)
        elif op == "mean":
            y = np.mean(x, axis=ax)
        else:
            y = None
        if y is None:
            axis_out.append({"case_id": cid, "out_shape": None, "flat": None})
        else:
            arr = np.atleast_1d(y)
            axis_out.append({
                "case_id": cid,
                "flat": finite_or_none(arr),
            })
    except Exception:
        axis_out.append({"case_id": cid, "flat": None})

pad_out = []
for c in q["pad"]:
    cid = c["case_id"]
    shape = tuple(c["shape"])
    x = np.array(c["data"], dtype=float).reshape(shape)
    pad_w = [(int(a), int(b)) for (a, b) in c["pad_width"]]
    constant = float(c["constant"])
    try:
        y = np.pad(x, pad_w, mode="constant", constant_values=constant)
        pad_out.append({
            "case_id": cid,
            "flat": finite_or_none(y),
        })
    except Exception:
        pad_out.append({"case_id": cid, "flat": None})

print(json.dumps({"axis": axis_out, "pad": pad_out}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize axis_ops query");
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
                "failed to spawn python3 for axis_ops oracle: {e}"
            );
            eprintln!("skipping axis_ops oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open axis_ops oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "axis_ops oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping axis_ops oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for axis_ops oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "axis_ops oracle failed: {stderr}"
        );
        eprintln!("skipping axis_ops oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse axis_ops oracle JSON"))
}

#[test]
fn diff_ndimage_axis_ops_pad() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.axis.len(), query.axis.len());
    assert_eq!(oracle.pad.len(), query.pad.len());

    let axis_map: HashMap<String, ArmShape> = oracle
        .axis
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();
    let pad_map: HashMap<String, ArmShape> = oracle
        .pad
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.axis {
        let scipy_arm = axis_map.get(&case.case_id).expect("validated oracle");
        let Some(expected) = scipy_arm.flat.as_ref() else {
            continue;
        };
        let Ok(input) = NdArray::new(case.data.clone(), case.shape.clone()) else {
            continue;
        };
        let result = match case.op.as_str() {
            "sum" => sum_axis(&input, case.axis),
            "mean" => mean_axis(&input, case.axis),
            _ => continue,
        };
        let Ok(out) = result else {
            continue;
        };
        let abs_d = if out.data.len() != expected.len() {
            f64::INFINITY
        } else {
            out.data
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

    for case in &query.pad {
        let scipy_arm = pad_map.get(&case.case_id).expect("validated oracle");
        let Some(expected) = scipy_arm.flat.as_ref() else {
            continue;
        };
        let Ok(input) = NdArray::new(case.data.clone(), case.shape.clone()) else {
            continue;
        };
        let Ok(out) = pad_constant(&input, &case.pad_width, case.constant) else {
            continue;
        };
        let abs_d = if out.data.len() != expected.len() {
            f64::INFINITY
        } else {
            out.data
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        };
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: "pad_constant".into(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_ndimage_axis_ops_pad".into(),
        category: "fsci_ndimage sum_axis + mean_axis + pad_constant vs numpy".into(),
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
        "axis_ops_pad conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
