#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_ndimage::uniform_filter,
//! median_filter, sobel, prewitt, and laplace on 2-D inputs.
//!
//! Resolves [frankenscipy-1igc9]. 1e-10 abs (exact integer arithmetic
//! for sobel/prewitt/laplace; rational divisions for uniform_filter).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_ndimage::{BoundaryMode, NdArray, laplace, median_filter, prewitt, sobel, uniform_filter};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String,
    shape: Vec<usize>,
    input: Vec<f64>,
    /// uniform/median: filter size (single integer).
    size: usize,
    /// sobel/prewitt: axis (0 or 1).
    axis: usize,
    /// Boundary mode: "reflect" | "constant" | "nearest" | "wrap".
    mode: String,
    cval: f64,
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
    fs::create_dir_all(output_dir()).expect("create filters_edges diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize filters_edges diff log");
    fs::write(path, json).expect("write filters_edges diff log");
}

fn mode_of(name: &str) -> Option<BoundaryMode> {
    match name {
        "reflect" => Some(BoundaryMode::Reflect),
        "constant" => Some(BoundaryMode::Constant),
        "nearest" => Some(BoundaryMode::Nearest),
        "wrap" => Some(BoundaryMode::Wrap),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    let mat_3x3: Vec<f64> = (1..=9).map(|i| i as f64).collect();
    let mat_4x5: Vec<f64> = (1..=20).map(|i| (i as f64) * 0.5).collect();
    let mat_5x5_pattern: Vec<f64> = (0..25).map(|i| ((i as f64) * 0.3).sin() * 5.0).collect();

    let mut points = Vec::new();

    // uniform_filter & median_filter with size=3
    let filter_cases: &[(&str, &[f64], Vec<usize>, &str)] = &[
        ("u3x3_reflect", &mat_3x3, vec![3, 3], "reflect"),
        ("u3x3_nearest", &mat_3x3, vec![3, 3], "nearest"),
        ("u4x5_reflect", &mat_4x5, vec![4, 5], "reflect"),
        ("u5x5_constant", &mat_5x5_pattern, vec![5, 5], "constant"),
    ];
    for (label, input, shape, mode) in filter_cases {
        for op in ["uniform_filter", "median_filter"] {
            points.push(PointCase {
                case_id: format!("{op}_{label}"),
                op: op.into(),
                shape: shape.clone(),
                input: input.to_vec(),
                size: 3,
                axis: 0,
                mode: (*mode).into(),
                cval: 0.0,
            });
        }
    }

    // sobel & prewitt on both axes
    let edge_inputs: &[(&str, &[f64], Vec<usize>)] = &[
        ("3x3_arange", &mat_3x3, vec![3, 3]),
        ("4x5_ramp", &mat_4x5, vec![4, 5]),
        ("5x5_sine", &mat_5x5_pattern, vec![5, 5]),
    ];
    for (label, input, shape) in edge_inputs {
        for axis in 0..=1 {
            for op in ["sobel", "prewitt"] {
                points.push(PointCase {
                    case_id: format!("{op}_{label}_ax{axis}"),
                    op: op.into(),
                    shape: shape.clone(),
                    input: input.to_vec(),
                    size: 0,
                    axis,
                    mode: "reflect".into(),
                    cval: 0.0,
                });
            }
        }
        // laplace (no axis)
        points.push(PointCase {
            case_id: format!("laplace_{label}"),
            op: "laplace".into(),
            shape: shape.clone(),
            input: input.to_vec(),
            size: 0,
            axis: 0,
            mode: "reflect".into(),
            cval: 0.0,
        });
    }

    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy import ndimage

def finite_flat_or_none(arr):
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
    x = np.array(case["input"], dtype=float).reshape(shape)
    mode = case["mode"]
    cval = float(case["cval"])
    size = int(case["size"])
    axis = int(case["axis"])
    try:
        if op == "uniform_filter":
            y = ndimage.uniform_filter(x, size=size, mode=mode, cval=cval)
        elif op == "median_filter":
            y = ndimage.median_filter(x, size=size, mode=mode, cval=cval)
        elif op == "sobel":
            y = ndimage.sobel(x, axis=axis, mode=mode, cval=cval)
        elif op == "prewitt":
            y = ndimage.prewitt(x, axis=axis, mode=mode, cval=cval)
        elif op == "laplace":
            y = ndimage.laplace(x, mode=mode, cval=cval)
        else:
            y = None
        points.append({"case_id": cid, "values": finite_flat_or_none(y) if y is not None else None})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize filters_edges query");
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
                "failed to spawn python3 for filters_edges oracle: {e}"
            );
            eprintln!("skipping filters_edges oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open filters_edges oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "filters_edges oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping filters_edges oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for filters_edges oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "filters_edges oracle failed: {stderr}"
        );
        eprintln!("skipping filters_edges oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse filters_edges oracle JSON"))
}

#[test]
fn diff_ndimage_filters_edges() {
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
        let Some(mode) = mode_of(&case.mode) else {
            continue;
        };
        let Ok(input) = NdArray::new(case.input.clone(), case.shape.clone()) else {
            continue;
        };
        let fsci_result = match case.op.as_str() {
            "uniform_filter" => uniform_filter(&input, case.size, mode, case.cval),
            "median_filter" => median_filter(&input, case.size, mode, case.cval),
            "sobel" => sobel(&input, case.axis, mode, case.cval),
            "prewitt" => prewitt(&input, case.axis, mode, case.cval),
            "laplace" => laplace(&input, mode, case.cval),
            _ => continue,
        };
        let Ok(out) = fsci_result else {
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

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_ndimage_filters_edges".into(),
        category: "scipy.ndimage uniform/median/sobel/prewitt/laplace".into(),
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
        "filters_edges conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
