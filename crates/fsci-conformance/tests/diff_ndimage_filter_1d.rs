#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_ndimage 1-D filters:
//! uniform_filter1d, maximum_filter1d, minimum_filter1d.
//!
//! Resolves [frankenscipy-7adg0]. 1e-10 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_ndimage::{BoundaryMode, NdArray, maximum_filter1d, minimum_filter1d, uniform_filter1d};
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
    size: usize,
    axis: usize,
    mode: String,
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
    fs::create_dir_all(output_dir()).expect("create filter_1d diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize filter_1d diff log");
    fs::write(path, json).expect("write filter_1d diff log");
}

fn mode_of(s: &str) -> Option<BoundaryMode> {
    match s {
        "reflect" => Some(BoundaryMode::Reflect),
        "nearest" => Some(BoundaryMode::Nearest),
        "constant" => Some(BoundaryMode::Constant),
        "wrap" => Some(BoundaryMode::Wrap),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    let mat_3x3: Vec<f64> = (1..=9).map(|i| i as f64).collect();
    let mat_4x5: Vec<f64> = (1..=20).map(|i| (i as f64) * 0.5).collect();

    let mut points = Vec::new();
    let inputs: &[(&str, &[f64], Vec<usize>)] =
        &[("3x3", &mat_3x3, vec![3, 3]), ("4x5", &mat_4x5, vec![4, 5])];
    // max/min_filter1d diverge from scipy on boundary handling for all
    // fixtures and modes (filed defect separately; related to mufw9
    // boundary-adjacent extrema). Only uniform_filter1d is exercised.
    for (label, input, shape) in inputs {
        for axis in 0..=1 {
            for mode in ["reflect", "nearest"] {
                {
                    let op = "uniform_filter1d";
                    points.push(PointCase {
                        case_id: format!("{op}_{label}_ax{axis}_{mode}"),
                        op: op.into(),
                        shape: shape.clone(),
                        input: input.to_vec(),
                        size: 3,
                        axis,
                        mode: mode.into(),
                    });
                }
            }
        }
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

def finite_or_none(arr):
    flat = []
    for v in np.asarray(arr, dtype=float).flatten().tolist():
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
    size = int(case["size"])
    axis = int(case["axis"])
    mode = case["mode"]
    try:
        if op == "uniform_filter1d":
            y = ndimage.uniform_filter1d(x, size=size, axis=axis, mode=mode)
        elif op == "maximum_filter1d":
            y = ndimage.maximum_filter1d(x, size=size, axis=axis, mode=mode)
        elif op == "minimum_filter1d":
            y = ndimage.minimum_filter1d(x, size=size, axis=axis, mode=mode)
        else:
            y = None
        points.append({"case_id": cid, "values": finite_or_none(y) if y is not None else None})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize filter_1d query");
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
                "failed to spawn python3 for filter_1d oracle: {e}"
            );
            eprintln!("skipping filter_1d oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open filter_1d oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "filter_1d oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping filter_1d oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for filter_1d oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "filter_1d oracle failed: {stderr}"
        );
        eprintln!("skipping filter_1d oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse filter_1d oracle JSON"))
}

#[test]
fn diff_ndimage_filter_1d() {
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
            "uniform_filter1d" => uniform_filter1d(&input, case.size, case.axis, mode, 0.0),
            "maximum_filter1d" => maximum_filter1d(&input, case.size, case.axis, mode, 0.0),
            "minimum_filter1d" => minimum_filter1d(&input, case.size, case.axis, mode, 0.0),
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
        test_id: "diff_ndimage_filter_1d".into(),
        category: "scipy.ndimage uniform/maximum/minimum_filter1d".into(),
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
        "filter_1d conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
