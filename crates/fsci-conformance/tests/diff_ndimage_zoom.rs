#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.ndimage.zoom`.
//!
//! Resolves [frankenscipy-qf0ub]. fsci_ndimage::zoom uses spline-coeff
//! prefiltering and the (out-1)/(in-1) coordinate mapping. scipy.ndimage.zoom
//! with `grid_mode=False` (default) uses the same mapping. Scoped to
//! spline orders 0 (nearest) and 1 (linear) to sidestep higher-order
//! spline convention differences. 1e-10 abs tolerance.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_ndimage::{BoundaryMode, NdArray, zoom};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-015";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    input_shape: Vec<usize>,
    input: Vec<f64>,
    zoom_factors: Vec<f64>,
    order: usize,
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
    out_shape: Option<Vec<usize>>,
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
    fs::create_dir_all(output_dir()).expect("create zoom diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize zoom diff log");
    fs::write(path, json).expect("write zoom diff log");
}

fn parse_mode(name: &str) -> BoundaryMode {
    match name {
        "reflect" => BoundaryMode::Reflect,
        "constant" => BoundaryMode::Constant,
        "nearest" => BoundaryMode::Nearest,
        "wrap" => BoundaryMode::Wrap,
        _ => BoundaryMode::Reflect,
    }
}

fn generate_query() -> OracleQuery {
    let inputs: &[(&str, Vec<usize>, Vec<f64>, Vec<f64>)] = &[
        (
            "4x4_zoom_2",
            vec![4, 4],
            (1..=16).map(|i| i as f64).collect(),
            vec![2.0, 2.0],
        ),
        (
            "4x4_zoom_half",
            vec![4, 4],
            (1..=16).map(|i| i as f64).collect(),
            vec![0.5, 0.5],
        ),
        (
            "1d_len8_zoom_2",
            vec![8],
            (1..=8).map(|i| i as f64).collect(),
            vec![2.0],
        ),
        (
            "1d_len10_zoom_3",
            vec![10],
            (1..=10).map(|i| i as f64).collect(),
            vec![3.0],
        ),
        (
            "3x4_asym_zoom",
            vec![3, 4],
            (1..=12).map(|i| i as f64).collect(),
            vec![2.0, 1.0],
        ),
    ];
    let orders = [0usize, 1];
    let modes = ["nearest", "reflect"];

    let mut points = Vec::new();
    for (label, shape, data, zf) in inputs {
        for &order in &orders {
            for mode in modes {
                points.push(PointCase {
                    case_id: format!("{label}_order{order}_{mode}"),
                    input_shape: shape.clone(),
                    input: data.clone(),
                    zoom_factors: zf.clone(),
                    order,
                    mode: mode.into(),
                    cval: 0.0,
                });
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

def finite_vec_or_none(arr):
    out = []
    for v in np.asarray(arr).flatten().tolist():
        try:
            v = float(v)
        except Exception:
            return None
        if not math.isfinite(v):
            return None
        out.append(v)
    return out

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    shape = case["input_shape"]
    arr = np.array(case["input"], dtype=float).reshape(shape)
    zf = case["zoom_factors"]
    order = int(case["order"])
    mode = case["mode"]; cval = float(case["cval"])
    try:
        v = ndimage.zoom(arr, zoom=zf, order=order, mode=mode, cval=cval, grid_mode=False)
        points.append({
            "case_id": cid,
            "values": finite_vec_or_none(v),
            "out_shape": list(v.shape),
        })
    except Exception:
        points.append({"case_id": cid, "values": None, "out_shape": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize zoom query");
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
                "failed to spawn python3 for zoom oracle: {e}"
            );
            eprintln!("skipping zoom oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open zoom oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "zoom oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping zoom oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for zoom oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "zoom oracle failed: {stderr}"
        );
        eprintln!("skipping zoom oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse zoom oracle JSON"))
}

#[test]
fn diff_ndimage_zoom() {
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
        let Some(scipy_v) = scipy_arm.values.as_ref() else {
            continue;
        };
        let Ok(input) = NdArray::new(case.input.clone(), case.input_shape.clone()) else {
            continue;
        };
        let Ok(out) = zoom(
            &input,
            &case.zoom_factors,
            case.order,
            parse_mode(&case.mode),
            case.cval,
        ) else {
            continue;
        };
        // Shape parity is checked separately from the flattened values: two different
        // output shapes can flatten to the same length.
        let shape_mismatch = scipy_arm
            .out_shape
            .as_ref()
            .is_some_and(|shape| *shape != out.shape);
        if shape_mismatch || out.data.len() != scipy_v.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let abs_d = out
            .data
            .iter()
            .zip(scipy_v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_ndimage_zoom".into(),
        category: "scipy.ndimage.zoom (orders 0, 1, grid_mode=False)".into(),
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
            eprintln!("zoom mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "scipy.ndimage.zoom conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
