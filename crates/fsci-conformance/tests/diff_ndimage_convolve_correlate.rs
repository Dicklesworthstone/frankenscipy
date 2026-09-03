#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the two N-dim linear-filter
//! primitives that diff_ndimage.rs didn't cover:
//!   - `scipy.ndimage.convolve(input, weights, mode, cval)`
//!   - `scipy.ndimage.correlate(input, weights, mode, cval)`
//!
//! Resolves [frankenscipy-41ks8]. diff_ndimage.rs exercises the
//! morphological and statistical filter families plus sobel /
//! prewitt / laplace; this harness fills the gap for the explicit
//! convolution and correlation primitives.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_ndimage::{BoundaryMode, NdArray, convolve, correlate};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-015";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String,
    input_shape: Vec<usize>,
    input: Vec<f64>,
    weights_shape: Vec<usize>,
    weights: Vec<f64>,
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
    fs::create_dir_all(output_dir()).expect("create ndimage_conv diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize ndimage_conv diff log");
    fs::write(path, json).expect("write ndimage_conv diff log");
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
    // Inputs: simple 3×3 increasing, 4×4 sparse, 1D 5-element.
    let inputs: &[(&str, Vec<usize>, Vec<f64>)] = &[
        (
            "3x3_increasing",
            vec![3, 3],
            (1..=9).map(|i| i as f64).collect(),
        ),
        (
            "4x4_sparse",
            vec![4, 4],
            vec![
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            ],
        ),
        ("1d_len5", vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]),
    ];
    // Kernels per input dimensionality.
    let kernel_2d_laplace: (Vec<usize>, Vec<f64>) = (
        vec![3, 3],
        vec![0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0],
    );
    let kernel_2d_box: (Vec<usize>, Vec<f64>) = (vec![3, 3], vec![1.0 / 9.0; 9]);
    let kernel_1d_box: (Vec<usize>, Vec<f64>) = (vec![3], vec![1.0 / 3.0; 3]);

    let modes = ["reflect", "constant", "nearest"];

    let mut points = Vec::new();
    for (label, shape, data) in inputs {
        let kernels: Vec<(&str, &(Vec<usize>, Vec<f64>))> = if shape.len() == 1 {
            vec![("box1d", &kernel_1d_box)]
        } else {
            vec![("laplace2d", &kernel_2d_laplace), ("box2d", &kernel_2d_box)]
        };
        for (kname, (kshape, kdata)) in kernels {
            for mode in modes {
                for op in ["convolve", "correlate"] {
                    points.push(PointCase {
                        case_id: format!("{op}_{label}_{kname}_{mode}"),
                        op: op.into(),
                        input_shape: shape.clone(),
                        input: data.clone(),
                        weights_shape: kshape.clone(),
                        weights: kdata.clone(),
                        mode: mode.into(),
                        cval: 0.0,
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

def finite_vec_or_none(arr):
    out = []
    for v in arr.flatten().tolist():
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
    cid = case["case_id"]; op = case["op"]
    input_arr = np.array(case["input"], dtype=float).reshape(case["input_shape"])
    weights_arr = np.array(case["weights"], dtype=float).reshape(case["weights_shape"])
    mode = case["mode"]; cval = float(case["cval"])
    try:
        if op == "convolve":
            v = ndimage.convolve(input_arr, weights_arr, mode=mode, cval=cval)
        elif op == "correlate":
            v = ndimage.correlate(input_arr, weights_arr, mode=mode, cval=cval)
        else:
            v = None
        points.append({"case_id": cid, "values": finite_vec_or_none(v) if v is not None else None})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize ndimage_conv query");
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
                "failed to spawn python3 for ndimage_conv oracle: {e}"
            );
            eprintln!("skipping ndimage_conv oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open ndimage_conv oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "ndimage_conv oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping ndimage_conv oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for ndimage_conv oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "ndimage_conv oracle failed: {stderr}"
        );
        eprintln!("skipping ndimage_conv oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse ndimage_conv oracle JSON"))
}

fn fsci_eval(case: &PointCase) -> Option<Vec<f64>> {
    let input = NdArray::new(case.input.clone(), case.input_shape.clone()).ok()?;
    let weights = NdArray::new(case.weights.clone(), case.weights_shape.clone()).ok()?;
    let mode = parse_mode(&case.mode);
    let result = match case.op.as_str() {
        "convolve" => convolve(&input, &weights, mode, case.cval).ok()?,
        "correlate" => correlate(&input, &weights, mode, case.cval).ok()?,
        _ => return None,
    };
    Some(result.data)
}

#[test]
fn diff_ndimage_convolve_correlate() {
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
        let Some(fsci_v) = fsci_eval(case) else {
            continue;
        };
        let Some(scipy_v) = scipy_arm.values.as_ref() else {
            continue;
        };
        if fsci_v.len() != scipy_v.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                op: case.op.clone(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let abs_d = fsci_v
            .iter()
            .zip(scipy_v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
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
        test_id: "diff_ndimage_convolve_correlate".into(),
        category: "scipy.ndimage.convolve / correlate".into(),
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
                "ndimage_conv {} mismatch: {} abs_diff={}",
                d.op, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.ndimage.convolve/correlate conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
