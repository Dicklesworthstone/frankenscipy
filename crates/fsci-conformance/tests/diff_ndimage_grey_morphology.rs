#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_ndimage's grey
//! morphological wrappers and compound ops:
//!   - `grey_erosion(input, size, mode, cval)`      vs scipy.ndimage.grey_erosion
//!   - `grey_dilation(input, size, mode, cval)`     vs scipy.ndimage.grey_dilation
//!   - `grey_opening(input, size, mode, cval)`      vs scipy.ndimage.grey_opening
//!   - `grey_closing(input, size, mode, cval)`      vs scipy.ndimage.grey_closing
//!   - `morphological_gradient(input, size, ...)`   vs scipy.ndimage.morphological_gradient
//!   - `white_tophat(input, size, mode, cval)`      vs scipy.ndimage.white_tophat
//!   - `black_tophat(input, size, mode, cval)`      vs scipy.ndimage.black_tophat
//!
//! Resolves [frankenscipy-4k4qy]. fsci's grey morph ops thin-wrap
//! minimum_filter / maximum_filter (already covered in diff_ndimage),
//! so this is mostly a parity-by-composition harness. scipy is called
//! with `size=(s,)*ndim` to get the same flat structuring element.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_ndimage::{
    BoundaryMode, NdArray, black_tophat, grey_closing, grey_dilation, grey_erosion, grey_opening,
    morphological_gradient, white_tophat,
};
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
    size: usize,
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
    fs::create_dir_all(output_dir()).expect("create grey_morph diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize grey_morph diff log");
    fs::write(path, json).expect("write grey_morph diff log");
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
    let inputs: &[(&str, Vec<usize>, Vec<f64>)] = &[
        (
            "4x4_increasing",
            vec![4, 4],
            (0..16).map(|i| i as f64).collect(),
        ),
        (
            "5x5_random_seeded",
            vec![5, 5],
            // Deterministic seed-driven values (not numpy.random).
            (0..25)
                .map(|i| ((i as f64) * 1.7).sin() * 5.0 + (i as f64) * 0.3)
                .collect(),
        ),
        (
            "3x3_dense",
            vec![3, 3],
            vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0],
        ),
        (
            "1d_len12",
            vec![12],
            vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0],
        ),
    ];

    let modes = ["reflect", "constant", "nearest"];
    let ops = [
        "erosion",
        "dilation",
        "opening",
        "closing",
        "morph_gradient",
        "white_tophat",
        "black_tophat",
    ];

    let mut points = Vec::new();
    for (label, shape, data) in inputs {
        for &size in &[3usize] {
            for mode in modes {
                for op in ops {
                    points.push(PointCase {
                        case_id: format!("{op}_{label}_size{size}_{mode}"),
                        op: op.into(),
                        input_shape: shape.clone(),
                        input: data.clone(),
                        size,
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

OPS = {
    "erosion":         ndimage.grey_erosion,
    "dilation":        ndimage.grey_dilation,
    "opening":         ndimage.grey_opening,
    "closing":         ndimage.grey_closing,
    "morph_gradient":  ndimage.morphological_gradient,
    "white_tophat":    ndimage.white_tophat,
    "black_tophat":    ndimage.black_tophat,
}

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
    cid = case["case_id"]; op = case["op"]
    shape = case["input_shape"]
    arr = np.array(case["input"], dtype=float).reshape(shape)
    s = case["size"]; mode = case["mode"]; cval = float(case["cval"])
    fn = OPS.get(op)
    try:
        v = fn(arr, size=(s,)*len(shape), mode=mode, cval=cval)
        points.append({"case_id": cid, "values": finite_vec_or_none(v)})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize grey_morph query");
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
                "failed to spawn python3 for grey_morph oracle: {e}"
            );
            eprintln!("skipping grey_morph oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open grey_morph oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "grey_morph oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping grey_morph oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for grey_morph oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "grey_morph oracle failed: {stderr}"
        );
        eprintln!("skipping grey_morph oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse grey_morph oracle JSON"))
}

fn fsci_eval(case: &PointCase) -> Option<Vec<f64>> {
    let input = NdArray::new(case.input.clone(), case.input_shape.clone()).ok()?;
    let mode = parse_mode(&case.mode);
    let result = match case.op.as_str() {
        "erosion" => grey_erosion(&input, case.size, mode, case.cval).ok()?,
        "dilation" => grey_dilation(&input, case.size, mode, case.cval).ok()?,
        "opening" => grey_opening(&input, case.size, mode, case.cval).ok()?,
        "closing" => grey_closing(&input, case.size, mode, case.cval).ok()?,
        "morph_gradient" => morphological_gradient(&input, case.size, mode, case.cval).ok()?,
        "white_tophat" => white_tophat(&input, case.size, mode, case.cval).ok()?,
        "black_tophat" => black_tophat(&input, case.size, mode, case.cval).ok()?,
        _ => return None,
    };
    Some(result.data)
}

#[test]
fn diff_ndimage_grey_morphology() {
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
        test_id: "diff_ndimage_grey_morphology".into(),
        category: "scipy.ndimage grey morph + tophat/gradient".into(),
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
                "grey_morph {} mismatch: {} abs_diff={}",
                d.op, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.ndimage grey morphology conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
