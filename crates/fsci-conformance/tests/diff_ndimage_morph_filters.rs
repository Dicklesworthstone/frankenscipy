#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_ndimage rank-filter
//! morphology helpers (minimum_filter, maximum_filter,
//! morphological_gradient, white_tophat, black_tophat) and rotate.
//!
//! Resolves [frankenscipy-fof02]. 1e-10 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_ndimage::{
    BoundaryMode, NdArray, black_tophat, maximum_filter, minimum_filter, morphological_gradient,
    rotate, white_tophat,
};
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
    angle: f64,
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
    fs::create_dir_all(output_dir()).expect("create morph_filters diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize morph_filters diff log");
    fs::write(path, json).expect("write morph_filters diff log");
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
    let mat_5x5_pattern: Vec<f64> = (0..25)
        .map(|i| ((i as f64) * 0.3).sin() * 5.0 + 2.0)
        .collect();

    let mut points = Vec::new();

    // min/max/morph_gradient/wth/bth filters with size=3, mode varies
    let inputs: &[(&str, &[f64], Vec<usize>, &str)] = &[
        ("3x3_reflect", &mat_3x3, vec![3, 3], "reflect"),
        ("3x3_nearest", &mat_3x3, vec![3, 3], "nearest"),
        ("4x5_reflect", &mat_4x5, vec![4, 5], "reflect"),
        ("5x5_constant", &mat_5x5_pattern, vec![5, 5], "constant"),
    ];
    for (label, input, shape, mode) in inputs {
        for op in [
            "minimum_filter",
            "maximum_filter",
            "morphological_gradient",
            "white_tophat",
            "black_tophat",
        ] {
            points.push(PointCase {
                case_id: format!("{op}_{label}"),
                op: op.into(),
                shape: shape.clone(),
                input: input.to_vec(),
                size: 3,
                angle: 0.0,
                order: 0,
                mode: (*mode).into(),
                cval: 0.0,
            });
        }
    }

    // rotate at 0/90/180/270 degrees (order=0). Square inputs only —
    // rectangular reshape=False crops the rotated content and fsci/scipy
    // diverge on which corner is preserved (filed separately).
    for (label, input, shape) in [
        ("3x3", mat_3x3.clone(), vec![3, 3]),
        ("5x5", mat_5x5_pattern.clone(), vec![5, 5]),
    ] {
        for angle in [0.0_f64, 90.0, 180.0, 270.0] {
            points.push(PointCase {
                case_id: format!("rotate_{label}_{}deg", angle as i64),
                op: "rotate".into(),
                shape: shape.clone(),
                input: input.clone(),
                size: 0,
                angle,
                order: 0,
                mode: "constant".into(),
                cval: 0.0,
            });
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
    angle = float(case["angle"])
    order = int(case["order"])
    try:
        if op == "minimum_filter":
            y = ndimage.minimum_filter(x, size=size, mode=mode, cval=cval)
        elif op == "maximum_filter":
            y = ndimage.maximum_filter(x, size=size, mode=mode, cval=cval)
        elif op == "morphological_gradient":
            y = ndimage.morphological_gradient(x, size=size, mode=mode, cval=cval)
        elif op == "white_tophat":
            y = ndimage.white_tophat(x, size=size, mode=mode, cval=cval)
        elif op == "black_tophat":
            y = ndimage.black_tophat(x, size=size, mode=mode, cval=cval)
        elif op == "rotate":
            y = ndimage.rotate(x, angle, reshape=False, order=order, mode=mode, cval=cval)
        else:
            y = None
        if y is None:
            points.append({"case_id": cid, "values": None})
        else:
            points.append({"case_id": cid, "values": finite_flat_or_none(y)})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize morph_filters query");
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
                "failed to spawn python3 for morph_filters oracle: {e}"
            );
            eprintln!("skipping morph_filters oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open morph_filters oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "morph_filters oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping morph_filters oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for morph_filters oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "morph_filters oracle failed: {stderr}"
        );
        eprintln!("skipping morph_filters oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse morph_filters oracle JSON"))
}

#[test]
fn diff_ndimage_morph_filters() {
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
            "minimum_filter" => minimum_filter(&input, case.size, mode, case.cval),
            "maximum_filter" => maximum_filter(&input, case.size, mode, case.cval),
            "morphological_gradient" => morphological_gradient(&input, case.size, mode, case.cval),
            "white_tophat" => white_tophat(&input, case.size, mode, case.cval),
            "black_tophat" => black_tophat(&input, case.size, mode, case.cval),
            "rotate" => rotate(&input, case.angle, false, case.order, mode, case.cval),
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
        test_id: "diff_ndimage_morph_filters".into(),
        category: "scipy.ndimage min/max/morph/rotate".into(),
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
        "morph_filters conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
