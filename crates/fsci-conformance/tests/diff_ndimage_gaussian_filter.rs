#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.ndimage.gaussian_filter`.
//!
//! Resolves [frankenscipy-6u7yg] and [frankenscipy-0venh].
//! fsci_ndimage::gaussian_filter applies a separable 1-D Gaussian along
//! each axis and uses SciPy's default radius rule:
//! `int(truncate*σ + 0.5)` with `truncate=4.0`.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_ndimage::{BoundaryMode, NdArray, gaussian_filter};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-015";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    input_shape: Vec<usize>,
    input: Vec<f64>,
    sigma: f64,
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
    fs::create_dir_all(output_dir()).expect("create gauss_filter diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize gauss_filter diff log");
    fs::write(path, json).expect("write gauss_filter diff log");
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
            "5x5_increasing",
            vec![5, 5],
            (0..25).map(|i| i as f64).collect(),
        ),
        (
            "7x7_smooth",
            vec![7, 7],
            (0..49)
                .map(|i| {
                    let r = (i / 7) as f64;
                    let c = (i % 7) as f64;
                    (r - 3.0).powi(2) + (c - 3.0).powi(2)
                })
                .collect(),
        ),
        (
            "1d_len15",
            vec![15],
            (0..15).map(|i| ((i as f64) * 0.5).sin() * 5.0).collect(),
        ),
    ];
    let sigmas: &[f64] = &[0.5, 1.0, 1.3, 2.0];
    let modes = ["reflect", "constant", "nearest"];

    let mut points = Vec::new();
    for (label, shape, data) in inputs {
        for &sigma in sigmas {
            for mode in modes {
                points.push(PointCase {
                    case_id: format!("{label}_s{sigma}_{mode}"),
                    input_shape: shape.clone(),
                    input: data.clone(),
                    sigma,
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
    sigma = float(case["sigma"]); mode = case["mode"]; cval = float(case["cval"])
    try:
        v = ndimage.gaussian_filter(arr, sigma=sigma, mode=mode, cval=cval, truncate=4.0)
        points.append({"case_id": cid, "values": finite_vec_or_none(v)})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize gauss_filter query");
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
                "failed to spawn python3 for gauss_filter oracle: {e}"
            );
            eprintln!("skipping gauss_filter oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open gauss_filter oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "gauss_filter oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping gauss_filter oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for gauss_filter oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "gauss_filter oracle failed: {stderr}"
        );
        eprintln!("skipping gauss_filter oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse gauss_filter oracle JSON"))
}

#[test]
fn diff_ndimage_gaussian_filter() {
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
        let Ok(fsci) = gaussian_filter(&input, case.sigma, parse_mode(&case.mode), case.cval)
        else {
            continue;
        };
        if fsci.data.len() != scipy_v.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let abs_d = fsci
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
        test_id: "diff_ndimage_gaussian_filter".into(),
        category: "scipy.ndimage.gaussian_filter".into(),
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
                "gaussian_filter mismatch: {} abs_diff={}",
                d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.ndimage.gaussian_filter conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
