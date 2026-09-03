#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.ndimage.distance_transform_edt`.
//!
//! Resolves [frankenscipy-sa627]. fsci_ndimage::distance_transform_edt
//! computes the Euclidean distance from each foreground pixel (nonzero)
//! to the nearest background pixel (zero). Currently 2-D only. All
//! probe inputs include at least one background pixel — fsci's
//! all-foreground fallback is non-standard so excluded.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_ndimage::{NdArray, distance_transform_edt};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-015";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    input_shape: Vec<usize>,
    input: Vec<f64>,
    sampling: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create edt diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize edt diff log");
    fs::write(path, json).expect("write edt diff log");
}

fn generate_query() -> OracleQuery {
    let scenarios: &[(&str, Vec<usize>, Vec<f64>, Option<Vec<f64>>)] = &[
        (
            "5x5_isolated_bg_center",
            vec![5, 5],
            vec![
                1.0, 1.0, 1.0, 1.0, 1.0, //
                1.0, 1.0, 1.0, 1.0, 1.0, //
                1.0, 1.0, 0.0, 1.0, 1.0, //
                1.0, 1.0, 1.0, 1.0, 1.0, //
                1.0, 1.0, 1.0, 1.0, 1.0, //
            ],
            None,
        ),
        (
            "5x5_corner_bg",
            vec![5, 5],
            vec![
                0.0, 1.0, 1.0, 1.0, 1.0, //
                1.0, 1.0, 1.0, 1.0, 1.0, //
                1.0, 1.0, 1.0, 1.0, 1.0, //
                1.0, 1.0, 1.0, 1.0, 1.0, //
                1.0, 1.0, 1.0, 1.0, 1.0, //
            ],
            None,
        ),
        (
            "6x6_two_bg_clusters",
            vec![6, 6],
            vec![
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, //
                1.0, 0.0, 0.0, 1.0, 1.0, 1.0, //
                1.0, 0.0, 0.0, 1.0, 1.0, 1.0, //
                1.0, 1.0, 1.0, 1.0, 0.0, 0.0, //
                1.0, 1.0, 1.0, 1.0, 0.0, 0.0, //
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, //
            ],
            None,
        ),
        (
            "5x5_bg_with_sampling",
            vec![5, 5],
            vec![
                1.0, 1.0, 1.0, 1.0, 1.0, //
                1.0, 1.0, 1.0, 1.0, 1.0, //
                1.0, 1.0, 0.0, 1.0, 1.0, //
                1.0, 1.0, 1.0, 1.0, 1.0, //
                1.0, 1.0, 1.0, 1.0, 1.0, //
            ],
            Some(vec![2.0, 1.0]),
        ),
        (
            "4x6_rect_bg_strip",
            vec![4, 6],
            vec![
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, //
                1.0, 1.0, 0.0, 0.0, 1.0, 1.0, //
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, //
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, //
            ],
            None,
        ),
    ];
    let points = scenarios
        .iter()
        .map(|(name, shape, input, sampling)| PointCase {
            case_id: (*name).into(),
            input_shape: shape.clone(),
            input: input.clone(),
            sampling: sampling.clone(),
        })
        .collect();
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
    sampling = case.get("sampling")
    try:
        if sampling is not None:
            v = ndimage.distance_transform_edt(arr, sampling=sampling)
        else:
            v = ndimage.distance_transform_edt(arr)
        points.append({"case_id": cid, "values": finite_vec_or_none(v)})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize edt query");
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
                "failed to spawn python3 for edt oracle: {e}"
            );
            eprintln!("skipping edt oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open edt oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "edt oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping edt oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for edt oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "edt oracle failed: {stderr}"
        );
        eprintln!("skipping edt oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse edt oracle JSON"))
}

#[test]
fn diff_ndimage_distance_transform_edt() {
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
        let Ok(out) = distance_transform_edt(&input, case.sampling.as_deref()) else {
            continue;
        };
        if out.data.len() != scipy_v.len() {
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
        test_id: "diff_ndimage_distance_transform_edt".into(),
        category: "scipy.ndimage.distance_transform_edt".into(),
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
            eprintln!("edt mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "scipy.ndimage.distance_transform_edt conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
