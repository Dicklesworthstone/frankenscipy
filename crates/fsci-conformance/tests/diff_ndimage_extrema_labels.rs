#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `fsci_ndimage::extrema_labels`.
//!
//! Resolves [frankenscipy-v59pa]. fsci returns (mins, maxs) per label
//! (1..=num_labels). Compared against scipy.ndimage.{minimum, maximum}
//! with index=1..num. 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_ndimage::{NdArray, extrema_labels};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-015";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    input_shape: Vec<usize>,
    input: Vec<f64>,
    labels: Vec<f64>,
    num_labels: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    mins: Option<Vec<f64>>,
    maxs: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    min_diff: f64,
    max_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create extrema diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize extrema diff log");
    fs::write(path, json).expect("write extrema diff log");
}

fn generate_query() -> OracleQuery {
    let scenarios: &[(&str, Vec<usize>, Vec<f64>, Vec<f64>, usize)] = &[
        (
            "3x3_two_regions",
            vec![3, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![1.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 2.0],
            2,
        ),
        (
            "4x4_three_regions",
            vec![4, 4],
            (1..=16).map(|i| i as f64).collect(),
            vec![
                1.0, 1.0, 0.0, 0.0, //
                1.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 2.0, 2.0, //
                3.0, 3.0, 2.0, 2.0, //
            ],
            3,
        ),
        (
            "1d_len10",
            vec![10],
            (1..=10).map(|i| i as f64).collect(),
            vec![1.0, 1.0, 1.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            2,
        ),
        (
            "5x5_isolated",
            vec![5, 5],
            (0..25).map(|i| i as f64 * 0.5 + 1.0).collect(),
            vec![
                1.0, 0.0, 2.0, 0.0, 3.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, //
                4.0, 0.0, 0.0, 0.0, 5.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, //
                6.0, 0.0, 7.0, 0.0, 8.0, //
            ],
            8,
        ),
    ];
    let points = scenarios
        .iter()
        .map(|(name, shape, input, labels, num)| PointCase {
            case_id: (*name).into(),
            input_shape: shape.clone(),
            input: input.clone(),
            labels: labels.clone(),
            num_labels: *num,
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

def vec_or_none(arr):
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
    inp = np.array(case["input"], dtype=float).reshape(shape)
    lbls = np.array(case["labels"], dtype=int).reshape(shape)
    num = int(case["num_labels"])
    idx = np.arange(1, num + 1)
    try:
        mn = ndimage.minimum(inp, labels=lbls, index=idx)
        mx = ndimage.maximum(inp, labels=lbls, index=idx)
        points.append({"case_id": cid, "mins": vec_or_none(mn), "maxs": vec_or_none(mx)})
    except Exception:
        points.append({"case_id": cid, "mins": None, "maxs": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize extrema query");
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
                "failed to spawn python3 for extrema oracle: {e}"
            );
            eprintln!("skipping extrema oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open extrema oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "extrema oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping extrema oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for extrema oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "extrema oracle failed: {stderr}"
        );
        eprintln!("skipping extrema oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse extrema oracle JSON"))
}

#[test]
fn diff_ndimage_extrema_labels() {
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

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(scipy_mins) = scipy_arm.mins.as_ref() else {
            continue;
        };
        let Some(scipy_maxs) = scipy_arm.maxs.as_ref() else {
            continue;
        };
        let Ok(input) = NdArray::new(case.input.clone(), case.input_shape.clone()) else {
            continue;
        };
        let Ok(labels) = NdArray::new(case.labels.clone(), case.input_shape.clone()) else {
            continue;
        };
        let (fsci_mins, fsci_maxs) = extrema_labels(&input, &labels, case.num_labels);
        if fsci_mins.len() != scipy_mins.len() || fsci_maxs.len() != scipy_maxs.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                min_diff: f64::INFINITY,
                max_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let min_diff = fsci_mins
            .iter()
            .zip(scipy_mins.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let max_diff = fsci_maxs
            .iter()
            .zip(scipy_maxs.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            min_diff,
            max_diff,
            pass: min_diff <= ABS_TOL && max_diff <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_ndimage_extrema_labels".into(),
        category: "scipy.ndimage.minimum + maximum (per-label)".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "extrema mismatch: {} min_diff={} max_diff={}",
                d.case_id, d.min_diff, d.max_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.ndimage extrema_labels conformance failed: {} cases",
        diffs.len()
    );
}
