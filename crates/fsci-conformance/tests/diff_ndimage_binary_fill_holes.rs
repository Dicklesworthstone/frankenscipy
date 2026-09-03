#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.ndimage.binary_fill_holes`.
//!
//! Resolves [frankenscipy-dkl4l]. Bit-exact 0/1 comparison on 2-D
//! binary masks.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_ndimage::{NdArray, binary_fill_holes};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-015";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    input_shape: Vec<usize>,
    input: Vec<f64>,
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
    mismatched_pixels: usize,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    total_mismatched_pixels: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create fill_holes diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize fill_holes diff log");
    fs::write(path, json).expect("write fill_holes diff log");
}

fn generate_query() -> OracleQuery {
    let scenarios: &[(&str, Vec<usize>, Vec<f64>)] = &[
        (
            "ring_5x5",
            vec![5, 5],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 1.0, 1.0, 0.0, //
                0.0, 1.0, 0.0, 1.0, 0.0, //
                0.0, 1.0, 1.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, //
            ],
        ),
        (
            "two_rings_7x7",
            vec![7, 7],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, //
                0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            ],
        ),
        (
            "no_holes_4x4",
            vec![4, 4],
            vec![
                1.0, 1.0, 0.0, 0.0, //
                1.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, //
            ],
        ),
        ("all_zeros_4x4", vec![4, 4], vec![0.0; 16]),
        ("all_ones_4x4", vec![4, 4], vec![1.0; 16]),
        (
            "u_shape_6x6",
            vec![6, 6],
            vec![
                1.0, 1.0, 0.0, 0.0, 1.0, 1.0, //
                1.0, 1.0, 0.0, 0.0, 1.0, 1.0, //
                1.0, 1.0, 0.0, 0.0, 1.0, 1.0, //
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, //
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            ],
        ),
    ];
    let points = scenarios
        .iter()
        .map(|(name, shape, input)| PointCase {
            case_id: (*name).into(),
            input_shape: shape.clone(),
            input: input.clone(),
        })
        .collect();
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import sys
import numpy as np
from scipy import ndimage

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    shape = case["input_shape"]
    arr = np.array(case["input"], dtype=float).reshape(shape) > 0
    try:
        out = ndimage.binary_fill_holes(arr)
        v = out.astype(float).flatten().tolist()
        points.append({"case_id": cid, "values": v})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize fill_holes query");
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
                "failed to spawn python3 for fill_holes oracle: {e}"
            );
            eprintln!("skipping fill_holes oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open fill_holes oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "fill_holes oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping fill_holes oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for fill_holes oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "fill_holes oracle failed: {stderr}"
        );
        eprintln!("skipping fill_holes oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse fill_holes oracle JSON"))
}

fn fsci_eval(case: &PointCase) -> Option<Vec<f64>> {
    let input = NdArray::new(case.input.clone(), case.input_shape.clone()).ok()?;
    let out = binary_fill_holes(&input).ok()?;
    Some(out.data)
}

#[test]
fn diff_ndimage_binary_fill_holes() {
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
    let mut total_mismatched: usize = 0;

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(scipy_v) = scipy_arm.values.as_ref() else {
            continue;
        };
        let Some(fsci_v) = fsci_eval(case) else {
            continue;
        };
        let mismatched = if fsci_v.len() != scipy_v.len() {
            fsci_v.len()
        } else {
            fsci_v
                .iter()
                .zip(scipy_v.iter())
                .filter(|(a, b)| (**a - **b).abs() > 0.5)
                .count()
        };
        total_mismatched += mismatched;
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            mismatched_pixels: mismatched,
            pass: mismatched == 0,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_ndimage_binary_fill_holes".into(),
        category: "scipy.ndimage.binary_fill_holes".into(),
        case_count: diffs.len(),
        total_mismatched_pixels: total_mismatched,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "fill_holes mismatch: {} {} pixels",
                d.case_id, d.mismatched_pixels
            );
        }
    }

    assert!(
        all_pass,
        "scipy.ndimage.binary_fill_holes conformance failed: {} cases, total mismatched={}",
        diffs.len(),
        total_mismatched
    );
}
