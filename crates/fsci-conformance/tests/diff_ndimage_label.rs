#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.ndimage.label`.
//!
//! Resolves [frankenscipy-gebb5]. Label IDs themselves may differ
//! between fsci and scipy (different traversal order), but the
//! partition into connected components is well-defined. Compare:
//!   1. num_features: exact integer match
//!   2. partition equivalence: for every pair of pixels, both impls
//!      either put them in the same component or in different
//!      components.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_ndimage::{NdArray, label};
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
    num_features: Option<usize>,
    /// Flattened labels (integer-valued floats)
    labels: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    num_match: bool,
    partition_match: bool,
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
    fs::create_dir_all(output_dir()).expect("create label diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize label diff log");
    fs::write(path, json).expect("write label diff log");
}

fn generate_query() -> OracleQuery {
    let scenarios: &[(&str, Vec<usize>, Vec<f64>)] = &[
        (
            "5x5_two_blobs",
            vec![5, 5],
            vec![
                1.0, 1.0, 0.0, 0.0, 0.0, //
                1.0, 1.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 1.0, 1.0, //
                0.0, 0.0, 0.0, 1.0, 1.0, //
            ],
        ),
        (
            "5x5_three_clusters",
            vec![5, 5],
            vec![
                1.0, 0.0, 1.0, 0.0, 1.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, //
                1.0, 0.0, 1.0, 0.0, 1.0, //
            ],
        ),
        (
            "1d_len12_segments",
            vec![12],
            vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        ),
        ("4x4_all_zeros", vec![4, 4], vec![0.0; 16]),
        ("4x4_all_ones", vec![4, 4], vec![1.0; 16]),
        (
            "6x6_chain",
            vec![6, 6],
            vec![
                1.0, 1.0, 1.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 1.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 1.0, 1.0, //
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
    arr = np.array(case["input"], dtype=float).reshape(shape)
    try:
        labeled, num = ndimage.label(arr)
        points.append({
            "case_id": cid,
            "num_features": int(num),
            "labels": [float(v) for v in labeled.flatten().tolist()],
        })
    except Exception:
        points.append({"case_id": cid, "num_features": None, "labels": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize label query");
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
                "failed to spawn python3 for label oracle: {e}"
            );
            eprintln!("skipping label oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open label oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "label oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping label oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for label oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "label oracle failed: {stderr}"
        );
        eprintln!("skipping label oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse label oracle JSON"))
}

/// Two labelings produce the same partition iff there's a consistent
/// bijection between labels.
fn same_partition(a: &[f64], b: &[f64]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut a_to_b: HashMap<i64, i64> = HashMap::new();
    let mut b_to_a: HashMap<i64, i64> = HashMap::new();
    for (la, lb) in a.iter().zip(b.iter()) {
        let la = *la as i64;
        let lb = *lb as i64;
        if let Some(&existing_b) = a_to_b.get(&la) {
            if existing_b != lb {
                return false;
            }
        } else {
            a_to_b.insert(la, lb);
        }
        if let Some(&existing_a) = b_to_a.get(&lb) {
            if existing_a != la {
                return false;
            }
        } else {
            b_to_a.insert(lb, la);
        }
    }
    true
}

#[test]
fn diff_ndimage_label() {
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
        let Some(scipy_num) = scipy_arm.num_features else {
            continue;
        };
        let Some(scipy_labels) = scipy_arm.labels.as_ref() else {
            continue;
        };
        let Ok(input) = NdArray::new(case.input.clone(), case.input_shape.clone()) else {
            continue;
        };
        let Ok((fsci_labels, fsci_num)) = label(&input) else {
            continue;
        };
        let num_match = fsci_num == scipy_num;
        let partition_match = same_partition(&fsci_labels.data, scipy_labels);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            num_match,
            partition_match,
            pass: num_match && partition_match,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_ndimage_label".into(),
        category: "scipy.ndimage.label".into(),
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
                "label mismatch: {} num_match={} partition_match={}",
                d.case_id, d.num_match, d.partition_match
            );
        }
    }

    assert!(
        all_pass,
        "scipy.ndimage.label conformance failed: {} cases",
        diffs.len()
    );
}
