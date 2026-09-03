#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `fsci_ndimage::histogram_labels`.
//!
//! Resolves [frankenscipy-4rer9]. Probed against scipy.ndimage.histogram
//! with labels= and index=1..num. Bit-exact integer comparison.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_ndimage::{NdArray, histogram_labels};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-015";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    input_shape: Vec<usize>,
    input: Vec<f64>,
    labels: Vec<f64>,
    num_labels: usize,
    min_val: f64,
    max_val: f64,
    nbins: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// Flattened per-label histograms (num × nbins).
    counts: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
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
    fs::create_dir_all(output_dir()).expect("create histogram_labels diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize histogram_labels diff log");
    fs::write(path, json).expect("write histogram_labels diff log");
}

fn generate_query() -> OracleQuery {
    let scenarios: &[(&str, Vec<usize>, Vec<f64>, Vec<f64>, usize, f64, f64, usize)] = &[
        (
            "2x4_two_labels_4bins",
            vec![2, 4],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![1.0, 1.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0],
            2,
            1.0,
            8.0,
            4,
        ),
        (
            "5x5_three_labels_5bins",
            vec![5, 5],
            (0..25).map(|i| i as f64).collect(),
            vec![
                1.0, 1.0, 0.0, 2.0, 2.0, //
                1.0, 1.0, 0.0, 2.0, 2.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, //
                3.0, 3.0, 0.0, 0.0, 0.0, //
                3.0, 3.0, 0.0, 0.0, 0.0, //
            ],
            3,
            0.0,
            25.0,
            5,
        ),
        (
            "1d_len12_3bins",
            vec![12],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            vec![1.0, 1.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            2,
            1.0,
            12.0,
            3,
        ),
    ];
    let points = scenarios
        .iter()
        .map(|(name, shape, input, labels, num, lo, hi, nb)| PointCase {
            case_id: (*name).into(),
            input_shape: shape.clone(),
            input: input.clone(),
            labels: labels.clone(),
            num_labels: *num,
            min_val: *lo,
            max_val: *hi,
            nbins: *nb,
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
    inp = np.array(case["input"], dtype=float).reshape(shape)
    lbls = np.array(case["labels"], dtype=int).reshape(shape)
    num = int(case["num_labels"])
    lo = float(case["min_val"]); hi = float(case["max_val"])
    nb = int(case["nbins"])
    try:
        packed = []
        for i in range(1, num + 1):
            h = ndimage.histogram(inp, min=lo, max=hi, bins=nb, labels=lbls, index=i)
            packed.extend(int(v) for v in h.tolist())
        points.append({"case_id": cid, "counts": packed})
    except Exception:
        points.append({"case_id": cid, "counts": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize histogram_labels query");
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
                "failed to spawn python3 for histogram_labels oracle: {e}"
            );
            eprintln!("skipping histogram_labels oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open histogram_labels oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "histogram_labels oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping histogram_labels oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for histogram_labels oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "histogram_labels oracle failed: {stderr}"
        );
        eprintln!("skipping histogram_labels oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse histogram_labels oracle JSON"))
}

#[test]
fn diff_ndimage_histogram_labels() {
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
        let Some(scipy_counts) = scipy_arm.counts.as_ref() else {
            continue;
        };
        let Ok(input) = NdArray::new(case.input.clone(), case.input_shape.clone()) else {
            continue;
        };
        let Ok(labels) = NdArray::new(case.labels.clone(), case.input_shape.clone()) else {
            continue;
        };
        let hists = histogram_labels(
            &input,
            &labels,
            case.num_labels,
            case.min_val,
            case.max_val,
            case.nbins,
        );
        let fsci_counts: Vec<usize> = hists.into_iter().flatten().collect();
        if fsci_counts.len() != scipy_counts.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                pass: false,
            });
            continue;
        }
        let pass = fsci_counts
            .iter()
            .zip(scipy_counts.iter())
            .all(|(a, b)| *a == *b);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_ndimage_histogram_labels".into(),
        category: "scipy.ndimage.histogram (per-label)".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("histogram_labels mismatch: {}", d.case_id);
        }
    }

    assert!(
        all_pass,
        "scipy.ndimage.histogram (per-label) conformance failed: {} cases",
        diffs.len()
    );
}
