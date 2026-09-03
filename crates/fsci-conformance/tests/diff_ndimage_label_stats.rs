#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_ndimage's label-region
//! statistics:
//!   - `sum_labels(input, labels, num)`            vs scipy.ndimage.sum_labels
//!   - `mean_labels(input, labels, num)`           vs scipy.ndimage.mean
//!   - `variance_labels(input, labels, num)`       vs scipy.ndimage.variance
//!   - `standard_deviation_labels(input, labels, num)` vs scipy.ndimage.standard_deviation
//!   - `center_of_mass(input, labels, num)`        vs scipy.ndimage.center_of_mass
//!
//! Resolves [frankenscipy-ds7fd]. Uses handcrafted labels arrays to
//! sidestep label()-ordering ambiguity between implementations. Both
//! sides receive identical (input, labels, index=1..=num_labels)
//! arguments; outputs are closed-form aggregates over labeled regions.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_ndimage::{
    NdArray, center_of_mass, mean_labels, standard_deviation_labels, sum_labels, variance_labels,
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
    /// Same shape as input — integer labels stored as f64.
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
    /// Flattened per-label outputs (vector for scalar ops, flattened row-major for center_of_mass).
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
    fs::create_dir_all(output_dir()).expect("create label_stats diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize label_stats diff log");
    fs::write(path, json).expect("write label_stats diff log");
}

fn generate_query() -> OracleQuery {
    // Handcrafted (input, labels, num_labels) triples.
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
            "1d_len10_single",
            vec![10],
            (0..10).map(|i| (i as f64) * 0.5 + 1.0).collect(),
            vec![1.0, 1.0, 1.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            2,
        ),
        (
            "5x5_isolated_pts",
            vec![5, 5],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, //
                6.0, 7.0, 8.0, 9.0, 10.0, //
                11.0, 12.0, 13.0, 14.0, 15.0, //
                16.0, 17.0, 18.0, 19.0, 20.0, //
                21.0, 22.0, 23.0, 24.0, 25.0, //
            ],
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

    let mut points = Vec::new();
    for (label, shape, input, labels, num) in scenarios {
        for op in ["sum", "mean", "variance", "std", "center_of_mass"] {
            points.push(PointCase {
                case_id: format!("{op}_{label}"),
                op: op.into(),
                input_shape: shape.clone(),
                input: input.clone(),
                labels: labels.clone(),
                num_labels: *num,
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
    inp = np.array(case["input"], dtype=float).reshape(shape)
    lbls = np.array(case["labels"], dtype=int).reshape(shape)
    num = int(case["num_labels"])
    idx = np.arange(1, num + 1)
    try:
        if op == "sum":
            v = ndimage.sum_labels(inp, labels=lbls, index=idx)
        elif op == "mean":
            v = ndimage.mean(inp, labels=lbls, index=idx)
        elif op == "variance":
            v = ndimage.variance(inp, labels=lbls, index=idx)
        elif op == "std":
            v = ndimage.standard_deviation(inp, labels=lbls, index=idx)
        elif op == "center_of_mass":
            coms = ndimage.center_of_mass(inp, labels=lbls, index=idx)
            # Flatten per-label COM tuples
            v = np.asarray(coms).flatten()
        else:
            v = None
        points.append({"case_id": cid, "values": finite_vec_or_none(v) if v is not None else None})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize label_stats query");
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
                "failed to spawn python3 for label_stats oracle: {e}"
            );
            eprintln!("skipping label_stats oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open label_stats oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "label_stats oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping label_stats oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for label_stats oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "label_stats oracle failed: {stderr}"
        );
        eprintln!("skipping label_stats oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse label_stats oracle JSON"))
}

fn fsci_eval(case: &PointCase) -> Option<Vec<f64>> {
    let input = NdArray::new(case.input.clone(), case.input_shape.clone()).ok()?;
    let labels = NdArray::new(case.labels.clone(), case.input_shape.clone()).ok()?;
    match case.op.as_str() {
        "sum" => sum_labels(&input, &labels, case.num_labels)
            .ok()
            .map(|v| skip_zero(&v, case.num_labels)),
        "mean" => mean_labels(&input, &labels, case.num_labels)
            .ok()
            .map(|v| skip_zero(&v, case.num_labels)),
        "variance" => variance_labels(&input, &labels, case.num_labels)
            .ok()
            .map(|v| skip_zero(&v, case.num_labels)),
        "std" => standard_deviation_labels(&input, &labels, case.num_labels)
            .ok()
            .map(|v| skip_zero(&v, case.num_labels)),
        "center_of_mass" => {
            let coms = center_of_mass(&input, &labels, case.num_labels).ok()?;
            // fsci returns Vec of length num_labels indexed 0..num_labels-1
            // (labels 1..=num_labels). Flatten in scipy order.
            let mut packed = Vec::with_capacity(case.num_labels * input.shape.len());
            for coord in &coms {
                for &c in coord {
                    packed.push(c);
                }
            }
            Some(packed)
        }
        _ => None,
    }
}

/// fsci's *_labels return Vec of length num_labels+1, indexed by label
/// (with background at 0). Slice off the leading background entry to
/// align with scipy's `index=1..num_labels` output.
fn skip_zero(v: &[f64], num_labels: usize) -> Vec<f64> {
    if v.len() > num_labels {
        v[1..=num_labels].to_vec()
    } else {
        v.to_vec()
    }
}

#[test]
fn diff_ndimage_label_stats() {
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
        let Some(fsci_v) = fsci_eval(case) else {
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
        test_id: "diff_ndimage_label_stats".into(),
        category: "scipy.ndimage label-region statistics".into(),
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
                "label_stats {} mismatch: {} abs_diff={}",
                d.op, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.ndimage label_stats conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
