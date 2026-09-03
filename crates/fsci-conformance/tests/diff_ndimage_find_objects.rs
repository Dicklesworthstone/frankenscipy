#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.ndimage.find_objects`.
//!
//! Resolves [frankenscipy-9sun8]. fsci returns
//! `Vec<Option<(min_indices, max_indices)>>` (max-inclusive bounds);
//! scipy returns list of `slice(start, stop)` tuples (stop-exclusive).
//! Conversion: scipy.start == fsci.min, scipy.stop == fsci.max + 1.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_ndimage::{NdArray, find_objects};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-015";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    input_shape: Vec<usize>,
    /// Pre-computed labels (integer-valued floats)
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
    /// For each label, [min_0, min_1, ..., max_0, max_1, ...] flattened.
    /// Some entry => Vec<usize>. None entry => empty Vec.
    boxes: Option<Vec<Vec<i64>>>,
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
    fs::create_dir_all(output_dir()).expect("create find_objects diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize find_objects diff log");
    fs::write(path, json).expect("write find_objects diff log");
}

fn generate_query() -> OracleQuery {
    let scenarios: &[(&str, Vec<usize>, Vec<f64>, usize)] = &[
        (
            "4x4_two_quadrants",
            vec![4, 4],
            vec![
                1.0, 1.0, 0.0, 0.0, //
                1.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 2.0, 2.0, //
                0.0, 0.0, 2.0, 2.0, //
            ],
            2,
        ),
        (
            "5x5_three_labels",
            vec![5, 5],
            vec![
                1.0, 1.0, 0.0, 2.0, 2.0, //
                1.0, 1.0, 0.0, 2.0, 2.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, //
                3.0, 0.0, 0.0, 0.0, 0.0, //
                3.0, 0.0, 0.0, 0.0, 0.0, //
            ],
            3,
        ),
        (
            "5x5_with_gap",
            vec![5, 5],
            vec![
                1.0, 0.0, 0.0, 0.0, 1.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 2.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, //
                3.0, 0.0, 0.0, 0.0, 3.0, //
            ],
            3,
        ),
        (
            "1d_len10",
            vec![10],
            vec![1.0, 1.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 3.0, 3.0],
            3,
        ),
    ];
    let points = scenarios
        .iter()
        .map(|(name, shape, labels, num)| PointCase {
            case_id: (*name).into(),
            input_shape: shape.clone(),
            labels: labels.clone(),
            num_labels: *num,
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
    labels = np.array(case["labels"], dtype=int).reshape(shape)
    try:
        objs = ndimage.find_objects(labels)
        # Convert to [min_0, min_1, ..., max_0, max_1, ...] per label;
        # empty list for None entries.
        out = []
        for o in objs:
            if o is None:
                out.append([])
            else:
                box = []
                for s in o:
                    box.append(int(s.start))
                for s in o:
                    box.append(int(s.stop) - 1)
                out.append(box)
        points.append({"case_id": cid, "boxes": out})
    except Exception:
        points.append({"case_id": cid, "boxes": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize find_objects query");
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
                "failed to spawn python3 for find_objects oracle: {e}"
            );
            eprintln!("skipping find_objects oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open find_objects oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "find_objects oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping find_objects oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for find_objects oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "find_objects oracle failed: {stderr}"
        );
        eprintln!("skipping find_objects oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse find_objects oracle JSON"))
}

#[test]
fn diff_ndimage_find_objects() {
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
        let Some(scipy_boxes) = scipy_arm.boxes.as_ref() else {
            continue;
        };
        let Ok(labels) = NdArray::new(case.labels.clone(), case.input_shape.clone()) else {
            continue;
        };
        let fsci_boxes = find_objects(&labels, case.num_labels);
        if fsci_boxes.len() != scipy_boxes.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                pass: false,
            });
            continue;
        }
        let mut ok = true;
        for (fb, sb) in fsci_boxes.iter().zip(scipy_boxes.iter()) {
            match fb {
                None => {
                    if !sb.is_empty() {
                        ok = false;
                        break;
                    }
                }
                Some((mins, maxs)) => {
                    let mut expected = Vec::with_capacity(mins.len() + maxs.len());
                    for v in mins {
                        expected.push(*v as i64);
                    }
                    for v in maxs {
                        expected.push(*v as i64);
                    }
                    if expected != *sb {
                        ok = false;
                        break;
                    }
                }
            }
        }
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            pass: ok,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_ndimage_find_objects".into(),
        category: "scipy.ndimage.find_objects".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("find_objects mismatch: {}", d.case_id);
        }
    }

    assert!(
        all_pass,
        "scipy.ndimage.find_objects conformance failed: {} cases",
        diffs.len()
    );
}
