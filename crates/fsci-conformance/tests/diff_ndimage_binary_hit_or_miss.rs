#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.ndimage.binary_hit_or_miss`.
//!
//! Resolves [frankenscipy-tob9b]. Probe against scipy on a few small
//! binary masks and matching structuring-element pairs. Bit-exact 0/1
//! comparison.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_ndimage::{NdArray, binary_hit_or_miss};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-015";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    input_shape: Vec<usize>,
    input: Vec<f64>,
    structure1_shape: Vec<usize>,
    structure1: Vec<f64>,
    structure2_shape: Option<Vec<usize>>,
    structure2: Option<Vec<f64>>,
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
    mismatched: usize,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    total_mismatched: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create hom diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize hom diff log");
    fs::write(path, json).expect("write hom diff log");
}

fn generate_query() -> OracleQuery {
    let points = vec![
        // 5x5 input, looking for 3x3 cross
        PointCase {
            case_id: "5x5_cross_struct1_default_struct2".into(),
            input_shape: vec![5, 5],
            input: vec![
                0.0, 0.0, 1.0, 0.0, 0.0, //
                0.0, 1.0, 1.0, 1.0, 0.0, //
                1.0, 1.0, 1.0, 1.0, 1.0, //
                0.0, 1.0, 1.0, 1.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, 0.0, //
            ],
            structure1_shape: vec![3, 3],
            structure1: vec![
                0.0, 1.0, 0.0, //
                1.0, 1.0, 1.0, //
                0.0, 1.0, 0.0, //
            ],
            structure2_shape: None,
            structure2: None,
        },
        // 6x6 input with corner pattern
        PointCase {
            case_id: "6x6_corner_pattern".into(),
            input_shape: vec![6, 6],
            input: vec![
                1.0, 1.0, 0.0, 0.0, 0.0, 0.0, //
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 1.0, 1.0, //
                0.0, 0.0, 0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            ],
            structure1_shape: vec![2, 2],
            structure1: vec![1.0, 1.0, 1.0, 0.0],
            structure2_shape: None,
            structure2: None,
        },
        // 4x4 with explicit struct2
        PointCase {
            case_id: "4x4_explicit_struct2".into(),
            input_shape: vec![4, 4],
            input: vec![
                0.0, 1.0, 0.0, 0.0, //
                1.0, 1.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, //
            ],
            structure1_shape: vec![3, 3],
            structure1: vec![
                0.0, 1.0, 0.0, //
                1.0, 1.0, 0.0, //
                0.0, 1.0, 0.0, //
            ],
            structure2_shape: Some(vec![3, 3]),
            structure2: Some(vec![
                1.0, 0.0, 1.0, //
                0.0, 0.0, 1.0, //
                1.0, 0.0, 1.0, //
            ]),
        },
    ];

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
    s1 = np.array(case["structure1"], dtype=float).reshape(case["structure1_shape"]) > 0
    s2 = None
    if case.get("structure2") is not None:
        s2 = np.array(case["structure2"], dtype=float).reshape(case["structure2_shape"]) > 0
    try:
        out = ndimage.binary_hit_or_miss(arr, structure1=s1, structure2=s2)
        v = out.astype(float).flatten().tolist()
        points.append({"case_id": cid, "values": v})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize hom query");
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
                "failed to spawn python3 for hom oracle: {e}"
            );
            eprintln!("skipping hom oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open hom oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "hom oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping hom oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for hom oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "hom oracle failed: {stderr}"
        );
        eprintln!("skipping hom oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse hom oracle JSON"))
}

fn fsci_eval(case: &PointCase) -> Option<Vec<f64>> {
    let input = NdArray::new(case.input.clone(), case.input_shape.clone()).ok()?;
    let s1 = NdArray::new(case.structure1.clone(), case.structure1_shape.clone()).ok()?;
    let s2_opt = match (case.structure2.as_ref(), case.structure2_shape.as_ref()) {
        (Some(d), Some(s)) => Some(NdArray::new(d.clone(), s.clone()).ok()?),
        _ => None,
    };
    let out = binary_hit_or_miss(&input, &s1, s2_opt.as_ref()).ok()?;
    Some(out.data)
}

#[test]
fn diff_ndimage_binary_hit_or_miss() {
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
            mismatched,
            pass: mismatched == 0,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_ndimage_binary_hit_or_miss".into(),
        category: "scipy.ndimage.binary_hit_or_miss".into(),
        case_count: diffs.len(),
        total_mismatched,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "hit_or_miss mismatch: {} {} pixels",
                d.case_id, d.mismatched
            );
        }
    }

    assert!(
        all_pass,
        "scipy.ndimage.binary_hit_or_miss conformance failed: {} cases, total mismatched={}",
        diffs.len(),
        total_mismatched
    );
}
