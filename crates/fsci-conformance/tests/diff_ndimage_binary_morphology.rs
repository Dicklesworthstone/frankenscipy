#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the binary morphological
//! operations in fsci_ndimage:
//!   - `binary_erosion(input, structure_size, iterations)`
//!   - `binary_dilation(input, structure_size, iterations)`
//!   - `binary_opening(input, structure_size, iterations)`
//!   - `binary_closing(input, structure_size, iterations)`
//!
//! Resolves [frankenscipy-sm8qx]. The fsci impls use a
//! `size × size × ...` N-d **box** structuring element with constant-0
//! boundary treatment. scipy.ndimage matches this when called with
//! `structure=np.ones((size,)*ndim)`, `border_value=0`. Output is
//! binary {0,1} per pixel so bit-exact agreement is expected.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_ndimage::{NdArray, binary_closing, binary_dilation, binary_erosion, binary_opening};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-015";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String,
    input_shape: Vec<usize>,
    input: Vec<f64>,
    structure_size: usize,
    iterations: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// 0/1 values (flattened, row-major) or None on python-side error.
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
    mismatched_pixels: usize,
    total_pixels: usize,
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
    fs::create_dir_all(output_dir()).expect("create binary_morph diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize binary_morph diff log");
    fs::write(path, json).expect("write binary_morph diff log");
}

fn generate_query() -> OracleQuery {
    let inputs: &[(&str, Vec<usize>, Vec<f64>)] = &[
        // 5×5 isolated point and small clump.
        (
            "5x5_point_and_clump",
            vec![5, 5],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 1.0, 0.0, 0.0, //
                0.0, 1.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 1.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, //
            ],
        ),
        // 5×5 cross / plus.
        (
            "5x5_cross",
            vec![5, 5],
            vec![
                0.0, 0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, 0.0, //
                1.0, 1.0, 1.0, 1.0, 1.0, //
                0.0, 0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, 0.0, //
            ],
        ),
        // 6×6 solid block surrounded by zeros (boundary effects).
        ("6x6_solid_block", vec![6, 6], {
            let mut v = vec![0.0; 36];
            for r in 1..5 {
                for c in 1..5 {
                    v[r * 6 + c] = 1.0;
                }
            }
            v
        }),
        // 1D 9-element.
        (
            "1d_len9_segments",
            vec![9],
            vec![0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        ),
        // All zeros.
        ("4x4_all_zero", vec![4, 4], vec![0.0; 16]),
        // All ones — boundary effects dominate for erosion.
        ("4x4_all_one", vec![4, 4], vec![1.0; 16]),
    ];

    let mut points = Vec::new();
    for (label, shape, data) in inputs {
        // Try structure sizes 3 and 5 where they fit. Iterations 1 and 2.
        let sizes: Vec<usize> = if shape.iter().any(|&n| n < 5) {
            vec![3]
        } else {
            vec![3, 5]
        };
        for &size in &sizes {
            for &iters in &[1usize, 2] {
                for op in ["erosion", "dilation", "opening", "closing"] {
                    points.push(PointCase {
                        case_id: format!("{op}_{label}_size{size}_iter{iters}"),
                        op: op.into(),
                        input_shape: shape.clone(),
                        input: data.clone(),
                        structure_size: size,
                        iterations: iters,
                    });
                }
            }
        }
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import sys
import numpy as np
from scipy import ndimage

OPS = {
    "erosion": ndimage.binary_erosion,
    "dilation": ndimage.binary_dilation,
    "opening": ndimage.binary_opening,
    "closing": ndimage.binary_closing,
}

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    shape = case["input_shape"]
    arr = np.array(case["input"], dtype=float).reshape(shape) > 0
    size = case["structure_size"]
    iters = case["iterations"]
    fn = OPS.get(op)
    try:
        structure = np.ones((size,)*len(shape), dtype=bool)
        out = fn(arr, structure=structure, iterations=iters, border_value=0)
        # cast bool to float 0/1
        vals = out.astype(float).flatten().tolist()
        points.append({"case_id": cid, "values": vals})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize binary_morph query");
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
                "failed to spawn python3 for binary_morph oracle: {e}"
            );
            eprintln!("skipping binary_morph oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open binary_morph oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "binary_morph oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping binary_morph oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for binary_morph oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "binary_morph oracle failed: {stderr}"
        );
        eprintln!("skipping binary_morph oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse binary_morph oracle JSON"))
}

fn fsci_eval(case: &PointCase) -> Option<Vec<f64>> {
    let input = NdArray::new(case.input.clone(), case.input_shape.clone()).ok()?;
    let result = match case.op.as_str() {
        "erosion" => binary_erosion(&input, case.structure_size, case.iterations).ok()?,
        "dilation" => binary_dilation(&input, case.structure_size, case.iterations).ok()?,
        "opening" => binary_opening(&input, case.structure_size, case.iterations).ok()?,
        "closing" => binary_closing(&input, case.structure_size, case.iterations).ok()?,
        _ => return None,
    };
    Some(result.data)
}

#[test]
fn diff_ndimage_binary_morphology() {
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
        let Some(fsci_v) = fsci_eval(case) else {
            continue;
        };
        let Some(scipy_v) = scipy_arm.values.as_ref() else {
            continue;
        };
        let total = fsci_v.len();
        let mismatched = if fsci_v.len() != scipy_v.len() {
            total // count mismatch as full-shape miss
        } else {
            fsci_v
                .iter()
                .zip(scipy_v.iter())
                // both are 0/1; compare exactly
                .filter(|(a, b)| (**a - **b).abs() > 0.5)
                .count()
        };
        total_mismatched += mismatched;
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            mismatched_pixels: mismatched,
            total_pixels: total,
            pass: mismatched == 0,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_ndimage_binary_morphology".into(),
        category: "scipy.ndimage.binary_{erosion,dilation,opening,closing}".into(),
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
                "binary_morph {} mismatch: {} {}/{} pixels",
                d.op, d.case_id, d.mismatched_pixels, d.total_pixels
            );
        }
    }

    assert!(
        all_pass,
        "scipy.ndimage binary morphology conformance failed: {} cases, total mismatched pixels={}",
        diffs.len(),
        total_mismatched
    );
}
