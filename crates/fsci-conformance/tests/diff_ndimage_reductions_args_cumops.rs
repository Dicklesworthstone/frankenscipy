#![forbid(unsafe_code)]
//! Live numpy parity for fsci_ndimage reductions / argmin-argmax /
//! cumulative ops / element-wise comparison helpers:
//! count_nonzero, argmax, argmin, cumsum_array, cumprod_array,
//! diff_array, equal_within.
//!
//! Resolves [frankenscipy-yr65w]. Tolerance: 1e-12 abs (integer ops
//! exact; float ops below double-precision noise).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_ndimage::{
    NdArray, argmax, argmin, count_nonzero, cumprod_array, cumsum_array, diff_array, equal_within,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String,
    shape: Vec<usize>,
    data: Vec<f64>,
    /// For `equal_within` only: second operand (same shape).
    data_b: Vec<f64>,
    /// For `equal_within` only: tolerance.
    tol: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// Scalar ops use this field.
    scalar: Option<f64>,
    /// Vector ops use this field (flattened).
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
    fs::create_dir_all(output_dir()).expect("create reductions diff dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize log");
    fs::write(path, json).expect("write log");
}

fn generate_query() -> OracleQuery {
    let pos_neg_3x4: Vec<f64> = vec![
        1.0, -2.0, 0.0, 3.0, -4.0, 5.0, 0.0, -6.0, 7.0, 8.0, -9.0, 10.0,
    ];
    let with_zeros: Vec<f64> = vec![0.0, 1.0, 0.0, 2.0, 3.0, 0.0, 4.0, 5.0];
    let strictly_pos: Vec<f64> = (1..=8).map(|i| (i as f64) * 0.5).collect();
    let small: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];

    let mut points = vec![
        Case {
            case_id: "count_nonzero_pn".into(),
            op: "count_nonzero".into(),
            shape: vec![3, 4],
            data: pos_neg_3x4.clone(),
            data_b: vec![],
            tol: 0.0,
        },
        Case {
            case_id: "count_nonzero_with_zeros".into(),
            op: "count_nonzero".into(),
            shape: vec![2, 4],
            data: with_zeros.clone(),
            data_b: vec![],
            tol: 0.0,
        },
        Case {
            case_id: "argmax_pn".into(),
            op: "argmax".into(),
            shape: vec![3, 4],
            data: pos_neg_3x4.clone(),
            data_b: vec![],
            tol: 0.0,
        },
        Case {
            case_id: "argmin_pn".into(),
            op: "argmin".into(),
            shape: vec![3, 4],
            data: pos_neg_3x4.clone(),
            data_b: vec![],
            tol: 0.0,
        },
        Case {
            case_id: "argmax_strictly_pos".into(),
            op: "argmax".into(),
            shape: vec![2, 4],
            data: strictly_pos.clone(),
            data_b: vec![],
            tol: 0.0,
        },
        Case {
            case_id: "argmin_strictly_pos".into(),
            op: "argmin".into(),
            shape: vec![2, 4],
            data: strictly_pos.clone(),
            data_b: vec![],
            tol: 0.0,
        },
        Case {
            case_id: "cumsum_small".into(),
            op: "cumsum".into(),
            shape: vec![4],
            data: small.clone(),
            data_b: vec![],
            tol: 0.0,
        },
        Case {
            case_id: "cumsum_3x4".into(),
            op: "cumsum".into(),
            shape: vec![3, 4],
            data: pos_neg_3x4.clone(),
            data_b: vec![],
            tol: 0.0,
        },
        Case {
            case_id: "cumprod_small".into(),
            op: "cumprod".into(),
            shape: vec![4],
            data: small,
            data_b: vec![],
            tol: 0.0,
        },
        Case {
            case_id: "diff_pn".into(),
            op: "diff".into(),
            shape: vec![3, 4],
            data: pos_neg_3x4,
            data_b: vec![],
            tol: 0.0,
        },
        Case {
            case_id: "diff_strictly_pos".into(),
            op: "diff".into(),
            shape: vec![8],
            data: strictly_pos.clone(),
            data_b: vec![],
            tol: 0.0,
        },
    ];
    // equal_within: identical and slightly different fixtures
    let a_eq = strictly_pos.clone();
    let b_eq_same = strictly_pos.clone();
    let b_eq_diff: Vec<f64> = strictly_pos
        .iter()
        .enumerate()
        .map(|(i, &v)| if i % 2 == 0 { v + 0.01 } else { v })
        .collect();
    points.push(Case {
        case_id: "equal_within_identical".into(),
        op: "equal_within".into(),
        shape: vec![2, 4],
        data: a_eq.clone(),
        data_b: b_eq_same,
        tol: 1.0e-12,
    });
    points.push(Case {
        case_id: "equal_within_loose_tol".into(),
        op: "equal_within".into(),
        shape: vec![2, 4],
        data: a_eq.clone(),
        data_b: b_eq_diff.clone(),
        tol: 0.1,
    });
    points.push(Case {
        case_id: "equal_within_tight_tol".into(),
        op: "equal_within".into(),
        shape: vec![2, 4],
        data: a_eq,
        data_b: b_eq_diff,
        tol: 1.0e-6,
    });
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np

def finite_or_none(arr):
    out = []
    for v in arr:
        if not math.isfinite(float(v)):
            return None
        out.append(float(v))
    return out

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    shape = tuple(int(s) for s in case["shape"])
    a = np.array(case["data"], dtype=float).reshape(shape)
    try:
        if op == "count_nonzero":
            v = int(np.count_nonzero(a))
            points.append({"case_id": cid, "scalar": float(v), "values": None})
        elif op == "argmax":
            v = int(np.argmax(a.ravel()))
            points.append({"case_id": cid, "scalar": float(v), "values": None})
        elif op == "argmin":
            v = int(np.argmin(a.ravel()))
            points.append({"case_id": cid, "scalar": float(v), "values": None})
        elif op == "cumsum":
            v = np.cumsum(a.ravel()).tolist()
            points.append({"case_id": cid, "scalar": None, "values": finite_or_none(v)})
        elif op == "cumprod":
            v = np.cumprod(a.ravel()).tolist()
            points.append({"case_id": cid, "scalar": None, "values": finite_or_none(v)})
        elif op == "diff":
            v = np.diff(a.ravel()).tolist()
            points.append({"case_id": cid, "scalar": None, "values": finite_or_none(v)})
        elif op == "equal_within":
            b = np.array(case["data_b"], dtype=float).reshape(shape)
            tol = float(case["tol"])
            mask = (np.abs(a - b) <= tol).astype(float)
            points.append({"case_id": cid, "scalar": None, "values": finite_or_none(mask.ravel().tolist())})
        else:
            points.append({"case_id": cid, "scalar": None, "values": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "scalar": None, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize query");
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
                "failed to spawn python3 for reductions oracle: {e}"
            );
            eprintln!("skipping reductions oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "reductions oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping reductions oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for reductions oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "reductions oracle failed: {stderr}"
        );
        eprintln!("skipping reductions oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse reductions oracle JSON"))
}

fn vec_max_diff(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn diff_ndimage_reductions_args_cumops() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let Some(arm) = pmap.get(&case.case_id) else {
            continue;
        };
        let Ok(arr) = NdArray::new(case.data.clone(), case.shape.clone()) else {
            continue;
        };

        let abs_d = match case.op.as_str() {
            "count_nonzero" => {
                let actual = count_nonzero(&arr) as f64;
                let Some(e) = arm.scalar else { continue };
                (actual - e).abs()
            }
            "argmax" => {
                let actual = argmax(&arr) as f64;
                let Some(e) = arm.scalar else { continue };
                (actual - e).abs()
            }
            "argmin" => {
                let actual = argmin(&arr) as f64;
                let Some(e) = arm.scalar else { continue };
                (actual - e).abs()
            }
            "cumsum" => {
                let actual = cumsum_array(&arr);
                let Some(e) = arm.values.as_ref() else {
                    continue;
                };
                vec_max_diff(&actual.data, e)
            }
            "cumprod" => {
                let actual = cumprod_array(&arr);
                let Some(e) = arm.values.as_ref() else {
                    continue;
                };
                vec_max_diff(&actual.data, e)
            }
            "diff" => {
                let actual = diff_array(&arr);
                let Some(e) = arm.values.as_ref() else {
                    continue;
                };
                vec_max_diff(&actual.data, e)
            }
            "equal_within" => {
                let Ok(arr_b) = NdArray::new(case.data_b.clone(), case.shape.clone()) else {
                    continue;
                };
                let Ok(actual) = equal_within(&arr, &arr_b, case.tol) else {
                    continue;
                };
                let Some(e) = arm.values.as_ref() else {
                    continue;
                };
                vec_max_diff(&actual.data, e)
            }
            _ => continue,
        };

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
        test_id: "diff_ndimage_reductions_args_cumops".into(),
        category:
            "fsci_ndimage count_nonzero/argmax/argmin/cumsum/cumprod/diff/equal_within vs numpy"
                .into(),
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
            eprintln!("{} mismatch: {} abs_diff={}", d.op, d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "reductions/args/cumops conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
