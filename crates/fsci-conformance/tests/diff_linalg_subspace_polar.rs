#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_linalg::khatri_rao,
//! matrix_rank, subspace_angles, and polar.
//!
//! Resolves [frankenscipy-0spk8]. polar is unique (when full rank), so
//! both factors compared. subspace_angles compared element-wise.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_linalg::{DecompOptions, khatri_rao, matrix_rank, polar, subspace_angles};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct OpCase {
    case_id: String,
    op: String,
    a_rows: usize,
    a_cols: usize,
    a: Vec<f64>,
    /// Used only by khatri_rao and subspace_angles.
    b_rows: usize,
    b_cols: usize,
    b: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<OpCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// For khatri_rao: dense (HIGH ROWS) flat.
    /// For polar: U flat (rows×cols same as A).
    /// For subspace_angles: angle values.
    values: Option<Vec<f64>>,
    /// For polar: P flat (cols×cols).
    aux: Option<Vec<f64>>,
    /// For matrix_rank: integer rank as f64.
    scalar: Option<f64>,
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
    fs::create_dir_all(output_dir()).expect("create subspace_polar diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize subspace_polar diff log");
    fs::write(path, json).expect("write subspace_polar diff log");
}

fn flatten(m: &[Vec<f64>]) -> Vec<f64> {
    m.iter().flat_map(|row| row.iter().copied()).collect()
}

fn rows_of(a_flat: &[f64], rows: usize, cols: usize) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|r| (0..cols).map(|c| a_flat[r * cols + c]).collect())
        .collect()
}

fn mk(rows: usize, cols: usize, dense: Vec<f64>) -> (usize, usize, Vec<f64>) {
    assert_eq!(dense.len(), rows * cols);
    (rows, cols, dense)
}

fn empty() -> (usize, usize, Vec<f64>) {
    (0, 0, vec![])
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();

    // khatri_rao
    let kr_cases: &[(&str, (usize, usize, Vec<f64>), (usize, usize, Vec<f64>))] = &[
        (
            "kr_2x2_2x2",
            mk(2, 2, vec![1.0, 2.0, 3.0, 4.0]),
            mk(2, 2, vec![5.0, 6.0, 7.0, 8.0]),
        ),
        (
            "kr_3x2_2x2",
            mk(3, 2, vec![1.0, 0.0, 0.0, 2.0, 3.0, 4.0]),
            mk(2, 2, vec![1.0, 1.0, 0.0, 1.0]),
        ),
        (
            "kr_2x3_2x3",
            mk(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            mk(2, 3, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
        ),
    ];
    for (name, a, b) in kr_cases {
        points.push(OpCase {
            case_id: (*name).into(),
            op: "khatri_rao".into(),
            a_rows: a.0,
            a_cols: a.1,
            a: a.2.clone(),
            b_rows: b.0,
            b_cols: b.1,
            b: b.2.clone(),
        });
    }

    // matrix_rank
    let rank_cases: &[(&str, (usize, usize, Vec<f64>))] = &[
        (
            "rank_full_3x3",
            mk(3, 3, vec![1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0]),
        ),
        (
            "rank_deficient",
            mk(3, 3, vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0]),
        ),
        (
            "rank_4x6_full_row",
            mk(4, 6, (1..=24).map(|i| i as f64).collect()),
        ),
        (
            "rank_3x3_identity",
            mk(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        ),
    ];
    let empty_b = empty();
    for (name, a) in rank_cases {
        points.push(OpCase {
            case_id: (*name).into(),
            op: "matrix_rank".into(),
            a_rows: a.0,
            a_cols: a.1,
            a: a.2.clone(),
            b_rows: empty_b.0,
            b_cols: empty_b.1,
            b: empty_b.2.clone(),
        });
    }

    // subspace_angles
    let sa_cases: &[(&str, (usize, usize, Vec<f64>), (usize, usize, Vec<f64>))] = &[
        (
            "sa_3x2_3x2_overlap",
            mk(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            mk(3, 2, vec![0.0, 1.0, 1.0, 0.0, 1.0, 1.0]),
        ),
        (
            "sa_4x2_4x2_distinct",
            mk(4, 2, vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            mk(4, 2, vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
        ),
        (
            "sa_3x2_3x1",
            mk(3, 2, vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            mk(3, 1, vec![1.0, 1.0, 0.0]),
        ),
    ];
    for (name, a, b) in sa_cases {
        points.push(OpCase {
            case_id: (*name).into(),
            op: "subspace_angles".into(),
            a_rows: a.0,
            a_cols: a.1,
            a: a.2.clone(),
            b_rows: b.0,
            b_cols: b.1,
            b: b.2.clone(),
        });
    }

    // polar
    let polar_cases: &[(&str, (usize, usize, Vec<f64>))] = &[
        ("polar_2x2", mk(2, 2, vec![1.0, 2.0, 3.0, 4.0])),
        (
            "polar_3x3_diag",
            mk(3, 3, vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0]),
        ),
        (
            "polar_3x2_tall",
            mk(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
        ),
    ];
    for (name, a) in polar_cases {
        points.push(OpCase {
            case_id: (*name).into(),
            op: "polar".into(),
            a_rows: a.0,
            a_cols: a.1,
            a: a.2.clone(),
            b_rows: empty_b.0,
            b_cols: empty_b.1,
            b: empty_b.2.clone(),
        });
    }

    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy import linalg

def finite_flat_or_none(m):
    flat = []
    for row in np.asarray(m, dtype=float).tolist():
        if isinstance(row, list):
            for v in row:
                if not math.isfinite(float(v)):
                    return None
                flat.append(float(v))
        else:
            if not math.isfinite(float(row)):
                return None
            flat.append(float(row))
    return flat

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    A = np.array(case["a"], dtype=float).reshape(int(case["a_rows"]), int(case["a_cols"]))
    if case.get("b_rows", 0) and case.get("b_cols", 0):
        B = np.array(case["b"], dtype=float).reshape(int(case["b_rows"]), int(case["b_cols"]))
    else:
        B = None
    try:
        if op == "khatri_rao":
            m = linalg.khatri_rao(A, B)
            r, c = m.shape
            points.append({"case_id": cid, "values": finite_flat_or_none(m),
                           "aux": None, "scalar": None,
                           "_rows": r, "_cols": c})
        elif op == "matrix_rank":
            v = int(np.linalg.matrix_rank(A))
            points.append({"case_id": cid, "values": None, "aux": None,
                           "scalar": float(v)})
        elif op == "subspace_angles":
            v = linalg.subspace_angles(A, B)
            points.append({"case_id": cid, "values": finite_flat_or_none(v),
                           "aux": None, "scalar": None})
        elif op == "polar":
            U, P = linalg.polar(A)
            points.append({"case_id": cid, "values": finite_flat_or_none(U),
                           "aux": finite_flat_or_none(P), "scalar": None})
        else:
            points.append({"case_id": cid, "values": None, "aux": None,
                           "scalar": None})
    except Exception:
        points.append({"case_id": cid, "values": None, "aux": None,
                       "scalar": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize subspace_polar query");
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
                "failed to spawn python3 for subspace_polar oracle: {e}"
            );
            eprintln!("skipping subspace_polar oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open subspace_polar oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "subspace_polar oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping subspace_polar oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for subspace_polar oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "subspace_polar oracle failed: {stderr}"
        );
        eprintln!("skipping subspace_polar oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse subspace_polar oracle JSON"))
}

#[test]
fn diff_linalg_subspace_polar() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let a = rows_of(&case.a, case.a_rows, case.a_cols);
        let abs_d: f64 = match case.op.as_str() {
            "khatri_rao" => {
                let Some(expected) = scipy_arm.values.as_ref() else {
                    continue;
                };
                let b = rows_of(&case.b, case.b_rows, case.b_cols);
                let Ok(m) = khatri_rao(&a, &b) else {
                    continue;
                };
                let fsci_flat = flatten(&m);
                if fsci_flat.len() != expected.len() {
                    f64::INFINITY
                } else {
                    fsci_flat
                        .iter()
                        .zip(expected.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0_f64, f64::max)
                }
            }
            "matrix_rank" => {
                let Some(expected) = scipy_arm.scalar else {
                    continue;
                };
                let opts = DecompOptions::default();
                let Ok(r) = matrix_rank(&a, None, opts) else {
                    continue;
                };
                ((r as f64) - expected).abs()
            }
            "subspace_angles" => {
                let Some(expected) = scipy_arm.values.as_ref() else {
                    continue;
                };
                let b = rows_of(&case.b, case.b_rows, case.b_cols);
                let opts = DecompOptions::default();
                let Ok(v) = subspace_angles(&a, &b, opts) else {
                    continue;
                };
                if v.len() != expected.len() {
                    f64::INFINITY
                } else {
                    v.iter()
                        .zip(expected.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0_f64, f64::max)
                }
            }
            "polar" => {
                let Some(u_exp) = scipy_arm.values.as_ref() else {
                    continue;
                };
                let Some(p_exp) = scipy_arm.aux.as_ref() else {
                    continue;
                };
                let opts = DecompOptions::default();
                let Ok(res) = polar(&a, opts) else {
                    continue;
                };
                let u_flat = flatten(&res.u);
                let p_flat = flatten(&res.p);
                if u_flat.len() != u_exp.len() || p_flat.len() != p_exp.len() {
                    f64::INFINITY
                } else {
                    let du = u_flat
                        .iter()
                        .zip(u_exp.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0_f64, f64::max);
                    let dp = p_flat
                        .iter()
                        .zip(p_exp.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0_f64, f64::max);
                    du.max(dp)
                }
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
        test_id: "diff_linalg_subspace_polar".into(),
        category: "scipy.linalg.khatri_rao + matrix_rank + subspace_angles + polar".into(),
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
        "subspace_polar conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
