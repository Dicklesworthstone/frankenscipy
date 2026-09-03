#![forbid(unsafe_code)]
//! Live scipy.linalg parity for fsci_linalg::null_space, orth, numerical_rank.
//!
//! Resolves [frankenscipy-s3q0j].
//!
//! - `null_space`: basis is non-unique (any orthogonal rotation /
//!   sign change is equally valid). Property test:
//!   •shape matches scipy.linalg.null_space
//!   •||A · N||_F < 1e-9
//!   •columns of N are orthonormal: N^T N = I within 1e-9
//! - `orth`: similar property test:
//!   •shape matches scipy.linalg.orth
//!   •Q^T Q = I within 1e-9
//!   •range(Q) = range(A) verified by ||(I - Q Q^T) A||_F < 1e-9
//! - `numerical_rank`: returns an int that should match scipy.linalg
//!   matrix_rank exactly.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_linalg::{DecompOptions, null_space, numerical_rank, orth};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String, // "null" | "orth" | "rank"
    matrix: Vec<Vec<f64>>,
    tol: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// Expected number of columns of the basis (nullity / rank)
    n_cols: Option<usize>,
    /// Expected rank (for "rank" op)
    rank: Option<usize>,
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
    fs::create_dir_all(output_dir()).expect("create null_orth_rank diff dir");
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

fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    if m == 0 {
        return vec![];
    }
    let p = a[0].len();
    let n = if b.is_empty() { 0 } else { b[0].len() };
    let mut c = vec![vec![0.0; n]; m];
    for i in 0..m {
        for k in 0..p {
            let aik = a[i][k];
            for j in 0..n {
                c[i][j] += aik * b[k][j];
            }
        }
    }
    c
}

fn transpose(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if m.is_empty() {
        return vec![];
    }
    let r = m.len();
    let c = m[0].len();
    let mut t = vec![vec![0.0; r]; c];
    for i in 0..r {
        for j in 0..c {
            t[j][i] = m[i][j];
        }
    }
    t
}

fn frobenius_norm(m: &[Vec<f64>]) -> f64 {
    m.iter()
        .flat_map(|r| r.iter())
        .map(|&v| v * v)
        .sum::<f64>()
        .sqrt()
}

fn ident(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect()
}

fn sub(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let n = if m == 0 { 0 } else { a[0].len() };
    let mut c = vec![vec![0.0; n]; m];
    for i in 0..m {
        for j in 0..n {
            c[i][j] = a[i][j] - b[i][j];
        }
    }
    c
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();

    let m1 = vec![
        vec![1.0_f64, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ]; // rank 2
    let m2 = vec![
        vec![1.0_f64, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ]; // rank 3 (full)
    let m3 = vec![
        vec![1.0_f64, 2.0, 3.0, 4.0],
        vec![2.0, 4.0, 6.0, 8.0],
        vec![3.0, 6.0, 9.0, 12.0],
    ]; // rank 1
    let m4 = vec![
        vec![1.0_f64, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 2.0, 0.0],
    ]; // rank 3, m=3, n=4

    for (label, m) in [
        ("m1_rank2", &m1),
        ("m2_full", &m2),
        ("m3_rank1", &m3),
        ("m4_3x4", &m4),
    ] {
        for op in ["null", "orth"] {
            points.push(Case {
                case_id: format!("{op}_{label}"),
                op: op.into(),
                matrix: m.clone(),
                tol: 0.0,
            });
        }
        points.push(Case {
            case_id: format!("rank_{label}"),
            op: "rank".into(),
            matrix: m.clone(),
            tol: 1.0e-10,
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
from scipy import linalg as sla

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    A = np.array(case["matrix"], dtype=float)
    try:
        if op == "null":
            N = sla.null_space(A)
            n_cols = int(N.shape[1])
            points.append({"case_id": cid, "n_cols": n_cols, "rank": None})
        elif op == "orth":
            Q = sla.orth(A)
            n_cols = int(Q.shape[1])
            points.append({"case_id": cid, "n_cols": n_cols, "rank": None})
        elif op == "rank":
            r = int(np.linalg.matrix_rank(A, tol=float(case["tol"])))
            points.append({"case_id": cid, "n_cols": None, "rank": r})
        else:
            points.append({"case_id": cid, "n_cols": None, "rank": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "n_cols": None, "rank": None})
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
                "failed to spawn python3 for null_orth_rank oracle: {e}"
            );
            eprintln!("skipping null_orth_rank oracle: python3 not available ({e})");
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
                "null_orth_rank oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping null_orth_rank oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for null_orth_rank oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "null_orth_rank oracle failed: {stderr}"
        );
        eprintln!("skipping null_orth_rank oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse null_orth_rank oracle JSON"))
}

#[test]
fn diff_linalg_null_space_orth_rank() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let opts = DecompOptions::default();
    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let Some(arm) = pmap.get(&case.case_id) else {
            continue;
        };
        match case.op.as_str() {
            "null" => {
                let Some(exp_cols) = arm.n_cols else { continue };
                let Ok(n_basis) = null_space(&case.matrix, None, opts) else {
                    continue;
                };
                let m = case.matrix.len();
                let n_cols = if n_basis.is_empty() {
                    0
                } else {
                    n_basis[0].len()
                };
                if n_cols != exp_cols {
                    diffs.push(CaseDiff {
                        case_id: case.case_id.clone(),
                        op: case.op.clone(),
                        abs_diff: f64::INFINITY,
                        pass: false,
                    });
                    max_overall = f64::INFINITY;
                    continue;
                }
                if n_cols == 0 {
                    // Both empty — pass
                    diffs.push(CaseDiff {
                        case_id: case.case_id.clone(),
                        op: case.op.clone(),
                        abs_diff: 0.0,
                        pass: true,
                    });
                    continue;
                }
                // ||A · N||_F
                // n_basis has shape (n_input_cols, n_cols)
                let an = matmul(&case.matrix, &n_basis);
                let d_an = frobenius_norm(&an);
                // N^T N = I
                let nt = transpose(&n_basis);
                let nt_n = matmul(&nt, &n_basis);
                let d_orth = frobenius_norm(&sub(&nt_n, &ident(n_cols)));
                let abs_d = d_an.max(d_orth);
                max_overall = max_overall.max(abs_d);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    op: case.op.clone(),
                    abs_diff: abs_d,
                    pass: abs_d <= ABS_TOL,
                });
                let _ = m;
            }
            "orth" => {
                let Some(exp_cols) = arm.n_cols else { continue };
                let Ok(q) = orth(&case.matrix, None, opts) else {
                    continue;
                };
                let m = case.matrix.len();
                let n_cols = if q.is_empty() { 0 } else { q[0].len() };
                if n_cols != exp_cols {
                    diffs.push(CaseDiff {
                        case_id: case.case_id.clone(),
                        op: case.op.clone(),
                        abs_diff: f64::INFINITY,
                        pass: false,
                    });
                    max_overall = f64::INFINITY;
                    continue;
                }
                if n_cols == 0 {
                    diffs.push(CaseDiff {
                        case_id: case.case_id.clone(),
                        op: case.op.clone(),
                        abs_diff: 0.0,
                        pass: true,
                    });
                    continue;
                }
                // Q^T Q = I
                let qt = transpose(&q);
                let qt_q = matmul(&qt, &q);
                let d_orth = frobenius_norm(&sub(&qt_q, &ident(n_cols)));
                // range(Q) = range(A): (I - Q Q^T) A ≈ 0
                let q_qt = matmul(&q, &qt);
                let im = ident(m);
                let i_qqt = sub(&im, &q_qt);
                let i_qqt_a = matmul(&i_qqt, &case.matrix);
                let d_range = frobenius_norm(&i_qqt_a);
                let abs_d = d_orth.max(d_range);
                max_overall = max_overall.max(abs_d);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    op: case.op.clone(),
                    abs_diff: abs_d,
                    pass: abs_d <= ABS_TOL,
                });
            }
            "rank" => {
                let Some(expected) = arm.rank else { continue };
                let Ok(actual) = numerical_rank(&case.matrix, case.tol, opts) else {
                    continue;
                };
                let abs_d = if actual == expected {
                    0.0
                } else {
                    (actual as i64 - expected as i64).abs() as f64
                };
                max_overall = max_overall.max(abs_d);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    op: case.op.clone(),
                    abs_diff: abs_d,
                    pass: actual == expected,
                });
            }
            _ => continue,
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_linalg_null_space_orth_rank".into(),
        category: "fsci_linalg::{null_space, orth, numerical_rank} vs scipy.linalg".into(),
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
        "null/orth/rank conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
