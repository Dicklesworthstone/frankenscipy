#![forbid(unsafe_code)]
//! Live scipy.sparse parity for fsci_sparse basic ops:
//! sparse_add, sparse_scale, sparse_transpose, sparse_nnz,
//! sparse_is_symmetric.
//!
//! Resolves [frankenscipy-wf3yt]. All deterministic; 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_sparse::{
    CooMatrix, CsrMatrix, FormatConvertible, Shape2D, sparse_add, sparse_is_symmetric, sparse_nnz,
    sparse_scale, sparse_transpose,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String, // "add" | "scale" | "transpose" | "nnz" | "is_sym"
    a_rows: usize,
    a_cols: usize,
    a_triplets: Vec<(usize, usize, f64)>,
    b_rows: usize,
    b_cols: usize,
    b_triplets: Vec<(usize, usize, f64)>,
    alpha: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// dense flatten for add/scale/transpose
    dense: Option<Vec<f64>>,
    #[allow(dead_code)]
    out_rows: Option<usize>,
    #[allow(dead_code)]
    out_cols: Option<usize>,
    /// scalar count for nnz
    nnz: Option<usize>,
    /// boolean for is_sym
    is_sym: Option<bool>,
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
    fs::create_dir_all(output_dir()).expect("create sparse_basic diff dir");
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

fn csr_to_dense(c: &CsrMatrix) -> Vec<f64> {
    let s = c.shape();
    let mut out = vec![0.0_f64; s.rows * s.cols];
    let indptr = c.indptr();
    let indices = c.indices();
    let data = c.data();
    for r in 0..s.rows {
        let start = indptr[r];
        let end = indptr[r + 1];
        for idx in start..end {
            out[r * s.cols + indices[idx]] += data[idx];
        }
    }
    out
}

fn build_csr(rows: usize, cols: usize, trips: &[(usize, usize, f64)]) -> Option<CsrMatrix> {
    let mut data = Vec::new();
    let mut rs = Vec::new();
    let mut cs = Vec::new();
    for &(r, c, v) in trips {
        data.push(v);
        rs.push(r);
        cs.push(c);
    }
    let coo = CooMatrix::from_triplets(Shape2D::new(rows, cols), data, rs, cs, true).ok()?;
    coo.to_csr().ok()
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();

    let a_3x3 = vec![
        (0, 0, 2.0_f64),
        (0, 2, -1.0),
        (1, 1, 3.0),
        (2, 0, 1.0),
        (2, 2, 4.0),
    ];
    let b_3x3 = vec![(0, 1, 1.0_f64), (1, 0, 0.5), (1, 2, -0.3), (2, 1, 2.0)];
    let a_5x4 = vec![
        (0, 0, 1.0_f64),
        (0, 2, 2.0),
        (1, 1, 3.0),
        (1, 3, -1.0),
        (2, 2, 4.0),
        (3, 0, 0.5),
        (4, 3, 1.5),
    ];
    let b_5x4 = vec![
        (0, 1, -0.5_f64),
        (1, 0, 0.7),
        (2, 3, 2.5),
        (3, 3, 1.0),
        (4, 0, 0.3),
    ];
    // symmetric
    let sym_3 = vec![
        (0, 0, 2.0_f64),
        (0, 1, 1.0),
        (0, 2, -0.5),
        (1, 0, 1.0),
        (1, 1, 3.0),
        (1, 2, 0.7),
        (2, 0, -0.5),
        (2, 1, 0.7),
        (2, 2, 4.0),
    ];
    // not symmetric
    let asym_3 = vec![(0, 1, 1.0_f64), (1, 0, -1.0), (1, 2, 2.0)];

    // sparse_add
    points.push(Case {
        case_id: "add_3x3".into(),
        op: "add".into(),
        a_rows: 3,
        a_cols: 3,
        a_triplets: a_3x3.clone(),
        b_rows: 3,
        b_cols: 3,
        b_triplets: b_3x3.clone(),
        alpha: 0.0,
    });
    points.push(Case {
        case_id: "add_5x4".into(),
        op: "add".into(),
        a_rows: 5,
        a_cols: 4,
        a_triplets: a_5x4.clone(),
        b_rows: 5,
        b_cols: 4,
        b_triplets: b_5x4.clone(),
        alpha: 0.0,
    });

    // sparse_scale
    for &alpha in &[0.0_f64, 1.0, -1.0, 2.5, 0.5] {
        points.push(Case {
            case_id: format!("scale_3x3_a{alpha}"),
            op: "scale".into(),
            a_rows: 3,
            a_cols: 3,
            a_triplets: a_3x3.clone(),
            b_rows: 0,
            b_cols: 0,
            b_triplets: vec![],
            alpha,
        });
    }

    // sparse_transpose
    points.push(Case {
        case_id: "transpose_3x3".into(),
        op: "transpose".into(),
        a_rows: 3,
        a_cols: 3,
        a_triplets: a_3x3.clone(),
        b_rows: 0,
        b_cols: 0,
        b_triplets: vec![],
        alpha: 0.0,
    });
    points.push(Case {
        case_id: "transpose_5x4".into(),
        op: "transpose".into(),
        a_rows: 5,
        a_cols: 4,
        a_triplets: a_5x4.clone(),
        b_rows: 0,
        b_cols: 0,
        b_triplets: vec![],
        alpha: 0.0,
    });

    // sparse_nnz
    points.push(Case {
        case_id: "nnz_3x3".into(),
        op: "nnz".into(),
        a_rows: 3,
        a_cols: 3,
        a_triplets: a_3x3.clone(),
        b_rows: 0,
        b_cols: 0,
        b_triplets: vec![],
        alpha: 0.0,
    });
    points.push(Case {
        case_id: "nnz_5x4".into(),
        op: "nnz".into(),
        a_rows: 5,
        a_cols: 4,
        a_triplets: a_5x4.clone(),
        b_rows: 0,
        b_cols: 0,
        b_triplets: vec![],
        alpha: 0.0,
    });
    points.push(Case {
        case_id: "nnz_explicit_zero_2x2".into(),
        op: "nnz".into(),
        a_rows: 2,
        a_cols: 2,
        a_triplets: vec![(0, 0, 0.0), (1, 1, 2.0)],
        b_rows: 0,
        b_cols: 0,
        b_triplets: vec![],
        alpha: 0.0,
    });

    // is_symmetric
    points.push(Case {
        case_id: "is_sym_yes".into(),
        op: "is_sym".into(),
        a_rows: 3,
        a_cols: 3,
        a_triplets: sym_3,
        b_rows: 0,
        b_cols: 0,
        b_triplets: vec![],
        alpha: 0.0,
    });
    points.push(Case {
        case_id: "is_sym_no".into(),
        op: "is_sym".into(),
        a_rows: 3,
        a_cols: 3,
        a_triplets: asym_3,
        b_rows: 0,
        b_cols: 0,
        b_triplets: vec![],
        alpha: 0.0,
    });

    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy.sparse import csr_matrix

def build(rows, cols, trips):
    if not trips:
        return csr_matrix((rows, cols))
    rs = [int(t[0]) for t in trips]
    cs = [int(t[1]) for t in trips]
    vs = [float(t[2]) for t in trips]
    return csr_matrix((vs, (rs, cs)), shape=(rows, cols)).astype(float)

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    a_rows = int(case["a_rows"]); a_cols = int(case["a_cols"])
    try:
        A = build(a_rows, a_cols, case["a_triplets"])
        if op == "add":
            B = build(int(case["b_rows"]), int(case["b_cols"]), case["b_triplets"])
            R = (A + B)
            D = np.asarray(R.todense())
            flat = [float(v) for v in D.flatten().tolist()]
            points.append({"case_id": cid, "dense": flat, "out_rows": int(D.shape[0]), "out_cols": int(D.shape[1]),
                           "nnz": None, "is_sym": None})
        elif op == "scale":
            alpha = float(case["alpha"])
            R = A.multiply(alpha)
            D = np.asarray(R.todense())
            flat = [float(v) for v in D.flatten().tolist()]
            points.append({"case_id": cid, "dense": flat, "out_rows": int(D.shape[0]), "out_cols": int(D.shape[1]),
                           "nnz": None, "is_sym": None})
        elif op == "transpose":
            R = A.T
            D = np.asarray(R.todense())
            flat = [float(v) for v in D.flatten().tolist()]
            points.append({"case_id": cid, "dense": flat, "out_rows": int(D.shape[0]), "out_cols": int(D.shape[1]),
                           "nnz": None, "is_sym": None})
        elif op == "nnz":
            n = int(A.nnz)
            points.append({"case_id": cid, "dense": None, "out_rows": None, "out_cols": None, "nnz": n, "is_sym": None})
        elif op == "is_sym":
            D = np.asarray(A.todense())
            sym = bool(np.allclose(D, D.T, atol=1e-12))
            points.append({"case_id": cid, "dense": None, "out_rows": None, "out_cols": None, "nnz": None, "is_sym": sym})
        else:
            points.append({"case_id": cid, "dense": None, "out_rows": None, "out_cols": None, "nnz": None, "is_sym": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "dense": None, "out_rows": None, "out_cols": None, "nnz": None, "is_sym": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize query");
    let mut child = match Command::new("python3")
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
                "failed to spawn python3 for sparse_basic oracle: {e}"
            );
            eprintln!("skipping sparse_basic oracle: python3 not available ({e})");
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
                "sparse_basic oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping sparse_basic oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for sparse_basic oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "sparse_basic oracle failed: {stderr}"
        );
        eprintln!("skipping sparse_basic oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse sparse_basic oracle JSON"))
}

#[test]
fn diff_sparse_basic_ops() {
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
        let Some(csr_a) = build_csr(case.a_rows, case.a_cols, &case.a_triplets) else {
            continue;
        };
        let abs_d = match case.op.as_str() {
            "add" => {
                let Some(csr_b) = build_csr(case.b_rows, case.b_cols, &case.b_triplets) else {
                    continue;
                };
                let Some(expected) = arm.dense.as_ref() else {
                    continue;
                };
                let result = sparse_add(&csr_a, &csr_b);
                let d = csr_to_dense(&result);
                if d.len() != expected.len() {
                    f64::INFINITY
                } else {
                    d.iter()
                        .zip(expected.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0_f64, f64::max)
                }
            }
            "scale" => {
                let Some(expected) = arm.dense.as_ref() else {
                    continue;
                };
                let result = sparse_scale(&csr_a, case.alpha);
                let d = csr_to_dense(&result);
                if d.len() != expected.len() {
                    f64::INFINITY
                } else {
                    d.iter()
                        .zip(expected.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0_f64, f64::max)
                }
            }
            "transpose" => {
                let Some(expected) = arm.dense.as_ref() else {
                    continue;
                };
                let result = sparse_transpose(&csr_a);
                let d = csr_to_dense(&result);
                if d.len() != expected.len() {
                    f64::INFINITY
                } else {
                    d.iter()
                        .zip(expected.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0_f64, f64::max)
                }
            }
            "nnz" => {
                let Some(expected) = arm.nnz else {
                    continue;
                };
                let actual = sparse_nnz(&csr_a);
                if actual == expected {
                    0.0
                } else {
                    (actual as i64 - expected as i64).abs() as f64
                }
            }
            "is_sym" => {
                let Some(expected) = arm.is_sym else {
                    continue;
                };
                let actual = sparse_is_symmetric(&csr_a, 1e-12);
                if actual == expected { 0.0 } else { 1.0 }
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
        test_id: "diff_sparse_basic_ops".into(),
        category: "fsci_sparse::{add, scale, transpose, nnz, is_symmetric} vs scipy.sparse".into(),
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
        "sparse_basic_ops conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
