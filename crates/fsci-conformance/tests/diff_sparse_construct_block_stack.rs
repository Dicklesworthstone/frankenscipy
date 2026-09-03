#![forbid(unsafe_code)]
//! Live scipy.sparse parity for fsci_sparse construction helpers.
//!
//! Resolves [frankenscipy-rdhku]. Covers four block-construction
//! helpers that previously had no dedicated diff harness:
//!   * block_diag(matrices)  vs scipy.sparse.block_diag
//!   * bmat(blocks)          vs scipy.sparse.bmat
//!   * vstack(blocks)        vs scipy.sparse.vstack
//!   * hstack(blocks)        vs scipy.sparse.hstack
//!
//! Each probe builds small CSR inputs in Rust, runs the fsci helper,
//! converts to a dense matrix, and compares element-wise to a scipy
//! oracle that performs the same construction.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_sparse::{
    CooMatrix, CsrMatrix, FormatConvertible, Shape2D, block_diag, bmat, hstack, vstack,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-14;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

/// Lightweight CSR description that survives JSON round-trips.
#[derive(Debug, Clone, Serialize)]
struct CooSpec {
    rows: usize,
    cols: usize,
    triplets: Vec<(usize, usize, f64)>,
}

#[derive(Debug, Clone, Serialize)]
struct CasePoint {
    case_id: String,
    /// "block_diag" | "vstack" | "hstack"
    op: String,
    /// Flat list of input matrices (for block_diag/vstack/hstack)
    inputs: Vec<CooSpec>,
    /// For bmat: row-major grid of (Some(idx in inputs) or None)
    /// stored as i64: -1 == None, else index into `inputs`
    bmat_layout_rows: usize,
    bmat_layout_cols: usize,
    bmat_layout: Vec<i64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<CasePoint>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct OraclePoint {
    case_id: String,
    rows: Option<usize>,
    cols: Option<usize>,
    /// Flattened row-major dense reconstruction
    dense: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<OraclePoint>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    op: String,
    rows: usize,
    cols: usize,
    max_abs_diff: f64,
    pass: bool,
    note: String,
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
    fs::create_dir_all(output_dir()).expect("create construct diff dir");
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

fn to_csr(spec: &CooSpec) -> CsrMatrix {
    let data: Vec<f64> = spec.triplets.iter().map(|t| t.2).collect();
    let rs: Vec<usize> = spec.triplets.iter().map(|t| t.0).collect();
    let cs: Vec<usize> = spec.triplets.iter().map(|t| t.1).collect();
    let coo = CooMatrix::from_triplets(Shape2D::new(spec.rows, spec.cols), data, rs, cs, true)
        .expect("COO build");
    coo.to_csr().expect("to_csr")
}

fn csr_to_dense(m: &CsrMatrix) -> (usize, usize, Vec<f64>) {
    let s = m.shape();
    let mut dense = vec![0.0_f64; s.rows * s.cols];
    let indptr = m.indptr();
    let indices = m.indices();
    let data = m.data();
    for i in 0..s.rows {
        for idx in indptr[i]..indptr[i + 1] {
            let j = indices[idx];
            dense[i * s.cols + j] = data[idx];
        }
    }
    (s.rows, s.cols, dense)
}

fn build_query() -> OracleQuery {
    let mut pts = Vec::new();

    // Shared building blocks
    let m_2x2 = CooSpec {
        rows: 2,
        cols: 2,
        triplets: vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0)],
    };
    let m_3x2 = CooSpec {
        rows: 3,
        cols: 2,
        triplets: vec![(0, 0, 5.0), (1, 1, 6.0), (2, 0, 7.0), (2, 1, 8.0)],
    };
    let m_2x3 = CooSpec {
        rows: 2,
        cols: 3,
        triplets: vec![(0, 0, 9.0), (0, 2, 10.0), (1, 1, 11.0)],
    };
    let m_3x3 = CooSpec {
        rows: 3,
        cols: 3,
        triplets: vec![(0, 0, 1.5), (1, 1, 2.5), (2, 2, 3.5), (0, 2, 0.5)],
    };

    // 1. block_diag of two 2x2
    pts.push(CasePoint {
        case_id: "block_diag_2x2_3x3".into(),
        op: "block_diag".into(),
        inputs: vec![m_2x2.clone(), m_3x3.clone()],
        bmat_layout_rows: 0,
        bmat_layout_cols: 0,
        bmat_layout: Vec::new(),
    });
    // 2. block_diag of 3 matrices
    pts.push(CasePoint {
        case_id: "block_diag_three".into(),
        op: "block_diag".into(),
        inputs: vec![m_2x2.clone(), m_3x2.clone(), m_2x3.clone()],
        bmat_layout_rows: 0,
        bmat_layout_cols: 0,
        bmat_layout: Vec::new(),
    });

    // 3. vstack of two same-cols matrices
    pts.push(CasePoint {
        case_id: "vstack_2x3_2x3".into(),
        op: "vstack".into(),
        inputs: vec![m_2x3.clone(), m_2x3.clone()],
        bmat_layout_rows: 0,
        bmat_layout_cols: 0,
        bmat_layout: Vec::new(),
    });

    // 4. hstack of two same-rows matrices
    pts.push(CasePoint {
        case_id: "hstack_2x2_2x3".into(),
        op: "hstack".into(),
        inputs: vec![m_2x2.clone(), m_2x3.clone()],
        bmat_layout_rows: 0,
        bmat_layout_cols: 0,
        bmat_layout: Vec::new(),
    });

    // 5. bmat 2x2 with None entries: [[A, None], [None, B]]
    pts.push(CasePoint {
        case_id: "bmat_2x2_diag".into(),
        op: "bmat".into(),
        inputs: vec![m_2x2.clone(), m_3x3.clone()],
        bmat_layout_rows: 2,
        bmat_layout_cols: 2,
        bmat_layout: vec![0, -1, -1, 1],
    });

    // 6. bmat 1x3 horizontal: [[A, B, C]]
    pts.push(CasePoint {
        case_id: "bmat_1x3".into(),
        op: "bmat".into(),
        inputs: vec![m_2x2.clone(), m_2x3.clone(), m_2x2.clone()],
        bmat_layout_rows: 1,
        bmat_layout_cols: 3,
        bmat_layout: vec![0, 1, 2],
    });

    // 7. bmat 2x1 vertical: [[A], [B]] with same cols
    pts.push(CasePoint {
        case_id: "bmat_2x1".into(),
        op: "bmat".into(),
        inputs: vec![m_2x3.clone(), m_2x3.clone()],
        bmat_layout_rows: 2,
        bmat_layout_cols: 1,
        bmat_layout: vec![0, 1],
    });

    OracleQuery { points: pts }
}

fn scipy_oracle_or_skip(q: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json, math, sys
import numpy as np
from scipy import sparse as sp

def to_csr(spec):
    data = [t[2] for t in spec["triplets"]]
    rows = [t[0] for t in spec["triplets"]]
    cols = [t[1] for t in spec["triplets"]]
    return sp.csr_matrix((data, (rows, cols)),
                         shape=(spec["rows"], spec["cols"]))

q = json.load(sys.stdin)
out = []
for c in q["points"]:
    cid = c["case_id"]
    op = c["op"]
    inputs = [to_csr(s) for s in c["inputs"]]
    try:
        if op == "block_diag":
            m = sp.block_diag(inputs).tocsr()
        elif op == "vstack":
            m = sp.vstack(inputs).tocsr()
        elif op == "hstack":
            m = sp.hstack(inputs).tocsr()
        elif op == "bmat":
            rows = int(c["bmat_layout_rows"])
            cols = int(c["bmat_layout_cols"])
            flat = c["bmat_layout"]
            blocks = []
            for i in range(rows):
                row = []
                for j in range(cols):
                    idx = flat[i * cols + j]
                    row.append(None if idx < 0 else inputs[idx])
                blocks.append(row)
            m = sp.bmat(blocks).tocsr()
        else:
            m = None
        if m is None:
            out.append({"case_id": cid, "rows": None, "cols": None, "dense": None})
        else:
            dense = m.toarray()
            if not np.all(np.isfinite(dense)):
                out.append({"case_id": cid, "rows": None, "cols": None, "dense": None})
            else:
                out.append({
                    "case_id": cid,
                    "rows": int(dense.shape[0]),
                    "cols": int(dense.shape[1]),
                    "dense": [float(v) for v in dense.flatten()],
                })
    except Exception:
        out.append({"case_id": cid, "rows": None, "cols": None, "dense": None})

print(json.dumps({"points": out}))
"#;
    let query_json = serde_json::to_string(q).expect("serialize");
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
                "python3 spawn failed: {e}"
            );
            eprintln!("skipping construct oracle: python3 unavailable ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping construct oracle: stdin write failed");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "oracle failed: {stderr}"
        );
        eprintln!("skipping construct oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse JSON"))
}

#[test]
fn diff_sparse_construct_block_stack() {
    let query = build_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();

    for (case, o) in query.points.iter().zip(oracle.points.iter()) {
        assert_eq!(case.case_id, o.case_id);
        let (Some(exp_rows), Some(exp_cols), Some(exp_dense)) = (o.rows, o.cols, o.dense.as_ref())
        else {
            continue;
        };

        let inputs_csr: Vec<CsrMatrix> = case.inputs.iter().map(to_csr).collect();

        let result_csr = match case.op.as_str() {
            "block_diag" => {
                let refs: Vec<&CsrMatrix> = inputs_csr.iter().collect();
                block_diag(&refs)
            }
            "vstack" => {
                let refs: Vec<&dyn FormatConvertible> = inputs_csr
                    .iter()
                    .map(|m| m as &dyn FormatConvertible)
                    .collect();
                vstack(&refs)
            }
            "hstack" => {
                let refs: Vec<&dyn FormatConvertible> = inputs_csr
                    .iter()
                    .map(|m| m as &dyn FormatConvertible)
                    .collect();
                hstack(&refs)
            }
            "bmat" => {
                let mut blocks: Vec<Vec<Option<&CsrMatrix>>> =
                    Vec::with_capacity(case.bmat_layout_rows);
                for i in 0..case.bmat_layout_rows {
                    let mut row: Vec<Option<&CsrMatrix>> =
                        Vec::with_capacity(case.bmat_layout_cols);
                    for j in 0..case.bmat_layout_cols {
                        let idx = case.bmat_layout[i * case.bmat_layout_cols + j];
                        row.push(if idx < 0 {
                            None
                        } else {
                            Some(&inputs_csr[idx as usize])
                        });
                    }
                    blocks.push(row);
                }
                bmat(&blocks)
            }
            other => panic!("unknown op {other}"),
        };

        let csr = match result_csr {
            Ok(m) => m,
            Err(e) => {
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    op: case.op.clone(),
                    rows: 0,
                    cols: 0,
                    max_abs_diff: f64::INFINITY,
                    pass: false,
                    note: format!("construct error: {e:?}"),
                });
                continue;
            }
        };
        let (rows, cols, dense) = csr_to_dense(&csr);
        if rows != exp_rows || cols != exp_cols {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                op: case.op.clone(),
                rows,
                cols,
                max_abs_diff: f64::INFINITY,
                pass: false,
                note: format!("shape mismatch: fsci {rows}x{cols} scipy {exp_rows}x{exp_cols}"),
            });
            continue;
        }
        let mut max_abs = 0.0_f64;
        for (a, e) in dense.iter().zip(exp_dense.iter()) {
            max_abs = max_abs.max((a - e).abs());
        }
        let pass = max_abs <= ABS_TOL;
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            rows,
            cols,
            max_abs_diff: max_abs,
            pass,
            note: String::new(),
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_sparse_construct_block_stack".into(),
        category: "fsci_sparse::{block_diag, bmat, vstack, hstack} vs scipy.sparse".into(),
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
                "construct mismatch: {} ({}) {}x{} max_abs={} note={}",
                d.case_id, d.op, d.rows, d.cols, d.max_abs_diff, d.note
            );
        }
    }

    assert!(all_pass, "construct parity failed: {} cases", diffs.len());
}
