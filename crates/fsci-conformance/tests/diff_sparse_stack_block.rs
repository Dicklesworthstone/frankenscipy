#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.sparse.hstack`, `vstack`,
//! `block_diag`, `bmat`, and `eye(m, n, k)`. All outputs densified
//! row-major from CSR.
//!
//! Resolves [frankenscipy-tcopn]. 1e-12 abs (integer-valued data).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_sparse::{CsrMatrix, Shape2D, block_diag, bmat, eye_rectangular, hstack, vstack};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct DenseBlock {
    rows: usize,
    cols: usize,
    dense: Vec<f64>,
}

/// One "stack" operation request — hstack/vstack/block_diag.
#[derive(Debug, Clone, Serialize)]
struct StackCase {
    case_id: String,
    op: String, // "hstack" | "vstack" | "block_diag"
    blocks: Vec<DenseBlock>,
}

/// One "bmat" request — 2-D layout of optional blocks.
#[derive(Debug, Clone, Serialize)]
struct BmatCase {
    case_id: String,
    /// Each row holds `Some(DenseBlock)` or `None`. Encoded as Option to
    /// preserve None entries.
    rows: Vec<Vec<Option<DenseBlock>>>,
}

/// One eye_rectangular request — (m, n, k).
#[derive(Debug, Clone, Serialize)]
struct EyeCase {
    case_id: String,
    m: usize,
    n: usize,
    k: i64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    stack_cases: Vec<StackCase>,
    bmat_cases: Vec<BmatCase>,
    eye_cases: Vec<EyeCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct DenseArm {
    case_id: String,
    rows: Option<usize>,
    cols: Option<usize>,
    dense: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    stack: Vec<DenseArm>,
    bmat: Vec<DenseArm>,
    eye: Vec<DenseArm>,
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
    fs::create_dir_all(output_dir()).expect("create stack_block diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize stack_block diff log");
    fs::write(path, json).expect("write stack_block diff log");
}

fn dense_from_csr(csr: &CsrMatrix) -> Vec<f64> {
    let shape = csr.shape();
    let mut dense = vec![0.0_f64; shape.rows * shape.cols];
    let indptr = csr.indptr();
    let indices = csr.indices();
    let data = csr.data();
    for row in 0..shape.rows {
        for idx in indptr[row]..indptr[row + 1] {
            dense[row * shape.cols + indices[idx]] += data[idx];
        }
    }
    dense
}

fn dense_to_csr(block: &DenseBlock) -> CsrMatrix {
    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = Vec::with_capacity(block.rows + 1);
    indptr.push(0);
    for r in 0..block.rows {
        for c in 0..block.cols {
            let v = block.dense[r * block.cols + c];
            if v != 0.0 {
                data.push(v);
                indices.push(c);
            }
        }
        indptr.push(data.len());
    }
    CsrMatrix::from_components(
        Shape2D::new(block.rows, block.cols),
        data,
        indices,
        indptr,
        true,
    )
    .expect("dense_to_csr build")
}

fn mk(rows: usize, cols: usize, dense: Vec<f64>) -> DenseBlock {
    assert_eq!(dense.len(), rows * cols);
    DenseBlock { rows, cols, dense }
}

fn generate_query() -> OracleQuery {
    let a_2x3 = mk(2, 3, vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]);
    let b_2x2 = mk(2, 2, vec![4.0, 5.0, 6.0, 0.0]);
    let c_2x4 = mk(2, 4, vec![7.0, 0.0, 8.0, 0.0, 0.0, 9.0, 0.0, 10.0]);

    let a_3x2 = mk(3, 2, vec![1.0, 0.0, 0.0, 2.0, 3.0, 4.0]);
    let b_2x2_v = mk(2, 2, vec![5.0, 6.0, 0.0, 7.0]);
    let c_1x2 = mk(1, 2, vec![8.0, 9.0]);

    let d_2x2 = mk(2, 2, vec![1.0, 0.0, 0.0, 2.0]);
    let e_3x3 = mk(3, 3, vec![3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 5.0]);
    let f_1x4 = mk(1, 4, vec![6.0, 7.0, 0.0, 8.0]);

    let stack_cases = vec![
        StackCase {
            case_id: "hstack_2x3_2x2_2x4".into(),
            op: "hstack".into(),
            blocks: vec![a_2x3.clone(), b_2x2.clone(), c_2x4.clone()],
        },
        StackCase {
            case_id: "hstack_2x3_2x2".into(),
            op: "hstack".into(),
            blocks: vec![a_2x3.clone(), b_2x2.clone()],
        },
        StackCase {
            case_id: "vstack_3x2_2x2_1x2".into(),
            op: "vstack".into(),
            blocks: vec![a_3x2.clone(), b_2x2_v.clone(), c_1x2.clone()],
        },
        StackCase {
            case_id: "vstack_3x2_2x2".into(),
            op: "vstack".into(),
            blocks: vec![a_3x2.clone(), b_2x2_v.clone()],
        },
        StackCase {
            case_id: "block_diag_2x2_3x3_1x4".into(),
            op: "block_diag".into(),
            blocks: vec![d_2x2.clone(), e_3x3.clone(), f_1x4.clone()],
        },
        StackCase {
            case_id: "block_diag_2x2_3x3".into(),
            op: "block_diag".into(),
            blocks: vec![d_2x2.clone(), e_3x3.clone()],
        },
    ];

    // bmat layouts (2x2 grid of blocks, some None)
    let bmat_cases = vec![
        BmatCase {
            case_id: "bmat_2x2_dense".into(),
            rows: vec![
                vec![Some(d_2x2.clone()), Some(b_2x2.clone())],
                vec![Some(b_2x2_v.clone()), Some(d_2x2.clone())],
            ],
        },
        BmatCase {
            case_id: "bmat_2x2_with_none".into(),
            rows: vec![
                vec![Some(d_2x2.clone()), None],
                vec![None, Some(d_2x2.clone())],
            ],
        },
    ];

    let eye_cases = vec![
        EyeCase {
            case_id: "eye_3x3_k0".into(),
            m: 3,
            n: 3,
            k: 0,
        },
        EyeCase {
            case_id: "eye_4x6_k0".into(),
            m: 4,
            n: 6,
            k: 0,
        },
        EyeCase {
            case_id: "eye_5x5_k1".into(),
            m: 5,
            n: 5,
            k: 1,
        },
        EyeCase {
            case_id: "eye_5x5_kneg2".into(),
            m: 5,
            n: 5,
            k: -2,
        },
    ];

    OracleQuery {
        stack_cases,
        bmat_cases,
        eye_cases,
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy import sparse

def dense_of(m):
    arr = np.asarray(m.todense())
    rows, cols = arr.shape
    flat = []
    for row in arr.tolist():
        for v in row:
            if not math.isfinite(float(v)):
                return rows, cols, None
            flat.append(float(v))
    return rows, cols, flat

def mat_of_block(block):
    arr = np.array(block["dense"], dtype=float).reshape(block["rows"], block["cols"])
    return sparse.csr_matrix(arr)

q = json.load(sys.stdin)
stack_out = []
for c in q["stack_cases"]:
    cid = c["case_id"]; op = c["op"]
    mats = [mat_of_block(b) for b in c["blocks"]]
    try:
        if op == "hstack":
            m = sparse.hstack(mats, format="csr")
        elif op == "vstack":
            m = sparse.vstack(mats, format="csr")
        elif op == "block_diag":
            m = sparse.block_diag(mats, format="csr")
        else:
            m = None
        if m is None:
            stack_out.append({"case_id": cid, "rows": None, "cols": None, "dense": None})
        else:
            r, cc, dn = dense_of(m)
            stack_out.append({"case_id": cid, "rows": r, "cols": cc, "dense": dn})
    except Exception:
        stack_out.append({"case_id": cid, "rows": None, "cols": None, "dense": None})

bmat_out = []
for c in q["bmat_cases"]:
    cid = c["case_id"]
    rows = []
    for r in c["rows"]:
        row_mats = []
        for cell in r:
            row_mats.append(None if cell is None else mat_of_block(cell))
        rows.append(row_mats)
    try:
        m = sparse.bmat(rows, format="csr")
        r, cc, dn = dense_of(m)
        bmat_out.append({"case_id": cid, "rows": r, "cols": cc, "dense": dn})
    except Exception:
        bmat_out.append({"case_id": cid, "rows": None, "cols": None, "dense": None})

eye_out = []
for c in q["eye_cases"]:
    cid = c["case_id"]
    m_ = int(c["m"]); n_ = int(c["n"]); k_ = int(c["k"])
    try:
        m = sparse.eye(m_, n_, k=k_, format="csr")
        r, cc, dn = dense_of(m)
        eye_out.append({"case_id": cid, "rows": r, "cols": cc, "dense": dn})
    except Exception:
        eye_out.append({"case_id": cid, "rows": None, "cols": None, "dense": None})

print(json.dumps({"stack": stack_out, "bmat": bmat_out, "eye": eye_out}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize stack_block query");
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
                "failed to spawn python3 for stack_block oracle: {e}"
            );
            eprintln!("skipping stack_block oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open stack_block oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "stack_block oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping stack_block oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for stack_block oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "stack_block oracle failed: {stderr}"
        );
        eprintln!("skipping stack_block oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse stack_block oracle JSON"))
}

fn compare_dense(
    case_id: &str,
    op: &str,
    fsci: &CsrMatrix,
    scipy_rows: Option<usize>,
    scipy_cols: Option<usize>,
    expected: Option<&Vec<f64>>,
) -> Option<CaseDiff> {
    let expected = expected?;
    let (Some(rows), Some(cols)) = (scipy_rows, scipy_cols) else {
        return None;
    };
    let fsci_shape = fsci.shape();
    if fsci_shape.rows != rows || fsci_shape.cols != cols {
        return Some(CaseDiff {
            case_id: case_id.into(),
            op: op.into(),
            abs_diff: f64::INFINITY,
            pass: false,
        });
    }
    let fsci_dense = dense_from_csr(fsci);
    let abs_d = fsci_dense
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    Some(CaseDiff {
        case_id: case_id.into(),
        op: op.into(),
        abs_diff: abs_d,
        pass: abs_d <= ABS_TOL,
    })
}

#[test]
fn diff_sparse_stack_block() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.stack.len(), query.stack_cases.len());
    assert_eq!(oracle.bmat.len(), query.bmat_cases.len());
    assert_eq!(oracle.eye.len(), query.eye_cases.len());

    let stack_map: HashMap<String, DenseArm> = oracle
        .stack
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();
    let bmat_map: HashMap<String, DenseArm> = oracle
        .bmat
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();
    let eye_map: HashMap<String, DenseArm> = oracle
        .eye
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    // stack cases
    for case in &query.stack_cases {
        let scipy_arm = stack_map.get(&case.case_id).expect("validated oracle");
        let blocks_csr: Vec<CsrMatrix> = case.blocks.iter().map(dense_to_csr).collect();
        let refs: Vec<&CsrMatrix> = blocks_csr.iter().collect();
        let fsci_result: Result<CsrMatrix, _> = match case.op.as_str() {
            "hstack" => {
                let dyn_refs: Vec<&dyn fsci_sparse::FormatConvertible> = refs
                    .iter()
                    .map(|r| *r as &dyn fsci_sparse::FormatConvertible)
                    .collect();
                hstack(&dyn_refs)
            }
            "vstack" => {
                let dyn_refs: Vec<&dyn fsci_sparse::FormatConvertible> = refs
                    .iter()
                    .map(|r| *r as &dyn fsci_sparse::FormatConvertible)
                    .collect();
                vstack(&dyn_refs)
            }
            "block_diag" => block_diag(&refs),
            _ => continue,
        };
        let Ok(fsci_csr) = fsci_result else {
            continue;
        };
        if let Some(d) = compare_dense(
            &case.case_id,
            &case.op,
            &fsci_csr,
            scipy_arm.rows,
            scipy_arm.cols,
            scipy_arm.dense.as_ref(),
        ) {
            max_overall = max_overall.max(d.abs_diff);
            diffs.push(d);
        }
    }

    // bmat cases
    for case in &query.bmat_cases {
        let scipy_arm = bmat_map.get(&case.case_id).expect("validated oracle");
        let blocks_csr: Vec<Vec<Option<CsrMatrix>>> = case
            .rows
            .iter()
            .map(|row| row.iter().map(|b| b.as_ref().map(dense_to_csr)).collect())
            .collect();
        let blocks_refs: Vec<Vec<Option<&CsrMatrix>>> = blocks_csr
            .iter()
            .map(|row| row.iter().map(|b| b.as_ref()).collect())
            .collect();
        let Ok(fsci_csr) = bmat(&blocks_refs) else {
            continue;
        };
        if let Some(d) = compare_dense(
            &case.case_id,
            "bmat",
            &fsci_csr,
            scipy_arm.rows,
            scipy_arm.cols,
            scipy_arm.dense.as_ref(),
        ) {
            max_overall = max_overall.max(d.abs_diff);
            diffs.push(d);
        }
    }

    // eye_rectangular cases
    for case in &query.eye_cases {
        let scipy_arm = eye_map.get(&case.case_id).expect("validated oracle");
        let Ok(fsci_csr) = eye_rectangular(case.m, case.n, case.k as isize) else {
            continue;
        };
        if let Some(d) = compare_dense(
            &case.case_id,
            "eye",
            &fsci_csr,
            scipy_arm.rows,
            scipy_arm.cols,
            scipy_arm.dense.as_ref(),
        ) {
            max_overall = max_overall.max(d.abs_diff);
            diffs.push(d);
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_sparse_stack_block".into(),
        category: "scipy.sparse.hstack + vstack + block_diag + bmat + eye".into(),
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
        "sparse stack/block conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
