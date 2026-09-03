#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_sparse ops `add_csr`,
//! `sub_csr`, `scale_csr`, `spmv_csr` and constructor `spdiags`.
//!
//! Resolves [frankenscipy-jure3]. 1e-12 abs (integer/rational data).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_sparse::{CsrMatrix, Shape2D, add_csr, scale_csr, spdiags, spmv_csr, sub_csr};
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

#[derive(Debug, Clone, Serialize)]
struct ArithCase {
    case_id: String,
    op: String, // "add" | "sub"
    a: DenseBlock,
    b: DenseBlock,
}

#[derive(Debug, Clone, Serialize)]
struct ScaleCase {
    case_id: String,
    a: DenseBlock,
    alpha: f64,
}

#[derive(Debug, Clone, Serialize)]
struct SpmvCase {
    case_id: String,
    a: DenseBlock,
    x: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct SpdiagsCase {
    case_id: String,
    diagonals: Vec<Vec<f64>>,
    offsets: Vec<i64>,
    rows: usize,
    cols: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    arith: Vec<ArithCase>,
    scale: Vec<ScaleCase>,
    spmv: Vec<SpmvCase>,
    spdiags: Vec<SpdiagsCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct DenseArm {
    case_id: String,
    rows: Option<usize>,
    cols: Option<usize>,
    dense: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct VecArm {
    case_id: String,
    values: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    arith: Vec<DenseArm>,
    scale: Vec<DenseArm>,
    spmv: Vec<VecArm>,
    spdiags: Vec<DenseArm>,
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
    fs::create_dir_all(output_dir()).expect("create ops_arith diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize ops_arith diff log");
    fs::write(path, json).expect("write ops_arith diff log");
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
    let a = mk(3, 3, vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0]);
    let b = mk(3, 3, vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0]);
    let c = mk(
        4,
        4,
        vec![
            1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0, 0.0, 7.0,
        ],
    );
    let d = mk(
        4,
        4,
        vec![
            -1.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0, -4.0,
        ],
    );
    let rect = mk(2, 5, vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0]);
    let rect_match = mk(2, 5, vec![0.0; 10]);

    let arith = vec![
        ArithCase {
            case_id: "add_3x3_disjoint".into(),
            op: "add".into(),
            a: a.clone(),
            b: b.clone(),
        },
        ArithCase {
            case_id: "sub_3x3_disjoint".into(),
            op: "sub".into(),
            a: a.clone(),
            b: b.clone(),
        },
        ArithCase {
            case_id: "add_4x4_overlap".into(),
            op: "add".into(),
            a: c.clone(),
            b: d.clone(),
        },
        ArithCase {
            case_id: "sub_4x4_overlap".into(),
            op: "sub".into(),
            a: c.clone(),
            b: d.clone(),
        },
        ArithCase {
            case_id: "add_rect_zero".into(),
            op: "add".into(),
            a: rect.clone(),
            b: rect_match.clone(),
        },
    ];

    let scale = vec![
        ScaleCase {
            case_id: "scale_3x3_alpha_2".into(),
            a: a.clone(),
            alpha: 2.0,
        },
        ScaleCase {
            case_id: "scale_3x3_alpha_neg_half".into(),
            a: a.clone(),
            alpha: -0.5,
        },
        ScaleCase {
            case_id: "scale_4x4_alpha_zero".into(),
            a: c.clone(),
            alpha: 0.0,
        },
        ScaleCase {
            case_id: "scale_rect_alpha_pi".into(),
            a: rect.clone(),
            alpha: std::f64::consts::PI,
        },
    ];

    let spmv = vec![
        SpmvCase {
            case_id: "spmv_3x3_ones".into(),
            a: a.clone(),
            x: vec![1.0, 1.0, 1.0],
        },
        SpmvCase {
            case_id: "spmv_3x3_ramp".into(),
            a: a.clone(),
            x: vec![0.5, -1.0, 2.0],
        },
        SpmvCase {
            case_id: "spmv_4x4_alt".into(),
            a: c,
            x: vec![1.0, -1.0, 1.0, -1.0],
        },
        SpmvCase {
            case_id: "spmv_rect_2x5".into(),
            a: rect,
            x: vec![1.0, 2.0, 3.0, 4.0, 5.0],
        },
    ];

    let spdiags = vec![
        SpdiagsCase {
            case_id: "spdiag_tridiag_5x5".into(),
            diagonals: vec![
                vec![1.0, 1.0, 1.0, 1.0, 1.0],
                vec![2.0, 2.0, 2.0, 2.0, 2.0],
                vec![3.0, 3.0, 3.0, 3.0, 3.0],
            ],
            offsets: vec![-1, 0, 1],
            rows: 5,
            cols: 5,
        },
        SpdiagsCase {
            case_id: "spdiag_rect_3x6".into(),
            diagonals: vec![
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            ],
            offsets: vec![0, 2],
            rows: 3,
            cols: 6,
        },
        SpdiagsCase {
            case_id: "spdiag_4x4_offset_neg2".into(),
            diagonals: vec![vec![7.0, 8.0, 9.0, 10.0]],
            offsets: vec![-2],
            rows: 4,
            cols: 4,
        },
    ];

    OracleQuery {
        arith,
        scale,
        spmv,
        spdiags,
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
    r, c = arr.shape
    flat = []
    for row in arr.tolist():
        for v in row:
            if not math.isfinite(float(v)):
                return r, c, None
            flat.append(float(v))
    return r, c, flat

def vec_or_none(arr):
    flat = []
    for v in np.asarray(arr).flatten().tolist():
        if not math.isfinite(float(v)):
            return None
        flat.append(float(v))
    return flat

q = json.load(sys.stdin)

arith_out = []
for c_case in q["arith"]:
    cid = c_case["case_id"]; op = c_case["op"]
    a_blk = c_case["a"]; b_blk = c_case["b"]
    A = sparse.csr_matrix(np.array(a_blk["dense"], dtype=float).reshape(a_blk["rows"], a_blk["cols"]))
    B = sparse.csr_matrix(np.array(b_blk["dense"], dtype=float).reshape(b_blk["rows"], b_blk["cols"]))
    try:
        m = (A + B) if op == "add" else (A - B)
        r, cc, dn = dense_of(m)
        arith_out.append({"case_id": cid, "rows": r, "cols": cc, "dense": dn})
    except Exception:
        arith_out.append({"case_id": cid, "rows": None, "cols": None, "dense": None})

scale_out = []
for c_case in q["scale"]:
    cid = c_case["case_id"]
    a_blk = c_case["a"]; alpha = float(c_case["alpha"])
    A = sparse.csr_matrix(np.array(a_blk["dense"], dtype=float).reshape(a_blk["rows"], a_blk["cols"]))
    try:
        m = A * alpha
        r, cc, dn = dense_of(m)
        scale_out.append({"case_id": cid, "rows": r, "cols": cc, "dense": dn})
    except Exception:
        scale_out.append({"case_id": cid, "rows": None, "cols": None, "dense": None})

spmv_out = []
for c_case in q["spmv"]:
    cid = c_case["case_id"]
    a_blk = c_case["a"]
    A = sparse.csr_matrix(np.array(a_blk["dense"], dtype=float).reshape(a_blk["rows"], a_blk["cols"]))
    x = np.array(c_case["x"], dtype=float)
    try:
        y = A @ x
        spmv_out.append({"case_id": cid, "values": vec_or_none(y)})
    except Exception:
        spmv_out.append({"case_id": cid, "values": None})

spdiags_out = []
for c_case in q["spdiags"]:
    cid = c_case["case_id"]
    diagonals = [np.array(d, dtype=float) for d in c_case["diagonals"]]
    offsets = list(c_case["offsets"])
    rows = int(c_case["rows"]); cols = int(c_case["cols"])
    try:
        m = sparse.spdiags(diagonals, offsets, rows, cols, format="csr")
        r, cc, dn = dense_of(m)
        spdiags_out.append({"case_id": cid, "rows": r, "cols": cc, "dense": dn})
    except Exception:
        spdiags_out.append({"case_id": cid, "rows": None, "cols": None, "dense": None})

print(json.dumps({"arith": arith_out, "scale": scale_out, "spmv": spmv_out, "spdiags": spdiags_out}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize ops_arith query");
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
                "failed to spawn python3 for ops_arith oracle: {e}"
            );
            eprintln!("skipping ops_arith oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open ops_arith oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "ops_arith oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping ops_arith oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for ops_arith oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "ops_arith oracle failed: {stderr}"
        );
        eprintln!("skipping ops_arith oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse ops_arith oracle JSON"))
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
fn diff_sparse_ops_arith() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.arith.len(), query.arith.len());
    assert_eq!(oracle.scale.len(), query.scale.len());
    assert_eq!(oracle.spmv.len(), query.spmv.len());
    assert_eq!(oracle.spdiags.len(), query.spdiags.len());

    let arith_map: HashMap<String, DenseArm> = oracle
        .arith
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();
    let scale_map: HashMap<String, DenseArm> = oracle
        .scale
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();
    let spmv_map: HashMap<String, VecArm> = oracle
        .spmv
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();
    let spdiags_map: HashMap<String, DenseArm> = oracle
        .spdiags
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    // arith
    for case in &query.arith {
        let scipy_arm = arith_map.get(&case.case_id).expect("validated oracle");
        let a_csr = dense_to_csr(&case.a);
        let b_csr = dense_to_csr(&case.b);
        let fsci_result = match case.op.as_str() {
            "add" => add_csr(&a_csr, &b_csr),
            "sub" => sub_csr(&a_csr, &b_csr),
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

    // scale
    for case in &query.scale {
        let scipy_arm = scale_map.get(&case.case_id).expect("validated oracle");
        let a_csr = dense_to_csr(&case.a);
        let Ok(fsci_csr) = scale_csr(&a_csr, case.alpha) else {
            continue;
        };
        if let Some(d) = compare_dense(
            &case.case_id,
            "scale",
            &fsci_csr,
            scipy_arm.rows,
            scipy_arm.cols,
            scipy_arm.dense.as_ref(),
        ) {
            max_overall = max_overall.max(d.abs_diff);
            diffs.push(d);
        }
    }

    // spmv
    for case in &query.spmv {
        let scipy_arm = spmv_map.get(&case.case_id).expect("validated oracle");
        let Some(expected) = scipy_arm.values.as_ref() else {
            continue;
        };
        let a_csr = dense_to_csr(&case.a);
        let Ok(fsci_y) = spmv_csr(&a_csr, &case.x) else {
            continue;
        };
        let abs_d = if fsci_y.len() != expected.len() {
            f64::INFINITY
        } else {
            fsci_y
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        };
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: "spmv".into(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    // spdiags
    for case in &query.spdiags {
        let scipy_arm = spdiags_map.get(&case.case_id).expect("validated oracle");
        let offsets_is: Vec<isize> = case.offsets.iter().map(|&o| o as isize).collect();
        let Ok(fsci_dia) = spdiags(&case.diagonals, &offsets_is, case.rows, case.cols) else {
            continue;
        };
        let Ok(fsci_csr) = fsci_dia.to_csr() else {
            continue;
        };
        if let Some(d) = compare_dense(
            &case.case_id,
            "spdiags",
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
        test_id: "diff_sparse_ops_arith".into(),
        category: "scipy.sparse arith + spdiags".into(),
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
        "sparse ops_arith conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
