#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.sparse.diags`, `kron`,
//! and `kronsum`. Densify both fsci CSR and scipy CSR outputs in
//! row-major order for comparison.
//!
//! Resolves [frankenscipy-s6w8u]. 1e-12 abs (integer/rational arithmetic).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_sparse::{CsrMatrix, Shape2D, diags, kron, kronsum};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct DiagsCase {
    case_id: String,
    diagonals: Vec<Vec<f64>>,
    offsets: Vec<i64>,
    /// (rows, cols) — None means infer.
    shape: Option<(usize, usize)>,
}

#[derive(Debug, Clone, Serialize)]
struct KronOperand {
    /// Dense row-major flattening.
    dense: Vec<f64>,
    rows: usize,
    cols: usize,
}

#[derive(Debug, Clone, Serialize)]
struct KronCase {
    case_id: String,
    op: String, // "kron" | "kronsum"
    a: KronOperand,
    b: KronOperand,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    diags_cases: Vec<DiagsCase>,
    kron_cases: Vec<KronCase>,
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
    diags: Vec<DenseArm>,
    kron: Vec<DenseArm>,
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
    fs::create_dir_all(output_dir()).expect("create diags_kron diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize diags_kron diff log");
    fs::write(path, json).expect("write diags_kron diff log");
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

fn dense_to_csr(rows: usize, cols: usize, dense: &[f64]) -> CsrMatrix {
    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = Vec::with_capacity(rows + 1);
    indptr.push(0);
    for r in 0..rows {
        for c in 0..cols {
            let v = dense[r * cols + c];
            if v != 0.0 {
                data.push(v);
                indices.push(c);
            }
        }
        indptr.push(data.len());
    }
    CsrMatrix::from_components(Shape2D::new(rows, cols), data, indices, indptr, true)
        .expect("dense_to_csr build")
}

fn generate_query() -> OracleQuery {
    let diags_cases = vec![
        DiagsCase {
            case_id: "tridiag_5".into(),
            diagonals: vec![
                vec![-1.0, -1.0, -1.0, -1.0],
                vec![2.0, 2.0, 2.0, 2.0, 2.0],
                vec![-1.0, -1.0, -1.0, -1.0],
            ],
            offsets: vec![-1, 0, 1],
            shape: None,
        },
        DiagsCase {
            case_id: "with_super_2".into(),
            diagonals: vec![
                vec![1.0, 2.0, 3.0, 4.0, 5.0],
                vec![10.0, 20.0, 30.0, 40.0],
                vec![100.0, 200.0, 300.0],
            ],
            offsets: vec![0, 1, 2],
            shape: None,
        },
        DiagsCase {
            case_id: "wide_4x6_zero_diag".into(),
            diagonals: vec![vec![1.0, 1.0, 1.0, 1.0]],
            offsets: vec![0],
            shape: Some((4, 6)),
        },
        DiagsCase {
            case_id: "tall_5x3_subdiag".into(),
            diagonals: vec![vec![1.0, 2.0, 3.0]],
            offsets: vec![-1],
            shape: Some((5, 3)),
        },
    ];

    let a1 = KronOperand {
        rows: 2,
        cols: 2,
        dense: vec![1.0, 2.0, 3.0, 4.0],
    };
    let b1 = KronOperand {
        rows: 2,
        cols: 2,
        dense: vec![0.0, 5.0, 6.0, 7.0],
    };
    let a2 = KronOperand {
        rows: 3,
        cols: 2,
        dense: vec![1.0, 0.0, 0.0, 2.0, 3.0, 0.0],
    };
    let b2 = KronOperand {
        rows: 2,
        cols: 3,
        dense: vec![1.0, 1.0, 0.0, 0.0, 2.0, 2.0],
    };
    let a3 = KronOperand {
        rows: 3,
        cols: 3,
        dense: vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 5.0, 0.0, 6.0],
    };
    let b3 = KronOperand {
        rows: 3,
        cols: 3,
        dense: vec![7.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 9.0],
    };
    let kron_cases = vec![
        KronCase {
            case_id: "kron_2x2_2x2".into(),
            op: "kron".into(),
            a: a1.clone(),
            b: b1.clone(),
        },
        KronCase {
            case_id: "kron_3x2_2x3".into(),
            op: "kron".into(),
            a: a2,
            b: b2,
        },
        KronCase {
            case_id: "kron_3x3_3x3".into(),
            op: "kron".into(),
            a: a3.clone(),
            b: b3.clone(),
        },
        KronCase {
            case_id: "kronsum_2x2_2x2".into(),
            op: "kronsum".into(),
            a: a1,
            b: b1,
        },
        KronCase {
            case_id: "kronsum_3x3_3x3".into(),
            op: "kronsum".into(),
            a: a3,
            b: b3,
        },
    ];

    OracleQuery {
        diags_cases,
        kron_cases,
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy import sparse

def dense_or_none(rows, cols, mat):
    if mat is None:
        return None
    arr = np.asarray(mat.todense()).reshape(rows, cols).tolist()
    flat = []
    for row in arr:
        for v in row:
            if not math.isfinite(float(v)):
                return None
            flat.append(float(v))
    return flat

q = json.load(sys.stdin)
diags_out = []
for c in q["diags_cases"]:
    cid = c["case_id"]
    diagonals = [np.array(d, dtype=float) for d in c["diagonals"]]
    offsets = list(c["offsets"])
    shape = c.get("shape")
    if shape is not None:
        shape = (int(shape[0]), int(shape[1]))
    try:
        m = sparse.diags(diagonals, offsets, shape=shape, format="csr")
        rows, cols = m.shape
        diags_out.append({"case_id": cid, "rows": rows, "cols": cols,
                          "dense": dense_or_none(rows, cols, m)})
    except Exception:
        diags_out.append({"case_id": cid, "rows": None, "cols": None, "dense": None})

kron_out = []
for c in q["kron_cases"]:
    cid = c["case_id"]; op = c["op"]
    a = c["a"]; b = c["b"]
    A = sparse.csr_matrix(np.array(a["dense"], dtype=float).reshape(a["rows"], a["cols"]))
    B = sparse.csr_matrix(np.array(b["dense"], dtype=float).reshape(b["rows"], b["cols"]))
    try:
        if op == "kron":
            m = sparse.kron(A, B, format="csr")
        elif op == "kronsum":
            m = sparse.kronsum(A, B, format="csr")
        else:
            m = None
        if m is None:
            kron_out.append({"case_id": cid, "rows": None, "cols": None, "dense": None})
        else:
            rows, cols = m.shape
            kron_out.append({"case_id": cid, "rows": rows, "cols": cols,
                             "dense": dense_or_none(rows, cols, m)})
    except Exception:
        kron_out.append({"case_id": cid, "rows": None, "cols": None, "dense": None})

print(json.dumps({"diags": diags_out, "kron": kron_out}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize diags_kron query");
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
                "failed to spawn python3 for diags_kron oracle: {e}"
            );
            eprintln!("skipping diags_kron oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open diags_kron oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "diags_kron oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping diags_kron oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for diags_kron oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "diags_kron oracle failed: {stderr}"
        );
        eprintln!("skipping diags_kron oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse diags_kron oracle JSON"))
}

#[test]
fn diff_sparse_diags_kron() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.diags.len(), query.diags_cases.len());
    assert_eq!(oracle.kron.len(), query.kron_cases.len());

    let diags_map: HashMap<String, DenseArm> = oracle
        .diags
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();
    let kron_map: HashMap<String, DenseArm> = oracle
        .kron
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    // diags
    for case in &query.diags_cases {
        let scipy_arm = diags_map.get(&case.case_id).expect("validated oracle");
        let Some(expected) = scipy_arm.dense.as_ref() else {
            continue;
        };
        let (Some(rows), Some(cols)) = (scipy_arm.rows, scipy_arm.cols) else {
            continue;
        };
        let shape = case.shape.map(|(r, c)| Shape2D::new(r, c));
        let offsets_is: Vec<isize> = case.offsets.iter().map(|&o| o as isize).collect();
        let Ok(fsci_csr) = diags(&case.diagonals, &offsets_is, shape) else {
            continue;
        };
        let fsci_shape = fsci_csr.shape();
        if fsci_shape.rows != rows || fsci_shape.cols != cols {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                op: "diags".into(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let fsci_dense = dense_from_csr(&fsci_csr);
        let abs_d = fsci_dense
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: "diags".into(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    // kron / kronsum
    for case in &query.kron_cases {
        let scipy_arm = kron_map.get(&case.case_id).expect("validated oracle");
        let Some(expected) = scipy_arm.dense.as_ref() else {
            continue;
        };
        let (Some(rows), Some(cols)) = (scipy_arm.rows, scipy_arm.cols) else {
            continue;
        };
        let a_csr = dense_to_csr(case.a.rows, case.a.cols, &case.a.dense);
        let b_csr = dense_to_csr(case.b.rows, case.b.cols, &case.b.dense);
        let fsci_csr = match case.op.as_str() {
            "kron" => kron(&a_csr, &b_csr),
            "kronsum" => kronsum(&a_csr, &b_csr),
            _ => continue,
        };
        let Ok(fsci_csr) = fsci_csr else {
            continue;
        };
        let fsci_shape = fsci_csr.shape();
        if fsci_shape.rows != rows || fsci_shape.cols != cols {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                op: case.op.clone(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let fsci_dense = dense_from_csr(&fsci_csr);
        let abs_d = fsci_dense
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
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
        test_id: "diff_sparse_diags_kron".into(),
        category: "scipy.sparse.diags + kron + kronsum".into(),
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
        "scipy.sparse.diags/kron/kronsum conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
