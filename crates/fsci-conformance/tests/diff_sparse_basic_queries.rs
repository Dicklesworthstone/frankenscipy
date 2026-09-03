#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the basic sparse-matrix query
//! primitives that diff_sparse.rs doesn't exercise:
//!   - `fsci_sparse::sparse_norm(A, "fro" | "1" | "inf")` vs
//!     `scipy.sparse.linalg.norm(A, ord='fro' | 1 | numpy.inf)`
//!   - `fsci_sparse::sparse_diagonal(A)`  vs `A.diagonal()`
//!   - `fsci_sparse::sparse_trace(A)`     vs `A.diagonal().sum()`
//!
//! Resolves [frankenscipy-xa3b9]. All three primitives are closed-form
//! over the sparse structure, so machine-precision agreement (1e-12)
//! is appropriate.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_sparse::{
    CooMatrix, FormatConvertible, Shape2D, sparse_diagonal, sparse_norm, sparse_trace,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-004";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    rows: usize,
    cols: usize,
    /// COO triplets (row, col, value).
    triplets: Vec<(usize, usize, f64)>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    norm_fro: Option<f64>,
    norm_1: Option<f64>,
    norm_inf: Option<f64>,
    diagonal: Option<Vec<f64>>,
    trace: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    arm: String,
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
    fs::create_dir_all(output_dir()).expect("create sparse_basic diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize sparse_basic diff log");
    fs::write(path, json).expect("write sparse_basic diff log");
}

fn generate_query() -> OracleQuery {
    let cases: Vec<(&str, usize, usize, Vec<(usize, usize, f64)>)> = vec![
        (
            "3x3_diagonal",
            3,
            3,
            vec![(0, 0, 1.0), (1, 1, -2.0), (2, 2, 3.5)],
        ),
        (
            "3x3_dense",
            3,
            3,
            vec![
                (0, 0, 4.0),
                (0, 1, -1.0),
                (0, 2, 0.5),
                (1, 0, -1.0),
                (1, 1, 4.0),
                (1, 2, -1.0),
                (2, 0, 0.5),
                (2, 1, -1.0),
                (2, 2, 4.0),
            ],
        ),
        (
            "4x4_sparse",
            4,
            4,
            vec![
                (0, 0, 1.0),
                (0, 3, 2.0),
                (1, 1, -1.5),
                (2, 0, -0.5),
                (2, 2, 7.0),
                (3, 3, -3.0),
            ],
        ),
        (
            "5x3_rectangular_tall",
            5,
            3,
            vec![
                (0, 0, 1.0),
                (0, 2, 2.0),
                (1, 1, -2.0),
                (2, 0, 3.0),
                (3, 2, -1.5),
                (4, 1, 0.25),
            ],
        ),
        (
            "2x5_rectangular_wide",
            2,
            5,
            vec![
                (0, 0, 1.0),
                (0, 1, -1.0),
                (0, 4, 0.5),
                (1, 2, 2.0),
                (1, 3, -2.0),
            ],
        ),
        ("5x5_empty", 5, 5, vec![]),
        (
            "6x6_negative_mixed",
            6,
            6,
            vec![
                (0, 0, -3.0),
                (1, 4, 2.5),
                (2, 1, -1.0),
                (3, 3, 4.0),
                (4, 2, 5.5),
                (5, 5, -6.0),
            ],
        ),
        ("1x1_singleton", 1, 1, vec![(0, 0, 9.0)]),
    ];
    let points = cases
        .into_iter()
        .map(|(name, rows, cols, triplets)| PointCase {
            case_id: name.into(),
            rows,
            cols,
            triplets,
        })
        .collect();
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

def vec_or_none(arr):
    out = []
    for v in np.asarray(arr).tolist():
        try:
            v = float(v)
        except Exception:
            return None
        if not math.isfinite(v):
            return None
        out.append(v)
    return out

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    rows = case["rows"]; cols = case["cols"]
    triplets = case["triplets"]
    if triplets:
        r = np.array([t[0] for t in triplets], dtype=int)
        c = np.array([t[1] for t in triplets], dtype=int)
        v = np.array([t[2] for t in triplets], dtype=float)
    else:
        r = np.zeros(0, dtype=int); c = np.zeros(0, dtype=int); v = np.zeros(0, dtype=float)
    try:
        A = sp.csr_matrix((v, (r, c)), shape=(rows, cols))
        nfro = fnone(spl.norm(A, ord='fro'))
        n1   = fnone(spl.norm(A, ord=1))
        ninf = fnone(spl.norm(A, ord=np.inf))
        diag = vec_or_none(A.diagonal())
        tr   = fnone(A.diagonal().sum())
        points.append({
            "case_id": cid,
            "norm_fro": nfro,
            "norm_1": n1,
            "norm_inf": ninf,
            "diagonal": diag,
            "trace": tr,
        })
    except Exception:
        points.append({
            "case_id": cid,
            "norm_fro": None, "norm_1": None, "norm_inf": None,
            "diagonal": None, "trace": None,
        })
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize sparse_basic query");
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
                "failed to spawn python3 for sparse_basic oracle: {e}"
            );
            eprintln!("skipping sparse_basic oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open sparse_basic oracle stdin");
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
fn diff_sparse_basic_queries() {
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
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        // Build fsci csr from triplets.
        let r: Vec<usize> = case.triplets.iter().map(|t| t.0).collect();
        let c: Vec<usize> = case.triplets.iter().map(|t| t.1).collect();
        let d: Vec<f64> = case.triplets.iter().map(|t| t.2).collect();
        let Ok(coo) = CooMatrix::from_triplets(Shape2D::new(case.rows, case.cols), d, r, c, false)
        else {
            continue;
        };
        let Ok(csr) = coo.to_csr() else { continue };

        // Norm comparisons only make sense on square matrices for fsci's
        // sparse_norm impl (it iterates by rows assuming shape().rows is the
        // canonical dimension); test on all but compare per-arm where
        // scipy has a value. fsci's impl computes "fro" purely from data
        // so it's well-defined for any shape; same for "1" (col_sums) and
        // "inf" (row_sums). We probe all three arms uniformly.
        if let Some(nfro) = scipy_arm.norm_fro {
            let f = sparse_norm(&csr, "fro").expect("ord fro");
            let abs_d = (f - nfro).abs();
            max_overall = max_overall.max(abs_d);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "norm_fro".into(),
                abs_diff: abs_d,
                pass: abs_d <= ABS_TOL,
            });
        }
        // sparse_norm '1' and 'inf' assume square matrix in fsci impl
        // (m bound on col_sums comes from shape().cols, ok for rect, but
        // skip if scipy returned None).
        if let Some(n1) = scipy_arm.norm_1 {
            let f = sparse_norm(&csr, "1").expect("ord 1");
            let abs_d = (f - n1).abs();
            max_overall = max_overall.max(abs_d);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "norm_1".into(),
                abs_diff: abs_d,
                pass: abs_d <= ABS_TOL,
            });
        }
        if let Some(ninf) = scipy_arm.norm_inf {
            let f = sparse_norm(&csr, "inf").expect("ord inf");
            let abs_d = (f - ninf).abs();
            max_overall = max_overall.max(abs_d);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "norm_inf".into(),
                abs_diff: abs_d,
                pass: abs_d <= ABS_TOL,
            });
        }
        if let Some(diag) = scipy_arm.diagonal.as_ref() {
            let f = sparse_diagonal(&csr);
            if f.len() == diag.len() {
                let abs_d = f
                    .iter()
                    .zip(diag.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                max_overall = max_overall.max(abs_d);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    arm: "diagonal".into(),
                    abs_diff: abs_d,
                    pass: abs_d <= ABS_TOL,
                });
            } else {
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    arm: "diagonal".into(),
                    abs_diff: f64::INFINITY,
                    pass: false,
                });
            }
        }
        if let Some(tr) = scipy_arm.trace {
            let f = sparse_trace(&csr);
            let abs_d = (f - tr).abs();
            max_overall = max_overall.max(abs_d);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "trace".into(),
                abs_diff: abs_d,
                pass: abs_d <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_sparse_basic_queries".into(),
        category: "scipy.sparse.linalg.norm + A.diagonal/trace".into(),
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
            eprintln!(
                "sparse_basic {} mismatch: {} abs_diff={}",
                d.arm, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.sparse basic-query conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
