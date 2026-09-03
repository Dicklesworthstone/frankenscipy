#![forbid(unsafe_code)]
//! Live scipy.sparse parity for fsci_sparse::{sparse_has_explicit_zeros,
//! sparse_eliminate_zeros}.
//!
//! Resolves [frankenscipy-2upd6]. Builds CSR matrices with explicit
//! zero entries (via raw triplet construction that bypasses
//! deduplication), then verifies:
//!   - has_explicit_zeros matches `(A.data == 0).any()`
//!   - eliminate_zeros matches scipy's `A.eliminate_zeros()` semantics
//!     (same dense reconstruction; nnz drops by the number of zeros)
//!
//! Tolerance: 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_sparse::{
    CooMatrix, FormatConvertible, Shape2D, sparse_eliminate_zeros, sparse_has_explicit_zeros,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    rows: usize,
    cols: usize,
    triplets: Vec<(usize, usize, f64)>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    has_zeros: Option<bool>,
    nnz_after: Option<usize>,
    /// Dense row-major after eliminate_zeros (same as before; just nnz changes).
    dense: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create eliminate_zeros diff dir");
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
    let points = vec![
        // No zeros — has_explicit_zeros=false, nnz unchanged.
        Case {
            case_id: "no_zeros".into(),
            rows: 4,
            cols: 4,
            triplets: vec![
                (0, 0, 1.0),
                (1, 1, 2.0),
                (2, 2, 3.0),
                (3, 3, 4.0),
                (0, 3, 0.5),
            ],
        },
        // Some explicit zeros sprinkled
        Case {
            case_id: "with_zeros".into(),
            rows: 4,
            cols: 4,
            triplets: vec![
                (0, 0, 1.0),
                (0, 1, 0.0), // explicit zero
                (1, 1, 2.0),
                (1, 2, 0.0), // explicit zero
                (2, 2, 3.0),
                (2, 3, 0.5),
                (3, 0, 0.0), // explicit zero
                (3, 3, 4.0),
            ],
        },
        // All zeros
        Case {
            case_id: "all_zeros".into(),
            rows: 3,
            cols: 3,
            triplets: vec![(0, 0, 0.0), (1, 1, 0.0), (2, 2, 0.0)],
        },
        // Diagonal with one zero in the middle
        Case {
            case_id: "diag_mid_zero".into(),
            rows: 5,
            cols: 5,
            triplets: (0..5)
                .map(|i| (i, i, if i == 2 { 0.0 } else { (i + 1) as f64 }))
                .collect(),
        },
    ];
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy.sparse import csr_matrix

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    rows = int(case["rows"]); cols = int(case["cols"])
    rs = [int(t[0]) for t in case["triplets"]]
    cs = [int(t[1]) for t in case["triplets"]]
    vs = [float(t[2]) for t in case["triplets"]]
    try:
        # Build COO then convert (preserves explicit zeros);
        # csr_matrix from COO data with explicit zeros doesn't dedupe by default
        from scipy.sparse import coo_matrix
        A_coo = coo_matrix((vs, (rs, cs)), shape=(rows, cols))
        A = A_coo.tocsr()
        has_zeros = bool((A.data == 0.0).any())
        # scipy eliminate_zeros works in place; do it on a copy
        B = A.copy()
        B.eliminate_zeros()
        nnz_after = int(B.nnz)
        dense_after = B.toarray().flatten().tolist()
        points.append({
            "case_id": cid,
            "has_zeros": has_zeros,
            "nnz_after": nnz_after,
            "dense": [float(v) for v in dense_after],
        })
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "has_zeros": None, "nnz_after": None, "dense": None})
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
                "failed to spawn python3 for eliminate_zeros oracle: {e}"
            );
            eprintln!("skipping eliminate_zeros oracle: python3 not available ({e})");
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
                "eliminate_zeros oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping eliminate_zeros oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for eliminate_zeros oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "eliminate_zeros oracle failed: {stderr}"
        );
        eprintln!("skipping eliminate_zeros oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse eliminate_zeros oracle JSON"))
}

fn csr_to_dense(
    rows: usize,
    cols: usize,
    indptr: &[usize],
    indices: &[usize],
    data: &[f64],
) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for i in 0..rows {
        for idx in indptr[i]..indptr[i + 1] {
            out[i * cols + indices[idx]] = data[idx];
        }
    }
    out
}

#[test]
fn diff_sparse_eliminate_zeros() {
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
        let (Some(ehz), Some(ennz), Some(edense)) =
            (arm.has_zeros, arm.nnz_after, arm.dense.as_ref())
        else {
            continue;
        };

        let mut data = Vec::with_capacity(case.triplets.len());
        let mut rows = Vec::with_capacity(case.triplets.len());
        let mut cols = Vec::with_capacity(case.triplets.len());
        for &(r, c, v) in &case.triplets {
            data.push(v);
            rows.push(r);
            cols.push(c);
        }
        // sum_duplicates=false so explicit zeros aren't collapsed.
        let Ok(coo) =
            CooMatrix::from_triplets(Shape2D::new(case.rows, case.cols), data, rows, cols, true)
        else {
            continue;
        };
        let Ok(csr) = coo.to_csr() else {
            continue;
        };

        // has_explicit_zeros
        let actual_hz = sparse_has_explicit_zeros(&csr);
        let pass = actual_hz == ehz;
        diffs.push(CaseDiff {
            case_id: format!("{}_has_zeros", case.case_id),
            op: "has_zeros".into(),
            abs_diff: if pass { 0.0 } else { 1.0 },
            pass,
        });

        // eliminate_zeros
        let cleaned = sparse_eliminate_zeros(&csr);
        let actual_nnz = cleaned.data().len();
        let nnz_pass = actual_nnz == ennz;
        diffs.push(CaseDiff {
            case_id: format!("{}_nnz_after", case.case_id),
            op: "nnz_after".into(),
            abs_diff: if nnz_pass {
                0.0
            } else {
                (actual_nnz as f64 - ennz as f64).abs()
            },
            pass: nnz_pass,
        });

        let actual_dense = csr_to_dense(
            cleaned.shape().rows,
            cleaned.shape().cols,
            cleaned.indptr(),
            cleaned.indices(),
            cleaned.data(),
        );
        let abs_d = if actual_dense.len() != edense.len() {
            f64::INFINITY
        } else {
            actual_dense
                .iter()
                .zip(edense.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        };
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: format!("{}_dense", case.case_id),
            op: "dense".into(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_sparse_eliminate_zeros".into(),
        category: "fsci_sparse::sparse_has_explicit_zeros + sparse_eliminate_zeros vs scipy.sparse"
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
        "eliminate_zeros conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
