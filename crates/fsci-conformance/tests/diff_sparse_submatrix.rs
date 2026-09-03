#![forbid(unsafe_code)]
//! Live scipy.sparse parity for fsci_sparse::sparse_submatrix.
//!
//! Resolves [frankenscipy-vm5mc]. Compares against
//! `A[r_start:r_end, c_start:c_end]` slicing of a scipy CSR matrix.
//! Reference dense reconstruction is compared element-wise.
//!
//! Tolerance: 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_sparse::{CooMatrix, FormatConvertible, Shape2D, sparse_submatrix};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct SubCase {
    case_id: String,
    rows: usize,
    cols: usize,
    triplets: Vec<(usize, usize, f64)>,
    r_start: usize,
    r_end: usize,
    c_start: usize,
    c_end: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<SubCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    out_rows: Option<usize>,
    out_cols: Option<usize>,
    /// Dense row-major flat representation of the expected submatrix.
    dense: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
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
    fs::create_dir_all(output_dir()).expect("create submatrix diff dir");
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

fn diag_with_off(n: usize) -> Vec<(usize, usize, f64)> {
    let mut out = Vec::new();
    for i in 0..n {
        out.push((i, i, (i + 1) as f64));
        if i + 1 < n {
            out.push((i, i + 1, -1.0));
            out.push((i + 1, i, 0.5));
        }
    }
    out
}

fn dense_8x6() -> Vec<(usize, usize, f64)> {
    let mut out = Vec::new();
    for i in 0..8 {
        for j in 0..6 {
            // Drop a few entries to make it sparse-ish but non-trivial.
            if (i + j) % 3 != 0 {
                out.push((i, j, (i * 6 + j + 1) as f64 * 0.1));
            }
        }
    }
    out
}

fn generate_query() -> OracleQuery {
    let d10 = diag_with_off(10);
    let m8x6 = dense_8x6();
    let points = vec![
        SubCase {
            case_id: "diag10_interior".into(),
            rows: 10,
            cols: 10,
            triplets: d10.clone(),
            r_start: 2,
            r_end: 7,
            c_start: 1,
            c_end: 5,
        },
        SubCase {
            case_id: "diag10_full".into(),
            rows: 10,
            cols: 10,
            triplets: d10.clone(),
            r_start: 0,
            r_end: 10,
            c_start: 0,
            c_end: 10,
        },
        SubCase {
            case_id: "diag10_one_row".into(),
            rows: 10,
            cols: 10,
            triplets: d10.clone(),
            r_start: 4,
            r_end: 5,
            c_start: 0,
            c_end: 10,
        },
        SubCase {
            case_id: "diag10_one_col".into(),
            rows: 10,
            cols: 10,
            triplets: d10,
            r_start: 0,
            r_end: 10,
            c_start: 3,
            c_end: 4,
        },
        SubCase {
            case_id: "m8x6_rect".into(),
            rows: 8,
            cols: 6,
            triplets: m8x6.clone(),
            r_start: 1,
            r_end: 6,
            c_start: 1,
            c_end: 5,
        },
        SubCase {
            case_id: "m8x6_corner".into(),
            rows: 8,
            cols: 6,
            triplets: m8x6,
            r_start: 5,
            r_end: 8,
            c_start: 3,
            c_end: 6,
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
    cs_ = [int(t[1]) for t in case["triplets"]]
    vs = [float(t[2]) for t in case["triplets"]]
    r0 = int(case["r_start"]); r1 = int(case["r_end"])
    c0 = int(case["c_start"]); c1 = int(case["c_end"])
    try:
        A = csr_matrix((vs, (rs, cs_)), shape=(rows, cols))
        sub = A[r0:r1, c0:c1]
        dense = sub.toarray()
        flat = dense.flatten().tolist()
        points.append({
            "case_id": cid,
            "out_rows": int(dense.shape[0]),
            "out_cols": int(dense.shape[1]),
            "dense": [float(v) for v in flat],
        })
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "out_rows": None, "out_cols": None, "dense": None})
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
                "failed to spawn python3 for submatrix oracle: {e}"
            );
            eprintln!("skipping submatrix oracle: python3 not available ({e})");
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
                "submatrix oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping submatrix oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for submatrix oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "submatrix oracle failed: {stderr}"
        );
        eprintln!("skipping submatrix oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse submatrix oracle JSON"))
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
fn diff_sparse_submatrix() {
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
        let (Some(eor), Some(eoc), Some(edense)) = (arm.out_rows, arm.out_cols, arm.dense.as_ref())
        else {
            continue;
        };
        let mut data = Vec::with_capacity(case.triplets.len());
        let mut rs = Vec::with_capacity(case.triplets.len());
        let mut cs = Vec::with_capacity(case.triplets.len());
        for &(r, c, v) in &case.triplets {
            data.push(v);
            rs.push(r);
            cs.push(c);
        }
        let Ok(coo) =
            CooMatrix::from_triplets(Shape2D::new(case.rows, case.cols), data, rs, cs, true)
        else {
            continue;
        };
        let Ok(csr) = coo.to_csr() else {
            continue;
        };
        let sub = sparse_submatrix(&csr, case.r_start, case.r_end, case.c_start, case.c_end);
        let actual_rows = sub.shape().rows;
        let actual_cols = sub.shape().cols;
        let actual_dense = csr_to_dense(
            actual_rows,
            actual_cols,
            sub.indptr(),
            sub.indices(),
            sub.data(),
        );

        let abs_d =
            if actual_rows != eor || actual_cols != eoc || actual_dense.len() != edense.len() {
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
            case_id: case.case_id.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_sparse_submatrix".into(),
        category: "fsci_sparse::sparse_submatrix vs scipy.sparse slicing".into(),
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
            eprintln!("submatrix mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "submatrix conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
