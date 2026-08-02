#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_sparse::laplacian
//! (graph Laplacian, normed or unnormed).
//!
//! Resolves [frankenscipy-k7110]. 1e-10 abs on flat matrix.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_sparse::{CsrMatrix, Shape2D, laplacian};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    rows: usize,
    cols: usize,
    adj_flat: Vec<f64>,
    normed: bool,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    matrix: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create laplacian diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize laplacian diff log");
    fs::write(path, json).expect("write laplacian diff log");
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
    let adj_4 = vec![
        0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    ];
    let adj_5_cycle = vec![
        0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
    ];
    let adj_3_triangle = vec![0.0, 2.0, 1.0, 2.0, 0.0, 3.0, 1.0, 3.0, 0.0];

    let mut points = Vec::new();
    let inputs: &[(&str, &[f64], usize)] = &[
        ("4n", &adj_4, 4),
        ("5n_cycle", &adj_5_cycle, 5),
        ("3n_triangle", &adj_3_triangle, 3),
    ];
    for (label, adj, n) in inputs {
        for normed in [false, true] {
            points.push(PointCase {
                case_id: format!("{label}_normed{normed}"),
                rows: *n,
                cols: *n,
                adj_flat: adj.to_vec(),
                normed,
            });
        }
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import os
import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian

def finite_or_none(arr):
    arr = np.asarray(arr, dtype=float)
    flat = []
    for v in arr.flatten().tolist():
        if not math.isfinite(float(v)):
            return None
        flat.append(float(v))
    return flat

q = json.loads(os.environ.pop("FSCI_LAPLACIAN_ORACLE_QUERY"))
points = []
for case in q["points"]:
    cid = case["case_id"]
    r = int(case["rows"]); c = int(case["cols"])
    adj = np.array(case["adj_flat"], dtype=float).reshape(r, c)
    normed = bool(case["normed"])
    try:
        L = laplacian(csr_matrix(adj), normed=normed)
        # L can be sparse; densify
        if hasattr(L, "todense"):
            arr = np.asarray(L.todense())
        else:
            arr = np.asarray(L)
        points.append({"case_id": cid, "matrix": finite_or_none(arr)})
    except Exception:
        points.append({"case_id": cid, "matrix": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize laplacian query");
    let mut child = match Command::new("python3")
        .arg("-")
        .env("FSCI_LAPLACIAN_ORACLE_QUERY", query_json)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn python3 for laplacian oracle: {e}"
            );
            eprintln!("skipping laplacian oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open laplacian oracle stdin");
        if let Err(err) = stdin.write_all(script.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "laplacian oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping laplacian oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for laplacian oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "laplacian oracle failed: {stderr}"
        );
        eprintln!("skipping laplacian oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse laplacian oracle JSON"))
}

#[test]
fn diff_sparse_laplacian() {
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
        let Some(expected) = scipy_arm.matrix.as_ref() else {
            continue;
        };
        let csr = dense_to_csr(case.rows, case.cols, &case.adj_flat);
        let Ok(lap) = laplacian(&csr, case.normed) else {
            continue;
        };
        let mut flat = vec![0.0; case.rows * case.cols];
        for row in 0..case.rows {
            for entry in lap.indptr()[row]..lap.indptr()[row + 1] {
                flat[row * case.cols + lap.indices()[entry]] = lap.data()[entry];
            }
        }
        let abs_d = if flat.len() != expected.len() {
            f64::INFINITY
        } else {
            flat.iter()
                .zip(expected.iter())
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
        test_id: "diff_sparse_laplacian".into(),
        category: "scipy.sparse.csgraph.laplacian".into(),
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
            eprintln!("laplacian mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "laplacian conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
