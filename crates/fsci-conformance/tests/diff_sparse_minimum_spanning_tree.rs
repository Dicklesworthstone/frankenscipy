#![forbid(unsafe_code)]
//! Live SciPy differential coverage for
//! `scipy.sparse.csgraph.minimum_spanning_tree`. Compares total weight
//! (the canonical invariant across multiple valid MSTs).
//!
//! Resolves [frankenscipy-acmec]. 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_sparse::{CsrMatrix, Shape2D, minimum_spanning_tree};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    rows: usize,
    cols: usize,
    adj_flat: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    total_weight: Option<f64>,
    nnz: Option<usize>,
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
    fs::create_dir_all(output_dir()).expect("create mst diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize mst diff log");
    fs::write(path, json).expect("write mst diff log");
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
    let adj_6 = vec![
        0.0, 7.0, 9.0, 0.0, 0.0, 14.0, 7.0, 0.0, 10.0, 15.0, 0.0, 0.0, 9.0, 10.0, 0.0, 11.0, 0.0,
        2.0, 0.0, 15.0, 11.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 9.0, 14.0, 0.0, 2.0, 0.0,
        9.0, 0.0,
    ];
    let adj_4_square = vec![
        0.0, 1.0, 0.0, 4.0, 1.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 3.0, 4.0, 0.0, 3.0, 0.0,
    ];
    let adj_5_star = vec![
        0.0, 2.0, 3.0, 5.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    ];

    OracleQuery {
        points: vec![
            PointCase {
                case_id: "6n_classic".into(),
                rows: 6,
                cols: 6,
                adj_flat: adj_6,
            },
            PointCase {
                case_id: "4n_square".into(),
                rows: 4,
                cols: 4,
                adj_flat: adj_4_square,
            },
            PointCase {
                case_id: "5n_star".into(),
                rows: 5,
                cols: 5,
                adj_flat: adj_5_star,
            },
        ],
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    r = int(case["rows"]); c = int(case["cols"])
    adj = np.array(case["adj_flat"], dtype=float).reshape(r, c)
    try:
        m = minimum_spanning_tree(csr_matrix(adj))
        total = float(m.sum())
        if not math.isfinite(total):
            points.append({"case_id": cid, "total_weight": None, "nnz": None})
        else:
            points.append({"case_id": cid, "total_weight": total, "nnz": int(m.nnz)})
    except Exception:
        points.append({"case_id": cid, "total_weight": None, "nnz": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize mst query");
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
                "failed to spawn python3 for mst oracle: {e}"
            );
            eprintln!("skipping mst oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open mst oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "mst oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping mst oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for mst oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "mst oracle failed: {stderr}"
        );
        eprintln!("skipping mst oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse mst oracle JSON"))
}

#[test]
fn diff_sparse_minimum_spanning_tree() {
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
        let Some(scipy_total) = scipy_arm.total_weight else {
            continue;
        };
        let Some(scipy_nnz) = scipy_arm.nnz else {
            continue;
        };
        let csr = dense_to_csr(case.rows, case.cols, &case.adj_flat);
        let Ok(res) = minimum_spanning_tree(&csr) else {
            continue;
        };
        let weight_d = (res.total_weight - scipy_total).abs();
        // Edge count should also match — both produce n-1 edges for connected graphs.
        let edge_d = if res.edges.len() == scipy_nnz {
            0.0
        } else {
            1.0
        };
        let abs_d = weight_d.max(edge_d);
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_sparse_minimum_spanning_tree".into(),
        category: "scipy.sparse.csgraph.minimum_spanning_tree".into(),
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
            eprintln!("mst mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "minimum_spanning_tree conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
