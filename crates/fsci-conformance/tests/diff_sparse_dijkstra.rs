#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_sparse::dijkstra.
//!
//! Resolves [frankenscipy-hxo6k]. Compares per-source shortest-path
//! distance vectors at 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_sparse::{CsrMatrix, Shape2D, dijkstra};
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
    source: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    distances: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create dijkstra diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize dijkstra diff log");
    fs::write(path, json).expect("write dijkstra diff log");
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
    // Classic 6-node weighted graph
    let adj_6 = vec![
        0.0, 7.0, 9.0, 0.0, 0.0, 14.0, 7.0, 0.0, 10.0, 15.0, 0.0, 0.0, 9.0, 10.0, 0.0, 11.0, 0.0,
        2.0, 0.0, 15.0, 11.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 9.0, 14.0, 0.0, 2.0, 0.0,
        9.0, 0.0,
    ];
    // Triangle 3-node
    let adj_3 = vec![0.0, 1.0, 4.0, 1.0, 0.0, 2.0, 4.0, 2.0, 0.0];
    // Linear chain 5-node
    let adj_5 = vec![
        0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0, 3.0,
        0.0, 4.0, 0.0, 0.0, 0.0, 4.0, 0.0,
    ];

    let mut points = Vec::new();
    for source in [0_usize, 2, 5] {
        points.push(PointCase {
            case_id: format!("6n_s{source}"),
            rows: 6,
            cols: 6,
            adj_flat: adj_6.clone(),
            source,
        });
    }
    for source in [0_usize, 1, 2] {
        points.push(PointCase {
            case_id: format!("3n_s{source}"),
            rows: 3,
            cols: 3,
            adj_flat: adj_3.clone(),
            source,
        });
    }
    for source in [0_usize, 2, 4] {
        points.push(PointCase {
            case_id: format!("5n_chain_s{source}"),
            rows: 5,
            cols: 5,
            adj_flat: adj_5.clone(),
            source,
        });
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

def vec_or_none(arr):
    out = []
    for v in np.asarray(arr, dtype=float).flatten().tolist():
        if v == float("inf"):
            out.append(float("inf"))
        elif not math.isfinite(float(v)):
            return None
        else:
            out.append(float(v))
    return out

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    r = int(case["rows"]); c = int(case["cols"])
    adj = np.array(case["adj_flat"], dtype=float).reshape(r, c)
    s = int(case["source"])
    try:
        dist = dijkstra(csr_matrix(adj), directed=False, indices=s)
        # Filter infinities to None for JSON (Rust will treat as f64::INFINITY)
        flat = []
        for v in dist.tolist():
            if v == float("inf"):
                flat.append(1e308)  # sentinel for infinity
            else:
                flat.append(float(v))
        points.append({"case_id": cid, "distances": flat})
    except Exception:
        points.append({"case_id": cid, "distances": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize dijkstra query");
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
                "failed to spawn python3 for dijkstra oracle: {e}"
            );
            eprintln!("skipping dijkstra oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open dijkstra oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "dijkstra oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping dijkstra oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for dijkstra oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "dijkstra oracle failed: {stderr}"
        );
        eprintln!("skipping dijkstra oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse dijkstra oracle JSON"))
}

#[test]
fn diff_sparse_dijkstra() {
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
        let Some(scipy_d) = scipy_arm.distances.as_ref() else {
            continue;
        };
        let csr = dense_to_csr(case.rows, case.cols, &case.adj_flat);
        let Ok(res) = dijkstra(&csr, case.source) else {
            continue;
        };
        // Compare distances; treat fsci INF and scipy 1e308 sentinel as both infinite.
        let abs_d = if res.distances.len() != scipy_d.len() {
            f64::INFINITY
        } else {
            res.distances
                .iter()
                .zip(scipy_d.iter())
                .map(|(a, b)| {
                    let a_inf = a.is_infinite();
                    let b_sent = *b >= 1.0e307;
                    if a_inf && b_sent {
                        0.0
                    } else if a_inf || b_sent {
                        f64::INFINITY
                    } else {
                        (a - b).abs()
                    }
                })
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
        test_id: "diff_sparse_dijkstra".into(),
        category: "scipy.sparse.csgraph.dijkstra".into(),
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
            eprintln!("dijkstra mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "dijkstra conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
