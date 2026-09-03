#![forbid(unsafe_code)]
//! Live networkx differential coverage for fsci_sparse graph summaries:
//! average_clustering, is_connected, degree_sequence.
//!
//! Resolves [frankenscipy-zsa85]. 1e-12 abs for the scalar avg
//! clustering; exact match for the boolean / degree vector.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_sparse::{CsrMatrix, Shape2D, average_clustering, degree_sequence, is_connected};
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
    avg_clustering: Option<f64>,
    is_connected: Option<bool>,
    degree_sequence: Option<Vec<i64>>,
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
    fs::create_dir_all(output_dir()).expect("create graph_summary diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize graph_summary diff log");
    fs::write(path, json).expect("write graph_summary diff log");
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
    let adj_6_connected = vec![
        0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
    ];
    let adj_5_two_comp = vec![
        0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    ];
    let adj_4_complete = vec![
        0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0,
    ];

    OracleQuery {
        points: vec![
            PointCase {
                case_id: "6n_connected".into(),
                rows: 6,
                cols: 6,
                adj_flat: adj_6_connected,
            },
            PointCase {
                case_id: "5n_two_comp".into(),
                rows: 5,
                cols: 5,
                adj_flat: adj_5_two_comp,
            },
            PointCase {
                case_id: "4n_complete".into(),
                rows: 4,
                cols: 4,
                adj_flat: adj_4_complete,
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
try:
    import networkx as nx
    HAVE_NX = True
except Exception:
    HAVE_NX = False

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    if not HAVE_NX:
        points.append({"case_id": cid, "avg_clustering": None,
                       "is_connected": None, "degree_sequence": None})
        continue
    r = int(case["rows"]); c = int(case["cols"])
    adj = np.array(case["adj_flat"], dtype=float).reshape(r, c)
    try:
        G = nx.from_numpy_array(adj)
        avg_cc = float(nx.average_clustering(G))
        conn = bool(nx.is_connected(G))
        # Degree sequence in canonical (sorted-by-node) order
        deg = [int(G.degree(i)) for i in range(r)]
        points.append({
            "case_id": cid,
            "avg_clustering": avg_cc if math.isfinite(avg_cc) else None,
            "is_connected": conn,
            "degree_sequence": deg,
        })
    except Exception:
        points.append({"case_id": cid, "avg_clustering": None,
                       "is_connected": None, "degree_sequence": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize graph_summary query");
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
                "failed to spawn python3 for graph_summary oracle: {e}"
            );
            eprintln!("skipping graph_summary oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open graph_summary oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "graph_summary oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping graph_summary oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for graph_summary oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "graph_summary oracle failed: {stderr}"
        );
        eprintln!("skipping graph_summary oracle: networkx not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse graph_summary oracle JSON"))
}

#[test]
fn diff_sparse_graph_summary() {
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
        let csr = dense_to_csr(case.rows, case.cols, &case.adj_flat);

        if let Some(expected) = scipy_arm.avg_clustering {
            let fsci_v = average_clustering(&csr);
            let abs_d = (fsci_v - expected).abs();
            max_overall = max_overall.max(abs_d);
            diffs.push(CaseDiff {
                case_id: format!("{}_avg_clustering", case.case_id),
                op: "avg_clustering".into(),
                abs_diff: abs_d,
                pass: abs_d <= ABS_TOL,
            });
        }

        if let Some(expected) = scipy_arm.is_connected {
            let fsci_v = is_connected(&csr);
            let pass = fsci_v == expected;
            diffs.push(CaseDiff {
                case_id: format!("{}_is_connected", case.case_id),
                op: "is_connected".into(),
                abs_diff: if pass { 0.0 } else { 1.0 },
                pass,
            });
        }

        if let Some(expected) = scipy_arm.degree_sequence.as_ref() {
            let fsci_v = degree_sequence(&csr);
            let abs_d = if fsci_v.len() != expected.len() {
                f64::INFINITY
            } else {
                fsci_v
                    .iter()
                    .zip(expected.iter())
                    .map(|(&a, &b)| ((a as i64) - b).unsigned_abs() as f64)
                    .fold(0.0_f64, f64::max)
            };
            diffs.push(CaseDiff {
                case_id: format!("{}_degree_sequence", case.case_id),
                op: "degree_sequence".into(),
                abs_diff: abs_d,
                pass: abs_d == 0.0,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_sparse_graph_summary".into(),
        category: "fsci_sparse avg_clustering + is_connected + degree_sequence vs networkx".into(),
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
        "graph_summary conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
