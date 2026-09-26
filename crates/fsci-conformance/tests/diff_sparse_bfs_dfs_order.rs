#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_sparse::breadth_first_order
//! and depth_first_order with SciPy's default `directed=True`.
//!
//! Resolves [frankenscipy-wy6ri]. Both implementations enumerate neighbours
//! in CSR storage order, so the visit ORDER and the PREDECESSOR tree are
//! compared exactly (SciPy's -9999 "no predecessor" maps to fsci's -1).
//!
//! br-szq1n.3: this file used to compare visit SETS only, against SciPy's
//! `directed=False`, which could not see that the old depth-first search
//! marked nodes visited at push time and so built the wrong predecessor tree.
//! The asymmetric cases below are the ones where that showed.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_sparse::{CsrMatrix, Shape2D, breadth_first_order, depth_first_order};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String,
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
    /// Visit order.
    order: Option<Vec<i64>>,
    /// Predecessor of each node, -1 for the source and unreached nodes.
    predecessors: Option<Vec<i64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    op: String,
    pass: bool,
    note: String,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create bfs_dfs diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize bfs_dfs diff log");
    fs::write(path, json).expect("write bfs_dfs diff log");
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
        0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0,
        0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    ];
    let adj_5_chain = vec![
        0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    ];

    let mut points = Vec::new();
    for source in [0_usize, 2, 5] {
        for op in ["bfs", "dfs"] {
            points.push(PointCase {
                case_id: format!("{op}_6n_s{source}"),
                op: op.into(),
                rows: 6,
                cols: 6,
                adj_flat: adj_6.clone(),
                source,
            });
        }
    }
    for source in [0_usize, 4] {
        for op in ["bfs", "dfs"] {
            points.push(PointCase {
                case_id: format!("{op}_5n_chain_s{source}"),
                op: op.into(),
                rows: 5,
                cols: 5,
                adj_flat: adj_5_chain.clone(),
                source,
            });
        }
    }
    // Directed, asymmetric graphs. 0->1, 0->2, 1->2: a depth-first search reaches
    // 2 through 1 (SciPy predecessor 1; push-time marking recorded 0).
    let adj_3_dag = vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    // 0->1, 0->2, 1->3, 3->2: SciPy DFS order [0,1,3,2], predecessor of 2 is 3.
    let adj_4_dag = vec![
        0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    ];
    // A directed cycle with a chord and an unreachable node (5 has no in-edges).
    let adj_6_directed = vec![
        0.0, 1.0, 0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, //
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, //
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, //
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let directed: &[(&str, &[f64], usize, &[usize])] = &[
        ("3n_dag", &adj_3_dag, 3, &[0]),
        ("4n_dag", &adj_4_dag, 4, &[0, 1]),
        ("6n_directed", &adj_6_directed, 6, &[0, 3, 5]),
    ];
    for (label, adj, n, sources) in directed {
        for &source in *sources {
            for op in ["bfs", "dfs"] {
                points.push(PointCase {
                    case_id: format!("{op}_{label}_s{source}"),
                    op: op.into(),
                    rows: *n,
                    cols: *n,
                    adj_flat: adj.to_vec(),
                    source,
                });
            }
        }
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, depth_first_order

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    r = int(case["rows"]); c = int(case["cols"])
    adj = np.array(case["adj_flat"], dtype=float).reshape(r, c)
    s = int(case["source"])
    try:
        if op == "bfs":
            order, pred = breadth_first_order(csr_matrix(adj), s, directed=True,
                                              return_predecessors=True)
        elif op == "dfs":
            order, pred = depth_first_order(csr_matrix(adj), s, directed=True,
                                            return_predecessors=True)
        else:
            order = None
        if order is None:
            points.append({"case_id": cid, "order": None, "predecessors": None})
        else:
            points.append({
                "case_id": cid,
                "order": [int(v) for v in order.tolist()],
                "predecessors": [-1 if int(p) < 0 else int(p) for p in pred.tolist()],
            })
    except Exception:
        points.append({"case_id": cid, "order": None, "predecessors": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize bfs_dfs query");
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
                "failed to spawn python3 for bfs_dfs oracle: {e}"
            );
            eprintln!("skipping bfs_dfs oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open bfs_dfs oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "bfs_dfs oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping bfs_dfs oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for bfs_dfs oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "bfs_dfs oracle failed: {stderr}"
        );
        eprintln!("skipping bfs_dfs oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse bfs_dfs oracle JSON"))
}

#[test]
fn diff_sparse_bfs_dfs_order() -> Result<(), String> {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return Ok(());
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let arms = ["bfs", "dfs"];
    let mut ledger = CompareLedger::new("diff_sparse_bfs_dfs_order", &arms);

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let csr = dense_to_csr(case.rows, case.cols, &case.adj_flat);
        let fsci_result = match case.op.as_str() {
            "bfs" => breadth_first_order(&csr, case.source, true),
            "dfs" => depth_first_order(&csr, case.source, true),
            other => return Err(format!("unknown op {other}")),
        };
        // SciPy raising on a valid traversal case is recorded as oracle_missing.
        let Some(((scipy_order, scipy_preds), (fsci_order, fsci_preds))) = ledger.both(
            &case.op,
            &case.case_id,
            scipy_arm
                .order
                .as_ref()
                .zip(scipy_arm.predecessors.as_ref()),
            fsci_result.ok(),
        ) else {
            continue;
        };
        let fsci_order: Vec<i64> = fsci_order.iter().map(|&v| v as i64).collect();
        let (pass, note) = if &fsci_order != scipy_order {
            (
                false,
                format!("order fsci={fsci_order:?} scipy={scipy_order:?}"),
            )
        } else if &fsci_preds != scipy_preds {
            (
                false,
                format!("predecessors fsci={fsci_preds:?} scipy={scipy_preds:?}"),
            )
        } else {
            (true, "ok".to_string())
        };
        ledger.compared(&case.op, &case.case_id, pass);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            pass,
            note,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_sparse_bfs_dfs_order".into(),
        category: "scipy.sparse.csgraph BFS/DFS order + predecessors (directed=True)".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("{} mismatch: {} note={}", d.op, d.case_id, d.note);
        }
    }

    assert_eq!(
        diffs.len(),
        query.points.len(),
        "bfs_dfs: compared {} of {} cases",
        diffs.len(),
        query.points.len()
    );

    assert!(
        all_pass,
        "bfs_dfs_order conformance failed: {} cases",
        diffs.len()
    );
    let min_per_arm = arms
        .iter()
        .map(|op| query.points.iter().filter(|c| c.op == *op).count())
        .min()
        .unwrap_or(0);
    ledger.finish(min_per_arm);
    Ok(())
}
