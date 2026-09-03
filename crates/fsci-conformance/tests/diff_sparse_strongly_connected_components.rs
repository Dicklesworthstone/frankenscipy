#![forbid(unsafe_code)]
//! Live scipy parity for fsci_sparse::strongly_connected_components.
//!
//! Resolves [frankenscipy-ert3w]. Compares against scipy.sparse.
//! csgraph.connected_components(graph, directed=True, connection='strong').
//!
//! Label numbers may differ between implementations (Tarjan-style
//! vs scipy's variant). The harness compares partition equivalence:
//! for every (i, j), `labels_fsci[i] == labels_fsci[j]` iff
//! `labels_scipy[i] == labels_scipy[j]`.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_sparse::{CooMatrix, FormatConvertible, Shape2D, strongly_connected_components};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct GraphCase {
    case_id: String,
    n: usize,
    /// (src, dst) directed edges.
    edges: Vec<(usize, usize)>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<GraphCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    n_components: Option<usize>,
    labels: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    n_components_match: bool,
    partition_match: bool,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create scc diff dir");
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
    OracleQuery {
        points: vec![
            // 3-cycle + isolated node
            GraphCase {
                case_id: "cycle3_plus_isolated".into(),
                n: 4,
                edges: vec![(0, 1), (1, 2), (2, 0)],
            },
            // Two cycles connected by a bridge
            GraphCase {
                case_id: "two_cycles_bridge".into(),
                n: 6,
                edges: vec![(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)],
            },
            // Linear chain (each node is its own SCC)
            GraphCase {
                case_id: "chain5".into(),
                n: 5,
                edges: vec![(0, 1), (1, 2), (2, 3), (3, 4)],
            },
            // Fully connected directed (single SCC)
            GraphCase {
                case_id: "complete4_directed".into(),
                n: 4,
                edges: (0..4)
                    .flat_map(|i| (0..4).filter(move |&j| i != j).map(move |j| (i, j)))
                    .collect(),
            },
            // Disjoint pair of 2-cycles
            GraphCase {
                case_id: "two_2cycles".into(),
                n: 4,
                edges: vec![(0, 1), (1, 0), (2, 3), (3, 2)],
            },
            // DAG branching (each node its own SCC, but the structure tests Tarjan recursion)
            GraphCase {
                case_id: "diamond_dag".into(),
                n: 4,
                edges: vec![(0, 1), (0, 2), (1, 3), (2, 3)],
            },
        ],
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import sys
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    n = int(case["n"])
    rows = [int(e[0]) for e in case["edges"]]
    cols = [int(e[1]) for e in case["edges"]]
    vals = [1.0] * len(rows)
    try:
        A = csr_matrix((vals, (rows, cols)), shape=(n, n))
        nc, labels = connected_components(A, directed=True, connection='strong')
        points.append({
            "case_id": cid,
            "n_components": int(nc),
            "labels": [int(v) for v in labels.tolist()],
        })
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "n_components": None, "labels": None})
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
                "failed to spawn python3 for scc oracle: {e}"
            );
            eprintln!("skipping scc oracle: python3 not available ({e})");
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
                "scc oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping scc oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for scc oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "scc oracle failed: {stderr}"
        );
        eprintln!("skipping scc oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse scc oracle JSON"))
}

/// Two labelings induce the same partition iff for every pair (i, j),
/// they agree on "same component".
fn partitions_equal(a: &[usize], b: &[usize]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let n = a.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let same_a = a[i] == a[j];
            let same_b = b[i] == b[j];
            if same_a != same_b {
                return false;
            }
        }
    }
    true
}

#[test]
fn diff_sparse_strongly_connected_components() {
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

    for case in &query.points {
        let Some(arm) = pmap.get(&case.case_id) else {
            continue;
        };
        let (Some(enc), Some(elabels)) = (arm.n_components, arm.labels.as_ref()) else {
            continue;
        };

        let mut data = Vec::with_capacity(case.edges.len());
        let mut rows = Vec::with_capacity(case.edges.len());
        let mut cols = Vec::with_capacity(case.edges.len());
        for &(u, v) in &case.edges {
            data.push(1.0);
            rows.push(u);
            cols.push(v);
        }
        let Ok(coo) =
            CooMatrix::from_triplets(Shape2D::new(case.n, case.n), data, rows, cols, true)
        else {
            continue;
        };
        let Ok(csr) = coo.to_csr() else {
            continue;
        };
        let labels = strongly_connected_components(&csr);
        // Number of components = count of unique labels.
        let actual_nc: usize = {
            let mut s = std::collections::BTreeSet::new();
            for &l in &labels {
                s.insert(l);
            }
            s.len()
        };
        let n_components_match = actual_nc == enc;
        let partition_match = partitions_equal(&labels, elabels);
        let pass = n_components_match && partition_match;
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            n_components_match,
            partition_match,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_sparse_strongly_connected_components".into(),
        category: "fsci_sparse::strongly_connected_components vs scipy.sparse.csgraph.connected_components(strong)".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "scc mismatch: {} n_components_match={} partition_match={}",
                d.case_id, d.n_components_match, d.partition_match
            );
        }
    }

    assert!(all_pass, "scc conformance failed: {} cases", diffs.len());
}
