#![forbid(unsafe_code)]
//! Live networkx differential coverage for fsci's
//! `proximity_cliques(data, eps)` — maximal-clique enumeration
//! via Bron-Kerbosch on the eps-radius graph.
//!
//! Resolves [frankenscipy-rqes5]. The canonical reference is
//! `networkx.find_cliques` (Bron-Kerbosch with pivoting).
//!
//! Bron-Kerbosch enumeration order is not canonical across
//! implementations: fsci and networkx may return the same
//! cliques in different orders, and may emit cliques as
//! lists in different vertex orders. So this harness
//! normalizes each clique to a sorted tuple and compares the
//! resulting sets.
//!
//! 4 fixtures × 1 eps each = 4 cases.

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_cluster::proximity_cliques;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-012";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    eps: f64,
    data: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    cliques: Option<Vec<Vec<u64>>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    rust_n_cliques: usize,
    nx_n_cliques: usize,
    set_match: bool,
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
    fs::create_dir_all(output_dir()).expect("create proximity_cliques diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize proximity_cliques diff log");
    fs::write(path, json).expect("write proximity_cliques diff log");
}

fn normalize_cliques<I: IntoIterator<Item = Vec<u64>>>(it: I) -> BTreeSet<Vec<u64>> {
    it.into_iter()
        .map(|mut c| {
            c.sort_unstable();
            c
        })
        .collect()
}

fn generate_query() -> OracleQuery {
    // Fixture 1: triangle plus an isolated vertex. Two
    // points-clique structures: a triangle (1 maximal 3-clique)
    // and an isolated point (1 maximal 1-clique).
    let triangle: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.5, 0.866],
        vec![10.0, 10.0],
    ];

    // Fixture 2: K4 cluster + K3 cluster, well separated.
    // Maximal cliques: {0,1,2,3} and {4,5,6}.
    let two_cliques: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![20.0, 20.0],
        vec![21.0, 20.0],
        vec![20.5, 20.866],
    ];

    // Fixture 3: a bowtie graph — two triangles sharing one
    // vertex. Bron-Kerbosch should find 2 maximal 3-cliques.
    let bowtie: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0],   // A
        vec![1.0, 0.0],   // B
        vec![0.5, 0.866], // C
        vec![1.5, 0.866], // D — within eps of C only? Need to design carefully
        vec![2.0, 0.0],   // E
    ];

    // Fixture 4: chain (path graph). Maximal cliques are the
    // n-1 edges (each a 2-clique), no triangles.
    let chain: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64, 0.0]).collect();

    OracleQuery {
        points: vec![
            PointCase {
                case_id: "triangle_plus_isolated".into(),
                eps: 1.5,
                data: triangle,
            },
            PointCase {
                case_id: "k4_plus_k3_separated".into(),
                eps: 2.0,
                data: two_cliques,
            },
            PointCase {
                case_id: "bowtie_share_vertex".into(),
                eps: 1.05,
                data: bowtie,
            },
            PointCase {
                case_id: "path_5_edges_only".into(),
                eps: 1.05,
                data: chain,
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
import networkx as nx

def build_graph(points, eps):
    n = len(points)
    P = np.asarray(points, dtype=np.float64)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    eps2 = eps * eps
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.sum((P[i] - P[j]) ** 2))
            if d <= eps2:
                g.add_edge(i, j)
    return g

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    out = None
    try:
        g = build_graph(case["data"], case["eps"])
        out = [sorted(int(v) for v in c) for c in nx.find_cliques(g)]
    except Exception:
        out = None
    points.append({"case_id": cid, "cliques": out})
print(json.dumps({"points": points}, allow_nan=False))
"#;
    let query_json = serde_json::to_string(query).expect("serialize proximity_cliques query");
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
                "failed to spawn python3 for proximity_cliques oracle: {e}"
            );
            eprintln!("skipping proximity_cliques oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open proximity_cliques oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "proximity_cliques oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping proximity_cliques oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for proximity_cliques oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "proximity_cliques oracle failed: {stderr}"
        );
        eprintln!("skipping proximity_cliques oracle: networkx not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse proximity_cliques oracle JSON"))
}

#[test]
fn diff_cluster_proximity_cliques() {
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

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(nx_cliques) = scipy_arm.cliques.as_ref() else {
            continue;
        };
        let rust_cliques = proximity_cliques(&case.data, case.eps);

        let rust_set = normalize_cliques(
            rust_cliques
                .iter()
                .map(|c| c.iter().map(|&x| x as u64).collect()),
        );
        let nx_set = normalize_cliques(nx_cliques.iter().cloned());

        let set_match = rust_set == nx_set;
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            rust_n_cliques: rust_set.len(),
            nx_n_cliques: nx_set.len(),
            set_match,
            pass: set_match,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_cluster_proximity_cliques".into(),
        category: "fsci_cluster::proximity_cliques (networkx reference)".into(),
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
                "proximity_cliques mismatch: {} rust_n={} nx_n={} set_match={}",
                d.case_id, d.rust_n_cliques, d.nx_n_cliques, d.set_match
            );
        }
    }

    assert!(
        all_pass,
        "proximity_cliques conformance failed across {} cases",
        diffs.len()
    );
}
