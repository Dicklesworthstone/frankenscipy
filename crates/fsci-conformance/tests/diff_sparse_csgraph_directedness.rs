#![forbid(unsafe_code)]
//! Live SciPy diff for `scipy.sparse.csgraph`'s `directed=` semantics (frankenscipy-szq1n.3).
//!
//! Every graph routine used to hard-code one directedness: shortest paths and traversals were
//! directed only, `connected_components` was undirected only, and there was no
//! `connection='strong'`. This compares each routine in BOTH modes with the incumbent on graphs
//! whose two directions differ:
//! - lower-triangle-only and upper-triangle-only storage of a weighted graph;
//! - asymmetric weights, where (i, j) and (j, i) both exist with different weights;
//! - a symmetric graph (both modes must agree with SciPy there too);
//! - a unit-weight grid, where many paths tie and the predecessor tree depends on SciPy's
//!   exact heap and scan order;
//! - a cycle-rich directed graph for strongly connected components;
//! - seeded random graphs (integer weights 1..4, so ties are common).
//!
//! Compared per graph and mode:
//! - `dijkstra` from every source: distances within DIST_ABS_TOL, predecessors exactly.
//! - `bellman_ford` from every source: distances within DIST_ABS_TOL, predecessors exactly.
//! - `floyd_warshall` and `johnson`: distances within DIST_ABS_TOL. fsci's johnson is
//!   Bellman-Ford per source, so its predecessors are not compared.
//! - `breadth_first_order` / `depth_first_order` from every source: order and predecessors
//!   exactly.
//! - `connected_components` weak and strong: component count and labels exactly (SciPy's
//!   numbering, not just the same partition).
//!
//! A graph with one negative edge checks that undirected Bellman-Ford rejects it as a negative
//! cycle in both libraries, while directed Bellman-Ford solves it. SciPy's "no predecessor"
//! -9999 maps to fsci's -1. Compared counts are asserted per routine.

use std::io::Write;
use std::process::Stdio;

use fsci_sparse::{
    Connection, CsrMatrix, Shape2D, bellman_ford, breadth_first_order, connected_components,
    depth_first_order, dijkstra, floyd_warshall, johnson,
};
use serde::{Deserialize, Serialize};

const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
/// Shortest-path distances: sums of at most n integer or small-decimal weights.
const DIST_ABS_TOL: f64 = 1e-12;

#[derive(Debug, Clone, Serialize)]
struct Graph {
    name: String,
    n: usize,
    /// Row-major dense adjacency; nonzero entries are edges.
    dense: Vec<f64>,
}

#[derive(Debug, Deserialize, PartialEq)]
struct Traversal {
    order: Vec<i64>,
    pred: Vec<i64>,
}

/// A distance row as the oracle sends it: JSON has no infinity, so unreachable is `null`.
type DistRow = Vec<Option<f64>>;

fn with_infinity(row: &[Option<f64>]) -> Vec<f64> {
    row.iter().map(|d| d.unwrap_or(f64::INFINITY)).collect()
}

#[derive(Debug, Deserialize)]
struct ModeAnswer {
    /// Per source; `None` when SciPy raised (negative cycle).
    dijkstra: Option<Vec<(DistRow, Vec<i64>)>>,
    bellman_ford: Option<Vec<(DistRow, Vec<i64>)>>,
    johnson: Option<Vec<DistRow>>,
    floyd: Option<Vec<DistRow>>,
    bfs: Vec<Traversal>,
    dfs: Vec<Traversal>,
    cc_weak: (usize, Vec<usize>),
    cc_strong: (usize, Vec<usize>),
}

#[derive(Debug, Deserialize)]
struct Answer {
    directed: ModeAnswer,
    undirected: ModeAnswer,
}

fn dense_to_csr(n: usize, dense: &[f64]) -> CsrMatrix {
    let (mut data, mut indices, mut indptr) = (Vec::new(), Vec::new(), vec![0]);
    for r in 0..n {
        for c in 0..n {
            let v = dense[r * n + c];
            if v != 0.0 {
                data.push(v);
                indices.push(c);
            }
        }
        indptr.push(data.len());
    }
    CsrMatrix::from_components(Shape2D::new(n, n), data, indices, indptr, true)
        .expect("dense_to_csr build")
}

/// splitmix64 on [0, 1).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1_u64 << 53) as f64
    }
}

fn graphs() -> Vec<Graph> {
    let n6 = 6;
    let symmetric: Vec<f64> = vec![
        0.0, 7.0, 9.0, 0.0, 0.0, 14.0, //
        7.0, 0.0, 10.0, 15.0, 0.0, 0.0, //
        9.0, 10.0, 0.0, 11.0, 0.0, 2.0, //
        0.0, 15.0, 11.0, 0.0, 6.0, 0.0, //
        0.0, 0.0, 0.0, 6.0, 0.0, 9.0, //
        14.0, 0.0, 2.0, 0.0, 9.0, 0.0,
    ];
    let triangle = |lower: bool| -> Vec<f64> {
        (0..n6 * n6)
            .map(|k| {
                let (r, c) = (k / n6, k % n6);
                if (lower && r > c) || (!lower && r < c) {
                    symmetric[k]
                } else {
                    0.0
                }
            })
            .collect()
    };
    let asymmetric: Vec<f64> = vec![
        0.0, 5.0, 0.0, 0.0, 2.0, //
        1.0, 0.0, 4.0, 0.0, 0.0, //
        0.0, 0.5, 0.0, 3.0, 0.0, //
        6.0, 0.0, 1.5, 0.0, 0.0, //
        0.0, 0.0, 0.0, 2.5, 0.0,
    ];
    // 3x3 unit grid: many equal-length paths.
    let mut grid = vec![0.0; 81];
    for r in 0..3 {
        for c in 0..3 {
            let i = r * 3 + c;
            if c + 1 < 3 {
                grid[i * 9 + i + 1] = 1.0;
                grid[(i + 1) * 9 + i] = 1.0;
            }
            if r + 1 < 3 {
                grid[i * 9 + i + 3] = 1.0;
                grid[(i + 3) * 9 + i] = 1.0;
            }
        }
    }
    // Cycles 0->1->2->0 and 3->4->5->3, a bridge 2->3, a sink 6 and a self-loop on 7.
    let mut cycles = vec![0.0; 64];
    for &(a, b) in &[
        (0, 1),
        (1, 2),
        (2, 0),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 3),
        (5, 6),
        (7, 7),
    ] {
        cycles[a * 8 + b] = 1.0;
    }
    let mut out = vec![
        Graph {
            name: "symmetric6".into(),
            n: n6,
            dense: symmetric.clone(),
        },
        Graph {
            name: "lower6".into(),
            n: n6,
            dense: triangle(true),
        },
        Graph {
            name: "upper6".into(),
            n: n6,
            dense: triangle(false),
        },
        Graph {
            name: "asymmetric5".into(),
            n: 5,
            dense: asymmetric,
        },
        Graph {
            name: "grid3x3".into(),
            n: 9,
            dense: grid,
        },
        Graph {
            name: "cycles8".into(),
            n: 8,
            dense: cycles,
        },
    ];
    let mut rng = Rng(0x0C59_2026);
    for k in 0..4 {
        let n = 9 + k;
        let dense = (0..n * n)
            .map(|idx| {
                let (r, c) = (idx / n, idx % n);
                if r != c && rng.next() < 0.28 {
                    (1.0 + (rng.next() * 4.0).floor()).min(4.0)
                } else {
                    0.0
                }
            })
            .collect();
        out.push(Graph {
            name: format!("random{n}_seed{k}"),
            n,
            dense,
        });
    }
    out
}

fn scipy_answers(graphs: &[Graph]) -> Option<Vec<Answer>> {
    let script = r#"
import json, sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import (dijkstra, bellman_ford, johnson, floyd_warshall,
    breadth_first_order, depth_first_order, connected_components, NegativeCycleError)

def pred(p):
    return [int(v) if v >= 0 else -1 for v in p]

def dist(row):
    return [float(x) if np.isfinite(x) else None for x in row]

def paths(fn, A, directed, n):
    try:
        d, p = fn(A, directed=directed, indices=list(range(n)), return_predecessors=True)
    except NegativeCycleError:
        return None
    return [(dist(d[s]), pred(p[s])) for s in range(n)]

def mode(A, n, directed):
    try:
        jd = [dist(row) for row in johnson(A, directed=directed)]
    except NegativeCycleError:
        jd = None
    try:
        fd = [dist(row) for row in floyd_warshall(A, directed=directed)]
    except NegativeCycleError:
        fd = None
    trav = lambda f: [dict(zip(("order", "pred"),
                               (lambda o, p: ([int(v) for v in o], pred(p)))(*f(A, s, directed=directed))))
                      for s in range(n)]
    cw = connected_components(A, directed=directed, connection="weak")
    cs = connected_components(A, directed=directed, connection="strong")
    return {"dijkstra": paths(dijkstra, A, directed, n) if (A.data >= 0).all() else None,
            "bellman_ford": paths(bellman_ford, A, directed, n),
            "johnson": jd, "floyd": fd,
            "bfs": trav(breadth_first_order), "dfs": trav(depth_first_order),
            "cc_weak": [int(cw[0]), [int(v) for v in cw[1]]],
            "cc_strong": [int(cs[0]), [int(v) for v in cs[1]]]}

out = []
for g in json.load(sys.stdin):
    n = g["n"]
    A = csr_matrix(np.array(g["dense"], dtype=float).reshape(n, n))
    out.append({"directed": mode(A, n, True), "undirected": mode(A, n, False)})
print(json.dumps(out))
"#;
    let query = serde_json::to_string(graphs).expect("serialize graphs");
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
                "failed to spawn the csgraph oracle: {e}"
            );
            eprintln!("skipping csgraph oracle: python not available ({e})");
            return None;
        }
    };
    child
        .stdin
        .as_mut()
        .expect("oracle stdin")
        .write_all(query.as_bytes())
        .expect("write oracle query");
    let output = child
        .wait_with_output()
        .expect("wait for the csgraph oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "csgraph oracle failed: {stderr}"
        );
        eprintln!("skipping csgraph oracle: scipy not available\n{stderr}");
        return None;
    }
    Some(serde_json::from_slice(&output.stdout).expect("parse csgraph oracle JSON"))
}

fn distances_close(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            (x.is_infinite() && y.is_infinite() && x.signum() == y.signum())
                || (x - y).abs() <= DIST_ABS_TOL
        })
}

#[derive(Default)]
struct Counts {
    paths: usize,
    matrices: usize,
    traversals: usize,
    components: usize,
    refusals: usize,
}

type PathFn =
    fn(&CsrMatrix, bool, usize) -> fsci_sparse::SparseResult<fsci_sparse::ShortestPathResult>;

#[allow(clippy::too_many_lines)]
fn compare_mode(
    graph: &Graph,
    csr: &CsrMatrix,
    directed: bool,
    answer: &ModeAnswer,
    counts: &mut Counts,
    failures: &mut Vec<String>,
) {
    let tag = format!("{} directed={directed}", graph.name);
    let has_negative = graph.dense.iter().any(|&w| w < 0.0);
    let path_arms: [(&str, PathFn, &Option<Vec<(DistRow, Vec<i64>)>>); 2] = [
        ("dijkstra", dijkstra, &answer.dijkstra),
        ("bellman_ford", bellman_ford, &answer.bellman_ford),
    ];
    for (name, fsci_fn, scipy) in path_arms {
        if name == "dijkstra" && has_negative {
            continue; // SciPy's dijkstra warns on negative weights; fsci runs Bellman-Ford.
        }
        match scipy {
            None => {
                counts.refusals += 1;
                if fsci_fn(csr, directed, 0).is_ok() {
                    failures.push(format!(
                        "{tag} {name}: SciPy raised a negative cycle, fsci did not"
                    ));
                }
            }
            Some(rows) => {
                for (s, (dist, pred)) in rows.iter().enumerate() {
                    counts.paths += 1;
                    let dist = with_infinity(dist);
                    match fsci_fn(csr, directed, s) {
                        Ok(r) => {
                            if !distances_close(&r.distances, &dist) || &r.predecessors != pred {
                                failures.push(format!(
                                    "{tag} {name} source {s}: fsci {:?} / {:?}, SciPy {dist:?} / {pred:?}",
                                    r.distances, r.predecessors
                                ));
                            }
                        }
                        Err(e) => {
                            failures.push(format!("{tag} {name} source {s}: fsci Err({e:?})"))
                        }
                    }
                }
            }
        }
    }

    // fsci's floyd_warshall does not detect negative cycles (that is bellman_ford's job), so
    // only the graphs SciPy solves are compared.
    if let Some(scipy) = &answer.floyd {
        let fw = floyd_warshall(csr, directed);
        counts.matrices += 1;
        let same = fw.len() == scipy.len()
            && fw
                .iter()
                .zip(scipy)
                .all(|(a, b)| distances_close(a, &with_infinity(b)));
        if !same {
            failures.push(format!(
                "{tag} floyd_warshall: fsci {fw:?}, SciPy {scipy:?}"
            ));
        }
    }
    match (&answer.johnson, johnson(csr, directed)) {
        (Some(scipy), Ok(ours)) => {
            counts.matrices += 1;
            let same = ours.len() == scipy.len()
                && ours
                    .iter()
                    .zip(scipy)
                    .all(|(a, b)| distances_close(&a.distances, &with_infinity(b)));
            if !same {
                failures.push(format!("{tag} johnson distances differ"));
            }
        }
        (None, Err(_)) => counts.refusals += 1,
        (s, o) => failures.push(format!(
            "{tag} johnson: SciPy {} / fsci {}",
            if s.is_some() { "solved" } else { "refused" },
            if o.is_ok() { "solved" } else { "refused" }
        )),
    }

    for (name, scipy) in [("bfs", &answer.bfs), ("dfs", &answer.dfs)] {
        for (s, want) in scipy.iter().enumerate() {
            counts.traversals += 1;
            let got = if name == "bfs" {
                breadth_first_order(csr, s, directed)
            } else {
                depth_first_order(csr, s, directed)
            };
            match got {
                Ok((order, pred)) => {
                    let order: Vec<i64> = order.iter().map(|&v| v as i64).collect();
                    if order != want.order || pred != want.pred {
                        failures.push(format!(
                            "{tag} {name} from {s}: fsci {order:?} / {pred:?}, SciPy {:?} / {:?}",
                            want.order, want.pred
                        ));
                    }
                }
                Err(e) => failures.push(format!("{tag} {name} from {s}: fsci Err({e:?})")),
            }
        }
    }

    for (connection, (n_components, labels)) in [
        (Connection::Weak, &answer.cc_weak),
        (Connection::Strong, &answer.cc_strong),
    ] {
        counts.components += 1;
        match connected_components(csr, directed, connection) {
            Ok(r) => {
                if r.n_components != *n_components || &r.labels != labels {
                    failures.push(format!(
                        "{tag} connected_components {connection:?}: fsci {} {:?}, SciPy {n_components} {labels:?}",
                        r.n_components, r.labels
                    ));
                }
            }
            Err(e) => failures.push(format!(
                "{tag} connected_components {connection:?}: Err({e:?})"
            )),
        }
    }
}

/// The must-miss arm. If SciPy's own directed and undirected answers agreed on every fixture,
/// a routine that ignored `directed` would pass every row above. So the fixtures must separate
/// the modes for shortest paths, traversals and strong components, and on the graphs where
/// they do, fsci's DIRECTED shortest paths must disagree with SciPy's UNDIRECTED answer: the
/// comparison can see the difference it is there to catch. Returns the number of graphs that
/// separate the modes for each of the three.
fn assert_modes_are_distinguishable(graphs: &[Graph], answers: &[Answer]) -> [usize; 3] {
    let mut separated = [0usize; 3];
    let mut detected = 0usize;
    for (graph, answer) in graphs.iter().zip(answers) {
        let (d, u) = (&answer.directed, &answer.undirected);
        if d.bfs != u.bfs || d.dfs != u.dfs {
            separated[1] += 1;
        }
        if d.cc_strong != u.cc_strong {
            separated[2] += 1;
        }
        let (Some(d_paths), Some(u_paths)) = (&d.dijkstra, &u.dijkstra) else {
            continue;
        };
        if d_paths == u_paths || graph.dense.iter().any(|&w| w < 0.0) {
            continue;
        }
        separated[0] += 1;
        let csr = dense_to_csr(graph.n, &graph.dense);
        let sees_it = u_paths.iter().enumerate().any(|(s, (dist, pred))| {
            dijkstra(&csr, true, s).is_ok_and(|r| {
                !distances_close(&r.distances, &with_infinity(dist)) || &r.predecessors != pred
            })
        });
        detected += usize::from(sees_it);
    }
    assert!(
        separated.iter().all(|&k| k > 0),
        "the fixtures do not separate the modes (paths, traversals, strong components): {separated:?}"
    );
    assert_eq!(
        detected, separated[0],
        "fsci's directed dijkstra matched SciPy's undirected answer on a graph whose modes differ"
    );
    separated
}

#[test]
fn diff_sparse_csgraph_directedness() {
    let mut graphs = graphs();
    // One negative edge: directed Bellman-Ford solves it, undirected it is a negative cycle.
    let mut negative = graphs[3].dense.clone();
    negative[2 * 5 + 1] = -0.25;
    graphs.push(Graph {
        name: "asymmetric5_negative_edge".into(),
        n: 5,
        dense: negative,
    });
    let Some(answers) = scipy_answers(&graphs) else {
        return;
    };
    assert_eq!(
        answers.len(),
        graphs.len(),
        "the oracle must answer every graph"
    );
    let separated = assert_modes_are_distinguishable(&graphs, &answers);
    println!(
        "graphs whose two modes differ in SciPy: {} shortest paths, {} traversals, {} strong components",
        separated[0], separated[1], separated[2]
    );
    let mut counts = Counts::default();
    let mut failures = Vec::new();
    for (graph, answer) in graphs.iter().zip(&answers) {
        let csr = dense_to_csr(graph.n, &graph.dense);
        compare_mode(
            graph,
            &csr,
            true,
            &answer.directed,
            &mut counts,
            &mut failures,
        );
        compare_mode(
            graph,
            &csr,
            false,
            &answer.undirected,
            &mut counts,
            &mut failures,
        );
    }
    println!(
        "{} graphs x 2 modes: {} shortest-path sources, {} distance matrices, {} traversals, \
         {} component labelings, {} negative-cycle refusals matched",
        graphs.len(),
        counts.paths,
        counts.matrices,
        counts.traversals,
        counts.components,
        counts.refusals
    );
    assert!(counts.paths > 0 && counts.traversals > 0 && counts.components > 0);
    assert!(
        counts.refusals > 0,
        "the negative-cycle arm was never exercised"
    );
    assert!(
        failures.is_empty(),
        "csgraph directedness disagrees: {failures:#?}"
    );
}
