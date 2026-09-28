#![forbid(unsafe_code)]
//! Live SciPy diff for the `scipy.sparse.csgraph` routines of frankenscipy-fdepw:
//! `maximum_flow` (Dinic and Edmonds–Karp), `maximum_bipartite_matching`,
//! `min_weight_full_bipartite_matching`, `breadth_first_tree`, `depth_first_tree`,
//! `reconstruct_path`, `construct_dist_matrix`, `csgraph_from_dense`, `csgraph_to_dense`,
//! `csgraph_masked_from_dense`, `csgraph_from_masked`, `csgraph_to_masked`, `yen`, and
//! `NegativeCycleError` from `bellman_ford`, `johnson`, `floyd_warshall` and `yen`.
//!
//! Every input is generated here from a seeded splitmix64 stream: small integer or dyadic
//! weights so that ties are common (several maximum flows, several optimal matchings, equally
//! long paths), rows stored out of column order, explicit zeros, duplicated entries, and NaN or
//! infinity where the routine gives them a meaning.
//!
//! Everything is compared EXACTLY: integers, index arrays and CSR structure, and floats as
//! values (NaN matches NaN; the sign of a zero is not asserted). fsci ports SciPy's loops
//! operation for operation, so there is no tolerance to name. Where SciPy raises, fsci must
//! refuse, and SciPy's `NegativeCycleError` must be `SparseError::NegativeCycle`.
//!
//! Tree and path matrices are compared after sorting each SciPy row by column: SciPy orders a
//! row's children with `np.argsort`, which is not stable past 16 elements, so its storage order
//! within a row is an accident of numpy's sort (fsci stores children in increasing order); the
//! matrix is the same. Flow matrices are compared in SciPy's own storage order.
//!
//! Must-hit arms: the fixtures have to separate SciPy's answer from a plausible wrong one — a
//! depth-first Ford–Fulkerson flow, a greedy matching, the lexicographically first optimal
//! assignment, a list of Yen paths without equal lengths, a mask that is exact equality — or
//! the test fails, since a comparison that never meets a tie cannot see a tie broken wrongly.

use std::collections::HashMap;
use std::io::Write;
use std::process::Stdio;

use fsci_conformance::CompareLedger;
use fsci_sparse::{
    CsrMatrix, MaskedGraph, MatchingPermType, MaximumFlowMethod, Shape2D, SparseError,
    bellman_ford, breadth_first_tree, construct_dist_matrix, csgraph_from_dense,
    csgraph_from_masked, csgraph_masked_from_dense, csgraph_to_dense, csgraph_to_masked,
    depth_first_tree, floyd_warshall, johnson, maximum_bipartite_matching, maximum_flow,
    min_weight_full_bipartite_matching, reconstruct_path, yen,
};
use serde::{Deserialize, Serialize};

const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
/// SciPy's "no predecessor"; fsci reads any negative predecessor as none.
const NULL_PRED: i64 = -9999;
const INF: f64 = f64::INFINITY;
const ARMS: [&str; 14] = [
    "maximum_flow",
    "maximum_bipartite_matching",
    "min_weight_full_bipartite_matching",
    "breadth_first_tree",
    "depth_first_tree",
    "reconstruct_path",
    "construct_dist_matrix",
    "csgraph_from_dense",
    "csgraph_to_dense",
    "csgraph_masked_from_dense",
    "csgraph_from_masked",
    "csgraph_to_masked",
    "yen",
    "negative_cycle",
];
/// The smallest arm (`negative_cycle`, `csgraph_from_masked`) has at least this many cases.
const MIN_PER_ARM: usize = 12;

/// splitmix64.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    fn chance(&mut self, p: f64) -> bool {
        self.unit() < p
    }

    fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[self.below(xs.len())]
    }

    fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            let j = self.below(i + 1);
            xs.swap(i, j);
        }
    }
}

/// Floats cross the pipe as text: JSON has no NaN or infinity.
fn text(x: f64) -> String {
    format!("{x:?}")
}

fn num(s: &str) -> f64 {
    s.parse()
        .unwrap_or_else(|_| panic!("unparseable float from the oracle: {s}"))
}

fn floats(rows: &[Vec<String>]) -> Vec<Vec<f64>> {
    rows.iter()
        .map(|row| row.iter().map(|s| num(s)).collect())
        .collect()
}

fn texts(rows: &[Vec<f64>]) -> Vec<Vec<String>> {
    rows.iter()
        .map(|row| row.iter().map(|&x| text(x)).collect())
        .collect()
}

/// NaN matches NaN; otherwise `==` (so 0.0 matches -0.0).
fn same(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || a == b
}

/// A CSR graph exactly as stored: explicit zeros, duplicates and row order are kept.
#[derive(Debug, Clone, Serialize)]
struct Graph {
    rows: usize,
    cols: usize,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<String>,
}

impl Graph {
    /// `entries` grouped by row, each row in the storage order wanted.
    fn new(rows: usize, cols: usize, entries: &[(usize, usize, f64)]) -> Self {
        assert!(entries.windows(2).all(|w| w[0].0 <= w[1].0));
        let mut indptr = vec![0usize; rows + 1];
        for &(r, _, _) in entries {
            indptr[r + 1] += 1;
        }
        for r in 0..rows {
            indptr[r + 1] += indptr[r];
        }
        Self {
            rows,
            cols,
            indptr,
            indices: entries.iter().map(|e| e.1).collect(),
            data: entries.iter().map(|e| text(e.2)).collect(),
        }
    }

    fn csr(&self) -> CsrMatrix {
        CsrMatrix::from_components(
            Shape2D::new(self.rows, self.cols),
            self.data.iter().map(|d| num(d)).collect(),
            self.indices.clone(),
            self.indptr.clone(),
            false,
        )
        .expect("generated graph is a valid CSR")
    }

    fn entries(&self) -> Vec<(usize, usize, f64)> {
        (0..self.rows)
            .flat_map(|r| (self.indptr[r]..self.indptr[r + 1]).map(move |e| (r, e)))
            .map(|(r, e)| (r, self.indices[e], num(&self.data[e])))
            .collect()
    }
}

/// Random entries: each off-diagonal pair with probability `density`, its weight from
/// `weights`; `shuffle` stores each row out of column order.
fn random_entries(
    rng: &mut Rng,
    rows: usize,
    cols: usize,
    density: f64,
    weights: &[f64],
    shuffle: bool,
) -> Vec<(usize, usize, f64)> {
    let mut entries = Vec::new();
    for r in 0..rows {
        let mut row: Vec<(usize, usize, f64)> = (0..cols)
            .filter(|&c| !(rows == cols && r == c))
            .filter(|_| rng.chance(density))
            .map(|c| (r, c, 0.0))
            .collect();
        for entry in &mut row {
            entry.2 = rng.pick(weights);
        }
        if shuffle {
            rng.shuffle(&mut row);
        }
        entries.extend(row);
    }
    entries
}

/// One call. Unused fields keep their defaults.
#[derive(Debug, Clone, Default, Serialize)]
struct Case {
    id: String,
    func: &'static str,
    graph: Option<Graph>,
    int_capacities: bool,
    source: usize,
    sink: usize,
    method: &'static str,
    directed: bool,
    perm: &'static str,
    maximize: bool,
    start: usize,
    pred: Vec<i64>,
    pred_matrix: Vec<Vec<i64>>,
    null_value: Option<String>,
    nan_null: bool,
    infinity_null: bool,
    dense: Vec<Vec<String>>,
    mask: Vec<Vec<bool>>,
    k: usize,
    unweighted: bool,
    kind: &'static str,
    /// SciPy is expected to raise (a ValueError; `NegativeCycleError` for `negative_cycle`).
    #[serde(skip)]
    expect_raise: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct CsrParts {
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct Answer {
    id: String,
    status: String,
    error: Option<String>,
    flow_value: Option<i64>,
    csr: Option<CsrParts>,
    raw_sorted: Option<bool>,
    perm: Option<Vec<i64>>,
    rows: Option<Vec<usize>>,
    cols: Option<Vec<usize>>,
    dense: Option<Vec<Vec<String>>>,
    mask: Option<Vec<Vec<bool>>>,
    dist: Option<Vec<String>>,
    preds: Option<Vec<Vec<i64>>>,
}

fn flow_cases(rng: &mut Rng) -> Vec<Case> {
    let mut cases = Vec::new();
    for g in 0..24 {
        let n = 4 + rng.below(9);
        let density = 0.25 + 0.3 * rng.unit();
        let shuffle = g % 3 == 1;
        let duplicate = g % 6 == 3;
        let mut entries = Vec::new();
        for r in 0..n {
            let mut row = Vec::new();
            for c in 0..n {
                if r == c || !rng.chance(density) {
                    continue;
                }
                let cap = if rng.chance(0.08) {
                    0.0
                } else if rng.chance(0.05) {
                    -(1.0 + rng.below(3) as f64)
                } else {
                    1.0 + rng.below(5) as f64
                };
                row.push((r, c, cap));
                if duplicate && rng.chance(0.2) {
                    row.push((r, c, 1.0 + rng.below(3) as f64));
                }
            }
            if shuffle {
                rng.shuffle(&mut row);
            }
            entries.extend(row);
        }
        let source = rng.below(n);
        let sink = (source + 1 + rng.below(n - 1)) % n;
        let graph = Graph::new(n, n, &entries);
        for method in ["dinic", "edmonds_karp"] {
            cases.push(Case {
                id: format!("flow{g}_{method}"),
                func: "maximum_flow",
                graph: Some(graph.clone()),
                int_capacities: true,
                source,
                sink,
                method,
                ..Case::default()
            });
        }
    }
    // SciPy's ValueErrors: a fractional capacity (float dtype), source == sink, a non-square
    // graph, a sink out of range.
    let square = Graph::new(3, 3, &[(0, 1, 2.0), (1, 2, 3.0)]);
    let refusals = [
        (
            "fractional",
            Graph::new(3, 3, &[(0, 1, 1.5), (1, 2, 3.0)]),
            false,
            0,
            2,
        ),
        ("source_is_sink", square.clone(), true, 1, 1),
        (
            "non_square",
            Graph::new(3, 4, &[(0, 1, 1.0), (1, 3, 2.0)]),
            true,
            0,
            1,
        ),
        ("sink_out_of_range", square, true, 0, 3),
    ];
    for (name, graph, int_capacities, source, sink) in refusals {
        cases.push(Case {
            id: format!("flow_refuse_{name}"),
            func: "maximum_flow",
            graph: Some(graph),
            int_capacities,
            source,
            sink,
            method: "dinic",
            expect_raise: true,
            ..Case::default()
        });
    }
    cases
}

fn matching_cases(rng: &mut Rng) -> Vec<Case> {
    let mut cases = Vec::new();
    let mut graphs = Vec::new();
    for g in 0..20 {
        let rows = 1 + rng.below(10);
        let cols = 1 + rng.below(10);
        let density = 0.15 + 0.35 * rng.unit();
        // Explicit zeros are edges too.
        let weights: &[f64] = if g % 5 == 2 { &[0.0, 1.0] } else { &[1.0] };
        let entries = random_entries(rng, rows, cols, density, weights, g % 3 == 1);
        graphs.push(Graph::new(rows, cols, &entries));
    }
    graphs.push(Graph::new(3, 0, &[]));
    graphs.push(Graph::new(0, 2, &[]));
    for (g, graph) in graphs.into_iter().enumerate() {
        for perm in ["row", "column"] {
            cases.push(Case {
                id: format!("match{g}_{perm}"),
                func: "maximum_bipartite_matching",
                graph: Some(graph.clone()),
                perm,
                ..Case::default()
            });
        }
    }
    cases
}

fn assignment_cases(rng: &mut Rng) -> Vec<Case> {
    let mut cases = Vec::new();
    for g in 0..30 {
        let n = 2 + rng.below(6);
        let (rows, cols) = match g % 3 {
            0 => (n, n),
            1 => (n, n + 1 + rng.below(3)),
            _ => (n + 1 + rng.below(3), n),
        };
        let density = 0.55 + 0.4 * rng.unit();
        let weights: &[f64] = if g % 2 == 0 {
            &[1.0, 2.0, 3.0]
        } else {
            &[0.5, 1.25, 2.0, 3.5, 7.0]
        };
        let mut entries = random_entries(rng, rows, cols, density, weights, g % 5 == 1);
        // An explicit zero or +inf is not an edge (SciPy removes it, warning for the zero).
        for entry in &mut entries {
            if rng.chance(0.05) {
                entry.2 = 0.0;
            } else if rng.chance(0.04) {
                entry.2 = INF;
            }
        }
        cases.push(Case {
            id: format!("assign{g}"),
            func: "min_weight_full_bipartite_matching",
            graph: Some(Graph::new(rows, cols, &entries)),
            maximize: g % 4 == 3,
            ..Case::default()
        });
    }
    // A NaN weight is an edge that no comparison prefers.
    for g in 0..3 {
        let n = 3 + g;
        let mut entries = random_entries(rng, n, n, 0.8, &[1.0, 2.0], false);
        let at = rng.below(entries.len());
        entries[at].2 = f64::NAN;
        cases.push(Case {
            id: format!("assign_nan{g}"),
            func: "min_weight_full_bipartite_matching",
            graph: Some(Graph::new(n, n, &entries)),
            ..Case::default()
        });
    }
    // No full matching: an empty column; a tall graph with one usable column.
    let infeasible = [
        Graph::new(3, 3, &[(0, 0, 1.0), (0, 1, 2.0), (1, 0, 1.0), (2, 1, 4.0)]),
        Graph::new(4, 2, &[(0, 0, 1.0), (1, 0, 2.0), (3, 0, 3.0)]),
    ];
    for (g, graph) in infeasible.into_iter().enumerate() {
        cases.push(Case {
            id: format!("assign_infeasible{g}"),
            func: "min_weight_full_bipartite_matching",
            graph: Some(graph),
            expect_raise: true,
            ..Case::default()
        });
    }
    cases
}

fn tree_cases(rng: &mut Rng) -> Vec<Case> {
    let mut cases = Vec::new();
    for g in 0..16 {
        let n = rng.pick(&[5, 8, 12, 17, 20, 24, 30]);
        let density = (2.5 / n as f64 + 0.1 * rng.unit()).min(0.5);
        let mut entries = random_entries(rng, n, n, density, &[1.0, 2.0, 3.0, 0.0], g % 4 == 1);
        if g % 5 == 4 {
            // A star: node 0 is the parent of many nodes, past numpy's 16-element sort.
            entries.retain(|e| e.0 != 0);
            let star: Vec<(usize, usize, f64)> = (1..n).map(|c| (0, c, 1.0)).collect();
            entries.splice(0..0, star);
        }
        let graph = Graph::new(n, n, &entries);
        let start = if g % 5 == 4 { 0 } else { rng.below(n) };
        for directed in [true, false] {
            for func in ["breadth_first_tree", "depth_first_tree"] {
                cases.push(Case {
                    id: format!("{func}{g}_{directed}"),
                    func,
                    graph: Some(graph.clone()),
                    start,
                    directed,
                    ..Case::default()
                });
            }
        }
    }
    cases
}

fn reconstruct_cases(rng: &mut Rng) -> Vec<Case> {
    let mut cases = Vec::new();
    for g in 0..16 {
        let n = 3 + rng.below(20);
        let mut entries = random_entries(rng, n, n, 0.3, &[1.0, 2.0, 3.0, 0.0], false);
        if g % 4 == 1 {
            // Duplicated entries add up in `csgraph[p, i]`.
            let doubled: Vec<(usize, usize, f64)> =
                entries.iter().flat_map(|&e| [e, (e.0, e.1, 0.5)]).collect();
            entries = doubled;
        }
        let pred: Vec<i64> = (0..n)
            .map(|_| {
                if rng.chance(0.25) {
                    rng.pick(&[NULL_PRED, -1])
                } else {
                    rng.below(n) as i64
                }
            })
            .collect();
        let graph = Graph::new(n, n, &entries);
        for directed in [true, false] {
            cases.push(Case {
                id: format!("reconstruct{g}_{directed}"),
                func: "reconstruct_path",
                graph: Some(graph.clone()),
                pred: pred.clone(),
                directed,
                ..Case::default()
            });
        }
    }
    cases
}

/// Row `i` of a predecessor matrix whose chains end at `i` or at a node with no predecessor:
/// each node's parent is a node placed before it, an edge into it when one exists.
fn random_predecessor_forest(
    rng: &mut Rng,
    n: usize,
    root: usize,
    has_edge: &dyn Fn(usize, usize) -> bool,
) -> Vec<i64> {
    let mut row = vec![NULL_PRED; n];
    let mut order: Vec<usize> = (0..n).filter(|&v| v != root).collect();
    rng.shuffle(&mut order);
    let mut placed = vec![root];
    for v in order {
        if !rng.chance(0.2) {
            let along_edges: Vec<usize> =
                placed.iter().copied().filter(|&p| has_edge(p, v)).collect();
            let parent = if !along_edges.is_empty() && rng.chance(0.8) {
                rng.pick(&along_edges)
            } else {
                rng.pick(&placed)
            };
            row[v] = parent as i64;
        }
        placed.push(v);
    }
    row
}

fn dist_matrix_cases(rng: &mut Rng) -> Vec<Case> {
    let mut cases = Vec::new();
    for g in 0..12 {
        let n = 3 + rng.below(8);
        let mut entries = random_entries(rng, n, n, 0.4, &[0.5, 1.25, 2.0, 3.5, 0.0], false);
        if g % 3 == 2 {
            // Duplicates: the lightest one is the edge's weight.
            let doubled: Vec<(usize, usize, f64)> =
                entries.iter().flat_map(|&e| [e, (e.0, e.1, 1.0)]).collect();
            entries = doubled;
        }
        let graph = Graph::new(n, n, &entries);
        let stored: Vec<(usize, usize)> = entries.iter().map(|e| (e.0, e.1)).collect();
        for directed in [true, false] {
            let has_edge = |p: usize, v: usize| {
                stored.contains(&(p, v)) || (!directed && stored.contains(&(v, p)))
            };
            let pred_matrix: Vec<Vec<i64>> = (0..n)
                .map(|root| random_predecessor_forest(rng, n, root, &has_edge))
                .collect();
            cases.push(Case {
                id: format!("dist{g}_{directed}"),
                func: "construct_dist_matrix",
                graph: Some(graph.clone()),
                pred_matrix,
                directed,
                null_value: Some(text(rng.pick(&[INF, -1.0, 0.0, 99.5]))),
                ..Case::default()
            });
        }
    }
    cases
}

fn dense_cases(rng: &mut Rng) -> Vec<Case> {
    let mut cases = Vec::new();
    let nulls = [
        Some(0.0),
        Some(1.5),
        Some(-2.0),
        None,
        Some(f64::NAN),
        Some(INF),
        Some(-INF),
    ];
    for g in 0..20 {
        let n = 1 + rng.below(6);
        let null = rng.pick(&nulls);
        let base = null.filter(|v: &f64| v.is_finite()).unwrap_or(0.0);
        // Values at, near (inside and outside np.isclose), and far from the null value.
        let pool = [
            0.0,
            1e-9,
            -5e-9,
            2e-8,
            1.0,
            2.5,
            f64::NAN,
            INF,
            -INF,
            base,
            base * (1.0 + 1e-6),
            base * (1.0 + 3e-5) + 3e-8,
            base + 5e-9,
            base - 2e-8,
        ];
        let dense: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..n).map(|_| rng.pick(&pool)).collect())
            .collect();
        let (nan_null, infinity_null) = (rng.chance(0.5), rng.chance(0.5));
        for func in ["csgraph_from_dense", "csgraph_masked_from_dense"] {
            cases.push(Case {
                id: format!("{func}{g}"),
                func,
                dense: texts(&dense),
                null_value: null.map(text),
                nan_null,
                infinity_null,
                ..Case::default()
            });
        }
    }
    for g in 0..12 {
        let n = 1 + rng.below(6);
        let pool = [0.0, 1.0, -2.5, 3.25, f64::NAN, INF];
        let dense: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..n).map(|_| rng.pick(&pool)).collect())
            .collect();
        let mask: Vec<Vec<bool>> = (0..n)
            .map(|_| (0..n).map(|_| rng.chance(0.4)).collect())
            .collect();
        cases.push(Case {
            id: format!("from_masked{g}"),
            func: "csgraph_from_masked",
            dense: texts(&dense),
            mask,
            ..Case::default()
        });
    }
    // Sparse to dense: duplicates (the smallest wins), explicit zeros, NaN (reads as
    // infinity), infinities, rows out of order.
    for g in 0..16 {
        let n = 1 + rng.below(7);
        let mut entries = random_entries(
            rng,
            n,
            n,
            0.5,
            &[0.0, 1.0, 2.0, -1.5, f64::NAN, INF, -INF],
            false,
        );
        let mut extra = Vec::new();
        for &(r, c, _) in &entries {
            if rng.chance(0.3) {
                extra.push((r, c, rng.pick(&[0.5, 3.0, -4.0])));
            }
        }
        entries.extend(extra);
        entries.sort_by_key(|e| e.0);
        if g % 2 == 1 {
            let mut start = 0;
            while start < entries.len() {
                let row = entries[start].0;
                let end = start + entries[start..].iter().take_while(|e| e.0 == row).count();
                rng.shuffle(&mut entries[start..end]);
                start = end;
            }
        }
        let graph = Graph::new(n, n, &entries);
        cases.push(Case {
            id: format!("to_dense{g}"),
            func: "csgraph_to_dense",
            graph: Some(graph.clone()),
            null_value: Some(text(rng.pick(&[0.0, INF, -1.0, f64::NAN]))),
            ..Case::default()
        });
        if g < 12 {
            cases.push(Case {
                id: format!("to_masked{g}"),
                func: "csgraph_to_masked",
                graph: Some(graph),
                ..Case::default()
            });
        }
    }
    cases
}

fn yen_cases(rng: &mut Rng) -> Vec<Case> {
    let mut cases = Vec::new();
    for g in 0..24 {
        let n = 4 + rng.below(7);
        let density = 0.35 + 0.25 * rng.unit();
        let weights: &[f64] = if g % 2 == 0 {
            &[1.0, 2.0]
        } else {
            &[1.0, 1.5, 2.0, 3.0, 0.0]
        };
        let entries = random_entries(rng, n, n, density, weights, g % 5 == 2);
        let source = rng.below(n);
        let sink = if g % 8 == 5 {
            source
        } else {
            (source + 1 + rng.below(n - 1)) % n
        };
        cases.push(Case {
            id: format!("yen{g}"),
            func: "yen",
            graph: Some(Graph::new(n, n, &entries)),
            source,
            sink,
            k: 1 + rng.below(8),
            directed: g % 3 != 0,
            unweighted: g % 4 == 1,
            ..Case::default()
        });
    }
    // Negative weights without a negative cycle (edges only run forward): Johnson reweighting.
    for g in 0..6 {
        let n = 5 + rng.below(4);
        let mut entries = Vec::new();
        for r in 0..n {
            for c in r + 1..n {
                if rng.chance(0.55) {
                    entries.push((r, c, rng.pick(&[-2.0, -1.0, 1.0, 2.0, 3.0])));
                }
            }
        }
        cases.push(Case {
            id: format!("yen_negative{g}"),
            func: "yen",
            graph: Some(Graph::new(n, n, &entries)),
            source: 0,
            sink: n - 1,
            k: 2 + rng.below(5),
            directed: true,
            ..Case::default()
        });
    }
    cases
}

fn negative_cycle_cases(rng: &mut Rng) -> Vec<Case> {
    let mut cases = Vec::new();
    let cycle = [(0, 1, 1.0), (1, 2, -2.5), (2, 0, 0.5)];
    for g in 0..3 {
        let n = 4 + rng.below(5);
        let mut dense = vec![vec![0.0; n]; n];
        for (r, row) in dense.iter_mut().enumerate() {
            for (c, w) in row.iter_mut().enumerate() {
                if r != c && rng.chance(0.3) {
                    *w = rng.pick(&[1.0, 2.0, 3.0]);
                }
            }
        }
        for &(r, c, w) in &cycle {
            dense[r][c] = w;
            dense[c][r] = 0.0;
        }
        let entries: Vec<(usize, usize, f64)> = dense
            .iter()
            .enumerate()
            .flat_map(|(r, row)| {
                row.iter()
                    .enumerate()
                    .filter(|&(_, &w)| w != 0.0)
                    .map(move |(c, &w)| (r, c, w))
            })
            .collect();
        let graph = Graph::new(n, n, &entries);
        for kind in ["bellman_ford", "johnson", "floyd_warshall", "yen"] {
            cases.push(Case {
                id: format!("negcycle{g}_{kind}"),
                func: "negative_cycle",
                graph: Some(graph.clone()),
                kind,
                source: 0,
                sink: n - 1,
                k: 2,
                directed: true,
                expect_raise: true,
                ..Case::default()
            });
        }
    }
    // Undirected, one negative edge is a negative cycle (walk it and back).
    let one_negative = Graph::new(4, 4, &[(0, 1, 2.0), (1, 2, -0.5), (2, 3, 1.0)]);
    for kind in ["bellman_ford", "johnson", "floyd_warshall", "yen"] {
        cases.push(Case {
            id: format!("negcycle_undirected_{kind}"),
            func: "negative_cycle",
            graph: Some(one_negative.clone()),
            kind,
            source: 0,
            sink: 3,
            k: 2,
            directed: false,
            expect_raise: true,
            ..Case::default()
        });
    }
    cases
}

fn scipy_answers(cases: &[Case]) -> Option<Vec<Answer>> {
    let script = r#"
import json, sys, warnings
import numpy as np
from scipy.sparse import csr_array
import scipy.sparse.csgraph as cg

# min_weight_full_bipartite_matching warns about explicit zeros; the value is what is compared.
warnings.simplefilter("ignore")

def s(x):
    return repr(float(x))

def graph(g, int_capacities=False):
    data = np.array([float(v) for v in g["data"]], dtype=np.float64)
    if int_capacities:
        data = data.astype(np.int32)
    return csr_array((data, np.array(g["indices"], dtype=np.int32),
                      np.array(g["indptr"], dtype=np.int32)), shape=(g["rows"], g["cols"]))

def parts(m, canonical):
    m = csr_array(m)
    raw_sorted = bool(m.has_sorted_indices)
    if canonical:
        m = m.sorted_indices()
    return {"csr": {"indptr": [int(v) for v in m.indptr],
                    "indices": [int(v) for v in m.indices],
                    "data": [s(v) for v in m.data]},
            "raw_sorted": raw_sorted}

def dense(a):
    return [[s(v) for v in row] for row in np.asarray(a, dtype=np.float64)]

def floats(rows):
    return np.array([[float(v) for v in row] for row in rows], dtype=np.float64)

def pred(v):
    return int(v) if v >= 0 else -1

def run(c):
    f = c["func"]
    null = None if c["null_value"] is None else float(c["null_value"])
    if f == "maximum_flow":
        r = cg.maximum_flow(graph(c["graph"], c["int_capacities"]), c["source"], c["sink"],
                            method=c["method"])
        return {"flow_value": int(r.flow_value), **parts(r.flow, False)}
    if f == "maximum_bipartite_matching":
        p = cg.maximum_bipartite_matching(graph(c["graph"]), perm_type=c["perm"])
        return {"perm": [int(v) for v in p]}
    if f == "min_weight_full_bipartite_matching":
        r, k = cg.min_weight_full_bipartite_matching(graph(c["graph"]), maximize=c["maximize"])
        return {"rows": [int(v) for v in r], "cols": [int(v) for v in k]}
    if f == "breadth_first_tree":
        t = cg.breadth_first_tree(graph(c["graph"]), c["start"], directed=c["directed"])
        return parts(t, True)
    if f == "depth_first_tree":
        t = cg.depth_first_tree(graph(c["graph"]), c["start"], directed=c["directed"])
        return parts(t, True)
    if f == "reconstruct_path":
        p = np.array(c["pred"], dtype=np.int32)
        return parts(cg.reconstruct_path(graph(c["graph"]), p, directed=c["directed"]), True)
    if f == "construct_dist_matrix":
        p = np.array(c["pred_matrix"], dtype=np.int32)
        d = cg.construct_dist_matrix(graph(c["graph"]), p, directed=c["directed"],
                                     null_value=null)
        return {"dense": dense(d)}
    if f == "csgraph_from_dense":
        return parts(cg.csgraph_from_dense(floats(c["dense"]), null_value=null,
                                           nan_null=c["nan_null"],
                                           infinity_null=c["infinity_null"]), False)
    if f == "csgraph_masked_from_dense":
        m = cg.csgraph_masked_from_dense(floats(c["dense"]), null_value=null,
                                         nan_null=c["nan_null"],
                                         infinity_null=c["infinity_null"])
        return {"dense": dense(m.data), "mask": np.ma.getmaskarray(m).tolist()}
    if f == "csgraph_from_masked":
        m = np.ma.masked_array(floats(c["dense"]), mask=np.array(c["mask"], dtype=bool))
        return parts(cg.csgraph_from_masked(m), False)
    if f == "csgraph_to_dense":
        return {"dense": dense(cg.csgraph_to_dense(graph(c["graph"]), null_value=null))}
    if f == "csgraph_to_masked":
        m = cg.csgraph_to_masked(graph(c["graph"]))
        return {"dense": dense(m.data), "mask": np.ma.getmaskarray(m).tolist()}
    if f == "yen":
        d, p = cg.yen(graph(c["graph"]), c["source"], c["sink"], c["k"],
                      directed=c["directed"], return_predecessors=True,
                      unweighted=c["unweighted"])
        return {"dist": [s(v) for v in d], "preds": [[pred(v) for v in row] for row in p]}
    if f == "negative_cycle":
        a, kind = graph(c["graph"]), c["kind"]
        if kind == "bellman_ford":
            cg.bellman_ford(a, directed=c["directed"], indices=c["source"])
        elif kind == "johnson":
            cg.johnson(a, directed=c["directed"])
        elif kind == "floyd_warshall":
            cg.floyd_warshall(a, directed=c["directed"])
        else:
            cg.yen(a, c["source"], c["sink"], c["k"], directed=c["directed"])
        return {}
    raise KeyError(f)

out = []
for c in json.load(sys.stdin):
    try:
        ans = {"status": "ok", **run(c)}
    except cg.NegativeCycleError as e:
        ans = {"status": "negative_cycle", "error": str(e)}
    except Exception as e:
        ans = {"status": "raised", "error": f"{type(e).__name__}: {e}"}
    ans["id"] = c["id"]
    out.append(ans)
print(json.dumps(out))
"#;
    let query = serde_json::to_string(cases).expect("serialize cases");
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
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

/// What fsci returned for a case.
#[derive(Debug)]
enum Fsci {
    Flow(i64, CsrMatrix),
    Perm(Vec<i64>),
    Assignment(Vec<usize>, Vec<usize>),
    Csr(CsrMatrix),
    Dense(Vec<Vec<f64>>),
    Masked(MaskedGraph),
    Paths(Vec<f64>, Vec<Vec<i64>>),
    Solved,
}

fn run_fsci(case: &Case) -> Result<Fsci, SparseError> {
    let graph = || case.graph.as_ref().expect("case has a graph").csr();
    let null = case.null_value.as_deref().map(num);
    Ok(match case.func {
        "maximum_flow" => {
            let method = if case.method == "dinic" {
                MaximumFlowMethod::Dinic
            } else {
                MaximumFlowMethod::EdmondsKarp
            };
            let r = maximum_flow(&graph(), case.source, case.sink, method)?;
            Fsci::Flow(r.flow_value, r.flow)
        }
        "maximum_bipartite_matching" => {
            let perm = if case.perm == "row" {
                MatchingPermType::Row
            } else {
                MatchingPermType::Column
            };
            Fsci::Perm(maximum_bipartite_matching(&graph(), perm))
        }
        "min_weight_full_bipartite_matching" => {
            let (rows, cols) = min_weight_full_bipartite_matching(&graph(), case.maximize)?;
            Fsci::Assignment(rows, cols)
        }
        "breadth_first_tree" => Fsci::Csr(breadth_first_tree(&graph(), case.start, case.directed)?),
        "depth_first_tree" => Fsci::Csr(depth_first_tree(&graph(), case.start, case.directed)?),
        "reconstruct_path" => Fsci::Csr(reconstruct_path(&graph(), &case.pred, case.directed)?),
        "construct_dist_matrix" => Fsci::Dense(construct_dist_matrix(
            &graph(),
            &case.pred_matrix,
            case.directed,
            null.expect("null value"),
        )?),
        "csgraph_from_dense" => Fsci::Csr(csgraph_from_dense(
            &floats(&case.dense),
            null,
            case.nan_null,
            case.infinity_null,
        )?),
        "csgraph_masked_from_dense" => Fsci::Masked(csgraph_masked_from_dense(
            &floats(&case.dense),
            null,
            case.nan_null,
            case.infinity_null,
        )?),
        "csgraph_from_masked" => Fsci::Csr(csgraph_from_masked(&MaskedGraph {
            data: floats(&case.dense),
            mask: case.mask.clone(),
        })?),
        "csgraph_to_dense" => Fsci::Dense(csgraph_to_dense(&graph(), null.expect("null value"))?),
        "csgraph_to_masked" => Fsci::Masked(csgraph_to_masked(&graph())?),
        "yen" => {
            let r = yen(
                &graph(),
                case.source,
                case.sink,
                case.k,
                case.directed,
                case.unweighted,
            )?;
            Fsci::Paths(r.distances, r.predecessors)
        }
        "negative_cycle" => {
            let g = graph();
            match case.kind {
                "bellman_ford" => {
                    bellman_ford(&g, case.directed, case.source)?;
                }
                "johnson" => {
                    johnson(&g, case.directed)?;
                }
                "floyd_warshall" => {
                    floyd_warshall(&g, case.directed)?;
                }
                _ => {
                    yen(&g, case.source, case.sink, case.k, case.directed, false)?;
                }
            }
            Fsci::Solved
        }
        other => panic!("unknown func {other}"),
    })
}

fn same_csr(m: &CsrMatrix, want: &CsrParts) -> bool {
    m.indptr() == want.indptr.as_slice()
        && m.indices() == want.indices.as_slice()
        && m.data().len() == want.data.len()
        && m.data()
            .iter()
            .zip(&want.data)
            .all(|(&a, b)| same(a, num(b)))
}

fn same_dense(ours: &[Vec<f64>], want: &[Vec<String>]) -> bool {
    ours.len() == want.len()
        && ours
            .iter()
            .zip(want)
            .all(|(a, b)| a.len() == b.len() && a.iter().zip(b).all(|(&x, y)| same(x, num(y))))
}

/// `None` when the oracle's answer lacks the field this output is compared on.
fn matches_scipy(ours: &Fsci, answer: &Answer) -> Option<bool> {
    Some(match ours {
        Fsci::Flow(value, flow) => {
            answer.flow_value? == *value && same_csr(flow, answer.csr.as_ref()?)
        }
        Fsci::Perm(perm) => answer.perm.as_ref()? == perm,
        Fsci::Assignment(rows, cols) => {
            answer.rows.as_ref()? == rows && answer.cols.as_ref()? == cols
        }
        Fsci::Csr(m) => same_csr(m, answer.csr.as_ref()?),
        Fsci::Dense(d) => same_dense(d, answer.dense.as_ref()?),
        Fsci::Masked(m) => {
            same_dense(&m.data, answer.dense.as_ref()?) && answer.mask.as_ref()? == &m.mask
        }
        Fsci::Paths(dist, preds) => {
            let want = answer.dist.as_ref()?;
            dist.len() == want.len()
                && dist.iter().zip(want).all(|(&a, b)| same(a, num(b)))
                && answer.preds.as_ref()? == preds
        }
        Fsci::Solved => true,
    })
}

fn check(case: &Case, answer: &Answer, ledger: &mut CompareLedger, failures: &mut Vec<String>) {
    let arm = case.func;
    let ours = run_fsci(case);
    if case.expect_raise || answer.status != "ok" {
        let error = answer.error.as_deref().unwrap_or("");
        let scipy_raised_as_expected = match (arm, answer.status.as_str()) {
            ("negative_cycle", status) => status == "negative_cycle",
            (_, "raised") if case.expect_raise => error.starts_with("ValueError"),
            ("min_weight_full_bipartite_matching", "raised") => {
                error == "ValueError: no full matching exists"
            }
            _ => false,
        };
        if !scipy_raised_as_expected {
            ledger.oracle_missing(
                arm,
                &case.id,
                &format!("SciPy status {} ({error})", answer.status),
            );
            failures.push(format!(
                "{}: SciPy status {} ({error}), expected_raise={}",
                case.id, answer.status, case.expect_raise
            ));
            return;
        }
        let refused = match (&ours, arm) {
            (Err(SparseError::NegativeCycle { .. }), "negative_cycle") => true,
            (
                Err(SparseError::InvalidArgument { message }),
                "min_weight_full_bipartite_matching",
            ) => message == "no full matching exists",
            (Err(_), other) => {
                other != "negative_cycle" && other != "min_weight_full_bipartite_matching"
            }
            (Ok(_), _) => false,
        };
        ledger.expected_raise(arm, &case.id, refused);
        if !refused {
            failures.push(format!(
                "{}: SciPy raised ({error}), fsci gave {ours:?}",
                case.id
            ));
        }
        return;
    }
    match &ours {
        Err(e) => {
            ledger.rust_failed(arm, &case.id, &e.to_string());
            failures.push(format!(
                "{}: fsci refused ({e}) where SciPy answered",
                case.id
            ));
        }
        Ok(result) => match matches_scipy(result, answer) {
            None => {
                ledger.oracle_missing(arm, &case.id, "the answer lacks the compared field");
                failures.push(format!("{}: incomplete oracle answer {answer:?}", case.id));
            }
            Some(pass) => {
                ledger.compared(arm, &case.id, pass);
                if !pass {
                    failures.push(format!("{}: fsci {result:?}\n  SciPy {answer:?}", case.id));
                }
            }
        },
    }
}

/// The must-miss arm of the comparators: a result that differs from SciPy's by one ULP, by one
/// moved index, or by NaN against a number must fail, and NaN against NaN must pass. Without
/// this an exact comparison that silently matched everything would read as parity.
fn assert_comparators_see_differences() {
    let m = Graph::new(2, 3, &[(0, 2, 1.5), (1, 0, f64::NAN)]).csr();
    let parts = |indices: [usize; 2], first: f64| CsrParts {
        indptr: vec![0, 1, 2],
        indices: indices.to_vec(),
        data: vec![text(first), text(f64::NAN)],
    };
    assert!(same_csr(&m, &parts([2, 0], 1.5)), "must-hit: identical CSR");
    assert!(!same_csr(&m, &parts([2, 0], 1.5f64.next_up())), "1 ULP");
    assert!(!same_csr(&m, &parts([1, 0], 1.5)), "moved index");
    assert!(
        !same_csr(&m, &parts([2, 0], f64::NAN)),
        "NaN against a number"
    );
    let dense = vec![vec![0.25, INF]];
    assert!(
        same_dense(&dense, &texts(&dense)),
        "must-hit: identical dense"
    );
    assert!(
        !same_dense(&dense, &texts(&[vec![0.25f64.next_down(), INF]])),
        "1 ULP"
    );
    assert!(
        !same_dense(&dense, &texts(&[vec![0.25, -INF]])),
        "sign of infinity"
    );
}

/// A depth-first Ford–Fulkerson over the dense residual graph, neighbours in increasing index:
/// a correct maximum flow that is not, in general, the one SciPy returns.
fn dfs_ford_fulkerson(cap: &[Vec<i64>], s: usize, t: usize) -> Vec<Vec<i64>> {
    let n = cap.len();
    let mut flow = vec![vec![0i64; n]; n];
    let neighbours: Vec<Vec<usize>> = (0..n)
        .map(|i| {
            (0..n)
                .filter(|&j| cap[i][j] != 0 || cap[j][i] != 0)
                .collect()
        })
        .collect();
    loop {
        let mut pred = vec![usize::MAX; n];
        let mut seen = vec![false; n];
        let mut stack = vec![s];
        seen[s] = true;
        while let Some(u) = stack.pop() {
            if u == t {
                break;
            }
            for &v in neighbours[u].iter().rev() {
                if !seen[v] && cap[u][v] - flow[u][v] > 0 {
                    seen[v] = true;
                    pred[v] = u;
                    stack.push(v);
                }
            }
        }
        if !seen[t] {
            return flow;
        }
        let mut df = i64::MAX;
        let mut v = t;
        while v != s {
            let u = pred[v];
            df = df.min(cap[u][v] - flow[u][v]);
            v = u;
        }
        let mut v = t;
        while v != s {
            let u = pred[v];
            flow[u][v] += df;
            flow[v][u] -= df;
            v = u;
        }
    }
}

fn next_permutation(p: &mut [usize]) -> bool {
    let n = p.len();
    if n < 2 {
        return false;
    }
    let mut i = n - 1;
    while i > 0 && p[i - 1] >= p[i] {
        i -= 1;
    }
    if i == 0 {
        return false;
    }
    let mut j = n - 1;
    while p[j] <= p[i - 1] {
        j -= 1;
    }
    p.swap(i - 1, j);
    p[i..].reverse();
    true
}

/// The must-hit counts: cases where SciPy's answer differs from a plausible wrong one.
#[derive(Debug, Default)]
struct Discrimination {
    flow_differs_from_dfs: usize,
    matching_differs_from_greedy: usize,
    assignment_not_lexicographic_first: usize,
    yen_equal_lengths: usize,
    isclose_masked_nonequal: usize,
    tree_rows_unsorted_in_scipy: usize,
}

#[allow(clippy::too_many_lines)]
fn discrimination(cases: &[Case], answers: &HashMap<&str, &Answer>) -> Discrimination {
    let mut d = Discrimination::default();
    for case in cases {
        let answer = answers[case.id.as_str()];
        if answer.status != "ok" || case.expect_raise {
            continue;
        }
        match case.func {
            "maximum_flow" => {
                let graph = case.graph.as_ref().expect("graph");
                let entries = graph.entries();
                let n = graph.rows;
                let mut pairs: Vec<(usize, usize)> = entries.iter().map(|e| (e.0, e.1)).collect();
                pairs.sort_unstable();
                pairs.dedup();
                if pairs.len() != entries.len() {
                    continue; // duplicated edges have no dense form
                }
                let mut cap = vec![vec![0i64; n]; n];
                for &(r, c, w) in &entries {
                    cap[r][c] = w as i64;
                }
                let dfs = dfs_ford_fulkerson(&cap, case.source, case.sink);
                let dfs_value: i64 = dfs[case.source].iter().sum();
                assert_eq!(
                    Some(dfs_value),
                    answer.flow_value,
                    "{}: the depth-first detector is not a maximum flow",
                    case.id
                );
                let parts = answer.csr.as_ref().expect("flow csr");
                let mut scipy = vec![vec![0i64; n]; n];
                for r in 0..n {
                    for e in parts.indptr[r]..parts.indptr[r + 1] {
                        scipy[r][parts.indices[e]] += num(&parts.data[e]) as i64;
                    }
                }
                d.flow_differs_from_dfs += usize::from(scipy != dfs);
            }
            "maximum_bipartite_matching" if case.perm == "column" => {
                let graph = case.graph.as_ref().expect("graph");
                let mut used = vec![false; graph.cols];
                let greedy: Vec<i64> = (0..graph.rows)
                    .map(|r| {
                        let free = graph.indices[graph.indptr[r]..graph.indptr[r + 1]]
                            .iter()
                            .copied()
                            .find(|&c| !used[c]);
                        free.map_or(-1, |c| {
                            used[c] = true;
                            c as i64
                        })
                    })
                    .collect();
                d.matching_differs_from_greedy +=
                    usize::from(answer.perm.as_ref() != Some(&greedy));
            }
            "min_weight_full_bipartite_matching" => {
                let graph = case.graph.as_ref().expect("graph");
                let n = graph.rows;
                if graph.cols != n || n > 7 {
                    continue;
                }
                let mut weight: Vec<Vec<Option<f64>>> = vec![vec![None; n]; n];
                let mut has_nan = false;
                for (r, c, w) in graph.entries() {
                    let w = if case.maximize { -w } else { w };
                    has_nan |= w.is_nan();
                    if w != 0.0 && w != INF {
                        weight[r][c] = Some(w);
                    }
                }
                if has_nan {
                    continue;
                }
                let total = |perm: &[usize]| -> Option<f64> {
                    perm.iter()
                        .enumerate()
                        .map(|(r, &c)| weight[r][c])
                        .sum::<Option<f64>>()
                };
                let mut perm: Vec<usize> = (0..n).collect();
                let mut best: Option<(f64, Vec<usize>)> = None;
                loop {
                    if let Some(t) = total(&perm)
                        && best.as_ref().is_none_or(|(b, _)| t < *b)
                    {
                        best = Some((t, perm.clone()));
                    }
                    if !next_permutation(&mut perm) {
                        break;
                    }
                }
                let (best_total, first) = best.expect("a full matching exists");
                let cols = answer.cols.as_ref().expect("cols");
                assert_eq!(
                    total(cols),
                    Some(best_total),
                    "{}: SciPy's assignment is not optimal by brute force",
                    case.id
                );
                d.assignment_not_lexicographic_first += usize::from(*cols != first);
            }
            "yen" => {
                let dist = answer.dist.as_ref().expect("dist");
                d.yen_equal_lengths += usize::from(dist.windows(2).any(|w| w[0] == w[1]));
            }
            "csgraph_masked_from_dense" => {
                let Some(null) = case
                    .null_value
                    .as_deref()
                    .map(num)
                    .filter(|v| v.is_finite())
                else {
                    continue;
                };
                let mask = answer.mask.as_ref().expect("mask");
                let hit = floats(&case.dense).iter().zip(mask).any(|(row, m)| {
                    row.iter()
                        .zip(m)
                        .any(|(&x, &masked)| masked && x.is_finite() && x != null)
                });
                d.isclose_masked_nonequal += usize::from(hit);
            }
            "breadth_first_tree" | "depth_first_tree" | "reconstruct_path" => {
                d.tree_rows_unsorted_in_scipy += usize::from(answer.raw_sorted == Some(false));
            }
            _ => {}
        }
    }
    d
}

#[test]
fn diff_sparse_csgraph_flow_matching_paths() {
    assert_comparators_see_differences();
    let mut rng = Rng(0x0C5F_DE9A_2026_0928);
    let mut cases = Vec::new();
    cases.extend(flow_cases(&mut rng));
    cases.extend(matching_cases(&mut rng));
    cases.extend(assignment_cases(&mut rng));
    cases.extend(tree_cases(&mut rng));
    cases.extend(reconstruct_cases(&mut rng));
    cases.extend(dist_matrix_cases(&mut rng));
    cases.extend(dense_cases(&mut rng));
    cases.extend(yen_cases(&mut rng));
    cases.extend(negative_cycle_cases(&mut rng));
    let mut ids: Vec<&str> = cases.iter().map(|c| c.id.as_str()).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), cases.len(), "case ids must be unique");

    let Some(answers) = scipy_answers(&cases) else {
        return;
    };
    assert_eq!(
        answers.len(),
        cases.len(),
        "the oracle must answer every case"
    );
    let by_id: HashMap<&str, &Answer> = answers.iter().map(|a| (a.id.as_str(), a)).collect();

    let mut ledger = CompareLedger::new("diff_sparse_csgraph_flow_matching_paths", &ARMS);
    let mut failures = Vec::new();
    for case in &cases {
        check(case, by_id[case.id.as_str()], &mut ledger, &mut failures);
    }
    let d = discrimination(&cases, &by_id);
    println!("{} cases; discrimination {d:?}", cases.len());
    for (arm, counts) in ledger.counts() {
        println!(
            "{arm}: compared {} failed {} rust_failed {} oracle_missing {}",
            counts.compared_cases,
            counts.compared_failed,
            counts.rust_failed,
            counts.oracle_missing
        );
    }
    assert!(
        failures.is_empty(),
        "csgraph flow/matching/path routines disagree with SciPy: {failures:#?}"
    );
    assert!(
        d.flow_differs_from_dfs > 0,
        "no flow case separates SciPy's maximum flow from a depth-first one"
    );
    assert!(
        d.matching_differs_from_greedy > 0,
        "no matching case separates Hopcroft–Karp from a greedy matching"
    );
    assert!(
        d.assignment_not_lexicographic_first > 0,
        "no assignment case has a tie SciPy breaks away from the lexicographically first optimum"
    );
    assert!(
        d.yen_equal_lengths > 0,
        "no Yen case has equally long paths"
    );
    assert!(
        d.isclose_masked_nonequal > 0,
        "no dense case masks a value that is only np.isclose to the null value"
    );
    ledger.finish(MIN_PER_ARM);
}
