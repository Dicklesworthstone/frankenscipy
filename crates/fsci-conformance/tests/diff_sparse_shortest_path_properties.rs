#![forbid(unsafe_code)]
//! Property-based coverage for fsci_sparse::shortest_path.
//!
//! Resolves [frankenscipy-n4nuc]. Single-source single-target
//! Dijkstra over a CSR-stored weighted graph. Properties:
//!   * dist(source, source) == 0; path == [source]
//!   * Known triangle: weights chosen so path is unique
//!   * Disconnected → (INFINITY, [])
//!   * Out-of-bounds source/target → (INFINITY, [])
//!   * Reconstructed path is a valid edge sequence (each step has
//!     an edge in the CSR adjacency)
//!   * Sum of edge weights along path == reported distance

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_sparse::{CooMatrix, CsrMatrix, FormatConvertible, Shape2D, shortest_path};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    pass: bool,
    note: String,
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
    fs::create_dir_all(output_dir()).expect("create shortest_path diff dir");
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

fn build_csr(n: usize, trips: &[(usize, usize, f64)]) -> CsrMatrix {
    let data: Vec<f64> = trips.iter().map(|t| t.2).collect();
    let rs: Vec<usize> = trips.iter().map(|t| t.0).collect();
    let cs: Vec<usize> = trips.iter().map(|t| t.1).collect();
    let coo = CooMatrix::from_triplets(Shape2D::new(n, n), data, rs, cs, true).unwrap();
    coo.to_csr().unwrap()
}

/// Compute edge weight from u → v in the graph. Returns None if no edge.
fn edge_weight(g: &CsrMatrix, u: usize, v: usize) -> Option<f64> {
    let indptr = g.indptr();
    let indices = g.indices();
    let data = g.data();
    let start = indptr[u];
    let end = indptr[u + 1];
    for idx in start..end {
        if indices[idx] == v {
            return Some(data[idx]);
        }
    }
    None
}

#[test]
fn diff_sparse_shortest_path_properties() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut check = |id: &str, ok: bool, note: String| {
        diffs.push(CaseDiff {
            case_id: id.into(),
            pass: ok,
            note,
        });
    };

    // === Graph 1: 5-node directed weighted ===
    //   0 → 1 (1.0)
    //   0 → 2 (4.0)
    //   1 → 2 (2.0)
    //   1 → 3 (5.0)
    //   2 → 3 (1.0)
    //   3 → 4 (3.0)
    // Shortest 0 → 4 via 0→1→2→3→4: 1+2+1+3 = 7
    let g1 = build_csr(
        5,
        &[
            (0, 1, 1.0),
            (0, 2, 4.0),
            (1, 2, 2.0),
            (1, 3, 5.0),
            (2, 3, 1.0),
            (3, 4, 3.0),
        ],
    );

    {
        let (d, p) = shortest_path(&g1, true, 0, 4);
        check(
            "g1_0_to_4_distance",
            (d - 7.0).abs() < ABS_TOL,
            format!("d={d} p={p:?}"),
        );
        check(
            "g1_0_to_4_path_starts_at_source",
            p.first() == Some(&0),
            format!("p={p:?}"),
        );
        check(
            "g1_0_to_4_path_ends_at_target",
            p.last() == Some(&4),
            format!("p={p:?}"),
        );
        // Each consecutive pair in path must have a valid edge
        let mut edges_valid = true;
        let mut path_sum = 0.0;
        for w in p.windows(2) {
            match edge_weight(&g1, w[0], w[1]) {
                Some(weight) => path_sum += weight,
                None => {
                    edges_valid = false;
                    break;
                }
            }
        }
        check("g1_path_edges_valid", edges_valid, format!("p={p:?}"));
        check(
            "g1_path_sum_equals_distance",
            (path_sum - d).abs() < ABS_TOL,
            format!("path_sum={path_sum} d={d}"),
        );
    }

    // === Source == target → distance 0 ===
    {
        let (d, p) = shortest_path(&g1, true, 2, 2);
        check(
            "self_distance_zero",
            d == 0.0 && p == vec![2],
            format!("d={d} p={p:?}"),
        );
    }

    // === Disconnected: no edge from 4 back to 0 in directed graph ===
    {
        let (d, p) = shortest_path(&g1, true, 4, 0);
        check(
            "disconnected_returns_infinity",
            d == f64::INFINITY && p.is_empty(),
            format!("d={d} p={p:?}"),
        );
    }

    // === Out-of-bounds source ===
    {
        let (d, p) = shortest_path(&g1, true, 99, 0);
        check(
            "oob_source_returns_infinity",
            d == f64::INFINITY && p.is_empty(),
            format!("d={d} p={p:?}"),
        );
    }

    // === Out-of-bounds target ===
    {
        let (d, p) = shortest_path(&g1, true, 0, 99);
        check(
            "oob_target_returns_infinity",
            d == f64::INFINITY && p.is_empty(),
            format!("d={d} p={p:?}"),
        );
    }

    // === Direct edge: shortest path is just the edge ===
    {
        let g2 = build_csr(3, &[(0, 1, 5.0), (1, 2, 7.0), (0, 2, 100.0)]);
        // Direct edge 0→2 weight 100; via 1 it's 5+7=12. So path via 1 wins.
        let (d, p) = shortest_path(&g2, true, 0, 2);
        check(
            "indirect_path_wins",
            (d - 12.0).abs() < ABS_TOL && p == vec![0, 1, 2],
            format!("d={d} p={p:?}"),
        );
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_sparse_shortest_path_properties".into(),
        category: "fsci_sparse::shortest_path property-based coverage".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("shortest_path mismatch: {} — {}", d.case_id, d.note);
        }
    }

    assert!(
        all_pass,
        "shortest_path coverage failed: {} cases",
        diffs.len()
    );
}
