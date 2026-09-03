#![forbid(unsafe_code)]
//! Property-based parity for fsci_sparse::reverse_cuthill_mckee.
//!
//! Resolves [frankenscipy-ldcck]. Two invariants:
//!   1. Output is a permutation of 0..n (covers all, no duplicates).
//!   2. Bandwidth(P A Pᵀ) ≤ Bandwidth(A) — the point of RCM.
//!
//! Exact permutation may differ from scipy (starting-node heuristics
//! vary), but the bandwidth-reduction property is the contract.

use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_sparse::{CsrMatrix, Shape2D, reverse_cuthill_mckee};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    original_bw: usize,
    perm_bw: usize,
    is_perm: bool,
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
    fs::create_dir_all(output_dir()).expect("create rcm diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize rcm diff log");
    fs::write(path, json).expect("write rcm diff log");
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

/// Bandwidth of a dense adjacency matrix: max |i - j| over nonzeros.
fn bandwidth(rows: usize, cols: usize, dense: &[f64]) -> usize {
    let mut bw = 0_usize;
    for i in 0..rows {
        for j in 0..cols {
            if dense[i * cols + j] != 0.0 {
                bw = bw.max(i.abs_diff(j));
            }
        }
    }
    bw
}

/// Build P A Pᵀ as a dense matrix, where P is given as a row-permutation
/// vector (perm[i] = original-row that goes to row i).
fn permute_matrix(rows: usize, cols: usize, dense: &[f64], perm: &[usize]) -> Vec<f64> {
    let mut out = vec![0.0_f64; rows * cols];
    for new_i in 0..rows {
        for new_j in 0..cols {
            let orig_i = perm[new_i];
            let orig_j = perm[new_j];
            out[new_i * cols + new_j] = dense[orig_i * cols + orig_j];
        }
    }
    out
}

fn fixtures() -> Vec<(&'static str, Vec<f64>, usize)> {
    vec![
        (
            "6n_p1",
            vec![
                0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
                1.0, 0.0, 1.0, 0.0,
            ],
            6,
        ),
        (
            "8n_grid",
            vec![
                0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
            ],
            8,
        ),
        (
            "5n_cycle",
            vec![
                0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
            ],
            5,
        ),
    ]
}

#[test]
fn diff_sparse_reverse_cuthill_mckee_properties() {
    let start = Instant::now();
    let mut diffs = Vec::new();

    for (label, adj, n) in fixtures() {
        let csr = dense_to_csr(n, n, &adj);
        let perm = reverse_cuthill_mckee(&csr);
        // Property 1: perm is a permutation
        let is_perm = perm.len() == n
            && perm.iter().copied().collect::<HashSet<_>>().len() == n
            && perm.iter().all(|&p| p < n);
        // Property 2: bandwidth reduced (or equal)
        let original_bw = bandwidth(n, n, &adj);
        let permuted = if is_perm {
            permute_matrix(n, n, &adj, &perm)
        } else {
            adj.clone()
        };
        let perm_bw = bandwidth(n, n, &permuted);
        let pass = is_perm && perm_bw <= original_bw;
        diffs.push(CaseDiff {
            case_id: label.into(),
            original_bw,
            perm_bw,
            is_perm,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_sparse_reverse_cuthill_mckee_properties".into(),
        category: "fsci_sparse::reverse_cuthill_mckee invariants".into(),
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
                "rcm fail: {} is_perm={} original_bw={} perm_bw={}",
                d.case_id, d.is_perm, d.original_bw, d.perm_bw
            );
        }
    }

    assert!(all_pass, "rcm conformance failed: {} cases", diffs.len());
}
