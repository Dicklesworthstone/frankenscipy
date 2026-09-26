#![forbid(unsafe_code)]
//! Property-based parity for fsci_sparse::pagerank.
//!
//! Resolves [frankenscipy-15d6n]. Verifies:
//!   1. Output sums to 1.0 (probability distribution)
//!   2. All values finite and non-negative
//!   3. For symmetric graphs, larger-degree nodes have at least as high a
//!      rank as smaller-degree nodes within the same connected component.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_sparse::{CsrMatrix, Shape2D, pagerank};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-6;
/// The one fixture the hub-rank property is designed for.
const HUB_FIXTURE: &str = "undirected_6n_hub";
/// One ledger arm per invariant. `sums_to_one` compares the rank sum against the analytic 1.0
/// through `pair`; the others are boolean properties. `hub_outranks_leaves` checks only
/// `HUB_FIXTURE`; every other arm checks every fixture.
const ARMS: [&str; 5] = [
    "sums_to_one",
    "all_finite",
    "all_nonneg",
    "len_eq_n",
    "hub_outranks_leaves",
];

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    sum: f64,
    min_val: f64,
    pass: bool,
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
    fs::create_dir_all(output_dir()).expect("create pagerank diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize pagerank diff log");
    fs::write(path, json).expect("write pagerank diff log");
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

#[test]
fn diff_sparse_pagerank_properties() {
    let start = Instant::now();
    let mut diffs = Vec::new();

    let fixtures: Vec<(&str, Vec<f64>, usize)> = vec![
        (
            "directed_4n",
            vec![
                0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
            ],
            4,
        ),
        (
            "undirected_5n_cycle",
            vec![
                0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
            ],
            5,
        ),
        (
            HUB_FIXTURE,
            // node 0 connects to all others
            vec![
                0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
            ],
            6,
        ),
    ];

    // Every arm but the hub arm checks every fixture; the hub arm checks only HUB_FIXTURE, so
    // it has the smallest designed case count.
    let hub_cases = fixtures
        .iter()
        .filter(|(label, _, _)| *label == HUB_FIXTURE)
        .count();
    let mut ledger = CompareLedger::new("diff_sparse_pagerank_properties", &ARMS);

    for (label, adj, n) in fixtures {
        let csr = dense_to_csr(n, n, &adj);
        let ranks = pagerank(&csr, 0.85, 200, 1e-9);

        let sum: f64 = ranks.iter().sum();
        let min_val = ranks.iter().copied().fold(f64::INFINITY, f64::min);
        let all_finite = ranks.iter().all(|v| v.is_finite());
        let all_nonneg = ranks.iter().all(|&v| v >= -ABS_TOL);
        let sums_to_one = (sum - 1.0).abs() <= ABS_TOL;
        let len_ok = ranks.len() == n;
        let pass = all_finite && all_nonneg && sums_to_one && len_ok;

        // The analytic reference is 1.0; pair records a non-finite sum as an fsci failure.
        if ledger
            .pair("sums_to_one", label, Some(1.0), Some(sum))
            .is_some()
        {
            ledger.compared("sums_to_one", label, sums_to_one);
        }
        ledger.compared("all_finite", label, all_finite);
        ledger.compared("all_nonneg", label, all_nonneg);
        ledger.compared("len_eq_n", label, len_ok);

        // For "undirected_6n_hub": node 0 has degree 5; nodes 1-5 each have degree 1.
        // So rank(0) > rank(any other).
        let mut extra_ok = true;
        if label == HUB_FIXTURE {
            if pass {
                let hub_rank = ranks[0];
                let other_max = ranks[1..].iter().copied().fold(f64::NEG_INFINITY, f64::max);
                extra_ok = hub_rank > other_max;
                ledger.compared("hub_outranks_leaves", label, extra_ok);
            } else {
                // The hub comparison is only defined on a rank vector that passed the base
                // invariants; recording it keeps the arm from reading as merely uncompared.
                ledger.rust_failed(
                    "hub_outranks_leaves",
                    label,
                    "rank vector failed the base invariants",
                );
            }
        }

        diffs.push(CaseDiff {
            case_id: label.into(),
            sum,
            min_val,
            pass: pass && extra_ok,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_sparse_pagerank_properties".into(),
        category: "fsci_sparse::pagerank invariants".into(),
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
            eprintln!(
                "pagerank fail: {} sum={} min={}",
                d.case_id, d.sum, d.min_val
            );
        }
    }

    assert!(
        all_pass,
        "pagerank conformance failed: {} cases",
        diffs.len()
    );
    // The smallest designed count: the hub arm's one fixture. The per-fixture arms record every
    // fixture (sums_to_one through pair with an always-present reference), so a fixture they did
    // not compare is a failure, not a gap.
    ledger.finish(hub_cases);
}
