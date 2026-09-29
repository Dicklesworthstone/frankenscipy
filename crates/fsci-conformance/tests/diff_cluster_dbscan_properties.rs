#![forbid(unsafe_code)]
//! Property-based coverage for fsci_cluster::dbscan.
//!
//! Resolves [frankenscipy-vq8mj]. sklearn is not available in the
//! conformance environment, but DBSCAN has clean mathematical
//! invariants that pin down the algorithm completely:
//!   * Two well-separated clusters of ≥ min_samples points each → all
//!     pairs within the same cluster share the same label
//!   * A point further than eps from every other point → noise (-1)
//!   * core_sample_indices includes every point whose ε-neighborhood
//!     has ≥ min_samples points
//!   * Permutation invariance: shuffling input rows produces the same
//!     partition modulo a remap of cluster IDs
//!   * Error paths: empty data, non-finite eps, min_samples=0

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_cluster::dbscan;
use fsci_conformance::{ArmCounts, CompareLedger};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
/// One ledger arm per property check, each a single fixed case recorded under its own name.
/// Every check is a boolean invariant of fsci's own output (integer labels and counts, so no
/// value can be NaN) or a fail-closed contract (`*_errors`, the call must refuse); there is no
/// SciPy side, the reference kind is analytic.
const ARMS: [&str; 15] = [
    "two_clusters_count",
    "two_clusters_first_group_same_label",
    "two_clusters_last_group_same_label",
    "two_clusters_distinct_labels",
    "two_clusters_no_noise",
    "noise_isolated_point_labeled_minus_1",
    "noise_cluster_count_one",
    "core_sample_indices_all_five",
    "core_sample_indices_sorted",
    "permutation_invariance_modulo_label_remap",
    "empty_data_errors",
    "zero_eps_errors",
    "nan_eps_errors",
    "zero_min_samples_errors",
    "non_finite_data_errors",
];

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
    fs::create_dir_all(output_dir()).expect("create dbscan diff dir");
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

/// Canonicalize labels by remapping cluster IDs to the order in which
/// they first appear (noise -1 stays -1). Useful for permutation-
/// invariance checks since DBSCAN's cluster numbering is order-dependent.
fn canonicalize(labels: &[i64]) -> Vec<i64> {
    let mut map: HashMap<i64, i64> = HashMap::new();
    let mut next_id = 0_i64;
    labels
        .iter()
        .map(|&l| {
            if l == -1 {
                -1
            } else {
                *map.entry(l).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                })
            }
        })
        .collect()
}

#[test]
fn diff_cluster_dbscan_properties() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut ledger = CompareLedger::new("diff_cluster_dbscan_properties", &ARMS);
    // Every check id is its own arm and its own case id; an undeclared id panics in the ledger.
    let mut check = |id: &str, ok: bool, note: String| {
        ledger.compared(id, id, ok);
        diffs.push(CaseDiff {
            case_id: id.into(),
            pass: ok,
            note,
        });
    };

    // === 1. Two well-separated clusters ===
    {
        let data: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![0.0, 0.2],
            vec![0.2, 0.0],
            vec![10.0, 10.0],
            vec![10.1, 10.0],
            vec![10.0, 10.1],
            vec![10.05, 10.05],
        ];
        let r = dbscan(&data, 0.5, 2).expect("dbscan");
        check(
            "two_clusters_count",
            r.n_clusters == 2,
            format!("n_clusters={} labels={:?}", r.n_clusters, r.labels),
        );
        // First 4 points share a label, last 4 share a label, no -1
        let first_labels = &r.labels[..4];
        let last_labels = &r.labels[4..];
        let all_first_same =
            first_labels.iter().all(|&l| l == first_labels[0]) && first_labels[0] >= 0;
        let all_last_same = last_labels.iter().all(|&l| l == last_labels[0]) && last_labels[0] >= 0;
        check(
            "two_clusters_first_group_same_label",
            all_first_same,
            format!("first={first_labels:?}"),
        );
        check(
            "two_clusters_last_group_same_label",
            all_last_same,
            format!("last={last_labels:?}"),
        );
        check(
            "two_clusters_distinct_labels",
            first_labels[0] != last_labels[0],
            format!("first={} last={}", first_labels[0], last_labels[0]),
        );
        check(
            "two_clusters_no_noise",
            r.labels.iter().all(|&l| l >= 0),
            format!("labels={:?}", r.labels),
        );
    }

    // === 2. Noise: lone point far from cluster ===
    {
        let data: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![0.0, 0.2],
            vec![100.0, 100.0], // isolated
        ];
        let r = dbscan(&data, 0.5, 2).expect("dbscan noise");
        check(
            "noise_isolated_point_labeled_minus_1",
            r.labels[3] == -1,
            format!("labels={:?}", r.labels),
        );
        check(
            "noise_cluster_count_one",
            r.n_clusters == 1,
            format!("n_clusters={}", r.n_clusters),
        );
    }

    // === 3. core_sample_indices: every point in a dense neighborhood is core ===
    {
        let data: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![0.2, 0.0],
            vec![0.05, 0.15],
            vec![0.1, 0.2],
        ];
        // eps=0.5 covers the cluster; min_samples=2 → every point has ≥2 neighbors
        let r = dbscan(&data, 0.5, 2).expect("dbscan core");
        check(
            "core_sample_indices_all_five",
            r.core_sample_indices.len() == 5,
            format!("core_idx={:?}", r.core_sample_indices),
        );
        check(
            "core_sample_indices_sorted",
            r.core_sample_indices.windows(2).all(|w| w[0] < w[1]),
            format!("core_idx={:?}", r.core_sample_indices),
        );
    }

    // === 4. Permutation invariance (modulo label remap) ===
    {
        let data: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![5.0, 5.0],
            vec![5.1, 5.0],
            vec![5.0, 5.1],
            vec![0.0, 0.2],
        ];
        let original = dbscan(&data, 0.5, 2).expect("original");
        // Reverse the input
        let reversed: Vec<Vec<f64>> = data.iter().rev().cloned().collect();
        let rev_labels = dbscan(&reversed, 0.5, 2).expect("reversed").labels;
        // Reverse rev_labels to match original ordering
        let rev_aligned: Vec<i64> = rev_labels.into_iter().rev().collect();
        check(
            "permutation_invariance_modulo_label_remap",
            canonicalize(&original.labels) == canonicalize(&rev_aligned),
            format!(
                "orig={:?} rev_aligned={:?}",
                canonicalize(&original.labels),
                canonicalize(&rev_aligned)
            ),
        );
    }

    // === 5. Error: empty data ===
    {
        let r = dbscan(&[], 0.5, 2);
        check("empty_data_errors", r.is_err(), format!("res={r:?}"));
    }

    // === 6. Error: eps == 0 ===
    {
        let data: Vec<Vec<f64>> = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let r = dbscan(&data, 0.0, 2);
        check("zero_eps_errors", r.is_err(), format!("res={r:?}"));
    }

    // === 7. Error: eps non-finite ===
    {
        let data: Vec<Vec<f64>> = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let r = dbscan(&data, f64::NAN, 2);
        check("nan_eps_errors", r.is_err(), format!("res={r:?}"));
    }

    // === 8. Error: min_samples == 0 ===
    {
        let data: Vec<Vec<f64>> = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let r = dbscan(&data, 0.5, 0);
        check("zero_min_samples_errors", r.is_err(), format!("res={r:?}"));
    }

    // === 9. Error: non-finite input ===
    {
        let data: Vec<Vec<f64>> = vec![vec![0.0, 0.0], vec![f64::NAN, 1.0]];
        let r = dbscan(&data, 0.5, 2);
        check("non_finite_data_errors", r.is_err(), format!("res={r:?}"));
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_cluster_dbscan_properties".into(),
        category: "fsci_cluster::dbscan property-based coverage".into(),
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
            eprintln!("dbscan mismatch: {} — {}", d.case_id, d.note);
        }
    }

    assert!(
        all_pass,
        "dbscan property coverage failed: {} cases",
        diffs.len()
    );
    // Every arm is one fixed case, so each must have compared it.
    ledger.finish(1);
}
