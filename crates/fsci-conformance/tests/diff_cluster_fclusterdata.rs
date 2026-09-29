#![forbid(unsafe_code)]
//! Live scipy differential coverage for fsci's
//! `fclusterdata(data, FclusterCriterion::MaxClust(k), method)` — chained
//! linkage + fcluster from raw data, equivalent to
//! `scipy.cluster.hierarchy.fclusterdata(X, t, criterion='maxclust',
//! method=method)`.
//!
//! Resolves [frankenscipy-i2fyz]. Existing `diff_cluster_fcluster`
//! covers `fcluster` against scipy's linkage matrix; this harness
//! covers the full data → labels pipeline.
//!
//! The labels themselves are compared, not only the partition: SciPy
//! numbers flat clusters in the order of its depth-first walk from the
//! root, and fsci ports that walk (frankenscipy-3h1yw). The evenly
//! spaced chain has every single-linkage merge at the same height;
//! SciPy's maxclust then keeps the whole tie, forming fewer than k
//! clusters, and fsci must do the same.
//!
//! 4 fixtures × 3 methods (Single/Complete/Average) = 12 cases.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_cluster::{FclusterCriterion::MaxClust, LinkageMethod, fclusterdata};
use fsci_conformance::{ArmCounts, CompareLedger};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-012";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    method: String,
    max_clusters: usize,
    data: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    labels: Option<Vec<i64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    method: String,
    labels_match: bool,
    coassign_match: bool,
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
    fs::create_dir_all(output_dir()).expect("create fclusterdata diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize fclusterdata diff log");
    fs::write(path, json).expect("write fclusterdata diff log");
}

fn method_for(name: &str) -> Option<LinkageMethod> {
    match name {
        "single" => Some(LinkageMethod::Single),
        "complete" => Some(LinkageMethod::Complete),
        "average" => Some(LinkageMethod::Average),
        _ => None,
    }
}

fn coassign_matrix(labels: &[i64]) -> Vec<Vec<bool>> {
    let n = labels.len();
    let mut m = vec![vec![false; n]; n];
    for i in 0..n {
        for j in 0..n {
            m[i][j] = labels[i] == labels[j];
        }
    }
    m
}

fn generate_query() -> OracleQuery {
    // Three well-separated point clouds in 2-D so the partition
    // is unambiguous, plus carefully chosen offsets so all
    // pairwise Euclidean distances are distinct.
    let two_clusters: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0],
        vec![0.1, 0.2],
        vec![0.3, 0.05],
        vec![10.0, 10.0],
        vec![10.2, 10.15],
        vec![10.4, 9.95],
    ];
    let three_clusters: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0],
        vec![0.1, 0.2],
        vec![0.25, 0.05],
        vec![5.0, 5.0],
        vec![5.15, 5.2],
        vec![5.3, 4.95],
        vec![10.0, 0.0],
        vec![10.2, 0.15],
        vec![10.35, -0.05],
    ];
    let four_clusters: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0],
        vec![0.1, 0.15],
        vec![0.25, 0.05],
        vec![5.0, 0.0],
        vec![5.15, 0.2],
        vec![5.3, -0.05],
        vec![0.0, 5.0],
        vec![0.15, 5.2],
        vec![5.0, 5.0],
        vec![5.2, 5.15],
    ];

    // Unit gaps: every single-linkage merge ties at height 1.
    let tied_chain: Vec<Vec<f64>> = (0..8).map(|i| vec![f64::from(i), 0.0]).collect();

    let fixtures: Vec<(&str, usize, Vec<Vec<f64>>)> = vec![
        ("two_clusters_n6_k2", 2, two_clusters),
        ("three_clusters_n9_k3", 3, three_clusters),
        ("four_clusters_n10_k4", 4, four_clusters),
        ("tied_chain_n8_k3", 3, tied_chain),
    ];
    let methods = ["single", "complete", "average"];
    let mut points = Vec::new();
    for (name, k, data) in &fixtures {
        for m in &methods {
            points.push(PointCase {
                case_id: format!("{name}_{m}"),
                method: (*m).to_string(),
                max_clusters: *k,
                data: data.clone(),
            });
        }
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import sys
import numpy as np
from scipy.cluster.hierarchy import fclusterdata

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    labels = None
    try:
        X = np.asarray(case["data"], dtype=np.float64)
        L = fclusterdata(
            X, t=case["max_clusters"], criterion="maxclust", method=case["method"]
        )
        labels = [int(v) for v in L.tolist()]
    except Exception as e:
        labels = None
    points.append({"case_id": cid, "labels": labels})
print(json.dumps({"points": points}, allow_nan=False))
"#;
    let query_json = serde_json::to_string(query).expect("serialize fclusterdata query");
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
                "failed to spawn python3 for fclusterdata oracle: {e}"
            );
            eprintln!("skipping fclusterdata oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open fclusterdata oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "fclusterdata oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping fclusterdata oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for fclusterdata oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "fclusterdata oracle failed: {stderr}"
        );
        eprintln!("skipping fclusterdata oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse fclusterdata oracle JSON"))
}

#[test]
fn diff_cluster_fclusterdata() {
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
    let mut ledger = CompareLedger::new(
        "diff_cluster_fclusterdata",
        &["single", "complete", "average"],
    );

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let method = method_for(&case.method).expect("generated linkage method");
        let rust_labels = fclusterdata(&case.data, MaxClust(case.max_clusters), method)
            .ok()
            .map(|v| v.into_iter().map(|x| x as i64).collect::<Vec<_>>());
        let Some((scipy_labels, rust_labels)) = ledger.both(
            &case.method,
            &case.case_id,
            scipy_arm.labels.as_ref(),
            rust_labels,
        ) else {
            continue;
        };

        let coassign = if rust_labels.len() == scipy_labels.len() {
            coassign_matrix(&rust_labels) == coassign_matrix(scipy_labels)
        } else {
            false
        };
        // Both sides are 1-based. The co-assignment check stays as a diagnostic: it tells a
        // numbering difference from a different partition.
        let labels_match = &rust_labels == scipy_labels;
        ledger.compared(&case.method, &case.case_id, labels_match);

        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            method: case.method.clone(),
            labels_match,
            coassign_match: coassign,
            pass: labels_match,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_cluster_fclusterdata".into(),
        category: "fsci_cluster::fclusterdata".into(),
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
                "fclusterdata mismatch: {} ({}) labels_match={} coassign_match={}",
                d.case_id, d.method, d.labels_match, d.coassign_match
            );
        }
    }

    assert!(
        all_pass,
        "fclusterdata conformance failed across {} cases",
        diffs.len()
    );
    ledger.finish(query.points.iter().filter(|c| c.method == "single").count());
}
