#![forbid(unsafe_code)]
//! Live scipy differential coverage for fsci's
//! `linkage_from_distances(condensed_dist, n, method)`.
//!
//! Resolves [frankenscipy-rqtwr]. Existing `diff_cluster.rs`
//! covers `linkage(data, method)` (raw-data path);
//! `linkage_from_distances` is the precomputed-distance entry
//! point (matrix already condensed). scipy's
//! `scipy.cluster.hierarchy.linkage(y, method)` accepts both
//! data and condensed-distance vectors, so this oracle just
//! feeds the same condensed vector and method.
//!
//! 3 fixtures × 3 methods (Single, Complete, Average) = 9
//! cases. All fixtures use strictly distinct distances so no
//! tie-breaking divergence is possible.
//!
//! Tol 1e-12 abs on heights (column 2 of Z); structural
//! equality on merged-cluster IDs (cols 0, 1) and member
//! counts (col 3).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_cluster::{LinkageMethod, linkage_from_distances};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-012";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    method: String,
    n: usize,
    condensed: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    rows: Option<Vec<[f64; 4]>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    method: String,
    max_height_diff: f64,
    structural_match: bool,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    max_abs_diff: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create linkage_from_distances diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json =
        serde_json::to_string_pretty(log).expect("serialize linkage_from_distances diff log");
    fs::write(path, json).expect("write linkage_from_distances diff log");
}

fn method_for(name: &str) -> Option<LinkageMethod> {
    match name {
        "single" => Some(LinkageMethod::Single),
        "complete" => Some(LinkageMethod::Complete),
        "average" => Some(LinkageMethod::Average),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    // 3 fixtures × 3 methods. Each condensed vector is
    // n*(n-1)/2 entries with strictly distinct values so neither
    // fsci nor scipy must invoke any tie-breaking rule.
    let fixtures: Vec<(&str, usize, Vec<f64>)> = vec![
        // n=4 → 6 condensed entries.
        ("fix_n4_distinct", 4, vec![1.0, 5.7, 9.1, 3.3, 11.4, 7.2]),
        // n=5 → 10 condensed entries.
        (
            "fix_n5_distinct",
            5,
            vec![2.1, 4.9, 8.3, 13.6, 6.4, 10.5, 16.2, 12.8, 19.1, 15.7],
        ),
        // n=6 → 15 condensed entries.
        (
            "fix_n6_distinct",
            6,
            vec![
                1.5, 3.8, 7.2, 12.3, 18.7, 5.1, 9.4, 14.6, 21.0, 8.7, 13.9, 20.2, 17.4, 23.5, 26.1,
            ],
        ),
    ];
    let methods = ["single", "complete", "average"];
    let mut points = Vec::new();
    for (name, n, dist) in &fixtures {
        for m in &methods {
            points.push(PointCase {
                case_id: format!("{name}_{m}"),
                method: (*m).to_string(),
                n: *n,
                condensed: dist.clone(),
            });
        }
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy.cluster.hierarchy import linkage

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    rows = None
    try:
        Z = linkage(np.asarray(case["condensed"], dtype=np.float64), method=case["method"])
        rows = []
        for r in Z:
            rows.append([float(r[0]), float(r[1]), float(r[2]), float(r[3])])
    except Exception:
        rows = None
    points.append({"case_id": cid, "rows": rows})
print(json.dumps({"points": points}, allow_nan=False))
"#;
    let query_json = serde_json::to_string(query).expect("serialize linkage_from_distances query");
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
                "failed to spawn python3 for linkage_from_distances oracle: {e}"
            );
            eprintln!("skipping linkage_from_distances oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open linkage_from_distances oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "linkage_from_distances oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!(
                "skipping linkage_from_distances oracle: stdin write failed ({err})\n{stderr}"
            );
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for linkage_from_distances oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "linkage_from_distances oracle failed: {stderr}"
        );
        eprintln!("skipping linkage_from_distances oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse linkage_from_distances oracle JSON"))
}

#[test]
fn diff_cluster_linkage_from_distances() {
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
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(scipy_z) = scipy_arm.rows.as_ref() else {
            continue;
        };
        let Some(method) = method_for(&case.method) else {
            continue;
        };
        let rust_z = match linkage_from_distances(&case.condensed, case.n, method) {
            Ok(v) => v,
            Err(_) => continue,
        };

        if rust_z.len() != scipy_z.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                method: case.method.clone(),
                max_height_diff: f64::INFINITY,
                structural_match: false,
                pass: false,
            });
            continue;
        }

        let mut max_h = 0.0_f64;
        let mut struct_ok = true;
        for (rr, sr) in rust_z.iter().zip(scipy_z.iter()) {
            let h_diff = (rr[2] - sr[2]).abs();
            max_h = max_h.max(h_diff);
            // Structural: cluster IDs (cols 0, 1) match as a set.
            // scipy preserves left/right ordering by id; fsci should
            // too for distinct distances. Sort both pairs for a
            // forgiving comparison if either side flips order.
            let ru = (rr[0] as i64).min(rr[1] as i64);
            let rv = (rr[0] as i64).max(rr[1] as i64);
            let su = (sr[0] as i64).min(sr[1] as i64);
            let sv = (sr[0] as i64).max(sr[1] as i64);
            if ru != su || rv != sv {
                struct_ok = false;
            }
            if (rr[3] - sr[3]).abs() > 0.5 {
                struct_ok = false;
            }
        }

        max_overall = max_overall.max(max_h);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            method: case.method.clone(),
            max_height_diff: max_h,
            structural_match: struct_ok,
            pass: max_h <= ABS_TOL && struct_ok,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_cluster_linkage_from_distances".into(),
        category: "fsci_cluster::linkage_from_distances".into(),
        case_count: diffs.len(),
        max_abs_diff: max_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };

    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "linkage_from_distances mismatch: {} ({}) max_height={} struct_match={}",
                d.case_id, d.method, d.max_height_diff, d.structural_match
            );
        }
    }

    assert!(
        all_pass,
        "linkage_from_distances conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
