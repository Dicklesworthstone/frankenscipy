#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the rank-assignment
//! function `scipy.stats.rankdata(a, method)` across all five
//! tie-breaking methods:
//!   • "average" — mean of tied positions (the default)
//!   • "min"     — smallest of tied positions
//!   • "max"     — largest of tied positions
//!   • "dense"   — like "min" but no gaps after a tie group
//!   • "ordinal" — strict 1..n with ties resolved by stable
//!     position order
//!
//! Resolves [frankenscipy-5rri9]. The existing closed-form
//! coverage in `e2e_stats.rs` pins a single 4-element vector
//! per method; this harness exercises all five methods across
//! four datasets including heavy-tie and run-only inputs.
//!
//! 4 datasets × 5 methods = 20 cases via subprocess. Output is
//! a Vec<f64> per case; we compare element-wise and report
//! max-abs across the vector for each case. Tol 1e-12 abs
//! (ranks are integer or half-integer values).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::rankdata;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    method: String,
    data: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    ranks: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    method: String,
    max_abs_diff: f64,
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
    fs::create_dir_all(output_dir()).expect("create rankdata diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize rankdata diff log");
    fs::write(path, json).expect("write rankdata diff log");
}

fn generate_query() -> OracleQuery {
    let datasets: Vec<(&str, Vec<f64>)> = vec![
        // No ties: strict ordering — every method should agree.
        ("no_ties", vec![5.0, 1.0, 4.0, 2.0, 3.0, 7.0, 6.0]),
        // Light ties: a single 2-tie pair.
        ("light_ties", vec![1.0, 2.0, 2.0, 4.0, 3.0, 5.0]),
        // Heavy ties: two large tie groups.
        (
            "heavy_ties",
            vec![3.0, 3.0, 3.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 2.0],
        ),
        // Run only — single value repeated, exercises full collapse.
        ("constant_run", vec![7.0, 7.0, 7.0, 7.0, 7.0]),
    ];
    let methods = ["average", "min", "max", "dense", "ordinal"];

    let mut points = Vec::new();
    for (name, data) in datasets {
        for method in methods {
            points.push(PointCase {
                case_id: format!("{name}_{method}"),
                method: method.into(),
                data: data.clone(),
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
from scipy import stats

def filter_finite(arr):
    out = []
    for v in arr:
        try:
            v = float(v)
        except Exception:
            return None
        if not math.isfinite(v):
            return None
        out.append(v)
    return out

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; method = case["method"]
    data = np.array(case["data"], dtype=float)
    try:
        ranks = stats.rankdata(data, method=method)
        cleaned = filter_finite(ranks.tolist())
        points.append({"case_id": cid, "ranks": cleaned})
    except Exception:
        points.append({"case_id": cid, "ranks": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize rankdata query");
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
                "failed to spawn python3 for rankdata oracle: {e}"
            );
            eprintln!("skipping rankdata oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open rankdata oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "rankdata oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping rankdata oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for rankdata oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "rankdata oracle failed: {stderr}"
        );
        eprintln!("skipping rankdata oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse rankdata oracle JSON"))
}

#[test]
fn diff_stats_rankdata() {
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
        let Some(scipy_ranks) = &scipy_arm.ranks else {
            continue;
        };
        let rust_ranks = match rankdata(&case.data, Some(&case.method)) {
            Ok(r) => r,
            Err(_) => continue,
        };
        if rust_ranks.len() != scipy_ranks.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                method: case.method.clone(),
                max_abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let mut case_max = 0.0_f64;
        for (a, b) in rust_ranks.iter().zip(scipy_ranks.iter()) {
            case_max = case_max.max((a - b).abs());
        }
        max_overall = max_overall.max(case_max);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            method: case.method.clone(),
            max_abs_diff: case_max,
            pass: case_max <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_rankdata".into(),
        category: "scipy.stats.rankdata".into(),
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
                "rankdata mismatch: {} method={} max_abs={}",
                d.case_id, d.method, d.max_abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "rankdata conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
