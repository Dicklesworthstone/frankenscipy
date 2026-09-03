#![forbid(unsafe_code)]
//! Live SciPy differential coverage for
//! `scipy.stats.percentileofscore(a, score, kind)` across all
//! four kinds (rank, weak, strict, mean).
//!
//! Resolves [frankenscipy-hlj21]. Originally surfaced via
//! diff_stats_summary_quantiles.rs which had to drop
//! percentileofscore because fsci's 'weak'/'strict' were
//! swapped vs scipy and 'rank' was missing the +0.5/n boost
//! when the score is in the data. Both fixed; harness now
//! ships green.
//!
//! 3 datasets × 4 kinds × varying score points = ~48 cases
//! via subprocess. Tol 1e-12 abs (closed-form rational
//! arithmetic over count/n).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::percentileofscore;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    data: Vec<f64>,
    score: f64,
    kind: String,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    value: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    kind: String,
    abs_diff: f64,
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
    fs::create_dir_all(output_dir()).expect("create percentileofscore diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize percentileofscore diff log");
    fs::write(path, json).expect("write percentileofscore diff log");
}

fn generate_query() -> OracleQuery {
    let datasets: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![
        // compact: 10 distinct integers; tests both in-data and not-in-data scores
        (
            "compact",
            (1..=10).map(|i| i as f64).collect(),
            vec![1.0, 3.0, 5.5, 7.0, 10.0, 0.0, 11.0],
        ),
        // ties: heavy ties at certain values
        (
            "ties",
            vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0],
            vec![2.0, 3.0, 4.0, 1.5, 2.5, 6.0],
        ),
        // spread: continuous-looking values
        (
            "spread",
            vec![
                -3.0, -1.5, -0.7, 0.0, 0.5, 1.2, 2.0, 3.5, 4.7, 6.0, 8.5, 12.0,
            ],
            vec![-1.5, 0.0, 1.0, 5.0, 12.0, -10.0, 100.0],
        ),
    ];
    let kinds = ["rank", "weak", "strict", "mean"];

    let mut points = Vec::new();
    for (name, data, scores) in &datasets {
        for kind in kinds {
            for &s in scores {
                points.push(PointCase {
                    case_id: format!("{name}_{kind}_s{s}"),
                    data: data.clone(),
                    score: s,
                    kind: kind.into(),
                });
            }
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

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    data = np.array(case["data"], dtype=float)
    score = float(case["score"])
    kind = case["kind"]
    try:
        val = float(stats.percentileofscore(data, score, kind=kind))
    except Exception:
        val = None
    points.append({"case_id": cid, "value": fnone(val)})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize percentileofscore query");
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
                "failed to spawn python3 for percentileofscore oracle: {e}"
            );
            eprintln!("skipping percentileofscore oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open percentileofscore oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "percentileofscore oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping percentileofscore oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for percentileofscore oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "percentileofscore oracle failed: {stderr}"
        );
        eprintln!("skipping percentileofscore oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse percentileofscore oracle JSON"))
}

#[test]
fn diff_stats_percentileofscore() {
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
        if let Some(scipy_v) = scipy_arm.value {
            let rust_v = percentileofscore(&case.data, case.score, Some(&case.kind));
            if rust_v.is_finite() {
                let abs_diff = (rust_v - scipy_v).abs();
                max_overall = max_overall.max(abs_diff);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    kind: case.kind.clone(),
                    abs_diff,
                    pass: abs_diff <= ABS_TOL,
                });
            }
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_percentileofscore".into(),
        category: "scipy.stats.percentileofscore".into(),
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
                "percentileofscore mismatch: {} kind={} abs={}",
                d.case_id, d.kind, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "percentileofscore conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
