#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the iterative
//! sigma-clipping outlier removal
//! `scipy.stats.sigmaclip(data, low, high)`.
//!
//! Resolves [frankenscipy-5f3v0]. Cross-checks both the
//! returned clipped-vector length and the (lower, upper)
//! bound scalars across 4 datasets × 3 (low, high) configs.
//!
//! 4 fixtures × 3 configs × 3 arms (n_clipped + lower +
//! upper) = 36 cases via subprocess. Tol 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::sigmaclip;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    data: Vec<f64>,
    low: f64,
    high: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    n_clipped: Option<i64>,
    lower: Option<f64>,
    upper: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    arm: String,
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
    fs::create_dir_all(output_dir()).expect("create sigmaclip diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize sigmaclip diff log");
    fs::write(path, json).expect("write sigmaclip diff log");
}

fn generate_query() -> OracleQuery {
    let datasets: Vec<(&str, Vec<f64>)> = vec![
        // Mostly-clean with one outlier
        (
            "near_clean",
            vec![
                1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98, 1.04, 0.96, 1.01, 0.99, 25.0,
            ],
        ),
        // Several outliers on both tails
        (
            "two_sided_outliers",
            vec![
                -50.0, 1.0, 1.2, 0.9, 1.1, 1.0, 0.8, 1.3, 1.05, 0.95, 50.0, 75.0, -40.0,
            ],
        ),
        // No outliers — should converge in 1 iteration
        (
            "no_outliers",
            vec![
                10.0, 10.5, 9.5, 10.2, 9.8, 10.4, 9.6, 10.1, 9.9, 10.3, 9.7, 10.0,
            ],
        ),
        // Heavy-tailed sample with multiple iterations needed
        (
            "heavy_tailed",
            vec![
                0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 3.0, -3.0, 5.0, -5.0, 8.0, -8.0,
                15.0, -15.0,
            ],
        ),
    ];
    // (low_sigma, high_sigma) configs
    let configs: &[(&str, f64, f64)] = &[
        ("3_3", 3.0, 3.0), // Standard 3σ both sides
        ("2_4", 2.0, 4.0), // Tighter low, looser high
        ("4_2", 4.0, 2.0), // Looser low, tighter high
    ];

    let mut points = Vec::new();
    for (name, data) in &datasets {
        for (cname, low, high) in configs {
            points.push(PointCase {
                case_id: format!("{name}_{cname}"),
                data: data.clone(),
                low: *low,
                high: *high,
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
    low = float(case["low"]); high = float(case["high"])
    try:
        clipped, lo, hi = stats.sigmaclip(data, low=low, high=high)
        points.append({
            "case_id": cid,
            "n_clipped": int(len(clipped)),
            "lower": fnone(lo),
            "upper": fnone(hi),
        })
    except Exception:
        points.append({"case_id": cid, "n_clipped": None,
                       "lower": None, "upper": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize sigmaclip query");
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
                "failed to spawn python3 for sigmaclip oracle: {e}"
            );
            eprintln!("skipping sigmaclip oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open sigmaclip oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "sigmaclip oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping sigmaclip oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for sigmaclip oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "sigmaclip oracle failed: {stderr}"
        );
        eprintln!("skipping sigmaclip oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse sigmaclip oracle JSON"))
}

#[test]
fn diff_stats_sigmaclip() {
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
        let result = sigmaclip(&case.data, case.low, case.high);

        if let Some(scipy_n) = scipy_arm.n_clipped {
            let abs_diff = (result.clipped.len() as i64 - scipy_n).unsigned_abs() as f64;
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "n_clipped".into(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
        if let Some(scipy_lo) = scipy_arm.lower
            && result.lower.is_finite()
        {
            let abs_diff = (result.lower - scipy_lo).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "lower".into(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
        if let Some(scipy_hi) = scipy_arm.upper
            && result.upper.is_finite()
        {
            let abs_diff = (result.upper - scipy_hi).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "upper".into(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_sigmaclip".into(),
        category: "scipy.stats.sigmaclip".into(),
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
                "sigmaclip mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "sigmaclip conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
