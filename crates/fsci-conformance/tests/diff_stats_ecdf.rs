#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the empirical CDF
//! evaluator `scipy.stats.ecdf(sample).cdf.evaluate(x)`.
//!
//! Resolves [frankenscipy-vel7p]. fsci's `ecdf(data, x_eval)`
//! returns the fraction of `data` ≤ each query point in
//! `x_eval`. scipy's `ecdf(sample).cdf.evaluate(x)` returns
//! the same value.
//!
//! 3 datasets × 7 query points = 21 cases via subprocess
//! (compared as a single max-abs-vector arm per (data, query
//! grid) pair = 3 cases). Tol 1e-12 abs (closed-form count/n
//! arithmetic — both libraries should agree exactly).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::ecdf;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    data: Vec<f64>,
    x_eval: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    cdf_values: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
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
    fs::create_dir_all(output_dir()).expect("create ecdf diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize ecdf diff log");
    fs::write(path, json).expect("write ecdf diff log");
}

fn generate_query() -> OracleQuery {
    let fixtures: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![
        // Compact integer data
        (
            "compact",
            (1..=10).map(|i| i as f64).collect(),
            vec![0.0, 1.0, 3.5, 5.0, 7.5, 10.0, 11.0],
        ),
        // Spread mixed-sign data
        (
            "spread",
            vec![-3.0, -1.5, -0.7, 0.0, 0.5, 1.2, 2.0, 3.5, 4.7, 6.0],
            vec![-5.0, -1.0, 0.0, 1.0, 3.0, 5.0, 10.0],
        ),
        // Data with ties
        (
            "ties",
            vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0],
            vec![0.5, 2.0, 2.5, 3.0, 4.0, 4.5, 6.0],
        ),
    ];

    let points = fixtures
        .into_iter()
        .map(|(name, data, x_eval)| PointCase {
            case_id: name.into(),
            data,
            x_eval,
        })
        .collect();
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy import stats

def vec_or_none(arr):
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
    cid = case["case_id"]
    data = np.array(case["data"], dtype=float)
    x_eval = np.array(case["x_eval"], dtype=float)
    val = None
    try:
        res = stats.ecdf(data)
        # cdf.evaluate gives the fraction <= each x.
        val = vec_or_none(res.cdf.evaluate(x_eval).tolist())
    except Exception:
        val = None
    points.append({"case_id": cid, "cdf_values": val})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize ecdf query");
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
                "failed to spawn python3 for ecdf oracle: {e}"
            );
            eprintln!("skipping ecdf oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open ecdf oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "ecdf oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping ecdf oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for ecdf oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "ecdf oracle failed: {stderr}"
        );
        eprintln!("skipping ecdf oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse ecdf oracle JSON"))
}

#[test]
fn diff_stats_ecdf() {
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
        let Some(scipy_vec) = &scipy_arm.cdf_values else {
            continue;
        };
        let rust_vec = ecdf(&case.data, &case.x_eval);
        if rust_vec.len() != scipy_vec.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let mut max_local = 0.0_f64;
        for (a, b) in rust_vec.iter().zip(scipy_vec.iter()) {
            if a.is_finite() {
                max_local = max_local.max((a - b).abs());
            }
        }
        max_overall = max_overall.max(max_local);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: max_local,
            pass: max_local <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_ecdf".into(),
        category: "scipy.stats.ecdf cdf.evaluate".into(),
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
            eprintln!("ecdf mismatch: {} abs={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "ecdf conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
