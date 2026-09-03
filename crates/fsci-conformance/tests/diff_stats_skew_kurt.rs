#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the standalone
//! sample skewness and kurtosis functions:
//!   • `scipy.stats.skew(a, bias=True)`
//!   • `scipy.stats.kurtosis(a, fisher=True, bias=True)`
//!
//! Resolves [frankenscipy-eucfl]. fsci's `skew(data)` and
//! `kurtosis(data)` use biased (population) central moments
//! and Fisher's excess-kurtosis definition (m4/m2² - 3),
//! which matches scipy's bias=True / fisher=True default.
//!
//! 5 datasets × 2 functions = 10 cases via subprocess.
//! Tol 1e-12 abs (closed-form moment ratios).
//!
//! Distinct from diff_stats_describe_misc.rs which exercises
//! these same quantities through the full `describe()`
//! 7-tuple — this harness pins the standalone entry points.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{kurtosis, skew};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    data: Vec<f64>,
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
    func: String,
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
    fs::create_dir_all(output_dir()).expect("create skew_kurt diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize skew_kurt diff log");
    fs::write(path, json).expect("write skew_kurt diff log");
}

fn generate_query() -> OracleQuery {
    let datasets: Vec<(&str, Vec<f64>)> = vec![
        (
            "near_normal",
            vec![
                -1.5, -0.9, -0.4, -0.1, 0.0, 0.2, 0.4, 0.6, 0.8, 1.1, 1.3, 1.6,
            ],
        ),
        (
            "skewed_pos",
            vec![
                0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.2, 1.6, 2.1, 2.8, 3.7, 5.0, 7.0, 10.0,
            ],
        ),
        (
            "skewed_neg",
            vec![
                -10.0, -7.0, -5.0, -3.7, -2.8, -2.1, -1.6, -1.2, -0.9, -0.7, -0.5, -0.4, -0.3,
                -0.2, -0.1,
            ],
        ),
        (
            "heavy_tailed",
            vec![
                -8.0, -3.0, -1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.1, 1.5,
                2.0, 3.0, 4.0, 6.0, 12.0,
            ],
        ),
        ("uniform_like", (1..=20).map(|i| i as f64).collect()),
    ];

    let mut points = Vec::new();
    for (name, data) in &datasets {
        for func in ["skew", "kurtosis"] {
            points.push(PointCase {
                case_id: format!("{func}_{name}"),
                func: func.into(),
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

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; func = case["func"]
    data = np.array(case["data"], dtype=float)
    val = None
    try:
        if func == "skew":
            val = float(stats.skew(data, bias=True))
        elif func == "kurtosis":
            val = float(stats.kurtosis(data, fisher=True, bias=True))
    except Exception:
        val = None
    points.append({"case_id": cid, "value": fnone(val)})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize skew_kurt query");
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
                "failed to spawn python3 for skew_kurt oracle: {e}"
            );
            eprintln!("skipping skew_kurt oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open skew_kurt oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "skew_kurt oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping skew_kurt oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for skew_kurt oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "skew_kurt oracle failed: {stderr}"
        );
        eprintln!("skipping skew_kurt oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse skew_kurt oracle JSON"))
}

#[test]
fn diff_stats_skew_kurt() {
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
            let rust_v = match case.func.as_str() {
                "skew" => skew(&case.data),
                "kurtosis" => kurtosis(&case.data),
                _ => continue,
            };
            if rust_v.is_finite() {
                let abs_diff = (rust_v - scipy_v).abs();
                max_overall = max_overall.max(abs_diff);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    func: case.func.clone(),
                    abs_diff,
                    pass: abs_diff <= ABS_TOL,
                });
            }
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_skew_kurt".into(),
        category: "scipy.stats.skew + kurtosis".into(),
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
                "skew_kurt {} mismatch: {} abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "skew_kurt conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
