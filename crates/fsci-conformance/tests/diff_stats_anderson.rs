#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the Anderson-Darling
//! normality test `scipy.stats.anderson(data, dist='norm')`.
//!
//! Resolves [frankenscipy-3hunc]. Cross-checks both:
//!   • `statistic` — the raw A² value. scipy 1.17.1 returns
//!     A² without the (1+0.75/N+2.25/N²) correction; the
//!     correction is applied only to the critical values.
//!   • The five critical values at significance levels
//!     [15%, 10%, 5%, 2.5%, 1%], computed as
//!     `around(_Avals_norm / (1 + 0.75/N + 2.25/N²), 3)`.
//!
//! Surfaced [frankenscipy-2oulp] (the original fsci anderson
//! applied the correction to the *statistic* and returned
//! N-independent critical values). Fix landed alongside this
//! harness, so the diff now passes.
//!
//! 4 datasets × (1 stat + 5 critical) = 24 cases via subprocess.
//! Tolerances:
//!   - statistic       : 1e-11 abs (closed-form A² accumulator;
//!     small drift can show up at extreme |z| where fsci's cdf
//!     clamp can shave the last bit relative to scipy's logcdf).
//!   - critical_values : 1e-12 abs (tabulated constants scaled
//!     by a closed-form rational and rounded identically).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::anderson;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const STAT_TOL: f64 = 1.0e-11;
const CRIT_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct DatasetCase {
    case_id: String,
    data: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<DatasetCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    statistic: Option<f64>,
    critical_values: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create anderson diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize anderson diff log");
    fs::write(path, json).expect("write anderson diff log");
}

fn generate_query() -> OracleQuery {
    let datasets: Vec<(&str, Vec<f64>)> = vec![
        (
            "near_normal_n12",
            vec![
                -1.5, -0.9, -0.4, -0.1, 0.0, 0.2, 0.4, 0.6, 0.8, 1.1, 1.3, 1.6,
            ],
        ),
        (
            "near_normal_n20",
            vec![
                -2.1, -1.6, -1.2, -0.9, -0.7, -0.4, -0.2, 0.0, 0.1, 0.3, 0.4, 0.6, 0.8, 1.0, 1.1,
                1.3, 1.5, 1.8, 2.0, 2.4,
            ],
        ),
        (
            "skewed",
            vec![
                0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.2, 1.6, 2.1, 2.8, 3.7, 5.0, 7.0, 10.0,
            ],
        ),
        (
            "bimodal",
            vec![
                -2.5, -2.4, -2.2, -2.0, -1.8, -1.5, 1.5, 1.8, 2.0, 2.2, 2.4, 2.5,
            ],
        ),
    ];

    let points = datasets
        .into_iter()
        .map(|(name, data)| DatasetCase {
            case_id: name.into(),
            data,
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

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

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
    try:
        res = stats.anderson(data, dist='norm')
        points.append({
            "case_id": cid,
            "statistic": fnone(res.statistic),
            "critical_values": vec_or_none(res.critical_values.tolist()),
        })
    except Exception:
        points.append({"case_id": cid, "statistic": None, "critical_values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize anderson query");
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
                "failed to spawn python3 for anderson oracle: {e}"
            );
            eprintln!("skipping anderson oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open anderson oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "anderson oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping anderson oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for anderson oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "anderson oracle failed: {stderr}"
        );
        eprintln!("skipping anderson oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse anderson oracle JSON"))
}

#[test]
fn diff_stats_anderson() {
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
        let result = anderson(&case.data, "norm");

        if let Some(scipy_stat) = scipy_arm.statistic
            && result.statistic.is_finite()
        {
            let abs_diff = (result.statistic - scipy_stat).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "statistic".into(),
                abs_diff,
                pass: abs_diff <= STAT_TOL,
            });
        }

        if let Some(scipy_crit) = &scipy_arm.critical_values {
            for (idx, &scipy_v) in scipy_crit.iter().enumerate() {
                if idx >= result.critical_values.len() {
                    break;
                }
                let rust_v = result.critical_values[idx];
                if rust_v.is_finite() {
                    let abs_diff = (rust_v - scipy_v).abs();
                    max_overall = max_overall.max(abs_diff);
                    diffs.push(CaseDiff {
                        case_id: case.case_id.clone(),
                        arm: format!("crit_{idx}"),
                        abs_diff,
                        pass: abs_diff <= CRIT_TOL,
                    });
                }
            }
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_anderson".into(),
        category: "scipy.stats.anderson(dist='norm')".into(),
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
                "anderson mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "anderson conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
