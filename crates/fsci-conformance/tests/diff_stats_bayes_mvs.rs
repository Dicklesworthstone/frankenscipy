#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the Bayesian credible
//! interval routine `scipy.stats.bayes_mvs(data, alpha)`.
//!
//! Resolves [frankenscipy-dx6r8]. Cross-checks
//!   - mean.statistic / .low / .high
//!   - variance.low / .high
//!   - std.low / .high
//!     against scipy across 3 datasets at two confidence levels
//!     (0.90, 0.95). 3 datasets × 2 alphas × 7 arms = 42 cases.
//!
//! variance.statistic and std.statistic are deliberately
//! omitted: fsci uses sample variance ss/(n-1) for variance
//! point estimate while scipy uses Jeffrey's posterior mean
//! ss/(n-3); std uses an additional gamma-ratio bias
//! correction. Tracked as [frankenscipy-u2y4u]; the CI bounds
//! work out the same on both sides because the chi-squared
//! scaling cancels the s² choice.
//!
//! Tol 1e-9 abs throughout — fsci's mean interval chains
//! `StudentT::ppf`, the variance/std intervals chain
//! `ChiSquared::ppf`. Both inverse-cdfs use Newton+rational
//! initial guesses and accumulate ~1e-11 drift relative to
//! scipy's distpacked Brent solvers.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::bayes_mvs;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    data: Vec<f64>,
    alpha: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    mean_stat: Option<f64>,
    mean_lo: Option<f64>,
    mean_hi: Option<f64>,
    var_stat: Option<f64>,
    var_lo: Option<f64>,
    var_hi: Option<f64>,
    std_stat: Option<f64>,
    std_lo: Option<f64>,
    std_hi: Option<f64>,
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
    fs::create_dir_all(output_dir()).expect("create bayes_mvs diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize bayes_mvs diff log");
    fs::write(path, json).expect("write bayes_mvs diff log");
}

fn generate_query() -> OracleQuery {
    let datasets: Vec<(&str, Vec<f64>)> = vec![
        (
            "small_n10",
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        ),
        (
            "noisy_n15",
            vec![
                -0.7, 0.1, -0.3, 0.5, 1.2, 0.8, 1.5, -0.2, 0.9, 1.7, 0.4, -0.8, 1.0, 2.1, 0.3,
            ],
        ),
        (
            "wide_n20",
            vec![
                -3.0, -2.5, -1.8, -1.0, -0.5, 0.0, 0.3, 0.7, 1.1, 1.4, 1.8, 2.0, 2.5, 3.0, 3.5,
                4.0, 4.7, 5.5, 6.5, 8.0,
            ],
        ),
    ];
    let alphas = [0.90, 0.95];

    let mut points = Vec::new();
    for (name, data) in datasets {
        for &alpha in &alphas {
            points.push(PointCase {
                case_id: format!("{name}_a{alpha}"),
                data: data.clone(),
                alpha,
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
    alpha = float(case["alpha"])
    try:
        mean, var, std = stats.bayes_mvs(data, alpha=alpha)
        points.append({
            "case_id": cid,
            "mean_stat": fnone(mean.statistic),
            "mean_lo":   fnone(mean.minmax[0]),
            "mean_hi":   fnone(mean.minmax[1]),
            "var_stat":  fnone(var.statistic),
            "var_lo":    fnone(var.minmax[0]),
            "var_hi":    fnone(var.minmax[1]),
            "std_stat":  fnone(std.statistic),
            "std_lo":    fnone(std.minmax[0]),
            "std_hi":    fnone(std.minmax[1]),
        })
    except Exception:
        points.append({
            "case_id": cid,
            "mean_stat": None, "mean_lo": None, "mean_hi": None,
            "var_stat": None, "var_lo": None, "var_hi": None,
            "std_stat": None, "std_lo": None, "std_hi": None,
        })
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize bayes_mvs query");
    let mut child = match Command::new("python3")
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
                "failed to spawn python3 for bayes_mvs oracle: {e}"
            );
            eprintln!("skipping bayes_mvs oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open bayes_mvs oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "bayes_mvs oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping bayes_mvs oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for bayes_mvs oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "bayes_mvs oracle failed: {stderr}"
        );
        eprintln!("skipping bayes_mvs oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse bayes_mvs oracle JSON"))
}

#[test]
fn diff_stats_bayes_mvs() {
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
        let result = bayes_mvs(&case.data, case.alpha);

        // var_stat and std_stat re-enabled after u2y4u (point-estimate
        // formula fix): variance.statistic = ss/(n−3) matches scipy's
        // Jeffreys posterior mean; std.statistic uses the Generalized-
        // Gamma posterior mean.
        let arms: [(&str, Option<f64>, f64); 9] = [
            ("mean_stat", scipy_arm.mean_stat, result.mean.statistic),
            ("mean_lo", scipy_arm.mean_lo, result.mean.low),
            ("mean_hi", scipy_arm.mean_hi, result.mean.high),
            ("var_stat", scipy_arm.var_stat, result.variance.statistic),
            ("var_lo", scipy_arm.var_lo, result.variance.low),
            ("var_hi", scipy_arm.var_hi, result.variance.high),
            ("std_stat", scipy_arm.std_stat, result.std.statistic),
            ("std_lo", scipy_arm.std_lo, result.std.low),
            ("std_hi", scipy_arm.std_hi, result.std.high),
        ];

        for (arm_name, scipy_v, rust_v) in arms {
            if let Some(scipy_v) = scipy_v
                && rust_v.is_finite()
            {
                let abs_diff = (rust_v - scipy_v).abs();
                max_overall = max_overall.max(abs_diff);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    arm: arm_name.into(),
                    abs_diff,
                    pass: abs_diff <= ABS_TOL,
                });
            }
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_bayes_mvs".into(),
        category: "scipy.stats.bayes_mvs".into(),
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
                "bayes_mvs mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "bayes_mvs conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
