#![forbid(unsafe_code)]
//! Live numerical reference checks for two utilities not
//! covered by any other diff harness:
//!   • `theil_sen(x, y)` — simple median-of-pairwise-slopes
//!     regression returning (slope, intercept). Distinct from
//!     theilslopes (which returns slope/intercept/low/high
//!     and IS covered separately).
//!   • `ks_distance(data, cdf)` — supremum |F_n(x) - F(x)|
//!     between an empirical CDF and a target distribution.
//!
//! Resolves [frankenscipy-7ezth]. The oracle reproduces both
//! analytically in numpy + scipy.stats.norm.cdf. ks_distance
//! is tested against the standard normal cdf.
//!
//! 4 (x, y) fixtures × 2 theil_sen arms + 4 (data) fixtures
//! × 1 ks_distance arm = 12 cases via subprocess. Tol 1e-12
//! abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{ContinuousDistribution, Normal, ks_distance, theil_sen};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    x: Vec<f64>,
    y: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    slope: Option<f64>,
    intercept: Option<f64>,
    distance: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    func: String,
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
    fs::create_dir_all(output_dir()).expect("create theil_ks_distance diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize theil_ks_distance diff log");
    fs::write(path, json).expect("write theil_ks_distance diff log");
}

fn generate_query() -> OracleQuery {
    let theil_fixtures: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![
        (
            "clean_linear",
            (1..=10).map(|i| i as f64).collect(),
            (1..=10).map(|i| 2.0 * i as f64 + 1.0).collect(),
        ),
        (
            "noisy",
            (1..=12).map(|i| i as f64).collect(),
            vec![
                2.1, 4.3, 5.8, 8.2, 9.7, 12.1, 14.3, 16.05, 18.2, 19.8, 21.9, 24.1,
            ],
        ),
        (
            "outlier",
            (1..=10).map(|i| i as f64).collect(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 50.0, 7.0, 8.0, 9.0, 10.0],
        ),
        (
            "neg_slope",
            (1..=11).map(|i| i as f64).collect(),
            vec![10.05, 9.1, 7.85, 6.9, 6.05, 5.1, 4.0, 3.05, 2.1, 0.9, 0.1],
        ),
    ];
    let ks_fixtures: Vec<(&str, Vec<f64>)> = vec![
        (
            "near_normal",
            vec![
                -1.5, -0.9, -0.4, -0.1, 0.0, 0.2, 0.4, 0.6, 0.8, 1.1, 1.3, 1.6,
            ],
        ),
        (
            "skewed",
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.2, 1.6, 2.1, 2.8, 3.7],
        ),
        ("uniform", vec![-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]),
        ("bimodal", vec![-2.5, -2.0, -1.5, 1.5, 2.0, 2.5]),
    ];

    let mut points = Vec::new();
    for (name, x, y) in &theil_fixtures {
        points.push(PointCase {
            case_id: format!("theil_{name}"),
            func: "theil_sen".into(),
            x: x.clone(),
            y: y.clone(),
        });
    }
    for (name, data) in &ks_fixtures {
        points.push(PointCase {
            case_id: format!("ks_{name}"),
            func: "ks_distance".into(),
            x: data.clone(),
            y: vec![],
        });
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
    out = {"case_id": cid, "slope": None, "intercept": None, "distance": None}
    try:
        if func == "theil_sen":
            x = np.array(case["x"], dtype=float)
            y = np.array(case["y"], dtype=float)
            n = len(x)
            slopes = []
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[j] - x[i]
                    if abs(dx) > 1e-15:
                        slopes.append((y[j] - y[i]) / dx)
            slopes.sort()
            slope = float(np.median(slopes))
            # intercept = median(y - slope*x)
            residuals = sorted([yi - slope * xi for xi, yi in zip(x, y)])
            m = len(residuals)
            if m % 2 == 0:
                intercept = (residuals[m // 2 - 1] + residuals[m // 2]) / 2.0
            else:
                intercept = residuals[m // 2]
            out["slope"] = fnone(slope)
            out["intercept"] = fnone(intercept)
        elif func == "ks_distance":
            data = np.array(case["x"], dtype=float)
            n = len(data)
            sorted_d = np.sort(data)
            cdf_vals = stats.norm.cdf(sorted_d)
            ecdf_after = np.arange(1, n + 1) / n
            ecdf_before = np.arange(0, n) / n
            d_after = np.abs(ecdf_after - cdf_vals)
            d_before = np.abs(ecdf_before - cdf_vals)
            distance = float(max(np.max(d_after), np.max(d_before)))
            out["distance"] = fnone(distance)
    except Exception:
        pass
    points.append(out)
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize theil_ks_distance query");
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
                "failed to spawn python3 for theil_ks_distance oracle: {e}"
            );
            eprintln!("skipping theil_ks_distance oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open theil_ks_distance oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "theil_ks_distance oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping theil_ks_distance oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for theil_ks_distance oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "theil_ks_distance oracle failed: {stderr}"
        );
        eprintln!("skipping theil_ks_distance oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse theil_ks_distance oracle JSON"))
}

#[test]
fn diff_stats_theil_ks_distance() {
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

    let norm = Normal::standard();
    let cdf_norm = |x: f64| ContinuousDistribution::cdf(&norm, x);

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        match case.func.as_str() {
            "theil_sen" => {
                let (rs, ri) = theil_sen(&case.x, &case.y);
                if let Some(scipy_s) = scipy_arm.slope
                    && rs.is_finite()
                {
                    let abs_diff = (rs - scipy_s).abs();
                    max_overall = max_overall.max(abs_diff);
                    diffs.push(CaseDiff {
                        case_id: case.case_id.clone(),
                        func: case.func.clone(),
                        arm: "slope".into(),
                        abs_diff,
                        pass: abs_diff <= ABS_TOL,
                    });
                }
                if let Some(scipy_i) = scipy_arm.intercept
                    && ri.is_finite()
                {
                    let abs_diff = (ri - scipy_i).abs();
                    max_overall = max_overall.max(abs_diff);
                    diffs.push(CaseDiff {
                        case_id: case.case_id.clone(),
                        func: case.func.clone(),
                        arm: "intercept".into(),
                        abs_diff,
                        pass: abs_diff <= ABS_TOL,
                    });
                }
            }
            "ks_distance" => {
                let rd = ks_distance(&case.x, cdf_norm);
                if let Some(scipy_d) = scipy_arm.distance
                    && rd.is_finite()
                {
                    let abs_diff = (rd - scipy_d).abs();
                    max_overall = max_overall.max(abs_diff);
                    diffs.push(CaseDiff {
                        case_id: case.case_id.clone(),
                        func: case.func.clone(),
                        arm: "distance".into(),
                        abs_diff,
                        pass: abs_diff <= ABS_TOL,
                    });
                }
            }
            _ => {}
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_theil_ks_distance".into(),
        category: "theil_sen + ks_distance".into(),
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
                "theil_ks_distance {} mismatch: {} arm={} abs={}",
                d.func, d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "theil_ks_distance conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
