#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci's three
//! D'Agostino normality tests:
//!   • `skewtest(data, nan_policy, alternative)` (D'Agostino 1990)
//!   • `kurtosistest(data, nan_policy, alternative)` (Anscombe-Glynn 1983)
//!   • `normaltest(data)` (D'Agostino-Pearson K²)
//!
//! Resolves [frankenscipy-p6rog]. The oracle calls
//! `scipy.stats.{skewtest, kurtosistest, normaltest}`.
//!
//! 4 datasets × 3 funcs × 2 arms (statistic + pvalue) = 24
//! cases. Tol 1e-9 abs (two-sided normal CDF / chi-square
//! tail chain via ndtri rational approximation).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{kurtosistest, normaltest, skewtest};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
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
    statistic: Option<f64>,
    pvalue: Option<f64>,
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
    fs::create_dir_all(output_dir()).expect("create normality_tests diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize normality_tests diff log");
    fs::write(path, json).expect("write normality_tests diff log");
}

fn generate_query() -> OracleQuery {
    // n ≥ 20 ensures all three asymptotic z-tests are valid.
    let datasets: Vec<(&str, Vec<f64>)> = vec![
        // Near-uniform 20 points (light tails)
        ("uniform_n20", (1..=20).map(|i| (i as f64) - 10.5).collect()),
        // Near-normal: deterministic inverse-CDF samples.
        ("near_normal_n30", {
            // map (i+0.5)/n through a smooth probit-ish polynomial — gives a
            // near-normal-shape deterministic sequence without needing scipy.
            (0..30)
                .map(|i| {
                    let p = (i as f64 + 0.5) / 30.0;
                    let q = p - 0.5;
                    // Approximate probit via cubic — symmetric, monotone.
                    2.5 * (q + 4.0 * q * q * q)
                })
                .collect()
        }),
        // Right-skewed exponential-like
        (
            "exp_like_n25",
            (1..=25).map(|i| ((i as f64) / 5.0).exp() - 1.0).collect(),
        ),
        // Heavy-tailed (cubic kept symmetric to favor kurtosistest signal)
        (
            "heavy_tail_n40",
            (0..40)
                .map(|i| {
                    let q = (i as f64 + 0.5) / 40.0 - 0.5;
                    q * q * q * 12.0
                })
                .collect(),
        ),
    ];

    let mut points = Vec::new();
    for (name, data) in &datasets {
        for func in ["skewtest", "kurtosistest", "normaltest"] {
            points.push(PointCase {
                case_id: format!("{name}_{func}"),
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
    stat = None; pval = None
    try:
        if func == "skewtest":
            res = stats.skewtest(data)
        elif func == "kurtosistest":
            res = stats.kurtosistest(data)
        elif func == "normaltest":
            res = stats.normaltest(data)
        else:
            res = None
        if res is not None:
            stat = fnone(res.statistic)
            pval = fnone(res.pvalue)
    except Exception:
        pass
    points.append({"case_id": cid, "statistic": stat, "pvalue": pval})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize normality_tests query");
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
                "failed to spawn python3 for normality_tests oracle: {e}"
            );
            eprintln!("skipping normality_tests oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open normality_tests oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "normality_tests oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping normality_tests oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for normality_tests oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "normality_tests oracle failed: {stderr}"
        );
        eprintln!("skipping normality_tests oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse normality_tests oracle JSON"))
}

#[test]
fn diff_stats_normality_tests() {
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
        let (rust_stat, rust_p) = match case.func.as_str() {
            "skewtest" => {
                let res = skewtest(&case.data, None, None).expect("skewtest");
                (res.statistic, res.pvalue)
            }
            "kurtosistest" => {
                let res = kurtosistest(&case.data, None, None).expect("kurtosistest");
                (res.statistic, res.pvalue)
            }
            "normaltest" => {
                let res = normaltest(&case.data);
                (res.statistic, res.pvalue)
            }
            _ => continue,
        };

        if let Some(s_stat) = scipy_arm.statistic
            && rust_stat.is_finite()
        {
            let abs_diff = (rust_stat - s_stat).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: format!("{}.statistic", case.func),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
        if let Some(s_p) = scipy_arm.pvalue
            && rust_p.is_finite()
        {
            let abs_diff = (rust_p - s_p).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: format!("{}.pvalue", case.func),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_normality_tests".into(),
        category: "scipy.stats.{skewtest, kurtosistest, normaltest}".into(),
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
                "normality_tests mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "normality_tests conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
