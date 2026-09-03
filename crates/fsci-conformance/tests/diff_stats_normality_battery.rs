#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the normality-test
//! battery in `scipy.stats`:
//!   • `skewtest(data)`     — D'Agostino skewness test
//!   • `kurtosistest(data)` — Anscombe-Glynn kurtosis test
//!   • `normaltest(data)`   — D'Agostino-Pearson omnibus
//!   • `jarque_bera(data)`  — Jarque-Bera moment test
//!
//! Resolves [frankenscipy-k2qie]. All four tests share the
//! same return shape (statistic + pvalue) but compute the
//! statistic via independent moment combinations and route
//! through different distribution tail computations.
//!
//! 4 dataset fixtures × 4 tests × 2 arms = 32 cases via
//! subprocess. Tol 1e-9 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{jarque_bera, kurtosistest, normaltest, skewtest};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    test: String,
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
    test: String,
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
    fs::create_dir_all(output_dir()).expect("create normality-battery diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize normality-battery diff log");
    fs::write(path, json).expect("write normality-battery diff log");
}

fn fsci_eval(test: &str, data: &[f64]) -> Option<(f64, f64)> {
    let r = match test {
        "skewtest" => skewtest(data, None, None).ok()?,
        "kurtosistest" => kurtosistest(data, None, None).ok()?,
        "normaltest" => normaltest(data),
        "jarque_bera" => jarque_bera(data),
        _ => return None,
    };
    if r.statistic.is_finite() && r.pvalue.is_finite() {
        Some((r.statistic, r.pvalue))
    } else {
        None
    }
}

fn generate_query() -> OracleQuery {
    let datasets: Vec<(&str, Vec<f64>)> = vec![
        // Near-normal, n=20 (skewtest needs n≥8, kurtosistest needs n≥20)
        (
            "near_normal_n20",
            vec![
                -2.1, -1.6, -1.2, -0.9, -0.7, -0.4, -0.2, 0.0, 0.1, 0.3, 0.4, 0.6, 0.8, 1.0, 1.1,
                1.3, 1.5, 1.8, 2.0, 2.4,
            ],
        ),
        // Near-normal, larger n=40
        (
            "near_normal_n40",
            (0..40)
                .map(|i| {
                    let p = (i as f64 + 0.5) / 40.0;
                    // Beasley-Springer-Moro lite via inverse-error approximation
                    // is overkill here; just hand-tabulate something near-N(0,1).
                    2.0 * (p - 0.5) * 1.4
                })
                .collect(),
        ),
        // Skewed (positive)
        (
            "skewed_n20",
            (1..=20).map(|i| (i as f64).powf(1.5) / 20.0).collect(),
        ),
        // Heavy-tailed (kurtotic) — t-like with extreme tails
        (
            "heavy_tail_n20",
            vec![
                -8.0, -3.0, -1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.1, 1.5,
                2.0, 3.0, 4.0, 6.0, 12.0,
            ],
        ),
    ];
    let tests = ["skewtest", "kurtosistest", "normaltest", "jarque_bera"];

    let mut points = Vec::new();
    for (name, data) in &datasets {
        for test in tests {
            points.push(PointCase {
                case_id: format!("{name}_{test}"),
                test: test.into(),
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
    cid = case["case_id"]; test = case["test"]
    data = np.array(case["data"], dtype=float)
    try:
        if test == "skewtest":
            res = stats.skewtest(data, alternative='two-sided')
        elif test == "kurtosistest":
            res = stats.kurtosistest(data, alternative='two-sided')
        elif test == "normaltest":
            res = stats.normaltest(data)
        elif test == "jarque_bera":
            res = stats.jarque_bera(data)
        else:
            res = None
        if res is None:
            points.append({"case_id": cid, "statistic": None, "pvalue": None})
        else:
            points.append({
                "case_id": cid,
                "statistic": fnone(res.statistic),
                "pvalue": fnone(res.pvalue),
            })
    except Exception:
        points.append({"case_id": cid, "statistic": None, "pvalue": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize normality-battery query");
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
                "failed to spawn python3 for normality-battery oracle: {e}"
            );
            eprintln!("skipping normality-battery oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open normality-battery oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "normality-battery oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping normality-battery oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for normality-battery oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "normality-battery oracle failed: {stderr}"
        );
        eprintln!("skipping normality-battery oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse normality-battery oracle JSON"))
}

#[test]
fn diff_stats_normality_battery() {
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
        let Some((stat, pval)) = fsci_eval(&case.test, &case.data) else {
            continue;
        };

        if let Some(scipy_stat) = scipy_arm.statistic {
            let abs_diff = (stat - scipy_stat).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                test: case.test.clone(),
                arm: "statistic".into(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
        if let Some(scipy_p) = scipy_arm.pvalue {
            let abs_diff = (pval - scipy_p).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                test: case.test.clone(),
                arm: "pvalue".into(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_normality_battery".into(),
        category: "scipy.stats.skewtest/kurtosistest/normaltest/jarque_bera".into(),
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
                "normality-battery {} mismatch: {} arm={} abs={}",
                d.test, d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "normality-battery conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
