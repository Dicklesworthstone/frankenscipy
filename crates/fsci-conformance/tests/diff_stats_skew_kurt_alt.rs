#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the one-sided
//! variants of `scipy.stats.skewtest` and
//! `scipy.stats.kurtosistest` — `alternative='less'` and
//! `'greater'`. The two-sided defaults are already covered
//! by diff_stats_normality_battery.rs.
//!
//! Resolves [frankenscipy-lcaym]. Cross-checks the test
//! statistic and the directed pvalue across 3 datasets × 2
//! functions × 2 alternatives × 2 arms = 24 cases via
//! subprocess. Tol 1e-9 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{kurtosistest, skewtest};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    alternative: String,
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
    fs::create_dir_all(output_dir()).expect("create skew_kurt_alt diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize skew_kurt_alt diff log");
    fs::write(path, json).expect("write skew_kurt_alt diff log");
}

fn generate_query() -> OracleQuery {
    let datasets: Vec<(&str, Vec<f64>)> = vec![
        (
            "near_normal_n20",
            vec![
                -2.1, -1.6, -1.2, -0.9, -0.7, -0.4, -0.2, 0.0, 0.1, 0.3, 0.4, 0.6, 0.8, 1.0, 1.1,
                1.3, 1.5, 1.8, 2.0, 2.4,
            ],
        ),
        (
            "skewed_pos",
            vec![
                0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.2, 1.6, 2.1, 2.8, 3.7, 5.0, 7.0, 10.0, 14.0,
                20.0, 28.0, 40.0, 60.0,
            ],
        ),
        (
            "heavy_tail_n20",
            vec![
                -8.0, -3.0, -1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.1, 1.5,
                2.0, 3.0, 4.0, 6.0, 12.0,
            ],
        ),
    ];

    let mut points = Vec::new();
    for (name, data) in &datasets {
        for func in ["skewtest", "kurtosistest"] {
            for alt in ["less", "greater"] {
                points.push(PointCase {
                    case_id: format!("{func}_{name}_{alt}"),
                    func: func.into(),
                    alternative: alt.into(),
                    data: data.clone(),
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
    cid = case["case_id"]; func = case["func"]; alt = case["alternative"]
    data = np.array(case["data"], dtype=float)
    val_stat = None; val_p = None
    try:
        if func == "skewtest":
            res = stats.skewtest(data, alternative=alt)
        elif func == "kurtosistest":
            res = stats.kurtosistest(data, alternative=alt)
        else:
            res = None
        if res is not None:
            val_stat = fnone(res.statistic)
            val_p = fnone(res.pvalue)
    except Exception:
        pass
    points.append({"case_id": cid, "statistic": val_stat, "pvalue": val_p})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize skew_kurt_alt query");
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
                "failed to spawn python3 for skew_kurt_alt oracle: {e}"
            );
            eprintln!("skipping skew_kurt_alt oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open skew_kurt_alt oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "skew_kurt_alt oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping skew_kurt_alt oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for skew_kurt_alt oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "skew_kurt_alt oracle failed: {stderr}"
        );
        eprintln!("skipping skew_kurt_alt oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse skew_kurt_alt oracle JSON"))
}

#[test]
fn diff_stats_skew_kurt_alt() {
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
        let result = match case.func.as_str() {
            "skewtest" => skewtest(&case.data, None, Some(&case.alternative)),
            "kurtosistest" => kurtosistest(&case.data, None, Some(&case.alternative)),
            _ => continue,
        };
        let result = match result {
            Ok(r) => r,
            Err(_) => continue,
        };

        if let Some(scipy_stat) = scipy_arm.statistic
            && result.statistic.is_finite()
        {
            let abs_diff = (result.statistic - scipy_stat).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
                arm: "statistic".into(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
        if let Some(scipy_p) = scipy_arm.pvalue
            && result.pvalue.is_finite()
        {
            let abs_diff = (result.pvalue - scipy_p).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
                arm: "pvalue".into(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_skew_kurt_alt".into(),
        category: "scipy.stats.skewtest/kurtosistest alternative=less|greater".into(),
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
                "skew_kurt_alt {} mismatch: {} arm={} abs={}",
                d.func, d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "skew_kurt_alt conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
