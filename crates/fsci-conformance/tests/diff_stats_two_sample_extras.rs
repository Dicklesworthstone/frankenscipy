#![forbid(unsafe_code)]
//! Live SciPy differential coverage for two two-sample tests
//! not exercised by any other diff harness:
//!   • `cramervonmises_2samp(x, y)` — CDF-distance test with
//!     bias-corrected statistic and Bessel-K-series p-value
//!     (auto method).
//!   • `mood_alternative(x, y, alternative)` — Mood's test for
//!     equal scale parameters with directional alternative
//!     (two-sided / less / greater).
//!
//! Resolves [frankenscipy-9q3hc]. The oracle calls
//! `scipy.stats.cramervonmises_2samp(x, y)` and
//! `scipy.stats.mood(x, y, alternative=...)`.
//!
//! 4 fixtures × cvm2samp 2 arms + 4 fixtures × mood_alt 3
//! alternatives × 2 arms = 32 cases. Tol 1e-2 abs for the cvm
//! pvalue (Bessel-K asymptotic series — same precision floor
//! we use for the 1-sample cvm harness) and 1e-9 abs for cvm
//! statistic and all mood arms.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{cramervonmises_2samp, mood_alternative};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const STAT_TOL: f64 = 1.0e-9;
const CVM_PVALUE_TOL: f64 = 1.0e-5;
const MOOD_PVALUE_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    x: Vec<f64>,
    y: Vec<f64>,
    alternative: String,
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
    fs::create_dir_all(output_dir()).expect("create two_sample_extras diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize two_sample_extras diff log");
    fs::write(path, json).expect("write two_sample_extras diff log");
}

fn generate_query() -> OracleQuery {
    let fixtures: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![
        (
            "balanced_n10",
            (1..=10).map(|i| i as f64).collect(),
            (1..=10).map(|i| (i as f64) + 0.5).collect(),
        ),
        (
            "shifted_n12",
            (1..=12).map(|i| i as f64).collect(),
            (5..=16).map(|i| i as f64).collect(),
        ),
        (
            "diff_scale_n14",
            vec![
                4.0, 5.0, 4.5, 5.2, 4.8, 5.1, 4.9, 5.3, 4.7, 5.0, 4.6, 5.4, 4.5, 5.5,
            ],
            vec![
                -2.0, 12.0, -1.0, 11.0, 0.0, 10.0, 1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0,
            ],
        ),
        (
            "skewed_n16",
            (1..=16).map(|i| (i as f64).sqrt() * 4.0).collect(),
            (1..=16).map(|i| i as f64).collect(),
        ),
    ];

    let mut points = Vec::new();
    for (name, x, y) in &fixtures {
        // Cramér–von Mises 2-sample (no alternative arg — 2-sided only).
        points.push(PointCase {
            case_id: format!("{name}_cvm2samp"),
            func: "cvm2samp".into(),
            x: x.clone(),
            y: y.clone(),
            alternative: String::new(),
        });
        for alt in ["two-sided", "less", "greater"] {
            points.push(PointCase {
                case_id: format!("{name}_mood_{alt}"),
                func: "mood_alt".into(),
                x: x.clone(),
                y: y.clone(),
                alternative: alt.into(),
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
    x = np.array(case["x"], dtype=float)
    y = np.array(case["y"], dtype=float)
    stat = None; pval = None
    try:
        if func == "cvm2samp":
            res = stats.cramervonmises_2samp(x, y)
            stat = fnone(res.statistic); pval = fnone(res.pvalue)
        elif func == "mood_alt":
            res = stats.mood(x, y, alternative=case["alternative"])
            stat = fnone(res.statistic); pval = fnone(res.pvalue)
    except Exception:
        pass
    points.append({"case_id": cid, "statistic": stat, "pvalue": pval})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize two_sample_extras query");
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
                "failed to spawn python3 for two_sample_extras oracle: {e}"
            );
            eprintln!("skipping two_sample_extras oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open two_sample_extras oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "two_sample_extras oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping two_sample_extras oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for two_sample_extras oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "two_sample_extras oracle failed: {stderr}"
        );
        eprintln!("skipping two_sample_extras oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse two_sample_extras oracle JSON"))
}

#[test]
fn diff_stats_two_sample_extras() {
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
        let (rust_stat, rust_p, pvalue_tol) = match case.func.as_str() {
            "cvm2samp" => {
                let r = cramervonmises_2samp(&case.x, &case.y);
                (r.statistic, r.pvalue, CVM_PVALUE_TOL)
            }
            "mood_alt" => {
                let r = mood_alternative(&case.x, &case.y, &case.alternative);
                (r.statistic, r.pvalue, MOOD_PVALUE_TOL)
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
                pass: abs_diff <= STAT_TOL,
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
                pass: abs_diff <= pvalue_tol,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_two_sample_extras".into(),
        category: "scipy.stats.{cramervonmises_2samp, mood(alternative)}".into(),
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
                "two_sample_extras mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "two_sample_extras conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
