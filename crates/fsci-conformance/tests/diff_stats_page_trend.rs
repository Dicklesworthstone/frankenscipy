#![forbid(unsafe_code)]
//! Live SciPy differential coverage for Page's L trend test
//! `scipy.stats.page_trend_test(data)`.
//!
//! Resolves [frankenscipy-78chv]. Page's test is a
//! non-parametric repeated-measures ANOVA where conditions
//! are pre-ordered along a hypothesized monotonic trend.
//! Within each subject (row) ranks are assigned to the k
//! conditions; the L statistic is the weighted sum of column
//! rank-sums by their predicted column index, normalized.
//!
//! 4 (n×k) fixtures × 2 arms (statistic + pvalue) = 8 cases
//! via subprocess. Tol 1e-9 abs.
//!
//! The harness uses scipy's default `predicted_ranks=None`
//! and `ranked=False` so both libraries rank within-row
//! using the same convention.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::page_trend_test;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    data: Vec<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create page_trend diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize page_trend diff log");
    fs::write(path, json).expect("write page_trend diff log");
}

fn generate_query() -> OracleQuery {
    // n×k repeated-measures matrices. Conditions (columns) are
    // pre-ordered along the hypothesized increasing trend.
    let fixtures: Vec<(&str, Vec<Vec<f64>>)> = vec![
        // Strong increasing trend: each row is monotone non-decreasing
        (
            "monotone_n6_k4",
            vec![
                vec![1.0, 2.0, 3.0, 4.0],
                vec![2.0, 3.0, 4.0, 5.0],
                vec![1.5, 2.5, 3.5, 4.5],
                vec![1.0, 2.0, 4.0, 5.0],
                vec![2.0, 3.0, 4.0, 6.0],
                vec![1.0, 3.0, 4.0, 5.0],
            ],
        ),
        // Mild increasing trend with noise
        (
            "noisy_n8_k3",
            vec![
                vec![3.0, 4.0, 5.0],
                vec![4.0, 5.0, 6.0],
                vec![3.0, 5.0, 4.0],
                vec![5.0, 6.0, 7.0],
                vec![4.0, 6.0, 5.0],
                vec![5.0, 7.0, 8.0],
                vec![6.0, 7.0, 9.0],
                vec![5.0, 8.0, 9.0],
            ],
        ),
        // No clear trend (negative test)
        (
            "no_trend_n5_k4",
            vec![
                vec![3.0, 1.0, 4.0, 2.0],
                vec![2.0, 4.0, 1.0, 3.0],
                vec![1.0, 3.0, 2.0, 4.0],
                vec![4.0, 2.0, 3.0, 1.0],
                vec![3.0, 4.0, 1.0, 2.0],
            ],
        ),
        // 5 conditions, 7 subjects, increasing trend
        (
            "increasing_n7_k5",
            vec![
                vec![1.0, 2.0, 3.0, 4.0, 5.0],
                vec![2.0, 3.0, 4.0, 5.0, 6.0],
                vec![1.5, 2.5, 3.5, 4.5, 5.5],
                vec![3.0, 4.0, 5.0, 6.0, 7.0],
                vec![2.5, 3.5, 4.5, 5.5, 6.5],
                vec![4.0, 5.0, 6.0, 7.0, 8.0],
                vec![3.5, 4.5, 5.5, 6.5, 7.5],
            ],
        ),
    ];

    let points = fixtures
        .into_iter()
        .map(|(name, data)| PointCase {
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

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    data = np.array(case["data"], dtype=float)
    try:
        # method='asymptotic' to match fsci's normal-tail pvalue path
        # (fsci does not implement the exact-permutation method).
        res = stats.page_trend_test(data, method='asymptotic')
        points.append({
            "case_id": cid,
            "statistic": fnone(res.statistic),
            "pvalue": fnone(res.pvalue),
        })
    except Exception:
        points.append({"case_id": cid, "statistic": None, "pvalue": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize page_trend query");
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
                "failed to spawn python3 for page_trend oracle: {e}"
            );
            eprintln!("skipping page_trend oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open page_trend oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "page_trend oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping page_trend oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for page_trend oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "page_trend oracle failed: {stderr}"
        );
        eprintln!("skipping page_trend oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse page_trend oracle JSON"))
}

#[test]
fn diff_stats_page_trend() {
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
        let data: Vec<&[f64]> = case.data.iter().map(|r| r.as_slice()).collect();
        let result = page_trend_test(&data);

        if let Some(scipy_stat) = scipy_arm.statistic
            && result.statistic.is_finite()
        {
            let abs_diff = (result.statistic - scipy_stat).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
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
                arm: "pvalue".into(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_page_trend".into(),
        category: "scipy.stats.page_trend_test".into(),
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
                "page_trend mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "page_trend conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
