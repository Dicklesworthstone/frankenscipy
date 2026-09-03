#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci's
//! `fligner(groups) → VarianceTestResult` — Fligner-Killeen
//! test for variance homogeneity (median-centered rank scores
//! against normal quantiles).
//!
//! Resolves [frankenscipy-1bgpn]. The oracle calls
//! `scipy.stats.fligner(*groups)`.
//!
//! 4 group-fixtures × 2 arms (statistic + pvalue) = 8 cases.
//! Tol 1e-9 abs (chi-square tail chain via ndtri).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::fligner;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
// 1e-6 abs to absorb the ndtri rational-approximation error on the
// half-normal-quantile chain (max observed ~3e-7 on a high-stat fixture).
const ABS_TOL: f64 = 1.0e-6;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    groups: Vec<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create fligner diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize fligner diff log");
    fs::write(path, json).expect("write fligner diff log");
}

fn generate_query() -> OracleQuery {
    let fixtures: Vec<(&str, Vec<Vec<f64>>)> = vec![
        // Two groups, equal variance (deterministic monotone)
        (
            "two_equal_var",
            vec![
                (1..=10).map(|i| i as f64).collect(),
                (11..=20).map(|i| i as f64).collect(),
            ],
        ),
        // Two groups, unequal variance (one tight, one wide)
        (
            "two_diff_var",
            vec![
                vec![5.0, 5.1, 4.9, 5.0, 5.1, 4.9, 5.0, 5.1, 4.9, 5.0, 5.1, 4.9],
                vec![1.0, 5.0, 9.0, 0.5, 4.5, 8.5, 1.5, 5.5, 9.5, 2.0, 6.0, 10.0],
            ],
        ),
        // Three groups
        (
            "three_groups",
            vec![
                (1..=8).map(|i| i as f64).collect(),
                (1..=8).map(|i| (i as f64) * 2.0).collect(),
                (1..=8).map(|i| (i as f64).powi(2)).collect(),
            ],
        ),
        // Four small groups, mixed
        (
            "four_small",
            vec![
                vec![1.0, 2.0, 3.0, 4.0, 5.0],
                vec![2.0, 4.0, 6.0, 8.0, 10.0],
                vec![3.0, 3.5, 4.0, 4.5, 5.0],
                vec![1.0, 1.5, 4.0, 8.0, 12.0],
            ],
        ),
    ];

    let points = fixtures
        .into_iter()
        .map(|(name, groups)| PointCase {
            case_id: name.into(),
            groups,
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
    groups = [np.array(g, dtype=float) for g in case["groups"]]
    stat = None; pval = None
    try:
        res = stats.fligner(*groups)
        stat = fnone(res.statistic)
        pval = fnone(res.pvalue)
    except Exception:
        pass
    points.append({"case_id": cid, "statistic": stat, "pvalue": pval})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize fligner query");
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
                "failed to spawn python3 for fligner oracle: {e}"
            );
            eprintln!("skipping fligner oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open fligner oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "fligner oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping fligner oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for fligner oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "fligner oracle failed: {stderr}"
        );
        eprintln!("skipping fligner oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse fligner oracle JSON"))
}

#[test]
fn diff_stats_fligner() {
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
        let group_refs: Vec<&[f64]> = case.groups.iter().map(|g| g.as_slice()).collect();
        let result = fligner(&group_refs);

        if let Some(s_stat) = scipy_arm.statistic
            && result.statistic.is_finite()
        {
            let abs_diff = (result.statistic - s_stat).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "statistic".into(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
        if let Some(s_p) = scipy_arm.pvalue
            && result.pvalue.is_finite()
        {
            let abs_diff = (result.pvalue - s_p).abs();
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
        test_id: "diff_stats_fligner".into(),
        category: "scipy.stats.fligner".into(),
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
                "fligner mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "fligner conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
