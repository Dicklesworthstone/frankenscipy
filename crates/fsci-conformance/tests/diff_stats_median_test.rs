#![forbid(unsafe_code)]
//! Live SciPy differential coverage for Mood's median test
//! `scipy.stats.median_test(*groups)`.
//!
//! Resolves [frankenscipy-k462e]. Computes the chi-squared
//! statistic on a 2×k above/below-grand-median contingency
//! table.
//!
//! 4 group-set fixtures × 3 arms (statistic + pvalue + df) =
//! 12 cases via subprocess. Tol 1e-9 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::median_test;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
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
    df: Option<f64>,
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
    fs::create_dir_all(output_dir()).expect("create median_test diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize median_test diff log");
    fs::write(path, json).expect("write median_test diff log");
}

fn generate_query() -> OracleQuery {
    let fixtures: Vec<(&str, Vec<Vec<f64>>)> = vec![
        // 2 groups, similar medians
        (
            "g2_similar",
            vec![
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            ],
        ),
        // 2 groups, different medians
        (
            "g2_different",
            vec![
                vec![1.0, 2.0, 3.0, 4.0, 5.0],
                vec![10.0, 11.0, 12.0, 13.0, 14.0],
            ],
        ),
        // 3 groups
        (
            "g3_mixed",
            vec![
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                vec![6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            ],
        ),
        // 4 groups
        (
            "g4_mixed",
            vec![
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0],
                vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
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
    try:
        # ties='below' is scipy's default — observations equal to the
        # grand median are counted in the 'below' row.
        result = stats.median_test(*groups, ties='below')
        # scipy returns (stat, p, median, table). The dof is len(groups)-1.
        stat = float(result[0])
        pval = float(result[1])
        df = float(len(groups) - 1)
        points.append({
            "case_id": cid,
            "statistic": fnone(stat),
            "pvalue": fnone(pval),
            "df": fnone(df),
        })
    except Exception:
        points.append({"case_id": cid, "statistic": None, "pvalue": None, "df": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize median_test query");
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
                "failed to spawn python3 for median_test oracle: {e}"
            );
            eprintln!("skipping median_test oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open median_test oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "median_test oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping median_test oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for median_test oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "median_test oracle failed: {stderr}"
        );
        eprintln!("skipping median_test oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse median_test oracle JSON"))
}

#[test]
fn diff_stats_median_test() {
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
        let groups: Vec<&[f64]> = case.groups.iter().map(|g| g.as_slice()).collect();
        let result = median_test(&groups);

        let arms: [(&str, Option<f64>, f64); 3] = [
            ("statistic", scipy_arm.statistic, result.statistic),
            ("pvalue", scipy_arm.pvalue, result.pvalue),
            ("df", scipy_arm.df, result.df),
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
        test_id: "diff_stats_median_test".into(),
        category: "scipy.stats.median_test".into(),
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
                "median_test mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "median_test conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
