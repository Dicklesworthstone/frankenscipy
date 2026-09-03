#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the nan_policy='omit'
//! variants of the variance-homogeneity / k-sample tests in
//! scipy.stats:
//!   • `levene(*groups, nan_policy='omit')`
//!   • `bartlett(*groups, nan_policy='omit')`
//!   • `kruskal(*groups, nan_policy='omit')`
//!
//! Resolves [frankenscipy-doiwy]. Default no-NaN paths are
//! covered by diff_stats_variance_tests.rs and
//! diff_stats_kruskal.rs; this harness exercises the
//! orthogonal NaN-omit code path.
//!
//! 3 group-set fixtures × 3 tests × 2 arms (statistic +
//! pvalue) = 18 cases via subprocess. Tol 1e-9 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{bartlett_with_nan_policy, kruskal_with_nan_policy, levene_with_nan_policy};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    test: String,
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
    fs::create_dir_all(output_dir()).expect("create variance_tests_nan diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize variance_tests_nan diff log");
    fs::write(path, json).expect("write variance_tests_nan diff log");
}

fn fsci_eval(test: &str, groups: &[&[f64]]) -> Option<(f64, f64)> {
    match test {
        "levene" => {
            let r = levene_with_nan_policy(groups, Some("omit")).ok()?;
            if r.statistic.is_finite() && r.pvalue.is_finite() {
                Some((r.statistic, r.pvalue))
            } else {
                None
            }
        }
        "bartlett" => {
            let r = bartlett_with_nan_policy(groups, Some("omit")).ok()?;
            if r.statistic.is_finite() && r.pvalue.is_finite() {
                Some((r.statistic, r.pvalue))
            } else {
                None
            }
        }
        "kruskal" => {
            let r = kruskal_with_nan_policy(groups, Some("omit")).ok()?;
            if r.statistic.is_finite() && r.pvalue.is_finite() {
                Some((r.statistic, r.pvalue))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    let nan = f64::NAN;
    let fixtures: Vec<(&str, Vec<Vec<f64>>)> = vec![
        // 2 groups, NaNs sprinkled in both
        (
            "g2_with_nan",
            vec![
                vec![1.0, 2.0, nan, 4.0, 5.0, 6.0, 7.0, nan, 9.0, 10.0],
                vec![3.0, nan, 5.0, 6.0, 7.0, 8.0, nan, 10.0, 11.0, 12.0],
            ],
        ),
        // 3 groups, NaNs in only one
        (
            "g3_one_nan_group",
            vec![
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                vec![3.0, 4.0, nan, 6.0, nan, 8.0],
                vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            ],
        ),
        // 4 groups, sparse NaNs
        (
            "g4_sparse_nan",
            vec![
                vec![1.0, 2.0, 3.0, 4.0, 5.0],
                vec![10.0, nan, 14.0, 16.0, 18.0],
                vec![5.0, 7.0, nan, 11.0, 13.0],
                vec![20.0, 21.0, 22.0, nan, 24.0],
            ],
        ),
    ];
    let tests = ["levene", "bartlett", "kruskal"];

    let mut points = Vec::new();
    for (name, groups) in &fixtures {
        for test in tests {
            points.push(PointCase {
                case_id: format!("{name}_{test}"),
                test: test.into(),
                groups: groups.clone(),
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
    groups = [np.array(g, dtype=float) for g in case["groups"]]
    val_stat = None; val_p = None
    try:
        if test == "levene":
            res = stats.levene(*groups, nan_policy='omit')
        elif test == "bartlett":
            res = stats.bartlett(*groups, nan_policy='omit')
        elif test == "kruskal":
            res = stats.kruskal(*groups, nan_policy='omit')
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
    let query_json = serde_json::to_string(query).expect("serialize variance_tests_nan query");
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
                "failed to spawn python3 for variance_tests_nan oracle: {e}"
            );
            eprintln!("skipping variance_tests_nan oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open variance_tests_nan oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "variance_tests_nan oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping variance_tests_nan oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for variance_tests_nan oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "variance_tests_nan oracle failed: {stderr}"
        );
        eprintln!("skipping variance_tests_nan oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse variance_tests_nan oracle JSON"))
}

#[test]
fn diff_stats_variance_tests_nan() {
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
        let Some((stat, pval)) = fsci_eval(&case.test, &groups) else {
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
        test_id: "diff_stats_variance_tests_nan".into(),
        category: "scipy.stats levene/bartlett/kruskal nan_policy='omit'".into(),
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
                "variance_tests_nan {} mismatch: {} arm={} abs={}",
                d.test, d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "variance_tests_nan conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
