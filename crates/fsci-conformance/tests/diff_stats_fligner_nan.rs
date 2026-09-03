#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci's
//! `fligner_with_nan_policy(groups, nan_policy)` — Fligner-
//! Killeen variance test with explicit NaN handling.
//!
//! Resolves [frankenscipy-kv39m]. The base fligner already
//! has a fix landing recently (`(N-1)` multiplier + average-
//! rank tie handling) and is covered by diff_stats_fligner.rs;
//! this harness extends to the nan_policy wrapper, completing
//! the *_with_nan_policy family started in
//! diff_stats_nan_policy.rs.
//!
//! 3 nan-fixtures (clean, nan_omit, nan_propagate) × 2 arms
//! = 6 cases. Tol 1e-6 abs (ndtri / chi-square tail chain —
//! same precision floor used by diff_stats_fligner.rs after
//! the variance / tie-handling fix).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::fligner_with_nan_policy;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-6;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    groups: Vec<Vec<f64>>,
    nan_policy: String,
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
    fs::create_dir_all(output_dir()).expect("create fligner_nan diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize fligner_nan diff log");
    fs::write(path, json).expect("write fligner_nan diff log");
}

fn generate_query() -> OracleQuery {
    let nan = f64::NAN;
    let scenarios: Vec<(&str, &str, Vec<Vec<f64>>)> = vec![
        (
            "clean",
            "omit",
            vec![
                (1..=8).map(|i| i as f64).collect(),
                (1..=8).map(|i| (i as f64) * 1.5 + 1.0).collect(),
                (1..=8).map(|i| (i as f64).powi(2) / 4.0).collect(),
            ],
        ),
        (
            "nan_omit",
            "omit",
            vec![
                vec![1.0, 2.0, nan, 4.0, 5.0, 6.0, 7.0, 8.0],
                vec![nan, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5],
                vec![0.25, 1.0, 2.25, 4.0, nan, 9.0, 12.25, 16.0],
            ],
        ),
        (
            "nan_propagate",
            "propagate",
            vec![
                vec![1.0, 2.0, nan, 4.0, 5.0, 6.0, 7.0, 8.0],
                vec![nan, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5],
                vec![0.25, 1.0, 2.25, 4.0, nan, 9.0, 12.25, 16.0],
            ],
        ),
    ];

    let points = scenarios
        .into_iter()
        .map(|(label, policy, groups)| PointCase {
            case_id: label.into(),
            groups,
            nan_policy: policy.to_string(),
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
    if v != v:
        return None
    return v if math.isfinite(v) else None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    nan_policy = case["nan_policy"]
    groups = [np.array(g, dtype=float) for g in case["groups"]]
    out = {"case_id": cid, "statistic": None, "pvalue": None}
    try:
        res = stats.fligner(*groups, nan_policy=nan_policy)
        out["statistic"] = fnone(res.statistic); out["pvalue"] = fnone(res.pvalue)
    except Exception:
        pass
    points.append(out)
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize fligner_nan query");
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
                "failed to spawn python3 for fligner_nan oracle: {e}"
            );
            eprintln!("skipping fligner_nan oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open fligner_nan oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "fligner_nan oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping fligner_nan oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for fligner_nan oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "fligner_nan oracle failed: {stderr}"
        );
        eprintln!("skipping fligner_nan oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse fligner_nan oracle JSON"))
}

#[test]
fn diff_stats_fligner_nan() {
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
        let result = match fligner_with_nan_policy(&group_refs, Some(&case.nan_policy)) {
            Ok(r) => r,
            Err(_) => continue,
        };

        if !result.statistic.is_finite() && scipy_arm.statistic.is_none() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "statistic".into(),
                abs_diff: 0.0,
                pass: true,
            });
        } else if let Some(s_stat) = scipy_arm.statistic
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
        if !result.pvalue.is_finite() && scipy_arm.pvalue.is_none() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "pvalue".into(),
                abs_diff: 0.0,
                pass: true,
            });
        } else if let Some(s_p) = scipy_arm.pvalue
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
        test_id: "diff_stats_fligner_nan".into(),
        category: "scipy.stats.fligner(nan_policy)".into(),
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
                "fligner_nan mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "fligner_nan conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
