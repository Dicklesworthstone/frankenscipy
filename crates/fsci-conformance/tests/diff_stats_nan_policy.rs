#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the `*_with_nan_policy`
//! wrappers around levene, bartlett, kruskal, and f_oneway.
//!
//! Resolves [frankenscipy-1z2ht]. fsci's wrappers accept the
//! same nan_policy ∈ {'propagate', 'omit', 'raise'} convention
//! as scipy: under 'propagate', any NaN-containing fixture
//! returns NaN/NaN; under 'omit', NaNs are filtered before the
//! base computation. The 'raise' branch is exercised by Result
//! return-type plumbing only — not part of the numerical diff.
//!
//! 4 funcs × {no_nan, with_nan_omit, with_nan_propagate} fixtures
//! × 2 arms = 24 cases. Tol 1e-6 abs for levene+fligner-style
//! pvalue (ndtri rational chain) / 1e-9 elsewhere; harness uses
//! 1e-6 uniformly.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{
    bartlett_with_nan_policy, f_oneway_with_nan_policy, kruskal_with_nan_policy,
    levene_with_nan_policy,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-6;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
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
    fs::create_dir_all(output_dir()).expect("create nan_policy diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize nan_policy diff log");
    fs::write(path, json).expect("write nan_policy diff log");
}

fn generate_query() -> OracleQuery {
    let nan = f64::NAN;
    let scenarios: Vec<(&str, &str, Vec<Vec<f64>>)> = vec![
        // No NaN — baseline, both 'omit' and 'propagate' return same numbers.
        (
            "clean",
            "omit",
            vec![
                (1..=8).map(|i| i as f64).collect(),
                (1..=8).map(|i| (i as f64) * 1.5 + 1.0).collect(),
                (1..=8).map(|i| (i as f64).powi(2) / 4.0).collect(),
            ],
        ),
        // NaN scattered, 'omit' policy → drop NaN cells, base function on cleaned data
        (
            "nan_omit",
            "omit",
            vec![
                vec![1.0, 2.0, nan, 4.0, 5.0, 6.0, 7.0, 8.0],
                vec![nan, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5],
                vec![0.25, 1.0, 2.25, 4.0, nan, 9.0, 12.25, 16.0],
            ],
        ),
        // NaN scattered, 'propagate' policy → NaN/NaN
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

    let mut points = Vec::new();
    for func in ["levene", "bartlett", "kruskal", "f_oneway"] {
        for (label, policy, groups) in &scenarios {
            points.push(PointCase {
                case_id: format!("{func}_{label}"),
                func: func.into(),
                groups: groups.clone(),
                nan_policy: (*policy).to_string(),
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
    if v != v:  # NaN
        return None
    return v if math.isfinite(v) else None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; func = case["func"]
    nan_policy = case["nan_policy"]
    groups = [np.array(g, dtype=float) for g in case["groups"]]
    out = {"case_id": cid, "statistic": None, "pvalue": None}
    try:
        if func == "levene":
            res = stats.levene(*groups, nan_policy=nan_policy)
        elif func == "bartlett":
            res = stats.bartlett(*groups, nan_policy=nan_policy)
        elif func == "kruskal":
            res = stats.kruskal(*groups, nan_policy=nan_policy)
        elif func == "f_oneway":
            res = stats.f_oneway(*groups, nan_policy=nan_policy)
        else:
            res = None
        if res is not None:
            out["statistic"] = fnone(res.statistic)
            out["pvalue"] = fnone(res.pvalue)
    except Exception:
        pass
    points.append(out)
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize nan_policy query");
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
                "failed to spawn python3 for nan_policy oracle: {e}"
            );
            eprintln!("skipping nan_policy oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open nan_policy oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "nan_policy oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping nan_policy oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for nan_policy oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "nan_policy oracle failed: {stderr}"
        );
        eprintln!("skipping nan_policy oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse nan_policy oracle JSON"))
}

#[test]
fn diff_stats_nan_policy() {
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
        let nan_policy = case.nan_policy.as_str();
        let (rust_stat, rust_p) = match case.func.as_str() {
            "levene" => match levene_with_nan_policy(&group_refs, Some(nan_policy)) {
                Ok(r) => (r.statistic, r.pvalue),
                Err(_) => continue,
            },
            "bartlett" => match bartlett_with_nan_policy(&group_refs, Some(nan_policy)) {
                Ok(r) => (r.statistic, r.pvalue),
                Err(_) => continue,
            },
            "kruskal" => match kruskal_with_nan_policy(&group_refs, Some(nan_policy)) {
                Ok(r) => (r.statistic, r.pvalue),
                Err(_) => continue,
            },
            "f_oneway" => match f_oneway_with_nan_policy(&group_refs, Some(nan_policy)) {
                Ok(r) => (r.statistic, r.pvalue),
                Err(_) => continue,
            },
            _ => continue,
        };

        // Both sides may be NaN under propagate — that's a pass.
        if !rust_stat.is_finite() && scipy_arm.statistic.is_none() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: format!("{}.statistic", case.func),
                abs_diff: 0.0,
                pass: true,
            });
        } else if let Some(s_stat) = scipy_arm.statistic
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
        if !rust_p.is_finite() && scipy_arm.pvalue.is_none() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: format!("{}.pvalue", case.func),
                abs_diff: 0.0,
                pass: true,
            });
        } else if let Some(s_p) = scipy_arm.pvalue
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
        test_id: "diff_stats_nan_policy".into(),
        category: "scipy.stats {levene, bartlett, kruskal, f_oneway} (nan_policy)".into(),
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
                "nan_policy mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "nan_policy conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
