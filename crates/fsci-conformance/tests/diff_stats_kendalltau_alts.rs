#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the directional
//! alternatives of `scipy.stats.kendalltau` —
//! `fsci_stats::kendalltau_alternative`.
//!
//! Resolves [frankenscipy-jrre2]. fsci's two-sided default
//! is already covered by diff_stats_correlation_basic.rs;
//! this harness exercises the `'less'` and `'greater'` tails
//! plus the explicit `'two-sided'` value for parity.
//!
//! 4 (x, y) fixtures × 3 alternatives × 2 arms = 24 cases.
//! Tol 1e-12 statistic / 1e-9 pvalue.
//!
//! Note: an earlier version of this harness forced the oracle
//! into `method='asymptotic'` and saw a 0.025 abs gap on the
//! uncorrelated fixture. That was a harness misconfiguration,
//! not an fsci defect — fsci uses the EXACT path for untied
//! n ≤ 25 and the asymptotic path otherwise. scipy's 'auto'
//! method does the same, so the two branches align cleanly
//! when the oracle uses 'auto' instead of forcing 'asymptotic'.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::kendalltau_alternative;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const STAT_TOL: f64 = 1.0e-12;
const PVALUE_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
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
    fs::create_dir_all(output_dir()).expect("create kendalltau_alts diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize kendalltau_alts diff log");
    fs::write(path, json).expect("write kendalltau_alts diff log");
}

fn generate_query() -> OracleQuery {
    // 4 fixtures with varied directional signal.
    let fixtures: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![
        // Strong positive monotone
        (
            "pos_monotone",
            (1..=15).map(|i| i as f64).collect(),
            (1..=15).map(|i| 2.5 * (i as f64) - 1.0).collect(),
        ),
        // Strong negative monotone
        (
            "neg_monotone",
            (1..=12).map(|i| i as f64).collect(),
            (1..=12).map(|i| -((i as f64).powi(2))).collect(),
        ),
        // Weak positive, deterministic noise
        (
            "weak_positive",
            (1..=20).map(|i| i as f64).collect(),
            (1..=20)
                .map(|i| (i as f64) + ((i as f64) * 0.5).sin() * 4.0)
                .collect(),
        ),
        // Uncorrelated permutation (exact path covers this, n=11 untied)
        (
            "uncorrelated",
            vec![5.0, 1.0, 6.0, 2.0, 7.0, 3.0, 8.0, 4.0, 9.0, 0.0, 10.0],
            vec![3.0, 8.0, 1.0, 5.0, 9.0, 2.0, 7.0, 4.0, 6.0, 10.0, 0.0],
        ),
    ];

    let alternatives = ["two-sided", "less", "greater"];
    let mut points = Vec::new();
    for (name, x, y) in &fixtures {
        for alt in &alternatives {
            points.push(PointCase {
                case_id: format!("{name}_{alt}"),
                x: x.clone(),
                y: y.clone(),
                alternative: (*alt).to_string(),
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
    cid = case["case_id"]
    x = np.array(case["x"], dtype=float)
    y = np.array(case["y"], dtype=float)
    alt = case["alternative"]
    stat = None; pval = None
    try:
        # method='auto' matches fsci's branching logic — fsci uses exact
        # for untied n ≤ 25 and asymptotic otherwise. scipy's 'auto'
        # picks exact for untied small samples too, so the two branches
        # align without forcing one method.
        res = stats.kendalltau(x, y, alternative=alt, method="auto")
        stat = fnone(res.statistic); pval = fnone(res.pvalue)
    except Exception:
        pass
    points.append({"case_id": cid, "statistic": stat, "pvalue": pval})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize kendalltau_alts query");
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
                "failed to spawn python3 for kendalltau_alts oracle: {e}"
            );
            eprintln!("skipping kendalltau_alts oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open kendalltau_alts oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "kendalltau_alts oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping kendalltau_alts oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for kendalltau_alts oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "kendalltau_alts oracle failed: {stderr}"
        );
        eprintln!("skipping kendalltau_alts oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse kendalltau_alts oracle JSON"))
}

#[test]
fn diff_stats_kendalltau_alts() {
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
        let result = kendalltau_alternative(&case.x, &case.y, &case.alternative);

        if let Some(s_stat) = scipy_arm.statistic
            && result.statistic.is_finite()
        {
            let abs_diff = (result.statistic - s_stat).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "statistic".into(),
                abs_diff,
                pass: abs_diff <= STAT_TOL,
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
                pass: abs_diff <= PVALUE_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_kendalltau_alts".into(),
        category: "scipy.stats.kendalltau(alternative, method='asymptotic')".into(),
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
                "kendalltau_alts mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "kendalltau_alts conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
