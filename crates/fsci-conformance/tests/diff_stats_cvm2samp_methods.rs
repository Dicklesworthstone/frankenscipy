#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci's
//! `cramervonmises_2samp_with_method(x, y, method)` —
//! explicit p-value method selection (Exact / Asymptotic) for
//! the two-sample Cramér-von Mises test.
//!
//! Resolves [frankenscipy-5tojt]. The existing
//! diff_stats_two_sample_extras.rs covers Auto-method only.
//! This harness exercises the two non-default branches against
//! `scipy.stats.cramervonmises_2samp(x, y, method='exact')`
//! and `method='asymptotic'`.
//!
//! 4 (x, y) fixtures × 2 methods × 2 arms = 16 cases. Per-arm
//! tols: 1e-9 statistic; 5e-3 asymptotic pvalue (Bessel-K
//! series floor — same precision floor we use for the 1-sample
//! cvm harness); 5e-2 exact pvalue (combinatorial enumeration
//! can differ on small n with ties between fsci's branch and
//! scipy's).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{Cvm2SampleMethod, cramervonmises_2samp_with_method};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const STAT_TOL: f64 = 1.0e-9;
const ASYMP_PVALUE_TOL: f64 = 5.0e-3;
const EXACT_PVALUE_TOL: f64 = 5.0e-2;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    method: String,
    x: Vec<f64>,
    y: Vec<f64>,
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
    fs::create_dir_all(output_dir()).expect("create cvm2samp_methods diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize cvm2samp_methods diff log");
    fs::write(path, json).expect("write cvm2samp_methods diff log");
}

fn generate_query() -> OracleQuery {
    let fixtures: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![
        // Identical-shape (slight shift)
        (
            "balanced_n10",
            (1..=10).map(|i| i as f64).collect(),
            (1..=10).map(|i| (i as f64) + 0.5).collect(),
        ),
        // Strong shift
        (
            "shifted_n12",
            (1..=12).map(|i| i as f64).collect(),
            (5..=16).map(|i| i as f64).collect(),
        ),
        // Different scales, same center
        (
            "diff_scale_n14",
            vec![
                4.0, 5.0, 4.5, 5.2, 4.8, 5.1, 4.9, 5.3, 4.7, 5.0, 4.6, 5.4, 4.5, 5.5,
            ],
            vec![
                -2.0, 12.0, -1.0, 11.0, 0.0, 10.0, 1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0,
            ],
        ),
        // Right-skewed vs near-normal
        (
            "skewed_n16",
            (1..=16).map(|i| (i as f64).sqrt() * 4.0).collect(),
            (1..=16).map(|i| i as f64).collect(),
        ),
    ];

    let mut points = Vec::new();
    for (name, x, y) in &fixtures {
        for method in ["exact", "asymptotic"] {
            points.push(PointCase {
                case_id: format!("{name}_{method}"),
                method: method.into(),
                x: x.clone(),
                y: y.clone(),
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
    cid = case["case_id"]; method = case["method"]
    x = np.array(case["x"], dtype=float)
    y = np.array(case["y"], dtype=float)
    stat = None; pval = None
    try:
        res = stats.cramervonmises_2samp(x, y, method=method)
        stat = fnone(res.statistic); pval = fnone(res.pvalue)
    except Exception:
        pass
    points.append({"case_id": cid, "statistic": stat, "pvalue": pval})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize cvm2samp_methods query");
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
                "failed to spawn python3 for cvm2samp_methods oracle: {e}"
            );
            eprintln!("skipping cvm2samp_methods oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open cvm2samp_methods oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "cvm2samp_methods oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping cvm2samp_methods oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for cvm2samp_methods oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "cvm2samp_methods oracle failed: {stderr}"
        );
        eprintln!("skipping cvm2samp_methods oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse cvm2samp_methods oracle JSON"))
}

#[test]
fn diff_stats_cvm2samp_methods() {
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
        let method = match case.method.as_str() {
            "exact" => Cvm2SampleMethod::Exact,
            "asymptotic" => Cvm2SampleMethod::Asymptotic,
            _ => continue,
        };
        let result = cramervonmises_2samp_with_method(&case.x, &case.y, method);
        let pvalue_tol = if case.method == "exact" {
            EXACT_PVALUE_TOL
        } else {
            ASYMP_PVALUE_TOL
        };

        if let Some(s_stat) = scipy_arm.statistic
            && result.statistic.is_finite()
        {
            let abs_diff = (result.statistic - s_stat).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: format!("{}.statistic", case.method),
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
                arm: format!("{}.pvalue", case.method),
                abs_diff,
                pass: abs_diff <= pvalue_tol,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_cvm2samp_methods".into(),
        category: "scipy.stats.cramervonmises_2samp(method=exact|asymptotic)".into(),
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
                "cvm2samp_methods mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "cvm2samp_methods conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
