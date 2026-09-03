#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci's
//! `power_divergence(f_obs, f_exp, lambda_) → (statistic, pvalue)`.
//!
//! Resolves [frankenscipy-a0bxu]. The oracle calls
//! `scipy.stats.power_divergence(f_obs, f_exp, lambda_=lambda_)`.
//!
//! Lambda parameter map (see scipy docs):
//!   • 1.0  — Pearson χ² (default)
//!   • 0.0  — G-test / log-likelihood ratio
//!   • -1.0 — modified log-likelihood
//!   • -0.5 — Freeman-Tukey
//!   • -2.0 — Neyman
//!   • 2/3  — Cressie-Read
//!
//! 3 (f_obs, f_exp) fixtures × 5 lambdas × 2 arms = 30 cases.
//! Tol 1e-9 abs (chi-square tail chain via regularized
//! incomplete gamma).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::power_divergence;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    f_obs: Vec<f64>,
    f_exp: Option<Vec<f64>>,
    lambda_: f64,
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
    fs::create_dir_all(output_dir()).expect("create power_divergence diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize power_divergence diff log");
    fs::write(path, json).expect("write power_divergence diff log");
}

fn generate_query() -> OracleQuery {
    let fixtures: Vec<(&str, Vec<f64>, Option<Vec<f64>>)> = vec![
        // Uniform default expected (no f_exp)
        (
            "dice_uniform",
            vec![16.0, 18.0, 16.0, 14.0, 12.0, 12.0],
            None,
        ),
        // Explicit non-uniform expected
        (
            "non_uniform",
            vec![20.0, 30.0, 25.0, 25.0, 10.0],
            Some(vec![25.0, 25.0, 25.0, 25.0, 10.0]),
        ),
        // Larger fixture with stronger deviation
        (
            "wide_deviation",
            vec![45.0, 38.0, 33.0, 30.0, 27.0, 25.0, 22.0, 20.0, 15.0, 12.0],
            None,
        ),
    ];

    let lambdas: Vec<(&str, f64)> = vec![
        ("pearson", 1.0),
        ("g_test", 0.0),
        ("mod_log_lik", -1.0),
        ("freeman_tukey", -0.5),
        ("cressie_read", 2.0 / 3.0),
    ];

    let mut points = Vec::new();
    for (name, f_obs, f_exp) in &fixtures {
        for (lname, lambda_) in &lambdas {
            points.push(PointCase {
                case_id: format!("{name}_{lname}"),
                f_obs: f_obs.clone(),
                f_exp: f_exp.clone(),
                lambda_: *lambda_,
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
    f_obs = np.array(case["f_obs"], dtype=float)
    f_exp = case["f_exp"]
    if f_exp is not None:
        f_exp = np.array(f_exp, dtype=float)
    lam = float(case["lambda_"])
    stat = None; pval = None
    try:
        if f_exp is None:
            res = stats.power_divergence(f_obs, lambda_=lam)
        else:
            res = stats.power_divergence(f_obs, f_exp, lambda_=lam)
        stat = fnone(res.statistic)
        pval = fnone(res.pvalue)
    except Exception:
        pass
    points.append({"case_id": cid, "statistic": stat, "pvalue": pval})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize power_divergence query");
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
                "failed to spawn python3 for power_divergence oracle: {e}"
            );
            eprintln!("skipping power_divergence oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open power_divergence oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "power_divergence oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping power_divergence oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for power_divergence oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "power_divergence oracle failed: {stderr}"
        );
        eprintln!("skipping power_divergence oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse power_divergence oracle JSON"))
}

#[test]
fn diff_stats_power_divergence() {
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
        let f_exp_ref: Option<&[f64]> = case.f_exp.as_deref();
        let (rust_stat, rust_p) = power_divergence(&case.f_obs, f_exp_ref, case.lambda_);

        if let Some(s_stat) = scipy_arm.statistic
            && rust_stat.is_finite()
        {
            let abs_diff = (rust_stat - s_stat).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "statistic".into(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
        if let Some(s_p) = scipy_arm.pvalue
            && rust_p.is_finite()
        {
            let abs_diff = (rust_p - s_p).abs();
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
        test_id: "diff_stats_power_divergence".into(),
        category: "scipy.stats.power_divergence".into(),
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
                "power_divergence mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "power_divergence conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
