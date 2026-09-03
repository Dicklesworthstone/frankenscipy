#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the goodness-of-fit
//! statistics:
//!   • `scipy.stats.chisquare(f_obs, f_exp)`
//!   • `scipy.stats.power_divergence(f_obs, f_exp, lambda_)`
//!
//! Resolves [frankenscipy-cl12n]. chisquare is the special
//! case `power_divergence(lambda=1)`. The harness exercises
//! all four canonical lambdas:
//!   - "pearson"    (1.0)  → Pearson chi²
//!   - "log-likelihood" (0.0) → G²
//!   - "cressie-read"   (2/3) → recommended balance
//!   - "mod-log-likelihood" (-1.0) → modified G²
//!
//! 3 (f_obs, f_exp) fixtures × 4 lambdas × 2 arms (statistic
//! + pvalue) = 24 cases via subprocess. Tol 1e-9 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{chisquare, power_divergence};
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
    fs::create_dir_all(output_dir()).expect("create chisquare_power diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize chisquare_power diff log");
    fs::write(path, json).expect("write chisquare_power diff log");
}

fn generate_query() -> OracleQuery {
    let fixtures: Vec<(&str, Vec<f64>, Option<Vec<f64>>)> = vec![
        // Uniform expected (default behavior)
        (
            "uniform_exp",
            vec![16.0, 18.0, 16.0, 14.0, 12.0, 12.0],
            None,
        ),
        // Custom expected; totals must match (180 == 180)
        (
            "custom_exp",
            vec![43.0, 37.0, 25.0, 39.0, 22.0, 14.0],
            Some(vec![30.0, 30.0, 30.0, 30.0, 30.0, 30.0]),
        ),
        // Larger counts; totals must match (186 == 186)
        (
            "varied",
            vec![89.0, 37.0, 30.0, 28.0, 2.0],
            Some(vec![70.0, 50.0, 30.0, 25.0, 11.0]),
        ),
    ];
    let lambdas: [(&str, f64); 4] = [
        ("pearson", 1.0),
        ("loglike", 0.0),
        ("cressie_read", 2.0 / 3.0),
        ("mod_loglike", -1.0),
    ];

    let mut points = Vec::new();
    for (name, f_obs, f_exp) in &fixtures {
        for (lname, lval) in lambdas {
            points.push(PointCase {
                case_id: format!("{name}_{lname}"),
                f_obs: f_obs.clone(),
                f_exp: f_exp.clone(),
                lambda_: lval,
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
    lam = float(case["lambda_"])
    try:
        if f_exp is None:
            res = stats.power_divergence(f_obs, lambda_=lam)
        else:
            res = stats.power_divergence(f_obs, np.array(f_exp, dtype=float),
                                         lambda_=lam)
        points.append({
            "case_id": cid,
            "statistic": fnone(res.statistic),
            "pvalue": fnone(res.pvalue),
        })
    except Exception:
        points.append({"case_id": cid, "statistic": None, "pvalue": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize chisquare_power query");
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
                "failed to spawn python3 for chisquare_power oracle: {e}"
            );
            eprintln!("skipping chisquare_power oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open chisquare_power oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "chisquare_power oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping chisquare_power oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for chisquare_power oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "chisquare_power oracle failed: {stderr}"
        );
        eprintln!("skipping chisquare_power oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse chisquare_power oracle JSON"))
}

#[test]
fn diff_stats_chisquare_power() {
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
        let exp_ref = case.f_exp.as_deref();
        let (stat, pval) = if (case.lambda_ - 1.0).abs() < 1e-12 {
            // Pearson lambda — also exercise the chisquare entry point.
            let (s_pd, p_pd) = power_divergence(&case.f_obs, exp_ref, case.lambda_);
            let (s_cs, p_cs) = chisquare(&case.f_obs, exp_ref);
            // chisquare and power_divergence(lambda=1) should be identical.
            // Use chisquare's output as the reported statistic so this case
            // also covers the chisquare wrapper.
            let stat_consistent = (s_pd.is_nan() && s_cs.is_nan()) || (s_pd - s_cs).abs() <= 1e-12;
            assert!(
                stat_consistent,
                "chisquare/power_divergence inconsistency: {} vs {}",
                s_pd, s_cs
            );
            let pval_consistent = (p_pd.is_nan() && p_cs.is_nan()) || (p_pd - p_cs).abs() <= 1e-12;
            assert!(
                pval_consistent,
                "chisquare/power_divergence pvalue inconsistency"
            );
            (s_cs, p_cs)
        } else {
            power_divergence(&case.f_obs, exp_ref, case.lambda_)
        };

        if let Some(scipy_stat) = scipy_arm.statistic
            && stat.is_finite()
        {
            let abs_diff = (stat - scipy_stat).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "statistic".into(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
        if let Some(scipy_p) = scipy_arm.pvalue
            && pval.is_finite()
        {
            let abs_diff = (pval - scipy_p).abs();
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
        test_id: "diff_stats_chisquare_power".into(),
        category: "scipy.stats.chisquare/power_divergence".into(),
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
                "chisquare_power mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "chisquare_power conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
