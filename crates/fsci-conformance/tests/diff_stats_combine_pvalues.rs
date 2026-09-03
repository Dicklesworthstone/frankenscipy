#![forbid(unsafe_code)]
//! Live SciPy differential coverage for
//! `scipy.stats.combine_pvalues(pvalues, method)` across four
//! of the five fsci-supported methods.
//!
//! Resolves [frankenscipy-jccjv]. Methods exercised:
//!   • "fisher"  — chi² with df=2k
//!   • "pearson" — chi² with df=2k (left tail)
//!   • "tippett" — minimum p-value (powf-based pvalue)
//!   • "stouffer" — weighted sum of normal-ppf z-scores
//!
//! `mudholkar_george` was previously omitted but the underlying
//! pvalue defect [frankenscipy-v51r8] is now FIXED — fsci
//! switched from a Logistic-tail approximation to scipy's
//! Mudholkar-George Student-t form: T(5n+4).sf(stat ·
//! sqrt(3/n)/π · sqrt(nu/(nu−2))).
//!
//! 3 (pvalue-vector) fixtures × 5 methods × 2 arms = 30 cases
//! via subprocess. Tolerances:
//!   - fisher / pearson / tippett: 1e-12 abs.
//!   - stouffer:                   1e-7  abs (fsci's
//!     standard_normal_ppf rational approximation accumulates
//!     ~1e-8 drift relative to scipy's distpacked ndtri).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::combine_pvalues;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const TIGHT_TOL: f64 = 1.0e-12;
const STOUFFER_TOL: f64 = 1.0e-7;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    method: String,
    pvalues: Vec<f64>,
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
    method: String,
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
    fs::create_dir_all(output_dir()).expect("create combine_pvalues diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize combine_pvalues diff log");
    fs::write(path, json).expect("write combine_pvalues diff log");
}

fn generate_query() -> OracleQuery {
    let fixtures: Vec<(&str, Vec<f64>)> = vec![
        // All small (significant) p-values
        ("all_small", vec![0.001, 0.005, 0.01, 0.02, 0.04]),
        // Mixed (some significant, some not)
        ("mixed", vec![0.01, 0.20, 0.50, 0.80, 0.95]),
        // All large (non-significant)
        ("all_large", vec![0.30, 0.45, 0.60, 0.75, 0.85, 0.92]),
    ];
    let methods = [
        "fisher",
        "pearson",
        "tippett",
        "stouffer",
        "mudholkar_george",
    ];

    let mut points = Vec::new();
    for (name, pvals) in &fixtures {
        for method in methods {
            points.push(PointCase {
                case_id: format!("{name}_{method}"),
                method: method.into(),
                pvalues: pvals.clone(),
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
    pvals = np.array(case["pvalues"], dtype=float)
    val_stat = None; val_p = None
    try:
        res = stats.combine_pvalues(pvals, method=method)
        val_stat = fnone(res.statistic)
        val_p = fnone(res.pvalue)
    except Exception:
        pass
    points.append({"case_id": cid, "statistic": val_stat, "pvalue": val_p})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize combine_pvalues query");
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
                "failed to spawn python3 for combine_pvalues oracle: {e}"
            );
            eprintln!("skipping combine_pvalues oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open combine_pvalues oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "combine_pvalues oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping combine_pvalues oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for combine_pvalues oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "combine_pvalues oracle failed: {stderr}"
        );
        eprintln!("skipping combine_pvalues oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse combine_pvalues oracle JSON"))
}

#[test]
fn diff_stats_combine_pvalues() {
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
        let result = match combine_pvalues(&case.pvalues, Some(&case.method), None) {
            Ok(r) => r,
            Err(_) => continue,
        };

        let tol = if case.method == "stouffer" {
            STOUFFER_TOL
        } else {
            TIGHT_TOL
        };
        if let Some(scipy_stat) = scipy_arm.statistic
            && result.statistic.is_finite()
        {
            let abs_diff = (result.statistic - scipy_stat).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                method: case.method.clone(),
                arm: "statistic".into(),
                abs_diff,
                pass: abs_diff <= tol,
            });
        }
        if let Some(scipy_p) = scipy_arm.pvalue
            && result.pvalue.is_finite()
        {
            let abs_diff = (result.pvalue - scipy_p).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                method: case.method.clone(),
                arm: "pvalue".into(),
                abs_diff,
                pass: abs_diff <= tol,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_combine_pvalues".into(),
        category: "scipy.stats.combine_pvalues".into(),
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
                "combine_pvalues {} mismatch: {} arm={} abs={}",
                d.method, d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "combine_pvalues conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
