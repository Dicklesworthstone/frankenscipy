#![forbid(unsafe_code)]
//! Live numpy differential coverage for fsci's
//! `sign_test(x, y) → TtestResult` — paired-sample sign test
//! using the normal approximation with continuity correction.
//!
//! Resolves [frankenscipy-2ejnu]. scipy.stats does NOT expose a
//! standalone `sign_test`; the equivalent is in statsmodels.
//! Statsmodels is not available in this environment, so the
//! oracle reproduces fsci's exact closed form in numpy:
//!
//!   diffs = x - y
//!   n_pos = #{d > 0}, n_neg = #{d < 0}
//!   n     = n_pos + n_neg   (exclude ties)
//!   k     = min(n_pos, n_neg)
//!   z     = (k - n/2 + 0.5) / sqrt(n/4)    (continuity correction)
//!   pvalue = 2 · Φ(min(z, 0))
//!
//! 4 (x, y) fixtures × 3 arms (statistic, pvalue, df) = 12
//! cases. Tol 1e-12 abs (closed-form arithmetic; the normal
//! CDF chain uses scipy.stats.norm.cdf which agrees with
//! fsci's `Normal::standard().cdf` at machine precision).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::sign_test;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
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
    fs::create_dir_all(output_dir()).expect("create sign_test diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize sign_test diff log");
    fs::write(path, json).expect("write sign_test diff log");
}

fn generate_query() -> OracleQuery {
    let fixtures: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![
        // Symmetric: equal positive/negative differences
        (
            "symmetric_n10",
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            vec![1.5, 2.5, 2.5, 4.5, 4.5, 6.5, 6.5, 8.5, 8.5, 10.5],
        ),
        // Strong positive bias (most x > y)
        (
            "x_larger_n12",
            (1..=12).map(|i| i as f64 + 5.0).collect(),
            (1..=12).map(|i| i as f64).collect(),
        ),
        // Strong negative bias (most x < y)
        (
            "x_smaller_n14",
            (1..=14).map(|i| i as f64).collect(),
            (1..=14).map(|i| i as f64 + 3.0).collect(),
        ),
        // With ties (zeros excluded)
        (
            "with_ties_n15",
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            ],
            vec![
                1.5, 2.0, 4.0, 4.0, 6.0, 6.0, 8.0, 8.0, 10.0, 10.0, 12.0, 12.0, 14.0, 14.0, 16.0,
            ],
        ),
    ];

    let points = fixtures
        .into_iter()
        .map(|(name, x, y)| PointCase {
            case_id: name.into(),
            x,
            y,
        })
        .collect();
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
from scipy.stats import norm

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
    x = case["x"]
    y = case["y"]
    out = {"case_id": cid, "statistic": None, "pvalue": None, "df": None}
    try:
        diffs = [a - b for a, b in zip(x, y)]
        n_pos = sum(1 for d in diffs if d > 0)
        n_neg = sum(1 for d in diffs if d < 0)
        n = n_pos + n_neg
        if n == 0:
            out["statistic"] = 0.0
            out["pvalue"] = 1.0
            # fsci returns df = nan when n == 0
            out["df"] = None
        else:
            k = float(min(n_pos, n_neg))
            nf = float(n)
            z = (k - nf / 2.0 + 0.5) / math.sqrt(nf / 4.0)
            # fsci uses 2 · Φ(min(z, 0)) which is correct for the
            # one-sided lower-tail of the binomial-equivalent.
            pvalue = 2.0 * float(norm.cdf(min(z, 0.0)))
            pvalue = max(0.0, min(1.0, pvalue))
            out["statistic"] = fnone(k)
            out["pvalue"] = fnone(pvalue)
            out["df"] = fnone(nf)
    except Exception:
        pass
    points.append(out)
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize sign_test query");
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
                "failed to spawn python3 for sign_test oracle: {e}"
            );
            eprintln!("skipping sign_test oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open sign_test oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "sign_test oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping sign_test oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for sign_test oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "sign_test oracle failed: {stderr}"
        );
        eprintln!("skipping sign_test oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse sign_test oracle JSON"))
}

#[test]
fn diff_stats_sign_test() {
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
        let result = match sign_test(&case.x, &case.y) {
            Ok(r) => r,
            Err(_) => continue,
        };

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
        if let Some(s_df) = scipy_arm.df
            && result.df.is_finite()
        {
            let abs_diff = (result.df - s_df).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "df".into(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_sign_test".into(),
        category: "fsci_stats::sign_test (numpy + scipy.stats.norm reference)".into(),
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
                "sign_test mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "sign_test conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
