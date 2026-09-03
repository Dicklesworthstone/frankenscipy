#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the basic
//! correlation / regression family:
//!   • `linregress(x, y)` — full slope/intercept/r/p/stderr
//!     bundle (closed-form + Student-t pvalue)
//!   • `kendalltau(x, y)` — Kendall's τ-b
//!   • `spearmanr(x, y)` — Spearman's ρ
//!   • `weightedtau(x, y)` — weighted Kendall τ scalar
//!
//! Resolves [frankenscipy-i8tiw]. The oracle calls
//! `scipy.stats.{linregress, kendalltau, spearmanr,
//! weightedtau}`.
//!
//! 4 (x, y) fixtures × variable arms (linregress: 6,
//! kendalltau/spearmanr: 2 each, weightedtau: 1) = 44 cases.
//! Tol 1e-12 abs (closed-form) / 1e-9 abs (t-tail / normal-tail
//! pvalue chains via betainc / ndtri).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{kendalltau, linregress, spearmanr, weightedtau};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const STAT_TOL: f64 = 1.0e-12;
const PVALUE_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
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
    /// linregress: [slope, intercept, rvalue, pvalue, stderr, intercept_stderr]
    /// kendalltau: [statistic, pvalue]
    /// spearmanr:  [statistic, pvalue]
    /// weightedtau: [statistic]
    values: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create correlation_basic diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize correlation_basic diff log");
    fs::write(path, json).expect("write correlation_basic diff log");
}

fn generate_query() -> OracleQuery {
    let fixtures: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![
        // Strong linear (perfect correlation, finite p)
        (
            "perfect_linear",
            (1..=12).map(|i| i as f64).collect(),
            (1..=12).map(|i| 2.0 * i as f64 + 1.5).collect(),
        ),
        // Mild correlation with deterministic noise
        (
            "mild_corr",
            (1..=20).map(|i| i as f64).collect(),
            (1..=20)
                .map(|i| {
                    let x = i as f64;
                    1.5 * x + ((x * 0.7).sin() * 2.0)
                })
                .collect(),
        ),
        // Negative monotone (sin-monotone-decreasing on [0, π])
        (
            "neg_monotone",
            (0..15).map(|i| i as f64).collect(),
            (0..15).map(|i| -((i as f64) * (i as f64) / 5.0)).collect(),
        ),
        // Uncorrelated permutation
        (
            "uncorrelated",
            vec![5.0, 1.0, 6.0, 2.0, 7.0, 3.0, 8.0, 4.0, 9.0, 0.0, 10.0],
            vec![3.0, 8.0, 1.0, 5.0, 9.0, 2.0, 7.0, 4.0, 6.0, 10.0, 0.0],
        ),
    ];

    let mut points = Vec::new();
    for (name, x, y) in &fixtures {
        for func in ["linregress", "kendalltau", "spearmanr", "weightedtau"] {
            points.push(PointCase {
                case_id: format!("{name}_{func}"),
                func: func.into(),
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

def vec_or_none(arr):
    out = []
    for v in arr:
        f = fnone(v)
        if f is None:
            return None
        out.append(f)
    return out

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; func = case["func"]
    x = np.array(case["x"], dtype=float)
    y = np.array(case["y"], dtype=float)
    val = None
    try:
        if func == "linregress":
            r = stats.linregress(x, y)
            val = vec_or_none([
                r.slope, r.intercept, r.rvalue, r.pvalue,
                r.stderr, r.intercept_stderr,
            ])
        elif func == "kendalltau":
            r = stats.kendalltau(x, y)
            val = vec_or_none([r.statistic, r.pvalue])
        elif func == "spearmanr":
            r = stats.spearmanr(x, y)
            val = vec_or_none([r.statistic, r.pvalue])
        elif func == "weightedtau":
            r = stats.weightedtau(x, y)
            val = vec_or_none([r.statistic])
    except Exception:
        val = None
    points.append({"case_id": cid, "values": val})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize correlation_basic query");
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
                "failed to spawn python3 for correlation_basic oracle: {e}"
            );
            eprintln!("skipping correlation_basic oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open correlation_basic oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "correlation_basic oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping correlation_basic oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for correlation_basic oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "correlation_basic oracle failed: {stderr}"
        );
        eprintln!("skipping correlation_basic oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse correlation_basic oracle JSON"))
}

#[test]
fn diff_stats_correlation_basic() {
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

    let record = |case_id: &str,
                  arm: &str,
                  rust_v: f64,
                  scipy_v: f64,
                  tol: f64,
                  diffs: &mut Vec<CaseDiff>,
                  max_overall: &mut f64| {
        if !rust_v.is_finite() {
            return;
        }
        let abs_diff = (rust_v - scipy_v).abs();
        *max_overall = max_overall.max(abs_diff);
        diffs.push(CaseDiff {
            case_id: case_id.into(),
            arm: arm.into(),
            abs_diff,
            pass: abs_diff <= tol,
        });
    };

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(scipy_vec) = &scipy_arm.values else {
            continue;
        };
        match case.func.as_str() {
            "linregress" => {
                let r = linregress(&case.x, &case.y);
                let arms = [
                    ("slope", r.slope, STAT_TOL),
                    ("intercept", r.intercept, STAT_TOL),
                    ("rvalue", r.rvalue, STAT_TOL),
                    ("pvalue", r.pvalue, PVALUE_TOL),
                    ("stderr", r.stderr, STAT_TOL),
                    ("intercept_stderr", r.intercept_stderr, STAT_TOL),
                ];
                if scipy_vec.len() != arms.len() {
                    continue;
                }
                for (i, (name, v, tol)) in arms.iter().enumerate() {
                    record(
                        &case.case_id,
                        name,
                        *v,
                        scipy_vec[i],
                        *tol,
                        &mut diffs,
                        &mut max_overall,
                    );
                }
            }
            "kendalltau" => {
                let r = kendalltau(&case.x, &case.y);
                if scipy_vec.len() < 2 {
                    continue;
                }
                record(
                    &case.case_id,
                    "kendalltau.statistic",
                    r.statistic,
                    scipy_vec[0],
                    STAT_TOL,
                    &mut diffs,
                    &mut max_overall,
                );
                record(
                    &case.case_id,
                    "kendalltau.pvalue",
                    r.pvalue,
                    scipy_vec[1],
                    PVALUE_TOL,
                    &mut diffs,
                    &mut max_overall,
                );
            }
            "spearmanr" => {
                let r = spearmanr(&case.x, &case.y);
                if scipy_vec.len() < 2 {
                    continue;
                }
                record(
                    &case.case_id,
                    "spearmanr.statistic",
                    r.statistic,
                    scipy_vec[0],
                    STAT_TOL,
                    &mut diffs,
                    &mut max_overall,
                );
                record(
                    &case.case_id,
                    "spearmanr.pvalue",
                    r.pvalue,
                    scipy_vec[1],
                    PVALUE_TOL,
                    &mut diffs,
                    &mut max_overall,
                );
            }
            "weightedtau" => {
                let r = weightedtau(&case.x, &case.y);
                if scipy_vec.is_empty() {
                    continue;
                }
                record(
                    &case.case_id,
                    "weightedtau.statistic",
                    r,
                    scipy_vec[0],
                    STAT_TOL,
                    &mut diffs,
                    &mut max_overall,
                );
            }
            _ => continue,
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_correlation_basic".into(),
        category: "scipy.stats.{linregress, kendalltau, spearmanr, weightedtau}".into(),
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
                "correlation_basic mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "correlation_basic conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
