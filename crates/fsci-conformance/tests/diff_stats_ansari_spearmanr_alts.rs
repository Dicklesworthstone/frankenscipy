#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the directional
//! alternatives of `scipy.stats.spearmanr` and
//! `scipy.stats.ansari`.
//!
//! Resolves [frankenscipy-3wlbl]. fsci's two-sided defaults
//! are already covered elsewhere; this harness exercises the
//! `'less'` and `'greater'` tails plus the explicit two-sided
//! variant for parity.
//!
//! 4 fixtures × 2 funcs (spearmanr_alt, ansari_alt) × 3
//! alternatives × 2 arms = 48 cases. Tol 1e-9 abs (Student-t-
//! tail chain via betainc; ansari exact distribution via
//! polynomial-product DP).
//!
//! Surfaced and fixed [frankenscipy-yy9f3]: pearsonr_alternative
//! and spearmanr_alternative both returned pvalue = 0 for all
//! tails when |r| = 1. Fixed inline by dispatching on the
//! alternative at the perfect-correlation boundary.
//!
//! Surfaced and fixed [frankenscipy-1vlpt]: ansari_alternative
//! used the standard-normal asymptotic for the pvalue while
//! scipy uses the exact convolution-recursion distribution for
//! small n + m < 55 (no ties). Implemented the exact path via
//! a 2-D polynomial-product DP over symmetric ranks; ansari_alt
//! is now re-enabled in this harness.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{ansari_alternative, spearmanr_alternative};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
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
    fs::create_dir_all(output_dir()).expect("create ansari_spearmanr_alts diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize ansari_spearmanr_alts diff log");
    fs::write(path, json).expect("write ansari_spearmanr_alts diff log");
}

fn generate_query() -> OracleQuery {
    // Two of the fixtures should produce strong directional signal so the
    // 'less' / 'greater' arms aren't degenerate (~0.5).
    let fixtures: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![
        // Equal scale / weak correlation — 2-sided will be near 1
        (
            "balanced",
            (1..=14).map(|i| i as f64).collect(),
            (1..=14)
                .map(|i| (i as f64) + 0.3 * ((i as f64) * 0.5).sin())
                .collect(),
        ),
        // Different scale, monotone (Ansari signal + strong Spearman ρ)
        (
            "diff_scale_monotone",
            vec![
                4.0, 5.0, 4.5, 5.2, 4.8, 5.1, 4.9, 5.3, 4.7, 5.0, 4.6, 5.4, 4.5, 5.5,
            ],
            vec![
                -2.0, 12.0, -1.0, 11.0, 0.0, 10.0, 1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0,
            ],
        ),
        // Strong negative monotone
        (
            "neg_monotone",
            (1..=15).map(|i| i as f64).collect(),
            (1..=15).map(|i| -((i as f64).powi(2))).collect(),
        ),
        // Strong positive monotone
        (
            "pos_monotone",
            (1..=18).map(|i| i as f64).collect(),
            (1..=18).map(|i| 2.5 * (i as f64) - 1.0).collect(),
        ),
    ];

    let alternatives = ["two-sided", "less", "greater"];
    let mut points = Vec::new();
    for (name, x, y) in &fixtures {
        for func in ["spearmanr_alt", "ansari_alt"] {
            for alt in &alternatives {
                points.push(PointCase {
                    case_id: format!("{name}_{func}_{alt}"),
                    func: func.to_string(),
                    x: x.clone(),
                    y: y.clone(),
                    alternative: (*alt).to_string(),
                });
            }
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
    cid = case["case_id"]; func = case["func"]
    x = np.array(case["x"], dtype=float)
    y = np.array(case["y"], dtype=float)
    alt = case["alternative"]
    stat = None; pval = None
    try:
        if func == "spearmanr_alt":
            res = stats.spearmanr(x, y, alternative=alt)
        elif func == "ansari_alt":
            res = stats.ansari(x, y, alternative=alt)
        else:
            res = None
        if res is not None:
            stat = fnone(res.statistic); pval = fnone(res.pvalue)
    except Exception:
        pass
    points.append({"case_id": cid, "statistic": stat, "pvalue": pval})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize ansari_spearmanr_alts query");
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
                "failed to spawn python3 for ansari_spearmanr_alts oracle: {e}"
            );
            eprintln!("skipping ansari_spearmanr_alts oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open ansari_spearmanr_alts oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "ansari_spearmanr_alts oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!(
                "skipping ansari_spearmanr_alts oracle: stdin write failed ({err})\n{stderr}"
            );
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for ansari_spearmanr_alts oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "ansari_spearmanr_alts oracle failed: {stderr}"
        );
        eprintln!("skipping ansari_spearmanr_alts oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse ansari_spearmanr_alts oracle JSON"))
}

#[test]
fn diff_stats_ansari_spearmanr_alts() {
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
        let (rust_stat, rust_p) = match case.func.as_str() {
            "spearmanr_alt" => {
                let r = spearmanr_alternative(&case.x, &case.y, &case.alternative);
                (r.statistic, r.pvalue)
            }
            "ansari_alt" => {
                let r = ansari_alternative(&case.x, &case.y, &case.alternative);
                (r.statistic, r.pvalue)
            }
            _ => continue,
        };

        if let Some(s_stat) = scipy_arm.statistic
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
        if let Some(s_p) = scipy_arm.pvalue
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
        test_id: "diff_stats_ansari_spearmanr_alts".into(),
        category: "scipy.stats.spearmanr(alternative)".into(),
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
                "ansari_spearmanr_alts mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "ansari_spearmanr_alts conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
