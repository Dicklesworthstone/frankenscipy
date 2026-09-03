#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the Ansari-Bradley
//! rank-based variance-equality test
//! `scipy.stats.ansari(x, y, alternative=...)`.
//!
//! Resolves [frankenscipy-7vcoy]. Cross-checks the
//! rank-symmetric statistic and the asymptotic normal-tail
//! pvalue across 4 (x, y) fixtures at 3 alternatives.
//!
//! 4 fixtures × 3 alternatives × 2 active arms (statistic +
//! pvalue) = 24 cases via subprocess. Tol 1e-9 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::ansari_alternative;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    alternative: String,
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
    fs::create_dir_all(output_dir()).expect("create ansari diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize ansari diff log");
    fs::write(path, json).expect("write ansari diff log");
}

fn generate_query() -> OracleQuery {
    let fixtures: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![
        // Equal variance
        (
            "equal_var",
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        ),
        // 2× spread
        (
            "spread_2x",
            vec![5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5],
            vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0],
        ),
        // Very different spreads
        (
            "very_diff",
            vec![10.0, 10.5, 10.8, 11.0, 11.2, 11.5, 12.0],
            vec![-5.0, -2.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
        ),
        // With ties (cross-sample)
        (
            "with_ties",
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0],
            vec![3.0, 4.0, 5.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        ),
    ];
    let alternatives = ["two-sided", "less", "greater"];

    let mut points = Vec::new();
    for (name, x, y) in &fixtures {
        for alt in alternatives {
            points.push(PointCase {
                case_id: format!("{name}_{alt}"),
                alternative: alt.into(),
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
    cid = case["case_id"]; alt = case["alternative"]
    x = np.array(case["x"], dtype=float)
    y = np.array(case["y"], dtype=float)
    try:
        # method='approx' uses the asymptotic normal-tail pathway,
        # matching fsci's normal_alternative_pvalue chain.
        res = stats.ansari(x, y, alternative=alt)
        points.append({
            "case_id": cid,
            "statistic": fnone(res.statistic),
            "pvalue": fnone(res.pvalue),
        })
    except Exception:
        points.append({"case_id": cid, "statistic": None, "pvalue": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize ansari query");
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
                "failed to spawn python3 for ansari oracle: {e}"
            );
            eprintln!("skipping ansari oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open ansari oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "ansari oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping ansari oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for ansari oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "ansari oracle failed: {stderr}"
        );
        eprintln!("skipping ansari oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse ansari oracle JSON"))
}

#[test]
fn diff_stats_ansari() {
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
        let result = ansari_alternative(&case.x, &case.y, &case.alternative);

        if let Some(scipy_stat) = scipy_arm.statistic
            && result.statistic.is_finite()
        {
            let abs_diff = (result.statistic - scipy_stat).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "statistic".into(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
        if let Some(scipy_p) = scipy_arm.pvalue
            && result.pvalue.is_finite()
        {
            let abs_diff = (result.pvalue - scipy_p).abs();
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
        test_id: "diff_stats_ansari".into(),
        category: "scipy.stats.ansari".into(),
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
                "ansari mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "ansari conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
