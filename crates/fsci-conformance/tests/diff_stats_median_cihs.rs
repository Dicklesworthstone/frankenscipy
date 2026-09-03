#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci's
//! `median_cihs(data, alpha) → (lo, hi)` — Hettmasperger-
//! Sheather binomial confidence interval for the median.
//!
//! Resolves [frankenscipy-n40oh]. The oracle calls
//! `scipy.stats.mstats.median_cihs(data, alpha=alpha)`.
//!
//! 3 datasets × 3 alphas (0.05, 0.10, 0.20) × 2 arms (lo, hi)
//! = 18 cases via subprocess. Tol 1e-9 abs (binomial-CDF chain
//! via the regularized incomplete beta — well below scipy's
//! own implementation noise but loose enough to absorb our
//! ndtri rational approximation).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::median_cihs;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    data: Vec<f64>,
    alpha: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    lo: Option<f64>,
    hi: Option<f64>,
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
    fs::create_dir_all(output_dir()).expect("create median_cihs diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize median_cihs diff log");
    fs::write(path, json).expect("write median_cihs diff log");
}

fn generate_query() -> OracleQuery {
    let datasets: Vec<(&str, Vec<f64>)> = vec![
        ("compact_n15", (1..=15).map(|i| i as f64).collect()),
        (
            "spread_n25",
            (1..=25).map(|i| (i as f64).powi(2) / 5.0).collect(),
        ),
        (
            "ties_n20",
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 6.0, 6.0, 7.0,
                7.0, 8.0, 9.0, 10.0,
            ],
        ),
    ];
    let alphas = vec![0.05, 0.10, 0.20];

    let mut points = Vec::new();
    for (name, data) in &datasets {
        for &alpha in &alphas {
            points.push(PointCase {
                case_id: format!("{name}_a{}", (alpha * 100.0_f64).round() as u32),
                data: data.clone(),
                alpha,
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
from scipy.stats import mstats

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
    data = np.array(case["data"], dtype=float)
    alpha = float(case["alpha"])
    out = {"case_id": cid, "lo": None, "hi": None}
    try:
        lo, hi = mstats.median_cihs(data, alpha=alpha)
        out["lo"] = fnone(lo)
        out["hi"] = fnone(hi)
    except Exception:
        pass
    points.append(out)
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize median_cihs query");
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
                "failed to spawn python3 for median_cihs oracle: {e}"
            );
            eprintln!("skipping median_cihs oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open median_cihs oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "median_cihs oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping median_cihs oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for median_cihs oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "median_cihs oracle failed: {stderr}"
        );
        eprintln!("skipping median_cihs oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse median_cihs oracle JSON"))
}

#[test]
fn diff_stats_median_cihs() {
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
        let (rust_lo, rust_hi) = median_cihs(&case.data, case.alpha);

        if let Some(s_lo) = scipy_arm.lo
            && rust_lo.is_finite()
        {
            let abs_diff = (rust_lo - s_lo).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "lo".into(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
        if let Some(s_hi) = scipy_arm.hi
            && rust_hi.is_finite()
        {
            let abs_diff = (rust_hi - s_hi).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "hi".into(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_median_cihs".into(),
        category: "scipy.stats.mstats.median_cihs".into(),
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
                "median_cihs mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "median_cihs conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
