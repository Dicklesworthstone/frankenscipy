#![forbid(unsafe_code)]
//! Live numerical reference checks for the Mann-Kendall trend
//! test `mannkendall(data)`.
//!
//! Resolves [frankenscipy-kgsr7]. fsci returns a tuple
//! `(tau, pvalue, trend_direction)` where `trend_direction`
//! ∈ {-1, 0, +1}. The oracle reproduces all three quantities
//! analytically in numpy + scipy.stats.norm.cdf using fsci's
//! exact normal-approximation pvalue with continuity
//! correction (S±1 in the numerator).
//!
//! 4 datasets × 3 arms = 12 cases via subprocess.
//! Tolerances:
//!   - tau          : 1e-12 abs (closed-form ratio)
//!   - pvalue       : 1e-9 abs  (normal cdf chain)
//!   - trend_dir    : 1e-12 abs (integer compare)

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::mannkendall;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const TAU_TOL: f64 = 1.0e-12;
const PVALUE_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    data: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    tau: Option<f64>,
    pvalue: Option<f64>,
    trend: Option<i64>,
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
    fs::create_dir_all(output_dir()).expect("create mannkendall diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize mannkendall diff log");
    fs::write(path, json).expect("write mannkendall diff log");
}

fn generate_query() -> OracleQuery {
    let fixtures: Vec<(&str, Vec<f64>)> = vec![
        // Strong increasing trend
        ("increasing", (1..=15).map(|i| i as f64).collect()),
        // Strong decreasing trend
        ("decreasing", (1..=15).rev().map(|i| i as f64).collect()),
        // No trend (alternating)
        (
            "no_trend",
            vec![5.0, 1.0, 6.0, 2.0, 7.0, 3.0, 8.0, 4.0, 9.0, 0.0, 10.0],
        ),
        // Mild upward trend with some reversals
        (
            "noisy_up",
            vec![
                1.0, 1.5, 1.2, 2.0, 1.8, 2.5, 2.3, 3.0, 2.8, 3.5, 3.2, 4.0, 3.8, 4.5, 4.2,
            ],
        ),
    ];

    let points = fixtures
        .into_iter()
        .map(|(name, data)| PointCase {
            case_id: name.into(),
            data,
        })
        .collect();
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
    data = np.array(case["data"], dtype=float)
    out = {"case_id": cid, "tau": None, "pvalue": None, "trend": None}
    try:
        n = len(data)
        if n < 3:
            points.append(out)
            continue
        s = 0
        for i in range(n):
            for j in range(i + 1, n):
                if data[j] > data[i]:
                    s += 1
                elif data[j] < data[i]:
                    s -= 1
        pairs = n * (n - 1) // 2
        tau = s / pairs
        var_s = n * (n - 1) * (2 * n + 5) / 18.0
        # Continuity correction matches fsci.
        if s > 0:
            z = (s - 1.0) / math.sqrt(var_s)
        elif s < 0:
            z = (s + 1.0) / math.sqrt(var_s)
        else:
            z = 0.0
        pvalue = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
        trend = 1 if s > 0 else (-1 if s < 0 else 0)
        out["tau"] = fnone(tau)
        out["pvalue"] = fnone(max(0.0, min(1.0, pvalue)))
        out["trend"] = trend
    except Exception:
        pass
    points.append(out)
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize mannkendall query");
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
                "failed to spawn python3 for mannkendall oracle: {e}"
            );
            eprintln!("skipping mannkendall oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open mannkendall oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "mannkendall oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping mannkendall oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for mannkendall oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "mannkendall oracle failed: {stderr}"
        );
        eprintln!("skipping mannkendall oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse mannkendall oracle JSON"))
}

#[test]
fn diff_stats_mannkendall() {
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
        let (rust_tau, rust_p, rust_trend) = mannkendall(&case.data);

        if let Some(scipy_tau) = scipy_arm.tau
            && rust_tau.is_finite()
        {
            let abs_diff = (rust_tau - scipy_tau).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "tau".into(),
                abs_diff,
                pass: abs_diff <= TAU_TOL,
            });
        }
        if let Some(scipy_p) = scipy_arm.pvalue
            && rust_p.is_finite()
        {
            let abs_diff = (rust_p - scipy_p).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "pvalue".into(),
                abs_diff,
                pass: abs_diff <= PVALUE_TOL,
            });
        }
        if let Some(scipy_trend) = scipy_arm.trend {
            let abs_diff = (rust_trend as i64 - scipy_trend).unsigned_abs() as f64;
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "trend".into(),
                abs_diff,
                pass: abs_diff <= TAU_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_mannkendall".into(),
        category: "mannkendall (numpy reference)".into(),
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
                "mannkendall mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "mannkendall conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
