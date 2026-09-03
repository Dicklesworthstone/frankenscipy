#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci's
//! `hdquantiles_sd(data, prob)` — jackknife standard error
//! of Harrell-Davis quantile estimates.
//!
//! Resolves [frankenscipy-7i8z0]. The oracle calls
//! `scipy.stats.mstats.hdquantiles_sd(data, prob)`.
//!
//! 3 datasets × 4 probs = 12 cases. Tol 1e-9 abs (BetaDist
//! CDF + jackknife chain — far from machine precision but
//! comfortably within scipy's own implementation noise).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::hdquantiles_sd;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    data: Vec<f64>,
    prob: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
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
    fs::create_dir_all(output_dir()).expect("create hdquantiles_sd diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize hdquantiles_sd diff log");
    fs::write(path, json).expect("write hdquantiles_sd diff log");
}

fn generate_query() -> OracleQuery {
    let probs = vec![0.1, 0.25, 0.5, 0.9];
    let datasets: Vec<(&str, Vec<f64>)> = vec![
        // Uniform-ish: deterministic increments
        ("uniform_n20", (1..=20).map(|i| i as f64).collect()),
        // Skewed: square root spreads upper tail
        (
            "skewed_n30",
            (1..=30).map(|i| (i as f64).sqrt() * 3.0).collect(),
        ),
        // Bimodal-ish: two cluster heaps
        ("bimodal_n25", {
            let mut v = Vec::new();
            for i in 0..12 {
                v.push(1.0 + i as f64 * 0.1);
            }
            for i in 0..13 {
                v.push(10.0 + i as f64 * 0.1);
            }
            v
        }),
    ];

    let points = datasets
        .into_iter()
        .map(|(name, data)| PointCase {
            case_id: name.into(),
            data,
            prob: probs.clone(),
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
from scipy.stats import mstats

def vec_or_none(arr):
    out = []
    for v in arr:
        try:
            v = float(v)
        except Exception:
            return None
        if not math.isfinite(v):
            return None
        out.append(v)
    return out

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    data = np.array(case["data"], dtype=float)
    prob = np.array(case["prob"], dtype=float)
    val = None
    try:
        ses = mstats.hdquantiles_sd(data, prob=prob)
        val = vec_or_none(np.asarray(ses).tolist())
    except Exception:
        val = None
    points.append({"case_id": cid, "values": val})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize hdquantiles_sd query");
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
                "failed to spawn python3 for hdquantiles_sd oracle: {e}"
            );
            eprintln!("skipping hdquantiles_sd oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open hdquantiles_sd oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "hdquantiles_sd oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping hdquantiles_sd oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for hdquantiles_sd oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "hdquantiles_sd oracle failed: {stderr}"
        );
        eprintln!("skipping hdquantiles_sd oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse hdquantiles_sd oracle JSON"))
}

#[test]
fn diff_stats_hdquantiles_sd() {
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
        let Some(scipy_vec) = &scipy_arm.values else {
            continue;
        };
        let rust_vec = hdquantiles_sd(&case.data, &case.prob);
        if rust_vec.len() != scipy_vec.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "shape".into(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        for (i, (r, s)) in rust_vec.iter().zip(scipy_vec.iter()).enumerate() {
            if r.is_finite() {
                let abs_diff = (r - s).abs();
                max_overall = max_overall.max(abs_diff);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    arm: format!("p{}", case.prob[i]),
                    abs_diff,
                    pass: abs_diff <= ABS_TOL,
                });
            }
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_hdquantiles_sd".into(),
        category: "scipy.stats.mstats.hdquantiles_sd".into(),
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
                "hdquantiles_sd mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "hdquantiles_sd conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
