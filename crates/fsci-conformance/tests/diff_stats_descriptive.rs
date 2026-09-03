#![forbid(unsafe_code)]
//! Live SciPy differential coverage for descriptive-statistics
//! utilities not exercised by any existing diff harness:
//!   • `scipy.stats.iqr(data)`
//!   • `scipy.stats.zscore(data)`            (ddof=0)
//!   • `scipy.stats.zscore(data, ddof=1)`
//!   • `scipy.stats.trim_mean(data, 0.10)`
//!   • `scipy.stats.trim_mean(data, 0.25)`
//!
//! Resolves [frankenscipy-tdrmu]. Each function is exercised
//! across 3 datasets (compact mixed, light-tail, heavy-tail).
//! Scalar arms compared directly; vector arms (zscore family)
//! compared element-wise with max-abs aggregation per case.
//!
//! 3 datasets × 5 arms = 15 cases via subprocess. Tol 1e-12
//! abs throughout — fsci's iqr/zscore/trim_mean defaults
//! match scipy's defaults (linear interpolation, ddof, floor
//! ncut) so the closed-form arithmetic should align exactly.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{iqr, trim_mean, zscore, zscore_ddof};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    arm: String,
    data: Vec<f64>,
    param: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    scalar: Option<f64>,
    vector: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create descriptive diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize descriptive diff log");
    fs::write(path, json).expect("write descriptive diff log");
}

fn generate_query() -> OracleQuery {
    let datasets: Vec<(&str, Vec<f64>)> = vec![
        (
            "compact",
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        ),
        (
            "light_tail",
            vec![-2.5, -1.7, -0.9, -0.3, 0.1, 0.4, 0.8, 1.2, 1.8, 2.1, 3.0],
        ),
        (
            "heavy_tail",
            vec![
                -50.0, -8.0, -3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 4.0, 7.0, 25.0, 100.0,
            ],
        ),
    ];

    let arms: [(&str, f64); 5] = [
        ("iqr", 0.0),
        ("zscore", 0.0),
        ("zscore_ddof1", 1.0),
        ("trim_mean_010", 0.10),
        ("trim_mean_025", 0.25),
    ];

    let mut points = Vec::new();
    for (name, data) in datasets {
        for (arm, param) in arms {
            points.push(PointCase {
                case_id: format!("{name}_{arm}"),
                arm: arm.into(),
                data: data.clone(),
                param,
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
    cid = case["case_id"]; arm = case["arm"]
    data = np.array(case["data"], dtype=float)
    param = float(case["param"])
    scalar = None; vector = None
    try:
        if arm == "iqr":
            scalar = fnone(stats.iqr(data))
        elif arm == "zscore":
            vector = vec_or_none(stats.zscore(data).tolist())
        elif arm == "zscore_ddof1":
            vector = vec_or_none(stats.zscore(data, ddof=1).tolist())
        elif arm == "trim_mean_010":
            scalar = fnone(stats.trim_mean(data, param))
        elif arm == "trim_mean_025":
            scalar = fnone(stats.trim_mean(data, param))
    except Exception:
        pass
    points.append({"case_id": cid, "scalar": scalar, "vector": vector})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize descriptive query");
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
                "failed to spawn python3 for descriptive oracle: {e}"
            );
            eprintln!("skipping descriptive oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open descriptive oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "descriptive oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping descriptive oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for descriptive oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "descriptive oracle failed: {stderr}"
        );
        eprintln!("skipping descriptive oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse descriptive oracle JSON"))
}

#[test]
fn diff_stats_descriptive() {
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
        let abs_diff: Option<f64> = match case.arm.as_str() {
            "iqr" => scipy_arm.scalar.map(|s| (iqr(&case.data) - s).abs()),
            "trim_mean_010" | "trim_mean_025" => scipy_arm.scalar.map(|s| {
                (trim_mean(&case.data, case.param).unwrap_or_else(|e| {
                    panic!(
                        "scipy produced a trim_mean for case {} (prop {}) but ours refused: {e:?}",
                        case.case_id, case.param
                    )
                }) - s)
                    .abs()
            }),
            "zscore" => scipy_arm.vector.as_ref().map(|v| {
                let r = zscore(&case.data);
                let mut m = 0.0_f64;
                for (a, b) in r.iter().zip(v.iter()) {
                    if a.is_finite() {
                        m = m.max((a - b).abs());
                    }
                }
                m
            }),
            "zscore_ddof1" => scipy_arm.vector.as_ref().map(|v| {
                let r = zscore_ddof(&case.data, 1);
                let mut m = 0.0_f64;
                for (a, b) in r.iter().zip(v.iter()) {
                    if a.is_finite() {
                        m = m.max((a - b).abs());
                    }
                }
                m
            }),
            _ => None,
        };
        if let Some(abs_diff) = abs_diff {
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: case.arm.clone(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_descriptive".into(),
        category: "scipy.stats.iqr/zscore/trim_mean".into(),
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
                "descriptive mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "descriptive conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
