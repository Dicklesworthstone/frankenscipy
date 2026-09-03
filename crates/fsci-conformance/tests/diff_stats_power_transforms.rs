#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the power-transform
//! log-likelihood functions:
//!   • `scipy.stats.boxcox_llf(lmb, data)` (positive data)
//!   • `scipy.stats.yeojohnson_llf(lmb, data)` (real data)
//!
//! Resolves [frankenscipy-9xu9l]. Both fsci and scipy use the
//! biased variance estimator (ddof=0) for these LLFs, so the
//! arithmetic should align to ~1e-12 abs across all
//! reasonable lambda values.
//!
//! 3 boxcox datasets × 4 lambdas + 3 yeojohnson datasets × 4
//! lambdas = 24 cases via subprocess. Tol 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{boxcox_llf, yeojohnson_llf};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    lmb: f64,
    data: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    value: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    func: String,
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
    fs::create_dir_all(output_dir()).expect("create power-transforms diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize power-transforms diff log");
    fs::write(path, json).expect("write power-transforms diff log");
}

fn fsci_eval(case: &PointCase) -> Option<f64> {
    let v = match case.func.as_str() {
        "boxcox_llf" => boxcox_llf(case.lmb, &case.data),
        "yeojohnson_llf" => yeojohnson_llf(case.lmb, &case.data),
        _ => return None,
    };
    if v.is_finite() { Some(v) } else { None }
}

fn generate_query() -> OracleQuery {
    let lambdas: [f64; 4] = [-1.0, 0.0, 0.5, 2.0];

    // Box-Cox: positive data only.
    let boxcox_datasets: Vec<(&str, Vec<f64>)> = vec![
        ("compact", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        (
            "wide_range",
            vec![0.5, 1.2, 2.5, 5.0, 10.0, 20.0, 50.0, 100.0],
        ),
        (
            "noisy_lognormal_like",
            vec![1.05, 2.7, 1.4, 3.1, 0.9, 4.2, 1.8, 2.0, 5.5, 1.25],
        ),
    ];

    // Yeo-Johnson: handles negatives and zero.
    let yj_datasets: Vec<(&str, Vec<f64>)> = vec![
        ("centered", vec![-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0]),
        ("asymmetric", vec![-2.0, -1.0, 0.5, 1.0, 2.5, 4.0, 6.0, 8.0]),
        (
            "all_negative",
            vec![-5.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5],
        ),
    ];

    let mut points = Vec::new();
    for (name, data) in boxcox_datasets {
        for &lmb in &lambdas {
            points.push(PointCase {
                case_id: format!("boxcox_{name}_l{lmb}"),
                func: "boxcox_llf".into(),
                lmb,
                data: data.clone(),
            });
        }
    }
    for (name, data) in yj_datasets {
        for &lmb in &lambdas {
            points.push(PointCase {
                case_id: format!("yj_{name}_l{lmb}"),
                func: "yeojohnson_llf".into(),
                lmb,
                data: data.clone(),
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
    cid = case["case_id"]; func = case["func"]
    lmb = float(case["lmb"])
    data = np.array(case["data"], dtype=float)
    try:
        if func == "boxcox_llf":
            value = float(stats.boxcox_llf(lmb, data))
        elif func == "yeojohnson_llf":
            value = float(stats.yeojohnson_llf(lmb, data))
        else:
            value = None
        points.append({"case_id": cid, "value": fnone(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize power-transforms query");
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
                "failed to spawn python3 for power-transforms oracle: {e}"
            );
            eprintln!("skipping power-transforms oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open power-transforms oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "power-transforms oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping power-transforms oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for power-transforms oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "power-transforms oracle failed: {stderr}"
        );
        eprintln!("skipping power-transforms oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse power-transforms oracle JSON"))
}

#[test]
fn diff_stats_power_transforms() {
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
        if let Some(scipy_v) = scipy_arm.value
            && let Some(rust_v) = fsci_eval(case)
        {
            let abs_diff = (rust_v - scipy_v).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_power_transforms".into(),
        category: "scipy.stats.boxcox_llf/yeojohnson_llf".into(),
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
                "power-transforms {} mismatch: {} abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "power-transforms conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
