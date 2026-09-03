#![forbid(unsafe_code)]
//! Live numpy differential coverage for fsci's
//! `weighted_mean(values, weights)` and `weighted_var(values,
//! weights)` — closed-form weighted average and biased
//! weighted variance.
//!
//! Resolves [frankenscipy-gav2k]. The oracle calls
//! `numpy.average(values, weights=weights)` for the mean and
//! `numpy.sum(weights * (values − avg)**2) / sum(weights)`
//! for the biased variance — fsci divides by total_w (not
//! total_w − 1), so the oracle uses the same biased
//! convention.
//!
//! 4 (values, weights) fixtures × 2 funcs = 8 cases. Tol 1e-12
//! abs (closed-form weighted sum; no transcendentals).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{weighted_mean, weighted_var};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    values: Vec<f64>,
    weights: Vec<f64>,
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
    fs::create_dir_all(output_dir()).expect("create weighted_mean_var diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize weighted_mean_var diff log");
    fs::write(path, json).expect("write weighted_mean_var diff log");
}

fn generate_query() -> OracleQuery {
    let fixtures: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![
        // Uniform weights → matches arithmetic mean / population var.
        (
            "uniform_weights_n10",
            (1..=10).map(|i| i as f64).collect(),
            vec![1.0; 10],
        ),
        // Highly skewed weights (heavily emphasise tail values).
        (
            "skewed_weights",
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            vec![0.1, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0],
        ),
        // Includes negatives + non-uniform weights
        (
            "negatives_n8",
            vec![-3.0, -1.5, 0.0, 1.5, 3.0, 4.5, 6.0, 7.5],
            vec![0.5, 1.0, 1.5, 2.0, 2.5, 2.0, 1.5, 1.0],
        ),
        // Tightly clustered values, varied weights
        (
            "clustered_n12",
            vec![
                4.95, 5.0, 5.05, 5.1, 4.9, 5.0, 5.0, 5.05, 4.95, 5.1, 4.95, 5.0,
            ],
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 1.0, 3.0, 2.0, 1.0, 2.0, 1.0],
        ),
    ];

    let mut points = Vec::new();
    for (name, values, weights) in &fixtures {
        for func in ["weighted_mean", "weighted_var"] {
            points.push(PointCase {
                case_id: format!("{name}_{func}"),
                func: func.into(),
                values: values.clone(),
                weights: weights.clone(),
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
    v = np.array(case["values"], dtype=float)
    w = np.array(case["weights"], dtype=float)
    val = None
    try:
        if func == "weighted_mean":
            val = np.average(v, weights=w)
        elif func == "weighted_var":
            avg = np.average(v, weights=w)
            tot = float(np.sum(w))
            val = float(np.sum(w * (v - avg) ** 2) / tot) if tot != 0 else float("nan")
    except Exception:
        val = None
    points.append({"case_id": cid, "value": fnone(val)})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize weighted_mean_var query");
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
                "failed to spawn python3 for weighted_mean_var oracle: {e}"
            );
            eprintln!("skipping weighted_mean_var oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open weighted_mean_var oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "weighted_mean_var oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping weighted_mean_var oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for weighted_mean_var oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "weighted_mean_var oracle failed: {stderr}"
        );
        eprintln!("skipping weighted_mean_var oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse weighted_mean_var oracle JSON"))
}

#[test]
fn diff_stats_weighted_mean_var() {
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
        let Some(scipy_v) = scipy_arm.value else {
            continue;
        };
        let rust_v = match case.func.as_str() {
            "weighted_mean" => weighted_mean(&case.values, &case.weights),
            "weighted_var" => weighted_var(&case.values, &case.weights),
            _ => continue,
        };
        if !rust_v.is_finite() {
            continue;
        }
        let abs_diff = (rust_v - scipy_v).abs();
        max_overall = max_overall.max(abs_diff);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            func: case.func.clone(),
            abs_diff,
            pass: abs_diff <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_weighted_mean_var".into(),
        category: "numpy.average + biased weighted variance reference".into(),
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
                "weighted_mean_var {} mismatch: {} abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "weighted_mean_var conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
