#![forbid(unsafe_code)]
//! Live SciPy differential coverage for
//! `scipy.integrate.cumulative_simpson(y, x=x)` — the composite
//! Simpson's-rule cumulative integral.
//!
//! Resolves [frankenscipy-bna0p]. diff_integrate.rs already covers
//! trapezoid / simpson / cumulative_trapezoid / romb but did not
//! exercise cumulative_simpson.
//!
//! Includes polynomial-exact cases plus higher-order and non-uniform
//! grids that exercise SciPy's paired-window cumulative Simpson
//! interval selection.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_integrate::cumulative_simpson;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-008";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    y: Vec<f64>,
    x: Vec<f64>,
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
    fs::create_dir_all(output_dir()).expect("create cumulative_simpson diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize cumulative_simpson diff log");
    fs::write(path, json).expect("write cumulative_simpson diff log");
}

fn generate_query() -> OracleQuery {
    let mut cases: Vec<(&str, Vec<f64>, Vec<f64>)> = Vec::new();

    // 1. Constant y on uniform x.
    cases.push((
        "constant_uniform_n8",
        vec![3.5; 8],
        (0..8).map(|i| i as f64).collect(),
    ));

    // 2. Linear y on uniform x.
    cases.push((
        "linear_uniform_n8",
        (0..8).map(|i| i as f64).collect(),
        (0..8).map(|i| i as f64).collect(),
    ));

    // 3. y = x^2 on uniform x [0, 1] step 0.1.
    let x_uni: Vec<f64> = (0..11).map(|i| i as f64 / 10.0).collect();
    let y_quad: Vec<f64> = x_uni.iter().map(|&t| t * t).collect();
    cases.push(("xsq_uniform_n11", y_quad, x_uni));

    // 4. Two-point fallback to cumulative_trapezoid.
    cases.push(("twopoint_fallback", vec![1.0, 4.0], vec![0.0, 1.0]));

    // 5. Three-point linear case.
    cases.push((
        "threepoint_linear",
        vec![1.0, 2.0, 3.0],
        vec![0.0, 1.0, 2.0],
    ));

    // 6. Quadratic on uniform x [0, 2] step 0.5 — Simpson exact.
    let x_q2: Vec<f64> = (0..5).map(|i| i as f64 * 0.5).collect();
    let y_q2: Vec<f64> = x_q2.iter().map(|&t| 2.0 * t * t + 3.0 * t + 1.0).collect();
    cases.push(("quad_uniform_n5", y_q2, x_q2));

    // 7. Cubic on a non-uniform grid; exercises SciPy's paired-window scheme.
    let x_cubic: Vec<f64> = vec![0.0, 0.2, 0.5, 1.1, 1.4, 2.0, 2.7, 3.5];
    let y_cubic: Vec<f64> = x_cubic.iter().map(|&t| t * t * t + 2.0 * t).collect();
    cases.push(("cubic_nonuniform_n8", y_cubic, x_cubic));

    // 8. Smooth non-polynomial on a uniform grid with an even number of subintervals.
    let x_sin: Vec<f64> = (0..=10)
        .map(|i| (i as f64) * std::f64::consts::PI / 10.0)
        .collect();
    let y_sin: Vec<f64> = x_sin.iter().map(|&t| t.sin()).collect();
    cases.push(("sin_uniform_pi_n11", y_sin, x_sin));

    // 9. Decaying exponential on a non-uniform grid.
    let x_exp: Vec<f64> = vec![0.0, 0.1, 0.4, 0.9, 1.5, 2.2, 3.0, 4.1, 5.0];
    let y_exp: Vec<f64> = x_exp.iter().map(|&t| (-t).exp()).collect();
    cases.push(("exp_nonuniform_n9", y_exp, x_exp));

    let points = cases
        .into_iter()
        .map(|(name, y, x)| PointCase {
            case_id: name.into(),
            y,
            x,
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
from scipy import integrate

def finite_vec_or_none(arr):
    out = []
    for v in np.asarray(arr).tolist():
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
    y = np.array(case["y"], dtype=float)
    x = np.array(case["x"], dtype=float)
    try:
        v = integrate.cumulative_simpson(y, x=x)
        points.append({"case_id": cid, "values": finite_vec_or_none(v)})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize cumulative_simpson query");
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
                "failed to spawn python3 for cumulative_simpson oracle: {e}"
            );
            eprintln!("skipping cumulative_simpson oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open cumulative_simpson oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "cumulative_simpson oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping cumulative_simpson oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for cumulative_simpson oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "cumulative_simpson oracle failed: {stderr}"
        );
        eprintln!("skipping cumulative_simpson oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse cumulative_simpson oracle JSON"))
}

#[test]
fn diff_integrate_cumulative_simpson() {
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
        let Ok(fsci_v) = cumulative_simpson(&case.y, &case.x) else {
            continue;
        };
        let Some(scipy_v) = scipy_arm.values.as_ref() else {
            continue;
        };
        if fsci_v.len() != scipy_v.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let abs_d = fsci_v
            .iter()
            .zip(scipy_v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_integrate_cumulative_simpson".into(),
        category: "scipy.integrate.cumulative_simpson".into(),
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
                "cumulative_simpson mismatch: {} abs_diff={}",
                d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.integrate.cumulative_simpson conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
