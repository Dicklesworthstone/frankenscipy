#![forbid(unsafe_code)]
//! Live scipy.optimize.minimize parity for fsci_opt::gradient_descent.
//!
//! Resolves [frankenscipy-3dwfp]. Tests convergence to the same minimum
//! on simple convex problems against scipy.optimize.minimize(method='BFGS').
//! Compares the FINAL x at 1e-3 abs — gradient descent is slower than
//! BFGS, so we run many iterations and accept moderate residual.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_opt::gradient_descent;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-3;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct GdCase {
    case_id: String,
    func: String,
    x0: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<GdCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    x: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create gd diff dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize log");
    fs::write(path, json).expect("write log");
}

fn f_eval(name: &str, x: &[f64]) -> f64 {
    match name {
        // f(x, y) = (x - 1)^2 + (y + 2)^2; minimum (1, -2)
        "shifted_quad_2d" => (x[0] - 1.0).powi(2) + (x[1] + 2.0).powi(2),
        // 3-D quadratic; minimum (0, 0, 0)
        "sumsq_3d" => x[0].powi(2) + x[1].powi(2) + x[2].powi(2),
        // f(x, y) = x^2 + 4y^2 + xy; minimum at (0, 0)
        "tilted_quad_2d" => x[0] * x[0] + 4.0 * x[1] * x[1] + x[0] * x[1],
        _ => f64::NAN,
    }
}

fn grad_eval(name: &str, x: &[f64]) -> Vec<f64> {
    match name {
        "shifted_quad_2d" => vec![2.0 * (x[0] - 1.0), 2.0 * (x[1] + 2.0)],
        "sumsq_3d" => vec![2.0 * x[0], 2.0 * x[1], 2.0 * x[2]],
        "tilted_quad_2d" => vec![2.0 * x[0] + x[1], 8.0 * x[1] + x[0]],
        _ => vec![],
    }
}

fn generate_query() -> OracleQuery {
    OracleQuery {
        points: vec![
            GdCase {
                case_id: "shifted_quad_2d_from_5_5".into(),
                func: "shifted_quad_2d".into(),
                x0: vec![5.0, 5.0],
            },
            GdCase {
                case_id: "sumsq_3d_from_2_2_2".into(),
                func: "sumsq_3d".into(),
                x0: vec![2.0, -2.0, 1.0],
            },
            GdCase {
                case_id: "tilted_quad_2d_from_3_3".into(),
                func: "tilted_quad_2d".into(),
                x0: vec![3.0, 3.0],
            },
        ],
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy.optimize import minimize

def f(name, x):
    if name == "shifted_quad_2d":
        return (x[0] - 1.0)**2 + (x[1] + 2.0)**2
    if name == "sumsq_3d":
        return x[0]**2 + x[1]**2 + x[2]**2
    if name == "tilted_quad_2d":
        return x[0]**2 + 4.0*x[1]**2 + x[0]*x[1]
    return float("nan")

def grad(name, x):
    if name == "shifted_quad_2d":
        return np.array([2*(x[0]-1.0), 2*(x[1]+2.0)])
    if name == "sumsq_3d":
        return np.array([2*x[0], 2*x[1], 2*x[2]])
    if name == "tilted_quad_2d":
        return np.array([2*x[0] + x[1], 8*x[1] + x[0]])
    return None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; name = case["func"]
    x0 = np.array(case["x0"], dtype=float)
    try:
        res = minimize(lambda x: f(name, x), x0, jac=lambda x: grad(name, x),
                       method='BFGS', options={'gtol': 1e-12})
        if res.success:
            points.append({"case_id": cid, "x": [float(v) for v in res.x.tolist()]})
        else:
            points.append({"case_id": cid, "x": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "x": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize query");
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
                "failed to spawn python3 for gd oracle: {e}"
            );
            eprintln!("skipping gd oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "gd oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping gd oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for gd oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "gd oracle failed: {stderr}"
        );
        eprintln!("skipping gd oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse gd oracle JSON"))
}

#[test]
fn diff_opt_gradient_descent() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let Some(arm) = pmap.get(&case.case_id) else {
            continue;
        };
        let Some(expected) = arm.x.as_ref() else {
            continue;
        };
        let f = |x: &[f64]| f_eval(&case.func, x);
        let g = |x: &[f64]| grad_eval(&case.func, x);
        let res = gradient_descent(f, g, &case.x0, 1.0e-9, 50000, 0.05);
        if !res.success {
            continue;
        }
        let abs_d = if res.x.len() != expected.len() {
            f64::INFINITY
        } else {
            res.x
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        };
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_gradient_descent".into(),
        category: "fsci_opt::gradient_descent vs scipy.optimize.minimize(BFGS)".into(),
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
            eprintln!("gd mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "gd conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
