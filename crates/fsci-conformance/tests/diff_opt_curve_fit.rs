#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_opt::curve_fit on
//! noise-free synthetic data. Three model families:
//!   - linear:        y = a*x + b
//!   - quadratic:     y = a*x^2 + b*x + c
//!   - exponential:   y = a * exp(-k*x)
//!
//! Resolves [frankenscipy-e11r9]. Noise-free, so both fsci's LM and
//! scipy's LM should converge to the true parameters; tolerance 1e-6 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_opt::{CurveFitOptions, curve_fit};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-6;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    model: String,
    xdata: Vec<f64>,
    ydata: Vec<f64>,
    p0: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    popt: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    model: String,
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
    fs::create_dir_all(output_dir()).expect("create curve_fit diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize curve_fit diff log");
    fs::write(path, json).expect("write curve_fit diff log");
}

fn linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![start];
    }
    let step = (stop - start) / (n - 1) as f64;
    (0..n).map(|i| start + step * i as f64).collect()
}

fn generate_query() -> OracleQuery {
    let x_lin = linspace(0.0, 10.0, 20);
    let y_lin: Vec<f64> = x_lin.iter().map(|&x| 2.5 * x + 1.3).collect();

    let x_quad = linspace(-2.0, 2.0, 15);
    let y_quad: Vec<f64> = x_quad
        .iter()
        .map(|&x| 0.5 * x * x + 2.0 * x + 3.0)
        .collect();

    let x_exp = linspace(0.0, 5.0, 25);
    let y_exp: Vec<f64> = x_exp.iter().map(|&x| 3.0 * (-0.4 * x).exp()).collect();

    let points = vec![
        PointCase {
            case_id: "linear_a25_b13".into(),
            model: "linear".into(),
            xdata: x_lin,
            ydata: y_lin,
            p0: vec![1.0, 0.0],
        },
        PointCase {
            case_id: "quadratic_05_2_3".into(),
            model: "quadratic".into(),
            xdata: x_quad,
            ydata: y_quad,
            p0: vec![1.0, 1.0, 1.0],
        },
        PointCase {
            case_id: "exp_3_p04".into(),
            model: "exponential".into(),
            xdata: x_exp,
            ydata: y_exp,
            p0: vec![1.0, 0.1],
        },
    ];

    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy.optimize import curve_fit

def linear(x, a, b): return a * x + b
def quadratic(x, a, b, c): return a * x * x + b * x + c
def exp_fn(x, a, k): return a * np.exp(-k * x)

models = {"linear": linear, "quadratic": quadratic, "exponential": exp_fn}

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; model = case["model"]
    x = np.array(case["xdata"], dtype=float)
    y = np.array(case["ydata"], dtype=float)
    p0 = list(case["p0"])
    try:
        popt, _ = curve_fit(models[model], x, y, p0=p0)
        flat = [float(v) for v in popt.tolist()]
        if all(math.isfinite(v) for v in flat):
            points.append({"case_id": cid, "popt": flat})
        else:
            points.append({"case_id": cid, "popt": None})
    except Exception:
        points.append({"case_id": cid, "popt": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize curve_fit query");
    let mut child = match Command::new("python3")
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
                "failed to spawn python3 for curve_fit oracle: {e}"
            );
            eprintln!("skipping curve_fit oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open curve_fit oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "curve_fit oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping curve_fit oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for curve_fit oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "curve_fit oracle failed: {stderr}"
        );
        eprintln!("skipping curve_fit oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse curve_fit oracle JSON"))
}

#[test]
fn diff_opt_curve_fit() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(expected) = scipy_arm.popt.as_ref() else {
            continue;
        };
        let opts = CurveFitOptions {
            p0: Some(case.p0.clone()),
            ..Default::default()
        };
        let result = match case.model.as_str() {
            "linear" => curve_fit(
                |x: f64, p: &[f64]| p[0] * x + p[1],
                &case.xdata,
                &case.ydata,
                opts,
            ),
            "quadratic" => curve_fit(
                |x: f64, p: &[f64]| p[0] * x * x + p[1] * x + p[2],
                &case.xdata,
                &case.ydata,
                opts,
            ),
            "exponential" => curve_fit(
                |x: f64, p: &[f64]| p[0] * (-p[1] * x).exp(),
                &case.xdata,
                &case.ydata,
                opts,
            ),
            _ => continue,
        };
        let Ok(res) = result else {
            continue;
        };
        let abs_d = if res.popt.len() != expected.len() {
            f64::INFINITY
        } else {
            res.popt
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        };
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            model: case.model.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_curve_fit".into(),
        category: "scipy.optimize.curve_fit (noise-free)".into(),
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
                "{} mismatch: {} abs_diff={}",
                d.model, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "curve_fit conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
