#![forbid(unsafe_code)]
//! Live scipy parity for fsci_opt::least_squares (Levenberg-Marquardt).
//!
//! Resolves [frankenscipy-5jvev]. Compares fsci's LM nonlinear least
//! squares against scipy.optimize.least_squares(method='lm') on 5
//! residual problems: exponential decay parameter fit, Gaussian peak
//! fit, circle-from-points fit, sine-curve parameter fit, and a simple
//! linear residual baseline.
//!
//! Pass criteria: (1) both solvers converge, AND (2) the cost
//! 0.5||r(x*)||² agrees within rel 1e-4 (both should find the same
//! local minimum) AND the parameters agree within abs 1e-3 OR rel 1e-3
//! (LM solvers can converge to mathematically equivalent points with
//! small numerical drift).

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_opt::{LeastSquaresOptions, least_squares};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const COST_REL_TOL: f64 = 1.0e-4;
const COST_ABS_TOL: f64 = 1.0e-10;
const PARAM_REL_TOL: f64 = 1.0e-3;
const PARAM_ABS_TOL: f64 = 1.0e-3;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct CasePoint {
    case_id: String,
    /// Problem kind: "exp_decay" | "gaussian" | "circle" | "sine" | "linear"
    kind: String,
    x0: Vec<f64>,
    /// Extra problem data (x for curve fits, points for circle, etc.)
    xs: Vec<f64>,
    ys: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<CasePoint>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct OraclePoint {
    case_id: String,
    converged: bool,
    x: Option<Vec<f64>>,
    cost: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<OraclePoint>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    fsci_converged: bool,
    scipy_converged: bool,
    fsci_cost: f64,
    scipy_cost: f64,
    max_param_abs_diff: f64,
    pass: bool,
    note: String,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create least_squares diff dir");
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

fn residuals_for(case: &CasePoint, params: &[f64]) -> Vec<f64> {
    match case.kind.as_str() {
        "exp_decay" => {
            // y = a * exp(-b * x); residuals = a*exp(-b*x) - y
            let a = params[0];
            let b = params[1];
            case.xs
                .iter()
                .zip(case.ys.iter())
                .map(|(&x, &y)| a * (-b * x).exp() - y)
                .collect()
        }
        "gaussian" => {
            // y = a * exp(-((x-c)/w)^2)
            let a = params[0];
            let c = params[1];
            let w = params[2];
            case.xs
                .iter()
                .zip(case.ys.iter())
                .map(|(&x, &y)| a * (-((x - c) / w).powi(2)).exp() - y)
                .collect()
        }
        "circle" => {
            // xs and ys are x,y of points; residual = sqrt((x-cx)^2 + (y-cy)^2) - r
            let cx = params[0];
            let cy = params[1];
            let r = params[2];
            case.xs
                .iter()
                .zip(case.ys.iter())
                .map(|(&x, &y)| ((x - cx).powi(2) + (y - cy).powi(2)).sqrt() - r)
                .collect()
        }
        "sine" => {
            // y = a * sin(b*x + c)
            let a = params[0];
            let b = params[1];
            let c = params[2];
            case.xs
                .iter()
                .zip(case.ys.iter())
                .map(|(&x, &y)| a * (b * x + c).sin() - y)
                .collect()
        }
        "linear" => {
            // y = m*x + c
            let m = params[0];
            let cc = params[1];
            case.xs
                .iter()
                .zip(case.ys.iter())
                .map(|(&x, &y)| m * x + cc - y)
                .collect()
        }
        other => panic!("unknown kind {other}"),
    }
}

fn build_query() -> OracleQuery {
    let mut pts = Vec::new();

    // 1. Exponential decay: y = 2.5 * exp(-0.8 * x), 20 points
    {
        let xs: Vec<f64> = (0..20).map(|i| i as f64 * 0.25).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 2.5 * (-0.8 * x).exp()).collect();
        pts.push(CasePoint {
            case_id: "exp_decay".into(),
            kind: "exp_decay".into(),
            x0: vec![1.0, 0.5],
            xs,
            ys,
        });
    }

    // 2. Gaussian peak: a=4.0, c=2.0, w=0.8
    {
        let xs: Vec<f64> = (0..25).map(|i| -1.0 + i as f64 * 0.25).collect();
        let ys: Vec<f64> = xs
            .iter()
            .map(|&x| 4.0 * (-((x - 2.0) / 0.8).powi(2)).exp())
            .collect();
        pts.push(CasePoint {
            case_id: "gaussian_peak".into(),
            kind: "gaussian".into(),
            x0: vec![1.0, 0.0, 1.0],
            xs,
            ys,
        });
    }

    // 3. Circle fit: true center (1, 2), radius 3, 12 points on circle
    {
        let n = 12;
        let xs: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 * 2.0 * std::f64::consts::PI / n as f64;
                1.0 + 3.0 * t.cos()
            })
            .collect();
        let ys: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 * 2.0 * std::f64::consts::PI / n as f64;
                2.0 + 3.0 * t.sin()
            })
            .collect();
        pts.push(CasePoint {
            case_id: "circle_fit".into(),
            kind: "circle".into(),
            x0: vec![0.0, 0.0, 1.0],
            xs,
            ys,
        });
    }

    // 4. Sine: y = 2.0 * sin(1.5 * x + 0.3)
    {
        let xs: Vec<f64> = (0..30).map(|i| i as f64 * 0.2).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 2.0 * (1.5 * x + 0.3).sin()).collect();
        pts.push(CasePoint {
            case_id: "sine_fit".into(),
            kind: "sine".into(),
            x0: vec![1.5, 1.4, 0.2],
            xs,
            ys,
        });
    }

    // 5. Linear baseline: y = 3.0 * x + 5.0
    {
        let xs: Vec<f64> = (0..15).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 3.0 * x + 5.0).collect();
        pts.push(CasePoint {
            case_id: "linear".into(),
            kind: "linear".into(),
            x0: vec![1.0, 0.0],
            xs,
            ys,
        });
    }

    OracleQuery { points: pts }
}

fn scipy_oracle_or_skip(q: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json, math, sys
import numpy as np
from scipy.optimize import least_squares

def res_fn(kind, xs, ys):
    if kind == "exp_decay":
        def f(p): return p[0] * np.exp(-p[1] * xs) - ys
        return f
    if kind == "gaussian":
        def f(p): return p[0] * np.exp(-((xs - p[1]) / p[2])**2) - ys
        return f
    if kind == "circle":
        def f(p):
            return np.sqrt((xs - p[0])**2 + (ys - p[1])**2) - p[2]
        return f
    if kind == "sine":
        def f(p): return p[0] * np.sin(p[1] * xs + p[2]) - ys
        return f
    if kind == "linear":
        def f(p): return p[0] * xs + p[1] - ys
        return f
    raise ValueError(kind)

q = json.load(sys.stdin)
out = []
for c in q["points"]:
    cid = c["case_id"]
    try:
        xs = np.array(c["xs"], dtype=float)
        ys = np.array(c["ys"], dtype=float)
        f = res_fn(c["kind"], xs, ys)
        result = least_squares(f, np.array(c["x0"], dtype=float), method='lm')
        cost = float(0.5 * np.sum(result.fun ** 2))
        if not np.all(np.isfinite(result.x)) or not math.isfinite(cost):
            out.append({"case_id": cid, "converged": False, "x": None, "cost": None})
        else:
            out.append({
                "case_id": cid,
                "converged": bool(result.success),
                "x": [float(v) for v in result.x],
                "cost": cost,
            })
    except Exception:
        out.append({"case_id": cid, "converged": False, "x": None, "cost": None})

print(json.dumps({"points": out}))
"#;
    let query_json = serde_json::to_string(q).expect("serialize");
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
                "python3 spawn failed: {e}"
            );
            eprintln!("skipping least_squares oracle: python3 unavailable ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping least_squares oracle: stdin write failed");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "oracle failed: {stderr}"
        );
        eprintln!("skipping least_squares oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse JSON"))
}

#[test]
fn diff_opt_least_squares() {
    let query = build_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();

    for (case, o) in query.points.iter().zip(oracle.points.iter()) {
        assert_eq!(case.case_id, o.case_id);
        let (Some(exp_x), Some(exp_cost)) = (o.x.as_ref(), o.cost) else {
            continue;
        };

        let case_clone = case.clone();
        let f = move |params: &[f64]| residuals_for(&case_clone, params);
        let result = match least_squares(f, &case.x0, LeastSquaresOptions::default()) {
            Ok(r) => r,
            Err(e) => {
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    fsci_converged: false,
                    scipy_converged: o.converged,
                    fsci_cost: f64::INFINITY,
                    scipy_cost: exp_cost,
                    max_param_abs_diff: f64::INFINITY,
                    pass: false,
                    note: format!("least_squares error: {e:?}"),
                });
                continue;
            }
        };

        let fsci_cost = result.cost;
        let abs_cost_d = (fsci_cost - exp_cost).abs();
        let denom_c = exp_cost.abs().max(1.0e-300);
        let rel_cost_d = abs_cost_d / denom_c;
        let cost_pass = rel_cost_d <= COST_REL_TOL || abs_cost_d <= COST_ABS_TOL;

        let mut max_param_abs = 0.0_f64;
        let mut param_pass = result.x.len() == exp_x.len();
        for (a, e) in result.x.iter().zip(exp_x.iter()) {
            let abs_d = (a - e).abs();
            let denom = e.abs().max(1.0e-300);
            let rel_d = abs_d / denom;
            max_param_abs = max_param_abs.max(abs_d);
            param_pass &= rel_d <= PARAM_REL_TOL || abs_d <= PARAM_ABS_TOL;
        }
        let converged_both = result.success && o.converged;
        let pass = converged_both && cost_pass && param_pass;
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            fsci_converged: result.success,
            scipy_converged: o.converged,
            fsci_cost,
            scipy_cost: exp_cost,
            max_param_abs_diff: max_param_abs,
            pass,
            note: String::new(),
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_opt_least_squares".into(),
        category: "fsci_opt::least_squares vs scipy.optimize.least_squares(method=lm)".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "least_squares mismatch: {} fsci_conv={} scipy_conv={} fsci_cost={} scipy_cost={} max_param_abs={} note={}",
                d.case_id,
                d.fsci_converged,
                d.scipy_converged,
                d.fsci_cost,
                d.scipy_cost,
                d.max_param_abs_diff,
                d.note
            );
        }
    }

    assert!(
        all_pass,
        "least_squares parity failed: {} cases",
        diffs.len()
    );
}
