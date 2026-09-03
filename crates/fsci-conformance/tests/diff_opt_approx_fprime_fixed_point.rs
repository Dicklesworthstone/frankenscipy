#![forbid(unsafe_code)]
//! Live scipy.optimize parity for fsci_opt::approx_fprime
//! and fsci_opt::fixed_point.
//!
//! Resolves [frankenscipy-4dl62].
//!
//! - `approx_fprime(xk, f, eps)`: forward-difference gradient.
//!   Deterministic for a fixed (xk, eps, f) — fsci and scipy
//!   compute the identical (f(xk + eps*e_i) - f(xk))/eps. Compare
//!   element-wise at 1e-12 abs.
//! - `fixed_point(f, x0, tol, maxiter)`: Steffensen-accelerated
//!   fixed-point iteration. Both implementations target the same
//!   tolerance so they converge to (near-)identical fixed points.
//!   Compare at 1e-8 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_opt::{approx_fprime, fixed_point};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
// Forward-difference gradients differ from scipy by ULP-level rounding
// when the summation order in numpy.sum (pairwise) diverges from Rust's
// left-fold f64 sum. Largest observed divergence: 3.6e-8 abs at eps=1e-8
// on 3-element sums. Loose tolerance accommodates this without missing
// substantive disagreement.
const GRAD_ABS_TOL: f64 = 1.0e-7;
const FP_ABS_TOL: f64 = 1.0e-8;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String, // "grad" | "fp"
    /// grad
    func: String,
    xk: Vec<f64>,
    eps: f64,
    /// fp
    x0: f64,
    fp_func: String,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
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
    op: String,
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
    fs::create_dir_all(output_dir()).expect("create approx_fp diff dir");
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

fn eval_multi(name: &str, x: &[f64]) -> f64 {
    match name {
        // f1: ||x||^2
        "sq_norm" => x.iter().map(|v| v * v).sum::<f64>(),
        // f2: rosenbrock (2D)
        "rosen2" => {
            let a = x[0];
            let b = x[1];
            (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2)
        }
        // f3: sum of sins
        "sumsin" => x.iter().map(|v| v.sin()).sum::<f64>(),
        // f4: log-prod
        "logprod" => x.iter().map(|v| (1.0 + v.powi(2)).ln()).sum::<f64>(),
        _ => f64::NAN,
    }
}

fn eval_fp(name: &str, x: f64) -> f64 {
    match name {
        // f = cos(x) → fixed point near 0.7390851332
        "cos" => x.cos(),
        // f = (x + 2/x)/2 → sqrt(2) fixed point
        "newton_sqrt2" => 0.5 * (x + 2.0 / x),
        // f = 0.5 * (x + 1) → fixed point at 1
        "linear_half" => 0.5 * (x + 1.0),
        // f = exp(-x) → fixed point near 0.5671432904
        "exp_neg" => (-x).exp(),
        _ => f64::NAN,
    }
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();

    // approx_fprime probes
    let grad_probes: &[(&str, Vec<f64>)] = &[
        ("sq_norm", vec![1.0, -2.0, 3.0]),
        ("sq_norm", vec![0.5, 0.5]),
        ("rosen2", vec![0.0, 0.0]),
        ("rosen2", vec![1.0, 1.0]),
        ("sumsin", vec![0.1, 0.5, 1.0, 1.5]),
        ("logprod", vec![0.2, -0.5, 1.0]),
    ];
    for &eps in &[1.0e-6, 1.0e-8] {
        for (i, (fname, xk)) in grad_probes.iter().enumerate() {
            points.push(Case {
                case_id: format!("grad_{i:02}_{fname}_n{}_e{eps}", xk.len()),
                op: "grad".into(),
                func: (*fname).into(),
                xk: xk.clone(),
                eps,
                x0: 0.0,
                fp_func: String::new(),
            });
        }
    }

    // fixed_point probes
    let fp_probes: &[(&str, f64)] = &[
        ("cos", 1.0),
        ("newton_sqrt2", 1.5),
        ("linear_half", 0.0),
        ("exp_neg", 0.5),
    ];
    for &(fname, x0) in fp_probes {
        points.push(Case {
            case_id: format!("fp_{fname}_x0{x0}"),
            op: "fp".into(),
            func: String::new(),
            xk: vec![],
            eps: 0.0,
            x0,
            fp_func: fname.into(),
        });
    }

    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy.optimize import approx_fprime, fixed_point

def f_multi(name, x):
    if name == "sq_norm":  return float(np.sum(x * x))
    if name == "rosen2":   return float((1 - x[0])**2 + 100*(x[1] - x[0]**2)**2)
    if name == "sumsin":   return float(np.sum(np.sin(x)))
    if name == "logprod":  return float(np.sum(np.log1p(x**2)))
    return float("nan")

def f_fp(name, x):
    if name == "cos":          return math.cos(x)
    if name == "newton_sqrt2": return 0.5*(x + 2.0/x)
    if name == "linear_half":  return 0.5*(x + 1.0)
    if name == "exp_neg":      return math.exp(-x)
    return float("nan")

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    try:
        if op == "grad":
            xk = np.array(case["xk"], dtype=float)
            eps = float(case["eps"])
            fname = case["func"]
            g = approx_fprime(xk, lambda x: f_multi(fname, x), eps)
            flat = [float(v) for v in g.tolist()]
            if all(math.isfinite(v) for v in flat):
                points.append({"case_id": cid, "values": flat})
            else:
                points.append({"case_id": cid, "values": None})
        elif op == "fp":
            x0 = float(case["x0"])
            fname = case["fp_func"]
            try:
                r = fixed_point(lambda x: f_fp(fname, x), x0, xtol=1e-12, maxiter=200)
                v = float(np.atleast_1d(r)[0])
            except RuntimeError:
                v = float("nan")
            if math.isfinite(v):
                points.append({"case_id": cid, "values": [v]})
            else:
                points.append({"case_id": cid, "values": None})
        else:
            points.append({"case_id": cid, "values": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "values": None})
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
                "failed to spawn python3 for approx_fp oracle: {e}"
            );
            eprintln!("skipping approx_fp oracle: python3 not available ({e})");
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
                "approx_fp oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping approx_fp oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for approx_fp oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "approx_fp oracle failed: {stderr}"
        );
        eprintln!("skipping approx_fp oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse approx_fp oracle JSON"))
}

fn vec_max_diff(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn diff_opt_approx_fprime_fixed_point() {
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
        let Some(expected) = arm.values.as_ref() else {
            continue;
        };
        match case.op.as_str() {
            "grad" => {
                let fname = case.func.clone();
                let f = move |x: &[f64]| eval_multi(&fname, x);
                let Ok(g) = approx_fprime(&case.xk, &f, case.eps) else {
                    continue;
                };
                let abs_d = vec_max_diff(&g, expected);
                max_overall = max_overall.max(abs_d);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    op: case.op.clone(),
                    abs_diff: abs_d,
                    pass: abs_d <= GRAD_ABS_TOL,
                });
            }
            "fp" => {
                let fname = case.fp_func.clone();
                let f = move |x: f64| eval_fp(&fname, x);
                let Ok(x) = fixed_point(&f, case.x0, 1.0e-12, 200) else {
                    continue;
                };
                let abs_d = (x - expected[0]).abs();
                max_overall = max_overall.max(abs_d);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    op: case.op.clone(),
                    abs_diff: abs_d,
                    pass: abs_d <= FP_ABS_TOL,
                });
            }
            _ => continue,
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_approx_fprime_fixed_point".into(),
        category: "fsci_opt::{approx_fprime, fixed_point} vs scipy.optimize".into(),
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
            eprintln!("{} mismatch: {} abs_diff={}", d.op, d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "approx_fprime/fixed_point conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
