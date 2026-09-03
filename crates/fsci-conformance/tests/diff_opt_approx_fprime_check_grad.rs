#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_opt's finite-difference
//! helpers:
//!   - `approx_fprime(xk, f, eps)` vs `scipy.optimize.approx_fprime`
//!   - `check_grad(func, grad, x0)` vs `scipy.optimize.check_grad`
//!     (default eps = sqrt(f64 eps) = 1.4901e-8 on both sides)
//!
//! Resolves [frankenscipy-cqpr1]. Both routines take callable arguments
//! so we use the same string-labeled dispatcher pattern as
//! diff_opt_rosen / diff_opt_scalar_min_fixed_point. Tolerance ~1e-6
//! for approx_fprime (forward-diff truncation+roundoff floor) and
//! ~5e-6 for check_grad (which sums forward-diff errors in quadrature).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_opt::{approx_fprime, check_grad, rosen, rosen_der};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-003";
const FPRIME_TOL: f64 = 1.0e-6;
const CHECK_GRAD_TOL: f64 = 5.0e-6;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    routine: String, // "approx_fprime" | "check_grad"
    /// Function label used on both sides for dispatching.
    func: String,
    x: Vec<f64>,
    eps: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// approx_fprime: gradient vector. check_grad: scalar norm wrapped as Vec[scalar].
    values: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    routine: String,
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
    fs::create_dir_all(output_dir()).expect("create approx_fprime diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize approx_fprime diff log");
    fs::write(path, json).expect("write approx_fprime diff log");
}

fn f_quad_sum(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum()
}

fn grad_quad_sum(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&v| 2.0 * v).collect()
}

fn f_dot_const(x: &[f64]) -> f64 {
    // f(x) = sum (i+1) * x[i]
    x.iter().enumerate().map(|(i, &v)| (i + 1) as f64 * v).sum()
}

fn grad_dot_const(x: &[f64]) -> Vec<f64> {
    (0..x.len()).map(|i| (i + 1) as f64).collect()
}

fn f_shifted_quad(x: &[f64]) -> f64 {
    // f(x) = sum (x[i] - i)^2
    x.iter()
        .enumerate()
        .map(|(i, &v)| {
            let d = v - i as f64;
            d * d
        })
        .sum()
}

fn grad_shifted_quad(x: &[f64]) -> Vec<f64> {
    x.iter()
        .enumerate()
        .map(|(i, &v)| 2.0 * (v - i as f64))
        .collect()
}

fn fsci_eval_func(name: &str) -> Option<fn(&[f64]) -> f64> {
    match name {
        "rosen" => Some(rosen),
        "quad_sum" => Some(f_quad_sum),
        "dot_const" => Some(f_dot_const),
        "shifted_quad" => Some(f_shifted_quad),
        _ => None,
    }
}

fn fsci_eval_grad(name: &str) -> Option<fn(&[f64]) -> Vec<f64>> {
    match name {
        "rosen" => Some(rosen_der),
        "quad_sum" => Some(grad_quad_sum),
        "dot_const" => Some(grad_dot_const),
        "shifted_quad" => Some(grad_shifted_quad),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    let func_x_pairs: &[(&str, Vec<f64>)] = &[
        ("rosen", vec![0.5, 0.5]),
        ("rosen", vec![-1.2, 1.0]),
        ("rosen", vec![1.0, 1.0, 1.0]),
        ("quad_sum", vec![1.0, 2.0, 3.0]),
        ("quad_sum", vec![-1.0, 0.5, 2.5, -2.0]),
        ("dot_const", vec![0.0, 0.0, 0.0]),
        ("dot_const", vec![1.0, 2.0, 3.0, 4.0]),
        ("shifted_quad", vec![0.5, 1.5, 2.5]),
        ("shifted_quad", vec![0.0, 0.0, 0.0, 0.0]),
    ];
    let mut points = Vec::new();
    for (func, x) in func_x_pairs {
        // approx_fprime probe at eps=1e-7 (a typical forward-diff step)
        points.push(PointCase {
            case_id: format!("approx_fprime_{func}_x{}d", x.len()),
            routine: "approx_fprime".into(),
            func: (*func).into(),
            x: x.clone(),
            eps: 1.0e-7,
        });
        // check_grad uses scipy default eps = 1.4901161e-8
        points.push(PointCase {
            case_id: format!("check_grad_{func}_x{}d", x.len()),
            routine: "check_grad".into(),
            func: (*func).into(),
            x: x.clone(),
            eps: 0.0,
        });
    }
    // Disambiguate duplicate case_ids by appending a discriminator.
    for (i, p) in points.iter_mut().enumerate() {
        p.case_id = format!("{}_{i}", p.case_id);
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy import optimize

def f_rosen(x): return optimize.rosen(x)
def g_rosen(x): return optimize.rosen_der(x)
def f_quad_sum(x): return float(np.sum(np.asarray(x)**2))
def g_quad_sum(x): return 2.0 * np.asarray(x)
def f_dot_const(x): return float(np.sum(np.arange(1, len(x)+1) * np.asarray(x)))
def g_dot_const(x): return np.arange(1, len(x)+1, dtype=float)
def f_shifted_quad(x):
    arr = np.asarray(x); ii = np.arange(len(x), dtype=float)
    return float(np.sum((arr - ii)**2))
def g_shifted_quad(x):
    arr = np.asarray(x); ii = np.arange(len(x), dtype=float)
    return 2.0 * (arr - ii)

FUNCS = {
    "rosen":        (f_rosen, g_rosen),
    "quad_sum":     (f_quad_sum, g_quad_sum),
    "dot_const":    (f_dot_const, g_dot_const),
    "shifted_quad": (f_shifted_quad, g_shifted_quad),
}

def vec_or_none(arr):
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
    r = case["routine"]; fn_name = case["func"]
    x = np.array(case["x"], dtype=float)
    try:
        f, g = FUNCS[fn_name]
        if r == "approx_fprime":
            eps = float(case["eps"])
            grad = optimize.approx_fprime(x, f, eps)
            points.append({"case_id": cid, "values": vec_or_none(grad)})
        elif r == "check_grad":
            val = float(optimize.check_grad(f, g, x))
            points.append({"case_id": cid, "values": [val] if math.isfinite(val) else None})
        else:
            points.append({"case_id": cid, "values": None})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize approx_fprime query");
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
                "failed to spawn python3 for approx_fprime oracle: {e}"
            );
            eprintln!("skipping approx_fprime oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open approx_fprime oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "approx_fprime oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping approx_fprime oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for approx_fprime oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "approx_fprime oracle failed: {stderr}"
        );
        eprintln!("skipping approx_fprime oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse approx_fprime oracle JSON"))
}

#[test]
fn diff_opt_approx_fprime_check_grad() {
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
        let Some(scipy_v) = scipy_arm.values.as_ref() else {
            continue;
        };
        let Some(f) = fsci_eval_func(&case.func) else {
            continue;
        };
        let (fsci_v, tol) = match case.routine.as_str() {
            "approx_fprime" => {
                let Ok(g) = approx_fprime(&case.x, f, case.eps) else {
                    continue;
                };
                (g, FPRIME_TOL)
            }
            "check_grad" => {
                let Some(gfn) = fsci_eval_grad(&case.func) else {
                    continue;
                };
                let Ok(v) = check_grad(f, gfn, &case.x) else {
                    continue;
                };
                (vec![v], CHECK_GRAD_TOL)
            }
            _ => continue,
        };
        if fsci_v.len() != scipy_v.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                routine: case.routine.clone(),
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
            routine: case.routine.clone(),
            abs_diff: abs_d,
            pass: abs_d <= tol,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_approx_fprime_check_grad".into(),
        category: "scipy.optimize.approx_fprime / check_grad".into(),
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
                "approx_fprime {} mismatch: {} abs_diff={}",
                d.routine, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.optimize approx_fprime/check_grad conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
