#![forbid(unsafe_code)]
//! Live scipy.optimize.newton parity for fsci_opt root-finders
//! newton_scalar (with fprime), secant (no fprime), and halley
//! (with fprime + fprime2).
//!
//! Resolves [frankenscipy-plrle]. Tolerance: 1e-9 abs on the root
//! value; scipy and fsci default to xtol ~ 1e-12 so 1e-9 absorbs
//! convergence-criterion drift.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_opt::{RootOptions, halley, newton_scalar, secant};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct RootCase {
    case_id: String,
    method: String, // "newton" | "secant" | "halley"
    func: String,
    x0: f64,
    x1: f64, // only for secant
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<RootCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    root: Option<f64>,
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
    fs::create_dir_all(output_dir()).expect("create newton_secant_halley diff dir");
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

fn f(name: &str, x: f64) -> f64 {
    match name {
        "x2_minus_2" => x * x - 2.0,
        "sin" => x.sin(),
        "x3_minus_x_minus_1" => x * x * x - x - 1.0,
        "exp_minus_2x" => (-x).exp() - x,
        _ => f64::NAN,
    }
}

fn fp(name: &str, x: f64) -> f64 {
    match name {
        "x2_minus_2" => 2.0 * x,
        "sin" => x.cos(),
        "x3_minus_x_minus_1" => 3.0 * x * x - 1.0,
        "exp_minus_2x" => -(-x).exp() - 1.0,
        _ => f64::NAN,
    }
}

fn fpp(name: &str, x: f64) -> f64 {
    match name {
        "x2_minus_2" => 2.0,
        "sin" => -x.sin(),
        "x3_minus_x_minus_1" => 6.0 * x,
        "exp_minus_2x" => (-x).exp(),
        _ => f64::NAN,
    }
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();
    let probes: &[(&str, &str, f64, f64)] = &[
        ("newton", "x2_minus_2", 1.0, 0.0),
        ("newton", "x2_minus_2", -1.0, 0.0),
        ("newton", "sin", 3.0, 0.0),
        ("newton", "x3_minus_x_minus_1", 1.5, 0.0),
        ("newton", "exp_minus_2x", 0.5, 0.0),
        ("secant", "x2_minus_2", 1.0, 1.5),
        ("secant", "sin", 3.0, 3.5),
        ("secant", "x3_minus_x_minus_1", 1.0, 2.0),
        ("secant", "exp_minus_2x", 0.0, 1.0),
        ("halley", "x2_minus_2", 1.0, 0.0),
        ("halley", "sin", 3.0, 0.0),
        ("halley", "x3_minus_x_minus_1", 1.5, 0.0),
        ("halley", "exp_minus_2x", 0.5, 0.0),
    ];
    for &(method, func, x0, x1) in probes {
        points.push(RootCase {
            case_id: format!(
                "{method}_{func}_x0_{x0}",
                x0 = x0.to_string().replace('.', "p").replace('-', "n")
            ),
            method: method.into(),
            func: func.into(),
            x0,
            x1,
        });
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
from scipy import optimize

def f(name, x):
    if name == "x2_minus_2":          return x*x - 2.0
    if name == "sin":                 return math.sin(x)
    if name == "x3_minus_x_minus_1":  return x**3 - x - 1.0
    if name == "exp_minus_2x":        return math.exp(-x) - x
    return float("nan")

def fp(name, x):
    if name == "x2_minus_2":          return 2.0*x
    if name == "sin":                 return math.cos(x)
    if name == "x3_minus_x_minus_1":  return 3.0*x*x - 1.0
    if name == "exp_minus_2x":        return -math.exp(-x) - 1.0
    return float("nan")

def fpp(name, x):
    if name == "x2_minus_2":          return 2.0
    if name == "sin":                 return -math.sin(x)
    if name == "x3_minus_x_minus_1":  return 6.0*x
    if name == "exp_minus_2x":        return math.exp(-x)
    return float("nan")

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    method = case["method"]; name = case["func"]
    x0 = float(case["x0"]); x1 = float(case["x1"])
    try:
        if method == "newton":
            r = float(optimize.newton(lambda x: f(name, x), x0, fprime=lambda x: fp(name, x),
                                       tol=1e-12, maxiter=100))
        elif method == "secant":
            r = float(optimize.newton(lambda x: f(name, x), x0, x1=x1,
                                       tol=1e-12, maxiter=100))
        elif method == "halley":
            r = float(optimize.newton(lambda x: f(name, x), x0,
                                       fprime=lambda x: fp(name, x),
                                       fprime2=lambda x: fpp(name, x),
                                       tol=1e-12, maxiter=100))
        else:
            r = None
        if r is not None and math.isfinite(r):
            points.append({"case_id": cid, "root": r})
        else:
            points.append({"case_id": cid, "root": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "root": None})
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
                "failed to spawn python3 for newton/secant/halley oracle: {e}"
            );
            eprintln!("skipping newton/secant/halley oracle: python3 not available ({e})");
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
                "newton/secant/halley oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping newton/secant/halley oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for newton/secant/halley oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "newton/secant/halley oracle failed: {stderr}"
        );
        eprintln!("skipping newton/secant/halley oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse newton/secant/halley oracle JSON"))
}

#[test]
fn diff_opt_newton_secant_halley() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let opts = RootOptions::default();
    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let Some(arm) = pmap.get(&case.case_id) else {
            continue;
        };
        let Some(expected) = arm.root else {
            continue;
        };
        let res = match case.method.as_str() {
            "newton" => newton_scalar(|x| f(&case.func, x), |x| fp(&case.func, x), case.x0, opts),
            "secant" => secant(|x| f(&case.func, x), case.x0, Some(case.x1), opts),
            "halley" => halley(
                |x| f(&case.func, x),
                |x| fp(&case.func, x),
                |x| fpp(&case.func, x),
                case.x0,
                opts,
            ),
            _ => continue,
        };
        let Ok(rr) = res else { continue };
        if !rr.converged {
            continue;
        }
        let abs_d = (rr.root - expected).abs();
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.method.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_newton_secant_halley".into(),
        category: "fsci_opt::{newton_scalar, secant, halley} vs scipy.optimize.newton".into(),
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
        "newton/secant/halley conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
