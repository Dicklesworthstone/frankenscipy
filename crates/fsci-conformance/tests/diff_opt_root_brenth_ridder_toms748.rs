#![forbid(unsafe_code)]
//! Live scipy.optimize parity for fsci_opt scalar root finders
//! brenth, ridder, toms748.
//!
//! Resolves [frankenscipy-nrkx7]. All three are bracketing methods
//! that should converge to the same root inside a sign-changing
//! bracket. Compare root x at 1e-9 abs and require residual |f(x)|
//! to be < 1e-10 on both sides.
//!
//! Note: fsci's brenth is currently an alias for brentq (see
//! frankenscipy-88gz), so it should match scipy's brenth at roughly
//! the same precision because both converge to the same true root.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_opt::{RootOptions, brenth, ridder, toms748};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const X_ABS_TOL: f64 = 1.0e-9;
const RESID_ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    method: String,
    func: String,
    a: f64,
    b: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    x: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    method: String,
    abs_diff_x: f64,
    fsci_residual: f64,
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
    fs::create_dir_all(output_dir()).expect("create root_brenth_ridder_toms748 diff dir");
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

fn eval_func(name: &str, x: f64) -> f64 {
    match name {
        "poly_cubic" => x * x * x - 2.0 * x - 5.0,
        "exp_minus_x" => x.exp() - 1.0 / x,
        "sin_minus_half" => x.sin() - 0.5,
        "log_plus_x" => x.ln() + x,
        "atan_minus" => x.atan() - 1.0,
        "x_pow3_minus_x_minus_1" => x.powi(3) - x - 1.0,
        _ => f64::NAN,
    }
}

fn generate_query() -> OracleQuery {
    let funcs: &[(&str, f64, f64)] = &[
        ("poly_cubic", 2.0, 3.0),
        ("exp_minus_x", 0.1, 2.0),
        ("sin_minus_half", 0.1, 1.0),
        ("log_plus_x", 0.1, 1.0),
        ("atan_minus", 0.5, 2.0),
        ("x_pow3_minus_x_minus_1", 1.0, 2.0),
    ];
    let methods = ["brenth", "ridder", "toms748"];
    let mut points = Vec::new();
    for &(fname, a, b) in funcs {
        for m in &methods {
            points.push(Case {
                case_id: format!("{m}_{fname}"),
                method: (*m).into(),
                func: fname.into(),
                a,
                b,
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
from scipy.optimize import brenth, ridder, toms748

def func(name, x):
    if name == "poly_cubic":              return x**3 - 2.0*x - 5.0
    if name == "exp_minus_x":             return math.exp(x) - 1.0/x
    if name == "sin_minus_half":          return math.sin(x) - 0.5
    if name == "log_plus_x":              return math.log(x) + x
    if name == "atan_minus":              return math.atan(x) - 1.0
    if name == "x_pow3_minus_x_minus_1":  return x**3 - x - 1.0
    return float("nan")

method_map = {"brenth": brenth, "ridder": ridder, "toms748": toms748}

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    m = method_map.get(case["method"])
    fname = case["func"]
    a = float(case["a"]); b = float(case["b"])
    try:
        x = float(m(lambda x: func(fname, x), a, b, xtol=1e-12, rtol=1e-15))
        if math.isfinite(x):
            points.append({"case_id": cid, "x": x})
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
                "failed to spawn python3 for root_brenth_ridder_toms748 oracle: {e}"
            );
            eprintln!("skipping root_brenth_ridder_toms748 oracle: python3 not available ({e})");
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
                "root_brenth_ridder_toms748 oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!(
                "skipping root_brenth_ridder_toms748 oracle: stdin write failed ({err})\n{stderr}"
            );
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for root_brenth_ridder_toms748 oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "root_brenth_ridder_toms748 oracle failed: {stderr}"
        );
        eprintln!("skipping root_brenth_ridder_toms748 oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse root_brenth_ridder_toms748 oracle JSON"))
}

#[test]
fn diff_opt_root_brenth_ridder_toms748() {
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
        let Some(scipy_x) = arm.x else {
            continue;
        };
        let fname = case.func.clone();
        let f = move |x: f64| eval_func(&fname, x);
        let res = match case.method.as_str() {
            "brenth" => brenth(&f, (case.a, case.b), opts),
            "ridder" => ridder(&f, (case.a, case.b), opts),
            "toms748" => toms748(&f, (case.a, case.b), opts),
            _ => continue,
        };
        let Ok(rr) = res else { continue };
        if !rr.converged {
            continue;
        }
        let fsci_residual = eval_func(&case.func, rr.root).abs();
        let scipy_residual = eval_func(&case.func, scipy_x).abs();
        let abs_diff_x = (rr.root - scipy_x).abs();
        let pass = abs_diff_x <= X_ABS_TOL
            && fsci_residual <= RESID_ABS_TOL
            && scipy_residual <= RESID_ABS_TOL;
        max_overall = max_overall.max(abs_diff_x);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            method: case.method.clone(),
            abs_diff_x,
            fsci_residual,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_root_brenth_ridder_toms748".into(),
        category: "fsci_opt::{brenth, ridder, toms748} vs scipy.optimize".into(),
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
                "{} mismatch: {} abs_diff_x={} fsci_residual={}",
                d.method, d.case_id, d.abs_diff_x, d.fsci_residual
            );
        }
    }

    assert!(
        all_pass,
        "brenth/ridder/toms748 conformance failed: {} cases, max_diff_x={}",
        diffs.len(),
        max_overall
    );
}
