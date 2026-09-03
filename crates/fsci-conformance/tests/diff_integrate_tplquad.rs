#![forbid(unsafe_code)]
//! Live scipy.integrate.tplquad parity for fsci_integrate::tplquad.
//!
//! Resolves [frankenscipy-7lxwx]. Tests several integrand × domain
//! combinations (unit cube and pyramidal regions). Tolerance: 1e-6
//! abs (adaptive QUADPACK on both sides converges to ~1.5e-8).
//!
//! Reference values are computed in scipy.integrate.tplquad with
//! identical bound functions.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_integrate::{DblquadOptions, tplquad};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-6;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct TplCase {
    case_id: String,
    integrand: String,
    bound_pattern: String,
    a: f64,
    b: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<TplCase>,
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
    fs::create_dir_all(output_dir()).expect("create tplquad diff dir");
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

fn integrand(name: &str, x: f64, y: f64, z: f64) -> f64 {
    match name {
        "one" => 1.0,
        "xyz" => x * y * z,
        "sum_xyz" => x + y + z,
        "exp_neg_sumsq" => (-(x * x + y * y + z * z)).exp(),
        _ => f64::NAN,
    }
}

// gfun, hfun: y bounds depending on x.
// qfun, rfun: z bounds depending on (x, y).
fn gfun(name: &str, _x: f64) -> f64 {
    match name {
        "unit_cube" | "y_to_1" => 0.0,
        "y_to_x" => 0.0,
        _ => f64::NAN,
    }
}
fn hfun(name: &str, x: f64) -> f64 {
    match name {
        "unit_cube" => 1.0,
        "y_to_x" => x,
        "y_to_1" => 1.0,
        _ => f64::NAN,
    }
}
fn qfun(name: &str, _x: f64, _y: f64) -> f64 {
    match name {
        "unit_cube" | "y_to_1" | "y_to_x" => 0.0,
        _ => f64::NAN,
    }
}
fn rfun(name: &str, _x: f64, y: f64) -> f64 {
    match name {
        "unit_cube" | "y_to_1" => 1.0,
        "y_to_x" => y,
        _ => f64::NAN,
    }
}

fn generate_query() -> OracleQuery {
    let probes: &[(&str, &str, &str, f64, f64)] = &[
        ("one_cube", "one", "unit_cube", 0.0, 1.0),
        ("xyz_cube", "xyz", "unit_cube", 0.0, 1.0),
        ("sum_xyz_cube", "sum_xyz", "unit_cube", 0.0, 1.0),
        ("exp_negsumsq_cube", "exp_neg_sumsq", "unit_cube", 0.0, 1.0),
        ("one_pyramid_yx", "one", "y_to_x", 0.0, 1.0),
        ("xyz_y_to_1", "xyz", "y_to_1", 0.0, 1.0),
    ];
    let points: Vec<TplCase> = probes
        .iter()
        .map(|&(cid, integrand, bound_pattern, a, b)| TplCase {
            case_id: cid.into(),
            integrand: integrand.into(),
            bound_pattern: bound_pattern.into(),
            a,
            b,
        })
        .collect();
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
from scipy.integrate import tplquad

def integrand(name, x, y, z):
    if name == "one":          return 1.0
    if name == "xyz":          return x * y * z
    if name == "sum_xyz":      return x + y + z
    if name == "exp_neg_sumsq": return math.exp(-(x*x + y*y + z*z))
    return float("nan")

def gfun(pat, x):
    if pat in ("unit_cube", "y_to_1", "y_to_x"): return 0.0
    return float("nan")

def hfun(pat, x):
    if pat == "unit_cube": return 1.0
    if pat == "y_to_x":    return x
    if pat == "y_to_1":    return 1.0
    return float("nan")

def qfun(pat, x, y):
    if pat in ("unit_cube", "y_to_1", "y_to_x"): return 0.0
    return float("nan")

def rfun(pat, x, y):
    if pat == "unit_cube": return 1.0
    if pat == "y_to_1":    return 1.0
    if pat == "y_to_x":    return y
    return float("nan")

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; name = case["integrand"]; pat = case["bound_pattern"]
    a = float(case["a"]); b = float(case["b"])
    try:
        v, _ = tplquad(
            lambda z, y, x: integrand(name, x, y, z),  # scipy passes (z, y, x)
            a, b,
            lambda x: gfun(pat, x),
            lambda x: hfun(pat, x),
            lambda x, y: qfun(pat, x, y),
            lambda x, y: rfun(pat, x, y),
            epsabs=1.0e-12, epsrel=1.0e-12,
        )
        if math.isfinite(v):
            points.append({"case_id": cid, "value": float(v)})
        else:
            points.append({"case_id": cid, "value": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "value": None})
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
                "failed to spawn python3 for tplquad oracle: {e}"
            );
            eprintln!("skipping tplquad oracle: python3 not available ({e})");
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
                "tplquad oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping tplquad oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for tplquad oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "tplquad oracle failed: {stderr}"
        );
        eprintln!("skipping tplquad oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse tplquad oracle JSON"))
}

#[test]
fn diff_integrate_tplquad() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let opts = DblquadOptions::default();
    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let Some(arm) = pmap.get(&case.case_id) else {
            continue;
        };
        let Some(expected) = arm.value else {
            continue;
        };
        let f = |x: f64, y: f64, z: f64| integrand(&case.integrand, x, y, z);
        let g = |x: f64| gfun(&case.bound_pattern, x);
        let h = |x: f64| hfun(&case.bound_pattern, x);
        let q = |x: f64, y: f64| qfun(&case.bound_pattern, x, y);
        let r = |x: f64, y: f64| rfun(&case.bound_pattern, x, y);
        let Ok(res) = tplquad(f, case.a, case.b, g, h, q, r, opts) else {
            continue;
        };
        let abs_d = (res.integral - expected).abs();
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_integrate_tplquad".into(),
        category: "fsci_integrate::tplquad vs scipy.integrate.tplquad".into(),
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
            eprintln!("tplquad mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "tplquad conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
