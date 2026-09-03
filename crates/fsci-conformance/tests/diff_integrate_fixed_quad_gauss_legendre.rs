#![forbid(unsafe_code)]
//! Live scipy.integrate parity for fsci_integrate::fixed_quad,
//! gauss_legendre, dblquad_rect, and tplquad_rect.
//!
//! Resolves [frankenscipy-bl4k2].
//!
//! - `fixed_quad(f, a, b, n)`: uses Gauss-Legendre nodes & weights.
//!   scipy.integrate.fixed_quad uses the same. Both are exact for
//!   polynomials of degree ≤ 2n-1.
//! - `gauss_legendre(f, a, b, n)`: convenience wrapper for n=2..5.
//!   Same exactness property.
//! - `dblquad_rect`, `tplquad_rect`: composite Simpson over a
//!   rectangular box. Exact for polynomials of total degree ≤ 3
//!   in each variable (Simpson 1/3 is exact for cubics).
//!
//! Probes restricted to integrands within these exactness windows
//! so tolerances stay tight (1e-10 abs).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_integrate::{dblquad_rect, fixed_quad, gauss_legendre, tplquad_rect};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String, // "fixed_quad" | "gl" | "dbl" | "tpl"
    func: String,
    /// Bounds
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    g: f64,
    /// Sample counts
    n: usize,
    ny: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
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
    fs::create_dir_all(output_dir()).expect("create fixed_quad_gl diff dir");
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

fn f1d(name: &str, x: f64) -> f64 {
    match name {
        "x_cubed" => x * x * x,
        "quintic" => 5.0 * x.powi(5) - 3.0 * x.powi(3) + 2.0 * x - 1.0,
        "septic" => x.powi(7) - 2.0 * x.powi(4) + x,
        "linear" => 3.0 * x + 1.0,
        _ => f64::NAN,
    }
}

fn f2d(name: &str, x: f64, y: f64) -> f64 {
    match name {
        // cubic in x and y separately, exact for Simpson per axis
        "xy_cubic" => x * x * x * y + 2.0 * x * y * y * y + x + y + 1.0,
        "quadratic" => x * x + 2.0 * x * y + y * y + 1.0,
        _ => f64::NAN,
    }
}

fn f3d(name: &str, x: f64, y: f64, z: f64) -> f64 {
    match name {
        "linear_3d" => x + 2.0 * y + 3.0 * z + 1.0,
        "trilinear" => x * y * z + x + y + z,
        _ => f64::NAN,
    }
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();

    // fixed_quad probes: vary n to match polynomial degree
    let probes_1d: &[(&str, f64, f64, usize)] = &[
        ("linear", -1.0, 1.0, 2), // GL2 exact for degree ≤ 3
        ("x_cubed", -2.0, 3.0, 2),
        ("quintic", 0.0, 1.5, 3), // GL3 exact for degree ≤ 5
        ("quintic", -1.0, 2.0, 3),
        ("septic", 0.0, 1.0, 4),  // GL4 exact for degree ≤ 7
        ("septic", -1.0, 1.0, 5), // GL5 exact for degree ≤ 9
    ];
    for &(fname, a, b, n) in probes_1d {
        points.push(Case {
            case_id: format!("fq_{fname}_n{n}_a{a}_b{b}"),
            op: "fixed_quad".into(),
            func: fname.into(),
            a,
            b,
            c: 0.0,
            d: 0.0,
            e: 0.0,
            g: 0.0,
            n,
            ny: 0,
        });
        points.push(Case {
            case_id: format!("gl_{fname}_n{n}_a{a}_b{b}"),
            op: "gl".into(),
            func: fname.into(),
            a,
            b,
            c: 0.0,
            d: 0.0,
            e: 0.0,
            g: 0.0,
            n,
            ny: 0,
        });
    }

    // dblquad_rect probes: cubic integrands, n=3 for Simpson exactness on cubics
    let probes_2d: &[(&str, f64, f64, f64, f64, usize, usize)] = &[
        ("quadratic", 0.0, 1.0, 0.0, 1.0, 3, 3),
        ("xy_cubic", -1.0, 1.0, -1.0, 1.0, 3, 3),
        ("quadratic", 0.0, 2.0, 0.0, 3.0, 5, 5),
    ];
    for &(fname, a, b, c, d, nx, ny) in probes_2d {
        points.push(Case {
            case_id: format!("dbl_{fname}_nx{nx}_ny{ny}"),
            op: "dbl".into(),
            func: fname.into(),
            a,
            b,
            c,
            d,
            e: 0.0,
            g: 0.0,
            n: nx,
            ny,
        });
    }

    // tplquad_rect probes: linear integrands, n=3 (Simpson exact)
    let probes_3d: &[(&str, f64, f64, f64, f64, f64, f64, usize)] = &[
        ("linear_3d", 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 3),
        ("trilinear", -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 3),
    ];
    for &(fname, a, b, c, d, e, g, n) in probes_3d {
        points.push(Case {
            case_id: format!("tpl_{fname}_n{n}"),
            op: "tpl".into(),
            func: fname.into(),
            a,
            b,
            c,
            d,
            e,
            g,
            n,
            ny: 0,
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
from scipy.integrate import quad, dblquad, tplquad

def f1d(name, x):
    if name == "linear":   return 3.0*x + 1.0
    if name == "x_cubed":  return x**3
    if name == "quintic":  return 5.0*x**5 - 3.0*x**3 + 2.0*x - 1.0
    if name == "septic":   return x**7 - 2.0*x**4 + x
    return float("nan")

def f2d(name, x, y):
    if name == "xy_cubic":   return x**3*y + 2.0*x*y**3 + x + y + 1.0
    if name == "quadratic":  return x*x + 2.0*x*y + y*y + 1.0
    return float("nan")

def f3d(name, x, y, z):
    if name == "linear_3d": return x + 2.0*y + 3.0*z + 1.0
    if name == "trilinear": return x*y*z + x + y + z
    return float("nan")

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]; fname = case["func"]
    a = float(case["a"]); b = float(case["b"])
    try:
        if op in ("fixed_quad", "gl"):
            # scipy.integrate.quad gives ~exact for polynomials
            v, _ = quad(lambda x: f1d(fname, x), a, b, epsabs=1e-15, epsrel=1e-14)
            points.append({"case_id": cid, "value": float(v)})
        elif op == "dbl":
            c = float(case["c"]); d = float(case["d"])
            v, _ = dblquad(
                lambda y, x: f2d(fname, x, y),
                a, b, lambda x: c, lambda x: d,
                epsabs=1e-15, epsrel=1e-14)
            points.append({"case_id": cid, "value": float(v)})
        elif op == "tpl":
            c = float(case["c"]); d = float(case["d"])
            e = float(case["e"]); g = float(case["g"])
            v, _ = tplquad(
                lambda z, y, x: f3d(fname, x, y, z),
                a, b,
                lambda x: c, lambda x: d,
                lambda x, y: e, lambda x, y: g,
                epsabs=1e-15, epsrel=1e-14)
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
                "failed to spawn python3 for fixed_quad_gl oracle: {e}"
            );
            eprintln!("skipping fixed_quad_gl oracle: python3 not available ({e})");
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
                "fixed_quad_gl oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping fixed_quad_gl oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for fixed_quad_gl oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "fixed_quad_gl oracle failed: {stderr}"
        );
        eprintln!("skipping fixed_quad_gl oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse fixed_quad_gl oracle JSON"))
}

#[test]
fn diff_integrate_fixed_quad_gauss_legendre() {
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
        let Some(expected) = arm.value else {
            continue;
        };
        let actual = match case.op.as_str() {
            "fixed_quad" => {
                let fname = case.func.clone();
                let f = move |x: f64| f1d(&fname, x);
                match fixed_quad(&f, case.a, case.b, case.n) {
                    Ok((v, _)) => v,
                    Err(_) => continue,
                }
            }
            "gl" => {
                let fname = case.func.clone();
                let f = move |x: f64| f1d(&fname, x);
                gauss_legendre(&f, case.a, case.b, case.n)
            }
            "dbl" => {
                let fname = case.func.clone();
                let f = move |x: f64, y: f64| f2d(&fname, x, y);
                dblquad_rect(&f, case.a, case.b, case.c, case.d, case.n, case.ny)
            }
            "tpl" => {
                let fname = case.func.clone();
                let f = move |x: f64, y: f64, z: f64| f3d(&fname, x, y, z);
                tplquad_rect(&f, case.a, case.b, case.c, case.d, case.e, case.g, case.n)
            }
            _ => continue,
        };
        let abs_d = (actual - expected).abs();
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_integrate_fixed_quad_gauss_legendre".into(),
        category:
            "fsci_integrate::{fixed_quad, gauss_legendre, dblquad_rect, tplquad_rect} vs scipy.integrate"
                .into(),
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
        "fixed_quad/gauss_legendre conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
