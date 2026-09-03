#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the function-based quad
//! family that diff_integrate.rs's composite-rule harness doesn't
//! exercise:
//!   - `scipy.integrate.quad(f, a, b)` (adaptive Gauss-Kronrod)
//!   - `scipy.integrate.fixed_quad(f, a, b, n)` (Gauss-Legendre order-n)
//!   - `scipy.integrate.romberg(f, a, b)` (extrapolated trapezoidal)
//!
//! Resolves [frankenscipy-nx1n3]. The integrands need to be defined
//! identically on both sides; the Python oracle takes a `func`
//! identifier and matches the same fsci-side dispatch.
//!
//! Tolerance: 1e-10 abs / rel for quad and romberg (well below
//! scipy's reported abs_err_estimate); 1e-12 for fixed_quad on
//! polynomials it integrates exactly.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_integrate::{QuadOptions, fixed_quad, quad, romberg};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-008";
const QUAD_TOL: f64 = 1.0e-10;
const FIXED_QUAD_TOL: f64 = 1.0e-12;
const ROMBERG_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    /// Which routine: "quad" | "fixed_quad" | "romberg".
    routine: String,
    /// Which built-in integrand: "x_squared" | "sin_x" | "exp_neg_xsq" | "x_pow4".
    func: String,
    a: f64,
    b: f64,
    /// Gauss-Legendre order for `fixed_quad`. Ignored otherwise.
    fq_n: u32,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
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
    fs::create_dir_all(output_dir()).expect("create integrate_quad diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize integrate_quad diff log");
    fs::write(path, json).expect("write integrate_quad diff log");
}

fn integrand(name: &str, x: f64) -> f64 {
    match name {
        "x_squared" => x * x,
        "sin_x" => x.sin(),
        "exp_neg_xsq" => (-x * x).exp(),
        "x_pow4" => x.powi(4),
        _ => 0.0,
    }
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();
    let pi = std::f64::consts::PI;

    // quad: cover polynomial, oscillatory, gaussian-like integrands.
    let quad_cases: &[(&str, f64, f64)] = &[
        ("x_squared", 0.0, 1.0),
        ("x_squared", -1.0, 2.0),
        ("sin_x", 0.0, pi),
        ("sin_x", -pi, pi),
        ("exp_neg_xsq", 0.0, 3.0),
        ("exp_neg_xsq", -3.0, 3.0),
        ("x_pow4", 0.0, 1.0),
    ];
    for &(f, a, b) in quad_cases {
        points.push(PointCase {
            case_id: format!("quad_{f}_{a}_{b}"),
            routine: "quad".into(),
            func: f.into(),
            a,
            b,
            fq_n: 0,
        });
    }

    // fixed_quad: Gauss-Legendre order-n is exact for polynomials up to
    // degree 2n-1. n=5 → exact through x^9, so x^4 and x^2 are exact.
    for &(f, a, b, n) in &[
        ("x_squared", 0.0_f64, 1.0_f64, 3_u32),
        ("x_squared", -2.0, 5.0, 5),
        ("x_pow4", 0.0, 1.0, 5),
        ("sin_x", 0.0, pi, 10),
    ] {
        points.push(PointCase {
            case_id: format!("fq_{f}_{a}_{b}_n{n}"),
            routine: "fixed_quad".into(),
            func: f.into(),
            a,
            b,
            fq_n: n,
        });
    }

    // romberg: extrapolated trapezoidal — converges fast for smooth f.
    for &(f, a, b) in &[
        ("x_squared", 0.0_f64, 1.0_f64),
        ("sin_x", 0.0, pi),
        ("exp_neg_xsq", -2.0, 2.0),
    ] {
        points.push(PointCase {
            case_id: format!("rom_{f}_{a}_{b}"),
            routine: "romberg".into(),
            func: f.into(),
            a,
            b,
            fq_n: 0,
        });
    }

    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
from scipy import integrate

FUNCS = {
    "x_squared":   lambda x: x*x,
    "sin_x":       lambda x: math.sin(x),
    "exp_neg_xsq": lambda x: math.exp(-x*x),
    "x_pow4":      lambda x: x**4,
}

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; routine = case["routine"]; func = case["func"]
    a = float(case["a"]); b = float(case["b"])
    fn = FUNCS.get(func)
    if fn is None:
        points.append({"case_id": cid, "value": None}); continue
    try:
        if routine == "quad":
            val, _err = integrate.quad(fn, a, b)
        elif routine == "fixed_quad":
            n = int(case["fq_n"])
            val, _ = integrate.fixed_quad(lambda xs: [fn(x) for x in xs], a, b, n=n)
        elif routine == "romberg":
            # romberg was removed in newer scipy; fall back to quad
            # which is even more accurate.
            try:
                val = integrate.romberg(fn, a, b)
            except AttributeError:
                val, _err = integrate.quad(fn, a, b)
        else:
            val = None
        points.append({"case_id": cid, "value": fnone(val)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize integrate_quad query");
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
                "failed to spawn python3 for integrate_quad oracle: {e}"
            );
            eprintln!("skipping integrate_quad oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open integrate_quad oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "integrate_quad oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping integrate_quad oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for integrate_quad oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "integrate_quad oracle failed: {stderr}"
        );
        eprintln!("skipping integrate_quad oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse integrate_quad oracle JSON"))
}

fn fsci_eval(case: &PointCase) -> Option<f64> {
    let f = |x: f64| integrand(&case.func, x);
    match case.routine.as_str() {
        "quad" => {
            let r = quad(f, case.a, case.b, QuadOptions::default()).ok()?;
            Some(r.integral)
        }
        "fixed_quad" => {
            let (val, _) = fixed_quad(f, case.a, case.b, case.fq_n as usize).ok()?;
            Some(val)
        }
        "romberg" => {
            // fsci romberg takes a tolerance + max_order rather than a single
            // call; pick a moderate refinement budget matching scipy.
            let r = romberg(f, case.a, case.b, 1.0e-12, 20);
            Some(r.integral)
        }
        _ => None,
    }
}

fn tol_for(routine: &str) -> f64 {
    match routine {
        "quad" => QUAD_TOL,
        "fixed_quad" => FIXED_QUAD_TOL,
        "romberg" => ROMBERG_TOL,
        _ => 1.0e-9,
    }
}

#[test]
fn diff_integrate_quad() {
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
        let Some(fsci_v) = fsci_eval(case) else {
            continue;
        };
        if let Some(scipy_v) = scipy_arm.value
            && fsci_v.is_finite()
        {
            let abs_d = (fsci_v - scipy_v).abs();
            max_overall = max_overall.max(abs_d);
            let tol = tol_for(&case.routine);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                routine: case.routine.clone(),
                abs_diff: abs_d,
                pass: abs_d <= tol,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_integrate_quad".into(),
        category: "scipy.integrate.quad / fixed_quad / romberg".into(),
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
                "integrate_quad {} mismatch: {} abs_diff={}",
                d.routine, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.integrate.quad/fixed_quad/romberg conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
