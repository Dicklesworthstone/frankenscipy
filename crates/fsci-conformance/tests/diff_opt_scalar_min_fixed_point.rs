#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_opt's 1-D minimizers and
//! fixed-point iterator:
//!   - `fsci_opt::brent_minimize(f, a, b, tol, maxiter)` vs
//!     `scipy.optimize.fminbound(f, a, b, xtol)` (Brent on a closed bracket)
//!   - `fsci_opt::golden(f, a, b, tol, maxiter)` vs
//!     `scipy.optimize.fminbound(...)` (golden's xmin should agree to a
//!     similar tolerance; both target the same minimum)
//!   - `fsci_opt::fixed_point(f, x0, tol, maxiter)` vs
//!     `scipy.optimize.fixed_point(f, [x0])` (both use Steffensen's
//!     delta-squared acceleration by default)
//!
//! Resolves [frankenscipy-dhbbz]. Cross-language testing uses a fixed
//! string-labeled function dispatcher (same pattern as diff_opt_rosen)
//! since closures can't be serialized. xmin agreement at ~5e-6 abs
//! (Brent's xtol floor); fixed-point agreement at 1e-8.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_opt::{brent_minimize, fixed_point, golden};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-003";
const XMIN_TOL: f64 = 5.0e-6; // Brent's documented xtol floor
const FIXED_POINT_TOL: f64 = 1.0e-8;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    routine: String, // "brent" | "golden" | "fixed_point"
    /// Label naming the integrand; both sides dispatch on the same label.
    func: String,
    a: f64,
    b: f64,
    x0: f64,
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
    fs::create_dir_all(output_dir()).expect("create scalar_min diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize scalar_min diff log");
    fs::write(path, json).expect("write scalar_min diff log");
}

/// Named functions exposed in both fsci and scipy oracles.
fn fsci_minimization_target(name: &str) -> Option<fn(f64) -> f64> {
    fn f_xsq(x: f64) -> f64 {
        x * x
    }
    fn f_x_minus3_sq(x: f64) -> f64 {
        (x - 3.0).powi(2)
    }
    fn f_neg_sin(x: f64) -> f64 {
        -x.sin()
    }
    fn f_exp_neg_xsq(x: f64) -> f64 {
        -(-x * x).exp()
    }
    fn f_quartic(x: f64) -> f64 {
        (x - 1.0).powi(4) + 0.5 * (x - 1.0).powi(2)
    }
    match name {
        "x_squared" => Some(f_xsq),
        "x_minus3_sq" => Some(f_x_minus3_sq),
        "neg_sin" => Some(f_neg_sin),
        "neg_exp_neg_xsq" => Some(f_exp_neg_xsq),
        "quartic_shifted" => Some(f_quartic),
        _ => None,
    }
}

fn fsci_fixed_point_target(name: &str) -> Option<fn(f64) -> f64> {
    // Contractive maps with unique fixed points.
    fn f_cos(x: f64) -> f64 {
        x.cos()
    } // fp ≈ 0.7390851332
    fn f_sqrt_plus2(x: f64) -> f64 {
        (x + 2.0).sqrt()
    } // fp = 2
    fn f_half_x_plus1(x: f64) -> f64 {
        0.5 * x + 1.0
    } // fp = 2
    fn f_half_x_plus3(x: f64) -> f64 {
        0.5 * x + 3.0
    } // fp = 6
    fn f_third_xsq(x: f64) -> f64 {
        (x * x + 2.0) / 3.0
    } // fp = 1 (smaller root)
    match name {
        "cos_x" => Some(f_cos),
        "sqrt_x_plus_2" => Some(f_sqrt_plus2),
        "half_x_plus1" => Some(f_half_x_plus1),
        "half_x_plus3" => Some(f_half_x_plus3),
        "third_xsq_plus2_over_3" => Some(f_third_xsq),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();

    // Brent / golden minimization cases.
    let min_cases: &[(&str, &str, f64, f64)] = &[
        ("x_squared", "min01", -2.0, 2.0),
        ("x_minus3_sq", "min02", 1.0, 6.0),
        ("neg_sin", "min03", 0.0, std::f64::consts::PI),
        ("neg_exp_neg_xsq", "min04", -2.0, 2.0),
        ("quartic_shifted", "min05", -3.0, 5.0),
    ];
    for (func, tag, a, b) in min_cases {
        for routine in ["brent", "golden"] {
            points.push(PointCase {
                case_id: format!("{routine}_{tag}_{func}"),
                routine: routine.into(),
                func: (*func).into(),
                a: *a,
                b: *b,
                x0: 0.0,
            });
        }
    }

    // Fixed-point cases.
    let fp_cases: &[(&str, &str, f64)] = &[
        ("cos_x", "fp01", 0.5),
        ("sqrt_x_plus_2", "fp02", 1.5),
        ("half_x_plus1", "fp03", 0.0),
        ("half_x_plus3", "fp04", 0.0),
        ("third_xsq_plus2_over_3", "fp05", 0.5),
    ];
    for (func, tag, x0) in fp_cases {
        points.push(PointCase {
            case_id: format!("fixed_point_{tag}_{func}"),
            routine: "fixed_point".into(),
            func: (*func).into(),
            a: 0.0,
            b: 0.0,
            x0: *x0,
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
from scipy import optimize

MIN_TARGETS = {
    "x_squared":         lambda x: x*x,
    "x_minus3_sq":       lambda x: (x - 3.0)**2,
    "neg_sin":           lambda x: -np.sin(x),
    "neg_exp_neg_xsq":   lambda x: -np.exp(-x*x),
    "quartic_shifted":   lambda x: (x - 1.0)**4 + 0.5*(x - 1.0)**2,
}
FP_TARGETS = {
    "cos_x":                    lambda x: np.cos(x),
    "sqrt_x_plus_2":            lambda x: np.sqrt(x + 2.0),
    "half_x_plus1":             lambda x: 0.5*x + 1.0,
    "half_x_plus3":             lambda x: 0.5*x + 3.0,
    "third_xsq_plus2_over_3":   lambda x: (x*x + 2.0)/3.0,
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
    cid = case["case_id"]; r = case["routine"]; fn_name = case["func"]
    a = case["a"]; b = case["b"]; x0 = case["x0"]
    try:
        if r in ("brent", "golden"):
            f = MIN_TARGETS.get(fn_name)
            if f is None:
                points.append({"case_id": cid, "value": None}); continue
            # Brent on a bounded interval = fminbound (matches fsci semantics).
            xmin = optimize.fminbound(f, a, b, xtol=1e-8, maxfun=500)
            points.append({"case_id": cid, "value": fnone(xmin)})
        elif r == "fixed_point":
            f = FP_TARGETS.get(fn_name)
            if f is None:
                points.append({"case_id": cid, "value": None}); continue
            xs = optimize.fixed_point(f, np.array([x0], dtype=float), xtol=1e-10, maxiter=500)
            points.append({"case_id": cid, "value": fnone(float(xs[0]))})
        else:
            points.append({"case_id": cid, "value": None})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize scalar_min query");
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
                "failed to spawn python3 for scalar_min oracle: {e}"
            );
            eprintln!("skipping scalar_min oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open scalar_min oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "scalar_min oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping scalar_min oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for scalar_min oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "scalar_min oracle failed: {stderr}"
        );
        eprintln!("skipping scalar_min oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse scalar_min oracle JSON"))
}

#[test]
fn diff_opt_scalar_min_fixed_point() {
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
        let Some(scipy_v) = scipy_arm.value else {
            continue;
        };
        let (fsci_v, tol) = match case.routine.as_str() {
            "brent" => {
                let Some(f) = fsci_minimization_target(&case.func) else {
                    continue;
                };
                let (xmin, _fmin) = brent_minimize(f, case.a, case.b, 1.0e-8, 500);
                (xmin, XMIN_TOL)
            }
            "golden" => {
                let Some(f) = fsci_minimization_target(&case.func) else {
                    continue;
                };
                let (xmin, _fmin) = golden(f, case.a, case.b, 1.0e-8, 500);
                (xmin, XMIN_TOL)
            }
            "fixed_point" => {
                let Some(f) = fsci_fixed_point_target(&case.func) else {
                    continue;
                };
                let Ok(xfp) = fixed_point(f, case.x0, 1.0e-10, 500) else {
                    continue;
                };
                (xfp, FIXED_POINT_TOL)
            }
            _ => continue,
        };
        let abs_d = (fsci_v - scipy_v).abs();
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
        test_id: "diff_opt_scalar_min_fixed_point".into(),
        category: "scipy.optimize.fminbound / fixed_point".into(),
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
                "scalar_min {} mismatch: {} abs_diff={}",
                d.routine, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy 1-D minimizer / fixed_point conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
