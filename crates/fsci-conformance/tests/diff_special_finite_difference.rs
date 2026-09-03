#![forbid(unsafe_code)]
//! Live numpy parity for fsci_special finite-difference helpers:
//! central_diff, central_diff2, gradient_approx, jacobian_approx,
//! hessian_approx.
//!
//! Resolves [frankenscipy-dkbum].
//! Compares against analytic derivatives computed in numpy.
//! Tolerance: 1e-6 abs (central differences are O(h^2) at h=1e-5).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::{central_diff, central_diff2, gradient_approx, hessian_approx, jacobian_approx};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-6;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
const H_STEP: f64 = 1.0e-5;

// Function identifiers shared between Rust closures and the python oracle.
// Each `func` string keys a fixed analytic form in both layers.

#[derive(Debug, Clone, Serialize)]
struct ScalarCase {
    case_id: String,
    func: String,
    x: f64,
    h: f64,
}

#[derive(Debug, Clone, Serialize)]
struct GradHessCase {
    case_id: String,
    func: String,
    x: Vec<f64>,
    h: f64,
}

#[derive(Debug, Clone, Serialize)]
struct JacobianCase {
    case_id: String,
    func: String,
    x: Vec<f64>,
    h: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    central1: Vec<ScalarCase>,
    central2: Vec<ScalarCase>,
    gradient: Vec<GradHessCase>,
    jacobian: Vec<JacobianCase>,
    hessian: Vec<GradHessCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct ScalarArm {
    case_id: String,
    value: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct VecArm {
    case_id: String,
    values: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct MatArm {
    case_id: String,
    /// Row-major flat.
    values: Option<Vec<f64>>,
    rows: Option<usize>,
    cols: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    central1: Vec<ScalarArm>,
    central2: Vec<ScalarArm>,
    gradient: Vec<VecArm>,
    jacobian: Vec<MatArm>,
    hessian: Vec<MatArm>,
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
    fs::create_dir_all(output_dir()).expect("create finite_diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize finite_diff log");
    fs::write(path, json).expect("write finite_diff log");
}

// ---- Rust-side analytic-closure dispatch ----

fn scalar_f(func: &str, x: f64) -> f64 {
    match func {
        "sin" => x.sin(),
        "cos" => x.cos(),
        "x3" => x * x * x,
        "exp_neg_x2" => (-x * x).exp(),
        "log1p_x2" => (1.0 + x * x).ln(),
        _ => f64::NAN,
    }
}

fn multi_f(func: &str, x: &[f64]) -> f64 {
    match func {
        // f(x,y) = x^2 + y^2
        "sumsq" => x[0] * x[0] + x[1] * x[1],
        // f(x,y) = x*sin(y) + y*cos(x)
        "trig_mix" => x[0] * x[1].sin() + x[1] * x[0].cos(),
        // f(x,y,z) = exp(x) - y*z + z^2
        "mixed3" => x[0].exp() - x[1] * x[2] + x[2] * x[2],
        _ => f64::NAN,
    }
}

fn vec_f(func: &str, x: &[f64]) -> Vec<f64> {
    match func {
        // F(x,y) = [x^2 + y, x*y]
        "vec2" => vec![x[0] * x[0] + x[1], x[0] * x[1]],
        // F(x,y,z) = [sin(x)+y, y*z, x+z*z]
        "vec3" => vec![x[0].sin() + x[1], x[1] * x[2], x[0] + x[2] * x[2]],
        _ => vec![],
    }
}

fn generate_query() -> OracleQuery {
    let central1 = vec![
        ScalarCase {
            case_id: "sin_at_0".into(),
            func: "sin".into(),
            x: 0.0,
            h: H_STEP,
        },
        ScalarCase {
            case_id: "sin_at_pi4".into(),
            func: "sin".into(),
            x: std::f64::consts::FRAC_PI_4,
            h: H_STEP,
        },
        ScalarCase {
            case_id: "cos_at_1".into(),
            func: "cos".into(),
            x: 1.0,
            h: H_STEP,
        },
        ScalarCase {
            case_id: "x3_at_2".into(),
            func: "x3".into(),
            x: 2.0,
            h: H_STEP,
        },
        ScalarCase {
            case_id: "exp_neg_x2_at_0p5".into(),
            func: "exp_neg_x2".into(),
            x: 0.5,
            h: H_STEP,
        },
        ScalarCase {
            case_id: "log1p_x2_at_1".into(),
            func: "log1p_x2".into(),
            x: 1.0,
            h: H_STEP,
        },
    ];

    let central2 = vec![
        ScalarCase {
            case_id: "d2_sin_at_pi4".into(),
            func: "sin".into(),
            x: std::f64::consts::FRAC_PI_4,
            h: 1.0e-4, // larger h for second derivative (error O(h^2) but cancellation worse)
        },
        ScalarCase {
            case_id: "d2_x3_at_2".into(),
            func: "x3".into(),
            x: 2.0,
            h: 1.0e-4,
        },
        ScalarCase {
            case_id: "d2_log1p_x2_at_1".into(),
            func: "log1p_x2".into(),
            x: 1.0,
            h: 1.0e-4,
        },
    ];

    let gradient = vec![
        GradHessCase {
            case_id: "grad_sumsq_at_1_2".into(),
            func: "sumsq".into(),
            x: vec![1.0, 2.0],
            h: H_STEP,
        },
        GradHessCase {
            case_id: "grad_trig_mix_at_0p5_1".into(),
            func: "trig_mix".into(),
            x: vec![0.5, 1.0],
            h: H_STEP,
        },
        GradHessCase {
            case_id: "grad_mixed3_at_0p1_2_minus1".into(),
            func: "mixed3".into(),
            x: vec![0.1, 2.0, -1.0],
            h: H_STEP,
        },
    ];

    let jacobian = vec![
        JacobianCase {
            case_id: "jac_vec2_at_1_2".into(),
            func: "vec2".into(),
            x: vec![1.0, 2.0],
            h: H_STEP,
        },
        JacobianCase {
            case_id: "jac_vec3_at_0p2_1_0p5".into(),
            func: "vec3".into(),
            x: vec![0.2, 1.0, 0.5],
            h: H_STEP,
        },
    ];

    let hessian = vec![
        GradHessCase {
            case_id: "hess_sumsq_at_1_2".into(),
            func: "sumsq".into(),
            x: vec![1.0, 2.0],
            h: 1.0e-4,
        },
        GradHessCase {
            case_id: "hess_trig_mix_at_0p5_1".into(),
            func: "trig_mix".into(),
            x: vec![0.5, 1.0],
            h: 1.0e-4,
        },
    ];

    OracleQuery {
        central1,
        central2,
        gradient,
        jacobian,
        hessian,
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np

def scalar_f(func, x):
    if func == "sin": return math.sin(x)
    if func == "cos": return math.cos(x)
    if func == "x3":  return x*x*x
    if func == "exp_neg_x2": return math.exp(-x*x)
    if func == "log1p_x2": return math.log1p(x*x)
    return float("nan")

def scalar_df(func, x):
    if func == "sin": return math.cos(x)
    if func == "cos": return -math.sin(x)
    if func == "x3":  return 3*x*x
    if func == "exp_neg_x2": return -2*x*math.exp(-x*x)
    if func == "log1p_x2": return (2*x) / (1.0 + x*x)
    return float("nan")

def scalar_d2f(func, x):
    if func == "sin": return -math.sin(x)
    if func == "cos": return -math.cos(x)
    if func == "x3":  return 6*x
    if func == "exp_neg_x2":
        return (4*x*x - 2) * math.exp(-x*x)
    if func == "log1p_x2":
        denom = 1.0 + x*x
        return (2*denom - 2*x*(2*x)) / (denom*denom)
    return float("nan")

def grad(func, x):
    if func == "sumsq":   # x[0]^2 + x[1]^2
        return [2*x[0], 2*x[1]]
    if func == "trig_mix": # x*sin(y) + y*cos(x)
        return [math.sin(x[1]) - x[1]*math.sin(x[0]), x[0]*math.cos(x[1]) + math.cos(x[0])]
    if func == "mixed3":   # exp(x) - y*z + z^2
        return [math.exp(x[0]), -x[2], -x[1] + 2*x[2]]
    return []

def vec_jac(func, x):
    # Returns row-major flat + (rows, cols)
    if func == "vec2":  # F(x,y) = [x^2+y, x*y]
        J = [[2*x[0], 1.0],
             [x[1],   x[0]]]
        return J
    if func == "vec3":  # F(x,y,z) = [sin(x)+y, y*z, x+z^2]
        J = [[math.cos(x[0]), 1.0,    0.0],
             [0.0,            x[2],   x[1]],
             [1.0,            0.0,    2*x[2]]]
        return J
    return None

def hess(func, x):
    if func == "sumsq":
        return [[2.0, 0.0], [0.0, 2.0]]
    if func == "trig_mix":
        # f = x*sin(y) + y*cos(x)
        # fxx = -y*cos(x); fxy = cos(y) - sin(x); fyy = -x*sin(y)
        a, b = x[0], x[1]
        return [
            [-b*math.cos(a), math.cos(b) - math.sin(a)],
            [math.cos(b) - math.sin(a), -a*math.sin(b)],
        ]
    return None

def finite_or_none_scalar(v):
    return float(v) if (v is not None and math.isfinite(v)) else None

def finite_or_none_vec(arr):
    if arr is None: return None
    out = []
    for v in arr:
        if not math.isfinite(float(v)):
            return None
        out.append(float(v))
    return out

q = json.load(sys.stdin)

c1 = []
for c in q["central1"]:
    v = scalar_df(c["func"], float(c["x"]))
    c1.append({"case_id": c["case_id"], "value": finite_or_none_scalar(v)})

c2 = []
for c in q["central2"]:
    v = scalar_d2f(c["func"], float(c["x"]))
    c2.append({"case_id": c["case_id"], "value": finite_or_none_scalar(v)})

g = []
for c in q["gradient"]:
    v = grad(c["func"], [float(t) for t in c["x"]])
    g.append({"case_id": c["case_id"], "values": finite_or_none_vec(v)})

j = []
for c in q["jacobian"]:
    M = vec_jac(c["func"], [float(t) for t in c["x"]])
    if M is None:
        j.append({"case_id": c["case_id"], "values": None, "rows": None, "cols": None})
    else:
        rows = len(M); cols = len(M[0]) if rows else 0
        flat = [v for row in M for v in row]
        j.append({"case_id": c["case_id"], "values": finite_or_none_vec(flat), "rows": rows, "cols": cols})

h = []
for c in q["hessian"]:
    H = hess(c["func"], [float(t) for t in c["x"]])
    if H is None:
        h.append({"case_id": c["case_id"], "values": None, "rows": None, "cols": None})
    else:
        rows = len(H); cols = len(H[0]) if rows else 0
        flat = [v for row in H for v in row]
        h.append({"case_id": c["case_id"], "values": finite_or_none_vec(flat), "rows": rows, "cols": cols})

print(json.dumps({
    "central1": c1, "central2": c2, "gradient": g, "jacobian": j, "hessian": h
}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize finite_diff query");
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
                "failed to spawn python3 for finite_diff oracle: {e}"
            );
            eprintln!("skipping finite_diff oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open finite_diff oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "finite_diff oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping finite_diff oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for finite_diff oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "finite_diff oracle failed: {stderr}"
        );
        eprintln!("skipping finite_diff oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse finite_diff oracle JSON"))
}

#[test]
fn diff_special_finite_difference() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    let c1_map: HashMap<String, ScalarArm> = oracle
        .central1
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();
    let c2_map: HashMap<String, ScalarArm> = oracle
        .central2
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();
    let g_map: HashMap<String, VecArm> = oracle
        .gradient
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();
    let j_map: HashMap<String, MatArm> = oracle
        .jacobian
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();
    let h_map: HashMap<String, MatArm> = oracle
        .hessian
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    // central_diff
    for case in &query.central1 {
        let Some(expected) = c1_map.get(&case.case_id).and_then(|a| a.value) else {
            continue;
        };
        let f = |x: f64| scalar_f(&case.func, x);
        let actual = central_diff(f, case.x, case.h);
        let abs_d = (actual - expected).abs();
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: "central_diff".into(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    // central_diff2 (relaxed tol for second derivative, h^2 cancellation)
    let d2_tol = 1.0e-4;
    for case in &query.central2 {
        let Some(expected) = c2_map.get(&case.case_id).and_then(|a| a.value) else {
            continue;
        };
        let f = |x: f64| scalar_f(&case.func, x);
        let actual = central_diff2(f, case.x, case.h);
        let abs_d = (actual - expected).abs();
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: "central_diff2".into(),
            abs_diff: abs_d,
            pass: abs_d <= d2_tol,
        });
    }

    // gradient_approx
    for case in &query.gradient {
        let Some(expected) = g_map.get(&case.case_id).and_then(|a| a.values.clone()) else {
            continue;
        };
        let f = |x: &[f64]| multi_f(&case.func, x);
        let actual = gradient_approx(f, &case.x, case.h);
        let abs_d = if actual.len() != expected.len() {
            f64::INFINITY
        } else {
            actual
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        };
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: "gradient_approx".into(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    // jacobian_approx
    for case in &query.jacobian {
        let arm = j_map.get(&case.case_id).expect("validated jacobian arm");
        let (Some(expected), Some(rows), Some(cols)) = (arm.values.clone(), arm.rows, arm.cols)
        else {
            continue;
        };
        let f = |x: &[f64]| vec_f(&case.func, x);
        let actual = jacobian_approx(f, &case.x, case.h);
        // fsci returns Vec<Vec<f64>>; flatten row-major (n rows = len(x) or m?).
        // jacobian_approx returns a vector of m rows × n cols where m = f(x).len().
        let flat: Vec<f64> = actual.iter().flat_map(|r| r.iter().copied()).collect();
        let abs_d = if actual.len() != rows || actual.first().map(|r| r.len()) != Some(cols) {
            f64::INFINITY
        } else {
            flat.iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        };
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: "jacobian_approx".into(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    // hessian_approx (relaxed tol due to compound differences)
    let h_tol = 1.0e-3;
    for case in &query.hessian {
        let arm = h_map.get(&case.case_id).expect("validated hessian arm");
        let (Some(expected), Some(rows), Some(cols)) = (arm.values.clone(), arm.rows, arm.cols)
        else {
            continue;
        };
        let f = |x: &[f64]| multi_f(&case.func, x);
        let actual = hessian_approx(f, &case.x, case.h);
        let flat: Vec<f64> = actual.iter().flat_map(|r| r.iter().copied()).collect();
        let abs_d = if actual.len() != rows || actual.first().map(|r| r.len()) != Some(cols) {
            f64::INFINITY
        } else {
            flat.iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        };
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: "hessian_approx".into(),
            abs_diff: abs_d,
            pass: abs_d <= h_tol,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_finite_difference".into(),
        category: "fsci_special FD helpers vs analytic derivatives".into(),
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
        "finite_difference conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
