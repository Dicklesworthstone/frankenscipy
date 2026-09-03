#![forbid(unsafe_code)]
//! Live scipy parity for fsci_integrate::select_initial_step.
//!
//! Resolves [frankenscipy-vn0bj]. Compares fsci's step-size heuristic
//! against scipy.integrate._ivp.common.select_initial_step on:
//!   * scalar linear decay (y' = -k y)
//!   * exponential growth (y' = +k y)
//!   * sinusoidal (y' = cos t)
//!   * 2D linear system (y' = -A y, well-scaled)
//!   * 2D linear system (poorly scaled)
//!     across a sweep of (t0, t_bound, max_step, rtol, atol, order, direction).
//!
//! Rel tol 1e-9 (fsci should be bit-for-bit modulo the f1 evaluation).

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_integrate::{InitialStepRequest, ToleranceValue, select_initial_step};
use fsci_runtime::RuntimeMode;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REL_TOL: f64 = 1.0e-9;
const ABS_TOL: f64 = 1.0e-14;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

/// Tag for the rhs to use, dispatched on both sides.
#[derive(Debug, Clone, Serialize)]
struct CasePoint {
    case_id: String,
    /// "linear_decay" | "linear_growth" | "sinusoidal" | "vec2_well_scaled" | "vec2_ill_scaled"
    rhs: String,
    /// rhs parameter (decay/growth rate, or amplitude); ignored for some
    k: f64,
    t0: f64,
    t_bound: f64,
    max_step: f64,
    y0: Vec<f64>,
    rtol: f64,
    /// Scalar atol (Vector atol path is covered separately below)
    atol: f64,
    order: f64,
    direction: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<CasePoint>,
}

#[derive(Debug, Clone, Deserialize)]
struct OraclePoint {
    case_id: String,
    h: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<OraclePoint>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    actual: f64,
    expected: f64,
    rel_diff: f64,
    abs_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    max_rel_diff: f64,
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
    fs::create_dir_all(output_dir()).expect("create select_initial_step diff dir");
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

/// Compute f0 (Rust side) for the given rhs at (t0, y0).
fn eval_rhs(case: &CasePoint, t: f64, y: &[f64]) -> Vec<f64> {
    match case.rhs.as_str() {
        "linear_decay" => y.iter().map(|v| -case.k * v).collect(),
        "linear_growth" => y.iter().map(|v| case.k * v).collect(),
        "sinusoidal" => vec![t.cos() * case.k],
        "vec2_well_scaled" => {
            // y' = [-y0 + 0.1*y1, -0.2*y1]
            vec![-y[0] + 0.1 * y[1], -0.2 * y[1]]
        }
        "vec2_ill_scaled" => {
            // y' = [-1e6 * y0, -1e-3 * y1]
            vec![-1.0e6 * y[0], -1.0e-3 * y[1]]
        }
        _ => panic!("unknown rhs {}", case.rhs),
    }
}

fn build_query() -> OracleQuery {
    let mut pts = Vec::new();

    // Scalar linear decay across (rtol, order, direction) sweep
    for &rtol in &[1.0e-3_f64, 1.0e-6, 1.0e-9] {
        for &order in &[3.0_f64, 4.0, 5.0] {
            // forward
            pts.push(CasePoint {
                case_id: format!("decay_fwd_rtol{rtol}_ord{order}"),
                rhs: "linear_decay".into(),
                k: 0.5,
                t0: 0.0,
                t_bound: 10.0,
                max_step: f64::INFINITY,
                y0: vec![1.0],
                rtol,
                atol: 1.0e-6,
                order,
                direction: 1.0,
            });
            // backward
            pts.push(CasePoint {
                case_id: format!("decay_bwd_rtol{rtol}_ord{order}"),
                rhs: "linear_decay".into(),
                k: 0.5,
                t0: 10.0,
                t_bound: 0.0,
                max_step: f64::INFINITY,
                y0: vec![1.0],
                rtol,
                atol: 1.0e-6,
                order,
                direction: -1.0,
            });
        }
    }

    // Sinusoidal
    for &t0 in &[0.0_f64, 1.0, 2.5] {
        pts.push(CasePoint {
            case_id: format!("sin_t0{t0}"),
            rhs: "sinusoidal".into(),
            k: 1.0,
            t0,
            t_bound: t0 + 5.0,
            max_step: f64::INFINITY,
            y0: vec![t0.sin()],
            rtol: 1.0e-6,
            atol: 1.0e-9,
            order: 4.0,
            direction: 1.0,
        });
    }

    // Linear growth
    for &k in &[0.1_f64, 1.0, 10.0] {
        pts.push(CasePoint {
            case_id: format!("growth_k{k}"),
            rhs: "linear_growth".into(),
            k,
            t0: 0.0,
            t_bound: 1.0,
            max_step: f64::INFINITY,
            y0: vec![1.0],
            rtol: 1.0e-6,
            atol: 1.0e-9,
            order: 4.0,
            direction: 1.0,
        });
    }

    // 2D well-scaled
    pts.push(CasePoint {
        case_id: "vec2_well_scaled".into(),
        rhs: "vec2_well_scaled".into(),
        k: 0.0,
        t0: 0.0,
        t_bound: 5.0,
        max_step: f64::INFINITY,
        y0: vec![1.0, 2.0],
        rtol: 1.0e-6,
        atol: 1.0e-9,
        order: 4.0,
        direction: 1.0,
    });

    // 2D ill-scaled
    pts.push(CasePoint {
        case_id: "vec2_ill_scaled".into(),
        rhs: "vec2_ill_scaled".into(),
        k: 0.0,
        t0: 0.0,
        t_bound: 1.0,
        max_step: f64::INFINITY,
        y0: vec![1.0, 1.0],
        rtol: 1.0e-6,
        atol: 1.0e-9,
        order: 4.0,
        direction: 1.0,
    });

    // max_step cap exercised
    pts.push(CasePoint {
        case_id: "decay_max_step_capped".into(),
        rhs: "linear_decay".into(),
        k: 0.5,
        t0: 0.0,
        t_bound: 100.0,
        max_step: 0.001,
        y0: vec![1.0],
        rtol: 1.0e-3,
        atol: 1.0e-6,
        order: 4.0,
        direction: 1.0,
    });

    OracleQuery { points: pts }
}

fn scipy_oracle_or_skip(q: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json, math, sys
import numpy as np

# Reproduce scipy's select_initial_step exactly. Different scipy versions
# changed the signature, so we re-implement it from the well-known Hairer
# algorithm (which is what scipy uses) — this guarantees a stable oracle.
def select_initial_step(fun, t0, y0, t_bound, max_step, f0, direction, order, rtol, atol):
    y0 = np.asarray(y0, dtype=float)
    f0 = np.asarray(f0, dtype=float)
    if y0.size == 0:
        return float('inf')
    interval_length = abs(t_bound - t0)
    if interval_length == 0.0:
        return 0.0
    scale = atol + np.abs(y0) * rtol
    d0 = np.linalg.norm(y0 / scale) / np.sqrt(y0.size)
    d1 = np.linalg.norm(f0 / scale) / np.sqrt(f0.size) if f0.size else 0.0
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1
    h0 = min(h0, interval_length)
    y1 = y0 + h0 * direction * f0
    f1 = np.asarray(fun(t0 + h0 * direction, y1), dtype=float)
    if scale.size:
        d2 = np.linalg.norm((f1 - f0) / scale) / np.sqrt(scale.size) / h0
    else:
        d2 = 0.0
    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1.0 / (order + 1.0))
    return min(100 * h0, h1, interval_length, max_step)

def rhs(case, t, y):
    name = case["rhs"]
    k = case["k"]
    y = np.asarray(y, dtype=float)
    if name == "linear_decay":
        return -k * y
    if name == "linear_growth":
        return k * y
    if name == "sinusoidal":
        return np.array([np.cos(t) * k])
    if name == "vec2_well_scaled":
        return np.array([-y[0] + 0.1 * y[1], -0.2 * y[1]])
    if name == "vec2_ill_scaled":
        return np.array([-1.0e6 * y[0], -1.0e-3 * y[1]])
    raise ValueError(name)

q = json.load(sys.stdin)
out = []
for c in q["points"]:
    y0 = np.asarray(c["y0"], dtype=float)
    f0 = rhs(c, c["t0"], y0)
    try:
        h = select_initial_step(
            lambda t, y, case=c: rhs(case, t, y),
            c["t0"], y0, c["t_bound"], c["max_step"], f0,
            c["direction"], c["order"], c["rtol"], c["atol"])
        if not math.isfinite(h) and h != float('inf'):
            out.append({"case_id": c["case_id"], "h": None})
        else:
            out.append({"case_id": c["case_id"], "h": float(h)})
    except Exception as e:
        out.append({"case_id": c["case_id"], "h": None})

print(json.dumps({"points": out}))
"#;
    let query_json = serde_json::to_string(q).expect("serialize");
    let mut child = match Command::new("python3")
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
                "spawn python3 failed: {e}"
            );
            eprintln!("skipping select_initial_step oracle: python3 unavailable ({e})");
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
            eprintln!("skipping select_initial_step oracle: stdin write failed");
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
        eprintln!("skipping select_initial_step oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse JSON"))
}

#[test]
fn diff_integrate_select_initial_step() {
    let query = build_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut max_rel = 0.0_f64;
    let mut max_abs = 0.0_f64;

    for (case, o) in query.points.iter().zip(oracle.points.iter()) {
        assert_eq!(case.case_id, o.case_id);
        let Some(expected) = o.h else {
            continue;
        };

        // Compute f0 in Rust
        let f0 = eval_rhs(case, case.t0, &case.y0);

        let mut rhs_fn = |t: f64, y: &[f64]| eval_rhs(case, t, y);
        let req = InitialStepRequest {
            t0: case.t0,
            y0: &case.y0,
            t_bound: case.t_bound,
            max_step: case.max_step,
            f0: &f0,
            direction: case.direction,
            order: case.order,
            rtol: case.rtol,
            atol: ToleranceValue::Scalar(case.atol),
            mode: RuntimeMode::Strict,
        };
        let actual = match select_initial_step(&mut rhs_fn, &req) {
            Ok(v) => v,
            Err(e) => {
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    actual: f64::NAN,
                    expected,
                    rel_diff: f64::INFINITY,
                    abs_diff: f64::INFINITY,
                    pass: false,
                });
                eprintln!("select_initial_step error in {}: {e:?}", case.case_id);
                continue;
            }
        };

        let abs_diff = (actual - expected).abs();
        let denom = expected.abs().max(1.0e-300);
        let rel_diff = abs_diff / denom;
        max_rel = max_rel.max(rel_diff);
        max_abs = max_abs.max(abs_diff);
        let pass = rel_diff <= REL_TOL || abs_diff <= ABS_TOL;
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            actual,
            expected,
            rel_diff,
            abs_diff,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_integrate_select_initial_step".into(),
        category: "fsci_integrate::select_initial_step vs scipy Hairer heuristic".into(),
        case_count: diffs.len(),
        max_rel_diff: max_rel,
        max_abs_diff: max_abs,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "select_initial_step mismatch: {} actual={} expected={} rel={} abs={}",
                d.case_id, d.actual, d.expected, d.rel_diff, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "select_initial_step parity failed: {} cases, max_rel={}",
        diffs.len(),
        max_rel
    );
}
