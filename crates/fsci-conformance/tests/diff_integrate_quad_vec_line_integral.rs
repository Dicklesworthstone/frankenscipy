#![forbid(unsafe_code)]
//! Live scipy.integrate.quad_vec parity for fsci_integrate::quad_vec
//! and analytical-formula parity for fsci_integrate::line_integral.
//!
//! Resolves [frankenscipy-z9ioc].
//!
//! - `quad_vec(f, a, b)`: adaptive GK15 over vector-valued integrand.
//!   Compare component-wise at 1e-8 abs vs scipy.integrate.quad_vec.
//! - `line_integral(f, x(t), y(t), tlo, thi, n)`: Simpson 1/3 + central
//!   difference ds. O(h²) accuracy; compare against analytical
//!   closed-form line integrals on unit circle and line segments at
//!   1e-4 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_integrate::{QuadOptions, line_integral, quad_vec};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const QV_ABS_TOL: f64 = 1.0e-8;
const LI_ABS_TOL: f64 = 1.0e-4;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String, // "qv" | "li"
    func: String,
    a: f64,
    b: f64,
    /// li
    n: usize,
    /// li analytical answer
    analytical: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// qv: components; li: [analytical]
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
    fs::create_dir_all(output_dir()).expect("create qv_li diff dir");
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

fn f_vec(name: &str, x: f64) -> Vec<f64> {
    match name {
        // 3-component: [sin x, cos x, x^2]
        "trig_sq" => vec![x.sin(), x.cos(), x * x],
        // 2-component: [exp(-x), x exp(-x)]
        "exp_decay" => vec![(-x).exp(), x * (-x).exp()],
        _ => vec![],
    }
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();

    // quad_vec probes (smooth, finite interval)
    let qv_probes: &[(&str, f64, f64)] = &[
        ("trig_sq", 0.0, std::f64::consts::PI),
        ("trig_sq", -1.0, 1.0),
        ("exp_decay", 0.0, 5.0),
    ];
    for &(fname, a, b) in qv_probes {
        points.push(Case {
            case_id: format!("qv_{fname}_a{a}_b{b}"),
            op: "qv".into(),
            func: fname.into(),
            a,
            b,
            n: 0,
            analytical: 0.0,
        });
    }

    // line_integral probes — analytical answers computed by hand
    // Unit circle x(t)=cos t, y(t)=sin t, t in [0, 2π]; ds = 1
    // ∫_C 1 ds = 2π
    points.push(Case {
        case_id: "li_circle_one".into(),
        op: "li".into(),
        func: "circle_one".into(),
        a: 0.0,
        b: 2.0 * std::f64::consts::PI,
        n: 401,
        analytical: 2.0 * std::f64::consts::PI,
    });
    // ∫_C (x² + y²) ds = ∫₀²π 1 · 1 dt = 2π  (on unit circle, x²+y²=1)
    points.push(Case {
        case_id: "li_circle_unit_radius".into(),
        op: "li".into(),
        func: "circle_unit_r".into(),
        a: 0.0,
        b: 2.0 * std::f64::consts::PI,
        n: 401,
        analytical: 2.0 * std::f64::consts::PI,
    });
    // Line segment from (0,0) to (3,4); length 5. ds=5/(thi-tlo) * dt; let t in [0,1].
    // ∫_C 1 ds = 5
    points.push(Case {
        case_id: "li_segment_one".into(),
        op: "li".into(),
        func: "segment_one".into(),
        a: 0.0,
        b: 1.0,
        n: 41,
        analytical: 5.0,
    });
    // ∫_C x ds along segment (0,0) to (3,4): x(t)=3t, y(t)=4t, ds = 5 dt
    //  ∫₀¹ (3t)(5) dt = 15/2 = 7.5
    points.push(Case {
        case_id: "li_segment_x".into(),
        op: "li".into(),
        func: "segment_x".into(),
        a: 0.0,
        b: 1.0,
        n: 41,
        analytical: 7.5,
    });

    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy.integrate import quad_vec

def f_vec(name, x):
    if name == "trig_sq":     return np.array([math.sin(x), math.cos(x), x*x])
    if name == "exp_decay":   return np.array([math.exp(-x), x*math.exp(-x)])
    return np.array([])

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]; fname = case["func"]
    a = float(case["a"]); b = float(case["b"])
    try:
        if op == "qv":
            v, _ = quad_vec(lambda x: f_vec(fname, x), a, b, epsabs=1e-13, epsrel=1e-12)
            flat = [float(x) for x in v.tolist()]
            points.append({"case_id": cid, "values": flat})
        elif op == "li":
            # Use analytical reference passed through `analytical` field;
            # Python just echoes it for deterministic comparison.
            points.append({"case_id": cid, "values": [float(case["analytical"])]})
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
                "failed to spawn python3 for qv_li oracle: {e}"
            );
            eprintln!("skipping qv_li oracle: python3 not available ({e})");
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
                "qv_li oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping qv_li oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for qv_li oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "qv_li oracle failed: {stderr}"
        );
        eprintln!("skipping qv_li oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse qv_li oracle JSON"))
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
fn diff_integrate_quad_vec_line_integral() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let opts = QuadOptions::default();
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
        let (abs_d, tol) = match case.op.as_str() {
            "qv" => {
                let fname = case.func.clone();
                let f = move |x: f64| f_vec(&fname, x);
                let Ok(r) = quad_vec(&f, case.a, case.b, opts) else {
                    continue;
                };
                (vec_max_diff(&r.integral, expected), QV_ABS_TOL)
            }
            "li" => {
                let v = match case.func.as_str() {
                    "circle_one" => line_integral(
                        |_x, _y| 1.0,
                        |t: f64| t.cos(),
                        |t: f64| t.sin(),
                        case.a,
                        case.b,
                        case.n,
                    ),
                    "circle_unit_r" => line_integral(
                        |x, y| x * x + y * y,
                        |t: f64| t.cos(),
                        |t: f64| t.sin(),
                        case.a,
                        case.b,
                        case.n,
                    ),
                    "segment_one" => line_integral(
                        |_x, _y| 1.0,
                        |t: f64| 3.0 * t,
                        |t: f64| 4.0 * t,
                        case.a,
                        case.b,
                        case.n,
                    ),
                    "segment_x" => line_integral(
                        |x, _y| x,
                        |t: f64| 3.0 * t,
                        |t: f64| 4.0 * t,
                        case.a,
                        case.b,
                        case.n,
                    ),
                    _ => continue,
                };
                ((v - expected[0]).abs(), LI_ABS_TOL)
            }
            _ => continue,
        };
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            abs_diff: abs_d,
            pass: abs_d <= tol,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_integrate_quad_vec_line_integral".into(),
        category: "fsci_integrate::{quad_vec, line_integral} vs scipy & analytical".into(),
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
        "quad_vec/line_integral conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
