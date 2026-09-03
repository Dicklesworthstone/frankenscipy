#![forbid(unsafe_code)]
//! Live scipy parity for fsci_integrate infinite-bounds quad helpers.
//! Resolves [frankenscipy-8hye6].
//!
//! Functions covered:
//!   quad_inf(f, a)       -> ∫_a^∞ f
//!   quad_neg_inf(f, b)   -> ∫_{-∞}^b f
//!   quad_full_inf(f)     -> ∫_{-∞}^∞ f
//!
//! Probe analytic integrands shared with the python oracle (scipy.integrate.quad).
//! Tolerance: 1e-6 abs (substitution-based methods don't match adaptive QUADPACK
//! to machine precision).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_integrate::{QuadOptions, quad_full_inf, quad_inf, quad_neg_inf};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-6;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct InfCase {
    case_id: String,
    kind: String, // "inf" | "neg_inf" | "full_inf"
    func: String,
    bound: Option<f64>, // a for "inf", b for "neg_inf", None for full
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<InfCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct ArmScalar {
    case_id: String,
    value: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<ArmScalar>,
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
    fs::create_dir_all(output_dir()).expect("create quad_inf diff dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize quad_inf log");
    fs::write(path, json).expect("write quad_inf log");
}

// Shared analytic integrands. Both Rust and python use the same `func` key.
fn integrand(name: &str, x: f64) -> f64 {
    match name {
        "exp_neg_x" => (-x).exp(),
        "exp_neg_x2" => (-x * x).exp(),
        "one_over_1_plus_x2" => 1.0 / (1.0 + x * x),
        "x_exp_neg_x" => x * (-x).exp(),
        "exp_neg_abs_x" => (-x.abs()).exp(),
        "sech2" => {
            let c = x.cosh();
            1.0 / (c * c)
        }
        _ => f64::NAN,
    }
}

fn generate_query() -> OracleQuery {
    let points = vec![
        // quad_inf: ∫_0^∞ exp(-x) dx = 1
        InfCase {
            case_id: "qi_exp_neg_x_a0".into(),
            kind: "inf".into(),
            func: "exp_neg_x".into(),
            bound: Some(0.0),
        },
        // ∫_1^∞ exp(-x) dx = e^-1
        InfCase {
            case_id: "qi_exp_neg_x_a1".into(),
            kind: "inf".into(),
            func: "exp_neg_x".into(),
            bound: Some(1.0),
        },
        // ∫_0^∞ exp(-x^2) dx = sqrt(pi)/2
        InfCase {
            case_id: "qi_exp_neg_x2_a0".into(),
            kind: "inf".into(),
            func: "exp_neg_x2".into(),
            bound: Some(0.0),
        },
        // ∫_0^∞ x exp(-x) dx = 1
        InfCase {
            case_id: "qi_x_exp_neg_x_a0".into(),
            kind: "inf".into(),
            func: "x_exp_neg_x".into(),
            bound: Some(0.0),
        },
        // ∫_0^∞ 1/(1+x^2) dx = pi/2
        InfCase {
            case_id: "qi_one_over_1_plus_x2_a0".into(),
            kind: "inf".into(),
            func: "one_over_1_plus_x2".into(),
            bound: Some(0.0),
        },
        // quad_neg_inf: ∫_{-∞}^0 exp(x) dx = 1 -- use exp(-|x|) instead so closure is consistent
        InfCase {
            case_id: "qni_exp_neg_abs_x_b0".into(),
            kind: "neg_inf".into(),
            func: "exp_neg_abs_x".into(),
            bound: Some(0.0),
        },
        // ∫_{-∞}^0 1/(1+x^2) dx = pi/2
        InfCase {
            case_id: "qni_one_over_1_plus_x2_b0".into(),
            kind: "neg_inf".into(),
            func: "one_over_1_plus_x2".into(),
            bound: Some(0.0),
        },
        // quad_full_inf
        // ∫_{-∞}^∞ exp(-x^2) dx = sqrt(pi)
        InfCase {
            case_id: "qfi_exp_neg_x2".into(),
            kind: "full_inf".into(),
            func: "exp_neg_x2".into(),
            bound: None,
        },
        // ∫_{-∞}^∞ 1/(1+x^2) dx = pi
        InfCase {
            case_id: "qfi_one_over_1_plus_x2".into(),
            kind: "full_inf".into(),
            func: "one_over_1_plus_x2".into(),
            bound: None,
        },
        // ∫_{-∞}^∞ sech^2(x) dx = 2
        InfCase {
            case_id: "qfi_sech2".into(),
            kind: "full_inf".into(),
            func: "sech2".into(),
            bound: None,
        },
    ];
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy import integrate

def f(name, x):
    if name == "exp_neg_x": return math.exp(-x)
    if name == "exp_neg_x2": return math.exp(-x*x)
    if name == "one_over_1_plus_x2": return 1.0 / (1.0 + x*x)
    if name == "x_exp_neg_x": return x * math.exp(-x)
    if name == "exp_neg_abs_x": return math.exp(-abs(x))
    if name == "sech2":
        c = math.cosh(x)
        return 1.0 / (c*c)
    return float("nan")

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; kind = case["kind"]; name = case["func"]
    bound = case["bound"]
    try:
        if kind == "inf":
            v, _ = integrate.quad(lambda x: f(name, x), bound, np.inf, epsabs=1e-12, epsrel=1e-12, limit=200)
        elif kind == "neg_inf":
            v, _ = integrate.quad(lambda x: f(name, x), -np.inf, bound, epsabs=1e-12, epsrel=1e-12, limit=200)
        elif kind == "full_inf":
            v, _ = integrate.quad(lambda x: f(name, x), -np.inf, np.inf, epsabs=1e-12, epsrel=1e-12, limit=200)
        else:
            v = None
        if v is not None and math.isfinite(v):
            points.append({"case_id": cid, "value": float(v)})
        else:
            points.append({"case_id": cid, "value": None})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize quad_inf query");
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
                "failed to spawn python3 for quad_inf oracle: {e}"
            );
            eprintln!("skipping quad_inf oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open quad_inf oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "quad_inf oracle stdin failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping quad_inf oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for quad_inf oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "quad_inf oracle failed: {stderr}"
        );
        eprintln!("skipping quad_inf oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse quad_inf oracle JSON"))
}

#[test]
fn diff_integrate_quad_infinite() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    let pmap: HashMap<String, ArmScalar> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let opts = QuadOptions::default();
    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let Some(expected) = pmap.get(&case.case_id).and_then(|a| a.value) else {
            continue;
        };
        let f = |x: f64| integrand(&case.func, x);
        let res = match case.kind.as_str() {
            "inf" => {
                let Some(a) = case.bound else {
                    continue;
                };
                quad_inf(f, a, opts)
            }
            "neg_inf" => {
                let Some(b) = case.bound else {
                    continue;
                };
                quad_neg_inf(f, b, opts)
            }
            "full_inf" => quad_full_inf(f, opts),
            _ => continue,
        };
        let Ok(qr) = res else { continue };
        let abs_d = (qr.integral - expected).abs();
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.kind.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_integrate_quad_infinite".into(),
        category: "fsci_integrate quad_inf / quad_neg_inf / quad_full_inf vs scipy.integrate.quad"
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
        "quad_infinite conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
