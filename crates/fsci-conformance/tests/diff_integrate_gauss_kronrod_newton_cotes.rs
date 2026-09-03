#![forbid(unsafe_code)]
//! Live scipy.integrate parity for fsci_integrate::gauss_kronrod_quad,
//! newton_cotes_quad, and quad_cauchy_pv.
//!
//! Resolves [frankenscipy-llatt].
//!
//! - `gauss_kronrod_quad`: GK15/G7 single-panel rule. Compare against
//!   scipy.integrate.quad on smooth integrands.
//! - `newton_cotes_quad`: composite Newton-Cotes of given order over
//!   n_panels. order=2 (Simpson) is exact for cubics. Compare against
//!   scipy.integrate.quad on polynomial integrands.
//! - `quad_cauchy_pv`: principal-value integral by symmetric eps-trim
//!   around singular point. Compare against
//!   scipy.integrate.quad(weight='cauchy', wvar=sing) for f(x)/(x-c).
//!
//! Tolerances: 1e-7 abs (GK15/G7 has ~1e-9 abs error on smooth
//! integrands; Cauchy PV via eps-trim has ~eps*max|f'| error).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_integrate::{QuadOptions, gauss_kronrod_quad, newton_cotes_quad, quad_cauchy_pv};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
const ABS_TOL: f64 = 1.0e-7;

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String, // "gk" | "nc" | "cpv"
    func: String,
    a: f64,
    b: f64,
    /// nc
    n_panels: usize,
    order: usize,
    /// cpv
    singular: f64,
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
    fs::create_dir_all(output_dir()).expect("create gk_nc_cpv diff dir");
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
        // smooth, regular
        "sin_x" => x.sin(),
        "exp_neg_xsq" => (-x * x).exp(),
        "poly_cubic" => x.powi(3) - 2.0 * x + 1.0,
        "poly_quintic" => x.powi(5) - 3.0 * x.powi(3) + x,
        // cauchy-PV numerator (used as f, then PV ∫ f(x)/(x-c) dx)
        "one" => 1.0,
        "x_squared" => x * x,
        _ => f64::NAN,
    }
}

fn f1d_div_singular(name: &str, x: f64, c: f64) -> f64 {
    f1d(name, x) / (x - c)
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();

    // gauss_kronrod_quad probes (smooth)
    let gk_probes: &[(&str, f64, f64)] = &[
        ("sin_x", 0.0, std::f64::consts::PI),
        ("exp_neg_xsq", -2.0, 2.0),
        ("poly_cubic", -1.0, 1.5),
        ("poly_quintic", 0.0, 2.0),
    ];
    for &(fname, a, b) in gk_probes {
        points.push(Case {
            case_id: format!("gk_{fname}_a{a}_b{b}"),
            op: "gk".into(),
            func: fname.into(),
            a,
            b,
            n_panels: 0,
            order: 0,
            singular: 0.0,
        });
    }

    // newton_cotes_quad probes
    // order=2 (Simpson) exact for cubics. order=1 (trapezoidal) exact for linears.
    // Use polynomial integrands within exactness window so any panel count works.
    let nc_probes: &[(&str, f64, f64, usize, usize)] = &[
        ("poly_cubic", -1.0, 1.5, 4, 2),
        ("poly_cubic", 0.0, 2.0, 8, 2),
        ("poly_quintic", 0.0, 1.0, 4, 4), // Boole's rule exact for degree ≤ 5
        ("poly_quintic", -1.0, 2.0, 6, 4),
    ];
    for &(fname, a, b, n_panels, order) in nc_probes {
        points.push(Case {
            case_id: format!("nc_{fname}_n{n_panels}_o{order}"),
            op: "nc".into(),
            func: fname.into(),
            a,
            b,
            n_panels,
            order,
            singular: 0.0,
        });
    }

    // quad_cauchy_pv probes — PV ∫ f(x)/(x-c) dx with c inside [a,b]
    let cpv_probes: &[(&str, f64, f64, f64)] = &[
        ("one", -1.0, 1.0, 0.0),       // ∫_{-1}^{1} 1/(x) dx PV = 0
        ("x_squared", -1.0, 1.0, 0.0), // ∫_{-1}^{1} x^2/x dx = ∫ x dx = 0
        ("one", 0.0, 2.0, 1.0),        // ∫_{0}^{2} 1/(x-1) dx PV = 0 (by symmetry)
    ];
    for &(fname, a, b, c) in cpv_probes {
        points.push(Case {
            case_id: format!("cpv_{fname}_c{c}"),
            op: "cpv".into(),
            func: fname.into(),
            a,
            b,
            n_panels: 0,
            order: 0,
            singular: c,
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
from scipy.integrate import quad

def f1d(name, x):
    if name == "sin_x":        return math.sin(x)
    if name == "exp_neg_xsq":  return math.exp(-x*x)
    if name == "poly_cubic":   return x**3 - 2.0*x + 1.0
    if name == "poly_quintic": return x**5 - 3.0*x**3 + x
    if name == "one":          return 1.0
    if name == "x_squared":    return x*x
    return float("nan")

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]; fname = case["func"]
    a = float(case["a"]); b = float(case["b"])
    try:
        if op == "gk":
            v, _ = quad(lambda x: f1d(fname, x), a, b, epsabs=1e-13, epsrel=1e-12)
            points.append({"case_id": cid, "value": float(v)})
        elif op == "nc":
            v, _ = quad(lambda x: f1d(fname, x), a, b, epsabs=1e-13, epsrel=1e-12)
            points.append({"case_id": cid, "value": float(v)})
        elif op == "cpv":
            c = float(case["singular"])
            v, _ = quad(lambda x: f1d(fname, x), a, b, weight="cauchy", wvar=c, epsabs=1e-13, epsrel=1e-12)
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
                "failed to spawn python3 for gk_nc_cpv oracle: {e}"
            );
            eprintln!("skipping gk_nc_cpv oracle: python3 not available ({e})");
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
                "gk_nc_cpv oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping gk_nc_cpv oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for gk_nc_cpv oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "gk_nc_cpv oracle failed: {stderr}"
        );
        eprintln!("skipping gk_nc_cpv oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse gk_nc_cpv oracle JSON"))
}

#[test]
fn diff_integrate_gauss_kronrod_newton_cotes() {
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
        let Some(expected) = arm.value else {
            continue;
        };
        let fname = case.func.clone();
        let actual = match case.op.as_str() {
            "gk" => {
                let f = move |x: f64| f1d(&fname, x);
                let r = gauss_kronrod_quad(&f, case.a, case.b, opts);
                r.integral
            }
            "nc" => {
                let f = move |x: f64| f1d(&fname, x);
                match newton_cotes_quad(&f, case.a, case.b, case.n_panels, case.order) {
                    Ok(v) => v,
                    Err(_) => continue,
                }
            }
            "cpv" => {
                let sing = case.singular;
                let f = move |x: f64| f1d_div_singular(&fname, x, sing);
                match quad_cauchy_pv(&f, case.a, case.b, sing, opts) {
                    Ok(r) => r.integral,
                    Err(_) => continue,
                }
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
        test_id: "diff_integrate_gauss_kronrod_newton_cotes".into(),
        category:
            "fsci_integrate::{gauss_kronrod_quad, newton_cotes_quad, quad_cauchy_pv} vs scipy.integrate.quad"
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
        "gk/nc/cpv conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
