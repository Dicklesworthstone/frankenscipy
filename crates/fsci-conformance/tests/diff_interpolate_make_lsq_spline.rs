#![forbid(unsafe_code)]
//! Live scipy parity for fsci_interpolate::make_lsq_spline.
//!
//! Resolves [frankenscipy-7iaj0]. Fits an least-squares B-spline
//! against scipy.interpolate.make_lsq_spline on linear (k=1), cubic
//! (k=3), and quintic (k=5) splines across a sine, quadratic, and
//! noisy quadratic dataset. Compares the spline evaluation at a
//! dense set of query points within rel tol 1e-6 (LS spline fit has
//! some implementation-dependent drift in coefficient assembly /
//! normal-equation solve, so a coefficient-level bit match is not
//! the right invariant; the evaluated curve is).

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_interpolate::make_lsq_spline;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REL_TOL: f64 = 1.0e-6;
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct CasePoint {
    case_id: String,
    x: Vec<f64>,
    y: Vec<f64>,
    t: Vec<f64>,
    k: usize,
    x_eval: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<CasePoint>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct OraclePoint {
    case_id: String,
    y_eval: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<OraclePoint>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    max_abs_diff: f64,
    max_rel_diff: f64,
    n_eval: usize,
    pass: bool,
    note: String,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create lsq_spline diff dir");
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

/// Build an open uniform knot vector with k+1-multiplicity at the ends
/// (this is the canonical form scipy.interpolate.make_lsq_spline expects).
fn uniform_knot_vector(x_min: f64, x_max: f64, n_interior: usize, k: usize) -> Vec<f64> {
    let mut t = Vec::with_capacity(2 * (k + 1) + n_interior);
    for _ in 0..=k {
        t.push(x_min);
    }
    if n_interior > 0 {
        let step = (x_max - x_min) / (n_interior + 1) as f64;
        for i in 1..=n_interior {
            t.push(x_min + step * i as f64);
        }
    }
    for _ in 0..=k {
        t.push(x_max);
    }
    t
}

fn build_query() -> OracleQuery {
    let mut pts = Vec::new();

    // Common evaluation grid (interior to avoid boundary artifacts).
    let make_eval = |a: f64, b: f64, n: usize| -> Vec<f64> {
        let pad = (b - a) * 0.02;
        let lo = a + pad;
        let hi = b - pad;
        (0..n)
            .map(|i| lo + (hi - lo) * i as f64 / (n - 1) as f64)
            .collect()
    };

    // Dataset 1: sine on [0, 2π], 40 samples
    {
        let n_samples = 40usize;
        let x: Vec<f64> = (0..n_samples)
            .map(|i| (i as f64) * (2.0 * std::f64::consts::PI / (n_samples - 1) as f64))
            .collect();
        let y: Vec<f64> = x.iter().map(|&v| v.sin()).collect();
        let x_eval = make_eval(x[0], x[n_samples - 1], 30);

        // Cubic spline
        let t_cubic = uniform_knot_vector(x[0], x[n_samples - 1], 8, 3);
        pts.push(CasePoint {
            case_id: "sine_cubic_k3".into(),
            x: x.clone(),
            y: y.clone(),
            t: t_cubic,
            k: 3,
            x_eval: x_eval.clone(),
        });
        // Quintic spline
        let t_q = uniform_knot_vector(x[0], x[n_samples - 1], 6, 5);
        pts.push(CasePoint {
            case_id: "sine_quintic_k5".into(),
            x,
            y,
            t: t_q,
            k: 5,
            x_eval,
        });
    }

    // Dataset 2: quadratic y = x^2 on [-2, 2], 30 samples — cubic spline should
    // recover this near-exactly (modulo LS coefficient drift)
    {
        let n_samples = 30usize;
        let x: Vec<f64> = (0..n_samples)
            .map(|i| -2.0 + 4.0 * i as f64 / (n_samples - 1) as f64)
            .collect();
        let y: Vec<f64> = x.iter().map(|&v| v * v).collect();
        let x_eval = make_eval(-2.0, 2.0, 25);

        let t_cubic = uniform_knot_vector(-2.0, 2.0, 5, 3);
        pts.push(CasePoint {
            case_id: "quadratic_cubic_k3".into(),
            x: x.clone(),
            y: y.clone(),
            t: t_cubic,
            k: 3,
            x_eval: x_eval.clone(),
        });

        // Linear spline (k=1)
        let t_lin = uniform_knot_vector(-2.0, 2.0, 8, 1);
        pts.push(CasePoint {
            case_id: "quadratic_linear_k1".into(),
            x,
            y,
            t: t_lin,
            k: 1,
            x_eval,
        });
    }

    // Dataset 3: deterministic "noisy" curve y = sin(x) + 0.1*cos(3x) on [0, 2π]
    {
        let n_samples = 60usize;
        let x: Vec<f64> = (0..n_samples)
            .map(|i| (i as f64) * (2.0 * std::f64::consts::PI / (n_samples - 1) as f64))
            .collect();
        let y: Vec<f64> = x.iter().map(|&v| v.sin() + 0.1 * (3.0 * v).cos()).collect();
        let x_eval = make_eval(x[0], x[n_samples - 1], 35);
        let t_cubic = uniform_knot_vector(x[0], x[n_samples - 1], 10, 3);
        pts.push(CasePoint {
            case_id: "noisy_cubic_k3".into(),
            x,
            y,
            t: t_cubic,
            k: 3,
            x_eval,
        });
    }

    OracleQuery { points: pts }
}

fn scipy_oracle_or_skip(q: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json, math, sys
import numpy as np
from scipy.interpolate import make_lsq_spline

q = json.load(sys.stdin)
out = []
for c in q["points"]:
    cid = c["case_id"]
    try:
        x = np.array(c["x"], dtype=float)
        y = np.array(c["y"], dtype=float)
        t = np.array(c["t"], dtype=float)
        spl = make_lsq_spline(x, y, t, k=c["k"])
        x_eval = np.array(c["x_eval"], dtype=float)
        y_eval = spl(x_eval)
        if not np.all(np.isfinite(y_eval)):
            out.append({"case_id": cid, "y_eval": None})
        else:
            out.append({"case_id": cid, "y_eval": [float(v) for v in y_eval]})
    except Exception:
        out.append({"case_id": cid, "y_eval": None})

print(json.dumps({"points": out}))
"#;
    let query_json = serde_json::to_string(q).expect("serialize");
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
                "python3 spawn failed: {e}"
            );
            eprintln!("skipping lsq_spline oracle: python3 unavailable ({e})");
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
            eprintln!("skipping lsq_spline oracle: stdin write failed");
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
        eprintln!("skipping lsq_spline oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse JSON"))
}

#[test]
fn diff_interpolate_make_lsq_spline() {
    let query = build_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();

    for (case, o) in query.points.iter().zip(oracle.points.iter()) {
        assert_eq!(case.case_id, o.case_id);
        let Some(expected) = o.y_eval.as_ref() else {
            continue;
        };

        let spline = match make_lsq_spline(&case.x, &case.y, &case.t, case.k) {
            Ok(s) => s,
            Err(e) => {
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    max_abs_diff: f64::INFINITY,
                    max_rel_diff: f64::INFINITY,
                    n_eval: 0,
                    pass: false,
                    note: format!("make_lsq_spline error: {e:?}"),
                });
                continue;
            }
        };

        let mut max_abs = 0.0_f64;
        let mut max_rel = 0.0_f64;
        for (xi, exp) in case.x_eval.iter().zip(expected.iter()) {
            let actual = spline.eval(*xi);
            let abs_d = (actual - exp).abs();
            let denom = exp.abs().max(1.0e-300);
            let rel_d = abs_d / denom;
            max_abs = max_abs.max(abs_d);
            max_rel = max_rel.max(rel_d);
        }

        let pass = max_rel <= REL_TOL || max_abs <= ABS_TOL;
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            max_abs_diff: max_abs,
            max_rel_diff: max_rel,
            n_eval: case.x_eval.len(),
            pass,
            note: String::new(),
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_interpolate_make_lsq_spline".into(),
        category: "fsci_interpolate::make_lsq_spline vs scipy.interpolate".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "lsq_spline mismatch: {} max_rel={} max_abs={} note={}",
                d.case_id, d.max_rel_diff, d.max_abs_diff, d.note
            );
        }
    }

    assert!(
        all_pass,
        "make_lsq_spline parity failed: {} cases",
        diffs.len()
    );
}
