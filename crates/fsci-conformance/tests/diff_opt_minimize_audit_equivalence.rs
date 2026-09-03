#![forbid(unsafe_code)]
//! Property test: fsci_opt::minimize_with_audit produces same
//! result as minimize for both success and edge cases.
//!
//! Resolves [frankenscipy-lg5aa].

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_opt::{MinimizeOptions, OptimizeMethod, minimize, minimize_with_audit, sync_audit_ledger};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;

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
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create minimize_audit diff dir");
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
fn diff_opt_minimize_audit_equivalence() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let ledger = sync_audit_ledger();

    // Three test objectives
    let quadratic = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>();
    let rosen = |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
    let shifted_quad =
        |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2) + (x[2] + 0.5).powi(2);

    let probes: Vec<(&str, Vec<f64>, OptimizeMethod, fn(&[f64]) -> f64)> = vec![
        (
            "quad_neldermead",
            vec![1.0, 1.0],
            OptimizeMethod::NelderMead,
            quadratic,
        ),
        (
            "quad_bfgs",
            vec![2.0, -1.0],
            OptimizeMethod::Bfgs,
            quadratic,
        ),
        (
            "rosen_neldermead",
            vec![0.0, 0.0],
            OptimizeMethod::NelderMead,
            rosen,
        ),
        (
            "rosen_powell",
            vec![0.0, 0.0],
            OptimizeMethod::Powell,
            rosen,
        ),
        (
            "shifted_neldermead",
            vec![0.0, 0.0, 0.0],
            OptimizeMethod::NelderMead,
            shifted_quad,
        ),
        (
            "shifted_bfgs",
            vec![0.0, 0.0, 0.0],
            OptimizeMethod::Bfgs,
            shifted_quad,
        ),
    ];

    for (label, x0, method, f) in probes {
        let opts = MinimizeOptions {
            method: Some(method),
            ..MinimizeOptions::default()
        };
        let plain = minimize(f, &x0, opts);
        let audited = minimize_with_audit(f, &x0, opts, &ledger);
        let pass = match (&plain, &audited) {
            (Ok(p), Ok(a)) => {
                let d_x = vec_max_diff(&p.x, &a.x);
                let pf = p.fun.unwrap_or(f64::NAN);
                let af = a.fun.unwrap_or(f64::NAN);
                let d_f = if pf.is_finite() && af.is_finite() {
                    (pf - af).abs()
                } else {
                    if pf.is_nan() && af.is_nan() {
                        0.0
                    } else {
                        f64::INFINITY
                    }
                };
                d_x <= ABS_TOL && d_f <= ABS_TOL && p.status == a.status
            }
            (Err(pe), Err(ae)) => format!("{pe:?}") == format!("{ae:?}"),
            _ => false,
        };
        let d = match (&plain, &audited) {
            (Ok(p), Ok(a)) => vec_max_diff(&p.x, &a.x),
            _ => 0.0,
        };
        diffs.push(CaseDiff {
            case_id: label.to_string(),
            op: format!("{method:?}"),
            abs_diff: d,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_minimize_audit_equivalence".into(),
        category: "fsci_opt::minimize_with_audit equivalent to minimize".into(),
        case_count: diffs.len(),
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
        "minimize_audit_equiv conformance failed: {} cases",
        diffs.len(),
    );
}
