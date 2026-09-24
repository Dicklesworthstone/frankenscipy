#![forbid(unsafe_code)]
//! Property-based test for fsci_opt::{line_search_wolfe1, line_search_wolfe2}.
//!
//! Resolves [frankenscipy-2tf1p]. Verify that the returned step
//! length α satisfies the Wolfe conditions on quadratic objectives:
//!   Armijo:    f(x + α*d) <= f(x) + c1 * α * g'd
//!   Curvature (Wolfe1): g(x + α*d)'d >= c2 * g'd
//!   Strong Wolfe (Wolfe2): |g(x + α*d)'d| <= c2 * |g'd|

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_opt::{WolfeParams, line_search_wolfe1, line_search_wolfe2};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";

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
    fs::create_dir_all(output_dir()).expect("create wolfe diff dir");
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

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[test]
fn diff_opt_line_search_wolfe() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let params = WolfeParams::default();

    // Quadratic: f(x) = x[0]² + x[1]², grad = 2*x.
    let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
    let g = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];

    let probes: &[(&str, Vec<f64>, Vec<f64>)] = &[
        // descent direction = -grad
        ("from_1_1", vec![1.0, 1.0], vec![-2.0, -2.0]),
        ("from_3_neg2", vec![3.0, -2.0], vec![-6.0, 4.0]),
        ("from_0p5_2", vec![0.5, 2.0], vec![-1.0, -4.0]),
    ];

    for (label, x0, direction) in probes {
        let f0 = f(x0);
        let g0 = g(x0);
        // Wolfe 1 (dcsrch). Every probe must produce a step: an error used to be skipped, so a
        // search that failed everywhere passed with nothing compared.
        {
            let res = line_search_wolfe1(&f, &g, x0, direction, f0, &g0, params)
                .unwrap_or_else(|e| panic!("wolfe1 {label}: {e}"));
            // Verify Armijo
            let xp: Vec<f64> = x0
                .iter()
                .zip(direction.iter())
                .map(|(xi, di)| xi + res.alpha * di)
                .collect();
            let f_at = f(&xp);
            let g_at = g(&xp);
            let dg0 = dot(&g0, direction);
            let dg_at = dot(&g_at, direction);
            let armijo_ok = f_at <= f0 + params.c1 * res.alpha * dg0 + 1e-10;
            let curvature_ok = dg_at >= params.c2 * dg0 - 1e-10;
            let pass = armijo_ok && curvature_ok && res.alpha > 0.0;
            diffs.push(CaseDiff {
                case_id: format!("wolfe1_{label}"),
                op: "wolfe1".into(),
                abs_diff: res.alpha,
                pass,
            });
        }
        // Wolfe 2 (strong)
        {
            let res = line_search_wolfe2(&f, &g, x0, direction, f0, &g0, params)
                .unwrap_or_else(|e| panic!("wolfe2 {label}: {e}"));
            let xp: Vec<f64> = x0
                .iter()
                .zip(direction.iter())
                .map(|(xi, di)| xi + res.alpha * di)
                .collect();
            let f_at = f(&xp);
            let g_at = g(&xp);
            let dg0 = dot(&g0, direction);
            let dg_at = dot(&g_at, direction);
            let armijo_ok = f_at <= f0 + params.c1 * res.alpha * dg0 + 1e-10;
            let strong_ok = dg_at.abs() <= params.c2 * dg0.abs() + 1e-10;
            let pass = armijo_ok && strong_ok && res.alpha > 0.0;
            diffs.push(CaseDiff {
                case_id: format!("wolfe2_{label}"),
                op: "wolfe2".into(),
                abs_diff: res.alpha,
                pass,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_line_search_wolfe".into(),
        category: "fsci_opt::{line_search_wolfe1, line_search_wolfe2} property test".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("{} mismatch: {} alpha={}", d.op, d.case_id, d.abs_diff);
        }
    }

    assert_eq!(
        diffs.len(),
        2 * probes.len(),
        "every probe must be compared"
    );
    assert!(
        all_pass,
        "line_search_wolfe conformance failed: {} cases",
        diffs.len(),
    );
}
