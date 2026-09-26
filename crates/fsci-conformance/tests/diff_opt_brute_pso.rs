#![forbid(unsafe_code)]
//! Property-based test for fsci_opt::{brute, pso}.
//!
//! Resolves [frankenscipy-gt9af]. brute is deterministic grid
//! search; pso is stochastic but reproducible with a seed. On
//! convex objectives both should find a solution near the true
//! global minimum. Tolerance generous for the grid (depends on
//! resolution); 1e-1 for pso (population-based stochastic).

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_opt::{brute, pso};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
const BRUTE_TOL: f64 = 0.5; // Coarse grid resolution
const PSO_TOL: f64 = 1.0e-2;

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
    compared: BTreeMap<String, ArmCounts>,
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
    fs::create_dir_all(output_dir()).expect("create brute_pso diff dir");
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

fn quadratic(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum()
}

fn shifted_quadratic(x: &[f64]) -> f64 {
    (x[0] - 1.5).powi(2) + (x[1] + 0.5).powi(2)
}

#[test]
fn diff_opt_brute_pso() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut max_overall = 0.0_f64;
    // The analytic global minimum (f* = 0 for both objectives) is the reference side; an fsci
    // error or a missing `fun` is `rust_failed`.
    let mut ledger = CompareLedger::new("diff_opt_brute_pso", &["brute", "pso"]);

    // brute — quadratic on [-2, 2]² with 21 grid points
    let brute_quad = brute(quadratic, &[(-2.0, 2.0), (-2.0, 2.0)], 21)
        .ok()
        .and_then(|res| res.fun);
    if let Some((f_star, f)) = ledger.pair("brute", "brute_quad", Some(0.0), brute_quad) {
        let f = f - f_star;
        max_overall = max_overall.max(f);
        ledger.compared("brute", "brute_quad", f <= BRUTE_TOL);
        diffs.push(CaseDiff {
            case_id: "brute_quad".into(),
            op: "brute".into(),
            abs_diff: f,
            pass: f <= BRUTE_TOL,
        });
    }
    let brute_shifted = brute(shifted_quadratic, &[(-2.0, 2.0), (-2.0, 2.0)], 41)
        .ok()
        .and_then(|res| res.fun);
    if let Some((f_star, f)) = ledger.pair("brute", "brute_shifted", Some(0.0), brute_shifted) {
        let f = f - f_star;
        max_overall = max_overall.max(f);
        ledger.compared("brute", "brute_shifted", f <= BRUTE_TOL);
        diffs.push(CaseDiff {
            case_id: "brute_shifted".into(),
            op: "brute".into(),
            abs_diff: f,
            pass: f <= BRUTE_TOL,
        });
    }

    // pso — quadratic on [-3, 3]² with 30 particles and 100 iters
    let (_, f1) = pso(quadratic, &[-3.0, -3.0], &[3.0, 3.0], 30, 100, 42);
    if let Some((f_star, f1)) = ledger.pair("pso", "pso_quad", Some(0.0), Some(f1)) {
        let f1 = f1 - f_star;
        max_overall = max_overall.max(f1);
        ledger.compared("pso", "pso_quad", f1 <= PSO_TOL);
        diffs.push(CaseDiff {
            case_id: "pso_quad".into(),
            op: "pso".into(),
            abs_diff: f1,
            pass: f1 <= PSO_TOL,
        });
    }
    let (_, f2) = pso(shifted_quadratic, &[-3.0, -3.0], &[3.0, 3.0], 30, 100, 42);
    if let Some((f_star, f2)) = ledger.pair("pso", "pso_shifted", Some(0.0), Some(f2)) {
        let f2 = f2 - f_star;
        max_overall = max_overall.max(f2);
        ledger.compared("pso", "pso_shifted", f2 <= PSO_TOL);
        diffs.push(CaseDiff {
            case_id: "pso_shifted".into(),
            op: "pso".into(),
            abs_diff: f2,
            pass: f2 <= PSO_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_brute_pso".into(),
        category: "fsci_opt::{brute, pso} property test".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        max_abs_diff: max_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("{} mismatch: {} f={}", d.op, d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "brute_pso conformance failed: {} cases, max_f={}",
        diffs.len(),
        max_overall
    );
    // Two objectives per method, each compared once.
    ledger.finish(2);
}
