#![forbid(unsafe_code)]
//! Property-based test for fsci_opt::augmented_lagrangian.
//!
//! Resolves [frankenscipy-p7e69]. The augmented-Lagrangian solver
//! minimizes f(x) subject to c(x) >= 0. Each subproblem is solved
//! via Nelder-Mead. Test on convex problems with analytical
//! solutions.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_opt::augmented_lagrangian;
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
const TOL: f64 = 5.0e-2;

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
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
    fs::create_dir_all(output_dir()).expect("create al diff dir");
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

#[test]
fn diff_opt_augmented_lagrangian() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut max_overall = 0.0_f64;
    // The analytic optimum is the reference side; an fsci error is `rust_failed`.
    let mut ledger = CompareLedger::new("diff_opt_augmented_lagrangian", &["augmented_lagrangian"]);

    // Problem 1: min x²+y² s.t. x+y >= 1
    // Analytical solution: (0.5, 0.5), f* = 0.5
    let f1 = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
    let c1 = |x: &[f64]| vec![x[0] + x[1] - 1.0];
    let res1 = augmented_lagrangian(f1, c1, &[0.0_f64, 0.0], 1, 20, 200).ok();
    if let Some((f_star1, f_at1)) = ledger.pair(
        "augmented_lagrangian",
        "al_linear_constraint",
        Some(0.5),
        res1.map(|r| f1(&r.x)),
    ) {
        let abs_d1 = (f_at1 - f_star1).abs();
        max_overall = max_overall.max(abs_d1);
        ledger.compared(
            "augmented_lagrangian",
            "al_linear_constraint",
            abs_d1 <= TOL,
        );
        diffs.push(CaseDiff {
            case_id: "al_linear_constraint".into(),
            abs_diff: abs_d1,
            pass: abs_d1 <= TOL,
        });
    }

    // Problem 2: min x²+y² s.t. x >= 1, y >= 1
    // Analytical solution: (1, 1), f* = 2.
    let f2 = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
    let c2 = |x: &[f64]| vec![x[0] - 1.0, x[1] - 1.0];
    let res2 = augmented_lagrangian(f2, c2, &[0.0_f64, 0.0], 2, 20, 200).ok();
    if let Some((f_star2, f_at2)) = ledger.pair(
        "augmented_lagrangian",
        "al_two_constraints",
        Some(2.0),
        res2.map(|r| f2(&r.x)),
    ) {
        let abs_d2 = (f_at2 - f_star2).abs();
        max_overall = max_overall.max(abs_d2);
        ledger.compared("augmented_lagrangian", "al_two_constraints", abs_d2 <= TOL);
        diffs.push(CaseDiff {
            case_id: "al_two_constraints".into(),
            abs_diff: abs_d2,
            pass: abs_d2 <= TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_augmented_lagrangian".into(),
        category: "fsci_opt::augmented_lagrangian property test".into(),
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
            eprintln!("al mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "augmented_lagrangian conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    // Two analytic problems, each compared once.
    ledger.finish(2);
}
