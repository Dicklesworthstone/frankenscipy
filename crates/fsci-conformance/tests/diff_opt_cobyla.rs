#![forbid(unsafe_code)]
//! Property-based test for fsci_opt::cobyla.
//!
//! Resolves [frankenscipy-m12tg]. COBYLA minimizes f(x) subject to
//! inequality constraints c_i(x) >= 0. Test on convex problems
//! with known analytical solutions. 5e-2 abs.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_opt::cobyla;
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
    fs::create_dir_all(output_dir()).expect("create cobyla diff dir");
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
fn diff_opt_cobyla() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut max_overall = 0.0_f64;
    // The analytic optimum is the reference side; an fsci error is `rust_failed`.
    let mut ledger = CompareLedger::new("diff_opt_cobyla", &["cobyla"]);

    // Problem 1: min x²+y² s.t. x+y >= 1.
    // Analytical solution: (0.5, 0.5), f* = 0.5.
    let f1 = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
    let c1: [fn(&[f64]) -> f64; 1] = [|x: &[f64]| x[0] + x[1] - 1.0];
    let res1 = cobyla(f1, &[0.0_f64, 0.0], &c1, 500, 0.5).ok();
    if let Some((f_star1, f_at1)) =
        ledger.pair("cobyla", "cobyla_linear", Some(0.5), res1.map(|r| f1(&r.x)))
    {
        let abs_d1 = (f_at1 - f_star1).abs();
        max_overall = max_overall.max(abs_d1);
        ledger.compared("cobyla", "cobyla_linear", abs_d1 <= TOL);
        diffs.push(CaseDiff {
            case_id: "cobyla_linear".into(),
            abs_diff: abs_d1,
            pass: abs_d1 <= TOL,
        });
    }

    // Problem 2: min (x-2)² + (y-1)² s.t. x >= 0, y >= 0
    // Analytical: (2, 1), f* = 0
    let f2 = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 1.0).powi(2);
    let c2: [fn(&[f64]) -> f64; 2] = [|x: &[f64]| x[0], |x: &[f64]| x[1]];
    let res2 = cobyla(f2, &[1.0_f64, 0.5], &c2, 500, 0.5).ok();
    if let Some((f_star2, f_at2)) = ledger.pair(
        "cobyla",
        "cobyla_two_pos",
        Some(0.0),
        res2.map(|r| f2(&r.x)),
    ) {
        // f* = 0, so this is f(x) itself.
        let f_at2 = f_at2 - f_star2;
        max_overall = max_overall.max(f_at2);
        ledger.compared("cobyla", "cobyla_two_pos", f_at2 <= TOL);
        diffs.push(CaseDiff {
            case_id: "cobyla_two_pos".into(),
            abs_diff: f_at2,
            pass: f_at2 <= TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_cobyla".into(),
        category: "fsci_opt::cobyla property test".into(),
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
            eprintln!("cobyla mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "cobyla conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    // Two analytic problems, each compared once.
    ledger.finish(2);
}
