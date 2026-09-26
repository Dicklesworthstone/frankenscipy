#![forbid(unsafe_code)]
//! Property-based test for fsci_opt::differential_evolution_constrained.
//!
//! Resolves [frankenscipy-o7gko]. Convert inequality constraints into
//! penalty terms in the inner DE call. Test on convex objectives with
//! known constrained minima.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_opt::{DifferentialEvolutionOptions, differential_evolution_constrained};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
const TOL: f64 = 1.0e-2;

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
    fs::create_dir_all(output_dir()).expect("create dec diff dir");
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
fn diff_opt_de_constrained() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut max_overall = 0.0_f64;
    // The analytic optimum is the reference side; an fsci error is `rust_failed`.
    let mut ledger = CompareLedger::new(
        "diff_opt_de_constrained",
        &["differential_evolution_constrained"],
    );

    let opts = DifferentialEvolutionOptions {
        maxiter: 500,
        popsize: 20,
        seed: Some(42),
        ..DifferentialEvolutionOptions::default()
    };

    // Problem 1: min x²+y² s.t. x+y >= 1, on [-2,2]²
    // Constraint violation: max(0, 1 - (x+y))
    // Analytical: (0.5, 0.5), f* = 0.5
    let f1 = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
    let cv1 = |x: &[f64]| (1.0 - (x[0] + x[1])).max(0.0);
    let res1 =
        differential_evolution_constrained(f1, &[(-2.0_f64, 2.0), (-2.0, 2.0)], cv1, opts.clone())
            .ok();
    if let Some((f_star1, f_at1)) = ledger.pair(
        "differential_evolution_constrained",
        "dec_linear_constraint",
        Some(0.5),
        res1.map(|r| f1(&r.x)),
    ) {
        let abs_d1 = (f_at1 - f_star1).abs();
        max_overall = max_overall.max(abs_d1);
        ledger.compared(
            "differential_evolution_constrained",
            "dec_linear_constraint",
            abs_d1 <= TOL,
        );
        diffs.push(CaseDiff {
            case_id: "dec_linear_constraint".into(),
            abs_diff: abs_d1,
            pass: abs_d1 <= TOL,
        });
    }

    // Problem 2: min (x-2)² + (y-2)² s.t. x² + y² <= 1 (disk constraint), on [-3, 3]²
    // Analytical: closest point on unit circle to (2, 2), i.e. (1/sqrt(2), 1/sqrt(2))
    // Distance: sqrt(2)*sqrt(2)·(2 - 1/sqrt(2)) = 2 - 1/sqrt(2) and squared distance ≈ 1.7157
    // Compute: ‖(2,2) − (1/√2, 1/√2)‖² = 2·(2 − 1/√2)²
    let target_f = 2.0 * (2.0 - 1.0 / 2.0_f64.sqrt()).powi(2);
    let f2 = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2);
    let cv2 = |x: &[f64]| ((x[0] * x[0] + x[1] * x[1]) - 1.0).max(0.0);
    let res2 =
        differential_evolution_constrained(f2, &[(-3.0_f64, 3.0), (-3.0, 3.0)], cv2, opts.clone())
            .ok();
    if let Some((target_f, f_at2)) = ledger.pair(
        "differential_evolution_constrained",
        "dec_disk_constraint",
        Some(target_f),
        res2.map(|r| f2(&r.x)),
    ) {
        let abs_d2 = (f_at2 - target_f).abs();
        max_overall = max_overall.max(abs_d2);
        ledger.compared(
            "differential_evolution_constrained",
            "dec_disk_constraint",
            abs_d2 <= TOL,
        );
        diffs.push(CaseDiff {
            case_id: "dec_disk_constraint".into(),
            abs_diff: abs_d2,
            pass: abs_d2 <= TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_de_constrained".into(),
        category: "fsci_opt::differential_evolution_constrained property test".into(),
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
            eprintln!("dec mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "dec conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    // Two analytic problems, each compared once.
    ledger.finish(2);
}
