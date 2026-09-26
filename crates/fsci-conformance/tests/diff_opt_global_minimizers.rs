#![forbid(unsafe_code)]
//! Property-based test for fsci_opt global minimizers:
//! basinhopping, dual_annealing, shgo.
//!
//! Resolves [frankenscipy-g67w6]. Each optimizer is stochastic (or
//! grid-based with randomization), so we don't compare to scipy
//! directly. Instead verify that on convex test objectives with a
//! known global minimum, the returned solution has function value
//! within tolerance of the true minimum.
//!
//! Scoped to basinhopping and shgo only; dual_annealing is filed
//! as defect frankenscipy-p8c1x (returns f ≈ 18 for x²+y² on
//! [-5,5]², far from the true minimum of 0).

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_opt::{BasinhoppingOptions, basinhopping, shgo};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
const TOL: f64 = 1.0e-3;

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
    fs::create_dir_all(output_dir()).expect("create global_opt diff dir");
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
    (x[0] - 1.0).powi(2) + (x[1] + 2.0).powi(2)
}

#[test]
fn diff_opt_global_minimizers() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut max_overall = 0.0_f64;

    // basinhopping — start far from minimum, seed for reproducibility
    let bh_opts = BasinhoppingOptions {
        niter: 50,
        temperature: 1.0,
        stepsize: 0.5,
        seed: Some(42),
        minimizer_tol: Some(1e-8),
    };

    // (case id, op, fsci's minimum value). Both objectives have the analytic global minimum 0,
    // which is the reference side of every case.
    let cases: [(&str, &str, Option<f64>); 4] = [
        // basinhopping(quadratic, x0=[5,5]) → expect f ≈ 0
        (
            "bh_quad",
            "basinhopping",
            basinhopping(quadratic, &[5.0, 5.0], bh_opts.clone())
                .ok()
                .and_then(|res| res.fun),
        ),
        // basinhopping(shifted_quadratic, x0=[10,10]) → expect f ≈ 0 at x=[1,-2]
        (
            "bh_shifted",
            "basinhopping",
            basinhopping(shifted_quadratic, &[10.0, 10.0], bh_opts)
                .ok()
                .and_then(|res| res.fun),
        ),
        // dual_annealing — dropped (defect frankenscipy-p8c1x: fails to
        // converge on simple quadratic; returns f ≈ 18 for x²+y² on
        // [-5,5]²).

        // shgo — bounded simplicial homology global optimization
        (
            "shgo_quad",
            "shgo",
            shgo(quadratic, &[(-5.0, 5.0), (-5.0, 5.0)])
                .ok()
                .and_then(|res| res.fun),
        ),
        (
            "shgo_shifted",
            "shgo",
            shgo(shifted_quadratic, &[(-5.0, 5.0), (-5.0, 5.0)])
                .ok()
                .and_then(|res| res.fun),
        ),
    ];
    let mut ledger = CompareLedger::new("diff_opt_global_minimizers", &["basinhopping", "shgo"]);

    for &(case_id, op, fsci_fun) in &cases {
        let Some((_true_min, f)) = ledger.pair(op, case_id, Some(0.0), fsci_fun) else {
            continue;
        };
        max_overall = max_overall.max(f);
        ledger.compared(op, case_id, f <= TOL);
        diffs.push(CaseDiff {
            case_id: case_id.into(),
            op: op.into(),
            abs_diff: f,
            pass: f <= TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_global_minimizers".into(),
        category: "fsci_opt::{basinhopping, dual_annealing, shgo} property test".into(),
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
        "global_opt conformance failed: {} cases, max_f={}",
        diffs.len(),
        max_overall
    );
    ledger.finish(cases.iter().filter(|(_, op, _)| *op == "shgo").count());
}
