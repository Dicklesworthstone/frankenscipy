#![forbid(unsafe_code)]
//! Property-based test for fsci_opt::projected_gradient_descent.
//!
//! Resolves [frankenscipy-j34ap]. Box-constrained minimization
//! of differentiable convex objectives. Verify the solution
//! satisfies KKT-like property: f(x) at the box-constrained minimum
//! is within tolerance of the analytical answer.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_opt::projected_gradient_descent;
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
    fs::create_dir_all(output_dir()).expect("create pgd diff dir");
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
fn diff_opt_projected_gradient_descent() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut max_overall = 0.0_f64;
    let arm = "projected_gradient_descent";
    let mut ledger = CompareLedger::new("diff_opt_projected_gradient_descent", &[arm]);

    // The analytic minimum always exists, so an unconverged fsci run is a failure, not a skip.
    // Quadratic: f(x) = x[0]² + x[1]². Min inside [-1,1]² at (0,0); f*=0.
    let f1 = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
    let g1 = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
    let res1 = projected_gradient_descent(
        f1,
        g1,
        &[0.5, 0.5],
        &[-1.0, -1.0],
        &[1.0, 1.0],
        1e-8,
        1000,
        0.1,
    );
    let fsci_f1 = res1.success.then(|| f1(&res1.x));
    if let Some((_, f_at)) = ledger.pair(arm, "pgd_quad_interior", Some(0.0), fsci_f1) {
        max_overall = max_overall.max(f_at);
        ledger.compared(arm, "pgd_quad_interior", f_at <= TOL);
        diffs.push(CaseDiff {
            case_id: "pgd_quad_interior".into(),
            abs_diff: f_at,
            pass: f_at <= TOL,
        });
    }

    // Shifted quadratic with min OUTSIDE the box: f(x) = (x[0]-3)² + (x[1]-3)².
    // Box [-1, 1]². Constrained min is at (1, 1) with f*=8.
    let f2 = |x: &[f64]| (x[0] - 3.0).powi(2) + (x[1] - 3.0).powi(2);
    let g2 = |x: &[f64]| vec![2.0 * (x[0] - 3.0), 2.0 * (x[1] - 3.0)];
    let res2 = projected_gradient_descent(
        f2,
        g2,
        &[0.0, 0.0],
        &[-1.0, -1.0],
        &[1.0, 1.0],
        1e-8,
        1000,
        0.1,
    );
    let fsci_f2 = res2.success.then(|| f2(&res2.x));
    if let Some((expected, f_at2)) = ledger.pair(arm, "pgd_quad_corner", Some(8.0), fsci_f2) {
        let abs_d2 = (f_at2 - expected).abs();
        max_overall = max_overall.max(abs_d2);
        ledger.compared(arm, "pgd_quad_corner", abs_d2 <= TOL);
        diffs.push(CaseDiff {
            case_id: "pgd_quad_corner".into(),
            abs_diff: abs_d2,
            pass: abs_d2 <= TOL,
        });
    }

    // 1D quadratic with min at x=2; box [0, 3]. Constrained min at 2.
    let f3 = |x: &[f64]| (x[0] - 2.0).powi(2);
    let g3 = |x: &[f64]| vec![2.0 * (x[0] - 2.0)];
    let res3 = projected_gradient_descent(f3, g3, &[0.5], &[0.0], &[3.0], 1e-8, 1000, 0.1);
    let fsci_x3 = res3.x.first().copied().filter(|_| res3.success);
    if let Some((expected, x3)) = ledger.pair(arm, "pgd_1d_interior", Some(2.0), fsci_x3) {
        let abs_d3 = (x3 - expected).abs();
        max_overall = max_overall.max(abs_d3);
        ledger.compared(arm, "pgd_1d_interior", abs_d3 <= TOL);
        diffs.push(CaseDiff {
            case_id: "pgd_1d_interior".into(),
            abs_diff: abs_d3,
            pass: abs_d3 <= TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_projected_gradient_descent".into(),
        category: "fsci_opt::projected_gradient_descent property test".into(),
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
            eprintln!("pgd mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "pgd conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    // Three inline fixtures, all on the one arm.
    ledger.finish(3);
}
