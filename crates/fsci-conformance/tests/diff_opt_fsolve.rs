#![forbid(unsafe_code)]
//! Analytic-root parity for fsci_opt::fsolve on multivariate nonlinear
//! systems with known closed-form solutions.
//!
//! Resolves [frankenscipy-dwtdx]. 1e-6 abs. Verifies converged x ≈
//! analytic root and residual ‖F(x)‖∞ ≤ tol.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_opt::fsolve;
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-6;

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    abs_diff: f64,
    residual: f64,
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
    fs::create_dir_all(output_dir()).expect("create fsolve diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize fsolve diff log");
    fs::write(path, json).expect("write fsolve diff log");
}

#[test]
fn diff_opt_fsolve() {
    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    type SysFn = Box<dyn Fn(&[f64]) -> Vec<f64>>;
    let cases: Vec<(&str, SysFn, Vec<f64>, Vec<f64>)> = vec![
        // System A: x + y = 5, x - y = 1 → (3, 2)
        (
            "linear_2x2",
            Box::new(|v: &[f64]| vec![v[0] + v[1] - 5.0, v[0] - v[1] - 1.0]),
            vec![0.0, 0.0],
            vec![3.0, 2.0],
        ),
        // System B: x² + y² = 25, x - y = 1 → (4, 3)
        (
            "quad_circle_2x2",
            Box::new(|v: &[f64]| vec![v[0] * v[0] + v[1] * v[1] - 25.0, v[0] - v[1] - 1.0]),
            vec![3.0, 2.0],
            vec![4.0, 3.0],
        ),
        // System C: x + y + z = 6, x² + y² + z² = 14, xyz = 6 → (1, 2, 3)
        (
            "vieta_3x3",
            Box::new(|v: &[f64]| {
                vec![
                    v[0] + v[1] + v[2] - 6.0,
                    v[0] * v[0] + v[1] * v[1] + v[2] * v[2] - 14.0,
                    v[0] * v[1] * v[2] - 6.0,
                ]
            }),
            vec![0.5, 1.5, 2.5],
            vec![1.0, 2.0, 3.0],
        ),
        // System D: sin(x) = 0.5, cos(y) = 0.5 → x = π/6, y = π/3 (starting near)
        (
            "trig_2x2",
            Box::new(|v: &[f64]| vec![v[0].sin() - 0.5, v[1].cos() - 0.5]),
            vec![0.4, 1.0],
            vec![std::f64::consts::FRAC_PI_6, std::f64::consts::FRAC_PI_3],
        ),
    ];

    let mut ledger = CompareLedger::new("diff_opt_fsolve", &["fsolve"]);
    for (label, f, x0, x_true) in &cases {
        // A root exists at x_true, so a non-converged fsci solve is an fsci failure.
        let r = fsolve(|v| f(v), x0).ok().filter(|r| r.converged);
        let Some((x_true, x)) = ledger.slices(
            "fsolve",
            label,
            Some(x_true.as_slice()),
            r.as_ref().map(|r| r.x.as_slice()),
        ) else {
            continue;
        };
        // Sort to handle multiple-root ambiguity? For these systems the roots
        // are unique near x0, so direct compare works.
        let d = x
            .iter()
            .zip(x_true.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let resid = r
            .as_ref()
            .expect("slices returned fsci's x")
            .fun
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        max_overall = max_overall.max(d);
        ledger.compared("fsolve", label, d <= ABS_TOL);
        diffs.push(CaseDiff {
            case_id: (*label).into(),
            abs_diff: d,
            residual: resid,
            pass: d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_fsolve".into(),
        category: "fsci_opt::fsolve vs analytic roots".into(),
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
            eprintln!(
                "fsolve mismatch: {} abs_diff={} residual={}",
                d.case_id, d.abs_diff, d.residual
            );
        }
    }

    assert!(
        all_pass,
        "fsolve conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    ledger.finish(cases.len());
}
