#![forbid(unsafe_code)]
//! Analytic-minimum parity for fsci_opt minimize methods on simple
//! quadratic objectives with known minima.
//!
//! Resolves [frankenscipy-mu0xg]. Each method must converge to within
//! 1e-4 of the analytic minimum.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_opt::{MinimizeOptions, bfgs, nelder_mead, powell};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-4;
const METHODS: [&str; 3] = ["bfgs", "powell", "nelder_mead"];

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    method: String,
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
    fs::create_dir_all(output_dir()).expect("create minimize diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize minimize diff log");
    fs::write(path, json).expect("write minimize diff log");
}

#[test]
fn diff_opt_minimize_quadratic() {
    let opts = MinimizeOptions::default();
    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    // Models: (label, f, x0, x_min)
    let cases: Vec<(&str, Box<dyn Fn(&[f64]) -> f64>, Vec<f64>, Vec<f64>)> = vec![
        // (x - 3)² + (y + 2)² → min at (3, -2)
        (
            "shifted_quad_2d",
            Box::new(|v: &[f64]| (v[0] - 3.0).powi(2) + (v[1] + 2.0).powi(2)),
            vec![0.0, 0.0],
            vec![3.0, -2.0],
        ),
        // x² + 4*y² + 9*z² → min at (0, 0, 0)
        (
            "ellipsoid_3d",
            Box::new(|v: &[f64]| v[0] * v[0] + 4.0 * v[1] * v[1] + 9.0 * v[2] * v[2]),
            vec![1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0],
        ),
        // (x - 1)² + (y - 1)² + (z - 1)² + (w - 1)² → min at (1,1,1,1)
        (
            "shifted_unit_4d",
            Box::new(|v: &[f64]| {
                (v[0] - 1.0).powi(2)
                    + (v[1] - 1.0).powi(2)
                    + (v[2] - 1.0).powi(2)
                    + (v[3] - 1.0).powi(2)
            }),
            vec![0.0, 0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0, 1.0],
        ),
    ];

    let mut ledger = CompareLedger::new("diff_opt_minimize_quadratic", &METHODS);
    for (label, f, x0, x_min) in &cases {
        for method in METHODS {
            let res = match method {
                "bfgs" => bfgs(&|v: &[f64]| f(v), x0, opts),
                "powell" => powell(&|v: &[f64]| f(v), x0, opts),
                "nelder_mead" => nelder_mead(&|v: &[f64]| f(v), x0, opts),
                other => panic!("unknown method {other}"),
            };
            let case_id = format!("{label}_{method}");
            // The analytic minimum exists, so an unsuccessful fsci run is an fsci failure.
            let fsci_x = res.ok().filter(|r| r.success).map(|r| r.x);
            let Some((x_min, x)) =
                ledger.slices(method, &case_id, Some(x_min.as_slice()), fsci_x.as_deref())
            else {
                continue;
            };
            let d = x
                .iter()
                .zip(x_min.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            max_overall = max_overall.max(d);
            ledger.compared(method, &case_id, d <= ABS_TOL);
            diffs.push(CaseDiff {
                case_id,
                method: method.into(),
                abs_diff: d,
                pass: d <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_minimize_quadratic".into(),
        category: "fsci_opt minimize methods on quadratic objectives".into(),
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
                "{} mismatch: {} abs_diff={}",
                d.method, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "minimize_quadratic conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    ledger.finish(cases.len());
}
