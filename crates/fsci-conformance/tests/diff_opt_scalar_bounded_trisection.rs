#![forbid(unsafe_code)]
//! Property-based test for fsci_opt::{minimize_scalar_bounded, minimize_trisection}.
//!
//! Resolves [frankenscipy-2vg7w]. Both find the minimum of a 1D
//! function on a bracket. Test on unimodal functions with known
//! analytical minima.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_opt::{minimize_scalar_bounded, minimize_trisection};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
const TOL: f64 = 1.0e-5;

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
    fs::create_dir_all(output_dir()).expect("create scalar_b1 diff dir");
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
fn diff_opt_scalar_bounded_trisection() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut max_overall = 0.0_f64;

    // Test functions with known minima:
    // f1(x) = (x-2)² — min at x=2, f*=0
    // f2(x) = x² + 1 — min at x=0, f*=1
    // f3(x) = (x - 0.5).sin() — many minima; restrict bracket to one peak
    let probes: &[(&str, fn(f64) -> f64, f64, f64, f64, f64)] = &[
        ("quad_min2", |x| (x - 2.0).powi(2), 0.0, 5.0, 2.0, 0.0),
        ("quad_min0", |x| x * x + 1.0, -3.0, 3.0, 0.0, 1.0),
        (
            "cubic_min",
            |x| (x - 1.5).powi(2) + 0.3 * (x - 1.5).abs(),
            0.0,
            3.0,
            1.5,
            0.0,
        ),
    ];

    let mut ledger = CompareLedger::new(
        "diff_opt_scalar_bounded_trisection",
        &["bounded", "trisection"],
    );

    for (label, f, a, b, x_star, f_star) in probes {
        // (x*, f*) as one vector, so a NaN in either that `d_x.max(d_f)` would drop is caught.
        let expected = [*x_star, *f_star];

        let bounded_id = format!("bounded_{label}");
        let (x1, f1) = minimize_scalar_bounded(*f, (*a, *b), 1e-8, 200);
        let got1 = [x1, f1];
        if let Some((_, got1)) = ledger.slices(
            "bounded",
            &bounded_id,
            Some(expected.as_slice()),
            Some(got1.as_slice()),
        ) {
            let d_x = (got1[0] - x_star).abs();
            let d_f = (got1[1] - f_star).abs();
            let abs_d_bounded = d_x.max(d_f);
            max_overall = max_overall.max(abs_d_bounded);
            ledger.compared("bounded", &bounded_id, abs_d_bounded <= TOL);
            diffs.push(CaseDiff {
                case_id: bounded_id,
                op: "bounded".into(),
                abs_diff: abs_d_bounded,
                pass: abs_d_bounded <= TOL,
            });
        }

        let trisection_id = format!("trisection_{label}");
        let (x2, f2) = minimize_trisection(*f, *a, *b, 1e-8, 1000);
        let got2 = [x2, f2];
        if let Some((_, got2)) = ledger.slices(
            "trisection",
            &trisection_id,
            Some(expected.as_slice()),
            Some(got2.as_slice()),
        ) {
            let d_x2 = (got2[0] - x_star).abs();
            let d_f2 = (got2[1] - f_star).abs();
            let abs_d_tri = d_x2.max(d_f2);
            max_overall = max_overall.max(abs_d_tri);
            ledger.compared("trisection", &trisection_id, abs_d_tri <= TOL);
            diffs.push(CaseDiff {
                case_id: trisection_id,
                op: "trisection".into(),
                abs_diff: abs_d_tri,
                pass: abs_d_tri <= TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_scalar_bounded_trisection".into(),
        category: "fsci_opt::{minimize_scalar_bounded, minimize_trisection} property test".into(),
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
            eprintln!("{} mismatch: {} abs_diff={}", d.op, d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "scalar_b1 conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    ledger.finish(probes.len());
}
