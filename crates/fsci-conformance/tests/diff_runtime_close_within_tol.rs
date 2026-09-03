#![forbid(unsafe_code)]
//! Branch coverage for fsci_runtime::{close_within_tol, within_tolerance}.
//!
//! Resolves [frankenscipy-mbcmf]. Exercise NaN-NaN equality, Inf-Inf
//! equality, atol/rtol formula on finite values, mixed NaN cases,
//! opposite-sign Inf.

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::{close_within_tol, within_tolerance};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    expected: bool,
    actual: bool,
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
    fs::create_dir_all(output_dir()).expect("create runtime_tol diff dir");
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
fn diff_runtime_close_within_tol() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = vec![
        // 1. NaN, NaN → true
        CaseDiff {
            case_id: "nan_nan".into(),
            expected: true,
            actual: close_within_tol(f64::NAN, f64::NAN, 1e-8, 1e-8),
            pass: false,
        },
        // 2. +Inf, +Inf → true
        CaseDiff {
            case_id: "pos_inf_pos_inf".into(),
            expected: true,
            actual: close_within_tol(f64::INFINITY, f64::INFINITY, 1e-8, 1e-8),
            pass: false,
        },
        // 3. -Inf, -Inf → true
        CaseDiff {
            case_id: "neg_inf_neg_inf".into(),
            expected: true,
            actual: close_within_tol(f64::NEG_INFINITY, f64::NEG_INFINITY, 1e-8, 1e-8),
            pass: false,
        },
        // 4. +Inf, -Inf → false
        CaseDiff {
            case_id: "pos_inf_neg_inf".into(),
            expected: false,
            actual: close_within_tol(f64::INFINITY, f64::NEG_INFINITY, 1e-8, 1e-8),
            pass: false,
        },
        // 5. NaN, 0.0 → false
        CaseDiff {
            case_id: "nan_zero".into(),
            expected: false,
            actual: close_within_tol(f64::NAN, 0.0, 1e-8, 1e-8),
            pass: false,
        },
        // 6. Finite within tol → true
        CaseDiff {
            case_id: "finite_within".into(),
            expected: true,
            actual: close_within_tol(1.0 + 1e-9, 1.0, 1e-8, 1e-8),
            pass: false,
        },
        // 7. Finite outside tol → false
        CaseDiff {
            case_id: "finite_outside".into(),
            expected: false,
            actual: close_within_tol(2.0, 1.0, 1e-8, 1e-8),
            pass: false,
        },
        // 8. Pure rtol (zero atol, large rtol) → true
        CaseDiff {
            case_id: "rtol_only".into(),
            expected: true,
            actual: close_within_tol(1.1, 1.0, 0.0, 0.2),
            pass: false,
        },
        // 9. within_tolerance is an alias for close_within_tol
        CaseDiff {
            case_id: "alias_within".into(),
            expected: true,
            actual: within_tolerance(2.0, 2.0 + 1e-12, 1e-9, 1e-9),
            pass: false,
        },
    ];

    // Compute pass flag
    for d in &mut diffs {
        d.pass = d.expected == d.actual;
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_runtime_close_within_tol".into(),
        category: "fsci_runtime::{close_within_tol, within_tolerance} branch coverage".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "close_within_tol mismatch: {} got {} (expected {})",
                d.case_id, d.actual, d.expected
            );
        }
    }

    assert!(
        all_pass,
        "close_within_tol coverage failed: {} cases",
        diffs.len(),
    );
}
