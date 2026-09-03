#![forbid(unsafe_code)]
//! Verify fsci_arrayapi::default_float_dtype returns Float64 for both
//! Strict and Hardened execution modes.
//!
//! Resolves [frankenscipy-zqyg0]. Trivial spec lookup — guard against
//! silent default-precision drift.

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_arrayapi::{DType, ExecutionMode, default_float_dtype};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
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
    fs::create_dir_all(output_dir()).expect("create dft diff dir");
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
fn diff_arrayapi_default_float_dtype() {
    let start = Instant::now();
    let diffs: Vec<CaseDiff> = vec![
        CaseDiff {
            case_id: "strict_float64".into(),
            pass: default_float_dtype(ExecutionMode::Strict) == DType::Float64,
        },
        CaseDiff {
            case_id: "hardened_float64".into(),
            pass: default_float_dtype(ExecutionMode::Hardened) == DType::Float64,
        },
    ];

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_arrayapi_default_float_dtype".into(),
        category: "fsci_arrayapi::default_float_dtype spec lookup".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("dft mismatch: {}", d.case_id);
        }
    }

    assert!(
        all_pass,
        "default_float_dtype coverage failed: {} cases",
        diffs.len(),
    );
}
