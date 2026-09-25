#![forbid(unsafe_code)]
//! Verify fsci_arrayapi::record_fail_closed appends events to the
//! shared audit ledger.
//!
//! Resolves [frankenscipy-2arqi].

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_arrayapi::{record_fail_closed, sync_audit_ledger};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    expected_delta: usize,
    actual_delta: usize,
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
    fs::create_dir_all(output_dir()).expect("create arrayapi_audit diff dir");
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

fn ledger_len(ledger: &fsci_arrayapi::SyncSharedAuditLedger) -> usize {
    let g = ledger.lock().expect("ledger lock");
    g.len()
}

#[test]
fn diff_arrayapi_audit_record_fail_closed() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let ledger = sync_audit_ledger();

    let before = ledger_len(&ledger);
    record_fail_closed(
        &ledger,
        "input-arrayapi",
        "broadcast_incompatible",
        "rejected",
    );
    let after = ledger_len(&ledger);
    diffs.push(CaseDiff {
        case_id: "single_record_appends".into(),
        expected_delta: 1,
        actual_delta: after - before,
        pass: after - before == 1,
    });

    // Five calls accumulate five events
    let before = ledger_len(&ledger);
    for i in 0..5 {
        record_fail_closed(&ledger, &format!("input-{i}"), "loop_reason", "rejected");
    }
    let after = ledger_len(&ledger);
    diffs.push(CaseDiff {
        case_id: "five_records_accumulate".into(),
        expected_delta: 5,
        actual_delta: after - before,
        pass: after - before == 5,
    });

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_arrayapi_audit_record_fail_closed".into(),
        category: "fsci_arrayapi::record_fail_closed append behavior".into(),
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
                "arrayapi_audit mismatch: {} expected_delta={} actual_delta={}",
                d.case_id, d.expected_delta, d.actual_delta
            );
        }
    }

    assert!(
        all_pass,
        "arrayapi audit_record coverage failed: {} cases",
        diffs.len(),
    );
}
