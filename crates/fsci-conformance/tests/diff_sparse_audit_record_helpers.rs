#![forbid(unsafe_code)]
//! Verify fsci_sparse::{record_fail_closed, record_bounded_recovery}
//! actually append events to the shared audit ledger.
//!
//! Resolves [frankenscipy-09uac]. Locks the ledger via the same
//! poison-recovering path the wrappers use, counts events before and
//! after each call.

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_sparse::{record_bounded_recovery, record_fail_closed, sync_audit_ledger};
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
    fs::create_dir_all(output_dir()).expect("create audit_rec diff dir");
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

fn ledger_len(ledger: &fsci_sparse::SyncSharedAuditLedger) -> usize {
    let g = ledger.lock().expect("ledger lock");
    g.len()
}

#[test]
fn diff_sparse_audit_record_helpers() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();

    let ledger = sync_audit_ledger();
    assert_eq!(ledger_len(&ledger), 0, "fresh ledger should be empty");

    // record_fail_closed adds 1
    let before = ledger_len(&ledger);
    record_fail_closed(&ledger, "input-A", "test_reason_A", "rejected");
    let after = ledger_len(&ledger);
    diffs.push(CaseDiff {
        case_id: "fail_closed_appends_one".into(),
        expected_delta: 1,
        actual_delta: after - before,
        pass: after - before == 1,
    });

    // record_bounded_recovery adds 1 more
    let before = ledger_len(&ledger);
    record_bounded_recovery(&ledger, "input-B", "restart_quad", "succeeded");
    let after = ledger_len(&ledger);
    diffs.push(CaseDiff {
        case_id: "bounded_recovery_appends_one".into(),
        expected_delta: 1,
        actual_delta: after - before,
        pass: after - before == 1,
    });

    // Multiple calls accumulate
    let before = ledger_len(&ledger);
    for i in 0..5 {
        record_fail_closed(&ledger, &format!("input-{i}"), "loop_reason", "rejected");
    }
    let after = ledger_len(&ledger);
    diffs.push(CaseDiff {
        case_id: "five_calls_append_five".into(),
        expected_delta: 5,
        actual_delta: after - before,
        pass: after - before == 5,
    });

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_sparse_audit_record_helpers".into(),
        category: "fsci_sparse::{record_fail_closed, record_bounded_recovery} append behavior"
            .into(),
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
                "audit_rec mismatch: {} expected_delta={} actual_delta={}",
                d.case_id, d.expected_delta, d.actual_delta
            );
        }
    }

    assert!(
        all_pass,
        "audit_record_helpers coverage failed: {} cases",
        diffs.len(),
    );
}
