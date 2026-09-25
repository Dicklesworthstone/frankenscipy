#![forbid(unsafe_code)]
//! Verify fsci_opt::{record_fail_closed, audit::record_alien_artifact_decision}
//! append events to the shared audit ledger.
//!
//! Resolves [frankenscipy-ckeph].

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_opt::record_fail_closed;
use fsci_opt::sync_audit_ledger;
use fsci_runtime::{DecisionEvidenceEntry, DecisionSignals, PolicyAction, RiskState, RuntimeMode};
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
    fs::create_dir_all(output_dir()).expect("create opt_audit diff dir");
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

fn ledger_len(ledger: &fsci_opt::SyncSharedAuditLedger) -> usize {
    let g = ledger.lock().expect("ledger lock");
    g.len()
}

#[test]
fn diff_opt_audit_record_helpers() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let ledger = sync_audit_ledger();

    // record_fail_closed
    let before = ledger_len(&ledger);
    record_fail_closed(&ledger, "opt-input-A", "non_finite_x0", "rejected");
    let after = ledger_len(&ledger);
    diffs.push(CaseDiff {
        case_id: "fail_closed_appends".into(),
        expected_delta: 1,
        actual_delta: after - before,
        pass: after - before == 1,
    });

    // record_alien_artifact_decision (constructed via DecisionEvidenceEntry)
    let entry = DecisionEvidenceEntry {
        mode: RuntimeMode::Strict,
        signals: DecisionSignals::new(12.0, 0.0, 0.1),
        logits: [0.0, 1.0, -1.0],
        posterior: [0.2, 0.7, 0.1],
        expected_losses: [35.5, 9.3, 25.6],
        action: PolicyAction::FullValidate,
        top_state: RiskState::IllConditioned,
        reason: String::from("opt audit unit test"),
    };
    let decision = entry.alien_artifact_decision();
    let before = ledger_len(&ledger);
    fsci_opt::audit::record_alien_artifact_decision(&ledger, "opt-input-B", decision, "validated");
    let after = ledger_len(&ledger);
    diffs.push(CaseDiff {
        case_id: "alien_decision_appends".into(),
        expected_delta: 1,
        actual_delta: after - before,
        pass: after - before == 1,
    });

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_audit_record_helpers".into(),
        category:
            "fsci_opt::{record_fail_closed, audit::record_alien_artifact_decision} append behavior"
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
                "opt_audit mismatch: {} expected_delta={} actual_delta={}",
                d.case_id, d.expected_delta, d.actual_delta
            );
        }
    }

    assert!(
        all_pass,
        "opt audit_record coverage failed: {} cases",
        diffs.len(),
    );
}
