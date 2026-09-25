#![forbid(unsafe_code)]
//! Verify arange_with_audit and from_slice_with_audit record fail-closed
//! ledger entries on error paths and remain silent on success paths.
//!
//! Resolves [frankenscipy-cq79a]. Success paths must not append to the
//! ledger. Error paths (zero step / non-finite step for arange,
//! overflow shape / value-length mismatch for from_slice) must append a
//! FailClosed entry tagged with operation::ErrorKind. The wrapped
//! non-audit functions still return the same Err result.

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_arrayapi::{
    ArangeRequest, ArrayApiErrorKind, CreationRequest, DType, ExecutionMode, MemoryOrder,
    ScalarValue, Shape, arange_with_audit, backend::CoreArrayBackend, from_slice_with_audit,
    sync_audit_ledger,
};
use fsci_runtime::{AuditAction, Fingerprinter};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    pass: bool,
    note: String,
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
    fs::create_dir_all(output_dir()).expect("create audit_creation diff dir");
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

fn ledger_snapshot(ledger: &fsci_arrayapi::SyncSharedAuditLedger) -> (usize, Vec<String>) {
    let g = ledger.lock().expect("acquire ledger");
    let reasons: Vec<String> = g
        .entries()
        .iter()
        .map(|e| match &e.action {
            AuditAction::FailClosed { reason } => reason.clone(),
            other => format!("{other:?}"),
        })
        .collect();
    (g.len(), reasons)
}

#[test]
fn diff_arrayapi_creation_audit_record() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let backend = CoreArrayBackend::new(ExecutionMode::Strict);

    // === arange_with_audit success path: empty ledger before & after ===
    {
        let ledger = sync_audit_ledger();
        let pre = ledger_snapshot(&ledger).0;
        let req = ArangeRequest {
            start: ScalarValue::F64(0.0),
            stop: ScalarValue::F64(3.0),
            step: ScalarValue::F64(1.0),
            dtype: Some(DType::Float64),
        };
        let res = arange_with_audit(&backend, &req, &ledger);
        let (post, _reasons) = ledger_snapshot(&ledger);
        diffs.push(CaseDiff {
            case_id: "arange_success_ok".into(),
            pass: res.is_ok(),
            note: format!("ok={}", res.is_ok()),
        });
        diffs.push(CaseDiff {
            case_id: "arange_success_no_ledger".into(),
            pass: pre == 0 && post == 0,
            note: format!("pre={pre} post={post}"),
        });
    }

    // === arange_with_audit zero-step → InvalidStep, ledger += 1 ===
    {
        let ledger = sync_audit_ledger();
        let req = ArangeRequest {
            start: ScalarValue::F64(0.0),
            stop: ScalarValue::F64(3.0),
            step: ScalarValue::F64(0.0),
            dtype: Some(DType::Float64),
        };
        let res = arange_with_audit(&backend, &req, &ledger);
        let (post, reasons) = ledger_snapshot(&ledger);
        let kind_matches = res
            .as_ref()
            .err()
            .is_some_and(|e| e.kind == ArrayApiErrorKind::InvalidStep);
        let reason_matches = reasons.first().is_some_and(|r| r == "arange::InvalidStep");
        diffs.push(CaseDiff {
            case_id: "arange_zero_step_err".into(),
            pass: kind_matches,
            note: format!("err={:?}", res.err()),
        });
        diffs.push(CaseDiff {
            case_id: "arange_zero_step_ledger".into(),
            pass: post == 1 && reason_matches,
            note: format!("post={post} reasons={reasons:?}"),
        });
    }

    // === arange_with_audit NaN-step → InvalidStep, ledger += 1 ===
    {
        let ledger = sync_audit_ledger();
        let req = ArangeRequest {
            start: ScalarValue::F64(0.0),
            stop: ScalarValue::F64(3.0),
            step: ScalarValue::F64(f64::NAN),
            dtype: Some(DType::Float64),
        };
        let res = arange_with_audit(&backend, &req, &ledger);
        let (post, reasons) = ledger_snapshot(&ledger);
        diffs.push(CaseDiff {
            case_id: "arange_nan_step_err".into(),
            pass: res.is_err(),
            note: format!("err={:?}", res.err()),
        });
        diffs.push(CaseDiff {
            case_id: "arange_nan_step_ledger".into(),
            pass: post == 1 && reasons.first().is_some_and(|r| r == "arange::InvalidStep"),
            note: format!("post={post} reasons={reasons:?}"),
        });
    }

    // === from_slice_with_audit success path: empty ledger ===
    {
        let ledger = sync_audit_ledger();
        let req = CreationRequest {
            shape: Shape::new(vec![2, 2]),
            dtype: DType::Float64,
            order: MemoryOrder::C,
        };
        let values = vec![
            ScalarValue::F64(1.0),
            ScalarValue::F64(2.0),
            ScalarValue::F64(3.0),
            ScalarValue::F64(4.0),
        ];
        let res = from_slice_with_audit(&backend, &values, &req, &ledger);
        let (post, _reasons) = ledger_snapshot(&ledger);
        diffs.push(CaseDiff {
            case_id: "from_slice_success_ok".into(),
            pass: res.is_ok(),
            note: format!("ok={}", res.is_ok()),
        });
        diffs.push(CaseDiff {
            case_id: "from_slice_success_no_ledger".into(),
            pass: post == 0,
            note: format!("post={post}"),
        });
    }

    // === from_slice_with_audit overflow shape → Overflow, ledger += 1 ===
    {
        let ledger = sync_audit_ledger();
        let req = CreationRequest {
            shape: Shape::new(vec![usize::MAX, 2]),
            dtype: DType::Float64,
            order: MemoryOrder::C,
        };
        let values = vec![ScalarValue::F64(1.0)];
        let res = from_slice_with_audit(&backend, &values, &req, &ledger);
        let (post, reasons) = ledger_snapshot(&ledger);
        let kind_matches = res
            .as_ref()
            .err()
            .is_some_and(|e| e.kind == ArrayApiErrorKind::Overflow);
        diffs.push(CaseDiff {
            case_id: "from_slice_overflow_err".into(),
            pass: kind_matches,
            note: format!("err={:?}", res.err()),
        });
        diffs.push(CaseDiff {
            case_id: "from_slice_overflow_ledger".into(),
            pass: post == 1 && reasons.first().is_some_and(|r| r == "from_slice::Overflow"),
            note: format!("post={post} reasons={reasons:?}"),
        });
    }

    // === from_slice_with_audit value-length mismatch → InvalidShape, ledger += 1 ===
    {
        let ledger = sync_audit_ledger();
        let req = CreationRequest {
            shape: Shape::new(vec![3]),
            dtype: DType::Float64,
            order: MemoryOrder::C,
        };
        // values_len = 2 but shape needs 3
        let values = vec![ScalarValue::F64(1.0), ScalarValue::F64(2.0)];
        let res = from_slice_with_audit(&backend, &values, &req, &ledger);
        let (post, reasons) = ledger_snapshot(&ledger);
        let kind_matches = res
            .as_ref()
            .err()
            .is_some_and(|e| e.kind == ArrayApiErrorKind::InvalidShape);
        diffs.push(CaseDiff {
            case_id: "from_slice_len_mismatch_err".into(),
            pass: kind_matches,
            note: format!("err={:?}", res.err()),
        });
        diffs.push(CaseDiff {
            case_id: "from_slice_len_mismatch_ledger".into(),
            pass: post == 1
                && reasons
                    .first()
                    .is_some_and(|r| r == "from_slice::InvalidShape"),
            note: format!("post={post} reasons={reasons:?}"),
        });
    }

    // === Ledger entries carry the canonical input fingerprint (frankenscipy-3cu8u.1) ===
    // `fsci_arrayapi::arange`'s records, rebuilt from the documented encoding: the backend's
    // namespace and mode, then `start`, `stop`, `step` (variant name, value), then `dtype`.
    {
        let ledger = sync_audit_ledger();
        let req = ArangeRequest {
            start: ScalarValue::F64(0.0),
            stop: ScalarValue::F64(3.0),
            step: ScalarValue::F64(0.0),
            dtype: Some(DType::Float64),
        };
        let _ = arange_with_audit(&backend, &req, &ledger);
        let g = ledger.lock().expect("acquire");
        let entry = &g.entries()[0];
        let mut expected = Fingerprinter::new("fsci_arrayapi::arange");
        expected.str("array_api").str("Strict");
        for value in [0.0, 3.0, 0.0] {
            expected.str("F64").f64(value);
        }
        expected.str("Some(Float64)");
        let expected_fp = expected.finish();
        diffs.push(CaseDiff {
            case_id: "arange_fingerprint_matches".into(),
            pass: entry.input_fingerprint == expected_fp,
            note: format!("got={} expected={}", entry.input_fingerprint, expected_fp),
        });
        diffs.push(CaseDiff {
            case_id: "arange_outcome_rejected".into(),
            pass: entry.outcome == "rejected",
            note: format!("outcome={}", entry.outcome),
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_arrayapi_creation_audit_record".into(),
        category: "arange_with_audit + from_slice_with_audit ledger semantics".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("audit_creation mismatch: {} — {}", d.case_id, d.note);
        }
    }

    assert!(
        all_pass,
        "arange/from_slice audit coverage failed: {} cases",
        diffs.len(),
    );
}
