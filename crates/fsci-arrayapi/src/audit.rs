//! Audit-ledger helpers for Array API hardened rejection paths.

pub use fsci_runtime::SyncSharedAuditLedger;
use fsci_runtime::{AuditAction, AuditEvent, AuditLedger, Fingerprinter, casp_now_unix_ms};

use crate::types::{IndexExpr, ScalarValue};

#[must_use]
pub fn sync_audit_ledger() -> SyncSharedAuditLedger {
    AuditLedger::shared()
}

/// Acquire the ledger guard, recovering from a poisoned mutex so
/// audit events still record after any prior thread panicked.
/// Resolves [frankenscipy-kt4od].
fn lock_or_recover(ledger: &SyncSharedAuditLedger) -> std::sync::MutexGuard<'_, AuditLedger> {
    match ledger.lock() {
        Ok(g) => g,
        Err(poisoned) => {
            ledger.clear_poison();
            poisoned.into_inner()
        }
    }
}

/// Feed one scalar to an audit fingerprint: its variant name, then its value (`f64`s by bits,
/// so `-0.0` and NaN payloads are distinguished; a complex value as `re` then `im`)
/// (frankenscipy-3cu8u.1).
pub fn fingerprint_scalar(fingerprinter: &mut Fingerprinter, value: ScalarValue) {
    match value {
        ScalarValue::Bool(value) => {
            fingerprinter.str("Bool").bool(value);
        }
        ScalarValue::I64(value) => {
            fingerprinter.str("I64").i64(value);
        }
        ScalarValue::U64(value) => {
            fingerprinter.str("U64").u64(value);
        }
        ScalarValue::F64(value) => {
            fingerprinter.str("F64").f64(value);
        }
        ScalarValue::ComplexF64 { re, im } => {
            fingerprinter.str("ComplexF64").f64(re).f64(im);
        }
    }
}

/// Feed a scalar list to an audit fingerprint: its length, then each scalar as
/// [`fingerprint_scalar`] does. [`crate::ArrayApiArray::fingerprint_into`] implementations
/// that store `ScalarValue`s feed their elements with this.
pub fn fingerprint_scalars(fingerprinter: &mut Fingerprinter, values: &[ScalarValue]) {
    fingerprinter.usize(values.len());
    for &value in values {
        fingerprint_scalar(fingerprinter, value);
    }
}

/// Feed an index expression to an audit fingerprint in full: its variant name, then every
/// slice (`start`/`stop` as a presence flag then the value, then `step`), every advanced index
/// list, or the mask shape.
pub(crate) fn fingerprint_index(fingerprinter: &mut Fingerprinter, index: &IndexExpr) {
    match index {
        IndexExpr::Basic { slices } => {
            fingerprinter.str("Basic").usize(slices.len());
            for slice in slices {
                for bound in [slice.start, slice.stop] {
                    fingerprinter.bool(bound.is_some());
                    if let Some(bound) = bound {
                        fingerprinter.i64(bound as i64);
                    }
                }
                fingerprinter.i64(slice.step as i64);
            }
        }
        IndexExpr::Advanced { indices } => {
            fingerprinter.str("Advanced").usize(indices.len());
            for axis in indices {
                fingerprinter.usize(axis.len());
                for &index in axis {
                    fingerprinter.i64(index as i64);
                }
            }
        }
        IndexExpr::BooleanMask { mask_shape } => {
            fingerprinter.str("BooleanMask").shape(&mask_shape.dims);
        }
    }
}

/// Record a fail-closed audit event. `fingerprint` is the call's [`Fingerprinter`] digest
/// over the backend configuration and every input (frankenscipy-3cu8u.1); it is stored as is.
pub fn record_fail_closed(
    ledger: &SyncSharedAuditLedger,
    fingerprint: &str,
    reason: &str,
    outcome: &str,
) {
    let event = AuditEvent::new(
        casp_now_unix_ms(),
        fingerprint,
        AuditAction::FailClosed {
            reason: reason.to_string(),
        },
        outcome.to_string(),
    );
    lock_or_recover(ledger).record(event);
}

pub(crate) fn record_array_api_error(
    ledger: &SyncSharedAuditLedger,
    operation: &str,
    fingerprint: &str,
    kind: crate::error::ArrayApiErrorKind,
) {
    record_fail_closed(
        ledger,
        fingerprint,
        &format!("{operation}::{kind:?}"),
        "rejected",
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_after_poison_still_lands_in_ledger() {
        // /mock-code-finder regression for [frankenscipy-h5hj3]:
        // mirrors the fsci-stats poison-recovery test for fsci-arrayapi
        // so a peer agent cannot silently revert lock_or_recover to the
        // 'if let Ok(mut g) = lock()' silent-drop pattern in this crate.
        let ledger = sync_audit_ledger();

        let poisoned_thread = {
            let l = ledger.clone();
            std::thread::spawn(move || {
                let _g = l.lock().expect("acquire");
                std::panic::resume_unwind(Box::new("poison fsci-arrayapi audit ledger on purpose"));
            })
            .join()
        };
        assert!(poisoned_thread.is_err(), "thread should have panicked");
        assert!(
            ledger.lock().is_err(),
            "ledger must be poisoned after panic"
        );

        record_fail_closed(
            &ledger,
            "shape mismatch",
            "broadcast::Incompatible",
            "rejected",
        );

        let g = ledger.lock().expect("ledger should recover after poison");
        assert_eq!(g.len(), 1, "the audit event must be recorded");
        assert!(matches!(
            &g.entries()[0].action,
            fsci_runtime::AuditAction::FailClosed { reason } if reason == "broadcast::Incompatible"
        ));
    }
}
