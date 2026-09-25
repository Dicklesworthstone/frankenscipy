//! Audit-ledger scaffolding for fsci-special (br-egba-2).
//!
//! Mirrors the workspace audit pattern: a `SyncSharedAuditLedger`
//! re-export, a `sync_audit_ledger()` factory, and `record_*`
//! helpers that crate-internal functions call at rejection / recovery
//! sites in Hardened mode.
//!
//! At the time of this slice, only a small subset of functions have
//! `_with_audit` variants wired up — `gamma` is the demonstrator. The
//! remaining 340+ public functions inherit a silent path until
//! dedicated wiring lands. `SyncSharedAuditLedger` comes from
//! fsci-runtime so a single ledger can be threaded across crate
//! boundaries.

pub use fsci_runtime::SyncSharedAuditLedger;
use fsci_runtime::{AuditAction, AuditEvent, AuditLedger, Fingerprinter, casp_now_unix_ms};

use crate::types::SpecialTensor;

/// Create a new shared audit ledger for synchronous contexts.
#[must_use]
pub fn sync_audit_ledger() -> SyncSharedAuditLedger {
    AuditLedger::shared()
}

/// Acquire the ledger guard, recovering from a poisoned mutex.
///
/// The previous `if let Ok(mut ledger) = ledger.lock()` pattern
/// silently dropped audit events when any prior thread panicked while
/// holding the guard, violating the fail-closed audit invariant.
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

/// Feed a [`SpecialTensor`] to an audit fingerprint in full: its variant name, then every
/// value (`Empty` has none), so two tensors share a fingerprint only if they are the same
/// variant with the same values bit for bit (frankenscipy-3cu8u.1).
pub(crate) fn fingerprint_tensor(fingerprinter: &mut Fingerprinter, tensor: &SpecialTensor) {
    match tensor {
        SpecialTensor::Empty => {
            fingerprinter.str("Empty");
        }
        SpecialTensor::RealScalar(value) => {
            fingerprinter.str("RealScalar").f64(*value);
        }
        SpecialTensor::ComplexScalar(value) => {
            fingerprinter
                .str("ComplexScalar")
                .complex(&[(value.re, value.im)]);
        }
        SpecialTensor::RealVec(values) => {
            fingerprinter.str("RealVec").f64s(values);
        }
        SpecialTensor::ComplexVec(values) => {
            let pairs: Vec<(f64, f64)> = values.iter().map(|z| (z.re, z.im)).collect();
            fingerprinter.str("ComplexVec").complex(&pairs);
        }
    }
}

/// Record a fail-closed audit event when Hardened mode rejects input.
///
/// `fingerprint` is the call's [`Fingerprinter`] digest over every input and mode
/// (frankenscipy-3cu8u.1); it is stored as is.
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

/// Record a bounded-recovery audit event when Hardened mode falls
/// back to a degraded but safe computation path. `fingerprint` is the call's
/// [`Fingerprinter`] digest.
pub fn record_bounded_recovery(
    ledger: &SyncSharedAuditLedger,
    fingerprint: &str,
    recovery_action: &str,
    outcome: &str,
) {
    let event = AuditEvent::new(
        casp_now_unix_ms(),
        fingerprint,
        AuditAction::BoundedRecovery {
            recovery_action: recovery_action.to_string(),
        },
        outcome.to_string(),
    );
    lock_or_recover(ledger).record(event);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_after_poison_still_lands_in_ledger() {
        // /mock-code-finder regression for [frankenscipy-h5hj3]:
        // lock_or_recover is duplicated across fsci-stats, fsci-special,
        // fsci-arrayapi, and fsci-opt audit.rs modules. fsci-stats has
        // a direct regression test; this test mirrors it for fsci-special
        // so a peer agent can't silently revert to the
        // 'if let Ok(mut g) = lock()' silent-drop pattern.
        let ledger = sync_audit_ledger();

        let poisoned_thread = {
            let l = ledger.clone();
            std::thread::spawn(move || {
                let _g = l.lock().expect("acquire");
                panic!("poison fsci-special audit ledger on purpose");
            })
            .join()
        };
        assert!(poisoned_thread.is_err(), "thread should have panicked");
        assert!(
            ledger.lock().is_err(),
            "ledger must be poisoned after panic"
        );

        record_fail_closed(&ledger, "x=NaN", "non_finite_input", "rejected");
        record_bounded_recovery(&ledger, "x=Inf", "clamp_to_max", "recovered");

        let g = ledger.lock().expect("ledger should recover after poison");
        assert_eq!(g.len(), 2, "both audit events must be recorded");
        let kinds: Vec<_> = g
            .entries()
            .iter()
            .map(|e| match &e.action {
                fsci_runtime::AuditAction::FailClosed { reason } => format!("FC:{reason}"),
                fsci_runtime::AuditAction::BoundedRecovery { recovery_action } => {
                    format!("BR:{recovery_action}")
                }
                _ => "OTHER".to_string(),
            })
            .collect();
        assert!(kinds.contains(&"FC:non_finite_input".to_string()));
        assert!(kinds.contains(&"BR:clamp_to_max".to_string()));
    }
}
