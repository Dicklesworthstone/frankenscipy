//! Audit-ledger helpers for optimization hardened rejection paths.

pub use fsci_runtime::SyncSharedAuditLedger;
use fsci_runtime::{AlienArtifactDecision, AuditAction, AuditEvent, AuditLedger, casp_now_unix_ms};

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

/// Record a fail-closed audit event. `fingerprint` is the call's
/// [`fsci_runtime::Fingerprinter`] digest over every input and option
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

/// Record a Spec §6 decision event. `fingerprint` is the call's
/// [`fsci_runtime::Fingerprinter`] digest.
pub fn record_alien_artifact_decision(
    ledger: &SyncSharedAuditLedger,
    fingerprint: &str,
    decision: AlienArtifactDecision,
    outcome: &str,
) {
    lock_or_recover(ledger).record_alien_artifact_decision(
        casp_now_unix_ms(),
        fingerprint,
        decision,
        outcome.to_string(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsci_runtime::{
        DecisionEvidenceEntry, DecisionSignals, PolicyAction, RiskState, RuntimeMode,
    };

    #[test]
    fn record_alien_artifact_decision_writes_structured_audit_event() {
        let entry = DecisionEvidenceEntry {
            mode: RuntimeMode::Strict,
            signals: DecisionSignals::new(12.0, 0.0, 0.1),
            logits: [0.0, 1.0, -1.0],
            posterior: [0.2, 0.7, 0.1],
            expected_losses: [35.5, 9.3, 25.6],
            action: PolicyAction::FullValidate,
            top_state: RiskState::IllConditioned,
            reason: String::from("opt audit structured decision"),
        };
        let ledger = sync_audit_ledger();

        record_alien_artifact_decision(
            &ledger,
            "optimize-input",
            entry.alien_artifact_decision(),
            "validated",
        );

        let guard = ledger.lock().expect("audit ledger lock");
        assert_eq!(guard.len(), 1);
        match &guard.entries()[0].action {
            AuditAction::AlienArtifactDecision { decision } => {
                assert_eq!(decision.state, RiskState::IllConditioned);
                assert_eq!(decision.action, PolicyAction::FullValidate);
                assert!(decision.loss_matrix[0][2] > decision.loss_matrix[0][0]);
            }
            other => unreachable!("expected structured decision event, got {other:?}"),
        }
    }
}
