//! Audit-ledger scaffolding for fsci-sparse (br-egba-4).
//!
//! Uses the canonical `fsci_runtime::SyncSharedAuditLedger` so a
//! single ledger can be threaded across crate boundaries. The crate
//! already has explicit Hardened-mode
//! validation in the format conversion routines (csr_to_csc_with_mode
//! and friends); `_with_audit` wrappers expose those rejections to a
//! forensic ledger without changing the existing error behavior.

pub use fsci_runtime::SyncSharedAuditLedger;
use fsci_runtime::{AuditAction, AuditEvent, AuditLedger, Fingerprinter, casp_now_unix_ms};

use crate::formats::CsrMatrix;

#[must_use]
pub fn sync_audit_ledger() -> SyncSharedAuditLedger {
    AuditLedger::shared()
}

/// Acquire the ledger guard, recovering from a poisoned mutex so audit
/// events still record after any prior thread panicked.
/// Resolves [frankenscipy-l2irg] for fsci-sparse.
fn lock_or_recover(ledger: &SyncSharedAuditLedger) -> std::sync::MutexGuard<'_, AuditLedger> {
    match ledger.lock() {
        Ok(g) => g,
        Err(poisoned) => {
            ledger.clear_poison();
            poisoned.into_inner()
        }
    }
}

/// Feed a CSR matrix to an audit fingerprint in full (frankenscipy-3cu8u.1): its shape
/// `[rows, cols]`, `data`, `indices`, `indptr`, then the canonical flags `sorted_indices` and
/// `deduplicated`. The index arrays go in as `shape` records, [`Fingerprinter`]'s
/// length-prefixed `usize` list.
pub(crate) fn fingerprint_csr(fingerprinter: &mut Fingerprinter, csr: &CsrMatrix) {
    let shape = csr.shape();
    let canonical = csr.canonical_meta();
    fingerprinter
        .shape(&[shape.rows, shape.cols])
        .f64s(csr.data())
        .shape(csr.indices())
        .shape(csr.indptr())
        .bool(canonical.sorted_indices)
        .bool(canonical.deduplicated);
}

/// Record a fail-closed audit event. `fingerprint` is the call's [`Fingerprinter`] digest over
/// every input and option (frankenscipy-3cu8u.1); it is stored as is.
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

/// Record a bounded-recovery audit event. `fingerprint` is the call's [`Fingerprinter`] digest.
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
        let ledger = sync_audit_ledger();

        let poisoned_thread = {
            let l = ledger.clone();
            std::thread::spawn(move || {
                let _g = l.lock().expect("acquire");
                std::panic::panic_any("poison fsci-sparse audit ledger on purpose");
            })
            .join()
        };
        assert!(poisoned_thread.is_err(), "thread should have panicked");
        assert!(
            ledger.lock().is_err(),
            "ledger must be poisoned after panic"
        );

        record_fail_closed(&ledger, "shape", "csr_to_csc::shape", "rejected");
        record_bounded_recovery(&ledger, "recover", "duplicate_indices_dedup", "recovered");

        let g = ledger.lock().expect("ledger should recover");
        assert_eq!(g.len(), 2);
    }

    /// frankenscipy-3cu8u.1: the audit fingerprints cover every input. `spsolve_with_audit`'s
    /// events used to carry constant labels (`b"non_finite"`, ...), the hardened CSR→CSC
    /// rejection hashed only the shape, and the CASP iterative solve hashed its rationale, so
    /// each differing pair below shared one fingerprint.
    #[test]
    fn audit_fingerprints_cover_every_input() {
        use crate::formats::Shape2D;
        use crate::linalg::{
            CaspIterativeSolveOptions, SolveOptions, casp_iterative_solve_with_audit,
            spsolve_with_audit,
        };
        use fsci_runtime::{RuntimeMode, SparseSolverPortfolio};

        fn only_fingerprint(ledger: &SyncSharedAuditLedger) -> String {
            let guard = lock_or_recover(ledger);
            assert_eq!(guard.len(), 1, "one call, one event");
            guard.entries()[0].input_fingerprint.clone()
        }
        // Symmetric, positive diagonal, row dominant: CASP picks CG.
        let spd = CsrMatrix::from_components(
            Shape2D::new(3, 3),
            vec![4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0],
            vec![0, 1, 0, 1, 2, 1, 2],
            vec![0, 2, 5, 7],
            true,
        )
        .expect("spd csr");

        // spsolve: non-finite rhs, rejected before any solve.
        let spsolve = |b: &[f64]| {
            let ledger = sync_audit_ledger();
            let mut portfolio = SparseSolverPortfolio::new(RuntimeMode::Strict, 16);
            let result =
                spsolve_with_audit(&spd, b, SolveOptions::default(), &mut portfolio, &ledger);
            assert!(result.is_err(), "non-finite rhs must be rejected");
            only_fingerprint(&ledger)
        };
        let base = spsolve(&[1.0, 2.0, f64::NAN]);
        assert!(base.starts_with("blake3:"), "{base}");
        assert_eq!(
            base,
            spsolve(&[1.0, 2.0, f64::NAN]),
            "same input, same fingerprint"
        );
        assert_ne!(base, spsolve(&[1.0, 3.0, f64::NAN]));

        // Hardened CSR→CSC: same shape and pattern, different values.
        let reject = |data: Vec<f64>| {
            let unsorted = CsrMatrix::from_components(
                Shape2D::new(2, 3),
                data,
                vec![2, 0, 1],
                vec![0, 2, 3],
                false,
            )
            .expect("non-canonical csr");
            let ledger = sync_audit_ledger();
            let result = crate::ops::csr_to_csc_with_mode_and_audit(
                &unsorted,
                RuntimeMode::Hardened,
                "audit-fingerprint",
                &ledger,
            );
            assert!(result.is_err(), "hardened rejects unsorted indices");
            only_fingerprint(&ledger)
        };
        let unsorted = reject(vec![1.0, 2.0, 3.0]);
        assert_eq!(unsorted, reject(vec![1.0, 2.0, 3.0]));
        assert_ne!(unsorted, reject(vec![1.0, 2.0, 4.0]));

        // CASP iterative solve: same routing and rationale, different rhs.
        let casp = |b: &[f64]| {
            let ledger = sync_audit_ledger();
            let solved = casp_iterative_solve_with_audit(
                &spd,
                b,
                None,
                CaspIterativeSolveOptions::default(),
                &ledger,
            )
            .expect("casp audited solve");
            (solved.decision.rationale, only_fingerprint(&ledger))
        };
        let (rationale, routed) = casp(&[5.0, 5.0, 3.0]);
        let (other_rationale, other_rhs) = casp(&[5.0, 5.0, 4.0]);
        assert_eq!(rationale, other_rationale, "both solves route the same way");
        assert_eq!(routed, casp(&[5.0, 5.0, 3.0]).1);
        assert_ne!(routed, other_rhs);
    }
}
