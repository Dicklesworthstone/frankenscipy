#![no_main]

use arbitrary::Arbitrary;
use fsci_cluster::kmeans_with_mode;
use fsci_linalg::{RecoveryPolicy, SolveOptions, solve_with_audit};
use fsci_runtime::{
    AuditAction, AuditLedger, RuntimeMode, SharedAuditLedger, SolverPortfolio,
    SyncSharedAuditLedger,
};
use fsci_signal::{SignalOptions, czt_with_mode_and_audit, lfilter_with_options, sync_audit_ledger};
use fsci_spatial::KDTree;
use libfuzzer_sys::fuzz_target;

const MAX_DIM: usize = 8;

#[derive(Debug, Clone, Arbitrary)]
struct SymmetryInput {
    routine_selector: u8,
    recovery_code: u8,
    rows: u8,
    cols: u8,
    k_clusters: u8,
    max_iter: u8,
    czt_m: u8,
    seed: u64,
    values: Vec<f64>,
    rhs: Vec<f64>,
}

fn decode_recovery(code: u8) -> RecoveryPolicy {
    match code % 4 {
        0 => RecoveryPolicy::Reject,
        1 => RecoveryPolicy::ProjectNonFinite,
        2 => RecoveryPolicy::Regularize,
        _ => RecoveryPolicy::ProjectAndRegularize,
    }
}

fn count_shared_actions(ledger: &SharedAuditLedger) -> (usize, usize) {
    let guard = ledger.lock().expect("shared audit ledger lock");
    let fail_closed = guard
        .entries()
        .iter()
        .filter(|e| matches!(e.action, AuditAction::FailClosed { .. }))
        .count();
    let bounded_recovery = guard
        .entries()
        .iter()
        .filter(|e| matches!(e.action, AuditAction::BoundedRecovery { .. }))
        .count();
    (fail_closed, bounded_recovery)
}

fn count_sync_actions(ledger: &SyncSharedAuditLedger) -> (usize, usize) {
    let guard = ledger.lock().expect("sync audit ledger lock");
    let fail_closed = guard
        .entries()
        .iter()
        .filter(|e| matches!(e.action, AuditAction::FailClosed { .. }))
        .count();
    let bounded_recovery = guard
        .entries()
        .iter()
        .filter(|e| matches!(e.action, AuditAction::BoundedRecovery { .. }))
        .count();
    (fail_closed, bounded_recovery)
}

fn build_matrix(rows: usize, cols: usize, values: &[f64]) -> Vec<Vec<f64>> {
    let mut mat = vec![vec![0.0_f64; cols]; rows];
    for (idx, &v) in values.iter().take(rows.saturating_mul(cols)).enumerate() {
        if cols > 0 {
            mat[idx / cols][idx % cols] = v;
        }
    }
    mat
}

fn build_vector(len: usize, values: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0_f64; len];
    for (idx, &v) in values.iter().take(len).enumerate() {
        out[idx] = v;
    }
    out
}

fn bits_slice(xs: &[f64]) -> Vec<u64> {
    xs.iter().map(|v| v.to_bits()).collect()
}

fuzz_target!(|input: SymmetryInput| {
    let recovery = decode_recovery(input.recovery_code);
    let rows = usize::from(input.rows) % (MAX_DIM + 1);
    let cols = usize::from(input.cols) % (MAX_DIM + 1);

    match input.routine_selector % 5 {
        0 => {
            // 1. fsci_linalg::solve_with_audit Strict vs Hardened
            let a = build_matrix(rows, cols, &input.values);
            let b = build_vector(rows, &input.rhs);

            let ledger_strict = AuditLedger::shared();
            let mut port_strict = SolverPortfolio::new(RuntimeMode::Strict, 32);
            let res_strict = solve_with_audit(
                &a,
                &b,
                SolveOptions {
                    mode: RuntimeMode::Strict,
                    recovery,
                    ..SolveOptions::default()
                },
                &mut port_strict,
                &ledger_strict,
            );
            let (fc_strict, br_strict) = count_shared_actions(&ledger_strict);
            assert_eq!(
                br_strict, 0,
                "solve Strict emitted BoundedRecovery on rows={rows}, cols={cols}"
            );
            if res_strict.is_err() {
                assert!(
                    fc_strict >= 1,
                    "solve Strict returned Err without FailClosed: rows={rows}, cols={cols}"
                );
            }

            let ledger_hard = AuditLedger::shared();
            let mut port_hard = SolverPortfolio::new(RuntimeMode::Hardened, 32);
            let res_hard = solve_with_audit(
                &a,
                &b,
                SolveOptions {
                    mode: RuntimeMode::Hardened,
                    recovery,
                    ..SolveOptions::default()
                },
                &mut port_hard,
                &ledger_hard,
            );
            let (fc_hard, br_hard) = count_shared_actions(&ledger_hard);
            if res_hard.is_err() {
                assert!(
                    fc_hard >= 1,
                    "solve Hardened returned Err without FailClosed: rows={rows}, cols={cols}, recovery={recovery:?}"
                );
            }

            if let Ok(ok_hard) = &res_hard {
                assert_eq!(
                    ok_hard.recovered,
                    br_hard > 0,
                    "solve Hardened recovered={} but BoundedRecovery count={br_hard}",
                    ok_hard.recovered
                );
                if let Ok(ok_strict) = &res_strict {
                    let same_bits = bits_slice(&ok_strict.x) == bits_slice(&ok_hard.x);
                    if !ok_hard.recovered {
                        assert!(
                            same_bits,
                            "solve Strict vs Hardened differed with recovered=false: rows={rows}, cols={cols}"
                        );
                    }
                    if !same_bits {
                        assert!(
                            br_hard > 0 && ok_hard.recovered,
                            "solve Strict vs Hardened differed without BoundedRecovery event"
                        );
                    }
                }
            }
        }
        1 => {
            // 2. fsci_cluster::kmeans_with_mode (covers W5.2 Strict FailClosed symmetry)
            let data = build_matrix(rows, cols, &input.values);
            let k = usize::from(input.k_clusters) % (MAX_DIM + 2);
            let max_iter = usize::from(input.max_iter) % 16;

            let ledger_strict = sync_audit_ledger();
            let res_strict = kmeans_with_mode(
                &data,
                k,
                max_iter,
                input.seed,
                RuntimeMode::Strict,
                Some(&ledger_strict),
            );
            let (fc_strict, br_strict) = count_sync_actions(&ledger_strict);
            assert_eq!(br_strict, 0, "kmeans Strict emitted BoundedRecovery");

            let ledger_hard = sync_audit_ledger();
            let res_hard = kmeans_with_mode(
                &data,
                k,
                max_iter,
                input.seed,
                RuntimeMode::Hardened,
                Some(&ledger_hard),
            );
            let (fc_hard, br_hard) = count_sync_actions(&ledger_hard);
            assert_eq!(br_hard, 0, "kmeans Hardened emitted BoundedRecovery");

            match (&res_strict, &res_hard) {
                (Err(e_s), _) => {
                    assert_eq!(
                        fc_strict, 1,
                        "kmeans Strict Err({e_s:?}) must record exactly 1 FailClosed, got {fc_strict} (rows={rows}, cols={cols}, k={k}, max_iter={max_iter})"
                    );
                }
                (Ok(_), _) => {
                    assert_eq!(fc_strict, 0, "kmeans Strict Ok recorded FailClosed");
                }
            }
            match &res_hard {
                Err(e_h) => {
                    assert_eq!(
                        fc_hard, 1,
                        "kmeans Hardened Err({e_h:?}) must record exactly 1 FailClosed, got {fc_hard}"
                    );
                }
                Ok(_) => {
                    assert_eq!(fc_hard, 0, "kmeans Hardened Ok recorded FailClosed");
                }
            }
            if let (Ok(s), Ok(h)) = (&res_strict, &res_hard) {
                assert_eq!(
                    s.inertia.to_bits(),
                    h.inertia.to_bits(),
                    "kmeans Strict vs Hardened inertia differed without recovery"
                );
            }
        }
        2 => {
            // 3. fsci_spatial::KDTree::new_with_mode (covers W5.2 Strict FailClosed symmetry)
            let data = build_matrix(rows, cols, &input.values);

            let ledger_strict = sync_audit_ledger();
            let res_strict =
                KDTree::new_with_mode(&data, RuntimeMode::Strict, Some(&ledger_strict));
            let (fc_strict, br_strict) = count_sync_actions(&ledger_strict);
            assert_eq!(br_strict, 0);

            let ledger_hard = sync_audit_ledger();
            let res_hard = KDTree::new_with_mode(&data, RuntimeMode::Hardened, Some(&ledger_hard));
            let (fc_hard, br_hard) = count_sync_actions(&ledger_hard);
            assert_eq!(br_hard, 0);

            if let Err(e_s) = &res_strict {
                assert_eq!(
                    fc_strict, 1,
                    "KDTree Strict Err({e_s:?}) must record exactly 1 FailClosed, got {fc_strict} (rows={rows}, cols={cols})"
                );
            } else {
                assert_eq!(fc_strict, 0);
            }

            if let Err(e_h) = &res_hard {
                assert_eq!(
                    fc_hard, 1,
                    "KDTree Hardened Err({e_h:?}) must record exactly 1 FailClosed, got {fc_hard} (rows={rows}, cols={cols})"
                );
            } else {
                assert_eq!(fc_hard, 0);
            }
        }
        3 => {
            // 4. fsci_signal::czt_with_mode_and_audit (covers W5.2 Strict FailClosed symmetry)
            let x = build_vector(rows, &input.values);
            let m = usize::from(input.czt_m) % (MAX_DIM + 2);
            let a_param = input.rhs.first().copied().map(|v| (v, 0.0));

            let ledger_strict = sync_audit_ledger();
            let res_strict = czt_with_mode_and_audit(
                &x,
                m,
                None,
                a_param,
                RuntimeMode::Strict,
                Some(&ledger_strict),
            );
            let (fc_strict, br_strict) = count_sync_actions(&ledger_strict);
            assert_eq!(br_strict, 0);

            let ledger_hard = sync_audit_ledger();
            let res_hard = czt_with_mode_and_audit(
                &x,
                m,
                None,
                a_param,
                RuntimeMode::Hardened,
                Some(&ledger_hard),
            );
            let (fc_hard, br_hard) = count_sync_actions(&ledger_hard);
            assert_eq!(br_hard, 0);

            if let Err(e_s) = &res_strict {
                assert_eq!(
                    fc_strict, 1,
                    "czt Strict Err({e_s:?}) must record exactly 1 FailClosed, got {fc_strict} (len={}, m={m})",
                    x.len()
                );
            } else {
                assert_eq!(fc_strict, 0);
            }

            if let Err(e_h) = &res_hard {
                assert_eq!(
                    fc_hard, 1,
                    "czt Hardened Err({e_h:?}) must record exactly 1 FailClosed, got {fc_hard} (len={}, m={m})",
                    x.len()
                );
            } else {
                assert_eq!(fc_hard, 0);
            }

            if let (Ok(s), Ok(h)) = (&res_strict, &res_hard) {
                assert_eq!(s.len(), h.len());
                for ((sr, si), (hr, hi)) in s.iter().zip(h.iter()) {
                    assert_eq!(sr.to_bits(), hr.to_bits());
                    assert_eq!(si.to_bits(), hi.to_bits());
                }
            }
        }
        _ => {
            // 5. fsci_signal::lfilter_with_options (Strict vs Hardened with opt-in non-finite projection)
            let b = [0.5_f64, 0.5_f64];
            let a = [1.0_f64, -0.25_f64];
            let x = build_vector(rows.max(1), &input.values);

            let ledger_strict = sync_audit_ledger();
            let res_strict = lfilter_with_options(
                &b,
                &a,
                &x,
                None,
                SignalOptions::new(RuntimeMode::Strict).with_recovery(recovery),
                Some(&ledger_strict),
            );

            let ledger_hard = sync_audit_ledger();
            let res_hard = lfilter_with_options(
                &b,
                &a,
                &x,
                None,
                SignalOptions::new(RuntimeMode::Hardened).with_recovery(recovery),
                Some(&ledger_hard),
            );
            let (fc_hard, br_hard) = count_sync_actions(&ledger_hard);

            if res_hard.is_err() {
                assert_eq!(
                    fc_hard, 1,
                    "lfilter Hardened Err must record 1 FailClosed, got {fc_hard}"
                );
            } else if let Ok(ok_hard) = &res_hard {
                assert_eq!(
                    ok_hard.recovered,
                    br_hard == 1,
                    "lfilter Hardened recovered={} but BoundedRecovery count={br_hard}",
                    ok_hard.recovered
                );
                if let Ok(ok_strict) = &res_strict {
                    let same_bits = bits_slice(&ok_strict.y) == bits_slice(&ok_hard.y);
                    if !ok_hard.recovered {
                        assert!(
                            same_bits,
                            "lfilter Strict vs Hardened differed with recovered=false"
                        );
                    }
                    if !same_bits {
                        assert!(
                            ok_hard.recovered && br_hard == 1,
                            "lfilter Strict vs Hardened differed without BoundedRecovery"
                        );
                    }
                }
            }
        }
    }
});
