#![no_main]

use std::time::{Duration, Instant};

use arbitrary::Arbitrary;
use fsci_linalg::{RecoveryPolicy, SolveOptions, solve_with_audit};
use fsci_runtime::{AuditAction, AuditLedger, RuntimeMode, SolverAction, SolverPortfolio};
use libfuzzer_sys::fuzz_target;

const MAX_DIM: usize = 10;
const MAX_STEPS: usize = 32;
const MAX_CALL_DURATION: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, Arbitrary)]
struct FloodMatrixSpec {
    dim: u8,
    scale_exp: i8,
    hilbert_like: bool,
    inject_nonfinite: bool,
    recovery_code: u8,
    values: Vec<f64>,
    rhs: Vec<f64>,
}

#[derive(Debug, Clone, Arbitrary)]
struct RcondFloodInput {
    hardened: bool,
    capacity: u8,
    steps: Vec<FloodMatrixSpec>,
}

fn decode_recovery(code: u8) -> RecoveryPolicy {
    match code % 4 {
        0 => RecoveryPolicy::Reject,
        1 => RecoveryPolicy::ProjectNonFinite,
        2 => RecoveryPolicy::Regularize,
        _ => RecoveryPolicy::ProjectAndRegularize,
    }
}

fn build_flood_system(spec: &FloodMatrixSpec) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = 2 + (usize::from(spec.dim) % (MAX_DIM - 1));
    let exp = i32::from(spec.scale_exp).clamp(-18, -10);
    let scale = 10.0_f64.powi(exp);

    let mut a = vec![vec![0.0_f64; n]; n];
    if spec.hilbert_like {
        for (i, row) in a.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                let base = 1.0 / ((i + j + 1) as f64);
                let perturb = spec
                    .values
                    .get(i * n + j)
                    .copied()
                    .filter(|v| v.is_finite())
                    .map_or(0.0, |v| (v % 1.0) * scale);
                *cell = base + perturb;
            }
        }
    } else {
        for (i, row) in a.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                let raw = spec
                    .values
                    .get(i * n + j)
                    .copied()
                    .filter(|v| v.is_finite())
                    .map_or(0.0, |v| v.clamp(-10.0, 10.0));
                *cell = if i == j {
                    if i + 1 == n {
                        scale
                    } else {
                        (n as f64) + 2.0 + raw.abs() * 0.1
                    }
                } else {
                    raw * 0.05
                };
            }
        }
        // Make last row nearly colinear with first row to force rcond ~ 1e-16
        if n >= 2 {
            for j in 0..n {
                a[n - 1][j] = a[0][j] + ((j + 1) as f64) * scale;
            }
        }
    }

    if spec.inject_nonfinite && !a.is_empty() {
        a[0][n - 1] = f64::NAN;
    }

    let mut b = vec![1.0_f64; n];
    for (i, slot) in b.iter_mut().enumerate() {
        if let Some(&v) = spec.rhs.get(i)
            && v.is_finite()
        {
            *slot = v.clamp(-100.0, 100.0);
        } else {
            *slot = (i + 1) as f64;
        }
    }

    (a, b)
}

fuzz_target!(|input: RcondFloodInput| {
    let mode = if input.hardened {
        RuntimeMode::Hardened
    } else {
        RuntimeMode::Strict
    };
    let capacity = (usize::from(input.capacity) % 24).max(4);
    let mut portfolio = SolverPortfolio::new(mode, capacity);
    let ledger = AuditLedger::shared();

    let mut prev_alarm = false;
    let mut expected_min_fail_closed = 0usize;

    for (step_idx, spec) in input.steps.iter().take(MAX_STEPS).enumerate() {
        let (a, b) = build_flood_system(spec);
        let recovery = decode_recovery(spec.recovery_code);
        let options = SolveOptions {
            mode,
            recovery,
            check_finite: true,
            ..SolveOptions::default()
        };

        let t0 = Instant::now();
        let res = solve_with_audit(&a, &b, options, &mut portfolio, &ledger);
        let elapsed = t0.elapsed();

        assert!(
            elapsed <= MAX_CALL_DURATION,
            "rcond_flood step {step_idx} exceeded time budget ({elapsed:?} > {MAX_CALL_DURATION:?}): mode={mode:?}, recovery={recovery:?}, n={}",
            a.len()
        );

        // 1. Portfolio evidence and policy controller ledgers must stay strictly bounded by capacity.
        assert!(
            portfolio.evidence_len() <= capacity,
            "rcond_flood step {step_idx}: portfolio.evidence_len() {} > capacity {capacity}",
            portfolio.evidence_len()
        );
        assert!(
            portfolio.policy_controller().ledger().len() <= capacity,
            "rcond_flood step {step_idx}: policy_controller.ledger().len() {} > capacity {capacity}",
            portfolio.policy_controller().ledger().len()
        );

        // 2. Anytime-valid e-process drift alarm invariants (W4.4).
        let ep = portfolio.policy_controller().eprocess();
        let e_val = ep.e_value();
        assert!(
            e_val.is_finite() && e_val >= 0.0,
            "rcond_flood step {step_idx}: non-finite or negative e_value={e_val}"
        );
        if prev_alarm {
            assert!(
                ep.is_alarm_triggered(),
                "rcond_flood step {step_idx}: e-process alarm must be monotone once triggered"
            );
        }
        prev_alarm = ep.is_alarm_triggered();

        if ep.is_alarm_triggered() || portfolio.calibrator().should_fallback() {
            let (chosen, _, _, _) = portfolio.select_action(1e-16, None);
            assert_eq!(
                chosen,
                SolverAction::SVDFallback,
                "rcond_flood step {step_idx}: drift alarm active but select_action returned {chosen:?}"
            );
        }

        // 3. Audit ledger and recovery flag invariants (W5.3).
        match res {
            Ok(solve_res) => {
                if mode == RuntimeMode::Strict || recovery == RecoveryPolicy::Reject {
                    assert!(
                        !solve_res.recovered,
                        "rcond_flood step {step_idx}: recovered=true under mode={mode:?}, recovery={recovery:?}"
                    );
                }
            }
            Err(_) => {
                expected_min_fail_closed += 1;
            }
        }
    }

    let guard = ledger.lock().expect("audit ledger lock");
    let fail_closed_count = guard
        .entries()
        .iter()
        .filter(|e| matches!(e.action, AuditAction::FailClosed { .. }))
        .count();
    assert!(
        fail_closed_count >= expected_min_fail_closed,
        "rcond_flood: expected at least {expected_min_fail_closed} FailClosed events, got {fail_closed_count} (mode={mode:?}, steps={})",
        input.steps.len().min(MAX_STEPS)
    );
});
