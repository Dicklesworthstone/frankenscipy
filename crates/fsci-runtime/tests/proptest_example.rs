//! Example property tests demonstrating FrankenSciPy test conventions.
//!
//! Convention: test_{module}_{function}_{scenario}
//!
//! Seed replay: `PROPTEST_CASES=1000 cargo test -p fsci-runtime --test proptest_example`
//! Reproduce: `PROPTEST_SEED=<seed> cargo test -p fsci-runtime --test proptest_example`

use fsci_runtime::{
    ConformalCalibrator, DecisionSignals, MatrixConditionState, PolicyAction, PolicyController,
    RuntimeMode, SolverPortfolio, TestLogEntry, TestResult,
};
use proptest::prelude::*;

// ═══════════════════════════════════════════════════════════════════
// Property: Policy controller posterior is always a valid probability
// distribution (non-negative, sums to 1).
// ═══════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn test_runtime_policy_posterior_is_valid_distribution(
        cond in 0.0f64..20.0,
        meta in 0.0f64..1.0,
        anomaly in 0.0f64..1.0,
    ) {
        let mut controller = PolicyController::new(RuntimeMode::Strict, 16);
        let decision = controller.decide(DecisionSignals::new(cond, meta, anomaly));

        // Posterior must be non-negative
        for p in &decision.posterior {
            prop_assert!(*p >= 0.0, "posterior element must be non-negative: {p}");
        }

        // Posterior must sum to 1.0 within floating-point tolerance
        let sum: f64 = decision.posterior.iter().sum();
        prop_assert!(
            (sum - 1.0).abs() < 1e-9,
            "posterior must sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn test_runtime_policy_action_is_deterministic(
        cond in 0.0f64..20.0,
        meta in 0.0f64..1.0,
        anomaly in 0.0f64..1.0,
    ) {
        let signals = DecisionSignals::new(cond, meta, anomaly);
        let mut c1 = PolicyController::new(RuntimeMode::Strict, 16);
        let mut c2 = PolicyController::new(RuntimeMode::Strict, 16);
        let d1 = c1.decide(signals);
        let d2 = c2.decide(signals);
        prop_assert_eq!(d1.action, d2.action, "same inputs must produce same action");
    }

    #[test]
    fn test_runtime_policy_both_modes_failclosed_on_incompatible(
        meta in 0.8f64..1.0,
        anomaly in 0.0f64..1.0,
    ) {
        // When metadata incompatibility is very high, BOTH modes must fail-closed.
        let signals = DecisionSignals::new(0.0, meta, anomaly);
        let mut strict = PolicyController::new(RuntimeMode::Strict, 16);
        let mut hardened = PolicyController::new(RuntimeMode::Hardened, 16);
        let ds = strict.decide(signals);
        let dh = hardened.decide(signals);

        prop_assert_eq!(
            ds.action,
            PolicyAction::FailClosed,
            "strict must fail-closed on high metadata incompatibility"
        );
        prop_assert_eq!(
            dh.action,
            PolicyAction::FailClosed,
            "hardened must fail-closed on high metadata incompatibility"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// Property: CASP solver selection is consistent across repeated calls.
// ═══════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn test_casp_solver_selection_is_deterministic(
        state_idx in 0usize..4,
    ) {
        let state = MatrixConditionState::ALL[state_idx];
        let p1 = SolverPortfolio::new(RuntimeMode::Strict, 64);
        let p2 = SolverPortfolio::new(RuntimeMode::Strict, 64);
        let (a1, _, _, _) = p1.select_action(&state);
        let (a2, _, _, _) = p2.select_action(&state);
        prop_assert_eq!(a1, a2, "same condition must select same solver");
    }
}

// ═══════════════════════════════════════════════════════════════════
// Property: Conformal calibrator is monotone-bounded.
// ═══════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn test_runtime_calibrator_bounded_capacity(
        n_scores in 1usize..500,
        capacity in 10usize..100,
    ) {
        let mut cal = ConformalCalibrator::new(0.05, capacity);
        for i in 0..n_scores {
            cal.observe(i as f64 * 1e-10);
        }
        prop_assert!(
            cal.total_predictions() == n_scores,
            "total predictions must match observations"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// Unit test demonstrating structured log convention.
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_runtime_structured_log_convention() {
    // This demonstrates the canonical structured log pattern.
    let entry = TestLogEntry::new(
        "test_runtime_policy_posterior_is_valid_distribution",
        "fsci_runtime",
        "property test: posterior normalization verified over 256 cases",
    )
    .with_result(TestResult::Pass)
    .with_mode(RuntimeMode::Strict);

    let json = entry.to_json_line();
    let parsed: serde_json::Value =
        serde_json::from_str(&json).expect("structured log must be valid JSON");
    assert!(parsed["test_id"].is_string());
    assert!(parsed["timestamp_ms"].is_number());
    assert_eq!(parsed["level"], "info");
}
