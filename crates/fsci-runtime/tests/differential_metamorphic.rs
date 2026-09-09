//! Differential oracle, metamorphic relation, and adversarial tests
//! for the CASP runtime policy engine (bd-3jh.19.6).
//!
//! - §1: Differential oracle (>=15 cases): compare against hand-computed
//!   logit/softmax/expected-loss/action values with configurable tolerance.
//! - §2: Metamorphic relations (>=6 cases): input transformations that
//!   preserve or predictably change outputs.
//! - §3: Adversarial (>=8 cases): evidence poisoning, rapid oscillation,
//!   empty evidence, NaN injection.
//!
//! All tests produce structured JSON log lines.

use fsci_runtime::{
    DecisionSignals, MatrixConditionState, PolicyAction, PolicyController, RiskState, RuntimeMode,
    SignalSequence, SolverAction, SolverPortfolio, TestLogEntry, TestResult, assert_close_slice,
};

// ── Structured log helper ────────────────────────────────────────

fn log_differential(test_id: &str, input_summary: &str, expected: &str, actual: &str, pass: bool) {
    let entry = TestLogEntry::new(test_id, "fsci_runtime::differential", input_summary)
        .with_result(if pass {
            TestResult::Pass
        } else {
            TestResult::Fail
        });
    let json = entry.to_json_line();
    // Parse to inject extra fields
    let mut v: serde_json::Value = serde_json::from_str(&json).unwrap();
    v["category"] = serde_json::Value::String("differential".into());
    v["expected"] = serde_json::Value::String(expected.into());
    v["actual"] = serde_json::Value::String(actual.into());
    v["pass"] = serde_json::Value::Bool(pass);
    eprintln!("{}", serde_json::to_string(&v).unwrap());
}

fn log_metamorphic(test_id: &str, relation: &str, pass: bool) {
    let entry =
        TestLogEntry::new(test_id, "fsci_runtime::metamorphic", relation).with_result(if pass {
            TestResult::Pass
        } else {
            TestResult::Fail
        });
    let json = entry.to_json_line();
    let mut v: serde_json::Value = serde_json::from_str(&json).unwrap();
    v["category"] = serde_json::Value::String("metamorphic".into());
    v["pass"] = serde_json::Value::Bool(pass);
    eprintln!("{}", serde_json::to_string(&v).unwrap());
}

fn log_adversarial(test_id: &str, scenario: &str, expected_behavior: &str, pass: bool) {
    let entry =
        TestLogEntry::new(test_id, "fsci_runtime::adversarial", scenario).with_result(if pass {
            TestResult::Pass
        } else {
            TestResult::Fail
        });
    let json = entry.to_json_line();
    let mut v: serde_json::Value = serde_json::from_str(&json).unwrap();
    v["category"] = serde_json::Value::String("adversarial".into());
    v["expected_behavior"] = serde_json::Value::String(expected_behavior.into());
    v["pass"] = serde_json::Value::Bool(pass);
    eprintln!("{}", serde_json::to_string(&v).unwrap());
}

// ── Hand-computed oracle functions ───────────────────────────────

/// Compute logits from signals (mirrors policy.rs::logits_from_signals).
fn oracle_logits(cond_log10: f64, meta: f64, anomaly: f64) -> [f64; 3] {
    let c = (cond_log10 / 16.0).clamp(0.0, 1.0);
    let m = meta.clamp(0.0, 1.0);
    let a = anomaly.clamp(0.0, 1.0);
    let compatible = 2.8 - 0.8 * c - 3.2 * m - 2.4 * a;
    let ill = -0.4 + 1.4 * c + 0.6 * a - 0.8 * m;
    let incompat = -2.0 + 3.5 * m + 0.7 * a;
    [compatible, ill, incompat]
}

/// Compute softmax (mirrors policy.rs::softmax).
fn oracle_softmax(logits: [f64; 3]) -> [f64; 3] {
    let max_l = logits.iter().fold(f64::NEG_INFINITY, |a, &v| a.max(v));
    let exps = logits.map(|v| (v - max_l).exp());
    let denom: f64 = exps.iter().sum();
    if denom == 0.0 {
        return [1.0, 0.0, 0.0];
    }
    exps.map(|v| v / denom)
}

/// Loss matrix for a given mode.
fn oracle_loss_matrix(mode: RuntimeMode) -> [[f64; 3]; 3] {
    match mode {
        RuntimeMode::Strict => [[0.0, 65.0, 200.0], [8.0, 4.0, 80.0], [40.0, 25.0, 1.0]],
        RuntimeMode::Hardened => [[0.0, 50.0, 180.0], [5.0, 3.0, 60.0], [55.0, 30.0, 1.0]],
    }
}

/// Expected losses for action given posterior and mode.
fn oracle_expected_losses(mode: RuntimeMode, posterior: [f64; 3]) -> [f64; 3] {
    let matrix = oracle_loss_matrix(mode);
    let mut losses = [0.0; 3];
    for (row_idx, row) in matrix.iter().enumerate() {
        losses[row_idx] = row.iter().zip(posterior.iter()).map(|(l, p)| l * p).sum();
    }
    losses
}

/// Select action with minimum expected loss (tie-break: higher index).
fn oracle_select_action(losses: [f64; 3]) -> PolicyAction {
    let actions = [
        PolicyAction::Allow,
        PolicyAction::FullValidate,
        PolicyAction::FailClosed,
    ];
    let mut best_idx = 0;
    let mut best_loss = losses[0];
    for (idx, &loss) in losses.iter().enumerate().skip(1) {
        if loss < best_loss {
            best_loss = loss;
            best_idx = idx;
        } else if (loss - best_loss).abs() <= 1e-12 && idx > best_idx {
            best_idx = idx;
        }
    }
    actions[best_idx]
}

/// Full oracle: signals → action.
fn oracle_decide(
    mode: RuntimeMode,
    cond: f64,
    meta: f64,
    anomaly: f64,
) -> (PolicyAction, [f64; 3], [f64; 3]) {
    let logits = oracle_logits(cond, meta, anomaly);
    let posterior = oracle_softmax(logits);
    let losses = oracle_expected_losses(mode, posterior);
    let action = oracle_select_action(losses);
    (action, posterior, losses)
}

const TOL: f64 = 1e-9;

// ═══════════════════════════════════════════════════════════════════
// §1  Differential Oracle Tests (>=15)
// ═══════════════════════════════════════════════════════════════════

macro_rules! diff_test {
    ($name:ident, $mode:expr, $cond:expr, $meta:expr, $anom:expr) => {
        #[test]
        fn $name() {
            let mode = $mode;
            let (cond, meta, anom) = ($cond, $meta, $anom);
            let (expected_action, expected_posterior, expected_losses) =
                oracle_decide(mode, cond, meta, anom);

            let mut ctrl = PolicyController::new(mode, 16);
            let d = ctrl.decide(DecisionSignals::new(cond, meta, anom));

            // Compare posterior
            assert_close_slice(&d.posterior, &expected_posterior, TOL, TOL);
            // Compare expected losses
            assert_close_slice(&d.expected_losses, &expected_losses, TOL, TOL);
            // Compare action
            assert_eq!(d.action, expected_action);

            let pass = true;
            log_differential(
                stringify!($name),
                &format!("cond={cond}, meta={meta}, anom={anom}, mode={mode:?}"),
                &format!("{expected_action:?}"),
                &format!("{:?}", d.action),
                pass,
            );
        }
    };
}

// D1-D5: Strict mode across signal space
diff_test!(diff_strict_zero, RuntimeMode::Strict, 0.0, 0.0, 0.0);
diff_test!(diff_strict_benign, RuntimeMode::Strict, 2.0, 0.0, 0.0);
diff_test!(diff_strict_high_cond, RuntimeMode::Strict, 16.0, 0.0, 0.0);
diff_test!(diff_strict_high_meta, RuntimeMode::Strict, 0.0, 1.0, 0.0);
diff_test!(diff_strict_high_anom, RuntimeMode::Strict, 0.0, 0.0, 1.0);

// D6-D10: Hardened mode across same space
diff_test!(diff_hard_zero, RuntimeMode::Hardened, 0.0, 0.0, 0.0);
diff_test!(diff_hard_benign, RuntimeMode::Hardened, 2.0, 0.0, 0.0);
diff_test!(diff_hard_high_cond, RuntimeMode::Hardened, 16.0, 0.0, 0.0);
diff_test!(diff_hard_high_meta, RuntimeMode::Hardened, 0.0, 1.0, 0.0);
diff_test!(diff_hard_high_anom, RuntimeMode::Hardened, 0.0, 0.0, 1.0);

// D11-D15: Mixed signals
diff_test!(diff_strict_mid_all, RuntimeMode::Strict, 8.0, 0.5, 0.5);
diff_test!(diff_hard_mid_all, RuntimeMode::Hardened, 8.0, 0.5, 0.5);
diff_test!(
    diff_strict_low_cond_high_meta,
    RuntimeMode::Strict,
    1.0,
    0.9,
    0.1
);
diff_test!(
    diff_hard_high_cond_mid_meta,
    RuntimeMode::Hardened,
    12.0,
    0.4,
    0.3
);
diff_test!(diff_strict_extreme, RuntimeMode::Strict, 100.0, 1.0, 1.0);

// D16-D18: Additional edge cases
diff_test!(diff_strict_tiny_meta, RuntimeMode::Strict, 0.0, 0.01, 0.0);
diff_test!(diff_strict_tiny_anom, RuntimeMode::Strict, 0.0, 0.0, 0.01);
diff_test!(diff_hard_boundary, RuntimeMode::Hardened, 8.0, 0.25, 0.3);

// ═══════════════════════════════════════════════════════════════════
// §2  Metamorphic Relation Tests (>=6)
// ═══════════════════════════════════════════════════════════════════

// M1: Signal clamping idempotence.
// Clamping negative signals to zero should produce identical output.
#[test]
fn meta_clamping_idempotence() {
    let mut c1 = PolicyController::new(RuntimeMode::Strict, 16);
    let mut c2 = PolicyController::new(RuntimeMode::Strict, 16);
    let d1 = c1.decide(DecisionSignals::new(-10.0, -1.0, -1.0));
    let d2 = c2.decide(DecisionSignals::new(0.0, 0.0, 0.0));
    assert_eq!(d1.action, d2.action);
    assert_close_slice(&d1.posterior, &d2.posterior, TOL, TOL);
    log_metamorphic("meta_clamping_idempotence", "neg_clamp(s) == zero(s)", true);
}

// M2: Signal clamping upper bound.
// Signals above the clamp range should produce identical output to clamp max.
#[test]
fn meta_clamping_upper_bound() {
    let mut c1 = PolicyController::new(RuntimeMode::Strict, 16);
    let mut c2 = PolicyController::new(RuntimeMode::Strict, 16);
    let d1 = c1.decide(DecisionSignals::new(1000.0, 5.0, 5.0));
    let d2 = c2.decide(DecisionSignals::new(16.0, 1.0, 1.0));
    assert_eq!(d1.action, d2.action);
    assert_close_slice(&d1.posterior, &d2.posterior, TOL, TOL);
    log_metamorphic("meta_clamping_upper_bound", "clamp(over_max) == max", true);
}

// M3: Mode does not affect posterior (only expected losses differ).
// Both modes use the same logit model, so posterior should be identical.
#[test]
fn meta_mode_preserves_posterior() {
    let s = DecisionSignals::new(5.0, 0.3, 0.2);
    let mut strict = PolicyController::new(RuntimeMode::Strict, 16);
    let mut hard = PolicyController::new(RuntimeMode::Hardened, 16);
    let ds = strict.decide(s);
    let dh = hard.decide(s);
    assert_close_slice(&ds.posterior, &dh.posterior, TOL, TOL);
    log_metamorphic(
        "meta_mode_preserves_posterior",
        "posterior(Strict, s) == posterior(Hardened, s)",
        true,
    );
}

// M4: Decision independence from ledger history.
// Two controllers with different histories should produce identical
// decisions for the same input.
#[test]
fn meta_decision_independent_of_history() {
    let s = DecisionSignals::new(5.0, 0.3, 0.2);
    let mut c1 = PolicyController::new(RuntimeMode::Strict, 16);
    // Warm up c1 with some decisions
    for i in 0..10 {
        c1.decide(DecisionSignals::new(i as f64, 0.0, 0.0));
    }
    let mut c2 = PolicyController::new(RuntimeMode::Strict, 16);
    let d1 = c1.decide(s);
    let d2 = c2.decide(s);
    assert_eq!(d1.action, d2.action);
    assert_close_slice(&d1.posterior, &d2.posterior, TOL, TOL);
    log_metamorphic(
        "meta_decision_independent_of_history",
        "decide(s, history1) == decide(s, history2)",
        true,
    );
}

// M5: Monotone metadata incompatibility.
// Increasing metadata_incompatibility_score (other signals held constant)
// should never decrease P(IncompatibleMetadata).
#[test]
fn meta_monotone_metadata_incompatibility() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 100);
    let mut prev_p_incompat = 0.0;
    for i in 0..=10 {
        let meta = i as f64 / 10.0;
        let d = ctrl.decide(DecisionSignals::new(0.0, meta, 0.0));
        assert!(
            d.posterior[2] >= prev_p_incompat - TOL,
            "P(Incompat) must be monotone in metadata: prev={prev_p_incompat}, curr={}",
            d.posterior[2]
        );
        prev_p_incompat = d.posterior[2];
    }
    log_metamorphic(
        "meta_monotone_metadata_incompatibility",
        "meta↑ => P(Incompat)↑",
        true,
    );
}

// M6: Monotone condition number → ill-conditioned posterior.
// Increasing condition_number_log10 should never decrease P(IllConditioned).
#[test]
fn meta_monotone_condition_number() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 100);
    let mut prev_p_ill = 0.0;
    for i in 0..=20 {
        let cond = i as f64;
        let d = ctrl.decide(DecisionSignals::new(cond, 0.0, 0.0));
        assert!(
            d.posterior[1] >= prev_p_ill - TOL,
            "P(IllCond) must be monotone in condition: prev={prev_p_ill}, curr={}",
            d.posterior[1]
        );
        prev_p_ill = d.posterior[1];
    }
    log_metamorphic(
        "meta_monotone_condition_number",
        "cond↑ => P(IllCond)↑",
        true,
    );
}

fn state_to_rcond(state: &MatrixConditionState) -> f64 {
    match state {
        MatrixConditionState::WellConditioned => 1e-2,
        MatrixConditionState::ModerateCondition => 1e-6,
        MatrixConditionState::IllConditioned => 1e-12,
        MatrixConditionState::NearSingular => 1e-18,
    }
}

// M7: Solver portfolio selection stability under condition state.
// For each condition state, the selected action should be stable across
// multiple calls (no nondeterminism).
#[test]
fn meta_solver_selection_stability() {
    let p = SolverPortfolio::new(RuntimeMode::Strict, 64);
    for state in &MatrixConditionState::ALL {
        let (a1, _, _, _) = p.select_action(state_to_rcond(state), None);
        let (a2, _, _, _) = p.select_action(state_to_rcond(state), None);
        assert_eq!(
            a1, a2,
            "solver selection must be deterministic for {state:?}"
        );
    }
    log_metamorphic(
        "meta_solver_selection_stability",
        "select(s) == select(s) for all states",
        true,
    );
}

// M8: Ledger capacity does not affect decision output.
#[test]
fn meta_ledger_capacity_irrelevant() {
    let s = DecisionSignals::new(5.0, 0.3, 0.2);
    let mut c1 = PolicyController::new(RuntimeMode::Strict, 1);
    let mut c2 = PolicyController::new(RuntimeMode::Strict, 1000);
    let d1 = c1.decide(s);
    let d2 = c2.decide(s);
    assert_eq!(d1.action, d2.action);
    assert_close_slice(&d1.posterior, &d2.posterior, TOL, TOL);
    log_metamorphic(
        "meta_ledger_capacity_irrelevant",
        "decide(s, cap=1) == decide(s, cap=1000)",
        true,
    );
}

// ═══════════════════════════════════════════════════════════════════
// §3  Adversarial Tests (>=8)
// ═══════════════════════════════════════════════════════════════════

// A1: Evidence poisoning – many benign signals then sudden hostile.
// The controller must correctly identify the hostile signal regardless
// of prior benign history.
#[test]
fn adv_evidence_poisoning_benign_then_hostile() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 100);
    for _ in 0..50 {
        ctrl.decide(DecisionSignals::new(2.0, 0.0, 0.0));
    }
    let d = ctrl.decide(DecisionSignals::new(0.0, 1.0, 0.0));
    assert_eq!(d.action, PolicyAction::FailClosed);
    assert_eq!(d.top_state, RiskState::IncompatibleMetadata);
    log_adversarial(
        "adv_evidence_poisoning_benign_then_hostile",
        "50 benign + 1 hostile",
        "FailClosed on hostile regardless of history",
        true,
    );
}

// A2: Rapid oscillation between Allow and FailClosed.
// Each decision should be correct regardless of the oscillation pattern.
#[test]
fn adv_rapid_oscillation_correctness() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 100);
    let mut decisions = Vec::new();
    for i in 0..40 {
        let meta = if i % 2 == 0 { 0.0 } else { 1.0 };
        let d = ctrl.decide(DecisionSignals::new(0.0, meta, 0.0));
        decisions.push((meta, d.action));
    }
    for (meta, action) in &decisions {
        if *meta > 0.5 {
            assert_eq!(*action, PolicyAction::FailClosed);
        } else {
            assert_eq!(*action, PolicyAction::Allow);
        }
    }
    log_adversarial(
        "adv_rapid_oscillation_correctness",
        "alternating benign/hostile signals",
        "correct action at each step",
        true,
    );
}

// A3: Empty ledger replay – controller with zero decisions should handle
// new input correctly.
#[test]
fn adv_empty_evidence_first_decision() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    assert!(ctrl.ledger().is_empty());
    let d = ctrl.decide(DecisionSignals::new(2.0, 0.0, 0.0));
    assert_eq!(d.action, PolicyAction::Allow);
    assert_eq!(ctrl.ledger().len(), 1);
    log_adversarial(
        "adv_empty_evidence_first_decision",
        "first decision on empty ledger",
        "Allow for benign signals",
        true,
    );
}

// A4: NaN injection – controller should not panic and should record
// the decision in the ledger as a deterministic fail-closed outcome.
#[test]
fn adv_nan_injection_no_panic() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    let d = ctrl.decide(DecisionSignals::new(f64::NAN, f64::NAN, f64::NAN));
    assert_eq!(ctrl.ledger().len(), 1);
    assert_eq!(d.action, PolicyAction::FailClosed);
    assert_eq!(d.top_state, RiskState::IncompatibleMetadata);
    assert_eq!(d.posterior, [0.0, 0.0, 1.0]);
    log_adversarial(
        "adv_nan_injection_no_panic",
        "all-NaN signals",
        "no panic, fail-closed action produced",
        true,
    );
}

// A5: Infinity injection – test graceful handling.
#[test]
fn adv_inf_injection_no_panic() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    let d = ctrl.decide(DecisionSignals::new(
        f64::INFINITY,
        f64::INFINITY,
        f64::INFINITY,
    ));
    assert_eq!(ctrl.ledger().len(), 1);
    assert_eq!(d.action, PolicyAction::FailClosed);
    assert_eq!(d.top_state, RiskState::IncompatibleMetadata);
    assert_eq!(d.posterior, [0.0, 0.0, 1.0]);
    log_adversarial(
        "adv_inf_injection_no_panic",
        "all-Infinity signals",
        "no panic, fail-closed action produced",
        true,
    );
}

// A6: Negative infinity injection.
#[test]
fn adv_neg_inf_injection_no_panic() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    let d = ctrl.decide(DecisionSignals::new(
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ));
    assert_eq!(ctrl.ledger().len(), 1);
    assert_eq!(d.action, PolicyAction::FailClosed);
    assert_eq!(d.top_state, RiskState::IncompatibleMetadata);
    assert_eq!(d.posterior, [0.0, 0.0, 1.0]);
    log_adversarial(
        "adv_neg_inf_injection_no_panic",
        "all-NEG_INFINITY signals",
        "no panic, fail-closed action produced",
        true,
    );
}

// A7: Ledger capacity exhaustion – verify FIFO under rapid writes.
#[test]
fn adv_ledger_fifo_exhaustion() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 3);
    for _ in 0..1000 {
        ctrl.decide(DecisionSignals::new(2.0, 0.0, 0.0));
    }
    assert_eq!(ctrl.ledger().len(), 3);
    let final_d = ctrl.decide(DecisionSignals::new(0.0, 1.0, 0.0));
    assert_eq!(final_d.action, PolicyAction::FailClosed);
    assert_eq!(ctrl.ledger().len(), 3);
    log_adversarial(
        "adv_ledger_fifo_exhaustion",
        "1000 writes to capacity-3 ledger",
        "FIFO eviction, correct final decision",
        true,
    );
}

// A8: Calibrator flooding – trigger and recover from fallback.
#[test]
fn adv_calibrator_flood_and_recovery() {
    let mut portfolio = SolverPortfolio::new(RuntimeMode::Strict, 64);
    // Feed enough violations to trigger calibrator fallback
    for _ in 0..40 {
        portfolio.observe_backward_error(0.0);
    }
    assert!(!portfolio.calibrator().should_fallback());

    for _ in 0..20 {
        portfolio.observe_backward_error(1.0);
    }
    // With fallback active, solver should return SVDFallback
    let (action, _, _, _) = portfolio.select_action(1e-2, None);
    assert_eq!(action, SolverAction::SVDFallback);
    log_adversarial(
        "adv_calibrator_flood_and_recovery",
        "flood calibrator with violations",
        "SVDFallback when calibrator triggers",
        true,
    );
}

// A9: Signal sequence with mixed NaN and valid signals.
#[test]
fn adv_sequence_mixed_nan_valid() {
    let mut seq = SignalSequence::new("adv_mixed_nan");
    seq.push(DecisionSignals::new(2.0, 0.0, 0.0));
    seq.push(DecisionSignals::new(f64::NAN, 0.0, 0.0));
    seq.push(DecisionSignals::new(2.0, 0.0, 0.0));

    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    for s in seq.iter() {
        let _ = ctrl.decide(*s);
    }
    // All 3 decisions recorded, controller didn't panic
    assert_eq!(ctrl.ledger().len(), 3);
    log_adversarial(
        "adv_sequence_mixed_nan_valid",
        "valid-NaN-valid signal sequence",
        "all decisions recorded without panic",
        true,
    );
}

// A10: Extreme condition number combined with extreme metadata.
#[test]
fn adv_extreme_combined_signals() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    let d = ctrl.decide(DecisionSignals::new(f64::MAX, f64::MAX, f64::MAX));
    // Should still produce a decision (after clamping)
    assert_eq!(ctrl.ledger().len(), 1);
    // All signals clamp to max → same as (16.0, 1.0, 1.0) effectively
    // but f64::MAX / 16.0 is still > 1.0, so clamped to 1.0
    let mut ctrl2 = PolicyController::new(RuntimeMode::Strict, 16);
    let d2 = ctrl2.decide(DecisionSignals::new(16.0, 1.0, 1.0));
    assert_eq!(d.action, d2.action);
    log_adversarial(
        "adv_extreme_combined_signals",
        "f64::MAX for all signals",
        "clamps to max range, same as clamped values",
        true,
    );
}

#[test]
fn test_serde_json_float_roundtrip_precision() {
    let test_floats = [
        -8.778143165047872e-97f64,
        0.0f64,
        -0.0f64,
        1.0f64,
        -1.0f64,
        0.5f64,
        -0.5f64,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        1.0e6,
        -1.0e6,
    ];
    for &x in &test_floats {
        let json = serde_json::to_string(&x).expect("serialize f64");
        let parsed: f64 = serde_json::from_str(&json).expect("deserialize f64");
        assert_eq!(
            x.to_bits(),
            parsed.to_bits(),
            "mismatch for {x:?}: original bits {:016x}, parsed bits {:016x}",
            x.to_bits(),
            parsed.to_bits()
        );
    }
}
