//! Comprehensive CASP runtime test suite (bd-3jh.19.5).
//!
//! >=300 unit test cases + >=8 property tests with structured JSON logging.

use fsci_runtime::{
    ConformalCalibrator, DecisionEvidenceEntry, DecisionSignals, MatrixConditionState,
    PolicyAction, PolicyController, PolicyDecision, PolicyEvidenceLedger, PortfolioEvidence,
    RiskState, RuntimeMode, SignalSequence, SolverAction, SolverEvidenceEntry, SolverPortfolio,
    TestLogEntry, TestLogLevel, TestResult, assert_close, assert_close_slice, casp_now_unix_ms,
    within_tolerance,
};
use proptest::prelude::*;

// ═══════════════════════════════════════════════════════════════════
// §1  RuntimeMode (10 tests)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn mode_strict_debug() {
    assert_eq!(format!("{:?}", RuntimeMode::Strict), "Strict");
}

#[test]
fn mode_hardened_debug() {
    assert_eq!(format!("{:?}", RuntimeMode::Hardened), "Hardened");
}

#[test]
fn mode_strict_eq() {
    assert_eq!(RuntimeMode::Strict, RuntimeMode::Strict);
}

#[test]
fn mode_hardened_eq() {
    assert_eq!(RuntimeMode::Hardened, RuntimeMode::Hardened);
}

#[test]
fn mode_strict_ne_hardened() {
    assert_ne!(RuntimeMode::Strict, RuntimeMode::Hardened);
}

#[test]
fn mode_clone() {
    let m = RuntimeMode::Strict;
    let m2 = m;
    assert_eq!(m, m2);
}

#[test]
fn mode_serde_strict_roundtrip() {
    let json = serde_json::to_string(&RuntimeMode::Strict).unwrap();
    let back: RuntimeMode = serde_json::from_str(&json).unwrap();
    assert_eq!(back, RuntimeMode::Strict);
}

#[test]
fn mode_serde_hardened_roundtrip() {
    let json = serde_json::to_string(&RuntimeMode::Hardened).unwrap();
    let back: RuntimeMode = serde_json::from_str(&json).unwrap();
    assert_eq!(back, RuntimeMode::Hardened);
}

#[test]
fn mode_serde_strict_json_value() {
    let json = serde_json::to_string(&RuntimeMode::Strict).unwrap();
    assert_eq!(json, "\"Strict\"");
}

#[test]
fn mode_serde_hardened_json_value() {
    let json = serde_json::to_string(&RuntimeMode::Hardened).unwrap();
    assert_eq!(json, "\"Hardened\"");
}

// ═══════════════════════════════════════════════════════════════════
// §2  DecisionSignals (30 tests)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn signals_new_stores_fields() {
    let s = DecisionSignals::new(1.0, 0.5, 0.3);
    assert_eq!(s.condition_number_log10, 1.0);
    assert_eq!(s.metadata_incompatibility_score, 0.5);
    assert_eq!(s.input_anomaly_score, 0.3);
}

#[test]
fn signals_zero() {
    let s = DecisionSignals::new(0.0, 0.0, 0.0);
    assert!(s.is_finite());
}

#[test]
fn signals_large_condition() {
    let s = DecisionSignals::new(1e18, 0.0, 0.0);
    assert!(s.is_finite());
}

#[test]
fn signals_negative_condition() {
    let s = DecisionSignals::new(-5.0, 0.0, 0.0);
    assert!(s.is_finite());
}

#[test]
fn signals_is_finite_all_normal() {
    assert!(DecisionSignals::new(2.0, 0.5, 0.1).is_finite());
}

#[test]
fn signals_nan_condition() {
    assert!(!DecisionSignals::new(f64::NAN, 0.0, 0.0).is_finite());
}

#[test]
fn signals_nan_metadata() {
    assert!(!DecisionSignals::new(0.0, f64::NAN, 0.0).is_finite());
}

#[test]
fn signals_nan_anomaly() {
    assert!(!DecisionSignals::new(0.0, 0.0, f64::NAN).is_finite());
}

#[test]
fn signals_inf_condition() {
    assert!(!DecisionSignals::new(f64::INFINITY, 0.0, 0.0).is_finite());
}

#[test]
fn signals_inf_metadata() {
    assert!(!DecisionSignals::new(0.0, f64::INFINITY, 0.0).is_finite());
}

#[test]
fn signals_inf_anomaly() {
    assert!(!DecisionSignals::new(0.0, 0.0, f64::INFINITY).is_finite());
}

#[test]
fn signals_neg_inf_condition() {
    assert!(!DecisionSignals::new(f64::NEG_INFINITY, 0.0, 0.0).is_finite());
}

#[test]
fn signals_neg_inf_metadata() {
    assert!(!DecisionSignals::new(0.0, f64::NEG_INFINITY, 0.0).is_finite());
}

#[test]
fn signals_neg_inf_anomaly() {
    assert!(!DecisionSignals::new(0.0, 0.0, f64::NEG_INFINITY).is_finite());
}

#[test]
fn signals_all_nan() {
    assert!(!DecisionSignals::new(f64::NAN, f64::NAN, f64::NAN).is_finite());
}

#[test]
fn signals_all_inf() {
    assert!(!DecisionSignals::new(f64::INFINITY, f64::INFINITY, f64::INFINITY).is_finite());
}

#[test]
fn signals_copy_semantics() {
    let s = DecisionSignals::new(1.0, 2.0, 3.0);
    let s2 = s;
    assert_eq!(s, s2);
}

#[test]
fn signals_clone_semantics() {
    let s = DecisionSignals::new(1.0, 2.0, 3.0);
    #[allow(clippy::clone_on_copy)]
    let s2 = s.clone();
    assert_eq!(s, s2);
}

#[test]
fn signals_serde_roundtrip() {
    let s = DecisionSignals::new(4.5, 0.7, 0.2);
    let json = serde_json::to_string(&s).unwrap();
    let back: DecisionSignals = serde_json::from_str(&json).unwrap();
    assert_eq!(back, s);
}

#[test]
fn signals_debug_format() {
    let s = DecisionSignals::new(1.0, 0.5, 0.3);
    let dbg = format!("{s:?}");
    assert!(dbg.contains("DecisionSignals"));
}

#[test]
fn signals_eq_same() {
    let a = DecisionSignals::new(1.0, 0.5, 0.3);
    let b = DecisionSignals::new(1.0, 0.5, 0.3);
    assert_eq!(a, b);
}

#[test]
fn signals_ne_different_cond() {
    let a = DecisionSignals::new(1.0, 0.5, 0.3);
    let b = DecisionSignals::new(2.0, 0.5, 0.3);
    assert_ne!(a, b);
}

#[test]
fn signals_ne_different_meta() {
    let a = DecisionSignals::new(1.0, 0.5, 0.3);
    let b = DecisionSignals::new(1.0, 0.6, 0.3);
    assert_ne!(a, b);
}

#[test]
fn signals_ne_different_anomaly() {
    let a = DecisionSignals::new(1.0, 0.5, 0.3);
    let b = DecisionSignals::new(1.0, 0.5, 0.4);
    assert_ne!(a, b);
}

#[test]
fn signals_subnormal_is_finite() {
    assert!(DecisionSignals::new(f64::MIN_POSITIVE / 2.0, 0.0, 0.0).is_finite());
}

#[test]
fn signals_max_is_finite() {
    assert!(DecisionSignals::new(f64::MAX, 0.0, 0.0).is_finite());
}

#[test]
fn signals_min_is_finite() {
    assert!(DecisionSignals::new(f64::MIN, 0.0, 0.0).is_finite());
}

#[test]
fn signals_epsilon_is_finite() {
    assert!(DecisionSignals::new(f64::EPSILON, f64::EPSILON, f64::EPSILON).is_finite());
}

#[test]
fn signals_serde_preserves_precision() {
    let s = DecisionSignals::new(1.23456789012345, 0.98765432109876, 0.11111111111111);
    let json = serde_json::to_string(&s).unwrap();
    let back: DecisionSignals = serde_json::from_str(&json).unwrap();
    assert_eq!(back.condition_number_log10, s.condition_number_log10);
}

#[test]
fn signals_negative_values_accepted() {
    let s = DecisionSignals::new(-10.0, -0.5, -0.9);
    assert!(s.is_finite());
}

// ═══════════════════════════════════════════════════════════════════
// §3  SignalSequence (25 tests)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn seq_new_empty() {
    let seq = SignalSequence::new("test");
    assert!(seq.is_empty());
    assert_eq!(seq.len(), 0);
    assert_eq!(seq.id, "test");
}

#[test]
fn seq_push_one() {
    let mut seq = SignalSequence::new("s1");
    seq.push(DecisionSignals::new(1.0, 0.0, 0.0));
    assert_eq!(seq.len(), 1);
    assert!(!seq.is_empty());
}

#[test]
fn seq_push_many() {
    let mut seq = SignalSequence::new("s2");
    for i in 0..100 {
        seq.push(DecisionSignals::new(i as f64, 0.0, 0.0));
    }
    assert_eq!(seq.len(), 100);
}

#[test]
fn seq_iter_count() {
    let mut seq = SignalSequence::new("iter");
    seq.push(DecisionSignals::new(1.0, 0.0, 0.0));
    seq.push(DecisionSignals::new(2.0, 0.0, 0.0));
    seq.push(DecisionSignals::new(3.0, 0.0, 0.0));
    assert_eq!(seq.iter().count(), 3);
}

#[test]
fn seq_iter_preserves_order() {
    let mut seq = SignalSequence::new("order");
    seq.push(DecisionSignals::new(1.0, 0.0, 0.0));
    seq.push(DecisionSignals::new(2.0, 0.0, 0.0));
    let vals: Vec<f64> = seq.iter().map(|s| s.condition_number_log10).collect();
    assert_eq!(vals, vec![1.0, 2.0]);
}

#[test]
fn seq_expected_actions_none_by_default() {
    let seq = SignalSequence::new("test");
    assert!(seq.expected_actions.is_none());
}

#[test]
fn seq_expected_actions_settable() {
    let mut seq = SignalSequence::new("test");
    seq.expected_actions = Some(vec!["Allow".into(), "FailClosed".into()]);
    assert_eq!(seq.expected_actions.as_ref().unwrap().len(), 2);
}

#[test]
fn seq_clone() {
    let mut seq = SignalSequence::new("orig");
    seq.push(DecisionSignals::new(1.0, 0.0, 0.0));
    let seq2 = seq.clone();
    assert_eq!(seq, seq2);
}

#[test]
fn seq_serde_roundtrip() {
    let mut seq = SignalSequence::new("serde");
    seq.push(DecisionSignals::new(5.0, 0.3, 0.1));
    let json = serde_json::to_string(&seq).unwrap();
    let back: SignalSequence = serde_json::from_str(&json).unwrap();
    assert_eq!(back, seq);
}

#[test]
fn seq_serde_omits_none_expected() {
    let seq = SignalSequence::new("test");
    let json = serde_json::to_string(&seq).unwrap();
    assert!(!json.contains("expected_actions"));
}

#[test]
fn seq_serde_includes_expected_when_set() {
    let mut seq = SignalSequence::new("test");
    seq.expected_actions = Some(vec!["Allow".into()]);
    let json = serde_json::to_string(&seq).unwrap();
    assert!(json.contains("expected_actions"));
}

#[test]
fn seq_debug_format() {
    let seq = SignalSequence::new("dbg");
    let dbg = format!("{seq:?}");
    assert!(dbg.contains("SignalSequence"));
}

#[test]
fn seq_id_from_string() {
    let seq = SignalSequence::new(String::from("owned"));
    assert_eq!(seq.id, "owned");
}

#[test]
fn seq_eq_same() {
    let a = SignalSequence::new("a");
    let b = SignalSequence::new("a");
    assert_eq!(a, b);
}

#[test]
fn seq_ne_different_id() {
    let a = SignalSequence::new("a");
    let b = SignalSequence::new("b");
    assert_ne!(a, b);
}

#[test]
fn seq_ne_different_signals() {
    let mut a = SignalSequence::new("x");
    a.push(DecisionSignals::new(1.0, 0.0, 0.0));
    let b = SignalSequence::new("x");
    assert_ne!(a, b);
}

#[test]
fn seq_replay_through_controller_strict() {
    let mut seq = SignalSequence::new("replay_strict");
    seq.push(DecisionSignals::new(2.0, 0.0, 0.0));
    seq.push(DecisionSignals::new(0.0, 1.0, 0.0));
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    let decisions: Vec<_> = seq.iter().map(|s| ctrl.decide(*s)).collect();
    assert_eq!(decisions[0].action, PolicyAction::Allow);
    assert_eq!(decisions[1].action, PolicyAction::FailClosed);
}

#[test]
fn seq_replay_through_controller_hardened() {
    let mut seq = SignalSequence::new("replay_hard");
    seq.push(DecisionSignals::new(2.0, 0.0, 0.0));
    seq.push(DecisionSignals::new(0.0, 1.0, 0.0));
    let mut ctrl = PolicyController::new(RuntimeMode::Hardened, 16);
    let decisions: Vec<_> = seq.iter().map(|s| ctrl.decide(*s)).collect();
    assert_eq!(decisions[0].action, PolicyAction::Allow);
    assert_eq!(decisions[1].action, PolicyAction::FailClosed);
}

#[test]
fn seq_empty_replay() {
    let seq = SignalSequence::new("empty");
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    let decisions: Vec<_> = seq.iter().map(|s| ctrl.decide(*s)).collect();
    assert!(decisions.is_empty());
    assert!(ctrl.ledger().is_empty());
}

#[test]
fn seq_large_replay() {
    let mut seq = SignalSequence::new("large");
    for _ in 0..50 {
        seq.push(DecisionSignals::new(2.0, 0.0, 0.0));
    }
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 100);
    let decisions: Vec<_> = seq.iter().map(|s| ctrl.decide(*s)).collect();
    assert_eq!(decisions.len(), 50);
    assert_eq!(ctrl.ledger().len(), 50);
}

#[test]
fn seq_nan_signals_in_sequence() {
    let mut seq = SignalSequence::new("nan_seq");
    seq.push(DecisionSignals::new(f64::NAN, 0.0, 0.0));
    assert!(!seq.iter().next().unwrap().is_finite());
}

#[test]
fn seq_mixed_finite_and_nan() {
    let mut seq = SignalSequence::new("mixed");
    seq.push(DecisionSignals::new(1.0, 0.0, 0.0));
    seq.push(DecisionSignals::new(f64::NAN, 0.0, 0.0));
    seq.push(DecisionSignals::new(3.0, 0.0, 0.0));
    let finites: Vec<bool> = seq.iter().map(|s| s.is_finite()).collect();
    assert_eq!(finites, vec![true, false, true]);
}

#[test]
fn seq_single_element() {
    let mut seq = SignalSequence::new("single");
    seq.push(DecisionSignals::new(5.0, 0.5, 0.5));
    assert_eq!(seq.len(), 1);
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    let d = ctrl.decide(*seq.iter().next().unwrap());
    assert!(d.posterior.iter().all(|p| *p >= 0.0));
}

#[test]
fn seq_iter_does_not_consume() {
    let mut seq = SignalSequence::new("no_consume");
    seq.push(DecisionSignals::new(1.0, 0.0, 0.0));
    let _ = seq.iter().count();
    let _ = seq.iter().count();
    assert_eq!(seq.len(), 1);
}

// ═══════════════════════════════════════════════════════════════════
// §4  PolicyController & PolicyDecision (70 tests)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn ctrl_new_strict() {
    let ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    assert_eq!(ctrl.mode(), RuntimeMode::Strict);
    assert!(ctrl.ledger().is_empty());
}

#[test]
fn ctrl_new_hardened() {
    let ctrl = PolicyController::new(RuntimeMode::Hardened, 16);
    assert_eq!(ctrl.mode(), RuntimeMode::Hardened);
}

#[test]
fn ctrl_benign_strict_allows() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    let d = ctrl.decide(DecisionSignals::new(2.0, 0.0, 0.0));
    assert_eq!(d.action, PolicyAction::Allow);
    assert_eq!(d.mode, RuntimeMode::Strict);
}

#[test]
fn ctrl_benign_hardened_allows() {
    let mut ctrl = PolicyController::new(RuntimeMode::Hardened, 16);
    let d = ctrl.decide(DecisionSignals::new(2.0, 0.0, 0.0));
    assert_eq!(d.action, PolicyAction::Allow);
}

#[test]
fn ctrl_high_meta_strict_failclosed() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    let d = ctrl.decide(DecisionSignals::new(0.0, 1.0, 0.0));
    assert_eq!(d.action, PolicyAction::FailClosed);
}

#[test]
fn ctrl_high_meta_hardened_failclosed() {
    let mut ctrl = PolicyController::new(RuntimeMode::Hardened, 16);
    let d = ctrl.decide(DecisionSignals::new(0.0, 1.0, 0.0));
    assert_eq!(d.action, PolicyAction::FailClosed);
}

#[test]
fn ctrl_high_meta_top_state_incompatible() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    let d = ctrl.decide(DecisionSignals::new(0.0, 1.0, 0.0));
    assert_eq!(d.top_state, RiskState::IncompatibleMetadata);
}

#[test]
fn ctrl_benign_top_state_compatible() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    let d = ctrl.decide(DecisionSignals::new(2.0, 0.0, 0.0));
    assert_eq!(d.top_state, RiskState::Compatible);
}

#[test]
fn ctrl_high_cond_strict() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    let d = ctrl.decide(DecisionSignals::new(16.0, 0.0, 0.0));
    // High condition → ill-conditioned state likely
    assert!(d.posterior[1] > d.posterior[0] || d.action == PolicyAction::FullValidate);
}

#[test]
fn ctrl_high_cond_hardened() {
    let mut ctrl = PolicyController::new(RuntimeMode::Hardened, 16);
    let d = ctrl.decide(DecisionSignals::new(16.0, 0.0, 0.0));
    assert!(d.posterior[1] > 0.0);
}

#[test]
fn ctrl_all_zeros_allows() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    let d = ctrl.decide(DecisionSignals::new(0.0, 0.0, 0.0));
    assert_eq!(d.action, PolicyAction::Allow);
    assert_eq!(d.top_state, RiskState::Compatible);
}

#[test]
fn ctrl_all_max_failclosed() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    let d = ctrl.decide(DecisionSignals::new(100.0, 1.0, 1.0));
    assert_eq!(d.action, PolicyAction::FailClosed);
}

#[test]
fn ctrl_posterior_normalized() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    let d = ctrl.decide(DecisionSignals::new(5.0, 0.3, 0.2));
    let sum: f64 = d.posterior.iter().sum();
    assert!((sum - 1.0).abs() < 1e-9);
}

#[test]
fn ctrl_posterior_nonneg() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    let d = ctrl.decide(DecisionSignals::new(5.0, 0.3, 0.2));
    for p in &d.posterior {
        assert!(*p >= 0.0);
    }
}

#[test]
fn ctrl_expected_losses_nonneg() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    let d = ctrl.decide(DecisionSignals::new(5.0, 0.3, 0.2));
    for l in &d.expected_losses {
        assert!(*l >= 0.0, "expected loss must be non-negative: {l}");
    }
}

#[test]
fn ctrl_reason_contains_mode() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    let d = ctrl.decide(DecisionSignals::new(2.0, 0.0, 0.0));
    assert!(d.reason.contains("Strict"));
}

#[test]
fn ctrl_reason_contains_mode_hardened() {
    let mut ctrl = PolicyController::new(RuntimeMode::Hardened, 16);
    let d = ctrl.decide(DecisionSignals::new(2.0, 0.0, 0.0));
    assert!(d.reason.contains("Hardened"));
}

#[test]
fn ctrl_reason_contains_top_state() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    let d = ctrl.decide(DecisionSignals::new(2.0, 0.0, 0.0));
    assert!(d.reason.contains("Compatible"));
}

#[test]
fn ctrl_ledger_records_decision() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    ctrl.decide(DecisionSignals::new(2.0, 0.0, 0.0));
    assert_eq!(ctrl.ledger().len(), 1);
}

#[test]
fn ctrl_ledger_records_multiple() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    for i in 0..5 {
        ctrl.decide(DecisionSignals::new(i as f64, 0.0, 0.0));
    }
    assert_eq!(ctrl.ledger().len(), 5);
}

#[test]
fn ctrl_ledger_latest_matches_last_decision() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    ctrl.decide(DecisionSignals::new(2.0, 0.0, 0.0));
    let d = ctrl.decide(DecisionSignals::new(0.0, 1.0, 0.0));
    let latest = ctrl.ledger().latest().unwrap();
    assert_eq!(latest.action, d.action);
}

#[test]
fn ctrl_deterministic_same_signals() {
    let s = DecisionSignals::new(5.0, 0.3, 0.2);
    let mut c1 = PolicyController::new(RuntimeMode::Strict, 16);
    let mut c2 = PolicyController::new(RuntimeMode::Strict, 16);
    assert_eq!(c1.decide(s).action, c2.decide(s).action);
}

#[test]
fn ctrl_deterministic_hardened() {
    let s = DecisionSignals::new(5.0, 0.3, 0.2);
    let mut c1 = PolicyController::new(RuntimeMode::Hardened, 16);
    let mut c2 = PolicyController::new(RuntimeMode::Hardened, 16);
    assert_eq!(c1.decide(s).action, c2.decide(s).action);
}

#[test]
fn ctrl_mode_accessor() {
    let ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    assert_eq!(ctrl.mode(), RuntimeMode::Strict);
}

#[test]
fn ctrl_ledger_accessor_empty() {
    let ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    assert!(ctrl.ledger().is_empty());
}

// Parametric: test 20 different signal combinations
macro_rules! ctrl_signal_test {
    ($name:ident, $mode:expr, $cond:expr, $meta:expr, $anom:expr, $expected_action:expr) => {
        #[test]
        fn $name() {
            let mut ctrl = PolicyController::new($mode, 16);
            let d = ctrl.decide(DecisionSignals::new($cond, $meta, $anom));
            assert_eq!(d.action, $expected_action);
        }
    };
}

ctrl_signal_test!(
    ctrl_strict_c0_m0_a0,
    RuntimeMode::Strict,
    0.0,
    0.0,
    0.0,
    PolicyAction::Allow
);
ctrl_signal_test!(
    ctrl_strict_c2_m0_a0,
    RuntimeMode::Strict,
    2.0,
    0.0,
    0.0,
    PolicyAction::Allow
);
ctrl_signal_test!(
    ctrl_strict_c0_m1_a0,
    RuntimeMode::Strict,
    0.0,
    1.0,
    0.0,
    PolicyAction::FailClosed
);
ctrl_signal_test!(
    ctrl_strict_c0_m0_a1,
    RuntimeMode::Strict,
    0.0,
    0.0,
    1.0,
    PolicyAction::FullValidate
);
ctrl_signal_test!(
    ctrl_strict_c8_m1_a02,
    RuntimeMode::Strict,
    8.0,
    1.0,
    0.2,
    PolicyAction::FailClosed
);
ctrl_signal_test!(
    ctrl_hard_c0_m0_a0,
    RuntimeMode::Hardened,
    0.0,
    0.0,
    0.0,
    PolicyAction::Allow
);
ctrl_signal_test!(
    ctrl_hard_c2_m0_a0,
    RuntimeMode::Hardened,
    2.0,
    0.0,
    0.0,
    PolicyAction::Allow
);
ctrl_signal_test!(
    ctrl_hard_c0_m1_a0,
    RuntimeMode::Hardened,
    0.0,
    1.0,
    0.0,
    PolicyAction::FailClosed
);
ctrl_signal_test!(
    ctrl_hard_c8_m1_a02,
    RuntimeMode::Hardened,
    8.0,
    1.0,
    0.2,
    PolicyAction::FailClosed
);
ctrl_signal_test!(
    ctrl_strict_c16_m0_a0,
    RuntimeMode::Strict,
    16.0,
    0.0,
    0.0,
    PolicyAction::FullValidate
);
ctrl_signal_test!(
    ctrl_hard_c16_m0_a0,
    RuntimeMode::Hardened,
    16.0,
    0.0,
    0.0,
    PolicyAction::FullValidate
);

// Boundary: mid-range metadata
#[test]
fn ctrl_strict_mid_meta_025() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    let d = ctrl.decide(DecisionSignals::new(0.0, 0.25, 0.0));
    // With low cond and anomaly, mid metadata → still Allow or FullValidate
    assert!(d.action == PolicyAction::Allow || d.action == PolicyAction::FullValidate);
}

#[test]
fn ctrl_hardened_mid_meta_03_anom_03() {
    let mut ctrl = PolicyController::new(RuntimeMode::Hardened, 16);
    let d = ctrl.decide(DecisionSignals::new(12.0, 0.25, 0.3));
    assert_eq!(d.action, PolicyAction::FullValidate);
}

// PolicyDecision fields
#[test]
fn decision_has_all_fields() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    let d = ctrl.decide(DecisionSignals::new(2.0, 0.0, 0.0));
    assert_eq!(d.mode, RuntimeMode::Strict);
    assert_eq!(d.posterior.len(), 3);
    assert_eq!(d.expected_losses.len(), 3);
    assert!(!d.reason.is_empty());
}

#[test]
fn decision_serde_roundtrip() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    let d = ctrl.decide(DecisionSignals::new(2.0, 0.1, 0.1));
    let json = serde_json::to_string(&d).unwrap();
    let back: PolicyDecision = serde_json::from_str(&json).unwrap();
    assert_eq!(back.action, d.action);
    assert_eq!(back.mode, d.mode);
}

#[test]
fn decision_clone() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    let d = ctrl.decide(DecisionSignals::new(2.0, 0.0, 0.0));
    let d2 = d.clone();
    assert_eq!(d, d2);
}

#[test]
fn decision_debug() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    let d = ctrl.decide(DecisionSignals::new(2.0, 0.0, 0.0));
    let dbg = format!("{d:?}");
    assert!(dbg.contains("PolicyDecision"));
}

// Multiple sequential decisions
#[test]
fn ctrl_sequential_decisions_independent() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    let d1 = ctrl.decide(DecisionSignals::new(2.0, 0.0, 0.0));
    let d2 = ctrl.decide(DecisionSignals::new(0.0, 1.0, 0.0));
    let d3 = ctrl.decide(DecisionSignals::new(2.0, 0.0, 0.0));
    // d1 and d3 should be identical (decisions are independent of history)
    assert_eq!(d1.action, d3.action);
    assert_ne!(d1.action, d2.action);
}

// Negative signals clamped
#[test]
fn ctrl_negative_signals_clamped() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    let d = ctrl.decide(DecisionSignals::new(-100.0, -1.0, -1.0));
    // All signals clamp to 0, so result should be same as all-zero
    let mut ctrl2 = PolicyController::new(RuntimeMode::Strict, 16);
    let d2 = ctrl2.decide(DecisionSignals::new(0.0, 0.0, 0.0));
    assert_eq!(d.action, d2.action);
}

// Very large condition number clamped
#[test]
fn ctrl_huge_condition_clamped() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    let d1 = ctrl.decide(DecisionSignals::new(1000.0, 0.0, 0.0));
    let mut ctrl2 = PolicyController::new(RuntimeMode::Strict, 16);
    let d2 = ctrl2.decide(DecisionSignals::new(16.0, 0.0, 0.0));
    // Both should produce same result due to clamping
    assert_eq!(d1.action, d2.action);
}

// Both modes agree on extreme signals
#[test]
fn ctrl_modes_agree_on_extreme_benign() {
    let s = DecisionSignals::new(0.0, 0.0, 0.0);
    let mut strict = PolicyController::new(RuntimeMode::Strict, 16);
    let mut hard = PolicyController::new(RuntimeMode::Hardened, 16);
    assert_eq!(strict.decide(s).action, hard.decide(s).action);
}

#[test]
fn ctrl_modes_agree_on_extreme_hostile() {
    let s = DecisionSignals::new(100.0, 1.0, 1.0);
    let mut strict = PolicyController::new(RuntimeMode::Strict, 16);
    let mut hard = PolicyController::new(RuntimeMode::Hardened, 16);
    assert_eq!(strict.decide(s).action, hard.decide(s).action);
}

// ═══════════════════════════════════════════════════════════════════
// §5  PolicyEvidenceLedger (30 tests)
// ═══════════════════════════════════════════════════════════════════

fn make_entry(action: PolicyAction) -> DecisionEvidenceEntry {
    DecisionEvidenceEntry {
        mode: RuntimeMode::Strict,
        signals: DecisionSignals::new(0.0, 0.0, 0.0),
        logits: [0.0; 3],
        posterior: [1.0, 0.0, 0.0],
        expected_losses: [0.0; 3],
        action,
        top_state: RiskState::Compatible,
        reason: "test".into(),
    }
}

#[test]
fn ledger_new_empty() {
    let l = PolicyEvidenceLedger::new(10);
    assert!(l.is_empty());
    assert_eq!(l.len(), 0);
}

#[test]
fn ledger_capacity() {
    let l = PolicyEvidenceLedger::new(42);
    assert_eq!(l.capacity(), 42);
}

#[test]
fn ledger_min_capacity_one() {
    let l = PolicyEvidenceLedger::new(0);
    assert_eq!(l.capacity(), 1);
}

#[test]
fn ledger_record_one() {
    let mut l = PolicyEvidenceLedger::new(10);
    l.record(make_entry(PolicyAction::Allow));
    assert_eq!(l.len(), 1);
}

#[test]
fn ledger_record_fills_to_capacity() {
    let mut l = PolicyEvidenceLedger::new(3);
    for _ in 0..3 {
        l.record(make_entry(PolicyAction::Allow));
    }
    assert_eq!(l.len(), 3);
}

#[test]
fn ledger_fifo_eviction() {
    let mut l = PolicyEvidenceLedger::new(2);
    l.record(make_entry(PolicyAction::Allow));
    l.record(make_entry(PolicyAction::FullValidate));
    l.record(make_entry(PolicyAction::FailClosed));
    assert_eq!(l.len(), 2);
    // Oldest (Allow) evicted, latest should be FailClosed
    assert_eq!(l.latest().unwrap().action, PolicyAction::FailClosed);
}

#[test]
fn ledger_latest_empty() {
    let l = PolicyEvidenceLedger::new(10);
    assert!(l.latest().is_none());
}

#[test]
fn ledger_latest_single() {
    let mut l = PolicyEvidenceLedger::new(10);
    l.record(make_entry(PolicyAction::FullValidate));
    assert_eq!(l.latest().unwrap().action, PolicyAction::FullValidate);
}

#[test]
fn ledger_latest_after_many() {
    let mut l = PolicyEvidenceLedger::new(10);
    l.record(make_entry(PolicyAction::Allow));
    l.record(make_entry(PolicyAction::FailClosed));
    assert_eq!(l.latest().unwrap().action, PolicyAction::FailClosed);
}

#[test]
fn ledger_is_empty_after_record() {
    let mut l = PolicyEvidenceLedger::new(10);
    l.record(make_entry(PolicyAction::Allow));
    assert!(!l.is_empty());
}

#[test]
fn ledger_capacity_one_works() {
    let mut l = PolicyEvidenceLedger::new(1);
    l.record(make_entry(PolicyAction::Allow));
    l.record(make_entry(PolicyAction::FailClosed));
    assert_eq!(l.len(), 1);
    assert_eq!(l.latest().unwrap().action, PolicyAction::FailClosed);
}

#[test]
fn ledger_serde_roundtrip() {
    let mut l = PolicyEvidenceLedger::new(10);
    l.record(make_entry(PolicyAction::Allow));
    let json = serde_json::to_string(&l).unwrap();
    let back: PolicyEvidenceLedger = serde_json::from_str(&json).unwrap();
    assert_eq!(back.len(), 1);
}

#[test]
fn ledger_clone() {
    let mut l = PolicyEvidenceLedger::new(10);
    l.record(make_entry(PolicyAction::Allow));
    let l2 = l.clone();
    assert_eq!(l, l2);
}

#[test]
fn ledger_debug() {
    let l = PolicyEvidenceLedger::new(10);
    let dbg = format!("{l:?}");
    assert!(dbg.contains("PolicyEvidenceLedger"));
}

#[test]
fn ledger_stress_100_entries_cap_5() {
    let mut l = PolicyEvidenceLedger::new(5);
    for _ in 0..100 {
        l.record(make_entry(PolicyAction::Allow));
    }
    assert_eq!(l.len(), 5);
}

#[test]
fn ledger_preserves_entry_fields() {
    let mut l = PolicyEvidenceLedger::new(10);
    let entry = DecisionEvidenceEntry {
        mode: RuntimeMode::Hardened,
        signals: DecisionSignals::new(5.0, 0.3, 0.1),
        logits: [1.0, 2.0, 3.0],
        posterior: [0.5, 0.3, 0.2],
        expected_losses: [10.0, 20.0, 30.0],
        action: PolicyAction::FullValidate,
        top_state: RiskState::IllConditioned,
        reason: "test reason".into(),
    };
    l.record(entry.clone());
    let latest = l.latest().unwrap();
    assert_eq!(latest.mode, RuntimeMode::Hardened);
    assert_eq!(latest.top_state, RiskState::IllConditioned);
    assert_eq!(latest.reason, "test reason");
}

// ═══════════════════════════════════════════════════════════════════
// §6  RiskState & PolicyAction enums (15 tests)
// ═══════════════════════════════════════════════════════════════════

const RISK_STATES: [RiskState; 3] = [
    RiskState::Compatible,
    RiskState::IllConditioned,
    RiskState::IncompatibleMetadata,
];

const POLICY_ACTIONS: [PolicyAction; 3] = [
    PolicyAction::Allow,
    PolicyAction::FullValidate,
    PolicyAction::FailClosed,
];

#[test]
fn risk_state_all_count() {
    assert_eq!(RISK_STATES.len(), 3);
}

#[test]
fn risk_state_all_order() {
    assert_eq!(RISK_STATES[0], RiskState::Compatible);
    assert_eq!(RISK_STATES[1], RiskState::IllConditioned);
    assert_eq!(RISK_STATES[2], RiskState::IncompatibleMetadata);
}

#[test]
fn risk_state_serde_roundtrip() {
    for state in &RISK_STATES {
        let json = serde_json::to_string(state).unwrap();
        let back: RiskState = serde_json::from_str(&json).unwrap();
        assert_eq!(*state, back);
    }
}

#[test]
fn risk_state_debug() {
    assert_eq!(format!("{:?}", RiskState::Compatible), "Compatible");
    assert_eq!(format!("{:?}", RiskState::IllConditioned), "IllConditioned");
}

#[test]
fn risk_state_clone() {
    let s = RiskState::Compatible;
    let s2 = s;
    assert_eq!(s, s2);
}

#[test]
fn policy_action_all_count() {
    assert_eq!(POLICY_ACTIONS.len(), 3);
}

#[test]
fn policy_action_all_order() {
    assert_eq!(POLICY_ACTIONS[0], PolicyAction::Allow);
    assert_eq!(POLICY_ACTIONS[1], PolicyAction::FullValidate);
    assert_eq!(POLICY_ACTIONS[2], PolicyAction::FailClosed);
}

#[test]
fn policy_action_serde_roundtrip() {
    for action in &POLICY_ACTIONS {
        let json = serde_json::to_string(action).unwrap();
        let back: PolicyAction = serde_json::from_str(&json).unwrap();
        assert_eq!(*action, back);
    }
}

#[test]
fn policy_action_debug() {
    assert_eq!(format!("{:?}", PolicyAction::Allow), "Allow");
    assert_eq!(format!("{:?}", PolicyAction::FailClosed), "FailClosed");
}

#[test]
fn policy_action_ne() {
    assert_ne!(PolicyAction::Allow, PolicyAction::FailClosed);
    assert_ne!(PolicyAction::Allow, PolicyAction::FullValidate);
    assert_ne!(PolicyAction::FullValidate, PolicyAction::FailClosed);
}

#[test]
fn risk_state_ne() {
    assert_ne!(RiskState::Compatible, RiskState::IllConditioned);
    assert_ne!(RiskState::Compatible, RiskState::IncompatibleMetadata);
    assert_ne!(RiskState::IllConditioned, RiskState::IncompatibleMetadata);
}

#[test]
fn evidence_entry_serde_roundtrip() {
    let entry = make_entry(PolicyAction::Allow);
    let json = serde_json::to_string(&entry).unwrap();
    let back: DecisionEvidenceEntry = serde_json::from_str(&json).unwrap();
    assert_eq!(back.action, entry.action);
}

#[test]
fn evidence_entry_clone() {
    let entry = make_entry(PolicyAction::FullValidate);
    let entry2 = entry.clone();
    assert_eq!(entry, entry2);
}

#[test]
fn evidence_entry_debug() {
    let entry = make_entry(PolicyAction::Allow);
    let dbg = format!("{entry:?}");
    assert!(dbg.contains("DecisionEvidenceEntry"));
}

#[test]
fn evidence_entry_eq() {
    let a = make_entry(PolicyAction::Allow);
    let b = make_entry(PolicyAction::Allow);
    assert_eq!(a, b);
}

// ═══════════════════════════════════════════════════════════════════
// §7  SolverPortfolio (40 tests)
// ═══════════════════════════════════════════════════════════════════

fn state_to_rcond(state: &MatrixConditionState) -> f64 {
    match state {
        MatrixConditionState::WellConditioned => 1e-2,
        MatrixConditionState::ModerateCondition => 1e-6,
        MatrixConditionState::IllConditioned => 1e-11,
        MatrixConditionState::NearSingular => 1e-16,
    }
}

#[test]
fn portfolio_new_strict() {
    let p = SolverPortfolio::new(RuntimeMode::Strict, 64);
    assert_eq!(p.mode(), RuntimeMode::Strict);
    assert_eq!(p.evidence_len(), 0);
}

#[test]
fn portfolio_new_hardened() {
    let p = SolverPortfolio::new(RuntimeMode::Hardened, 64);
    assert_eq!(p.mode(), RuntimeMode::Hardened);
}

#[test]
fn portfolio_lu_for_well_conditioned() {
    let p = SolverPortfolio::new(RuntimeMode::Strict, 64);
    let (a, _, _, _) = p.select_action(1e-2, None);
    assert_eq!(a, SolverAction::DirectLU);
}

// frankenscipy-7tb8d.2: with the calibrated loss matrix, LU has the least expected loss at
// every conditioning; the hand-set matrix sent moderate conditioning to QR and ill
// conditioning to the SVD, which the measurements do not support.
#[test]
fn portfolio_lu_at_every_conditioning() {
    for mode in [RuntimeMode::Strict, RuntimeMode::Hardened] {
        let p = SolverPortfolio::new(mode, 64);
        for state in &MatrixConditionState::ALL {
            let (a, _, _, _) = p.select_action(state_to_rcond(state), None);
            assert_eq!(a, SolverAction::DirectLU, "{mode:?} {state:?}");
        }
    }
}

// frankenscipy-7tb8d.2: the fallback order the calibration implies: QR once LU has failed its
// backward error, the SVD only once QR has too.
#[test]
fn portfolio_fallback_order_is_qr_then_svd() {
    let p = SolverPortfolio::new(RuntimeMode::Strict, 64);
    for state in &MatrixConditionState::ALL {
        let rcond = state_to_rcond(state);
        let (second, ..) = p
            .select_action_excluding(rcond, None, &[SolverAction::DirectLU])
            .expect("QR and SVD remain");
        assert_eq!(second, SolverAction::PivotedQR, "{state:?}");
        let (third, ..) = p
            .select_action_excluding(
                rcond,
                None,
                &[SolverAction::DirectLU, SolverAction::PivotedQR],
            )
            .expect("SVD remains");
        assert_eq!(third, SolverAction::SVDFallback, "{state:?}");
    }
}

#[test]
fn portfolio_hardened_lu_for_well() {
    let p = SolverPortfolio::new(RuntimeMode::Hardened, 64);
    let (a, _, _, _) = p.select_action(1e-2, None);
    assert_eq!(a, SolverAction::DirectLU);
}

#[test]
fn portfolio_posterior_well_conditioned() {
    let p = SolverPortfolio::new(RuntimeMode::Strict, 64);
    let (_, posterior, _, _) = p.select_action(1e-2, None);
    assert_eq!(posterior, [1.0, 0.0, 0.0, 0.0]);
}

#[test]
fn portfolio_posterior_moderate() {
    let p = SolverPortfolio::new(RuntimeMode::Strict, 64);
    let (_, posterior, _, _) = p.select_action(1e-6, None);
    assert_eq!(posterior, [0.0, 1.0, 0.0, 0.0]);
}

#[test]
fn portfolio_posterior_ill() {
    let p = SolverPortfolio::new(RuntimeMode::Strict, 64);
    let (_, posterior, _, _) = p.select_action(1e-11, None);
    assert_eq!(posterior, [0.0, 0.0, 1.0, 0.0]);
}

#[test]
fn portfolio_posterior_near_singular() {
    let p = SolverPortfolio::new(RuntimeMode::Strict, 64);
    let (_, posterior, _, _) = p.select_action(1e-16, None);
    assert_eq!(posterior, [0.0, 0.0, 0.0, 1.0]);
}

/// At a state's center the posterior is one-hot, so the expected losses are that state's
/// column of the loss matrix, exactly. The matrix is measured (frankenscipy-7tb8d.2), so the
/// comparison is with it, not with typed numbers.
fn assert_losses_are_column(rcond: f64, state: usize) {
    let p = SolverPortfolio::new(RuntimeMode::Strict, 64);
    let (_, posterior, losses, _) = p.select_action(rcond, None);
    let mut one_hot = [0.0; 4];
    one_hot[state] = 1.0;
    assert_eq!(posterior, one_hot, "rcond {rcond:e}");
    let m = SolverPortfolio::default_loss_matrix();
    for (a, &loss) in losses.iter().enumerate() {
        assert_eq!(
            loss.to_bits(),
            m[a][state].to_bits(),
            "{:?} at rcond {rcond:e}: {loss:e} vs {:e}",
            SolverAction::ALL[a],
            m[a][state]
        );
    }
}

#[test]
fn portfolio_expected_losses_well() {
    assert_losses_are_column(1e-2, 0);
}

#[test]
fn portfolio_chosen_loss_is_minimum() {
    let p = SolverPortfolio::new(RuntimeMode::Strict, 64);
    for state in &MatrixConditionState::ALL {
        let (action, _, losses, chosen_loss) = p.select_action(state_to_rcond(state), None);
        // chosen_loss should be losses[action.index()] for general solvers
        let idx = action.index();
        assert_close(chosen_loss, losses[idx], 1e-12, 0.0);
    }
}

#[test]
fn portfolio_default_loss_matrix_shape() {
    let m = SolverPortfolio::default_loss_matrix();
    assert_eq!(m.len(), 6);
    for row in &m {
        assert_eq!(row.len(), 4);
    }
}

/// The ranking the calibration measured (frankenscipy-7tb8d.2), state by state: LU no costlier
/// than QR, QR cheaper than the SVD, the diagonal and triangular fast paths no costlier than
/// LU, and every loss at least its action's flop cost (LU 1, QR 2, SVD 31.5).
fn calibrated_ranking_holds(m: &[[f64; 4]; 6]) -> bool {
    (0..4).all(|s| {
        m[0][s] <= m[1][s]
            && m[1][s] < m[2][s]
            && m[3][s] <= m[0][s]
            && m[4][s] <= m[0][s]
            && m[0][s] >= 1.0
            && m[1][s] >= 2.0
            && m[2][s] >= 31.5
    }) && m.iter().flatten().all(|v| v.is_finite() && *v >= 0.0)
}

#[test]
fn portfolio_default_loss_matrix_values() {
    assert!(calibrated_ranking_holds(
        &SolverPortfolio::default_loss_matrix()
    ));
    // Must-miss: the hand-set matrix the calibration replaced ranks QR first at moderate
    // conditioning and the SVD first at ill conditioning.
    let hand_set = [
        [1.0, 5.0, 40.0, 120.0],
        [3.0, 1.0, 8.0, 45.0],
        [15.0, 10.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 100.0],
        [0.0, 0.0, 0.0, 100.0],
        [0.0, 0.0, 0.0, 100.0],
    ];
    assert!(!calibrated_ranking_holds(&hand_set));
}

fn make_solver_evidence(action: SolverAction) -> SolverEvidenceEntry {
    SolverEvidenceEntry {
        component: "test",
        matrix_shape: (4, 4),
        rcond_estimate: 0.5,
        chosen_action: action,
        posterior: vec![1.0, 0.0, 0.0, 0.0],
        expected_losses: vec![1.0, 3.0, 15.0, 0.0, 0.0],
        chosen_expected_loss: 1.0,
        fallback_active: false,
        backward_error: None,
    }
}

#[test]
fn portfolio_record_evidence() {
    let mut p = SolverPortfolio::new(RuntimeMode::Strict, 64);
    p.record_evidence(make_solver_evidence(SolverAction::DirectLU));
    assert_eq!(p.evidence_len(), 1);
}

#[test]
fn portfolio_evidence_bounded() {
    let mut p = SolverPortfolio::new(RuntimeMode::Strict, 3);
    for _ in 0..10 {
        p.record_evidence(make_solver_evidence(SolverAction::DirectLU));
    }
    assert_eq!(p.evidence_len(), 3);
}

#[test]
fn portfolio_evidence_min_capacity_one() {
    let mut p = SolverPortfolio::new(RuntimeMode::Strict, 0);
    p.record_evidence(make_solver_evidence(SolverAction::DirectLU));
    p.record_evidence(make_solver_evidence(SolverAction::PivotedQR));
    assert_eq!(p.evidence_len(), 1);
}

#[test]
fn portfolio_jsonl_empty() {
    let p = SolverPortfolio::new(RuntimeMode::Strict, 64);
    assert!(p.serialize_jsonl().is_empty());
}

#[test]
fn portfolio_jsonl_single() {
    let mut p = SolverPortfolio::new(RuntimeMode::Strict, 64);
    p.record_evidence(make_solver_evidence(SolverAction::DirectLU));
    let jsonl = p.serialize_jsonl();
    let parsed: serde_json::Value = serde_json::from_str(&jsonl).unwrap();
    assert_eq!(parsed["component"], "test");
}

#[test]
fn portfolio_jsonl_multiple() {
    let mut p = SolverPortfolio::new(RuntimeMode::Strict, 64);
    p.record_evidence(make_solver_evidence(SolverAction::DirectLU));
    p.record_evidence(make_solver_evidence(SolverAction::PivotedQR));
    let jsonl = p.serialize_jsonl();
    let lines: Vec<&str> = jsonl.lines().collect();
    assert_eq!(lines.len(), 2);
}

#[test]
fn portfolio_calibrator_accessor() {
    let p = SolverPortfolio::new(RuntimeMode::Strict, 64);
    assert!(!p.calibrator().should_fallback());
}

#[test]
fn portfolio_observe_backward_error() {
    let mut p = SolverPortfolio::new(RuntimeMode::Strict, 64);
    p.observe_backward_error(1e-15);
    assert_eq!(p.calibrator().total_predictions(), 1);
}

#[test]
fn portfolio_calibrator_fallback_overrides() {
    let mut p = SolverPortfolio::new(RuntimeMode::Strict, 64);
    // Feed enough violations to trigger fallback
    for _ in 0..50 {
        p.observe_backward_error(1e-15);
    }
    for _ in 0..10 {
        p.observe_backward_error(1.0);
    }
    // Should now fallback
    let (action, _, _, _) = p.select_action(1e-2, None);
    assert_eq!(action, SolverAction::SVDFallback);
}

#[test]
fn portfolio_calibrator_nan_backward_error_falls_back() {
    let mut p = SolverPortfolio::new(RuntimeMode::Strict, 64);
    for _ in 0..10 {
        p.observe_backward_error(f64::NAN);
    }
    let (action, _, _, _) = p.select_action(1e-2, None);
    assert_eq!(action, SolverAction::SVDFallback);
}

#[test]
fn portfolio_deterministic_selection() {
    let p1 = SolverPortfolio::new(RuntimeMode::Strict, 64);
    let p2 = SolverPortfolio::new(RuntimeMode::Strict, 64);
    for state in &MatrixConditionState::ALL {
        let (a1, _, _, _) = p1.select_action(state_to_rcond(state), None);
        let (a2, _, _, _) = p2.select_action(state_to_rcond(state), None);
        assert_eq!(a1, a2);
    }
}

#[test]
fn portfolio_clone() {
    let p = SolverPortfolio::new(RuntimeMode::Strict, 64);
    let p2 = p.clone();
    let (a1, _, _, _) = p.select_action(1e-2, None);
    let (a2, _, _, _) = p2.select_action(1e-2, None);
    assert_eq!(a1, a2);
}

// ═══════════════════════════════════════════════════════════════════
// §8  MatrixConditionState & SolverAction enums (20 tests)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn mcs_all_count() {
    assert_eq!(MatrixConditionState::ALL.len(), 4);
}

#[test]
fn mcs_index_well() {
    assert_eq!(MatrixConditionState::WellConditioned.index(), 0);
}

#[test]
fn mcs_index_moderate() {
    assert_eq!(MatrixConditionState::ModerateCondition.index(), 1);
}

#[test]
fn mcs_index_ill() {
    assert_eq!(MatrixConditionState::IllConditioned.index(), 2);
}

#[test]
fn mcs_index_near_singular() {
    assert_eq!(MatrixConditionState::NearSingular.index(), 3);
}

#[test]
fn mcs_serde_roundtrip() {
    for s in &MatrixConditionState::ALL {
        let json = serde_json::to_string(s).unwrap();
        let back: MatrixConditionState = serde_json::from_str(&json).unwrap();
        assert_eq!(*s, back);
    }
}

#[test]
fn mcs_debug() {
    assert_eq!(
        format!("{:?}", MatrixConditionState::WellConditioned),
        "WellConditioned"
    );
}

#[test]
fn sa_all_count() {
    assert_eq!(SolverAction::ALL.len(), 6);
}

#[test]
fn sa_index_lu() {
    assert_eq!(SolverAction::DirectLU.index(), 0);
}

#[test]
fn sa_index_qr() {
    assert_eq!(SolverAction::PivotedQR.index(), 1);
}

#[test]
fn sa_index_svd() {
    assert_eq!(SolverAction::SVDFallback.index(), 2);
}

#[test]
fn sa_index_diag() {
    assert_eq!(SolverAction::DiagonalFastPath.index(), 3);
}

#[test]
fn sa_index_tri() {
    assert_eq!(SolverAction::TriangularFastPath.index(), 4);
}

#[test]
fn sa_index_symmetric() {
    assert_eq!(SolverAction::SymmetricFastPath.index(), 5);
    assert_eq!(SolverAction::ALL[5], SolverAction::SymmetricFastPath);
}

#[test]
fn sa_serde_roundtrip() {
    for a in &SolverAction::ALL {
        let json = serde_json::to_string(a).unwrap();
        let back: SolverAction = serde_json::from_str(&json).unwrap();
        assert_eq!(*a, back);
    }
}

#[test]
fn sa_debug() {
    assert_eq!(format!("{:?}", SolverAction::DirectLU), "DirectLU");
}

#[test]
fn sa_ne() {
    assert_ne!(SolverAction::DirectLU, SolverAction::PivotedQR);
}

#[test]
fn sa_clone() {
    let a = SolverAction::SVDFallback;
    let b = a;
    assert_eq!(a, b);
}

#[test]
fn solver_evidence_entry_serializes() {
    let entry = make_solver_evidence(SolverAction::DirectLU);
    let json = serde_json::to_string(&entry).unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["component"], "test");
    assert_eq!(v["chosen_action"], "DirectLU");
}

#[test]
fn solver_evidence_entry_clone() {
    let entry = make_solver_evidence(SolverAction::PivotedQR);
    let entry2 = entry.clone();
    assert_eq!(entry, entry2);
}

#[test]
fn solver_evidence_entry_fields() {
    let entry = make_solver_evidence(SolverAction::SVDFallback);
    assert_eq!(entry.component, "test");
    assert_eq!(entry.matrix_shape, (4, 4));
    assert_eq!(entry.rcond_estimate, 0.5);
    assert!(!entry.fallback_active);
}

// ═══════════════════════════════════════════════════════════════════
// §9  ConformalCalibrator (35 tests)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn cal_new_defaults() {
    let cal = ConformalCalibrator::new(0.05, 100);
    assert!(!cal.should_fallback());
    assert_eq!(cal.empirical_miscoverage(), 0.0);
    assert_eq!(cal.total_predictions(), 0);
    assert_close(cal.alpha(), 0.05, 1e-12, 0.0);
}

#[test]
fn cal_alpha_clamped_low() {
    let cal = ConformalCalibrator::new(0.0, 100);
    assert_close(cal.alpha(), 0.001, 1e-12, 0.0);
}

#[test]
fn cal_alpha_clamped_high() {
    let cal = ConformalCalibrator::new(1.0, 100);
    assert_close(cal.alpha(), 0.5, 1e-12, 0.0);
}

#[test]
fn cal_capacity_min_10() {
    let mut cal = ConformalCalibrator::new(0.05, 1);
    for _ in 0..20 {
        cal.observe(0.0);
    }
    // capacity clamped to min 10
    assert_eq!(cal.total_predictions(), 20);
}

#[test]
fn cal_no_fallback_few_scores() {
    let mut cal = ConformalCalibrator::new(0.05, 100);
    for _ in 0..5 {
        cal.observe(1.0); // all violations
    }
    // < 10 scores, should never fallback
    assert!(!cal.should_fallback());
}

#[test]
fn cal_no_fallback_all_good() {
    let mut cal = ConformalCalibrator::new(0.05, 100);
    for _ in 0..50 {
        cal.observe(1e-15);
    }
    assert!(!cal.should_fallback());
}

#[test]
fn cal_fallback_high_violations() {
    let mut cal = ConformalCalibrator::new(0.05, 100);
    for _ in 0..50 {
        cal.observe(1e-15);
    }
    for _ in 0..10 {
        cal.observe(1.0);
    }
    // 10/60 ≈ 0.167 > 0.07
    assert!(cal.should_fallback());
}

#[test]
fn cal_miscoverage_zero_initially() {
    let cal = ConformalCalibrator::new(0.05, 100);
    assert_eq!(cal.empirical_miscoverage(), 0.0);
}

#[test]
fn cal_miscoverage_all_good() {
    let mut cal = ConformalCalibrator::new(0.05, 100);
    for _ in 0..20 {
        cal.observe(0.0);
    }
    assert_eq!(cal.empirical_miscoverage(), 0.0);
}

#[test]
fn cal_miscoverage_all_bad() {
    let mut cal = ConformalCalibrator::new(0.05, 100);
    for _ in 0..20 {
        cal.observe(1.0);
    }
    assert_close(cal.empirical_miscoverage(), 1.0, 1e-12, 0.0);
}

#[test]
fn cal_miscoverage_half() {
    let mut cal = ConformalCalibrator::new(0.05, 100);
    for _ in 0..10 {
        cal.observe(0.0);
    }
    for _ in 0..10 {
        cal.observe(1.0);
    }
    assert_close(cal.empirical_miscoverage(), 0.5, 1e-12, 0.0);
}

#[test]
fn cal_total_predictions_tracks() {
    let mut cal = ConformalCalibrator::new(0.05, 100);
    for _ in 0..25 {
        cal.observe(0.0);
    }
    assert_eq!(cal.total_predictions(), 25);
}

#[test]
fn cal_bounded_capacity() {
    let mut cal = ConformalCalibrator::new(0.05, 20);
    for _ in 0..100 {
        cal.observe(0.0);
    }
    assert_eq!(cal.total_predictions(), 100);
}

#[test]
fn cal_set_violation_threshold() {
    let mut cal = ConformalCalibrator::new(0.05, 100);
    cal.set_violation_threshold(0.5);
    for _ in 0..20 {
        cal.observe(0.3); // below new threshold
    }
    assert_eq!(cal.empirical_miscoverage(), 0.0);
}

#[test]
fn cal_set_violation_threshold_negative_clamps() {
    let mut cal = ConformalCalibrator::new(0.05, 100);
    cal.set_violation_threshold(-1.0);
    // threshold clamped to 0.0, so any positive score is a violation
    cal.observe(0.001);
    assert!(cal.empirical_miscoverage() > 0.0);
}

#[test]
fn cal_clone() {
    let mut cal = ConformalCalibrator::new(0.05, 100);
    cal.observe(0.0);
    let cal2 = cal.clone();
    assert_eq!(cal.total_predictions(), cal2.total_predictions());
}

#[test]
fn cal_debug() {
    let cal = ConformalCalibrator::new(0.05, 100);
    let dbg = format!("{cal:?}");
    assert!(dbg.contains("ConformalCalibrator"));
}

#[test]
fn cal_eviction_reduces_violations() {
    let mut cal = ConformalCalibrator::new(0.05, 20);
    // Fill with all violations
    for _ in 0..20 {
        cal.observe(1.0);
    }
    assert!(cal.should_fallback());
    // Now feed good scores, evicting bad ones
    for _ in 0..20 {
        cal.observe(0.0);
    }
    assert!(!cal.should_fallback());
}

#[test]
fn cal_exact_threshold_boundary() {
    let mut cal = ConformalCalibrator::new(0.05, 100);
    // alpha + epsilon = 0.07. Need exactly 7% violations to be at boundary
    for _ in 0..93 {
        cal.observe(0.0);
    }
    for _ in 0..7 {
        cal.observe(1.0);
    }
    // 7/100 = 0.07, NOT > 0.07, so no fallback
    assert!(!cal.should_fallback());
}

#[test]
fn cal_just_over_threshold() {
    let mut cal = ConformalCalibrator::new(0.05, 100);
    for _ in 0..92 {
        cal.observe(0.0);
    }
    for _ in 0..8 {
        cal.observe(1.0);
    }
    // 8/100 = 0.08 > 0.07 → fallback
    assert!(cal.should_fallback());
}

#[test]
fn cal_alpha_05_needs_more_violations() {
    let mut cal = ConformalCalibrator::new(0.5, 100);
    for _ in 0..47 {
        cal.observe(0.0);
    }
    for _ in 0..53 {
        cal.observe(1.0);
    }
    // 53/100 = 0.53 > 0.50 + 0.02 = 0.52 → fallback
    assert!(cal.should_fallback());
}

// ═══════════════════════════════════════════════════════════════════
// §10  Adversarial Sequences (25 tests)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn adv_rapid_oscillation() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 100);
    for i in 0..20 {
        let meta = if i % 2 == 0 { 0.0 } else { 1.0 };
        let d = ctrl.decide(DecisionSignals::new(2.0, meta, 0.0));
        if meta > 0.5 {
            assert_eq!(d.action, PolicyAction::FailClosed);
        } else {
            assert_eq!(d.action, PolicyAction::Allow);
        }
    }
    assert_eq!(ctrl.ledger().len(), 20);
}

#[test]
fn adv_rapid_oscillation_hardened() {
    let mut ctrl = PolicyController::new(RuntimeMode::Hardened, 100);
    for i in 0..20 {
        let meta = if i % 2 == 0 { 0.0 } else { 1.0 };
        let d = ctrl.decide(DecisionSignals::new(2.0, meta, 0.0));
        if meta > 0.5 {
            assert_eq!(d.action, PolicyAction::FailClosed);
        } else {
            assert_eq!(d.action, PolicyAction::Allow);
        }
    }
}

#[test]
fn adv_gradual_escalation() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 100);
    let mut saw_failclosed = false;
    for i in 0..100 {
        let meta = i as f64 / 100.0;
        let d = ctrl.decide(DecisionSignals::new(0.0, meta, 0.0));
        if d.action == PolicyAction::FailClosed {
            saw_failclosed = true;
        }
    }
    assert!(
        saw_failclosed,
        "must eventually fail-closed as metadata rises"
    );
}

#[test]
fn adv_gradual_cond_escalation() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 100);
    let mut saw_validate = false;
    for i in 0..100 {
        let cond = i as f64 * 0.5;
        let d = ctrl.decide(DecisionSignals::new(cond, 0.0, 0.0));
        if d.action == PolicyAction::FullValidate {
            saw_validate = true;
        }
    }
    assert!(
        saw_validate,
        "must eventually full-validate as condition rises"
    );
}

#[test]
fn adv_all_extreme_values() {
    let extremes = [0.0, 1.0, 16.0, 100.0, f64::MAX, f64::MIN_POSITIVE];
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 1000);
    for &c in &extremes {
        for &m in &extremes {
            for &a in &extremes {
                if DecisionSignals::new(c, m, a).is_finite() {
                    let d = ctrl.decide(DecisionSignals::new(c, m, a));
                    // Must always produce valid posterior
                    let sum: f64 = d.posterior.iter().sum();
                    assert!((sum - 1.0).abs() < 1e-9);
                }
            }
        }
    }
}

#[test]
fn adv_evidence_poisoning_attempt() {
    // Simulate evidence poisoning: many benign then sudden hostile
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 100);
    for _ in 0..50 {
        ctrl.decide(DecisionSignals::new(2.0, 0.0, 0.0));
    }
    // Hostile signal must still be independently evaluated
    let d = ctrl.decide(DecisionSignals::new(0.0, 1.0, 0.0));
    assert_eq!(d.action, PolicyAction::FailClosed);
}

#[test]
fn adv_ledger_overflow_then_hostile() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 5);
    for _ in 0..100 {
        ctrl.decide(DecisionSignals::new(2.0, 0.0, 0.0));
    }
    assert_eq!(ctrl.ledger().len(), 5);
    let d = ctrl.decide(DecisionSignals::new(0.0, 1.0, 0.0));
    assert_eq!(d.action, PolicyAction::FailClosed);
}

#[test]
fn adv_nan_through_controller() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    let d = ctrl.decide(DecisionSignals::new(f64::NAN, 0.0, 0.0));
    assert_eq!(ctrl.ledger().len(), 1);
    assert_eq!(d.action, PolicyAction::FailClosed);
    assert_eq!(d.top_state, RiskState::IncompatibleMetadata);
    assert_eq!(d.posterior, [0.0, 0.0, 1.0]);
    assert!(d.expected_losses.iter().all(|loss| loss.is_finite()));
}

#[test]
fn adv_inf_through_controller() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    let d = ctrl.decide(DecisionSignals::new(f64::INFINITY, 0.0, 0.0));
    assert_eq!(ctrl.ledger().len(), 1);
    assert_eq!(d.action, PolicyAction::FailClosed);
    assert_eq!(d.top_state, RiskState::IncompatibleMetadata);
    assert_eq!(d.posterior, [0.0, 0.0, 1.0]);
    assert!(d.expected_losses.iter().all(|loss| loss.is_finite()));
}

#[test]
fn adv_neg_inf_through_controller() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    let d = ctrl.decide(DecisionSignals::new(f64::NEG_INFINITY, 0.0, 0.0));
    assert_eq!(ctrl.ledger().len(), 1);
    assert_eq!(d.action, PolicyAction::FailClosed);
    assert_eq!(d.top_state, RiskState::IncompatibleMetadata);
    assert_eq!(d.posterior, [0.0, 0.0, 1.0]);
    assert!(d.expected_losses.iter().all(|loss| loss.is_finite()));
}

#[test]
fn adv_alternating_modes_same_signals() {
    let s = DecisionSignals::new(5.0, 0.3, 0.2);
    let mut strict = PolicyController::new(RuntimeMode::Strict, 16);
    let mut hard = PolicyController::new(RuntimeMode::Hardened, 16);
    let ds = strict.decide(s);
    let dh = hard.decide(s);
    // Both should produce valid posteriors
    assert!((ds.posterior.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    assert!((dh.posterior.iter().sum::<f64>() - 1.0).abs() < 1e-9);
}

#[test]
fn adv_stress_1000_decisions() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 50);
    for i in 0..1000 {
        let d = ctrl.decide(DecisionSignals::new(
            (i % 20) as f64,
            (i % 10) as f64 / 10.0,
            (i % 5) as f64 / 5.0,
        ));
        assert!(d.posterior.iter().all(|p| *p >= 0.0));
    }
    assert_eq!(ctrl.ledger().len(), 50);
}

#[test]
fn adv_portfolio_oscillation() {
    let p = SolverPortfolio::new(RuntimeMode::Strict, 64);
    for i in 0..100 {
        let state = &MatrixConditionState::ALL[i % 4];
        let (action, _, _, _) = p.select_action(state_to_rcond(state), None);
        // frankenscipy-7tb8d.2: the calibrated choice is LU in every state, so the decision
        // does not oscillate as the state does.
        assert_eq!(action, SolverAction::DirectLU, "{state:?}");
    }
}

#[test]
fn adv_calibrator_flood_good_then_bad() {
    let mut cal = ConformalCalibrator::new(0.05, 50);
    for _ in 0..40 {
        cal.observe(0.0);
    }
    assert!(!cal.should_fallback());
    for _ in 0..10 {
        cal.observe(1.0);
    }
    assert!(cal.should_fallback());
}

#[test]
fn adv_calibrator_recovery() {
    let mut cal = ConformalCalibrator::new(0.05, 30);
    for _ in 0..20 {
        cal.observe(1.0);
    }
    assert!(cal.should_fallback());
    for _ in 0..30 {
        cal.observe(0.0);
    }
    assert!(!cal.should_fallback());
}

// ═══════════════════════════════════════════════════════════════════
// §11  Structured JSON Logging (15 tests)
// ═══════════════════════════════════════════════════════════════════

fn log_decision(test_id: &str, d: &PolicyDecision) -> String {
    let entry = TestLogEntry::new(
        test_id,
        "fsci_runtime::policy",
        format!("action={:?}", d.action),
    )
    .with_result(TestResult::Pass)
    .with_mode(d.mode);
    entry.to_json_line()
}

#[test]
fn log_entry_has_test_id() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    let d = ctrl.decide(DecisionSignals::new(2.0, 0.0, 0.0));
    let json = log_decision("log_entry_has_test_id", &d);
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["test_id"], "log_entry_has_test_id");
}

#[test]
fn log_entry_has_timestamp() {
    let entry = TestLogEntry::new("ts_test", "test", "msg");
    let json = entry.to_json_line();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(v["timestamp_ms"].is_number());
}

#[test]
fn log_entry_has_level() {
    let entry = TestLogEntry::new("lvl_test", "test", "msg");
    let v: serde_json::Value = serde_json::from_str(&entry.to_json_line()).unwrap();
    assert_eq!(v["level"], "info");
}

#[test]
fn log_entry_with_result_pass() {
    let entry = TestLogEntry::new("r_test", "test", "msg").with_result(TestResult::Pass);
    let v: serde_json::Value = serde_json::from_str(&entry.to_json_line()).unwrap();
    assert_eq!(v["result"], "pass");
}

#[test]
fn log_entry_with_result_fail() {
    let entry = TestLogEntry::new("f_test", "test", "msg").with_result(TestResult::Fail);
    let v: serde_json::Value = serde_json::from_str(&entry.to_json_line()).unwrap();
    assert_eq!(v["result"], "fail");
}

#[test]
fn log_entry_with_result_skip() {
    let entry = TestLogEntry::new("s_test", "test", "msg").with_result(TestResult::Skip);
    let v: serde_json::Value = serde_json::from_str(&entry.to_json_line()).unwrap();
    assert_eq!(v["result"], "skip");
}

#[test]
fn log_entry_with_result_warn() {
    let entry = TestLogEntry::new("w_test", "test", "msg").with_result(TestResult::Warn);
    let v: serde_json::Value = serde_json::from_str(&entry.to_json_line()).unwrap();
    assert_eq!(v["result"], "warn");
}

#[test]
fn log_entry_with_seed() {
    let entry = TestLogEntry::new("seed_test", "test", "msg").with_seed(12345);
    let v: serde_json::Value = serde_json::from_str(&entry.to_json_line()).unwrap();
    assert_eq!(v["seed"], 12345);
}

#[test]
fn log_entry_with_mode_strict() {
    let entry = TestLogEntry::new("mode_test", "test", "msg").with_mode(RuntimeMode::Strict);
    let v: serde_json::Value = serde_json::from_str(&entry.to_json_line()).unwrap();
    assert_eq!(v["mode"], "Strict");
}

#[test]
fn log_entry_with_mode_hardened() {
    let entry = TestLogEntry::new("mode_test", "test", "msg").with_mode(RuntimeMode::Hardened);
    let v: serde_json::Value = serde_json::from_str(&entry.to_json_line()).unwrap();
    assert_eq!(v["mode"], "Hardened");
}

#[test]
fn log_entry_with_fixture() {
    let entry = TestLogEntry::new("fix_test", "test", "msg").with_fixture("fixture_001");
    let v: serde_json::Value = serde_json::from_str(&entry.to_json_line()).unwrap();
    assert_eq!(v["fixture_id"], "fixture_001");
}

#[test]
fn log_entry_omits_none_fields() {
    let entry = TestLogEntry::new("omit_test", "test", "msg");
    let v: serde_json::Value = serde_json::from_str(&entry.to_json_line()).unwrap();
    assert!(v.get("seed").is_none());
    assert!(v.get("fixture_id").is_none());
    assert!(v.get("mode").is_none());
    assert!(v.get("result").is_none());
}

#[test]
fn log_entry_full_chain() {
    let entry = TestLogEntry::new("full", "fsci_runtime", "full chain test")
        .with_result(TestResult::Pass)
        .with_seed(42)
        .with_mode(RuntimeMode::Strict)
        .with_fixture("golden_001");
    let v: serde_json::Value = serde_json::from_str(&entry.to_json_line()).unwrap();
    assert_eq!(v["test_id"], "full");
    assert_eq!(v["result"], "pass");
    assert_eq!(v["seed"], 42);
    assert_eq!(v["mode"], "Strict");
    assert_eq!(v["fixture_id"], "golden_001");
}

#[test]
fn log_level_variants() {
    assert_eq!(format!("{:?}", TestLogLevel::Info), "Info");
    assert_eq!(format!("{:?}", TestLogLevel::Warn), "Warn");
    assert_eq!(format!("{:?}", TestLogLevel::Error), "Error");
    assert_eq!(format!("{:?}", TestLogLevel::Debug), "Debug");
}

#[test]
fn log_result_variants() {
    assert_eq!(format!("{:?}", TestResult::Pass), "Pass");
    assert_eq!(format!("{:?}", TestResult::Fail), "Fail");
    assert_eq!(format!("{:?}", TestResult::Skip), "Skip");
    assert_eq!(format!("{:?}", TestResult::Warn), "Warn");
}

// ═══════════════════════════════════════════════════════════════════
// §12  Test helpers (10 tests)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn helper_assert_close_negative() {
    assert_close(-1.0, -1.0, 1e-12, 1e-12);
}

#[test]
fn helper_assert_close_zero() {
    assert_close(0.0, 0.0, 1e-12, 1e-12);
}

#[test]
fn helper_assert_close_large() {
    assert_close(1e15, 1e15, 1.0, 1e-12);
}

#[test]
fn helper_within_tolerance_true() {
    assert!(within_tolerance(1.001, 1.0, 0.01, 0.0));
}

#[test]
fn helper_within_tolerance_false() {
    assert!(!within_tolerance(2.0, 1.0, 0.01, 0.0));
}

#[test]
fn helper_assert_close_slice_3() {
    assert_close_slice(&[0.5, 0.3, 0.2], &[0.5, 0.3, 0.2], 1e-12, 1e-12);
}

#[test]
fn helper_casp_now_nonzero() {
    assert!(casp_now_unix_ms() > 0);
}

#[test]
fn helper_casp_now_monotonic() {
    let t1 = casp_now_unix_ms();
    let t2 = casp_now_unix_ms();
    assert!(t2 >= t1);
}

// ═══════════════════════════════════════════════════════════════════
// §13  Golden-value tests (15 tests)
// ═══════════════════════════════════════════════════════════════════

// Golden: all-zero signals in Strict mode
#[test]
fn golden_strict_zero_signals() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    let d = ctrl.decide(DecisionSignals::new(0.0, 0.0, 0.0));
    // logits: compatible=2.8, ill=-0.4, incompat=-2.0
    // softmax([2.8, -0.4, -2.0])
    let e28 = (2.8f64).exp();
    let em04 = (-0.4f64).exp();
    let em2 = (-2.0f64).exp();
    let denom = e28 + em04 + em2;
    assert_close(d.posterior[0], e28 / denom, 1e-9, 1e-9);
    assert_close(d.posterior[1], em04 / denom, 1e-9, 1e-9);
    assert_close(d.posterior[2], em2 / denom, 1e-9, 1e-9);
}

// Golden: all-one signals (cond=16 for max clamped cond=1)
#[test]
fn golden_strict_unit_signals() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    let d = ctrl.decide(DecisionSignals::new(16.0, 1.0, 1.0));
    // cond=1.0, meta=1.0, anomaly=1.0
    // compatible = 2.8 - 0.8*1 - 3.2*1 - 2.4*1 = 2.8-0.8-3.2-2.4 = -3.6
    // ill = -0.4 + 1.4*1 + 0.6*1 - 0.8*1 = -0.4+1.4+0.6-0.8 = 0.8
    // incompat = -2.0 + 3.5*1 + 0.7*1 = 2.2
    let logits: [f64; 3] = [-3.6, 0.8, 2.2];
    let max_l: f64 = 2.2;
    let exps: Vec<f64> = logits.iter().map(|l| (l - max_l).exp()).collect();
    let denom: f64 = exps.iter().sum();
    assert_close(d.posterior[0], exps[0] / denom, 1e-9, 1e-9);
    assert_close(d.posterior[1], exps[1] / denom, 1e-9, 1e-9);
    assert_close(d.posterior[2], exps[2] / denom, 1e-9, 1e-9);
    assert_eq!(d.action, PolicyAction::FailClosed);
}

// Golden: strict loss matrix values
#[test]
fn golden_strict_loss_matrix() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    // Pure Compatible state: posterior = [~1, ~0, ~0]
    let d = ctrl.decide(DecisionSignals::new(0.0, 0.0, 0.0));
    // expected_losses ≈ [L_allow * p_compat, L_validate * p_compat, L_failclosed * p_compat]
    // = [0*1, 8*1, 40*1] approximately (dominant term)
    // Allow has lowest loss → action=Allow
    assert_eq!(d.action, PolicyAction::Allow);
    assert!(d.expected_losses[0] < d.expected_losses[1]);
    assert!(d.expected_losses[1] < d.expected_losses[2]);
}

// Golden: hardened loss matrix values
#[test]
fn golden_hardened_loss_matrix() {
    let mut ctrl = PolicyController::new(RuntimeMode::Hardened, 16);
    let d = ctrl.decide(DecisionSignals::new(0.0, 0.0, 0.0));
    assert_eq!(d.action, PolicyAction::Allow);
    assert!(d.expected_losses[0] < d.expected_losses[1]);
}

// Golden: high cond but compatible still dominant when meta=0 anomaly=0
#[test]
fn golden_strict_high_cond_compatible_dominant() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    // At (16,0,0): cond=1.0, logits: compat=2.0, ill=1.0, incompat=-2.0
    // Compatible is still the top state but FullValidate wins on expected loss
    let d = ctrl.decide(DecisionSignals::new(16.0, 0.0, 0.0));
    assert_eq!(d.top_state, RiskState::Compatible);
    assert_eq!(d.action, PolicyAction::FullValidate);
}

// Golden: pure incompatible signal
#[test]
fn golden_strict_pure_incompatible() {
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    let d = ctrl.decide(DecisionSignals::new(0.0, 1.0, 0.0));
    assert_eq!(d.top_state, RiskState::IncompatibleMetadata);
    assert_eq!(d.action, PolicyAction::FailClosed);
}

// Golden: verify logit coefficients
#[test]
fn golden_logit_compatible_at_zero() {
    // At (0,0,0): compatible logit = 2.8
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    let d = ctrl.decide(DecisionSignals::new(0.0, 0.0, 0.0));
    // Compatible should be overwhelmingly dominant
    assert!(d.posterior[0] > 0.9);
}

#[test]
fn golden_logit_ill_conditioned_at_16() {
    // At (16,0,0): cond=1.0, logits: compat=2.0, ill=1.0, incompat=-2.0
    // Compatible is still highest posterior, ill is second
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    let d = ctrl.decide(DecisionSignals::new(16.0, 0.0, 0.0));
    assert!(d.posterior[0] > d.posterior[1]); // compat > ill
    assert!(d.posterior[1] > d.posterior[2]); // ill > incompat
}

#[test]
fn golden_logit_incompatible_at_meta1() {
    // At (0,1,0): incompat logit = -2+3.5 = 1.5
    let mut ctrl = PolicyController::new(RuntimeMode::Strict, 16);
    let d = ctrl.decide(DecisionSignals::new(0.0, 1.0, 0.0));
    assert!(d.posterior[2] > d.posterior[0]);
    assert!(d.posterior[2] > d.posterior[1]);
}

// Golden: solver portfolio expected losses verification
// Golden: the expected losses at each state's center are that state's column of the loss
// matrix, bit for bit (see `assert_losses_are_column`).
#[test]
fn golden_portfolio_well_cond_losses() {
    assert_losses_are_column(1e-2, 0);
}

#[test]
fn golden_portfolio_moderate_losses() {
    assert_losses_are_column(1e-6, 1);
}

#[test]
fn golden_portfolio_ill_losses() {
    assert_losses_are_column(1e-11, 2);
}

#[test]
fn golden_portfolio_near_singular_losses() {
    assert_losses_are_column(1e-16, 3);
}

#[test]
fn golden_calibrator_threshold_default() {
    let cal = ConformalCalibrator::new(0.05, 100);
    // Default violation threshold is 1e-8
    let mut cal2 = cal.clone();
    cal2.observe(1e-9); // below threshold
    assert_eq!(cal2.empirical_miscoverage(), 0.0);
    cal2.observe(1e-7); // above threshold
    assert_eq!(cal2.empirical_miscoverage(), 0.5);
}

// ═══════════════════════════════════════════════════════════════════
// §14  Property tests (>=10 with 1000 cases each)
// ═══════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn prop_posterior_normalized(
        cond in 0.0f64..100.0,
        meta in 0.0f64..1.0,
        anomaly in 0.0f64..1.0,
    ) {
        let mut ctrl = PolicyController::new(RuntimeMode::Strict, 4);
        let d = ctrl.decide(DecisionSignals::new(cond, meta, anomaly));
        let sum: f64 = d.posterior.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-9, "sum={sum}");
        for p in &d.posterior {
            prop_assert!(*p >= 0.0, "negative posterior: {p}");
        }
    }

    #[test]
    fn prop_posterior_normalized_hardened(
        cond in 0.0f64..100.0,
        meta in 0.0f64..1.0,
        anomaly in 0.0f64..1.0,
    ) {
        let mut ctrl = PolicyController::new(RuntimeMode::Hardened, 4);
        let d = ctrl.decide(DecisionSignals::new(cond, meta, anomaly));
        let sum: f64 = d.posterior.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-9, "sum={sum}");
    }

    #[test]
    fn prop_determinism(
        cond in 0.0f64..100.0,
        meta in 0.0f64..1.0,
        anomaly in 0.0f64..1.0,
    ) {
        let s = DecisionSignals::new(cond, meta, anomaly);
        let mut c1 = PolicyController::new(RuntimeMode::Strict, 4);
        let mut c2 = PolicyController::new(RuntimeMode::Strict, 4);
        prop_assert_eq!(c1.decide(s).action, c2.decide(s).action);
    }

    #[test]
    fn prop_ledger_bounded(
        n in 1usize..200,
        cap in 1usize..50,
    ) {
        let mut ctrl = PolicyController::new(RuntimeMode::Strict, cap);
        for i in 0..n {
            ctrl.decide(DecisionSignals::new(i as f64, 0.0, 0.0));
        }
        prop_assert!(ctrl.ledger().len() <= cap);
    }

    #[test]
    fn prop_expected_losses_nonneg(
        cond in 0.0f64..100.0,
        meta in 0.0f64..1.0,
        anomaly in 0.0f64..1.0,
    ) {
        let mut ctrl = PolicyController::new(RuntimeMode::Strict, 4);
        let d = ctrl.decide(DecisionSignals::new(cond, meta, anomaly));
        for l in &d.expected_losses {
            prop_assert!(*l >= -1e-12, "negative loss: {l}");
        }
    }

    #[test]
    fn prop_action_in_valid_set(
        cond in 0.0f64..100.0,
        meta in 0.0f64..1.0,
        anomaly in 0.0f64..1.0,
        mode_idx in 0usize..2,
    ) {
        let mode = if mode_idx == 0 { RuntimeMode::Strict } else { RuntimeMode::Hardened };
        let mut ctrl = PolicyController::new(mode, 4);
        let d = ctrl.decide(DecisionSignals::new(cond, meta, anomaly));
        prop_assert!(
            d.action == PolicyAction::Allow
            || d.action == PolicyAction::FullValidate
            || d.action == PolicyAction::FailClosed
        );
    }

    #[test]
    fn prop_solver_action_bounded(
        state_idx in 0usize..4,
    ) {
        let state = MatrixConditionState::ALL[state_idx];
        let p = SolverPortfolio::new(RuntimeMode::Strict, 64);
        let (action, _, _, _) = p.select_action(state_to_rcond(&state), None);
        prop_assert!(action.index() < 5);
    }

    #[test]
    fn prop_calibrator_total_matches_observations(
        n in 1usize..500,
        cap in 10usize..200,
    ) {
        let mut cal = ConformalCalibrator::new(0.05, cap);
        for i in 0..n {
            cal.observe(i as f64 * 1e-10);
        }
        prop_assert_eq!(cal.total_predictions(), n);
    }

    #[test]
    fn prop_is_finite_consistency(
        c in proptest::num::f64::ANY,
        m in proptest::num::f64::ANY,
        a in proptest::num::f64::ANY,
    ) {
        let s = DecisionSignals::new(c, m, a);
        let expected = c.is_finite() && m.is_finite() && a.is_finite();
        prop_assert_eq!(s.is_finite(), expected);
    }

    #[test]
    fn prop_high_meta_always_failclosed(
        meta in 0.85f64..1.0,
        anomaly in 0.0f64..1.0,
    ) {
        let s = DecisionSignals::new(0.0, meta, anomaly);
        let mut strict = PolicyController::new(RuntimeMode::Strict, 4);
        let mut hard = PolicyController::new(RuntimeMode::Hardened, 4);
        prop_assert_eq!(strict.decide(s).action, PolicyAction::FailClosed);
        prop_assert_eq!(hard.decide(s).action, PolicyAction::FailClosed);
    }

    #[test]
    fn prop_mode_does_not_affect_top_state(
        cond in 0.0f64..20.0,
        meta in 0.0f64..1.0,
        anomaly in 0.0f64..1.0,
    ) {
        let s = DecisionSignals::new(cond, meta, anomaly);
        let mut strict = PolicyController::new(RuntimeMode::Strict, 4);
        let mut hard = PolicyController::new(RuntimeMode::Hardened, 4);
        // Top state depends on posterior which depends on logits (same for both modes)
        prop_assert_eq!(strict.decide(s).top_state, hard.decide(s).top_state);
    }

    #[test]
    fn prop_negative_signals_same_as_zero(
        cond in -100.0f64..0.0,
        meta in -1.0f64..0.0,
        anomaly in -1.0f64..0.0,
    ) {
        let mut c1 = PolicyController::new(RuntimeMode::Strict, 4);
        let mut c2 = PolicyController::new(RuntimeMode::Strict, 4);
        let d1 = c1.decide(DecisionSignals::new(cond, meta, anomaly));
        let d2 = c2.decide(DecisionSignals::new(0.0, 0.0, 0.0));
        // All clamp to 0 → identical results
        prop_assert_eq!(d1.action, d2.action);
    }
}
