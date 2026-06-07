#![forbid(unsafe_code)]

//! Bayesian risk-state decision engine with asymmetric loss matrix.
//!
//! The [`PolicyController`] evaluates [`DecisionSignals`] against a
//! log-odds model to produce a posterior over risk states, then selects
//! the action with minimum expected loss under the mode-specific loss matrix.
//!
//! # Status
//!
//! `fsci-linalg::{solve, solve_with_casp, solve_with_audit}` invokes
//! `PolicyController::decide` for finite solve inputs before CASP solver
//! action selection and applies `FailClosed` / `FullValidate` decisions
//! to the production solve path. Other production crates still use local
//! Hardened-mode validators until they are wired into this controller.

use serde::{Deserialize, Serialize};

use crate::evidence::{DecisionEvidenceEntry, PolicyEvidenceLedger};
use crate::mode::RuntimeMode;
use crate::signals::DecisionSignals;

/// Risk states for the policy decision model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskState {
    Compatible,
    IllConditioned,
    IncompatibleMetadata,
}

impl RiskState {
    pub(crate) const ALL: [Self; 3] = [
        Self::Compatible,
        Self::IllConditioned,
        Self::IncompatibleMetadata,
    ];

    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Compatible => 0,
            Self::IllConditioned => 1,
            Self::IncompatibleMetadata => 2,
        }
    }
}

/// Actions available to the policy controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyAction {
    Allow,
    FullValidate,
    FailClosed,
}

impl PolicyAction {
    pub(crate) const ALL: [Self; 3] = [Self::Allow, Self::FullValidate, Self::FailClosed];
}

/// Result of a policy decision evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyDecision {
    pub mode: RuntimeMode,
    pub action: PolicyAction,
    pub top_state: RiskState,
    pub posterior: [f64; 3],
    pub expected_losses: [f64; 3],
    pub reason: String,
}

/// Bayesian risk-state decision engine.
///
/// Each call to [`decide()`](Self::decide) is independent: the ledger
/// records history for audit but does not influence future decisions.
#[derive(Debug, Clone)]
pub struct PolicyController {
    mode: RuntimeMode,
    ledger: PolicyEvidenceLedger,
}

impl PolicyController {
    #[must_use]
    pub fn new(mode: RuntimeMode, ledger_capacity: usize) -> Self {
        Self {
            mode,
            ledger: PolicyEvidenceLedger::new(ledger_capacity),
        }
    }

    #[must_use]
    pub const fn mode(&self) -> RuntimeMode {
        self.mode
    }

    #[must_use]
    pub const fn ledger(&self) -> &PolicyEvidenceLedger {
        &self.ledger
    }

    pub fn decide(&mut self, signals: DecisionSignals) -> PolicyDecision {
        if !signals.is_finite() {
            let logits = [-1.0e30, -1.0e30, 0.0];
            let posterior = [0.0, 0.0, 1.0];
            let expected_losses = expected_loss(self.mode, posterior);
            let action = PolicyAction::FailClosed;
            let top_state = RiskState::IncompatibleMetadata;
            let reason = format!(
                "mode={:?}; top_state={:?}; p=1.000000; non_finite_signals=true",
                self.mode, top_state
            );

            self.ledger.record(DecisionEvidenceEntry {
                mode: self.mode,
                signals,
                logits,
                posterior,
                expected_losses,
                action,
                top_state,
                reason: reason.clone(),
            });

            return PolicyDecision {
                mode: self.mode,
                action,
                top_state,
                posterior,
                expected_losses,
                reason,
            };
        }

        let logits = logits_from_signals(signals);
        let posterior = softmax(logits);
        let expected_losses = expected_loss(self.mode, posterior);
        let (action, action_idx) = select_action(expected_losses);
        let (top_state, top_state_prob) = top_risk_state(posterior);
        let reason = format!(
            "mode={:?}; top_state={:?}; p={top_state_prob:.6}; cond_log10={:.3}; metadata={:.3}; anomaly={:.3}",
            self.mode,
            top_state,
            signals.condition_number_log10,
            signals.metadata_incompatibility_score,
            signals.input_anomaly_score
        );

        self.ledger.record(DecisionEvidenceEntry {
            mode: self.mode,
            signals,
            logits,
            posterior,
            expected_losses,
            action,
            top_state,
            reason: reason.clone(),
        });

        debug_assert!(
            action_idx < PolicyAction::ALL.len(),
            "action index must be bounded"
        );

        PolicyDecision {
            mode: self.mode,
            action,
            top_state,
            posterior,
            expected_losses,
            reason,
        }
    }
}

// ── Internal functions ──────────────────────────────────────────────

pub(crate) fn logits_from_signals(signals: DecisionSignals) -> [f64; 3] {
    let cond = (signals.condition_number_log10 / 16.0).clamp(0.0, 1.0);
    let metadata = signals.metadata_incompatibility_score.clamp(0.0, 1.0);
    let anomaly = signals.input_anomaly_score.clamp(0.0, 1.0);

    let compatible = 2.8 - 0.8 * cond - 3.2 * metadata - 2.4 * anomaly;
    let ill_conditioned = -0.4 + 1.4 * cond + 0.6 * anomaly - 0.8 * metadata;
    let incompatible = -2.0 + 3.5 * metadata + 0.7 * anomaly;

    [compatible, ill_conditioned, incompatible]
}

pub(crate) fn softmax(logits: [f64; 3]) -> [f64; 3] {
    let max_logit = logits.iter().fold(f64::NEG_INFINITY, |acc, v| acc.max(*v));
    let exps = logits.map(|v| (v - max_logit).exp());
    let denom = exps.iter().sum::<f64>();
    if denom <= 0.0 {
        return [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
    }
    exps.map(|v| v / denom)
}

#[must_use]
pub const fn decision_loss_matrix(mode: RuntimeMode) -> [[f64; 3]; 3] {
    match mode {
        RuntimeMode::Strict => [[0.0, 65.0, 200.0], [8.0, 4.0, 80.0], [40.0, 25.0, 1.0]],
        RuntimeMode::Hardened => [[0.0, 50.0, 180.0], [5.0, 3.0, 60.0], [55.0, 30.0, 1.0]],
    }
}

pub(crate) fn expected_loss(mode: RuntimeMode, posterior: [f64; 3]) -> [f64; 3] {
    let matrix = decision_loss_matrix(mode);
    let mut losses = [0.0; 3];
    for (row_idx, row) in matrix.iter().enumerate() {
        losses[row_idx] = row
            .iter()
            .zip(posterior.iter())
            .map(|(loss, prob)| loss * prob)
            .sum();
    }
    losses
}

pub(crate) fn select_action(expected_losses: [f64; 3]) -> (PolicyAction, usize) {
    let mut best_idx = 0usize;
    let mut best_loss = expected_losses[0];
    for (idx, loss) in expected_losses.iter().enumerate().skip(1) {
        if loss < &best_loss {
            best_loss = *loss;
            best_idx = idx;
            continue;
        }
        if (*loss - best_loss).abs() <= 1e-12 && idx > best_idx {
            best_idx = idx;
        }
    }
    (PolicyAction::ALL[best_idx], best_idx)
}

pub(crate) fn top_risk_state(posterior: [f64; 3]) -> (RiskState, f64) {
    let mut best_idx = 0usize;
    let mut best_prob = posterior[0];
    for (idx, prob) in posterior.iter().enumerate().skip(1) {
        if prob > &best_prob {
            best_idx = idx;
            best_prob = *prob;
        }
    }
    (RiskState::ALL[best_idx], best_prob)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decide_fail_closes_on_non_finite_signals() {
        let mut ctrl = PolicyController::new(RuntimeMode::Strict, 4);
        let decision = ctrl.decide(DecisionSignals::new(f64::NAN, 0.0, 0.0));

        assert_eq!(decision.action, PolicyAction::FailClosed);
        assert_eq!(decision.top_state, RiskState::IncompatibleMetadata);
        assert_eq!(decision.posterior, [0.0, 0.0, 1.0]);
        assert!(decision.expected_losses.iter().all(|loss| loss.is_finite()));
        assert!(decision.reason.contains("non_finite_signals=true"));

        let entry = ctrl.ledger().latest().expect("ledger entry");
        assert_eq!(entry.action, PolicyAction::FailClosed);
        assert_eq!(entry.posterior, [0.0, 0.0, 1.0]);
        assert!(entry.logits.iter().all(|logit| logit.is_finite()));
    }

    #[test]
    fn policy_decision_golden_snapshot() {
        let cases = [
            (
                "strict",
                RuntimeMode::Strict,
                DecisionSignals::new(5.0, 0.3, 0.2),
            ),
            (
                "hardened",
                RuntimeMode::Hardened,
                DecisionSignals::new(5.0, 0.3, 0.2),
            ),
            (
                "nonfinite",
                RuntimeMode::Strict,
                DecisionSignals::new(f64::NAN, 0.0, 0.0),
            ),
        ];

        for (case, mode, signals) in cases {
            let mut ctrl = PolicyController::new(mode, 4);
            let decision = ctrl.decide(signals);
            let entry = ctrl.ledger().latest().expect("ledger entry");

            assert_eq!(entry.action, decision.action);
            assert_eq!(entry.top_state, decision.top_state);
            assert_eq!(entry.posterior, decision.posterior);
            assert_eq!(entry.expected_losses, decision.expected_losses);
            assert_eq!(entry.reason, decision.reason);

            println!(
                concat!(
                    "POLICY_DECISION_GOLDEN case={} mode={:?} action={:?} top={:?} ",
                    "posterior={:016x},{:016x},{:016x} ",
                    "losses={:016x},{:016x},{:016x} reason={}"
                ),
                case,
                decision.mode,
                decision.action,
                decision.top_state,
                decision.posterior[0].to_bits(),
                decision.posterior[1].to_bits(),
                decision.posterior[2].to_bits(),
                decision.expected_losses[0].to_bits(),
                decision.expected_losses[1].to_bits(),
                decision.expected_losses[2].to_bits(),
                decision.reason
            );
        }
    }
}
