#![forbid(unsafe_code)]

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimeMode {
    Strict,
    Hardened,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskState {
    Compatible,
    IllConditioned,
    IncompatibleMetadata,
}

impl RiskState {
    const ALL: [Self; 3] = [
        Self::Compatible,
        Self::IllConditioned,
        Self::IncompatibleMetadata,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyAction {
    Allow,
    FullValidate,
    FailClosed,
}

impl PolicyAction {
    const ALL: [Self; 3] = [Self::Allow, Self::FullValidate, Self::FailClosed];
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DecisionSignals {
    pub condition_number_log10: f64,
    pub metadata_incompatibility_score: f64,
    pub input_anomaly_score: f64,
}

impl DecisionSignals {
    #[must_use]
    pub fn new(
        condition_number_log10: f64,
        metadata_incompatibility_score: f64,
        input_anomaly_score: f64,
    ) -> Self {
        Self {
            condition_number_log10,
            metadata_incompatibility_score,
            input_anomaly_score,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyDecision {
    pub mode: RuntimeMode,
    pub action: PolicyAction,
    pub top_state: RiskState,
    pub posterior: [f64; 3],
    pub expected_losses: [f64; 3],
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionEvidenceEntry {
    pub mode: RuntimeMode,
    pub signals: DecisionSignals,
    pub logits: [f64; 3],
    pub posterior: [f64; 3],
    pub expected_losses: [f64; 3],
    pub action: PolicyAction,
    pub top_state: RiskState,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyEvidenceLedger {
    capacity: usize,
    entries: VecDeque<DecisionEvidenceEntry>,
}

impl PolicyEvidenceLedger {
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: VecDeque::new(),
        }
    }

    pub fn record(&mut self, entry: DecisionEvidenceEntry) {
        if self.entries.len() == self.capacity {
            let _ = self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[must_use]
    pub fn latest(&self) -> Option<&DecisionEvidenceEntry> {
        self.entries.back()
    }

    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }
}

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

fn logits_from_signals(signals: DecisionSignals) -> [f64; 3] {
    let cond = (signals.condition_number_log10 / 16.0).clamp(0.0, 1.0);
    let metadata = signals.metadata_incompatibility_score.clamp(0.0, 1.0);
    let anomaly = signals.input_anomaly_score.clamp(0.0, 1.0);

    // Log-odds model tuned for fail-closed behavior under metadata incompatibility.
    let compatible = 2.8 - 0.8 * cond - 3.2 * metadata - 2.4 * anomaly;
    let ill_conditioned = -0.4 + 1.4 * cond + 0.6 * anomaly - 0.8 * metadata;
    let incompatible = -2.0 + 3.5 * metadata + 0.7 * anomaly;

    [compatible, ill_conditioned, incompatible]
}

fn softmax(logits: [f64; 3]) -> [f64; 3] {
    let max_logit = logits.iter().fold(f64::NEG_INFINITY, |acc, v| acc.max(*v));
    let exps = logits.map(|v| (v - max_logit).exp());
    let denom = exps.iter().sum::<f64>();
    if denom == 0.0 {
        return [1.0, 0.0, 0.0];
    }
    exps.map(|v| v / denom)
}

fn loss_matrix(mode: RuntimeMode) -> [[f64; 3]; 3] {
    // rows: action (allow, full_validate, fail_closed)
    // cols: state (compatible, ill_conditioned, incompatible_metadata)
    match mode {
        RuntimeMode::Strict => [[0.0, 65.0, 200.0], [8.0, 4.0, 80.0], [40.0, 25.0, 1.0]],
        RuntimeMode::Hardened => [[0.0, 50.0, 180.0], [5.0, 3.0, 60.0], [55.0, 30.0, 1.0]],
    }
}

fn expected_loss(mode: RuntimeMode, posterior: [f64; 3]) -> [f64; 3] {
    let matrix = loss_matrix(mode);
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

fn select_action(expected_losses: [f64; 3]) -> (PolicyAction, usize) {
    let mut best_idx = 0usize;
    let mut best_loss = expected_losses[0];
    for (idx, loss) in expected_losses.iter().enumerate().skip(1) {
        if loss < &best_loss {
            best_loss = *loss;
            best_idx = idx;
            continue;
        }

        // Tie-break toward safer action in order: FailClosed > FullValidate > Allow.
        if (*loss - best_loss).abs() <= 1e-12 && idx > best_idx {
            best_idx = idx;
        }
    }
    (PolicyAction::ALL[best_idx], best_idx)
}

fn top_risk_state(posterior: [f64; 3]) -> (RiskState, f64) {
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
    fn strict_mode_fails_closed_on_incompatible_metadata() {
        let mut controller = PolicyController::new(RuntimeMode::Strict, 16);
        let decision = controller.decide(DecisionSignals::new(8.0, 1.0, 0.2));
        assert_eq!(decision.action, PolicyAction::FailClosed);
        assert_eq!(decision.top_state, RiskState::IncompatibleMetadata);
    }

    #[test]
    fn hardened_mode_prefers_full_validation_on_mid_risk() {
        let mut controller = PolicyController::new(RuntimeMode::Hardened, 16);
        let decision = controller.decide(DecisionSignals::new(12.0, 0.25, 0.3));
        assert_eq!(decision.action, PolicyAction::FullValidate);
    }

    #[test]
    fn ledger_is_bounded() {
        let mut controller = PolicyController::new(RuntimeMode::Strict, 2);
        for i in 0..4 {
            let _ = controller.decide(DecisionSignals::new(i as f64, 0.0, 0.0));
        }
        assert_eq!(controller.ledger().len(), 2);
    }

    #[test]
    fn posterior_is_normalized() {
        let logits = logits_from_signals(DecisionSignals::new(2.0, 0.1, 0.1));
        let posterior = softmax(logits);
        let sum = posterior.iter().sum::<f64>();
        assert!((sum - 1.0).abs() < 1e-9);
    }
}
