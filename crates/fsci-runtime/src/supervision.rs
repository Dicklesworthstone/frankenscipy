#![forbid(unsafe_code)]

//! Structured runtime supervision for bounded recovery and fail-closed orchestration.
//!
//! Tracks solver health across decisions, enforcing:
//! 1. Bounded recovery budgets: finite allocation of fallback events before circuit breaking.
//! 2. Escalation policies: progressive degradation under consecutive failures (Healthy -> Degraded -> Quarantined -> FailClosed).
//! 3. Fail-closed guarantees: automatic refusal of solve requests when safety contracts are breached.

use crate::policy::PolicyAction;
use crate::signals::DecisionSignals;
use serde::{Deserialize, Serialize};

/// Current operational health status under supervision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SupervisionStatus {
    /// Nominal state; low failure and fallback rates.
    Healthy,
    /// Elevated fallback rate or condition anomalies observed.
    Degraded,
    /// Circuit breaker tripped due to consecutive failures or exhausted recovery budget.
    Quarantined,
    /// Unrecoverable integrity violation; all subsequent actions fail closed.
    FailClosed,
}

/// Configuration parameters for policy supervision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SupervisionConfig {
    /// Maximum consecutive fallbacks permitted before transitioning to Degraded.
    pub max_consecutive_fallbacks: usize,
    /// Maximum consecutive fail-closed decisions permitted before transitioning to Quarantined.
    pub max_consecutive_failures: usize,
    /// Total recovery budget (maximum fallback events allowed before quarantine).
    pub recovery_budget: usize,
    /// Number of consecutive direct solves required to clear Degraded state back to Healthy.
    pub cooldown_steps: usize,
}

impl Default for SupervisionConfig {
    fn default() -> Self {
        Self {
            max_consecutive_fallbacks: 3,
            max_consecutive_failures: 2,
            recovery_budget: 16,
            cooldown_steps: 5,
        }
    }
}

/// Audit event emitted during supervision state transitions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SupervisionEvent {
    pub previous_status: SupervisionStatus,
    pub new_status: SupervisionStatus,
    pub step: usize,
    pub reason: String,
}

/// Bounded supervisor for policy controllers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicySupervisor {
    config: SupervisionConfig,
    status: SupervisionStatus,
    consecutive_fallbacks: usize,
    consecutive_failures: usize,
    consecutive_successes: usize,
    recovery_budget_remaining: usize,
    total_steps: usize,
    events: Vec<SupervisionEvent>,
}

impl PolicySupervisor {
    /// Create a new supervisor with the given configuration.
    #[must_use]
    pub fn new(config: SupervisionConfig) -> Self {
        let budget = config.recovery_budget;
        Self {
            config,
            status: SupervisionStatus::Healthy,
            consecutive_fallbacks: 0,
            consecutive_failures: 0,
            consecutive_successes: 0,
            recovery_budget_remaining: budget,
            total_steps: 0,
            events: Vec::new(),
        }
    }

    /// Return the supervision configuration.
    #[must_use]
    pub const fn config(&self) -> &SupervisionConfig {
        &self.config
    }

    /// Current operational status.
    #[must_use]
    pub const fn status(&self) -> SupervisionStatus {
        self.status
    }

    /// Number of consecutive fallback solves currently observed.
    #[must_use]
    pub const fn consecutive_fallbacks(&self) -> usize {
        self.consecutive_fallbacks
    }

    /// Number of consecutive fail-closed outcomes currently observed.
    #[must_use]
    pub const fn consecutive_failures(&self) -> usize {
        self.consecutive_failures
    }

    /// Number of remaining fallback recovery events allowed.
    #[must_use]
    pub const fn recovery_budget_remaining(&self) -> usize {
        self.recovery_budget_remaining
    }

    /// Total decisions supervised.
    #[must_use]
    pub const fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// History of supervision state transitions.
    #[must_use]
    pub fn events(&self) -> &[SupervisionEvent] {
        &self.events
    }

    /// Record a status transition event.
    fn transition_to(&mut self, new_status: SupervisionStatus, reason: String) {
        if self.status != new_status {
            self.events.push(SupervisionEvent {
                previous_status: self.status,
                new_status,
                step: self.total_steps,
                reason,
            });
            self.status = new_status;
        }
    }

    /// Observe a proposed policy action, evaluate against supervisory bounds,
    /// and return the enforced policy action.
    pub fn observe_and_enforce(
        &mut self,
        proposed_action: PolicyAction,
        signals: &DecisionSignals,
    ) -> PolicyAction {
        self.total_steps += 1;

        // Severe signal violations trigger immediate fail-closed state
        if !signals.is_finite() {
            self.transition_to(
                SupervisionStatus::FailClosed,
                "Non-finite signals observed; fail-closed latch tripped".to_string(),
            );
            return PolicyAction::FailClosed;
        }

        // Active quarantine or fail-closed state unconditionally enforces FailClosed
        if self.status == SupervisionStatus::Quarantined
            || self.status == SupervisionStatus::FailClosed
        {
            return PolicyAction::FailClosed;
        }

        match proposed_action {
            PolicyAction::Allow => {
                self.consecutive_fallbacks = 0;
                self.consecutive_failures = 0;
                self.consecutive_successes += 1;

                // Cooldown transition from Degraded back to Healthy
                if self.status == SupervisionStatus::Degraded
                    && self.consecutive_successes >= self.config.cooldown_steps
                {
                    self.transition_to(
                        SupervisionStatus::Healthy,
                        format!(
                            "Cooldown satisfied ({} consecutive direct solves)",
                            self.consecutive_successes
                        ),
                    );
                }
                PolicyAction::Allow
            }
            PolicyAction::FullValidate => {
                self.consecutive_successes = 0;
                self.consecutive_failures = 0;
                self.consecutive_fallbacks += 1;

                if self.recovery_budget_remaining > 0 {
                    self.recovery_budget_remaining -= 1;
                }

                if self.recovery_budget_remaining == 0 {
                    self.transition_to(
                        SupervisionStatus::Quarantined,
                        "Recovery budget exhausted; circuit breaker tripped".to_string(),
                    );
                    PolicyAction::FailClosed
                } else if self.consecutive_fallbacks >= self.config.max_consecutive_fallbacks {
                    self.transition_to(
                        SupervisionStatus::Degraded,
                        format!(
                            "Max consecutive fallbacks reached ({})",
                            self.consecutive_fallbacks
                        ),
                    );
                    PolicyAction::FullValidate
                } else {
                    PolicyAction::FullValidate
                }
            }
            PolicyAction::FailClosed => {
                self.consecutive_successes = 0;
                self.consecutive_fallbacks = 0;
                self.consecutive_failures += 1;

                if self.consecutive_failures >= self.config.max_consecutive_failures {
                    self.transition_to(
                        SupervisionStatus::Quarantined,
                        format!(
                            "Consecutive failures threshold reached ({})",
                            self.consecutive_failures
                        ),
                    );
                }
                PolicyAction::FailClosed
            }
        }
    }

    /// Reset supervisor to initial healthy state with replenished budget.
    pub fn reset(&mut self) {
        let budget = self.config.recovery_budget;
        self.status = SupervisionStatus::Healthy;
        self.consecutive_fallbacks = 0;
        self.consecutive_failures = 0;
        self.consecutive_successes = 0;
        self.recovery_budget_remaining = budget;
        self.total_steps = 0;
        self.events.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nominal_direct_solves_remain_healthy() {
        let config = SupervisionConfig::default();
        let mut supervisor = PolicySupervisor::new(config);
        assert_eq!(supervisor.status(), SupervisionStatus::Healthy);

        let signals = DecisionSignals {
            condition_number_log10: 2.0,
            metadata_incompatibility_score: 0.0,
            input_anomaly_score: 0.1,
        };

        for _ in 0..10 {
            let action = supervisor.observe_and_enforce(PolicyAction::Allow, &signals);
            assert_eq!(action, PolicyAction::Allow);
            assert_eq!(supervisor.status(), SupervisionStatus::Healthy);
        }
        assert_eq!(supervisor.events().len(), 0);
    }

    #[test]
    fn consecutive_fallbacks_trigger_degradation() {
        let config = SupervisionConfig {
            max_consecutive_fallbacks: 2,
            max_consecutive_failures: 2,
            recovery_budget: 10,
            cooldown_steps: 3,
        };
        let mut supervisor = PolicySupervisor::new(config);
        let signals = DecisionSignals {
            condition_number_log10: 8.0,
            metadata_incompatibility_score: 0.1,
            input_anomaly_score: 0.2,
        };

        // First fallback: still healthy
        let a1 = supervisor.observe_and_enforce(PolicyAction::FullValidate, &signals);
        assert_eq!(a1, PolicyAction::FullValidate);
        assert_eq!(supervisor.status(), SupervisionStatus::Healthy);

        // Second fallback: transitions to Degraded
        let a2 = supervisor.observe_and_enforce(PolicyAction::FullValidate, &signals);
        assert_eq!(a2, PolicyAction::FullValidate);
        assert_eq!(supervisor.status(), SupervisionStatus::Degraded);
        assert_eq!(supervisor.events().len(), 1);

        // Cooldown: 3 direct solves return to Healthy
        for _ in 0..2 {
            supervisor.observe_and_enforce(PolicyAction::Allow, &signals);
            assert_eq!(supervisor.status(), SupervisionStatus::Degraded);
        }
        supervisor.observe_and_enforce(PolicyAction::Allow, &signals);
        assert_eq!(supervisor.status(), SupervisionStatus::Healthy);
    }

    #[test]
    fn budget_exhaustion_trips_quarantine() {
        let config = SupervisionConfig {
            max_consecutive_fallbacks: 5,
            max_consecutive_failures: 5,
            recovery_budget: 2,
            cooldown_steps: 3,
        };
        let mut supervisor = PolicySupervisor::new(config);
        let signals = DecisionSignals {
            condition_number_log10: 5.0,
            metadata_incompatibility_score: 0.0,
            input_anomaly_score: 0.0,
        };

        supervisor.observe_and_enforce(PolicyAction::FullValidate, &signals);
        assert_eq!(supervisor.recovery_budget_remaining(), 1);

        // Exhausts budget on second fallback
        let a2 = supervisor.observe_and_enforce(PolicyAction::FullValidate, &signals);
        assert_eq!(a2, PolicyAction::FailClosed);
        assert_eq!(supervisor.status(), SupervisionStatus::Quarantined);
        assert_eq!(supervisor.recovery_budget_remaining(), 0);

        // Future requests fail closed while quarantined
        let a3 = supervisor.observe_and_enforce(PolicyAction::Allow, &signals);
        assert_eq!(a3, PolicyAction::FailClosed);

        // Reset replenishes
        supervisor.reset();
        assert_eq!(supervisor.status(), SupervisionStatus::Healthy);
        assert_eq!(supervisor.recovery_budget_remaining(), 2);
    }

    #[test]
    fn non_finite_signals_trip_fail_closed() {
        let config = SupervisionConfig::default();
        let mut supervisor = PolicySupervisor::new(config);
        let invalid = DecisionSignals {
            condition_number_log10: f64::NAN,
            metadata_incompatibility_score: 0.0,
            input_anomaly_score: 0.0,
        };

        let action = supervisor.observe_and_enforce(PolicyAction::Allow, &invalid);
        assert_eq!(action, PolicyAction::FailClosed);
        assert_eq!(supervisor.status(), SupervisionStatus::FailClosed);
    }
}
