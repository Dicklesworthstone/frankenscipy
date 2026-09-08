#![forbid(unsafe_code)]

//! Anytime-valid invariant monitors (e-processes) for solver correctness sentinels.
//!
//! An e-process is a nonnegative sequence $(E_t)_{t \ge 0}$ such that
//! $\mathbb{E}[E_\tau] \le 1$ for any stopping time $\tau$ under the null hypothesis
//! $H_0: \mathbb{E}[X_t] \le \mu_0$.
//!
//! By Ville's inequality:
//! $$\mathbb{P}_{H_0}\left(\exists t \ge 1: E_t \ge \frac{1}{\alpha}\right) \le \alpha$$
//!
//! This provides continuous, sequential quality assurance over streaming solver outputs
//! without inflating family-wise error rates or requiring fixed sample sizes.

use serde::{Deserialize, Serialize};

/// Configuration for an anytime-valid e-process invariant monitor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EProcessConfig {
    /// Null hypothesis upper bound $\mu_0 \in (0, 1)$ on expected sentinel discrepancy.
    pub null_bound: f64,
    /// False alarm rate budget $\alpha \in (0, 1)$ (e.g., 0.01 for 99% confidence).
    pub alpha: f64,
    /// Maximum predictable betting parameter $\lambda_{\max} > 0$.
    pub max_lambda: f64,
}

impl Default for EProcessConfig {
    fn default() -> Self {
        Self {
            null_bound: 1.0e-3,
            alpha: 0.01,
            max_lambda: 2.0,
        }
    }
}

impl EProcessConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), &'static str> {
        if !(0.0 < self.null_bound && self.null_bound < 1.0) {
            return Err("null_bound must be in (0, 1)");
        }
        if !(0.0 < self.alpha && self.alpha < 1.0) {
            return Err("alpha must be in (0, 1)");
        }
        if self.max_lambda <= 0.0 {
            return Err("max_lambda must be positive");
        }
        Ok(())
    }
}

/// Status reported by the monitor after observing a sentinel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EProcessStatus {
    /// Process is within calibrated bounds; no invariant violation detected.
    Nominal,
    /// Invariant violation detected: e-value has crossed $1 / \alpha$.
    AlarmTriggered,
}

/// Sequential e-process invariant monitor for numerical solver sentinels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EProcessMonitor {
    config: EProcessConfig,
    step_count: usize,
    log_e_value: f64,
    running_sum: f64,
    alarm_triggered: bool,
    alarm_step: Option<usize>,
}

impl EProcessMonitor {
    /// Create a new monitor with the given configuration.
    pub fn new(config: EProcessConfig) -> Result<Self, &'static str> {
        config.validate()?;
        Ok(Self {
            config,
            step_count: 0,
            log_e_value: 0.0,
            running_sum: 0.0,
            alarm_triggered: false,
            alarm_step: None,
        })
    }

    /// Return the monitor configuration.
    #[must_use]
    pub const fn config(&self) -> &EProcessConfig {
        &self.config
    }

    /// Number of observations processed so far.
    #[must_use]
    pub const fn step_count(&self) -> usize {
        self.step_count
    }

    /// Current e-value $E_t = \exp(\log E_t)$.
    #[must_use]
    pub fn e_value(&self) -> f64 {
        self.log_e_value.exp()
    }

    /// Current log e-value $\log E_t$.
    #[must_use]
    pub const fn log_e_value(&self) -> f64 {
        self.log_e_value
    }

    /// Threshold $1 / \alpha$ required to trigger an anytime-valid alarm.
    #[must_use]
    pub fn threshold(&self) -> f64 {
        1.0 / self.config.alpha
    }

    /// Log threshold $-\log(\alpha)$.
    #[must_use]
    pub fn log_threshold(&self) -> f64 {
        -self.config.alpha.ln()
    }

    /// Whether an alarm has been raised.
    #[must_use]
    pub const fn is_alarm_triggered(&self) -> bool {
        self.alarm_triggered
    }

    /// Step at which the alarm was first raised, if any.
    #[must_use]
    pub const fn alarm_step(&self) -> Option<usize> {
        self.alarm_step
    }

    /// Compute predictable parameter $\lambda_t$ from past data $\mathcal{F}_{t-1}$.
    fn predictable_lambda(&self) -> f64 {
        if self.step_count == 0 {
            return 0.5 * self.config.max_lambda;
        }
        let empirical_mean = self.running_sum / (self.step_count as f64);
        let gap = empirical_mean - self.config.null_bound;
        if gap <= 0.0 {
            0.1 * self.config.max_lambda
        } else {
            (4.0 * gap).clamp(0.1 * self.config.max_lambda, self.config.max_lambda)
        }
    }

    /// Ingest a new bounded sentinel observation $X_t \in [0, 1]$ (e.g. normalized residual).
    pub fn observe(&mut self, discrepancy: f64) -> EProcessStatus {
        let x = discrepancy.clamp(0.0, 1.0);
        let lambda = self.predictable_lambda();

        // Sub-Gaussian exponential mixture step: log(e_t) = lambda * (x - mu_0) - lambda^2 / 8
        let log_increment = lambda * (x - self.config.null_bound) - (lambda * lambda) / 8.0;
        self.log_e_value += log_increment;
        self.step_count += 1;
        self.running_sum += x;

        if !self.alarm_triggered && self.log_e_value >= self.log_threshold() {
            self.alarm_triggered = true;
            self.alarm_step = Some(self.step_count);
        }

        if self.alarm_triggered {
            EProcessStatus::AlarmTriggered
        } else {
            EProcessStatus::Nominal
        }
    }

    /// Reset the monitor state to step 0 ($E_0 = 1$).
    pub fn reset(&mut self) {
        self.step_count = 0;
        self.log_e_value = 0.0;
        self.running_sum = 0.0;
        self.alarm_triggered = false;
        self.alarm_step = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_validation_rejects_invalid_bounds() {
        assert!(
            EProcessConfig {
                null_bound: 0.0,
                alpha: 0.01,
                max_lambda: 2.0
            }
            .validate()
            .is_err()
        );
        assert!(
            EProcessConfig {
                null_bound: 1.0,
                alpha: 0.01,
                max_lambda: 2.0
            }
            .validate()
            .is_err()
        );
        assert!(
            EProcessConfig {
                null_bound: 0.1,
                alpha: 0.0,
                max_lambda: 2.0
            }
            .validate()
            .is_err()
        );
        assert!(
            EProcessConfig {
                null_bound: 0.1,
                alpha: 1.0,
                max_lambda: 2.0
            }
            .validate()
            .is_err()
        );
        assert!(
            EProcessConfig {
                null_bound: 0.1,
                alpha: 0.05,
                max_lambda: -1.0
            }
            .validate()
            .is_err()
        );
        assert!(
            EProcessConfig {
                null_bound: 0.05,
                alpha: 0.01,
                max_lambda: 1.5
            }
            .validate()
            .is_ok()
        );
    }

    #[test]
    fn nominal_stream_does_not_trigger_alarm() {
        let config = EProcessConfig {
            null_bound: 0.1,
            alpha: 0.01,
            max_lambda: 2.0,
        };
        let mut monitor = EProcessMonitor::new(config).unwrap();
        assert_eq!(monitor.step_count(), 0);
        assert!((monitor.e_value() - 1.0).abs() < 1e-12);

        // Inputs strictly below null_bound
        for _ in 0..100 {
            let status = monitor.observe(0.05);
            assert_eq!(status, EProcessStatus::Nominal);
        }

        assert!(!monitor.is_alarm_triggered());
        assert!(monitor.e_value() < monitor.threshold());
    }

    #[test]
    fn degraded_stream_triggers_alarm_quickly() {
        let config = EProcessConfig {
            null_bound: 0.05,
            alpha: 0.01,
            max_lambda: 2.0,
        };
        let mut monitor = EProcessMonitor::new(config).unwrap();

        // Elevated discrepancy (0.8 vs 0.05 null bound)
        let mut triggered_at = None;
        for step in 1..=50 {
            let status = monitor.observe(0.8);
            if status == EProcessStatus::AlarmTriggered {
                triggered_at = Some(step);
                break;
            }
        }

        assert!(triggered_at.is_some(), "alarm should have triggered");
        assert!(monitor.is_alarm_triggered());
        assert_eq!(monitor.alarm_step(), triggered_at);
        assert!(monitor.e_value() >= monitor.threshold());

        // Reset restores nominal initial state
        monitor.reset();
        assert_eq!(monitor.step_count(), 0);
        assert!(!monitor.is_alarm_triggered());
        assert!((monitor.e_value() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn serde_roundtrip_preserves_monitor_state() {
        let config = EProcessConfig {
            null_bound: 0.1,
            alpha: 0.05,
            max_lambda: 2.0,
        };
        let mut monitor = EProcessMonitor::new(config).unwrap();
        monitor.observe(0.2);
        monitor.observe(0.3);

        let json = serde_json::to_string(&monitor).unwrap();
        let restored: EProcessMonitor = serde_json::from_str(&json).unwrap();

        assert_eq!(monitor.step_count(), restored.step_count());
        assert!((monitor.log_e_value() - restored.log_e_value()).abs() < 1e-12);
        assert_eq!(monitor.is_alarm_triggered(), restored.is_alarm_triggered());
    }
}
