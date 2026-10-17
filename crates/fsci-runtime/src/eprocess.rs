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
///
/// Implements the Waudby-Smith & Ramdas betting e-process on bounded observations
/// $X_t \in [0, 1]$ under $H_0: \mathbb{E}[X_t \mid \mathcal{F}_{t-1}] \le \mu_0$:
/// $$E_t = \prod_{s=1}^t \bigl(1 + \lambda_s (X_s - \mu_0)\bigr), \qquad
///   \lambda_s \in \bigl[0, \min(\lambda_{\max}, 0.95 / \mu_0)\bigr)$$
/// Because $\lambda_t$ is $\mathcal{F}_{t-1}$-measurable and $1 + \lambda_t (X_t - \mu_0) > 0$,
/// $\mathbb{E}_{H_0}[E_t \mid \mathcal{F}_{t-1}] \le E_{t-1}$ with $E_0 = 1$, so Ville's
/// inequality guarantees $\mathbb{P}_{H_0}(\exists t \ge 1: E_t \ge 1/\alpha) \le \alpha$
/// at any stopping time without fixed window sizes (frankenscipy-7tb8d.4).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EProcessMonitor {
    config: EProcessConfig,
    step_count: usize,
    log_e_value: f64,
    running_sum: f64,
    #[serde(default)]
    ewma_mean: f64,
    alarm_triggered: bool,
    alarm_step: Option<usize>,
}

const EWMA_GAMMA: f64 = 0.05;

impl EProcessMonitor {
    /// Create a new monitor with the given configuration.
    pub fn new(config: EProcessConfig) -> Result<Self, &'static str> {
        config.validate()?;
        let ewma_mean = config.null_bound;
        Ok(Self {
            config,
            step_count: 0,
            log_e_value: 0.0,
            running_sum: 0.0,
            ewma_mean,
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

    /// Compute predictable betting parameter $\lambda_t \in [0, 1/\mu_0)$ from past data
    /// $\mathcal{F}_{t-1}$.
    ///
    /// Uses a predictable EWMA estimate $\tilde{\mu}_{t-1}$ of recent exceedance rate so
    /// changepoints at arbitrary $t_0 \gg 1$ are detected without burying $\log E_t$ during
    /// nominal prefix periods, while strictly preserving the supermartingale property under $H_0$.
    fn predictable_lambda(&self) -> f64 {
        let mu0 = self.config.null_bound;
        let cap = self.config.max_lambda.min(0.95 / mu0);
        if self.step_count == 0 {
            return 0.0;
        }
        let gap = self.ewma_mean - mu0;
        if gap <= 0.0 {
            0.0
        } else {
            let denom = (mu0 * (1.0 - mu0)).max(1.0e-12);
            let kelly = gap / denom;
            // Smoothly ramp from quadratic to full Kelly as gap approaches mu0 so
            // single-observation fluctuations under H0 do not erode log_e_value.
            let ramp = (gap / mu0).clamp(0.0, 1.0);
            (kelly * ramp).clamp(0.0, cap)
        }
    }

    /// Ingest a new bounded sentinel observation $X_t \in [0, 1]$ (e.g. exceedance indicator
    /// $\mathbf{1}\{\omega_t > \text{tol}\}$ or normalized residual).
    pub fn observe(&mut self, discrepancy: f64) -> EProcessStatus {
        let x = discrepancy.clamp(0.0, 1.0);
        let lambda = self.predictable_lambda();

        // Betting e-process step: E_t = E_{t-1} * (1 + lambda_t * (X_t - mu_0))
        let delta = (lambda * (x - self.config.null_bound)).max(-0.999_999);
        let log_increment = delta.ln_1p();
        self.log_e_value += log_increment;
        self.step_count += 1;
        self.running_sum += x;
        self.ewma_mean = (1.0 - EWMA_GAMMA) * self.ewma_mean + EWMA_GAMMA * x;

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
        self.ewma_mean = self.config.null_bound;
        self.alarm_triggered = false;
        self.alarm_step = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn splitmix64(state: &mut u64) -> f64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

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

    /// frankenscipy-7tb8d.4 acceptance 1: under H0 (Bernoulli(mu0)) over 10,000 seeded streams
    /// of length 5,000, the anytime-valid false-alarm rate is <= alpha + 3-sigma slack.
    #[test]
    fn ville_inequality_holds_over_10000_seeded_bernoulli_null_streams() {
        let mu0 = 0.05;
        let alpha = 0.01;
        let num_streams = 10_000usize;
        let horizon = 5_000usize;
        let config = EProcessConfig {
            null_bound: mu0,
            alpha,
            max_lambda: 2.5,
        };

        let mut false_alarms = 0usize;
        for stream_idx in 0..num_streams {
            let mut rng = 0xD1B5_4A32_D192_ED03_u64 ^ (stream_idx as u64).wrapping_mul(0x9E37_79B9);
            let mut monitor = EProcessMonitor::new(config.clone()).unwrap();
            for _ in 0..horizon {
                let x = if splitmix64(&mut rng) < mu0 { 1.0 } else { 0.0 };
                if monitor.observe(x) == EProcessStatus::AlarmTriggered {
                    false_alarms += 1;
                    break;
                }
            }
        }

        let false_alarm_rate = false_alarms as f64 / num_streams as f64;
        let three_sigma = 3.0 * (alpha * (1.0 - alpha) / num_streams as f64).sqrt();
        println!(
            "Ville H0 test: false_alarms={false_alarms}/{num_streams} (rate={false_alarm_rate:.5}, bound={:.5})",
            alpha + three_sigma
        );
        assert!(
            false_alarm_rate <= alpha + three_sigma,
            "false_alarm_rate={false_alarm_rate:.5} exceeded alpha + 3*sigma = {:.5} (mu0={mu0}, alpha={alpha})",
            alpha + three_sigma
        );
    }

    /// frankenscipy-7tb8d.4 acceptance 2: under an injected shift from Bernoulli(mu0) to
    /// Bernoulli(3*mu0) at t = 1000, alarm triggers quickly with documented median delay.
    #[test]
    fn changepoint_shift_to_3x_mu0_at_t1000_triggers_alarm_quickly() {
        let mu0 = 0.05;
        let shifted = 3.0 * mu0;
        let alpha = 0.01;
        let shift_step = 1_000usize;
        let horizon = 2_500usize;
        let num_trials = 200usize;
        let config = EProcessConfig {
            null_bound: mu0,
            alpha,
            max_lambda: 2.5,
        };

        let mut delays = Vec::with_capacity(num_trials);
        for trial in 0..num_trials {
            let mut rng = 0xA5A5_5A5A_F00D_CAFE_u64 ^ (trial as u64).wrapping_mul(0x517C_C1B7);
            let mut monitor = EProcessMonitor::new(config.clone()).unwrap();
            for step in 1..=horizon {
                let p = if step <= shift_step { mu0 } else { shifted };
                let x = if splitmix64(&mut rng) < p { 1.0 } else { 0.0 };
                if monitor.observe(x) == EProcessStatus::AlarmTriggered {
                    if step > shift_step {
                        delays.push(step - shift_step);
                    }
                    break;
                }
            }
        }

        assert!(
            delays.len() >= num_trials * 9 / 10,
            "at least 90% of post-shift streams should alarm within horizon, got {}/{}",
            delays.len(),
            num_trials
        );
        delays.sort_unstable();
        let median_delay = delays[delays.len() / 2];
        println!(
            "Changepoint t=1000 Bernoulli({mu0}) -> Bernoulli({shifted}): detected={}/{num_trials}, median_delay={median_delay} steps",
            delays.len()
        );
        assert!(
            median_delay <= 150,
            "median detection delay {median_delay} exceeded 150 steps after t=1000 shift"
        );
    }
}
