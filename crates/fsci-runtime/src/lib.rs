#![forbid(unsafe_code)]

//! FrankenSciPy runtime: CASP (Condition-Aware Solver Portfolio) engine
//! and policy decision infrastructure.
//!
//! ## Module layout
//!
//! | Module      | Contents                                                   |
//! |-------------|-----------------------------------------------------------|
//! | `mode`      | [`RuntimeMode`] enum (Strict / Hardened)                   |
//! | `signals`   | [`DecisionSignals`], [`SignalSequence`] for replay testing |
//! | `evidence`  | [`PolicyEvidenceLedger`], [`DecisionEvidenceEntry`]        |
//! | `policy`    | [`PolicyController`], loss matrices, risk-state model      |
//! | `warnings`  | SciPy's warning classes, [`warn`] and [`catch_warnings`]    |
//! | `deterministic_math` | [`deterministic_exp`], [`deterministic_log10`]: the same bits on every platform |

#[doc(hidden)]
#[cfg_attr(
    not(feature = "bench-harness"),
    doc = "```compile_fail\nuse fsci_runtime::booking_claim;\n```\n```compile_fail\nuse fsci_runtime::scipy_incumbent;\n```"
)]
pub struct __BenchHarnessGatedDoctests;

#[cfg(feature = "bench-harness")]
pub mod booking_claim;
mod calibrated_losses;
mod deterministic_math;
pub mod eprocess;
pub mod evidence;
pub mod mode;
pub mod policy;
#[cfg(feature = "bench-harness")]
pub mod scipy_incumbent;
pub mod signals;
pub mod supervision;
pub mod warnings;

// ── Re-exports: preserve the flat public API ────────────────────────
#[cfg(feature = "bench-harness")]
pub use booking_claim::{BookingClaim, ClaimRejection, FleetBooking};
pub use deterministic_math::{deterministic_exp, deterministic_log10};
pub use eprocess::{EProcessConfig, EProcessMonitor, EProcessStatus};
pub use evidence::{
    AlienArtifactDecision, AuditAction, AuditEvent, AuditLedger, AuditScope, DecisionEvidenceEntry,
    EvidenceValue, Fingerprinter, PolicyEvidenceLedger, SharedAuditLedger, SyncSharedAuditLedger,
    audit_finish, audit_recover, audit_reject,
};
pub use mode::{HARDENED_MAX_DIM, RecoveryPolicy, RuntimeMode};
pub use policy::{PolicyAction, PolicyController, PolicyDecision, RiskState, decision_loss_matrix};
pub use signals::{DecisionSignals, SignalSequence};
pub use supervision::{PolicySupervisor, SupervisionConfig, SupervisionEvent, SupervisionStatus};
pub use warnings::{Warning, WarningCategory, catch_warnings, warn};

use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════
// CASP — Condition-Aware Solver Portfolio (§0.4)
// ═══════════════════════════════════════════════════════════════════

/// Read access to a CASP portfolio's recorded decisions, the same for all five portfolios
/// (frankenscipy-7tb8d.11). Each keeps its newest `evidence_capacity` entries, oldest first.
pub trait PortfolioEvidence {
    /// One recorded decision.
    type Entry: Serialize;
    /// The portfolio's name in its JSONL lines: `solver`, `sparse`, `opt`, `ode` or `hyper`.
    const NAME: &'static str;

    /// The runtime mode the portfolio decides in.
    fn mode(&self) -> RuntimeMode;

    /// The recorded decisions, oldest first.
    fn evidence(&self) -> &VecDeque<Self::Entry>;

    /// The recorded decisions as JSONL, one object per line, oldest first: `portfolio` and
    /// `mode`, then the entry's own fields. A line that fails to serialize is left out.
    fn serialize_jsonl(&self) -> String {
        #[derive(Serialize)]
        struct Line<'a, E> {
            portfolio: &'static str,
            mode: RuntimeMode,
            #[serde(flatten)]
            entry: &'a E,
        }
        let mode = self.mode();
        let mut output = Vec::with_capacity(self.evidence().len().saturating_mul(256));
        for entry in self.evidence() {
            let entry_start = output.len();
            if entry_start != 0 {
                output.push(b'\n');
            }
            let line = Line {
                portfolio: Self::NAME,
                mode,
                entry,
            };
            if serde_json::to_writer(&mut output, &line).is_err() {
                output.truncate(entry_start);
            }
        }
        String::from_utf8(output).expect("serde_json always emits UTF-8")
    }
}

/// [`PortfolioEvidence`] for the five portfolios, whose fields share names.
macro_rules! portfolio_evidence {
    ($($portfolio:ty => $entry:ty, $name:literal;)+) => {$(
        impl PortfolioEvidence for $portfolio {
            type Entry = $entry;
            const NAME: &'static str = $name;

            fn mode(&self) -> RuntimeMode {
                self.mode
            }

            fn evidence(&self) -> &VecDeque<$entry> {
                &self.evidence
            }
        }
    )+};
}

portfolio_evidence! {
    SolverPortfolio => SolverEvidenceEntry, "solver";
    SparseSolverPortfolio => SparseSolverEvidenceEntry, "sparse";
    OptSolverPortfolio => OptSolverEvidenceEntry, "opt";
    OdeSolverPortfolio => OdeSolverEvidenceEntry, "ode";
    HyperSolverPortfolio => HyperSolverEvidenceEntry, "hyper";
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatrixConditionState {
    WellConditioned,
    ModerateCondition,
    IllConditioned,
    NearSingular,
}

impl MatrixConditionState {
    pub const ALL: [Self; 4] = [
        Self::WellConditioned,
        Self::ModerateCondition,
        Self::IllConditioned,
        Self::NearSingular,
    ];

    #[must_use]
    pub fn index(self) -> usize {
        match self {
            Self::WellConditioned => 0,
            Self::ModerateCondition => 1,
            Self::IllConditioned => 2,
            Self::NearSingular => 3,
        }
    }
}

/// Structural evidence provided to the portfolio (§0.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StructuralEvidence {
    General,
    Diagonal,
    Triangular,
    /// Exactly symmetric (and neither diagonal nor triangular), or declared symmetric /
    /// positive definite: SciPy factors it as symmetric (Cholesky, then Bunch–Kaufman LDLᵀ).
    Symmetric,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolverAction {
    DirectLU,
    PivotedQR,
    SVDFallback,
    DiagonalFastPath,
    TriangularFastPath,
    /// SciPy's factorization of a symmetric matrix: Cholesky (`potrf`), falling back to
    /// Bunch–Kaufman LDLᵀ (`sytrf`) where a pivot is not positive. Fails only on an exactly
    /// singular `D`, or on a non-positive-definite matrix declared positive definite.
    SymmetricFastPath,
}

impl SolverAction {
    pub const ALL: [Self; 6] = [
        Self::DirectLU,
        Self::PivotedQR,
        Self::SVDFallback,
        Self::DiagonalFastPath,
        Self::TriangularFastPath,
        Self::SymmetricFastPath,
    ];

    #[must_use]
    pub fn index(self) -> usize {
        match self {
            Self::DirectLU => 0,
            Self::PivotedQR => 1,
            Self::SVDFallback => 2,
            Self::DiagonalFastPath => 3,
            Self::TriangularFastPath => 4,
            Self::SymmetricFastPath => 5,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolverEvidenceEntry {
    pub component: &'static str,
    pub matrix_shape: (usize, usize),
    pub rcond_estimate: f64,
    pub chosen_action: SolverAction,
    pub posterior: Vec<f64>,
    pub expected_losses: Vec<f64>,
    pub chosen_expected_loss: f64,
    pub fallback_active: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backward_error: Option<f64>,
}

/// Expected-loss solver selection engine (§0.4 alien-artifact).
///
/// The loss matrix (6 actions × 4 states) is measured, not typed: [`Self::default_loss_matrix`]
/// is the generated `calibrated_losses::SOLVER_LOSS_MATRIX`, from
/// `cargo run --release -p fsci-linalg --bin casp_calibrate`, and its report
/// `artifacts/casp-calibration-solver.json` holds the corpus, every matrix's outcome per action,
/// the failure rates and the rule (frankenscipy-7tb8d.2). Measured, LU's backward error passes
/// the portfolio's acceptance test at every conditioning except where pivot growth defeats it,
/// so LU has the least expected loss in every state. QR (twice LU's flops) is the fallback
/// when LU's backward error fails. The SVD refuses a square system whose numerical rank is short
/// (its rank cutoff), about a quarter of the ill-conditioned matrices, so it comes last. The
/// hand-set matrix this replaced sent moderate conditioning to QR and ill conditioning to the SVD.
///
/// Decision: a* = argmin_a Σ_s L(a,s) × P(s|evidence), where an action with attempts recorded
/// at this rcond decade has the failure term of its loss replaced by what they showed
/// (frankenscipy-3iekl, [`learned_loss`]).
#[derive(Debug, Clone)]
pub struct SolverPortfolio {
    mode: RuntimeMode,
    loss_matrix: [[f64; 4]; 6],
    evidence: VecDeque<SolverEvidenceEntry>,
    evidence_capacity: usize,
    calibrator: ConformalCalibrator,
    /// Offline Conformal Risk Control (CRC) escalation rule with finite-sample guarantee
    /// (frankenscipy-7tb8d.3).
    crc: ConformalRiskControl,
    /// Persistent risk-state policy controller, runtime supervisor, and anytime-valid
    /// e-process drift monitor (frankenscipy-7tb8d.4).
    policy_controller: PolicyController,
    /// Dirichlet outcome counts over the four condition states, one row per decade of rcond
    /// (see [`rcond_decade`]). Filled by [`SolverPortfolio::record_outcome`].
    outcome_counts: [[f64; 4]; RCOND_DECADES],
    /// `[failures, trials]` per rcond decade and action, from the same recorded outcomes
    /// (frankenscipy-3iekl). They replace the calibrated failure term of an action's expected
    /// loss (see [`learned_loss`]). Unlike the state counts they can move the decision: the
    /// calibrated losses rank LU first in every state, so no state posterior can.
    failure_counts: [[[f64; 2]; 6]; RCOND_DECADES],
}

/// How an attempted dense solve turned out, as fed back to the portfolio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttemptOutcome {
    /// Returned a solution meeting the accuracy test.
    Ok,
    /// Returned a solution whose backward error failed the accuracy test.
    Inaccurate,
    /// Returned an error (singular factor, breakdown, ...).
    Failed,
}

/// Decades of rcond the outcome counts are kept for: `[1e0, 1e-1)`, ..., `[1e-16, 1e-17)`, and
/// everything at or below 1e-17 (including 0 and NaN) in the last one.
const RCOND_DECADES: usize = 18;

/// Pseudo-count mass of the prior map in the Dirichlet posterior (per state row, total 4).
const PRIOR_MASS: f64 = 4.0;

/// Per-state smoothing used when ATTRIBUTING an outcome to states, so that a state the prior
/// map gives zero weight can still accumulate evidence. It is not part of the decision
/// posterior (see [`SolverPortfolio::posterior`]).
const STATE_SMOOTHING: f64 = 0.05;

/// P(`AttemptOutcome::Ok` | state, action) for the observation model. Rows follow
/// `SolverAction::index`, columns the condition states (well, moderate, ill, near-singular).
/// Hand-set, like the loss matrix; a calibrated table is frankenscipy-7tb8d.2.
const OK_PROBABILITY: [[f64; 4]; 6] = [
    [0.99, 0.95, 0.40, 0.05], // DirectLU
    [0.99, 0.98, 0.70, 0.20], // PivotedQR
    [0.99, 0.99, 0.95, 0.90], // SVDFallback
    [0.99, 0.99, 0.99, 0.30], // DiagonalFastPath
    [0.99, 0.99, 0.95, 0.30], // TriangularFastPath
    [0.99, 0.99, 0.95, 0.30], // SymmetricFastPath
];

/// Share of the non-`Ok` probability that is `Inaccurate` (the rest is `Failed`).
const INACCURATE_SHARE: f64 = 0.7;

/// Pseudo-trials the calibrated failure rate carries in the Beta posterior of an action's
/// failure rate at an rcond decade (frankenscipy-3iekl).
const FAILURE_PRIOR_TRIALS: f64 = 4.0;

/// An action's expected loss once attempts with it at this rcond decade have been recorded
/// (frankenscipy-3iekl). The calibrated loss is `cost + p0 × r`: `p0` the measured failure rate
/// under the state posterior and `r` the expected recovery cost it implies. The recorded
/// attempts replace `p0` by the Beta posterior mean `(m·p0 + failures) / (m + trials)`, with `m`
/// = [`FAILURE_PRIOR_TRIALS`], and keep `r`. An action the corpus never saw fail implies no
/// `r`; its failure is priced at the cheapest other general solver, the one that would recover.
fn learned_loss(
    calibrated: f64,
    action: usize,
    posterior: [f64; 4],
    failures: f64,
    trials: f64,
) -> f64 {
    let cost = calibrated_losses::SOLVER_ACTION_COST[action];
    let p0: f64 = calibrated_losses::SOLVER_FAILURE_RATE[action]
        .iter()
        .zip(posterior)
        .map(|(rate, p)| rate * p)
        .sum();
    let failure_term = calibrated - cost;
    let recovery = if p0 > 0.0 && failure_term > 0.0 {
        failure_term / p0
    } else {
        (0..3)
            .filter(|&other| other != action)
            .map(|other| calibrated_losses::SOLVER_ACTION_COST[other])
            .fold(f64::INFINITY, f64::min)
    };
    let failure_rate = (FAILURE_PRIOR_TRIALS * p0 + failures) / (FAILURE_PRIOR_TRIALS + trials);
    cost + failure_rate * recovery
}

/// The decade row an rcond value's outcomes are counted in.
fn rcond_decade(rcond: f64) -> usize {
    if !rcond.is_finite() || rcond <= 0.0 {
        return RCOND_DECADES - 1;
    }
    // Not the platform libm: a last-bit difference at a decade would count the outcome in
    // another row on another platform (fsci-runtime deterministic_math).
    let decade = (-deterministic_log10(rcond)).floor();
    if decade <= 0.0 {
        0
    } else {
        (decade as usize).min(RCOND_DECADES - 1)
    }
}

impl SolverPortfolio {
    #[must_use]
    pub fn new(mode: RuntimeMode, evidence_capacity: usize) -> Self {
        let evidence_capacity = evidence_capacity.max(1);
        let supervision_config = match mode {
            RuntimeMode::Strict => SupervisionConfig {
                max_consecutive_fallbacks: 3,
                max_consecutive_failures: usize::MAX,
                recovery_budget: usize::MAX,
                cooldown_steps: 5,
            },
            RuntimeMode::Hardened => SupervisionConfig::default(),
        };
        Self {
            mode,
            loss_matrix: Self::default_loss_matrix(),
            evidence: VecDeque::with_capacity(evidence_capacity),
            evidence_capacity,
            calibrator: ConformalCalibrator::new(calibrated_losses::SOLVER_CRC_ALPHA, 200),
            crc: ConformalRiskControl::default_calibrated(),
            policy_controller: PolicyController::with_supervision(
                mode,
                evidence_capacity,
                supervision_config,
            ),
            outcome_counts: [[0.0; 4]; RCOND_DECADES],
            failure_counts: [[[0.0; 2]; 6]; RCOND_DECADES],
        }
    }

    /// Posterior over the four condition states at `rcond`.
    ///
    /// A Dirichlet posterior whose prior is the fixed rcond map ([`Self::condition_posterior`])
    /// with pseudo-count mass [`PRIOR_MASS`], updated by the outcome counts recorded for this
    /// rcond decade. With no outcomes recorded for the decade it returns the prior map EXACTLY,
    /// so a fresh portfolio decides bit-for-bit as the map alone did (frankenscipy-7tb8d.1).
    ///
    /// The decision posterior carries NO smoothing mass: smoothing is only used to attribute
    /// an outcome to states in [`Self::record_outcome`]. Putting it here would move ~1% onto
    /// NearSingular after a single SUCCESSFUL LU at rcond 1e-3 and, with LU's loss of 120
    /// there, flip the next decision away from the action that just succeeded.
    #[must_use]
    pub fn posterior(&self, rcond: f64) -> [f64; 4] {
        let counts = &self.outcome_counts[rcond_decade(rcond)];
        let observed: f64 = counts.iter().sum();
        if observed == 0.0 {
            return Self::condition_posterior(rcond);
        }
        let prior = Self::condition_posterior(rcond);
        let total = PRIOR_MASS + observed;
        let mut posterior = [0.0; 4];
        for (state, p) in posterior.iter_mut().enumerate() {
            *p = (PRIOR_MASS * prior[state] + counts[state]) / total;
        }
        posterior
    }

    /// The posterior used to ATTRIBUTE an outcome: [`Self::posterior`] plus
    /// [`STATE_SMOOTHING`] per state, so a state the prior map gives zero weight can still
    /// receive responsibility for an outcome only it explains.
    fn smoothed_posterior(&self, rcond: f64) -> [f64; 4] {
        let prior = Self::condition_posterior(rcond);
        let counts = &self.outcome_counts[rcond_decade(rcond)];
        let observed: f64 = counts.iter().sum();
        let total = PRIOR_MASS + 4.0 * STATE_SMOOTHING + observed;
        let mut posterior = [0.0; 4];
        for (state, p) in posterior.iter_mut().enumerate() {
            *p = (PRIOR_MASS * prior[state] + STATE_SMOOTHING + counts[state]) / total;
        }
        posterior
    }

    /// Feed back how an attempt with `action` at `rcond` turned out.
    ///
    /// The condition state is latent, so the outcome contributes a fractional count to each
    /// state in proportion to its responsibility P(state | outcome, action) under the current
    /// (smoothed) posterior and the [`OK_PROBABILITY`] observation model. Repeated inaccurate
    /// LU results at an rcond the map calls well conditioned therefore move that decade's
    /// posterior toward the ill-conditioned states, and the next decision there changes.
    pub fn record_outcome(&mut self, rcond: f64, action: SolverAction, outcome: AttemptOutcome) {
        let tally = &mut self.failure_counts[rcond_decade(rcond)][action.index()];
        tally[1] += 1.0;
        if outcome != AttemptOutcome::Ok {
            tally[0] += 1.0;
        }
        let prior = self.smoothed_posterior(rcond);
        let ok = OK_PROBABILITY[action.index()];
        let mut responsibility = [0.0; 4];
        for state in 0..4 {
            let likelihood = match outcome {
                AttemptOutcome::Ok => ok[state],
                AttemptOutcome::Inaccurate => (1.0 - ok[state]) * INACCURATE_SHARE,
                AttemptOutcome::Failed => (1.0 - ok[state]) * (1.0 - INACCURATE_SHARE),
            };
            responsibility[state] = prior[state] * likelihood;
        }
        let norm: f64 = responsibility.iter().sum();
        if !norm.is_finite() || norm <= 0.0 {
            return;
        }
        let row = &mut self.outcome_counts[rcond_decade(rcond)];
        for (count, r) in row.iter_mut().zip(responsibility) {
            *count += r / norm;
        }
    }

    /// Outcome counts recorded for `rcond`'s decade (for diagnostics and audit).
    #[must_use]
    pub fn outcome_counts(&self, rcond: f64) -> [f64; 4] {
        self.outcome_counts[rcond_decade(rcond)]
    }

    #[must_use]
    pub const fn default_loss_matrix() -> [[f64; 4]; 6] {
        calibrated_losses::SOLVER_LOSS_MATRIX
    }

    /// Select optimal action via expected-loss minimization.
    /// Returns (action, posterior, expected_losses, chosen_loss).
    pub fn select_action(
        &self,
        rcond: f64,
        structure: Option<StructuralEvidence>,
    ) -> (SolverAction, [f64; 4], [f64; 6], f64) {
        self.select_action_excluding(rcond, structure, &[])
            .expect("the three general solvers are never all excluded here")
    }

    /// [`Self::select_action`] restricted to the actions NOT in `excluded`: the fallback path
    /// re-ranks the remaining actions under the posterior as it stands after the failed
    /// attempts were recorded, instead of walking a ranking fixed before the first attempt.
    /// `None` when every applicable action is excluded.
    #[must_use]
    pub fn select_action_excluding(
        &self,
        rcond: f64,
        structure: Option<StructuralEvidence>,
        excluded: &[SolverAction],
    ) -> Option<(SolverAction, [f64; 4], [f64; 6], f64)> {
        let posterior = self.posterior(rcond);
        let losses = self.compute_expected_losses(posterior, rcond);

        // If conformal calibrator or anytime-valid e-process detects drift, override to SVDFallback
        if (self.calibrator.should_fallback()
            || self.policy_controller.eprocess().is_alarm_triggered())
            && !excluded.contains(&SolverAction::SVDFallback)
        {
            return Some((
                SolverAction::SVDFallback,
                posterior,
                losses,
                losses[SolverAction::SVDFallback.index()],
            ));
        }

        // argmin over expected losses
        // We consider general solvers (0, 1, 2) and the one fast path the structure admits
        // (3, 4 or 5). Use stack-allocated array to avoid heap allocation in tight loops.
        let mut candidates = [0, 1, 2, 0];
        let mut count = 3;
        match structure {
            Some(StructuralEvidence::Diagonal) => {
                candidates[3] = 3;
                count = 4;
            }
            Some(StructuralEvidence::Triangular) => {
                candidates[3] = 4;
                count = 4;
            }
            Some(StructuralEvidence::Symmetric) => {
                candidates[3] = 5;
                count = 4;
            }
            Some(StructuralEvidence::General) | None => {}
        }

        let mut best: Option<(usize, f64)> = None;
        for &idx in candidates.iter().take(count) {
            if excluded.contains(&SolverAction::ALL[idx]) {
                continue;
            }
            let loss = losses[idx];
            best = match best {
                None => Some((idx, loss)),
                Some((_, best_loss)) if loss < best_loss => Some((idx, loss)),
                Some((best_idx, best_loss)) if (loss - best_loss).abs() <= 1e-12 => {
                    // Tie-break toward safer action for general solvers (higher index = safer:
                    // LU < QR < SVD) or toward fast paths if they have equal expected loss.
                    if (idx < 3 && idx > best_idx) || (idx >= 3 && best_idx < 3) {
                        Some((idx, loss))
                    } else {
                        Some((best_idx, best_loss))
                    }
                }
                keep => keep,
            };
        }

        best.map(|(idx, loss)| {
            let chosen = SolverAction::ALL[idx];
            if chosen == SolverAction::DirectLU
                && self.crc.should_escalate(rcond)
                && !excluded.contains(&SolverAction::PivotedQR)
            {
                let qr_idx = SolverAction::PivotedQR.index();
                (SolverAction::PivotedQR, posterior, losses, losses[qr_idx])
            } else {
                (chosen, posterior, losses, loss)
            }
        })
    }

    /// Record solver evidence for audit trail and calibration.
    pub fn record_evidence(&mut self, entry: SolverEvidenceEntry) {
        if let Some(err) = entry.backward_error {
            let exceeded =
                err > self.calibrator.violation_threshold || !err.is_finite() || err < 0.0;
            self.calibrator.observe(err);
            let _ = self.policy_controller.observe_exceedance(exceeded);
        }
        if self.evidence.len() >= self.evidence_capacity {
            let _ = self.evidence.pop_front();
        }
        self.evidence.push_back(entry);
    }

    /// Update conformal calibrator and e-process drift monitor with observed backward error.
    pub fn observe_backward_error(&mut self, backward_error: f64) {
        let exceeded = backward_error > self.calibrator.violation_threshold
            || !backward_error.is_finite()
            || backward_error < 0.0;
        self.calibrator.observe(backward_error);
        let _ = self.policy_controller.observe_exceedance(exceeded);
    }

    #[must_use]
    pub fn evidence_len(&self) -> usize {
        self.evidence.len()
    }

    /// A digest of everything [`Self::select_action`] reads: the mode, the loss matrix, the
    /// outcome and failure counts, the offline CRC threshold, and the online calibrator's state,
    /// as a [`Fingerprinter`] `"blake3:…"` (frankenscipy-7tb8d.12). Two portfolios with one
    /// digest decide identically. A solve certificate records the digest of the portfolio that
    /// made its decision, so the decision replays against a snapshot of that portfolio, and a
    /// snapshot in any other state is refused rather than giving a different answer.
    #[must_use]
    pub fn state_digest(&self) -> String {
        let mut fingerprinter = Fingerprinter::new("fsci_runtime::SolverPortfolio::state");
        fingerprinter
            .str(&format!("{:?}", self.mode))
            .f64s(self.loss_matrix.as_flattened())
            .f64s(self.outcome_counts.as_flattened())
            .f64s(self.failure_counts.as_flattened().as_flattened());
        self.calibrator.fingerprint_into(&mut fingerprinter);
        self.crc.fingerprint_into(&mut fingerprinter);
        if self.policy_controller.eprocess().is_alarm_triggered() {
            fingerprinter.bool(true);
        }
        fingerprinter.finish()
    }

    #[must_use]
    pub const fn mode(&self) -> RuntimeMode {
        self.mode
    }

    #[must_use]
    pub fn calibrator(&self) -> &ConformalCalibrator {
        &self.calibrator
    }

    /// Target risk level `alpha` (default `0.05`) for Conformal Risk Control escalation
    /// (frankenscipy-7tb8d.3).
    #[must_use]
    pub const fn alpha(&self) -> f64 {
        self.crc.alpha()
    }

    /// Set the portfolio's target risk level `alpha` (clamped to `[0.001, 0.5]`), updating both
    /// the offline [`ConformalRiskControl`] escalation threshold `t_hat` and the online drift
    /// monitor's `alpha` (frankenscipy-7tb8d.3).
    pub fn set_alpha(&mut self, alpha: f64) {
        self.crc.set_alpha(alpha);
        self.calibrator.alpha = self.crc.alpha();
    }

    /// Offline Conformal Risk Control escalation state (frankenscipy-7tb8d.3).
    #[must_use]
    pub const fn crc(&self) -> &ConformalRiskControl {
        &self.crc
    }

    /// Mutable access to the offline Conformal Risk Control escalation state.
    pub fn crc_mut(&mut self) -> &mut ConformalRiskControl {
        &mut self.crc
    }

    /// Replace the portfolio's offline Conformal Risk Control calibration.
    pub fn set_crc(&mut self, crc: ConformalRiskControl) {
        self.calibrator.alpha = crc.alpha();
        self.crc = crc;
    }

    #[must_use]
    pub const fn policy_controller(&self) -> &PolicyController {
        &self.policy_controller
    }

    pub fn policy_controller_mut(&mut self) -> &mut PolicyController {
        &mut self.policy_controller
    }

    /// Σ_s L(a,s) · P(s), with the failure term of every action that has recorded attempts at
    /// `rcond`'s decade replaced by what they showed ([`learned_loss`]). An action with none
    /// keeps the calibrated value exactly, so a fresh portfolio decides as before, bit for bit.
    fn compute_expected_losses(&self, posterior: [f64; 4], rcond: f64) -> [f64; 6] {
        let mut losses = [0.0; 6];
        for (action_idx, row) in self.loss_matrix.iter().enumerate() {
            losses[action_idx] = row.iter().zip(posterior.iter()).map(|(l, p)| l * p).sum();
        }
        let tallies = &self.failure_counts[rcond_decade(rcond)];
        for (action_idx, loss) in losses.iter_mut().enumerate() {
            let [failures, trials] = tallies[action_idx];
            if trials > 0.0 {
                *loss = learned_loss(*loss, action_idx, posterior, failures, trials);
            }
        }
        losses
    }

    /// `[failures, trials]` recorded per action at `rcond`'s decade, in `SolverAction::ALL`
    /// order (for diagnostics and audit).
    #[must_use]
    pub fn failure_counts(&self, rcond: f64) -> [[f64; 2]; 6] {
        self.failure_counts[rcond_decade(rcond)]
    }

    /// The PRIOR map from rcond to condition states: piecewise-linear in `log10(rcond)` between
    /// the state centres -2, -6, -11 and -16 (all mass on the end states outside that range,
    /// on NearSingular for a non-finite or non-positive rcond). [`Self::posterior`] updates it
    /// with recorded outcomes.
    fn condition_posterior(rcond: f64) -> [f64; 4] {
        // Guard: if rcond is NaN/Inf, assume worst case (NearSingular)
        if !rcond.is_finite() || rcond <= 0.0 {
            return [0.0, 0.0, 0.0, 1.0];
        }

        // Centers of states (log10 rcond):
        // Well: -2.0, Mod: -6.0, Ill: -11.0, NearSing: -16.0
        let centers = [-2.0, -6.0, -11.0, -16.0];
        if rcond >= 1e-2 {
            return [1.0, 0.0, 0.0, 0.0];
        }
        if rcond <= 1e-16 {
            return [0.0, 0.0, 0.0, 1.0];
        }

        let log_r = deterministic_log10(rcond);
        let mut p = [0.0; 4];

        // Find the interval [centers[i+1], centers[i]] that contains log_r
        for i in 0..3 {
            let c_upper = centers[i];
            let c_lower = centers[i + 1];
            if log_r <= c_upper && log_r >= c_lower {
                let weight_upper = (log_r - c_lower) / (c_upper - c_lower);
                p[i] = weight_upper;
                p[i + 1] = 1.0 - weight_upper;
                break;
            }
        }
        p
    }
}

/// Conformal calibration guard (§12.1).
/// Tracks nonconformity scores and triggers SVD fallback when
/// empirical miscoverage exceeds target rate.
#[derive(Debug, Clone)]
pub struct ConformalCalibrator {
    alpha: f64,
    scores: VecDeque<f64>,
    capacity: usize,
    violation_threshold: f64,
    coverage_violations: usize,
    total_predictions: usize,
}

impl ConformalCalibrator {
    /// Feed the calibrator's whole state to a fingerprint: its parameters, its counters and
    /// its score window, oldest first.
    fn fingerprint_into(&self, fingerprinter: &mut Fingerprinter) {
        let scores: Vec<f64> = self.scores.iter().copied().collect();
        fingerprinter
            .f64(self.alpha)
            .usize(self.capacity)
            .f64(self.violation_threshold)
            .usize(self.coverage_violations)
            .usize(self.total_predictions)
            .f64s(&scores);
    }

    #[must_use]
    pub fn new(alpha: f64, capacity: usize) -> Self {
        let capacity = capacity.max(10);
        Self {
            alpha: alpha.clamp(0.001, 0.5),
            scores: VecDeque::with_capacity(capacity),
            capacity,
            violation_threshold: 1e-8,
            coverage_violations: 0,
            total_predictions: 0,
        }
    }

    /// Record a nonconformity score (e.g., backward error).
    pub fn observe(&mut self, score: f64) {
        let normalized = if score.is_finite() && score >= 0.0 {
            score
        } else {
            // Treat invalid scores as violations to fail closed.
            f64::INFINITY
        };
        if self.scores.len() >= self.capacity {
            // Remove oldest and adjust violation count
            if let Some(old) = self.scores.pop_front()
                && old > self.violation_threshold
            {
                self.coverage_violations = self.coverage_violations.saturating_sub(1);
            }
        }
        self.total_predictions += 1;
        if normalized > self.violation_threshold {
            self.coverage_violations += 1;
        }
        self.scores.push_back(normalized);
    }

    /// Check if empirical miscoverage rate exceeds alpha + epsilon.
    /// When true, the portfolio should fall back to SVD (safest action).
    #[must_use]
    pub fn should_fallback(&self) -> bool {
        if self.scores.len() < 10 {
            return false; // Not enough data for calibration
        }
        let empirical_miscoverage = self.coverage_violations as f64 / self.scores.len() as f64;
        let epsilon = 0.02; // tolerance band
        empirical_miscoverage > self.alpha + epsilon
    }

    #[must_use]
    pub fn empirical_miscoverage(&self) -> f64 {
        if self.scores.is_empty() {
            return 0.0;
        }
        self.coverage_violations as f64 / self.scores.len() as f64
    }

    #[must_use]
    pub fn total_predictions(&self) -> usize {
        self.total_predictions
    }

    #[must_use]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Set the violation threshold for nonconformity scores.
    pub fn set_violation_threshold(&mut self, threshold: f64) {
        self.violation_threshold = threshold.max(0.0);
        // Recompute violations after threshold change
        self.coverage_violations = self
            .scores
            .iter()
            .filter(|&&s| s > self.violation_threshold)
            .count();
    }
}

/// One calibration problem for [`ConformalRiskControl`] (frankenscipy-7tb8d.3).
///
/// For an escalation threshold `t in [0.0, 1.0]` on `rcond`:
/// - when `rcond > t`, the unescalated solver (e.g. `DirectLU`) runs and incurs indicator loss
///   `1.0` if `unescalated_exceeded` (`omega_i(DirectLU) > tol`) and `0.0` otherwise;
/// - when `rcond <= t` (or `!rcond.is_finite()`), the portfolio escalates to the conservative
///   solver (e.g. `PivotedQR`) and incurs indicator loss `1.0` if `escalated_exceeded` and `0.0`
///   otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CrcCalibrationSample {
    pub rcond: f64,
    pub unescalated_exceeded: bool,
    pub escalated_exceeded: bool,
}

impl CrcCalibrationSample {
    /// Evaluate the bounded loss `L_i(t) = 1{omega_i(action_i(t)) > tol} in {0.0, 1.0}` at
    /// escalation threshold `t`.
    #[must_use]
    pub fn loss_at(&self, t: f64) -> f64 {
        let escalated = !self.rcond.is_finite() || self.rcond <= t;
        let exceeded = if escalated {
            self.escalated_exceeded
        } else {
            self.unescalated_exceeded
        };
        if exceeded { 1.0 } else { 0.0 }
    }

    /// Check the Conformal Risk Control precondition that `L_i(t)` is monotone non-increasing
    /// in `t`: escalating at `rcond <= t` never turns an accepted solve into an exceedance.
    #[must_use]
    pub const fn is_monotone(&self) -> bool {
        !self.escalated_exceeded || self.unescalated_exceeded
    }
}

/// Offline Conformal Risk Control (CRC) escalation threshold selector with a finite-sample
/// risk guarantee (Angelopoulos, Bates, Fisch, Lei, Schuster, 2022, *"Conformal Risk Control"*,
/// frankenscipy-7tb8d.3).
///
/// # Loss and Escalation Parameter
/// For problem $i$ with reciprocal condition estimate $\text{rcond}_i \in [0, 1]$ and
/// conservativeness parameter $t \in [0, 1]$ (the `rcond` threshold at which `DirectLU`
/// escalates to `PivotedQR`), the loss is the indicator of a tolerance exceedance:
///
/// $$L_i(t) = \mathbf{1}\{\omega_i(\text{action}_i(t)) > \text{tol}\} \in [0, 1],$$
///
/// which is bounded by $B = 1$, right-continuous, and monotone non-increasing in $t$.
///
/// # Calibration Rule
/// Given $n$ calibration problems with empirical risk
/// $\hat{R}_n(t) = \frac{1}{n}\sum_{i=1}^n L_i(t)$, CRC selects:
///
/// $$\hat{t} = \inf\left\{ t \in [0, 1] : \frac{n}{n + 1}\hat{R}_n(t) + \frac{B}{n + 1} \le \alpha \right\} \quad (B = 1).$$
///
/// # Finite-Sample Guarantee and Exchangeability Assumption
/// **Guarantee (Angelopoulos et al., 2022, Theorem 1):** Suppose the $n$ calibration loss
/// functions $(L_1, \dots, L_n)$ and a future test problem $L_{n+1}$ are **exchangeable**, each
/// $L_i(t) \in [0, 1]$ is monotone non-increasing and right-continuous in $t$, and
/// $L_i(1) = 0$. Then the expected loss on the new problem satisfies the finite-sample bound:
///
/// $$\mathbb{E}\left[L_{n+1}(\hat{t})\right] \le \alpha.$$
///
/// Online non-exchangeable drift is monitored separately by [`PolicyController`]'s anytime-valid
/// e-process and [`ConformalCalibrator`] (frankenscipy-7tb8d.4).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConformalRiskControl {
    alpha: f64,
    t_hat: f64,
    n_calibration: usize,
    empirical_risk_at_t_hat: f64,
    seed: u64,
    exceedance_rconds: Vec<f64>,
    unrecoverable_count: usize,
}

impl ConformalRiskControl {
    /// Construct the default CRC calibration generated from the W4.2 corpus
    /// (`crates/fsci-runtime/src/calibrated_losses.rs`, frankenscipy-7tb8d.3).
    #[must_use]
    pub fn default_calibrated() -> Self {
        let exceedance_rconds = calibrated_losses::SOLVER_CRC_EXCEEDANCE_RCONDS.to_vec();
        let n_calibration = calibrated_losses::SOLVER_CRC_N;
        let alpha = calibrated_losses::SOLVER_CRC_ALPHA;
        let t_hat = calibrated_losses::SOLVER_CRC_T_HAT;
        let unrecovered = exceedance_rconds.iter().filter(|&&r| r > t_hat).count();
        let empirical_risk_at_t_hat = if n_calibration > 0 {
            unrecovered as f64 / n_calibration as f64
        } else {
            0.0
        };
        Self {
            alpha,
            t_hat,
            n_calibration,
            empirical_risk_at_t_hat,
            seed: calibrated_losses::SOLVER_CRC_SEED,
            exceedance_rconds,
            unrecoverable_count: 0,
        }
    }

    /// Calibrate $\hat{t} = \inf\{ t \in [0, 1] : \frac{n}{n+1}\hat{R}_n(t) + \frac{1}{n+1} \le \alpha \}$
    /// over `samples`.
    #[must_use]
    pub fn calibrate(samples: &[CrcCalibrationSample], alpha: f64, seed: u64) -> Self {
        let alpha = alpha.clamp(0.001, 0.5);
        let n_calibration = samples.len();
        let mut exceedance_rconds = Vec::new();
        let mut unrecoverable_count = 0usize;
        for sample in samples {
            if sample.escalated_exceeded {
                unrecoverable_count += 1;
            } else if sample.unescalated_exceeded {
                exceedance_rconds.push(sample.rcond.clamp(0.0, 1.0));
            }
        }
        exceedance_rconds.sort_by(f64::total_cmp);
        let (t_hat, empirical_risk_at_t_hat) = Self::solve_t_hat(
            n_calibration,
            &exceedance_rconds,
            unrecoverable_count,
            alpha,
        );
        Self {
            alpha,
            t_hat,
            n_calibration,
            empirical_risk_at_t_hat,
            seed,
            exceedance_rconds,
            unrecoverable_count,
        }
    }

    fn solve_t_hat(
        n_calibration: usize,
        exceedance_rconds: &[f64],
        unrecoverable_count: usize,
        alpha: f64,
    ) -> (f64, f64) {
        if n_calibration == 0 {
            return (1.0, 0.0);
        }
        let mut candidates = Vec::with_capacity(exceedance_rconds.len() + 2);
        candidates.push(0.0);
        for &r in exceedance_rconds {
            if candidates.last().copied() != Some(r) {
                candidates.push(r);
            }
        }
        if candidates.last().copied() != Some(1.0) {
            candidates.push(1.0);
        }
        let denom = n_calibration as f64 + 1.0;
        for t in candidates {
            let unrecovered =
                unrecoverable_count + exceedance_rconds.iter().filter(|&&r| r > t).count();
            let crc_bound = (unrecovered as f64 + 1.0) / denom;
            if crc_bound <= alpha + 1e-15 {
                let r_hat = unrecovered as f64 / n_calibration as f64;
                return (t, r_hat);
            }
        }
        let unrecovered = unrecoverable_count;
        (1.0, unrecovered as f64 / n_calibration as f64)
    }

    /// Verify that every sample's loss function $L_i(t)$ is monotone non-increasing across
    /// all candidate thresholds $t \in [0, 1]$. Returns `Err(index)` naming the first
    /// non-monotone sample if violated.
    pub fn verify_monotonicity(samples: &[CrcCalibrationSample]) -> Result<(), usize> {
        let mut grid = Vec::with_capacity(samples.len() + 2);
        grid.push(0.0);
        for s in samples {
            if s.rcond.is_finite() {
                grid.push(s.rcond.clamp(0.0, 1.0));
            }
        }
        grid.push(1.0);
        grid.sort_by(f64::total_cmp);
        grid.dedup_by(|a, b| a.to_bits() == b.to_bits());

        for (idx, sample) in samples.iter().enumerate() {
            if !sample.is_monotone() {
                return Err(idx);
            }
            for pair in grid.windows(2) {
                if sample.loss_at(pair[1]) > sample.loss_at(pair[0]) {
                    return Err(idx);
                }
            }
        }
        Ok(())
    }

    /// Compute empirical risk $\hat{R}_n(t) = \frac{1}{n}\sum_{i=1}^n L_i(t)$.
    #[must_use]
    pub fn empirical_risk(samples: &[CrcCalibrationSample], t: f64) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum: f64 = samples.iter().map(|s| s.loss_at(t)).sum();
        sum / samples.len() as f64
    }

    /// Compute the CRC finite-sample objective $\frac{n}{n+1}\hat{R}_n(t) + \frac{1}{n+1}$.
    #[must_use]
    pub fn crc_objective(samples: &[CrcCalibrationSample], t: f64) -> f64 {
        let n = samples.len() as f64;
        let sum: f64 = samples.iter().map(|s| s.loss_at(t)).sum();
        (sum + 1.0) / (n + 1.0)
    }

    /// Update the target risk level $\alpha \in [0.001, 0.5]$ and recompute $\hat{t}$ over the
    /// stored calibration distribution.
    pub fn set_alpha(&mut self, alpha: f64) {
        self.alpha = alpha.clamp(0.001, 0.5);
        let (t_hat, r_hat) = Self::solve_t_hat(
            self.n_calibration,
            &self.exceedance_rconds,
            self.unrecoverable_count,
            self.alpha,
        );
        self.t_hat = t_hat;
        self.empirical_risk_at_t_hat = r_hat;
    }

    /// Override $\hat{t}$ directly (for testing or custom threshold injection).
    pub fn set_t_hat(&mut self, t_hat: f64) {
        self.t_hat = t_hat.clamp(0.0, 1.0);
        if self.n_calibration > 0 {
            let unrecovered = self.unrecoverable_count
                + self
                    .exceedance_rconds
                    .iter()
                    .filter(|&&r| r > self.t_hat)
                    .count();
            self.empirical_risk_at_t_hat = unrecovered as f64 / self.n_calibration as f64;
        }
    }

    /// Whether a problem with reciprocal condition estimate `rcond` should escalate from
    /// `DirectLU` to `PivotedQR` under $\hat{t}$.
    #[must_use]
    pub fn should_escalate(&self, rcond: f64) -> bool {
        self.t_hat > 0.0 && (!rcond.is_finite() || rcond <= self.t_hat)
    }

    #[must_use]
    pub const fn alpha(&self) -> f64 {
        self.alpha
    }

    #[must_use]
    pub const fn t_hat(&self) -> f64 {
        self.t_hat
    }

    #[must_use]
    pub const fn n_calibration(&self) -> usize {
        self.n_calibration
    }

    #[must_use]
    pub const fn empirical_risk_at_t_hat(&self) -> f64 {
        self.empirical_risk_at_t_hat
    }

    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    fn fingerprint_into(&self, fingerprinter: &mut Fingerprinter) {
        fingerprinter
            .f64(self.alpha)
            .f64(self.t_hat)
            .usize(self.n_calibration)
            .u64(self.seed);
    }
}

/// Exact one-sided $(1 - \delta)$ Clopper-Pearson upper confidence bound on a binomial
/// exceedance probability given $k$ exceedances in $n$ trials (frankenscipy-7tb8d.3).
///
/// Solves $\sum_{j=0}^{k} \binom{n}{j} p^j (1 - p)^{n - j} = \delta$ by 64-step bisection on
/// $p \in [0, 1]$ (for $k = 0$, returns $1 - \delta^{1/n}$; for $k \ge n$, returns $1.0$).
#[must_use]
pub fn clopper_pearson_upper_bound(k: usize, n: usize, delta: f64) -> f64 {
    if n == 0 || k >= n {
        return 1.0;
    }
    let delta = delta.clamp(1e-15, 1.0 - 1e-15);
    if k == 0 {
        return 1.0 - delta.powf(1.0 / n as f64);
    }
    let binom_cdf = |p: f64| -> f64 {
        if p <= 0.0 {
            return 1.0;
        }
        if p >= 1.0 {
            return 0.0;
        }
        let q = 1.0 - p;
        let ratio = p / q;
        let mut term = q.powi(n as i32);
        let mut sum = term;
        for j in 1..=k {
            term *= (n - j + 1) as f64 / j as f64 * ratio;
            sum += term;
        }
        sum
    };
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    for _ in 0..64 {
        let mid = 0.5 * (lo + hi);
        if binom_cdf(mid) > delta {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Timestamp utility for CASP evidence entries.
#[must_use]
pub fn casp_now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

// ═══════════════════════════════════════════════════════════════════
// CASP Sparse — Iterative & Direct Sparse Solver Portfolio
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SparseConditionState {
    SymmetricPositiveDefinite,
    GeneralWellConditioned,
    Indefinite,
    IllConditioned,
}

impl SparseConditionState {
    pub const ALL: [Self; 4] = [
        Self::SymmetricPositiveDefinite,
        Self::GeneralWellConditioned,
        Self::Indefinite,
        Self::IllConditioned,
    ];

    #[must_use]
    pub fn index(self) -> usize {
        match self {
            Self::SymmetricPositiveDefinite => 0,
            Self::GeneralWellConditioned => 1,
            Self::Indefinite => 2,
            Self::IllConditioned => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SparseSolverAction {
    ConjugateGradient,
    MinRes,
    BiCGSTAB,
    GMRES,
    QMR,
    SuperLU,
}

impl SparseSolverAction {
    pub const ALL: [Self; 6] = [
        Self::ConjugateGradient,
        Self::MinRes,
        Self::BiCGSTAB,
        Self::GMRES,
        Self::QMR,
        Self::SuperLU,
    ];

    #[must_use]
    pub fn index(self) -> usize {
        match self {
            Self::ConjugateGradient => 0,
            Self::MinRes => 1,
            Self::BiCGSTAB => 2,
            Self::GMRES => 3,
            Self::QMR => 4,
            Self::SuperLU => 5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparseStructuralEvidence {
    pub is_symmetric: bool,
    pub is_positive_definite_hint: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparseSolverEvidenceEntry {
    pub component: &'static str,
    pub matrix_shape: (usize, usize),
    pub nnz: usize,
    pub cond_estimate: f64,
    pub chosen_action: SparseSolverAction,
    pub posterior: Vec<f64>,
    pub expected_losses: Vec<f64>,
    pub chosen_expected_loss: f64,
    pub fallback_active: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relative_residual: Option<f64>,
}

fn sparse_cond_decade(cond_estimate: f64) -> usize {
    if !cond_estimate.is_finite() || cond_estimate <= 0.0 {
        return RCOND_DECADES - 1;
    }
    let decade = deterministic_log10(cond_estimate.max(1.0)).floor();
    if decade <= 0.0 {
        0
    } else {
        (decade as usize).min(RCOND_DECADES - 1)
    }
}

/// P(AttemptOutcome::Ok | SparseSolverAction, SparseConditionState) used by
/// [`SparseSolverPortfolio::record_outcome`] to attribute outcomes across the 4 latent
/// condition states (SPD, GeneralWellConditioned, Indefinite, IllConditioned).
const SPARSE_OK_PROBABILITY: [[f64; 4]; 6] = [
    [0.98, 0.05, 0.05, 0.02], // ConjugateGradient
    [0.92, 0.10, 0.94, 0.05], // MinRes
    [0.90, 0.96, 0.55, 0.08], // BiCGSTAB
    [0.94, 0.94, 0.88, 0.45], // GMRES
    [0.92, 0.92, 0.86, 0.25], // QMR
    [0.99, 0.99, 0.99, 0.98], // SuperLU
];

/// Sparse solver selection portfolio engine.
///
/// Loss matrix: [`Self::default_loss_matrix`], 6 actions × 4 states (SPD, general well
/// conditioned, indefinite, ill conditioned). Posterior combines a continuous feature prior
/// (`log10(cond_estimate)` and [`SparseStructuralEvidence`]) with per-decade Dirichlet
/// outcome counts and per-action failure counts (`frankenscipy-7tb8d.7`).
#[derive(Debug, Clone)]
pub struct SparseSolverPortfolio {
    mode: RuntimeMode,
    loss_matrix: [[f64; 4]; 6],
    evidence: VecDeque<SparseSolverEvidenceEntry>,
    evidence_capacity: usize,
    calibrator: ConformalCalibrator,
    outcome_counts: [[f64; 4]; RCOND_DECADES],
    failure_counts: [[[f64; 2]; 6]; RCOND_DECADES],
}

impl SparseSolverPortfolio {
    #[must_use]
    pub fn new(mode: RuntimeMode, evidence_capacity: usize) -> Self {
        let evidence_capacity = evidence_capacity.max(1);
        Self {
            mode,
            loss_matrix: Self::default_loss_matrix(),
            evidence: VecDeque::with_capacity(evidence_capacity),
            evidence_capacity,
            calibrator: ConformalCalibrator::new(0.05, 200),
            outcome_counts: [[0.0; 4]; RCOND_DECADES],
            failure_counts: [[[0.0; 2]; 6]; RCOND_DECADES],
        }
    }

    #[must_use]
    pub const fn default_loss_matrix() -> [[f64; 4]; 6] {
        [
            // SPD,   GenWell, Indef,  IllCond
            [1.0, 40.0, 50.0, 100.0], // ConjugateGradient (symmetric PD)
            [3.0, 30.0, 1.0, 60.0],   // MinRes (symmetric indefinite)
            [5.0, 1.0, 12.0, 60.0],   // BiCGSTAB (non-symmetric well-conditioned)
            [8.0, 6.0, 7.0, 10.0],    // GMRES (non-symmetric harder / pre-direct band)
            [10.0, 4.5, 3.0, 32.0],   // QMR (non-symmetric moderate / indefinite band)
            [25.0, 20.0, 18.0, 2.0],  // SuperLU (ill-conditioned direct LU)
        ]
    }

    /// Dirichlet posterior over the four sparse condition states at `cond_estimate`.
    #[must_use]
    pub fn posterior(
        &self,
        cond_estimate: f64,
        structure: Option<SparseStructuralEvidence>,
    ) -> [f64; 4] {
        let counts = &self.outcome_counts[sparse_cond_decade(cond_estimate)];
        let observed: f64 = counts.iter().sum();
        if observed == 0.0 {
            return Self::condition_posterior(cond_estimate, structure);
        }
        let prior = Self::condition_posterior(cond_estimate, structure);
        let total = PRIOR_MASS + observed;
        let mut posterior = [0.0; 4];
        for (state, p) in posterior.iter_mut().enumerate() {
            *p = (PRIOR_MASS * prior[state] + counts[state]) / total;
        }
        posterior
    }

    fn smoothed_posterior(
        &self,
        cond_estimate: f64,
        structure: Option<SparseStructuralEvidence>,
    ) -> [f64; 4] {
        let prior = Self::condition_posterior(cond_estimate, structure);
        let counts = &self.outcome_counts[sparse_cond_decade(cond_estimate)];
        let observed: f64 = counts.iter().sum();
        let total = PRIOR_MASS + 4.0 * STATE_SMOOTHING + observed;
        let mut posterior = [0.0; 4];
        for (state, p) in posterior.iter_mut().enumerate() {
            *p = (PRIOR_MASS * prior[state] + STATE_SMOOTHING + counts[state]) / total;
        }
        posterior
    }

    /// Feed back how an attempt with `action` at `cond_estimate` turned out (`frankenscipy-7tb8d.7`).
    pub fn record_outcome(
        &mut self,
        cond_estimate: f64,
        structure: Option<SparseStructuralEvidence>,
        action: SparseSolverAction,
        outcome: AttemptOutcome,
    ) {
        let decade = sparse_cond_decade(cond_estimate);
        let tally = &mut self.failure_counts[decade][action.index()];
        tally[1] += 1.0;
        if outcome != AttemptOutcome::Ok {
            tally[0] += 1.0;
        }
        let prior = self.smoothed_posterior(cond_estimate, structure);
        let ok = SPARSE_OK_PROBABILITY[action.index()];
        let mut responsibility = [0.0; 4];
        for state in 0..4 {
            let likelihood = match outcome {
                AttemptOutcome::Ok => ok[state],
                AttemptOutcome::Inaccurate => (1.0 - ok[state]) * INACCURATE_SHARE,
                AttemptOutcome::Failed => (1.0 - ok[state]) * (1.0 - INACCURATE_SHARE),
            };
            responsibility[state] = prior[state] * likelihood;
        }
        let norm: f64 = responsibility.iter().sum();
        if !norm.is_finite() || norm <= 0.0 {
            return;
        }
        let row = &mut self.outcome_counts[decade];
        for (count, r) in row.iter_mut().zip(responsibility) {
            *count += r / norm;
        }
    }

    /// Select optimal sparse solver via expected-loss minimization over structurally applicable
    /// actions (`ConjugateGradient` requires symmetry without a negative-definite hint;
    /// `MinRes` requires symmetry).
    pub fn select_action(
        &self,
        cond_estimate: f64,
        structure: Option<SparseStructuralEvidence>,
    ) -> (SparseSolverAction, [f64; 4], [f64; 6], f64) {
        let posterior = self.posterior(cond_estimate, structure);
        let losses = self.compute_expected_losses(posterior, cond_estimate);

        // If conformal calibrator triggers drift, fallback to SuperLU direct solver
        if self.calibrator.should_fallback() {
            return (
                SparseSolverAction::SuperLU,
                posterior,
                losses,
                losses[SparseSolverAction::SuperLU.index()],
            );
        }

        let is_sym = structure.is_some_and(|s| s.is_symmetric);
        let pd_hint = structure.and_then(|s| s.is_positive_definite_hint);

        let mut best_idx = SparseSolverAction::SuperLU.index();
        let mut best_loss = f64::INFINITY;

        for (idx, &loss) in losses.iter().enumerate() {
            let action = SparseSolverAction::ALL[idx];
            let applicable = match action {
                SparseSolverAction::ConjugateGradient => is_sym && pd_hint != Some(false),
                SparseSolverAction::MinRes => is_sym,
                SparseSolverAction::BiCGSTAB
                | SparseSolverAction::GMRES
                | SparseSolverAction::QMR
                | SparseSolverAction::SuperLU => true,
            };
            if !applicable {
                continue;
            }
            if loss < best_loss {
                best_loss = loss;
                best_idx = idx;
            } else if (loss - best_loss).abs() <= 1e-12 && idx > best_idx {
                best_idx = idx;
            }
        }

        (
            SparseSolverAction::ALL[best_idx],
            posterior,
            losses,
            best_loss,
        )
    }

    pub fn record_evidence(&mut self, entry: SparseSolverEvidenceEntry) {
        if let Some(res) = entry.relative_residual {
            self.calibrator.observe(res);
        }
        if self.evidence.len() >= self.evidence_capacity {
            let _ = self.evidence.pop_front();
        }
        self.evidence.push_back(entry);
    }

    pub fn observe_relative_residual(&mut self, residual: f64) {
        self.calibrator.observe(residual);
    }

    #[must_use]
    pub fn evidence_len(&self) -> usize {
        self.evidence.len()
    }

    #[must_use]
    pub const fn mode(&self) -> RuntimeMode {
        self.mode
    }

    #[must_use]
    pub fn calibrator(&self) -> &ConformalCalibrator {
        &self.calibrator
    }

    fn compute_expected_losses(&self, posterior: [f64; 4], cond_estimate: f64) -> [f64; 6] {
        let mut losses = [0.0; 6];
        let tallies = &self.failure_counts[sparse_cond_decade(cond_estimate)];
        for (action_idx, row) in self.loss_matrix.iter().enumerate() {
            let base: f64 = row.iter().zip(posterior.iter()).map(|(l, p)| l * p).sum();
            let [failures, trials] = tallies[action_idx];
            losses[action_idx] = if trials > 0.0 {
                let fail_rate = failures / trials;
                base + fail_rate * row[SparseConditionState::IllConditioned.index()]
            } else {
                base
            };
        }
        losses
    }

    /// Continuous prior over `(SymmetricPositiveDefinite, GeneralWellConditioned, Indefinite,
    /// IllConditioned)` driven by `deterministic_log10(cond_estimate)` and structural evidence
    /// (`frankenscipy-7tb8d.7`).
    fn condition_posterior(
        cond_estimate: f64,
        structure: Option<SparseStructuralEvidence>,
    ) -> [f64; 4] {
        if !cond_estimate.is_finite() || cond_estimate <= 0.0 || cond_estimate >= 1e8 {
            return [0.0, 0.0, 0.0, 1.0];
        }

        let is_sym = structure.is_some_and(|s| s.is_symmetric);
        let pd_hint = structure.and_then(|s| s.is_positive_definite_hint);

        let log_k = deterministic_log10(cond_estimate.max(1.0)).clamp(0.0, 8.0);
        let w_ill = if log_k <= 4.0 {
            0.005 + 0.015 * (log_k / 4.0)
        } else {
            let u = (log_k - 4.0) / 4.0;
            0.02 + 0.98 * (u * u)
        };
        let w_well = 1.0 - w_ill;

        if is_sym {
            match pd_hint {
                Some(true) => [w_well, 0.0, 0.0, w_ill],
                Some(false) => [0.0, 0.0, w_well, w_ill],
                None => {
                    let indef_share = 0.25 + 0.25 * (log_k / 8.0);
                    [
                        (1.0 - indef_share) * w_well,
                        0.0,
                        indef_share * w_well,
                        w_ill,
                    ]
                }
            }
        } else {
            let indef_share = if log_k <= 3.0 {
                0.02 * (log_k / 3.0)
            } else {
                let u = ((log_k - 3.0) / 2.0).min(1.0);
                0.02 + 0.83 * u
            };
            [
                0.0,
                (1.0 - indef_share) * w_well,
                indef_share * w_well,
                w_ill,
            ]
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// CASP Opt — Continuous & Global Optimization Solver Portfolio
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LandscapeConditionState {
    SmoothConvex,
    IllConditionedValley,
    NonConvexMultiModal,
    NoisyNonSmooth,
}

impl LandscapeConditionState {
    pub const ALL: [Self; 4] = [
        Self::SmoothConvex,
        Self::IllConditionedValley,
        Self::NonConvexMultiModal,
        Self::NoisyNonSmooth,
    ];

    #[must_use]
    pub fn index(self) -> usize {
        match self {
            Self::SmoothConvex => 0,
            Self::IllConditionedValley => 1,
            Self::NonConvexMultiModal => 2,
            Self::NoisyNonSmooth => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptSolverAction {
    BFGS,
    LBFGSB,
    NelderMead,
    DIRECT,
    TrustRegionNewtonCG,
}

impl OptSolverAction {
    pub const ALL: [Self; 5] = [
        Self::BFGS,
        Self::LBFGSB,
        Self::NelderMead,
        Self::DIRECT,
        Self::TrustRegionNewtonCG,
    ];

    #[must_use]
    pub fn index(self) -> usize {
        match self {
            Self::BFGS => 0,
            Self::LBFGSB => 1,
            Self::NelderMead => 2,
            Self::DIRECT => 3,
            Self::TrustRegionNewtonCG => 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptSolverEvidenceEntry {
    pub component: &'static str,
    pub dimension: usize,
    pub condition_number_estimate: f64,
    pub chosen_action: OptSolverAction,
    pub posterior: Vec<f64>,
    pub expected_losses: Vec<f64>,
    pub chosen_expected_loss: f64,
    pub fallback_active: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gradient_norm: Option<f64>,
}

/// Optimization solver selection portfolio engine.
///
/// Loss matrix: [`Self::default_loss_matrix`], 5 actions × 4 states (smooth convex, valley,
/// multimodal, noisy non-smooth). It is hand-set, not yet measured: the dense
/// [`SolverPortfolio`]'s is calibrated (frankenscipy-7tb8d.2), and this one is to reuse that
/// harness once its features are real. The table that stood here disagreed with the constant.
#[derive(Debug, Clone)]
pub struct OptSolverPortfolio {
    mode: RuntimeMode,
    loss_matrix: [[f64; 4]; 5],
    evidence: VecDeque<OptSolverEvidenceEntry>,
    evidence_capacity: usize,
    calibrator: ConformalCalibrator,
}

impl OptSolverPortfolio {
    #[must_use]
    pub fn new(mode: RuntimeMode, evidence_capacity: usize) -> Self {
        let evidence_capacity = evidence_capacity.max(1);
        let mut calibrator = ConformalCalibrator::new(0.05, 200);
        // Default optimizer tolerances (BFGS/L-BFGS-B gtol = 1e-5, Nelder-Mead xatol = 1e-4)
        // terminate with gradient norms up to ~1e-3; scores above 1e-2 indicate non-stationarity.
        calibrator.set_violation_threshold(1e-2);
        Self {
            mode,
            loss_matrix: Self::default_loss_matrix(),
            evidence: VecDeque::with_capacity(evidence_capacity),
            evidence_capacity,
            calibrator,
        }
    }

    #[must_use]
    pub const fn default_loss_matrix() -> [[f64; 4]; 5] {
        [
            // SmoothConvex, Valley,  MultiModal, NoisyNonSmooth
            [1.0, 15.0, 120.0, 150.0], // BFGS
            [2.0, 20.0, 120.0, 150.0], // LBFGSB
            [30.0, 30.0, 40.0, 1.0],   // NelderMead
            [80.0, 40.0, 2.0, 20.0],   // DIRECT
            [8.0, 2.0, 70.0, 180.0],   // TrustRegionNewtonCG
        ]
    }

    /// Select among all five actions (an unconstrained problem).
    pub fn select_action(
        &self,
        hessian_cond_estimate: f64,
        is_noisy_or_discontinuous: bool,
        is_multimodal: bool,
    ) -> (OptSolverAction, [f64; 4], [f64; 5], f64) {
        self.select_action_among(
            hessian_cond_estimate,
            is_noisy_or_discontinuous,
            is_multimodal,
            &OptSolverAction::ALL,
        )
    }

    /// Select the minimum-expected-loss action among `allowed` only (an empty slice
    /// means all actions).
    ///
    /// br-szq1n.9: feasibility is a hard requirement, not a loss term. A bounded
    /// problem can only be answered by a method that honours bounds (L-BFGS-B,
    /// Nelder-Mead, and DIRECT on a finite box); scoring BFGS lower than L-BFGS-B must not be able to
    /// route it to a method that ignores the bounds and returns an infeasible point.
    /// Filtering the candidate set before the argmin is also what makes L-BFGS-B
    /// reachable: its loss row is dominated by BFGS whenever both are candidates.
    pub fn select_action_among(
        &self,
        hessian_cond_estimate: f64,
        is_noisy_or_discontinuous: bool,
        is_multimodal: bool,
        allowed: &[OptSolverAction],
    ) -> (OptSolverAction, [f64; 4], [f64; 5], f64) {
        let allowed: &[OptSolverAction] = if allowed.is_empty() {
            &OptSolverAction::ALL
        } else {
            allowed
        };
        let posterior = Self::landscape_posterior(
            hessian_cond_estimate,
            is_noisy_or_discontinuous,
            is_multimodal,
        );
        let losses = self.compute_expected_losses(posterior);

        // Minimum expected loss over the allowed actions, in ALL order; an exact
        // tie goes to the later action.
        let mut best: Option<(OptSolverAction, f64)> = None;
        for action in OptSolverAction::ALL {
            if !allowed.contains(&action) {
                continue;
            }
            let loss = losses[action.index()];
            best = match best {
                None => Some((action, loss)),
                Some((_, best_loss)) if loss < best_loss => Some((action, loss)),
                Some((_, best_loss)) if (loss - best_loss).abs() <= 1e-12 => Some((action, loss)),
                keep => keep,
            };
        }
        let (argmin, argmin_loss) = best.expect("allowed is non-empty");

        // Calibrator drift: prefer the conservative method for the landscape, but
        // never one outside the feasible set.
        if self.calibrator.should_fallback() {
            let preferred = if is_multimodal {
                OptSolverAction::DIRECT
            } else {
                OptSolverAction::NelderMead
            };
            let fallback_action = if allowed.contains(&preferred) {
                preferred
            } else {
                argmin
            };
            return (
                fallback_action,
                posterior,
                losses,
                losses[fallback_action.index()],
            );
        }

        (argmin, posterior, losses, argmin_loss)
    }

    pub fn record_evidence(&mut self, entry: OptSolverEvidenceEntry) {
        if let Some(gnorm) = entry.gradient_norm {
            self.calibrator.observe(gnorm);
        }
        if self.evidence.len() >= self.evidence_capacity {
            let _ = self.evidence.pop_front();
        }
        self.evidence.push_back(entry);
    }

    pub fn observe_step_failure(&mut self, failure_score: f64) {
        self.calibrator.observe(failure_score);
    }

    #[must_use]
    pub fn evidence_len(&self) -> usize {
        self.evidence.len()
    }

    #[must_use]
    pub const fn mode(&self) -> RuntimeMode {
        self.mode
    }

    #[must_use]
    pub fn calibrator(&self) -> &ConformalCalibrator {
        &self.calibrator
    }

    fn compute_expected_losses(&self, posterior: [f64; 4]) -> [f64; 5] {
        let mut losses = [0.0; 5];
        for (action_idx, row) in self.loss_matrix.iter().enumerate() {
            losses[action_idx] = row.iter().zip(posterior.iter()).map(|(l, p)| l * p).sum();
        }
        losses
    }

    fn landscape_posterior(
        hessian_cond_estimate: f64,
        is_noisy_or_discontinuous: bool,
        is_multimodal: bool,
    ) -> [f64; 4] {
        if is_noisy_or_discontinuous {
            return [0.05, 0.05, 0.1, 0.8]; // NoisyNonSmooth
        }
        if is_multimodal {
            return [0.05, 0.1, 0.8, 0.05]; // NonConvexMultiModal
        }
        if !hessian_cond_estimate.is_finite() || hessian_cond_estimate >= 1e6 {
            return [0.05, 0.85, 0.05, 0.05]; // IllConditionedValley
        }
        if hessian_cond_estimate < 1e3 {
            [0.85, 0.05, 0.05, 0.05] // SmoothConvex
        } else {
            [0.3, 0.6, 0.05, 0.05]
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// CASP Integrate — ODE & IVP Solver Portfolio with Stiffness Tracking
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StiffnessConditionState {
    NonStiffSmooth,
    MildlyStiff,
    Stiff,
    AlgebraicConstrained,
}

impl StiffnessConditionState {
    pub const ALL: [Self; 4] = [
        Self::NonStiffSmooth,
        Self::MildlyStiff,
        Self::Stiff,
        Self::AlgebraicConstrained,
    ];

    #[must_use]
    pub fn index(self) -> usize {
        match self {
            Self::NonStiffSmooth => 0,
            Self::MildlyStiff => 1,
            Self::Stiff => 2,
            Self::AlgebraicConstrained => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OdeSolverAction {
    RK45,
    RK23,
    DOP853,
    Radau,
    BDF,
}

impl OdeSolverAction {
    pub const ALL: [Self; 5] = [Self::RK45, Self::RK23, Self::DOP853, Self::Radau, Self::BDF];

    #[must_use]
    pub fn index(self) -> usize {
        match self {
            Self::RK45 => 0,
            Self::RK23 => 1,
            Self::DOP853 => 2,
            Self::Radau => 3,
            Self::BDF => 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OdeSolverEvidenceEntry {
    pub component: &'static str,
    pub system_dim: usize,
    pub stiffness_ratio_estimate: f64,
    pub chosen_action: OdeSolverAction,
    pub posterior: Vec<f64>,
    pub expected_losses: Vec<f64>,
    pub chosen_expected_loss: f64,
    pub fallback_active: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step_rejections: Option<usize>,
}

/// Online stiffness detection for adaptive step and solver control (`frankenscipy-7tb8d.9`).
#[derive(Debug, Clone)]
pub struct StiffnessDetector {
    stiffness_threshold: f64,
    consecutive_rejections: usize,
    rejection_threshold: usize,
    rejection_history: VecDeque<bool>,
    history_capacity: usize,
    hairer_wanner_streak: usize,
    last_stiffness_indicator: f64,
}

impl StiffnessDetector {
    #[must_use]
    pub fn new(stiffness_threshold: f64, rejection_threshold: usize) -> Self {
        Self {
            stiffness_threshold: stiffness_threshold.max(1.0),
            consecutive_rejections: 0,
            rejection_threshold: rejection_threshold.max(1),
            rejection_history: VecDeque::with_capacity(100),
            history_capacity: 100,
            hairer_wanner_streak: 0,
            last_stiffness_indicator: 0.0,
        }
    }

    /// Record a step outcome (true = accepted, false = rejected due to stability/tolerance).
    pub fn observe_step(&mut self, accepted: bool) {
        if accepted {
            self.consecutive_rejections = 0;
        } else {
            self.consecutive_rejections += 1;
        }
        if self.rejection_history.len() >= self.history_capacity {
            let _ = self.rejection_history.pop_front();
        }
        self.rejection_history.push_back(!accepted);
    }

    /// Record a per-step Hairer-Wanner embedded-pair stiffness test summary (`frankenscipy-7tb8d.9`).
    pub fn observe_hairer_wanner(&mut self, stiff_streak: usize, stiffness_indicator: f64) {
        self.hairer_wanner_streak = stiff_streak;
        if stiffness_indicator.is_finite() && stiffness_indicator > 0.0 {
            self.last_stiffness_indicator = self.last_stiffness_indicator.max(stiffness_indicator);
        }
        if stiff_streak >= self.rejection_threshold
            && stiffness_indicator >= self.stiffness_threshold
        {
            self.consecutive_rejections = self.consecutive_rejections.max(self.rejection_threshold);
        }
    }

    #[must_use]
    pub const fn hairer_wanner_streak(&self) -> usize {
        self.hairer_wanner_streak
    }

    #[must_use]
    pub const fn last_stiffness_indicator(&self) -> f64 {
        self.last_stiffness_indicator
    }

    /// Check if stiffness is detected based on consecutive rejections or ratio estimate.
    #[must_use]
    pub fn is_stiff(&self, stiffness_ratio_estimate: f64) -> bool {
        if self.consecutive_rejections >= self.rejection_threshold {
            return true;
        }
        stiffness_ratio_estimate.is_finite() && stiffness_ratio_estimate >= self.stiffness_threshold
    }

    #[must_use]
    pub fn rejection_rate(&self) -> f64 {
        if self.rejection_history.is_empty() {
            return 0.0;
        }
        let rejections = self.rejection_history.iter().filter(|&&r| r).count();
        rejections as f64 / self.rejection_history.len() as f64
    }
}

impl Default for StiffnessDetector {
    fn default() -> Self {
        Self::new(1e3, 3)
    }
}

/// ODE solver selection portfolio engine (`frankenscipy-7tb8d.9`).
///
/// Loss matrix: [`Self::default_loss_matrix`], 5 actions × 4 states (`NonStiffSmooth`,
/// `MildlyStiff`, `Stiff`, `AlgebraicConstrained`), where every action (`DOP853`, `RK45`,
/// `RK23`, `BDF`, `Radau`) is the unique Bayes-optimal choice in its regime.
#[derive(Debug, Clone)]
pub struct OdeSolverPortfolio {
    mode: RuntimeMode,
    loss_matrix: [[f64; 4]; 5],
    evidence: VecDeque<OdeSolverEvidenceEntry>,
    evidence_capacity: usize,
    calibrator: ConformalCalibrator,
    detector: StiffnessDetector,
}

impl OdeSolverPortfolio {
    #[must_use]
    pub fn new(mode: RuntimeMode, evidence_capacity: usize) -> Self {
        let evidence_capacity = evidence_capacity.max(1);
        let mut calibrator = ConformalCalibrator::new(0.05, 200);
        calibrator.set_violation_threshold(0.5);
        Self {
            mode,
            loss_matrix: Self::default_loss_matrix(),
            evidence: VecDeque::with_capacity(evidence_capacity),
            evidence_capacity,
            calibrator,
            detector: StiffnessDetector::default(),
        }
    }

    #[must_use]
    pub const fn default_loss_matrix() -> [[f64; 4]; 5] {
        [
            // NonStiff, MildlyStiff, Stiff,  Algebraic
            [1.0, 8.0, 200.0, 300.0],  // RK45 (5th-order explicit default)
            [4.0, 1.0, 200.0, 300.0],  // RK23 (3rd-order explicit for mildly stiff / coarse tol)
            [0.7, 35.0, 200.0, 300.0], // DOP853 (8th-order explicit for ultra-smooth / tight tol)
            [35.0, 12.0, 2.0, 1.0],    // Radau (L-stable implicit RK for algebraic/DAE & fallback)
            [25.0, 10.0, 1.0, 15.0],   // BDF (variable-order implicit multistep for stiff ODEs)
        ]
    }

    pub fn select_action(
        &self,
        stiffness_ratio_estimate: f64,
        is_algebraic_dae: bool,
    ) -> (OdeSolverAction, [f64; 4], [f64; 5], f64) {
        self.select_action_excluding(stiffness_ratio_estimate, is_algebraic_dae, 1e-3, &[])
    }

    /// Select optimal ODE solver accounting for caller relative tolerance `rtol`
    /// (`rtol <= 1e-7` favors 8th-order `DOP853` on non-stiff systems; `rtol >= 1e-2`
    /// favors 3rd-order `RK23` on non-stiff systems).
    pub fn select_action_with_tol(
        &self,
        stiffness_ratio_estimate: f64,
        is_algebraic_dae: bool,
        rtol: f64,
    ) -> (OdeSolverAction, [f64; 4], [f64; 5], f64) {
        self.select_action_excluding(stiffness_ratio_estimate, is_algebraic_dae, rtol, &[])
    }

    /// Select optimal ODE solver excluding any actions in `excluded` (used for bounded fallback
    /// retry when an initial attempt fails).
    pub fn select_action_excluding(
        &self,
        stiffness_ratio_estimate: f64,
        is_algebraic_dae: bool,
        rtol: f64,
        excluded: &[OdeSolverAction],
    ) -> (OdeSolverAction, [f64; 4], [f64; 5], f64) {
        let online_stiff = self.detector.is_stiff(stiffness_ratio_estimate);
        let effective_stiffness =
            if !is_algebraic_dae && !online_stiff && stiffness_ratio_estimate < 80.0 {
                if rtol.is_finite() && rtol > 0.0 && rtol <= 1e-7 {
                    stiffness_ratio_estimate.min(0.1)
                } else if rtol.is_finite() && rtol >= 1e-2 {
                    stiffness_ratio_estimate.max(120.0)
                } else {
                    stiffness_ratio_estimate
                }
            } else {
                stiffness_ratio_estimate
            };
        let posterior =
            Self::stiffness_posterior(effective_stiffness, is_algebraic_dae, online_stiff);
        let losses = self.compute_expected_losses(posterior);

        // If conformal calibrator triggers drift and Radau is not excluded, fallback to Radau
        if self.calibrator.should_fallback() && !excluded.contains(&OdeSolverAction::Radau) {
            return (
                OdeSolverAction::Radau,
                posterior,
                losses,
                losses[OdeSolverAction::Radau.index()],
            );
        }

        let mut best_idx = None;
        let mut best_loss = f64::INFINITY;

        for (idx, &loss) in losses.iter().enumerate() {
            let action = OdeSolverAction::ALL[idx];
            if excluded.contains(&action) {
                continue;
            }
            if loss < best_loss {
                best_loss = loss;
                best_idx = Some(idx);
            } else if (loss - best_loss).abs() <= 1e-12 && best_idx.is_none_or(|b| idx > b) {
                best_idx = Some(idx);
            }
        }

        let chosen_idx = best_idx.unwrap_or_else(|| OdeSolverAction::Radau.index());
        (
            OdeSolverAction::ALL[chosen_idx],
            posterior,
            losses,
            losses[chosen_idx],
        )
    }

    /// Record an evidence entry and feed the conformal calibrator once (`frankenscipy-7tb8d.9`).
    pub fn record_evidence(&mut self, entry: OdeSolverEvidenceEntry) {
        if let Some(rejections) = entry.step_rejections {
            self.calibrator.observe(rejections as f64);
        }
        self.push_evidence_only(entry);
    }

    /// Append an evidence entry (such as a failed first attempt before a fallback retry) without
    /// adding a duplicate observation to the conformal calibrator (`frankenscipy-7tb8d.9`).
    pub fn push_evidence_only(&mut self, entry: OdeSolverEvidenceEntry) {
        if self.evidence.len() >= self.evidence_capacity {
            let _ = self.evidence.pop_front();
        }
        self.evidence.push_back(entry);
    }

    /// Update the online stiffness detector with a step/attempt outcome without double-feeding
    /// the conformal calibrator (`frankenscipy-7tb8d.9`).
    pub fn observe_step(&mut self, accepted: bool) {
        self.detector.observe_step(accepted);
    }

    /// Update the online stiffness detector with a Hairer-Wanner per-step stiffness summary.
    pub fn observe_hairer_wanner(&mut self, stiff_streak: usize, stiffness_indicator: f64) {
        self.detector
            .observe_hairer_wanner(stiff_streak, stiffness_indicator);
    }

    #[must_use]
    pub fn evidence_len(&self) -> usize {
        self.evidence.len()
    }

    #[must_use]
    pub const fn mode(&self) -> RuntimeMode {
        self.mode
    }

    #[must_use]
    pub fn calibrator(&self) -> &ConformalCalibrator {
        &self.calibrator
    }

    #[must_use]
    pub fn detector(&self) -> &StiffnessDetector {
        &self.detector
    }

    fn compute_expected_losses(&self, posterior: [f64; 4]) -> [f64; 5] {
        let mut losses = [0.0; 5];
        for (action_idx, row) in self.loss_matrix.iter().enumerate() {
            losses[action_idx] = row.iter().zip(posterior.iter()).map(|(l, p)| l * p).sum();
        }
        losses
    }

    fn stiffness_posterior(
        stiffness_ratio_estimate: f64,
        is_algebraic_dae: bool,
        online_stiff: bool,
    ) -> [f64; 4] {
        if is_algebraic_dae {
            return [0.0, 0.05, 0.1, 0.85]; // AlgebraicConstrained -> Radau
        }
        if online_stiff || stiffness_ratio_estimate >= 1e3 {
            return [0.0, 0.1, 0.85, 0.05]; // Stiff -> BDF
        }
        if stiffness_ratio_estimate >= 80.0 {
            return [0.08, 0.90, 0.01, 0.01]; // MildlyStiff -> RK23
        }
        if stiffness_ratio_estimate <= 0.0 || !stiffness_ratio_estimate.is_finite() {
            return [0.0, 0.1, 0.8, 0.1];
        }
        if stiffness_ratio_estimate < 0.2 {
            return [0.98, 0.01, 0.005, 0.005]; // Ultra-smooth -> DOP853
        }
        [0.85, 0.1, 0.03, 0.02] // NonStiffSmooth -> RK45
    }
}

// ═══════════════════════════════════════════════════════════════════
// CASP Special — Hypergeometric Branch Selection Portfolio
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HyperConditionState {
    NearZeroDirectSeries,
    TransformDomain,
    LargeArgumentAsymptotic,
    BoundaryNearPole,
}

impl HyperConditionState {
    pub const ALL: [Self; 4] = [
        Self::NearZeroDirectSeries,
        Self::TransformDomain,
        Self::LargeArgumentAsymptotic,
        Self::BoundaryNearPole,
    ];

    #[must_use]
    pub fn index(self) -> usize {
        match self {
            Self::NearZeroDirectSeries => 0,
            Self::TransformDomain => 1,
            Self::LargeArgumentAsymptotic => 2,
            Self::BoundaryNearPole => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HyperBranchAction {
    DirectSeries,
    KummerTransform,
    PfaffTransform,
    Asymptotic,
    ContinuedFraction,
    GuardedFallback,
}

impl HyperBranchAction {
    pub const ALL: [Self; 6] = [
        Self::DirectSeries,
        Self::KummerTransform,
        Self::PfaffTransform,
        Self::Asymptotic,
        Self::ContinuedFraction,
        Self::GuardedFallback,
    ];

    #[must_use]
    pub fn index(self) -> usize {
        match self {
            Self::DirectSeries => 0,
            Self::KummerTransform => 1,
            Self::PfaffTransform => 2,
            Self::Asymptotic => 3,
            Self::ContinuedFraction => 4,
            Self::GuardedFallback => 5,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HyperSolverEvidenceEntry {
    pub component: &'static str,
    pub function_kind: &'static str,
    pub z_abs: f64,
    pub parameter_stability_margin: f64,
    pub chosen_action: HyperBranchAction,
    pub posterior: Vec<f64>,
    pub expected_losses: Vec<f64>,
    pub chosen_expected_loss: f64,
    pub fallback_active: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub term_count: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimated_truncation_error: Option<f64>,
}

/// Hypergeometric branch selection portfolio engine.
///
/// Loss matrix: [`Self::default_loss_matrix`], 6 actions × 4 states (near zero, transform,
/// asymptotic, boundary near a pole). It is hand-set, not yet measured: the dense
/// [`SolverPortfolio`]'s is calibrated (frankenscipy-7tb8d.2), and this one is to reuse that
/// harness once its features are real. The table that stood here disagreed with the constant.
#[derive(Debug, Clone)]
pub struct HyperSolverPortfolio {
    mode: RuntimeMode,
    loss_matrix: [[f64; 4]; 6],
    evidence: VecDeque<HyperSolverEvidenceEntry>,
    evidence_capacity: usize,
    calibrator: ConformalCalibrator,
}

impl HyperSolverPortfolio {
    #[must_use]
    pub fn new(mode: RuntimeMode, evidence_capacity: usize) -> Self {
        let evidence_capacity = evidence_capacity.max(1);
        Self {
            mode,
            loss_matrix: Self::default_loss_matrix(),
            evidence: VecDeque::with_capacity(evidence_capacity),
            evidence_capacity,
            calibrator: ConformalCalibrator::new(0.05, 200),
        }
    }

    #[must_use]
    pub const fn default_loss_matrix() -> [[f64; 4]; 6] {
        [
            // NearZero, Transform, Asymptotic, BoundaryNearPole
            [1.0, 25.0, 80.0, 100.0], // DirectSeries
            [15.0, 1.0, 60.0, 80.0],  // KummerTransform
            [20.0, 2.0, 60.0, 80.0],  // PfaffTransform
            [50.0, 30.0, 1.0, 60.0],  // Asymptotic
            [25.0, 15.0, 15.0, 10.0], // ContinuedFraction
            [35.0, 20.0, 20.0, 1.0],  // GuardedFallback
        ]
    }

    pub fn select_action(
        &self,
        z: f64,
        stability_margin: f64,
        is_negative_real_confluent: bool,
    ) -> (HyperBranchAction, [f64; 4], [f64; 6], f64) {
        let posterior = Self::branch_posterior(z, stability_margin, is_negative_real_confluent);
        let allowed: &[HyperBranchAction] = if is_negative_real_confluent {
            &[
                HyperBranchAction::DirectSeries,
                HyperBranchAction::KummerTransform,
                HyperBranchAction::Asymptotic,
                HyperBranchAction::GuardedFallback,
            ]
        } else if z < -0.5 {
            &[
                HyperBranchAction::DirectSeries,
                HyperBranchAction::PfaffTransform,
                HyperBranchAction::Asymptotic,
                HyperBranchAction::ContinuedFraction,
                HyperBranchAction::GuardedFallback,
            ]
        } else {
            &HyperBranchAction::ALL
        };
        self.select_action_for_allowed(posterior, allowed)
    }

    /// Select the minimum-expected-loss branch action restricted to `allowed` actions for the
    /// hypergeometric function family (`0F1`, `1F1`, or `2F1`).
    ///
    /// # Panics
    ///
    /// Panics if `allowed` is empty.
    pub fn select_action_for_allowed(
        &self,
        posterior: [f64; 4],
        allowed: &[HyperBranchAction],
    ) -> (HyperBranchAction, [f64; 4], [f64; 6], f64) {
        assert!(
            !allowed.is_empty(),
            "HyperSolverPortfolio::select_action_for_allowed requires at least one allowed action"
        );
        let losses = self.compute_expected_losses(posterior);

        let mut best: Option<(HyperBranchAction, f64)> = None;
        for &action in allowed {
            let loss = losses[action.index()];
            best = Some(match best {
                None => (action, loss),
                Some((prev_action, prev_loss)) => {
                    if loss < prev_loss
                        || ((loss - prev_loss).abs() <= 1e-12
                            && action.index() > prev_action.index())
                    {
                        (action, loss)
                    } else {
                        (prev_action, prev_loss)
                    }
                }
            });
        }
        let (argmin, argmin_loss) = best.expect("allowed is non-empty");

        // If conformal calibrator triggers drift, fallback to GuardedFallback if allowed.
        if self.calibrator.should_fallback() {
            let fallback_action = if allowed.contains(&HyperBranchAction::GuardedFallback) {
                HyperBranchAction::GuardedFallback
            } else {
                argmin
            };
            return (
                fallback_action,
                posterior,
                losses,
                losses[fallback_action.index()],
            );
        }

        (argmin, posterior, losses, argmin_loss)
    }

    pub fn record_evidence(&mut self, entry: HyperSolverEvidenceEntry) {
        // Never feed `term_count` (an integer iteration cap like 300 or 500) into the
        // calibrator: that compared term counts against `violation_threshold = 1e-8` and
        // latched `GuardedFallback` after 10 calls (frankenscipy-7tb8d.10). Only a genuine
        // accuracy signal (`estimated_truncation_error`) is observed.
        if let Some(err) = entry.estimated_truncation_error {
            self.calibrator.observe(err);
        }
        if self.evidence.len() >= self.evidence_capacity {
            let _ = self.evidence.pop_front();
        }
        self.evidence.push_back(entry);
    }

    pub fn observe_relative_error(&mut self, error: f64) {
        self.calibrator.observe(error);
    }

    /// Set the nonconformity violation threshold on this portfolio's calibrator.
    pub fn set_calibrator_violation_threshold(&mut self, threshold: f64) {
        self.calibrator.set_violation_threshold(threshold);
    }

    #[must_use]
    pub fn evidence_len(&self) -> usize {
        self.evidence.len()
    }

    #[must_use]
    pub const fn mode(&self) -> RuntimeMode {
        self.mode
    }

    #[must_use]
    pub fn calibrator(&self) -> &ConformalCalibrator {
        &self.calibrator
    }

    pub fn calibrator_mut(&mut self) -> &mut ConformalCalibrator {
        &mut self.calibrator
    }

    fn compute_expected_losses(&self, posterior: [f64; 4]) -> [f64; 6] {
        let mut losses = [0.0; 6];
        for (action_idx, row) in self.loss_matrix.iter().enumerate() {
            losses[action_idx] = row.iter().zip(posterior.iter()).map(|(l, p)| l * p).sum();
        }
        losses
    }

    fn branch_posterior(
        z: f64,
        stability_margin: f64,
        is_negative_real_confluent: bool,
    ) -> [f64; 4] {
        if !z.is_finite() || !stability_margin.is_finite() || stability_margin <= 1e-12 {
            return [0.0, 0.0, 0.0, 1.0]; // BoundaryNearPole
        }
        if (z - 1.0).abs() <= 0.05 {
            return [0.0, 0.02, 0.01, 0.97]; // BoundaryNearPole
        }
        let abs_z = z.abs();
        if abs_z >= 10.0 {
            return [0.0, 0.05, 0.9, 0.05]; // LargeArgumentAsymptotic
        }
        if is_negative_real_confluent || (z < -0.5 && abs_z < 10.0) {
            return [0.02, 0.95, 0.02, 0.01]; // TransformDomain
        }
        if abs_z <= 1.0 {
            [0.85, 0.05, 0.02, 0.08] // NearZeroDirectSeries
        } else {
            [0.35, 0.45, 0.15, 0.05]
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Test Helpers — Shared assertion and logging utilities (§bd-3jh.5)
// ═══════════════════════════════════════════════════════════════════

/// Structured test log entry for forensic comparison across runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestLogEntry {
    pub test_id: String,
    pub timestamp_ms: u64,
    pub level: TestLogLevel,
    pub module: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fixture_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<RuntimeMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<TestResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact_refs: Option<Vec<String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TestLogLevel {
    Info,
    Warn,
    Error,
    Debug,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TestResult {
    Pass,
    Fail,
    Skip,
    Warn,
}

impl TestLogEntry {
    #[must_use]
    pub fn new(
        test_id: impl Into<String>,
        module: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            test_id: test_id.into(),
            timestamp_ms: casp_now_unix_ms(),
            level: TestLogLevel::Info,
            module: module.into(),
            message: message.into(),
            seed: None,
            fixture_id: None,
            mode: None,
            result: None,
            artifact_refs: None,
        }
    }

    #[must_use]
    pub fn with_result(mut self, result: TestResult) -> Self {
        self.result = Some(result);
        self
    }

    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    #[must_use]
    pub fn with_mode(mut self, mode: RuntimeMode) -> Self {
        self.mode = Some(mode);
        self
    }

    #[must_use]
    pub fn with_fixture(mut self, fixture_id: impl Into<String>) -> Self {
        self.fixture_id = Some(fixture_id.into());
        self
    }

    /// Serialize to JSON line for structured logging.
    #[must_use]
    pub fn to_json_line(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| String::from("{}"))
    }
}

/// Returns true if `actual` and `expected` are close within a combined
/// absolute and relative tolerance, matching SciPy's
/// `numpy.testing.assert_allclose(..., equal_nan=True)` semantics.
///
/// Per frankenscipy-gtcn: also handles NaN/Inf without the naive
/// `(actual - expected).abs() <= atol + rtol * expected.abs()` pathology
/// that returns false for (+Inf == +Inf) and (NaN == NaN).
#[must_use]
pub fn close_within_tol(actual: f64, expected: f64, atol: f64, rtol: f64) -> bool {
    if actual.is_nan() && expected.is_nan() {
        return true;
    }
    if actual.is_infinite() || expected.is_infinite() {
        return actual == expected;
    }
    (actual - expected).abs() <= atol + rtol * expected.abs()
}

/// Assert two f64 values are close within combined absolute and relative tolerance.
///
/// Uses the formula: |actual - expected| <= atol + rtol * |expected|,
/// with NaN-NaN and Inf-Inf handled per numpy.testing.assert_allclose
/// (equal_nan=True). Per frankenscipy-gtcn.
pub fn assert_close(actual: f64, expected: f64, atol: f64, rtol: f64) {
    let tol = atol + rtol * expected.abs();
    assert!(
        close_within_tol(actual, expected, atol, rtol),
        "assert_close failed: actual={actual} expected={expected} diff={} tol={tol} (atol={atol}, rtol={rtol})",
        (actual - expected).abs()
    );
}

/// Assert two f64 slices are element-wise close within tolerance.
pub fn assert_close_slice(actual: &[f64], expected: &[f64], atol: f64, rtol: f64) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "assert_close_slice: length mismatch: actual={} expected={}",
        actual.len(),
        expected.len()
    );
    for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let tol = atol + rtol * e.abs();
        assert!(
            close_within_tol(*a, *e, atol, rtol),
            "assert_close_slice[{idx}]: actual={a} expected={e} diff={} tol={tol} (atol={atol}, rtol={rtol})",
            (a - e).abs()
        );
    }
}

/// Assert two 2D f64 matrices (`Vec<Vec<f64>>`) are element-wise close.
pub fn assert_close_matrix(actual: &[Vec<f64>], expected: &[Vec<f64>], atol: f64, rtol: f64) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "assert_close_matrix: row count mismatch: actual={} expected={}",
        actual.len(),
        expected.len()
    );
    for (row_idx, (a_row, e_row)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            a_row.len(),
            e_row.len(),
            "assert_close_matrix: column count mismatch at row {row_idx}"
        );
        for (col_idx, (a, e)) in a_row.iter().zip(e_row.iter()).enumerate() {
            let tol = atol + rtol * e.abs();
            assert!(
                close_within_tol(*a, *e, atol, rtol),
                "assert_close_matrix[{row_idx},{col_idx}]: actual={a} expected={e} diff={} tol={tol}",
                (a - e).abs()
            );
        }
    }
}

/// Check if a value is within absolute tolerance of expected. Handles
/// NaN-NaN and Inf-Inf correctly per frankenscipy-gtcn.
#[must_use]
pub fn within_tolerance(actual: f64, expected: f64, atol: f64, rtol: f64) -> bool {
    close_within_tol(actual, expected, atol, rtol)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[inline(never)]
    fn condition_posterior_former(rcond: f64) -> [f64; 4] {
        if !rcond.is_finite() || rcond <= 0.0 {
            return [0.0, 0.0, 0.0, 1.0];
        }

        let log_r = deterministic_log10(rcond.max(1e-25));
        let centers = [-2.0, -6.0, -11.0, -16.0];
        let mut posterior = [0.0; 4];

        if log_r >= centers[0] {
            posterior[0] = 1.0;
        } else if log_r <= centers[3] {
            posterior[3] = 1.0;
        } else {
            for i in 0..3 {
                let upper = centers[i];
                let lower = centers[i + 1];
                if log_r <= upper && log_r >= lower {
                    let upper_weight = (log_r - lower) / (upper - lower);
                    posterior[i] = upper_weight;
                    posterior[i + 1] = 1.0 - upper_weight;
                    break;
                }
            }
        }
        posterior
    }

    #[test]
    fn condition_posterior_endpoint_fast_paths_match_former_bits() {
        let well_boundary = 1e-2_f64.to_bits();
        let singular_boundary = 1e-16_f64.to_bits();
        let inputs = [
            f64::NAN,
            f64::NEG_INFINITY,
            f64::INFINITY,
            -1.0,
            -0.0,
            0.0,
            f64::MIN_POSITIVE,
            1e-25,
            f64::from_bits(singular_boundary - 1),
            1e-16,
            f64::from_bits(singular_boundary + 1),
            1e-12,
            1e-6,
            f64::from_bits(well_boundary - 1),
            1e-2,
            f64::from_bits(well_boundary + 1),
            1.0,
            f64::MAX,
        ];

        for rcond in inputs {
            let actual = SolverPortfolio::condition_posterior(rcond).map(f64::to_bits);
            let former = condition_posterior_former(rcond).map(f64::to_bits);
            assert_eq!(actual, former, "rcond={rcond:?}");
        }
    }

    /// The pre-7tb8d.1 decision rule, reproduced independently: argmin of the fixed map's
    /// expected losses with the safer-action tie-break.
    fn former_decision(rcond: f64, structure: Option<StructuralEvidence>) -> SolverAction {
        let p = condition_posterior_former(rcond);
        let m = SolverPortfolio::default_loss_matrix();
        let loss = |a: usize| (0..4).map(|s| m[a][s] * p[s]).sum::<f64>();
        let mut candidates = vec![0, 1, 2];
        match structure {
            Some(StructuralEvidence::Diagonal) => candidates.push(3),
            Some(StructuralEvidence::Triangular) => candidates.push(4),
            Some(StructuralEvidence::Symmetric) => candidates.push(5),
            _ => {}
        }
        let mut best = candidates[0];
        for &idx in &candidates[1..] {
            let (l, b) = (loss(idx), loss(best));
            if l < b
                || ((l - b).abs() <= 1e-12 && ((idx < 3 && idx > best) || (idx >= 3 && best < 3)))
            {
                best = idx;
            }
        }
        SolverAction::ALL[best]
    }

    // frankenscipy-7tb8d.1: with no outcomes recorded, the Dirichlet posterior IS the prior map,
    // so a fresh portfolio decides exactly as before on 200 log-spaced rcond values and the
    // special values.
    #[test]
    fn fresh_portfolio_decides_exactly_as_the_fixed_map() {
        let portfolio = SolverPortfolio::new(RuntimeMode::Strict, 8);
        let mut rconds: Vec<f64> = (0..200)
            .map(|i| 10f64.powf(-(i as f64) * 18.0 / 199.0))
            .collect();
        rconds.extend([0.0, -1.0, f64::NAN, f64::INFINITY, 1e-2, 1e-16]);
        for rcond in rconds {
            for structure in [
                None,
                Some(StructuralEvidence::Diagonal),
                Some(StructuralEvidence::Triangular),
                Some(StructuralEvidence::Symmetric),
            ] {
                let (action, posterior, _, _) = portfolio.select_action(rcond, structure);
                assert_eq!(
                    posterior.map(f64::to_bits),
                    condition_posterior_former(rcond).map(f64::to_bits),
                    "posterior at rcond={rcond:e}"
                );
                assert_eq!(action, former_decision(rcond, structure), "rcond={rcond:e}");
            }
        }
    }

    // frankenscipy-7tb8d.1: the posterior LEARNS. At rcond = 1e-3 the map says mostly
    // well-conditioned and LU wins; 50 recorded "LU was inaccurate" outcomes there must move
    // mass to the ill-conditioned states. Before that bead the posterior was a pure function of
    // rcond, so this failed.
    //
    // Whether the moved posterior moves the DECISION depends on the loss matrix. Under a
    // state-dependent one, the hand-set matrix fixed here, it does. Under the calibrated one
    // (frankenscipy-7tb8d.2), LU has the least loss in every state, so the posterior alone
    // cannot: LU fails by pivot growth, which a conditioning state does not describe. What moves
    // it there is the recorded failure rate of LU itself (frankenscipy-3iekl).
    #[test]
    fn recorded_outcomes_move_the_posterior_and_the_decision() {
        const HAND_SET: [[f64; 4]; 6] = [
            [1.0, 5.0, 40.0, 120.0],
            [3.0, 1.0, 8.0, 45.0],
            [15.0, 10.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 100.0],
            [0.0, 0.0, 0.0, 100.0],
            [0.0, 0.0, 0.0, 100.0],
        ];
        let mut portfolio = SolverPortfolio::new(RuntimeMode::Strict, 8);
        portfolio.loss_matrix = HAND_SET;
        let rcond = 1e-3;
        let (before, p0, _, _) = portfolio.select_action(rcond, None);
        assert_eq!(before, SolverAction::DirectLU);
        for _ in 0..50 {
            portfolio.record_outcome(rcond, SolverAction::DirectLU, AttemptOutcome::Inaccurate);
        }
        let (after, p1, losses, _) = portfolio.select_action(rcond, None);
        println!(
            "rcond 1e-3: prior {p0:?} -> posterior {p1:?}, losses {losses:?}, {before:?} -> {after:?}"
        );
        assert!(p1[2] + p1[3] > p0[2] + p0[3] + 0.5, "{p0:?} -> {p1:?}");
        assert!(matches!(
            after,
            SolverAction::PivotedQR | SolverAction::SVDFallback
        ));
        // Another decade is untouched.
        assert_eq!(portfolio.select_action(0.5, None).0, SolverAction::DirectLU);

        // Negative arm: successes at the same rcond keep LU -- including the FIRST one, which
        // a smoothed decision posterior would have flipped to QR.
        let mut ok = SolverPortfolio::new(RuntimeMode::Strict, 8);
        ok.loss_matrix = HAND_SET;
        ok.record_outcome(rcond, SolverAction::DirectLU, AttemptOutcome::Ok);
        assert_eq!(ok.select_action(rcond, None).0, SolverAction::DirectLU);
        for _ in 1..50 {
            ok.record_outcome(rcond, SolverAction::DirectLU, AttemptOutcome::Ok);
        }
        assert_eq!(ok.select_action(rcond, None).0, SolverAction::DirectLU);
        assert!(ok.posterior(rcond)[0] >= p0[0] - 0.05);

        // Under the calibrated losses the same outcomes move the posterior, and LU's own
        // failure rate moves the choice (frankenscipy-3iekl).
        let mut calibrated = SolverPortfolio::new(RuntimeMode::Strict, 8);
        for _ in 0..50 {
            calibrated.record_outcome(rcond, SolverAction::DirectLU, AttemptOutcome::Inaccurate);
        }
        let (choice, p2, ..) = calibrated.select_action(rcond, None);
        assert!(p2[2] + p2[3] > p0[2] + p0[3] + 0.5, "{p0:?} -> {p2:?}");
        assert_eq!(choice, SolverAction::PivotedQR);

        // frankenscipy-7tb8d.1: 50 consecutive inaccurate LU outcomes at rcond = 1e-9 shift
        // posterior mass toward IllConditioned/NearSingular and switch select_action(1e-9)
        // from DirectLU to PivotedQR or SVDFallback.
        let rcond_1e9 = 1e-9;
        let mut p_1e9 = SolverPortfolio::new(RuntimeMode::Strict, 8);
        let (before_1e9, prior_1e9, _, _) = p_1e9.select_action(rcond_1e9, None);
        assert_eq!(before_1e9, SolverAction::DirectLU);
        for _ in 0..50 {
            p_1e9.record_outcome(
                rcond_1e9,
                SolverAction::DirectLU,
                AttemptOutcome::Inaccurate,
            );
        }
        let (after_1e9, post_1e9, _, _) = p_1e9.select_action(rcond_1e9, None);
        assert!(
            post_1e9[2] + post_1e9[3] > prior_1e9[2] + prior_1e9[3] + 0.35,
            "{prior_1e9:?} -> {post_1e9:?}"
        );
        assert!(matches!(
            after_1e9,
            SolverAction::PivotedQR | SolverAction::SVDFallback
        ));
    }

    #[test]
    fn persistent_policy_controller_and_eprocess_alarm_in_portfolio() {
        let mut portfolio = SolverPortfolio::new(RuntimeMode::Strict, 16);
        assert_eq!(portfolio.policy_controller().ledger().len(), 0);
        assert!(
            !portfolio
                .policy_controller()
                .eprocess()
                .is_alarm_triggered()
        );

        // 180 nominal solves keep both the e-process and the 200-sample conformal window nominal.
        for _ in 0..180 {
            portfolio.observe_backward_error(1e-15);
        }
        assert!(
            !portfolio
                .policy_controller()
                .eprocess()
                .is_alarm_triggered()
        );
        assert_eq!(
            portfolio.select_action(1e-2, None).0,
            SolverAction::DirectLU
        );

        // A sudden changepoint burst of 8 backward-error exceedances (8/188 = 4.25% < 5% alpha,
        // so the 200-sample ConformalCalibrator has NOT yet triggered) already trips the
        // anytime-valid EProcessMonitor and overrides select_action to SVDFallback.
        for _ in 0..8 {
            portfolio.observe_backward_error(1e-3);
        }
        assert!(
            portfolio
                .policy_controller()
                .eprocess()
                .is_alarm_triggered()
        );
        assert!(!portfolio.calibrator().should_fallback());
        assert_eq!(
            portfolio.select_action(1e-2, None).0,
            SolverAction::SVDFallback
        );
    }

    /// frankenscipy-3iekl: under the calibrated losses, recorded failures of an action raise ITS
    /// expected loss at that rcond decade until the next solver wins, and recorded successes
    /// lower it again. A fresh portfolio decides exactly as the calibrated matrix does.
    #[test]
    fn recorded_failures_move_the_calibrated_decision() {
        let rcond = 1e-3;
        let mut portfolio = SolverPortfolio::new(RuntimeMode::Strict, 8);
        let (cold, posterior, cold_losses, _) = portfolio.select_action(rcond, None);
        assert_eq!(cold, SolverAction::DirectLU);
        for (a, row) in SolverPortfolio::default_loss_matrix().iter().enumerate() {
            let calibrated: f64 = row.iter().zip(posterior).map(|(l, p)| l * p).sum();
            assert_eq!(cold_losses[a].to_bits(), calibrated.to_bits(), "action {a}");
        }

        // LU keeps failing at this decade: it holds for a few failures, then QR takes over.
        let mut failures = 0;
        while portfolio.select_action(rcond, None).0 == SolverAction::DirectLU {
            portfolio.record_outcome(rcond, SolverAction::DirectLU, AttemptOutcome::Inaccurate);
            failures += 1;
            assert!(failures <= 10, "LU still chosen after {failures} failures");
        }
        let (switched, _, losses, _) = portfolio.select_action(rcond, None);
        println!("rcond 1e-3: QR after {failures} LU failures, losses {losses:?}");
        assert_eq!(switched, SolverAction::PivotedQR);
        assert!(failures >= 2, "one failure must not be enough ({failures})");
        assert_eq!(
            portfolio.failure_counts(rcond)[SolverAction::DirectLU.index()],
            [f64::from(failures), f64::from(failures)]
        );

        // Must-miss: the failures stay at their decade.
        for other in [0.5, 1e-2, 1e-4, 1e-9] {
            assert_eq!(
                portfolio.select_action(other, None).0,
                SolverAction::DirectLU,
                "rcond {other:e}"
            );
        }

        // Successes bring LU back.
        let mut successes = 0;
        while portfolio.select_action(rcond, None).0 != SolverAction::DirectLU {
            portfolio.record_outcome(rcond, SolverAction::DirectLU, AttemptOutcome::Ok);
            successes += 1;
            assert!(successes <= 10, "LU not back after {successes} successes");
        }
        println!("rcond 1e-3: LU back after {successes} successes");

        // The tallies are part of the state a certificate replays against.
        let mut untallied = portfolio.clone();
        untallied.failure_counts = [[[0.0; 2]; 6]; RCOND_DECADES];
        assert_ne!(portfolio.state_digest(), untallied.state_digest());
    }

    // frankenscipy-7tb8d.1: the fallback re-ranks the REMAINING actions under the updated
    // posterior, and returns None once every applicable action has been tried.
    #[test]
    fn select_action_excluding_reranks_the_remaining_actions() {
        let mut portfolio = SolverPortfolio::new(RuntimeMode::Strict, 8);
        let rcond = 1e-3;
        portfolio.record_outcome(rcond, SolverAction::DirectLU, AttemptOutcome::Failed);
        let (second, ..) = portfolio
            .select_action_excluding(rcond, None, &[SolverAction::DirectLU])
            .expect("QR and SVD remain");
        assert_ne!(second, SolverAction::DirectLU);
        let all = [
            SolverAction::DirectLU,
            SolverAction::PivotedQR,
            SolverAction::SVDFallback,
        ];
        assert!(
            portfolio
                .select_action_excluding(rcond, None, &all)
                .is_none()
        );
        assert!(portfolio.outcome_counts(rcond).iter().sum::<f64>() > 0.99);
    }

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
        let logits = policy::logits_from_signals(DecisionSignals::new(2.0, 0.1, 0.1));
        let posterior = policy::softmax(logits);
        let sum = posterior.iter().sum::<f64>();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    // ═══ CASP tests ═══

    #[test]
    fn casp_selects_lu_for_well_conditioned() {
        let portfolio = SolverPortfolio::new(RuntimeMode::Strict, 64);
        let (action, _, _, _) = portfolio.select_action(1e-2, None);
        assert_eq!(action, SolverAction::DirectLU);
    }

    // frankenscipy-7tb8d.2: under the calibrated losses LU is chosen at every conditioning
    // (the hand-set matrix chose QR at 1e-6 and the SVD at 1e-12 and 1e-18), and the SVD has the
    // largest expected loss of the three general solvers.
    #[test]
    fn casp_selects_lu_at_every_conditioning() {
        let portfolio = SolverPortfolio::new(RuntimeMode::Strict, 64);
        for rcond in [1e-6, 1e-12, 1e-18] {
            let (action, _, losses, _) = portfolio.select_action(rcond, None);
            assert_eq!(action, SolverAction::DirectLU, "rcond {rcond:e}");
            assert!(
                losses[SolverAction::SVDFallback.index()] > losses[SolverAction::PivotedQR.index()],
                "rcond {rcond:e}: {losses:?}"
            );
        }
    }

    #[test]
    fn casp_soft_transitions() {
        let portfolio = SolverPortfolio::new(RuntimeMode::Strict, 64);
        // Mid-way between -2 (Well) and -6 (Mod) is -4 (rcond=1e-4)
        let (_, posterior, _, _) = portfolio.select_action(1e-4, None);
        assert_close(posterior[0], 0.5, 1e-10, 1e-10);
        assert_close(posterior[1], 0.5, 1e-10, 1e-10);
    }

    #[test]
    fn casp_selects_fast_path_when_available() {
        let portfolio = SolverPortfolio::new(RuntimeMode::Strict, 64);
        let (action, _, _, _) = portfolio.select_action(1e-2, Some(StructuralEvidence::Diagonal));
        assert_eq!(action, SolverAction::DiagonalFastPath);
    }

    // frankenscipy-7tb8d.14: the symmetric factorization is a candidate only for symmetric
    // evidence. Under the calibrated losses (frankenscipy-7tb8d.2) it is chosen for symmetric
    // evidence at every conditioning, near singular included: it failed no solvable corpus
    // matrix, where the SVD failed most near-singular ones.
    #[test]
    fn casp_offers_the_symmetric_path_only_for_symmetric_evidence() {
        let portfolio = SolverPortfolio::new(RuntimeMode::Strict, 64);
        let symmetric = Some(StructuralEvidence::Symmetric);
        for rcond in [1e-2, 1e-6, 1e-11, 1e-16] {
            assert_eq!(
                portfolio.select_action(rcond, symmetric).0,
                SolverAction::SymmetricFastPath
            );
            for other in [None, Some(StructuralEvidence::General)] {
                assert_ne!(
                    portfolio.select_action(rcond, other).0,
                    SolverAction::SymmetricFastPath
                );
            }
        }
        let excluded = portfolio
            .select_action_excluding(1e-2, symmetric, &[SolverAction::SymmetricFastPath])
            .expect("the general solvers remain");
        assert_eq!(excluded.0, SolverAction::DirectLU);
    }

    #[test]
    fn casp_evidence_is_bounded() {
        let mut portfolio = SolverPortfolio::new(RuntimeMode::Strict, 3);
        for index in 0..5 {
            portfolio.record_evidence(SolverEvidenceEntry {
                component: "test",
                matrix_shape: (index, index),
                rcond_estimate: 0.5,
                chosen_action: SolverAction::DirectLU,
                posterior: vec![1.0, 0.0, 0.0, 0.0],
                expected_losses: vec![1.0, 3.0, 15.0, 0.0, 0.0],
                chosen_expected_loss: 1.0,
                fallback_active: false,
                backward_error: None,
            });
        }
        assert_eq!(portfolio.evidence_len(), 3);
        assert_eq!(
            portfolio
                .evidence
                .iter()
                .map(|entry| entry.matrix_shape)
                .collect::<Vec<_>>(),
            [(2, 2), (3, 3), (4, 4)]
        );
    }

    #[test]
    fn casp_jsonl_serialization() {
        let mut portfolio = SolverPortfolio::new(RuntimeMode::Strict, 16);
        portfolio.record_evidence(SolverEvidenceEntry {
            component: "test\"entry",
            matrix_shape: (4, 4),
            rcond_estimate: 0.01,
            chosen_action: SolverAction::PivotedQR,
            posterior: vec![0.0, 1.0, 0.0, 0.0],
            expected_losses: vec![5.0, 1.0, 10.0, 0.0, 0.0],
            chosen_expected_loss: 1.0,
            fallback_active: false,
            backward_error: None,
        });
        portfolio.record_evidence(SolverEvidenceEntry {
            component: "test\nentry",
            matrix_shape: (8, 2),
            rcond_estimate: 1e-12,
            chosen_action: SolverAction::SVDFallback,
            posterior: vec![0.0, 0.0, 1.0, 0.0],
            expected_losses: vec![40.0, 8.0, 1.0, 0.0, 0.0],
            chosen_expected_loss: 1.0,
            fallback_active: true,
            backward_error: Some(1e-14),
        });
        // Each line is the entry's own JSON with `portfolio` and `mode` spliced in front.
        let former = portfolio
            .evidence
            .iter()
            .filter_map(|entry| serde_json::to_string(entry).ok())
            .map(|json| {
                format!(
                    "{{\"portfolio\":\"solver\",\"mode\":\"Strict\",{}",
                    &json[1..]
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let jsonl = portfolio.serialize_jsonl();
        assert_eq!(jsonl, former);
        let parsed = serde_json::from_str::<serde_json::Value>(
            jsonl.lines().next().expect("missing JSONL evidence entry"),
        )
        .expect("invalid JSONL evidence entry");
        assert_eq!(parsed["component"], "test\"entry");
    }

    /// frankenscipy-7tb8d.2: the dense portfolio's loss matrix is the committed calibration, bit
    /// for bit, and the calibration is its own rule applied to its own measurements. A hand
    /// edit of the generated module, of the report, or of both alike fails here; regenerate
    /// both with `casp_calibrate` instead.
    #[test]
    fn solver_loss_matrix_is_the_committed_calibration() {
        let report: serde_json::Value = serde_json::from_str(include_str!(
            "../../../artifacts/casp-calibration-solver.json"
        ))
        .expect("the calibration report parses");
        let table = |key: &str| -> Vec<Vec<f64>> {
            report[key]
                .as_array()
                .expect("a table of the report")
                .iter()
                .map(|row| {
                    row.as_array()
                        .expect("a row")
                        .iter()
                        .map(|v| v.as_f64().expect("a number"))
                        .collect()
                })
                .collect()
        };
        let actions: Vec<&str> = report["actions"]
            .as_array()
            .expect("actions")
            .iter()
            .map(|a| a.as_str().expect("an action name"))
            .collect();
        let names: Vec<String> = SolverAction::ALL.iter().map(|a| format!("{a:?}")).collect();
        assert_eq!(actions, names, "the report's rows are SolverAction::ALL");

        let recorded = table("loss_matrix");
        let matrix = SolverPortfolio::default_loss_matrix();
        let cost: Vec<f64> = report["cost"]
            .as_array()
            .expect("cost")
            .iter()
            .map(|v| v.as_f64().expect("a cost"))
            .collect();
        let (recovery, total) = (table("recovery_weight"), table("total_weight"));
        for (a, row) in matrix.iter().enumerate() {
            for (s, &loss) in row.iter().enumerate() {
                assert_eq!(
                    loss.to_bits(),
                    recorded[a][s].to_bits(),
                    "{} in state {s}: the constant {loss:e}, the report {:e}",
                    names[a],
                    recorded[a][s]
                );
                // The report's rule, recomputed from its measurements in the generator's order.
                assert!(total[a][s] > 0.0, "{} unmeasured in state {s}", names[a]);
                let rule = cost[a] + recovery[a][s] / total[a][s];
                assert_eq!(
                    rule.to_bits(),
                    loss.to_bits(),
                    "{} in state {s}: the rule gives {rule:e}, the loss is {loss:e}",
                    names[a]
                );
            }
        }

        // frankenscipy-3iekl: the loss's two terms, as generated, are the report's too.
        let rates = table("failure_rate");
        for a in 0..6 {
            assert_eq!(
                calibrated_losses::SOLVER_ACTION_COST[a].to_bits(),
                cost[a].to_bits(),
                "{} cost",
                names[a]
            );
            for s in 0..4 {
                assert_eq!(
                    calibrated_losses::SOLVER_FAILURE_RATE[a][s].to_bits(),
                    rates[a][s].to_bits(),
                    "{} failure rate in state {s}",
                    names[a]
                );
            }
        }

        // frankenscipy-7tb8d.3: Conformal Risk Control constants match the report's corpus.
        let seed_str = report["seed"]
            .as_str()
            .expect("seed string")
            .trim_start_matches("0x");
        let report_seed = u64::from_str_radix(seed_str, 16).expect("hex seed");
        assert_eq!(calibrated_losses::SOLVER_CRC_SEED, report_seed);
        let matrices = report["matrices"].as_u64().expect("matrices") as usize;
        let unsolvable = report["unsolvable"].as_u64().expect("unsolvable") as usize;
        assert_eq!(calibrated_losses::SOLVER_CRC_N, matrices - unsolvable);
        let mut samples = Vec::new();
        for cell in report["cells"].as_array().expect("cells") {
            let outcomes = cell["outcomes"].as_array().expect("outcomes");
            let solvable = outcomes
                .iter()
                .any(|o| !o["failed"].as_bool().unwrap_or(true));
            if !solvable {
                continue;
            }
            let rcond = cell["rcond_estimate"].as_f64().unwrap_or(0.0);
            let lu_failed = outcomes
                .iter()
                .find(|o| o["action"].as_str() == Some("DirectLU"))
                .is_some_and(|o| o["failed"].as_bool().unwrap_or(true));
            let qr_failed = outcomes
                .iter()
                .find(|o| o["action"].as_str() == Some("PivotedQR"))
                .is_some_and(|o| o["failed"].as_bool().unwrap_or(true));
            samples.push(CrcCalibrationSample {
                rcond,
                unescalated_exceeded: lu_failed,
                escalated_exceeded: qr_failed,
            });
        }
        assert_eq!(samples.len(), calibrated_losses::SOLVER_CRC_N);
        let calibrated_crc = ConformalRiskControl::calibrate(
            &samples,
            calibrated_losses::SOLVER_CRC_ALPHA,
            report_seed,
        );
        assert_eq!(calibrated_crc, ConformalRiskControl::default_calibrated());
        assert_eq!(
            calibrated_crc.t_hat().to_bits(),
            calibrated_losses::SOLVER_CRC_T_HAT.to_bits()
        );
    }

    /// frankenscipy-7tb8d.2: every action does work on the calibration corpus, as the first
    /// choice on some matrix or as the fallback that succeeds when the first choice fails,
    /// except the ones named here, which are retained only as later fallbacks. A recalibration
    /// that changes the set fails here until someone decides again.
    ///
    /// `SVDFallback`: refuses a square system whose numerical rank is short (the rank cutoff
    /// `max(m, n)·eps·σ_max`), and on the full-rank ones it accepts, Householder QR is backward
    /// stable too, at `4n³/3` flops against its `21n³`. On the corpus it is never chosen, never
    /// the fallback that succeeds and never the only action that succeeds; whether to remove it
    /// from the dense portfolio or give it a job is frankenscipy-qmgrm.
    #[test]
    fn every_solver_action_is_reached_or_named_as_retained() {
        const RETAINED_UNREACHED: [SolverAction; 1] = [SolverAction::SVDFallback];
        let report: serde_json::Value = serde_json::from_str(include_str!(
            "../../../artifacts/casp-calibration-solver.json"
        ))
        .expect("the calibration report parses");
        let unreached = |report: &serde_json::Value| -> Vec<SolverAction> {
            let count = |key: &str, action: SolverAction| {
                report[key][format!("{action:?}")]
                    .as_u64()
                    .expect("the report's counts cover every action")
            };
            SolverAction::ALL
                .into_iter()
                .filter(|&a| count("chosen_on_corpus", a) == 0 && count("fallback_success", a) == 0)
                .collect()
        };
        assert_eq!(unreached(&report), RETAINED_UNREACHED);
        let listed: Vec<&str> = report["unreached"]
            .as_array()
            .expect("the report's unreached list")
            .iter()
            .map(|a| a.as_str().expect("an action name"))
            .collect();
        assert_eq!(listed, ["SVDFallback"], "the report's own list agrees");

        // Must-miss: the one matrix QR rescues is all that keeps it reached. Without it, QR
        // would join the unreached set and the assertion above would fail.
        assert!(report["fallback_success"]["PivotedQR"].as_u64() > Some(0));
        let mut without_rescue = report.clone();
        without_rescue["fallback_success"]["PivotedQR"] = 0.into();
        assert_eq!(
            unreached(&without_rescue),
            [SolverAction::PivotedQR, SolverAction::SVDFallback]
        );
    }

    /// frankenscipy-7tb8d.11: every portfolio's recorded decision reads back through its JSONL
    /// with the portfolio, the mode, the action and, bit for bit, the posterior and the losses.
    #[test]
    fn every_portfolio_round_trips_a_decision_through_jsonl() {
        fn check<P: PortfolioEvidence>(
            portfolio: &P,
            action: impl Serialize,
            posterior: &[f64],
            losses: &[f64],
        ) {
            let jsonl = portfolio.serialize_jsonl();
            assert_eq!(jsonl.lines().count(), 1, "{jsonl}");
            let line: serde_json::Value = serde_json::from_str(&jsonl).expect("one JSON line");
            assert_eq!(line["portfolio"], P::NAME, "{jsonl}");
            assert_eq!(line["mode"], "Hardened", "{jsonl}");
            assert_eq!(
                line["chosen_action"],
                serde_json::to_value(action).expect("action"),
                "{jsonl}"
            );
            let bits = |values: &[f64]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            for (key, expected) in [("posterior", posterior), ("expected_losses", losses)] {
                let read: Vec<f64> = line[key]
                    .as_array()
                    .expect(key)
                    .iter()
                    .map(|v| v.as_f64().expect("number"))
                    .collect();
                assert_eq!(bits(&read), bits(expected), "{key} in {jsonl}");
            }
        }
        let mode = RuntimeMode::Hardened;

        let mut solver = SolverPortfolio::new(mode, 4);
        let (action, posterior, losses, chosen) =
            solver.select_action(3.7e-7, Some(StructuralEvidence::General));
        solver.record_evidence(SolverEvidenceEntry {
            component: "test",
            matrix_shape: (3, 3),
            rcond_estimate: 3.7e-7,
            chosen_action: action,
            posterior: posterior.to_vec(),
            expected_losses: losses.to_vec(),
            chosen_expected_loss: chosen,
            fallback_active: false,
            backward_error: Some(1.25e-16),
        });
        check(&solver, action, &posterior, &losses);

        let mut sparse = SparseSolverPortfolio::new(mode, 4);
        let (action, posterior, losses, chosen) = sparse.select_action(
            1.3e5,
            Some(SparseStructuralEvidence {
                is_symmetric: true,
                is_positive_definite_hint: Some(true),
            }),
        );
        sparse.record_evidence(SparseSolverEvidenceEntry {
            component: "test",
            matrix_shape: (5, 5),
            nnz: 13,
            cond_estimate: 1.3e5,
            chosen_action: action,
            posterior: posterior.to_vec(),
            expected_losses: losses.to_vec(),
            chosen_expected_loss: chosen,
            fallback_active: false,
            relative_residual: None,
        });
        check(&sparse, action, &posterior, &losses);

        let mut opt = OptSolverPortfolio::new(mode, 4);
        let (action, posterior, losses, chosen) = opt.select_action(2.2e4, false, true);
        opt.record_evidence(OptSolverEvidenceEntry {
            component: "test",
            dimension: 7,
            condition_number_estimate: 2.2e4,
            chosen_action: action,
            posterior: posterior.to_vec(),
            expected_losses: losses.to_vec(),
            chosen_expected_loss: chosen,
            fallback_active: true,
            gradient_norm: Some(3.1e-9),
        });
        check(&opt, action, &posterior, &losses);

        let mut ode = OdeSolverPortfolio::new(mode, 4);
        let (action, posterior, losses, chosen) = ode.select_action(250.0, false);
        ode.record_evidence(OdeSolverEvidenceEntry {
            component: "test",
            system_dim: 2,
            stiffness_ratio_estimate: 250.0,
            chosen_action: action,
            posterior: posterior.to_vec(),
            expected_losses: losses.to_vec(),
            chosen_expected_loss: chosen,
            fallback_active: false,
            step_rejections: Some(1),
        });
        check(&ode, action, &posterior, &losses);

        let mut hyper = HyperSolverPortfolio::new(mode, 4);
        let (action, posterior, losses, chosen) = hyper.select_action(0.7, 0.3, false);
        hyper.record_evidence(HyperSolverEvidenceEntry {
            component: "test",
            function_kind: "hyp2f1",
            z_abs: 0.7,
            parameter_stability_margin: 0.3,
            chosen_action: action,
            posterior: posterior.to_vec(),
            expected_losses: losses.to_vec(),
            chosen_expected_loss: chosen,
            fallback_active: false,
            term_count: Some(31),
            estimated_truncation_error: Some(1e-15),
        });
        check(&hyper, action, &posterior, &losses);
    }

    #[test]
    fn conformal_calibrator_no_fallback_initially() {
        let cal = ConformalCalibrator::new(0.05, 100);
        assert!(!cal.should_fallback());
        assert_eq!(cal.empirical_miscoverage(), 0.0);
    }

    #[test]
    fn conformal_calibrator_triggers_fallback_on_high_violations() {
        let mut cal = ConformalCalibrator::new(0.05, 100);
        // Feed 50 good scores, then 10 bad ones
        for _ in 0..50 {
            cal.observe(1e-15);
        }
        assert!(!cal.should_fallback());
        for _ in 0..10 {
            cal.observe(1.0); // high nonconformity scores
        }
        // 10/60 ≈ 0.167 > 0.05 + 0.02 = 0.07 → should fallback
        assert!(cal.should_fallback());
    }

    #[test]
    fn conformal_calibrator_is_bounded() {
        let mut cal = ConformalCalibrator::new(0.05, 20);
        for _ in 0..30 {
            cal.observe(1e-15);
        }
        assert_eq!(cal.scores.len(), 20);
    }

    // ═══ Module structure tests ═══

    #[test]
    fn signal_sequence_basic_usage() {
        let mut seq = SignalSequence::new("test_adversarial");
        seq.push(DecisionSignals::new(2.0, 0.0, 0.0));
        seq.push(DecisionSignals::new(8.0, 1.0, 0.5));
        assert_eq!(seq.len(), 2);
        assert!(!seq.is_empty());
        assert_eq!(seq.id, "test_adversarial");
    }

    #[test]
    fn signal_sequence_replay_through_controller() {
        let mut seq = SignalSequence::new("replay_test");
        seq.push(DecisionSignals::new(2.0, 0.0, 0.0));
        seq.push(DecisionSignals::new(0.0, 1.0, 0.0));

        let mut controller = PolicyController::new(RuntimeMode::Strict, 16);
        let decisions: Vec<_> = seq.iter().map(|s| controller.decide(*s)).collect();

        assert_eq!(decisions[0].action, PolicyAction::Allow);
        assert_eq!(decisions[1].action, PolicyAction::FailClosed);
        assert_eq!(controller.ledger().len(), 2);
    }

    #[test]
    fn decision_signals_is_finite() {
        let good = DecisionSignals::new(2.0, 0.5, 0.1);
        assert!(good.is_finite());

        let nan = DecisionSignals::new(f64::NAN, 0.0, 0.0);
        assert!(!nan.is_finite());

        let inf = DecisionSignals::new(0.0, f64::INFINITY, 0.0);
        assert!(!inf.is_finite());
    }

    // ═══ Test helper tests ═══

    #[test]
    fn test_helpers_assert_close_exact() {
        assert_close(1.0, 1.0, 1e-12, 1e-12);
    }

    #[test]
    fn test_helpers_assert_close_within_atol() {
        assert_close(1.0 + 1e-13, 1.0, 1e-12, 0.0);
    }

    #[test]
    fn test_helpers_assert_close_within_rtol() {
        assert_close(100.0 + 1e-10, 100.0, 0.0, 1e-11);
    }

    #[test]
    #[should_panic(expected = "assert_close failed")]
    fn test_helpers_assert_close_rejects_far() {
        assert_close(1.0, 2.0, 1e-12, 1e-12);
    }

    #[test]
    fn test_helpers_assert_close_slice_ok() {
        assert_close_slice(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], 1e-12, 1e-12);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn test_helpers_assert_close_slice_length_mismatch() {
        assert_close_slice(&[1.0, 2.0], &[1.0], 1e-12, 1e-12);
    }

    #[test]
    fn test_helpers_assert_close_matrix_ok() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert_close_matrix(&a, &a, 1e-12, 1e-12);
    }

    #[test]
    fn test_helpers_within_tolerance() {
        assert!(within_tolerance(1.0, 1.0, 1e-12, 1e-12));
        assert!(!within_tolerance(1.0, 2.0, 1e-12, 1e-12));
    }

    #[test]
    fn close_within_tol_exact_equality_passes() {
        // /mock-code-finder for [frankenscipy-ryfk0]:
        // exact-equal values must pass even at zero tolerance.
        for &v in &[0.0_f64, 1.0, -3.5, 1e10, -1e-10] {
            assert!(close_within_tol(v, v, 0.0, 0.0));
            assert!(close_within_tol(v, v, 1e-12, 1e-12));
        }
    }

    #[test]
    fn close_within_tol_at_threshold_accepts() {
        // Boundary inclusion: |a - e| ≤ atol + rtol · |e| (≤, not <).
        // For e=10, rtol=0.1, atol=0: tol = 1.0 exactly. So a=11 must pass.
        assert!(close_within_tol(11.0, 10.0, 0.0, 0.1));
        assert!(close_within_tol(9.0, 10.0, 0.0, 0.1));
        // Above the threshold rejects.
        assert!(!close_within_tol(11.0001, 10.0, 0.0, 0.1));
    }

    #[test]
    fn close_within_tol_atol_only_for_zero_expected() {
        // When expected=0, rtol contributes 0 — only atol matters.
        assert!(close_within_tol(0.5, 0.0, 0.5, 1.0));
        assert!(close_within_tol(0.5, 0.0, 0.5, 0.0));
        assert!(!close_within_tol(0.6, 0.0, 0.5, 0.0));
    }

    #[test]
    fn close_within_tol_rtol_scales_with_expected_magnitude() {
        // rtol scales with |expected| not |actual|. For rtol=0.01:
        //   expected=100, actual=101: tol=1.0, |diff|=1.0 → pass
        //   expected=1, actual=100:   tol=0.01, |diff|=99 → fail
        assert!(close_within_tol(101.0, 100.0, 0.0, 0.01));
        assert!(!close_within_tol(100.0, 1.0, 0.0, 0.01));
        // Asymmetry check: tolerance based on EXPECTED, so swapping
        // actual and expected can flip the result.
        assert!(close_within_tol(0.0001, 0.001, 0.0, 1.0));
        // Now swap — expected is much smaller, rtol·|expected| collapses.
        assert!(!close_within_tol(0.001, 0.0001, 0.0, 1.0));
    }

    #[test]
    fn within_tolerance_handles_nan_and_infinity() {
        // Per frankenscipy-gtcn: (NaN, NaN) accepted; (Inf, Inf) accepted
        // but sign-preserving; (Inf, -Inf) rejected.
        assert!(within_tolerance(f64::NAN, f64::NAN, 1e-12, 1e-12));
        assert!(within_tolerance(f64::INFINITY, f64::INFINITY, 1e-12, 1e-12));
        assert!(within_tolerance(
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            1e-12,
            1e-12
        ));
        assert!(!within_tolerance(
            f64::INFINITY,
            f64::NEG_INFINITY,
            1e-12,
            1e-12
        ));
        assert!(!within_tolerance(f64::NAN, 1.0, 1e-12, 1e-12));
    }

    #[test]
    fn test_helpers_log_entry_serializes() {
        let entry = TestLogEntry::new("test_foo", "fsci_linalg", "solve passed")
            .with_result(TestResult::Pass)
            .with_seed(42)
            .with_mode(RuntimeMode::Strict);
        let json = entry.to_json_line();
        let parsed =
            serde_json::from_str::<serde_json::Value>(&json).expect("invalid test log entry JSON");
        assert_eq!(parsed["test_id"], "test_foo");
        assert_eq!(parsed["result"], "pass");
        assert_eq!(parsed["seed"], 42);
        assert_eq!(parsed["mode"], "Strict");
    }

    #[test]
    fn test_helpers_log_entry_omits_none_fields() {
        let entry = TestLogEntry::new("test_bar", "fsci_integrate", "quad converged");
        let json = entry.to_json_line();
        let parsed =
            serde_json::from_str::<serde_json::Value>(&json).expect("invalid test log entry JSON");
        assert!(parsed.get("seed").is_none());
        assert!(parsed.get("fixture_id").is_none());
        assert!(parsed.get("mode").is_none());
        assert!(parsed.get("result").is_none());
    }

    #[test]
    fn test_sparse_solver_portfolio_spd_selects_cg() {
        let p = SparseSolverPortfolio::new(RuntimeMode::Strict, 16);
        let (action, _post, _losses, _best) = p.select_action(
            100.0,
            Some(SparseStructuralEvidence {
                is_symmetric: true,
                is_positive_definite_hint: Some(true),
            }),
        );
        assert_eq!(action, SparseSolverAction::ConjugateGradient);
    }

    #[test]
    fn test_sparse_solver_portfolio_general_well_selects_bicgstab() {
        let p = SparseSolverPortfolio::new(RuntimeMode::Strict, 16);
        let (action, _post, _losses, _best) = p.select_action(
            100.0,
            Some(SparseStructuralEvidence {
                is_symmetric: false,
                is_positive_definite_hint: None,
            }),
        );
        assert_eq!(action, SparseSolverAction::BiCGSTAB);
    }

    #[test]
    fn test_sparse_solver_portfolio_indefinite_selects_minres() {
        let p = SparseSolverPortfolio::new(RuntimeMode::Strict, 16);
        let (action, _post, _losses, _best) = p.select_action(
            100.0,
            Some(SparseStructuralEvidence {
                is_symmetric: true,
                is_positive_definite_hint: Some(false),
            }),
        );
        assert_eq!(action, SparseSolverAction::MinRes);
    }

    #[test]
    fn test_sparse_solver_portfolio_ill_conditioned_selects_superlu() {
        let p = SparseSolverPortfolio::new(RuntimeMode::Strict, 16);
        let (action, _post, _losses, _best) = p.select_action(1e9, None);
        assert_eq!(action, SparseSolverAction::SuperLU);
    }

    #[test]
    fn test_sparse_solver_portfolio_every_action_is_reachable_and_outcomes_move_posterior() {
        let p = SparseSolverPortfolio::new(RuntimeMode::Strict, 16);
        let sym_pd = Some(SparseStructuralEvidence {
            is_symmetric: true,
            is_positive_definite_hint: Some(true),
        });
        let sym_indef = Some(SparseStructuralEvidence {
            is_symmetric: true,
            is_positive_definite_hint: Some(false),
        });
        let nonsym = Some(SparseStructuralEvidence {
            is_symmetric: false,
            is_positive_definite_hint: None,
        });

        let (a_cg, _, _, _) = p.select_action(100.0, sym_pd);
        let (a_minres, _, _, _) = p.select_action(100.0, sym_indef);
        let (a_bicgstab, _, _, _) = p.select_action(100.0, nonsym);
        let (a_qmr, _, _, _) = p.select_action(1e5, nonsym);
        let (a_gmres, _, _, _) = p.select_action(2e6, nonsym);
        let (a_superlu, _, _, _) = p.select_action(1e8, nonsym);

        assert_eq!(a_cg, SparseSolverAction::ConjugateGradient);
        assert_eq!(a_minres, SparseSolverAction::MinRes);
        assert_eq!(a_bicgstab, SparseSolverAction::BiCGSTAB);
        assert_eq!(a_qmr, SparseSolverAction::QMR);
        assert_eq!(a_gmres, SparseSolverAction::GMRES);
        assert_eq!(a_superlu, SparseSolverAction::SuperLU);

        let reached = [a_cg, a_minres, a_bicgstab, a_qmr, a_gmres, a_superlu];
        for action in SparseSolverAction::ALL {
            assert!(
                reached.contains(&action),
                "{action:?} must be reachable by SparseSolverPortfolio::select_action"
            );
        }

        // Continuous prior: distinct condition numbers in the same bucket produce distinct posteriors.
        let post_10 = p.posterior(10.0, nonsym);
        let post_1000 = p.posterior(1000.0, nonsym);
        assert!(post_1000[3] > post_10[3]);

        // Dirichlet update: repeated CG failures at kappa=100 move the posterior toward
        // IllConditioned and flip the selected action away from CG.
        let mut p_learned = SparseSolverPortfolio::new(RuntimeMode::Strict, 16);
        let before = p_learned.posterior(100.0, sym_pd);
        for _ in 0..8 {
            p_learned.record_outcome(
                100.0,
                sym_pd,
                SparseSolverAction::ConjugateGradient,
                AttemptOutcome::Inaccurate,
            );
        }
        let after = p_learned.posterior(100.0, sym_pd);
        assert!(after[3] > before[3]);
        let (action_after, _, _, _) = p_learned.select_action(100.0, sym_pd);
        assert_ne!(action_after, SparseSolverAction::ConjugateGradient);
    }

    #[test]
    fn test_opt_solver_portfolio_smooth_convex_selects_bfgs() {
        let p = OptSolverPortfolio::new(RuntimeMode::Strict, 16);
        let (action, _, _, _) = p.select_action(100.0, false, false);
        assert_eq!(action, OptSolverAction::BFGS);
    }

    #[test]
    fn test_opt_solver_portfolio_valley_selects_trust_region_newton_cg() {
        let p = OptSolverPortfolio::new(RuntimeMode::Strict, 16);
        let (action, _, _, _) = p.select_action(1e7, false, false);
        assert_eq!(action, OptSolverAction::TrustRegionNewtonCG);
    }

    #[test]
    fn test_opt_solver_portfolio_multimodal_selects_direct() {
        let p = OptSolverPortfolio::new(RuntimeMode::Strict, 16);
        let (action, _, _, _) = p.select_action(100.0, false, true);
        assert_eq!(action, OptSolverAction::DIRECT);
    }

    #[test]
    fn test_opt_solver_portfolio_noisy_selects_nelder_mead() {
        let p = OptSolverPortfolio::new(RuntimeMode::Strict, 16);
        let (action, _, _, _) = p.select_action(100.0, true, false);
        assert_eq!(action, OptSolverAction::NelderMead);
    }

    // br-szq1n.9: every action must be the argmin for SOME problem, or it is dead code.
    // L-BFGS-B is only reachable through the feasible set: its loss row is dominated
    // by BFGS whenever both are candidates.
    #[test]
    fn test_opt_solver_portfolio_every_action_is_reachable() {
        let p = OptSolverPortfolio::new(RuntimeMode::Strict, 16);
        let bounded = [OptSolverAction::LBFGSB, OptSolverAction::DIRECT];
        let mut reached: Vec<OptSolverAction> = Vec::new();
        for &(cond, noisy, multimodal) in &[
            (100.0, false, false),
            (1e7, false, false),
            (100.0, false, true),
            (100.0, true, false),
        ] {
            reached.push(p.select_action(cond, noisy, multimodal).0);
            reached.push(p.select_action_among(cond, noisy, multimodal, &bounded).0);
        }
        for action in OptSolverAction::ALL {
            assert!(reached.contains(&action), "{action:?} is never selected");
        }
    }

    #[test]
    fn test_opt_solver_portfolio_bounded_never_selects_an_infeasible_action() {
        let mut p = OptSolverPortfolio::new(RuntimeMode::Strict, 16);
        let bounded_only = [OptSolverAction::LBFGSB];
        for &(cond, noisy, multimodal) in &[
            (100.0, false, false),
            (1e7, false, false),
            (100.0, false, true),
            (100.0, true, false),
        ] {
            let (action, _, _, _) = p.select_action_among(cond, noisy, multimodal, &bounded_only);
            // Without the feasible set this is BFGS / TrustRegionNewtonCG / DIRECT /
            // NelderMead, none of which reads a non-box bound.
            assert_eq!(
                action,
                OptSolverAction::LBFGSB,
                "state ({cond}, {noisy}, {multimodal})"
            );
        }
        // Drift must not escape the feasible set either.
        for _ in 0..400 {
            p.observe_step_failure(1.0);
        }
        assert!(
            p.calibrator().should_fallback(),
            "drift fallback must be active"
        );
        let (action, _, _, _) = p.select_action_among(100.0, true, false, &bounded_only);
        assert_eq!(action, OptSolverAction::LBFGSB);
    }

    #[test]
    fn test_ode_solver_portfolio_nonstiff_selects_rk45() {
        let p = OdeSolverPortfolio::new(RuntimeMode::Strict, 16);
        let (action, _, _, _) = p.select_action(1.0, false);
        assert_eq!(action, OdeSolverAction::RK45);
    }

    #[test]
    fn test_ode_solver_portfolio_stiff_selects_bdf() {
        let p = OdeSolverPortfolio::new(RuntimeMode::Strict, 16);
        let (action, _, _, _) = p.select_action(1e6, false);
        assert_eq!(action, OdeSolverAction::BDF);
    }

    #[test]
    fn test_ode_solver_portfolio_algebraic_selects_radau() {
        let p = OdeSolverPortfolio::new(RuntimeMode::Strict, 16);
        let (action, _, _, _) = p.select_action(10.0, true);
        assert_eq!(action, OdeSolverAction::Radau);
    }

    #[test]
    fn test_ode_solver_portfolio_every_action_is_reachable_and_single_calibrator_feed() {
        let mut p = OdeSolverPortfolio::new(RuntimeMode::Strict, 16);
        let (a_dop853, _, _, _) = p.select_action(0.05, false);
        let (a_rk45, _, _, _) = p.select_action(1.0, false);
        let (a_rk23, _, _, _) = p.select_action(200.0, false);
        let (a_bdf, _, _, _) = p.select_action(1e6, false);
        let (a_radau, _, _, _) = p.select_action(10.0, true);

        assert_eq!(a_dop853, OdeSolverAction::DOP853);
        assert_eq!(a_rk45, OdeSolverAction::RK45);
        assert_eq!(a_rk23, OdeSolverAction::RK23);
        assert_eq!(a_bdf, OdeSolverAction::BDF);
        assert_eq!(a_radau, OdeSolverAction::Radau);

        // Also reachable via tolerance-driven selection on non-stiff problems:
        assert_eq!(
            p.select_action_with_tol(1.0, false, 1e-9).0,
            OdeSolverAction::DOP853
        );
        assert_eq!(
            p.select_action_with_tol(1.0, false, 1e-3).0,
            OdeSolverAction::RK45
        );
        assert_eq!(
            p.select_action_with_tol(1.0, false, 2e-2).0,
            OdeSolverAction::RK23
        );

        let reached = [a_dop853, a_rk45, a_rk23, a_bdf, a_radau];
        for action in OdeSolverAction::ALL {
            assert!(
                reached.contains(&action),
                "{action:?} must be reachable by OdeSolverPortfolio"
            );
        }

        // Single calibrator feed per call (observe_step + record_evidence):
        for i in 0..5 {
            p.observe_step(true);
            p.record_evidence(OdeSolverEvidenceEntry {
                component: "fsci-integrate",
                system_dim: 2,
                stiffness_ratio_estimate: 1.0,
                chosen_action: OdeSolverAction::RK45,
                posterior: vec![0.85, 0.1, 0.03, 0.02],
                expected_losses: vec![13.65, 15.5, 16.095, 31.03, 22.58],
                chosen_expected_loss: 13.65,
                fallback_active: false,
                step_rejections: Some(0),
            });
            assert_eq!(
                p.calibrator().total_predictions(),
                i + 1,
                "calibrator observation count must equal number of calls (1x, not 2x)"
            );
        }
    }

    #[test]
    fn test_stiffness_detector_online_trigger() {
        let mut detector = StiffnessDetector::new(1e3, 3);
        assert!(!detector.is_stiff(1.0));
        detector.observe_step(false);
        detector.observe_step(false);
        assert!(!detector.is_stiff(1.0));
        detector.observe_step(false);
        assert!(detector.is_stiff(1.0)); // 3 consecutive rejections
        detector.observe_step(true);
        assert!(!detector.is_stiff(1.0)); // reset by acceptance
    }

    #[test]
    fn test_hyper_solver_portfolio_near_zero_selects_direct_series() {
        let p = HyperSolverPortfolio::new(RuntimeMode::Strict, 16);
        let (action, _, _, _) = p.select_action(0.2, 1.0, false);
        assert_eq!(action, HyperBranchAction::DirectSeries);
    }

    #[test]
    fn test_hyper_solver_portfolio_transform_selects_kummer_or_pfaff() {
        let p = HyperSolverPortfolio::new(RuntimeMode::Strict, 16);
        let (action, _, _, _) = p.select_action(-3.0, 1.0, true);
        assert_eq!(action, HyperBranchAction::KummerTransform);

        let (action_pfaff, _, _, _) = p.select_action(-3.0, 1.0, false);
        assert_eq!(action_pfaff, HyperBranchAction::PfaffTransform);
    }

    #[test]
    fn test_hyper_solver_portfolio_50_benign_calls_never_latch_guarded_fallback() {
        let mut p = HyperSolverPortfolio::new(RuntimeMode::Strict, 64);
        for _ in 0..50 {
            let (action, posterior, losses, chosen) = p.select_action(0.2, 1.0, false);
            assert_eq!(action, HyperBranchAction::DirectSeries);
            assert!(!p.calibrator().should_fallback());
            p.record_evidence(HyperSolverEvidenceEntry {
                component: "fsci-special",
                function_kind: "hyp2f1",
                z_abs: 0.2,
                parameter_stability_margin: 1.0,
                chosen_action: action,
                posterior: posterior.to_vec(),
                expected_losses: losses.to_vec(),
                chosen_expected_loss: chosen,
                fallback_active: false,
                term_count: Some(500),
                estimated_truncation_error: Some(f64::EPSILON),
            });
        }
        assert!(!p.calibrator().should_fallback());
    }

    #[test]
    fn test_hyper_solver_portfolio_asymptotic_selects_asymptotic() {
        let p = HyperSolverPortfolio::new(RuntimeMode::Strict, 16);
        let (action, _, _, _) = p.select_action(25.0, 1.0, false);
        assert_eq!(action, HyperBranchAction::Asymptotic);
    }

    #[test]
    fn test_hyper_solver_portfolio_boundary_near_pole_selects_guarded_fallback() {
        let p = HyperSolverPortfolio::new(RuntimeMode::Strict, 16);
        let (action, _, _, _) = p.select_action(1.0, 1.0, false);
        assert_eq!(action, HyperBranchAction::GuardedFallback);

        let (action_pole, _, _, _) = p.select_action(0.5, 1e-15, false);
        assert_eq!(action_pole, HyperBranchAction::GuardedFallback);
    }

    /// frankenscipy-7tb8d.3: hand-computed Conformal Risk Control threshold-selection unit test
    /// at `n = 19, alpha = 0.1`.
    ///
    /// With `n = 19`, `B = 1`, and `alpha = 0.10`:
    /// `(19/20) * Rhat_19(t) + 1/20 <= 0.10  <=>  sum_{i=1}^{19} L_i(t) <= 1`.
    /// Suppose 3 of the 19 problems fail under `DirectLU` (and succeed under `PivotedQR`) at
    /// `rcond` values `1e-12`, `1e-8`, and `1e-4`:
    /// - At `t = 0`: 3 unrecovered exceedances -> `(3 + 1) / 20 = 0.20 > 0.10`.
    /// - At `t = 1e-12`: 2 unrecovered exceedances (`1e-8`, `1e-4`) -> `(2 + 1) / 20 = 0.15 > 0.10`.
    /// - At `t = 1e-8`: 1 unrecovered exceedance (`1e-4`) -> `(1 + 1) / 20 = 0.10 <= 0.10`.
    /// Hence `t_hat = 1e-8` and `Rhat_19(t_hat) = 1 / 19` exactly.
    #[test]
    fn crc_threshold_selection_matches_hand_computed_n19_alpha01() {
        let mut samples = Vec::with_capacity(19);
        for k in 1..=19 {
            let rcond = 10f64.powi(-k);
            let lu_exceeded = matches!(k, 4 | 8 | 12);
            samples.push(CrcCalibrationSample {
                rcond,
                unescalated_exceeded: lu_exceeded,
                escalated_exceeded: false,
            });
        }
        assert_eq!(samples.len(), 19);
        assert_eq!(ConformalRiskControl::verify_monotonicity(&samples), Ok(()));

        let crc = ConformalRiskControl::calibrate(&samples, 0.10, 0x7b8d_0003);
        assert_eq!(crc.n_calibration(), 19);
        assert_eq!(crc.alpha().to_bits(), 0.10_f64.to_bits());
        assert_eq!(crc.t_hat().to_bits(), 1e-8_f64.to_bits());
        assert_eq!(
            crc.empirical_risk_at_t_hat().to_bits(),
            (1.0_f64 / 19.0_f64).to_bits()
        );
        assert!((ConformalRiskControl::crc_objective(&samples, 0.0) - 0.20).abs() < 1e-15);
        assert!((ConformalRiskControl::crc_objective(&samples, 1e-12) - 0.15).abs() < 1e-15);
        assert!((ConformalRiskControl::crc_objective(&samples, 1e-8) - 0.10).abs() < 1e-15);
        assert!((ConformalRiskControl::crc_objective(&samples, 1e-4) - 0.05).abs() < 1e-15);

        // Wire into SolverPortfolio and verify select_action escalates at rcond <= t_hat.
        let mut portfolio = SolverPortfolio::new(RuntimeMode::Strict, 16);
        portfolio.set_crc(crc);
        assert_eq!(portfolio.alpha().to_bits(), 0.10_f64.to_bits());
        assert_eq!(
            portfolio.select_action(1e-8, None).0,
            SolverAction::PivotedQR
        );
        assert_eq!(
            portfolio.select_action(1e-12, None).0,
            SolverAction::PivotedQR
        );
        assert_eq!(
            portfolio.select_action(1e-7, None).0,
            SolverAction::DirectLU
        );

        // Lowering alpha on the default portfolio to 0.005 (< 2/322 = 0.00621) escalates past
        // Wilkinson n=128 (rcond = 0.0078125).
        let mut default_p = SolverPortfolio::new(RuntimeMode::Strict, 16);
        assert_eq!(default_p.alpha(), 0.05);
        assert_eq!(default_p.crc().t_hat(), 0.0);
        default_p.set_alpha(0.005);
        assert_eq!(default_p.alpha(), 0.005);
        assert_eq!(default_p.crc().t_hat(), 0.0078125);
        assert_eq!(
            default_p.select_action(0.0078125, None).0,
            SolverAction::PivotedQR
        );
        assert_eq!(
            default_p.select_action(0.02, None).0,
            SolverAction::DirectLU
        );
    }

    /// frankenscipy-7tb8d.3: verify the Conformal Risk Control precondition that `L_i(t)` is
    /// monotone non-increasing in `t` over every matrix in the committed calibration corpus, and
    /// verify that a synthetic non-monotone sample is rejected.
    #[test]
    fn crc_loss_is_monotone_non_increasing_in_t_over_calibration_corpus() {
        let report: serde_json::Value = serde_json::from_str(include_str!(
            "../../../artifacts/casp-calibration-solver.json"
        ))
        .expect("calibration report parses");
        let cells = report["cells"].as_array().expect("cells");
        let mut all_samples = Vec::with_capacity(cells.len());
        for cell in cells {
            let rcond = cell["rcond_estimate"].as_f64().unwrap_or(0.0);
            let outcomes = cell["outcomes"].as_array().expect("outcomes");
            let lu_failed = outcomes
                .iter()
                .find(|o| o["action"].as_str() == Some("DirectLU"))
                .is_some_and(|o| o["failed"].as_bool().unwrap_or(true));
            let qr_failed = outcomes
                .iter()
                .find(|o| o["action"].as_str() == Some("PivotedQR"))
                .is_some_and(|o| o["failed"].as_bool().unwrap_or(true));
            all_samples.push(CrcCalibrationSample {
                rcond,
                unescalated_exceeded: lu_failed,
                escalated_exceeded: qr_failed,
            });
        }
        assert_eq!(all_samples.len(), 324);
        assert_eq!(
            ConformalRiskControl::verify_monotonicity(&all_samples),
            Ok(()),
            "every cell in the W4.2 corpus must satisfy L_i(t2) <= L_i(t1) for t1 <= t2"
        );

        // Negative arm: a sample where escalating to QR fails while LU succeeded violates monotonicity.
        let mut broken = all_samples;
        broken[42] = CrcCalibrationSample {
            rcond: 1e-6,
            unescalated_exceeded: false,
            escalated_exceeded: true,
        };
        assert_eq!(ConformalRiskControl::verify_monotonicity(&broken), Err(42));
    }

    /// frankenscipy-7tb8d.3: held-out 50/50 split test by seed with Clopper-Pearson 95% upper
    /// bound, plus the shifted-corpus (`kappa` bins `1e12..1e16` only) negative-case test.
    #[test]
    fn crc_held_out_50_50_split_and_shifted_corpus_negative_case() {
        let report: serde_json::Value = serde_json::from_str(include_str!(
            "../../../artifacts/casp-calibration-solver.json"
        ))
        .expect("calibration report parses");
        let cells = report["cells"].as_array().expect("cells");

        #[derive(Clone, Debug)]
        struct CorpusEntry {
            family: String,
            n: usize,
            rcond: f64,
            lu_bwd: f64,
            qr_bwd: f64,
            lu_fwd: f64,
            lu_failed: bool,
            qr_failed: bool,
            svd_failed: bool,
        }

        let mut solvable = Vec::new();
        for cell in cells {
            let outcomes = cell["outcomes"].as_array().expect("outcomes");
            if !outcomes
                .iter()
                .any(|o| !o["failed"].as_bool().unwrap_or(true))
            {
                continue;
            }
            let get = |action: &str| {
                outcomes
                    .iter()
                    .find(|o| o["action"].as_str() == Some(action))
                    .expect("general action present")
            };
            let lu = get("DirectLU");
            let qr = get("PivotedQR");
            let svd = get("SVDFallback");
            solvable.push(CorpusEntry {
                family: cell["family"].as_str().unwrap_or("").to_string(),
                n: cell["n"].as_u64().unwrap_or(0) as usize,
                rcond: cell["rcond_estimate"].as_f64().unwrap_or(0.0),
                lu_bwd: lu["backward_error"].as_f64().unwrap_or(f64::INFINITY),
                qr_bwd: qr["backward_error"].as_f64().unwrap_or(f64::INFINITY),
                lu_fwd: lu["forward_error"].as_f64().unwrap_or(f64::INFINITY),
                lu_failed: lu["failed"].as_bool().unwrap_or(true),
                qr_failed: qr["failed"].as_bool().unwrap_or(true),
                svd_failed: svd["failed"].as_bool().unwrap_or(true),
            });
        }
        assert_eq!(solvable.len(), 321);

        let splitmix64 = |state: &mut u64| -> u64 {
            *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = *state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let split_50_50 =
            |items: &[CorpusEntry], seed: u64| -> (Vec<CorpusEntry>, Vec<CorpusEntry>) {
                let mut indices: Vec<usize> = (0..items.len()).collect();
                let mut rng = seed;
                for i in (1..indices.len()).rev() {
                    let j = (splitmix64(&mut rng) % (i as u64 + 1)) as usize;
                    indices.swap(i, j);
                }
                let half = items.len() / 2;
                let cal = indices[..half].iter().map(|&i| items[i].clone()).collect();
                let held_out = indices[half..].iter().map(|&i| items[i].clone()).collect();
                (cal, held_out)
            };

        let seed = calibrated_losses::SOLVER_CRC_SEED;
        let (cal_entries, held_out_entries) = split_50_50(&solvable, seed);
        let cal_samples: Vec<CrcCalibrationSample> = cal_entries
            .iter()
            .map(|e| CrcCalibrationSample {
                rcond: e.rcond,
                unescalated_exceeded: e.lu_failed,
                escalated_exceeded: e.qr_failed,
            })
            .collect();
        let held_out_samples: Vec<CrcCalibrationSample> = held_out_entries
            .iter()
            .map(|e| CrcCalibrationSample {
                rcond: e.rcond,
                unescalated_exceeded: e.lu_failed,
                escalated_exceeded: e.qr_failed,
            })
            .collect();

        let alpha = 0.05;
        let crc = ConformalRiskControl::calibrate(&cal_samples, alpha, seed);
        let held_out_exceedances = held_out_samples
            .iter()
            .filter(|s| s.loss_at(crc.t_hat()) > 0.0)
            .count();
        let held_out_rate = held_out_exceedances as f64 / held_out_samples.len() as f64;
        let cp_upper_95 =
            clopper_pearson_upper_bound(held_out_exceedances, held_out_samples.len(), 0.05);
        let cp_slack = (cp_upper_95 - held_out_rate).max(0.0);

        let mut worst_held_out: Vec<(f64, &CorpusEntry)> = held_out_entries
            .iter()
            .map(|e| {
                let bwd = if crc.should_escalate(e.rcond) {
                    e.qr_bwd
                } else {
                    e.lu_bwd
                };
                (bwd, e)
            })
            .collect();
        worst_held_out.sort_by(|a, b| b.0.total_cmp(&a.0));

        println!(
            "CRC held-out 50/50 split: n={}, alpha={:.4}, t_hat={:.6e}, Rhat_n(t_hat)={:.6}, held_out_exceedances={}/{}, held_out_rate={:.6}, cp_95_upper_bound={:.6}",
            crc.n_calibration(),
            crc.alpha(),
            crc.t_hat(),
            crc.empirical_risk_at_t_hat(),
            held_out_exceedances,
            held_out_samples.len(),
            held_out_rate,
            cp_upper_95
        );
        for (rank, (bwd, entry)) in worst_held_out.iter().take(10).enumerate() {
            println!(
                "  worst_held_out[{}]: family={} n={} rcond={:.3e} bwd={:.3e} fwd={:.3e}",
                rank + 1,
                entry.family,
                entry.n,
                entry.rcond,
                bwd,
                entry.lu_fwd
            );
        }

        assert!(
            held_out_rate <= alpha + cp_slack,
            "held_out_rate {held_out_rate} > alpha {alpha} + cp_slack {cp_slack}"
        );
        assert!(
            cp_upper_95 <= alpha,
            "95% Clopper-Pearson upper bound {cp_upper_95} exceeds alpha {alpha}"
        );

        // Negative case on deliberately shifted corpus (kappa bins 1e12..1e16 only):
        // Arm A (backward-error contract omega_i > 1e-8): the pre-CRC fixed-threshold rule
        // (which escalated rcond <= 1e-8 to SVDFallback) exceeds alpha on the held-out half of
        // the shifted corpus, whereas CRC recalibrated on the calibration half of the shifted
        // corpus has 0.0 held-out exceedance (<= alpha).
        // Arm B (condition-guard escalation threshold t on rcond vs fixed NearSingular boundary 1e-16
        // for forward-error control > 1e-4): the fixed threshold t_fixed = 1e-16 exceeds alpha
        // on the held-out half of the shifted corpus, whereas CRC recalibrated on the shifted
        // calibration half selects t_hat = 1.910e-15 and keeps held-out exceedance = 0.0208 <= alpha.
        let shifted: Vec<CorpusEntry> = solvable
            .iter()
            .filter(|e| {
                ["1e12", "1e13", "1e14", "1e15", "1e16", "1e-12", "1e-16"]
                    .iter()
                    .any(|tag| e.family.contains(tag))
            })
            .cloned()
            .collect();
        assert_eq!(shifted.len(), 96);
        let (shifted_cal, shifted_held_out) = split_50_50(&shifted, seed);
        assert_eq!(shifted_cal.len(), 48);
        assert_eq!(shifted_held_out.len(), 48);

        let fixed_svd_rule_exceedances = shifted_held_out
            .iter()
            .filter(|e| e.rcond <= 1e-8 && e.svd_failed)
            .count();
        let fixed_svd_rule_rate = fixed_svd_rule_exceedances as f64 / shifted_held_out.len() as f64;
        let shifted_bwd_cal: Vec<CrcCalibrationSample> = shifted_cal
            .iter()
            .map(|e| CrcCalibrationSample {
                rcond: e.rcond,
                unescalated_exceeded: e.lu_failed,
                escalated_exceeded: e.qr_failed,
            })
            .collect();
        let shifted_bwd_crc = ConformalRiskControl::calibrate(&shifted_bwd_cal, alpha, seed);
        let shifted_bwd_crc_rate = shifted_held_out
            .iter()
            .filter(|e| {
                let s = CrcCalibrationSample {
                    rcond: e.rcond,
                    unescalated_exceeded: e.lu_failed,
                    escalated_exceeded: e.qr_failed,
                };
                s.loss_at(shifted_bwd_crc.t_hat()) > 0.0
            })
            .count() as f64
            / shifted_held_out.len() as f64;

        let fixed_rcond_threshold = 1e-16_f64;
        let fwd_tol = 1e-4_f64;
        let shifted_fwd_cal: Vec<CrcCalibrationSample> = shifted_cal
            .iter()
            .map(|e| CrcCalibrationSample {
                rcond: e.rcond,
                unescalated_exceeded: e.lu_fwd > fwd_tol,
                escalated_exceeded: false,
            })
            .collect();
        let shifted_fwd_held_out: Vec<CrcCalibrationSample> = shifted_held_out
            .iter()
            .map(|e| CrcCalibrationSample {
                rcond: e.rcond,
                unescalated_exceeded: e.lu_fwd > fwd_tol,
                escalated_exceeded: false,
            })
            .collect();
        let fixed_rcond_held_out_rate =
            ConformalRiskControl::empirical_risk(&shifted_fwd_held_out, fixed_rcond_threshold);
        let recalibrated_fwd_crc = ConformalRiskControl::calibrate(&shifted_fwd_cal, alpha, seed);
        let recalibrated_fwd_held_out_rate = ConformalRiskControl::empirical_risk(
            &shifted_fwd_held_out,
            recalibrated_fwd_crc.t_hat(),
        );

        println!(
            "CRC shifted corpus (kappa 1e12..1e16, n_cal={}, n_held_out={}): \
             bwd_fixed_rule_rate={:.4} vs bwd_crc_rate={:.4} (t_hat={:.3e}); \
             fwd_fixed_1e-16_rate={:.4} vs fwd_crc_rate={:.4} (t_hat={:.3e})",
            shifted_cal.len(),
            shifted_held_out.len(),
            fixed_svd_rule_rate,
            shifted_bwd_crc_rate,
            shifted_bwd_crc.t_hat(),
            fixed_rcond_held_out_rate,
            recalibrated_fwd_held_out_rate,
            recalibrated_fwd_crc.t_hat()
        );

        assert!(
            fixed_svd_rule_rate > alpha,
            "fixed SVD escalation rule must exceed alpha on shifted corpus: {fixed_svd_rule_rate}"
        );
        assert!(
            shifted_bwd_crc_rate <= alpha,
            "CRC recalibrated backward-error rate must be <= alpha: {shifted_bwd_crc_rate}"
        );
        assert!(
            fixed_rcond_held_out_rate > alpha,
            "fixed rcond=1e-16 rule must exceed alpha on shifted corpus: {fixed_rcond_held_out_rate}"
        );
        assert!(
            recalibrated_fwd_held_out_rate <= alpha,
            "CRC recalibrated on shifted corpus must keep held-out exceedance <= alpha: {recalibrated_fwd_held_out_rate}"
        );
    }
}
