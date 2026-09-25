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
//! | `scipy_incumbent` | [`ScipyIncumbent`]: the one live-SciPy oracle resolver |
//! | `booking_claim` | [`BookingClaim`]: verifies the fleet measurement booking a timed row cites |

pub mod booking_claim;
pub mod eprocess;
pub mod evidence;
pub mod mode;
pub mod policy;
pub mod scipy_incumbent;
pub mod signals;
pub mod supervision;

// ── Re-exports: preserve the flat public API ────────────────────────
pub use booking_claim::{BookingClaim, ClaimRejection, FleetBooking};
pub use eprocess::{EProcessConfig, EProcessMonitor, EProcessStatus};
pub use evidence::{
    AlienArtifactDecision, AuditAction, AuditEvent, AuditLedger, AuditScope, DecisionEvidenceEntry,
    EvidenceValue, Fingerprinter, PolicyEvidenceLedger, SharedAuditLedger, SyncSharedAuditLedger,
    audit_finish, audit_recover, audit_reject,
};
pub use mode::{HARDENED_MAX_DIM, RuntimeMode};
pub use policy::{PolicyAction, PolicyController, PolicyDecision, RiskState, decision_loss_matrix};
pub use signals::{DecisionSignals, SignalSequence};
pub use supervision::{PolicySupervisor, SupervisionConfig, SupervisionEvent, SupervisionStatus};

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
/// Loss matrix (6 actions × 4 states):
///
/// | Action \ State     | WellCond | Moderate | IllCond | NearSingular |
/// |--------------------|----------|----------|---------|--------------|
/// | DirectLU           |        1 |        5 |      40 |          120 |
/// | PivotedQR          |        3 |        1 |       8 |           45 |
/// | SVDFallback        |       15 |       10 |       1 |            1 |
/// | DiagonalFastPath   |        0 |        0 |       0 |          100 |
/// | TriangularFastPath |        0 |        0 |       0 |          100 |
/// | SymmetricFastPath  |        0 |        0 |       0 |          100 |
///
/// Decision: a* = argmin_a Σ_s L(a,s) × P(s|evidence)
#[derive(Debug, Clone)]
pub struct SolverPortfolio {
    mode: RuntimeMode,
    loss_matrix: [[f64; 4]; 6],
    evidence: VecDeque<SolverEvidenceEntry>,
    evidence_capacity: usize,
    calibrator: ConformalCalibrator,
    /// Dirichlet outcome counts over the four condition states, one row per decade of rcond
    /// (see [`rcond_decade`]). Filled by [`SolverPortfolio::record_outcome`].
    outcome_counts: [[f64; 4]; RCOND_DECADES],
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

/// The decade row an rcond value's outcomes are counted in.
fn rcond_decade(rcond: f64) -> usize {
    if !rcond.is_finite() || rcond <= 0.0 {
        return RCOND_DECADES - 1;
    }
    let decade = (-rcond.log10()).floor();
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
        Self {
            mode,
            loss_matrix: Self::default_loss_matrix(),
            evidence: VecDeque::with_capacity(evidence_capacity),
            evidence_capacity,
            calibrator: ConformalCalibrator::new(0.05, 200),
            outcome_counts: [[0.0; 4]; RCOND_DECADES],
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
        [
            [1.0, 5.0, 40.0, 120.0], // DirectLU
            [3.0, 1.0, 8.0, 45.0],   // PivotedQR
            [15.0, 10.0, 1.0, 1.0],  // SVDFallback
            [0.0, 0.0, 0.0, 100.0],  // DiagonalFastPath
            [0.0, 0.0, 0.0, 100.0],  // TriangularFastPath
            [0.0, 0.0, 0.0, 100.0],  // SymmetricFastPath
        ]
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
        let losses = self.compute_expected_losses(posterior);

        // If conformal calibrator detects drift, override to SVDFallback
        if self.calibrator.should_fallback() && !excluded.contains(&SolverAction::SVDFallback) {
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

        best.map(|(idx, loss)| (SolverAction::ALL[idx], posterior, losses, loss))
    }

    /// Record solver evidence for audit trail and calibration.
    pub fn record_evidence(&mut self, entry: SolverEvidenceEntry) {
        if let Some(err) = entry.backward_error {
            self.calibrator.observe(err);
        }
        if self.evidence.len() >= self.evidence_capacity {
            let _ = self.evidence.pop_front();
        }
        self.evidence.push_back(entry);
    }

    /// Update conformal calibrator with observed backward error.
    pub fn observe_backward_error(&mut self, backward_error: f64) {
        self.calibrator.observe(backward_error);
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

    fn compute_expected_losses(&self, posterior: [f64; 4]) -> [f64; 6] {
        let mut losses = [0.0; 6];
        for (action_idx, row) in self.loss_matrix.iter().enumerate() {
            losses[action_idx] = row.iter().zip(posterior.iter()).map(|(l, p)| l * p).sum();
        }
        losses
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

        let log_r = rcond.log10();
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

/// Sparse solver selection portfolio engine.
///
/// Loss matrix (6 actions × 4 states):
///
/// | Action \ State       | SPD | GenWell | Indef | IllCond |
/// |----------------------|-----|---------|-------|---------|
/// | ConjugateGradient    |   1 |     150 |   200 |     250 |
/// | MinRes               |   3 |     100 |     2 |     200 |
/// | BiCGSTAB             |   6 |       1 |    45 |     180 |
/// | GMRES                |  10 |       4 |     6 |      60 |
/// | QMR                  |  12 |       5 |     8 |      70 |
/// | SuperLU              |  40 |      30 |    25 |       2 |
#[derive(Debug, Clone)]
pub struct SparseSolverPortfolio {
    mode: RuntimeMode,
    loss_matrix: [[f64; 4]; 6],
    evidence: VecDeque<SparseSolverEvidenceEntry>,
    evidence_capacity: usize,
    calibrator: ConformalCalibrator,
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
        }
    }

    #[must_use]
    pub const fn default_loss_matrix() -> [[f64; 4]; 6] {
        [
            // SPD,   GenWell, Indef,  IllCond
            [1.0, 40.0, 50.0, 100.0], // ConjugateGradient
            [3.0, 30.0, 2.0, 80.0],   // MinRes
            [5.0, 1.0, 15.0, 60.0],   // BiCGSTAB
            [8.0, 4.0, 5.0, 30.0],    // GMRES
            [10.0, 5.0, 6.0, 35.0],   // QMR
            [25.0, 20.0, 18.0, 2.0],  // SuperLU
        ]
    }

    /// Select optimal sparse solver via expected-loss minimization.
    pub fn select_action(
        &self,
        cond_estimate: f64,
        structure: Option<SparseStructuralEvidence>,
    ) -> (SparseSolverAction, [f64; 4], [f64; 6], f64) {
        let posterior = Self::condition_posterior(cond_estimate, structure);

        // If conformal calibrator triggers drift, fallback to SuperLU direct solver
        if self.calibrator.should_fallback() {
            let losses = self.compute_expected_losses(posterior);
            return (
                SparseSolverAction::SuperLU,
                posterior,
                losses,
                losses[SparseSolverAction::SuperLU.index()],
            );
        }

        let losses = self.compute_expected_losses(posterior);
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

    fn compute_expected_losses(&self, posterior: [f64; 4]) -> [f64; 6] {
        let mut losses = [0.0; 6];
        for (action_idx, row) in self.loss_matrix.iter().enumerate() {
            losses[action_idx] = row.iter().zip(posterior.iter()).map(|(l, p)| l * p).sum();
        }
        losses
    }

    fn condition_posterior(
        cond_estimate: f64,
        structure: Option<SparseStructuralEvidence>,
    ) -> [f64; 4] {
        if !cond_estimate.is_finite() || cond_estimate <= 0.0 {
            return [0.0, 0.0, 0.0, 1.0];
        }

        let is_sym = structure.is_some_and(|s| s.is_symmetric);
        let pd_hint = structure.and_then(|s| s.is_positive_definite_hint);

        if cond_estimate >= 1e8 {
            return [0.0, 0.0, 0.0, 1.0]; // IllConditioned
        }

        if is_sym {
            if pd_hint == Some(true) {
                if cond_estimate < 1e4 {
                    [0.98, 0.0, 0.0, 0.02]
                } else {
                    [0.78, 0.0, 0.0, 0.22]
                }
            } else if pd_hint == Some(false) {
                if cond_estimate < 1e4 {
                    [0.0, 0.0, 0.97, 0.03]
                } else {
                    [0.0, 0.0, 0.70, 0.30]
                }
            } else if cond_estimate < 1e4 {
                [0.70, 0.0, 0.28, 0.02]
            } else {
                [0.45, 0.0, 0.35, 0.20]
            }
        } else if cond_estimate < 1e4 {
            [0.0, 0.96, 0.02, 0.02]
        } else {
            [0.0, 0.60, 0.15, 0.25]
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
/// Loss matrix (5 actions × 4 states):
///
/// | Action \ State          | SmoothConvex | Valley | MultiModal | NoisyNonSmooth |
/// |-------------------------|--------------|--------|------------|----------------|
/// | BFGS                    |            1 |     15 |        120 |            150 |
/// | LBFGSB                  |            2 |     20 |        120 |            150 |
/// | NelderMead              |           50 |     60 |         80 |              2 |
/// | DIRECT                  |           80 |     40 |          2 |             10 |
/// | TrustRegionNewtonCG     |            8 |      2 |         70 |            180 |
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
        Self {
            mode,
            loss_matrix: Self::default_loss_matrix(),
            evidence: VecDeque::with_capacity(evidence_capacity),
            evidence_capacity,
            calibrator: ConformalCalibrator::new(0.05, 200),
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

/// Online stiffness detection for adaptive step and solver control.
#[derive(Debug, Clone)]
pub struct StiffnessDetector {
    stiffness_threshold: f64,
    consecutive_rejections: usize,
    rejection_threshold: usize,
    rejection_history: VecDeque<bool>,
    history_capacity: usize,
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

/// ODE solver selection portfolio engine.
///
/// Loss matrix (5 actions × 4 states):
///
/// | Action \ State  | NonStiff | MildlyStiff | Stiff | Algebraic |
/// |-----------------|----------|-------------|-------|-----------|
/// | RK45            |        1 |          25 |   200 |       300 |
/// | RK23            |        5 |          40 |   250 |       300 |
/// | DOP853          |        2 |          35 |   250 |       150 |
/// | Radau           |       35 |           3 |     2 |         1 |
/// | BDF             |       25 |           2 |     1 |        15 |
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
        Self {
            mode,
            loss_matrix: Self::default_loss_matrix(),
            evidence: VecDeque::with_capacity(evidence_capacity),
            evidence_capacity,
            calibrator: ConformalCalibrator::new(0.05, 200),
            detector: StiffnessDetector::default(),
        }
    }

    #[must_use]
    pub const fn default_loss_matrix() -> [[f64; 4]; 5] {
        [
            // NonStiff, MildlyStiff, Stiff,  Algebraic
            [1.0, 25.0, 200.0, 300.0], // RK45
            [5.0, 40.0, 250.0, 300.0], // RK23
            [2.0, 35.0, 250.0, 150.0], // DOP853
            [35.0, 3.0, 2.0, 1.0],     // Radau
            [25.0, 2.0, 1.0, 15.0],    // BDF
        ]
    }

    pub fn select_action(
        &self,
        stiffness_ratio_estimate: f64,
        is_algebraic_dae: bool,
    ) -> (OdeSolverAction, [f64; 4], [f64; 5], f64) {
        let online_stiff = self.detector.is_stiff(stiffness_ratio_estimate);
        let posterior =
            Self::stiffness_posterior(stiffness_ratio_estimate, is_algebraic_dae, online_stiff);

        // If conformal calibrator triggers drift, fallback to Radau
        if self.calibrator.should_fallback() {
            let losses = self.compute_expected_losses(posterior);
            return (
                OdeSolverAction::Radau,
                posterior,
                losses,
                losses[OdeSolverAction::Radau.index()],
            );
        }

        let losses = self.compute_expected_losses(posterior);
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

        (OdeSolverAction::ALL[best_idx], posterior, losses, best_loss)
    }

    pub fn record_evidence(&mut self, entry: OdeSolverEvidenceEntry) {
        if let Some(rejections) = entry.step_rejections {
            self.calibrator.observe(rejections as f64);
        }
        if self.evidence.len() >= self.evidence_capacity {
            let _ = self.evidence.pop_front();
        }
        self.evidence.push_back(entry);
    }

    pub fn observe_step(&mut self, accepted: bool) {
        self.detector.observe_step(accepted);
        if !accepted {
            self.calibrator.observe(1.0);
        } else {
            self.calibrator.observe(0.0);
        }
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
            return [0.0, 0.05, 0.1, 0.85]; // AlgebraicConstrained
        }
        if online_stiff || stiffness_ratio_estimate >= 1e5 {
            return [0.0, 0.1, 0.85, 0.05]; // Stiff
        }
        if stiffness_ratio_estimate >= 1e2 {
            return [0.1, 0.7, 0.15, 0.05]; // MildlyStiff
        }
        if stiffness_ratio_estimate <= 0.0 || !stiffness_ratio_estimate.is_finite() {
            return [0.0, 0.1, 0.8, 0.1];
        }
        [0.85, 0.1, 0.03, 0.02] // NonStiffSmooth
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
}

/// Hypergeometric branch selection portfolio engine.
///
/// Loss matrix (6 actions × 4 states):
///
/// | Action \ State       | NearZero | Transform | Asymptotic | BoundaryNearPole |
/// |----------------------|----------|-----------|------------|------------------|
/// | DirectSeries         |        1 |        40 |        150 |              200 |
/// | KummerTransform      |       20 |         2 |        100 |              200 |
/// | PfaffTransform       |       25 |         3 |         90 |              200 |
/// | Asymptotic           |       80 |        50 |          2 |              100 |
/// | ContinuedFraction    |       30 |        10 |         20 |                5 |
/// | GuardedFallback      |      100 |        80 |         70 |                1 |
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

        // If conformal calibrator triggers drift, fallback to GuardedFallback
        if self.calibrator.should_fallback() {
            let losses = self.compute_expected_losses(posterior);
            return (
                HyperBranchAction::GuardedFallback,
                posterior,
                losses,
                losses[HyperBranchAction::GuardedFallback.index()],
            );
        }

        let losses = self.compute_expected_losses(posterior);
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

        (
            HyperBranchAction::ALL[best_idx],
            posterior,
            losses,
            best_loss,
        )
    }

    pub fn record_evidence(&mut self, entry: HyperSolverEvidenceEntry) {
        if let Some(terms) = entry.term_count {
            self.calibrator.observe(terms as f64);
        }
        if self.evidence.len() >= self.evidence_capacity {
            let _ = self.evidence.pop_front();
        }
        self.evidence.push_back(entry);
    }

    pub fn observe_relative_error(&mut self, error: f64) {
        self.calibrator.observe(error);
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

        let log_r = rcond.max(1e-25).log10();
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
    // mass to the ill-conditioned states and change the decision. Before this bead the
    // posterior was a pure function of rcond, so this failed.
    #[test]
    fn recorded_outcomes_move_the_posterior_and_the_decision() {
        let mut portfolio = SolverPortfolio::new(RuntimeMode::Strict, 8);
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
        ok.record_outcome(rcond, SolverAction::DirectLU, AttemptOutcome::Ok);
        assert_eq!(ok.select_action(rcond, None).0, SolverAction::DirectLU);
        for _ in 1..50 {
            ok.record_outcome(rcond, SolverAction::DirectLU, AttemptOutcome::Ok);
        }
        assert_eq!(ok.select_action(rcond, None).0, SolverAction::DirectLU);
        assert!(ok.posterior(rcond)[0] >= p0[0] - 0.05);
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

    #[test]
    fn casp_selects_qr_for_moderate() {
        let portfolio = SolverPortfolio::new(RuntimeMode::Strict, 64);
        let (action, _, _, _) = portfolio.select_action(1e-6, None);
        assert_eq!(action, SolverAction::PivotedQR);
    }

    #[test]
    fn casp_selects_svd_for_ill_conditioned() {
        let portfolio = SolverPortfolio::new(RuntimeMode::Strict, 64);
        let (action, _, _, _) = portfolio.select_action(1e-12, None);
        assert_eq!(action, SolverAction::SVDFallback);
    }

    #[test]
    fn casp_selects_svd_for_near_singular() {
        let portfolio = SolverPortfolio::new(RuntimeMode::Strict, 64);
        let (action, _, _, _) = portfolio.select_action(1e-18, None);
        assert_eq!(action, SolverAction::SVDFallback);
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
    // evidence, and loses to SVD where the posterior is all near-singular.
    #[test]
    fn casp_offers_the_symmetric_path_only_for_symmetric_evidence() {
        let portfolio = SolverPortfolio::new(RuntimeMode::Strict, 64);
        let symmetric = Some(StructuralEvidence::Symmetric);
        for rcond in [1e-2, 1e-6, 1e-11] {
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
        assert_eq!(
            portfolio.select_action(1e-16, symmetric).0,
            SolverAction::SVDFallback
        );
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
}
