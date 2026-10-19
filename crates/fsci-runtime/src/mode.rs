#![forbid(unsafe_code)]

//! Runtime mode definitions for Strict (SciPy-compatible) and Hardened operation.

use serde::{Deserialize, Serialize};

/// Operational mode governing compatibility/safety trade-offs.
///
/// - **Strict**: Match SciPy behavior as closely as possible; clamping semantics
///   replicate SciPy exactly.
/// - **Hardened**: Extra safety layer beyond SciPy; adds finite-check rejection
///   and tighter validation. Higher FailClosed+Compatible cost (55 vs 40) makes
///   it slightly less eager to reject good inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimeMode {
    Strict,
    Hardened,
}

/// Maximum dimension allowed in Hardened mode for memory/compute bounded defense (§0.3).
pub const HARDENED_MAX_DIM: usize = 10_000;

/// Opt-in Hardened-mode bounded recovery policy (`frankenscipy-3cu8u.3`).
///
/// In [`RuntimeMode::Strict`], this policy is ignored and recovery paths are unreachable.
/// In [`RuntimeMode::Hardened`], the default (`Reject`) fails closed on non-finite or
/// near-singular inputs; opt-in variants perform bounded recovery and emit an
/// `AuditAction::BoundedRecovery` event carrying the original input fingerprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum RecoveryPolicy {
    /// Default Hardened behavior: fail closed on non-finite or near-singular inputs.
    #[default]
    Reject,
    /// Replace non-finite (`NaN` / `±Inf`) entries with neutral `0.0`, recording count and
    /// positions digest in a `BoundedRecovery` event (`recovery_action: "nonfinite_projection"`).
    ProjectNonFinite,
    /// Regularize near-singular dense systems (`rcond_est < 1e-14`) via SVD Tikhonov filter
    /// factors `σ_i / (σ_i² + λ²)` with `λ = σ_max · √ε · n`, recording `λ`, `rcond_est`,
    /// relative residual `‖b - A·x‖ / ‖b‖`, and filtered mode count in a `BoundedRecovery`
    /// event (`recovery_action: "tikhonov_regularization"`).
    RegularizeNearSingular,
    /// Apply both non-finite projection and Tikhonov regularization when needed.
    ProjectAndRegularize,
}

#[allow(non_upper_case_globals)]
impl RecoveryPolicy {
    /// Alias for [`RecoveryPolicy::ProjectNonFinite`].
    pub const Project: Self = Self::ProjectNonFinite;
    /// Alias for [`RecoveryPolicy::RegularizeNearSingular`].
    pub const Regularize: Self = Self::RegularizeNearSingular;

    /// Whether this policy permits non-finite projection in Hardened mode.
    #[must_use]
    pub const fn allows_projection(self) -> bool {
        matches!(self, Self::ProjectNonFinite | Self::ProjectAndRegularize)
    }

    /// Whether this policy permits Tikhonov regularization in Hardened mode.
    #[must_use]
    pub const fn allows_regularization(self) -> bool {
        matches!(
            self,
            Self::RegularizeNearSingular | Self::ProjectAndRegularize
        )
    }
}
