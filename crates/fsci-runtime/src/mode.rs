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
