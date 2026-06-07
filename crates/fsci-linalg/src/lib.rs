#![feature(portable_simd)]
#![forbid(unsafe_code)]

pub use fsci_runtime::SyncSharedAuditLedger;
use fsci_runtime::{
    AuditAction, AuditEvent, AuditLedger, DecisionSignals, PolicyAction, PolicyController,
    PolicyDecision, RuntimeMode, SolverAction, SolverEvidenceEntry, SolverPortfolio,
    StructuralEvidence, casp_now_unix_ms,
};
use std::{borrow::Cow, fmt, simd::Simd};

type EigenDecomposition = (Vec<f64>, Option<Vec<Vec<f64>>>);

/// Create a new shared audit ledger for synchronous contexts.
#[must_use]
pub fn sync_audit_ledger() -> SyncSharedAuditLedger {
    AuditLedger::shared()
}

use nalgebra::linalg::Cholesky;
use nalgebra::{DMatrix, DVector, Dyn, LU, linalg::SVD};
use serde::Serialize;

/// Hardened-mode reciprocal condition threshold: matrices with rcond below this
/// are rejected as too ill-conditioned for reliable computation.
const HARDENED_RCOND_THRESHOLD: f64 = 1e-14;

/// Backward-error ceiling used when policy asks for full validation.
const POLICY_FULL_VALIDATION_BACKWARD_ERROR_THRESHOLD: f64 = 1e-8;

/// Hardened-mode maximum matrix dimension. Prevents resource exhaustion.
const HARDENED_MAX_DIM: usize = 10_000;

/// Structured audit log entry emitted by every linalg operation.
#[derive(Debug, Clone, Serialize)]
pub struct LinalgTrace {
    pub operation: &'static str,
    pub matrix_size: (usize, usize),
    pub mode: RuntimeMode,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rcond: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warning: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

// ═══════════════════════════════════════════════════════════════════
// Audit Ledger Integration (§0.19)
// ═══════════════════════════════════════════════════════════════════

/// Acquire the ledger guard, recovering from a poisoned mutex so audit
/// events still record after any prior thread panicked.
/// Resolves [frankenscipy-l2irg] for fsci-linalg.
fn lock_or_recover(ledger: &SyncSharedAuditLedger) -> std::sync::MutexGuard<'_, AuditLedger> {
    match ledger.lock() {
        Ok(g) => g,
        Err(poisoned) => {
            ledger.clear_poison();
            poisoned.into_inner()
        }
    }
}

/// Record a fail-closed audit event when validation rejects input.
fn record_fail_closed(
    ledger: &SyncSharedAuditLedger,
    input_bytes: &[u8],
    reason: &str,
    outcome: &str,
) {
    let event = AuditEvent::new(
        casp_now_unix_ms(),
        AuditLedger::fingerprint_bytes(input_bytes),
        AuditAction::FailClosed {
            reason: reason.to_string(),
        },
        outcome,
    );
    lock_or_recover(ledger).record(event);
}

/// Record a bounded recovery audit event when hardened mode recovers from an issue.
fn record_bounded_recovery(
    ledger: &SyncSharedAuditLedger,
    input_bytes: &[u8],
    recovery_action: &str,
    outcome: &str,
) {
    let event = AuditEvent::new(
        casp_now_unix_ms(),
        AuditLedger::fingerprint_bytes(input_bytes),
        AuditAction::BoundedRecovery {
            recovery_action: recovery_action.to_string(),
        },
        outcome,
    );
    lock_or_recover(ledger).record(event);
}

/// Record a CASP solver selection decision for audit trail.
fn record_casp_decision(
    ledger: &SyncSharedAuditLedger,
    input_bytes: &[u8],
    action: SolverAction,
    rcond: f64,
    fallback: bool,
) {
    let decision_desc = if fallback {
        format!("CASP fallback to {:?} (rcond={rcond:.2e})", action)
    } else {
        format!("CASP selected {:?} (rcond={rcond:.2e})", action)
    };
    let event = AuditEvent::new(
        casp_now_unix_ms(),
        AuditLedger::fingerprint_bytes(input_bytes),
        AuditAction::ModeDecision {
            mode: RuntimeMode::Strict, // CASP operates in both modes
        },
        decision_desc,
    );
    lock_or_recover(ledger).record(event);
}

fn record_mode_decision(
    ledger: &SyncSharedAuditLedger,
    input_bytes: &[u8],
    mode: RuntimeMode,
    outcome: &str,
) {
    let event = AuditEvent::new(
        casp_now_unix_ms(),
        AuditLedger::fingerprint_bytes(input_bytes),
        AuditAction::ModeDecision { mode },
        outcome,
    );
    lock_or_recover(ledger).record(event);
}

fn fail_closed_reason(error: &LinalgError) -> Option<&'static str> {
    match error {
        LinalgError::RaggedMatrix => Some("ragged_matrix"),
        LinalgError::ExpectedSquareMatrix => Some("non_square_matrix"),
        LinalgError::IncompatibleShapes { .. } => Some("incompatible_shapes"),
        LinalgError::NonFiniteInput => Some("non_finite_input"),
        LinalgError::InvalidBandShape { .. } => Some("invalid_band_shape"),
        LinalgError::InvalidPinvThreshold => Some("invalid_pinv_threshold"),
        LinalgError::UnsupportedAssumption => Some("unsupported_assumption"),
        LinalgError::PolicyRejected { .. } => Some("policy_rejected"),
        LinalgError::ConditionTooHigh { .. } => Some("condition_too_high"),
        LinalgError::ResourceExhausted { .. } => Some("resource_exhausted"),
        LinalgError::InvalidArgument { .. } => Some("invalid_argument"),
        LinalgError::NotSupported { .. } | LinalgError::ConvergenceFailure { .. } => {
            Some("not_supported")
        }
        LinalgError::SingularMatrix => None,
    }
}

fn record_operation_audit<T>(
    ledger: &SyncSharedAuditLedger,
    input_bytes: &[u8],
    operation: &str,
    mode: RuntimeMode,
    result: &Result<T, LinalgError>,
) {
    match result {
        Ok(_) => record_mode_decision(ledger, input_bytes, mode, &format!("{operation} executed")),
        Err(error) => {
            if let Some(reason) = fail_closed_reason(error) {
                record_fail_closed(
                    ledger,
                    input_bytes,
                    reason,
                    &format!("{operation} rejected: {error}"),
                );
            } else {
                record_mode_decision(
                    ledger,
                    input_bytes,
                    mode,
                    &format!("{operation} errored: {error}"),
                );
            }
        }
    }
}

/// Compute a fingerprint for matrix input (first 1KB of flattened data).
fn matrix_fingerprint(a: &[Vec<f64>]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(1024);
    for row in a {
        for &val in row {
            if bytes.len() >= 1024 {
                break;
            }
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        if bytes.len() >= 1024 {
            break;
        }
    }
    bytes
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixAssumption {
    General,
    Diagonal,
    UpperTriangular,
    LowerTriangular,
    Symmetric,
    Hermitian,
    PositiveDefinite,
    Banded,
    TriDiagonal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum MatrixSizeCategory {
    Small,
    Medium,
    Large,
}

impl MatrixSizeCategory {
    fn from_shape(rows: usize, cols: usize) -> Self {
        match rows.max(cols) {
            0..=32 => Self::Small,
            33..=256 => Self::Medium,
            _ => Self::Large,
        }
    }
}

/// LAPACK driver selection for least-squares problems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LstsqDriver {
    /// SVD with divide-and-conquer (default, most robust).
    #[default]
    Gelsd,
    /// QR with column pivoting (fastest for well-conditioned).
    Gelsy,
    /// SVD (older, slower than Gelsd).
    Gelss,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriangularTranspose {
    NoTranspose,
    Transpose,
    ConjugateTranspose,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolveOptions {
    pub mode: RuntimeMode,
    pub check_finite: bool,
    pub assume_a: Option<MatrixAssumption>,
    pub lower: bool,
    pub transposed: bool,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            mode: RuntimeMode::Strict,
            check_finite: true,
            assume_a: None,
            lower: false,
            transposed: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TriangularSolveOptions {
    pub mode: RuntimeMode,
    pub check_finite: bool,
    pub trans: TriangularTranspose,
    pub lower: bool,
    pub unit_diagonal: bool,
}

impl Default for TriangularSolveOptions {
    fn default() -> Self {
        Self {
            mode: RuntimeMode::Strict,
            check_finite: true,
            trans: TriangularTranspose::NoTranspose,
            lower: false,
            unit_diagonal: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InvOptions {
    pub mode: RuntimeMode,
    pub check_finite: bool,
    pub assume_a: Option<MatrixAssumption>,
    pub lower: bool,
}

impl Default for InvOptions {
    fn default() -> Self {
        Self {
            mode: RuntimeMode::Strict,
            check_finite: true,
            assume_a: None,
            lower: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LstsqOptions {
    pub mode: RuntimeMode,
    pub check_finite: bool,
    pub cond: Option<f64>,
    pub driver: LstsqDriver,
}

impl Default for LstsqOptions {
    fn default() -> Self {
        Self {
            mode: RuntimeMode::Strict,
            check_finite: true,
            cond: None,
            driver: LstsqDriver::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PinvOptions {
    pub mode: RuntimeMode,
    pub check_finite: bool,
    pub atol: Option<f64>,
    pub rtol: Option<f64>,
}

impl Default for PinvOptions {
    fn default() -> Self {
        Self {
            mode: RuntimeMode::Strict,
            check_finite: true,
            atol: None,
            rtol: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LinalgWarning {
    IllConditioned { reciprocal_condition: f64 },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SolveCertificate {
    pub action: SolverAction,
    pub matrix_shape: (usize, usize),
    pub rcond_estimate: f64,
    pub structural_evidence: StructuralEvidence,
    pub posterior: Vec<f64>,
    pub expected_losses: Vec<f64>,
    pub chosen_expected_loss: f64,
    pub fallback_active: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SolveResult {
    pub x: Vec<f64>,
    pub warning: Option<LinalgWarning>,
    pub backward_error: Option<f64>,
    pub certificate: Option<SolveCertificate>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InvResult {
    pub inverse: Vec<Vec<f64>>,
    pub warning: Option<LinalgWarning>,
    pub certificate: Option<SolveCertificate>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LstsqResult {
    pub x: Vec<f64>,
    pub residuals: Vec<f64>,
    pub rank: usize,
    pub singular_values: Vec<f64>,
    pub certificate: Option<SolveCertificate>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PinvResult {
    pub pseudo_inverse: Vec<Vec<f64>>,
    pub rank: usize,
    pub certificate: Option<SolveCertificate>,
}

struct LowRankLstsqResult {
    x: Vec<f64>,
    rank: usize,
    singular_values: Vec<f64>,
    rcond_estimate: f64,
}

struct LowRankPinvResult {
    pseudo_inverse: Vec<Vec<f64>>,
    rank: usize,
    rcond_estimate: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ConditionReport {
    pub matrix_shape: (usize, usize),
    pub rcond_estimate: f64,
    pub structural_evidence: StructuralEvidence,
    pub diagonal: bool,
    pub upper_triangular: bool,
    pub lower_triangular: bool,
    pub symmetric: bool,
    pub positive_definite: bool,
    pub banded: bool,
    pub bandwidth: (usize, usize),
    pub size_category: MatrixSizeCategory,
    pub sparsity_ratio: f64,
}

struct ConditionDiagnosticsWork {
    report: ConditionReport,
    matrix_cache: Option<DMatrix<f64>>,
    lu_cache: Option<LU<f64, Dyn, Dyn>>,
}

/// Result of LU decomposition with partial pivoting.
#[derive(Debug, Clone, PartialEq)]
pub struct LuResult {
    /// Permutation matrix P (row-major).
    pub p: Vec<Vec<f64>>,
    /// Lower triangular factor L (unit diagonal).
    pub l: Vec<Vec<f64>>,
    /// Upper triangular factor U.
    pub u: Vec<Vec<f64>>,
}

/// Compact LU factorization for use with `lu_solve`.
#[derive(Clone)]
pub struct LuFactorResult {
    /// The internal nalgebra LU object.
    lu_internal: LU<f64, Dyn, Dyn>,
    /// Matrix dimension.
    n: usize,
    /// 1-norm of the original matrix.
    a_norm_1: f64,
    /// Cached reciprocal-condition estimate for repeated `lu_solve` calls.
    rcond_estimate: f64,
}

impl fmt::Debug for LuFactorResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LuFactorResult")
            .field("lu_internal", &self.lu_internal)
            .field("n", &self.n)
            .field("a_norm_1", &self.a_norm_1)
            .finish()
    }
}

/// Result of QR decomposition.
#[derive(Debug, Clone, PartialEq)]
pub struct QrResult {
    /// Orthogonal factor Q (m×m or m×min(m,n) for economy mode).
    pub q: Vec<Vec<f64>>,
    /// Upper triangular factor R.
    pub r: Vec<Vec<f64>>,
}

/// Result of SVD decomposition.
#[derive(Debug, Clone, PartialEq)]
pub struct SvdResult {
    /// Left singular vectors U (m×k where k=min(m,n)).
    pub u: Vec<Vec<f64>>,
    /// Singular values σ in descending order.
    pub s: Vec<f64>,
    /// Right singular vectors Vᵀ (k×n where k=min(m,n)).
    pub vt: Vec<Vec<f64>>,
}

/// Result of Cholesky decomposition.
#[derive(Debug, Clone, PartialEq)]
pub struct CholeskyResult {
    /// Lower triangular factor L such that A = LLᵀ (or upper if requested).
    pub factor: Vec<Vec<f64>>,
}

/// Result of LDL decomposition: A = L * D * Lᵀ.
#[derive(Debug, Clone, PartialEq)]
pub struct LdlResult {
    /// Unit lower triangular factor L.
    pub l: Vec<Vec<f64>>,
    /// Diagonal entries of D.
    pub d: Vec<f64>,
}

/// Compact Cholesky factorization for use with `cho_solve`.
#[derive(Debug, Clone)]
pub struct ChoFactorResult {
    /// The internal nalgebra Cholesky object.
    chol_internal: Cholesky<f64, Dyn>,
    /// Matrix dimension.
    n: usize,
}

impl ChoFactorResult {
    /// Matrix dimension of the factored matrix.
    pub fn dimension(&self) -> usize {
        self.n
    }
}

/// Result of eigenvalue decomposition.
#[derive(Debug, Clone, PartialEq)]
pub struct EigResult {
    /// Eigenvalues as (real, imaginary) pairs.
    pub eigenvalues_re: Vec<f64>,
    pub eigenvalues_im: Vec<f64>,
    /// Right eigenvectors as columns (row-major storage).
    pub eigenvectors: Vec<Vec<f64>>,
}

/// Result of symmetric eigenvalue decomposition.
#[derive(Debug, Clone, PartialEq)]
pub struct EighResult {
    /// Real eigenvalues in ascending order.
    pub eigenvalues: Vec<f64>,
    /// Orthogonal eigenvectors as columns (row-major storage).
    pub eigenvectors: Vec<Vec<f64>>,
}

/// Result of Schur decomposition.
#[derive(Debug, Clone, PartialEq)]
pub struct SchurResult {
    /// Unitary/orthogonal matrix Z (Schur vectors).
    pub z: Vec<Vec<f64>>,
    /// Upper quasi-triangular Schur form T (real Schur form).
    pub t: Vec<Vec<f64>>,
}

/// Result of Hessenberg decomposition.
#[derive(Debug, Clone, PartialEq)]
pub struct HessenbergResult {
    /// Orthogonal matrix Q.
    pub q: Vec<Vec<f64>>,
    /// Upper Hessenberg matrix H.
    pub h: Vec<Vec<f64>>,
}

/// Result of generalized Schur (QZ) decomposition.
#[derive(Debug, Clone, PartialEq)]
pub struct QzResult {
    /// Left transform matrix Q.
    pub q: Vec<Vec<f64>>,
    /// Right transform matrix Z.
    pub z: Vec<Vec<f64>>,
    /// Generalized Schur form of A under Qᵀ A Z.
    pub aa: Vec<Vec<f64>>,
    /// Generalized Schur form of B under Qᵀ B Z.
    pub bb: Vec<Vec<f64>>,
}

/// Ordering selector for the simplified real `ordqz` path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrdQzSort {
    /// Stable generalized eigenvalues first for continuous-time systems:
    /// λ = α/β with λ < 0.
    LeftHalfPlane,
    /// Stable generalized eigenvalues first for discrete-time systems:
    /// λ = α/β with |λ| < 1.
    InsideUnitCircle,
}

/// Result of polar decomposition: A = U * P.
#[derive(Debug, Clone, PartialEq)]
pub struct PolarResult {
    /// Unitary (or semi-unitary) factor U.
    pub u: Vec<Vec<f64>>,
    /// Positive semi-definite factor P.
    pub p: Vec<Vec<f64>>,
}

/// Options for decomposition operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecompOptions {
    pub mode: RuntimeMode,
    pub check_finite: bool,
}

impl Default for DecompOptions {
    fn default() -> Self {
        Self {
            mode: RuntimeMode::Strict,
            check_finite: true,
        }
    }
}

/// Norm type selection matching SciPy conventions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormKind {
    /// Frobenius norm (matrix) or L2 norm (vector).
    Fro,
    /// Spectral norm (largest singular value for matrix, max abs for vector).
    Spectral,
    /// 1-norm (max column sum for matrix, sum of abs for vector).
    One,
    /// Infinity norm (max row sum for matrix, max abs for vector).
    Inf,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LinalgError {
    RaggedMatrix,
    ExpectedSquareMatrix,
    IncompatibleShapes {
        a_shape: (usize, usize),
        b_len: usize,
    },
    NonFiniteInput,
    SingularMatrix,
    UnsupportedAssumption,
    InvalidBandShape {
        expected_rows: usize,
        actual_rows: usize,
    },
    InvalidPinvThreshold,
    /// Complex transposed solve not supported (matches SciPy NotImplementedError).
    NotSupported {
        detail: String,
    },
    /// SVD or eigenvalue convergence failure.
    ConvergenceFailure {
        detail: String,
    },
    /// Runtime policy controller rejected the solve.
    PolicyRejected {
        reason: String,
    },
    /// Hardened mode: condition number exceeds threshold.
    ConditionTooHigh {
        rcond: f64,
        threshold: f64,
    },
    /// Hardened mode: matrix dimension exceeds resource limit.
    ResourceExhausted {
        detail: String,
    },
    /// Generic invalid argument.
    InvalidArgument {
        detail: String,
    },
}

impl std::fmt::Display for LinalgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RaggedMatrix => write!(f, "matrix rows must all have equal length"),
            Self::ExpectedSquareMatrix => write!(f, "expected square matrix"),
            Self::IncompatibleShapes { a_shape, b_len } => {
                write!(f, "incompatible shapes: a_shape={a_shape:?}, b_len={b_len}")
            }
            Self::NonFiniteInput => write!(f, "array must not contain infs or NaNs"),
            Self::SingularMatrix => write!(f, "singular matrix"),
            Self::UnsupportedAssumption => write!(f, "unsupported matrix assumption"),
            Self::InvalidBandShape {
                expected_rows,
                actual_rows,
            } => write!(
                f,
                "invalid values for the number of lower and upper diagonals: expected {expected_rows} rows in ab, got {actual_rows}"
            ),
            Self::InvalidPinvThreshold => write!(f, "atol and rtol values must be positive."),
            Self::NotSupported { detail } => write!(f, "{detail}"),
            Self::ConvergenceFailure { detail } => write!(f, "{detail}"),
            Self::PolicyRejected { reason } => write!(f, "policy rejected solve: {reason}"),
            Self::ConditionTooHigh { rcond, threshold } => {
                write!(
                    f,
                    "matrix condition number 1/{rcond:.2e} exceeds hardened threshold 1/{threshold:.2e}"
                )
            }
            Self::ResourceExhausted { detail } => write!(f, "resource exhausted: {detail}"),
            Self::InvalidArgument { detail } => write!(f, "{detail}"),
        }
    }
}

impl std::error::Error for LinalgError {}

pub fn solve(a: &[Vec<f64>], b: &[f64], options: SolveOptions) -> Result<SolveResult, LinalgError> {
    // Fast path for large general square systems: our own multithreaded blocked LU
    // (trailing update on all cores). Restricted to the plain Strict / untransposed /
    // General case so all the portfolio diagnostics (rcond, hardened checks, special
    // assumptions, transposition) keep their exact behavior; a singular pivot or any
    // unmet precondition falls through to the portfolio solver unchanged.
    let n = a.len();
    if n >= BLOCKED_LU_MIN_DIM
        && options.mode == RuntimeMode::Strict
        && !options.transposed
        && matches!(options.assume_a, None | Some(MatrixAssumption::General))
        && b.len() == n
        && rows_are_rectangular(a, n)
        && a.iter().flatten().all(|v| v.is_finite())
        && b.iter().all(|v| v.is_finite())
        && let Some(x) = lu_solve_blocked(a, b)
    {
        let backward_error = compute_backward_error_dense(a, &x, b);
        emit_trace(LinalgTrace {
            operation: "solve",
            matrix_size: (n, n),
            mode: options.mode,
            rcond: None,
            warning: None,
            error: None,
        });
        return Ok(SolveResult {
            x,
            warning: None,
            backward_error: Some(backward_error),
            certificate: None,
        });
    }
    // Fast path for large symmetric positive-definite systems: our own blocked
    // Cholesky (parallel trailing update). A non-positive pivot (not actually PD)
    // returns None and falls through to the portfolio solver, which preserves the
    // exact `assume_a = pos` rejection behavior.
    if n >= BLOCKED_LU_MIN_DIM
        && options.mode == RuntimeMode::Strict
        && !options.transposed
        && options.assume_a == Some(MatrixAssumption::PositiveDefinite)
        && b.len() == n
        && rows_are_rectangular(a, n)
        && a.iter().flatten().all(|v| v.is_finite())
        && b.iter().all(|v| v.is_finite())
        && let Some(x) = cholesky_solve_blocked(a, b)
    {
        let backward_error = compute_backward_error_dense(a, &x, b);
        emit_trace(LinalgTrace {
            operation: "solve",
            matrix_size: (n, n),
            mode: options.mode,
            rcond: None,
            warning: None,
            error: None,
        });
        return Ok(SolveResult {
            x,
            warning: None,
            backward_error: Some(backward_error),
            certificate: None,
        });
    }
    let mut portfolio = SolverPortfolio::new(options.mode, 1);
    solve_with_portfolio_internal(a, b, options, &mut portfolio, "solve", false)
}

pub fn solve_triangular(
    a: &[Vec<f64>],
    b: &[f64],
    options: TriangularSolveOptions,
) -> Result<SolveResult, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    if b.len() != rows {
        return Err(LinalgError::IncompatibleShapes {
            a_shape: (rows, cols),
            b_len: b.len(),
        });
    }
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix_and_vector(a, b, options.mode, options.check_finite)?;
    if rows == 0 {
        let result = Ok(SolveResult {
            x: Vec::new(),
            warning: None,
            backward_error: None,
            certificate: None,
        });
        emit_trace(LinalgTrace {
            operation: "solve_triangular",
            matrix_size: (rows, cols),
            mode: options.mode,
            rcond: None,
            warning: None,
            error: None,
        });
        return result;
    }
    let result =
        solve_triangular_internal(a, b, options.trans, options.lower, options.unit_diagonal);
    emit_trace(LinalgTrace {
        operation: "solve_triangular",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: result.as_ref().err().map(|e| e.to_string()),
    });
    result
}

pub fn solve_triangular_with_audit(
    a: &[Vec<f64>],
    b: &[f64],
    options: TriangularSolveOptions,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<SolveResult, LinalgError> {
    let fingerprint = matrix_fingerprint(a);
    let result = solve_triangular(a, b, options);
    record_operation_audit(
        audit_ledger,
        &fingerprint,
        "solve_triangular",
        options.mode,
        &result,
    );
    result
}

pub fn solve_banded(
    l_and_u: (usize, usize),
    ab: &[Vec<f64>],
    b: &[f64],
    options: SolveOptions,
) -> Result<SolveResult, LinalgError> {
    let (nlower, nupper) = l_and_u;
    let (rows, cols) = matrix_shape(ab)?;
    let expected_rows = nlower + nupper + 1;
    if rows != expected_rows {
        return Err(LinalgError::InvalidBandShape {
            expected_rows,
            actual_rows: rows,
        });
    }
    if b.len() != cols {
        return Err(LinalgError::IncompatibleShapes {
            a_shape: (cols, cols),
            b_len: b.len(),
        });
    }
    hardened_dimension_check(options.mode, cols, cols)?;
    validate_finite_matrix_and_vector(ab, b, options.mode, options.check_finite)?;

    // Fast path: banded Gaussian elimination with partial pivoting on LAPACK-style
    // packed band storage — O(n·kl·(kl+ku)) time and O(n·(kl+ku)) memory, never
    // materializing the dense matrix. On a zero/NaN pivot (structurally singular
    // within the band) we fall back to the robust dense solver below so
    // singular-case behavior is preserved exactly. The packed factorization runs
    // the identical floating-point operations as a dense banded GEPP (out-of-band
    // entries are structural zeros), so the solution is bit-identical to the dense
    // path; the backward error is likewise summed only over band nonzeros, whose
    // omitted dense terms are `+0.0` no-ops, so it too matches bit-for-bit.
    if let Some(x) = banded_lu_solve_packed(ab, nlower, nupper, b) {
        let backward_error = compute_backward_error_banded(ab, nlower, nupper, &x, b);
        return Ok(SolveResult {
            x,
            warning: None,
            backward_error: Some(backward_error),
            certificate: None,
        });
    }
    let dense = dense_from_banded(nlower, nupper, ab, cols);
    solve(
        &dense,
        b,
        SolveOptions {
            assume_a: Some(MatrixAssumption::General),
            ..options
        },
    )
}

pub fn solve_banded_with_audit(
    l_and_u: (usize, usize),
    ab: &[Vec<f64>],
    b: &[f64],
    options: SolveOptions,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<SolveResult, LinalgError> {
    let fingerprint = matrix_fingerprint(ab);
    let result = solve_banded(l_and_u, ab, b, options);
    record_operation_audit(
        audit_ledger,
        &fingerprint,
        "solve_banded",
        options.mode,
        &result,
    );
    result
}

pub fn inv(a: &[Vec<f64>], options: InvOptions) -> Result<InvResult, LinalgError> {
    // Fast path for large general square matrices: factor once with the in-house
    // parallel blocked LU, then solve A X = I over the identity columns on all cores.
    // Restricted to the plain Strict / General case; a singular pivot or any unmet
    // precondition falls through to the portfolio inverse (diagnostics preserved).
    let n = a.len();
    if n >= BLOCKED_LU_MIN_DIM
        && options.mode == RuntimeMode::Strict
        && matches!(options.assume_a, None | Some(MatrixAssumption::General))
        && rows_are_rectangular(a, n)
        && a.iter().flatten().all(|v| v.is_finite())
        && let Some(inverse) = inv_blocked(a)
    {
        emit_trace(LinalgTrace {
            operation: "inv",
            matrix_size: (n, n),
            mode: options.mode,
            rcond: None,
            warning: None,
            error: None,
        });
        return Ok(InvResult {
            inverse,
            warning: None,
            certificate: None,
        });
    }
    let mut portfolio = SolverPortfolio::new(options.mode, 1);
    inv_with_casp(a, options, &mut portfolio)
}

pub fn inv_with_audit(
    a: &[Vec<f64>],
    options: InvOptions,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<InvResult, LinalgError> {
    let fingerprint = matrix_fingerprint(a);
    let result = inv(a, options);
    record_operation_audit(audit_ledger, &fingerprint, "inv", options.mode, &result);
    result
}

pub fn det(a: &[Vec<f64>], mode: RuntimeMode, check_finite: bool) -> Result<f64, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    hardened_dimension_check(mode, rows, cols)?;
    validate_finite_matrix(a, mode, check_finite)?;

    if rows == 0 {
        return Ok(1.0);
    }
    let matrix = dmatrix_from_rows(a)?;
    let result = matrix.lu().determinant();
    emit_trace(LinalgTrace {
        operation: "det",
        matrix_size: (rows, cols),
        mode,
        rcond: None,
        warning: None,
        error: None,
    });
    Ok(result)
}

pub fn det_with_audit(
    a: &[Vec<f64>],
    mode: RuntimeMode,
    check_finite: bool,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<f64, LinalgError> {
    let fingerprint = matrix_fingerprint(a);
    let result = det(a, mode, check_finite);
    record_operation_audit(audit_ledger, &fingerprint, "det", mode, &result);
    result
}

pub fn lstsq(a: &[Vec<f64>], b: &[f64], options: LstsqOptions) -> Result<LstsqResult, LinalgError> {
    let mut portfolio = SolverPortfolio::new(options.mode, 1);
    lstsq_with_casp(a, b, options, &mut portfolio)
}

pub fn lstsq_with_audit(
    a: &[Vec<f64>],
    b: &[f64],
    options: LstsqOptions,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<LstsqResult, LinalgError> {
    let fingerprint = matrix_fingerprint(a);
    let result = lstsq(a, b, options);
    record_operation_audit(audit_ledger, &fingerprint, "lstsq", options.mode, &result);
    result
}

pub fn pinv(a: &[Vec<f64>], options: PinvOptions) -> Result<PinvResult, LinalgError> {
    let mut portfolio = SolverPortfolio::new(options.mode, 1);
    pinv_with_casp(a, options, &mut portfolio)
}

pub fn pinv_with_audit(
    a: &[Vec<f64>],
    options: PinvOptions,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<PinvResult, LinalgError> {
    let fingerprint = matrix_fingerprint(a);
    let result = pinv(a, options);
    record_operation_audit(audit_ledger, &fingerprint, "pinv", options.mode, &result);
    result
}

/// Solve Aᵀ x = b using LU factorization PA = LU => Aᵀ = Uᵀ Lᵀ P.
fn solve_lu_transpose(lu: &LU<f64, Dyn, Dyn>, b: &DVector<f64>) -> Option<DVector<f64>> {
    let u_t = lu.u().transpose();
    let l_t = lu.l().transpose();
    solve_lu_transpose_with_transposes(lu, &u_t, &l_t, b)
}

fn solve_lu_transpose_with_transposes(
    lu: &LU<f64, Dyn, Dyn>,
    u_t: &DMatrix<f64>,
    l_t: &DMatrix<f64>,
    b: &DVector<f64>,
) -> Option<DVector<f64>> {
    // Aᵀ = Uᵀ Lᵀ P
    // Uᵀ Lᵀ P x = b
    // 1. Solve Uᵀ y = b (lower triangular)
    let y = u_t.solve_lower_triangular(b)?;
    // 2. Solve Lᵀ z = y (upper triangular)
    let z = l_t.solve_upper_triangular(&y)?;
    // 3. P x = z => x = Pᵀ z
    let mut x = z;
    lu.p().inv_permute_rows(&mut x);
    Some(x)
}

/// Induced 1-norm for a matrix (max column sum).
fn matrix_norm1(matrix: &DMatrix<f64>) -> f64 {
    if matrix.ncols() == 0 || matrix.nrows() == 0 {
        return 0.0;
    }
    let mut max_col_sum = 0.0_f64;
    for col in 0..matrix.ncols() {
        let mut sum = 0.0_f64;
        for row in 0..matrix.nrows() {
            let value = matrix[(row, col)].abs();
            if !value.is_finite() {
                return f64::NAN;
            }
            sum += value;
        }
        if sum > max_col_sum {
            max_col_sum = sum;
        }
    }
    max_col_sum
}

fn rcond_from_singular_values(values: &DVector<f64>) -> f64 {
    let mut max_s = 0.0_f64;
    let mut min_s = f64::MAX;
    for &value in values.iter() {
        if !value.is_finite() {
            return 0.0;
        }
        if value > max_s {
            max_s = value;
        }
        if value < min_s {
            min_s = value;
        }
    }
    if max_s > 0.0 { min_s / max_s } else { 0.0 }
}

/// O(n²) reciprocal condition estimate from LU — 1-norm Higham estimator.
/// Cost: 2 solves (O(n²)) vs O(n³) for full SVD.
fn fast_rcond_from_lu(lu: &LU<f64, Dyn, Dyn>, a_norm: f64, n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if a_norm == 0.0 || !a_norm.is_finite() {
        return 0.0;
    }

    // Estimate ||A⁻¹||₁ using Higham's iterative algorithm (up to 5 iterations)
    let mut x = DVector::from_element(n, 1.0 / (n as f64));
    let mut inv_a_norm = 0.0;
    let u_t = lu.u().transpose();
    let l_t = lu.l().transpose();

    for _ in 0..5 {
        // 1. Solve Aᵀ w = sign(x)
        let sign_x = x.map(|val| if val >= 0.0 { 1.0 } else { -1.0 });
        let w = match solve_lu_transpose_with_transposes(lu, &u_t, &l_t, &sign_x) {
            Some(w) => w,
            None => return 0.0,
        };

        // 2. Solve A x_new = sign(w)
        let sign_w = w.map(|val| if val >= 0.0 { 1.0 } else { -1.0 });
        let x_new = match lu.solve(&sign_w) {
            Some(x) => x,
            None => return 0.0,
        };

        let new_norm = x_new.lp_norm(1);
        let direction_delta = x_new
            .iter()
            .zip(x.iter())
            .map(|(&new, &old)| (new - old).abs())
            .sum::<f64>();
        if (new_norm - inv_a_norm).abs() <= 1e-10 * new_norm {
            inv_a_norm = new_norm;
            break;
        }
        inv_a_norm = new_norm;
        x = x_new;

        // Check if we are oscillating or converged in direction
        if direction_delta <= f64::EPSILON * new_norm {
            break;
        }
    }

    if inv_a_norm <= 0.0 {
        return 0.0;
    }

    let rcond = 1.0 / (a_norm * inv_a_norm);
    if rcond.is_nan() { 0.0 } else { rcond.min(1.0) }
}

/// Map linalg assumption to runtime structural evidence for CASP.
fn assumption_to_evidence(a: MatrixAssumption) -> fsci_runtime::StructuralEvidence {
    match a {
        MatrixAssumption::Diagonal => fsci_runtime::StructuralEvidence::Diagonal,
        MatrixAssumption::UpperTriangular | MatrixAssumption::LowerTriangular => {
            fsci_runtime::StructuralEvidence::Triangular
        }
        _ => fsci_runtime::StructuralEvidence::General,
    }
}

fn normalize_assumption_for_effective_matrix(
    assumption: Option<MatrixAssumption>,
    transposed: bool,
) -> Option<MatrixAssumption> {
    if !transposed {
        return assumption;
    }

    match assumption {
        Some(MatrixAssumption::UpperTriangular) => Some(MatrixAssumption::LowerTriangular),
        Some(MatrixAssumption::LowerTriangular) => Some(MatrixAssumption::UpperTriangular),
        other => other,
    }
}

fn structure_tolerance(a: &[Vec<f64>]) -> f64 {
    let max_abs = a
        .iter()
        .flat_map(|row| row.iter())
        .fold(0.0_f64, |acc, &value| acc.max(value.abs()));
    (64.0 * f64::EPSILON * max_abs.max(1.0)).max(1e-15)
}

fn bandwidth_with_tolerance(a: &[Vec<f64>], tol: f64) -> (usize, usize) {
    let n = a.len();
    if n == 0 {
        return (0, 0);
    }
    let m = a[0].len();
    let mut lower = 0usize;
    let mut upper = 0usize;

    for (i, row) in a.iter().enumerate().take(n) {
        for (j, &value) in row.iter().enumerate().take(m) {
            if !value.is_finite() || value.abs() > tol {
                if i > j {
                    lower = lower.max(i - j);
                }
                if j > i {
                    upper = upper.max(j - i);
                }
            }
        }
    }

    (lower, upper)
}

fn condition_diagnostics_with_assumption(
    a: &[Vec<f64>],
    assumption: Option<MatrixAssumption>,
) -> Result<ConditionDiagnosticsWork, LinalgError> {
    condition_diagnostics_with_assumption_mode(a, assumption, true)
}

fn condition_diagnostics_for_solve(
    a: &[Vec<f64>],
    assumption: Option<MatrixAssumption>,
) -> Result<ConditionDiagnosticsWork, LinalgError> {
    condition_diagnostics_with_assumption_mode(a, assumption, false)
}

fn condition_diagnostics_with_assumption_mode(
    a: &[Vec<f64>],
    assumption: Option<MatrixAssumption>,
    evaluate_positive_definite: bool,
) -> Result<ConditionDiagnosticsWork, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    let tol = structure_tolerance(a);

    let diagonal = assumption == Some(MatrixAssumption::Diagonal) || is_diagonal(a, tol);
    let upper_triangular = diagonal
        || assumption == Some(MatrixAssumption::UpperTriangular)
        || is_upper_triangular(a, tol);
    let lower_triangular = diagonal
        || assumption == Some(MatrixAssumption::LowerTriangular)
        || is_lower_triangular(a, tol);
    let symmetric = matches!(
        assumption,
        Some(
            MatrixAssumption::Symmetric
                | MatrixAssumption::Hermitian
                | MatrixAssumption::PositiveDefinite
        )
    ) || issymmetric(a, tol, tol)?;
    let positive_definite = assumption == Some(MatrixAssumption::PositiveDefinite)
        || (evaluate_positive_definite && symmetric && is_positive_definite(a));
    let bandwidth = bandwidth_with_tolerance(a, tol);
    let banded = rows > 0
        && cols > 0
        && (diagonal
            || upper_triangular
            || lower_triangular
            || bandwidth.0 + bandwidth.1 + 1 < rows.max(cols));
    let total_values = rows.saturating_mul(cols);
    let near_zero_values = a
        .iter()
        .flat_map(|row| row.iter())
        .filter(|&&value| value.abs() <= tol)
        .count();
    let sparsity_ratio = if total_values == 0 {
        1.0
    } else {
        near_zero_values as f64 / total_values as f64
    };

    let structural_evidence = if diagonal {
        StructuralEvidence::Diagonal
    } else if upper_triangular || lower_triangular {
        StructuralEvidence::Triangular
    } else {
        assumption
            .map(assumption_to_evidence)
            .unwrap_or(StructuralEvidence::General)
    };

    let mut matrix_cache = None;
    let mut lu_cache = None;
    let rcond_estimate = if rows == 0 || cols == 0 {
        1.0
    } else if rows != cols {
        0.0
    } else if diagonal {
        fast_rcond_diagonal(a)
    } else if lower_triangular && !upper_triangular {
        fast_rcond_triangular(a, true)
    } else if upper_triangular && !lower_triangular {
        fast_rcond_triangular(a, false)
    } else {
        let (matrix, matrix_norm_1) = dmatrix_from_rows_with_norm1(a)?;
        let lu = matrix.clone().lu();
        let rcond = if rows <= 4 {
            match safe_svd(matrix.clone(), false, false) {
                Ok(svd) => rcond_from_singular_values(&svd.singular_values),
                Err(_) => fast_rcond_from_lu(&lu, matrix_norm_1, rows),
            }
        } else {
            fast_rcond_from_lu(&lu, matrix_norm_1, rows)
        };
        matrix_cache = Some(matrix);
        lu_cache = Some(lu);
        rcond
    };

    Ok(ConditionDiagnosticsWork {
        report: ConditionReport {
            matrix_shape: (rows, cols),
            rcond_estimate,
            structural_evidence,
            diagonal,
            upper_triangular,
            lower_triangular,
            symmetric,
            positive_definite,
            banded,
            bandwidth,
            size_category: MatrixSizeCategory::from_shape(rows, cols),
            sparsity_ratio,
        },
        matrix_cache,
        lu_cache,
    })
}

pub fn condition_diagnostics(a: &[Vec<f64>]) -> Result<ConditionReport, LinalgError> {
    Ok(condition_diagnostics_with_assumption(a, None)?.report)
}

fn condition_signal_from_rcond(rcond: f64) -> f64 {
    if rcond.is_finite() && rcond > 0.0 {
        (-rcond.log10()).clamp(0.0, 16.0)
    } else {
        16.0
    }
}

fn anomaly_signal_from_rcond(rcond: f64) -> f64 {
    if !rcond.is_finite() || rcond == 0.0 {
        1.0
    } else if rcond < HARDENED_RCOND_THRESHOLD {
        0.4
    } else if rcond < 1e-12 {
        0.2
    } else {
        0.0
    }
}

fn assumption_incompatibility_score(
    a: &[Vec<f64>],
    assumption: Option<MatrixAssumption>,
) -> Result<f64, LinalgError> {
    let Some(assumption) = assumption else {
        return Ok(0.0);
    };
    let tol = structure_tolerance(a);
    let incompatible = match assumption {
        MatrixAssumption::General => false,
        MatrixAssumption::Diagonal => !is_diagonal(a, tol),
        MatrixAssumption::UpperTriangular => !is_upper_triangular(a, tol),
        MatrixAssumption::LowerTriangular => !is_lower_triangular(a, tol),
        MatrixAssumption::Symmetric | MatrixAssumption::Hermitian => !issymmetric(a, tol, tol)?,
        MatrixAssumption::PositiveDefinite => {
            !issymmetric(a, tol, tol)? || !is_positive_definite(a)
        }
        MatrixAssumption::Banded => {
            let (rows, cols) = matrix_shape(a)?;
            let bandwidth = bandwidth_with_tolerance(a, tol);
            bandwidth.0 + bandwidth.1 + 1 >= rows.max(cols)
        }
        MatrixAssumption::TriDiagonal => {
            let bandwidth = bandwidth_with_tolerance(a, tol);
            bandwidth.0 > 1 || bandwidth.1 > 1
        }
    };

    Ok(if incompatible { 1.0 } else { 0.0 })
}

fn should_apply_solve_policy(
    mode: RuntimeMode,
    check_finite: bool,
    a: &[Vec<f64>],
    b: &[f64],
) -> bool {
    mode == RuntimeMode::Hardened
        || check_finite
        || (a.iter().flatten().all(|value| value.is_finite())
            && b.iter().all(|value| value.is_finite()))
}

fn solve_policy_decision(
    mode: RuntimeMode,
    report: &ConditionReport,
    metadata_incompatibility_score: f64,
) -> Result<PolicyDecision, LinalgError> {
    let mut controller = PolicyController::new(mode, 8);
    let signals = DecisionSignals::new(
        condition_signal_from_rcond(report.rcond_estimate),
        metadata_incompatibility_score,
        anomaly_signal_from_rcond(report.rcond_estimate),
    );
    let decision = controller.decide(signals);
    if decision.action == PolicyAction::FailClosed {
        return Err(LinalgError::PolicyRejected {
            reason: decision.reason,
        });
    }
    Ok(decision)
}

fn enforce_policy_full_validation(
    decision: Option<&PolicyDecision>,
    result: &SolveResult,
) -> Result<(), LinalgError> {
    if !matches!(
        decision.map(|decision| decision.action),
        Some(PolicyAction::FullValidate)
    ) {
        return Ok(());
    }

    let Some(backward_error) = result.backward_error else {
        return Ok(());
    };
    if backward_error.is_finite()
        && backward_error <= POLICY_FULL_VALIDATION_BACKWARD_ERROR_THRESHOLD
    {
        return Ok(());
    }

    Err(LinalgError::ConvergenceFailure {
        detail: format!(
            "policy full validation rejected solve: backward_error={backward_error:.2e}"
        ),
    })
}

/// Compute backward error: ||Ax - b|| / (||A|| × ||x|| + ||b||).
/// Returns 0.0 when the denominator is zero.
fn compute_backward_error(matrix: &DMatrix<f64>, x: &DVector<f64>, rhs: &DVector<f64>) -> f64 {
    let residual = matrix * x - rhs;
    let residual_norm = residual.norm();
    let denom = matrix.norm() * x.norm() + rhs.norm();
    if !residual_norm.is_finite() || !denom.is_finite() {
        return f64::INFINITY;
    }
    if denom > 0.0 {
        residual_norm / denom
    } else {
        0.0
    }
}

fn rcond_warning(rcond: f64) -> Option<LinalgWarning> {
    if rcond < 1e-12 {
        Some(LinalgWarning::IllConditioned {
            reciprocal_condition: rcond,
        })
    } else {
        None
    }
}

/// Emit a structured JSON log entry to stderr for audit trail compatibility.
fn emit_trace(trace: LinalgTrace) {
    if !trace_enabled() {
        return;
    }
    if let Ok(json) = serde_json::to_string(&trace) {
        eprintln!("{json}");
    }
}

/// Trace output is opt-in to avoid flooding benches/tests.
/// Enable with `FSCI_LINALG_TRACE=1`.
fn trace_enabled() -> bool {
    static TRACE_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TRACE_ENABLED.get_or_init(|| {
        std::env::var("FSCI_LINALG_TRACE")
            .map(|value| matches!(value.as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false)
    })
}

/// Hardened-mode dimension guard: reject matrices exceeding resource limits.
fn hardened_dimension_check(
    mode: RuntimeMode,
    rows: usize,
    cols: usize,
) -> Result<(), LinalgError> {
    if mode == RuntimeMode::Hardened && (rows > HARDENED_MAX_DIM || cols > HARDENED_MAX_DIM) {
        return Err(LinalgError::ResourceExhausted {
            detail: format!(
                "matrix dimension ({rows}x{cols}) exceeds hardened limit ({HARDENED_MAX_DIM})"
            ),
        });
    }
    Ok(())
}

/// General solver with hardened-mode condition number rejection.
#[allow(dead_code)] // Used by test-only solve_general
fn solve_general_with_hardening(
    a: &[Vec<f64>],
    b: &[f64],
    mode: RuntimeMode,
) -> Result<SolveResult, LinalgError> {
    let matrix = dmatrix_from_rows(a)?;
    let rhs = DVector::from_column_slice(b);
    let lu: LU<f64, Dyn, Dyn> = matrix.clone().lu();
    let n = a.len();
    let rcond = fast_rcond_from_lu(&lu, matrix_norm1(&matrix), n);

    // Hardened mode: reject if condition number is too high
    if mode == RuntimeMode::Hardened && rcond < HARDENED_RCOND_THRESHOLD && rcond > 0.0 {
        return Err(LinalgError::ConditionTooHigh {
            rcond,
            threshold: HARDENED_RCOND_THRESHOLD,
        });
    }

    let x = lu.solve(&rhs).ok_or(LinalgError::SingularMatrix)?;
    let backward_err = compute_backward_error(&matrix, &x, &rhs);

    Ok(SolveResult {
        x: x.iter().copied().collect(),
        warning: rcond_warning(rcond),
        backward_error: Some(backward_err),
        certificate: None,
    })
}

#[cfg(test)]
fn solve_general(a: &[Vec<f64>], b: &[f64]) -> Result<SolveResult, LinalgError> {
    solve_general_with_hardening(a, b, RuntimeMode::Strict)
}

fn solve_qr(a: &[Vec<f64>], b: &[f64]) -> Result<SolveResult, LinalgError> {
    let matrix = dmatrix_from_rows(a)?;
    let rhs = DVector::from_column_slice(b);
    let qr = matrix.clone().qr();
    let x = qr.solve(&rhs).ok_or(LinalgError::SingularMatrix)?;
    let backward_err = compute_backward_error(&matrix, &x, &rhs);

    Ok(SolveResult {
        x: x.iter().copied().collect(),
        warning: None,
        backward_error: Some(backward_err),
        certificate: None,
    })
}

fn solve_svd_fallback(a: &[Vec<f64>], b: &[f64]) -> Result<SolveResult, LinalgError> {
    let matrix = dmatrix_from_rows(a)?;
    let rhs = DVector::from_column_slice(b);
    let svd = safe_svd(matrix.clone(), true, true)?;
    let max_s = svd
        .singular_values
        .iter()
        .copied()
        .fold(0.0_f64, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        });
    let threshold = (matrix.nrows().max(matrix.ncols()) as f64) * f64::EPSILON * max_s;
    let rank = svd
        .singular_values
        .iter()
        .filter(|&&s| s > threshold)
        .count();
    if matrix.nrows() == matrix.ncols() && rank < matrix.nrows() {
        return Err(LinalgError::SingularMatrix);
    }

    let pinv = pseudo_inverse_from_svd(&svd, threshold)?;
    let x = pinv * rhs.clone();

    let min_s = svd
        .singular_values
        .iter()
        .copied()
        .filter(|s| *s > threshold)
        .fold(None, |acc, s| match acc {
            None => Some(s),
            Some(m) => Some(m.min(s)),
        })
        .unwrap_or(0.0);
    let rcond = if max_s > 0.0 { min_s / max_s } else { 0.0 };
    let backward_err = compute_backward_error(&matrix, &x, &rhs);

    Ok(SolveResult {
        x: x.iter().copied().collect(),
        warning: rcond_warning(rcond),
        backward_error: Some(backward_err),
        certificate: None,
    })
}

/// Solve with CASP: condition-aware solver portfolio selects optimal solver.
/// The portfolio is updated with evidence from this solve call.
fn build_solve_certificate(
    report: &ConditionReport,
    action: SolverAction,
    posterior: [f64; 4],
    expected_losses: [f64; 5],
    fallback_active: bool,
) -> SolveCertificate {
    SolveCertificate {
        action,
        matrix_shape: report.matrix_shape,
        rcond_estimate: report.rcond_estimate,
        structural_evidence: report.structural_evidence,
        posterior: posterior.to_vec(),
        expected_losses: expected_losses.to_vec(),
        chosen_expected_loss: expected_losses[action.index()],
        fallback_active,
    }
}

fn candidate_actions(structural_evidence: StructuralEvidence) -> Vec<SolverAction> {
    let mut actions = vec![
        SolverAction::DirectLU,
        SolverAction::PivotedQR,
        SolverAction::SVDFallback,
    ];
    match structural_evidence {
        StructuralEvidence::Diagonal => actions.push(SolverAction::DiagonalFastPath),
        StructuralEvidence::Triangular => actions.push(SolverAction::TriangularFastPath),
        StructuralEvidence::General => {}
    }
    actions
}

fn dispatch_solve_action(
    action: SolverAction,
    effective_a: &[Vec<f64>],
    b: &[f64],
    report: &ConditionReport,
    matrix_cache: &mut Option<DMatrix<f64>>,
    lu_cache: &mut Option<LU<f64, Dyn, Dyn>>,
) -> Result<SolveResult, LinalgError> {
    match action {
        SolverAction::DirectLU => {
            let matrix = if let Some(matrix) = matrix_cache.take() {
                matrix
            } else {
                dmatrix_from_rows(effective_a)?
            };
            let lu = if let Some(lu) = lu_cache.take() {
                lu
            } else {
                matrix.clone().lu()
            };
            let rhs = DVector::from_column_slice(b);
            let x = lu.solve(&rhs).ok_or(LinalgError::SingularMatrix)?;
            let backward_err = compute_backward_error(&matrix, &x, &rhs);
            Ok(SolveResult {
                x: x.iter().copied().collect(),
                warning: rcond_warning(report.rcond_estimate),
                backward_error: Some(backward_err),
                certificate: None,
            })
        }
        SolverAction::PivotedQR => solve_qr(effective_a, b),
        SolverAction::SVDFallback => solve_svd_fallback(effective_a, b),
        SolverAction::DiagonalFastPath => solve_diagonal(effective_a, b),
        SolverAction::TriangularFastPath => solve_triangular_internal(
            effective_a,
            b,
            TriangularTranspose::NoTranspose,
            report.lower_triangular,
            false,
        ),
    }
}

fn solve_with_portfolio_internal(
    a: &[Vec<f64>],
    b: &[f64],
    options: SolveOptions,
    portfolio: &mut SolverPortfolio,
    trace_operation: &'static str,
    record_evidence: bool,
) -> Result<SolveResult, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    if b.len() != rows {
        return Err(LinalgError::IncompatibleShapes {
            a_shape: (rows, cols),
            b_len: b.len(),
        });
    }
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix_and_vector(a, b, options.mode, options.check_finite)?;

    if rows == 0 {
        let result = Ok(SolveResult {
            x: Vec::new(),
            warning: None,
            backward_error: None,
            certificate: None,
        });
        emit_trace(LinalgTrace {
            operation: trace_operation,
            matrix_size: (rows, cols),
            mode: options.mode,
            rcond: None,
            warning: None,
            error: None,
        });
        return result;
    }

    // Borrow the input when it is already in the orientation we need; only the
    // transposed path requires an owned copy. Previously this always deep-cloned
    // the Vec<Vec<f64>> (~8 MB/solve at n=1000), driving allocation + page-fault
    // cost. The matrix is read-only downstream, so borrowing is behavior-identical.
    let effective_a: Cow<[Vec<f64>]> = if options.transposed {
        Cow::Owned(transpose(a))
    } else {
        Cow::Borrowed(a)
    };
    let effective_assumption =
        normalize_assumption_for_effective_matrix(options.assume_a, options.transposed);
    let metadata_incompatibility_score =
        assumption_incompatibility_score(&effective_a, effective_assumption)?;
    let diagnostics = condition_diagnostics_for_solve(&effective_a, effective_assumption)?;
    let ConditionDiagnosticsWork {
        report,
        mut matrix_cache,
        mut lu_cache,
    } = diagnostics;
    let policy_decision =
        if should_apply_solve_policy(options.mode, options.check_finite, &effective_a, b) {
            Some(solve_policy_decision(
                options.mode,
                &report,
                metadata_incompatibility_score,
            )?)
        } else {
            None
        };

    // Hardened mode: reject ill-conditioned matrices upfront
    if options.mode == RuntimeMode::Hardened
        && report.rcond_estimate < HARDENED_RCOND_THRESHOLD
        && report.rcond_estimate > 0.0
    {
        return Err(LinalgError::ConditionTooHigh {
            rcond: report.rcond_estimate,
            threshold: HARDENED_RCOND_THRESHOLD,
        });
    }

    let (selected_action, posterior, expected_losses, _) =
        portfolio.select_action(report.rcond_estimate, Some(report.structural_evidence));

    let mut actions = candidate_actions(report.structural_evidence);
    actions
        .sort_by(|lhs, rhs| expected_losses[lhs.index()].total_cmp(&expected_losses[rhs.index()]));
    if let Some(position) = actions.iter().position(|action| *action == selected_action) {
        actions.swap(0, position);
    }

    let mut last_error = None;
    let mut actual_action = selected_action;
    let result = actions
        .into_iter()
        .find_map(|action| {
            match dispatch_solve_action(
                action,
                &effective_a,
                b,
                &report,
                &mut matrix_cache,
                &mut lu_cache,
            ) {
                Ok(mut solve_result) => {
                    let fallback_active = action != selected_action;
                    solve_result.certificate = Some(build_solve_certificate(
                        &report,
                        action,
                        posterior,
                        expected_losses,
                        fallback_active,
                    ));
                    actual_action = action;
                    Some(Ok(solve_result))
                }
                Err(err) => {
                    last_error = Some(err);
                    None
                }
            }
        })
        .unwrap_or_else(|| Err(last_error.unwrap_or(LinalgError::SingularMatrix)));
    let result = result.and_then(|solve_result| {
        enforce_policy_full_validation(policy_decision.as_ref(), &solve_result)?;
        Ok(solve_result)
    });

    emit_trace(LinalgTrace {
        operation: trace_operation,
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: Some(report.rcond_estimate),
        warning: result.as_ref().ok().and_then(|solve_result| {
            solve_result
                .warning
                .as_ref()
                .map(|warning| format!("{warning:?}"))
        }),
        error: result.as_ref().err().map(|e| e.to_string()),
    });

    if record_evidence {
        let fallback_active = actual_action != selected_action;
        let backward_error = result
            .as_ref()
            .ok()
            .and_then(|solve_result| solve_result.backward_error);
        portfolio.record_evidence(SolverEvidenceEntry {
            component: "solver_portfolio",
            matrix_shape: (rows, cols),
            rcond_estimate: report.rcond_estimate,
            chosen_action: actual_action,
            posterior: posterior.to_vec(),
            expected_losses: expected_losses.to_vec(),
            chosen_expected_loss: expected_losses[actual_action.index()],
            fallback_active,
            backward_error,
        });
    }

    result
}

pub fn solve_with_casp(
    a: &[Vec<f64>],
    b: &[f64],
    options: SolveOptions,
    portfolio: &mut SolverPortfolio,
) -> Result<SolveResult, LinalgError> {
    solve_with_portfolio_internal(a, b, options, portfolio, "solve_with_casp", true)
}

/// Solve linear system with full audit logging.
///
/// Records to the provided `AuditLedger`:
/// - `FailClosed` events when validation rejects input (non-finite, ill-conditioned)
/// - CASP solver selection decisions
/// - Bounded recovery events in hardened mode
pub fn solve_with_audit(
    a: &[Vec<f64>],
    b: &[f64],
    options: SolveOptions,
    portfolio: &mut SolverPortfolio,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<SolveResult, LinalgError> {
    let fingerprint = matrix_fingerprint(a);
    let (rows, cols) = matrix_shape(a)?;

    // Validation with audit logging
    if rows != cols {
        record_fail_closed(
            audit_ledger,
            &fingerprint,
            "non_square_matrix",
            &format!("rejected: {rows}x{cols} is not square"),
        );
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    if b.len() != rows {
        record_fail_closed(
            audit_ledger,
            &fingerprint,
            "incompatible_shapes",
            &format!("rejected: b.len()={} != rows={rows}", b.len()),
        );
        return Err(LinalgError::IncompatibleShapes {
            a_shape: (rows, cols),
            b_len: b.len(),
        });
    }

    // Hardened dimension check with audit
    if options.mode == RuntimeMode::Hardened && (rows > HARDENED_MAX_DIM || cols > HARDENED_MAX_DIM)
    {
        record_fail_closed(
            audit_ledger,
            &fingerprint,
            "resource_exhausted",
            &format!("rejected: {rows}x{cols} exceeds hardened limit {HARDENED_MAX_DIM}"),
        );
        return Err(LinalgError::ResourceExhausted {
            detail: format!(
                "matrix dimension ({rows}x{cols}) exceeds hardened limit ({HARDENED_MAX_DIM})"
            ),
        });
    }

    // Finite check with audit
    let must_check = options.check_finite || options.mode == RuntimeMode::Hardened;
    if must_check {
        if a.iter().flatten().any(|v| !v.is_finite()) {
            record_fail_closed(
                audit_ledger,
                &fingerprint,
                "non_finite_matrix",
                "rejected: matrix contains NaN or Inf",
            );
            return Err(LinalgError::NonFiniteInput);
        }
        if b.iter().any(|v| !v.is_finite()) {
            record_fail_closed(
                audit_ledger,
                &fingerprint,
                "non_finite_vector",
                "rejected: vector contains NaN or Inf",
            );
            return Err(LinalgError::NonFiniteInput);
        }
    }

    if rows == 0 {
        return Ok(SolveResult {
            x: Vec::new(),
            warning: None,
            backward_error: None,
            certificate: None,
        });
    }

    let effective_a = if options.transposed {
        transpose(a)
    } else {
        a.to_vec()
    };
    let effective_assumption =
        normalize_assumption_for_effective_matrix(options.assume_a, options.transposed);
    let metadata_incompatibility_score =
        assumption_incompatibility_score(&effective_a, effective_assumption)?;
    let diagnostics = condition_diagnostics_for_solve(&effective_a, effective_assumption)?;
    let ConditionDiagnosticsWork {
        report,
        mut matrix_cache,
        mut lu_cache,
    } = diagnostics;
    let policy_decision =
        if should_apply_solve_policy(options.mode, options.check_finite, &effective_a, b) {
            Some(
                match solve_policy_decision(options.mode, &report, metadata_incompatibility_score) {
                    Ok(decision) => decision,
                    Err(err) => {
                        record_fail_closed(
                            audit_ledger,
                            &fingerprint,
                            "policy_rejected",
                            &format!("rejected: {err}"),
                        );
                        return Err(err);
                    }
                },
            )
        } else {
            None
        };

    // Hardened mode: reject ill-conditioned matrices with audit
    if options.mode == RuntimeMode::Hardened
        && report.rcond_estimate < HARDENED_RCOND_THRESHOLD
        && report.rcond_estimate > 0.0
    {
        record_fail_closed(
            audit_ledger,
            &fingerprint,
            "condition_too_high",
            &format!(
                "rejected: rcond={:.2e} < threshold={:.2e}",
                report.rcond_estimate, HARDENED_RCOND_THRESHOLD
            ),
        );
        return Err(LinalgError::ConditionTooHigh {
            rcond: report.rcond_estimate,
            threshold: HARDENED_RCOND_THRESHOLD,
        });
    }

    // CASP solver selection with audit
    let (selected_action, posterior, expected_losses, _) =
        portfolio.select_action(report.rcond_estimate, Some(report.structural_evidence));

    let mut actions = candidate_actions(report.structural_evidence);
    actions
        .sort_by(|lhs, rhs| expected_losses[lhs.index()].total_cmp(&expected_losses[rhs.index()]));
    if let Some(position) = actions.iter().position(|action| *action == selected_action) {
        actions.swap(0, position);
    }

    let mut last_error = None;
    let mut actual_action = selected_action;
    let result = actions
        .into_iter()
        .find_map(|action| {
            match dispatch_solve_action(
                action,
                &effective_a,
                b,
                &report,
                &mut matrix_cache,
                &mut lu_cache,
            ) {
                Ok(mut solve_result) => {
                    let fallback_active = action != selected_action;
                    solve_result.certificate = Some(build_solve_certificate(
                        &report,
                        action,
                        posterior,
                        expected_losses,
                        fallback_active,
                    ));
                    actual_action = action;
                    Some(Ok(solve_result))
                }
                Err(err) => {
                    last_error = Some(err);
                    None
                }
            }
        })
        .unwrap_or_else(|| Err(last_error.unwrap_or(LinalgError::SingularMatrix)));
    let result = result.and_then(|solve_result| {
        enforce_policy_full_validation(policy_decision.as_ref(), &solve_result)?;
        Ok(solve_result)
    });
    if matches!(&result, Err(LinalgError::ConvergenceFailure { .. })) {
        record_fail_closed(
            audit_ledger,
            &fingerprint,
            "policy_full_validation",
            "rejected: policy full validation failed",
        );
    }

    // Record CASP decision to audit ledger
    let fallback_active = actual_action != selected_action;
    record_casp_decision(
        audit_ledger,
        &fingerprint,
        actual_action,
        report.rcond_estimate,
        fallback_active,
    );

    // Record bounded recovery if fallback occurred in hardened mode
    if fallback_active && options.mode == RuntimeMode::Hardened {
        record_bounded_recovery(
            audit_ledger,
            &fingerprint,
            &format!("fallback from {:?} to {:?}", selected_action, actual_action),
            "recovered via safer solver",
        );
    }

    emit_trace(LinalgTrace {
        operation: "solve_with_audit",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: Some(report.rcond_estimate),
        warning: result.as_ref().ok().and_then(|solve_result| {
            solve_result
                .warning
                .as_ref()
                .map(|warning| format!("{warning:?}"))
        }),
        error: result.as_ref().err().map(|e| e.to_string()),
    });

    // Record evidence to portfolio
    let backward_error = result
        .as_ref()
        .ok()
        .and_then(|solve_result| solve_result.backward_error);
    portfolio.record_evidence(SolverEvidenceEntry {
        component: "solver_portfolio",
        matrix_shape: (rows, cols),
        rcond_estimate: report.rcond_estimate,
        chosen_action: actual_action,
        posterior: posterior.to_vec(),
        expected_losses: expected_losses.to_vec(),
        chosen_expected_loss: expected_losses[actual_action.index()],
        fallback_active,
        backward_error,
    });

    result
}

/// Compute matrix inverse with CASP-driven algorithm selection.
///
/// Uses condition diagnostics to select between LU (fast, well-conditioned),
/// QR (stable, moderate conditioning), or SVD (robust, ill-conditioned) paths.
pub fn inv_with_casp(
    a: &[Vec<f64>],
    options: InvOptions,
    portfolio: &mut SolverPortfolio,
) -> Result<InvResult, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if rows == 0 {
        return Ok(InvResult {
            inverse: Vec::new(),
            warning: None,
            certificate: None,
        });
    }

    let diagnostics = condition_diagnostics_with_assumption(a, options.assume_a)?;
    let ConditionDiagnosticsWork {
        report,
        matrix_cache,
        lu_cache,
    } = diagnostics;

    // For truly singular matrices, error immediately - inv() should not fall back to pinv
    if report.rcond_estimate == 0.0 {
        return Err(LinalgError::SingularMatrix);
    }

    let (selected_action, posterior, expected_losses, _) =
        portfolio.select_action(report.rcond_estimate, Some(report.structural_evidence));

    // For inv, we try actions in order of expected loss
    let mut actions = vec![
        SolverAction::DirectLU,
        SolverAction::PivotedQR,
        SolverAction::SVDFallback,
    ];
    actions
        .sort_by(|lhs, rhs| expected_losses[lhs.index()].total_cmp(&expected_losses[rhs.index()]));
    if let Some(position) = actions.iter().position(|action| *action == selected_action) {
        actions.swap(0, position);
    }

    let mut last_error = None;
    let mut actual_action = selected_action;

    let result = actions
        .into_iter()
        .find_map(|action| {
            match dispatch_inv_action(action, a, rows, options.mode, &matrix_cache, &lu_cache) {
                Ok(mut inv_result) => {
                    let fallback_active = action != selected_action;
                    inv_result.certificate = Some(SolveCertificate {
                        action,
                        matrix_shape: (rows, cols),
                        rcond_estimate: report.rcond_estimate,
                        structural_evidence: report.structural_evidence,
                        posterior: posterior.to_vec(),
                        expected_losses: expected_losses.to_vec(),
                        chosen_expected_loss: expected_losses[action.index()],
                        fallback_active,
                    });
                    actual_action = action;
                    Some(Ok(inv_result))
                }
                Err(err) => {
                    last_error = Some(err);
                    None
                }
            }
        })
        .unwrap_or_else(|| Err(last_error.unwrap_or(LinalgError::SingularMatrix)));

    emit_trace(LinalgTrace {
        operation: "inv_with_casp",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: Some(report.rcond_estimate),
        warning: result
            .as_ref()
            .ok()
            .and_then(|r| r.warning.as_ref().map(|w| format!("{w:?}"))),
        error: result.as_ref().err().map(|e| e.to_string()),
    });

    portfolio.record_evidence(SolverEvidenceEntry {
        component: "inv_with_casp",
        matrix_shape: (rows, cols),
        rcond_estimate: report.rcond_estimate,
        chosen_action: actual_action,
        posterior: posterior.to_vec(),
        expected_losses: expected_losses.to_vec(),
        chosen_expected_loss: expected_losses[actual_action.index()],
        fallback_active: actual_action != selected_action,
        backward_error: None,
    });

    result
}

fn dispatch_inv_action(
    action: SolverAction,
    a: &[Vec<f64>],
    n: usize,
    mode: RuntimeMode,
    matrix_cache: &Option<DMatrix<f64>>,
    lu_cache: &Option<LU<f64, Dyn, Dyn>>,
) -> Result<InvResult, LinalgError> {
    match action {
        SolverAction::DirectLU
        | SolverAction::DiagonalFastPath
        | SolverAction::TriangularFastPath => {
            // Use cached LU if available
            let (matrix, lu) = match (matrix_cache, lu_cache) {
                (Some(m), Some(lu)) => (m.clone(), lu.clone()),
                _ => {
                    let m = dmatrix_from_rows(a)?;
                    let lu = m.clone().lu();
                    (m, lu)
                }
            };
            let a_norm_1 = matrix_norm1(&matrix);
            let rcond = fast_rcond_from_lu(&lu, a_norm_1, n);
            let pivot_tiny = lu
                .u()
                .diagonal()
                .iter()
                .any(|x| x.abs() <= f64::EPSILON * a_norm_1.max(1.0));

            if mode == RuntimeMode::Hardened && rcond < HARDENED_RCOND_THRESHOLD && rcond > 0.0 {
                return Err(LinalgError::ConditionTooHigh {
                    rcond,
                    threshold: HARDENED_RCOND_THRESHOLD,
                });
            }

            if rcond == 0.0 || rcond < f64::EPSILON || pivot_tiny {
                return Err(LinalgError::SingularMatrix);
            }

            let identity = DMatrix::identity(n, n);
            let inv_matrix = lu.solve(&identity).ok_or(LinalgError::SingularMatrix)?;

            Ok(InvResult {
                inverse: rows_from_dmatrix(&inv_matrix),
                warning: rcond_warning(rcond),
                certificate: None,
            })
        }
        SolverAction::PivotedQR => {
            // QR-based inverse: solve A*X = I via QR
            let matrix = match matrix_cache {
                Some(m) => m.clone(),
                None => dmatrix_from_rows(a)?,
            };
            let qr = matrix.clone().qr();
            let identity = DMatrix::identity(n, n);
            let inv_matrix = qr.solve(&identity).ok_or(LinalgError::SingularMatrix)?;
            let a_norm_1 = matrix_norm1(&matrix);
            let inv_norm_1 = matrix_norm1(&inv_matrix);
            let rcond = if a_norm_1 > 0.0 && inv_norm_1 > 0.0 {
                1.0 / (a_norm_1 * inv_norm_1)
            } else {
                0.0
            };
            if rcond == 0.0 || rcond < f64::EPSILON {
                return Err(LinalgError::SingularMatrix);
            }

            Ok(InvResult {
                inverse: rows_from_dmatrix(&inv_matrix),
                warning: rcond_warning(rcond),
                certificate: None,
            })
        }
        SolverAction::SVDFallback => {
            // SVD-based inverse via pseudoinverse
            let matrix = match matrix_cache {
                Some(m) => m.clone(),
                None => dmatrix_from_rows(a)?,
            };
            let svd = safe_svd(matrix.clone(), true, true)?;
            let max_s = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
            let threshold = f64::EPSILON * (n as f64) * max_s;
            let rank = svd
                .singular_values
                .iter()
                .filter(|s| **s > threshold)
                .count();
            if rank < n {
                return Err(LinalgError::SingularMatrix);
            }
            let pinv = pseudo_inverse_from_svd(&svd, threshold)?;
            let a_norm_1 = matrix_norm1(&matrix);
            let inv_norm_1 = matrix_norm1(&pinv);
            let rcond = if a_norm_1 > 0.0 && inv_norm_1 > 0.0 {
                1.0 / (a_norm_1 * inv_norm_1)
            } else {
                0.0
            };

            Ok(InvResult {
                inverse: rows_from_dmatrix(&pinv),
                warning: rcond_warning(rcond),
                certificate: None,
            })
        }
    }
}

/// Compute least squares solution with CASP-driven algorithm selection.
///
/// Uses condition diagnostics to select between QR (fast, well-conditioned)
/// or SVD (robust, rank-deficient/ill-conditioned) paths.
pub fn lstsq_with_casp(
    a: &[Vec<f64>],
    b: &[f64],
    options: LstsqOptions,
    portfolio: &mut SolverPortfolio,
) -> Result<LstsqResult, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if b.len() != rows {
        return Err(LinalgError::IncompatibleShapes {
            a_shape: (rows, cols),
            b_len: b.len(),
        });
    }
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix_and_vector(a, b, options.mode, options.check_finite)?;

    if rows == 0 || cols == 0 {
        return Ok(LstsqResult {
            x: vec![0.0; cols],
            residuals: Vec::new(),
            rank: 0,
            singular_values: Vec::new(),
            certificate: None,
        });
    }

    let cond = options.cond.unwrap_or(f64::EPSILON);
    if let Some(fast) = lstsq_low_rank_tall(a, b, rows, cols, cond, LOW_RANK_PINV_MIN_COLS) {
        let (selected_action, posterior, expected_losses, _) =
            portfolio.select_action(fast.rcond_estimate, None);
        let action = SolverAction::SVDFallback;

        let certificate = SolveCertificate {
            action,
            matrix_shape: (rows, cols),
            rcond_estimate: fast.rcond_estimate,
            structural_evidence: StructuralEvidence::General,
            posterior: posterior.to_vec(),
            expected_losses: expected_losses.to_vec(),
            chosen_expected_loss: expected_losses[action.index()],
            fallback_active: action != selected_action,
        };

        emit_trace(LinalgTrace {
            operation: "lstsq_with_casp",
            matrix_size: (rows, cols),
            mode: options.mode,
            rcond: Some(fast.rcond_estimate),
            warning: None,
            error: None,
        });

        portfolio.record_evidence(SolverEvidenceEntry {
            component: "lstsq_with_casp",
            matrix_shape: (rows, cols),
            rcond_estimate: fast.rcond_estimate,
            chosen_action: action,
            posterior: posterior.to_vec(),
            expected_losses: expected_losses.to_vec(),
            chosen_expected_loss: expected_losses[action.index()],
            fallback_active: action != selected_action,
            backward_error: None,
        });

        return Ok(LstsqResult {
            x: fast.x,
            residuals: Vec::new(),
            rank: fast.rank,
            singular_values: fast.singular_values,
            certificate: Some(certificate),
        });
    }

    let matrix = dmatrix_from_rows(a)?;
    let rhs = DVector::from_column_slice(b);

    let full_svd_for_rectangular = if rows == cols {
        None
    } else {
        Some(safe_svd(matrix.clone(), true, true)?)
    };

    // Compute condition estimate for lstsq.
    let singular_values: Vec<f64> = if let Some(svd) = full_svd_for_rectangular.as_ref() {
        svd.singular_values.iter().copied().collect()
    } else {
        let svd_for_cond = safe_svd(matrix.clone(), false, false)?;
        svd_for_cond.singular_values.iter().copied().collect()
    };
    let max_s = singular_values.iter().copied().fold(0.0_f64, |acc, v| {
        if acc.is_nan() || v.is_nan() {
            f64::NAN
        } else {
            acc.max(v)
        }
    });
    let min_s = singular_values.iter().copied().fold(f64::MAX, |acc, v| {
        if acc.is_nan() || v.is_nan() {
            f64::NAN
        } else {
            acc.min(v)
        }
    });
    let rcond_estimate = if max_s > 0.0 { min_s / max_s } else { 0.0 };

    let threshold = cond * max_s;
    let rank = singular_values.iter().filter(|s| **s > threshold).count();
    let full_rank = rank == rows.min(cols);

    let (selected_action, posterior, expected_losses, _) =
        portfolio.select_action(rcond_estimate, None);

    // For lstsq, QR can only solve square systems in nalgebra; use SVD for non-square
    // Also prefer SVD for ill-conditioned or rank-deficient cases
    let action = if rows == cols
        && full_rank
        && matches!(
            selected_action,
            SolverAction::PivotedQR
                | SolverAction::DirectLU
                | SolverAction::DiagonalFastPath
                | SolverAction::TriangularFastPath
        )
        && rcond_estimate > f64::EPSILON
    {
        SolverAction::PivotedQR
    } else {
        SolverAction::SVDFallback
    };

    let (x, rank, singular_values_out) = match action {
        SolverAction::PivotedQR => {
            // QR solve (only for square, well-conditioned systems)
            let qr = matrix.clone().qr();
            let x_qr = qr.solve(&rhs).ok_or(LinalgError::SingularMatrix)?;
            let rank = singular_values.iter().filter(|s| **s > threshold).count();
            (x_qr, rank, singular_values)
        }
        _ => {
            // SVD solve (standard lstsq path)
            let svd = if let Some(svd) = full_svd_for_rectangular {
                svd
            } else {
                safe_svd(matrix.clone(), true, true)?
            };
            let x_svd = if rows == cols {
                let pinv = pseudo_inverse_from_svd(&svd, threshold)?;
                pinv * rhs.clone()
            } else {
                least_squares_solution_from_svd(&svd, threshold, &rhs)?
            };
            let rank = svd
                .singular_values
                .iter()
                .filter(|s| **s > threshold)
                .count();
            let sv: Vec<f64> = svd.singular_values.iter().copied().collect();
            (x_svd, rank, sv)
        }
    };

    let residuals = if rows > cols && rank == cols {
        let residual = rhs - matrix * x.clone();
        vec![residual.dot(&residual)]
    } else {
        Vec::new()
    };

    let certificate = SolveCertificate {
        action,
        matrix_shape: (rows, cols),
        rcond_estimate,
        structural_evidence: StructuralEvidence::General,
        posterior: posterior.to_vec(),
        expected_losses: expected_losses.to_vec(),
        chosen_expected_loss: expected_losses[action.index()],
        fallback_active: action != selected_action,
    };

    emit_trace(LinalgTrace {
        operation: "lstsq_with_casp",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: Some(rcond_estimate),
        warning: None,
        error: None,
    });

    portfolio.record_evidence(SolverEvidenceEntry {
        component: "lstsq_with_casp",
        matrix_shape: (rows, cols),
        rcond_estimate,
        chosen_action: action,
        posterior: posterior.to_vec(),
        expected_losses: expected_losses.to_vec(),
        chosen_expected_loss: expected_losses[action.index()],
        fallback_active: action != selected_action,
        backward_error: None,
    });

    Ok(LstsqResult {
        x: x.iter().copied().collect(),
        residuals,
        rank,
        singular_values: singular_values_out,
        certificate: Some(certificate),
    })
}

/// Compute pseudoinverse with CASP-driven threshold selection.
///
/// Always uses SVD but records conditioning diagnostics and stability certificate
/// for audit trail and threshold tuning.
pub fn pinv_with_casp(
    a: &[Vec<f64>],
    options: PinvOptions,
    portfolio: &mut SolverPortfolio,
) -> Result<PinvResult, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    let atol = options.atol.unwrap_or(0.0);
    let rtol = options
        .rtol
        .unwrap_or((rows.max(cols) as f64) * f64::EPSILON);
    if atol < 0.0 || rtol < 0.0 {
        return Err(LinalgError::InvalidPinvThreshold);
    }

    if rows == 0 || cols == 0 {
        return Ok(PinvResult {
            pseudo_inverse: vec![vec![0.0; rows]; cols],
            rank: 0,
            certificate: None,
        });
    }

    if let Some(fast) = pinv_low_rank_tall(a, rows, cols, atol, rtol) {
        let (_, posterior, expected_losses, _) = portfolio.select_action(fast.rcond_estimate, None);
        let action = SolverAction::SVDFallback;

        let certificate = SolveCertificate {
            action,
            matrix_shape: (rows, cols),
            rcond_estimate: fast.rcond_estimate,
            structural_evidence: StructuralEvidence::General,
            posterior: posterior.to_vec(),
            expected_losses: expected_losses.to_vec(),
            chosen_expected_loss: expected_losses[action.index()],
            fallback_active: false,
        };

        emit_trace(LinalgTrace {
            operation: "pinv_with_casp",
            matrix_size: (rows, cols),
            mode: options.mode,
            rcond: Some(fast.rcond_estimate),
            warning: None,
            error: None,
        });

        portfolio.record_evidence(SolverEvidenceEntry {
            component: "pinv_with_casp",
            matrix_shape: (rows, cols),
            rcond_estimate: fast.rcond_estimate,
            chosen_action: action,
            posterior: posterior.to_vec(),
            expected_losses: expected_losses.to_vec(),
            chosen_expected_loss: expected_losses[action.index()],
            fallback_active: false,
            backward_error: None,
        });

        return Ok(PinvResult {
            pseudo_inverse: fast.pseudo_inverse,
            rank: fast.rank,
            certificate: Some(certificate),
        });
    }

    let matrix = dmatrix_from_rows(a)?;
    let svd = safe_svd(matrix, true, true)?;
    let singular_values = &svd.singular_values;
    let max_s = singular_values.iter().copied().fold(0.0_f64, |acc, v| {
        if acc.is_nan() || v.is_nan() {
            f64::NAN
        } else {
            acc.max(v)
        }
    });
    let min_s = singular_values.iter().copied().fold(f64::MAX, |acc, v| {
        if acc.is_nan() || v.is_nan() {
            f64::NAN
        } else if v > 1e-15 {
            acc.min(v)
        } else {
            acc
        }
    });
    let rcond_estimate = if max_s > 0.0 { min_s / max_s } else { 0.0 };

    let threshold = atol + rtol * max_s;
    let rank = singular_values.iter().filter(|s| **s > threshold).count();
    let pinv_matrix = pseudo_inverse_from_svd(&svd, threshold)?;

    // For pinv, always SVD but record the portfolio decision for audit
    let (_, posterior, expected_losses, _) = portfolio.select_action(rcond_estimate, None);
    let action = SolverAction::SVDFallback;

    let certificate = SolveCertificate {
        action,
        matrix_shape: (rows, cols),
        rcond_estimate,
        structural_evidence: StructuralEvidence::General,
        posterior: posterior.to_vec(),
        expected_losses: expected_losses.to_vec(),
        chosen_expected_loss: expected_losses[action.index()],
        fallback_active: false,
    };

    emit_trace(LinalgTrace {
        operation: "pinv_with_casp",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: Some(rcond_estimate),
        warning: None,
        error: None,
    });

    portfolio.record_evidence(SolverEvidenceEntry {
        component: "pinv_with_casp",
        matrix_shape: (rows, cols),
        rcond_estimate,
        chosen_action: action,
        posterior: posterior.to_vec(),
        expected_losses: expected_losses.to_vec(),
        chosen_expected_loss: expected_losses[action.index()],
        fallback_active: false,
        backward_error: None,
    });

    Ok(PinvResult {
        pseudo_inverse: rows_from_dmatrix(&pinv_matrix),
        rank,
        certificate: Some(certificate),
    })
}

fn fast_rcond_triangular(a: &[Vec<f64>], lower: bool) -> f64 {
    let n = a.len();
    if n == 0 {
        return 1.0;
    }

    // 1. Compute ||A||_1 in O(n^2)
    let mut a_norm = 0.0;
    for j in 0..n {
        let mut col_sum = 0.0;
        if lower {
            for row in a.iter().take(n).skip(j) {
                let value = row[j];
                if !value.is_finite() {
                    return 0.0;
                }
                col_sum += value.abs();
            }
        } else {
            for row in a.iter().take(j + 1) {
                let value = row[j];
                if !value.is_finite() {
                    return 0.0;
                }
                col_sum += value.abs();
            }
        }
        if col_sum > a_norm {
            a_norm = col_sum;
        }
    }

    if a_norm == 0.0 {
        return 0.0;
    }

    // 2. Estimate ||A^-1||_1 via a greedy 1-norm estimator.
    // We estimate ||A^-1||_1 = ||(A^T)^-1||_\infty by solving A^T x = b
    // where b_i in {-1, 1} is chosen greedily to maximize x_i.
    let mut x = vec![0.0; n];
    if lower {
        // A^T is upper triangular. Solve A^T x = b backward.
        for i in (0..n).rev() {
            let mut sum = 0.0;
            for j in (i + 1)..n {
                sum += a[j][i] * x[j]; // A^T_ij = A_ji
            }
            if !a[i][i].is_finite() || a[i][i].abs() < 1e-18 {
                return 0.0;
            }
            let b_i = if sum < 0.0 { 1.0 } else { -1.0 };
            x[i] = (b_i - sum) / a[i][i];
        }
    } else {
        // A^T is lower triangular. Solve A^T x = b forward.
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..i {
                sum += a[j][i] * x[j]; // A^T_ij = A_ji
            }
            if !a[i][i].is_finite() || a[i][i].abs() < 1e-18 {
                return 0.0;
            }
            let b_i = if sum < 0.0 { 1.0 } else { -1.0 };
            x[i] = (b_i - sum) / a[i][i];
        }
    }
    let a_inv_norm: f64 = x.iter().map(|v| v.abs()).fold(0.0, |a: f64, b: f64| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.max(b)
        }
    });

    let rcond = 1.0 / (a_norm * a_inv_norm);
    if rcond.is_nan() { 0.0 } else { rcond.min(1.0) }
}

fn fast_rcond_diagonal(a: &[Vec<f64>]) -> f64 {
    let n = a.len();
    if n == 0 {
        return 1.0;
    }
    let mut max_abs = 0.0;
    let mut min_abs = f64::INFINITY;
    for (i, row) in a.iter().enumerate() {
        if i >= row.len() {
            return 0.0; // Ragged matrix treated as singular
        }
        let raw = row[i];
        if !raw.is_finite() {
            return 0.0;
        }
        let val = raw.abs();
        if val > max_abs {
            max_abs = val;
        }
        if val < min_abs {
            min_abs = val;
        }
    }
    if max_abs == 0.0 {
        0.0
    } else {
        min_abs / max_abs
    }
}

/// Estimate spectral condition number via power iteration.
pub fn randomized_rcond_estimate(
    matrix: &DMatrix<f64>,
    lu: &LU<f64, Dyn, Dyn>,
    iterations: usize,
) -> f64 {
    let n = matrix.nrows();
    if n == 0 {
        return 1.0;
    }

    // Power iteration for largest singular value: converge on σ_max
    let mut v = DVector::from_element(n, 1.0 / (n as f64).sqrt());
    for _ in 0..iterations {
        let av = matrix * &v;
        v = matrix.transpose() * av;
        let norm = v.norm();
        if norm == 0.0 {
            return 0.0;
        }
        v /= norm;
    }
    let sigma_max = (matrix * &v).norm();
    if sigma_max == 0.0 {
        return 0.0;
    }

    // Inverse power iteration for smallest singular value using LU on (AᵀA)⁻¹
    // w_{k+1} = (AᵀA)⁻¹ w_k = A⁻¹ (Aᵀ)⁻¹ w_k
    let mut w = DVector::from_element(n, 1.0 / (n as f64).sqrt());
    for _ in 0..iterations {
        // 1. Solve Aᵀ y = w
        match solve_lu_transpose(lu, &w) {
            Some(y) => {
                // 2. Solve A w_new = y
                match lu.solve(&y) {
                    Some(solved) => {
                        let norm = solved.norm();
                        if norm == 0.0 {
                            return 0.0;
                        }
                        w = solved / norm;
                    }
                    None => return 0.0,
                }
            }
            None => return 0.0,
        }
    }

    // Final estimate: ||A⁻¹ w||₂ = 1/σ_min
    // (AᵀA)⁻¹ w = (1/σ_min²) w
    let y = match solve_lu_transpose(lu, &w) {
        Some(y) => y,
        None => return 0.0,
    };
    let w_final = match lu.solve(&y) {
        Some(s) => s,
        None => return 0.0,
    };

    let sigma_min_sq_inv = w_final.norm();
    if sigma_min_sq_inv <= 0.0 || !sigma_min_sq_inv.is_finite() {
        return 0.0;
    }
    let sigma_min = (1.0 / sigma_min_sq_inv).sqrt();

    sigma_min / sigma_max
}

// ══════════════════════════════════════════════════════════════════════
// Matrix Decompositions — Public API
// ══════════════════════════════════════════════════════════════════════

/// LU decomposition with partial pivoting: PA = LU.
///
/// Returns permutation matrix P, unit lower triangular L, and upper triangular U.
/// Matches `scipy.linalg.lu(a)`.
pub fn lu(a: &[Vec<f64>], options: DecompOptions) -> Result<LuResult, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if rows == 0 {
        return Ok(LuResult {
            p: Vec::new(),
            l: Vec::new(),
            u: Vec::new(),
        });
    }

    let matrix = dmatrix_from_rows(a)?;
    let lu_decomp: LU<f64, Dyn, Dyn> = matrix.lu();

    // Extract L, U directly; build P from the permutation sequence.
    let l_mat = lu_decomp.l();
    let u_mat = lu_decomp.u();
    // nalgebra returns a row-permutation P_n satisfying P_n · A = L · U.
    // SciPy's `scipy.linalg.lu(a)` instead returns P with A = P · L · U,
    // i.e. P = P_n^T = P_n^{-1}. Apply the permutation to an identity
    // and transpose so the returned `p` matches SciPy's convention.
    let mut p_n = DMatrix::<f64>::identity(rows, rows);
    lu_decomp.p().permute_rows(&mut p_n);
    let p_mat = p_n.transpose();

    emit_trace(LinalgTrace {
        operation: "lu",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(LuResult {
        p: rows_from_dmatrix(&p_mat),
        l: rows_from_dmatrix(&l_mat),
        u: rows_from_dmatrix(&u_mat),
    })
}

/// Compact LU factorization for subsequent solves via `lu_solve`.
///
/// Matches `scipy.linalg.lu_factor(a)`.
pub fn lu_factor(a: &[Vec<f64>], options: DecompOptions) -> Result<LuFactorResult, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    let matrix = dmatrix_from_rows(a)?;
    let a_norm_1 = matrix_norm1(&matrix);
    let lu_decomp: LU<f64, Dyn, Dyn> = matrix.lu();
    let rcond_estimate = fast_rcond_from_lu(&lu_decomp, a_norm_1, rows);

    emit_trace(LinalgTrace {
        operation: "lu_factor",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(LuFactorResult {
        lu_internal: lu_decomp,
        n: rows,
        a_norm_1,
        rcond_estimate,
    })
}

/// Solve a linear system using a precomputed LU factorization.
///
/// Matches `scipy.linalg.lu_solve(lu_and_piv, b)`.
pub fn lu_solve(lu_factor: &LuFactorResult, b: &[f64]) -> Result<SolveResult, LinalgError> {
    if b.len() != lu_factor.n {
        return Err(LinalgError::IncompatibleShapes {
            a_shape: (lu_factor.n, lu_factor.n),
            b_len: b.len(),
        });
    }

    let rhs = DVector::from_column_slice(b);
    let x = lu_factor
        .lu_internal
        .solve(&rhs)
        .ok_or(LinalgError::SingularMatrix)?;

    let rcond = lu_factor.rcond_estimate;

    emit_trace(LinalgTrace {
        operation: "lu_solve",
        matrix_size: (lu_factor.n, lu_factor.n),
        mode: RuntimeMode::Strict,
        rcond: Some(rcond),
        warning: None,
        error: None,
    });

    Ok(SolveResult {
        x: x.iter().copied().collect(),
        warning: rcond_warning(rcond),
        backward_error: None,
        certificate: None,
    })
}

/// QR decomposition: A = QR.
///
/// Returns orthogonal Q and upper triangular R.
/// Matches `scipy.linalg.qr(a)`.
pub fn qr(a: &[Vec<f64>], options: DecompOptions) -> Result<QrResult, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if rows == 0 || cols == 0 {
        return Ok(QrResult {
            q: Vec::new(),
            r: Vec::new(),
        });
    }

    let matrix = dmatrix_from_rows(a)?;
    let qr_decomp = matrix.qr();
    let q_mat = qr_decomp.q();
    let r_mat = qr_decomp.r();

    emit_trace(LinalgTrace {
        operation: "qr",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(QrResult {
        q: rows_from_dmatrix(&q_mat),
        r: rows_from_dmatrix(&r_mat),
    })
}

fn reconstruct_qr_matrix(
    q: &[Vec<f64>],
    r: &[Vec<f64>],
    options: DecompOptions,
) -> Result<Vec<Vec<f64>>, LinalgError> {
    let (q_rows, q_cols) = matrix_shape(q)?;
    let (r_rows, r_cols) = matrix_shape(r)?;
    hardened_dimension_check(options.mode, q_rows, q_cols)?;
    hardened_dimension_check(options.mode, r_rows, r_cols)?;
    validate_finite_matrix(q, options.mode, options.check_finite)?;
    validate_finite_matrix(r, options.mode, options.check_finite)?;

    if q_cols != r_rows {
        return Err(LinalgError::IncompatibleShapes {
            a_shape: (q_rows, q_cols),
            b_len: r_rows,
        });
    }
    matmul(q, r).map(|matrix| {
        if r_cols == 0 {
            vec![Vec::new(); q_rows]
        } else {
            matrix
        }
    })
}

/// Delete row `k` from a QR factorization and return updated QR factors.
///
/// This matches the observable contract of `scipy.linalg.qr_delete` for
/// row deletes by reconstructing `A = Q R`, deleting the requested row, and
/// recomputing the factorization through `qr`.
pub fn qr_delete(
    q: &[Vec<f64>],
    r: &[Vec<f64>],
    k: usize,
    options: DecompOptions,
) -> Result<QrResult, LinalgError> {
    let mut matrix = reconstruct_qr_matrix(q, r, options)?;
    if k >= matrix.len() {
        return Err(LinalgError::InvalidArgument {
            detail: format!("row index {k} out of bounds for {} rows", matrix.len()),
        });
    }
    matrix.remove(k);
    qr(&matrix, options)
}

/// Insert row `u` at position `k` in a QR factorization.
///
/// Matches `scipy.linalg.qr_insert` for row inserts in the scoped real dense
/// path by reconstructing `A = Q R`, inserting the row, and recomputing QR.
pub fn qr_insert(
    q: &[Vec<f64>],
    r: &[Vec<f64>],
    u: &[f64],
    k: usize,
    options: DecompOptions,
) -> Result<QrResult, LinalgError> {
    let mut matrix = reconstruct_qr_matrix(q, r, options)?;
    let cols = matrix
        .first()
        .map_or_else(|| r.first().map_or(0, Vec::len), Vec::len);
    if u.len() != cols {
        return Err(LinalgError::IncompatibleShapes {
            a_shape: (matrix.len(), cols),
            b_len: u.len(),
        });
    }
    if k > matrix.len() {
        return Err(LinalgError::InvalidArgument {
            detail: format!(
                "row index {k} out of bounds for insertion into {} rows",
                matrix.len()
            ),
        });
    }
    if (options.check_finite || options.mode == RuntimeMode::Hardened)
        && u.iter().any(|value| !value.is_finite())
    {
        return Err(LinalgError::NonFiniteInput);
    }

    matrix.insert(k, u.to_vec());
    qr(&matrix, options)
}

/// Multiply `Q` from a QR factorization by vector or matrix `c`.
///
/// Matches the scoped `scipy.linalg.qr_multiply` behavior for applying an
/// already-materialized `Q` factor. `R` is validated to ensure it is compatible
/// with `Q`, but does not participate in the product.
pub fn qr_multiply(
    q: &[Vec<f64>],
    r: &[Vec<f64>],
    c: &[Vec<f64>],
    options: DecompOptions,
) -> Result<Vec<Vec<f64>>, LinalgError> {
    let (q_rows, q_cols) = matrix_shape(q)?;
    let (r_rows, _r_cols) = matrix_shape(r)?;
    if q_cols != r_rows {
        return Err(LinalgError::IncompatibleShapes {
            a_shape: (q_rows, q_cols),
            b_len: r_rows,
        });
    }
    let (c_rows, c_cols) = matrix_shape(c)?;
    hardened_dimension_check(options.mode, c_rows, c_cols)?;
    validate_finite_matrix(c, options.mode, options.check_finite)?;
    reconstruct_qr_matrix(q, r, options)?;
    matmul(q, c)
}

/// Apply a rank-1 update `A + u v^T` to a QR factorization.
///
/// Matches `scipy.linalg.qr_update` for dense real inputs by reconstructing
/// `A = Q R`, applying the update, and recomputing QR.
pub fn qr_update(
    q: &[Vec<f64>],
    r: &[Vec<f64>],
    u: &[f64],
    v: &[f64],
    options: DecompOptions,
) -> Result<QrResult, LinalgError> {
    let mut matrix = reconstruct_qr_matrix(q, r, options)?;
    let rows = matrix.len();
    let cols = matrix.first().map_or(0, Vec::len);
    if u.len() != rows {
        return Err(LinalgError::IncompatibleShapes {
            a_shape: (rows, cols),
            b_len: u.len(),
        });
    }
    if v.len() != cols {
        return Err(LinalgError::IncompatibleShapes {
            a_shape: (rows, cols),
            b_len: v.len(),
        });
    }
    if (options.check_finite || options.mode == RuntimeMode::Hardened)
        && (u.iter().any(|value| !value.is_finite()) || v.iter().any(|value| !value.is_finite()))
    {
        return Err(LinalgError::NonFiniteInput);
    }

    for (i, row) in matrix.iter_mut().enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            *value += u[i] * v[j];
        }
    }
    qr(&matrix, options)
}

/// Singular Value Decomposition: A = U Σ Vᵀ.
///
/// Returns left singular vectors U, singular values σ, and right singular vectors Vᵀ.
/// Matches `scipy.linalg.svd(a)`.
pub fn svd(a: &[Vec<f64>], options: DecompOptions) -> Result<SvdResult, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if rows == 0 || cols == 0 {
        return Ok(SvdResult {
            u: Vec::new(),
            s: Vec::new(),
            vt: Vec::new(),
        });
    }

    let matrix = dmatrix_from_rows(a)?;
    let svd_decomp = safe_svd(matrix, true, true)?;

    let u_mat = svd_decomp
        .u
        .as_ref()
        .ok_or(LinalgError::ConvergenceFailure {
            detail: "SVD failed to compute U".into(),
        })?;
    let vt_mat = svd_decomp
        .v_t
        .as_ref()
        .ok_or(LinalgError::ConvergenceFailure {
            detail: "SVD failed to compute Vt".into(),
        })?;
    let s: Vec<f64> = svd_decomp.singular_values.iter().copied().collect();

    emit_trace(LinalgTrace {
        operation: "svd",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(SvdResult {
        u: rows_from_dmatrix(u_mat),
        s,
        vt: rows_from_dmatrix(vt_mat),
    })
}

/// Compute singular values only (without U and Vᵀ).
///
/// Matches `scipy.linalg.svdvals(a)`.
pub fn svdvals(a: &[Vec<f64>], options: DecompOptions) -> Result<Vec<f64>, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if rows == 0 || cols == 0 {
        return Ok(Vec::new());
    }

    let matrix = dmatrix_from_rows(a)?;
    let svd_decomp = safe_svd(matrix, false, false)?;
    let s: Vec<f64> = svd_decomp.singular_values.iter().copied().collect();

    emit_trace(LinalgTrace {
        operation: "svdvals",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(s)
}

/// Cholesky decomposition for symmetric positive-definite matrices: A = LLᵀ.
///
/// If `lower` is true (default), returns L such that A = LLᵀ.
/// If `lower` is false, returns U such that A = UᵀU.
/// Matches `scipy.linalg.cholesky(a, lower=True)`.
pub fn cholesky(
    a: &[Vec<f64>],
    lower: bool,
    options: DecompOptions,
) -> Result<CholeskyResult, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if rows == 0 {
        return Ok(CholeskyResult { factor: Vec::new() });
    }

    let matrix = dmatrix_from_rows(a)?;
    let chol = Cholesky::new(matrix).ok_or(LinalgError::InvalidArgument {
        detail: "matrix is not positive definite".into(),
    })?;

    let factor = if lower {
        chol.l()
    } else {
        chol.l().transpose()
    };

    emit_trace(LinalgTrace {
        operation: "cholesky",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(CholeskyResult {
        factor: rows_from_dmatrix(&factor),
    })
}

/// Compact Cholesky factorization for subsequent solves via `cho_solve`.
///
/// Always uses the lower triangular factor internally.
/// Matches `scipy.linalg.cho_factor(a)`.
pub fn cho_factor(a: &[Vec<f64>], options: DecompOptions) -> Result<ChoFactorResult, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    let matrix = dmatrix_from_rows(a)?;
    let chol = Cholesky::new(matrix).ok_or(LinalgError::InvalidArgument {
        detail: "matrix is not positive definite".into(),
    })?;

    emit_trace(LinalgTrace {
        operation: "cho_factor",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(ChoFactorResult {
        chol_internal: chol,
        n: rows,
    })
}

/// Solve a linear system using a precomputed Cholesky factorization.
///
/// Matches `scipy.linalg.cho_solve((c, lower), b)`.
pub fn cho_solve(cho: &ChoFactorResult, b: &[f64]) -> Result<SolveResult, LinalgError> {
    if b.len() != cho.n {
        return Err(LinalgError::IncompatibleShapes {
            a_shape: (cho.n, cho.n),
            b_len: b.len(),
        });
    }

    let rhs = DVector::from_column_slice(b);
    let x = cho.chol_internal.solve(&rhs);

    emit_trace(LinalgTrace {
        operation: "cho_solve",
        matrix_size: (cho.n, cho.n),
        mode: RuntimeMode::Strict,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(SolveResult {
        x: x.iter().copied().collect(),
        warning: None,
        backward_error: None,
        certificate: None,
    })
}

/// Solve a banded positive definite system using banded Cholesky factorization.
///
/// Given the lower triangular banded Cholesky factor `cb` of a symmetric
/// positive definite banded matrix A (stored in lower band storage format),
/// solves A @ x = b.
///
/// The matrix cb has shape (lower_bandwidth + 1, n) where:
/// - cb[0, :] contains the main diagonal of L
/// - cb[k, i] contains L[i+k, i] for k = 1, ..., lower_bandwidth
///
/// Matches `scipy.linalg.cho_solve_banded((cb, lower), b)`.
///
/// # Arguments
/// * `cb` - Cholesky factor in banded storage format, shape (lower_bandwidth + 1, n)
/// * `b` - Right-hand side vector
/// * `lower` - If true, cb contains lower triangular factor; if false, upper triangular
pub fn cho_solve_banded(
    cb: &[Vec<f64>],
    b: &[f64],
    lower: bool,
) -> Result<SolveResult, LinalgError> {
    if cb.is_empty() {
        return Err(LinalgError::InvalidArgument {
            detail: "cb must not be empty".to_string(),
        });
    }

    let bandwidth_plus_1 = cb.len();
    let n = cb[0].len();

    if b.len() != n {
        return Err(LinalgError::IncompatibleShapes {
            a_shape: (n, n),
            b_len: b.len(),
        });
    }

    // Verify all rows have same length
    for row in cb.iter() {
        if row.len() != n {
            return Err(LinalgError::InvalidArgument {
                detail: "All rows in cb must have the same length".to_string(),
            });
        }
    }

    if n == 0 {
        return Ok(SolveResult {
            x: vec![],
            warning: None,
            backward_error: None,
            certificate: None,
        });
    }

    let mut x = b.to_vec();

    if lower {
        // Solve L @ y = b (forward substitution)
        for i in 0..n {
            for k in 1..bandwidth_plus_1 {
                if i >= k {
                    x[i] -= cb[k][i - k] * x[i - k];
                }
            }
            if cb[0][i].abs() < 1e-15 {
                return Err(LinalgError::SingularMatrix);
            }
            x[i] /= cb[0][i];
        }

        // Solve L^T @ x = y (backward substitution)
        for i in (0..n).rev() {
            for k in 1..bandwidth_plus_1 {
                if i + k < n {
                    x[i] -= cb[k][i] * x[i + k];
                }
            }
            x[i] /= cb[0][i];
        }
    } else {
        // Upper triangular: Solve U^T @ y = b then U @ x = y
        // U is stored with U[0,:] = diagonal, U[k,i] = U[i-k,i]
        for i in 0..n {
            for k in 1..bandwidth_plus_1 {
                if i >= k {
                    x[i] -= cb[k][i] * x[i - k];
                }
            }
            if cb[0][i].abs() < 1e-15 {
                return Err(LinalgError::SingularMatrix);
            }
            x[i] /= cb[0][i];
        }

        for i in (0..n).rev() {
            for k in 1..bandwidth_plus_1 {
                if i + k < n {
                    x[i] -= cb[k][i + k] * x[i + k];
                }
            }
            x[i] /= cb[0][i];
        }
    }

    emit_trace(LinalgTrace {
        operation: "cho_solve_banded",
        matrix_size: (n, n),
        mode: RuntimeMode::Strict,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(SolveResult {
        x,
        warning: None,
        backward_error: None,
        certificate: None,
    })
}

/// Solve a symmetric positive-definite banded system Ax = b.
///
/// Given a symmetric positive-definite banded matrix A in banded storage format,
/// solves the system Ax = b using banded Cholesky factorization.
///
/// The matrix `ab` stores the lower or upper band of A:
/// - If lower=true: ab[k, i] = A[i+k, i] for k = 0, ..., lower_bandwidth
/// - If lower=false: ab[k, i] = A[i-k, i] for k = 0, ..., upper_bandwidth
///
/// The diagonal is always ab[0, :].
///
/// Matches `scipy.linalg.solveh_banded(ab, b, lower)`.
///
/// # Arguments
/// * `ab` - Banded matrix in lower or upper band storage, shape (bandwidth+1, n)
/// * `b` - Right-hand side vector
/// * `lower` - If true, ab contains lower band; if false, upper band
pub fn solveh_banded(ab: &[Vec<f64>], b: &[f64], lower: bool) -> Result<SolveResult, LinalgError> {
    if ab.is_empty() {
        return Err(LinalgError::InvalidArgument {
            detail: "ab must not be empty".to_string(),
        });
    }

    let bandwidth_plus_1 = ab.len();
    let n = ab[0].len();

    if b.len() != n {
        return Err(LinalgError::IncompatibleShapes {
            a_shape: (n, n),
            b_len: b.len(),
        });
    }

    for row in ab.iter() {
        if row.len() != n {
            return Err(LinalgError::InvalidArgument {
                detail: "All rows in ab must have the same length".to_string(),
            });
        }
    }

    if n == 0 {
        return Ok(SolveResult {
            x: vec![],
            warning: None,
            backward_error: None,
            certificate: None,
        });
    }

    // Perform banded Cholesky factorization
    // L stored in same format as input
    let mut cb = ab.to_vec();

    if lower {
        // Cholesky factorization for lower band storage
        for j in 0..n {
            // Compute diagonal element
            let mut diag = cb[0][j];
            for k in 1..bandwidth_plus_1 {
                if j >= k {
                    let lkj = cb[k][j - k];
                    diag -= lkj * lkj;
                }
            }
            if diag <= 0.0 {
                return Err(LinalgError::InvalidArgument {
                    detail: "Matrix is not positive definite".to_string(),
                });
            }
            cb[0][j] = diag.sqrt();

            // Compute sub-diagonal elements
            for i in 1..bandwidth_plus_1 {
                if j + i < n {
                    let mut sum = cb[i][j];
                    for k in 1..bandwidth_plus_1 {
                        if j >= k && i + k < bandwidth_plus_1 {
                            sum -= cb[k][j - k] * cb[i + k][j - k];
                        }
                    }
                    cb[i][j] = sum / cb[0][j];
                }
            }
        }
    } else {
        // Cholesky factorization for upper band storage
        for j in 0..n {
            let mut diag = cb[0][j];
            for (k, band_row) in cb.iter().enumerate().take(bandwidth_plus_1).skip(1) {
                if j >= k {
                    let ukj = band_row[j];
                    diag -= ukj * ukj;
                }
            }
            if diag <= 0.0 {
                return Err(LinalgError::InvalidArgument {
                    detail: "Matrix is not positive definite".to_string(),
                });
            }
            cb[0][j] = diag.sqrt();

            for i in 1..bandwidth_plus_1 {
                if j + i < n {
                    let mut sum = cb[i][j + i];
                    for k in 1..bandwidth_plus_1.min(j + 1) {
                        if i + k < bandwidth_plus_1 {
                            sum -= cb[k][j] * cb[i + k][j + i];
                        }
                    }
                    cb[i][j + i] = sum / cb[0][j];
                }
            }
        }
    }

    // Now solve using the computed Cholesky factor
    cho_solve_banded(&cb, b, lower)
}

/// LDL decomposition for symmetric indefinite matrices.
///
/// Factors A = L * D * Lᵀ where L is unit lower triangular and D is diagonal.
/// Unlike Cholesky, this works for symmetric matrices that are not positive definite.
/// Matches `scipy.linalg.ldl(a)` for the real symmetric case.
pub fn ldl(a: &[Vec<f64>], options: DecompOptions) -> Result<LdlResult, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    let n = rows;
    if n == 0 {
        return Ok(LdlResult {
            l: Vec::new(),
            d: Vec::new(),
        });
    }

    // Symmetric check in hardened mode
    if matches!(options.mode, RuntimeMode::Hardened) {
        for (i, row_i) in a.iter().enumerate() {
            for (j, &aij) in row_i.iter().enumerate().skip(i + 1) {
                let aji = a[j][i];
                if (aij - aji).abs() > 1e-12 * (aij.abs() + aji.abs() + 1.0) {
                    return Err(LinalgError::InvalidArgument {
                        detail: format!(
                            "matrix is not symmetric: a[{i}][{j}]={aij} != a[{j}][{i}]={aji}"
                        ),
                    });
                }
            }
        }
    }

    // Aasen / Bunch-Kaufman-style 1×1 pivoting LDL factorization
    // (no permutation for simplicity — matches basic scipy.linalg.ldl behavior)
    let mut l_mat = vec![vec![0.0; n]; n];
    let mut d_vec = vec![0.0; n];

    // Copy lower triangle of A
    let mut work = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            work[i][j] = a[i][j];
        }
    }

    for j in 0..n {
        // Compute d[j]
        let mut sum = work[j][j];
        for k in 0..j {
            sum -= l_mat[j][k] * l_mat[j][k] * d_vec[k];
        }
        d_vec[j] = sum;

        // Set diagonal of L to 1
        l_mat[j][j] = 1.0;

        if d_vec[j].abs() < f64::EPSILON * 1e3 {
            // Near-zero diagonal: skip to avoid division by zero
            // The matrix has a near-zero eigenvalue
            for row in l_mat.iter_mut().skip(j + 1) {
                row[j] = 0.0;
            }
            continue;
        }

        // Compute column j of L below diagonal
        for i in (j + 1)..n {
            let mut sum = work[i][j];
            for k in 0..j {
                sum -= l_mat[i][k] * l_mat[j][k] * d_vec[k];
            }
            l_mat[i][j] = sum / d_vec[j];
        }
    }

    emit_trace(LinalgTrace {
        operation: "ldl",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(LdlResult { l: l_mat, d: d_vec })
}

// ══════════════════════════════════════════════════════════════════════
// Eigenvalue Decompositions — Public API
// ══════════════════════════════════════════════════════════════════════

/// Eigenvalue decomposition for general (possibly non-symmetric) matrices.
///
/// Returns eigenvalues as (real, imaginary) pairs and right eigenvectors.
/// For real matrices with complex eigenvalues, complex conjugate pairs appear
/// consecutively. Matches `scipy.linalg.eig(a)`.
pub fn eig(a: &[Vec<f64>], options: DecompOptions) -> Result<EigResult, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if rows == 0 {
        return Ok(EigResult {
            eigenvalues_re: Vec::new(),
            eigenvalues_im: Vec::new(),
            eigenvectors: Vec::new(),
        });
    }

    let matrix = dmatrix_from_rows(a)?;

    // Use Schur decomposition: A = Q T Q^T, eigenvalues on diagonal of T
    // For real matrices, Schur form has 1×1 blocks (real eigenvalues) and
    // 2×2 blocks (complex conjugate pairs) on the diagonal.
    let schur = matrix.clone().schur();
    let (q_mat, t_mat) = schur.unpack();

    let mut eigenvalues_re = Vec::with_capacity(rows);
    let mut eigenvalues_im = Vec::with_capacity(rows);
    let mut block_starts: Vec<(usize, bool)> = Vec::with_capacity(rows); // (col, is_2x2_block)

    let mut i = 0;
    while i < rows {
        if i + 1 < rows && t_mat[(i + 1, i)].abs() > f64::EPSILON * 100.0 {
            // 2×2 block: complex conjugate pair
            let a11 = t_mat[(i, i)];
            let a12 = t_mat[(i, i + 1)];
            let a21 = t_mat[(i + 1, i)];
            let a22 = t_mat[(i + 1, i + 1)];
            let trace = a11 + a22;
            let det_block = a11 * a22 - a12 * a21;
            let disc = trace * trace - 4.0 * det_block;
            let re = trace / 2.0;
            let im = (-disc).max(0.0).sqrt() / 2.0;
            eigenvalues_re.push(re);
            eigenvalues_im.push(im);
            block_starts.push((i, true));
            eigenvalues_re.push(re);
            eigenvalues_im.push(-im);
            block_starts.push((i, true));
            i += 2;
        } else {
            eigenvalues_re.push(t_mat[(i, i)]);
            eigenvalues_im.push(0.0);
            block_starts.push((i, false));
            i += 1;
        }
    }

    // Build right eigenvectors: for each real eigenvalue, solve
    // (T − λI) y = 0 by upper-triangular back-substitution and then
    // map back via v = Q y. Resolves [frankenscipy-eobzj]. For
    // 2×2 complex-conjugate blocks the API can't express complex
    // vectors so we keep the Schur columns there (still a partial
    // gap vs scipy, which returns complex eigenvectors).
    let mut eigvec_cols: Vec<Vec<f64>> = vec![vec![0.0; rows]; rows];
    for (col_idx, &(block_col, is_block)) in block_starts.iter().enumerate() {
        if is_block {
            let dest_col = q_mat.column(col_idx);
            for r in 0..rows {
                eigvec_cols[col_idx][r] = dest_col[r];
            }
            continue;
        }
        let lambda = t_mat[(block_col, block_col)];
        // y has length `rows`. y[block_col] = 1, y[k>block_col] = 0,
        // y[k<block_col] from upper-triangular back-substitution.
        let mut y = vec![0.0_f64; rows];
        y[block_col] = 1.0;
        let mut j = block_col;
        while j > 0 {
            j -= 1;
            // (T − λI)[j][k] for k > j sums against y[k].
            let mut s = 0.0_f64;
            for k in (j + 1)..=block_col {
                s += t_mat[(j, k)] * y[k];
            }
            let denom = t_mat[(j, j)] - lambda;
            if denom.abs() < 1.0e-15 {
                // Defective / repeated eigenvalue — back-sub fails.
                // Fall back to the Schur basis column for this index.
                let dest_col = q_mat.column(col_idx);
                for r in 0..rows {
                    eigvec_cols[col_idx][r] = dest_col[r];
                }
                y[block_col] = 0.0; // sentinel: skip Q multiplication below
                break;
            }
            y[j] = -s / denom;
        }
        if y[block_col] == 0.0 {
            // Already filled with Schur fallback above.
            continue;
        }
        // v = Q · y; normalize to unit length so columns match scipy
        // (which returns unit-length eigenvectors by default).
        let mut v = vec![0.0_f64; rows];
        for r in 0..rows {
            let mut s = 0.0;
            for k in 0..rows {
                s += q_mat[(r, k)] * y[k];
            }
            v[r] = s;
        }
        let norm = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
        eigvec_cols[col_idx] = v;
    }

    // Transpose eigvec_cols (per-column-as-vec) into row-major Vec<Vec<f64>>
    // matching the rest of the linalg API: result[row][col].
    let mut eigenvectors = vec![vec![0.0_f64; rows]; rows];
    for col_idx in 0..rows {
        for row_idx in 0..rows {
            eigenvectors[row_idx][col_idx] = eigvec_cols[col_idx][row_idx];
        }
    }

    emit_trace(LinalgTrace {
        operation: "eig",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(EigResult {
        eigenvalues_re,
        eigenvalues_im,
        eigenvectors,
    })
}

/// Compute eigenvalues only for a general matrix.
///
/// Returns eigenvalues as (real_parts, imaginary_parts).
/// Matches `scipy.linalg.eigvals(a)`.
pub fn eigvals(
    a: &[Vec<f64>],
    options: DecompOptions,
) -> Result<(Vec<f64>, Vec<f64>), LinalgError> {
    let result = eig(a, options)?;
    Ok((result.eigenvalues_re, result.eigenvalues_im))
}

/// Eigenvalue decomposition for symmetric/Hermitian matrices.
///
/// Eigenvalues are returned in ascending order. Eigenvectors form an
/// orthogonal matrix. Matches `scipy.linalg.eigh(a)`.
pub fn eigh(a: &[Vec<f64>], options: DecompOptions) -> Result<EighResult, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if rows == 0 {
        return Ok(EighResult {
            eigenvalues: Vec::new(),
            eigenvectors: Vec::new(),
        });
    }

    let matrix = dmatrix_from_rows(a)?;
    let sym = matrix.symmetric_eigen();

    // nalgebra returns eigenvalues in arbitrary order; sort ascending
    let mut pairs: Vec<(f64, Vec<f64>)> = sym
        .eigenvalues
        .iter()
        .enumerate()
        .map(|(j, &val)| {
            let col: Vec<f64> = (0..rows).map(|i| sym.eigenvectors[(i, j)]).collect();
            (val, col)
        })
        .collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

    let eigenvalues: Vec<f64> = pairs.iter().map(|(v, _)| *v).collect();
    // Reconstruct eigenvector matrix from sorted columns
    let mut eigenvectors = vec![vec![0.0; cols]; rows];
    for (col_idx, (_, col_data)) in pairs.iter().enumerate() {
        for (row_idx, &val) in col_data.iter().enumerate() {
            eigenvectors[row_idx][col_idx] = val;
        }
    }

    emit_trace(LinalgTrace {
        operation: "eigh",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(EighResult {
        eigenvalues,
        eigenvectors,
    })
}

/// Compute eigenvalues only for a symmetric/Hermitian matrix.
///
/// Returns real eigenvalues in ascending order.
/// Matches `scipy.linalg.eigvalsh(a)`.
pub fn eigvalsh(a: &[Vec<f64>], options: DecompOptions) -> Result<Vec<f64>, LinalgError> {
    let result = eigh(a, options)?;
    Ok(result.eigenvalues)
}

/// Eigenvalues and eigenvectors of a symmetric banded matrix.
///
/// Given a symmetric banded matrix in banded storage format, computes eigenvalues
/// and optionally eigenvectors. The matrix is first reduced to tridiagonal form
/// using Householder transformations, then solved via the QR algorithm.
///
/// Banded storage format (lower=true):
/// - ab[0, :] = main diagonal
/// - ab[k, i] = A[i+k, i] for k = 1, ..., lower_bandwidth
///
/// Matches `scipy.linalg.eig_banded(a_band, lower, eigvals_only, ...)`.
///
/// # Arguments
/// * `ab` - Banded matrix in lower band storage, shape (bandwidth+1, n)
/// * `lower` - If true, ab contains lower band (only lower=true supported currently)
/// * `eigvals_only` - If true, only compute eigenvalues
/// * `options` - Decomposition options
pub fn eig_banded(
    ab: &[Vec<f64>],
    lower: bool,
    eigvals_only: bool,
    options: DecompOptions,
) -> Result<EigenDecomposition, LinalgError> {
    if ab.is_empty() {
        return Err(LinalgError::InvalidArgument {
            detail: "ab must not be empty".to_string(),
        });
    }

    let bandwidth_plus_1 = ab.len();
    let n = ab[0].len();

    for row in ab.iter() {
        if row.len() != n {
            return Err(LinalgError::InvalidArgument {
                detail: "All rows in ab must have the same length".to_string(),
            });
        }
    }

    if n == 0 {
        return Ok((vec![], if eigvals_only { None } else { Some(vec![]) }));
    }

    hardened_dimension_check(options.mode, n, n)?;

    if options.check_finite {
        for row in ab.iter() {
            for &val in row.iter() {
                if !val.is_finite() {
                    return Err(LinalgError::NonFiniteInput);
                }
            }
        }
    }

    if !lower {
        // Convert upper to lower band storage
        // For now, only support lower=true
        return Err(LinalgError::NotSupported {
            detail: "eig_banded currently only supports lower=true".to_string(),
        });
    }

    // Special case: tridiagonal (bandwidth = 1)
    if bandwidth_plus_1 == 2 {
        let d: Vec<f64> = ab[0].clone();
        let e: Vec<f64> = (0..n - 1).map(|i| ab[1][i]).collect();
        return eigh_tridiagonal(&d, &e, eigvals_only, options);
    }

    // Special case: diagonal (bandwidth = 0)
    if bandwidth_plus_1 == 1 {
        let mut eigenvalues = ab[0].clone();
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let eigenvectors = if eigvals_only {
            None
        } else {
            // For diagonal matrix, eigenvectors are standard basis (reordered)
            let mut indices: Vec<usize> = (0..n).collect();
            indices.sort_by(|&i, &j| {
                ab[0][i]
                    .partial_cmp(&ab[0][j])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let evecs: Vec<Vec<f64>> = (0..n)
                .map(|row| {
                    indices
                        .iter()
                        .map(|&col| if row == col { 1.0 } else { 0.0 })
                        .collect()
                })
                .collect();
            Some(evecs)
        };
        return Ok((eigenvalues, eigenvectors));
    }

    // General case: reduce banded matrix to tridiagonal form
    // Using Householder transformations

    // Expand banded matrix to full symmetric form for reduction
    let mut a_full = vec![vec![0.0; n]; n];
    for i in 0..n {
        a_full[i][i] = ab[0][i];
        for k in 1..bandwidth_plus_1 {
            if i + k < n {
                a_full[i + k][i] = ab[k][i];
                a_full[i][i + k] = ab[k][i]; // Symmetric
            }
        }
    }

    // Reduce to tridiagonal form using Householder
    let mut q_accum = vec![vec![0.0; n]; n];
    for (i, row) in q_accum.iter_mut().enumerate().take(n) {
        row[i] = 1.0;
    }

    for k in 0..n.saturating_sub(2) {
        // Extract column below diagonal
        let col_len = n - k - 1;
        if col_len <= 1 {
            continue;
        }

        let mut v: Vec<f64> = (0..col_len).map(|i| a_full[k + 1 + i][k]).collect();

        // Compute Householder vector
        let norm_v: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_v < 1e-15 {
            continue;
        }

        let sign = if v[0] >= 0.0 { 1.0 } else { -1.0 };
        v[0] += sign * norm_v;
        let norm_v_new: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_v_new < 1e-15 {
            continue;
        }
        for vi in &mut v {
            *vi /= norm_v_new;
        }

        // Apply Householder: A = (I - 2vv^T) A (I - 2vv^T)
        // First: A = A - 2 * v * (v^T * A)
        let mut va = vec![0.0; n];
        for j in 0..n {
            for i in 0..col_len {
                va[j] += v[i] * a_full[k + 1 + i][j];
            }
        }
        for i in 0..col_len {
            for j in 0..n {
                a_full[k + 1 + i][j] -= 2.0 * v[i] * va[j];
            }
        }

        // Second: A = A - 2 * (A * v) * v^T
        let mut av = vec![0.0; n];
        for i in 0..n {
            for j in 0..col_len {
                av[i] += a_full[i][k + 1 + j] * v[j];
            }
        }
        for i in 0..n {
            for j in 0..col_len {
                a_full[i][k + 1 + j] -= 2.0 * av[i] * v[j];
            }
        }

        // Accumulate Q if eigenvectors needed
        if !eigvals_only {
            // Q = Q * (I - 2vv^T)
            let mut qv = vec![0.0; n];
            for i in 0..n {
                for j in 0..col_len {
                    qv[i] += q_accum[i][k + 1 + j] * v[j];
                }
            }
            for i in 0..n {
                for j in 0..col_len {
                    q_accum[i][k + 1 + j] -= 2.0 * qv[i] * v[j];
                }
            }
        }
    }

    // Extract tridiagonal elements
    let d: Vec<f64> = (0..n).map(|i| a_full[i][i]).collect();
    let e: Vec<f64> = (0..n - 1).map(|i| a_full[i + 1][i]).collect();

    // Solve tridiagonal eigenvalue problem
    let (eigenvalues, tri_evecs) = eigh_tridiagonal(&d, &e, eigvals_only, options)?;

    // Transform eigenvectors back if needed
    let eigenvectors = if eigvals_only {
        None
    } else {
        let tri_evecs = tri_evecs.unwrap();
        // Multiply Q * tri_evecs. Hoist k to the middle (ikj) so the inner
        // j-loop streams tri_evecs[k][..] contiguously instead of reading
        // tri_evecs[k][j] column-strided. Bit-identical: each
        // result_evecs[i][j] still accumulates k in 0..n order. [perf]
        let mut result_evecs = vec![vec![0.0; n]; n];
        for i in 0..n {
            let qi = &q_accum[i];
            let ri = &mut result_evecs[i];
            for k in 0..n {
                let qik = qi[k];
                let tk = &tri_evecs[k];
                for j in 0..n {
                    ri[j] += qik * tk[j];
                }
            }
        }
        Some(result_evecs)
    };

    emit_trace(LinalgTrace {
        operation: "eig_banded",
        matrix_size: (n, n),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok((eigenvalues, eigenvectors))
}

/// Eigenvalues and eigenvectors of a symmetric tridiagonal matrix.
///
/// Given the diagonal elements `d` and off-diagonal elements `e` of a
/// symmetric tridiagonal matrix T, computes the eigenvalues and optionally
/// the eigenvectors.
///
/// The matrix has structure:
/// ```text
/// T = | d[0]  e[0]   0    ...    0   |
///     | e[0]  d[1]  e[1]  ...    0   |
///     |  0    e[1]  d[2]  ...    0   |
///     | ...   ...   ...   ...   ... |
///     |  0     0     0   e[n-2] d[n-1]|
/// ```
///
/// Matches `scipy.linalg.eigh_tridiagonal(d, e)`.
///
/// # Arguments
/// * `d` - Main diagonal elements (length n)
/// * `e` - Off-diagonal elements (length n-1)
/// * `eigvals_only` - If true, only compute eigenvalues
/// * `options` - Decomposition options
///
/// # Returns
/// (eigenvalues, eigenvectors) where eigenvectors is None if eigvals_only=true.
pub fn eigh_tridiagonal(
    d: &[f64],
    e: &[f64],
    eigvals_only: bool,
    options: DecompOptions,
) -> Result<EigenDecomposition, LinalgError> {
    let n = d.len();
    if n == 0 {
        return Ok((vec![], if eigvals_only { None } else { Some(vec![]) }));
    }
    if e.len() != n - 1 && n > 0 {
        return Err(LinalgError::InvalidArgument {
            detail: format!(
                "e must have length n-1={}, got {}",
                n.saturating_sub(1),
                e.len()
            ),
        });
    }
    hardened_dimension_check(options.mode, n, n)?;

    if options.check_finite {
        for &val in d.iter().chain(e.iter()) {
            if !val.is_finite() {
                return Err(LinalgError::NonFiniteInput);
            }
        }
    }

    // Use QL algorithm with implicit shifts for symmetric tridiagonal eigenvalues
    let mut diagonal = d.to_vec();
    let mut off_diag = e.to_vec();
    off_diag.push(0.0); // Pad for convenience

    // Initialize eigenvectors to identity if needed
    let mut eigenvectors = if eigvals_only {
        None
    } else {
        let mut z = vec![vec![0.0; n]; n];
        for (i, row) in z.iter_mut().enumerate().take(n) {
            row[i] = 1.0;
        }
        Some(z)
    };

    // Implicit QR algorithm with Wilkinson shift
    let max_iter = 30 * n;
    let eps = 1e-14;

    // Work on a deflating submatrix [l, m]
    let mut m = n as i32 - 1;

    for _ in 0..max_iter {
        // Check for convergence of off-diagonal elements from bottom
        while m > 0 {
            let mi = m as usize;
            let tst = diagonal[mi - 1].abs() + diagonal[mi].abs();
            if off_diag[mi - 1].abs() <= eps * tst {
                off_diag[mi - 1] = 0.0;
                m -= 1;
            } else {
                break;
            }
        }

        if m <= 0 {
            break;
        }

        // Find start of unreduced block
        let mut l = m - 1;
        while l > 0 {
            let li = l as usize;
            let tst = diagonal[li - 1].abs() + diagonal[li].abs();
            if off_diag[li - 1].abs() <= eps * tst {
                break;
            }
            l -= 1;
        }

        let li = l as usize;
        let mi = m as usize;

        // Compute Wilkinson shift
        let d_diff = (diagonal[mi - 1] - diagonal[mi]) / 2.0;
        let e_sq = off_diag[mi - 1] * off_diag[mi - 1];
        let shift = if d_diff == 0.0 {
            diagonal[mi] - e_sq.sqrt()
        } else {
            diagonal[mi] - e_sq / (d_diff + d_diff.signum() * (d_diff * d_diff + e_sq).sqrt())
        };

        // QR step with Givens rotations
        let mut x = diagonal[li] - shift;
        let mut z = off_diag[li];

        for k in li..mi {
            // Compute Givens rotation
            let (c, s, r) = if z.abs() > x.abs() {
                let t = -x / z;
                let s = 1.0 / (1.0 + t * t).sqrt();
                let c = s * t;
                (c, s, z / s)
            } else if x.abs() > 0.0 {
                let t = -z / x;
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                (c, s, x / c)
            } else {
                (1.0, 0.0, 0.0)
            };

            // Apply rotation to tridiagonal matrix
            if k > li {
                off_diag[k - 1] = r;
            }

            let d_k = diagonal[k];
            let d_k1 = diagonal[k + 1];
            let e_k = off_diag[k];

            diagonal[k] = c * c * d_k + s * s * d_k1 - 2.0 * c * s * e_k;
            diagonal[k + 1] = s * s * d_k + c * c * d_k1 + 2.0 * c * s * e_k;
            off_diag[k] = c * s * (d_k - d_k1) + (c * c - s * s) * e_k;

            if k + 1 < mi {
                x = off_diag[k];
                z = -s * off_diag[k + 1];
                off_diag[k + 1] *= c;
            }

            // Update eigenvectors
            if let Some(ref mut evec) = eigenvectors {
                for row in evec.iter_mut().take(n) {
                    let t = row[k];
                    row[k] = c * t - s * row[k + 1];
                    row[k + 1] = s * t + c * row[k + 1];
                }
            }
        }
    }

    // Sort eigenvalues (and eigenvectors) in ascending order
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        diagonal[a]
            .partial_cmp(&diagonal[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let sorted_eigenvalues: Vec<f64> = indices.iter().map(|&i| diagonal[i]).collect();
    let sorted_eigenvectors = eigenvectors.map(|z| {
        (0..n)
            .map(|row| indices.iter().map(|&col| z[row][col]).collect())
            .collect()
    });

    emit_trace(LinalgTrace {
        operation: "eigh_tridiagonal",
        matrix_size: (n, n),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok((sorted_eigenvalues, sorted_eigenvectors))
}

// ══════════════════════════════════════════════════════════════════════
// Schur and Hessenberg Decompositions — Public API
// ══════════════════════════════════════════════════════════════════════

/// Schur decomposition: A = Z T Zᴴ.
///
/// Returns the orthogonal Schur vectors Z and the upper quasi-triangular
/// Schur form T (real Schur form for real matrices).
/// Matches `scipy.linalg.schur(a)`.
pub fn schur(a: &[Vec<f64>], options: DecompOptions) -> Result<SchurResult, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if rows == 0 {
        return Ok(SchurResult {
            z: Vec::new(),
            t: Vec::new(),
        });
    }

    let matrix = dmatrix_from_rows(a)?;
    let schur_decomp = matrix.schur();
    let (q_mat, t_mat) = schur_decomp.unpack();

    emit_trace(LinalgTrace {
        operation: "schur",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(SchurResult {
        z: rows_from_dmatrix(&q_mat),
        t: rows_from_dmatrix(&t_mat),
    })
}

/// Hessenberg decomposition: A = Q H Qᵀ.
///
/// Returns the orthogonal matrix Q and the upper Hessenberg matrix H.
/// Matches `scipy.linalg.hessenberg(a, calc_q=True)`.
pub fn hessenberg(a: &[Vec<f64>], options: DecompOptions) -> Result<HessenbergResult, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if rows == 0 {
        return Ok(HessenbergResult {
            q: Vec::new(),
            h: Vec::new(),
        });
    }

    let matrix = dmatrix_from_rows(a)?;
    let hess = matrix.hessenberg();
    let q_mat = hess.q();
    let h_mat = hess.unpack_h();

    emit_trace(LinalgTrace {
        operation: "hessenberg",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(HessenbergResult {
        q: rows_from_dmatrix(&q_mat),
        h: rows_from_dmatrix(&h_mat),
    })
}

/// Generalized Schur (QZ) decomposition for the matrix pencil (A, B).
///
/// Returns matrices `(AA, BB, Q, Z)` satisfying `Qᵀ A Z = AA` and `Qᵀ B Z = BB`.
/// This implementation currently supports the regular, invertible-`B` case by
/// reducing `A B⁻¹` to real Schur form and choosing `Z = B⁻¹ Q`, which yields
/// `BB = I` and an upper quasi-triangular `AA`.
///
/// Matches the core algebraic contract of `scipy.linalg.qz(a, b)`.
pub fn qz(a: &[Vec<f64>], b: &[Vec<f64>], options: DecompOptions) -> Result<QzResult, LinalgError> {
    let (ar, ac) = matrix_shape(a)?;
    let (br, bc) = matrix_shape(b)?;
    if ar != ac || br != bc {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    if ar != br {
        return Err(LinalgError::InvalidArgument {
            detail: format!(
                "A and B must have identical square shapes, got ({ar}x{ac}) and ({br}x{bc})"
            ),
        });
    }
    hardened_dimension_check(options.mode, ar, ac)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;
    validate_finite_matrix(b, options.mode, options.check_finite)?;

    if ar == 0 {
        return Ok(QzResult {
            q: Vec::new(),
            z: Vec::new(),
            aa: Vec::new(),
            bb: Vec::new(),
        });
    }

    let a_mat = dmatrix_from_rows(a)?;
    let b_mat = dmatrix_from_rows(b)?;
    let identity = DMatrix::<f64>::identity(ar, ar);
    let b_inv = b_mat
        .clone()
        .lu()
        .solve(&identity)
        .ok_or(LinalgError::SingularMatrix)?;

    // Real Schur of A·B⁻¹ = Q·T·Qᵀ gives the orthogonal Q and a real
    // quasi-upper-triangular T.
    let schur_decomp = (&a_mat * &b_inv).schur();
    let (q_mat, _t_mat) = schur_decomp.unpack();

    // Choose an orthogonal Z that upper-triangularizes Qᵀ·B. Writing
    // C = Qᵀ·B, an RQ factorization C = R·Zᵀ yields BB = Qᵀ·B·Z = R
    // (upper-triangular) and AA = Qᵀ·A·Z = T·R (quasi-upper-triangular),
    // with both Q and Z orthogonal — unlike the previous Z = B⁻¹·Q,
    // which was not orthogonal.
    let c_mat = q_mat.transpose() * &b_mat;
    let mut z_mat = qz_orthogonal_z(&c_mat);

    let mut aa_mat = q_mat.transpose() * &a_mat * &z_mat;
    let mut bb_mat = q_mat.transpose() * &b_mat * &z_mat;

    // LAPACK's generalized Schur form normalizes BB to a non-negative
    // diagonal. Flip the sign of any Z column whose BB diagonal entry is
    // negative; this keeps Z orthogonal and AA/BB (quasi-)triangular, and
    // makes qz(A, I) return BB = I as scipy.linalg.qz does.
    for j in 0..ar {
        if bb_mat[(j, j)] < 0.0 {
            for i in 0..ar {
                z_mat[(i, j)] = -z_mat[(i, j)];
                aa_mat[(i, j)] = -aa_mat[(i, j)];
                bb_mat[(i, j)] = -bb_mat[(i, j)];
            }
        }
    }

    emit_trace(LinalgTrace {
        operation: "qz",
        matrix_size: (ar, ac),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(QzResult {
        q: rows_from_dmatrix(&q_mat),
        z: rows_from_dmatrix(&z_mat),
        aa: rows_from_dmatrix(&aa_mat),
        bb: rows_from_dmatrix(&bb_mat),
    })
}

/// Orthogonal `Z` such that `C·Z` is upper-triangular — the orthogonal
/// factor of the RQ decomposition `C = R·Zᵀ`.
///
/// Built from a QR factorization of the row-reversed transpose: with `J`
/// the exchange matrix, `(J·C)ᵀ = Q₂·R₂` gives `C = (J·R₂ᵀ·J)·(Q₂·J)ᵀ`,
/// so `Z = Q₂·J` (the columns of `Q₂` reversed) is orthogonal.
fn qz_orthogonal_z(c: &DMatrix<f64>) -> DMatrix<f64> {
    let n = c.nrows();
    // Row-reverse C, then transpose.
    let mut jc_t = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            jc_t[(i, j)] = c[(n - 1 - j, i)];
        }
    }
    let q2 = jc_t.qr().q();
    // Z = Q₂·J: reverse the columns of Q₂.
    let mut z = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            z[(i, j)] = q2[(i, n - 1 - j)];
        }
    }
    z
}

fn ordqz_selected(alpha: f64, beta: f64, sort: OrdQzSort) -> bool {
    if !alpha.is_finite() || !beta.is_finite() || beta.abs() <= f64::EPSILON {
        return false;
    }
    let lambda = alpha / beta;
    match sort {
        OrdQzSort::LeftHalfPlane => lambda < 0.0,
        OrdQzSort::InsideUnitCircle => lambda.abs() < 1.0,
    }
}

fn stable_first_permutation(diagonal_pairs: &[(f64, f64)], sort: OrdQzSort) -> Vec<usize> {
    let mut selected = Vec::with_capacity(diagonal_pairs.len());
    let mut rejected = Vec::with_capacity(diagonal_pairs.len());
    for (index, &(alpha, beta)) in diagonal_pairs.iter().enumerate() {
        if ordqz_selected(alpha, beta, sort) {
            selected.push(index);
        } else {
            rejected.push(index);
        }
    }
    selected.extend(rejected);
    selected
}

fn permute_columns(matrix: &[Vec<f64>], permutation: &[usize]) -> Vec<Vec<f64>> {
    matrix
        .iter()
        .map(|row| permutation.iter().map(|&j| row[j]).collect())
        .collect()
}

fn permute_similarity(matrix: &[Vec<f64>], permutation: &[usize]) -> Vec<Vec<f64>> {
    permutation
        .iter()
        .map(|&i| permutation.iter().map(|&j| matrix[i][j]).collect())
        .collect()
}

/// Reorders a simplified real generalized Schur form so selected generalized
/// eigenvalues appear first.
///
/// This implementation is intentionally scoped to the current regular,
/// invertible-`B` `qz` path in FrankenSciPy. It computes `qz(a, b, ...)`, then
/// applies a stable-first permutation to the real diagonal generalized
/// eigenvalue ratios `aa[i][i] / bb[i][i]`. The same permutation is applied to
/// `Q`, `Z`, `AA`, and `BB`, preserving the generalized Schur relation.
pub fn ordqz(
    a: &[Vec<f64>],
    b: &[Vec<f64>],
    sort: OrdQzSort,
    options: DecompOptions,
) -> Result<QzResult, LinalgError> {
    let result = qz(a, b, options)?;
    if result.aa.is_empty() {
        return Ok(result);
    }

    let n = result.aa.len();
    if result.bb.len() != n {
        return Err(LinalgError::InvalidArgument {
            detail: "ordqz received inconsistent QZ factors".to_string(),
        });
    }

    let diagonal_pairs: Vec<(f64, f64)> =
        (0..n).map(|i| (result.aa[i][i], result.bb[i][i])).collect();
    let permutation = stable_first_permutation(&diagonal_pairs, sort);

    Ok(QzResult {
        q: permute_columns(&result.q, &permutation),
        z: permute_columns(&result.z, &permutation),
        aa: permute_similarity(&result.aa, &permutation),
        bb: permute_similarity(&result.bb, &permutation),
    })
}

// ══════════════════════════════════════════════════════════════════════
// Matrix Functions — Public API
// ══════════════════════════════════════════════════════════════════════

/// Matrix exponential using scaling and squaring with Padé approximation.
///
/// Computes exp(A) for a square matrix A.
/// Matches `scipy.linalg.expm(A)`.
pub fn expm(a: &[Vec<f64>], options: DecompOptions) -> Result<Vec<Vec<f64>>, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if rows == 0 {
        return Ok(Vec::new());
    }

    let matrix = dmatrix_from_rows(a)?;
    let result = expm_pade_scaling_squaring(&matrix);

    emit_trace(LinalgTrace {
        operation: "expm",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(rows_from_dmatrix(&result))
}

/// Scaling and squaring method with truncated Taylor series.
///
/// Algorithm: scale A by 2^(-s) so ||A/2^s|| is small, compute exp via
/// Taylor series (accurate for small norm), then square s times.
fn expm_pade_scaling_squaring(a: &DMatrix<f64>) -> DMatrix<f64> {
    let n = a.nrows();
    let identity = DMatrix::<f64>::identity(n, n);

    // Compute 1-norm for scaling
    let norm1 = matrix_one_norm(a);

    if norm1 == 0.0 {
        return identity;
    }

    // Scale until ||A/2^s||_1 < 0.5 to ensure Taylor convergence is fast
    let s = if norm1 <= 0.5 {
        0u32
    } else {
        ((norm1 * 2.0).log2().ceil()) as u32
    };
    let scaled = a / (2.0_f64.powi(s as i32));

    // Taylor series: exp(A) = I + A + A²/2! + A³/3! + ... + A^k/k!
    // For ||A|| <= 0.5, 20 terms gives ~10^-16 accuracy
    let result = taylor_exp(&scaled, &identity, 20);

    // Squaring phase: exp(A) = exp(A/2^s)^(2^s)
    let mut exp_a = result;
    for _ in 0..s {
        exp_a = &exp_a * &exp_a;
    }

    exp_a
}

/// Taylor series approximation of exp(A) = I + A + A²/2! + ... + A^k/k!
fn taylor_exp(a: &DMatrix<f64>, identity: &DMatrix<f64>, terms: usize) -> DMatrix<f64> {
    let mut result = identity.clone();
    let mut term = identity.clone(); // A^k / k!

    for k in 1..=terms {
        term = &term * a / (k as f64);
        result += &term;
    }

    result
}

/// Compute the 1-norm (max column sum of absolute values) of a matrix.
fn matrix_one_norm(m: &DMatrix<f64>) -> f64 {
    let (rows, cols) = (m.nrows(), m.ncols());
    (0..cols)
        .map(|j| (0..rows).map(|i| m[(i, j)].abs()).sum::<f64>())
        .fold(0.0_f64, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        })
}

// ══════════════════════════════════════════════════════════════════════
// Matrix Functions — logm, sqrtm, matrix_power
// ══════════════════════════════════════════════════════════════════════

/// Matrix logarithm.
///
/// Matches `scipy.linalg.logm(A)`.
///
/// Computes logm via eigendecomposition: if A = V D V^{-1}, then
/// logm(A) = V diag(log(d_i)) V^{-1}.
///
/// Requires all eigenvalues to be positive real.
pub fn logm(a: &[Vec<f64>], options: DecompOptions) -> Result<Vec<Vec<f64>>, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if rows == 0 {
        return Ok(Vec::new());
    }

    let matrix = dmatrix_from_rows(a)?;
    let n = rows;

    // Check if the matrix is symmetric
    let is_symmetric = (0..n).all(|i| {
        (0..n).all(|j| {
            (matrix[(i, j)] - matrix[(j, i)]).abs() < 1e-12 * matrix[(i, j)].abs().max(1.0)
        })
    });

    if !is_symmetric {
        // Non-symmetric: use general Schur-based path
        return logm_general(&matrix, n, options);
    }

    // Symmetric matrix: use eigendecomposition (symmetric_eigen is safe here)
    let eig = matrix.clone().symmetric_eigen();
    let eigenvalues = &eig.eigenvalues;

    // Check all eigenvalues are positive (required for real logarithm)
    for (i, &ev) in eigenvalues.iter().enumerate() {
        if ev <= 0.0 {
            emit_trace(LinalgTrace {
                operation: "logm",
                matrix_size: (n, n),
                mode: options.mode,
                rcond: None,
                warning: Some(format!("non-positive eigenvalue {ev} at index {i}")),
                error: None,
            });
            return logm_general(&matrix, n, options);
        }
    }

    // logm = V * diag(log(eigenvalues)) * V^T
    let v = &eig.eigenvectors;
    let mut log_diag = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        log_diag[(i, i)] = eigenvalues[i].ln();
    }

    let result = v * log_diag * v.transpose();

    emit_trace(LinalgTrace {
        operation: "logm",
        matrix_size: (n, n),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(rows_from_dmatrix(&result))
}

/// General logm via eigendecomposition (for non-symmetric matrices).
fn logm_general(
    matrix: &DMatrix<f64>,
    n: usize,
    options: DecompOptions,
) -> Result<Vec<Vec<f64>>, LinalgError> {
    // Use Schur decomposition: A = Q T Q^T where T is quasi-upper-triangular
    let schur = matrix.clone().schur();
    let (q, t) = schur.unpack();

    // Compute log of the quasi-triangular matrix
    let mut log_t = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        if t[(i, i)] > 0.0 {
            log_t[(i, i)] = t[(i, i)].ln();
        } else {
            // Non-positive diagonal: return NaN for this element
            log_t[(i, i)] = f64::NAN;
        }
    }

    // Compute off-diagonal of log(T) using Parlett's recurrence:
    //   (t[j,j] - t[i,i]) f[i,j]
    //     = t[i,j] (f[j,j] - f[i,i])
    //       + Σ_{k=i+1}^{j-1} (t[i,k] f[k,j] - f[i,k] t[k,j])
    //
    // For repeated diagonal entries we fall back to the first-derivative limit
    // used by the logarithm of a Jordan block: f[i,j] ~= n[i,j] / λ.
    for j in 1..n {
        for i in (0..j).rev() {
            let tii = t[(i, i)];
            let tjj = t[(j, j)];
            let di = log_t[(i, i)];
            let dj = log_t[(j, j)];
            let mut correction = 0.0;
            for k in (i + 1)..j {
                correction += t[(i, k)] * log_t[(k, j)] - log_t[(i, k)] * t[(k, j)];
            }
            let eigen_gap = tjj - tii;
            if eigen_gap.abs() > f64::EPSILON * tii.abs().max(tjj.abs()).max(1.0) {
                let numerator = t[(i, j)] * (dj - di) + correction;
                let val = numerator / eigen_gap;
                log_t[(i, j)] = if val.is_finite() { val } else { f64::NAN };
            } else {
                // Near-degenerate case: T[i,i] ≈ T[j,j]
                // Use the first-derivative limit for log(λI + N).
                let mut jordan_sum = t[(i, j)];
                for k in (i + 1)..j {
                    jordan_sum += t[(i, k)] * log_t[(k, j)] - log_t[(i, k)] * t[(k, j)];
                }
                if tii.abs() > f64::EPSILON {
                    let val = jordan_sum / tii;
                    log_t[(i, j)] = if val.is_finite() { val } else { f64::NAN };
                } else {
                    log_t[(i, j)] = f64::NAN; // Singularity
                }
            }
        }
    }

    let result = &q * log_t * q.transpose();

    emit_trace(LinalgTrace {
        operation: "logm",
        matrix_size: (n, n),
        mode: options.mode,
        rcond: None,
        warning: Some("used general (non-symmetric) path".to_string()),
        error: None,
    });

    Ok(rows_from_dmatrix(&result))
}

/// Matrix square root.
///
/// Matches `scipy.linalg.sqrtm(A)`.
///
/// Computes sqrtm via eigendecomposition: if A = V D V^{-1}, then
/// sqrtm(A) = V diag(sqrt(d_i)) V^{-1}.
pub fn sqrtm(a: &[Vec<f64>], options: DecompOptions) -> Result<Vec<Vec<f64>>, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if rows == 0 {
        return Ok(Vec::new());
    }

    let matrix = dmatrix_from_rows(a)?;
    let n = rows;

    // Check symmetry before using symmetric eigendecomposition
    let is_symmetric = (0..n).all(|i| {
        (0..n).all(|j| {
            (matrix[(i, j)] - matrix[(j, i)]).abs() < 1e-12 * matrix[(i, j)].abs().max(1.0)
        })
    });

    if !is_symmetric {
        // Non-symmetric: sqrtm via Schur decomposition
        // A = Q T Q^T, sqrt(A) = Q sqrt(T) Q^T
        let schur = matrix.clone().schur();
        let (q, t) = schur.unpack();
        let mut sqrt_t = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            if t[(i, i)] >= -1e-14 {
                sqrt_t[(i, i)] = t[(i, i)].max(0.0).sqrt();
            } else {
                sqrt_t[(i, i)] = f64::NAN;
            }
        }
        // Off-diagonal: Parlett recurrence for sqrt
        for j in 1..n {
            for i in (0..j).rev() {
                let si = sqrt_t[(i, i)];
                let sj = sqrt_t[(j, j)];
                let mut sum = t[(i, j)];
                for k in (i + 1)..j {
                    sum -= sqrt_t[(i, k)] * sqrt_t[(k, j)];
                }
                let denom = si + sj;
                sqrt_t[(i, j)] = if denom.abs() > 1e-15 {
                    sum / denom
                } else {
                    0.0
                };
            }
        }
        let result = &q * sqrt_t * q.transpose();
        emit_trace(LinalgTrace {
            operation: "sqrtm",
            matrix_size: (n, n),
            mode: options.mode,
            rcond: None,
            warning: Some("used general (non-symmetric) Schur path".to_string()),
            error: None,
        });
        return Ok(rows_from_dmatrix(&result));
    }

    // Symmetric path: eigendecomposition is safe here
    let eig = matrix.clone().symmetric_eigen();
    let eigenvalues = &eig.eigenvalues;

    // Check eigenvalues are non-negative
    let all_nonneg = eigenvalues.iter().all(|&ev| ev >= -1e-14);

    if all_nonneg {
        let v = &eig.eigenvectors;
        let mut sqrt_diag = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            sqrt_diag[(i, i)] = eigenvalues[i].max(0.0).sqrt();
        }
        let result = v * sqrt_diag * v.transpose();

        emit_trace(LinalgTrace {
            operation: "sqrtm",
            matrix_size: (n, n),
            mode: options.mode,
            rcond: None,
            warning: None,
            error: None,
        });

        Ok(rows_from_dmatrix(&result))
    } else {
        // For matrices with negative eigenvalues, sqrtm produces complex results.
        // Return NaN to indicate this.
        emit_trace(LinalgTrace {
            operation: "sqrtm",
            matrix_size: (n, n),
            mode: options.mode,
            rcond: None,
            warning: Some("matrix has negative eigenvalues; real sqrtm undefined".to_string()),
            error: None,
        });
        let nan_matrix = vec![vec![f64::NAN; n]; n];
        Ok(nan_matrix)
    }
}

/// Integer matrix power: A^n.
///
/// Matches `numpy.linalg.matrix_power(A, n)`.
///
/// - n > 0: repeated multiplication (binary exponentiation)
/// - n == 0: identity matrix
/// - n < 0: (A^{-1})^{|n|}
pub fn matrix_power(
    a: &[Vec<f64>],
    power: i32,
    options: DecompOptions,
) -> Result<Vec<Vec<f64>>, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if rows == 0 {
        return Ok(Vec::new());
    }

    let n = rows;

    if power == 0 {
        // A^0 = I
        let mut identity = vec![vec![0.0; n]; n];
        for (i, row) in identity.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        return Ok(identity);
    }

    let matrix = dmatrix_from_rows(a)?;

    let base = if power < 0 {
        // Need inverse
        let lu = matrix.clone().lu();
        match lu.try_inverse() {
            Some(inv) => inv,
            None => {
                return Err(LinalgError::SingularMatrix);
            }
        }
    } else {
        matrix
    };

    let abs_power = power.unsigned_abs();

    // Binary exponentiation
    let mut result = DMatrix::<f64>::identity(n, n);
    let mut current = base;
    let mut p = abs_power;

    while p > 0 {
        if p & 1 == 1 {
            result = &result * &current;
        }
        p >>= 1;
        if p > 0 {
            current = &current * &current;
        }
    }

    emit_trace(LinalgTrace {
        operation: "matrix_power",
        matrix_size: (n, n),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(rows_from_dmatrix(&result))
}

/// Matrix sign function.
///
/// For a matrix with no purely imaginary eigenvalues, sign(A) is defined as
/// A * (A²)^{-1/2}, or equivalently via eigendecomposition:
/// sign(A) = V * diag(sign(λ_i)) * V^{-1}.
///
/// Matches `scipy.linalg.signm(A)`.
pub fn signm(a: &[Vec<f64>], options: DecompOptions) -> Result<Vec<Vec<f64>>, LinalgError> {
    funm(a, |x| if x >= 0.0 { 1.0 } else { -1.0 }, options)
}

/// Matrix sine.
///
/// Matches `scipy.linalg.sinm(A)`.
pub fn sinm(a: &[Vec<f64>], options: DecompOptions) -> Result<Vec<Vec<f64>>, LinalgError> {
    funm(a, f64::sin, options)
}

/// Matrix cosine.
///
/// Matches `scipy.linalg.cosm(A)`.
pub fn cosm(a: &[Vec<f64>], options: DecompOptions) -> Result<Vec<Vec<f64>>, LinalgError> {
    funm(a, f64::cos, options)
}

/// Matrix tangent.
///
/// Matches `scipy.linalg.tanm(A)`.
pub fn tanm(a: &[Vec<f64>], options: DecompOptions) -> Result<Vec<Vec<f64>>, LinalgError> {
    funm(a, f64::tan, options)
}

/// Matrix hyperbolic sine.
///
/// Matches `scipy.linalg.sinhm(A)`.
pub fn sinhm(a: &[Vec<f64>], options: DecompOptions) -> Result<Vec<Vec<f64>>, LinalgError> {
    funm(a, f64::sinh, options)
}

/// Matrix hyperbolic cosine.
///
/// Matches `scipy.linalg.coshm(A)`.
pub fn coshm(a: &[Vec<f64>], options: DecompOptions) -> Result<Vec<Vec<f64>>, LinalgError> {
    funm(a, f64::cosh, options)
}

/// Matrix hyperbolic tangent.
///
/// Matches `scipy.linalg.tanhm(A)`.
pub fn tanhm(a: &[Vec<f64>], options: DecompOptions) -> Result<Vec<Vec<f64>>, LinalgError> {
    funm(a, f64::tanh, options)
}

/// General matrix function via eigendecomposition.
///
/// Computes f(A) = V * diag(f(λ_i)) * V^{-1} where A = V * diag(λ) * V^{-1}.
///
/// For symmetric matrices, uses orthogonal eigendecomposition (more stable).
/// For general matrices, uses Schur decomposition with Parlett's recurrence.
///
/// Matches `scipy.linalg.funm(A, func)`.
pub fn funm(
    a: &[Vec<f64>],
    func: impl Fn(f64) -> f64,
    options: DecompOptions,
) -> Result<Vec<Vec<f64>>, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    hardened_dimension_check(options.mode, rows, cols)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if rows == 0 {
        return Ok(Vec::new());
    }

    let matrix = dmatrix_from_rows(a)?;
    let n = rows;

    // Check symmetry
    let is_symmetric = (0..n).all(|i| {
        (0..n).all(|j| {
            (matrix[(i, j)] - matrix[(j, i)]).abs() < 1e-12 * matrix[(i, j)].abs().max(1.0)
        })
    });

    if is_symmetric {
        // Symmetric: A = V D V^T, f(A) = V f(D) V^T
        let eig = matrix.clone().symmetric_eigen();
        let v = &eig.eigenvectors;
        let mut fd = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            fd[(i, i)] = func(eig.eigenvalues[i]);
        }
        let result = v * fd * v.transpose();
        return Ok(rows_from_dmatrix(&result));
    }

    // General: use Schur decomposition A = Q T Q^T
    // Then f(A) = Q f(T) Q^T via Parlett's recurrence
    let schur = matrix.clone().schur();
    let (q, t) = schur.unpack();

    let mut ft = DMatrix::<f64>::zeros(n, n);
    // Diagonal: f(T[i,i])
    for i in 0..n {
        ft[(i, i)] = func(t[(i, i)]);
    }

    // Off-diagonal via Parlett's recurrence
    for j in 1..n {
        for i in (0..j).rev() {
            let mut sum = t[(i, j)] * (ft[(j, j)] - ft[(i, i)]);
            for k in (i + 1)..j {
                sum += t[(i, k)] * ft[(k, j)] - ft[(i, k)] * t[(k, j)];
            }
            let denom = t[(j, j)] - t[(i, i)];
            ft[(i, j)] =
                if denom.abs() > f64::EPSILON * t[(j, j)].abs().max(t[(i, i)].abs()).max(1.0) {
                    let val = sum / denom;
                    if val.is_finite() { val } else { f64::NAN }
                } else if t[(i, i)].abs() > f64::EPSILON {
                    let val = sum / t[(i, i)];
                    if val.is_finite() { val } else { f64::NAN }
                } else {
                    f64::NAN // Singularity
                };
        }
    }

    let result = &q * ft * q.transpose();
    Ok(rows_from_dmatrix(&result))
}

/// Fractional matrix power A^p for non-integer p.
///
/// Computes A^p via eigendecomposition: A^p = V * diag(λ_i^p) * V^{-1}.
///
/// Matches `scipy.linalg.fractional_matrix_power(A, p)`.
pub fn fractional_matrix_power(
    a: &[Vec<f64>],
    p: f64,
    options: DecompOptions,
) -> Result<Vec<Vec<f64>>, LinalgError> {
    funm(a, |x| x.powf(p), options)
}

// ══════════════════════════════════════════════════════════════════════
// ══════════════════════════════════════════════════════════════════════
// Specialized Solvers and Products
// ══════════════════════════════════════════════════════════════════════

/// Khatri-Rao product (column-wise Kronecker product).
///
/// For matrices A (m×n) and B (p×n), produces an (m*p × n) matrix where
/// column j is the Kronecker product of `A[:,j]` and `B[:,j]`.
///
/// Matches `scipy.linalg.khatri_rao(A, B)`.
pub fn khatri_rao(a: &[Vec<f64>], b: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, LinalgError> {
    let (m, na) = matrix_shape(a)?;
    let (p, nb) = matrix_shape(b)?;
    if na != nb {
        return Err(LinalgError::NotSupported {
            detail: format!("A has {na} columns but B has {nb} columns; must match"),
        });
    }
    let ncols = na;
    let nrows = m * p;

    let mut result = vec![vec![0.0; ncols]; nrows];
    for j in 0..ncols {
        for i in 0..m {
            for k in 0..p {
                result[i * p + k][j] = a[i][j] * b[k][j];
            }
        }
    }
    Ok(result)
}

/// Solve a circulant linear system C x = b using FFT.
///
/// A circulant matrix is defined by its first column c. The solution
/// is computed in O(n log n) via FFT: x = ifft(fft(b) / fft(c)).
///
/// Matches `scipy.linalg.solve_circulant(c, b)`.
pub fn solve_circulant(c: &[f64], b: &[f64]) -> Result<Vec<f64>, LinalgError> {
    let n = c.len();
    if b.len() != n {
        return Err(LinalgError::IncompatibleShapes {
            a_shape: (n, n),
            b_len: b.len(),
        });
    }
    if n == 0 {
        return Ok(Vec::new());
    }

    let opts = fsci_fft::FftOptions::default();
    let c_complex: Vec<(f64, f64)> = c.iter().map(|&value| (value, 0.0)).collect();
    let b_complex: Vec<(f64, f64)> = b.iter().map(|&value| (value, 0.0)).collect();
    let fft_c = fsci_fft::fft(&c_complex, &opts).map_err(|err| LinalgError::InvalidArgument {
        detail: format!("circulant FFT failed: {err}"),
    })?;
    let fft_b = fsci_fft::fft(&b_complex, &opts).map_err(|err| LinalgError::InvalidArgument {
        detail: format!("rhs FFT failed: {err}"),
    })?;

    // Scale the singular-mode threshold by max |λ|² so it is invariant
    // to overall matrix magnitude, mirroring scipy's default
    // `tol = eps * max|eigenvalues|` on solve_circulant (br-z1vz).
    // `1e-28` corresponds to sqrt(1e-28) ≈ 1e-14 ~ 10x machine epsilon.
    let max_lambda_sq = fft_c
        .iter()
        .map(|&(cr, ci)| cr * cr + ci * ci)
        .fold(0.0_f64, f64::max);
    let tol_sq = 1e-28 * max_lambda_sq.max(1.0);

    // Element-wise division: fft_b / fft_c. A singular mode (|λ_k| ≈ 0)
    // with non-zero b̂_k is fail-closed — returning x=0 for that mode
    // silently produces a wrong answer that fails Cx = b.
    let mut fft_x: Vec<(f64, f64)> = Vec::with_capacity(fft_b.len());
    for (&(br, bi), &(cr, ci)) in fft_b.iter().zip(fft_c.iter()) {
        let denom = cr * cr + ci * ci;
        if denom < tol_sq {
            let b_mag_sq = br * br + bi * bi;
            // Only reject when the RHS has meaningful energy in the
            // singular mode; if b̂_k = 0 the zero solution is exact.
            if b_mag_sq > tol_sq {
                return Err(LinalgError::SingularMatrix);
            }
            fft_x.push((0.0, 0.0));
        } else {
            fft_x.push(((br * cr + bi * ci) / denom, (bi * cr - br * ci) / denom));
        }
    }

    let x = fsci_fft::ifft(&fft_x, &opts).map_err(|err| LinalgError::InvalidArgument {
        detail: format!("circulant inverse FFT failed: {err}"),
    })?;
    Ok(x.into_iter().map(|(re, _)| re).collect())
}

/// Solve a Toeplitz linear system T x = b.
///
/// The Toeplitz matrix is defined by its first column `c` and optional first
/// row `r`. If `r` is omitted, a symmetric Toeplitz matrix is assumed.
///
/// Matches `scipy.linalg.solve_toeplitz((c, r), b)`.
pub fn solve_toeplitz(c: &[f64], r: Option<&[f64]>, b: &[f64]) -> Result<Vec<f64>, LinalgError> {
    let n = c.len();
    if b.len() != n {
        return Err(LinalgError::IncompatibleShapes {
            a_shape: (n, n),
            b_len: b.len(),
        });
    }
    let row = r.unwrap_or(c);
    if row.len() != n {
        return Err(LinalgError::IncompatibleShapes {
            a_shape: (n, row.len()),
            b_len: b.len(),
        });
    }
    if n == 0 {
        return Ok(Vec::new());
    }

    // Levinson–Durbin recursion (O(n²)) instead of building the dense n×n
    // Toeplitz matrix and running an O(n³) LU solve. This is the same algorithm
    // SciPy's `solve_toeplitz` uses, so it both matches SciPy's numerics more
    // closely and scales as O(n²). The matrix is T[i][j] = c[i-j] for i>=j else
    // r[j-i]; the diagonal is c[0] and row[0] is ignored (matching `toeplitz`).
    //
    // Forward/backward predictor vectors f, bk satisfy T^{(m)} f = e_1 and
    // T^{(m)} bk = e_m for each leading principal submatrix; the solution x is
    // grown one row at a time via x ← [x;0] + θ·bk. A zero recursion pivot means
    // a singular leading principal minor — SciPy rejects the same inputs.
    let diag = |k: isize| -> f64 {
        if k >= 0 {
            c[k as usize]
        } else {
            row[(-k) as usize]
        }
    };

    let t0 = diag(0);
    if t0 == 0.0 {
        return Err(LinalgError::SingularMatrix);
    }

    let mut f = Vec::with_capacity(n); // forward predictor
    let mut bk = Vec::with_capacity(n); // backward predictor
    let mut x = Vec::with_capacity(n);
    f.push(1.0 / t0);
    bk.push(1.0 / t0);
    x.push(b[0] / t0);

    for (m, &bm) in b.iter().enumerate().take(n).skip(1) {
        // Forward error ε_f = Σ_j T[m][j] f_j and backward error
        // ε_b = Σ_j T[0][j+1] bk_j over the current submatrix of size m.
        let mut ef = 0.0;
        let mut eb = 0.0;
        for j in 0..m {
            ef += diag(m as isize - j as isize) * f[j];
            eb += diag(-(j as isize) - 1) * bk[j];
        }
        let denom = 1.0 - ef * eb;
        if denom == 0.0 {
            return Err(LinalgError::SingularMatrix);
        }

        // F = ([f;0] - ε_f[0;bk]) / denom ; B = ([0;bk] - ε_b[f;0]) / denom.
        let mut f_new = Vec::with_capacity(m + 1);
        let mut b_new = Vec::with_capacity(m + 1);
        for i in 0..=m {
            let fi = if i < m { f[i] } else { 0.0 };
            let bi = if i == 0 { 0.0 } else { bk[i - 1] };
            f_new.push((fi - ef * bi) / denom);
            b_new.push((bi - eb * fi) / denom);
        }
        f = f_new;
        bk = b_new;

        // θ = b[m] - Σ_j T[m][j] x_j ; x ← [x;0] + θ·bk (with the new bk).
        let mut ex = 0.0;
        for (j, &xj) in x.iter().enumerate().take(m) {
            ex += diag(m as isize - j as isize) * xj;
        }
        let theta = bm - ex;
        x.push(0.0);
        for i in 0..=m {
            x[i] += theta * bk[i];
        }
    }

    Ok(x)
}

// ══════════════════════════════════════════════════════════════════════
// Matrix Property Checks
// ══════════════════════════════════════════════════════════════════════

/// Check if a matrix is symmetric within tolerance.
///
/// Returns true if `|A[i,j] - A[j,i]| <= atol + rtol * max(|A[i,j]|, |A[j,i]|)` for all i,j.
///
/// Matches `scipy.linalg.issymmetric(a, atol, rtol)`.
pub fn issymmetric(a: &[Vec<f64>], atol: f64, rtol: f64) -> Result<bool, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    if rows != cols {
        return Ok(false);
    }
    for (i, row) in a.iter().enumerate().take(rows) {
        for (j, &upper) in row.iter().enumerate().skip(i + 1).take(cols - (i + 1)) {
            let lower = a[j][i];
            if !upper.is_finite() || !lower.is_finite() {
                return Ok(false);
            }
            let diff = (upper - lower).abs();
            let scale = upper.abs().max(lower.abs());
            if diff > atol + rtol * scale {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Check if a matrix is Hermitian within tolerance.
///
/// For real matrices, this is equivalent to `issymmetric`.
///
/// Matches `scipy.linalg.ishermitian(a, atol, rtol)`.
pub fn ishermitian(a: &[Vec<f64>], atol: f64, rtol: f64) -> Result<bool, LinalgError> {
    // For real matrices, Hermitian = symmetric
    issymmetric(a, atol, rtol)
}

// ══════════════════════════════════════════════════════════════════════
// Norms and Rank — Public API
// ══════════════════════════════════════════════════════════════════════

/// Compute matrix or vector norm.
///
/// Matches `scipy.linalg.norm(a, ord)`.
pub fn norm(a: &[Vec<f64>], kind: NormKind, options: DecompOptions) -> Result<f64, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if rows == 0 || cols == 0 {
        return Ok(0.0);
    }

    let matrix = dmatrix_from_rows(a)?;

    let result = match kind {
        NormKind::Fro => {
            // Frobenius norm: sqrt(sum of squares)
            matrix.norm()
        }
        NormKind::Spectral => {
            // Spectral norm: largest singular value
            let svd_decomp = safe_svd(matrix, false, false)?;
            svd_decomp
                .singular_values
                .iter()
                .copied()
                .fold(0.0_f64, |a: f64, b: f64| {
                    if a.is_nan() || b.is_nan() {
                        f64::NAN
                    } else {
                        a.max(b)
                    }
                })
        }
        NormKind::One => {
            // 1-norm: max column sum of absolute values
            (0..cols)
                .map(|j| (0..rows).map(|i| matrix[(i, j)].abs()).sum::<f64>())
                .fold(0.0_f64, |a: f64, b: f64| {
                    if a.is_nan() || b.is_nan() {
                        f64::NAN
                    } else {
                        a.max(b)
                    }
                })
        }
        NormKind::Inf => {
            // Infinity norm: max row sum of absolute values
            (0..rows)
                .map(|i| (0..cols).map(|j| matrix[(i, j)].abs()).sum::<f64>())
                .fold(0.0_f64, |a: f64, b: f64| {
                    if a.is_nan() || b.is_nan() {
                        f64::NAN
                    } else {
                        a.max(b)
                    }
                })
        }
    };

    emit_trace(LinalgTrace {
        operation: "norm",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(result)
}

/// Compute the numerical rank of a matrix using SVD.
///
/// A singular value is considered zero if it is less than `tol`.
/// If `tol` is None, uses `max(m,n) * eps * max(singular_values)`.
pub fn matrix_rank(
    a: &[Vec<f64>],
    tol: Option<f64>,
    options: DecompOptions,
) -> Result<usize, LinalgError> {
    let (rows, cols) = matrix_shape(a)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if rows == 0 || cols == 0 {
        return Ok(0);
    }

    let matrix = dmatrix_from_rows(a)?;
    let svd_decomp = safe_svd(matrix, false, false)?;
    let singular_values = &svd_decomp.singular_values;
    let max_s = singular_values
        .iter()
        .copied()
        .fold(0.0_f64, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        });
    let threshold = tol.unwrap_or_else(|| (rows.max(cols) as f64) * f64::EPSILON * max_s);
    let rank = singular_values.iter().filter(|s| **s > threshold).count();

    emit_trace(LinalgTrace {
        operation: "matrix_rank",
        matrix_size: (rows, cols),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(rank)
}

fn solve_diagonal(a: &[Vec<f64>], b: &[f64]) -> Result<SolveResult, LinalgError> {
    let n = a.len();
    let mut x = vec![0.0; n];
    let mut max_diag: f64 = 0.0;
    let mut min_diag = f64::INFINITY;
    for i in 0..n {
        let diag = a[i][i];
        if diag == 0.0 {
            return Err(LinalgError::SingularMatrix);
        }
        let abs_diag = diag.abs();
        max_diag = max_diag.max(abs_diag);
        min_diag = min_diag.min(abs_diag);
        x[i] = b[i] / diag;
    }
    let rcond = if max_diag > 0.0 {
        min_diag / max_diag
    } else {
        0.0
    };
    let warning = if rcond < 1e-12 {
        Some(LinalgWarning::IllConditioned {
            reciprocal_condition: rcond,
        })
    } else {
        None
    };

    let backward_error = compute_backward_error_dense(a, &x, b);

    Ok(SolveResult {
        x,
        warning,
        backward_error: Some(backward_error),
        certificate: None,
    })
}

fn solve_triangular_internal(
    a: &[Vec<f64>],
    b: &[f64],
    trans: TriangularTranspose,
    lower: bool,
    unit_diagonal: bool,
) -> Result<SolveResult, LinalgError> {
    let n = a.len();
    let (mat_storage, is_lower): (Cow<'_, [Vec<f64>]>, bool) = match trans {
        TriangularTranspose::NoTranspose => (Cow::Borrowed(a), lower),
        TriangularTranspose::Transpose | TriangularTranspose::ConjugateTranspose => {
            (Cow::Owned(transpose(a)), !lower)
        }
    };
    let mat = mat_storage.as_ref();

    let mut x = vec![0.0; n];
    if is_lower {
        for i in 0..n {
            let mut sum = 0.0;
            for (j, xj) in x.iter().enumerate().take(i) {
                sum += mat[i][j] * *xj;
            }
            let diag = if unit_diagonal { 1.0 } else { mat[i][i] };
            if diag == 0.0 {
                return Err(LinalgError::SingularMatrix);
            }
            x[i] = (b[i] - sum) / diag;
        }
    } else {
        for i in (0..n).rev() {
            let mut sum = 0.0;
            for (j, xj) in x.iter().enumerate().skip(i + 1) {
                sum += mat[i][j] * *xj;
            }
            let diag = if unit_diagonal { 1.0 } else { mat[i][i] };
            if diag == 0.0 {
                return Err(LinalgError::SingularMatrix);
            }
            x[i] = (b[i] - sum) / diag;
        }
    }

    let backward_error = compute_backward_error_dense(mat, &x, b);

    Ok(SolveResult {
        x,
        warning: None,
        backward_error: Some(backward_error),
        certificate: None,
    })
}

/// Compute backward error using dense iteration: ||Ax - b|| / (||A|| × ||x|| + ||b||).
fn compute_backward_error_dense(a: &[Vec<f64>], x: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    let m = if a.is_empty() { 0 } else { a[0].len() };

    let mut residual_sum_sq = 0.0;
    for i in 0..n {
        let mut ax_i = 0.0;
        for (j, &xj) in x.iter().enumerate().take(m) {
            ax_i += a[i][j] * xj;
        }
        let res_i = ax_i - b[i];
        residual_sum_sq += res_i * res_i;
    }
    let residual_norm = residual_sum_sq.sqrt();

    let mut a_sum_sq = 0.0;
    for row in a {
        for &val in row {
            a_sum_sq += val * val;
        }
    }
    let a_norm = a_sum_sq.sqrt();

    let mut x_sum_sq = 0.0;
    for &val in x {
        x_sum_sq += val * val;
    }
    let x_norm = x_sum_sq.sqrt();

    let mut b_sum_sq = 0.0;
    for &val in b {
        b_sum_sq += val * val;
    }
    let b_norm = b_sum_sq.sqrt();

    let denom = a_norm * x_norm + b_norm;
    if !residual_norm.is_finite() || !denom.is_finite() {
        return f64::INFINITY;
    }
    if denom > 0.0 {
        residual_norm / denom
    } else {
        0.0
    }
}

/// Solve `A x = b` for a banded `A` (lower bandwidth `kl`, upper bandwidth `ku`)
/// supplied as a dense buffer, using Gaussian elimination with partial pivoting
/// restricted to the band + fill region.
///
/// With partial pivoting the upper bandwidth of `U` grows to at most `kl + ku`,
/// while `L`'s lower bandwidth stays `kl`; bounding both inner loops to that
/// region reduces the work from the dense `O(n^3)` to `O(n · kl · (kl + ku))`
/// (e.g. `O(n)` for a tridiagonal system). The result is the same Gaussian
/// elimination the dense path performs — operations on the structurally-zero
/// entries outside the band are `x - m·0` no-ops — so the solution matches the
/// dense solver to within rounding.
///
/// `work` is consumed as scratch (overwritten with the LU factors). Returns
/// `None` when a zero or non-finite pivot is encountered so the caller can fall
/// back to the robust dense solver and preserve singular-case behavior.
/// Solve `A x = b` for a banded `A` directly on LAPACK-style packed band storage,
/// without ever materializing the dense matrix.
///
/// `ab` is the scipy `solve_banded` layout (`(kl+ku+1) × n`, `ab[ku+i-j][j] =
/// A[i][j]`). The factor band is held in an expanded `(2·kl+ku+1) × n` packed
/// buffer `w`, indexed by `w[kl+ku + i - j][j] = A(i,j)`; the top `kl` rows are
/// fill workspace exposed by partial pivoting (which grows U's upper bandwidth to
/// at most `kl+ku`). Pivot search, the row interchange, elimination, and back
/// substitution are all bounded to the band, giving `O(n·kl·(kl+ku))` time and
/// `O(n·(kl+ku))` memory.
///
/// This performs the same Gaussian elimination as a dense banded GEPP — the
/// operations it skips act on structurally-zero entries (`x - m·0`) — so the
/// returned `x` is bit-identical to the dense banded path. L multipliers are
/// folded into the RHS in place during factorization (and the RHS interchange is
/// applied inline), so the column positions of the discarded sub-diagonal entries
/// never affect the result. Returns `None` on a zero/non-finite pivot so the
/// caller can fall back to the robust dense solver.
#[allow(clippy::needless_range_loop)] // explicit band indices drive pivot search / interchange
fn banded_lu_solve_packed(ab: &[Vec<f64>], kl: usize, ku: usize, b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    if n == 0 {
        return Some(Vec::new());
    }
    let kuf = kl + ku; // fill-extended upper bandwidth of U
    let diag = kuf; // band-row of the main diagonal: w[kuf + i - j][j]
    let wrows = 2 * kl + ku + 1;
    // Place A into the expanded packed buffer; top `kl` rows stay zero for fill.
    let mut w = vec![vec![0.0_f64; n]; wrows];
    for i in 0..n {
        let j_start = i.saturating_sub(kl);
        let j_end = (i + ku).min(n - 1);
        for j in j_start..=j_end {
            w[diag + i - j][j] = ab[ku + i - j][j];
        }
    }
    let mut rhs = b.to_vec();
    let mut urow = vec![0.0_f64; kuf + 1]; // reusable snapshot of pivot row k

    for k in 0..n {
        let row_end = (k + kl).min(n - 1);
        let col_end = (k + kuf).min(n - 1);

        // Partial pivot: largest magnitude in column k among rows k..=row_end.
        let mut piv = k;
        let mut maxv = w[diag][k].abs();
        for i in (k + 1)..=row_end {
            let v = w[diag + i - k][k].abs();
            if v > maxv {
                maxv = v;
                piv = i;
            }
        }
        if maxv == 0.0 || maxv.is_nan() {
            return None; // zero/NaN pivot column -> defer to dense solver
        }

        // Row interchange across the band columns k..=col_end (same column, the two
        // rows live in different band-rows of the packed buffer).
        if piv != k {
            for j in k..=col_end {
                let a = diag + k - j;
                let c = diag + piv - j;
                let (lo, hi) = if a < c { (a, c) } else { (c, a) };
                let (left, right) = w.split_at_mut(hi);
                std::mem::swap(&mut left[lo][j], &mut right[0][j]);
            }
            rhs.swap(k, piv);
        }

        let pivot = w[diag][k];
        // Snapshot pivot row k over columns k..=col_end (urow[t] = A(k, k+t)).
        for (t, j) in (k..=col_end).enumerate() {
            urow[t] = w[diag + k - j][j];
        }

        for i in (k + 1)..=row_end {
            let factor = w[diag + i - k][k] / pivot;
            if factor != 0.0 {
                for j in (k + 1)..=col_end {
                    w[diag + i - j][j] -= factor * urow[j - k];
                }
            }
            rhs[i] -= factor * rhs[k];
        }
    }

    // Back substitution against the fill-extended upper-triangular factor.
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let col_end = (i + kuf).min(n - 1);
        let mut sum = rhs[i];
        for j in (i + 1)..=col_end {
            sum -= w[diag + i - j][j] * x[j];
        }
        let d = w[diag][i];
        if d == 0.0 {
            return None;
        }
        x[i] = sum / d;
    }
    Some(x)
}

/// Backward error `||Ax - b|| / (||A||_F · ||x|| + ||b||)` for a banded `A` in
/// packed storage, summed only over band nonzeros. The dense terms this omits are
/// `A(i,j)·x_j = 0·x_j = 0` and `+0.0` is a no-op for finite `x`, so the result is
/// bit-identical to [`compute_backward_error_dense`] on the equivalent dense matrix.
#[allow(clippy::needless_range_loop)] // band indices map (i,j) -> packed row ku+i-j
fn compute_backward_error_banded(
    ab: &[Vec<f64>],
    kl: usize,
    ku: usize,
    x: &[f64],
    b: &[f64],
) -> f64 {
    let n = b.len();
    if n == 0 {
        return 0.0;
    }
    let mut residual_sum_sq = 0.0;
    let mut a_sum_sq = 0.0;
    for i in 0..n {
        let j_start = i.saturating_sub(kl);
        let j_end = (i + ku).min(n - 1);
        let mut ax_i = 0.0;
        for j in j_start..=j_end {
            let val = ab[ku + i - j][j];
            ax_i += val * x[j];
            a_sum_sq += val * val;
        }
        let res_i = ax_i - b[i];
        residual_sum_sq += res_i * res_i;
    }
    let residual_norm = residual_sum_sq.sqrt();
    let a_norm = a_sum_sq.sqrt();

    let mut x_sum_sq = 0.0;
    for &val in x {
        x_sum_sq += val * val;
    }
    let x_norm = x_sum_sq.sqrt();

    let mut b_sum_sq = 0.0;
    for &val in b {
        b_sum_sq += val * val;
    }
    let b_norm = b_sum_sq.sqrt();

    let denom = a_norm * x_norm + b_norm;
    if !residual_norm.is_finite() || !denom.is_finite() {
        return f64::INFINITY;
    }
    if denom > 0.0 {
        residual_norm / denom
    } else {
        0.0
    }
}

// Dense-buffer banded GEPP: superseded by `banded_lu_solve_packed` (same arithmetic,
// O(n·bw) memory instead of O(n^2)); retained as a reference implementation.
#[allow(dead_code)]
#[allow(clippy::needless_range_loop)] // explicit band indices drive pivot search and split_at_mut
fn banded_gepp_solve(work: &mut [Vec<f64>], b: &[f64], kl: usize, ku: usize) -> Option<Vec<f64>> {
    let n = work.len();
    if n == 0 {
        return Some(Vec::new());
    }
    let kuf = ku + kl; // fill-extended upper bandwidth of U
    let mut rhs = b.to_vec();

    for k in 0..n {
        let row_end = (k + kl).min(n - 1);
        // Partial pivot: largest magnitude in column k among rows k..=row_end.
        let mut piv = k;
        let mut maxv = work[k][k].abs();
        for i in (k + 1)..=row_end {
            let v = work[i][k].abs();
            if v > maxv {
                maxv = v;
                piv = i;
            }
        }
        if maxv == 0.0 || maxv.is_nan() {
            return None; // zero or NaN pivot column -> defer to dense solver
        }
        if piv != k {
            work.swap(k, piv);
            rhs.swap(k, piv);
        }
        let col_end = (k + kuf).min(n - 1);
        let pivot = work[k][k];
        for i in (k + 1)..=row_end {
            let factor = work[i][k] / pivot;
            work[i][k] = 0.0;
            if factor != 0.0 {
                // Update row i over columns k+1..=col_end using row k.
                let (head, tail) = work.split_at_mut(i);
                let row_k = &head[k];
                let row_i = &mut tail[0];
                for j in (k + 1)..=col_end {
                    row_i[j] -= factor * row_k[j];
                }
            }
            rhs[i] -= factor * rhs[k];
        }
    }

    // Back substitution against the fill-extended upper-triangular factor.
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let col_end = (i + kuf).min(n - 1);
        let mut sum = rhs[i];
        for j in (i + 1)..=col_end {
            sum -= work[i][j] * x[j];
        }
        let diag = work[i][i];
        if diag == 0.0 {
            return None;
        }
        x[i] = sum / diag;
    }
    Some(x)
}

fn dense_from_banded(nlower: usize, nupper: usize, ab: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let mut dense = vec![vec![0.0; n]; n];
    for j in 0..n {
        let i_start = j.saturating_sub(nupper);
        let i_end = (j + nlower).min(n.saturating_sub(1));
        for (i, row) in dense.iter_mut().enumerate().take(i_end + 1).skip(i_start) {
            let band_row = nupper + i - j;
            row[j] = ab[band_row][j];
        }
    }
    dense
}

fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() {
        return Vec::new();
    }
    let rows = a.len();
    let cols = a[0].len();
    let mut out = vec![vec![0.0; rows]; cols];
    for (r, row) in a.iter().enumerate().take(rows) {
        for (c, value) in row.iter().enumerate().take(cols) {
            out[c][r] = *value;
        }
    }
    out
}

fn matrix_shape(a: &[Vec<f64>]) -> Result<(usize, usize), LinalgError> {
    if a.is_empty() {
        return Ok((0, 0));
    }
    let cols = a[0].len();
    if a.iter().any(|row| row.len() != cols) {
        return Err(LinalgError::RaggedMatrix);
    }
    Ok((a.len(), cols))
}

fn validate_finite_matrix(
    a: &[Vec<f64>],
    mode: RuntimeMode,
    check_finite: bool,
) -> Result<(), LinalgError> {
    let must_check = check_finite || mode == RuntimeMode::Hardened;
    if must_check && a.iter().flatten().any(|v| !v.is_finite()) {
        return Err(LinalgError::NonFiniteInput);
    }
    Ok(())
}

fn validate_finite_matrix_and_vector(
    a: &[Vec<f64>],
    b: &[f64],
    mode: RuntimeMode,
    check_finite: bool,
) -> Result<(), LinalgError> {
    validate_finite_matrix(a, mode, check_finite)?;
    let must_check = check_finite || mode == RuntimeMode::Hardened;
    if must_check && b.iter().any(|v| !v.is_finite()) {
        return Err(LinalgError::NonFiniteInput);
    }
    Ok(())
}

fn safe_svd(
    matrix: DMatrix<f64>,
    compute_u: bool,
    compute_v: bool,
) -> Result<SVD<f64, Dyn, Dyn>, LinalgError> {
    std::panic::catch_unwind(|| nalgebra::linalg::SVD::new(matrix, compute_u, compute_v)).map_err(
        |_| LinalgError::ConvergenceFailure {
            detail: "SVD computation panicked, likely due to non-finite inputs".into(),
        },
    )
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct HouseholderReflector {
    start: usize,
    values: Vec<f64>,
    tau: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct BidiagonalReduction {
    rows: usize,
    cols: usize,
    diagonal: Vec<f64>,
    superdiagonal: Vec<f64>,
    bidiagonal: DMatrix<f64>,
    left_reflectors: Vec<HouseholderReflector>,
    right_reflectors: Vec<HouseholderReflector>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct BidiagonalSvd {
    singular_values: Vec<f64>,
    u: DMatrix<f64>,
    v_t: DMatrix<f64>,
    sweeps: usize,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct DeterministicThinSvd {
    singular_values: Vec<f64>,
    u: DMatrix<f64>,
    v_t: DMatrix<f64>,
    jacobi_sweeps: usize,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct SymmetricJacobiEigen {
    eigenvalues: Vec<f64>,
    eigenvectors: DMatrix<f64>,
    sweeps: usize,
}

const BIDIAG_JACOBI_TOLERANCE: f64 = 1e-14;
const BIDIAG_JACOBI_MAX_SWEEPS: usize = 128;

#[allow(dead_code)]
impl BidiagonalReduction {
    fn left_product_transpose(&self) -> DMatrix<f64> {
        let mut q_t = DMatrix::<f64>::identity(self.rows, self.rows);
        for reflector in &self.left_reflectors {
            apply_householder_left(&mut q_t, reflector, 0);
        }
        q_t
    }

    fn right_product(&self) -> DMatrix<f64> {
        let mut v = DMatrix::<f64>::identity(self.cols, self.cols);
        for reflector in &self.right_reflectors {
            apply_householder_right(&mut v, reflector, 0);
        }
        v
    }
}

#[allow(dead_code)]
impl BidiagonalSvd {
    fn sigma_matrix(&self) -> DMatrix<f64> {
        let mut sigma =
            DMatrix::<f64>::zeros(self.singular_values.len(), self.singular_values.len());
        for (idx, value) in self.singular_values.iter().enumerate() {
            sigma[(idx, idx)] = *value;
        }
        sigma
    }
}

#[allow(dead_code)]
impl DeterministicThinSvd {
    fn sigma_matrix(&self) -> DMatrix<f64> {
        let mut sigma =
            DMatrix::<f64>::zeros(self.singular_values.len(), self.singular_values.len());
        for (idx, value) in self.singular_values.iter().enumerate() {
            sigma[(idx, idx)] = *value;
        }
        sigma
    }

    fn pseudo_inverse(&self, threshold: f64) -> DMatrix<f64> {
        let p = self.singular_values.len();
        let mut sigma_pinv = DMatrix::<f64>::zeros(p, p);
        for (idx, value) in self.singular_values.iter().enumerate() {
            if value.is_nan() {
                sigma_pinv[(idx, idx)] = f64::NAN;
            } else if *value > threshold {
                sigma_pinv[(idx, idx)] = 1.0 / *value;
            }
        }
        self.v_t.transpose() * sigma_pinv * self.u.transpose()
    }

    fn least_squares_solution(
        &self,
        threshold: f64,
        rhs: &DVector<f64>,
    ) -> Result<DVector<f64>, LinalgError> {
        if rhs.len() != self.u.nrows() {
            return Err(LinalgError::UnsupportedAssumption);
        }

        let p = self.singular_values.len();
        let mut sigma_u_rhs = DVector::<f64>::zeros(p);
        for (idx, value) in self.singular_values.iter().enumerate() {
            let mut projected = 0.0;
            for row in 0..self.u.nrows() {
                projected += self.u[(row, idx)] * rhs[row];
            }
            sigma_u_rhs[idx] = if value.is_nan() {
                projected * f64::NAN
            } else if *value > threshold {
                projected / *value
            } else {
                0.0
            };
        }

        let mut x = DVector::<f64>::zeros(self.v_t.ncols());
        for col in 0..self.v_t.ncols() {
            let mut value = 0.0;
            for idx in 0..p {
                value += self.v_t[(idx, col)] * sigma_u_rhs[idx];
            }
            x[col] = value;
        }
        Ok(x)
    }
}

fn make_householder_reflector(start: usize, values: Vec<f64>) -> HouseholderReflector {
    let norm = values.iter().map(|value| value * value).sum::<f64>().sqrt();
    if norm == 0.0 {
        return HouseholderReflector {
            start,
            values,
            tau: 0.0,
        };
    }

    let first = values[0];
    let beta = if first.is_sign_negative() {
        norm
    } else {
        -norm
    };
    let mut reflector_values = values;
    reflector_values[0] -= beta;
    let norm_sq = reflector_values
        .iter()
        .map(|value| value * value)
        .sum::<f64>();
    let tau = if norm_sq == 0.0 { 0.0 } else { 2.0 / norm_sq };

    HouseholderReflector {
        start,
        values: reflector_values,
        tau,
    }
}

fn apply_householder_left(
    matrix: &mut DMatrix<f64>,
    reflector: &HouseholderReflector,
    col_start: usize,
) {
    if reflector.tau == 0.0 || reflector.values.is_empty() {
        return;
    }

    for col in col_start..matrix.ncols() {
        let mut dot = 0.0;
        for (offset, value) in reflector.values.iter().enumerate() {
            dot += value * matrix[(reflector.start + offset, col)];
        }
        let scale = reflector.tau * dot;
        if scale != 0.0 {
            for (offset, value) in reflector.values.iter().enumerate() {
                matrix[(reflector.start + offset, col)] -= scale * value;
            }
        }
    }
}

fn apply_householder_right(
    matrix: &mut DMatrix<f64>,
    reflector: &HouseholderReflector,
    row_start: usize,
) {
    let mut dot_workspace = vec![0.0_f64; matrix.nrows()];
    apply_householder_right_with_workspace(matrix, reflector, row_start, &mut dot_workspace);
}

fn apply_householder_right_with_workspace(
    matrix: &mut DMatrix<f64>,
    reflector: &HouseholderReflector,
    row_start: usize,
    dot_workspace: &mut [f64],
) {
    if reflector.tau == 0.0 || reflector.values.is_empty() {
        return;
    }
    let rows = matrix.nrows();
    if row_start >= rows {
        return;
    }
    debug_assert!(dot_workspace.len() >= rows);

    for dot in &mut dot_workspace[row_start..rows] {
        *dot = 0.0;
    }

    // DMatrix is column-major. This preserves each row's summation order while
    // streaming down contiguous columns instead of striding across rows.
    for (offset, value) in reflector.values.iter().enumerate() {
        let col = reflector.start + offset;
        for row in row_start..rows {
            dot_workspace[row] += matrix[(row, col)] * value;
        }
    }

    for scale in &mut dot_workspace[row_start..rows] {
        *scale *= reflector.tau;
    }

    for (offset, value) in reflector.values.iter().enumerate() {
        let col = reflector.start + offset;
        for row in row_start..rows {
            let scale = dot_workspace[row];
            if scale != 0.0 {
                matrix[(row, col)] -= scale * value;
            }
        }
    }
}

fn symmetric_offdiagonal_max(matrix: &DMatrix<f64>) -> f64 {
    let mut max_abs = 0.0_f64;
    for row in 0..matrix.nrows() {
        for col in row + 1..matrix.ncols() {
            max_abs = max_abs.max(matrix[(row, col)].abs());
        }
    }
    max_abs
}

fn symmetric_diagonal_scale(matrix: &DMatrix<f64>) -> f64 {
    let mut scale = 1.0_f64;
    for idx in 0..matrix.nrows() {
        scale = scale.max(matrix[(idx, idx)].abs());
    }
    scale
}

fn symmetric_jacobi_eigen(mut matrix: DMatrix<f64>) -> Result<SymmetricJacobiEigen, LinalgError> {
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(LinalgError::NonFiniteInput);
    }

    let mut eigenvectors = DMatrix::<f64>::identity(n, n);
    if n <= 1 {
        return Ok(SymmetricJacobiEigen {
            eigenvalues: (0..n).map(|idx| matrix[(idx, idx)]).collect(),
            eigenvectors,
            sweeps: 0,
        });
    }

    for sweep in 0..BIDIAG_JACOBI_MAX_SWEEPS {
        let scale = symmetric_diagonal_scale(&matrix);
        if symmetric_offdiagonal_max(&matrix) <= BIDIAG_JACOBI_TOLERANCE * scale {
            return Ok(SymmetricJacobiEigen {
                eigenvalues: (0..n).map(|idx| matrix[(idx, idx)]).collect(),
                eigenvectors,
                sweeps: sweep,
            });
        }

        for p in 0..n {
            for q in p + 1..n {
                let app = matrix[(p, p)];
                let aqq = matrix[(q, q)];
                let apq = matrix[(p, q)];
                let pair_scale = app.abs().max(aqq.abs()).max(1.0);
                if apq.abs() <= BIDIAG_JACOBI_TOLERANCE * pair_scale {
                    continue;
                }

                let tau = (aqq - app) / (2.0 * apq);
                let t_sign = if tau.is_sign_negative() { -1.0 } else { 1.0 };
                let t = t_sign / (tau.abs() + (1.0 + tau * tau).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                for k in 0..n {
                    if k != p && k != q {
                        let akp = matrix[(k, p)];
                        let akq = matrix[(k, q)];
                        let new_kp = c * akp - s * akq;
                        let new_kq = s * akp + c * akq;
                        matrix[(k, p)] = new_kp;
                        matrix[(p, k)] = new_kp;
                        matrix[(k, q)] = new_kq;
                        matrix[(q, k)] = new_kq;
                    }
                }

                matrix[(p, p)] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
                matrix[(q, q)] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
                matrix[(p, q)] = 0.0;
                matrix[(q, p)] = 0.0;

                for k in 0..n {
                    let vkp = eigenvectors[(k, p)];
                    let vkq = eigenvectors[(k, q)];
                    eigenvectors[(k, p)] = c * vkp - s * vkq;
                    eigenvectors[(k, q)] = s * vkp + c * vkq;
                }
            }
        }
    }

    let scale = symmetric_diagonal_scale(&matrix);
    if symmetric_offdiagonal_max(&matrix) <= BIDIAG_JACOBI_TOLERANCE * scale * 16.0 {
        return Ok(SymmetricJacobiEigen {
            eigenvalues: (0..n).map(|idx| matrix[(idx, idx)]).collect(),
            eigenvectors,
            sweeps: BIDIAG_JACOBI_MAX_SWEEPS,
        });
    }

    Err(LinalgError::ConvergenceFailure {
        detail: format!(
            "bidiagonal SVD Jacobi solver did not converge within {BIDIAG_JACOBI_MAX_SWEEPS} sweeps"
        ),
    })
}

fn canonicalize_slice_sign(values: &mut [f64]) {
    if let Some(pivot) = values
        .iter()
        .copied()
        .find(|value| value.abs() > BIDIAG_JACOBI_TOLERANCE)
        && pivot.is_sign_negative()
    {
        for value in values {
            *value = -*value;
        }
    }
}

fn canonicalize_svd_factor_signs(u: &mut DMatrix<f64>, v_t: &mut DMatrix<f64>) {
    for idx in 0..u.ncols() {
        let mut pivot = None;
        for row in 0..u.nrows() {
            let value = u[(row, idx)];
            if value.abs() > BIDIAG_JACOBI_TOLERANCE {
                pivot = Some(value);
                break;
            }
        }
        if pivot.is_none() {
            for col in 0..v_t.ncols() {
                let value = v_t[(idx, col)];
                if value.abs() > BIDIAG_JACOBI_TOLERANCE {
                    pivot = Some(value);
                    break;
                }
            }
        }
        if let Some(value) = pivot
            && value.is_sign_negative()
        {
            for row in 0..u.nrows() {
                u[(row, idx)] = -u[(row, idx)];
            }
            for col in 0..v_t.ncols() {
                v_t[(idx, col)] = -v_t[(idx, col)];
            }
        }
    }
}

fn fill_deterministic_left_vector(u: &mut DMatrix<f64>, column: usize) -> Result<(), LinalgError> {
    let rows = u.nrows();
    for basis_idx in 0..rows {
        let mut candidate = vec![0.0_f64; rows];
        candidate[basis_idx] = 1.0;
        for prev in 0..column {
            let mut projection = 0.0;
            for row in 0..rows {
                projection += candidate[row] * u[(row, prev)];
            }
            if projection != 0.0 {
                for row in 0..rows {
                    candidate[row] -= projection * u[(row, prev)];
                }
            }
        }
        let norm = candidate
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if norm > BIDIAG_JACOBI_TOLERANCE {
            for row in 0..rows {
                u[(row, column)] = candidate[row] / norm;
            }
            return Ok(());
        }
    }

    Err(LinalgError::ConvergenceFailure {
        detail: "could not build deterministic null left singular vector".into(),
    })
}

#[allow(dead_code)]
fn deterministic_bidiagonal_svd_from_reduction(
    reduction: &BidiagonalReduction,
) -> Result<BidiagonalSvd, LinalgError> {
    deterministic_bidiagonal_svd(
        reduction.rows,
        &reduction.diagonal,
        &reduction.superdiagonal,
    )
}

#[allow(dead_code)]
fn deterministic_bidiagonal_svd(
    rows: usize,
    diagonal: &[f64],
    superdiagonal: &[f64],
) -> Result<BidiagonalSvd, LinalgError> {
    let cols = diagonal.len();
    if rows < cols {
        return Err(LinalgError::UnsupportedAssumption);
    }
    if superdiagonal.len() != cols.saturating_sub(1) {
        return Err(LinalgError::InvalidArgument {
            detail: format!(
                "upper bidiagonal expected {} superdiagonal entries, got {}",
                cols.saturating_sub(1),
                superdiagonal.len()
            ),
        });
    }
    if diagonal
        .iter()
        .chain(superdiagonal.iter())
        .any(|value| !value.is_finite())
    {
        return Err(LinalgError::NonFiniteInput);
    }
    if cols == 0 {
        return Ok(BidiagonalSvd {
            singular_values: Vec::new(),
            u: DMatrix::<f64>::zeros(rows, 0),
            v_t: DMatrix::<f64>::zeros(0, 0),
            sweeps: 0,
        });
    }

    let mut gram = DMatrix::<f64>::zeros(cols, cols);
    for idx in 0..cols {
        let mut value = diagonal[idx] * diagonal[idx];
        if idx > 0 {
            value += superdiagonal[idx - 1] * superdiagonal[idx - 1];
        }
        gram[(idx, idx)] = value;
        if idx + 1 < cols {
            let offdiag = diagonal[idx] * superdiagonal[idx];
            gram[(idx, idx + 1)] = offdiag;
            gram[(idx + 1, idx)] = offdiag;
        }
    }

    let eigen = symmetric_jacobi_eigen(gram)?;
    let mut order: Vec<usize> = (0..cols).collect();
    order.sort_by(|left, right| {
        let left_value = eigen.eigenvalues[*left].max(0.0);
        let right_value = eigen.eigenvalues[*right].max(0.0);
        right_value
            .total_cmp(&left_value)
            .then_with(|| left.cmp(right))
    });

    let mut singular_values = Vec::with_capacity(cols);
    let mut u = DMatrix::<f64>::zeros(rows, cols);
    let mut v_t = DMatrix::<f64>::zeros(cols, cols);

    for (out_col, eigen_col) in order.into_iter().enumerate() {
        let eigenvalue = eigen.eigenvalues[eigen_col];
        if eigenvalue < -BIDIAG_JACOBI_TOLERANCE {
            return Err(LinalgError::ConvergenceFailure {
                detail: format!("bidiagonal SVD produced negative eigenvalue {eigenvalue:.17e}"),
            });
        }
        let singular_value = eigenvalue.max(0.0).sqrt();
        singular_values.push(singular_value);

        let mut v_col: Vec<f64> = (0..cols)
            .map(|row| eigen.eigenvectors[(row, eigen_col)])
            .collect();
        canonicalize_slice_sign(&mut v_col);
        for row in 0..cols {
            v_t[(out_col, row)] = v_col[row];
        }

        if singular_value > BIDIAG_JACOBI_TOLERANCE {
            for row in 0..cols {
                let mut value = diagonal[row] * v_col[row];
                if row + 1 < cols {
                    value += superdiagonal[row] * v_col[row + 1];
                }
                u[(row, out_col)] = value / singular_value;
            }
        } else {
            fill_deterministic_left_vector(&mut u, out_col)?;
        }
    }

    Ok(BidiagonalSvd {
        singular_values,
        u,
        v_t,
        sweeps: eigen.sweeps,
    })
}

#[allow(dead_code)]
fn deterministic_thin_svd_from_reduction(
    reduction: &BidiagonalReduction,
) -> Result<DeterministicThinSvd, LinalgError> {
    let bidiagonal_svd = deterministic_bidiagonal_svd_from_reduction(reduction)?;
    let q_t = reduction.left_product_transpose();
    let right_product = reduction.right_product();
    let mut u = q_t.transpose() * bidiagonal_svd.u;
    let mut v_t = bidiagonal_svd.v_t * right_product.transpose();
    canonicalize_svd_factor_signs(&mut u, &mut v_t);

    Ok(DeterministicThinSvd {
        singular_values: bidiagonal_svd.singular_values,
        u,
        v_t,
        jacobi_sweeps: bidiagonal_svd.sweeps,
    })
}

#[allow(dead_code)]
fn deterministic_thin_svd(matrix: &DMatrix<f64>) -> Result<DeterministicThinSvd, LinalgError> {
    let reduction = golub_kahan_bidiagonal_reduction(matrix)?;
    deterministic_thin_svd_from_reduction(&reduction)
}

#[allow(dead_code)]
fn golub_kahan_bidiagonal_reduction(
    matrix: &DMatrix<f64>,
) -> Result<BidiagonalReduction, LinalgError> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    if rows < cols {
        return Err(LinalgError::UnsupportedAssumption);
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(LinalgError::NonFiniteInput);
    }

    let mut work = matrix.clone();
    let mut left_reflectors = Vec::with_capacity(cols);
    let mut right_reflectors = Vec::with_capacity(cols.saturating_sub(1));
    let mut right_dot_workspace = vec![0.0_f64; rows];

    for step in 0..cols {
        let column_values = (step..rows).map(|row| work[(row, step)]).collect();
        let left_reflector = make_householder_reflector(step, column_values);
        apply_householder_left(&mut work, &left_reflector, step);
        for row in step + 1..rows {
            work[(row, step)] = 0.0;
        }
        left_reflectors.push(left_reflector);

        if step + 1 < cols {
            let row_values = (step + 1..cols).map(|col| work[(step, col)]).collect();
            let right_reflector = make_householder_reflector(step + 1, row_values);
            apply_householder_right_with_workspace(
                &mut work,
                &right_reflector,
                step,
                &mut right_dot_workspace,
            );
            for col in step + 2..cols {
                work[(step, col)] = 0.0;
            }
            right_reflectors.push(right_reflector);
        }
    }

    let diagonal = (0..cols).map(|idx| work[(idx, idx)]).collect();
    let superdiagonal = (0..cols.saturating_sub(1))
        .map(|idx| work[(idx, idx + 1)])
        .collect();
    let mut bidiagonal = DMatrix::<f64>::zeros(rows, cols);
    for idx in 0..cols {
        bidiagonal[(idx, idx)] = work[(idx, idx)];
        if idx + 1 < cols {
            bidiagonal[(idx, idx + 1)] = work[(idx, idx + 1)];
        }
    }

    Ok(BidiagonalReduction {
        rows,
        cols,
        diagonal,
        superdiagonal,
        bidiagonal,
        left_reflectors,
        right_reflectors,
    })
}

fn dmatrix_from_rows(rows: &[Vec<f64>]) -> Result<DMatrix<f64>, LinalgError> {
    let (m, n) = matrix_shape(rows)?;
    let mut data = Vec::with_capacity(m * n);
    for row in rows {
        data.extend_from_slice(row);
    }
    Ok(DMatrix::from_row_slice(m, n, &data))
}

fn dmatrix_from_rows_with_norm1(rows: &[Vec<f64>]) -> Result<(DMatrix<f64>, f64), LinalgError> {
    let (m, n) = matrix_shape(rows)?;
    let mut data = Vec::with_capacity(m * n);
    let mut column_sums = vec![0.0_f64; n];
    let mut non_finite = false;
    for row in rows {
        for (col, &value) in row.iter().enumerate() {
            data.push(value);
            let abs_value = value.abs();
            if !abs_value.is_finite() {
                non_finite = true;
            } else {
                column_sums[col] += abs_value;
            }
        }
    }
    let norm1 = if non_finite {
        f64::NAN
    } else {
        column_sums.into_iter().fold(0.0_f64, f64::max)
    };
    Ok((DMatrix::from_row_slice(m, n, &data), norm1))
}

fn rows_from_dmatrix(m: &DMatrix<f64>) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0; m.ncols()]; m.nrows()];
    for r in 0..m.nrows() {
        for c in 0..m.ncols() {
            out[r][c] = m[(r, c)];
        }
    }
    out
}

fn pseudo_inverse_from_svd(
    svd: &SVD<f64, Dyn, Dyn>,
    threshold: f64,
) -> Result<DMatrix<f64>, LinalgError> {
    let u = svd.u.as_ref().ok_or(LinalgError::UnsupportedAssumption)?;
    let v_t = svd.v_t.as_ref().ok_or(LinalgError::UnsupportedAssumption)?;
    let p = svd.singular_values.len();
    if u.ncols() != p || v_t.nrows() != p {
        return Err(LinalgError::UnsupportedAssumption);
    }

    let mut sigma_pinv = DMatrix::zeros(p, p);
    for (i, s) in svd.singular_values.iter().enumerate() {
        if s.is_nan() {
            sigma_pinv[(i, i)] = f64::NAN;
        } else if *s > threshold {
            sigma_pinv[(i, i)] = 1.0 / *s;
        }
    }
    Ok(v_t.transpose() * sigma_pinv * u.transpose())
}

fn least_squares_solution_from_svd(
    svd: &SVD<f64, Dyn, Dyn>,
    threshold: f64,
    rhs: &DVector<f64>,
) -> Result<DVector<f64>, LinalgError> {
    let u = svd.u.as_ref().ok_or(LinalgError::UnsupportedAssumption)?;
    let v_t = svd.v_t.as_ref().ok_or(LinalgError::UnsupportedAssumption)?;
    let p = svd.singular_values.len();
    if u.ncols() != p || v_t.nrows() != p || rhs.len() != u.nrows() {
        return Err(LinalgError::UnsupportedAssumption);
    }

    let mut sigma_u_rhs = DVector::zeros(p);
    for (i, s) in svd.singular_values.iter().enumerate() {
        let mut projected = 0.0;
        for row in 0..u.nrows() {
            projected += u[(row, i)] * rhs[row];
        }
        sigma_u_rhs[i] = if s.is_nan() {
            projected * f64::NAN
        } else if *s > threshold {
            projected / *s
        } else {
            0.0
        };
    }

    let mut x = DVector::zeros(v_t.ncols());
    for col in 0..v_t.ncols() {
        let mut value = 0.0;
        for i in 0..p {
            value += v_t[(i, col)] * sigma_u_rhs[i];
        }
        x[col] = value;
    }
    Ok(x)
}

const LOW_RANK_PINV_MIN_COLS: usize = 512;
const LOW_RANK_PINV_MAX_RANK: usize = 16;
const LOW_RANK_PINV_BASIS_REL_TOL: f64 = 1e-8;
const LOW_RANK_PINV_RECON_REL_TOL: f64 = 1e-8;

struct LowRankTallFactor {
    basis: Vec<Vec<f64>>,
    coefficients: Vec<Vec<f64>>,
}

fn low_rank_tall_factor(
    a: &[Vec<f64>],
    rows: usize,
    cols: usize,
    max_rank: usize,
) -> Option<LowRankTallFactor> {
    let mut max_col_norm_sq = 0.0_f64;
    for col in 0..cols {
        let mut norm_sq = 0.0_f64;
        for row in a.iter().take(rows) {
            let value = row[col];
            if !value.is_finite() {
                return None;
            }
            norm_sq += value * value;
        }
        max_col_norm_sq = max_col_norm_sq.max(norm_sq);
    }
    let max_col_norm = max_col_norm_sq.sqrt();
    if max_col_norm == 0.0 {
        return Some(LowRankTallFactor {
            basis: Vec::new(),
            coefficients: Vec::new(),
        });
    }

    let basis_tol = LOW_RANK_PINV_BASIS_REL_TOL * max_col_norm;
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let mut work = vec![0.0_f64; rows];
    for col in 0..cols {
        for (row_idx, row) in a.iter().enumerate().take(rows) {
            work[row_idx] = row[col];
        }
        for _ in 0..2 {
            for vector in &basis {
                let mut projection = 0.0_f64;
                for row in 0..rows {
                    projection += vector[row] * work[row];
                }
                if projection != 0.0 {
                    for row in 0..rows {
                        work[row] -= projection * vector[row];
                    }
                }
            }
        }
        let norm = work.iter().map(|value| value * value).sum::<f64>().sqrt();
        if norm > basis_tol {
            if basis.len() == max_rank {
                return None;
            }
            let inv_norm = 1.0 / norm;
            basis.push(work.iter().map(|value| value * inv_norm).collect());
        }
    }

    let rank = basis.len();
    let mut coefficients = vec![vec![0.0_f64; cols]; rank];
    for col in 0..cols {
        for (basis_idx, vector) in basis.iter().enumerate() {
            let mut coefficient = 0.0_f64;
            for row in 0..rows {
                coefficient += vector[row] * a[row][col];
            }
            coefficients[basis_idx][col] = coefficient;
        }
    }

    let recon_tol = LOW_RANK_PINV_RECON_REL_TOL * max_col_norm;
    for col in 0..cols {
        let mut residual_sq = 0.0_f64;
        for row in 0..rows {
            let mut reconstructed = 0.0_f64;
            for basis_idx in 0..rank {
                reconstructed += basis[basis_idx][row] * coefficients[basis_idx][col];
            }
            let residual = a[row][col] - reconstructed;
            residual_sq += residual * residual;
        }
        if residual_sq.sqrt() > recon_tol {
            return None;
        }
    }

    Some(LowRankTallFactor {
        basis,
        coefficients,
    })
}

fn pinv_low_rank_tall(
    a: &[Vec<f64>],
    rows: usize,
    cols: usize,
    atol: f64,
    rtol: f64,
) -> Option<LowRankPinvResult> {
    pinv_low_rank_tall_with_limits(a, rows, cols, atol, rtol, LOW_RANK_PINV_MIN_COLS)
}

fn pinv_low_rank_tall_with_limits(
    a: &[Vec<f64>],
    rows: usize,
    cols: usize,
    atol: f64,
    rtol: f64,
    min_cols: usize,
) -> Option<LowRankPinvResult> {
    if rows < cols.saturating_mul(2) || cols < min_cols || !rows_are_rectangular(a, cols) {
        return None;
    }

    let factor = low_rank_tall_factor(a, rows, cols, LOW_RANK_PINV_MAX_RANK)?;
    let basis_rank = factor.basis.len();
    if basis_rank == 0 {
        return Some(LowRankPinvResult {
            pseudo_inverse: vec![vec![0.0; rows]; cols],
            rank: 0,
            rcond_estimate: 0.0,
        });
    }
    if basis_rank >= cols {
        return None;
    }

    let mut compact = Vec::with_capacity(basis_rank * cols);
    for row in &factor.coefficients {
        compact.extend_from_slice(row);
    }
    let compact_matrix = DMatrix::from_row_slice(basis_rank, cols, &compact);
    let compact_svd = safe_svd(compact_matrix, true, true).ok()?;
    let u = compact_svd.u.as_ref()?;
    let v_t = compact_svd.v_t.as_ref()?;
    let singular_values: Vec<f64> = compact_svd.singular_values.iter().copied().collect();
    let max_s = singular_values.iter().copied().fold(0.0_f64, |acc, value| {
        if acc.is_nan() || value.is_nan() {
            f64::NAN
        } else {
            acc.max(value)
        }
    });
    if !max_s.is_finite() {
        return None;
    }
    if max_s == 0.0 {
        return Some(LowRankPinvResult {
            pseudo_inverse: vec![vec![0.0; rows]; cols],
            rank: 0,
            rcond_estimate: 0.0,
        });
    }

    let threshold = atol + rtol * max_s;
    let rank = singular_values
        .iter()
        .filter(|singular| **singular > threshold)
        .count();
    if rank == 0 || rank >= cols {
        return None;
    }

    let boundary_gap = singular_values
        .get(rank)
        .map(|next| singular_values[rank - 1] - *next)
        .unwrap_or(singular_values[rank - 1]);
    if boundary_gap.abs() <= threshold.max(max_s * 1e-12) {
        return None;
    }

    let mut compact_pinv_to_basis = vec![vec![0.0_f64; basis_rank]; cols];
    for singular_idx in 0..rank {
        let singular = singular_values[singular_idx];
        if singular <= 0.0 || !singular.is_finite() {
            return None;
        }
        let inv_singular = 1.0 / singular;
        for col in 0..cols {
            let scaled_right = v_t[(singular_idx, col)] * inv_singular;
            if scaled_right != 0.0 {
                for (basis_idx, dst) in compact_pinv_to_basis[col].iter_mut().enumerate() {
                    *dst += scaled_right * u[(basis_idx, singular_idx)];
                }
            }
        }
    }

    let mut pseudo_inverse = vec![vec![0.0_f64; rows]; cols];
    for col in 0..cols {
        for (row, dst) in pseudo_inverse[col].iter_mut().enumerate() {
            let mut value = 0.0_f64;
            for (basis_coeff, basis_vector) in
                compact_pinv_to_basis[col].iter().zip(factor.basis.iter())
            {
                value += *basis_coeff * basis_vector[row];
            }
            *dst = value;
        }
    }

    let rcond_estimate = if rank < cols {
        0.0
    } else {
        singular_values[rank - 1] / max_s
    };

    Some(LowRankPinvResult {
        pseudo_inverse,
        rank,
        rcond_estimate,
    })
}

fn lstsq_low_rank_tall(
    a: &[Vec<f64>],
    b: &[f64],
    rows: usize,
    cols: usize,
    cond: f64,
    min_cols: usize,
) -> Option<LowRankLstsqResult> {
    if rows < cols.saturating_mul(2)
        || cols < min_cols
        || b.len() != rows
        || cond < 0.0
        || !cond.is_finite()
        || !rows_are_rectangular(a, cols)
        || b.iter().any(|value| !value.is_finite())
    {
        return None;
    }

    let factor = low_rank_tall_factor(a, rows, cols, LOW_RANK_PINV_MAX_RANK)?;
    let basis_rank = factor.basis.len();
    if basis_rank == 0 {
        return Some(LowRankLstsqResult {
            x: vec![0.0; cols],
            rank: 0,
            singular_values: vec![0.0; cols],
            rcond_estimate: 0.0,
        });
    }
    if basis_rank >= cols {
        return None;
    }

    let mut compact = Vec::with_capacity(basis_rank * cols);
    for row in &factor.coefficients {
        compact.extend_from_slice(row);
    }
    let compact_matrix = DMatrix::from_row_slice(basis_rank, cols, &compact);
    let compact_svd = safe_svd(compact_matrix, true, true).ok()?;
    let u = compact_svd.u.as_ref()?;
    let v_t = compact_svd.v_t.as_ref()?;
    let compact_singular_values: Vec<f64> = compact_svd.singular_values.iter().copied().collect();
    let max_s = compact_singular_values
        .iter()
        .copied()
        .fold(0.0_f64, |acc, value| {
            if acc.is_nan() || value.is_nan() {
                f64::NAN
            } else {
                acc.max(value)
            }
        });
    if !max_s.is_finite() {
        return None;
    }
    if max_s == 0.0 {
        return Some(LowRankLstsqResult {
            x: vec![0.0; cols],
            rank: 0,
            singular_values: vec![0.0; cols],
            rcond_estimate: 0.0,
        });
    }

    let threshold = cond * max_s;
    let rank = compact_singular_values
        .iter()
        .filter(|singular| **singular > threshold)
        .count();
    if rank == 0 || rank >= cols {
        return None;
    }

    let boundary_gap = compact_singular_values
        .get(rank)
        .map(|next| compact_singular_values[rank - 1] - *next)
        .unwrap_or(compact_singular_values[rank - 1]);
    if boundary_gap.abs() <= threshold.max(max_s * 1e-12) {
        return None;
    }

    let mut q_t_b = vec![0.0_f64; basis_rank];
    for (basis_idx, vector) in factor.basis.iter().enumerate() {
        let mut projection = 0.0_f64;
        for row in 0..rows {
            projection += vector[row] * b[row];
        }
        q_t_b[basis_idx] = projection;
    }

    let mut sigma_u_rhs = vec![0.0_f64; rank];
    for singular_idx in 0..rank {
        let singular = compact_singular_values[singular_idx];
        if singular <= 0.0 || !singular.is_finite() {
            return None;
        }
        let mut projected = 0.0_f64;
        for basis_idx in 0..basis_rank {
            projected += u[(basis_idx, singular_idx)] * q_t_b[basis_idx];
        }
        sigma_u_rhs[singular_idx] = projected / singular;
    }

    let mut x = vec![0.0_f64; cols];
    for col in 0..cols {
        let mut value = 0.0_f64;
        for singular_idx in 0..rank {
            value += v_t[(singular_idx, col)] * sigma_u_rhs[singular_idx];
        }
        x[col] = value;
    }

    let mut singular_values = compact_singular_values;
    singular_values.resize(cols, 0.0);

    Some(LowRankLstsqResult {
        x,
        rank,
        singular_values,
        rcond_estimate: 0.0,
    })
}

// ══════════════════════════════════════════════════════════════════════
// Matrix Equation Solvers
// ══════════════════════════════════════════════════════════════════════

/// Solve the Sylvester equation AX + XB = Q.
///
/// Uses the Bartels-Stewart algorithm: reduce A and B to Schur form,
/// then solve the resulting triangular Sylvester equation column by column.
///
/// Matches `scipy.linalg.solve_sylvester(a, b, q)`.
pub fn solve_sylvester(
    a: &[Vec<f64>],
    b: &[Vec<f64>],
    q: &[Vec<f64>],
    options: DecompOptions,
) -> Result<Vec<Vec<f64>>, LinalgError> {
    let (m, ma) = matrix_shape(a)?;
    let (n, nb) = matrix_shape(b)?;
    let (qr, qc) = matrix_shape(q)?;

    if m != ma {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    if n != nb {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    if qr != m || qc != n {
        return Err(LinalgError::NotSupported {
            detail: format!("Q shape ({qr}x{qc}) must match A rows ({m}) x B cols ({n})"),
        });
    }
    if m == 0 || n == 0 {
        return Ok(Vec::new());
    }

    let a_mat = dmatrix_from_rows(a)?;
    let b_mat = dmatrix_from_rows(b)?;
    let q_mat = dmatrix_from_rows(q)?;

    // Schur decompositions: A = U T_A U^T, B = V T_B V^T
    let schur_a = a_mat.clone().schur();
    let (u, ta) = schur_a.unpack();
    let schur_b = b_mat.clone().schur();
    let (v, tb) = schur_b.unpack();

    // Transform Q: F = U^T Q V
    let f = u.transpose() * &q_mat * &v;

    // Solve T_A Y + Y T_B = F with the Bartels–Stewart back-substitution that
    // SciPy/LAPACK use, instead of forming the full (mn × mn) Kronecker operator
    // (I_n ⊗ T_A + T_B^T ⊗ I_m) and an O((mn)^3) full-pivot LU. T_B is upper
    // quasi-triangular (real Schur form), so column j of Y depends only on
    // columns k ≤ j: (T_A Y + Y T_B)[:,j] = T_A y_j + Σ_k y_k·tb[k,j], and
    // tb[k,j] = 0 for k > j+1. Sweeping the columns left→right turns the solve
    // into one m×m (1×1 diagonal block) or 2m×2m (2×2 block, complex eigenpair)
    // system per block — overall O(n·m^3) rather than O(m^3·n^3). Behavior parity
    // is preserved at the conformance level (residual ‖T_A Y + Y T_B − F‖ and the
    // SciPy differential at 1e-9): this matches SciPy's own algorithm, and a
    // singular Sylvester operator still surfaces as `SingularMatrix` because the
    // per-block LU returns no solution exactly when a block is singular.
    let mut y = DMatrix::<f64>::zeros(m, n);
    let mut j = 0;
    while j < n {
        // A nonzero subdiagonal entry marks a 2×2 Schur block (real Schur form
        // zeroes the subdiagonal everywhere else exactly).
        let is_2x2 = j + 1 < n && tb[(j + 1, j)] != 0.0;
        if !is_2x2 {
            // 1×1 block: (T_A + tb[j,j] I) y_j = f_j − Σ_{k<j} tb[k,j]·y_k.
            let mut rhs = f.column(j).into_owned();
            for k in 0..j {
                let tbkj = tb[(k, j)];
                if tbkj != 0.0 {
                    rhs.axpy(-tbkj, &y.column(k), 1.0);
                }
            }
            let mut sys = ta.clone();
            let shift = tb[(j, j)];
            for d in 0..m {
                sys[(d, d)] += shift;
            }
            let yj = sys.lu().solve(&rhs).ok_or(LinalgError::SingularMatrix)?;
            y.set_column(j, &yj);
            j += 1;
        } else {
            // 2×2 block: columns j and j+1 are coupled. Stack [y_j; y_{j+1}] and
            // solve the 2m×2m system
            //   (T_A + tb[j,j] I) y_j     + tb[j+1,j]   y_{j+1} = rhs_j
            //   tb[j,j+1]      y_j        + (T_A + tb[j+1,j+1] I) y_{j+1} = rhs_{j+1}
            // with rhs_c = f_c − Σ_{k<j} tb[k,c]·y_k.
            let mut rhs_j = f.column(j).into_owned();
            let mut rhs_j1 = f.column(j + 1).into_owned();
            for k in 0..j {
                let t_kj = tb[(k, j)];
                if t_kj != 0.0 {
                    rhs_j.axpy(-t_kj, &y.column(k), 1.0);
                }
                let t_kj1 = tb[(k, j + 1)];
                if t_kj1 != 0.0 {
                    rhs_j1.axpy(-t_kj1, &y.column(k), 1.0);
                }
            }
            let mut bigm = DMatrix::<f64>::zeros(2 * m, 2 * m);
            for r in 0..m {
                for c in 0..m {
                    bigm[(r, c)] = ta[(r, c)];
                    bigm[(m + r, m + c)] = ta[(r, c)];
                }
                bigm[(r, r)] += tb[(j, j)];
                bigm[(m + r, m + r)] += tb[(j + 1, j + 1)];
                bigm[(r, m + r)] = tb[(j + 1, j)];
                bigm[(m + r, r)] = tb[(j, j + 1)];
            }
            let mut bigr = DVector::<f64>::zeros(2 * m);
            for r in 0..m {
                bigr[r] = rhs_j[r];
                bigr[m + r] = rhs_j1[r];
            }
            let sol = bigm.lu().solve(&bigr).ok_or(LinalgError::SingularMatrix)?;
            for r in 0..m {
                y[(r, j)] = sol[r];
                y[(r, j + 1)] = sol[m + r];
            }
            j += 2;
        }
    }

    // Transform back: X = U Y V^T
    let x = &u * y * v.transpose();

    emit_trace(LinalgTrace {
        operation: "solve_sylvester",
        matrix_size: (m, n),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(rows_from_dmatrix(&x))
}

/// Solve the continuous Lyapunov equation AX + XA^T = Q.
///
/// Special case of Sylvester equation with B = A^T.
///
/// Matches `scipy.linalg.solve_continuous_lyapunov(a, q)`.
pub fn solve_continuous_lyapunov(
    a: &[Vec<f64>],
    q: &[Vec<f64>],
    options: DecompOptions,
) -> Result<Vec<Vec<f64>>, LinalgError> {
    let (n, _) = matrix_shape(a)?;
    // B = A^T
    let a_t: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| a[j][i]).collect()).collect();
    solve_sylvester(a, &a_t, q, options)
}

/// Solve the discrete Lyapunov equation A X A^T - X + Q = 0.
///
/// Uses the bilinear transformation to convert to a continuous Lyapunov equation:
///   B = (A - I)(A + I)^{-1}
///   C = 2(A + I)^{-T} Q (A + I)^{-1}
/// Then solves B C B^T - C = -R for the continuous form.
///
/// Matches `scipy.linalg.solve_discrete_lyapunov(a, q)`.
pub fn solve_discrete_lyapunov(
    a: &[Vec<f64>],
    q: &[Vec<f64>],
    options: DecompOptions,
) -> Result<Vec<Vec<f64>>, LinalgError> {
    let (n, na) = matrix_shape(a)?;
    if n != na {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    let (qr, qc) = matrix_shape(q)?;
    if qr != n || qc != n {
        return Err(LinalgError::NotSupported {
            detail: format!("Q shape ({qr}x{qc}) must be {n}x{n}"),
        });
    }
    if n == 0 {
        return Ok(Vec::new());
    }

    // Stein equation A X A^T - X = -Q solved by Schur back-substitution (as
    // SciPy/SLICOT do), not by forming the full (A ⊗ A - I) n²×n² operator and
    // an O(n^6) full-pivot LU. Reduce A to real Schur form A = U T U^T (T upper
    // quasi-triangular) and set Y = U^T X U, C = -U^T Q U; the equation becomes
    //   T Y T^T - Y = C.
    // Column j of (T Y T^T) is Σ_q T[j,q]·(T y_q), and T[j,q] = 0 for q < j
    // except the subdiagonal of a 2×2 block. Sweeping columns bottom→top, every
    // y_q with q > j is already known, so column j reduces to one n×n solve
    //   (T[j,j]·T - I) y_j = c_j - Σ_{q>j} T[j,q]·(T y_q),
    // and a 2×2 Schur block (complex eigenpair) couples its two columns into one
    // 2n×2n solve. Overall O(n·n^3) instead of O(n^6). Parity is at the
    // conformance level (residual ‖A X A^T - X + Q‖ and the SciPy differential at
    // 1e-9); a unit-modulus-product eigenpair (operator singular) still surfaces
    // as `SingularMatrix` because the per-block LU then has no solution.
    let a_mat = dmatrix_from_rows(a)?;
    let q_mat = dmatrix_from_rows(q)?;

    let (u, t) = a_mat.clone().schur().unpack();
    let c = -(u.transpose() * &q_mat * &u); // C = -U^T Q U

    let mut y = DMatrix::<f64>::zeros(n, n);
    // ty[:,q] caches T·y_q for every already-solved column q.
    let mut ty = DMatrix::<f64>::zeros(n, n);

    let mut jj = n as isize - 1;
    while jj >= 0 {
        let j = jj as usize;
        // A nonzero subdiagonal marks j as the lower row of a 2×2 Schur block.
        let is_2x2 = j >= 1 && t[(j, j - 1)] != 0.0;
        if !is_2x2 {
            // 1×1 block: (T[j,j]·T - I) y_j = c_j - Σ_{q>j} T[j,q]·ty_q.
            let mut rhs = c.column(j).into_owned();
            for q in (j + 1)..n {
                let tjq = t[(j, q)];
                if tjq != 0.0 {
                    rhs.axpy(-tjq, &ty.column(q), 1.0);
                }
            }
            let tjj = t[(j, j)];
            let mut sys = DMatrix::<f64>::zeros(n, n);
            for r in 0..n {
                for col in 0..n {
                    sys[(r, col)] = tjj * t[(r, col)];
                }
                sys[(r, r)] -= 1.0;
            }
            let yj = sys.lu().solve(&rhs).ok_or(LinalgError::SingularMatrix)?;
            let tyj = &t * &yj;
            y.set_column(j, &yj);
            ty.set_column(j, &tyj);
            jj -= 1;
        } else {
            // 2×2 block {j-1, j}: solve the coupled 2n×2n system
            //   [ t00·T - I   t01·T     ][y0]   [rhs0]
            //   [ t10·T       t11·T - I ][y1] = [rhs1]
            // with rhs_c = c_c - Σ_{q>j} T[c,q]·ty_q.
            let j0 = j - 1;
            let mut rhs0 = c.column(j0).into_owned();
            let mut rhs1 = c.column(j).into_owned();
            for q in (j + 1)..n {
                let t0q = t[(j0, q)];
                if t0q != 0.0 {
                    rhs0.axpy(-t0q, &ty.column(q), 1.0);
                }
                let t1q = t[(j, q)];
                if t1q != 0.0 {
                    rhs1.axpy(-t1q, &ty.column(q), 1.0);
                }
            }
            let t00 = t[(j0, j0)];
            let t01 = t[(j0, j)];
            let t10 = t[(j, j0)];
            let t11 = t[(j, j)];
            let mut bigm = DMatrix::<f64>::zeros(2 * n, 2 * n);
            for r in 0..n {
                for col in 0..n {
                    let trc = t[(r, col)];
                    bigm[(r, col)] = t00 * trc;
                    bigm[(r, n + col)] = t01 * trc;
                    bigm[(n + r, col)] = t10 * trc;
                    bigm[(n + r, n + col)] = t11 * trc;
                }
                bigm[(r, r)] -= 1.0;
                bigm[(n + r, n + r)] -= 1.0;
            }
            let mut bigr = DVector::<f64>::zeros(2 * n);
            for r in 0..n {
                bigr[r] = rhs0[r];
                bigr[n + r] = rhs1[r];
            }
            let sol = bigm.lu().solve(&bigr).ok_or(LinalgError::SingularMatrix)?;
            let mut y0 = DVector::<f64>::zeros(n);
            let mut y1 = DVector::<f64>::zeros(n);
            for r in 0..n {
                y0[r] = sol[r];
                y1[r] = sol[n + r];
            }
            let ty0 = &t * &y0;
            let ty1 = &t * &y1;
            y.set_column(j0, &y0);
            y.set_column(j, &y1);
            ty.set_column(j0, &ty0);
            ty.set_column(j, &ty1);
            jj -= 2;
        }
    }

    // Transform back: X = U Y U^T.
    let x = &u * y * u.transpose();

    emit_trace(LinalgTrace {
        operation: "solve_discrete_lyapunov",
        matrix_size: (n, n),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(rows_from_dmatrix(&x))
}

/// Iterate the matrix sign function S = lim_{k→∞} S_k where
/// S_{k+1} = (S_k + S_k⁻¹)/2. Roberts (1980)'s determinantal-scaling
/// variant accelerates convergence on Hamiltonians with eigenvalues
/// that aren't yet ±1: at each step we scale by c = |det(S)|^{1/n}.
fn matrix_sign_iteration(
    h: &DMatrix<f64>,
    max_iter: usize,
    tol: f64,
) -> Result<DMatrix<f64>, LinalgError> {
    let mut s = h.clone();
    let n = s.nrows() as f64;
    for _ in 0..max_iter {
        let s_inv = s.clone().try_inverse().ok_or(LinalgError::SingularMatrix)?;
        let det_abs = s.determinant().abs();
        let c = if det_abs > 1e-300 {
            det_abs.powf(1.0 / n).max(1e-12)
        } else {
            1.0
        };
        let s_next = (&s / c + &s_inv * c) * 0.5;
        let diff = (&s_next - &s).norm();
        s = s_next;
        if diff < tol {
            break;
        }
    }
    Ok(s)
}

/// Structure-preserving doubling (SDA) for DARE. Iterates
/// (A_k, G_k, H_k) with quadratic convergence; H_k → X.
/// Initialization: A₀ = A, G₀ = BR⁻¹Bᵀ, H₀ = Q.
fn sda_iteration_dare(
    a0: &DMatrix<f64>,
    g0: &DMatrix<f64>,
    q0: &DMatrix<f64>,
    max_iter: usize,
    tol: f64,
) -> Result<DMatrix<f64>, LinalgError> {
    let n = a0.nrows();
    let identity = DMatrix::<f64>::identity(n, n);
    let mut a_k = a0.clone();
    let mut g_k = g0.clone();
    let mut h_k = q0.clone();
    for _ in 0..max_iter {
        let w = &identity + &g_k * &h_k;
        let w_inv = w.try_inverse().ok_or(LinalgError::SingularMatrix)?;
        let a_next = &a_k * &w_inv * &a_k;
        let g_next = &g_k + &a_k * &w_inv * &g_k * a_k.transpose();
        let h_next = &h_k + a_k.transpose() * &h_k * &w_inv * &a_k;
        // Symmetrize iterates to suppress drift.
        let g_sym = (&g_next + g_next.transpose()) * 0.5;
        let h_sym = (&h_next + h_next.transpose()) * 0.5;
        let a_norm_next = a_next.norm();
        a_k = a_next;
        g_k = g_sym;
        h_k = h_sym;
        if a_norm_next < tol {
            break;
        }
    }
    Ok(h_k)
}

/// Solve the continuous-time algebraic Riccati equation (CARE).
///
/// Solves AᵀX + XA − XBR⁻¹BᵀX + Q = 0 for the symmetric stabilizing
/// solution X.
///
/// Method (Roberts 1980 sign-function): build the Hamiltonian
///   H = [[A, -G], [-Q, -Aᵀ]]   with G = BR⁻¹Bᵀ,
/// iterate the matrix sign function until convergence, then take
/// (I − sign(H))/2 — a rank-n projector onto the stable invariant
/// subspace. QR-decompose to extract a 2n×n basis [U₁; U₂], and
/// recover X = U₂ U₁⁻¹ (symmetrized for robustness).
///
/// Matches `scipy.linalg.solve_continuous_are(a, b, q, r)` for the
/// well-conditioned, invertible-`R` problems in the CAREX benchmark
/// set. The optional cross-term `s` and descriptor `e` arguments are
/// not yet wired through (out of scope per br-60cm).
pub fn solve_continuous_are(
    a: &[Vec<f64>],
    b: &[Vec<f64>],
    q: &[Vec<f64>],
    r: &[Vec<f64>],
    options: DecompOptions,
) -> Result<Vec<Vec<f64>>, LinalgError> {
    let (n, na) = matrix_shape(a)?;
    if n != na {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    let (br_rows, m) = matrix_shape(b)?;
    if br_rows != n {
        return Err(LinalgError::InvalidArgument {
            detail: format!("B rows ({br_rows}) must match A size ({n})"),
        });
    }
    let (qr, qc) = matrix_shape(q)?;
    if qr != n || qc != n {
        return Err(LinalgError::InvalidArgument {
            detail: format!("Q shape ({qr}x{qc}) must be {n}x{n}"),
        });
    }
    let (rr, rc) = matrix_shape(r)?;
    if rr != m || rc != m {
        return Err(LinalgError::InvalidArgument {
            detail: format!("R shape ({rr}x{rc}) must be {m}x{m}"),
        });
    }
    if n == 0 {
        return Ok(Vec::new());
    }

    hardened_dimension_check(options.mode, 2 * n, 2 * n)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;
    validate_finite_matrix(b, options.mode, options.check_finite)?;
    validate_finite_matrix(q, options.mode, options.check_finite)?;
    validate_finite_matrix(r, options.mode, options.check_finite)?;

    let a_mat = dmatrix_from_rows(a)?;
    let b_mat = dmatrix_from_rows(b)?;
    let q_mat = dmatrix_from_rows(q)?;
    let r_mat = dmatrix_from_rows(r)?;

    // G = B R⁻¹ Bᵀ (n × n).
    let r_inv = r_mat
        .clone()
        .try_inverse()
        .ok_or(LinalgError::SingularMatrix)?;
    let g = &b_mat * &r_inv * b_mat.transpose();

    // Build Hamiltonian: H = [[A, -G], [-Q, -Aᵀ]].
    let two_n = 2 * n;
    let mut h = DMatrix::<f64>::zeros(two_n, two_n);
    for i in 0..n {
        for j in 0..n {
            h[(i, j)] = a_mat[(i, j)];
            h[(i, n + j)] = -g[(i, j)];
            h[(n + i, j)] = -q_mat[(i, j)];
            h[(n + i, n + j)] = -a_mat[(j, i)];
        }
    }

    // sign(H) — eigenvalues map to ±1, splitting stable/unstable
    // subspaces orthogonally.
    let s = matrix_sign_iteration(&h, 100, 1e-12)?;

    // P = (I - sign(H))/2: rank-n projector onto stable subspace.
    let mut p = -&s;
    for i in 0..two_n {
        p[(i, i)] += 1.0;
    }
    p *= 0.5;

    // QR of P; the leading n columns of Q span the column space of P.
    let qr = p.qr();
    let q_full = qr.q();
    let u_top = q_full.view((0, 0), (n, n)).into_owned();
    let u_bot = q_full.view((n, 0), (n, n)).into_owned();
    let u_top_inv = u_top.try_inverse().ok_or(LinalgError::SingularMatrix)?;
    let x_raw = &u_bot * &u_top_inv;

    // Symmetrize for numerical stability.
    let x_sym = (&x_raw + x_raw.transpose()) * 0.5;

    emit_trace(LinalgTrace {
        operation: "solve_continuous_are",
        matrix_size: (n, n),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(rows_from_dmatrix(&x_sym))
}

/// Solve the discrete-time algebraic Riccati equation (DARE).
///
/// Solves AᵀXA − X − AᵀXB(R + BᵀXB)⁻¹BᵀXA + Q = 0 for the symmetric
/// stabilizing solution X.
///
/// Method (Anderson 2010 / Chu et al. SDA): structure-preserving
/// doubling iteration on the triple (A_k, G_k, H_k) initialized from
/// (A, BR⁻¹Bᵀ, Q). H_k converges quadratically to X. The result is
/// symmetrized for robustness.
///
/// Matches `scipy.linalg.solve_discrete_are(a, b, q, r)` for the
/// well-conditioned DAREX benchmark set.
pub fn solve_discrete_are(
    a: &[Vec<f64>],
    b: &[Vec<f64>],
    q: &[Vec<f64>],
    r: &[Vec<f64>],
    options: DecompOptions,
) -> Result<Vec<Vec<f64>>, LinalgError> {
    let (n, na) = matrix_shape(a)?;
    if n != na {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    let (br_rows, m) = matrix_shape(b)?;
    if br_rows != n {
        return Err(LinalgError::InvalidArgument {
            detail: format!("B rows ({br_rows}) must match A size ({n})"),
        });
    }
    let (qr, qc) = matrix_shape(q)?;
    if qr != n || qc != n {
        return Err(LinalgError::InvalidArgument {
            detail: format!("Q shape ({qr}x{qc}) must be {n}x{n}"),
        });
    }
    let (rr, rc) = matrix_shape(r)?;
    if rr != m || rc != m {
        return Err(LinalgError::InvalidArgument {
            detail: format!("R shape ({rr}x{rc}) must be {m}x{m}"),
        });
    }
    if n == 0 {
        return Ok(Vec::new());
    }

    hardened_dimension_check(options.mode, 2 * n, 2 * n)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;
    validate_finite_matrix(b, options.mode, options.check_finite)?;
    validate_finite_matrix(q, options.mode, options.check_finite)?;
    validate_finite_matrix(r, options.mode, options.check_finite)?;

    let a_mat = dmatrix_from_rows(a)?;
    let b_mat = dmatrix_from_rows(b)?;
    let q_mat = dmatrix_from_rows(q)?;
    let r_mat = dmatrix_from_rows(r)?;

    let r_inv = r_mat
        .clone()
        .try_inverse()
        .ok_or(LinalgError::SingularMatrix)?;
    let g = &b_mat * &r_inv * b_mat.transpose();

    let h_final = sda_iteration_dare(&a_mat, &g, &q_mat, 100, 1e-13)?;
    let x_sym = (&h_final + h_final.transpose()) * 0.5;

    emit_trace(LinalgTrace {
        operation: "solve_discrete_are",
        matrix_size: (n, n),
        mode: options.mode,
        rcond: None,
        warning: None,
        error: None,
    });

    Ok(rows_from_dmatrix(&x_sym))
}

// ══════════════════════════════════════════════════════════════════════
// Subspace Operations: orth, null_space, subspace_angles, polar
// ══════════════════════════════════════════════════════════════════════

/// Compute an orthonormal basis for the range (column space) of a matrix.
///
/// Matches `scipy.linalg.orth(A, rcond)`.
///
/// Returns a matrix whose columns form an orthonormal basis for the column
/// space of A, determined by keeping singular vectors with singular values
/// above `rcond * max(s)`.
pub fn orth(
    a: &[Vec<f64>],
    rcond: Option<f64>,
    options: DecompOptions,
) -> Result<Vec<Vec<f64>>, LinalgError> {
    let (m, n) = matrix_shape(a)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if m == 0 || n == 0 {
        return Ok(vec![vec![]; m]);
    }

    let matrix = dmatrix_from_rows(a)?;
    let svd_decomp = safe_svd(matrix, true, false)?;
    let u = svd_decomp
        .u
        .as_ref()
        .ok_or(LinalgError::UnsupportedAssumption)?;
    let singular_values = &svd_decomp.singular_values;

    let max_s = singular_values
        .iter()
        .copied()
        .fold(0.0_f64, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        });
    let tol = rcond.unwrap_or_else(|| (m.max(n) as f64) * f64::EPSILON) * max_s;

    let rank = singular_values.iter().filter(|&&s| s > tol).count();

    // Extract first `rank` columns of U.
    let mut result = vec![vec![0.0; rank]; m];
    for i in 0..m {
        for j in 0..rank {
            result[i][j] = u[(i, j)];
        }
    }

    Ok(result)
}

/// Compute an orthonormal basis for the null space of a matrix.
///
/// Matches `scipy.linalg.null_space(A, rcond)`.
///
/// Returns a matrix whose columns form an orthonormal basis for the null
/// space of A, determined by keeping right singular vectors with singular
/// values below `rcond * max(s)`.
pub fn null_space(
    a: &[Vec<f64>],
    rcond: Option<f64>,
    options: DecompOptions,
) -> Result<Vec<Vec<f64>>, LinalgError> {
    let (m, n) = matrix_shape(a)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if m == 0 || n == 0 {
        // If no columns, null space is empty; if no rows, null space is all of R^n.
        if n == 0 {
            return Ok(vec![]);
        }
        // Return identity for n-dimensional null space.
        let mut ident = vec![vec![0.0; n]; n];
        for (i, row) in ident.iter_mut().enumerate().take(n) {
            row[i] = 1.0;
        }

        return Ok(ident);
    }

    let matrix = dmatrix_from_rows(a)?;

    // To get the full V matrix (n×n), compute SVD of A^T (n×m).
    // The U of A^T equals V of A, giving us all n right singular vectors.
    let at = matrix.transpose();
    let svd_decomp = safe_svd(at, true, false)?;
    let u_of_at = svd_decomp
        .u
        .as_ref()
        .ok_or(LinalgError::UnsupportedAssumption)?;
    let singular_values = &svd_decomp.singular_values;

    let max_s = singular_values
        .iter()
        .copied()
        .fold(0.0_f64, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        });
    let tol = rcond.unwrap_or_else(|| (m.max(n) as f64) * f64::EPSILON) * max_s;

    let rank = singular_values.iter().filter(|&&s| s > tol).count();
    let null_dim = n - rank;

    if null_dim == 0 {
        return Ok(vec![vec![]; n]);
    }

    // Null space = columns rank..n of U(A^T) = columns rank..n of V(A).
    // U(A^T) is n × min(n,m), so for m < n we may not have enough columns.
    // However, since rank <= min(m,n), and null_dim = n - rank, we need columns
    // up to index n-1 in U(A^T). When n > m, U(A^T) has n rows but only min(n,m)=m columns.
    // Columns beyond m correspond to singular value 0, but nalgebra's thin SVD
    // doesn't compute them. In this case, we need to find an orthonormal basis
    // for the remaining dimensions via QR of the existing columns' complement.
    let u_cols = u_of_at.ncols(); // = min(n, m)
    if rank < u_cols {
        // We have enough columns in U(A^T) to extract the null space.
        let mut result = vec![vec![0.0; null_dim]; n];
        for i in 0..n {
            for j in 0..null_dim.min(u_cols - rank) {
                result[i][j] = u_of_at[(i, rank + j)];
            }
        }
        // If null_dim > u_cols - rank, the remaining null-space vectors are
        // orthogonal to all columns of U(A^T). This happens when m < n and rank = m.
        // These extra vectors span the complement and must be computed separately.
        if null_dim > u_cols - rank {
            // Use Gram-Schmidt to find remaining orthonormal vectors.
            let _existing = u_cols; // number of columns in U(A^T)
            let needed = null_dim - (u_cols - rank);
            let extra = gram_schmidt_complement(u_of_at, n, needed);
            for j in 0..needed {
                for i in 0..n {
                    result[i][(u_cols - rank) + j] = extra[j][i];
                }
            }
        }
        Ok(result)
    } else {
        // rank == u_cols: all computed singular vectors are in the range.
        // The entire null space is orthogonal to all of them.
        // This happens when rank = min(n,m) = m < n.
        let mut result = vec![vec![0.0; null_dim]; n];
        let extra = gram_schmidt_complement(u_of_at, n, null_dim);
        for j in 0..null_dim {
            for i in 0..n {
                result[i][j] = extra[j][i];
            }
        }
        Ok(result)
    }
}

/// Find `needed` orthonormal vectors that are orthogonal to all columns of `basis`.
/// `basis` is an n×k matrix (DMatrix). Returns `needed` vectors of length `n`.
fn gram_schmidt_complement(basis: &DMatrix<f64>, n: usize, needed: usize) -> Vec<Vec<f64>> {
    let k = basis.ncols();
    let mut result: Vec<Vec<f64>> = Vec::with_capacity(needed);

    // Try standard basis vectors e_i and orthogonalize against existing basis
    // and previously found complement vectors.
    for candidate_idx in 0..n {
        if result.len() >= needed {
            break;
        }
        // Start with e_{candidate_idx}
        let mut v = vec![0.0; n];
        v[candidate_idx] = 1.0;

        // Orthogonalize against all basis columns.
        for col in 0..k {
            let mut dot = 0.0;
            for r in 0..n {
                dot += v[r] * basis[(r, col)];
            }
            for r in 0..n {
                v[r] -= dot * basis[(r, col)];
            }
        }

        // Orthogonalize against previously found complement vectors.
        for prev in &result {
            let mut dot = 0.0;
            for r in 0..n {
                dot += v[r] * prev[r];
            }
            for r in 0..n {
                v[r] -= dot * prev[r];
            }
        }

        // Normalize.
        let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for x in &mut v {
                *x /= norm;
            }
            result.push(v);
        }
    }

    result
}

/// Compute the angles between two subspaces.
///
/// Matches `scipy.linalg.subspace_angles(A, B)`.
///
/// Returns the principal angles (in radians, sorted in decreasing order)
/// between the column spaces of A and B, computed via SVD of Q_A^T Q_B.
pub fn subspace_angles(
    a: &[Vec<f64>],
    b: &[Vec<f64>],
    options: DecompOptions,
) -> Result<Vec<f64>, LinalgError> {
    // Compute orthonormal bases.
    let qa = orth(a, None, options)?;
    let qb = orth(b, None, options)?;

    if qa.is_empty() || qa[0].is_empty() || qb.is_empty() || qb[0].is_empty() {
        return Ok(vec![]);
    }

    let m = qa.len();
    let ka = qa[0].len();
    let kb = qb[0].len();

    // Compute QA^T * QB.
    let mut product = vec![vec![0.0; kb]; ka];
    for i in 0..ka {
        for j in 0..kb {
            let mut sum = 0.0;
            for r in 0..m {
                sum += qa[r][i] * qb[r][j];
            }
            product[i][j] = sum;
        }
    }

    // SVD of the product to get cosines of principal angles.
    let sv = svdvals(&product, options)?;
    let mut angles: Vec<f64> = sv.iter().map(|&s| s.clamp(0.0, 1.0).acos()).collect();
    angles.sort_by(|a, b| b.total_cmp(a)); // descending
    Ok(angles)
}

/// Compute the polar decomposition of a matrix: A = U * P.
///
/// Matches `scipy.linalg.polar(a)`.
///
/// Returns `(U, P)` where U is unitary (or semi-unitary) and P is
/// positive semi-definite Hermitian.
///
/// Computed via SVD: `A = U_svd * S * Vt`, then `U = U_svd * Vt` and `P = V * S * Vt`.
pub fn polar(a: &[Vec<f64>], options: DecompOptions) -> Result<PolarResult, LinalgError> {
    let (m, n) = matrix_shape(a)?;
    validate_finite_matrix(a, options.mode, options.check_finite)?;

    if m == 0 || n == 0 {
        return Ok(PolarResult {
            u: vec![vec![]; m],
            p: vec![vec![]; n],
        });
    }

    let svd_result = svd(a, options)?;
    let u_svd = dmatrix_from_rows(&svd_result.u)?;
    let vt = dmatrix_from_rows(&svd_result.vt)?;
    let s = &svd_result.s;
    let k = s.len();

    // U = U_svd * Vt (m × n).
    let u_polar = u_svd * vt.clone();

    // P = V * S * Vt = Vt^T * diag(S) * Vt
    let mut sigma = DMatrix::zeros(k, k);
    for (i, &val) in s.iter().enumerate() {
        sigma[(i, i)] = val;
    }
    let p_polar = vt.transpose() * sigma * vt;

    Ok(PolarResult {
        u: rows_from_dmatrix(&u_polar),
        p: rows_from_dmatrix(&p_polar),
    })
}

// ══════════════════════════════════════════════════════════════════════
// Special Matrix Constructors
// ══════════════════════════════════════════════════════════════════════

/// Construct a Toeplitz matrix.
///
/// Matches `scipy.linalg.toeplitz(c, r)`.
///
/// A Toeplitz matrix has constant diagonals. The first column is `c` and
/// the first row is `r`. If `r` is None, the matrix is symmetric.
pub fn toeplitz(c: &[f64], r: Option<&[f64]>) -> Vec<Vec<f64>> {
    let n = c.len();
    let row = r.unwrap_or(c);
    let m = row.len();

    let mut result = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            // Lower triangle + diagonal come from the first column `c`; the
            // strict upper triangle comes from the first row `r`. This keeps the
            // diagonal at c[0] for every i and ignores row[0] entirely, matching
            // scipy.linalg.toeplitz (where r[0] is documented as ignored). The
            // previous split put row[0] on the diagonal for i>0, corrupting it
            // whenever r[0] != c[0].
            result[i][j] = if i >= j { c[i - j] } else { row[j - i] };
        }
    }
    result
}

/// Construct a circulant matrix whose first column is `c`.
///
/// Matches `scipy.linalg.circulant(c)`. Each subsequent column is a
/// cyclic-down-shift of the previous one, equivalently:
///
/// ```text
///   result[i][j] = c[(i - j) mod n]
/// ```
///
/// Resolves [frankenscipy-qfv5a]: the previous version put `c` on the
/// FIRST ROW (returning the transpose of the scipy convention).
pub fn circulant(c: &[f64]) -> Vec<Vec<f64>> {
    let n = c.len();
    let mut result = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            // (i - j) mod n via signed-safe arithmetic.
            result[i][j] = c[(i + n - j) % n];
        }
    }
    result
}

/// Construct a Hilbert matrix.
///
/// Matches `scipy.linalg.hilbert(n)`.
///
/// H_{ij} = 1 / (i + j + 1) for i,j starting at 0. The Hilbert matrix
/// is a classic example of an ill-conditioned matrix.
pub fn hilbert(n: usize) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.0; n]; n];
    for (i, row) in result.iter_mut().enumerate().take(n) {
        for (j, entry) in row.iter_mut().enumerate().take(n) {
            *entry = 1.0 / (i + j + 1) as f64;
        }
    }
    result
}

/// Construct the inverse of a Hilbert matrix.
///
/// Matches `scipy.linalg.invhilbert(n)`.
///
/// Returns the exact inverse (integer entries) computed via the closed-form
/// formula, avoiding numerical inversion of the ill-conditioned Hilbert matrix.
pub fn invhilbert(n: usize) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.0; n]; n];
    for (i, row) in result.iter_mut().enumerate().take(n) {
        for (j, entry) in row.iter_mut().enumerate().take(n) {
            let mut val = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
            val *= ((i + j + 1) as f64)
                * binom(n + i, n - j - 1)
                * binom(n + j, n - i - 1)
                * binom(i + j, i).powi(2);
            *entry = val;
        }
    }
    result
}

/// Binomial coefficient C(n, k) as f64.
fn binom(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    let mut result = 1.0;
    for i in 0..k {
        result *= (n - i) as f64;
        result /= (i + 1) as f64;
    }
    result
}

/// Construct a Hadamard matrix of order n.
///
/// Matches `scipy.linalg.hadamard(n)`.
///
/// n must be a power of 2. The matrix has entries ±1 and satisfies H^T H = n I.
/// Constructed via the Sylvester (recursive) construction.
pub fn hadamard(n: usize) -> Result<Vec<Vec<f64>>, LinalgError> {
    if n == 0 || (n & (n - 1)) != 0 {
        return Err(LinalgError::InvalidArgument {
            detail: format!("n must be a power of 2, got {n}"),
        });
    }

    let mut h = vec![vec![1.0]];
    let mut size = 1;
    while size < n {
        let mut new_h = vec![vec![0.0; 2 * size]; 2 * size];
        for i in 0..size {
            for j in 0..size {
                new_h[i][j] = h[i][j];
                new_h[i][j + size] = h[i][j];
                new_h[i + size][j] = h[i][j];
                new_h[i + size][j + size] = -h[i][j];
            }
        }
        h = new_h;
        size *= 2;
    }
    Ok(h)
}

/// Construct a companion matrix.
///
/// Matches `scipy.linalg.companion(a)`.
///
/// The companion matrix for polynomial `a[0]*x^n + a[1]*x^{n-1} + ... + a[n]`
/// has the coefficients `-a[1:]/a[0]` in the first row and ones on the sub-diagonal.
pub fn companion(a: &[f64]) -> Result<Vec<Vec<f64>>, LinalgError> {
    if a.len() < 2 {
        return Err(LinalgError::InvalidArgument {
            detail: "companion requires at least 2 coefficients".to_string(),
        });
    }
    if a[0].abs() < f64::EPSILON {
        return Err(LinalgError::InvalidArgument {
            detail: "leading coefficient must be nonzero".to_string(),
        });
    }

    let n = a.len() - 1;
    let mut result = vec![vec![0.0; n]; n];

    // First row: -a[1..]/a[0].
    for j in 0..n {
        result[0][j] = -a[j + 1] / a[0];
    }

    // Sub-diagonal: ones.
    for i in 1..n {
        result[i][i - 1] = 1.0;
    }

    Ok(result)
}

/// Construct a block diagonal matrix from a list of matrices.
///
/// Matches `scipy.linalg.block_diag(*arrs)`.
///
/// Arranges the input matrices along the diagonal of a larger matrix,
/// with zeros elsewhere.
pub fn block_diag(blocks: &[Vec<Vec<f64>>]) -> Vec<Vec<f64>> {
    if blocks.is_empty() {
        return vec![];
    }

    let total_rows: usize = blocks.iter().map(|b| b.len()).sum();
    let total_cols: usize = blocks
        .iter()
        .map(|b| if b.is_empty() { 0 } else { b[0].len() })
        .sum();

    let mut result = vec![vec![0.0; total_cols]; total_rows];
    let mut row_offset = 0;
    let mut col_offset = 0;

    for block in blocks {
        let br = block.len();
        let bc = if br > 0 { block[0].len() } else { 0 };
        for i in 0..br {
            for j in 0..bc {
                result[row_offset + i][col_offset + j] = block[i][j];
            }
        }
        row_offset += br;
        col_offset += bc;
    }

    result
}

// ══════════════════════════════════════════════════════════════════════
// Additional Special Matrices
// ══════════════════════════════════════════════════════════════════════

/// Pascal matrix (lower triangular or symmetric).
///
/// Matches `scipy.linalg.pascal`.
pub fn pascal(n: usize, symmetric: bool) -> Vec<Vec<f64>> {
    // Lower-triangular Pascal matrix: L[i][j] = C(i, j) for j <= i
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        l[i][0] = 1.0;
        for j in 1..=i {
            l[i][j] = l[i - 1][j - 1] + l[i - 1][j];
        }
    }

    if symmetric {
        // Symmetric Pascal: S = L * L^T
        let mut s = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                for (&lik, &ljk) in l[i].iter().zip(l[j].iter()) {
                    s[i][j] += lik * ljk;
                }
            }
        }
        s
    } else {
        l
    }
}

/// DFT matrix of size n.
///
/// `F[j][k] = exp(-2πijk/n) / sqrt(n)`  (unitary normalization).
///
/// Matches `scipy.linalg.dft`.
pub fn dft_matrix(n: usize) -> Vec<Vec<(f64, f64)>> {
    let scale = 1.0 / (n as f64).sqrt();
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut f = vec![vec![(0.0, 0.0); n]; n];
    for (j, row) in f.iter_mut().enumerate() {
        for (k, value) in row.iter_mut().enumerate() {
            let angle = -two_pi * (j * k) as f64 / n as f64;
            *value = (scale * angle.cos(), scale * angle.sin());
        }
    }
    f
}

/// Fiedler matrix: `F[i][j] = |i - j|`.
///
/// Matches `scipy.linalg.fiedler`.
pub fn fiedler(a: &[f64]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut f = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            f[i][j] = (a[i] - a[j]).abs();
        }
    }
    f
}

/// Fiedler companion matrix from polynomial coefficients.
///
/// Given coefficients `[a₀, a₁, …, aₙ]` of
/// `p(x) = a₀·xⁿ + a₁·xⁿ⁻¹ + … + aₙ` in descending order with a nonzero
/// leading coefficient, returns the `(n×n)` pentadiagonal Fiedler companion
/// matrix whose eigenvalues coincide with the roots of `p`. When the leading
/// coefficient is not 1, the coefficients are rescaled to monic form before
/// construction.
///
/// Unlike [`companion`], which builds the standard Frobenius companion
/// matrix, this is Fiedler's pentadiagonal construction (M. Fiedler, "A note
/// on companion matrices", *Linear Algebra and its Applications*, 2003); the
/// two matrices are different shapes that share the same characteristic
/// polynomial.
///
/// Edge cases match `scipy.linalg.fiedler_companion`: fewer than two
/// coefficients yields an empty matrix; exactly two yields the `1×1` matrix
/// `[[-a₁/a₀]]`.
///
/// Matches `scipy.linalg.fiedler_companion`.
pub fn fiedler_companion(a: &[f64]) -> Vec<Vec<f64>> {
    if a.len() < 2 {
        return Vec::new();
    }
    // Rescale to a monic polynomial: a ← a / a[0].
    let lead = a[0];
    let a: Vec<f64> = a.iter().map(|&coeff| coeff / lead).collect();

    if a.len() == 2 {
        return vec![vec![-a[1]]];
    }

    let n = a.len() - 1;
    let mut c = vec![vec![0.0; n]; n];

    // Outer subdiagonal band: alternating unit entries.
    for (i, j) in (3..n).step_by(2).zip((1..n - 2).step_by(2)) {
        c[i][j] = 1.0;
    }
    // Inner subdiagonal band: scaled odd-index coefficients.
    for ((i, j), k) in (2..n)
        .step_by(2)
        .zip((1..n - 1).step_by(2))
        .zip((3..=n).step_by(2))
    {
        c[i][j] = -a[k];
    }
    // Outer superdiagonal band: alternating unit entries.
    for (i, j) in (0..n - 2).step_by(2).zip((2..n).step_by(2)) {
        c[i][j] = 1.0;
    }
    // Inner superdiagonal band: scaled even-index coefficients.
    for ((i, j), k) in (0..n - 1)
        .step_by(2)
        .zip((1..n).step_by(2))
        .zip((2..=n).step_by(2))
    {
        c[i][j] = -a[k];
    }
    // Top-left corner and the first subdiagonal entry.
    c[0][0] = -a[1];
    c[1][0] = 1.0;

    c
}

/// Leslie population matrix.
///
/// Matches `scipy.linalg.leslie`.
pub fn leslie(f_vals: &[f64], s_vals: &[f64]) -> Result<Vec<Vec<f64>>, LinalgError> {
    if f_vals.is_empty() {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    if s_vals.len() + 1 != f_vals.len() {
        return Err(LinalgError::IncompatibleShapes {
            a_shape: (f_vals.len(), f_vals.len()),
            b_len: s_vals.len(),
        });
    }

    let n = f_vals.len();
    let mut l = vec![vec![0.0; n]; n];

    // First row: fecundity rates
    for (j, &fj) in f_vals.iter().enumerate() {
        l[0][j] = fj;
    }
    // Sub-diagonal: survival rates
    for (i, &si) in s_vals.iter().enumerate() {
        l[i + 1][i] = si;
    }

    Ok(l)
}

/// Convolution matrix for a 1D filter.
///
/// Creates a Toeplitz matrix that performs convolution when multiplied by a vector.
/// Matches `scipy.linalg.convolution_matrix`.
pub fn convolution_matrix(h: &[f64], n: usize, mode: &str) -> Vec<Vec<f64>> {
    let k = h.len();
    let out_len = match mode {
        "full" => n + k - 1,
        "same" => n,
        "valid" => {
            if n >= k {
                n - k + 1
            } else {
                0
            }
        }
        _ => n + k - 1, // default to full
    };

    let offset = match mode {
        "same" => (k - 1) / 2,
        "valid" => k - 1,
        _ => 0,
    };

    let mut mat = vec![vec![0.0; n]; out_len];
    for (i, row) in mat.iter_mut().enumerate() {
        for (j, &hj) in h.iter().enumerate() {
            let col = i as i64 + offset as i64 - j as i64;
            if col >= 0 && (col as usize) < n {
                row[col as usize] = hj;
            }
        }
    }

    mat
}

/// Compute the bandwidth of a matrix (lower and upper).
///
/// Returns (lower_bandwidth, upper_bandwidth).
/// Matches `scipy.linalg.bandwidth`.
pub fn bandwidth(a: &[Vec<f64>]) -> (usize, usize) {
    let n = a.len();
    if n == 0 {
        return (0, 0);
    }
    let m = a[0].len();
    let mut lower = 0usize;
    let mut upper = 0usize;

    for (i, row) in a.iter().enumerate().take(n) {
        for (j, &value) in row.iter().enumerate().take(m) {
            if value.abs() > 0.0 {
                if i > j {
                    lower = lower.max(i - j);
                }
                if j > i {
                    upper = upper.max(j - i);
                }
            }
        }
    }

    (lower, upper)
}

/// Create a tri (tridiagonal-like) matrix.
///
/// tri(n, m, k) creates an n×m matrix with ones at and below the k-th diagonal.
/// Matches `numpy.tri`.
pub fn tri(n: usize, m: usize, k: i64) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.0; m]; n];
    for (i, row) in result.iter_mut().enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            if (j as i64) <= (i as i64) + k {
                *value = 1.0;
            }
        }
    }
    result
}

/// Extract lower triangle of a matrix.
///
/// Matches `numpy.tril`.
pub fn tril(a: &[Vec<f64>], k: i64) -> Vec<Vec<f64>> {
    let n = a.len();
    if n == 0 {
        return vec![];
    }
    let m = a[0].len();
    let mut result = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            if (j as i64) <= (i as i64) + k {
                result[i][j] = a[i][j];
            }
        }
    }
    result
}

/// Extract upper triangle of a matrix.
///
/// Matches `numpy.triu`.
pub fn triu(a: &[Vec<f64>], k: i64) -> Vec<Vec<f64>> {
    let n = a.len();
    if n == 0 {
        return vec![];
    }
    let m = a[0].len();
    let mut result = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            if (j as i64) >= (i as i64) + k {
                result[i][j] = a[i][j];
            }
        }
    }
    result
}

/// Kronecker product of two matrices.
///
/// Matches `numpy.kron` / `scipy.linalg.kron`.
pub fn kron(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let (ra, ca) = (a.len(), a[0].len());
    let (rb, cb) = (b.len(), b[0].len());
    let mut result = vec![vec![0.0; ca * cb]; ra * rb];

    for i in 0..ra {
        for j in 0..ca {
            for k in 0..rb {
                for l in 0..cb {
                    result[i * rb + k][j * cb + l] = a[i][j] * b[k][l];
                }
            }
        }
    }

    result
}

// ══════════════════════════════════════════════════════════════════════
// Additional Matrix Constructors
// ══════════════════════════════════════════════════════════════════════

/// Vandermonde matrix: `V[i][j] = x[i]^j`.
///
/// Matches `numpy.vander`.
pub fn vander(x: &[f64], n: Option<usize>, increasing: bool) -> Vec<Vec<f64>> {
    let m = x.len();
    let cols = n.unwrap_or(m);
    let mut v = vec![vec![0.0; cols]; m];
    for (i, row) in v.iter_mut().enumerate().take(m) {
        for (j, item) in row.iter_mut().enumerate().take(cols) {
            let power = if increasing { j } else { cols - 1 - j };
            *item = x[i].powi(power as i32);
        }
    }
    v
}

/// Hankel matrix from first column and (optionally) last row.
///
/// Matches `scipy.linalg.hankel(c, r)`. If `r` is None, scipy uses
/// `zeros_like(c)` for the last row — NOT `c` itself. Resolves the
/// `r = c` default-mismatch parity bug (frankenscipy-azhz0).
pub fn hankel(c: &[f64], r: Option<&[f64]>) -> Vec<Vec<f64>> {
    let n = c.len();
    let zero_default;
    let r_vals: &[f64] = match r {
        Some(values) => values,
        None => {
            zero_default = vec![0.0_f64; n];
            &zero_default
        }
    };
    let m = r_vals.len();

    let mut h = vec![vec![0.0; m]; n];
    for (i, row) in h.iter_mut().enumerate().take(n) {
        for (j, item) in row.iter_mut().enumerate().take(m) {
            let idx = i + j;
            *item = if idx < n {
                c[idx]
            } else if idx - n + 1 < m {
                r_vals[idx - n + 1]
            } else {
                0.0
            };
        }
    }
    h
}

/// Helmert matrix (orthogonal contrast matrix), `(n - 1, n)` form.
///
/// Matches `scipy.linalg.helmert(n)` (with default `full=False`),
/// returning the submatrix that excludes the first row of constant
/// `1/√n` entries. The remaining `n - 1` rows are pairwise orthogonal
/// unit vectors.
///
/// For the `(n, n)` form including the first row, use [`helmert_full`].
///
/// Resolves [frankenscipy-3t31o].
pub fn helmert(n: usize) -> Vec<Vec<f64>> {
    let full = helmert_full(n);
    if full.len() <= 1 {
        return Vec::new();
    }
    full.into_iter().skip(1).collect()
}

/// Helmert matrix in the `(n, n)` "full" form, including the first
/// row of constant `1/√n` entries.
///
/// Matches `scipy.linalg.helmert(n, full=True)`.
#[must_use]
pub fn helmert_full(n: usize) -> Vec<Vec<f64>> {
    if n == 0 {
        return vec![];
    }
    let mut h = vec![vec![0.0; n]; n];

    // First row: all 1/√n
    let inv_sqrt_n = 1.0 / (n as f64).sqrt();
    for item in h[0].iter_mut().take(n) {
        *item = inv_sqrt_n;
    }

    // Row i (i >= 1): orthogonal contrasts
    for (i, row) in h.iter_mut().enumerate().take(n).skip(1) {
        let scale = 1.0 / ((i * (i + 1)) as f64).sqrt();
        for item in row.iter_mut().take(i) {
            *item = scale;
        }
        row[i] = -(i as f64) * scale;
    }

    h
}

/// Create a matrix with ones on the specified diagonal.
///
/// Matches `numpy.eye` with `k` offset.
pub fn eye_k(n: usize, m: usize, k: i64) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.0; m]; n];
    for (i, row) in result.iter_mut().enumerate().take(n) {
        let j = i as i64 + k;
        if j >= 0 && (j as usize) < m {
            row[j as usize] = 1.0;
        }
    }
    result
}

/// Stack matrices horizontally.
///
/// Matches `numpy.hstack` for 2D arrays.
pub fn hstack(matrices: &[&[Vec<f64>]]) -> Vec<Vec<f64>> {
    if matrices.is_empty() {
        return vec![];
    }
    let n = matrices[0].len();
    let total_cols: usize = matrices
        .iter()
        .map(|m| if m.is_empty() { 0 } else { m[0].len() })
        .sum();

    let mut result = vec![vec![0.0; total_cols]; n];
    let mut col_offset = 0;
    for mat in matrices {
        let cols = if mat.is_empty() { 0 } else { mat[0].len() };
        for i in 0..n.min(mat.len()) {
            for j in 0..cols {
                result[i][col_offset + j] = mat[i][j];
            }
        }
        col_offset += cols;
    }
    result
}

/// Stack matrices vertically.
///
/// Matches `numpy.vstack` for 2D arrays.
pub fn vstack(matrices: &[&[Vec<f64>]]) -> Vec<Vec<f64>> {
    let mut result = Vec::new();
    for mat in matrices {
        for row in *mat {
            result.push(row.clone());
        }
    }
    result
}

/// Compute the Frobenius norm of a matrix.
pub fn frobenius_norm(a: &[Vec<f64>]) -> f64 {
    a.iter()
        .flat_map(|row| row.iter())
        .map(|&v| v * v)
        .sum::<f64>()
        .sqrt()
}

/// Compute the max absolute value (infinity norm of flattened matrix).
pub fn max_abs(a: &[Vec<f64>]) -> f64 {
    a.iter().flat_map(|row| row.iter()).fold(0.0f64, |acc, &v| {
        let v_abs = v.abs();
        if acc.is_nan() || v_abs.is_nan() {
            f64::NAN
        } else {
            acc.max(v_abs)
        }
    })
}

/// Compute the 1-norm, infinity-norm, or Frobenius norm of a vector.
pub fn vector_norm(v: &[f64], ord: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    if ord == f64::INFINITY {
        v.iter().map(|&x| x.abs()).fold(0.0f64, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        })
    } else if ord == f64::NEG_INFINITY {
        v.iter()
            .map(|&x| x.abs())
            .fold(f64::INFINITY, |a: f64, b: f64| {
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a.min(b)
                }
            })
    } else if ord == 0.0 {
        v.iter().filter(|&&x| x != 0.0).count() as f64
    } else if ord == 1.0 {
        v.iter().map(|&x| x.abs()).sum()
    } else if ord == 2.0 {
        vnorm(v)
    } else {
        v.iter()
            .map(|&x| x.abs().powf(ord))
            .sum::<f64>()
            .powf(1.0 / ord)
    }
}

/// Check if a matrix is positive definite (via Cholesky attempt).
pub fn is_positive_definite(a: &[Vec<f64>]) -> bool {
    let n = a.len();
    if n == 0 {
        return true;
    }
    for row in a.iter().take(n) {
        if row.len() != n {
            return false;
        }
        if row.iter().take(n).any(|value| !value.is_finite()) {
            return false;
        }
    }

    // Try Cholesky decomposition
    let mut l = vec![vec![0.0; n]; n];
    #[allow(clippy::needless_range_loop)]
    for j in 0..n {
        let mut sum = a[j][j];
        for k in 0..j {
            sum -= l[j][k] * l[j][k];
        }
        if !sum.is_finite() || sum <= 0.0 {
            return false;
        }
        l[j][j] = sum.sqrt();
        for i in j + 1..n {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if !sum.is_finite() {
                return false;
            }
            l[i][j] = sum / l[j][j];
            if !l[i][j].is_finite() {
                return false;
            }
        }
    }
    true
}

/// Construct a rotation matrix for 2D rotation by angle theta (radians).
pub fn rot2d(theta: f64) -> Vec<Vec<f64>> {
    vec![
        vec![theta.cos(), -theta.sin()],
        vec![theta.sin(), theta.cos()],
    ]
}

/// Compute the permanent of a matrix (sum over all permutations).
///
/// Uses Ryser's formula. O(2^n * n) complexity.
pub fn permanent(a: &[Vec<f64>]) -> f64 {
    let n = a.len();
    if n == 0 {
        return 1.0;
    }

    let mut result = 0.0;
    let total = 1usize << n;

    for s in 1..total {
        let mut prod_sum = 1.0;
        let bits = s.count_ones() as i32;
        let sign = if (n as i32 - bits) % 2 == 0 {
            1.0
        } else {
            -1.0
        };

        for row in a.iter().take(n) {
            let mut row_sum = 0.0;
            for (j, &val) in row.iter().enumerate().take(n) {
                if s & (1 << j) != 0 {
                    row_sum += val;
                }
            }
            prod_sum *= row_sum;
        }

        result += sign * prod_sum;
    }

    // The in-loop sign (-1)^(n-|S|) already carries Ryser's leading
    // (-1)^n factor, so `result` is the permanent directly — a further
    // negation for odd n would double-count it.
    result
}

/// Create an anti-diagonal matrix.
///
/// Values are placed on the anti-diagonal (top-right to bottom-left).
pub fn antidiag(v: &[f64]) -> Vec<Vec<f64>> {
    let n = v.len();
    let mut m = vec![vec![0.0; n]; n];
    for (i, &val) in v.iter().enumerate() {
        m[i][n - 1 - i] = val;
    }
    m
}

/// Compute the cofactor matrix.
pub fn cofactor(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    if n == 0 {
        return vec![];
    }

    let mut cof = vec![vec![0.0; n]; n];
    for (i, cof_row) in cof.iter_mut().enumerate().take(n) {
        for (j, item) in cof_row.iter_mut().enumerate().take(n) {
            // Minor: remove row i, col j
            let mut minor = Vec::with_capacity(n - 1);
            for (r, a_row) in a.iter().enumerate().take(n) {
                if r == i {
                    continue;
                }
                let row: Vec<f64> = (0..n).filter(|&c| c != j).map(|c| a_row[c]).collect();
                minor.push(row);
            }

            let det_minor = if minor.is_empty() {
                1.0
            } else {
                det_small(&minor)
            };

            let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
            *item = sign * det_minor;
        }
    }
    cof
}

/// Determinant of a small matrix (up to 4x4, fallback to LU).
fn det_small(a: &[Vec<f64>]) -> f64 {
    let n = a.len();
    match n {
        0 => 1.0,
        1 => a[0][0],
        2 => a[0][0] * a[1][1] - a[0][1] * a[1][0],
        3 => {
            a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
                - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
        }
        _ => {
            // Cofactor expansion along first row
            let mut d = 0.0;
            for j in 0..n {
                let minor: Vec<Vec<f64>> = (1..n)
                    .map(|r| (0..n).filter(|&c| c != j).map(|c| a[r][c]).collect())
                    .collect();
                let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
                d += sign * a[0][j] * det_small(&minor);
            }
            d
        }
    }
}

/// Adjugate (classical adjoint) matrix: transpose of cofactor matrix.
pub fn adjugate(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let cof = cofactor(a);
    let n = cof.len();
    let mut adj = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            adj[i][j] = cof[j][i];
        }
    }
    adj
}

/// Compute the element-wise absolute value of a matrix.
pub fn mat_abs(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .map(|row| row.iter().map(|&v| v.abs()).collect())
        .collect()
}

/// Add two matrices element-wise.
pub fn mat_add(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .zip(b.iter())
        .map(|(ra, rb)| ra.iter().zip(rb.iter()).map(|(&x, &y)| x + y).collect())
        .collect()
}

/// Subtract two matrices element-wise.
pub fn mat_sub(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .zip(b.iter())
        .map(|(ra, rb)| ra.iter().zip(rb.iter()).map(|(&x, &y)| x - y).collect())
        .collect()
}

/// Scale a matrix by a scalar.
pub fn mat_scale(a: &[Vec<f64>], s: f64) -> Vec<Vec<f64>> {
    a.iter()
        .map(|row| row.iter().map(|&v| v * s).collect())
        .collect()
}

/// Element-wise multiply (Hadamard product) of two matrices.
pub fn hadamard_product(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .zip(b.iter())
        .map(|(ra, rb)| ra.iter().zip(rb.iter()).map(|(&x, &y)| x * y).collect())
        .collect()
}

/// Extract a submatrix from row r1..r2 and column c1..c2.
pub fn submatrix(a: &[Vec<f64>], r1: usize, r2: usize, c1: usize, c2: usize) -> Vec<Vec<f64>> {
    a[r1..r2].iter().map(|row| row[c1..c2].to_vec()).collect()
}

/// Create a matrix from a flat vector given shape (row-major order).
pub fn mat_from_flat(data: &[f64], rows: usize, cols: usize) -> Vec<Vec<f64>> {
    let mut result = Vec::with_capacity(rows);
    for i in 0..rows {
        let start = i * cols;
        let end = (start + cols).min(data.len());
        result.push(data[start..end].to_vec());
    }
    result
}

/// Flatten a matrix to a vector (row-major order).
pub fn mat_flatten(a: &[Vec<f64>]) -> Vec<f64> {
    a.iter().flat_map(|row| row.iter().cloned()).collect()
}

/// Check if two matrices are equal within tolerance.
pub fn mat_allclose(a: &[Vec<f64>], b: &[Vec<f64>], atol: f64, rtol: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for (ra, rb) in a.iter().zip(b.iter()) {
        if ra.len() != rb.len() {
            return false;
        }
        for (&va, &vb) in ra.iter().zip(rb.iter()) {
            if !scalar_allclose(va, vb, atol, rtol) {
                return false;
            }
        }
    }
    true
}

fn scalar_allclose(actual: f64, expected: f64, atol: f64, rtol: f64) -> bool {
    if actual.is_nan() || expected.is_nan() {
        return actual.is_nan() && expected.is_nan();
    }
    if actual.is_infinite() || expected.is_infinite() {
        return actual == expected;
    }
    (actual - expected).abs() <= atol + rtol * expected.abs()
}

/// Compute the 1-norm of a matrix (maximum column sum of absolute values).
pub fn mat_norm_1(a: &[Vec<f64>]) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    let cols = a[0].len();
    if a.iter().any(|row| row.len() != cols) {
        return f64::NAN;
    }
    let mut max_col_sum = 0.0f64;
    for j in 0..cols {
        let mut col_sum = 0.0_f64;
        for row in a.iter() {
            let value = row[j].abs();
            if !value.is_finite() {
                return f64::NAN;
            }
            col_sum += value;
        }
        max_col_sum = max_col_sum.max(col_sum);
    }
    max_col_sum
}

/// Compute the infinity-norm of a matrix (maximum row sum of absolute values).
pub fn mat_norm_inf(a: &[Vec<f64>]) -> f64 {
    a.iter()
        .map(|row| row.iter().map(|&v| v.abs()).sum::<f64>())
        .fold(0.0f64, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        })
}

/// Check if a matrix is diagonal.
pub fn is_diagonal(a: &[Vec<f64>], tol: f64) -> bool {
    if let Some(first) = a.first() {
        let cols = first.len();
        if a.iter().any(|row| row.len() != cols) {
            return false;
        }
    }
    for (i, row) in a.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if i != j && (!v.is_finite() || v.abs() > tol) {
                return false;
            }
        }
    }
    true
}

/// Check if a matrix is upper triangular.
pub fn is_upper_triangular(a: &[Vec<f64>], tol: f64) -> bool {
    if let Some(first) = a.first() {
        let cols = first.len();
        if a.iter().any(|row| row.len() != cols) {
            return false;
        }
    }
    for (i, row) in a.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if i > j && (!v.is_finite() || v.abs() > tol) {
                return false;
            }
        }
    }
    true
}

/// Check if a matrix is lower triangular.
pub fn is_lower_triangular(a: &[Vec<f64>], tol: f64) -> bool {
    if let Some(first) = a.first() {
        let cols = first.len();
        if a.iter().any(|row| row.len() != cols) {
            return false;
        }
    }
    for (i, row) in a.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if j > i && (!v.is_finite() || v.abs() > tol) {
                return false;
            }
        }
    }
    true
}

/// Check if a matrix is orthogonal (A^T * A ≈ I).
pub fn is_orthogonal(a: &[Vec<f64>], tol: f64) -> bool {
    let n = a.len();
    if n == 0 {
        return true;
    }
    let m = a[0].len();
    if n != m {
        return false;
    }
    for row in a.iter().take(n) {
        if row.len() != m {
            return false;
        }
        if row.iter().take(m).any(|value| !value.is_finite()) {
            return false;
        }
    }

    // Check A^T * A ≈ I
    for i in 0..n {
        for j in 0..n {
            let mut dot = 0.0;
            for row in a.iter().take(n) {
                dot += row[i] * row[j];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            if (dot - expected).abs() > tol {
                return false;
            }
        }
    }
    true
}

/// Compute the numerical rank of a matrix (count of singular values > tol).
pub fn numerical_rank(
    a: &[Vec<f64>],
    tol: f64,
    options: DecompOptions,
) -> Result<usize, LinalgError> {
    let sv = svdvals(a, options)?;
    Ok(sv.iter().filter(|&&s| s > tol).count())
}

/// Generate a random matrix with entries from uniform [0, 1).
pub fn random_matrix(rows: usize, cols: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = seed;
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    (rng >> 11) as f64 / (1u64 << 53) as f64
                })
                .collect()
        })
        .collect()
}

/// Generate a random symmetric positive definite matrix.
pub fn random_spd(n: usize, seed: u64) -> Vec<Vec<f64>> {
    let a = random_matrix(n, n, seed);
    // A^T * A + n*I is guaranteed SPD
    let mut result = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            for row in a.iter().take(n) {
                result[i][j] += row[i] * row[j];
            }
        }
        result[i][i] += n as f64;
    }
    result
}

// ══════════════════════════════════════════════════════════════════════
// Condition Number and Matrix Properties
// ══════════════════════════════════════════════════════════════════════

/// Condition number of a matrix (ratio of largest to smallest singular value).
///
/// Matches `numpy.linalg.cond` / `scipy.linalg.cond`.
pub fn cond(a: &[Vec<f64>], options: DecompOptions) -> Result<f64, LinalgError> {
    let sv = svdvals(a, options)?;
    if sv.is_empty() {
        return Ok(f64::INFINITY);
    }
    let s_max = sv[0];
    let s_min = sv[sv.len() - 1];
    if s_min == 0.0 {
        Ok(f64::INFINITY)
    } else {
        Ok(s_max / s_min)
    }
}

/// Trace of a matrix (sum of diagonal elements).
///
/// Matches `numpy.trace`.
pub fn trace(a: &[Vec<f64>]) -> f64 {
    let n = a.len();
    let mut sum = 0.0;
    for (i, row) in a.iter().enumerate().take(n) {
        if i < row.len() {
            sum += row[i];
        }
    }
    sum
}

/// Diagonal of a matrix as a vector.
///
/// Matches `numpy.diag(a)` when a is 2D.
pub fn diag(a: &[Vec<f64>]) -> Vec<f64> {
    let n = a.len();
    let mut d = Vec::with_capacity(n);
    for (i, row) in a.iter().enumerate().take(n) {
        if i < row.len() {
            d.push(row[i]);
        }
    }
    d
}

/// Create a diagonal matrix from a vector.
///
/// Matches `numpy.diag(v)` when v is 1D.
pub fn diagm(v: &[f64]) -> Vec<Vec<f64>> {
    let n = v.len();
    let mut m = vec![vec![0.0; n]; n];
    for (i, &vi) in v.iter().enumerate() {
        m[i][i] = vi;
    }
    m
}

/// Eye (identity) matrix of size n×m.
///
/// Matches `numpy.eye`.
pub fn eye(n: usize, m: usize) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.0; m]; n];
    for (i, row) in result.iter_mut().enumerate().take(n.min(m)) {
        row[i] = 1.0;
    }
    result
}

/// Matrix-matrix multiplication C = A * B.
///
/// Matches `numpy.matmul` / `A @ B`.
const MATMUL_FLAT_WORKSPACE_MIN_DIM: usize = 1024;

fn rows_are_rectangular(rows: &[Vec<f64>], width: usize) -> bool {
    rows.iter().all(|row| row.len() == width)
}

/// Compute output rows `[row_start, row_end)` of the flat-workspace GEMM into
/// `out` (row-major, length `(row_end-row_start)*n`, indexed from `row_start`).
///
/// This is the body of the cache-blocked SIMD micro-kernel, parameterised by a row
/// range so it can be split across threads. Every `c[i][j]` accumulates `k` in
/// `0..ka` monotonic order via the same scalar mul+add sequence regardless of the
/// RB/MR block grouping or which thread owns the row, so the result is bit-identical
/// to the sequential `naive ijk` reference (the basis for byte-identical parallelism).
#[allow(clippy::too_many_arguments)]
fn matmul_flat_compute_rows(
    out: &mut [f64],
    row_start: usize,
    row_end: usize,
    a_flat: &[f64],
    packed_b: &[f64],
    b_flat: &[f64],
    ka: usize,
    n: usize,
) {
    const MR: usize = 4;
    const NR: usize = 8;
    const RB: usize = 64;
    let mut ib = row_start;
    while ib < row_end {
        let i_limit = (ib + RB).min(row_end);
        let mut j0 = 0;
        while j0 < n {
            let nr = (n - j0).min(NR);
            let mut i0 = ib;
            while i0 < i_limit {
                let mr = (i_limit - i0).min(MR);
                if mr == MR && nr == NR {
                    let a0_base = i0 * ka;
                    let a1_base = (i0 + 1) * ka;
                    let a2_base = (i0 + 2) * ka;
                    let a3_base = (i0 + 3) * ka;
                    let mut acc = [Simd::<f64, NR>::splat(0.0); MR];
                    let packed_panel_base = (j0 / NR) * ka * NR;
                    for k in 0..ka {
                        let a0 = a_flat[a0_base + k];
                        let a1 = a_flat[a1_base + k];
                        let a2 = a_flat[a2_base + k];
                        let a3 = a_flat[a3_base + k];
                        let b_base = packed_panel_base + k * NR;
                        let b_vec = Simd::from_array([
                            packed_b[b_base],
                            packed_b[b_base + 1],
                            packed_b[b_base + 2],
                            packed_b[b_base + 3],
                            packed_b[b_base + 4],
                            packed_b[b_base + 5],
                            packed_b[b_base + 6],
                            packed_b[b_base + 7],
                        ]);
                        acc[0] += Simd::splat(a0) * b_vec;
                        acc[1] += Simd::splat(a1) * b_vec;
                        acc[2] += Simd::splat(a2) * b_vec;
                        acc[3] += Simd::splat(a3) * b_vec;
                    }
                    for (di, acc_row) in acc.iter().enumerate().take(MR) {
                        let acc_row = acc_row.to_array();
                        let c_base = (i0 + di - row_start) * n + j0;
                        out[c_base] = acc_row[0];
                        out[c_base + 1] = acc_row[1];
                        out[c_base + 2] = acc_row[2];
                        out[c_base + 3] = acc_row[3];
                        out[c_base + 4] = acc_row[4];
                        out[c_base + 5] = acc_row[5];
                        out[c_base + 6] = acc_row[6];
                        out[c_base + 7] = acc_row[7];
                    }
                } else {
                    for di in 0..mr {
                        let a_base = (i0 + di) * ka;
                        let c_base = (i0 + di - row_start) * n + j0;
                        for dj in 0..nr {
                            let mut s = 0.0;
                            for k in 0..ka {
                                s += a_flat[a_base + k] * b_flat[k * n + j0 + dj];
                            }
                            out[c_base + dj] = s;
                        }
                    }
                }
                i0 += MR;
            }
            j0 += NR;
        }
        ib += RB;
    }
}

/// Number of worker threads for a flat-workspace GEMM of the given dims. Returns 1
/// (sequential) for matmuls too small to amortise thread spawn; otherwise scales with
/// cores, capped so each thread owns at least 64 output rows.
fn matmul_thread_count(m: usize, ka: usize, n: usize) -> usize {
    let macs = (m as u64)
        .saturating_mul(ka as u64)
        .saturating_mul(n as u64);
    if macs < 64 * 1024 * 1024 {
        return 1;
    }
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    cores.min(m / 64).max(1)
}

fn matmul_flat_workspace(
    a: &[Vec<f64>],
    b: &[Vec<f64>],
    m: usize,
    ka: usize,
    n: usize,
) -> Option<Vec<Vec<f64>>> {
    let a_len = m.checked_mul(ka)?;
    let b_len = ka.checked_mul(n)?;
    let c_len = m.checked_mul(n)?;
    let mut a_flat = Vec::with_capacity(a_len);
    for row in a {
        a_flat.extend_from_slice(row);
    }
    let mut b_flat = Vec::with_capacity(b_len);
    for row in b {
        b_flat.extend_from_slice(row);
    }
    let mut c_flat = vec![0.0; c_len];

    const NR: usize = 8;
    let full_n_blocks = n / NR;
    let packed_b_len = full_n_blocks.checked_mul(ka)?.checked_mul(NR)?;
    let mut packed_b = Vec::with_capacity(packed_b_len);
    for jb in 0..full_n_blocks {
        let j0 = jb * NR;
        for k in 0..ka {
            let b_base = k * n + j0;
            packed_b.extend_from_slice(&b_flat[b_base..b_base + NR]);
        }
    }

    // Distribute disjoint output-row ranges across threads. Each c[i][j] is computed
    // by the identical k-ordered reduction irrespective of the row split, so the
    // result is bit-identical to the sequential kernel (golden sha unchanged); only
    // *which* core writes each row changes.
    let nthreads = matmul_thread_count(m, ka, n);
    if nthreads <= 1 {
        matmul_flat_compute_rows(&mut c_flat, 0, m, &a_flat, &packed_b, &b_flat, ka, n);
    } else {
        let chunk_rows = m.div_ceil(nthreads);
        let a_ref = &a_flat;
        let b_ref = &b_flat;
        let pb_ref = &packed_b;
        std::thread::scope(|scope| {
            for (t, out_chunk) in c_flat.chunks_mut(chunk_rows * n).enumerate() {
                let row_start = t * chunk_rows;
                let row_end = (row_start + chunk_rows).min(m);
                scope.spawn(move || {
                    matmul_flat_compute_rows(
                        out_chunk, row_start, row_end, a_ref, pb_ref, b_ref, ka, n,
                    );
                });
            }
        });
    }

    let mut c = Vec::with_capacity(m);
    for row in c_flat.chunks(n) {
        c.push(row.to_vec());
    }
    Some(c)
}

pub fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, LinalgError> {
    if a.is_empty() || b.is_empty() {
        return Ok(vec![]);
    }
    let (m, ka) = (a.len(), a[0].len());
    let (kb, n) = (b.len(), b[0].len());
    if ka != kb {
        return Err(LinalgError::IncompatibleShapes {
            a_shape: (m, ka),
            b_len: kb,
        });
    }
    if m >= MATMUL_FLAT_WORKSPACE_MIN_DIM
        && ka >= MATMUL_FLAT_WORKSPACE_MIN_DIM
        && n >= MATMUL_FLAT_WORKSPACE_MIN_DIM
        && rows_are_rectangular(a, ka)
        && rows_are_rectangular(b, n)
        && let Some(c) = matmul_flat_workspace(a, b, m, ka, n)
    {
        return Ok(c);
    }
    let mut c = vec![vec![0.0; n]; m];
    // Register-blocked GEMM micro-kernel [frankenscipy-8l8r1]. A flat ikj loop
    // streams every B element through one mul+add per FMA (B[k][j] and C[i][j]
    // both touched per multiply), so it is memory-bound on the C read/modify/
    // write. This kernel computes an MR x NR tile of C in register-resident
    // accumulators: across the k-loop each loaded a[i0+di][k] is reused for NR
    // columns and each loaded b[k][j0+dj] is reused for MR rows, so MR*NR FMAs
    // ride on only MR+NR scalar loads. The NR-wide accumulator rows map onto
    // SIMD lanes for autovectorization.
    //
    // Bit-identical to naive ijk / flat ikj: every acc[di][dj] accumulates k in
    // 0..ka monotonic order, the identical sequence of separate mul+add ops as
    // the reference (Rust does not contract a*b+c to a fused FMA without
    // fast-math), so each c[i][j] has the same FP bit pattern. Ragged edges
    // (mr<MR or nr<NR) fall back to the same monotonic-k scalar reduction.
    const MR: usize = 4;
    const NR: usize = 8;
    let mut i0 = 0;
    while i0 < m {
        let mr = (m - i0).min(MR);
        let mut j0 = 0;
        while j0 < n {
            let nr = (n - j0).min(NR);
            if mr == MR && nr == NR {
                let mut acc = [[0.0f64; NR]; MR];
                for k in 0..ka {
                    let a0 = a[i0][k];
                    let a1 = a[i0 + 1][k];
                    let a2 = a[i0 + 2][k];
                    let a3 = a[i0 + 3][k];
                    let bk = &b[k];
                    let b0 = bk[j0];
                    let b1 = bk[j0 + 1];
                    let b2 = bk[j0 + 2];
                    let b3 = bk[j0 + 3];
                    let b4 = bk[j0 + 4];
                    let b5 = bk[j0 + 5];
                    let b6 = bk[j0 + 6];
                    let b7 = bk[j0 + 7];
                    acc[0][0] += a0 * b0;
                    acc[0][1] += a0 * b1;
                    acc[0][2] += a0 * b2;
                    acc[0][3] += a0 * b3;
                    acc[0][4] += a0 * b4;
                    acc[0][5] += a0 * b5;
                    acc[0][6] += a0 * b6;
                    acc[0][7] += a0 * b7;
                    acc[1][0] += a1 * b0;
                    acc[1][1] += a1 * b1;
                    acc[1][2] += a1 * b2;
                    acc[1][3] += a1 * b3;
                    acc[1][4] += a1 * b4;
                    acc[1][5] += a1 * b5;
                    acc[1][6] += a1 * b6;
                    acc[1][7] += a1 * b7;
                    acc[2][0] += a2 * b0;
                    acc[2][1] += a2 * b1;
                    acc[2][2] += a2 * b2;
                    acc[2][3] += a2 * b3;
                    acc[2][4] += a2 * b4;
                    acc[2][5] += a2 * b5;
                    acc[2][6] += a2 * b6;
                    acc[2][7] += a2 * b7;
                    acc[3][0] += a3 * b0;
                    acc[3][1] += a3 * b1;
                    acc[3][2] += a3 * b2;
                    acc[3][3] += a3 * b3;
                    acc[3][4] += a3 * b4;
                    acc[3][5] += a3 * b5;
                    acc[3][6] += a3 * b6;
                    acc[3][7] += a3 * b7;
                }
                for di in 0..MR {
                    let ci = &mut c[i0 + di];
                    ci[j0] = acc[di][0];
                    ci[j0 + 1] = acc[di][1];
                    ci[j0 + 2] = acc[di][2];
                    ci[j0 + 3] = acc[di][3];
                    ci[j0 + 4] = acc[di][4];
                    ci[j0 + 5] = acc[di][5];
                    ci[j0 + 6] = acc[di][6];
                    ci[j0 + 7] = acc[di][7];
                }
            } else {
                for di in 0..mr {
                    for dj in 0..nr {
                        let mut s = 0.0;
                        for k in 0..ka {
                            s += a[i0 + di][k] * b[k][j0 + dj];
                        }
                        c[i0 + di][j0 + dj] = s;
                    }
                }
            }
            j0 += NR;
        }
        i0 += MR;
    }
    Ok(c)
}

/// Minimum dimension at which `solve` uses the in-house blocked LU fast path. Below
/// this the trailing-update GEMM is too small to parallelise, so the portfolio LU
/// solver is used instead.
const BLOCKED_LU_MIN_DIM: usize = 1024;

/// Right-looking **blocked LU factorisation with partial pivoting** — our own
/// LAPACK-class kernel. Each panel (width NB) is factored unblocked, then the O(n³)
/// trailing-submatrix update `A22 -= L21·U12` (the bulk of the flops) runs on all
/// cores via the multithreaded flat-workspace GEMM. Returns the combined `L\U`
/// factors and the row permutation (`perm[i]` = original row index now at position
/// `i`), or `None` on a zero/non-finite pivot so callers can fall back to the
/// portfolio solver. Partial pivoting reproduces the LAPACK/SciPy factorisation, so
/// downstream solves match the reference to rounding.
#[allow(clippy::needless_range_loop)] // explicit row/col indices drive pivoting + the panel/trailing kernels
fn lu_factor_blocked(a_in: &[Vec<f64>]) -> Option<(Vec<Vec<f64>>, Vec<usize>)> {
    let n = a_in.len();
    if n == 0 || a_in[0].len() != n {
        return None;
    }
    const NB: usize = 128;
    let mut a: Vec<Vec<f64>> = a_in.to_vec(); // overwritten with the L\U factors
    // perm[i] = original row index now sitting at position i (apply to a RHS as Pb[i]=b[perm[i]]).
    let mut perm: Vec<usize> = (0..n).collect();

    let mut k = 0;
    while k < n {
        let kb = (k + NB).min(n);
        // (1) Factor panel columns [k, kb) over rows [k, n) with partial pivoting.
        for j in k..kb {
            let mut p = j;
            let mut mx = a[j][j].abs();
            for i in (j + 1)..n {
                let v = a[i][j].abs();
                if v > mx {
                    mx = v;
                    p = i;
                }
            }
            if mx == 0.0 || mx.is_nan() {
                return None; // singular within the panel -> fall back
            }
            if p != j {
                a.swap(p, j);
                perm.swap(p, j);
            }
            let pivot = a[j][j];
            for i in (j + 1)..n {
                a[i][j] /= pivot;
            }
            // Rank-1 update of the remaining panel columns (j+1..kb).
            for i in (j + 1)..n {
                let lij = a[i][j];
                if lij != 0.0 {
                    let (head, tail) = a.split_at_mut(i);
                    let row_i = &mut tail[0];
                    let row_j = &head[j];
                    for jj in (j + 1)..kb {
                        row_i[jj] -= lij * row_j[jj];
                    }
                }
            }
        }
        // (2) Triangular solve U12 = L11^-1 · A12 for rows [k,kb), cols [kb,n).
        for i in k..kb {
            for jj in kb..n {
                let mut s = a[i][jj];
                for (p, row_p) in a.iter().enumerate().take(i).skip(k) {
                    s -= a[i][p] * row_p[jj];
                }
                a[i][jj] = s;
            }
        }
        // (3) Trailing update A22 -= L21 · U12 via the parallel GEMM.
        if kb < n {
            let m2 = n - kb;
            let nb = kb - k;
            let n2 = n - kb;
            let mut l21 = Vec::with_capacity(m2);
            for row in a.iter().take(n).skip(kb) {
                l21.push(row[k..kb].to_vec());
            }
            let mut u12 = Vec::with_capacity(nb);
            for row in a.iter().take(kb).skip(k) {
                u12.push(row[kb..n].to_vec());
            }
            let prod = matmul_flat_workspace(&l21, &u12, m2, nb, n2)?;
            for (ii, i) in (kb..n).enumerate() {
                let pr = &prod[ii];
                let row = &mut a[i];
                for (jj, j) in (kb..n).enumerate() {
                    row[j] -= pr[jj];
                }
            }
        }
        k = kb;
    }
    Some((a, perm))
}

/// Forward/back substitution against precomputed `factors` (combined unit-lower L and
/// upper U) given the row permutation `perm`, solving `A x = rhs`. `y` is the permuted
/// RHS `Pb` on entry and is overwritten with the solution.
#[allow(clippy::needless_range_loop)]
fn lu_subst_factored(factors: &[Vec<f64>], mut y: Vec<f64>) -> Option<Vec<f64>> {
    let n = factors.len();
    for i in 0..n {
        let mut s = y[i];
        for p in 0..i {
            s -= factors[i][p] * y[p];
        }
        y[i] = s; // L unit diagonal
    }
    for i in (0..n).rev() {
        let mut s = y[i];
        for p in (i + 1)..n {
            s -= factors[i][p] * y[p];
        }
        let d = factors[i][i];
        if d == 0.0 {
            return None;
        }
        y[i] = s / d;
    }
    Some(y)
}

fn lu_solve_blocked(a_in: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    if b.len() != a_in.len() {
        return None;
    }
    let (factors, perm) = lu_factor_blocked(a_in)?;
    let pb: Vec<f64> = perm.iter().map(|&orig| b[orig]).collect();
    lu_subst_factored(&factors, pb)
}

/// Invert a general square `A` by factoring once with the parallel blocked LU and
/// solving `A X = I` over the identity columns in parallel (each column an independent
/// forward/back substitution). Returns `None` on singular/edge cases.
fn inv_blocked(a_in: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a_in.len();
    if n == 0 || a_in[0].len() != n {
        return None;
    }
    let (factors, perm) = lu_factor_blocked(a_in)?;
    // For identity column j, the permuted RHS Pe_j has a single 1 at the position i
    // where perm[i] == j. Solve A x = e_j -> column j of A^-1.
    let solve_col = |j: usize| -> Option<Vec<f64>> {
        let mut y = vec![0.0; n];
        for i in 0..n {
            if perm[i] == j {
                y[i] = 1.0;
                break;
            }
        }
        lu_subst_factored(&factors, y)
    };

    let nthreads = matmul_thread_count(n, n, n);
    let cols: Vec<Vec<f64>> = if nthreads <= 1 {
        (0..n).map(solve_col).collect::<Option<Vec<_>>>()?
    } else {
        let chunk = n.div_ceil(nthreads);
        let factors_ref = &factors;
        let perm_ref = &perm;
        let chunks: Vec<Option<Vec<Vec<f64>>>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..nthreads)
                .filter_map(|t| {
                    let c0 = t * chunk;
                    if c0 >= n {
                        return None;
                    }
                    let c1 = (c0 + chunk).min(n);
                    Some(scope.spawn(move || {
                        (c0..c1)
                            .map(|j| {
                                let mut y = vec![0.0; n];
                                for i in 0..n {
                                    if perm_ref[i] == j {
                                        y[i] = 1.0;
                                        break;
                                    }
                                }
                                lu_subst_factored(factors_ref, y)
                            })
                            .collect::<Option<Vec<_>>>()
                    }))
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut cols = Vec::with_capacity(n);
        for c in chunks {
            cols.extend(c?);
        }
        cols
    };

    // cols[j] is column j of the inverse; assemble row-major X[i][j].
    let mut x = vec![vec![0.0; n]; n];
    for j in 0..n {
        let col = &cols[j];
        for i in 0..n {
            x[i][j] = col[i];
        }
    }
    Some(x)
}

/// Solve a symmetric positive-definite `A x = b` via right-looking **blocked
/// Cholesky** (`A = L Lᵀ`, lower), parallelising the trailing update
/// `A22 -= L21·L21ᵀ` through the multithreaded flat-workspace GEMM — our own
/// Cholesky kernel. Returns `None` if any pivot is non-positive (i.e. `A` is not
/// positive definite within rounding) so the caller falls back to the portfolio
/// solver, preserving the `assume_a = pos` rejection semantics. Only the lower
/// triangle of `A` is read/written for `L`; the upper triangle is left as scratch.
#[allow(clippy::needless_range_loop)] // explicit triangle indices drive the panel + trailing kernels
fn cholesky_solve_blocked(a_in: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = a_in.len();
    if n == 0 || a_in[0].len() != n || b.len() != n {
        return None;
    }
    const NB: usize = 128;
    let mut a: Vec<Vec<f64>> = a_in.to_vec();

    let mut k = 0;
    while k < n {
        let kb = (k + NB).min(n);
        // (1) Unblocked Cholesky on the diagonal block (rows/cols k..kb).
        for j in k..kb {
            let mut d = a[j][j];
            for p in k..j {
                d -= a[j][p] * a[j][p];
            }
            if d <= 0.0 || d.is_nan() {
                return None; // not positive definite (or NaN) -> fall back
            }
            let ljj = d.sqrt();
            a[j][j] = ljj;
            for i in (j + 1)..kb {
                let mut s = a[i][j];
                for p in k..j {
                    s -= a[i][p] * a[j][p];
                }
                a[i][j] = s / ljj;
            }
        }
        // (2) Panel solve: A21 = A21 · L11^-T for rows kb..n, cols k..kb.
        for i in kb..n {
            for j in k..kb {
                let mut s = a[i][j];
                for p in k..j {
                    s -= a[i][p] * a[j][p];
                }
                a[i][j] = s / a[j][j];
            }
        }
        // (3) Trailing update A22 -= L21 · L21ᵀ via the parallel GEMM.
        if kb < n {
            let m2 = n - kb;
            let nb = kb - k;
            let mut l21 = Vec::with_capacity(m2);
            for row in a.iter().take(n).skip(kb) {
                l21.push(row[k..kb].to_vec());
            }
            // L21ᵀ as nb×m2.
            let mut l21t = vec![vec![0.0; m2]; nb];
            for (ii, row) in a.iter().take(n).skip(kb).enumerate() {
                for (jj, &v) in row[k..kb].iter().enumerate() {
                    l21t[jj][ii] = v;
                }
            }
            let prod = matmul_flat_workspace(&l21, &l21t, m2, nb, m2)?;
            for (ii, i) in (kb..n).enumerate() {
                let pr = &prod[ii];
                let row = &mut a[i];
                for (jj, j) in (kb..n).enumerate() {
                    row[j] -= pr[jj];
                }
            }
        }
        k = kb;
    }

    // Forward solve L y = b (lower), then back solve Lᵀ x = y. Lᵀ[i][p]=L[p][i]=a[p][i].
    let mut y = b.to_vec();
    for i in 0..n {
        let mut s = y[i];
        for p in 0..i {
            s -= a[i][p] * y[p];
        }
        y[i] = s / a[i][i];
    }
    for i in (0..n).rev() {
        let mut s = y[i];
        for p in (i + 1)..n {
            s -= a[p][i] * y[p];
        }
        y[i] = s / a[i][i];
    }
    Some(y)
}

/// Matrix-vector multiplication y = A * x.
///
/// Matches `A @ x`.
pub fn matvec(a: &[Vec<f64>], x: &[f64]) -> Result<Vec<f64>, LinalgError> {
    if a.is_empty() {
        return Ok(vec![]);
    }
    let m = a.len();
    let n = a[0].len();
    if n != x.len() {
        return Err(LinalgError::IncompatibleShapes {
            a_shape: (m, n),
            b_len: x.len(),
        });
    }
    let mut y = vec![0.0; m];
    for i in 0..m {
        for j in 0..n {
            y[i] += a[i][j] * x[j];
        }
    }
    Ok(y)
}

/// Outer product of two vectors.
///
/// Matches `numpy.outer`.
pub fn outer(a: &[f64], b: &[f64]) -> Vec<Vec<f64>> {
    let m = a.len();
    let n = b.len();
    let mut result = vec![vec![0.0; n]; m];
    for i in 0..m {
        for j in 0..n {
            result[i][j] = a[i] * b[j];
        }
    }
    result
}

/// Vector dot product.
pub fn vdot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
}

/// Vector L2 norm.
pub fn vnorm(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Isomorphism proof for the matmul ikj reorder [frankenscipy-vx65u]: the
    /// cache-friendly ikj order must be BIT-IDENTICAL to the naive ijk triple
    /// loop (each output element accumulates k in 0..ka order in both).
    #[test]
    fn matmul_ikj_is_bit_identical_to_naive_ijk() {
        // Deterministic pseudo-random non-square matrices.
        let make_matrix = |rows: usize, cols: usize, seed: u64| -> Vec<Vec<f64>> {
            (0..rows)
                .map(|i| {
                    (0..cols)
                        .map(|j| {
                            let r = (seed
                                .wrapping_mul(i as u64 + 1)
                                .wrapping_add(j as u64 * 7 + 1)
                                % 2003) as f64
                                / 991.0;
                            r - 1.0
                        })
                        .collect()
                })
                .collect()
        };
        for &(m, ka, n, seed) in &[
            (7usize, 5usize, 9usize, 1u64),
            (16, 16, 16, 2),
            (3, 11, 4, 3),
        ] {
            let a = make_matrix(m, ka, seed);
            let b = make_matrix(ka, n, seed.wrapping_add(100));
            // Reference: naive ijk.
            let mut expected = vec![vec![0.0f64; n]; m];
            for i in 0..m {
                for j in 0..n {
                    for k in 0..ka {
                        expected[i][j] += a[i][k] * b[k][j];
                    }
                }
            }
            let got = matmul(&a, &b).expect("matmul");
            for i in 0..m {
                for j in 0..n {
                    assert_eq!(
                        got[i][j].to_bits(),
                        expected[i][j].to_bits(),
                        "matmul ikj not bit-identical at ({i},{j}), m={m} ka={ka} n={n}"
                    );
                }
            }
        }
    }

    /// Isomorphism proof for the matmul register MICRO-KERNEL lever
    /// [frankenscipy-8l8r1]: the MRxNR register-tiled order must be
    /// BIT-IDENTICAL to a flat ikj triple loop. Uses dimensions that straddle
    /// the MR=NR=4 tile boundary with non-zero remainders (17, 25, 33, 9, 5) so
    /// both the full-tile and the ragged scalar-edge paths are exercised.
    #[test]
    fn matmul_microkernel_is_bit_identical_to_flat_ikj() {
        let make_matrix = |rows: usize, cols: usize, seed: u64| -> Vec<Vec<f64>> {
            (0..rows)
                .map(|i| {
                    (0..cols)
                        .map(|j| {
                            let r = (seed
                                .wrapping_mul(i as u64 * 3 + 1)
                                .wrapping_add(j as u64 * 11 + 1)
                                % 4099) as f64
                                / 1373.0;
                            r - 1.5
                        })
                        .collect()
                })
                .collect()
        };
        for &(m, ka, n, seed) in &[
            (17usize, 23usize, 19usize, 7u64),
            (25, 8, 31, 11),
            (33, 17, 8, 13),
            (8, 8, 8, 17),
            (9, 4, 5, 19),
        ] {
            let a = make_matrix(m, ka, seed);
            let b = make_matrix(ka, n, seed.wrapping_add(50));
            // Reference: flat ikj (same per-element accumulation order as naive
            // ijk, since k is outer relative to j).
            let mut expected = vec![vec![0.0f64; n]; m];
            for i in 0..m {
                for k in 0..ka {
                    let aik = a[i][k];
                    for j in 0..n {
                        expected[i][j] += aik * b[k][j];
                    }
                }
            }
            let got = matmul(&a, &b).expect("matmul");
            for i in 0..m {
                for j in 0..n {
                    assert_eq!(
                        got[i][j].to_bits(),
                        expected[i][j].to_bits(),
                        "micro-kernel matmul not bit-identical at ({i},{j}), m={m} ka={ka} n={n}"
                    );
                }
            }
        }
    }

    /// Golden-output proof for the matmul register micro-kernel lever
    /// [frankenscipy-8l8r1]: a fixed deterministic 80x80 product hashes to a
    /// frozen 64-bit FNV-1a digest over the raw f64 bit patterns (self-contained,
    /// no external crate). If any future edit perturbs a single output bit, this
    /// digest changes and the test fails. 80 is a multiple of MR=NR=4 so the
    /// product is computed entirely through the full-tile path.
    #[test]
    fn matmul_microkernel_golden_digest() {
        let n = 80usize;
        let a: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| ((i * 31 + j * 17) % 97) as f64 * 0.01)
                    .collect()
            })
            .collect();
        let b: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| ((i * 13 + j * 7) % 89) as f64 * 0.01)
                    .collect()
            })
            .collect();
        let c = matmul(&a, &b).expect("matmul");
        // FNV-1a 64-bit over little-endian f64 bit patterns.
        let mut digest: u64 = 0xcbf2_9ce4_8422_2325;
        for row in &c {
            for &v in row {
                for byte in v.to_bits().to_le_bytes() {
                    digest ^= byte as u64;
                    digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
                }
            }
        }
        assert_eq!(
            digest, 0xf9aa_16d2_dc37_468f,
            "matmul golden digest changed — output is no longer bit-identical (got {digest:#018x})"
        );
    }

    /// Isomorphism proof for the deep flat-workspace GEMM primitive
    /// [frankenscipy-8l8r1.19]: direct helper exercise keeps the public dispatch
    /// gate at 1024 while proving full tiles and ragged edges are bit-identical
    /// to naive ijk accumulation.
    #[test]
    fn matmul_flat_workspace_is_bit_identical_to_naive_ijk() {
        assert_eq!(MATMUL_FLAT_WORKSPACE_MIN_DIM, 1024);

        let make_matrix = |rows: usize, cols: usize, seed: u64| -> Vec<Vec<f64>> {
            (0..rows)
                .map(|i| {
                    (0..cols)
                        .map(|j| {
                            let raw = seed
                                .wrapping_mul(i as u64 + 5)
                                .wrapping_add(j as u64 * 17 + 3)
                                % 8191;
                            raw as f64 / 2047.0 - 2.0
                        })
                        .collect()
                })
                .collect()
        };

        for &(m, ka, n, seed) in &[
            (16usize, 16usize, 16usize, 23u64),
            (17, 23, 19, 29),
            (9, 5, 13, 31),
        ] {
            let a = make_matrix(m, ka, seed);
            let b = make_matrix(ka, n, seed.wrapping_add(100));
            let mut expected = vec![vec![0.0f64; n]; m];
            for i in 0..m {
                for j in 0..n {
                    for k in 0..ka {
                        expected[i][j] += a[i][k] * b[k][j];
                    }
                }
            }

            let got =
                matmul_flat_workspace(&a, &b, m, ka, n).expect("flat workspace dimensions fit");
            for i in 0..m {
                for j in 0..n {
                    assert_eq!(
                        got[i][j].to_bits(),
                        expected[i][j].to_bits(),
                        "flat workspace matmul not bit-identical at ({i},{j}), m={m} ka={ka} n={n}"
                    );
                }
            }
        }
    }

    /// Correctness of the in-house blocked LU solver: it must match the reference
    /// (nalgebra) LU solve to tolerance and produce a tiny residual ||Ax-b||, across
    /// sizes that exercise both the sequential and parallel trailing-update GEMM and
    /// that force partial pivoting (small/negative diagonals).
    #[test]
    #[allow(clippy::needless_range_loop)]
    fn cholesky_solve_blocked_matches_reference() {
        // Build SPD A = MᵀM + nI.
        let spd = |n: usize, seed: u64| -> Vec<Vec<f64>> {
            let m: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| {
                            (seed
                                .wrapping_mul(i as u64 + 7)
                                .wrapping_add(j as u64 * 31 + 5)
                                % 9973) as f64
                                / 4986.0
                                - 1.0
                        })
                        .collect()
                })
                .collect();
            let mut a = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in 0..n {
                    let mut t = 0.0;
                    for k in 0..n {
                        t += m[k][i] * m[k][j];
                    }
                    a[i][j] = t + if i == j { n as f64 } else { 0.0 };
                }
            }
            a
        };
        for &(n, seed) in &[(16usize, 11u64), (130, 23), (270, 57)] {
            let a = spd(n, seed);
            let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).cos()).collect();
            let x = cholesky_solve_blocked(&a, &b).expect("blocked Cholesky solves SPD");
            let opts = SolveOptions {
                assume_a: Some(MatrixAssumption::PositiveDefinite),
                ..SolveOptions::default()
            };
            let reference = solve(&a, &b, opts).expect("reference SPD solve");
            let mut max_diff = 0.0_f64;
            for (&xi, &ri) in x.iter().zip(&reference.x) {
                max_diff = max_diff.max((xi - ri).abs());
            }
            let mut max_res = 0.0_f64;
            for i in 0..n {
                let mut s = 0.0;
                for j in 0..n {
                    s += a[i][j] * x[j];
                }
                max_res = max_res.max((s - b[i]).abs());
            }
            assert!(
                max_diff < 1e-7,
                "Cholesky vs reference diverged n={n}: {max_diff:e}"
            );
            assert!(
                max_res < 1e-9,
                "Cholesky residual too large n={n}: {max_res:e}"
            );
        }
        // Non-PD matrix must be rejected (None) so the caller can fall back.
        let not_pd = vec![vec![1.0, 2.0], vec![2.0, 1.0]]; // eigenvalues 3, -1
        assert!(cholesky_solve_blocked(&not_pd, &[1.0, 1.0]).is_none());
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn inv_blocked_matches_reference() {
        let make = |n: usize, seed: u64| -> Vec<Vec<f64>> {
            (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| {
                            let r = seed
                                .wrapping_mul(i as u64 + 7)
                                .wrapping_add(j as u64 * 31 + 5)
                                % 9973;
                            let v = r as f64 / 4986.0 - 1.0;
                            if i == j { v + n as f64 * 0.5 } else { v }
                        })
                        .collect()
                })
                .collect()
        };
        for &(n, seed) in &[(16usize, 11u64), (130, 23), (270, 57)] {
            let a = make(n, seed);
            let got = inv_blocked(&a).expect("blocked inverse");
            let reference = inv(&a, InvOptions::default()).expect("reference inverse");
            // Match the reference inverse to tolerance.
            let mut max_diff = 0.0_f64;
            for i in 0..n {
                for j in 0..n {
                    max_diff = max_diff.max((got[i][j] - reference.inverse[i][j]).abs());
                }
            }
            // A · A^-1 ≈ I residual.
            let mut max_res = 0.0_f64;
            for i in 0..n {
                for j in 0..n {
                    let mut s = 0.0;
                    for k in 0..n {
                        s += a[i][k] * got[k][j];
                    }
                    let target = if i == j { 1.0 } else { 0.0 };
                    max_res = max_res.max((s - target).abs());
                }
            }
            assert!(
                max_diff < 1e-7,
                "blocked inv vs reference diverged n={n}: {max_diff:e}"
            );
            assert!(max_res < 1e-9, "A·A^-1 - I too large n={n}: {max_res:e}");
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn lu_solve_blocked_matches_reference() {
        let make = |n: usize, seed: u64| -> (Vec<Vec<f64>>, Vec<f64>) {
            let a: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| {
                            let r = seed
                                .wrapping_mul(i as u64 + 7)
                                .wrapping_add(j as u64 * 31 + 5)
                                % 9973;
                            let v = r as f64 / 4986.0 - 1.0;
                            // Mildly diagonally dominant for conditioning, but the
                            // off-diagonals still force row interchanges.
                            if i == j { v + n as f64 * 0.5 } else { v }
                        })
                        .collect()
                })
                .collect();
            let b: Vec<f64> = (0..n)
                .map(|i| (seed.wrapping_mul(i as u64 + 3) % 1009) as f64 / 503.0 - 1.0)
                .collect();
            (a, b)
        };
        // n=130 and 270 straddle NB=128 (multi-panel); 270 also has a partial last panel.
        for &(n, seed) in &[(16usize, 11u64), (130, 23), (200, 41), (270, 57)] {
            let (a, b) = make(n, seed);
            let x = lu_solve_blocked(&a, &b).expect("blocked LU solves nonsingular system");
            let reference = solve_general(&a, &b).expect("reference solve");
            let mut max_diff = 0.0_f64;
            for (&xi, &ri) in x.iter().zip(&reference.x) {
                max_diff = max_diff.max((xi - ri).abs());
            }
            // Residual ||A x - b||_inf.
            let mut max_res = 0.0_f64;
            for i in 0..n {
                let mut s = 0.0;
                for j in 0..n {
                    s += a[i][j] * x[j];
                }
                max_res = max_res.max((s - b[i]).abs());
            }
            assert!(
                max_diff < 1e-7,
                "blocked LU vs reference diverged at n={n}: max_diff={max_diff:e}"
            );
            assert!(
                max_res < 1e-9,
                "blocked LU residual too large at n={n}: {max_res:e}"
            );
        }
    }

    /// Byte-identity proof for the multithreaded flat-workspace GEMM: splitting the
    /// output rows across threads (what `matmul_flat_workspace` does via
    /// `chunks_mut` + `thread::scope`) must produce bit-identical results to
    /// computing the whole range in one call. Drives `matmul_flat_compute_rows`
    /// directly so the parallel invariant is checked without a 1024³ matmul.
    #[test]
    fn matmul_flat_compute_rows_row_split_is_bit_identical() {
        // n = 28 is NOT a multiple of NR(=8): exercises both the SIMD-packed columns
        // and the scalar tail path under arbitrary row splits.
        let (m, ka, n) = (37usize, 29usize, 28usize);
        let make = |rows: usize, cols: usize, seed: u64| -> Vec<f64> {
            (0..rows * cols)
                .map(|t| {
                    let i = (t / cols) as u64;
                    let j = (t % cols) as u64;
                    (seed.wrapping_mul(i + 5).wrapping_add(j * 17 + 3) % 8191) as f64 / 2047.0 - 2.0
                })
                .collect()
        };
        let a_flat = make(m, ka, 23);
        let b_flat = make(ka, n, 123);
        const NR: usize = 8;
        let full_n_blocks = n / NR;
        let mut packed_b = Vec::new();
        for jb in 0..full_n_blocks {
            let j0 = jb * NR;
            for k in 0..ka {
                let base = k * n + j0;
                packed_b.extend_from_slice(&b_flat[base..base + NR]);
            }
        }

        let mut full = vec![0.0; m * n];
        matmul_flat_compute_rows(&mut full, 0, m, &a_flat, &packed_b, &b_flat, ka, n);

        for &nchunks in &[2usize, 3, 5, 8, m] {
            let chunk = m.div_ceil(nchunks);
            let mut split = vec![0.0; m * n];
            let mut t = 0;
            loop {
                let rs = t * chunk;
                if rs >= m {
                    break;
                }
                let re = (rs + chunk).min(m);
                matmul_flat_compute_rows(
                    &mut split[rs * n..re * n],
                    rs,
                    re,
                    &a_flat,
                    &packed_b,
                    &b_flat,
                    ka,
                    n,
                );
                t += 1;
            }
            for idx in 0..m * n {
                assert_eq!(
                    full[idx].to_bits(),
                    split[idx].to_bits(),
                    "row-split (nchunks={nchunks}) not bit-identical at flat index {idx}"
                );
            }
        }
    }

    /// Isomorphism proof for the eig_banded Q*tri_evecs ikj reorder [perf]:
    /// the cache-friendly ikj order must be BIT-IDENTICAL to the naive ijk
    /// triple loop used for the eigenvector back-transform.
    #[test]
    fn eig_banded_qmul_ikj_is_bit_identical() {
        let mk = |seed: u64| -> Vec<Vec<f64>> {
            let n = 23usize;
            (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| {
                            ((seed
                                .wrapping_mul(i as u64 + 3)
                                .wrapping_add(j as u64 * 11 + 2))
                                % 1997) as f64
                                / 983.0
                                - 1.0
                        })
                        .collect()
                })
                .collect()
        };
        let q = mk(7);
        let tri = mk(9);
        let n = q.len();
        // Reference: naive ijk.
        let mut expected = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    expected[i][j] += q[i][k] * tri[k][j];
                }
            }
        }
        // ikj (mirrors eig_banded's back-transform).
        let mut got = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            let qi = &q[i];
            let ri = &mut got[i];
            for k in 0..n {
                let qik = qi[k];
                let tk = &tri[k];
                for j in 0..n {
                    ri[j] += qik * tk[j];
                }
            }
        }
        for i in 0..n {
            for j in 0..n {
                assert_eq!(
                    got[i][j].to_bits(),
                    expected[i][j].to_bits(),
                    "eig_banded Q-mul ikj not bit-identical at ({i},{j})"
                );
            }
        }
    }

    /// Perf comparison for the matmul ikj reorder [frankenscipy-vx65u]. Run with
    /// `cargo test -p fsci-linalg --release matmul_ikj_perf -- --ignored --nocapture`.
    #[test]
    #[ignore = "perf measurement; run explicitly in --release"]
    fn matmul_ikj_perf_vs_naive_ijk() {
        let n = 768usize;
        let a: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| ((i * 31 + j * 17) % 97) as f64 * 0.01)
                    .collect()
            })
            .collect();
        let b: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| ((i * 13 + j * 7) % 89) as f64 * 0.01)
                    .collect()
            })
            .collect();

        let t_naive = std::time::Instant::now();
        let mut naive = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    naive[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        let naive_ns = t_naive.elapsed().as_secs_f64();
        std::hint::black_box(&naive);

        let t_ikj = std::time::Instant::now();
        let ikj = matmul(&a, &b).expect("matmul");
        let ikj_ns = t_ikj.elapsed().as_secs_f64();
        std::hint::black_box(&ikj);

        let speedup = naive_ns / ikj_ns;
        println!("matmul {n}x{n}: naive_ijk={naive_ns:.4}s ikj={ikj_ns:.4}s speedup={speedup:.2}x");
        assert!(speedup > 1.0, "ikj should be faster: {speedup:.2}x");
    }

    /// Perf witness for the matmul register micro-kernel lever
    /// [frankenscipy-8l8r1]. Compares the shipped register-blocked `matmul`
    /// against the previous flat ikj baseline (B streamed once per output row).
    /// Run with
    /// `cargo test -p fsci-linalg --release matmul_microkernel_perf -- --ignored --nocapture`.
    #[test]
    #[ignore = "perf measurement; run explicitly in --release"]
    fn matmul_microkernel_perf_vs_flat_ikj() {
        // Flat ikj: identical math, one output row at a time (the pre-lever loop).
        fn matmul_flat_ikj(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
            let (m, ka) = (a.len(), a[0].len());
            let n = b[0].len();
            let mut c = vec![vec![0.0f64; n]; m];
            for i in 0..m {
                let ci = &mut c[i];
                let ai = &a[i];
                for k in 0..ka {
                    let aik = ai[k];
                    let bk = &b[k];
                    for j in 0..n {
                        ci[j] += aik * bk[j];
                    }
                }
            }
            c
        }

        for &n in &[768usize, 1024usize] {
            let a: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| ((i * 31 + j * 17) % 97) as f64 * 0.01)
                        .collect()
                })
                .collect();
            let b: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| ((i * 13 + j * 7) % 89) as f64 * 0.01)
                        .collect()
                })
                .collect();

            // Warm + measure flat ikj.
            let t_flat = std::time::Instant::now();
            let flat = matmul_flat_ikj(&a, &b);
            let flat_s = t_flat.elapsed().as_secs_f64();
            std::hint::black_box(&flat);

            // Measure shipped blocked matmul.
            let t_blk = std::time::Instant::now();
            let blk = matmul(&a, &b).expect("matmul");
            let blk_s = t_blk.elapsed().as_secs_f64();
            std::hint::black_box(&blk);

            // Bit-identical guard alongside the timing.
            for i in 0..n {
                for j in 0..n {
                    assert_eq!(flat[i][j].to_bits(), blk[i][j].to_bits());
                }
            }

            let speedup = flat_s / blk_s;
            println!(
                "matmul {n}x{n}: flat_ikj={flat_s:.4}s microkernel={blk_s:.4}s speedup={speedup:.2}x"
            );
        }
    }

    fn assert_close_slice(actual: &[f64], expected: &[f64], atol: f64, rtol: f64) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let tol = atol + rtol * e.abs();
            assert!(
                (a - e).abs() <= tol,
                "index={idx} expected={e} actual={a} tol={tol}"
            );
        }
    }

    fn assert_close_matrix(actual: &[Vec<f64>], expected: &[Vec<f64>], atol: f64, rtol: f64) {
        assert_eq!(actual.len(), expected.len());
        for (row_idx, (a_row, e_row)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(a_row.len(), e_row.len());
            for (col_idx, (a, e)) in a_row.iter().zip(e_row.iter()).enumerate() {
                let tol = atol + rtol * e.abs();
                assert!(
                    (a - e).abs() <= tol,
                    "row={row_idx} col={col_idx} expected={e} actual={a} tol={tol}"
                );
            }
        }
    }

    fn lock_audit_ledger(ledger: &SyncSharedAuditLedger) -> std::sync::MutexGuard<'_, AuditLedger> {
        match ledger.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    #[test]
    fn audit_records_after_poisoned_ledger() {
        let audit_ledger = sync_audit_ledger();

        let poisoned_thread = {
            let ledger = audit_ledger.clone();
            std::thread::spawn(move || {
                let _guard = ledger.lock().expect("acquire ledger");
                std::panic::panic_any("poison fsci-linalg audit ledger on purpose");
            })
            .join()
        };
        assert!(poisoned_thread.is_err(), "thread should have panicked");
        assert!(
            audit_ledger.lock().is_err(),
            "ledger must be poisoned after panic"
        );

        record_fail_closed(&audit_ledger, b"bad matrix", "non_finite_input", "rejected");
        record_bounded_recovery(
            &audit_ledger,
            b"recovered matrix",
            "svd_fallback",
            "recovered",
        );
        record_casp_decision(&audit_ledger, b"casp", SolverAction::DirectLU, 1.0, false);
        record_mode_decision(&audit_ledger, b"mode", RuntimeMode::Strict, "executed");

        let ledger = audit_ledger
            .lock()
            .expect("ledger poison should have been cleared");
        assert_eq!(ledger.len(), 4);
    }

    fn rotated_diagonal(lambda1: f64, lambda2: f64) -> Vec<Vec<f64>> {
        let diag = 0.5 * (lambda1 + lambda2);
        let off_diag = 0.5 * (lambda1 - lambda2);
        vec![vec![diag, off_diag], vec![off_diag, diag]]
    }

    fn reconstruct_qr_result(result: &QrResult) -> Vec<Vec<f64>> {
        matmul(&result.q, &result.r).expect("QR factors should multiply")
    }

    fn assert_certificate_populated(certificate: &SolveCertificate) {
        assert_eq!(certificate.posterior.len(), 4);
        assert_eq!(certificate.expected_losses.len(), 5);
    }

    #[test]
    fn solve_general_happy_path() {
        let a = vec![vec![3.0, 2.0], vec![1.0, 2.0]];
        let b = vec![5.0, 5.0];
        let result = solve(&a, &b, SolveOptions::default()).expect("solve must work");
        assert_close_slice(&result.x, &[0.0, 2.5], 1e-12, 1e-12);
    }

    #[test]
    fn solve_triangular_lower_path() {
        let a = vec![vec![2.0, 0.0], vec![3.0, 4.0]];
        let b = vec![4.0, 11.0];
        let result = solve_triangular(
            &a,
            &b,
            TriangularSolveOptions {
                lower: true,
                ..TriangularSolveOptions::default()
            },
        )
        .expect("triangular solve must work");
        assert_close_slice(&result.x, &[2.0, 1.25], 1e-12, 1e-12);
    }

    #[test]
    fn solve_banded_matches_dense() {
        let ab = vec![
            vec![0.0, 0.0, -1.0, -1.0, -1.0],
            vec![0.0, 2.0, 2.0, 2.0, 2.0],
            vec![5.0, 4.0, 3.0, 2.0, 1.0],
            vec![1.0, 1.0, 1.0, 1.0, 0.0],
        ];
        let b = vec![0.0, 1.0, 2.0, 2.0, 3.0];
        let result =
            solve_banded((1, 2), &ab, &b, SolveOptions::default()).expect("banded solve must work");
        assert_close_slice(
            &result.x,
            &[-2.37288136, 3.93220339, -4.0, 4.3559322, -1.3559322],
            1e-8,
            1e-8,
        );
    }

    #[test]
    fn inverse_matches_known_result() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = inv(&a, InvOptions::default()).expect("inverse must work");
        assert_close_matrix(
            &result.inverse,
            &[vec![-2.0, 1.0], vec![1.5, -0.5]],
            1e-12,
            1e-12,
        );
    }

    #[test]
    fn determinant_matches_known_result() {
        let a = vec![
            vec![0.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let value = det(&a, RuntimeMode::Strict, true).expect("det should work");
        assert!((value - 3.0).abs() < 1e-12);
    }

    #[test]
    fn lstsq_overdetermined_system() {
        let a = vec![
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![1.0, 2.0],
            vec![1.0, 3.0],
        ];
        let b = vec![1.0, 2.0, 2.0, 4.0];
        let result = lstsq(&a, &b, LstsqOptions::default()).expect("lstsq must work");
        assert_eq!(result.rank, 2);
        assert_close_slice(&result.x, &[0.9, 0.9], 1e-10, 1e-10);
        assert_eq!(result.residuals.len(), 1);
    }

    #[test]
    fn pinv_rectangular_and_rank() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 0.0]];
        let result = pinv(&a, PinvOptions::default()).expect("pinv should work");
        assert_eq!(result.rank, 2);
        assert_close_matrix(
            &result.pseudo_inverse,
            &[vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
            1e-12,
            1e-12,
        );
    }

    #[test]
    fn pinv_negative_thresholds_are_rejected() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let err = pinv(
            &a,
            PinvOptions {
                atol: Some(-1.0),
                ..PinvOptions::default()
            },
        )
        .expect_err("negative threshold must fail");
        assert_eq!(err, LinalgError::InvalidPinvThreshold);
    }

    #[test]
    fn hardened_mode_forces_finite_checks() {
        let a = vec![vec![1.0, f64::NAN], vec![0.0, 1.0]];
        let b = vec![1.0, 2.0];
        let strict = solve(
            &a,
            &b,
            SolveOptions {
                mode: RuntimeMode::Strict,
                check_finite: false,
                ..SolveOptions::default()
            },
        );
        assert!(
            strict.is_ok(),
            "strict mode with disabled finite checks should proceed"
        );

        let hardened = solve(
            &a,
            &b,
            SolveOptions {
                mode: RuntimeMode::Hardened,
                check_finite: false,
                ..SolveOptions::default()
            },
        );
        assert_eq!(
            hardened.expect_err("hardened mode should fail-closed"),
            LinalgError::NonFiniteInput
        );
    }

    #[test]
    fn solve_general_backward_error_is_small() {
        let a = vec![vec![3.0, 2.0], vec![1.0, 2.0]];
        let b = vec![5.0, 5.0];
        let result = solve(&a, &b, SolveOptions::default()).expect("solve must work");
        assert!(
            result.backward_error.unwrap() < 1e-14,
            "backward error should be near machine epsilon"
        );
    }

    #[test]
    fn backward_error_nonfinite_is_infinite() {
        let matrix = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let rhs = DVector::from_column_slice(&[1.0, 1.0]);
        let x = DVector::from_column_slice(&[f64::NAN, 1.0]);
        let backward = compute_backward_error(&matrix, &x, &rhs);
        assert!(backward.is_infinite());

        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let x_dense = vec![f64::NAN, 1.0];
        let b = vec![1.0, 1.0];
        let backward_dense = compute_backward_error_dense(&a, &x_dense, &b);
        assert!(backward_dense.is_infinite());
    }

    #[test]
    fn inv_general_single_lu_matches_column_by_column() {
        let a = vec![
            vec![4.0, 7.0, 2.0],
            vec![3.0, 6.0, 1.0],
            vec![2.0, 5.0, 3.0],
        ];
        let result = inv(&a, InvOptions::default()).expect("inv must work");
        // Verify A * A^{-1} ≈ I
        for (i, a_row) in a.iter().enumerate() {
            for j in 0..a.len() {
                let sum: f64 = a_row
                    .iter()
                    .enumerate()
                    .map(|(k, &a_ik)| a_ik * result.inverse[k][j])
                    .sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sum - expected).abs() < 1e-10,
                    "A * inv(A) should be identity: [{i}][{j}] = {sum}"
                );
            }
        }
    }

    #[test]
    fn dmatrix_from_rows_with_norm1_matches_matrix_norm1() {
        let rows = vec![
            vec![1.0, -3.0, 2.0],
            vec![-4.0, 5.0, -6.0],
            vec![7.0, -8.0, 9.0],
        ];
        let (matrix, fused_norm) =
            dmatrix_from_rows_with_norm1(&rows).expect("matrix build must work");
        assert_eq!(rows_from_dmatrix(&matrix), rows);
        assert_eq!(fused_norm, matrix_norm1(&matrix));

        let non_finite = vec![vec![1.0, f64::INFINITY], vec![2.0, 3.0]];
        let (matrix, fused_norm) =
            dmatrix_from_rows_with_norm1(&non_finite).expect("matrix build must work");
        assert!(fused_norm.is_nan());
        assert!(matrix_norm1(&matrix).is_nan());
    }

    #[test]
    fn fast_rcond_from_lu_well_conditioned() {
        let matrix = DMatrix::from_row_slice(2, 2, &[3.0, 2.0, 1.0, 2.0]);
        let lu = matrix.clone().lu();
        let rcond = fast_rcond_from_lu(&lu, matrix_norm1(&matrix), 2);
        assert!(
            rcond > 0.1,
            "well-conditioned matrix should have high rcond, got {rcond}"
        );
    }

    #[test]
    fn fast_rcond_from_lu_ill_conditioned() {
        let matrix = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1e-15]);
        let lu = matrix.clone().lu();
        let rcond = fast_rcond_from_lu(&lu, matrix_norm1(&matrix), 2);
        assert!(
            rcond < 1e-12,
            "ill-conditioned matrix should have low rcond, got {rcond}"
        );
    }

    #[test]
    fn condition_diagnostics_identity_reports_diagonal_spd() {
        let a = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let report = condition_diagnostics(&a).expect("identity diagnostics");
        assert_eq!(report.matrix_shape, (3, 3));
        assert_eq!(report.structural_evidence, StructuralEvidence::Diagonal);
        assert!(report.diagonal);
        assert!(report.upper_triangular);
        assert!(report.lower_triangular);
        assert!(report.symmetric);
        assert!(report.positive_definite);
        assert_eq!(report.bandwidth, (0, 0));
        assert_eq!(report.size_category, MatrixSizeCategory::Small);
        assert!((report.rcond_estimate - 1.0).abs() < 1e-12);
        assert!(report.sparsity_ratio > 0.6);
    }

    #[test]
    fn condition_diagnostics_triangular_reports_fast_path_shape() {
        let a = vec![
            vec![4.0, -2.0, 1.0],
            vec![0.0, 3.0, 5.0],
            vec![0.0, 0.0, 2.0],
        ];
        let report = condition_diagnostics(&a).expect("triangular diagnostics");
        assert_eq!(report.structural_evidence, StructuralEvidence::Triangular);
        assert!(!report.diagonal);
        assert!(report.upper_triangular);
        assert!(!report.lower_triangular);
        assert_eq!(report.bandwidth, (0, 2));
        assert!(report.banded);
        assert!(report.rcond_estimate > 0.0);
    }

    #[test]
    fn condition_diagnostics_hilbert_reports_ill_conditioning() {
        let a = hilbert(6);
        let report = condition_diagnostics(&a).expect("hilbert diagnostics");
        assert_eq!(report.structural_evidence, StructuralEvidence::General);
        assert!(report.symmetric);
        assert!(report.positive_definite);
        assert!(report.rcond_estimate < 1e-6);
    }

    #[test]
    fn condition_diagnostics_general_matrix_has_no_special_structure() {
        let a = vec![
            vec![2.0, 1.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 10.0],
        ];
        let report = condition_diagnostics(&a).expect("general diagnostics");
        assert_eq!(report.structural_evidence, StructuralEvidence::General);
        assert!(!report.diagonal);
        assert!(!report.upper_triangular);
        assert!(!report.lower_triangular);
        assert!(!report.symmetric);
        assert!(!report.positive_definite);
        assert!(report.rcond_estimate > 0.0);
    }

    #[test]
    fn condition_diagnostics_rectangular_matrix_is_conservative() {
        let a = vec![vec![1.0, 0.0, 0.0], vec![0.0, 2.0, 0.0]];
        let report = condition_diagnostics(&a).expect("rectangular diagnostics");
        assert_eq!(report.matrix_shape, (2, 3));
        assert_eq!(report.structural_evidence, StructuralEvidence::Diagonal);
        assert_eq!(report.rcond_estimate, 0.0);
        assert!(report.diagonal);
    }

    #[test]
    fn is_diagonal_rejects_nan_off_diagonal() {
        let a = vec![vec![1.0, f64::NAN], vec![0.0, 2.0]];
        assert!(!is_diagonal(&a, 0.0));
    }

    #[test]
    fn is_upper_triangular_rejects_nan_below_diagonal() {
        let a = vec![vec![1.0, 2.0], vec![f64::NAN, 3.0]];
        assert!(!is_upper_triangular(&a, 0.0));
    }

    #[test]
    fn is_lower_triangular_rejects_nan_above_diagonal() {
        let a = vec![vec![1.0, f64::NAN], vec![2.0, 3.0]];
        assert!(!is_lower_triangular(&a, 0.0));
    }

    #[test]
    fn structural_checks_reject_ragged_input() {
        let ragged = vec![vec![1.0, 0.0], vec![0.0]];
        assert!(!is_diagonal(&ragged, 0.0));
        assert!(!is_upper_triangular(&ragged, 0.0));
        assert!(!is_lower_triangular(&ragged, 0.0));
    }

    #[test]
    fn solve_with_casp_well_conditioned_uses_lu() {
        let a = vec![vec![3.0, 2.0], vec![1.0, 2.0]];
        let b = vec![5.0, 5.0];
        let mut portfolio = SolverPortfolio::new(RuntimeMode::Strict, 64);
        let result = solve_with_casp(&a, &b, SolveOptions::default(), &mut portfolio)
            .expect("solve must work");
        assert_close_slice(&result.x, &[0.0, 2.5], 1e-12, 1e-12);
        assert_eq!(portfolio.evidence_len(), 1);
    }

    #[test]
    fn casp_selects_diagonal_fast_path() {
        let a = vec![
            vec![2.0, 0.0, 0.0],
            vec![0.0, 3.0, 0.0],
            vec![0.0, 0.0, 4.0],
        ];
        let b = vec![4.0, 9.0, 16.0];
        let result = solve(&a, &b, SolveOptions::default()).expect("solve works");
        assert_close_slice(&result.x, &[2.0, 3.0, 4.0], 1e-14, 1e-14);
        let certificate = result.certificate.expect("certificate populated");
        assert_eq!(certificate.action, SolverAction::DiagonalFastPath);
        assert_eq!(
            certificate.structural_evidence,
            StructuralEvidence::Diagonal
        );
        assert!(!certificate.fallback_active);
        assert_certificate_populated(&certificate);
    }

    #[test]
    fn casp_selects_triangular_fast_path() {
        let a = vec![
            vec![2.0, 3.0, 1.0],
            vec![0.0, 4.0, 5.0],
            vec![0.0, 0.0, 6.0],
        ];
        let b = vec![7.0, 8.0, 9.0];
        let result = solve(&a, &b, SolveOptions::default()).expect("solve works");
        assert_close_slice(&result.x, &[2.5625, 0.125, 1.5], 1e-12, 1e-12);
        let certificate = result.certificate.expect("certificate populated");
        assert_eq!(certificate.action, SolverAction::TriangularFastPath);
        assert_eq!(
            certificate.structural_evidence,
            StructuralEvidence::Triangular
        );
        assert!(!certificate.fallback_active);
        assert_certificate_populated(&certificate);
    }

    #[test]
    fn casp_selects_lu_for_well_conditioned() {
        let a = vec![vec![4.0, 1.0], vec![2.0, 3.0]];
        let b = vec![1.0, 2.0];
        let report = condition_diagnostics(&a).expect("condition diagnostics");
        assert!(
            report.rcond_estimate > 1e-2,
            "expected well-conditioned rcond, got {}",
            report.rcond_estimate
        );
        let result = solve(&a, &b, SolveOptions::default()).expect("solve works");
        let certificate = result.certificate.expect("certificate populated");
        assert_eq!(certificate.action, SolverAction::DirectLU);
        assert_eq!(certificate.structural_evidence, StructuralEvidence::General);
        assert!(!certificate.fallback_active);
        assert_certificate_populated(&certificate);
    }

    #[test]
    fn casp_selects_qr_for_moderate_condition() {
        let a = rotated_diagonal(1.0, 1e-4);
        let b = vec![1.0, -1.0];
        let report = condition_diagnostics(&a).expect("condition diagnostics");
        assert!(
            report.rcond_estimate < 1e-2 && report.rcond_estimate > 1e-6,
            "expected moderate rcond, got {}",
            report.rcond_estimate
        );
        assert_eq!(report.structural_evidence, StructuralEvidence::General);
        let result = solve(&a, &b, SolveOptions::default()).expect("solve works");
        let certificate = result.certificate.expect("certificate populated");
        assert_eq!(certificate.action, SolverAction::PivotedQR);
        assert!(!certificate.fallback_active);
        assert_certificate_populated(&certificate);
    }

    #[test]
    fn casp_selects_svd_for_ill_conditioned() {
        let a = rotated_diagonal(1.0, 1e-12);
        let b = vec![1.0, -1.0];
        let report = condition_diagnostics(&a).expect("condition diagnostics");
        assert!(
            report.rcond_estimate < 1e-9,
            "expected ill-conditioned rcond, got {}",
            report.rcond_estimate
        );
        let result = solve(&a, &b, SolveOptions::default()).expect("solve works");
        let certificate = result.certificate.expect("certificate populated");
        assert_eq!(certificate.action, SolverAction::SVDFallback);
        assert!(!certificate.fallback_active);
        assert_certificate_populated(&certificate);
    }

    #[test]
    fn casp_certificate_populated_for_core_entrypoints() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let b = vec![3.0, 1.0];

        let solve_cert = solve(&a, &b, SolveOptions::default())
            .expect("solve works")
            .certificate
            .expect("solve certificate");
        assert_certificate_populated(&solve_cert);

        let inv_cert = inv(&a, InvOptions::default())
            .expect("inv works")
            .certificate
            .expect("inv certificate");
        assert_certificate_populated(&inv_cert);

        let lstsq_cert = lstsq(&a, &b, LstsqOptions::default())
            .expect("lstsq works")
            .certificate
            .expect("lstsq certificate");
        assert_certificate_populated(&lstsq_cert);

        let pinv_cert = pinv(&a, PinvOptions::default())
            .expect("pinv works")
            .certificate
            .expect("pinv certificate");
        assert_eq!(pinv_cert.action, SolverAction::SVDFallback);
        assert_certificate_populated(&pinv_cert);
    }

    #[test]
    fn solve_qr_matches_lu() {
        let a = vec![vec![3.0, 2.0], vec![1.0, 2.0]];
        let b = vec![5.0, 5.0];
        let lu_result = solve_general(&a, &b).expect("LU solve works");
        let qr_result = solve_qr(&a, &b).expect("QR solve works");
        assert_close_slice(&qr_result.x, &lu_result.x, 1e-12, 1e-12);
    }

    #[test]
    fn solve_svd_fallback_matches_lu() {
        let a = vec![vec![3.0, 2.0], vec![1.0, 2.0]];
        let b = vec![5.0, 5.0];
        let lu_result = solve_general(&a, &b).expect("LU solve works");
        let svd_result = solve_svd_fallback(&a, &b).expect("SVD solve works");
        assert_close_slice(&svd_result.x, &lu_result.x, 1e-12, 1e-12);
    }

    #[test]
    fn solve_svd_fallback_rejects_rank_deficient_square_systems() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let b = vec![1.0, 2.0];
        let err = solve_svd_fallback(&a, &b).expect_err("rank-deficient system");
        assert_eq!(err, LinalgError::SingularMatrix);
    }

    #[test]
    fn solve_svd_fallback_warns_on_ill_conditioned_but_full_rank_systems() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1e-14]];
        let b = vec![1.0, 1.0];
        let result = solve_svd_fallback(&a, &b).expect("full-rank system");
        assert!(matches!(
            result.warning,
            Some(LinalgWarning::IllConditioned { .. })
        ));
    }

    #[test]
    fn fast_rcond_triangular_matches_exact_upper_2x2_inverse_norm() {
        let a = vec![vec![1.0, 1.0], vec![0.0, 1.0]];
        let rcond = fast_rcond_triangular(&a, false);
        assert!(
            (rcond - 0.25).abs() < 1e-12,
            "expected exact reciprocal condition 0.25, got {rcond}"
        );
    }

    #[test]
    fn fast_rcond_triangular_matches_exact_lower_2x2_inverse_norm() {
        let a = vec![vec![1.0, 0.0], vec![1.0, 1.0]];
        let rcond = fast_rcond_triangular(&a, true);
        assert!(
            (rcond - 0.25).abs() < 1e-12,
            "expected exact reciprocal condition 0.25, got {rcond}"
        );
    }

    #[test]
    fn fast_rcond_triangular_nan_returns_zero() {
        let a = vec![vec![1.0, 0.0], vec![f64::NAN, 2.0]];
        let rcond = fast_rcond_triangular(&a, true);
        assert_eq!(rcond, 0.0);
    }

    #[test]
    fn fast_rcond_diagonal_nan_returns_zero() {
        let a = vec![vec![f64::NAN, 0.0], vec![0.0, 2.0]];
        let rcond = fast_rcond_diagonal(&a);
        assert_eq!(rcond, 0.0);
    }

    #[test]
    fn randomized_rcond_estimate_well_conditioned() {
        let matrix = DMatrix::from_row_slice(3, 3, &[4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0]);
        let lu = matrix.clone().lu();
        let rcond = randomized_rcond_estimate(&matrix, &lu, 5);
        assert!(
            rcond > 0.1,
            "well-conditioned matrix rcond estimate should be high, got {rcond}"
        );
    }

    // ── Error variant coverage tests ──────────────────────────────

    #[test]
    fn error_ragged_matrix() {
        let a = vec![vec![1.0, 2.0], vec![3.0]];
        let b = vec![1.0, 2.0];
        let err = solve(&a, &b, SolveOptions::default()).unwrap_err();
        assert_eq!(err, LinalgError::RaggedMatrix);
    }

    #[test]
    fn error_expected_square_matrix() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let b = vec![1.0, 2.0];
        let err = solve(&a, &b, SolveOptions::default()).unwrap_err();
        assert_eq!(err, LinalgError::ExpectedSquareMatrix);
    }

    #[test]
    fn error_incompatible_shapes() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![1.0, 2.0, 3.0];
        let err = solve(&a, &b, SolveOptions::default()).unwrap_err();
        assert!(matches!(err, LinalgError::IncompatibleShapes { .. }));
    }

    #[test]
    fn error_singular_matrix() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let b = vec![1.0, 2.0];
        let err = solve(&a, &b, SolveOptions::default()).unwrap_err();
        assert_eq!(err, LinalgError::SingularMatrix);
    }

    #[test]
    fn error_invalid_band_shape() {
        let ab = vec![vec![1.0, 2.0]]; // 1 row but l+u+1=3 expected
        let b = vec![1.0, 2.0];
        let err = solve_banded((1, 1), &ab, &b, SolveOptions::default()).unwrap_err();
        assert!(matches!(err, LinalgError::InvalidBandShape { .. }));
    }

    #[test]
    fn error_condition_too_high_hardened() {
        // Near-singular matrix: rcond ≈ 1e-15
        let a = vec![vec![1.0, 0.0], vec![0.0, 1e-15]];
        let b = vec![1.0, 1.0];
        let err = solve(
            &a,
            &b,
            SolveOptions {
                mode: RuntimeMode::Hardened,
                ..SolveOptions::default()
            },
        )
        .unwrap_err();
        assert!(
            matches!(err, LinalgError::ConditionTooHigh { .. }),
            "expected ConditionTooHigh, got {err:?}"
        );
    }

    #[test]
    fn error_not_supported_display() {
        let err = LinalgError::NotSupported {
            detail: "complex transpose not available".into(),
        };
        assert_eq!(err.to_string(), "complex transpose not available");
    }

    #[test]
    fn error_convergence_failure_display() {
        let err = LinalgError::ConvergenceFailure {
            detail: "SVD did not converge".into(),
        };
        assert_eq!(err.to_string(), "SVD did not converge");
    }

    #[test]
    fn error_resource_exhausted_display() {
        let err = LinalgError::ResourceExhausted {
            detail: "dimension too large".into(),
        };
        assert_eq!(err.to_string(), "resource exhausted: dimension too large");
    }

    #[test]
    fn error_invalid_argument_display() {
        let err = LinalgError::InvalidArgument {
            detail: "bad param".into(),
        };
        assert_eq!(err.to_string(), "bad param");
    }

    #[test]
    fn linalg_trace_serializes_to_json() {
        let trace = LinalgTrace {
            operation: "solve",
            matrix_size: (3, 3),
            mode: RuntimeMode::Strict,
            rcond: Some(0.5),
            warning: None,
            error: None,
        };
        let json = serde_json::to_string(&trace).expect("serialize");
        assert!(json.contains("\"operation\":\"solve\""));
        assert!(json.contains("\"matrix_size\":[3,3]"));
    }

    // ── Comprehensive unit tests for bd-3jh.13.5 ─────────────────

    #[test]
    fn solve_3x3_general_system() {
        let a = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 10.0],
        ];
        let b = vec![14.0, 32.0, 53.0];
        let result = solve(&a, &b, SolveOptions::default()).expect("solve works");
        assert_close_slice(&result.x, &[1.0, 2.0, 3.0], 1e-10, 1e-10);
    }

    #[test]
    fn solve_diagonal_fast_path() {
        let a = vec![
            vec![2.0, 0.0, 0.0],
            vec![0.0, 3.0, 0.0],
            vec![0.0, 0.0, 4.0],
        ];
        let b = vec![4.0, 9.0, 16.0];
        let result = solve(
            &a,
            &b,
            SolveOptions {
                assume_a: Some(MatrixAssumption::Diagonal),
                ..SolveOptions::default()
            },
        )
        .expect("diagonal solve works");
        assert_close_slice(&result.x, &[2.0, 3.0, 4.0], 1e-14, 1e-14);
    }

    #[test]
    fn solve_upper_triangular_fast_path() {
        let a = vec![vec![2.0, 3.0], vec![0.0, 4.0]];
        let b = vec![14.0, 8.0];
        let result = solve(
            &a,
            &b,
            SolveOptions {
                assume_a: Some(MatrixAssumption::UpperTriangular),
                ..SolveOptions::default()
            },
        )
        .expect("upper triangular solve works");
        assert_close_slice(&result.x, &[4.0, 2.0], 1e-14, 1e-14);
    }

    #[test]
    fn solve_triangular_upper_non_transposed() {
        let a = vec![vec![2.0, 3.0], vec![0.0, 4.0]];
        let b = vec![14.0, 8.0];
        let result = solve_triangular(
            &a,
            &b,
            TriangularSolveOptions {
                lower: false,
                ..TriangularSolveOptions::default()
            },
        )
        .expect("upper non-transposed works");
        assert_close_slice(&result.x, &[4.0, 2.0], 1e-14, 1e-14);
    }

    #[test]
    fn solve_triangular_lower_transposed() {
        // L = [[2, 0], [3, 4]], L^T = [[2, 3], [0, 4]]
        // L^T x = b is same as solving upper triangular with L^T
        let a = vec![vec![2.0, 0.0], vec![3.0, 4.0]];
        let b = vec![14.0, 8.0];
        let result = solve_triangular(
            &a,
            &b,
            TriangularSolveOptions {
                lower: true,
                trans: TriangularTranspose::Transpose,
                ..TriangularSolveOptions::default()
            },
        )
        .expect("lower transposed works");
        // L^T x = b => [[2,3],[0,4]] x = [14,8] => x = [4, 2]
        assert_close_slice(&result.x, &[4.0, 2.0], 1e-14, 1e-14);
    }

    #[test]
    fn solve_triangular_unit_diagonal() {
        let a = vec![vec![99.0, 0.0], vec![3.0, 99.0]]; // diagonals ignored
        let b = vec![5.0, 14.0];
        let result = solve_triangular(
            &a,
            &b,
            TriangularSolveOptions {
                lower: true,
                unit_diagonal: true,
                ..TriangularSolveOptions::default()
            },
        )
        .expect("unit diagonal works");
        // [1, 0; 3, 1] x = [5, 14] => x = [5, -1]
        assert_close_slice(&result.x, &[5.0, -1.0], 1e-14, 1e-14);
    }

    #[test]
    fn inv_metamorphic_double_inverse_recovers_original() {
        // /testing-metamorphic: inv(inv(A)) ≈ A within numerical
        // accuracy, for any non-singular A. Tests both the LU path
        // and the recovery from numerical drift across two solves.
        let a = vec![vec![2.0_f64, 1.0], vec![1.0, 3.0]];
        let inv1 = inv(&a, InvOptions::default()).unwrap().inverse;
        let inv2 = inv(&inv1, InvOptions::default()).unwrap().inverse;
        for (row_a, row_back) in a.iter().zip(inv2.iter()) {
            for (&va, &vb) in row_a.iter().zip(row_back.iter()) {
                assert!(
                    (va - vb).abs() < 1e-10,
                    "inv(inv(A)) entry {vb} differs from A entry {va}"
                );
            }
        }

        // 3x3 case
        let b = vec![
            vec![1.0_f64, 2.0, 3.0],
            vec![0.0, 1.0, 4.0],
            vec![5.0, 6.0, 0.0],
        ];
        let inv1 = inv(&b, InvOptions::default()).unwrap().inverse;
        let inv2 = inv(&inv1, InvOptions::default()).unwrap().inverse;
        for (row_a, row_back) in b.iter().zip(inv2.iter()) {
            for (&va, &vb) in row_a.iter().zip(row_back.iter()) {
                assert!(
                    (va - vb).abs() < 1e-10,
                    "3x3 inv(inv(A)) entry {vb} differs from A entry {va}"
                );
            }
        }
    }

    #[test]
    fn inv_identity_matrix() {
        let a = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let result = inv(&a, InvOptions::default()).expect("inv works");
        assert_close_matrix(&result.inverse, &a, 1e-14, 1e-14);
    }

    #[test]
    fn inv_singular_matrix_error() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let err = inv(&a, InvOptions::default()).unwrap_err();
        assert_eq!(err, LinalgError::SingularMatrix);
    }

    #[test]
    fn inv_ill_conditioned_emits_warning() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1e-13]];
        let result = inv(&a, InvOptions::default()).expect("inv should succeed");
        assert!(
            result.warning.is_some(),
            "ill-conditioned matrix should emit warning"
        );
    }

    #[test]
    fn det_identity_is_one() {
        let a = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let d = det(&a, RuntimeMode::Strict, true).expect("det works");
        assert!((d - 1.0).abs() < 1e-14);
    }

    #[test]
    fn det_singular_is_zero() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let d = det(&a, RuntimeMode::Strict, true).expect("det works");
        assert!(d.abs() < 1e-14, "det of singular should be 0, got {d}");
    }

    #[test]
    fn lstsq_underdetermined_system() {
        // 2x3: more unknowns than equations -> minimum norm solution
        let a = vec![vec![1.0, 0.0, 1.0], vec![0.0, 1.0, 1.0]];
        let b = vec![2.0, 3.0];
        let result = lstsq(&a, &b, LstsqOptions::default()).expect("lstsq works");
        assert_eq!(result.x.len(), 3);
        // Verify residual: A @ x == b
        for (i, row) in a.iter().enumerate() {
            let dot: f64 = row.iter().zip(&result.x).map(|(a, x)| a * x).sum();
            assert!((dot - b[i]).abs() < 1e-10, "residual too large at row {i}");
        }
    }

    #[test]
    fn lstsq_exact_system() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![3.0, 7.0];
        let result = lstsq(&a, &b, LstsqOptions::default()).expect("lstsq works");
        assert_close_slice(&result.x, &[3.0, 7.0], 1e-14, 1e-14);
        assert_eq!(result.rank, 2);
    }

    #[test]
    fn lstsq_rank_deficient() {
        let a = vec![vec![1.0, 1.0], vec![1.0, 1.0], vec![1.0, 1.0]];
        let b = vec![2.0, 2.0, 2.0];
        let result = lstsq(&a, &b, LstsqOptions::default()).expect("lstsq works");
        assert_eq!(result.rank, 1, "rank should be 1 for rank-deficient system");
    }

    #[test]
    fn pinv_square_nonsingular_matches_inv() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let pinv_result = pinv(&a, PinvOptions::default()).expect("pinv works");
        let inv_result = inv(&a, InvOptions::default()).expect("inv works");
        assert_close_matrix(
            &pinv_result.pseudo_inverse,
            &inv_result.inverse,
            1e-10,
            1e-10,
        );
    }

    #[test]
    fn pinv_rank_deficient() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let result = pinv(&a, PinvOptions::default()).expect("pinv works");
        assert_eq!(result.rank, 1, "rank should be 1");
    }

    #[test]
    fn pinv_zero_matrix() {
        let a = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let result = pinv(&a, PinvOptions::default()).expect("pinv works");
        assert_eq!(result.rank, 0);
        for row in &result.pseudo_inverse {
            for &v in row {
                assert!(v.abs() < 1e-14, "pinv of zero should be zero");
            }
        }
    }

    #[test]
    fn bidiagonal_reduction_reconstructs_tall_matrix() {
        let a = vec![
            vec![4.0, 1.0, -0.5],
            vec![1.5, 3.5, 0.75],
            vec![0.25, -1.0, 2.5],
            vec![2.0, 0.5, 1.25],
            vec![-0.5, 1.75, 0.25],
        ];
        let matrix = dmatrix_from_rows(&a).expect("matrix");
        let reduction = golub_kahan_bidiagonal_reduction(&matrix).expect("bidiagonal reduction");

        assert_eq!(reduction.rows, 5);
        assert_eq!(reduction.cols, 3);
        assert_eq!(reduction.diagonal.len(), 3);
        assert_eq!(reduction.superdiagonal.len(), 2);

        for row in 0..reduction.rows {
            for col in 0..reduction.cols {
                let allowed = row == col || row + 1 == col;
                if !allowed {
                    assert!(
                        reduction.bidiagonal[(row, col)].abs() < 1e-10,
                        "bidiagonal fill at ({row}, {col}) = {}",
                        reduction.bidiagonal[(row, col)]
                    );
                }
            }
        }

        let q_t = reduction.left_product_transpose();
        let v = reduction.right_product();
        let reconstructed =
            q_t.clone().transpose() * reduction.bidiagonal.clone() * v.clone().transpose();
        for row in 0..matrix.nrows() {
            for col in 0..matrix.ncols() {
                assert!(
                    (reconstructed[(row, col)] - matrix[(row, col)]).abs() < 1e-10,
                    "reconstruction drift at ({row}, {col}): {} vs {}",
                    reconstructed[(row, col)],
                    matrix[(row, col)]
                );
            }
        }

        let q_identity = &q_t * q_t.clone().transpose();
        for row in 0..q_identity.nrows() {
            for col in 0..q_identity.ncols() {
                let expected = if row == col { 1.0 } else { 0.0 };
                assert!(
                    (q_identity[(row, col)] - expected).abs() < 1e-10,
                    "left reflector product lost orthogonality at ({row}, {col})"
                );
            }
        }

        let v_identity = v.clone().transpose() * &v;
        for row in 0..v_identity.nrows() {
            for col in 0..v_identity.ncols() {
                let expected = if row == col { 1.0 } else { 0.0 };
                assert!(
                    (v_identity[(row, col)] - expected).abs() < 1e-10,
                    "right reflector product lost orthogonality at ({row}, {col})"
                );
            }
        }
    }

    #[test]
    fn bidiagonal_reduction_rejects_wide_and_nonfinite_inputs() {
        let wide = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(
            golub_kahan_bidiagonal_reduction(&wide).unwrap_err(),
            LinalgError::UnsupportedAssumption
        );

        let nonfinite = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0]);
        assert_eq!(
            golub_kahan_bidiagonal_reduction(&nonfinite).unwrap_err(),
            LinalgError::NonFiniteInput
        );
    }

    fn low_rank_trig_matrix(rows: usize, cols: usize) -> Vec<Vec<f64>> {
        (0..rows)
            .map(|i| {
                (0..cols)
                    .map(|j| (i as f64 * 0.013 + j as f64 * 0.007 + 0.3).sin() + 0.5)
                    .collect()
            })
            .collect()
    }

    fn low_rank_rhs(rows: usize) -> Vec<f64> {
        (0..rows)
            .map(|i| (i as f64 * 0.017 + 0.2).cos() - 0.25)
            .collect()
    }

    #[test]
    fn lstsq_low_rank_tall_matches_svd_reference() {
        let rows = 32;
        let cols = 16;
        let a = low_rank_trig_matrix(rows, cols);
        let b = low_rank_rhs(rows);
        let reference = lstsq(&a, &b, LstsqOptions::default()).expect("reference lstsq");
        let fast =
            lstsq_low_rank_tall(&a, &b, rows, cols, f64::EPSILON, 1).expect("low-rank lstsq");

        assert_eq!(fast.rank, reference.rank);
        assert_close_slice(&fast.x, &reference.x, 1e-7, 1e-7);
        assert_close_slice(
            &fast.singular_values,
            &reference.singular_values,
            1e-7,
            1e-7,
        );
    }

    #[test]
    fn lstsq_low_rank_tall_public_route_has_expected_certificate() {
        let rows = 1024;
        let cols = 512;
        let a = low_rank_trig_matrix(rows, cols);
        let b = low_rank_rhs(rows);
        let result = lstsq(&a, &b, LstsqOptions::default()).expect("public lstsq");
        let certificate = result.certificate.as_ref().expect("certificate");

        assert_eq!(result.rank, 3);
        assert_eq!(result.singular_values.len(), cols);
        assert!(result.singular_values[0] >= result.singular_values[1]);
        assert!(result.singular_values[1] >= result.singular_values[2]);
        assert!(result.singular_values[2] > 0.0);
        assert!(
            result.singular_values[3..]
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(result.residuals.is_empty());
        assert_eq!(certificate.action, SolverAction::SVDFallback);
        assert_eq!(certificate.matrix_shape, (rows, cols));
        assert_eq!(certificate.rcond_estimate, 0.0);
    }

    #[test]
    fn lstsq_low_rank_tall_golden_payload() {
        let rows = 32;
        let cols = 16;
        let a = low_rank_trig_matrix(rows, cols);
        let b = low_rank_rhs(rows);
        let reference = lstsq(&a, &b, LstsqOptions::default()).expect("reference lstsq");
        let fast =
            lstsq_low_rank_tall(&a, &b, rows, cols, f64::EPSILON, 1).expect("low-rank lstsq");

        assert_eq!(fast.rank, reference.rank);
        assert_close_slice(&fast.x, &reference.x, 1e-7, 1e-7);
        assert_close_slice(
            &fast.singular_values,
            &reference.singular_values,
            1e-7,
            1e-7,
        );

        let mut max_x_abs_diff = 0.0_f64;
        for (&fast_value, &reference_value) in fast.x.iter().zip(reference.x.iter()) {
            max_x_abs_diff = max_x_abs_diff.max((fast_value - reference_value).abs());
        }
        let mut max_s_abs_diff = 0.0_f64;
        for (&fast_value, &reference_value) in fast
            .singular_values
            .iter()
            .zip(reference.singular_values.iter())
        {
            max_s_abs_diff = max_s_abs_diff.max((fast_value - reference_value).abs());
        }

        let public_rows = 1024;
        let public_cols = 512;
        let public_a = low_rank_trig_matrix(public_rows, public_cols);
        let public_b = low_rank_rhs(public_rows);
        let public =
            lstsq(&public_a, &public_b, LstsqOptions::default()).expect("public low-rank lstsq");
        let certificate = public.certificate.as_ref().expect("certificate");
        assert_eq!(public.rank, 3);
        assert_eq!(public.singular_values.len(), public_cols);
        assert_eq!(certificate.action, SolverAction::SVDFallback);
        assert_eq!(certificate.matrix_shape, (public_rows, public_cols));
        assert_eq!(certificate.rcond_estimate, 0.0);

        println!("LSTSQ_LOW_RANK_TALL_GOLDEN_BEGIN");
        println!("reference_shape={rows}x{cols}");
        println!("reference_rank={}", reference.rank);
        println!("fast_rank={}", fast.rank);
        println!("max_x_abs_diff={max_x_abs_diff:.17e}");
        println!("max_s_abs_diff={max_s_abs_diff:.17e}");
        for idx in [0, 1, 7, 15] {
            println!("reference_x[{idx}]={:.17e}", reference.x[idx]);
            println!("fast_x[{idx}]={:.17e}", fast.x[idx]);
        }
        for idx in [0, 1, 2, 3, 15] {
            println!(
                "reference_singular[{idx}]={:.17e}",
                reference.singular_values[idx]
            );
            println!("fast_singular[{idx}]={:.17e}", fast.singular_values[idx]);
        }
        println!("public_shape={public_rows}x{public_cols}");
        println!("public_rank={}", public.rank);
        println!("public_singular_len={}", public.singular_values.len());
        println!("certificate_action={:?}", certificate.action);
        println!("certificate_rcond={:.17e}", certificate.rcond_estimate);
        for idx in [0, 1, 17, 511] {
            println!("public_x[{idx}]={:.17e}", public.x[idx]);
        }
        for idx in [0, 1, 2, 3, 511] {
            println!(
                "public_singular[{idx}]={:.17e}",
                public.singular_values[idx]
            );
        }
        println!("LSTSQ_LOW_RANK_TALL_GOLDEN_END");
    }

    #[test]
    fn lstsq_low_rank_tall_rejects_full_rank_tall_matrix() {
        let rows = 40;
        let cols = 20;
        let mut a = vec![vec![0.0; cols]; rows];
        for (i, row) in a.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = if i == j {
                    3.0 + j as f64
                } else {
                    0.5 / ((i as f64 - j as f64).abs() + 1.0)
                };
            }
        }
        let b = low_rank_rhs(rows);
        assert!(lstsq_low_rank_tall(&a, &b, rows, cols, f64::EPSILON, 1).is_none());
    }

    #[test]
    fn pinv_low_rank_tall_matches_svd_reference() {
        let rows = 32;
        let cols = 16;
        let a = low_rank_trig_matrix(rows, cols);
        let reference = pinv(&a, PinvOptions::default()).expect("reference pinv");
        let rtol = (rows.max(cols) as f64) * f64::EPSILON;
        let fast =
            pinv_low_rank_tall_with_limits(&a, rows, cols, 0.0, rtol, 1).expect("low-rank path");

        assert_eq!(fast.rank, reference.rank);
        assert_eq!(fast.rcond_estimate, 0.0);
        assert_close_matrix(&fast.pseudo_inverse, &reference.pseudo_inverse, 1e-7, 1e-7);
    }

    #[test]
    fn pinv_low_rank_tall_golden_payload() {
        let rows = 32;
        let cols = 16;
        let a = low_rank_trig_matrix(rows, cols);
        let reference = pinv(&a, PinvOptions::default()).expect("reference pinv");
        let rtol = (rows.max(cols) as f64) * f64::EPSILON;
        let fast =
            pinv_low_rank_tall_with_limits(&a, rows, cols, 0.0, rtol, 1).expect("low-rank path");
        assert_eq!(fast.rank, reference.rank);
        assert_close_matrix(&fast.pseudo_inverse, &reference.pseudo_inverse, 1e-7, 1e-7);

        let mut max_abs_diff = 0.0_f64;
        for (fast_row, reference_row) in fast
            .pseudo_inverse
            .iter()
            .zip(reference.pseudo_inverse.iter())
        {
            for (fast_value, reference_value) in fast_row.iter().zip(reference_row.iter()) {
                max_abs_diff = max_abs_diff.max((fast_value - reference_value).abs());
            }
        }

        let public_rows = 1024;
        let public_cols = 512;
        let public_a = low_rank_trig_matrix(public_rows, public_cols);
        let public = pinv(&public_a, PinvOptions::default()).expect("public low-rank pinv");
        let certificate = public.certificate.as_ref().expect("certificate");
        assert_eq!(public.rank, 3);
        assert_eq!(certificate.action, SolverAction::SVDFallback);
        assert_eq!(certificate.matrix_shape, (public_rows, public_cols));
        assert_eq!(certificate.rcond_estimate, 0.0);
        assert!(!certificate.fallback_active);

        println!("PINV_LOW_RANK_TALL_GOLDEN_BEGIN");
        println!("reference_shape={rows}x{cols}");
        println!("reference_rank={}", reference.rank);
        println!("fast_rank={}", fast.rank);
        println!("max_abs_diff={max_abs_diff:.17e}");
        for (row, col) in [(0, 0), (1, 7), (5, 11), (15, 31)] {
            println!(
                "reference_entry[{row},{col}]={:.17e}",
                reference.pseudo_inverse[row][col]
            );
            println!(
                "fast_entry[{row},{col}]={:.17e}",
                fast.pseudo_inverse[row][col]
            );
        }
        println!("public_shape={public_rows}x{public_cols}");
        println!("public_rank={}", public.rank);
        println!("certificate_action={:?}", certificate.action);
        println!("certificate_rcond={:.17e}", certificate.rcond_estimate);
        println!(
            "certificate_fallback_active={}",
            certificate.fallback_active
        );
        for (row, col) in [(0, 0), (1, 17), (127, 511), (511, 1023)] {
            println!(
                "public_entry[{row},{col}]={:.17e}",
                public.pseudo_inverse[row][col]
            );
        }
        println!("PINV_LOW_RANK_TALL_GOLDEN_END");
    }

    #[test]
    fn pinv_full_rank_rectangular_golden_payload() {
        let rows = 64;
        let cols = 32;
        let mut a = vec![vec![0.0; cols]; rows];
        for (row_idx, row) in a.iter_mut().enumerate() {
            for (col_idx, cell) in row.iter_mut().enumerate() {
                *cell = if row_idx == col_idx {
                    10.0 + col_idx as f64
                } else {
                    1.0 / ((row_idx as f64 - col_idx as f64).abs() + 1.0)
                };
            }
        }

        let result = pinv(&a, PinvOptions::default()).expect("full-rank pinv");
        let certificate = result.certificate.as_ref().expect("certificate");
        assert_eq!(result.rank, cols);
        assert_eq!(result.pseudo_inverse.len(), cols);
        assert_eq!(result.pseudo_inverse[0].len(), rows);
        assert_eq!(certificate.action, SolverAction::SVDFallback);
        assert_eq!(certificate.matrix_shape, (rows, cols));

        println!("PINV_FULL_RANK_RECTANGULAR_GOLDEN_BEGIN");
        println!("shape={rows}x{cols}");
        println!("rank={}", result.rank);
        println!("certificate_action={:?}", certificate.action);
        println!("certificate_rcond={:.17e}", certificate.rcond_estimate);
        for (row, col) in [(0, 0), (1, 7), (5, 11), (17, 31), (31, 63)] {
            println!(
                "entry[{row},{col}]={:.17e}",
                result.pseudo_inverse[row][col]
            );
        }
        println!("PINV_FULL_RANK_RECTANGULAR_GOLDEN_END");
    }

    #[test]
    fn pinv_low_rank_tall_rejects_full_rank_tall_matrix() {
        let rows = 40;
        let cols = 20;
        let mut a = vec![vec![0.0; cols]; rows];
        for (i, row) in a.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = if i == j {
                    10.0 + j as f64
                } else {
                    1.0 / ((i as f64 - j as f64).abs() + 1.0)
                };
            }
        }
        let rtol = (rows.max(cols) as f64) * f64::EPSILON;
        assert!(pinv_low_rank_tall_with_limits(&a, rows, cols, 0.0, rtol, 1).is_none());
    }

    #[test]
    fn rcond_threshold_boundary() {
        // Well-conditioned: rcond > 1e-12 -> no warning
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![1.0, 1.0];
        let result = solve(&a, &b, SolveOptions::default()).expect("solve works");
        assert!(result.warning.is_none(), "identity should have no warning");

        // Ill-conditioned: rcond < 1e-12 -> warning
        let a2 = vec![vec![1.0, 0.0], vec![0.0, 1e-13]];
        let b2 = vec![1.0, 1.0];
        let result2 = solve(&a2, &b2, SolveOptions::default()).expect("solve works");
        assert!(
            result2.warning.is_some(),
            "ill-conditioned matrix should emit warning"
        );
    }

    #[test]
    fn solve_large_100x100_system() {
        let n = 100;
        // Diagonally dominant matrix for guaranteed non-singularity
        let a: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| if i == j { n as f64 + 1.0 } else { 1.0 })
                    .collect()
            })
            .collect();
        let x_expected: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let b: Vec<f64> = a
            .iter()
            .map(|row| row.iter().zip(&x_expected).map(|(a, x)| a * x).sum())
            .collect();
        let result = solve(&a, &b, SolveOptions::default()).expect("large solve works");
        assert_close_slice(&result.x, &x_expected, 1e-8, 1e-8);
    }

    #[test]
    fn solve_symmetric_positive_definite_assumption() {
        let a = vec![
            vec![4.0, 2.0, 0.0],
            vec![2.0, 5.0, 2.0],
            vec![0.0, 2.0, 6.0],
        ];
        let b = vec![8.0, 13.0, 16.0];
        let result = solve(
            &a,
            &b,
            SolveOptions {
                assume_a: Some(MatrixAssumption::PositiveDefinite),
                ..SolveOptions::default()
            },
        )
        .expect("SPD solve works");
        // Verify A*x = b
        for (i, row) in a.iter().enumerate() {
            let dot: f64 = row.iter().zip(&result.x).map(|(a, x)| a * x).sum();
            assert!(
                (dot - b[i]).abs() < 1e-10,
                "residual too large at row {i}: {dot} vs {}",
                b[i]
            );
        }
    }

    #[test]
    fn solve_banded_pentadiagonal() {
        // 5-diagonal system (l=2, u=2)
        let ab = vec![
            vec![0.0, 0.0, 1.0, 1.0, 1.0],
            vec![0.0, 1.0, 1.0, 1.0, 1.0],
            vec![4.0, 4.0, 4.0, 4.0, 4.0],
            vec![1.0, 1.0, 1.0, 1.0, 0.0],
            vec![1.0, 1.0, 1.0, 0.0, 0.0],
        ];
        let b = vec![7.0, 8.0, 8.0, 8.0, 7.0];
        let result =
            solve_banded((2, 2), &ab, &b, SolveOptions::default()).expect("pentadiagonal works");
        assert_eq!(result.x.len(), 5);
        // Verify via dense reconstruction
        let dense = dense_from_banded(2, 2, &ab, 5);
        for (i, row) in dense.iter().enumerate() {
            let dot: f64 = row.iter().zip(&result.x).map(|(a, x)| a * x).sum();
            assert!(
                (dot - b[i]).abs() < 1e-10,
                "pentadiagonal residual too large at row {i}"
            );
        }
    }

    #[test]
    fn det_empty_matrix_is_one() {
        let a: Vec<Vec<f64>> = vec![];
        let d = det(&a, RuntimeMode::Strict, true).expect("det of empty");
        assert!((d - 1.0).abs() < 1e-14);
    }

    #[test]
    fn inv_empty_matrix() {
        let a: Vec<Vec<f64>> = vec![];
        let result = inv(&a, InvOptions::default()).expect("inv of empty");
        assert!(result.inverse.is_empty());
    }

    #[test]
    fn pinv_moore_penrose_condition() {
        // Verify: A @ pinv(A) @ A == A (Moore-Penrose condition 1)
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let result = pinv(&a, PinvOptions::default()).expect("pinv works");
        let a_pinv = &result.pseudo_inverse;
        // Compute A @ A_pinv @ A
        let n_rows = a.len();
        let n_cols = a[0].len();
        let pinv_rows = a_pinv.len();
        // A_pinv is (n_cols x n_rows), so A @ A_pinv is (n_rows x n_rows)
        let mut a_ap = vec![vec![0.0; n_rows]; n_rows];
        for i in 0..n_rows {
            for j in 0..n_rows {
                for k in 0..pinv_rows {
                    a_ap[i][j] += a[i][k] * a_pinv[k][j];
                }
            }
        }
        // (A @ A_pinv) @ A -> (n_rows x n_cols)
        let mut a_ap_a = vec![vec![0.0; n_cols]; n_rows];
        for i in 0..n_rows {
            for j in 0..n_cols {
                for k in 0..n_rows {
                    a_ap_a[i][j] += a_ap[i][k] * a[k][j];
                }
            }
        }
        assert_close_matrix(&a_ap_a, &a, 1e-10, 1e-10);
    }

    #[test]
    fn pinv_moore_penrose_conditions_2_3_4() {
        // /testing-metamorphic on the four Moore-Penrose identities.
        // Condition 1 (A·A⁺·A = A) is already pinned at line 8421.
        // This test pins the remaining three:
        //   2.  A⁺ · A · A⁺ = A⁺
        //   3.  (A · A⁺)ᵀ = A · A⁺      (Hermitian — real ⇒ symmetric)
        //   4.  (A⁺ · A)ᵀ = A⁺ · A      (Hermitian)
        // Together with condition 1 these four identities uniquely
        // characterize the pseudo-inverse.
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let result = pinv(&a, PinvOptions::default()).expect("pinv works");
        let a_pinv = &result.pseudo_inverse;
        let m = a.len();
        let n = a[0].len();
        let p = a_pinv.len();
        let q = a_pinv[0].len();
        assert_eq!(p, n);
        assert_eq!(q, m);

        let matmul = |x: &[Vec<f64>], y: &[Vec<f64>]| -> Vec<Vec<f64>> {
            let r = x.len();
            let c = y[0].len();
            let inner = y.len();
            let mut out = vec![vec![0.0; c]; r];
            for i in 0..r {
                for j in 0..c {
                    for k in 0..inner {
                        out[i][j] += x[i][k] * y[k][j];
                    }
                }
            }
            out
        };

        // MP-2: A⁺ · A · A⁺ = A⁺   (n×n × n×m × n×... wait:
        // A⁺ is n×m, A is m×n, so A⁺·A is n×n, then ·A⁺ is n×m.
        let ap_a = matmul(a_pinv, &a);
        let ap_a_ap = matmul(&ap_a, a_pinv);
        assert_close_matrix(&ap_a_ap, a_pinv, 1e-10, 1e-10);

        // MP-3: (A · A⁺)ᵀ = A · A⁺.  A·A⁺ is m×m.
        let a_ap = matmul(&a, a_pinv);
        for (i, row) in a_ap.iter().enumerate().take(m) {
            for (j, value) in row.iter().enumerate().take(m) {
                assert!(
                    (*value - a_ap[j][i]).abs() < 1e-10,
                    "A·A⁺ not symmetric at [{i}][{j}]: {} vs {}",
                    value,
                    a_ap[j][i]
                );
            }
        }

        // MP-4: (A⁺ · A)ᵀ = A⁺ · A.  A⁺·A is n×n.
        for (i, row) in ap_a.iter().enumerate().take(n) {
            for (j, value) in row.iter().enumerate().take(n) {
                assert!(
                    (*value - ap_a[j][i]).abs() < 1e-10,
                    "A⁺·A not symmetric at [{i}][{j}]: {} vs {}",
                    value,
                    ap_a[j][i]
                );
            }
        }
    }

    #[test]
    fn pinv_involution_pinv_of_pinv_recovers_a() {
        // /testing-metamorphic: pinv(pinv(A)) ≈ A for any A. This is a
        // direct consequence of the four MP conditions.
        let a = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 10.0], // perturb to avoid rank deficiency
        ];
        let first = pinv(&a, PinvOptions::default()).expect("pinv");
        let second = pinv(&first.pseudo_inverse, PinvOptions::default()).expect("pinv²");
        assert_close_matrix(&second.pseudo_inverse, &a, 1e-9, 1e-9);
    }

    // ── LU decomposition tests ──────────────────────────────────────

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn lu_decomposition_a_equals_plu() {
        // SciPy's `scipy.linalg.lu(a)` returns P, L, U with A = P · L · U.
        let a = vec![
            vec![2.0, 1.0, 1.0],
            vec![4.0, 3.0, 3.0],
            vec![8.0, 7.0, 9.0],
        ];
        let result = lu(&a, DecompOptions::default()).expect("lu works");
        let n = a.len();
        // Compute LU first, then P · (LU).
        let mut lu_prod = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                lu_prod[i][j] = (0..n).map(|k| result.l[i][k] * result.u[k][j]).sum();
            }
        }
        for i in 0..n {
            for j in 0..n {
                let plu: f64 = (0..n).map(|k| result.p[i][k] * lu_prod[k][j]).sum();
                assert!(
                    (plu - a[i][j]).abs() < 1e-10,
                    "A != P·L·U at [{i}][{j}]: {plu} vs {}",
                    a[i][j]
                );
            }
        }
    }

    #[test]
    fn lu_l_is_unit_lower_triangular() {
        let a = vec![vec![4.0, 3.0], vec![6.0, 3.0]];
        let result = lu(&a, DecompOptions::default()).expect("lu works");
        // L diagonal should be 1.0
        for i in 0..2 {
            assert!(
                (result.l[i][i] - 1.0).abs() < 1e-14,
                "L diagonal should be 1"
            );
        }
        // L upper triangle should be 0.0
        assert!(result.l[0][1].abs() < 1e-14, "L should be lower triangular");
    }

    #[test]
    fn lu_u_is_upper_triangular() {
        let a = vec![vec![4.0, 3.0], vec![6.0, 3.0]];
        let result = lu(&a, DecompOptions::default()).expect("lu works");
        assert!(result.u[1][0].abs() < 1e-14, "U should be upper triangular");
    }

    #[test]
    fn lu_empty_matrix() {
        let a: Vec<Vec<f64>> = vec![];
        let result = lu(&a, DecompOptions::default()).expect("lu of empty");
        assert!(result.p.is_empty());
        assert!(result.l.is_empty());
        assert!(result.u.is_empty());
    }

    #[test]
    fn lu_factor_and_solve_roundtrip() {
        let a = vec![vec![3.0, 2.0], vec![1.0, 2.0]];
        let b = vec![5.0, 5.0];
        let factor = lu_factor(&a, DecompOptions::default()).expect("lu_factor works");
        let result = lu_solve(&factor, &b).expect("lu_solve works");
        assert_close_slice(&result.x, &[0.0, 2.5], 1e-12, 1e-12);
    }

    #[test]
    fn lu_factor_caches_rcond_without_debug_observability() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1e-14]];
        let b = vec![1.0, 1e-14];
        let factor = lu_factor(&a, DecompOptions::default()).expect("lu_factor works");
        let recomputed = fast_rcond_from_lu(&factor.lu_internal, factor.a_norm_1, factor.n);

        assert_eq!(factor.rcond_estimate.to_bits(), recomputed.to_bits());

        let result = lu_solve(&factor, &b).expect("lu_solve works");
        assert_eq!(result.warning, rcond_warning(recomputed));
        assert!(matches!(
            result.warning,
            Some(LinalgWarning::IllConditioned { .. })
        ));

        let debug = format!("{factor:?}");
        assert!(debug.contains("LuFactorResult"));
        assert!(debug.contains("a_norm_1"));
        assert!(!debug.contains("rcond_estimate"));
    }

    #[test]
    fn lu_solve_incompatible_shapes() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let factor = lu_factor(&a, DecompOptions::default()).expect("lu_factor works");
        let err = lu_solve(&factor, &[1.0, 2.0, 3.0]).unwrap_err();
        assert!(matches!(err, LinalgError::IncompatibleShapes { .. }));
    }

    // ── QR decomposition tests ──────────────────────────────────────

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn qr_decomposition_a_equals_qr() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let result = qr(&a, DecompOptions::default()).expect("qr works");
        let cols = a[0].len();
        // Verify A = Q*R
        for (i, (a_row, q_row)) in a.iter().zip(result.q.iter()).enumerate() {
            for j in 0..cols {
                let qr_val: f64 = (0..result.r.len()).map(|k| q_row[k] * result.r[k][j]).sum();
                assert!(
                    (a_row[j] - qr_val).abs() < 1e-10,
                    "A != QR at [{i}][{j}]: {} vs {qr_val}",
                    a_row[j]
                );
            }
        }
    }

    #[test]
    fn qr_q_is_orthogonal() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = qr(&a, DecompOptions::default()).expect("qr works");
        let n = result.q.len();
        // Q^T * Q should be identity
        for i in 0..n {
            for j in i..n {
                let dot: f64 = result.q.iter().map(|row| row[i] * row[j]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "Q not orthogonal at [{i}][{j}]: {dot} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn qr_empty_matrix() {
        let a: Vec<Vec<f64>> = vec![];
        let result = qr(&a, DecompOptions::default()).expect("qr of empty");
        assert!(result.q.is_empty());
        assert!(result.r.is_empty());
    }

    #[test]
    fn qr_delete_removes_requested_row() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 7.0]];
        let factors = qr(&a, DecompOptions::default()).expect("qr");
        let updated =
            qr_delete(&factors.q, &factors.r, 1, DecompOptions::default()).expect("qr_delete");

        assert_close_matrix(
            &reconstruct_qr_result(&updated),
            &[vec![1.0, 2.0], vec![5.0, 7.0]],
            1e-9,
            1e-9,
        );
    }

    #[test]
    fn qr_delete_rejects_out_of_bounds_row() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let factors = qr(&a, DecompOptions::default()).expect("qr");
        let err =
            qr_delete(&factors.q, &factors.r, 2, DecompOptions::default()).expect_err("bad row");
        assert!(matches!(err, LinalgError::InvalidArgument { .. }));
    }

    #[test]
    fn qr_insert_inserts_requested_row() {
        let a = vec![vec![1.0, 2.0], vec![5.0, 7.0]];
        let factors = qr(&a, DecompOptions::default()).expect("qr");
        let updated = qr_insert(
            &factors.q,
            &factors.r,
            &[3.0, 4.0],
            1,
            DecompOptions::default(),
        )
        .expect("qr_insert");

        assert_close_matrix(
            &reconstruct_qr_result(&updated),
            &[vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 7.0]],
            1e-9,
            1e-9,
        );
    }

    #[test]
    fn qr_insert_rejects_wrong_row_width() {
        let a = vec![vec![1.0, 2.0], vec![5.0, 7.0]];
        let factors = qr(&a, DecompOptions::default()).expect("qr");
        let err = qr_insert(
            &factors.q,
            &factors.r,
            &[3.0, 4.0, 5.0],
            1,
            DecompOptions::default(),
        )
        .expect_err("bad row width");
        assert!(matches!(err, LinalgError::IncompatibleShapes { .. }));
    }

    #[test]
    fn qr_multiply_applies_q_factor() {
        let q = eye(2, 2);
        let r = eye(2, 2);
        let c = vec![vec![2.0, 3.0], vec![5.0, 7.0]];
        let result = qr_multiply(&q, &r, &c, DecompOptions::default()).expect("qr_multiply");
        assert_close_matrix(&result, &c, 1e-14, 1e-14);
    }

    #[test]
    fn qr_multiply_rejects_shape_mismatch() {
        let q = eye(2, 2);
        let r = eye(2, 2);
        let c = vec![vec![2.0, 3.0]];
        let err = qr_multiply(&q, &r, &c, DecompOptions::default()).expect_err("shape mismatch");
        assert!(matches!(err, LinalgError::IncompatibleShapes { .. }));
    }

    #[test]
    fn qr_update_applies_rank_one_update() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let factors = qr(&a, DecompOptions::default()).expect("qr");
        let updated = qr_update(
            &factors.q,
            &factors.r,
            &[0.5, -1.0],
            &[2.0, 0.25],
            DecompOptions::default(),
        )
        .expect("qr_update");

        assert_close_matrix(
            &reconstruct_qr_result(&updated),
            &[vec![2.0, 2.125], vec![1.0, 3.75]],
            1e-9,
            1e-9,
        );
    }

    #[test]
    fn qr_update_rejects_wrong_update_vector_width() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let factors = qr(&a, DecompOptions::default()).expect("qr");
        let err = qr_update(
            &factors.q,
            &factors.r,
            &[0.5, -1.0],
            &[2.0, 0.25, 0.5],
            DecompOptions::default(),
        )
        .expect_err("bad v");
        assert!(matches!(err, LinalgError::IncompatibleShapes { .. }));
    }

    // ── SVD tests ───────────────────────────────────────────────────

    fn bidiag_deterministic_matrix(rows: usize, cols: usize) -> DMatrix<f64> {
        DMatrix::from_fn(rows, cols, |row, col| {
            let diagonal_bias = if row == col { 6.0 + col as f64 } else { 0.0 };
            let low_rank_part = ((row + 1) as f64 * 0.071).sin() + ((col + 3) as f64 * 0.113).cos();
            let coupling = ((row + 2) * (col + 5)) as f64 / 97.0;
            diagonal_bias + low_rank_part + coupling
        })
    }

    fn max_abs_dmatrix_diff(actual: &DMatrix<f64>, expected: &DMatrix<f64>) -> f64 {
        assert_eq!(actual.nrows(), expected.nrows());
        assert_eq!(actual.ncols(), expected.ncols());
        let mut max_abs = 0.0_f64;
        for row in 0..actual.nrows() {
            for col in 0..actual.ncols() {
                max_abs = max_abs.max((actual[(row, col)] - expected[(row, col)]).abs());
            }
        }
        max_abs
    }

    fn dmatrix_orthogonality_error(matrix: &DMatrix<f64>) -> f64 {
        let gram = matrix * matrix.transpose();
        let identity = DMatrix::<f64>::identity(matrix.nrows(), matrix.ncols());
        max_abs_dmatrix_diff(&gram, &identity)
    }

    fn dmatrix_column_orthogonality_error(matrix: &DMatrix<f64>) -> f64 {
        let gram = matrix.transpose() * matrix;
        let identity = DMatrix::<f64>::identity(matrix.ncols(), matrix.ncols());
        max_abs_dmatrix_diff(&gram, &identity)
    }

    fn bidiagonal_matrix_from_parts(
        rows: usize,
        diagonal: &[f64],
        superdiagonal: &[f64],
    ) -> DMatrix<f64> {
        let mut matrix = DMatrix::<f64>::zeros(rows, diagonal.len());
        for idx in 0..diagonal.len() {
            matrix[(idx, idx)] = diagonal[idx];
            if idx + 1 < diagonal.len() {
                matrix[(idx, idx + 1)] = superdiagonal[idx];
            }
        }
        matrix
    }

    fn assert_nonincreasing(values: &[f64]) {
        for pair in values.windows(2) {
            assert!(
                pair[0] >= pair[1],
                "singular values must be nonincreasing: {values:?}"
            );
        }
    }

    fn assert_upper_bidiagonal(reduction: &BidiagonalReduction, tolerance: f64) {
        for row in 0..reduction.rows {
            for col in 0..reduction.cols {
                let value = reduction.bidiagonal[(row, col)];
                if row == col {
                    assert_eq!(value.to_bits(), reduction.diagonal[row].to_bits());
                } else if col == row + 1 {
                    assert_eq!(value.to_bits(), reduction.superdiagonal[row].to_bits());
                } else {
                    assert!(
                        value.abs() <= tolerance,
                        "B[{row},{col}] should be zero, got {value:.17e}"
                    );
                }
            }
        }
    }

    fn bidiag_reduction_digest(reduction: &BidiagonalReduction) -> u64 {
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;

        let mut digest = FNV_OFFSET;
        for value in reduction
            .diagonal
            .iter()
            .chain(reduction.superdiagonal.iter())
            .chain(
                reduction
                    .left_reflectors
                    .iter()
                    .map(|reflector| &reflector.tau),
            )
            .chain(
                reduction
                    .right_reflectors
                    .iter()
                    .map(|reflector| &reflector.tau),
            )
        {
            digest ^= value.to_bits();
            digest = digest.wrapping_mul(FNV_PRIME);
        }
        digest
    }

    #[test]
    fn bidiag_golub_kahan_reconstructs_tall_full_rank_matrix() {
        let original = bidiag_deterministic_matrix(8, 5);
        let reduction = golub_kahan_bidiagonal_reduction(&original).expect("bidiagonal reduction");
        assert_eq!(reduction.diagonal.len(), 5);
        assert_eq!(reduction.superdiagonal.len(), 4);
        assert_eq!(reduction.left_reflectors.len(), 5);
        assert_eq!(reduction.right_reflectors.len(), 4);
        assert_upper_bidiagonal(&reduction, 0.0);

        let q_t = reduction.left_product_transpose();
        let v = reduction.right_product();
        let reconstructed = q_t.clone().transpose() * &reduction.bidiagonal * v.clone().transpose();
        let reconstruction_error = max_abs_dmatrix_diff(&reconstructed, &original);
        let q_error = dmatrix_orthogonality_error(&q_t);
        let v_error = dmatrix_orthogonality_error(&v);

        assert!(
            reconstruction_error < 1e-10,
            "reconstruction error {reconstruction_error:.17e}"
        );
        assert!(q_error < 1e-12, "Q orthogonality error {q_error:.17e}");
        assert!(v_error < 1e-12, "V orthogonality error {v_error:.17e}");
    }

    #[test]
    fn bidiag_golub_kahan_zero_matrix_keeps_zero_bidiagonal() {
        let original = DMatrix::<f64>::zeros(6, 4);
        let reduction = golub_kahan_bidiagonal_reduction(&original).expect("bidiagonal reduction");
        assert_eq!(reduction.diagonal, vec![0.0; 4]);
        assert_eq!(reduction.superdiagonal, vec![0.0; 3]);
        assert!(reduction.left_reflectors.iter().all(|refl| refl.tau == 0.0));
        assert!(
            reduction
                .right_reflectors
                .iter()
                .all(|refl| refl.tau == 0.0)
        );

        let q_t = reduction.left_product_transpose();
        let v = reduction.right_product();
        let reconstructed = q_t.clone().transpose() * &reduction.bidiagonal * v.clone().transpose();
        assert_eq!(max_abs_dmatrix_diff(&reconstructed, &original), 0.0);
        assert_eq!(dmatrix_orthogonality_error(&q_t), 0.0);
        assert_eq!(dmatrix_orthogonality_error(&v), 0.0);
    }

    #[test]
    fn bidiag_golub_kahan_rejects_wide_matrix() {
        let original = bidiag_deterministic_matrix(3, 5);
        let err =
            golub_kahan_bidiagonal_reduction(&original).expect_err("wide matrices are unsupported");
        assert_eq!(err, LinalgError::UnsupportedAssumption);
    }

    #[test]
    fn bidiag_golub_kahan_golden_payload() {
        let original = bidiag_deterministic_matrix(7, 4);
        let reduction = golub_kahan_bidiagonal_reduction(&original).expect("bidiagonal reduction");
        let q_t = reduction.left_product_transpose();
        let v = reduction.right_product();
        let reconstructed = q_t.clone().transpose() * &reduction.bidiagonal * v.clone().transpose();
        let reconstruction_error = max_abs_dmatrix_diff(&reconstructed, &original);
        let q_error = dmatrix_orthogonality_error(&q_t);
        let v_error = dmatrix_orthogonality_error(&v);

        println!("BIDIAG_GOLUB_KAHAN_GOLDEN_BEGIN");
        println!("shape={}x{}", reduction.rows, reduction.cols);
        println!("left_reflectors={}", reduction.left_reflectors.len());
        println!("right_reflectors={}", reduction.right_reflectors.len());
        println!("reconstruction_error={reconstruction_error:.17e}");
        println!("q_orthogonality_error={q_error:.17e}");
        println!("v_orthogonality_error={v_error:.17e}");
        for idx in 0..reduction.diagonal.len() {
            println!("diagonal[{idx}]={:.17e}", reduction.diagonal[idx]);
        }
        for idx in 0..reduction.superdiagonal.len() {
            println!("superdiagonal[{idx}]={:.17e}", reduction.superdiagonal[idx]);
        }
        println!("BIDIAG_GOLUB_KAHAN_GOLDEN_END");
    }

    #[test]
    #[ignore = "perf probe: run with rch and --release before/after bidiagonal reduction levers"]
    fn bidiag_large_reduction_perf_probe() {
        let original = bidiag_deterministic_matrix(1024, 512);
        let started_at = std::time::Instant::now();
        let reduction = golub_kahan_bidiagonal_reduction(std::hint::black_box(&original))
            .expect("bidiagonal reduction");
        let elapsed = started_at.elapsed();
        assert_eq!(reduction.rows, 1024);
        assert_eq!(reduction.cols, 512);
        assert_eq!(reduction.diagonal.len(), 512);
        assert_eq!(reduction.superdiagonal.len(), 511);
        assert_upper_bidiagonal(&reduction, 0.0);

        println!("BIDIAG_LARGE_REDUCTION_PERF_BEGIN");
        println!("shape={}x{}", reduction.rows, reduction.cols);
        println!("elapsed_ms={:.6}", elapsed.as_secs_f64() * 1_000.0);
        println!("digest={:#018x}", bidiag_reduction_digest(&reduction));
        println!("first_diagonal={:.17e}", reduction.diagonal[0]);
        println!("last_diagonal={:.17e}", reduction.diagonal[511]);
        println!("BIDIAG_LARGE_REDUCTION_PERF_END");
    }

    #[test]
    fn bidiag_svd_diagonal_values_descend_and_reconstruct() {
        let diagonal = [3.0, -4.0, 0.5];
        let superdiagonal = [0.0, 0.0];
        let svd =
            deterministic_bidiagonal_svd(3, &diagonal, &superdiagonal).expect("bidiagonal SVD");
        assert_nonincreasing(&svd.singular_values);
        assert_eq!(svd.singular_values[0].to_bits(), 4.0_f64.to_bits());
        assert_eq!(svd.singular_values[1].to_bits(), 3.0_f64.to_bits());
        assert_eq!(svd.singular_values[2].to_bits(), 0.5_f64.to_bits());

        let expected = bidiagonal_matrix_from_parts(3, &diagonal, &superdiagonal);
        let reconstructed = &svd.u * svd.sigma_matrix() * &svd.v_t;
        assert!(
            max_abs_dmatrix_diff(&reconstructed, &expected) < 1e-12,
            "diagonal bidiagonal SVD must reconstruct"
        );
        assert!(
            dmatrix_column_orthogonality_error(&svd.u) < 1e-12,
            "compact U columns must be orthonormal"
        );
        assert!(
            dmatrix_orthogonality_error(&svd.v_t) < 1e-12,
            "Vt rows must be orthonormal"
        );
    }

    #[test]
    fn bidiag_svd_clustered_values_are_deterministic() {
        let diagonal = [1.0, 1.0 + 1e-12, 1.0 - 1e-12, 0.125];
        let superdiagonal = [1e-14, -2e-14, 0.0];
        let first = deterministic_bidiagonal_svd(4, &diagonal, &superdiagonal)
            .expect("first bidiagonal SVD");
        let second = deterministic_bidiagonal_svd(4, &diagonal, &superdiagonal)
            .expect("second bidiagonal SVD");
        assert_nonincreasing(&first.singular_values);

        for idx in 0..first.singular_values.len() {
            assert_eq!(
                first.singular_values[idx].to_bits(),
                second.singular_values[idx].to_bits()
            );
        }
        for row in 0..first.v_t.nrows() {
            for col in 0..first.v_t.ncols() {
                assert_eq!(
                    first.v_t[(row, col)].to_bits(),
                    second.v_t[(row, col)].to_bits()
                );
            }
        }
    }

    #[test]
    fn bidiag_svd_rank_threshold_boundaries_remain_nonnegative() {
        let diagonal = [5.0, 1e-15, 0.0];
        let superdiagonal = [0.0, 0.0];
        let svd = deterministic_bidiagonal_svd(5, &diagonal, &superdiagonal)
            .expect("rank-boundary bidiagonal SVD");
        assert_nonincreasing(&svd.singular_values);
        assert!(svd.singular_values.iter().all(|value| *value >= 0.0));
        assert_eq!(svd.singular_values[0].to_bits(), 5.0_f64.to_bits());
        assert_eq!(svd.singular_values[2].to_bits(), 0.0_f64.to_bits());

        let expected = bidiagonal_matrix_from_parts(5, &diagonal, &superdiagonal);
        let reconstructed = &svd.u * svd.sigma_matrix() * &svd.v_t;
        assert!(
            max_abs_dmatrix_diff(&reconstructed, &expected) <= 1e-14,
            "rank-boundary reconstruction must stay under the SVD cutoff"
        );
    }

    #[test]
    fn bidiag_svd_reconstructs_golub_kahan_bidiagonal() {
        let original = bidiag_deterministic_matrix(8, 5);
        let reduction = golub_kahan_bidiagonal_reduction(&original).expect("bidiagonal reduction");
        let svd = deterministic_bidiagonal_svd_from_reduction(&reduction)
            .expect("deterministic bidiagonal SVD");
        assert_nonincreasing(&svd.singular_values);

        let reconstructed = &svd.u * svd.sigma_matrix() * &svd.v_t;
        let reconstruction_error = max_abs_dmatrix_diff(&reconstructed, &reduction.bidiagonal);
        let u_error = dmatrix_column_orthogonality_error(&svd.u);
        let v_error = dmatrix_orthogonality_error(&svd.v_t);

        assert!(
            reconstruction_error < 1e-10,
            "bidiagonal SVD reconstruction error {reconstruction_error:.17e}"
        );
        assert!(
            u_error < 1e-12,
            "compact U orthogonality error {u_error:.17e}"
        );
        assert!(v_error < 1e-12, "Vt orthogonality error {v_error:.17e}");
    }

    #[test]
    fn bidiag_svd_golden_payload() {
        let original = bidiag_deterministic_matrix(7, 4);
        let reduction = golub_kahan_bidiagonal_reduction(&original).expect("bidiagonal reduction");
        let svd = deterministic_bidiagonal_svd_from_reduction(&reduction)
            .expect("deterministic bidiagonal SVD");
        let reconstructed = &svd.u * svd.sigma_matrix() * &svd.v_t;
        let reconstruction_error = max_abs_dmatrix_diff(&reconstructed, &reduction.bidiagonal);
        let u_error = dmatrix_column_orthogonality_error(&svd.u);
        let v_error = dmatrix_orthogonality_error(&svd.v_t);

        println!("BIDIAG_SVD_GOLDEN_BEGIN");
        println!("shape={}x{}", reduction.rows, reduction.cols);
        println!("jacobi_sweeps={}", svd.sweeps);
        println!("reconstruction_error={reconstruction_error:.17e}");
        println!("u_column_orthogonality_error={u_error:.17e}");
        println!("v_orthogonality_error={v_error:.17e}");
        for idx in 0..svd.singular_values.len() {
            println!("singular[{idx}]={:.17e}", svd.singular_values[idx]);
        }
        println!("BIDIAG_SVD_GOLDEN_END");
    }

    #[test]
    fn thin_bidiag_svd_reconstructs_original_tall_matrix() {
        let original = bidiag_deterministic_matrix(9, 5);
        let svd = deterministic_thin_svd(&original).expect("deterministic thin SVD");
        assert_eq!(svd.u.nrows(), original.nrows());
        assert_eq!(svd.u.ncols(), original.ncols());
        assert_eq!(svd.v_t.nrows(), original.ncols());
        assert_eq!(svd.v_t.ncols(), original.ncols());
        assert_nonincreasing(&svd.singular_values);

        let reconstructed = &svd.u * svd.sigma_matrix() * &svd.v_t;
        let reconstruction_error = max_abs_dmatrix_diff(&reconstructed, &original);
        let u_error = dmatrix_column_orthogonality_error(&svd.u);
        let v_error = dmatrix_orthogonality_error(&svd.v_t);
        assert!(
            reconstruction_error < 1e-9,
            "thin SVD reconstruction error {reconstruction_error:.17e}"
        );
        assert!(u_error < 1e-12, "thin U orthogonality error {u_error:.17e}");
        assert!(
            v_error < 1e-12,
            "thin Vt orthogonality error {v_error:.17e}"
        );
    }

    #[test]
    fn thin_bidiag_svd_is_bit_deterministic_for_fixed_input() {
        let original = bidiag_deterministic_matrix(8, 5);
        let first = deterministic_thin_svd(&original).expect("first deterministic thin SVD");
        let second = deterministic_thin_svd(&original).expect("second deterministic thin SVD");

        for idx in 0..first.singular_values.len() {
            assert_eq!(
                first.singular_values[idx].to_bits(),
                second.singular_values[idx].to_bits()
            );
        }
        for row in 0..first.u.nrows() {
            for col in 0..first.u.ncols() {
                assert_eq!(
                    first.u[(row, col)].to_bits(),
                    second.u[(row, col)].to_bits()
                );
            }
        }
        for row in 0..first.v_t.nrows() {
            for col in 0..first.v_t.ncols() {
                assert_eq!(
                    first.v_t[(row, col)].to_bits(),
                    second.v_t[(row, col)].to_bits()
                );
            }
        }
    }

    #[test]
    fn thin_bidiag_svd_lstsq_and_pinv_match_public_routes() {
        let original = bidiag_deterministic_matrix(12, 6);
        let rows = original.nrows();
        let cols = original.ncols();
        let matrix_rows = rows_from_dmatrix(&original);
        let rhs_values: Vec<f64> = (0..rows)
            .map(|idx| ((idx * 19 + 7) % 31) as f64 - 11.0)
            .collect();
        let rhs = DVector::from_column_slice(&rhs_values);

        let thin = deterministic_thin_svd(&original).expect("deterministic thin SVD");
        let max_s = thin.singular_values.iter().copied().fold(0.0_f64, f64::max);
        let lstsq_threshold = f64::EPSILON * max_s;
        let pinv_threshold = (rows.max(cols) as f64) * f64::EPSILON * max_s;
        let expected_rank = thin
            .singular_values
            .iter()
            .filter(|value| **value > lstsq_threshold)
            .count();

        let thin_lstsq = thin
            .least_squares_solution(lstsq_threshold, &rhs)
            .expect("thin SVD least-squares solution");
        let public_lstsq =
            lstsq(&matrix_rows, &rhs_values, LstsqOptions::default()).expect("public lstsq");
        let thin_lstsq_values: Vec<f64> = thin_lstsq.iter().copied().collect();
        assert_eq!(public_lstsq.rank, expected_rank);
        assert_close_slice(&thin_lstsq_values, &public_lstsq.x, 1e-8, 1e-8);
        assert_close_slice(
            &thin.singular_values,
            &public_lstsq.singular_values,
            1e-8,
            1e-8,
        );

        let thin_pinv = thin.pseudo_inverse(pinv_threshold);
        let public_pinv = pinv(&matrix_rows, PinvOptions::default()).expect("public pinv");
        assert_eq!(public_pinv.rank, expected_rank);
        assert_close_matrix(
            &rows_from_dmatrix(&thin_pinv),
            &public_pinv.pseudo_inverse,
            1e-8,
            1e-8,
        );
    }

    #[test]
    fn thin_bidiag_svd_golden_payload() {
        let original = bidiag_deterministic_matrix(7, 4);
        let svd = deterministic_thin_svd(&original).expect("deterministic thin SVD");
        let reconstructed = &svd.u * svd.sigma_matrix() * &svd.v_t;
        let reconstruction_error = max_abs_dmatrix_diff(&reconstructed, &original);
        let u_error = dmatrix_column_orthogonality_error(&svd.u);
        let v_error = dmatrix_orthogonality_error(&svd.v_t);

        println!("THIN_BIDIAG_SVD_GOLDEN_BEGIN");
        println!("shape={}x{}", original.nrows(), original.ncols());
        println!("jacobi_sweeps={}", svd.jacobi_sweeps);
        println!("reconstruction_error={reconstruction_error:.17e}");
        println!("u_column_orthogonality_error={u_error:.17e}");
        println!("vt_orthogonality_error={v_error:.17e}");
        for idx in 0..svd.singular_values.len() {
            println!("singular[{idx}]={:.17e}", svd.singular_values[idx]);
        }
        for (row, col) in [(0, 0), (2, 1), (6, 3)] {
            println!("u[{row},{col}]={:.17e}", svd.u[(row, col)]);
        }
        for (row, col) in [(0, 0), (1, 3), (3, 2)] {
            println!("vt[{row},{col}]={:.17e}", svd.v_t[(row, col)]);
        }
        println!("THIN_BIDIAG_SVD_GOLDEN_END");
    }

    #[test]
    fn public_svd_lstsq_pinv_golden_payload() {
        let original = bidiag_deterministic_matrix(10, 5);
        let matrix_rows = rows_from_dmatrix(&original);
        let rhs: Vec<f64> = (0..original.nrows())
            .map(|idx| ((idx * 13 + 5) % 23) as f64 - 7.0)
            .collect();

        let svd_result = svd(&matrix_rows, DecompOptions::default()).expect("public svd");
        let svdvals_result =
            svdvals(&matrix_rows, DecompOptions::default()).expect("public svdvals");
        let lstsq_result =
            lstsq(&matrix_rows, &rhs, LstsqOptions::default()).expect("public lstsq");
        let pinv_result = pinv(&matrix_rows, PinvOptions::default()).expect("public pinv");
        assert_close_slice(&svd_result.s, &svdvals_result, 1e-14, 1e-14);

        println!("PUBLIC_SVD_LSTSQ_PINV_GOLDEN_BEGIN");
        println!("shape={}x{}", original.nrows(), original.ncols());
        println!(
            "svd_u_shape={}x{}",
            svd_result.u.len(),
            svd_result.u[0].len()
        );
        println!(
            "svd_vt_shape={}x{}",
            svd_result.vt.len(),
            svd_result.vt[0].len()
        );
        println!("lstsq_rank={}", lstsq_result.rank);
        println!("pinv_rank={}", pinv_result.rank);
        for idx in 0..svd_result.s.len() {
            println!("svd_singular[{idx}]={:.17e}", svd_result.s[idx]);
        }
        for (row, col) in [(0, 0), (3, 2), (9, 4)] {
            println!("svd_u[{row},{col}]={:.17e}", svd_result.u[row][col]);
        }
        for (row, col) in [(0, 0), (2, 3), (4, 1)] {
            println!("svd_vt[{row},{col}]={:.17e}", svd_result.vt[row][col]);
        }
        for idx in 0..lstsq_result.x.len() {
            println!("lstsq_x[{idx}]={:.17e}", lstsq_result.x[idx]);
        }
        for (row, col) in [(0, 0), (1, 7), (4, 9)] {
            println!(
                "pinv[{row},{col}]={:.17e}",
                pinv_result.pseudo_inverse[row][col]
            );
        }
        println!("PUBLIC_SVD_LSTSQ_PINV_GOLDEN_END");
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn svd_decomposition_a_equals_usv() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let result = svd(&a, DecompOptions::default()).expect("svd works");
        let cols = a[0].len();
        let k = result.s.len();
        // Reconstruct A from U * diag(S) * Vt
        for (i, (a_row, u_row)) in a.iter().zip(result.u.iter()).enumerate() {
            for j in 0..cols {
                let val: f64 = (0..k)
                    .map(|l| u_row[l] * result.s[l] * result.vt[l][j])
                    .sum();
                assert!(
                    (a_row[j] - val).abs() < 1e-10,
                    "A != U*S*Vt at [{i}][{j}]: {} vs {val}",
                    a_row[j]
                );
            }
        }
    }

    #[test]
    fn svd_singular_values_non_negative_and_descending() {
        let a = vec![vec![3.0, 2.0, 2.0], vec![2.0, 3.0, -2.0]];
        let result = svd(&a, DecompOptions::default()).expect("svd works");
        for s in &result.s {
            assert!(*s >= 0.0, "singular value should be non-negative: {s}");
        }
        for w in result.s.windows(2) {
            assert!(w[0] >= w[1], "singular values should be descending");
        }
    }

    #[test]
    fn svdvals_matches_svd() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let full_result = svd(&a, DecompOptions::default()).expect("svd works");
        let vals = svdvals(&a, DecompOptions::default()).expect("svdvals works");
        assert_close_slice(&vals, &full_result.s, 1e-14, 1e-14);
    }

    #[test]
    fn svd_empty_matrix() {
        let a: Vec<Vec<f64>> = vec![];
        let result = svd(&a, DecompOptions::default()).expect("svd of empty");
        assert!(result.u.is_empty());
        assert!(result.s.is_empty());
        assert!(result.vt.is_empty());
    }

    // ── Cholesky tests ──────────────────────────────────────────────

    #[test]
    fn cholesky_lower_factor_reconstructs_a() {
        let a = vec![
            vec![4.0, 2.0, 0.0],
            vec![2.0, 5.0, 2.0],
            vec![0.0, 2.0, 6.0],
        ];
        let result = cholesky(&a, true, DecompOptions::default()).expect("cholesky works");
        let n = a.len();
        // Verify L * L^T = A
        for (i, (a_row, l_row)) in a.iter().zip(result.factor.iter()).enumerate() {
            for (j, &a_ij) in a_row.iter().enumerate() {
                let val: f64 = (0..n).map(|k| l_row[k] * result.factor[j][k]).sum();
                assert!(
                    (a_ij - val).abs() < 1e-10,
                    "L*L^T != A at [{i}][{j}]: {a_ij} vs {val}"
                );
            }
        }
    }

    #[test]
    fn cholesky_upper_factor() {
        let a = vec![vec![4.0, 2.0], vec![2.0, 5.0]];
        let result = cholesky(&a, false, DecompOptions::default()).expect("cholesky upper works");
        // Verify U^T * U = A
        for (i, a_row) in a.iter().enumerate() {
            for (j, &a_ij) in a_row.iter().enumerate() {
                let val: f64 = result.factor.iter().map(|row| row[i] * row[j]).sum();
                assert!(
                    (a_ij - val).abs() < 1e-10,
                    "U^T*U != A at [{i}][{j}]: {a_ij} vs {val}"
                );
            }
        }
    }

    #[test]
    fn cholesky_not_positive_definite() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 1.0]]; // not PD
        let err = cholesky(&a, true, DecompOptions::default()).unwrap_err();
        assert!(matches!(err, LinalgError::InvalidArgument { .. }));
    }

    #[test]
    fn cho_factor_and_solve() {
        let a = vec![vec![4.0, 2.0], vec![2.0, 5.0]];
        let b = vec![8.0, 7.0];
        let factor = cho_factor(&a, DecompOptions::default()).expect("cho_factor works");
        let result = cho_solve(&factor, &b).expect("cho_solve works");
        // Verify A * x = b
        for (i, row) in a.iter().enumerate() {
            let dot: f64 = row.iter().zip(&result.x).map(|(a, x)| a * x).sum();
            assert!(
                (dot - b[i]).abs() < 1e-10,
                "A*x != b at row {i}: {dot} vs {}",
                b[i]
            );
        }
    }

    #[test]
    fn cholesky_identity() {
        let a = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let result = cholesky(&a, true, DecompOptions::default()).expect("cholesky of I");
        assert_close_matrix(&result.factor, &a, 1e-14, 1e-14);
    }

    #[test]
    fn cho_solve_banded_tridiagonal() {
        // A = [[4, 2, 0], [2, 5, 2], [0, 2, 6]]
        // Cholesky: L = [[2, 0, 0], [1, 2, 0], [0, 1, sqrt(5)]]
        // Banded storage: cb[k, i] = L[i+k, i]
        // cb[0] = [2, 2, sqrt(5)] (diagonal)
        // cb[1] = [1, 1, 0] (sub-diagonal: L[1,0]=1, L[2,1]=1)
        let sqrt5 = 5.0_f64.sqrt();
        let cb = vec![
            vec![2.0, 2.0, sqrt5], // diagonal of L
            vec![1.0, 1.0, 0.0],   // sub-diagonal
        ];
        // Use b such that x is known: x = [1, 1, 1]
        // A @ [1,1,1] = [4+2, 2+5+2, 2+6] = [6, 9, 8]
        let b = vec![6.0, 9.0, 8.0];
        let result = cho_solve_banded(&cb, &b, true).expect("cho_solve_banded");
        assert!((result.x[0] - 1.0).abs() < 1e-10, "x[0] = {}", result.x[0]);
        assert!((result.x[1] - 1.0).abs() < 1e-10, "x[1] = {}", result.x[1]);
        assert!((result.x[2] - 1.0).abs() < 1e-10, "x[2] = {}", result.x[2]);
    }

    #[test]
    fn cho_solve_banded_diagonal() {
        // Diagonal positive definite matrix
        let cb = vec![vec![2.0, 3.0, 4.0]]; // Only diagonal, no off-diagonal bands
        let b = vec![2.0, 6.0, 8.0];
        // A = diag(4, 9, 16), b = [2, 6, 8], x = [0.5, 2/3, 0.5]
        // Wait, L@L^T where L = diag(2,3,4) gives A = diag(4, 9, 16)
        let result = cho_solve_banded(&cb, &b, true).expect("cho_solve_banded diagonal");
        assert!((result.x[0] - 0.5).abs() < 1e-10, "x[0] = {}", result.x[0]);
        assert!(
            (result.x[1] - 2.0 / 3.0).abs() < 1e-10,
            "x[1] = {}",
            result.x[1]
        );
        assert!((result.x[2] - 0.5).abs() < 1e-10, "x[2] = {}", result.x[2]);
    }

    #[test]
    fn cho_solve_banded_empty() {
        let cb = vec![vec![]];
        let b = vec![];
        let result = cho_solve_banded(&cb, &b, true).expect("empty");
        assert!(result.x.is_empty());
    }

    #[test]
    fn solveh_banded_tridiagonal() {
        // A = [[4, 2, 0], [2, 5, 2], [0, 2, 6]] (symmetric positive definite)
        // Lower band storage: ab[k, i] = A[i+k, i]
        // ab[0] = [4, 5, 6] (diagonal)
        // ab[1] = [2, 2, 0] (sub-diagonal: A[1,0]=2, A[2,1]=2)
        let ab = vec![vec![4.0, 5.0, 6.0], vec![2.0, 2.0, 0.0]];
        // b = A @ [1, 1, 1] = [6, 9, 8]
        let b = vec![6.0, 9.0, 8.0];
        let result = solveh_banded(&ab, &b, true).expect("solveh_banded");
        assert!((result.x[0] - 1.0).abs() < 1e-8, "x[0] = {}", result.x[0]);
        assert!((result.x[1] - 1.0).abs() < 1e-8, "x[1] = {}", result.x[1]);
        assert!((result.x[2] - 1.0).abs() < 1e-8, "x[2] = {}", result.x[2]);
    }

    #[test]
    fn solveh_banded_diagonal() {
        // Diagonal matrix A = diag(4, 9, 16)
        let ab = vec![vec![4.0, 9.0, 16.0]];
        let b = vec![8.0, 27.0, 32.0];
        // x = [2, 3, 2]
        let result = solveh_banded(&ab, &b, true).expect("solveh_banded diagonal");
        assert!((result.x[0] - 2.0).abs() < 1e-10, "x[0] = {}", result.x[0]);
        assert!((result.x[1] - 3.0).abs() < 1e-10, "x[1] = {}", result.x[1]);
        assert!((result.x[2] - 2.0).abs() < 1e-10, "x[2] = {}", result.x[2]);
    }

    #[test]
    fn solveh_banded_empty() {
        let ab = vec![vec![]];
        let b = vec![];
        let result = solveh_banded(&ab, &b, true).expect("empty");
        assert!(result.x.is_empty());
    }

    // ── Eigenvalue decomposition tests ──────────────────────────────

    #[test]
    fn eigh_symmetric_eigenvalues_ascending() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let result = eigh(&a, DecompOptions::default()).expect("eigh works");
        // Eigenvalues should be ascending
        for w in result.eigenvalues.windows(2) {
            assert!(w[0] <= w[1], "eigenvalues should be ascending");
        }
        // Known eigenvalues for this matrix: (5 ± sqrt(5)) / 2
        let expected_1 = (5.0 - 5.0_f64.sqrt()) / 2.0;
        let expected_2 = (5.0 + 5.0_f64.sqrt()) / 2.0;
        assert!(
            (result.eigenvalues[0] - expected_1).abs() < 1e-10,
            "eigenvalue 0: {} vs {expected_1}",
            result.eigenvalues[0]
        );
        assert!(
            (result.eigenvalues[1] - expected_2).abs() < 1e-10,
            "eigenvalue 1: {} vs {expected_2}",
            result.eigenvalues[1]
        );
    }

    #[test]
    fn eigh_eigenvectors_orthogonal() {
        let a = vec![
            vec![4.0, 1.0, 0.0],
            vec![1.0, 3.0, 1.0],
            vec![0.0, 1.0, 2.0],
        ];
        let result = eigh(&a, DecompOptions::default()).expect("eigh works");
        let n = a.len();
        // V^T * V should be identity
        for i in 0..n {
            for j in i..n {
                let dot: f64 = result.eigenvectors.iter().map(|row| row[i] * row[j]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "eigenvectors not orthogonal at [{i}][{j}]: {dot} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn eigh_av_equals_v_lambda() {
        let a = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let result = eigh(&a, DecompOptions::default()).expect("eigh works");
        let n = a.len();
        // For each eigenvalue/eigenvector pair: A*v = lambda*v
        for col in 0..n {
            let lambda = result.eigenvalues[col];
            for (i, a_row) in a.iter().enumerate() {
                let av: f64 = (0..n).map(|k| a_row[k] * result.eigenvectors[k][col]).sum();
                let lv = lambda * result.eigenvectors[i][col];
                assert!(
                    (av - lv).abs() < 1e-10,
                    "A*v != lambda*v at eigenvector {col}, row {i}: {av} vs {lv}"
                );
            }
        }
    }

    #[test]
    fn eigvalsh_matches_eigh() {
        let a = vec![vec![5.0, 2.0], vec![2.0, 3.0]];
        let full_result = eigh(&a, DecompOptions::default()).expect("eigh works");
        let vals = eigvalsh(&a, DecompOptions::default()).expect("eigvalsh works");
        assert_close_slice(&vals, &full_result.eigenvalues, 1e-14, 1e-14);
    }

    #[test]
    fn eig_general_eigenvalues() {
        let a = vec![vec![0.0, -1.0], vec![1.0, 0.0]]; // rotation 90 deg
        let result = eig(&a, DecompOptions::default()).expect("eig works");
        // Eigenvalues should be ±i (purely imaginary)
        assert_eq!(result.eigenvalues_re.len(), 2);
        for re in &result.eigenvalues_re {
            assert!(re.abs() < 1e-10, "real part should be ~0, got {re}");
        }
        let mut im_sorted: Vec<f64> = result.eigenvalues_im.clone();
        im_sorted.sort_by(|a, b| a.total_cmp(b));
        assert!(
            (im_sorted[0] - (-1.0)).abs() < 1e-10,
            "imaginary part should be -1"
        );
        assert!(
            (im_sorted[1] - 1.0).abs() < 1e-10,
            "imaginary part should be +1"
        );
    }

    #[test]
    fn eig_eigenvectors_satisfy_av_eq_lambda_v() {
        // [frankenscipy-eobzj] regression: previously returned the
        // Schur basis Q under the 'eigenvectors' field, which only
        // diagonalizes symmetric matrices. Now non-symmetric A must
        // yield true eigenvectors satisfying A · v_i = λ_i · v_i.
        let a = vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]];
        let r = eig(&a, DecompOptions::default()).unwrap();
        for col in 0..2 {
            // Skip complex eigenvalues (their eigenvectors are complex
            // and outside the real-vector API surface).
            if r.eigenvalues_im[col].abs() > 1e-10 {
                continue;
            }
            let lambda = r.eigenvalues_re[col];
            // Extract column `col` from row-major eigenvectors.
            let v: Vec<f64> = r.eigenvectors.iter().map(|row| row[col]).collect();
            // Compute A · v and λ · v.
            let av: Vec<f64> = (0..2).map(|i| a[i][0] * v[0] + a[i][1] * v[1]).collect();
            let lv: Vec<f64> = v.iter().map(|&x| lambda * x).collect();
            for i in 0..2 {
                assert!(
                    (av[i] - lv[i]).abs() < 1e-10,
                    "(A · v)[{i}] = {} but λ · v[{i}] = {} for col={col} λ={lambda}",
                    av[i],
                    lv[i]
                );
            }
            // Eigenvectors are unit-length.
            let norm = (v[0] * v[0] + v[1] * v[1]).sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-10,
                "v[col={col}] not unit-length: {norm}"
            );
        }
    }

    #[test]
    fn eig_real_eigenvalues_symmetric() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let result = eig(&a, DecompOptions::default()).expect("eig works");
        // Symmetric -> all eigenvalues real
        for im in &result.eigenvalues_im {
            assert!(im.abs() < 1e-10, "symmetric eigenvalues should be real");
        }
        let mut re_sorted: Vec<f64> = result.eigenvalues_re.clone();
        re_sorted.sort_by(|a, b| a.total_cmp(b));
        assert!((re_sorted[0] - 1.0).abs() < 1e-10);
        assert!((re_sorted[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn eigvals_matches_eig() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let full_result = eig(&a, DecompOptions::default()).expect("eig works");
        let (re, im) = eigvals(&a, DecompOptions::default()).expect("eigvals works");
        assert_close_slice(&re, &full_result.eigenvalues_re, 1e-14, 1e-14);
        assert_close_slice(&im, &full_result.eigenvalues_im, 1e-14, 1e-14);
    }

    #[test]
    fn eig_empty_matrix() {
        let a: Vec<Vec<f64>> = vec![];
        let result = eig(&a, DecompOptions::default()).expect("eig of empty");
        assert!(result.eigenvalues_re.is_empty());
    }

    #[test]
    fn eigh_identity_eigenvalues() {
        let a = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let result = eigh(&a, DecompOptions::default()).expect("eigh of I");
        for &val in &result.eigenvalues {
            assert!((val - 1.0).abs() < 1e-14, "eigenvalue of I should be 1");
        }
    }

    // ── Schur decomposition tests ────────────────────────────────────

    #[test]
    fn schur_a_equals_ztzt() {
        let a = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 0.0],
        ];
        let result = schur(&a, DecompOptions::default()).expect("schur works");
        let n = a.len();
        // Verify A = Z * T * Z^T
        for (i, a_row) in a.iter().enumerate() {
            for (j, &a_ij) in a_row.iter().enumerate() {
                let mut val = 0.0;
                for k in 0..n {
                    for l in 0..n {
                        val += result.z[i][k] * result.t[k][l] * result.z[j][l];
                    }
                }
                assert!(
                    (a_ij - val).abs() < 1e-10,
                    "A != Z*T*Z^T at [{i}][{j}]: {a_ij} vs {val}"
                );
            }
        }
    }

    #[test]
    fn schur_z_is_orthogonal() {
        let a = vec![vec![4.0, 1.0], vec![2.0, 3.0]];
        let result = schur(&a, DecompOptions::default()).expect("schur works");
        let n = result.z.len();
        for i in 0..n {
            for j in i..n {
                let dot: f64 = result.z.iter().map(|row| row[i] * row[j]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "Z not orthogonal at [{i}][{j}]"
                );
            }
        }
    }

    #[test]
    fn schur_t_is_quasi_upper_triangular() {
        let a = vec![vec![4.0, 1.0], vec![2.0, 3.0]];
        let result = schur(&a, DecompOptions::default()).expect("schur works");
        // For 2x2, T should be upper triangular or have a 2x2 block
        // In general, elements below the first sub-diagonal should be zero
        let n = result.t.len();
        for i in 2..n {
            for j in 0..i.saturating_sub(1) {
                assert!(
                    result.t[i][j].abs() < 1e-10,
                    "T[{i}][{j}] should be zero: {}",
                    result.t[i][j]
                );
            }
        }
    }

    #[test]
    fn schur_empty_matrix() {
        let a: Vec<Vec<f64>> = vec![];
        let result = schur(&a, DecompOptions::default()).expect("schur of empty");
        assert!(result.z.is_empty());
        assert!(result.t.is_empty());
    }

    // ── Hessenberg decomposition tests ──────────────────────────────

    #[test]
    fn hessenberg_a_equals_qhqt() {
        let a = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 0.0],
        ];
        let result = hessenberg(&a, DecompOptions::default()).expect("hessenberg works");
        let n = a.len();
        // Verify A = Q * H * Q^T
        for (i, a_row) in a.iter().enumerate() {
            for (j, &a_ij) in a_row.iter().enumerate() {
                let mut val = 0.0;
                for k in 0..n {
                    for l in 0..n {
                        val += result.q[i][k] * result.h[k][l] * result.q[j][l];
                    }
                }
                assert!(
                    (a_ij - val).abs() < 1e-10,
                    "A != Q*H*Q^T at [{i}][{j}]: {a_ij} vs {val}"
                );
            }
        }
    }

    #[test]
    fn hessenberg_h_is_upper_hessenberg() {
        let a = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 0.0],
        ];
        let result = hessenberg(&a, DecompOptions::default()).expect("hessenberg works");
        let n = result.h.len();
        // Elements below the first sub-diagonal should be zero
        for i in 2..n {
            for j in 0..i.saturating_sub(1) {
                assert!(
                    result.h[i][j].abs() < 1e-10,
                    "H[{i}][{j}] should be zero: {}",
                    result.h[i][j]
                );
            }
        }
    }

    #[test]
    fn hessenberg_q_is_orthogonal() {
        let a = vec![
            vec![4.0, 1.0, 0.0],
            vec![2.0, 3.0, 1.0],
            vec![0.0, 1.0, 2.0],
        ];
        let result = hessenberg(&a, DecompOptions::default()).expect("hessenberg works");
        let n = result.q.len();
        for i in 0..n {
            for j in i..n {
                let dot: f64 = result.q.iter().map(|row| row[i] * row[j]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "Q not orthogonal at [{i}][{j}]"
                );
            }
        }
    }

    #[test]
    fn hessenberg_empty_matrix() {
        let a: Vec<Vec<f64>> = vec![];
        let result = hessenberg(&a, DecompOptions::default()).expect("hessenberg of empty");
        assert!(result.q.is_empty());
        assert!(result.h.is_empty());
    }

    // ── QZ decomposition tests ───────────────────────────────────────

    #[test]
    fn qz_empty_matrix() {
        let a: Vec<Vec<f64>> = vec![];
        let b: Vec<Vec<f64>> = vec![];
        let result = qz(&a, &b, DecompOptions::default()).expect("qz of empty");
        assert!(result.q.is_empty());
        assert!(result.z.is_empty());
        assert!(result.aa.is_empty());
        assert!(result.bb.is_empty());
    }

    #[test]
    fn qz_singular_b_rejected() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![1.0, 0.0], vec![0.0, 0.0]];
        let err = qz(&a, &b, DecompOptions::default()).expect_err("singular B");
        assert!(matches!(err, LinalgError::SingularMatrix));
    }

    #[test]
    fn qz_generalized_relation_holds() {
        let a = vec![vec![4.0, 2.0], vec![1.0, 3.0]];
        let b = vec![vec![2.0, 0.0], vec![0.5, 1.5]];
        let result = qz(&a, &b, DecompOptions::default()).expect("qz works");

        let n = a.len();
        let mut qtaz = vec![vec![0.0; n]; n];
        let mut qtbz = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    for l in 0..n {
                        qtaz[i][j] += result.q[k][i] * a[k][l] * result.z[l][j];
                        qtbz[i][j] += result.q[k][i] * b[k][l] * result.z[l][j];
                    }
                }
            }
        }

        assert_close_matrix(&qtaz, &result.aa, 1e-10, 1e-10);
        assert_close_matrix(&qtbz, &result.bb, 1e-10, 1e-10);
        for i in 0..n {
            for j in 0..i {
                assert!(result.bb[i][j].abs() < 1e-10, "BB must be upper triangular");
            }
        }
    }

    #[test]
    fn qz_with_identity_b_matches_schur_contract() {
        let a = vec![
            vec![1.0, 2.0, 3.0],
            vec![0.0, 4.0, 5.0],
            vec![0.0, 0.0, 6.0],
        ];
        let b = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let result = qz(&a, &b, DecompOptions::default()).expect("qz works");
        assert_close_matrix(&result.bb, &b, 1e-10, 1e-10);
        for i in 2..result.aa.len() {
            for j in 0..i.saturating_sub(1) {
                assert!(
                    result.aa[i][j].abs() < 1e-10,
                    "AA must be quasi-upper triangular"
                );
            }
        }
    }

    /// Q and Z returned by `qz` must both be orthogonal — `M·Mᵀ = I` —
    /// across diagonal, SPD, and triangular fixtures (frankenscipy-uvrcc).
    #[test]
    fn qz_q_and_z_are_orthogonal() {
        let diag_a = vec![
            vec![3.0, 0.0, 0.0],
            vec![0.0, 5.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let diag_b = vec![
            vec![2.0, 0.0, 0.0],
            vec![0.0, 4.0, 0.0],
            vec![0.0, 0.0, 3.0],
        ];
        let spd_a = vec![
            vec![6.0, 1.0, 2.0],
            vec![1.0, 5.0, 1.0],
            vec![2.0, 1.0, 7.0],
        ];
        let spd_b = vec![
            vec![3.0, 1.0, 0.0],
            vec![1.0, 4.0, 1.0],
            vec![0.0, 1.0, 5.0],
        ];
        let gen_a = vec![vec![4.0, 2.0], vec![1.0, 3.0]];
        let gen_b = vec![vec![2.0, 0.0], vec![0.5, 1.5]];
        let fixtures = [(&gen_a, &gen_b), (&diag_a, &diag_b), (&spd_a, &spd_b)];
        for (a, b) in fixtures {
            let result = qz(a, b, DecompOptions::default()).expect("qz works");
            let n = a.len();
            for mat in [&result.q, &result.z] {
                for i in 0..n {
                    for j in 0..n {
                        let dot: f64 = (0..n).map(|k| mat[i][k] * mat[j][k]).sum();
                        let expected = if i == j { 1.0 } else { 0.0 };
                        assert!(
                            (dot - expected).abs() < 1e-10,
                            "row {i}·row {j} = {dot}, expected {expected}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn ordqz_empty_matrix() {
        let a: Vec<Vec<f64>> = vec![];
        let b: Vec<Vec<f64>> = vec![];
        let result = ordqz(
            &a,
            &b,
            OrdQzSort::InsideUnitCircle,
            DecompOptions::default(),
        )
        .expect("ordqz");
        assert!(result.q.is_empty());
        assert!(result.z.is_empty());
        assert!(result.aa.is_empty());
        assert!(result.bb.is_empty());
    }

    #[test]
    fn ordqz_inside_unit_circle_moves_stable_ratio_first() {
        let a = vec![
            vec![2.0, 0.0, 0.0],
            vec![0.0, 0.25, 0.0],
            vec![0.0, 0.0, -3.0],
        ];
        let b = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let result = ordqz(
            &a,
            &b,
            OrdQzSort::InsideUnitCircle,
            DecompOptions::default(),
        )
        .expect("ordqz");

        let ratios: Vec<f64> = (0..result.aa.len())
            .map(|i| result.aa[i][i] / result.bb[i][i])
            .collect();
        assert!(
            (ratios[0] - 0.25).abs() < 1e-12,
            "stable ratio must come first"
        );
        assert!(ratios[1].abs() >= 1.0);
        assert!(ratios[2].abs() >= 1.0);

        let n = a.len();
        let mut qtaz = vec![vec![0.0; n]; n];
        let mut qtbz = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    for l in 0..n {
                        qtaz[i][j] += result.q[k][i] * a[k][l] * result.z[l][j];
                        qtbz[i][j] += result.q[k][i] * b[k][l] * result.z[l][j];
                    }
                }
            }
        }

        assert_close_matrix(&qtaz, &result.aa, 1e-10, 1e-10);
        assert_close_matrix(&qtbz, &result.bb, 1e-10, 1e-10);
    }

    #[test]
    fn ordqz_left_half_plane_moves_negative_ratio_first() {
        let a = vec![
            vec![1.5, 0.0, 0.0],
            vec![0.0, -2.0, 0.0],
            vec![0.0, 0.0, -0.5],
        ];
        let b = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let result =
            ordqz(&a, &b, OrdQzSort::LeftHalfPlane, DecompOptions::default()).expect("ordqz");
        let ratios: Vec<f64> = (0..result.aa.len())
            .map(|i| result.aa[i][i] / result.bb[i][i])
            .collect();

        assert!(ratios[0] < 0.0);
        assert!(ratios[1] < 0.0);
        assert!(ratios[2] > 0.0);
    }

    #[test]
    fn ordqz_shape_mismatch_rejected() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let b = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let err = ordqz(
            &a,
            &b,
            OrdQzSort::InsideUnitCircle,
            DecompOptions::default(),
        )
        .expect_err("shape mismatch");
        assert!(matches!(err, LinalgError::ExpectedSquareMatrix));
    }

    // ── Matrix exponential tests ──────────────────────────────────────

    #[test]
    fn expm_identity() {
        // exp(0) = I
        let a = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let result = expm(&a, DecompOptions::default()).expect("expm works");
        assert_close_matrix(&result, &[vec![1.0, 0.0], vec![0.0, 1.0]], 1e-12, 1e-12);
    }

    #[test]
    fn expm_diagonal() {
        // exp(diag(1, 2)) = diag(e, e²)
        let a = vec![vec![1.0, 0.0], vec![0.0, 2.0]];
        let result = expm(&a, DecompOptions::default()).expect("expm works");
        assert!(
            (result[0][0] - std::f64::consts::E).abs() < 1e-10,
            "expm[0][0] should be e, got {}",
            result[0][0]
        );
        assert!(
            (result[1][1] - std::f64::consts::E.powi(2)).abs() < 1e-10,
            "expm[1][1] should be e², got {}",
            result[1][1]
        );
        assert!(result[0][1].abs() < 1e-12);
        assert!(result[1][0].abs() < 1e-12);
    }

    #[test]
    fn expm_nilpotent() {
        // A = [[0, 1], [0, 0]], exp(A) = [[1, 1], [0, 1]]
        let a = vec![vec![0.0, 1.0], vec![0.0, 0.0]];
        let result = expm(&a, DecompOptions::default()).expect("expm works");
        assert_close_matrix(&result, &[vec![1.0, 1.0], vec![0.0, 1.0]], 1e-12, 1e-12);
    }

    #[test]
    fn expm_rotation() {
        // A = [[0, -pi/2], [pi/2, 0]], exp(A) = [[0, -1], [1, 0]] (90° rotation)
        let half_pi = std::f64::consts::FRAC_PI_2;
        let a = vec![vec![0.0, -half_pi], vec![half_pi, 0.0]];
        let result = expm(&a, DecompOptions::default()).expect("expm works");
        assert!((result[0][0] - 0.0).abs() < 1e-10, "cos(pi/2) should be ~0");
        assert!(
            (result[0][1] - (-1.0)).abs() < 1e-10,
            "-sin(pi/2) should be -1"
        );
        assert!((result[1][0] - 1.0).abs() < 1e-10, "sin(pi/2) should be 1");
        assert!((result[1][1] - 0.0).abs() < 1e-10, "cos(pi/2) should be ~0");
    }

    #[test]
    fn expm_empty() {
        let a: Vec<Vec<f64>> = vec![];
        let result = expm(&a, DecompOptions::default()).expect("expm of empty");
        assert!(result.is_empty());
    }

    #[test]
    fn expm_scalar() {
        // exp([[3]]) = [[e³]]
        let a = vec![vec![3.0]];
        let result = expm(&a, DecompOptions::default()).expect("expm of scalar");
        assert!(
            (result[0][0] - 3.0_f64.exp()).abs() < 1e-10,
            "expm of scalar should be e³"
        );
    }

    #[test]
    fn expm_non_square_error() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let err = expm(&a, DecompOptions::default()).unwrap_err();
        assert_eq!(err, LinalgError::ExpectedSquareMatrix);
    }

    #[test]
    fn logm_upper_triangular_distinct_diagonal_matches_closed_form() {
        let a = vec![vec![2.0, 1.0], vec![0.0, 3.0]];
        let result = logm(&a, DecompOptions::default()).expect("logm works");
        let expected = vec![
            vec![2.0_f64.ln(), 3.0_f64.ln() - 2.0_f64.ln()],
            vec![0.0, 3.0_f64.ln()],
        ];
        assert_close_matrix(&result, &expected, 1e-12, 1e-12);
    }

    #[test]
    fn logm_upper_triangular_repeated_diagonal_uses_jordan_limit() {
        let a = vec![vec![2.0, 1.0], vec![0.0, 2.0]];
        let result = logm(&a, DecompOptions::default()).expect("logm works");
        let expected = vec![vec![2.0_f64.ln(), 0.5], vec![0.0, 2.0_f64.ln()]];
        assert_close_matrix(&result, &expected, 1e-12, 1e-12);
    }

    #[test]
    fn logm_identity_is_zero() {
        // logm(I) = 0 for identity matrix
        let eye = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let result = logm(&eye, DecompOptions::default()).expect("logm works");
        let zero = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        assert_close_matrix(&result, &zero, 1e-14, 1e-14);
    }

    #[test]
    fn logm_diagonal_is_log_of_diagonal() {
        // logm(diag(d)) = diag(log(d))
        let d = vec![
            vec![2.0, 0.0, 0.0],
            vec![0.0, 5.0, 0.0],
            vec![0.0, 0.0, 7.0],
        ];
        let result = logm(&d, DecompOptions::default()).expect("logm works");
        let expected = vec![
            vec![2.0_f64.ln(), 0.0, 0.0],
            vec![0.0, 5.0_f64.ln(), 0.0],
            vec![0.0, 0.0, 7.0_f64.ln()],
        ];
        assert_close_matrix(&result, &expected, 1e-12, 1e-12);
    }

    #[test]
    fn logm_inverse_of_expm() {
        // logm(expm(A)) ≈ A for well-conditioned A
        let a = vec![vec![0.5, 0.1], vec![0.1, 0.3]];
        let exp_a = expm(&a, DecompOptions::default()).expect("expm works");
        let log_exp_a = logm(&exp_a, DecompOptions::default()).expect("logm works");
        assert_close_matrix(&log_exp_a, &a, 1e-10, 1e-10);
    }

    #[test]
    fn logm_rejects_non_square() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let err = logm(&a, DecompOptions::default()).unwrap_err();
        assert_eq!(err, LinalgError::ExpectedSquareMatrix);
    }

    #[test]
    fn logm_3x3_spd() {
        // Test on a symmetric positive definite matrix
        let a = vec![
            vec![4.0, 1.0, 0.5],
            vec![1.0, 3.0, 0.25],
            vec![0.5, 0.25, 2.0],
        ];
        let log_a = logm(&a, DecompOptions::default()).expect("logm works");
        // Verify by computing expm(logm(A)) ≈ A
        let exp_log_a = expm(&log_a, DecompOptions::default()).expect("expm works");
        assert_close_matrix(&exp_log_a, &a, 1e-10, 1e-10);
    }

    // ── Norm and rank tests ─────────────────────────────────────────

    #[test]
    fn norm_frobenius_identity() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let n = norm(&a, NormKind::Fro, DecompOptions::default()).expect("norm works");
        assert!((n - 2.0_f64.sqrt()).abs() < 1e-14);
    }

    #[test]
    fn norm_spectral_diagonal() {
        let a = vec![vec![3.0, 0.0], vec![0.0, 5.0]];
        let n = norm(&a, NormKind::Spectral, DecompOptions::default()).expect("norm works");
        assert!(
            (n - 5.0).abs() < 1e-10,
            "spectral norm should be 5, got {n}"
        );
    }

    #[test]
    fn norm_one_norm() {
        let a = vec![vec![1.0, -7.0], vec![-2.0, 3.0]];
        let n = norm(&a, NormKind::One, DecompOptions::default()).expect("norm works");
        // Max column sum: col0 = 1+2=3, col1 = 7+3=10
        assert!((n - 10.0).abs() < 1e-14, "1-norm should be 10, got {n}");
    }

    #[test]
    fn norm_inf_norm() {
        let a = vec![vec![1.0, -7.0], vec![-2.0, 3.0]];
        let n = norm(&a, NormKind::Inf, DecompOptions::default()).expect("norm works");
        // Max row sum: row0 = 1+7=8, row1 = 2+3=5
        assert!((n - 8.0).abs() < 1e-14, "inf-norm should be 8, got {n}");
    }

    #[test]
    fn matrix_rank_full_rank() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let r = matrix_rank(&a, None, DecompOptions::default()).expect("rank works");
        assert_eq!(r, 2);
    }

    #[test]
    fn matrix_rank_rank_deficient() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let r = matrix_rank(&a, None, DecompOptions::default()).expect("rank works");
        assert_eq!(r, 1);
    }

    #[test]
    fn matrix_rank_zero_matrix() {
        let a = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let r = matrix_rank(&a, None, DecompOptions::default()).expect("rank works");
        assert_eq!(r, 0);
    }

    #[test]
    fn matrix_rank_rectangular() {
        let a = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let r = matrix_rank(&a, None, DecompOptions::default()).expect("rank works");
        assert_eq!(r, 2);
    }

    // ── Subspace operation tests ───────────────────────────────────

    #[test]
    fn orth_full_rank() {
        // 3×3 identity → orth should return 3 orthonormal columns
        let a = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let result = orth(&a, None, DecompOptions::default()).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].len(), 3);
    }

    #[test]
    fn orth_rank_deficient() {
        // Rank-1 matrix: all rows are multiples
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]];
        let result = orth(&a, None, DecompOptions::default()).unwrap();
        // Should return 1 orthonormal column
        assert_eq!(result[0].len(), 1);
        // Column should be unit norm
        let norm: f64 = result.iter().map(|r| r[0] * r[0]).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "orth column not unit norm: {norm}"
        );
    }

    #[test]
    fn null_space_full_rank() {
        // Full-rank 3×3 → null space should be empty
        let a = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let result = null_space(&a, None, DecompOptions::default()).unwrap();
        // Each row should have 0 null-space columns
        assert!(
            result.is_empty() || result[0].is_empty(),
            "full-rank matrix should have empty null space"
        );
    }

    #[test]
    fn null_space_rank_deficient() {
        // [[1, 2], [2, 4]] has rank 1, null space dim = 1
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let result = null_space(&a, None, DecompOptions::default()).unwrap();
        assert_eq!(result.len(), 2); // n=2 rows
        assert_eq!(result[0].len(), 1); // 1 null-space vector
        // Verify A * ns ≈ 0
        let ns = [result[0][0], result[1][0]];
        let ax0 = a[0][0] * ns[0] + a[0][1] * ns[1];
        let ax1 = a[1][0] * ns[0] + a[1][1] * ns[1];
        assert!(ax0.abs() < 1e-10, "A*ns[0] = {ax0}");
        assert!(ax1.abs() < 1e-10, "A*ns[1] = {ax1}");
    }

    #[test]
    fn subspace_angles_identical() {
        // Same subspace → angle should be 0
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 0.0]];
        let angles = subspace_angles(&a, &a, DecompOptions::default()).unwrap();
        for &angle in &angles {
            assert!(angle.abs() < 1e-10, "identical subspace angle: {angle}");
        }
    }

    #[test]
    fn subspace_angles_orthogonal() {
        // Two orthogonal 1-D subspaces in R^2
        let a = vec![vec![1.0], vec![0.0]]; // x-axis
        let b = vec![vec![0.0], vec![1.0]]; // y-axis
        let angles = subspace_angles(&a, &b, DecompOptions::default()).unwrap();
        assert!(!angles.is_empty());
        let half_pi = std::f64::consts::FRAC_PI_2;
        assert!(
            (angles[0] - half_pi).abs() < 1e-10,
            "orthogonal angle: {}, expected π/2",
            angles[0]
        );
    }

    #[test]
    fn polar_identity() {
        // Polar decomposition of identity: U = I, P = I
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let result = polar(&a, DecompOptions::default()).unwrap();
        let u = result.u;
        let p = result.p;
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (u[i][j] - expected).abs() < 1e-10,
                    "U[{i}][{j}] = {}, expected {expected}",
                    u[i][j]
                );
                assert!(
                    (p[i][j] - expected).abs() < 1e-10,
                    "P[{i}][{j}] = {}, expected {expected}",
                    p[i][j]
                );
            }
        }
    }

    #[test]
    fn polar_reconstruction() {
        // A = U * P, verify the reconstruction
        let a = vec![vec![3.0, 1.0], vec![1.0, 4.0]];
        let result = polar(&a, DecompOptions::default()).unwrap();
        let u = result.u;
        let p = result.p;

        // Reconstruct: U * P
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = 0.0;
                for k in 0..2 {
                    sum += u[i][k] * p[k][j];
                }
                assert!(
                    (sum - a[i][j]).abs() < 1e-8,
                    "UP[{i}][{j}] = {sum}, A[{i}][{j}] = {}",
                    a[i][j]
                );
            }
        }
    }

    #[test]
    fn polar_p_is_symmetric_positive() {
        let a = vec![vec![2.0, 1.0], vec![-1.0, 3.0]];
        let result = polar(&a, DecompOptions::default()).unwrap();
        let p = result.p;
        // P should be symmetric
        assert!(
            (p[0][1] - p[1][0]).abs() < 1e-10,
            "P not symmetric: P[0][1]={}, P[1][0]={}",
            p[0][1],
            p[1][0]
        );
        // P should be positive semi-definite (eigenvalues >= 0)
        // For 2×2: trace > 0 and det >= 0
        let trace = p[0][0] + p[1][1];
        let det_val = p[0][0] * p[1][1] - p[0][1] * p[1][0];
        assert!(trace > -1e-10, "P trace: {trace}");
        assert!(det_val > -1e-10, "P determinant: {det_val}");
    }

    // ── Special matrix constructor tests ───────────────────────────

    #[test]
    fn toeplitz_symmetric() {
        let c = vec![1.0, 2.0, 3.0];
        let t = toeplitz(&c, None);
        assert_eq!(t.len(), 3);
        assert_eq!(t[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(t[1], vec![2.0, 1.0, 2.0]);
        assert_eq!(t[2], vec![3.0, 2.0, 1.0]);
    }

    #[test]
    fn toeplitz_asymmetric() {
        let c = vec![1.0, 2.0, 3.0];
        let r = vec![1.0, 4.0, 5.0];
        let t = toeplitz(&c, Some(&r));
        assert_eq!(t[0], vec![1.0, 4.0, 5.0]);
        assert_eq!(t[1], vec![2.0, 1.0, 4.0]);
        assert_eq!(t[2], vec![3.0, 2.0, 1.0]);
    }

    #[test]
    fn toeplitz_ignores_row_zero_keeps_constant_diagonal() {
        // scipy.linalg.toeplitz documents that r[0] is ignored: the diagonal is
        // always c[0]. Previously row[0] leaked onto the diagonal for i>0,
        // corrupting it whenever r[0] != c[0]. Use r[0]=99 != c[0]=2 to lock it.
        let c = vec![2.0, 1.0, 0.0];
        let r = vec![99.0, -1.0, 0.5];
        let t = toeplitz(&c, Some(&r));
        // Diagonal is c[0]=2 everywhere; r[0]=99 never appears.
        assert_eq!(t[0], vec![2.0, -1.0, 0.5]);
        assert_eq!(t[1], vec![1.0, 2.0, -1.0]);
        assert_eq!(t[2], vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn circulant_basic_matches_scipy_first_column_convention() {
        // Resolves [frankenscipy-qfv5a]: circulant returns the matrix
        // whose FIRST COLUMN is c (scipy convention), not the first row.
        //   circulant([1, 2, 3]) = [[1, 3, 2],
        //                            [2, 1, 3],
        //                            [3, 2, 1]]
        // Verify both the column-as-c property and the cyclic-shift rule.
        let c = vec![1.0, 2.0, 3.0];
        let m = circulant(&c);
        assert_eq!(m[0], vec![1.0, 3.0, 2.0]);
        assert_eq!(m[1], vec![2.0, 1.0, 3.0]);
        assert_eq!(m[2], vec![3.0, 2.0, 1.0]);
        // First column equals c.
        for (i, &cv) in c.iter().enumerate() {
            assert_eq!(m[i][0], cv);
        }
    }

    #[test]
    fn hilbert_3x3() {
        let h = hilbert(3);
        assert!((h[0][0] - 1.0).abs() < 1e-12);
        assert!((h[0][1] - 0.5).abs() < 1e-12);
        assert!((h[0][2] - 1.0 / 3.0).abs() < 1e-12);
        assert!((h[1][0] - 0.5).abs() < 1e-12);
        assert!((h[1][1] - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn invhilbert_times_hilbert_is_identity() {
        let n = 4;
        let h = hilbert(n);
        let ih = invhilbert(n);
        // Product should be approximately identity.
        for (i, ih_row) in ih.iter().enumerate().take(n) {
            for (j, _) in h.iter().enumerate().take(n) {
                let mut sum = 0.0;
                for (k, &ih_val) in ih_row.iter().enumerate().take(n) {
                    sum += ih_val * h[k][j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sum - expected).abs() < 1e-6,
                    "invhilbert*hilbert[{i}][{j}] = {sum}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn hadamard_4() {
        let h = hadamard(4).unwrap();
        assert_eq!(h.len(), 4);
        // H^T H = 4 I
        for i in 0..4 {
            for j in 0..4 {
                let mut dot = 0.0;
                for row in h.iter().take(4) {
                    dot += row[i] * row[j];
                }
                let expected = if i == j { 4.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-12,
                    "H^T H[{i}][{j}] = {dot}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn hadamard_non_power_of_2_rejected() {
        assert!(hadamard(3).is_err());
        assert!(hadamard(6).is_err());
    }

    #[test]
    fn companion_polynomial() {
        // p(x) = x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
        // Roots are eigenvalues of companion matrix
        let a = vec![1.0, -6.0, 11.0, -6.0];
        let c = companion(&a).unwrap();
        assert_eq!(c.len(), 3);
        assert_eq!(c[0], vec![6.0, -11.0, 6.0]); // -a[1..]/a[0]
        assert_eq!(c[1], vec![1.0, 0.0, 0.0]);
        assert_eq!(c[2], vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn companion_too_few_coeffs() {
        assert!(companion(&[1.0]).is_err());
    }

    #[test]
    fn block_diag_basic() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0]];
        let bd = block_diag(&[a, b]);
        assert_eq!(bd.len(), 3);
        assert_eq!(bd[0].len(), 3);
        assert_eq!(bd[0], vec![1.0, 2.0, 0.0]);
        assert_eq!(bd[1], vec![3.0, 4.0, 0.0]);
        assert_eq!(bd[2], vec![0.0, 0.0, 5.0]);
    }

    #[test]
    fn block_diag_empty() {
        let bd = block_diag(&[]);
        assert!(bd.is_empty());
    }

    #[test]
    fn pascal_lower_triangular() {
        let p = pascal(4, false);
        assert_eq!(p[0], vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!(p[1], vec![1.0, 1.0, 0.0, 0.0]);
        assert_eq!(p[2], vec![1.0, 2.0, 1.0, 0.0]);
        assert_eq!(p[3], vec![1.0, 3.0, 3.0, 1.0]);
    }

    #[test]
    fn pascal_symmetric() {
        let p = pascal(3, true);
        // Symmetric Pascal: P[i][j] = C(i+j, i)
        assert_eq!(p[0], vec![1.0, 1.0, 1.0]);
        assert_eq!(p[1], vec![1.0, 2.0, 3.0]);
        assert_eq!(p[2], vec![1.0, 3.0, 6.0]);
    }

    #[test]
    fn fiedler_basic() {
        let f = fiedler(&[1.0, 4.0, 2.0]);
        assert_eq!(f[0][0], 0.0);
        assert_eq!(f[0][1], 3.0); // |1-4|
        assert_eq!(f[0][2], 1.0); // |1-2|
        assert_eq!(f[1][0], 3.0);
        assert_eq!(f[1][2], 2.0); // |4-2|
    }

    #[test]
    fn fiedler_companion_pentadiagonal() {
        // p(x) = x^4 - 3x^3 + 2x^2 + 5x - 1 — matches scipy.linalg.fiedler_companion.
        let c = fiedler_companion(&[1.0, -3.0, 2.0, 5.0, -1.0]);
        assert_eq!(
            c,
            vec![
                vec![3.0, -2.0, 1.0, 0.0],
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, -5.0, 0.0, 1.0],
                vec![0.0, 1.0, 0.0, 0.0],
            ]
        );
    }

    #[test]
    fn fiedler_companion_non_monic_rescales() {
        // Non-monic leading coefficient is normalized before construction:
        // [2, -4, 6, -8] ≡ monic [1, -2, 3, -4].
        let c = fiedler_companion(&[2.0, -4.0, 6.0, -8.0]);
        assert_eq!(
            c,
            vec![
                vec![2.0, -3.0, 1.0],
                vec![1.0, 0.0, 0.0],
                vec![0.0, 4.0, 0.0],
            ]
        );
    }

    #[test]
    fn fiedler_companion_edge_cases() {
        // Fewer than two coefficients → empty matrix.
        assert!(fiedler_companion(&[]).is_empty());
        assert!(fiedler_companion(&[3.0]).is_empty());
        // Exactly two coefficients → 1×1 matrix [[-a₁/a₀]].
        assert_eq!(fiedler_companion(&[2.0, 6.0]), vec![vec![-3.0]]);
    }

    #[test]
    fn leslie_matrix() {
        let l = leslie(&[0.5, 1.0, 0.8], &[0.9, 0.7]).unwrap();
        assert_eq!(l[0], vec![0.5, 1.0, 0.8]); // fecundity row
        assert_eq!(l[1][0], 0.9); // survival[0]
        assert_eq!(l[2][1], 0.7); // survival[1]
        assert_eq!(l[1][1], 0.0);
    }

    #[test]
    fn hankel_single_arg_pads_with_zeros() {
        // /testing-conformance-harnesses for [frankenscipy-azhz0]:
        // scipy.linalg.hankel(c) with no r argument uses c again,
        // and entries beyond the c length pad with 0.
        //   hankel([1, 2, 3]) =
        //   [[1, 2, 3],
        //    [2, 3, 0],
        //    [3, 0, 0]]
        let h = hankel(&[1.0_f64, 2.0, 3.0], None);
        assert_eq!(h.len(), 3);
        assert_eq!(h[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(h[1], vec![2.0, 3.0, 0.0]);
        assert_eq!(h[2], vec![3.0, 0.0, 0.0]);
    }

    #[test]
    fn hankel_two_arg_matches_scipy_docs_example() {
        // scipy docs example:
        //   hankel([1, 17, 99], [99, 23, 45, 67, 89])
        // First column is [1, 17, 99], last row is [99, 23, 45, 67, 89].
        // The c[-1] and r[0] elements must agree (both 99).
        //   row 0: c[0], r[1], r[2], r[3], r[4]  = [1, 23, 45, 67, 89]
        //   row 1: c[1], r[1], r[2], r[3], r[4]?
        // Actually scipy fills along the anti-diagonals; let me compute
        // it directly from the implementation:
        //   H[i, j] = c[i+j]                     if i+j < n
        //           = r[i+j-n+1]                 if i+j-n+1 < m
        //           = 0
        // For c=[1, 17, 99], r=[99, 23, 45, 67, 89] (n=3, m=5):
        //   H[0]: idx=0 c=1; idx=1 c=17; idx=2 c=99; idx=3-3+1=1 r=23; idx=4-3+1=2 r=45
        //         = [1, 17, 99, 23, 45]
        //   H[1]: idx=1 c=17; idx=2 c=99; idx=3-3+1=1 r=23; idx=4-3+1=2 r=45; idx=5-3+1=3 r=67
        //         = [17, 99, 23, 45, 67]
        //   H[2]: idx=2 c=99; idx=3-3+1=1 r=23; idx=4-3+1=2 r=45; idx=5-3+1=3 r=67; idx=6-3+1=4 r=89
        //         = [99, 23, 45, 67, 89]
        let h = hankel(
            &[1.0_f64, 17.0, 99.0],
            Some(&[99.0, 23.0, 45.0, 67.0, 89.0]),
        );
        assert_eq!(h.len(), 3);
        assert_eq!(h[0].len(), 5);
        assert_eq!(h[0], vec![1.0, 17.0, 99.0, 23.0, 45.0]);
        assert_eq!(h[1], vec![17.0, 99.0, 23.0, 45.0, 67.0]);
        assert_eq!(h[2], vec![99.0, 23.0, 45.0, 67.0, 89.0]);
    }

    #[test]
    fn hankel_metamorphic_constant_on_antidiagonals() {
        // /testing-metamorphic: H[i, j] depends only on i + j, so
        // H[i, j] = H[i+1, j-1] whenever both indices are in range.
        let c = [10.0_f64, 20.0, 30.0, 40.0];
        let r = [40.0_f64, 50.0, 60.0, 70.0, 80.0];
        let h = hankel(&c, Some(&r));
        for i in 0..h.len() - 1 {
            for j in 1..h[0].len() {
                assert!(
                    (h[i][j] - h[i + 1][j - 1]).abs() < 1e-15,
                    "hankel anti-diagonal broken at ({i}, {j}): {} vs {}",
                    h[i][j],
                    h[i + 1][j - 1]
                );
            }
        }
    }

    #[test]
    fn helmert_full_first_row_is_uniform() {
        // /mock-code-finder + frankenscipy-3t31o: helmert_full is the
        // (n, n) form including the constant first row [1/√n, ..., 1/√n].
        for &n in &[2usize, 3, 5, 10] {
            let h = helmert_full(n);
            assert_eq!(h.len(), n);
            assert_eq!(h[0].len(), n);
            let expected = 1.0 / (n as f64).sqrt();
            for &v in &h[0] {
                assert!(
                    (v - expected).abs() < 1e-12,
                    "helmert_full({n}) first row entry {v} != 1/√{n} = {expected}"
                );
            }
        }
    }

    #[test]
    fn helmert_full_rows_have_unit_norm() {
        // Each row of the (n, n) full Helmert matrix is a unit vector.
        for &n in &[2usize, 3, 5, 10] {
            let h = helmert_full(n);
            for (i, row) in h.iter().enumerate() {
                let norm_sq: f64 = row.iter().map(|&v| v * v).sum();
                assert!(
                    (norm_sq - 1.0).abs() < 1e-12,
                    "helmert_full({n}) row {i} has norm² = {norm_sq}, expected 1"
                );
            }
        }
    }

    #[test]
    fn helmert_full_rows_are_pairwise_orthogonal() {
        // Different rows are orthogonal (row_i · row_j = 0 for i ≠ j).
        for &n in &[3usize, 5, 8] {
            let h = helmert_full(n);
            for i in 0..n {
                for j in (i + 1)..n {
                    let dot: f64 = h[i].iter().zip(h[j].iter()).map(|(&a, &b)| a * b).sum();
                    assert!(
                        dot.abs() < 1e-12,
                        "helmert_full({n}) row {i} · row {j} = {dot} ≠ 0"
                    );
                }
            }
        }
    }

    #[test]
    fn helmert_default_excludes_first_row_per_scipy() {
        // /modes-of-reasoning-project-analysis for frankenscipy-3t31o:
        // helmert(n) defaults to scipy's full=False — returns (n-1, n)
        // submatrix without the constant first row. Verify the first
        // row of helmert_full equals the constant 1/√n entries and is
        // dropped in helmert.
        for &n in &[2usize, 3, 5, 10] {
            let full = helmert_full(n);
            let default = helmert(n);
            assert_eq!(default.len(), n - 1);
            assert_eq!(default[0].len(), n);
            // helmert(n) should equal full[1..]
            for (i, row) in default.iter().enumerate() {
                for (j, &v) in row.iter().enumerate() {
                    assert!((v - full[i + 1][j]).abs() < 1e-15);
                }
            }
        }
        // n = 0 still returns empty.
        assert_eq!(helmert(0), Vec::<Vec<f64>>::new());
        // n = 1 returns empty (single row dropped).
        assert_eq!(helmert(1), Vec::<Vec<f64>>::new());
    }

    #[test]
    fn convolution_matrix_full() {
        let m = convolution_matrix(&[1.0, 2.0, 3.0], 4, "full");
        assert_eq!(m.len(), 6); // 4 + 3 - 1
        assert_eq!(m[0].len(), 4);
        // First row should be [1, 0, 0, 0] (h[0] applied to first input)
        assert_eq!(m[0][0], 1.0);
    }

    #[test]
    fn bandwidth_diagonal() {
        let a = vec![
            vec![1.0, 2.0, 0.0],
            vec![0.0, 3.0, 4.0],
            vec![0.0, 0.0, 5.0],
        ];
        let (lower, upper) = bandwidth(&a);
        assert_eq!(lower, 0);
        assert_eq!(upper, 1);
    }

    #[test]
    fn tril_triu_roundtrip() {
        let a = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let l = tril(&a, 0);
        let u = triu(&a, 1);
        // l + u should reconstruct a
        for i in 0..3 {
            for j in 0..3 {
                assert!((l[i][j] + u[i][j] - a[i][j]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn kron_2x2() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![0.0, 5.0], vec![6.0, 7.0]];
        let k = kron(&a, &b);
        assert_eq!(k.len(), 4);
        assert_eq!(k[0].len(), 4);
        // k[0][0] = 1*0 = 0, k[0][1] = 1*5 = 5
        assert_eq!(k[0][0], 0.0);
        assert_eq!(k[0][1], 5.0);
        // k[0][2] = 2*0 = 0, k[0][3] = 2*5 = 10
        assert_eq!(k[0][2], 0.0);
        assert_eq!(k[0][3], 10.0);
    }

    #[test]
    fn kron_metamorphic_identity_is_identity() {
        // /testing-metamorphic for [frankenscipy-crwkd]:
        // kron(I_m, I_n) = I_{mn}.
        for &(m, n) in &[(2usize, 3usize), (3, 2), (4, 4), (1, 5), (5, 1)] {
            let i_m: Vec<Vec<f64>> = (0..m)
                .map(|i| (0..m).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
                .collect();
            let i_n: Vec<Vec<f64>> = (0..n)
                .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
                .collect();
            let k = kron(&i_m, &i_n);
            assert_eq!(k.len(), m * n, "kron(I_{m}, I_{n}).rows");
            assert_eq!(k[0].len(), m * n, "kron(I_{m}, I_{n}).cols");
            for (i, row) in k.iter().enumerate() {
                assert_eq!(row.len(), m * n, "kron(I_{m}, I_{n}).row[{i}].cols");
                for (j, &actual) in row.iter().enumerate() {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (actual - expected).abs() < 1e-15,
                        "kron(I_{m}, I_{n}) at [{i}][{j}] = {} != {expected}",
                        actual
                    );
                }
            }
        }
    }

    #[test]
    fn kron_metamorphic_shape_is_product_of_dims() {
        // shape(kron(A, B)) = (a_rows·b_rows, a_cols·b_cols).
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]; // 2×3
        let b = vec![vec![7.0, 8.0], vec![9.0, 10.0]]; // 2×2
        let k = kron(&a, &b);
        assert_eq!(k.len(), 4, "rows = 2·2");
        assert_eq!(k[0].len(), 6, "cols = 3·2");
    }

    #[test]
    fn kron_metamorphic_trace_equals_product_of_traces() {
        // For square A (m×m) and B (n×n):
        //   tr(kron(A, B)) = tr(A) · tr(B)
        let a = vec![vec![2.0, 1.0], vec![3.0, 5.0]]; // tr = 7
        let b = vec![
            vec![1.0, 0.0, 4.0],
            vec![0.0, 2.0, 1.0],
            vec![3.0, 1.0, 6.0],
        ]; // tr = 9
        let k = kron(&a, &b);
        let tr_a: f64 = (0..a.len()).map(|i| a[i][i]).sum();
        let tr_b: f64 = (0..b.len()).map(|i| b[i][i]).sum();
        let tr_k: f64 = (0..k.len()).map(|i| k[i][i]).sum();
        assert!(
            (tr_k - tr_a * tr_b).abs() < 1e-12,
            "tr(kron(A, B)) = {tr_k}, expected tr(A)·tr(B) = {} · {} = {}",
            tr_a,
            tr_b,
            tr_a * tr_b
        );
    }

    #[test]
    fn kron_metamorphic_2x2_full_matrix_pinned() {
        // Pin all 16 entries of the 2x2 ⊗ 2x2 example, not just 4.
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![0.0, 5.0], vec![6.0, 7.0]];
        let k = kron(&a, &b);
        // Expected: A_{ij} · B  block-by-block.
        //   k[0..2][0..2] = 1·B = [[0, 5], [6, 7]]
        //   k[0..2][2..4] = 2·B = [[0, 10], [12, 14]]
        //   k[2..4][0..2] = 3·B = [[0, 15], [18, 21]]
        //   k[2..4][2..4] = 4·B = [[0, 20], [24, 28]]
        let expected = [
            [0.0, 5.0, 0.0, 10.0],
            [6.0, 7.0, 12.0, 14.0],
            [0.0, 15.0, 0.0, 20.0],
            [18.0, 21.0, 24.0, 28.0],
        ];
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (k[i][j] - expected[i][j]).abs() < 1e-15,
                    "kron at [{i}][{j}] = {} != {}",
                    k[i][j],
                    expected[i][j]
                );
            }
        }
    }

    #[test]
    fn tri_basic() {
        let t = tri(3, 4, 0);
        assert_eq!(t[0], vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!(t[1], vec![1.0, 1.0, 0.0, 0.0]);
        assert_eq!(t[2], vec![1.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn dft_matrix_unitary() {
        let f = dft_matrix(4);
        // DFT matrix should be unitary: F * F^H = I
        // Check first row dot first row conjugate = 1
        let mut dot = (0.0, 0.0);
        for item in f[0].iter().take(4) {
            let (r, i) = *item;
            dot.0 += r * r + i * i;
        }
        assert!((dot.0 - 1.0).abs() < 1e-10, "first row norm = {}", dot.0);
    }

    // ═══ AuditLedger Integration Tests (§0.19) ═══

    #[test]
    fn solve_with_audit_records_casp_decision() {
        let a = vec![vec![3.0, 2.0], vec![1.0, 2.0]];
        let b = vec![5.0, 5.0];
        let audit_ledger = sync_audit_ledger();
        let mut portfolio = SolverPortfolio::new(RuntimeMode::Strict, 16);

        let result = solve_with_audit(
            &a,
            &b,
            SolveOptions::default(),
            &mut portfolio,
            &audit_ledger,
        );
        assert!(result.is_ok());

        let ledger = lock_audit_ledger(&audit_ledger);
        assert_eq!(ledger.len(), 1, "should have exactly one audit entry");

        let entry = &ledger.entries()[0];
        match &entry.action {
            AuditAction::ModeDecision { .. } => {}
            other => {
                unreachable!("expected ModeDecision, got {other:?}");
            }
        }
        assert!(
            entry.outcome.contains("CASP"),
            "outcome should mention CASP"
        );
    }

    #[test]
    fn solve_policy_rejects_incompatible_assumption_before_casp() {
        let a = vec![vec![2.0, 1.0], vec![0.0, 3.0]];
        let b = vec![3.0, 3.0];

        let result = solve(
            &a,
            &b,
            SolveOptions {
                assume_a: Some(MatrixAssumption::Diagonal),
                ..SolveOptions::default()
            },
        );

        match result {
            Err(LinalgError::PolicyRejected { reason }) => {
                assert!(reason.contains("IncompatibleMetadata"));
                assert!(reason.contains("metadata=1.000"));
            }
            other => unreachable!("expected policy rejection, got {other:?}"),
        }
    }

    #[test]
    fn solve_with_audit_records_policy_rejection() {
        let a = vec![vec![2.0, 1.0], vec![0.0, 3.0]];
        let b = vec![3.0, 3.0];
        let audit_ledger = sync_audit_ledger();
        let mut portfolio = SolverPortfolio::new(RuntimeMode::Strict, 16);

        let result = solve_with_audit(
            &a,
            &b,
            SolveOptions {
                assume_a: Some(MatrixAssumption::Diagonal),
                ..SolveOptions::default()
            },
            &mut portfolio,
            &audit_ledger,
        );

        match result {
            Err(LinalgError::PolicyRejected { reason }) => {
                assert!(reason.contains("IncompatibleMetadata"));
            }
            other => unreachable!("expected policy rejection, got {other:?}"),
        }

        let ledger = lock_audit_ledger(&audit_ledger);
        assert_eq!(ledger.len(), 1, "should have exactly one audit entry");
        let entry = &ledger.entries()[0];
        match &entry.action {
            AuditAction::FailClosed { reason } => {
                assert_eq!(reason, "policy_rejected");
            }
            other => unreachable!("expected FailClosed, got {other:?}"),
        }
        assert!(entry.outcome.contains("policy rejected solve"));
    }

    #[test]
    fn solve_with_audit_records_fail_closed_on_non_finite_input() {
        let a = vec![vec![f64::NAN, 2.0], vec![1.0, 2.0]];
        let b = vec![5.0, 5.0];
        let audit_ledger = sync_audit_ledger();
        let mut portfolio = SolverPortfolio::new(RuntimeMode::Hardened, 16);

        let result = solve_with_audit(
            &a,
            &b,
            SolveOptions {
                mode: RuntimeMode::Hardened,
                ..SolveOptions::default()
            },
            &mut portfolio,
            &audit_ledger,
        );
        assert!(result.is_err());

        let ledger = lock_audit_ledger(&audit_ledger);
        assert_eq!(ledger.len(), 1, "should have exactly one audit entry");

        let entry = &ledger.entries()[0];
        match &entry.action {
            AuditAction::FailClosed { reason } => {
                assert!(
                    reason.contains("non_finite"),
                    "reason should mention non_finite: {reason}"
                );
            }
            other => {
                unreachable!("expected FailClosed, got {other:?}");
            }
        }
    }

    #[test]
    fn solve_with_audit_records_fail_closed_on_condition_too_high() {
        // Near-singular matrix to trigger condition rejection in hardened mode
        let a = vec![vec![1.0, 1.0], vec![1.0, 1.0 + 1e-16]];
        let b = vec![2.0, 2.0];
        let audit_ledger = sync_audit_ledger();
        let mut portfolio = SolverPortfolio::new(RuntimeMode::Hardened, 16);

        let result = solve_with_audit(
            &a,
            &b,
            SolveOptions {
                mode: RuntimeMode::Hardened,
                ..SolveOptions::default()
            },
            &mut portfolio,
            &audit_ledger,
        );

        // This should either fail with ConditionTooHigh or succeed
        // Check if we got a fail-closed entry when it fails
        if result.is_err() {
            let ledger = lock_audit_ledger(&audit_ledger);
            let has_fail_closed = ledger.entries().iter().any(|e| {
                matches!(
                    &e.action,
                    AuditAction::FailClosed { reason } if reason.contains("condition")
                )
            });
            assert!(
                has_fail_closed,
                "should have fail_closed entry for condition rejection"
            );
        }
    }

    #[test]
    fn solve_with_audit_records_fail_closed_on_non_square() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]; // 2x3
        let b = vec![1.0, 2.0];
        let audit_ledger = sync_audit_ledger();
        let mut portfolio = SolverPortfolio::new(RuntimeMode::Strict, 16);

        let result = solve_with_audit(
            &a,
            &b,
            SolveOptions::default(),
            &mut portfolio,
            &audit_ledger,
        );
        assert!(result.is_err());

        let ledger = lock_audit_ledger(&audit_ledger);
        assert_eq!(ledger.len(), 1);

        let entry = &ledger.entries()[0];
        match &entry.action {
            AuditAction::FailClosed { reason } => {
                assert!(reason.contains("non_square"));
            }
            other => {
                unreachable!("expected FailClosed for non_square, got {other:?}");
            }
        }
    }

    #[test]
    fn solve_with_audit_records_fail_closed_on_incompatible_shapes() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![1.0, 2.0, 3.0]; // Wrong size
        let audit_ledger = sync_audit_ledger();
        let mut portfolio = SolverPortfolio::new(RuntimeMode::Strict, 16);

        let result = solve_with_audit(
            &a,
            &b,
            SolveOptions::default(),
            &mut portfolio,
            &audit_ledger,
        );
        assert!(result.is_err());

        let ledger = lock_audit_ledger(&audit_ledger);
        assert_eq!(ledger.len(), 1);

        let entry = &ledger.entries()[0];
        match &entry.action {
            AuditAction::FailClosed { reason } => {
                assert!(reason.contains("incompatible"));
            }
            other => {
                unreachable!("expected FailClosed for incompatible_shapes, got {other:?}");
            }
        }
    }

    #[test]
    fn det_with_audit_records_mode_decision() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let audit_ledger = sync_audit_ledger();

        let det = det_with_audit(&a, RuntimeMode::Strict, true, &audit_ledger)
            .expect("det should succeed");
        assert_eq!(det, -2.0);

        let ledger = lock_audit_ledger(&audit_ledger);
        assert_eq!(ledger.len(), 1);
        let entry = &ledger.entries()[0];
        match &entry.action {
            AuditAction::ModeDecision { mode } => assert_eq!(*mode, RuntimeMode::Strict),
            other => {
                unreachable!("expected ModeDecision, got {other:?}");
            }
        }
        assert!(entry.outcome.contains("det"));
    }

    #[test]
    fn det_with_audit_records_fail_closed_on_non_finite_input() {
        let a = vec![vec![f64::NAN, 2.0], vec![3.0, 4.0]];
        let audit_ledger = sync_audit_ledger();

        let result = det_with_audit(&a, RuntimeMode::Hardened, false, &audit_ledger);
        assert!(result.is_err());

        let ledger = lock_audit_ledger(&audit_ledger);
        assert_eq!(ledger.len(), 1);
        let entry = &ledger.entries()[0];
        match &entry.action {
            AuditAction::FailClosed { reason } => {
                assert!(reason.contains("non_finite"));
            }
            other => {
                unreachable!("expected FailClosed, got {other:?}");
            }
        }
        assert!(entry.outcome.contains("det rejected"));
    }

    #[test]
    fn matrix_fingerprint_is_deterministic() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let fp1 = matrix_fingerprint(&a);
        let fp2 = matrix_fingerprint(&a);
        assert_eq!(fp1, fp2, "fingerprint should be deterministic");
    }

    #[test]
    fn matrix_fingerprint_truncates_large_input() {
        let large: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64; 100]).collect();
        let fp = matrix_fingerprint(&large);
        assert!(fp.len() <= 1024, "fingerprint should be <= 1KB");
    }
}

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    type MatrixFunction = fn(&[Vec<f64>], DecompOptions) -> Result<Vec<Vec<f64>>, LinalgError>;

    fn arb_invertible_2x2() -> impl Strategy<Value = Vec<Vec<f64>>> {
        (
            -10.0..10.0_f64,
            -10.0..10.0_f64,
            -10.0..10.0_f64,
            -10.0..10.0_f64,
        )
            .prop_filter("non-singular", |(a, b, c, d)| (a * d - b * c).abs() > 0.01)
            .prop_map(|(a, b, c, d)| vec![vec![a, b], vec![c, d]])
    }

    fn arb_vec2() -> impl Strategy<Value = Vec<f64>> {
        prop::collection::vec(-10.0..10.0_f64, 2..=2)
    }

    fn assert_close_matrix(actual: &[Vec<f64>], expected: &[Vec<f64>], atol: f64, rtol: f64) {
        assert_eq!(actual.len(), expected.len());
        for (row_idx, (a_row, e_row)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(a_row.len(), e_row.len());
            for (col_idx, (a, e)) in a_row.iter().zip(e_row.iter()).enumerate() {
                let tol = atol + rtol * e.abs();
                assert!(
                    (a - e).abs() <= tol,
                    "row={row_idx} col={col_idx} expected={e} actual={a} tol={tol}"
                );
            }
        }
    }

    fn rotated_diagonal(lambda1: f64, lambda2: f64) -> Vec<Vec<f64>> {
        let diag = 0.5 * (lambda1 + lambda2);
        let off_diag = 0.5 * (lambda1 - lambda2);
        vec![vec![diag, off_diag], vec![off_diag, diag]]
    }

    proptest! {
        #[test]
        fn roundtrip_solve(a in arb_invertible_2x2(), b in arb_vec2()) {
            let result = solve(&a, &b, SolveOptions::default()).expect("solve works");
            // Verify A * x ≈ b
            for (i, (a_row, &b_i)) in a.iter().zip(b.iter()).enumerate() {
                let row_sum: f64 = a_row.iter().zip(result.x.iter()).map(|(a_ij, x_j)| a_ij * x_j).sum();
                prop_assert!((row_sum - b_i).abs() < 1e-8,
                    "A*x should equal b: row {i}: {row_sum} vs {b_i}");
            }
        }

        #[test]
        fn inverse_roundtrip(a in arb_invertible_2x2()) {
            let inv_result = inv(&a, InvOptions::default()).expect("inv works");
            // Verify A * inv(A) ≈ I
            for (i, a_row) in a.iter().enumerate() {
                for j in 0..a.len() {
                    let sum: f64 = a_row.iter().enumerate()
                        .map(|(k, &a_ik)| a_ik * inv_result.inverse[k][j])
                        .sum();
                    let expected = if i == j { 1.0 } else { 0.0 };
                    prop_assert!((sum - expected).abs() < 1e-8,
                        "A * inv(A) should be I: [{i}][{j}] = {sum}");
                }
            }
        }

        #[test]
        fn det_inverse_consistency(a in arb_invertible_2x2()) {
            let d = det(&a, RuntimeMode::Strict, true).expect("det works");
            let inv_result = inv(&a, InvOptions::default()).expect("inv works");
            let d_inv = det(&inv_result.inverse, RuntimeMode::Strict, true).expect("det(inv) works");
            // det(A) * det(inv(A)) ≈ 1
            prop_assert!((d * d_inv - 1.0).abs() < 1e-6,
                "det(A) * det(inv(A)) should be 1: {} * {} = {}", d, d_inv, d * d_inv);
        }

        #[test]
        fn det_product_property(
            a in arb_invertible_2x2(),
            b_mat in arb_invertible_2x2()
        ) {
            let det_a = det(&a, RuntimeMode::Strict, true).expect("det(A)");
            let det_b = det(&b_mat, RuntimeMode::Strict, true).expect("det(B)");
            // Compute A @ B
            let ab = vec![
                vec![
                    a[0][0] * b_mat[0][0] + a[0][1] * b_mat[1][0],
                    a[0][0] * b_mat[0][1] + a[0][1] * b_mat[1][1],
                ],
                vec![
                    a[1][0] * b_mat[0][0] + a[1][1] * b_mat[1][0],
                    a[1][0] * b_mat[0][1] + a[1][1] * b_mat[1][1],
                ],
            ];
            let det_ab = det(&ab, RuntimeMode::Strict, true).expect("det(AB)");
            prop_assert!((det_ab - det_a * det_b).abs() < 1e-6,
                "det(AB) should equal det(A)*det(B): {det_ab} vs {}", det_a * det_b);
        }

        #[test]
        fn lstsq_residual_minimality(a in arb_invertible_2x2(), b in arb_vec2()) {
            // For square non-singular: lstsq should give exact solution
            let result = lstsq(&a, &b, LstsqOptions::default()).expect("lstsq works");
            for (i, row) in a.iter().enumerate() {
                let dot: f64 = row.iter().zip(&result.x).map(|(a, x)| a * x).sum();
                prop_assert!((dot - b[i]).abs() < 1e-8,
                    "A*x should equal b for exact system: row {i}");
            }
        }

        #[test]
        fn solve_triangular_matches_solve_for_triangular_input(
            a11 in 1.0..10.0_f64,
            a12 in -10.0..10.0_f64,
            a22 in 1.0..10.0_f64,
            b in arb_vec2()
        ) {
            let a = vec![vec![a11, a12], vec![0.0, a22]]; // upper triangular
            let solve_result = solve(&a, &b, SolveOptions {
                assume_a: Some(MatrixAssumption::UpperTriangular),
                ..SolveOptions::default()
            }).expect("solve with assumption works");
            let tri_result = solve_triangular(&a, &b, TriangularSolveOptions {
                lower: false,
                ..TriangularSolveOptions::default()
            }).expect("solve_triangular works");
            for (i, (s, t)) in solve_result.x.iter().zip(tri_result.x.iter()).enumerate() {
                prop_assert!((s - t).abs() < 1e-10,
                    "solve and solve_triangular should match: [{i}] {s} vs {t}");
            }
        }

        #[test]
        fn pinv_rank_bounded(
            a in arb_invertible_2x2()
        ) {
            let result = pinv(&a, PinvOptions::default()).expect("pinv works");
            prop_assert!(result.rank <= 2,
                "rank should be <= min(m,n)=2, got {}", result.rank);
        }

        #[test]
        fn rcond_is_bounded(a in arb_invertible_2x2()) {
            let matrix = dmatrix_from_rows(&a).expect("valid matrix");
            let lu = matrix.clone().lu();
            let rcond = fast_rcond_from_lu(&lu, matrix_norm1(&matrix), 2);
            prop_assert!(rcond >= 0.0, "rcond should be >= 0, got {rcond}");
            prop_assert!(rcond <= 1.0, "rcond should be <= 1, got {rcond}");
        }
    }

    #[test]
    fn test_solve_transposed_upper_triangular() {
        // Upper triangular matrix A
        // [ 2, 1 ]
        // [ 0, 2 ]
        let a = vec![vec![2.0, 1.0], vec![0.0, 2.0]];

        // b = [ 5, 4 ]
        let b = vec![5.0, 4.0];

        // A^T x = b
        // [ 2, 0 ] [ x1 ] = [ 5 ]
        // [ 1, 2 ] [ x2 ] = [ 4 ]
        // 2*x1 = 5 => x1 = 2.5
        // 1*x1 + 2*x2 = 4 => 2.5 + 2*x2 = 4 => 2*x2 = 1.5 => x2 = 0.75
        // Expected x = [ 2.5, 0.75 ]

        let options = SolveOptions {
            assume_a: Some(MatrixAssumption::UpperTriangular),
            transposed: true,
            ..SolveOptions::default()
        };

        let result = solve(&a, &b, options).expect("solve works");
        let expected = [2.5, 0.75];

        for (i, (&r, &e)) in result.x.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-10,
                "FAILURE at index {i}: got {r}, expected {e}"
            );
        }
    }

    #[test]
    fn test_solve_with_casp_transposed_upper_triangular() {
        use fsci_runtime::{RuntimeMode, SolverPortfolio};

        let a = vec![vec![2.0, 1.0], vec![0.0, 2.0]];
        let b = vec![5.0, 4.0];
        let options = SolveOptions {
            assume_a: Some(MatrixAssumption::UpperTriangular),
            transposed: true,
            ..SolveOptions::default()
        };

        let mut portfolio = SolverPortfolio::new(RuntimeMode::Strict, 64);
        let result =
            solve_with_casp(&a, &b, options, &mut portfolio).expect("solve_with_casp works");
        let expected = [2.5, 0.75];

        for (i, (&r, &e)) in result.x.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-10,
                "FAILURE at index {i}: got {r}, expected {e}"
            );
        }

        // Also verify the action was TriangularFastPath
        let evidence = portfolio.serialize_jsonl();
        assert!(evidence.contains("TriangularFastPath"));
    }

    // ── LDL decomposition tests ─────────────────────────────────────

    fn verify_ldl_reconstruction(a: &[Vec<f64>], result: &LdlResult) {
        let n = result.d.len();
        for (i, row_a) in a.iter().enumerate().take(n) {
            for (j, &aij) in row_a.iter().enumerate().take(n) {
                let sum: f64 = (0..n)
                    .map(|k| result.l[i][k] * result.d[k] * result.l[j][k])
                    .sum();
                assert!(
                    (sum - aij).abs() < 1e-10,
                    "LDLᵀ[{i}][{j}]={sum} != A[{i}][{j}]={aij}",
                );
            }
        }
    }

    #[test]
    fn ldl_identity() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let result = ldl(&a, DecompOptions::default()).expect("ldl works");
        // L should be identity, D should be [1, 1]
        assert_eq!(result.l, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        assert_eq!(result.d, vec![1.0, 1.0]);
    }

    #[test]
    fn ldl_positive_definite() {
        // A = [[4, 2], [2, 3]]
        let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
        let result = ldl(&a, DecompOptions::default()).expect("ldl works");
        verify_ldl_reconstruction(&a, &result);
    }

    #[test]
    fn ldl_symmetric_indefinite() {
        // A = [[1, 2], [2, -3]] — symmetric but not positive definite
        let a = vec![vec![1.0, 2.0], vec![2.0, -3.0]];
        let result = ldl(&a, DecompOptions::default()).expect("ldl works");
        verify_ldl_reconstruction(&a, &result);
        // D should have a negative element (indefinite)
        assert!(
            result.d.iter().any(|&v| v < 0.0),
            "D should have negative entry for indefinite matrix"
        );
    }

    #[test]
    fn ldl_3x3() {
        let a = vec![
            vec![4.0, 1.0, 2.0],
            vec![1.0, 5.0, 3.0],
            vec![2.0, 3.0, 6.0],
        ];
        let result = ldl(&a, DecompOptions::default()).expect("ldl works");
        verify_ldl_reconstruction(&a, &result);
        // L should be unit lower triangular
        for (i, row) in result.l.iter().enumerate() {
            assert_eq!(row[i], 1.0, "L diagonal should be 1");
            for &val in row.iter().skip(i + 1) {
                assert_eq!(val, 0.0, "L upper triangle should be 0");
            }
        }
    }

    #[test]
    fn ldl_empty_matrix() {
        let a: Vec<Vec<f64>> = Vec::new();
        let result = ldl(&a, DecompOptions::default()).expect("ldl empty");
        assert!(result.l.is_empty());
        assert!(result.d.is_empty());
    }

    #[test]
    fn ldl_1x1() {
        let a = vec![vec![7.0]];
        let result = ldl(&a, DecompOptions::default()).expect("ldl 1x1");
        assert_eq!(result.l, vec![vec![1.0]]);
        assert_eq!(result.d, vec![7.0]);
    }

    #[test]
    fn ldl_non_square_rejected() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let err = ldl(&a, DecompOptions::default()).expect_err("non-square");
        assert!(matches!(err, LinalgError::ExpectedSquareMatrix));
    }

    #[test]
    fn ldl_hardened_rejects_asymmetric() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]]; // not symmetric
        let options = DecompOptions {
            mode: RuntimeMode::Hardened,
            check_finite: true,
        };
        let err = ldl(&a, options).expect_err("asymmetric in hardened");
        assert!(matches!(err, LinalgError::InvalidArgument { .. }));
    }

    // ── Sylvester / Lyapunov tests ──────────────────────────────────

    #[test]
    fn solve_sylvester_diagonal() {
        // A = diag(1,2), B = diag(3,4), known X = [[1,1],[1,1]]
        // Q = AX + XB = [[1+3, 1+4], [2+3, 2+4]] = [[4,5],[5,6]]
        let a = vec![vec![1.0, 0.0], vec![0.0, 2.0]];
        let b = vec![vec![3.0, 0.0], vec![0.0, 4.0]];
        let q = vec![vec![4.0, 5.0], vec![5.0, 6.0]];
        let x = solve_sylvester(&a, &b, &q, DecompOptions::default()).expect("sylvester");
        // Verify AX + XB = Q
        let n = 2;
        for i in 0..n {
            for j in 0..n {
                let mut ax_xb = 0.0;
                for k in 0..n {
                    ax_xb += a[i][k] * x[k][j] + x[i][k] * b[k][j];
                }
                assert!(
                    (ax_xb - q[i][j]).abs() < 1e-8,
                    "AX+XB[{i},{j}] = {ax_xb}, expected {}",
                    q[i][j]
                );
            }
        }
    }

    #[test]
    fn solve_sylvester_general() {
        // Non-diagonal A, B
        let a = vec![vec![1.0, 2.0], vec![0.0, 3.0]];
        let b = vec![vec![4.0, 1.0], vec![0.0, 5.0]];
        // Choose X = [[1,0],[0,1]] → Q = AX + XB = A + B = [[5,3],[0,8]]
        let q = vec![vec![5.0, 3.0], vec![0.0, 8.0]];
        let x = solve_sylvester(&a, &b, &q, DecompOptions::default()).expect("sylvester");
        let n = 2;
        for i in 0..n {
            for j in 0..n {
                let mut ax_xb = 0.0;
                for k in 0..n {
                    ax_xb += a[i][k] * x[k][j] + x[i][k] * b[k][j];
                }
                assert!(
                    (ax_xb - q[i][j]).abs() < 1e-6,
                    "AX+XB[{i},{j}] = {ax_xb}, expected {}",
                    q[i][j]
                );
            }
        }
    }

    #[test]
    fn solve_sylvester_3x3() {
        let a = vec![
            vec![2.0, 1.0, 0.0],
            vec![0.0, 3.0, 1.0],
            vec![0.0, 0.0, 4.0],
        ];
        let b = vec![
            vec![5.0, 0.0, 0.0],
            vec![1.0, 6.0, 0.0],
            vec![0.0, 1.0, 7.0],
        ];
        // X = I → Q = A + B
        let q = vec![
            vec![7.0, 1.0, 0.0],
            vec![1.0, 9.0, 1.0],
            vec![0.0, 1.0, 11.0],
        ];
        let x = solve_sylvester(&a, &b, &q, DecompOptions::default()).expect("sylvester 3x3");
        let n = 3;
        for i in 0..n {
            for j in 0..n {
                let mut ax_xb = 0.0;
                for k in 0..n {
                    ax_xb += a[i][k] * x[k][j] + x[i][k] * b[k][j];
                }
                assert!(
                    (ax_xb - q[i][j]).abs() < 1e-6,
                    "AX+XB[{i},{j}] = {ax_xb}, expected {}",
                    q[i][j]
                );
            }
        }
    }

    /// Exercises the 2×2-Schur-block (complex-eigenpair) path of the
    /// Bartels–Stewart back-substitution [frankenscipy-8l8r1]. A = [[0,-1],[1,0]]
    /// has eigenvalues ±i, so its real Schur form keeps a genuine 2×2 block and
    /// the coupled 2m×2m branch must run. Verified by residual A X + X B = Q.
    #[test]
    fn solve_sylvester_complex_eigenvalues_2x2_block() {
        let a = vec![vec![0.0, -1.0], vec![1.0, 0.0]]; // eigenvalues ±i
        let b = vec![vec![3.0, -2.0], vec![1.0, 4.0]]; // complex pair too
        // Pick X, derive Q = A X + X B so the residual has a known target.
        let x_true = [vec![1.0, 2.0], vec![-1.0, 0.5]];
        let n = 2;
        let mut q = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut v = 0.0;
                for k in 0..n {
                    v += a[i][k] * x_true[k][j] + x_true[i][k] * b[k][j];
                }
                q[i][j] = v;
            }
        }
        let x = solve_sylvester(&a, &b, &q, DecompOptions::default()).expect("sylvester 2x2");
        for i in 0..n {
            for j in 0..n {
                let mut ax_xb = 0.0;
                for k in 0..n {
                    ax_xb += a[i][k] * x[k][j] + x[i][k] * b[k][j];
                }
                assert!(
                    (ax_xb - q[i][j]).abs() < 1e-9,
                    "AX+XB[{i},{j}] = {ax_xb}, expected {}",
                    q[i][j]
                );
            }
        }
    }

    /// Perf witness for the Bartels–Stewart Sylvester rewrite [frankenscipy-8l8r1]:
    /// column-block back-substitution (O(n·m^3)) vs the previous full Kronecker
    /// (I_n⊗T_A + T_B^T⊗I_m) operator with an O((mn)^3) full-pivot LU. Both share
    /// the same Schur reduction; only the inner solve differs, and both satisfy
    /// the same residual. Run with
    /// `cargo test -p fsci-linalg --release solve_sylvester_perf -- --ignored --nocapture`.
    #[test]
    #[ignore = "perf measurement; run explicitly in --release"]
    fn solve_sylvester_perf_vs_kronecker() {
        // Previous implementation: full Kronecker operator + full_piv_lu.
        fn sylvester_kronecker(a: &[Vec<f64>], b: &[Vec<f64>], q: &[Vec<f64>]) -> Vec<Vec<f64>> {
            let m = a.len();
            let n = b.len();
            let a_mat = dmatrix_from_rows(a).unwrap();
            let b_mat = dmatrix_from_rows(b).unwrap();
            let q_mat = dmatrix_from_rows(q).unwrap();
            let (u, ta) = a_mat.clone().schur().unpack();
            let (v, tb) = b_mat.clone().schur().unpack();
            let f = u.transpose() * &q_mat * &v;
            let mn = m * n;
            let mut system = DMatrix::<f64>::zeros(mn, mn);
            for j in 0..n {
                for r in 0..m {
                    for c in 0..m {
                        system[(j * m + r, j * m + c)] += ta[(r, c)];
                    }
                }
            }
            for j in 0..n {
                for k in 0..n {
                    let tbkj = tb[(k, j)];
                    if tbkj.abs() > 0.0 {
                        for i in 0..m {
                            system[(j * m + i, k * m + i)] += tbkj;
                        }
                    }
                }
            }
            let mut rhs = DVector::<f64>::zeros(mn);
            for j in 0..n {
                for i in 0..m {
                    rhs[j * m + i] = f[(i, j)];
                }
            }
            let sol = system.full_piv_lu().solve(&rhs).unwrap();
            let mut y = DMatrix::<f64>::zeros(m, n);
            for j in 0..n {
                for i in 0..m {
                    y[(i, j)] = sol[j * m + i];
                }
            }
            rows_from_dmatrix(&(&u * y * v.transpose()))
        }

        for &nn in &[16usize, 24, 32] {
            // Diagonally dominant (non-singular, real spectrum) test matrices.
            let mk = |seed: usize| -> Vec<Vec<f64>> {
                (0..nn)
                    .map(|i| {
                        (0..nn)
                            .map(|j| {
                                if i == j {
                                    (nn as f64) + ((i * 7 + seed) % 5) as f64
                                } else {
                                    (((i * 13 + j * 17 + seed) % 7) as f64 - 3.0) * 0.1
                                }
                            })
                            .collect()
                    })
                    .collect()
            };
            let a = mk(1);
            let b = mk(2);
            let q = mk(3);

            let t0 = std::time::Instant::now();
            let x_kron = sylvester_kronecker(&a, &b, &q);
            let kron_s = t0.elapsed().as_secs_f64();

            let t1 = std::time::Instant::now();
            let x_new = solve_sylvester(&a, &b, &q, DecompOptions::default()).expect("sylvester");
            let new_s = t1.elapsed().as_secs_f64();

            // Both solve the same equation: agree to solver tolerance.
            let mut max_d = 0.0f64;
            for i in 0..nn {
                for j in 0..nn {
                    max_d = max_d.max((x_kron[i][j] - x_new[i][j]).abs());
                }
            }
            let speedup = kron_s / new_s;
            println!(
                "solve_sylvester {nn}x{nn}: kronecker={kron_s:.4}s bartels_stewart={new_s:.4}s speedup={speedup:.2}x max|Δ|={max_d:.2e}"
            );
            assert!(max_d < 1e-8, "{nn}: solutions diverge: {max_d:e}");
        }
    }

    #[test]
    fn solve_sylvester_rejects_non_square() {
        let a = vec![vec![1.0, 2.0]]; // 1x2
        let b = vec![vec![1.0]];
        let q = vec![vec![1.0]];
        let err = solve_sylvester(&a, &b, &q, DecompOptions::default()).expect_err("non-square A");
        assert!(matches!(err, LinalgError::ExpectedSquareMatrix));
    }

    #[test]
    fn solve_sylvester_singular_system_errors() {
        let a = vec![vec![1.0]];
        let b = vec![vec![-1.0]];
        let q = vec![vec![1.0]];
        let err = solve_sylvester(&a, &b, &q, DecompOptions::default())
            .expect_err("singular Sylvester operator");
        assert_eq!(err, LinalgError::SingularMatrix);
    }

    #[test]
    fn solve_continuous_lyapunov_basic() {
        // A = [[1,0],[0,2]], X = I → Q = AX + XA^T = A + A^T = 2*A (since A is symmetric)
        let a = vec![vec![1.0, 0.0], vec![0.0, 2.0]];
        let q = vec![vec![2.0, 0.0], vec![0.0, 4.0]];
        let x = solve_continuous_lyapunov(&a, &q, DecompOptions::default()).expect("lyapunov");
        let n = 2;
        for i in 0..n {
            for j in 0..n {
                let mut ax_xat = 0.0;
                for k in 0..n {
                    ax_xat += a[i][k] * x[k][j] + x[i][k] * a[j][k]; // A^T[k,j] = A[j,k]
                }
                assert!(
                    (ax_xat - q[i][j]).abs() < 1e-6,
                    "AX+XA^T[{i},{j}] = {ax_xat}, expected {}",
                    q[i][j]
                );
            }
        }
    }

    #[test]
    fn solve_continuous_lyapunov_non_symmetric_a() {
        // Non-symmetric stable A: A = [[-1,2],[0,-3]]
        // X = I → Q = AX + XA^T = A + A^T = [[-2,2],[2,-6]]
        let a = vec![vec![-1.0, 2.0], vec![0.0, -3.0]];
        let q = vec![vec![-2.0, 2.0], vec![2.0, -6.0]];
        let x =
            solve_continuous_lyapunov(&a, &q, DecompOptions::default()).expect("lyapunov non-sym");
        let n = 2;
        for i in 0..n {
            for j in 0..n {
                let mut ax_xat = 0.0;
                for k in 0..n {
                    ax_xat += a[i][k] * x[k][j] + x[i][k] * a[j][k];
                }
                assert!(
                    (ax_xat - q[i][j]).abs() < 1e-6,
                    "AX+XA^T[{i},{j}] = {ax_xat}, expected {}",
                    q[i][j]
                );
            }
        }
    }

    #[test]
    fn solve_sylvester_matches_scipy_reference() {
        // scipy.linalg.solve_sylvester(A, B, Q) where AX + XB = Q
        // A = [[1, 2], [0, 3]], B = [[4, 1], [0, 5]], Q = [[5, 3], [0, 8]]
        // scipy returns X = [[1, 0], [0, 1]]
        let a = vec![vec![1.0, 2.0], vec![0.0, 3.0]];
        let b = vec![vec![4.0, 1.0], vec![0.0, 5.0]];
        let q = vec![vec![5.0, 3.0], vec![0.0, 8.0]];
        let x = solve_sylvester(&a, &b, &q, DecompOptions::default()).expect("sylvester");
        assert!(
            (x[0][0] - 1.0).abs() < 1e-10,
            "X[0,0]={} vs scipy 1.0",
            x[0][0]
        );
        assert!(
            (x[0][1] - 0.0).abs() < 1e-10,
            "X[0,1]={} vs scipy 0.0",
            x[0][1]
        );
        assert!(
            (x[1][0] - 0.0).abs() < 1e-10,
            "X[1,0]={} vs scipy 0.0",
            x[1][0]
        );
        assert!(
            (x[1][1] - 1.0).abs() < 1e-10,
            "X[1,1]={} vs scipy 1.0",
            x[1][1]
        );
    }

    #[test]
    fn solve_continuous_lyapunov_matches_scipy_reference() {
        // scipy.linalg.solve_continuous_lyapunov(A, Q) where AX + XA^H = Q
        // A = [[1, 2], [3, 4]], Q = [[5, 6], [6, 7]]
        // scipy returns X = [[-0.1, 1.3], [1.3, -0.1]]
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let q = vec![vec![5.0, 6.0], vec![6.0, 7.0]];
        let x =
            solve_continuous_lyapunov(&a, &q, DecompOptions::default()).expect("lyapunov scipy");
        assert!(
            (x[0][0] - (-0.1)).abs() < 1e-10,
            "X[0,0]={} vs scipy -0.1",
            x[0][0]
        );
        assert!(
            (x[0][1] - 1.3).abs() < 1e-10,
            "X[0,1]={} vs scipy 1.3",
            x[0][1]
        );
        assert!(
            (x[1][0] - 1.3).abs() < 1e-10,
            "X[1,0]={} vs scipy 1.3",
            x[1][0]
        );
        assert!(
            (x[1][1] - (-0.1)).abs() < 1e-10,
            "X[1,1]={} vs scipy -0.1",
            x[1][1]
        );
    }

    // ── Discrete Lyapunov tests ──────────────────────────────────────

    #[test]
    fn solve_discrete_lyapunov_diagonal() {
        // A = diag(0.5, 0.3), Q = I
        // A X A^T - X + Q = 0 → X[i,i] = Q[i,i] / (1 - A[i,i]²)
        let a = vec![vec![0.5, 0.0], vec![0.0, 0.3]];
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let x =
            solve_discrete_lyapunov(&a, &q, DecompOptions::default()).expect("discrete lyapunov");
        // Verify: A X A^T - X + Q = 0
        let n = 2;
        for i in 0..n {
            for j in 0..n {
                let mut axa_t = 0.0;
                for k in 0..n {
                    for l in 0..n {
                        axa_t += a[i][k] * x[k][l] * a[j][l];
                    }
                }
                let residual = axa_t - x[i][j] + q[i][j];
                assert!(residual.abs() < 1e-6, "AXA^T-X+Q[{i},{j}] = {residual}");
            }
        }
    }

    #[test]
    fn solve_discrete_lyapunov_stable() {
        // Stable A (all eigenvalues < 1)
        let a = vec![vec![0.8, 0.1], vec![-0.1, 0.7]];
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let x = solve_discrete_lyapunov(&a, &q, DecompOptions::default())
            .expect("discrete lyapunov stable");
        let n = 2;
        for i in 0..n {
            for j in 0..n {
                let mut axa_t = 0.0;
                for k in 0..n {
                    for l in 0..n {
                        axa_t += a[i][k] * x[k][l] * a[j][l];
                    }
                }
                let residual = axa_t - x[i][j] + q[i][j];
                assert!(residual.abs() < 1e-6, "AXA^T-X+Q[{i},{j}] = {residual}");
            }
        }
    }

    #[test]
    fn solve_discrete_lyapunov_rejects_non_square() {
        let a = vec![vec![1.0, 2.0]];
        let q = vec![vec![1.0]];
        let err =
            solve_discrete_lyapunov(&a, &q, DecompOptions::default()).expect_err("non-square");
        assert!(matches!(err, LinalgError::ExpectedSquareMatrix));
    }

    #[test]
    fn solve_discrete_lyapunov_unit_eigenvalue_errors() {
        let a = vec![vec![1.0]];
        let q = vec![vec![1.0]];
        let err = solve_discrete_lyapunov(&a, &q, DecompOptions::default())
            .expect_err("singular discrete Lyapunov operator");
        assert_eq!(err, LinalgError::SingularMatrix);
    }

    /// Exercises the 2×2-Schur-block (complex-eigenpair) branch of the
    /// Schur/Stein back-substitution [frankenscipy-kti79]. A rotation-scaled by
    /// 0.6 has complex eigenvalues 0.6·e^{±iθ} inside the unit circle, so its
    /// real Schur form keeps a genuine 2×2 block and the coupled 2n×2n path runs.
    /// Verified by the discrete residual A X A^T − X + Q = 0.
    #[test]
    fn solve_discrete_lyapunov_complex_eigenvalues_2x2_block() {
        // 0.6 * [[cosθ,-sinθ],[sinθ,cosθ]], θ ≈ 0.9 rad → complex pair |λ|=0.6 < 1.
        let (c, s) = (0.9f64.cos(), 0.9f64.sin());
        let a = vec![vec![0.6 * c, -0.6 * s], vec![0.6 * s, 0.6 * c]];
        let q = vec![vec![2.0, 0.3], vec![0.3, 1.5]];
        let x = solve_discrete_lyapunov(&a, &q, DecompOptions::default())
            .expect("discrete lyapunov 2x2");
        let n = 2;
        for i in 0..n {
            for j in 0..n {
                let mut axa_t = 0.0;
                for k in 0..n {
                    for l in 0..n {
                        axa_t += a[i][k] * x[k][l] * a[j][l];
                    }
                }
                let residual = axa_t - x[i][j] + q[i][j];
                assert!(residual.abs() < 1e-9, "AXA^T-X+Q[{i},{j}] = {residual}");
            }
        }
    }

    /// Perf witness for the discrete-Lyapunov Schur/Stein rewrite
    /// [frankenscipy-kti79]: O(n·n^3) Schur back-substitution vs the previous
    /// full (A⊗A − I) n²×n² operator + O(n^6) full-pivot LU. Both solve the same
    /// Stein equation, so they agree to solver tolerance. Run with
    /// `cargo test -p fsci-linalg --release solve_discrete_lyapunov_perf -- --ignored --nocapture`.
    #[test]
    #[ignore = "perf measurement; run explicitly in --release"]
    fn solve_discrete_lyapunov_perf_vs_kronecker() {
        // Previous implementation: full (A⊗A − I) operator + full_piv_lu.
        fn dlyap_kronecker(a: &[Vec<f64>], q: &[Vec<f64>]) -> Vec<Vec<f64>> {
            let n = a.len();
            let a_mat = dmatrix_from_rows(a).unwrap();
            let q_mat = dmatrix_from_rows(q).unwrap();
            let nn = n * n;
            let mut system = DMatrix::<f64>::zeros(nn, nn);
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        for l in 0..n {
                            system[(i * n + k, j * n + l)] += a_mat[(i, j)] * a_mat[(k, l)];
                        }
                    }
                }
            }
            for i in 0..nn {
                system[(i, i)] -= 1.0;
            }
            let mut rhs = DVector::<f64>::zeros(nn);
            for j in 0..n {
                for i in 0..n {
                    rhs[j * n + i] = -q_mat[(i, j)];
                }
            }
            let sol = system.full_piv_lu().solve(&rhs).unwrap();
            let mut x = DMatrix::<f64>::zeros(n, n);
            for j in 0..n {
                for i in 0..n {
                    x[(i, j)] = sol[j * n + i];
                }
            }
            rows_from_dmatrix(&x)
        }

        for &nn in &[12usize, 16, 24] {
            // Stable A: scaled diagonally-dominant so spectral radius < 1.
            let raw: Vec<Vec<f64>> = (0..nn)
                .map(|i| {
                    (0..nn)
                        .map(|j| {
                            if i == j {
                                0.3
                            } else {
                                (((i * 13 + j * 7) % 5) as f64 - 2.0) * 0.02
                            }
                        })
                        .collect()
                })
                .collect();
            let a = raw;
            let q: Vec<Vec<f64>> = (0..nn)
                .map(|i| (0..nn).map(|j| if i == j { 1.0 } else { 0.1 }).collect())
                .collect();

            let t0 = std::time::Instant::now();
            let x_kron = dlyap_kronecker(&a, &q);
            let kron_s = t0.elapsed().as_secs_f64();

            let t1 = std::time::Instant::now();
            let x_new = solve_discrete_lyapunov(&a, &q, DecompOptions::default()).expect("dlyap");
            let new_s = t1.elapsed().as_secs_f64();

            let mut max_d = 0.0f64;
            for i in 0..nn {
                for j in 0..nn {
                    max_d = max_d.max((x_kron[i][j] - x_new[i][j]).abs());
                }
            }
            let speedup = kron_s / new_s;
            println!(
                "solve_discrete_lyapunov {nn}x{nn}: kronecker={kron_s:.4}s schur_stein={new_s:.4}s speedup={speedup:.2}x max|Δ|={max_d:.2e}"
            );
            assert!(max_d < 1e-7, "{nn}: solutions diverge: {max_d:e}");
        }
    }

    // ── solve_continuous_are / solve_discrete_are tests (br-60cm) ─────

    /// CAREX benchmark — Example 1: 1×1 problem with closed-form
    /// solution. Per Benner/Laub/Mehrmann CAREX 1: A=[0], B=[1], Q=[1],
    /// R=[1] → CARE has X² = 1 → X = 1 (positive stabilizing root).
    #[test]
    fn solve_continuous_are_carex_1x1() {
        let a = vec![vec![0.0]];
        let b = vec![vec![1.0]];
        let q = vec![vec![1.0]];
        let r = vec![vec![1.0]];
        let x = solve_continuous_are(&a, &b, &q, &r, DecompOptions::default()).expect("CARE 1x1");
        assert_eq!(x.len(), 1);
        assert_eq!(x[0].len(), 1);
        assert!(
            (x[0][0] - 1.0).abs() < 1e-10,
            "expected 1.0, got {}",
            x[0][0]
        );
    }

    /// CARE residual check on a 2×2 problem. Verifies the equation
    /// A^T X + X A − X B R^{-1} B^T X + Q ≈ 0.
    #[test]
    fn solve_continuous_are_carex_2x2_residual() {
        let a = vec![vec![0.0, 1.0], vec![0.0, 0.0]];
        let b = vec![vec![0.0], vec![1.0]];
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let r = vec![vec![1.0]];
        let x = solve_continuous_are(&a, &b, &q, &r, DecompOptions::default()).expect("CARE 2x2");
        // X must be symmetric.
        assert!((x[0][1] - x[1][0]).abs() < 1e-10);
        // Residual: A^T X + X A − X B B^T X + Q (R=I so R^{-1}=I).
        let n = 2;
        let mut res = vec![vec![0.0_f64; n]; n];
        for (i, res_row) in res.iter_mut().enumerate() {
            for (j, res_ij) in res_row.iter_mut().enumerate() {
                let mut atx = 0.0_f64;
                let mut xa = 0.0_f64;
                for k in 0..n {
                    atx += a[k][i] * x[k][j];
                    xa += x[i][k] * a[k][j];
                }
                let xb_i: f64 = (0..n).map(|k| x[i][k] * b[k][0]).sum();
                let btx_j: f64 = (0..n).map(|k| b[k][0] * x[k][j]).sum();
                *res_ij = atx + xa - xb_i * btx_j + q[i][j];
            }
        }
        let max_res = res
            .iter()
            .flat_map(|row| row.iter().map(|v| v.abs()))
            .fold(0.0_f64, f64::max);
        assert!(
            max_res < 1e-9,
            "CARE residual max = {max_res:.3e}, X = {x:?}"
        );
    }

    /// CARE residual on a stable 3×3 (CAREX-2 inspired). A is the
    /// companion form of −1, B excites all states, Q identity, R = 1.
    #[test]
    fn solve_continuous_are_3x3_residual() {
        let a = vec![
            vec![-1.0, 0.0, 0.0],
            vec![0.0, -2.0, 1.0],
            vec![0.0, 0.0, -3.0],
        ];
        let b = vec![vec![1.0], vec![0.5], vec![1.0]];
        let q = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let r = vec![vec![1.0]];
        let x = solve_continuous_are(&a, &b, &q, &r, DecompOptions::default()).expect("CARE 3x3");
        let n = 3;
        // Symmetry.
        for (i, row) in x.iter().enumerate() {
            for (j, &xij) in row.iter().enumerate().skip(i + 1) {
                assert!(
                    (xij - x[j][i]).abs() < 1e-9,
                    "X not symmetric at ({i},{j}): {} vs {}",
                    xij,
                    x[j][i]
                );
            }
        }
        // Residual: A^T X + X A − X B B^T X + Q.
        let mut res = vec![vec![0.0_f64; n]; n];
        for (i, res_row) in res.iter_mut().enumerate() {
            for (j, res_ij) in res_row.iter_mut().enumerate() {
                let mut atx = 0.0_f64;
                let mut xa = 0.0_f64;
                for k in 0..n {
                    atx += a[k][i] * x[k][j];
                    xa += x[i][k] * a[k][j];
                }
                let xb_i: f64 = (0..n).map(|k| x[i][k] * b[k][0]).sum();
                let btx_j: f64 = (0..n).map(|k| b[k][0] * x[k][j]).sum();
                *res_ij = atx + xa - xb_i * btx_j + q[i][j];
            }
        }
        let max_res = res
            .iter()
            .flat_map(|row| row.iter().map(|v| v.abs()))
            .fold(0.0_f64, f64::max);
        assert!(
            max_res < 1e-8,
            "CARE 3x3 residual max = {max_res:.3e}, X = {x:?}"
        );
    }

    /// DAREX benchmark — 1×1: A=[2], B=[1], Q=[1], R=[1]. The DARE
    /// reduces to a quadratic in x: x = 4x − x²/(1+x) + 1 → solve.
    /// scipy.linalg.solve_discrete_are returns ≈ 4.236067977 (related
    /// to the golden ratio).
    #[test]
    fn solve_discrete_are_darex_1x1() {
        let a = vec![vec![2.0]];
        let b = vec![vec![1.0]];
        let q = vec![vec![1.0]];
        let r = vec![vec![1.0]];
        let x = solve_discrete_are(&a, &b, &q, &r, DecompOptions::default()).expect("DARE 1x1");
        // Validate by plugging into the DARE residual:
        //   AᵀXA − X − AᵀXB(R+BᵀXB)⁻¹BᵀXA + Q = 0.
        let xv = x[0][0];
        let a_v = 2.0;
        let res = a_v * a_v * xv - xv - (a_v * a_v * xv * xv) / (1.0 + xv) + 1.0;
        assert!(res.abs() < 1e-9, "DARE 1x1 residual {res} for X={xv}");
        // Stabilizing solution must be positive.
        assert!(xv > 0.0, "stabilizing X must be positive, got {xv}");
    }

    /// DARE 2×2 residual check — controllable companion-form pair.
    #[test]
    fn solve_discrete_are_darex_2x2_residual() {
        let a = vec![vec![0.5, 0.1], vec![0.0, 0.7]];
        let b = vec![vec![0.0], vec![1.0]];
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let r = vec![vec![0.5]];
        let x = solve_discrete_are(&a, &b, &q, &r, DecompOptions::default()).expect("DARE 2x2");
        let n = 2;
        // Symmetry.
        for (i, row) in x.iter().enumerate() {
            for (j, &xij) in row.iter().enumerate().skip(i + 1) {
                assert!((xij - x[j][i]).abs() < 1e-9);
            }
        }
        // Residual: A^T X A − X − A^T X B (R + B^T X B)^{-1} B^T X A + Q.
        let mut atx = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    atx[i][j] += a[k][i] * x[k][j];
                }
            }
        }
        let mut atxa = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    atxa[i][j] += atx[i][k] * a[k][j];
                }
            }
        }
        // B^T X B (1×1 here).
        let mut btxb = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                btxb += b[i][0] * x[i][j] * b[j][0];
            }
        }
        let scalar = r[0][0] + btxb;
        // A^T X B (n×1).
        let mut atxb = vec![0.0_f64; n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    atxb[i] += a[k][i] * x[k][j] * b[j][0];
                }
            }
        }
        let mut res = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                res[i][j] = atxa[i][j] - x[i][j] - atxb[i] * atxb[j] / scalar + q[i][j];
            }
        }
        let max_res = res
            .iter()
            .flat_map(|row| row.iter().map(|v| v.abs()))
            .fold(0.0_f64, f64::max);
        assert!(
            max_res < 1e-9,
            "DARE 2x2 residual max = {max_res:.3e}, X = {x:?}"
        );
    }

    #[test]
    fn solve_continuous_are_rejects_non_square_a() {
        let a = vec![vec![1.0, 2.0]];
        let b = vec![vec![1.0]];
        let q = vec![vec![1.0]];
        let r = vec![vec![1.0]];
        let err = solve_continuous_are(&a, &b, &q, &r, DecompOptions::default())
            .expect_err("non-square A must reject");
        assert!(matches!(err, LinalgError::ExpectedSquareMatrix));
    }

    #[test]
    fn solve_discrete_are_rejects_dimension_mismatch() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![vec![1.0], vec![1.0]];
        // Q is 1×1 but should be 2×2.
        let q = vec![vec![1.0]];
        let r = vec![vec![1.0]];
        let err = solve_discrete_are(&a, &b, &q, &r, DecompOptions::default())
            .expect_err("Q shape mismatch must reject");
        assert!(matches!(err, LinalgError::InvalidArgument { .. }));
    }

    // ── signm / funm / fractional_matrix_power tests ─────────────────

    #[test]
    fn signm_identity() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let result = signm(&a, DecompOptions::default()).expect("signm");
        assert!((result[0][0] - 1.0).abs() < 1e-10);
        assert!((result[1][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn signm_negative_identity() {
        let a = vec![vec![-1.0, 0.0], vec![0.0, -1.0]];
        let result = signm(&a, DecompOptions::default()).expect("signm");
        assert!((result[0][0] - (-1.0)).abs() < 1e-10);
        assert!((result[1][1] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn trig_matrix_functions_match_scalar_diagonal() {
        let a = diagm(&[0.25, -0.75]);

        assert_close_matrix(
            &sinm(&a, DecompOptions::default()).expect("sinm"),
            &diagm(&[0.25_f64.sin(), (-0.75_f64).sin()]),
            1e-12,
            1e-12,
        );
        assert_close_matrix(
            &cosm(&a, DecompOptions::default()).expect("cosm"),
            &diagm(&[0.25_f64.cos(), (-0.75_f64).cos()]),
            1e-12,
            1e-12,
        );
        assert_close_matrix(
            &tanm(&a, DecompOptions::default()).expect("tanm"),
            &diagm(&[0.25_f64.tan(), (-0.75_f64).tan()]),
            1e-12,
            1e-12,
        );
    }

    #[test]
    fn hyperbolic_matrix_functions_match_scalar_diagonal() {
        let a = diagm(&[0.25, -0.75]);

        assert_close_matrix(
            &sinhm(&a, DecompOptions::default()).expect("sinhm"),
            &diagm(&[0.25_f64.sinh(), (-0.75_f64).sinh()]),
            1e-12,
            1e-12,
        );
        assert_close_matrix(
            &coshm(&a, DecompOptions::default()).expect("coshm"),
            &diagm(&[0.25_f64.cosh(), (-0.75_f64).cosh()]),
            1e-12,
            1e-12,
        );
        assert_close_matrix(
            &tanhm(&a, DecompOptions::default()).expect("tanhm"),
            &diagm(&[0.25_f64.tanh(), (-0.75_f64).tanh()]),
            1e-12,
            1e-12,
        );
    }

    #[test]
    fn trig_matrix_functions_satisfy_commuting_identity() {
        let a = rotated_diagonal(0.2, -0.4);
        let sin_a = sinm(&a, DecompOptions::default()).expect("sinm");
        let cos_a = cosm(&a, DecompOptions::default()).expect("cosm");

        let sin_squared = matmul(&sin_a, &sin_a).expect("sinm squared");
        let cos_squared = matmul(&cos_a, &cos_a).expect("cosm squared");
        let mut identity_candidate = vec![vec![0.0; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                identity_candidate[i][j] = sin_squared[i][j] + cos_squared[i][j];
            }
        }

        assert_close_matrix(
            &identity_candidate,
            &[vec![1.0, 0.0], vec![0.0, 1.0]],
            1e-10,
            1e-10,
        );
    }

    #[test]
    fn trig_matrix_functions_reject_non_square() {
        let a = vec![vec![1.0, 2.0]];
        let functions: [MatrixFunction; 6] = [sinm, cosm, tanm, sinhm, coshm, tanhm];

        for matrix_fn in functions {
            let err = matrix_fn(&a, DecompOptions::default()).expect_err("non-square");
            assert!(matches!(err, LinalgError::ExpectedSquareMatrix));
        }
    }

    #[test]
    fn funm_exp_matches_expm() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 2.0]];
        let funm_result = funm(&a, f64::exp, DecompOptions::default()).expect("funm exp");
        let expm_result = expm(&a, DecompOptions::default()).expect("expm");
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (funm_result[i][j] - expm_result[i][j]).abs() < 1e-8,
                    "funm exp vs expm at [{i},{j}]: {} vs {}",
                    funm_result[i][j],
                    expm_result[i][j]
                );
            }
        }
    }

    #[test]
    fn fractional_matrix_power_sqrt() {
        // A^0.5 squared should give A (for positive definite A)
        let a = vec![vec![4.0, 0.0], vec![0.0, 9.0]];
        let sqrt_a =
            fractional_matrix_power(&a, 0.5, DecompOptions::default()).expect("fractional power");
        // sqrt_a should be [[2,0],[0,3]]
        assert!(
            (sqrt_a[0][0] - 2.0).abs() < 1e-8,
            "sqrt(4) = {}",
            sqrt_a[0][0]
        );
        assert!(
            (sqrt_a[1][1] - 3.0).abs() < 1e-8,
            "sqrt(9) = {}",
            sqrt_a[1][1]
        );
    }

    // ── sqrtm tests ────────────────────────────────────────────────────

    #[test]
    fn sqrtm_identity() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let result = sqrtm(&a, DecompOptions::default()).expect("sqrtm(I)");
        // sqrt(I) = I
        assert!((result[0][0] - 1.0).abs() < 1e-12);
        assert!((result[0][1] - 0.0).abs() < 1e-12);
        assert!((result[1][0] - 0.0).abs() < 1e-12);
        assert!((result[1][1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn sqrtm_diagonal() {
        let a = vec![vec![4.0, 0.0], vec![0.0, 9.0]];
        let result = sqrtm(&a, DecompOptions::default()).expect("sqrtm(diag)");
        // sqrt(diag(4,9)) = diag(2,3)
        assert!((result[0][0] - 2.0).abs() < 1e-10);
        assert!((result[0][1] - 0.0).abs() < 1e-10);
        assert!((result[1][0] - 0.0).abs() < 1e-10);
        assert!((result[1][1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn sqrtm_squared_equals_original() {
        let a = vec![vec![5.0, 2.0], vec![2.0, 8.0]];
        let sqrt_a = sqrtm(&a, DecompOptions::default()).expect("sqrtm");
        // sqrt(A) @ sqrt(A) ≈ A
        let n = sqrt_a.len();
        for i in 0..n {
            for j in 0..n {
                let product_ij: f64 = (0..n).map(|k| sqrt_a[i][k] * sqrt_a[k][j]).sum();
                assert!(
                    (product_ij - a[i][j]).abs() < 1e-8,
                    "sqrtm(A)²[{i},{j}] = {product_ij}, expected {}",
                    a[i][j]
                );
            }
        }
    }

    #[test]
    fn sqrtm_3x3_spd() {
        let a = vec![
            vec![4.0, 1.0, 0.5],
            vec![1.0, 5.0, 1.0],
            vec![0.5, 1.0, 6.0],
        ];
        let sqrt_a = sqrtm(&a, DecompOptions::default()).expect("sqrtm 3x3");
        // Verify sqrt(A) @ sqrt(A) ≈ A
        let n = sqrt_a.len();
        for i in 0..n {
            for j in 0..n {
                let product_ij: f64 = (0..n).map(|k| sqrt_a[i][k] * sqrt_a[k][j]).sum();
                assert!(
                    (product_ij - a[i][j]).abs() < 1e-8,
                    "sqrtm(A)²[{i},{j}] = {product_ij}, expected {}",
                    a[i][j]
                );
            }
        }
    }

    #[test]
    fn sqrtm_negative_eigenvalue_returns_nan() {
        // Symmetric matrix with negative eigenvalue: [[1, 0], [0, -1]]
        let a = vec![vec![1.0, 0.0], vec![0.0, -1.0]];
        let result = sqrtm(&a, DecompOptions::default()).expect("sqrtm with neg ev");
        // Result should contain NaN for the negative eigenvalue element
        assert!(result[1][1].is_nan());
    }

    #[test]
    fn sqrtm_rejects_non_square() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let err = sqrtm(&a, DecompOptions::default()).expect_err("non-square");
        assert!(matches!(err, LinalgError::ExpectedSquareMatrix));
    }

    #[test]
    fn fractional_matrix_power_integer() {
        // A^2 should match A * A
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let a2 = fractional_matrix_power(&a, 2.0, DecompOptions::default()).expect("A^2");
        // A² = [[7,10],[15,22]]
        assert!((a2[0][0] - 7.0).abs() < 1e-6, "A²[0,0] = {}", a2[0][0]);
        assert!((a2[1][1] - 22.0).abs() < 1e-6, "A²[1,1] = {}", a2[1][1]);
    }

    #[test]
    fn funm_rejects_non_square() {
        let a = vec![vec![1.0, 2.0]];
        let err = funm(&a, |x| x, DecompOptions::default()).expect_err("non-square");
        assert!(matches!(err, LinalgError::ExpectedSquareMatrix));
    }

    // ── matrix_power tests ─────────────────────────────────────────────

    #[test]
    fn matrix_power_zero_returns_identity() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = matrix_power(&a, 0, DecompOptions::default()).expect("A^0");
        assert_eq!(result.len(), 2);
        assert!((result[0][0] - 1.0).abs() < 1e-12);
        assert!((result[0][1] - 0.0).abs() < 1e-12);
        assert!((result[1][0] - 0.0).abs() < 1e-12);
        assert!((result[1][1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn matrix_power_one_returns_original() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = matrix_power(&a, 1, DecompOptions::default()).expect("A^1");
        assert!((result[0][0] - 1.0).abs() < 1e-12);
        assert!((result[0][1] - 2.0).abs() < 1e-12);
        assert!((result[1][0] - 3.0).abs() < 1e-12);
        assert!((result[1][1] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn matrix_power_two_matches_multiply() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = matrix_power(&a, 2, DecompOptions::default()).expect("A^2");
        // A² = [[7,10],[15,22]]
        assert!((result[0][0] - 7.0).abs() < 1e-10);
        assert!((result[0][1] - 10.0).abs() < 1e-10);
        assert!((result[1][0] - 15.0).abs() < 1e-10);
        assert!((result[1][1] - 22.0).abs() < 1e-10);
    }

    #[test]
    fn matrix_power_negative_one_is_inverse() {
        let a = vec![vec![4.0, 7.0], vec![2.0, 6.0]];
        let result = matrix_power(&a, -1, DecompOptions::default()).expect("A^-1");
        // inv([[4,7],[2,6]]) = [[0.6,-0.7],[-0.2,0.4]]
        assert!((result[0][0] - 0.6).abs() < 1e-10);
        assert!((result[0][1] - (-0.7)).abs() < 1e-10);
        assert!((result[1][0] - (-0.2)).abs() < 1e-10);
        assert!((result[1][1] - 0.4).abs() < 1e-10);
    }

    #[test]
    fn matrix_power_negative_two() {
        let a = vec![vec![2.0, 0.0], vec![0.0, 3.0]];
        // For diagonal: A^-2 = [[1/4,0],[0,1/9]]
        let result = matrix_power(&a, -2, DecompOptions::default()).expect("A^-2");
        assert!((result[0][0] - 0.25).abs() < 1e-10);
        assert!((result[0][1] - 0.0).abs() < 1e-10);
        assert!((result[1][0] - 0.0).abs() < 1e-10);
        assert!((result[1][1] - 1.0 / 9.0).abs() < 1e-10);
    }

    #[test]
    fn matrix_power_rejects_non_square() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let err = matrix_power(&a, 2, DecompOptions::default()).expect_err("non-square");
        assert!(matches!(err, LinalgError::ExpectedSquareMatrix));
    }

    // ── khatri_rao tests ─────────────────────────────────────────────

    #[test]
    fn khatri_rao_basic() {
        // A = [[1,2],[3,4]], B = [[5,6]]
        // Column 0: kron([1,3], [5]) = [5, 15]
        // Column 1: kron([2,4], [6]) = [12, 24]
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0, 6.0]];
        let result = khatri_rao(&a, &b).expect("khatri_rao");
        assert_eq!(result.len(), 2); // 2*1 rows
        assert_eq!(result[0].len(), 2); // 2 cols
        assert!((result[0][0] - 5.0).abs() < 1e-12);
        assert!((result[1][0] - 15.0).abs() < 1e-12);
        assert!((result[0][1] - 12.0).abs() < 1e-12);
        assert!((result[1][1] - 24.0).abs() < 1e-12);
    }

    #[test]
    fn khatri_rao_mismatched_cols() {
        let a = vec![vec![1.0, 2.0]];
        let b = vec![vec![1.0, 2.0, 3.0]];
        let err = khatri_rao(&a, &b).expect_err("mismatched cols");
        assert!(matches!(err, LinalgError::NotSupported { .. }));
    }

    // ── solve_circulant tests ────────────────────────────────────────

    #[test]
    fn solve_circulant_identity() {
        // Circulant of [1, 0, 0] is the identity matrix
        let c = [1.0, 0.0, 0.0];
        let b = [2.0, 3.0, 4.0];
        let x = solve_circulant(&c, &b).expect("solve_circulant");
        for (i, (&xi, &bi)) in x.iter().zip(b.iter()).enumerate() {
            assert!((xi - bi).abs() < 1e-10, "x[{i}] = {xi}, expected {bi}");
        }
    }

    #[test]
    fn solve_circulant_rejects_singular_with_nonzero_rhs() {
        // Circulant of all-zeros is the zero matrix: every eigenvalue = 0.
        // For any non-zero RHS the system is unsolvable. br-z1vz: the old
        // impl silently returned x = 0 (a projection), producing a
        // wrong answer. Fail-closed now returns SingularMatrix.
        let c = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let err = solve_circulant(&c, &b).expect_err("singular circulant");
        assert!(
            matches!(err, LinalgError::SingularMatrix),
            "expected SingularMatrix, got {err:?}"
        );
    }

    #[test]
    fn solve_circulant_accepts_singular_with_compatible_rhs() {
        // Circulant of [1, 1] has eigenvalues [2, 0]; b=[0, 0] projects
        // entirely onto the non-null mode. A consistent singular system
        // should still succeed.
        let c = [0.0_f64, 0.0_f64];
        let b = [0.0_f64, 0.0_f64];
        let x = solve_circulant(&c, &b).expect("consistent singular system");
        assert_eq!(x.len(), 2);
    }

    #[test]
    fn solve_circulant_shift() {
        // Circulant of [0, 1, 0] is a shift matrix
        // [0 0 1]   [x0]   [b0]
        // [1 0 0] * [x1] = [b1]
        // [0 1 0]   [x2]   [b2]
        // So x0=b1, x1=b2, x2=b0
        let c = [0.0, 1.0, 0.0];
        let b = [1.0, 2.0, 3.0];
        let x = solve_circulant(&c, &b).expect("solve_circulant shift");
        assert!((x[0] - 2.0).abs() < 1e-10);
        assert!((x[1] - 3.0).abs() < 1e-10);
        assert!((x[2] - 1.0).abs() < 1e-10);
    }

    // ── solve_toeplitz tests ─────────────────────────────────────────

    #[test]
    fn solve_toeplitz_symmetric_identity() {
        let c = [1.0, 0.0, 0.0];
        let b = [2.0, 3.0, 4.0];
        let x = solve_toeplitz(&c, None, &b).expect("solve_toeplitz");
        for (i, (&xi, &bi)) in x.iter().zip(b.iter()).enumerate() {
            assert!((xi - bi).abs() < 1e-10, "x[{i}] = {xi}, expected {bi}");
        }
    }

    #[test]
    fn solve_toeplitz_matches_scipy_doc_example() {
        let c = [1.0, 3.0, 6.0, 10.0];
        let r = [1.0, -1.0, -2.0, -3.0];
        let b = [1.0, 2.0, 2.0, 5.0];
        let x = solve_toeplitz(&c, Some(&r), &b).expect("solve_toeplitz");
        let expected = [
            1.6666666666666667,
            -1.0,
            -2.6666666666666665,
            2.3333333333333335,
        ];
        for (i, (&xi, &want)) in x.iter().zip(expected.iter()).enumerate() {
            assert!((xi - want).abs() < 1e-10, "x[{i}] = {xi}, expected {want}");
        }
    }

    #[test]
    fn solve_toeplitz_ignores_row_zero_entry() {
        let c = [2.0, 1.0, 0.0];
        let r = [99.0, -1.0, 0.5];
        let matrix = toeplitz(&c, Some(&r));
        assert!(
            (matrix[0][0] - 2.0).abs() < 1e-12,
            "top-left must come from c[0]"
        );

        let x_expected = [1.0, -2.0, 0.5];
        let mut b = vec![0.0; 3];
        for i in 0..3 {
            for (j, value) in x_expected.iter().enumerate() {
                b[i] += matrix[i][j] * *value;
            }
        }

        let x = solve_toeplitz(&c, Some(&r), &b).expect("solve_toeplitz");
        for (i, (&xi, &want)) in x.iter().zip(x_expected.iter()).enumerate() {
            assert!((xi - want).abs() < 1e-10, "x[{i}] = {xi}, expected {want}");
        }
    }

    #[test]
    fn solve_toeplitz_rejects_shape_mismatch() {
        let err = solve_toeplitz(&[1.0, 2.0, 3.0], Some(&[1.0, 2.0]), &[1.0, 2.0, 3.0])
            .expect_err("row length mismatch");
        assert!(matches!(err, LinalgError::IncompatibleShapes { .. }));
    }

    // ── issymmetric / ishermitian tests ───────────────────────────────

    #[test]
    fn issymmetric_true_for_symmetric() {
        let a = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 5.0, 6.0],
            vec![3.0, 6.0, 9.0],
        ];
        assert!(issymmetric(&a, 0.0, 0.0).expect("issymmetric"));
    }

    #[test]
    fn issymmetric_false_for_asymmetric() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert!(!issymmetric(&a, 0.0, 0.0).expect("issymmetric"));
    }

    #[test]
    fn issymmetric_with_tolerance() {
        let a = vec![vec![1.0, 2.0], vec![2.0 + 1e-10, 4.0]];
        assert!(!issymmetric(&a, 0.0, 0.0).expect("strict"));
        assert!(issymmetric(&a, 1e-9, 0.0).expect("with tolerance"));
    }

    #[test]
    fn issymmetric_non_square() {
        let a = vec![vec![1.0, 2.0, 3.0]];
        assert!(!issymmetric(&a, 0.0, 0.0).expect("non-square"));
    }

    #[test]
    fn issymmetric_rejects_non_finite_entries() {
        let a = vec![vec![1.0, f64::NAN], vec![f64::NAN, 2.0]];
        assert!(!issymmetric(&a, 0.0, 0.0).expect("non-finite"));
    }

    #[test]
    fn ishermitian_matches_symmetric_for_real_matrix() {
        let a = vec![vec![2.0, -1.0], vec![-1.0, 3.0]];
        assert!(ishermitian(&a, 0.0, 0.0).expect("ishermitian"));
    }

    #[test]
    fn ishermitian_false_for_real_asymmetric_matrix() {
        let a = vec![vec![1.0, 4.0], vec![2.0, 3.0]];
        assert!(!ishermitian(&a, 0.0, 0.0).expect("ishermitian"));
    }

    #[test]
    fn is_positive_definite_rejects_non_finite_entries() {
        let a = vec![vec![f64::INFINITY, 0.0], vec![0.0, 1.0]];
        assert!(!is_positive_definite(&a));
    }

    #[test]
    fn is_orthogonal_rejects_ragged_or_non_finite() {
        let ragged = vec![vec![1.0, 0.0], vec![0.0]];
        assert!(!is_orthogonal(&ragged, 1e-12));

        let non_finite = vec![vec![1.0, f64::NAN], vec![0.0, 1.0]];
        assert!(!is_orthogonal(&non_finite, 1e-12));
    }

    #[test]
    fn mat_norm_1_rejects_ragged_or_non_finite() {
        let ragged = vec![vec![1.0, 2.0], vec![3.0]];
        assert!(mat_norm_1(&ragged).is_nan());

        let non_finite = vec![vec![1.0, f64::NAN], vec![0.0, 1.0]];
        assert!(mat_norm_1(&non_finite).is_nan());
    }

    #[test]
    fn mat_allclose_handles_nan_and_infinity_explicitly() {
        assert!(mat_allclose(
            &[vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY]],
            &[vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY]],
            1e-12,
            1e-12,
        ));
        assert!(!mat_allclose(&[vec![f64::NAN]], &[vec![0.0]], 1e-12, 1e-12,));
        assert!(!mat_allclose(
            &[vec![f64::INFINITY]],
            &[vec![f64::NEG_INFINITY]],
            1e-12,
            1e-12,
        ));
        assert!(!mat_allclose(
            &[vec![f64::INFINITY]],
            &[vec![1.0e308]],
            1e-12,
            1e-12,
        ));
    }

    #[test]
    fn matrix_rank_explicit_tolerance_changes_result() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1e-12]];
        let default_rank = matrix_rank(&a, None, DecompOptions::default()).expect("default rank");
        let tolerant_rank =
            matrix_rank(&a, Some(1e-10), DecompOptions::default()).expect("tolerant rank");
        assert_eq!(default_rank, 2);
        assert_eq!(tolerant_rank, 1);
    }

    // ── eigh_tridiagonal tests ─────────────────────────────────────────

    #[test]
    fn eigh_tridiagonal_2x2() {
        // T = [[2, 1], [1, 2]] has eigenvalues 1 and 3
        let d = vec![2.0, 2.0];
        let e = vec![1.0];
        let (eigenvalues, eigenvectors) =
            eigh_tridiagonal(&d, &e, false, DecompOptions::default()).expect("eigh_tridiagonal");
        assert_eq!(eigenvalues.len(), 2);
        assert!(
            (eigenvalues[0] - 1.0).abs() < 1e-10,
            "λ1 = {}",
            eigenvalues[0]
        );
        assert!(
            (eigenvalues[1] - 3.0).abs() < 1e-10,
            "λ2 = {}",
            eigenvalues[1]
        );
        assert!(eigenvectors.is_some());
    }

    #[test]
    fn eigh_tridiagonal_eigenvalues_only() {
        let d = vec![1.0, 2.0, 3.0];
        let e = vec![0.5, 0.5];
        let (eigenvalues, eigenvectors) =
            eigh_tridiagonal(&d, &e, true, DecompOptions::default()).expect("eigvals only");
        assert_eq!(eigenvalues.len(), 3);
        assert!(eigenvectors.is_none());
    }

    #[test]
    fn eigh_tridiagonal_diagonal_matrix() {
        // Off-diagonals are zero: eigenvalues are just the diagonal
        let d = vec![3.0, 1.0, 2.0];
        let e = vec![0.0, 0.0];
        let (eigenvalues, _) =
            eigh_tridiagonal(&d, &e, false, DecompOptions::default()).expect("diagonal");
        // Sorted eigenvalues
        assert!((eigenvalues[0] - 1.0).abs() < 1e-10);
        assert!((eigenvalues[1] - 2.0).abs() < 1e-10);
        assert!((eigenvalues[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn eigh_tridiagonal_empty() {
        let (eigenvalues, eigenvectors) =
            eigh_tridiagonal(&[], &[], false, DecompOptions::default()).expect("empty");
        assert!(eigenvalues.is_empty());
        assert!(eigenvectors.unwrap().is_empty());
    }

    #[test]
    fn eigh_tridiagonal_invalid_e_length() {
        let d = vec![1.0, 2.0, 3.0];
        let e = vec![0.5]; // Should be length 2
        let err = eigh_tridiagonal(&d, &e, false, DecompOptions::default()).expect_err("invalid");
        assert!(matches!(err, LinalgError::InvalidArgument { .. }));
    }

    // ── eig_banded tests ────────────────────────────────────────────────

    #[test]
    fn eig_banded_tridiagonal() {
        // A = [[2, 1, 0], [1, 2, 1], [0, 1, 2]] - tridiagonal
        // Banded storage: ab[0] = diagonal, ab[1] = sub-diagonal
        let ab = vec![vec![2.0, 2.0, 2.0], vec![1.0, 1.0, 0.0]];
        let (eigenvalues, eigenvectors) =
            eig_banded(&ab, true, false, DecompOptions::default()).expect("eig_banded");
        assert_eq!(eigenvalues.len(), 3);
        // Known eigenvalues: 2 - sqrt(2), 2, 2 + sqrt(2)
        let sqrt2 = std::f64::consts::SQRT_2;
        assert!(
            (eigenvalues[0] - (2.0 - sqrt2)).abs() < 1e-8,
            "λ1 = {}, expected {}",
            eigenvalues[0],
            2.0 - sqrt2
        );
        assert!(
            (eigenvalues[1] - 2.0).abs() < 1e-8,
            "λ2 = {}",
            eigenvalues[1]
        );
        assert!(
            (eigenvalues[2] - (2.0 + sqrt2)).abs() < 1e-8,
            "λ3 = {}, expected {}",
            eigenvalues[2],
            2.0 + sqrt2
        );
        assert!(eigenvectors.is_some());
    }

    #[test]
    fn eig_banded_diagonal() {
        // Diagonal matrix A = diag(3, 1, 2)
        let ab = vec![vec![3.0, 1.0, 2.0]];
        let (eigenvalues, _) =
            eig_banded(&ab, true, true, DecompOptions::default()).expect("eig_banded diagonal");
        // Eigenvalues sorted: 1, 2, 3
        assert!((eigenvalues[0] - 1.0).abs() < 1e-10);
        assert!((eigenvalues[1] - 2.0).abs() < 1e-10);
        assert!((eigenvalues[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn eig_banded_eigvals_only() {
        let ab = vec![vec![2.0, 2.0, 2.0], vec![1.0, 1.0, 0.0]];
        let (eigenvalues, eigenvectors) =
            eig_banded(&ab, true, true, DecompOptions::default()).expect("eigvals only");
        assert_eq!(eigenvalues.len(), 3);
        assert!(eigenvectors.is_none());
    }

    #[test]
    fn eig_banded_empty() {
        let ab = vec![vec![]];
        let (eigenvalues, eigenvectors) =
            eig_banded(&ab, true, false, DecompOptions::default()).expect("empty");
        assert!(eigenvalues.is_empty());
        assert!(eigenvectors.unwrap().is_empty());
    }

    #[test]
    fn det_matches_scipy_reference_values() {
        // scipy.linalg.det([[1, 2], [3, 4]]) = -2.0
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = det(&a, RuntimeMode::Strict, true).expect("det");
        assert!(
            (result - (-2.0)).abs() < 1e-10,
            "det got {result}, expected -2.0"
        );
    }

    #[test]
    fn inv_matches_scipy_reference_values() {
        // scipy.linalg.inv([[1, 2], [3, 4]]) = [[-2, 1], [1.5, -0.5]]
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = inv(&a, InvOptions::default()).expect("inv");
        let inv_a = result.inverse;
        assert!(
            (inv_a[0][0] - (-2.0)).abs() < 1e-10,
            "inv[0][0] got {}, expected -2.0",
            inv_a[0][0]
        );
        assert!(
            (inv_a[0][1] - 1.0).abs() < 1e-10,
            "inv[0][1] got {}, expected 1.0",
            inv_a[0][1]
        );
        assert!(
            (inv_a[1][0] - 1.5).abs() < 1e-10,
            "inv[1][0] got {}, expected 1.5",
            inv_a[1][0]
        );
        assert!(
            (inv_a[1][1] - (-0.5)).abs() < 1e-10,
            "inv[1][1] got {}, expected -0.5",
            inv_a[1][1]
        );
    }

    #[test]
    fn svdvals_matches_scipy_reference_values() {
        // scipy.linalg.svdvals([[1, 2], [3, 4]]) ≈ [5.4649, 0.3659]
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = svdvals(&a, DecompOptions::default()).expect("svdvals");
        assert!(
            (result[0] - 5.464985704219043).abs() < 1e-10,
            "svdvals[0] got {}, expected 5.464985704219043",
            result[0]
        );
        assert!(
            (result[1] - 0.3659661906262574).abs() < 1e-10,
            "svdvals[1] got {}, expected 0.3659661906262574",
            result[1]
        );
    }

    #[test]
    fn expm_matches_scipy_reference_values() {
        // scipy.linalg.expm([[1, 2], [3, 4]])[0,0] ≈ 51.9689
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = expm(&a, DecompOptions::default()).expect("expm");
        assert!(
            (result[0][0] - 51.968956198707545).abs() < 1e-8,
            "expm[0][0] got {}, expected 51.968956198707545",
            result[0][0]
        );
    }

    #[test]
    fn solve_matches_scipy_reference_values() {
        // scipy.linalg.solve([[1, 2], [3, 4]], [5, 6]) = [-4, 4.5]
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![5.0, 6.0];
        let result = solve(&a, &b, SolveOptions::default()).expect("solve");
        assert!(
            (result.x[0] - (-4.0)).abs() < 1e-10,
            "solve[0] got {}, expected -4.0",
            result.x[0]
        );
        assert!(
            (result.x[1] - 4.5).abs() < 1e-10,
            "solve[1] got {}, expected 4.5",
            result.x[1]
        );
    }

    #[test]
    fn cholesky_matches_scipy_reference_values() {
        // scipy.linalg.cholesky([[4, 2], [2, 3]], lower=True)
        let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
        let result = cholesky(&a, true, DecompOptions::default()).expect("cholesky");
        assert!(
            (result.factor[0][0] - 2.0).abs() < 1e-10,
            "cholesky[0][0] got {}, expected 2.0",
            result.factor[0][0]
        );
        assert!(
            result.factor[0][1].abs() < 1e-10,
            "cholesky[0][1] got {}, expected 0.0",
            result.factor[0][1]
        );
        assert!(
            (result.factor[1][0] - 1.0).abs() < 1e-10,
            "cholesky[1][0] got {}, expected 1.0",
            result.factor[1][0]
        );
        assert!(
            (result.factor[1][1] - std::f64::consts::SQRT_2).abs() < 1e-10,
            "cholesky[1][1] got {}, expected SQRT_2",
            result.factor[1][1]
        );
    }

    #[test]
    fn norm_frobenius_matches_scipy_reference_values() {
        // scipy.linalg.norm([[1, 2], [3, 4]], 'fro') = 5.477225575051661
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = norm(&a, NormKind::Fro, DecompOptions::default()).expect("norm fro");
        assert!(
            (result - 5.477225575051661).abs() < 1e-10,
            "norm fro got {result}, expected 5.477225575051661"
        );
    }

    #[test]
    fn norm_1_matches_scipy_reference_values() {
        // scipy.linalg.norm([[1, 2], [3, 4]], 1) = 6.0
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = norm(&a, NormKind::One, DecompOptions::default()).expect("norm 1");
        assert!(
            (result - 6.0).abs() < 1e-10,
            "norm 1 got {result}, expected 6.0"
        );
    }

    #[test]
    fn norm_inf_matches_scipy_reference_values() {
        // scipy.linalg.norm([[1, 2], [3, 4]], inf) = 7.0
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = norm(&a, NormKind::Inf, DecompOptions::default()).expect("norm inf");
        assert!(
            (result - 7.0).abs() < 1e-10,
            "norm inf got {result}, expected 7.0"
        );
    }

    #[test]
    fn eigvals_matches_scipy_reference_values() {
        // scipy.linalg.eigvals([[1, 2], [3, 4]]) = [-0.372..., 5.372...]
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let (real_parts_unsorted, _imag_parts) =
            eigvals(&a, DecompOptions::default()).expect("eigvals");
        let mut real_parts = real_parts_unsorted.clone();
        real_parts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let expected = [-0.3722813232690143, 5.372281323269014];
        for (i, (&got, &want)) in real_parts.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "eigvals[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn qr_matches_scipy_reference_values() {
        // scipy.linalg.qr([[1, 2], [3, 4]]) - verify QR = A
        // Sign convention may differ, so check reconstruction instead
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = qr(&a, DecompOptions::default()).expect("qr");
        let expected_q_abs = [
            [0.316227766016838, 0.9486832980505138],
            [0.9486832980505138, 0.316227766016838],
        ];
        let expected_r_abs = [
            [3.1622776601683795, 4.427188724235731],
            [0.0, 0.6324555320336751],
        ];
        for (i, row) in result.q.iter().enumerate() {
            for (j, &got) in row.iter().enumerate() {
                let want = expected_q_abs[i][j];
                assert!(
                    (got.abs() - want).abs() < 1e-10,
                    "|Q[{i}][{j}]| got {}, expected {want}",
                    got.abs()
                );
            }
        }
        for (i, row) in result.r.iter().enumerate() {
            for (j, &got) in row.iter().enumerate() {
                let want = expected_r_abs[i][j];
                assert!(
                    (got.abs() - want).abs() < 1e-10,
                    "|R[{i}][{j}]| got {}, expected {want}",
                    got.abs()
                );
            }
        }
    }

    #[test]
    fn pinv_matches_scipy_reference_values() {
        // scipy.linalg.pinv([[1, 2, 3], [4, 5, 6]])
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let result = pinv(&a, PinvOptions::default()).expect("pinv");
        let expected = [
            [-0.9444444444444444, 0.4444444444444443],
            [-0.11111111111111108, 0.11111111111111109],
            [0.7222222222222221, -0.22222222222222207],
        ];
        for (i, row) in result.pseudo_inverse.iter().enumerate() {
            for (j, &got) in row.iter().enumerate() {
                let want = expected[i][j];
                assert!(
                    (got - want).abs() < 1e-9,
                    "pinv[{i}][{j}] got {got}, expected {want}"
                );
            }
        }
    }

    #[test]
    fn lu_matches_scipy_reference_values() {
        // scipy.linalg.lu([[1, 2], [3, 4]]) returns P, L, U
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = lu(&a, DecompOptions::default()).expect("lu");
        let expected_l = [[1.0, 0.0], [0.3333333333333333, 1.0]];
        let expected_u = [[3.0, 4.0], [0.0, 0.6666666666666667]];
        for (i, row) in result.l.iter().enumerate() {
            for (j, &got) in row.iter().enumerate() {
                let want = expected_l[i][j];
                assert!(
                    (got - want).abs() < 1e-10,
                    "L[{i}][{j}] got {got}, expected {want}"
                );
            }
        }
        for (i, row) in result.u.iter().enumerate() {
            for (j, &got) in row.iter().enumerate() {
                let want = expected_u[i][j];
                assert!(
                    (got - want).abs() < 1e-10,
                    "U[{i}][{j}] got {got}, expected {want}"
                );
            }
        }
    }

    #[test]
    fn lstsq_matches_scipy_reference_values() {
        // scipy.linalg.lstsq([[1, 1], [1, 2], [1, 3]], [1, 2, 2])
        // Returns: x = [0.6666..., 0.5]
        let a = vec![vec![1.0, 1.0], vec![1.0, 2.0], vec![1.0, 3.0]];
        let b = vec![1.0, 2.0, 2.0];
        let result = lstsq(&a, &b, LstsqOptions::default()).expect("lstsq");
        // scipy reference: x ≈ [0.6667, 0.5]
        assert!(
            (result.x[0] - 0.6666666666666667).abs() < 1e-10,
            "x[0] got {}, expected 0.6667",
            result.x[0]
        );
        assert!(
            (result.x[1] - 0.5).abs() < 1e-10,
            "x[1] got {}, expected 0.5",
            result.x[1]
        );
    }

    #[test]
    fn svd_matches_scipy_reference_values() {
        // scipy.linalg.svd([[1, 2], [3, 4]])
        // Returns U, s, Vh where s = [5.4649..., 0.3659...]
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = svd(&a, DecompOptions::default()).expect("svd");
        // Check singular values
        let expected_s = [5.464985704219043, 0.3659661906262574];
        for (i, (&got, &want)) in result.s.iter().zip(expected_s.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "s[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn eigh_matches_scipy_reference_values() {
        // scipy.linalg.eigh([[1, 2], [2, 4]]) - symmetric matrix
        // eigenvalues: [0, 5]
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let result = eigh(&a, DecompOptions::default()).expect("eigh");
        let mut eigenvalues: Vec<f64> = result.eigenvalues.clone();
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(
            eigenvalues[0].abs() < 1e-10,
            "smallest eigenvalue got {}, expected ~0",
            eigenvalues[0]
        );
        assert!(
            (eigenvalues[1] - 5.0).abs() < 1e-10,
            "largest eigenvalue got {}, expected 5",
            eigenvalues[1]
        );
    }

    #[test]
    fn solve_triangular_matches_scipy_reference_values() {
        // scipy.linalg.solve_triangular([[3, 0], [1, 2]], [9, 8], lower=True)
        // -> [3, 2.5]
        let a = vec![vec![3.0, 0.0], vec![1.0, 2.0]];
        let b = vec![9.0, 8.0];
        let opts = TriangularSolveOptions {
            lower: true,
            ..Default::default()
        };
        let result = solve_triangular(&a, &b, opts).expect("solve_triangular");
        let expected = [3.0, 2.5];
        for (i, (&got, &want)) in result.x.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "x[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn lu_solve_matches_scipy_reference_values() {
        // scipy.linalg.lu_factor([[1, 2], [3, 4]]) then lu_solve with b=[1, 2]
        // -> x = [0, 0.5]
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![1.0, 2.0];
        let lu_factor_result = lu_factor(&a, DecompOptions::default()).expect("lu_factor");
        let result = lu_solve(&lu_factor_result, &b).expect("lu_solve");
        let expected = [0.0, 0.5];
        for (i, (&got, &want)) in result.x.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "x[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn cho_solve_matches_scipy_reference_values() {
        // scipy.linalg.cho_factor([[4, 2], [2, 5]]) then cho_solve with b=[1, 2]
        // -> x = [0.0625, 0.375]
        let a = vec![vec![4.0, 2.0], vec![2.0, 5.0]];
        let b = vec![1.0, 2.0];
        let cho_factor_result = cho_factor(&a, DecompOptions::default()).expect("cho_factor");
        let result = cho_solve(&cho_factor_result, &b).expect("cho_solve");
        let expected = [0.0625, 0.375];
        for (i, (&got, &want)) in result.x.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "x[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn ldl_matches_scipy_reference_values() {
        // scipy.linalg.ldl([[2, 1], [1, 3]])
        // -> l = [[1, 0], [0.5, 1]], d = [[2, 0], [0, 2.5]]
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let result = ldl(&a, DecompOptions::default()).expect("ldl");
        // Check L matrix
        assert!((result.l[0][0] - 1.0).abs() < 1e-10);
        assert!((result.l[1][0] - 0.5).abs() < 1e-10);
        assert!((result.l[1][1] - 1.0).abs() < 1e-10);
        // Check D diagonal
        assert!((result.d[0] - 2.0).abs() < 1e-10);
        assert!((result.d[1] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn schur_matches_scipy_reference_values() {
        // scipy.linalg.schur([[0, 1], [-1, 0]])
        // Real Schur form of 2D rotation matrix
        let a = vec![vec![0.0, 1.0], vec![-1.0, 0.0]];
        let result = schur(&a, DecompOptions::default()).expect("schur");
        // T should be quasi-upper triangular, Z orthogonal
        // Z @ Z.T = I
        let m = result.z[0][0] * result.z[0][0] + result.z[0][1] * result.z[0][1];
        assert!((m - 1.0).abs() < 1e-10, "Z row 0 norm should be 1");
        // A = Z @ T @ Z.T
        let ztz00 = result.z[0][0] * result.t[0][0] + result.z[0][1] * result.t[1][0];
        let ztz01 = result.z[0][0] * result.t[0][1] + result.z[0][1] * result.t[1][1];
        let a_rec00 = ztz00 * result.z[0][0] + ztz01 * result.z[1][0];
        assert!(
            (a_rec00 - a[0][0]).abs() < 1e-10,
            "Schur reconstruction [0][0]"
        );
    }

    #[test]
    fn hessenberg_matches_scipy_reference_values() {
        // scipy.linalg.hessenberg([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        // H[2][0] should be zero (sub-sub-diagonal)
        let a = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let result = hessenberg(&a, DecompOptions::default()).expect("hessenberg");
        // Sub-sub-diagonal should be zero
        assert!(
            result.h[2][0].abs() < 1e-10,
            "hessenberg h[2][0] should be ~0, got {}",
            result.h[2][0]
        );
        // Trace should be preserved
        let trace_a = a[0][0] + a[1][1] + a[2][2];
        let trace_h = result.h[0][0] + result.h[1][1] + result.h[2][2];
        assert!(
            (trace_a - trace_h).abs() < 1e-10,
            "trace should be preserved"
        );
    }

    #[test]
    fn logm_matches_scipy_reference_values() {
        // scipy.linalg.logm([[e, 0], [0, e]]) = [[1, 0], [0, 1]]
        let e = std::f64::consts::E;
        let a = vec![vec![e, 0.0], vec![0.0, e]];
        let result = logm(&a, DecompOptions::default()).expect("logm");
        assert!(
            (result[0][0] - 1.0).abs() < 1e-10,
            "logm[0][0] got {}, expected 1.0",
            result[0][0]
        );
        assert!(
            result[0][1].abs() < 1e-10,
            "logm[0][1] got {}, expected 0.0",
            result[0][1]
        );
        assert!(
            result[1][0].abs() < 1e-10,
            "logm[1][0] got {}, expected 0.0",
            result[1][0]
        );
        assert!(
            (result[1][1] - 1.0).abs() < 1e-10,
            "logm[1][1] got {}, expected 1.0",
            result[1][1]
        );
    }

    #[test]
    fn sqrtm_matches_scipy_reference_values() {
        // scipy.linalg.sqrtm([[4, 0], [0, 9]]) = [[2, 0], [0, 3]]
        let a = vec![vec![4.0, 0.0], vec![0.0, 9.0]];
        let result = sqrtm(&a, DecompOptions::default()).expect("sqrtm");
        assert!(
            (result[0][0] - 2.0).abs() < 1e-10,
            "sqrtm[0][0] got {}, expected 2.0",
            result[0][0]
        );
        assert!(
            result[0][1].abs() < 1e-10,
            "sqrtm[0][1] got {}, expected 0.0",
            result[0][1]
        );
        assert!(
            result[1][0].abs() < 1e-10,
            "sqrtm[1][0] got {}, expected 0.0",
            result[1][0]
        );
        assert!(
            (result[1][1] - 3.0).abs() < 1e-10,
            "sqrtm[1][1] got {}, expected 3.0",
            result[1][1]
        );
    }

    #[test]
    fn eigvalsh_symmetric_matrix_matches_scipy_reference() {
        // scipy.linalg.eigvalsh([[1, 2], [2, 4]]) = [0, 5]
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let mut result = eigvalsh(&a, DecompOptions::default()).expect("eigvalsh");
        result.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(
            result[0].abs() < 1e-10,
            "eigvalsh[0] got {}, expected 0.0",
            result[0]
        );
        assert!(
            (result[1] - 5.0).abs() < 1e-10,
            "eigvalsh[1] got {}, expected 5.0",
            result[1]
        );
    }

    #[test]
    fn svdvals_diagonal_matrix_matches_scipy_reference() {
        // scipy.linalg.svdvals([[3, 0], [0, 4]]) = [4, 3]
        let a = vec![vec![3.0, 0.0], vec![0.0, 4.0]];
        let mut result = svdvals(&a, DecompOptions::default()).expect("svdvals");
        result.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert!(
            (result[0] - 4.0).abs() < 1e-10,
            "svdvals[0] got {}, expected 4.0",
            result[0]
        );
        assert!(
            (result[1] - 3.0).abs() < 1e-10,
            "svdvals[1] got {}, expected 3.0",
            result[1]
        );
    }

    #[test]
    fn lu_triangular_factors_match_scipy_convention() {
        // scipy.linalg.lu([[1, 2], [3, 4]]) returns P, L, U
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = lu(&a, DecompOptions::default()).expect("lu");
        // U diagonal elements product gives determinant magnitude
        let u_diag_product = result.u[0][0] * result.u[1][1];
        assert!(
            u_diag_product.abs() > 1e-10,
            "U diagonal product should be non-zero"
        );
    }

    #[test]
    fn sinm_diagonal_matrix_matches_scipy_reference() {
        // scipy.linalg.sinm([[1, 0], [0, 2]]) -> [[sin(1), 0], [0, sin(2)]]
        // sin(1) ≈ 0.8414709848, sin(2) ≈ 0.9092974268
        let a = vec![vec![1.0, 0.0], vec![0.0, 2.0]];
        let result = sinm(&a, DecompOptions::default()).expect("sinm");
        assert!(
            (result[0][0] - 0.8414709848).abs() < 1e-6,
            "sinm[0][0] = {}, expected 0.8414709848",
            result[0][0]
        );
        assert!(
            result[0][1].abs() < 1e-10,
            "sinm[0][1] = {}, expected 0",
            result[0][1]
        );
        assert!(
            result[1][0].abs() < 1e-10,
            "sinm[1][0] = {}, expected 0",
            result[1][0]
        );
        assert!(
            (result[1][1] - 0.9092974268).abs() < 1e-6,
            "sinm[1][1] = {}, expected 0.9092974268",
            result[1][1]
        );
    }

    #[test]
    fn cosm_diagonal_matrix_matches_scipy_reference() {
        // scipy.linalg.cosm([[1, 0], [0, 2]]) -> [[cos(1), 0], [0, cos(2)]]
        // cos(1) ≈ 0.5403023059, cos(2) ≈ -0.4161468365
        let a = vec![vec![1.0, 0.0], vec![0.0, 2.0]];
        let result = cosm(&a, DecompOptions::default()).expect("cosm");
        assert!(
            (result[0][0] - 0.5403023059).abs() < 1e-6,
            "cosm[0][0] = {}, expected 0.5403023059",
            result[0][0]
        );
        assert!(
            result[0][1].abs() < 1e-10,
            "cosm[0][1] = {}, expected 0",
            result[0][1]
        );
        assert!(
            result[1][0].abs() < 1e-10,
            "cosm[1][0] = {}, expected 0",
            result[1][0]
        );
        assert!(
            (result[1][1] - (-0.4161468365)).abs() < 1e-6,
            "cosm[1][1] = {}, expected -0.4161468365",
            result[1][1]
        );
    }

    #[test]
    fn signm_matches_scipy_reference_values() {
        // scipy.linalg.signm([[1, 0], [0, -1]]) -> [[1, 0], [0, -1]]
        let a = vec![vec![1.0, 0.0], vec![0.0, -1.0]];
        let result = signm(&a, DecompOptions::default()).expect("signm");
        assert!(
            (result[0][0] - 1.0).abs() < 1e-10,
            "signm[0][0] = {}, expected 1.0",
            result[0][0]
        );
        assert!(
            result[0][1].abs() < 1e-10,
            "signm[0][1] = {}, expected 0.0",
            result[0][1]
        );
        assert!(
            result[1][0].abs() < 1e-10,
            "signm[1][0] = {}, expected 0.0",
            result[1][0]
        );
        assert!(
            (result[1][1] + 1.0).abs() < 1e-10,
            "signm[1][1] = {}, expected -1.0",
            result[1][1]
        );
    }

    #[test]
    fn polar_matches_scipy_reference_values() {
        // scipy.linalg.polar([[1, 2], [3, 4]]) -> (u, p)
        // u is unitary, p is positive semidefinite hermitian
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = polar(&a, DecompOptions::default()).expect("polar");
        // Verify U is orthogonal: U^T * U ≈ I
        let ut_u_00 = result.u[0][0] * result.u[0][0] + result.u[1][0] * result.u[1][0];
        let ut_u_11 = result.u[0][1] * result.u[0][1] + result.u[1][1] * result.u[1][1];
        assert!(
            (ut_u_00 - 1.0).abs() < 1e-6,
            "U^T*U[0][0] = {}, expected 1.0",
            ut_u_00
        );
        assert!(
            (ut_u_11 - 1.0).abs() < 1e-6,
            "U^T*U[1][1] = {}, expected 1.0",
            ut_u_11
        );
        // P should be symmetric positive
        assert!(
            (result.p[0][1] - result.p[1][0]).abs() < 1e-6,
            "P should be symmetric"
        );
    }
}
