#![forbid(unsafe_code)]

pub use fsci_runtime::SyncSharedAuditLedger;
use fsci_runtime::{AuditAction, AuditEvent, AuditLedger, RuntimeMode, casp_now_unix_ms};

/// Create a shared audit ledger for integrate validation APIs.
#[must_use]
pub fn sync_audit_ledger() -> SyncSharedAuditLedger {
    AuditLedger::shared()
}

pub(crate) fn audit_fingerprint(context: &str, detail: impl Into<String>) -> Vec<u8> {
    format!("{context}:{}", detail.into()).into_bytes()
}

/// Acquire the ledger guard, recovering from a poisoned mutex so audit
/// events still record after any prior thread panicked.
/// Resolves [frankenscipy-l2irg] for fsci-integrate.
fn lock_or_recover(ledger: &SyncSharedAuditLedger) -> std::sync::MutexGuard<'_, AuditLedger> {
    match ledger.lock() {
        Ok(g) => g,
        Err(poisoned) => {
            ledger.clear_poison();
            poisoned.into_inner()
        }
    }
}

pub(crate) fn record_fail_closed(
    ledger: Option<&SyncSharedAuditLedger>,
    input_bytes: &[u8],
    reason: &str,
    outcome: &str,
) {
    let Some(ledger) = ledger else {
        return;
    };

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

pub(crate) fn record_bounded_recovery(
    ledger: Option<&SyncSharedAuditLedger>,
    input_bytes: &[u8],
    recovery_action: &str,
    outcome: &str,
) {
    let Some(ledger) = ledger else {
        return;
    };

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

pub const EPS: f64 = f64::EPSILON;
pub const MIN_RTOL: f64 = 100.0 * EPS;

#[derive(Debug, Clone, PartialEq)]
pub enum ToleranceValue {
    Scalar(f64),
    Vector(Vec<f64>),
}

impl ToleranceValue {
    fn map(self, mut f: impl FnMut(f64) -> f64) -> Self {
        match self {
            Self::Scalar(value) => Self::Scalar(f(value)),
            Self::Vector(values) => Self::Vector(values.into_iter().map(f).collect()),
        }
    }

    fn any(&self, mut predicate: impl FnMut(f64) -> bool) -> bool {
        match self {
            Self::Scalar(value) => predicate(*value),
            Self::Vector(values) => values.iter().copied().any(predicate),
        }
    }

    fn len_if_vector(&self) -> Option<usize> {
        match self {
            Self::Scalar(_) => None,
            Self::Vector(values) => Some(values.len()),
        }
    }

    pub(crate) fn into_scalar(self) -> Option<f64> {
        match self {
            Self::Scalar(value) => Some(value),
            Self::Vector(_) => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ToleranceWarning {
    RtolClamped { minimum: f64 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedTolerance {
    pub rtol: ToleranceValue,
    pub atol: ToleranceValue,
    pub mode: RuntimeMode,
    pub warnings: Vec<ToleranceWarning>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntegrateValidationError {
    EmptyY0,
    FirstStepMustBePositive,
    FirstStepExceedsBounds,
    MaxStepMustBePositive,
    NonFiniteFirstStep,
    NonFiniteMaxStep,
    NonFiniteRtol,
    NonFiniteAtol,
    AtolWrongShape {
        expected: usize,
        actual: usize,
    },
    AtolMustBePositive,
    NonFiniteY0,
    NonFiniteSpan,
    NonFiniteF0,
    RhsWrongShape {
        expected: usize,
        actual: usize,
    },
    NonFiniteEventDirection {
        index: usize,
    },
    EventMaxEventsMustBePositive {
        index: usize,
    },
    NonFiniteEventValue {
        index: usize,
    },
    TEvalOutOfSpan,
    TEvalNotSorted,
    NotYetImplemented {
        function: &'static str,
    },
    QuadInvalidBounds {
        detail: String,
    },
    QuadInvalidTolerance {
        detail: String,
    },
    LebedevOrderUnavailable {
        order: i64,
    },
    /// The integrator stopped before reaching every requested time (e.g. the step
    /// size underflowed on a finite-time blow-up). Carries the solver's message.
    IntegrationFailed {
        message: String,
    },
}

impl std::fmt::Display for IntegrateValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyY0 => write!(f, "`y0` must contain at least one state variable."),
            Self::FirstStepMustBePositive => write!(f, "`first_step` must be positive."),
            Self::FirstStepExceedsBounds => write!(f, "`first_step` exceeds bounds."),
            Self::MaxStepMustBePositive => write!(f, "`max_step` must be positive."),
            Self::NonFiniteFirstStep => write!(f, "`first_step` must be finite."),
            Self::NonFiniteMaxStep => write!(f, "`max_step` must not be NaN."),
            Self::NonFiniteRtol => write!(f, "`rtol` must not be NaN."),
            Self::NonFiniteAtol => write!(f, "`atol` must not be NaN."),
            Self::AtolWrongShape { .. } => write!(f, "`atol` has wrong shape."),
            Self::AtolMustBePositive => write!(f, "`atol` must be positive."),
            Self::NonFiniteY0 => write!(f, "`y0` must be finite in Hardened mode."),
            Self::NonFiniteSpan => write!(f, "`t_span` must be finite."),
            Self::NonFiniteF0 => write!(f, "`f0` must be finite in Hardened mode."),
            Self::RhsWrongShape { expected, actual } => write!(
                f,
                "right-hand side returned {actual} derivative values, expected {expected}."
            ),
            Self::NonFiniteEventDirection { index } => {
                write!(f, "event {index} has a non-finite direction.")
            }
            Self::EventMaxEventsMustBePositive { index } => {
                write!(f, "event {index} max_events must be positive.")
            }
            Self::NonFiniteEventValue { index } => {
                write!(f, "event {index} returned a non-finite value.")
            }
            Self::TEvalOutOfSpan => write!(f, "Values in `t_eval` are not within `t_span`."),
            Self::TEvalNotSorted => write!(f, "Values in `t_eval` are not properly sorted."),
            Self::NotYetImplemented { function } => {
                write!(f, "`{function}` is planned but not implemented yet.")
            }
            Self::QuadInvalidBounds { detail } => write!(f, "{detail}"),
            Self::QuadInvalidTolerance { detail } => write!(f, "{detail}"),
            Self::LebedevOrderUnavailable { order } => write!(
                f,
                "Lebedev order {order} not available. Available orders are 3, 5, 7, \
                 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 41, 47, 53, 59, \
                 65, 71, 77, 83, 89, 95, 101, 107, 113, 119, 125, 131."
            ),
            Self::IntegrationFailed { message } => write!(f, "integration failed: {message}"),
        }
    }
}

impl std::error::Error for IntegrateValidationError {}

pub(crate) fn validate_rhs_shape(
    actual: usize,
    expected: usize,
) -> Result<(), IntegrateValidationError> {
    if actual != expected {
        return Err(IntegrateValidationError::RhsWrongShape { expected, actual });
    }
    Ok(())
}

pub fn validate_first_step(
    first_step: f64,
    t0: f64,
    t_bound: f64,
) -> Result<f64, IntegrateValidationError> {
    validate_first_step_with_audit(first_step, t0, t_bound, None)
}

pub fn validate_first_step_with_audit(
    first_step: f64,
    t0: f64,
    t_bound: f64,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<f64, IntegrateValidationError> {
    if !first_step.is_finite() {
        let fingerprint = audit_fingerprint(
            "validate_first_step",
            format!("first_step={first_step};t0={t0};t_bound={t_bound}"),
        );
        record_fail_closed(
            audit_ledger,
            &fingerprint,
            "first_step_must_be_finite",
            "rejected",
        );
        return Err(IntegrateValidationError::NonFiniteFirstStep);
    }
    if first_step <= 0.0 {
        let fingerprint = audit_fingerprint(
            "validate_first_step",
            format!("first_step={first_step};t0={t0};t_bound={t_bound}"),
        );
        record_fail_closed(
            audit_ledger,
            &fingerprint,
            "first_step_must_be_positive",
            "rejected",
        );
        return Err(IntegrateValidationError::FirstStepMustBePositive);
    }
    if first_step > (t_bound - t0).abs() {
        let fingerprint = audit_fingerprint(
            "validate_first_step",
            format!("first_step={first_step};t0={t0};t_bound={t_bound}"),
        );
        record_fail_closed(
            audit_ledger,
            &fingerprint,
            "first_step_exceeds_bounds",
            "rejected",
        );
        return Err(IntegrateValidationError::FirstStepExceedsBounds);
    }
    Ok(first_step)
}

pub fn validate_max_step(max_step: f64) -> Result<f64, IntegrateValidationError> {
    validate_max_step_with_audit(max_step, None)
}

pub fn validate_max_step_with_audit(
    max_step: f64,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<f64, IntegrateValidationError> {
    if max_step.is_nan() {
        let fingerprint = audit_fingerprint("validate_max_step", format!("max_step={max_step}"));
        record_fail_closed(
            audit_ledger,
            &fingerprint,
            "max_step_must_not_be_nan",
            "rejected",
        );
        return Err(IntegrateValidationError::NonFiniteMaxStep);
    }
    if max_step <= 0.0 {
        let fingerprint = audit_fingerprint("validate_max_step", format!("max_step={max_step}"));
        record_fail_closed(
            audit_ledger,
            &fingerprint,
            "max_step_must_be_positive",
            "rejected",
        );
        return Err(IntegrateValidationError::MaxStepMustBePositive);
    }
    Ok(max_step)
}

pub fn validate_tol(
    rtol: ToleranceValue,
    atol: ToleranceValue,
    n: usize,
    mode: RuntimeMode,
) -> Result<ValidatedTolerance, IntegrateValidationError> {
    validate_tol_with_audit(rtol, atol, n, mode, None)
}

pub fn validate_tol_with_audit(
    rtol: ToleranceValue,
    atol: ToleranceValue,
    n: usize,
    mode: RuntimeMode,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<ValidatedTolerance, IntegrateValidationError> {
    let mut warnings = Vec::new();
    let fingerprint = audit_ledger.map(|_| {
        audit_fingerprint(
            "validate_tol",
            format!("rtol={rtol:?};atol={atol:?};n={n};mode={mode:?}"),
        )
    });
    let fingerprint = fingerprint.as_deref().unwrap_or_default();
    // NaN in rtol or atol falls through every `<` and `<=` predicate
    // silently. Reject up front so Hardened callers see a fail-closed
    // error rather than NaN propagating into the adaptive step controller.
    // Per frankenscipy-i9vw.
    if rtol.any(|x| x.is_nan()) {
        record_fail_closed(
            audit_ledger,
            fingerprint,
            "rtol_must_not_be_nan",
            "rejected",
        );
        return Err(IntegrateValidationError::NonFiniteRtol);
    }
    if atol.any(|x| x.is_nan()) {
        record_fail_closed(
            audit_ledger,
            fingerprint,
            "atol_must_not_be_nan",
            "rejected",
        );
        return Err(IntegrateValidationError::NonFiniteAtol);
    }
    let needs_clamp = rtol.any(|x| x < MIN_RTOL);
    let rtol = if needs_clamp {
        warnings.push(ToleranceWarning::RtolClamped { minimum: MIN_RTOL });
        if mode == RuntimeMode::Hardened {
            record_bounded_recovery(
                audit_ledger,
                fingerprint,
                "clamp_rtol_to_min",
                &format!("clamped_rtol_to_{MIN_RTOL:.3e}"),
            );
        }
        rtol.map(|x| x.max(MIN_RTOL))
    } else {
        rtol
    };

    if let Some(len) = atol.len_if_vector()
        && len != n
    {
        record_fail_closed(audit_ledger, fingerprint, "atol_wrong_shape", "rejected");
        return Err(IntegrateValidationError::AtolWrongShape {
            expected: n,
            actual: len,
        });
    }

    if atol.any(|x| x < 0.0) {
        record_fail_closed(
            audit_ledger,
            fingerprint,
            "atol_must_be_positive",
            "rejected",
        );
        return Err(IntegrateValidationError::AtolMustBePositive);
    }

    Ok(ValidatedTolerance {
        rtol,
        atol,
        mode,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── validate_tol scalar tests ────────────────────────────────

    // 1. rtol within range -> passthrough (Strict)
    #[test]
    fn test_validation_tol_scalar_rtol_within_range_strict() {
        let report = validate_tol(
            ToleranceValue::Scalar(1e-3),
            ToleranceValue::Scalar(1e-6),
            3,
            RuntimeMode::Strict,
        )
        .expect("valid tolerances");
        assert!(report.warnings.is_empty());
        assert_eq!(report.rtol, ToleranceValue::Scalar(1e-3));
    }

    // 1b. rtol within range -> passthrough (Hardened)
    #[test]
    fn test_validation_tol_scalar_rtol_within_range_hardened() {
        let report = validate_tol(
            ToleranceValue::Scalar(1e-3),
            ToleranceValue::Scalar(1e-6),
            3,
            RuntimeMode::Hardened,
        )
        .expect("valid tolerances");
        assert!(report.warnings.is_empty());
        assert_eq!(report.mode, RuntimeMode::Hardened);
    }

    // 2. rtol below MIN_RTOL -> clamped with warning
    #[test]
    fn test_validation_tol_scalar_rtol_below_min_clamped() {
        let report = validate_tol(
            ToleranceValue::Scalar(1e-30),
            ToleranceValue::Scalar(1e-8),
            3,
            RuntimeMode::Strict,
        )
        .expect("tolerance should validate");
        assert_eq!(
            report,
            ValidatedTolerance {
                rtol: ToleranceValue::Scalar(MIN_RTOL),
                atol: ToleranceValue::Scalar(1e-8),
                mode: RuntimeMode::Strict,
                warnings: vec![ToleranceWarning::RtolClamped { minimum: MIN_RTOL }],
            }
        );
    }

    // 3. rtol = 0.0 -> clamped
    #[test]
    fn test_validation_tol_scalar_rtol_zero_clamped() {
        let report = validate_tol(
            ToleranceValue::Scalar(0.0),
            ToleranceValue::Scalar(1e-6),
            1,
            RuntimeMode::Strict,
        )
        .expect("zero rtol should be clamped");
        assert!(!report.warnings.is_empty());
        assert!(matches!(report.rtol, ToleranceValue::Scalar(_)));
        if let ToleranceValue::Scalar(v) = report.rtol {
            assert_eq!(v, MIN_RTOL);
        }
    }

    // 4. negative rtol -> clamped (SciPy clamps, doesn't error)
    #[test]
    fn test_validation_tol_scalar_negative_rtol_clamped() {
        // SciPy's behavior: negative rtol gets clamped to MIN_RTOL
        let report = validate_tol(
            ToleranceValue::Scalar(-1.0),
            ToleranceValue::Scalar(1e-6),
            1,
            RuntimeMode::Strict,
        )
        .expect("negative rtol should be clamped");
        assert!(!report.warnings.is_empty());
    }

    // ── validate_tol vector tests ────────────────────────────────

    // 5. matching dimension -> passthrough
    #[test]
    fn test_validation_tol_vector_matching_dim() {
        let report = validate_tol(
            ToleranceValue::Scalar(1e-3),
            ToleranceValue::Vector(vec![1e-6, 1e-7, 1e-8]),
            3,
            RuntimeMode::Strict,
        )
        .expect("matching vector atol should succeed");
        assert!(report.warnings.is_empty());
    }

    #[test]
    fn test_validation_tol_borrowed_vector_scans_preserve_bits() {
        let rtol = vec![1e-3, MIN_RTOL, f64::INFINITY, 1e-6];
        let atol = vec![0.0, -0.0, 1e-9, f64::INFINITY];
        let expected_rtol_bits = rtol.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
        let expected_atol_bits = atol.iter().map(|value| value.to_bits()).collect::<Vec<_>>();

        let report = validate_tol(
            ToleranceValue::Vector(rtol),
            ToleranceValue::Vector(atol),
            4,
            RuntimeMode::Strict,
        )
        .expect("valid vector tolerances should remain unchanged");

        assert_eq!(report.mode, RuntimeMode::Strict);
        assert!(report.warnings.is_empty());
        assert!(matches!(
            &report.rtol,
            ToleranceValue::Vector(actual_rtol)
                if actual_rtol
                    .iter()
                    .map(|value| value.to_bits())
                    .eq(expected_rtol_bits.iter().copied())
        ));
        assert!(matches!(
            &report.atol,
            ToleranceValue::Vector(actual_atol)
                if actual_atol
                    .iter()
                    .map(|value| value.to_bits())
                    .eq(expected_atol_bits.iter().copied())
        ));
    }

    // 6. wrong dimension -> AtolWrongShape error
    #[test]
    fn test_validation_tol_vector_wrong_dim() {
        let err = validate_tol(
            ToleranceValue::Scalar(1e-6),
            ToleranceValue::Vector(vec![1e-9, 1e-9]),
            3,
            RuntimeMode::Strict,
        )
        .expect_err("wrong atol shape must fail");
        assert_eq!(
            err,
            IntegrateValidationError::AtolWrongShape {
                expected: 3,
                actual: 2
            }
        );
    }

    // 7. negative element -> AtolMustBePositive error
    #[test]
    fn test_validation_tol_vector_negative_element() {
        let err = validate_tol(
            ToleranceValue::Scalar(1e-6),
            ToleranceValue::Vector(vec![1e-9, -1e-9, 1e-9]),
            3,
            RuntimeMode::Hardened,
        )
        .expect_err("negative atol must fail");
        assert_eq!(err, IntegrateValidationError::AtolMustBePositive);
    }

    // 8. NaN input in atol -> rejected (was accepted pre-i9vw)
    #[test]
    fn test_validation_tol_nan_atol_rejected() {
        // Per frankenscipy-i9vw: NaN atol now fails closed rather than
        // falling through every `<` predicate silently.
        let err = validate_tol(
            ToleranceValue::Scalar(1e-3),
            ToleranceValue::Scalar(f64::NAN),
            1,
            RuntimeMode::Strict,
        )
        .expect_err("NaN atol should fail closed");
        assert_eq!(err, IntegrateValidationError::NonFiniteAtol);
    }

    #[test]
    fn test_validation_tol_nan_rtol_rejected() {
        let err = validate_tol(
            ToleranceValue::Scalar(f64::NAN),
            ToleranceValue::Scalar(1e-6),
            1,
            RuntimeMode::Strict,
        )
        .expect_err("NaN rtol should fail closed");
        assert_eq!(err, IntegrateValidationError::NonFiniteRtol);
    }

    // 9. Inf input in atol -> accepted (SciPy allows)
    #[test]
    fn test_validation_tol_inf_atol() {
        let report = validate_tol(
            ToleranceValue::Scalar(1e-3),
            ToleranceValue::Scalar(f64::INFINITY),
            1,
            RuntimeMode::Strict,
        );
        assert!(report.is_ok());
    }

    // ── validate_first_step tests ────────────────────────────────

    // 10. positive within bounds -> accepted
    #[test]
    fn test_validation_first_step_positive_within_bounds() {
        let result = validate_first_step(0.5, 0.0, 1.0);
        assert_eq!(result.unwrap(), 0.5);
    }

    // 11. zero -> FirstStepMustBePositive error
    #[test]
    fn test_validation_first_step_zero() {
        let err = validate_first_step(0.0, 0.0, 1.0).expect_err("must reject zero");
        assert_eq!(err, IntegrateValidationError::FirstStepMustBePositive);
    }

    // 12. negative -> FirstStepMustBePositive error
    #[test]
    fn test_validation_first_step_negative() {
        let err = validate_first_step(-0.1, 0.0, 1.0).expect_err("must reject negative");
        assert_eq!(err, IntegrateValidationError::FirstStepMustBePositive);
    }

    // 13. exceeds bounds -> FirstStepExceedsBounds error
    #[test]
    fn test_validation_first_step_exceeds_bounds() {
        let err = validate_first_step(2.0, 0.0, 1.0).expect_err("must reject out-of-bounds step");
        assert_eq!(err, IntegrateValidationError::FirstStepExceedsBounds);
    }

    #[test]
    fn test_validation_first_step_nan_rejected() {
        let err = validate_first_step(f64::NAN, 0.0, 1.0).expect_err("must reject NaN step");
        assert_eq!(err, IntegrateValidationError::NonFiniteFirstStep);
    }

    // ── validate_max_step tests ──────────────────────────────────

    // 14. positive -> accepted
    #[test]
    fn test_validation_max_step_positive() {
        assert_eq!(validate_max_step(1.0).unwrap(), 1.0);
    }

    // 15. zero -> MaxStepMustBePositive error
    #[test]
    fn test_validation_max_step_zero() {
        let err = validate_max_step(0.0).expect_err("must reject zero");
        assert_eq!(err, IntegrateValidationError::MaxStepMustBePositive);
    }

    // 16. negative -> MaxStepMustBePositive error
    #[test]
    fn test_validation_max_step_negative() {
        let err = validate_max_step(-1.0).expect_err("must reject negative max step");
        assert_eq!(err, IntegrateValidationError::MaxStepMustBePositive);
    }

    // 17. Inf -> accepted (SciPy allows this as the default)
    #[test]
    fn test_validation_max_step_infinity() {
        assert_eq!(validate_max_step(f64::INFINITY).unwrap(), f64::INFINITY);
    }

    #[test]
    fn test_validation_max_step_nan_rejected() {
        let err = validate_max_step(f64::NAN).expect_err("must reject NaN max step");
        assert_eq!(err, IntegrateValidationError::NonFiniteMaxStep);
    }

    // ── Mode-specific tests ──────────────────────────────────────

    // 18. Strict mode: rtol clamping preserves SciPy semantics
    #[test]
    fn test_validation_tol_strict_clamping_scipy_semantics() {
        let report = validate_tol(
            ToleranceValue::Scalar(1e-20),
            ToleranceValue::Scalar(1e-8),
            2,
            RuntimeMode::Strict,
        )
        .expect("strict mode should clamp");
        assert_eq!(report.mode, RuntimeMode::Strict);
        assert!(!report.warnings.is_empty());
        assert!(matches!(report.rtol, ToleranceValue::Scalar(_)));
        if let ToleranceValue::Scalar(v) = report.rtol {
            assert!(v >= MIN_RTOL);
        }
    }

    // 19. Hardened mode: rtol clamping + finite check
    #[test]
    fn test_validation_tol_hardened_clamping() {
        let report = validate_tol(
            ToleranceValue::Scalar(1e-20),
            ToleranceValue::Scalar(1e-8),
            2,
            RuntimeMode::Hardened,
        )
        .expect("hardened mode should clamp");
        assert_eq!(report.mode, RuntimeMode::Hardened);
        assert!(!report.warnings.is_empty());
    }

    // 20. Round-trip: reasonable inputs -> no warnings
    #[test]
    fn test_validation_tol_roundtrip_no_warnings() {
        let report = validate_tol(
            ToleranceValue::Scalar(1e-3),
            ToleranceValue::Vector(vec![1e-6, 1e-7]),
            2,
            RuntimeMode::Strict,
        )
        .expect("reasonable inputs");
        assert!(report.warnings.is_empty());
        assert_eq!(report.atol, ToleranceValue::Vector(vec![1e-6, 1e-7]));
    }

    // ── Edge cases ───────────────────────────────────────────────

    // 21. Empty system (n=0)
    #[test]
    fn test_validation_tol_empty_system() {
        let report = validate_tol(
            ToleranceValue::Scalar(1e-3),
            ToleranceValue::Scalar(1e-6),
            0,
            RuntimeMode::Strict,
        )
        .expect("empty system should be valid");
        assert!(report.warnings.is_empty());
    }

    // 22. Very large n
    #[test]
    fn test_validation_tol_large_n() {
        let report = validate_tol(
            ToleranceValue::Scalar(1e-3),
            ToleranceValue::Scalar(1e-6),
            10000,
            RuntimeMode::Strict,
        )
        .expect("large n should be valid with scalar atol");
        assert!(report.warnings.is_empty());
    }

    // 23. Extreme small tolerance (1e-300)
    #[test]
    fn test_validation_tol_extreme_small() {
        let report = validate_tol(
            ToleranceValue::Scalar(1e-300),
            ToleranceValue::Scalar(1e-300),
            1,
            RuntimeMode::Strict,
        )
        .expect("extreme small should be clamped");
        assert!(!report.warnings.is_empty());
    }

    // 24. Extreme large tolerance (1e300)
    #[test]
    fn test_validation_tol_extreme_large() {
        let report = validate_tol(
            ToleranceValue::Scalar(1e300),
            ToleranceValue::Scalar(1e300),
            1,
            RuntimeMode::Strict,
        )
        .expect("extreme large should be valid");
        assert!(report.warnings.is_empty());
    }

    // 25. First step at exact boundary
    #[test]
    fn test_validation_first_step_exact_boundary() {
        let result = validate_first_step(1.0, 0.0, 1.0);
        assert_eq!(result.unwrap(), 1.0);
    }

    // 26. Backward integration first step
    #[test]
    fn test_validation_first_step_backward() {
        let result = validate_first_step(0.5, 1.0, 0.0);
        assert_eq!(result.unwrap(), 0.5);
    }

    // 27. Vector atol with zero elements
    #[test]
    fn test_validation_tol_vector_zero_element() {
        let report = validate_tol(
            ToleranceValue::Scalar(1e-3),
            ToleranceValue::Vector(vec![0.0, 1e-6]),
            2,
            RuntimeMode::Strict,
        )
        .expect("zero atol element should be valid");
        assert!(report.warnings.is_empty());
    }

    #[test]
    fn test_validation_tol_hardened_clamp_records_bounded_recovery() {
        let audit_ledger = sync_audit_ledger();
        let report = validate_tol_with_audit(
            ToleranceValue::Scalar(0.0),
            ToleranceValue::Scalar(1e-6),
            1,
            RuntimeMode::Hardened,
            Some(&audit_ledger),
        )
        .expect("hardened clamp should succeed");
        assert_eq!(report.rtol, ToleranceValue::Scalar(MIN_RTOL));

        let ledger = audit_ledger.lock().expect("lock");
        assert_eq!(ledger.len(), 1);
        assert!(matches!(
            ledger.entries()[0].action,
            AuditAction::BoundedRecovery {
                ref recovery_action
            } if recovery_action.as_str().eq("clamp_rtol_to_min")
        ));
    }

    #[test]
    fn test_validation_first_step_records_fail_closed() {
        let audit_ledger = sync_audit_ledger();
        let err = validate_first_step_with_audit(0.0, 0.0, 1.0, Some(&audit_ledger))
            .expect_err("zero first_step must fail");
        assert_eq!(err, IntegrateValidationError::FirstStepMustBePositive);

        let ledger = audit_ledger.lock().expect("lock");
        assert_eq!(ledger.len(), 1);
        assert!(matches!(
            ledger.entries()[0].action,
            AuditAction::FailClosed { ref reason } if reason == "first_step_must_be_positive"
        ));
    }

    #[test]
    fn test_validation_tol_nan_records_fail_closed() {
        let audit_ledger = sync_audit_ledger();
        let expected_fingerprint = AuditLedger::fingerprint_bytes(&audit_fingerprint(
            "validate_tol",
            format!(
                "rtol={:?};atol={:?};n=1;mode={:?}",
                ToleranceValue::Scalar(f64::NAN),
                ToleranceValue::Scalar(1e-6),
                RuntimeMode::Hardened,
            ),
        ));
        let err = validate_tol_with_audit(
            ToleranceValue::Scalar(f64::NAN),
            ToleranceValue::Scalar(1e-6),
            1,
            RuntimeMode::Hardened,
            Some(&audit_ledger),
        )
        .expect_err("NaN rtol must fail closed");
        assert_eq!(err, IntegrateValidationError::NonFiniteRtol);

        let ledger = audit_ledger.lock().expect("lock");
        assert_eq!(ledger.len(), 1);
        assert_eq!(ledger.entries()[0].input_fingerprint, expected_fingerprint);
        assert!(matches!(
            ledger.entries()[0].action,
            AuditAction::FailClosed { ref reason } if reason == "rtol_must_not_be_nan"
        ));
    }

    #[test]
    fn test_validation_audit_records_after_poisoned_ledger() {
        let audit_ledger = sync_audit_ledger();

        let poisoned_thread = {
            let ledger = audit_ledger.clone();
            std::thread::spawn(move || {
                let _guard = ledger.lock().expect("acquire ledger");
                std::panic::panic_any("poison fsci-integrate audit ledger on purpose");
            })
            .join()
        };
        assert!(poisoned_thread.is_err(), "thread should have panicked");
        assert!(
            audit_ledger.lock().is_err(),
            "ledger must be poisoned after panic"
        );

        record_fail_closed(
            Some(&audit_ledger),
            b"bad first step",
            "first_step_must_be_positive",
            "rejected",
        );
        record_bounded_recovery(
            Some(&audit_ledger),
            b"small rtol",
            "clamp_rtol_to_min",
            "recovered",
        );

        let ledger = audit_ledger
            .lock()
            .expect("ledger poison should have been cleared");
        assert_eq!(ledger.len(), 2);
    }
}
