#![forbid(unsafe_code)]

use fsci_runtime::RuntimeMode;

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

    fn any(self, mut predicate: impl FnMut(f64) -> bool) -> bool {
        match self {
            Self::Scalar(value) => predicate(value),
            Self::Vector(values) => values.into_iter().any(predicate),
        }
    }

    fn len_if_vector(&self) -> Option<usize> {
        match self {
            Self::Scalar(_) => None,
            Self::Vector(values) => Some(values.len()),
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
    FirstStepMustBePositive,
    FirstStepExceedsBounds,
    MaxStepMustBePositive,
    AtolWrongShape { expected: usize, actual: usize },
    AtolMustBePositive,
}

impl std::fmt::Display for IntegrateValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FirstStepMustBePositive => write!(f, "`first_step` must be positive."),
            Self::FirstStepExceedsBounds => write!(f, "`first_step` exceeds bounds."),
            Self::MaxStepMustBePositive => write!(f, "`max_step` must be positive."),
            Self::AtolWrongShape { .. } => write!(f, "`atol` has wrong shape."),
            Self::AtolMustBePositive => write!(f, "`atol` must be positive."),
        }
    }
}

impl std::error::Error for IntegrateValidationError {}

pub fn validate_first_step(
    first_step: f64,
    t0: f64,
    t_bound: f64,
) -> Result<f64, IntegrateValidationError> {
    if first_step <= 0.0 {
        return Err(IntegrateValidationError::FirstStepMustBePositive);
    }
    if first_step > (t_bound - t0).abs() {
        return Err(IntegrateValidationError::FirstStepExceedsBounds);
    }
    Ok(first_step)
}

pub fn validate_max_step(max_step: f64) -> Result<f64, IntegrateValidationError> {
    if max_step <= 0.0 {
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
    let mut warnings = Vec::new();
    let needs_clamp = rtol.clone().any(|x| x < MIN_RTOL);
    let rtol = if needs_clamp {
        warnings.push(ToleranceWarning::RtolClamped { minimum: MIN_RTOL });
        rtol.map(|x| x.max(MIN_RTOL))
    } else {
        rtol
    };

    if let Some(len) = atol.len_if_vector()
        && len != n
    {
        return Err(IntegrateValidationError::AtolWrongShape {
            expected: n,
            actual: len,
        });
    }

    if atol.clone().any(|x| x < 0.0) {
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

    #[test]
    fn first_step_must_be_positive() {
        let err = validate_first_step(0.0, 0.0, 1.0).expect_err("must reject non-positive step");
        assert_eq!(err.to_string(), "`first_step` must be positive.");
    }

    #[test]
    fn first_step_respects_bounds() {
        let err = validate_first_step(2.0, 0.0, 1.0).expect_err("must reject out-of-bounds step");
        assert_eq!(err.to_string(), "`first_step` exceeds bounds.");
    }

    #[test]
    fn max_step_must_be_positive() {
        let err = validate_max_step(-1.0).expect_err("must reject negative max step");
        assert_eq!(err.to_string(), "`max_step` must be positive.");
    }

    #[test]
    fn validate_tol_clamps_small_rtol() {
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

    #[test]
    fn validate_tol_rejects_bad_atol_shape() {
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

    #[test]
    fn validate_tol_rejects_negative_atol() {
        let err = validate_tol(
            ToleranceValue::Scalar(1e-6),
            ToleranceValue::Vector(vec![1e-9, -1e-9, 1e-9]),
            3,
            RuntimeMode::Hardened,
        )
        .expect_err("negative atol must fail");
        assert_eq!(err, IntegrateValidationError::AtolMustBePositive);
    }
}
