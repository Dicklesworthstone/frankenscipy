#![forbid(unsafe_code)]

use crate::types::{MinimizeOptions, OptError, OptimizeMethod, OptimizeResult};

pub fn minimize<F>(
    _fun: F,
    x0: &[f64],
    options: MinimizeOptions,
) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    if x0.is_empty() {
        return Err(OptError::InvalidArgument {
            detail: String::from("x0 must be a finite 1-D vector with at least one element"),
        });
    }
    if x0.iter().any(|v| !v.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: String::from("x0 must not contain NaN or Inf"),
        });
    }

    match options.method.unwrap_or(OptimizeMethod::Bfgs) {
        OptimizeMethod::Bfgs => bfgs(x0, options),
        OptimizeMethod::ConjugateGradient => cg_pr_plus(x0, options),
        OptimizeMethod::Powell => powell(x0, options),
    }
}

pub fn bfgs(x0: &[f64], _options: MinimizeOptions) -> Result<OptimizeResult, OptError> {
    Ok(OptimizeResult::not_implemented(
        x0,
        "BFGS kernel not implemented yet",
    ))
}

pub fn cg_pr_plus(x0: &[f64], _options: MinimizeOptions) -> Result<OptimizeResult, OptError> {
    Ok(OptimizeResult::not_implemented(
        x0,
        "CG(PR+) kernel not implemented yet",
    ))
}

pub fn powell(x0: &[f64], _options: MinimizeOptions) -> Result<OptimizeResult, OptError> {
    Ok(OptimizeResult::not_implemented(
        x0,
        "Powell kernel not implemented yet",
    ))
}
