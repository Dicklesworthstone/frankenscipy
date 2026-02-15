#![forbid(unsafe_code)]

use crate::types::{ConvergenceStatus, OptError, RootMethod, RootOptions};

#[derive(Debug, Clone, PartialEq)]
pub struct RootResult {
    pub root: f64,
    pub converged: bool,
    pub status: ConvergenceStatus,
    pub iterations: usize,
    pub function_calls: usize,
    pub method: RootMethod,
    pub message: String,
}

impl RootResult {
    #[must_use]
    pub fn not_implemented(method: RootMethod) -> Self {
        Self {
            root: f64::NAN,
            converged: false,
            status: ConvergenceStatus::NotImplemented,
            iterations: 0,
            function_calls: 0,
            method,
            message: format!("{method:?} root kernel not implemented yet"),
        }
    }
}

pub fn root_scalar<F>(
    _f: F,
    bracket: Option<(f64, f64)>,
    x0: Option<f64>,
    x1: Option<f64>,
    options: RootOptions,
) -> Result<RootResult, OptError>
where
    F: Fn(f64) -> f64,
{
    let method = if let Some(method) = options.method {
        method
    } else if bracket.is_some() {
        RootMethod::Brentq
    } else if x0.is_some() && x1.is_some() {
        RootMethod::Ridder
    } else {
        return Err(OptError::InvalidArgument {
            detail: String::from(
                "unable to select a root solver: provide bracket or explicit method",
            ),
        });
    };

    match method {
        RootMethod::Brentq => {
            let interval = require_bracket(bracket)?;
            brentq(interval, options)
        }
        RootMethod::Brenth => {
            let interval = require_bracket(bracket)?;
            brenth(interval, options)
        }
        RootMethod::Bisect => {
            let interval = require_bracket(bracket)?;
            bisect(interval, options)
        }
        RootMethod::Ridder => {
            let interval = require_bracket(bracket)?;
            ridder(interval, options)
        }
    }
}

pub fn brentq(bracket: (f64, f64), options: RootOptions) -> Result<RootResult, OptError> {
    validate_root_options(options)?;
    validate_bracket_finite(bracket)?;
    Ok(RootResult::not_implemented(RootMethod::Brentq))
}

pub fn brenth(bracket: (f64, f64), options: RootOptions) -> Result<RootResult, OptError> {
    validate_root_options(options)?;
    validate_bracket_finite(bracket)?;
    Ok(RootResult::not_implemented(RootMethod::Brenth))
}

pub fn bisect(bracket: (f64, f64), options: RootOptions) -> Result<RootResult, OptError> {
    validate_root_options(options)?;
    validate_bracket_finite(bracket)?;
    Ok(RootResult::not_implemented(RootMethod::Bisect))
}

pub fn ridder(bracket: (f64, f64), options: RootOptions) -> Result<RootResult, OptError> {
    validate_root_options(options)?;
    validate_bracket_finite(bracket)?;
    Ok(RootResult::not_implemented(RootMethod::Ridder))
}

fn require_bracket(bracket: Option<(f64, f64)>) -> Result<(f64, f64), OptError> {
    bracket.ok_or_else(|| OptError::InvalidArgument {
        detail: String::from("bracket is required for bracketing root methods"),
    })
}

fn validate_root_options(options: RootOptions) -> Result<(), OptError> {
    if options.xtol <= 0.0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("xtol must be > 0"),
        });
    }
    if options.maxiter == 0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("maxiter must be >= 1"),
        });
    }
    Ok(())
}

fn validate_bracket_finite(bracket: (f64, f64)) -> Result<(), OptError> {
    if !bracket.0.is_finite() || !bracket.1.is_finite() {
        return Err(OptError::NonFiniteInput {
            detail: String::from("bracket endpoints must be finite"),
        });
    }
    if bracket.0 >= bracket.1 {
        return Err(OptError::InvalidBounds {
            detail: String::from("bracket must satisfy a < b"),
        });
    }
    Ok(())
}
