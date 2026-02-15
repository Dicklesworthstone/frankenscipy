#![forbid(unsafe_code)]

use crate::types::OptError;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WolfeParams {
    pub c1: f64,
    pub c2: f64,
    pub amax: f64,
    pub amin: f64,
    pub maxiter: usize,
}

impl Default for WolfeParams {
    fn default() -> Self {
        Self {
            c1: 1.0e-4,
            c2: 0.9,
            amax: 50.0,
            amin: 1.0e-8,
            maxiter: 10,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineSearchResult {
    pub alpha: f64,
    pub f_at_alpha: f64,
    pub directional_derivative: f64,
    pub evaluations: usize,
}

pub fn validate_wolfe_params(params: WolfeParams) -> Result<(), OptError> {
    if !(0.0 < params.c1 && params.c1 < params.c2 && params.c2 < 1.0) {
        return Err(OptError::InvalidArgument {
            detail: String::from("Wolfe constants must satisfy 0 < c1 < c2 < 1"),
        });
    }
    if params.amin <= 0.0 || params.amax <= 0.0 || params.amin >= params.amax {
        return Err(OptError::InvalidArgument {
            detail: String::from("line-search alpha bounds must satisfy 0 < amin < amax"),
        });
    }
    if params.maxiter == 0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("line-search maxiter must be >= 1"),
        });
    }
    Ok(())
}

pub fn line_search_wolfe1(_params: WolfeParams) -> Result<LineSearchResult, OptError> {
    Err(OptError::NotImplemented {
        detail: String::from("Wolfe1 line-search kernel is not implemented yet"),
    })
}

pub fn line_search_wolfe2(_params: WolfeParams) -> Result<LineSearchResult, OptError> {
    Err(OptError::NotImplemented {
        detail: String::from("Wolfe2 line-search kernel is not implemented yet"),
    })
}
