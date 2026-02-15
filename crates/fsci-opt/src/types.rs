#![forbid(unsafe_code)]

use fsci_runtime::RuntimeMode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizeMethod {
    Bfgs,
    ConjugateGradient,
    Powell,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootMethod {
    Brentq,
    Brenth,
    Bisect,
    Ridder,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergenceStatus {
    Success,
    MaxIterations,
    MaxEvaluations,
    PrecisionLoss,
    NanEncountered,
    OutOfBounds,
    CallbackStop,
    NotImplemented,
    InvalidInput,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OptimizeResult {
    pub x: Vec<f64>,
    pub fun: Option<f64>,
    pub success: bool,
    pub status: ConvergenceStatus,
    pub message: String,
    pub nfev: usize,
    pub njev: usize,
    pub nhev: usize,
    pub nit: usize,
    pub jac: Option<Vec<f64>>,
    pub hess_inv: Option<Vec<Vec<f64>>>,
    pub maxcv: Option<f64>,
}

impl OptimizeResult {
    #[must_use]
    pub fn not_implemented(seed: &[f64], message: impl Into<String>) -> Self {
        Self {
            x: seed.to_vec(),
            fun: None,
            success: false,
            status: ConvergenceStatus::NotImplemented,
            message: message.into(),
            nfev: 0,
            njev: 0,
            nhev: 0,
            nit: 0,
            jac: None,
            hess_inv: None,
            maxcv: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MinimizeOptions {
    pub method: Option<OptimizeMethod>,
    pub tol: Option<f64>,
    pub maxiter: Option<usize>,
    pub maxfev: Option<usize>,
    pub mode: RuntimeMode,
}

impl Default for MinimizeOptions {
    fn default() -> Self {
        Self {
            method: None,
            tol: None,
            maxiter: None,
            maxfev: None,
            mode: RuntimeMode::Strict,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RootOptions {
    pub method: Option<RootMethod>,
    pub xtol: f64,
    pub rtol: f64,
    pub maxiter: usize,
    pub mode: RuntimeMode,
}

impl Default for RootOptions {
    fn default() -> Self {
        Self {
            method: None,
            xtol: 2.0e-12,
            rtol: 8.881_784_197_001_252e-16,
            maxiter: 100,
            mode: RuntimeMode::Strict,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptError {
    InvalidArgument { detail: String },
    InvalidBounds { detail: String },
    SignChangeRequired { detail: String },
    NonFiniteInput { detail: String },
    NotImplemented { detail: String },
}

impl std::fmt::Display for OptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArgument { detail } => write!(f, "{detail}"),
            Self::InvalidBounds { detail } => write!(f, "{detail}"),
            Self::SignChangeRequired { detail } => write!(f, "{detail}"),
            Self::NonFiniteInput { detail } => write!(f, "{detail}"),
            Self::NotImplemented { detail } => write!(f, "{detail}"),
        }
    }
}

impl std::error::Error for OptError {}
