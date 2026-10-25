#![forbid(unsafe_code)]

use fsci_runtime::RuntimeMode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OdeSolverState {
    Running,
    Finished,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepOutcome {
    pub message: Option<String>,
    pub state: OdeSolverState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepFailure {
    RuntimeError(&'static str),
    StepSizeTooSmall,
    ConvergenceFailure,
    SolverError,
    NonFiniteState,
    InvalidJacobian(crate::IntegrateValidationError),
    NotYetImplemented(&'static str),
}

pub trait OdeSolver {
    fn mode(&self) -> RuntimeMode;

    fn state(&self) -> OdeSolverState;

    fn step(&mut self) -> Result<StepOutcome, StepFailure>;
}
