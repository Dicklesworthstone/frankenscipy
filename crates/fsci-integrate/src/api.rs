#![forbid(unsafe_code)]

use fsci_opt::root::brentq;
use fsci_opt::types::RootOptions;
use fsci_runtime::{
    AuditScope, Fingerprinter, OdeSolverAction, OdeSolverEvidenceEntry, OdeSolverPortfolio,
    RuntimeMode,
};

use crate::IntegrateValidationError;
use crate::bdf::{BdfSolver, BdfSolverConfig};
use crate::rk::{RK23_TABLEAU, RK45_TABLEAU, RkSolver, RkSolverConfig};
use crate::solver::{OdeSolver, OdeSolverState, StepFailure, StepOutcome};
use crate::validation::{
    SyncSharedAuditLedger, ToleranceValue, fail_closed, fingerprint_optional_f64,
    fingerprint_tolerance, validate_first_step_scoped, validate_max_step_scoped,
    validate_tol_scoped,
};

pub type EventFn = fn(f64, &[f64]) -> f64;

#[derive(Debug, Clone, Copy)]
pub struct EventSpec {
    pub func: EventFn,
    pub direction: f64,
    pub max_events: Option<usize>,
}

impl PartialEq for EventSpec {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::fn_addr_eq(self.func, other.func)
            && self.direction.to_bits() == other.direction.to_bits()
            && self.max_events == other.max_events
    }
}

impl EventSpec {
    pub fn new(func: EventFn) -> Self {
        Self {
            func,
            direction: 0.0,
            max_events: None,
        }
    }

    pub fn terminal(func: EventFn) -> Self {
        Self {
            func,
            direction: 0.0,
            max_events: Some(1),
        }
    }

    pub fn with_direction(mut self, direction: f64) -> Self {
        self.direction = direction;
        self
    }

    pub fn with_max_events(mut self, max_events: usize) -> Self {
        self.max_events = Some(max_events.max(1));
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverKind {
    Rk23,
    Rk45,
    Dop853,
    Radau,
    Bdf,
    Lsoda,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OdeSolution {
    pub knots: Vec<f64>,
    pub values: Vec<Vec<f64>>,
    pub alt_segment: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SolveIvpOptions<'a> {
    pub t_span: (f64, f64),
    pub y0: &'a [f64],
    pub method: SolverKind,
    pub t_eval: Option<&'a [f64]>,
    pub dense_output: bool,
    pub events: Option<Vec<EventSpec>>,
    pub rtol: f64,
    pub atol: ToleranceValue,
    pub first_step: Option<f64>,
    pub max_step: f64,
    pub mode: RuntimeMode,
}

impl Default for SolveIvpOptions<'_> {
    fn default() -> Self {
        Self {
            t_span: (0.0, 0.0),
            y0: &[],
            method: SolverKind::Rk45,
            t_eval: None,
            dense_output: false,
            events: None,
            rtol: 1e-3,
            atol: ToleranceValue::Scalar(1e-6),
            first_step: None,
            max_step: f64::INFINITY,
            mode: RuntimeMode::Strict,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SolveIvpResult {
    pub t: Vec<f64>,
    pub y: Vec<Vec<f64>>,
    pub sol: Option<OdeSolution>,
    pub t_events: Option<Vec<Vec<f64>>>,
    pub y_events: Option<Vec<Vec<Vec<f64>>>>,
    pub nfev: usize,
    pub njev: usize,
    pub nlu: usize,
    pub status: i32,
    pub message: String,
    pub success: bool,
}

const MSG_SUCCESS: &str = "The solver successfully reached the end of the integration interval.";
const MSG_FAILED: &str = "Integration step failed.";

fn event_active(old_val: f64, new_val: f64, direction: f64) -> bool {
    let up = old_val <= 0.0 && new_val >= 0.0;
    let down = old_val >= 0.0 && new_val <= 0.0;
    if direction > 0.0 {
        up
    } else if direction < 0.0 {
        down
    } else {
        up || down
    }
}

fn validate_t_eval_with_audit(
    t_eval: &[f64],
    t0: f64,
    tf: f64,
    audit: Option<&AuditScope<'_>>,
) -> Result<(), IntegrateValidationError> {
    let t_min = t0.min(tf);
    let t_max = t0.max(tf);
    if t_eval.iter().any(|&te| te < t_min || te > t_max) {
        fail_closed(audit, "t_eval_out_of_span", "rejected");
        return Err(IntegrateValidationError::TEvalOutOfSpan);
    }

    let is_sorted = if tf >= t0 {
        t_eval.windows(2).all(|window| window[1] > window[0])
    } else {
        t_eval.windows(2).all(|window| window[1] < window[0])
    };
    if !is_sorted {
        fail_closed(audit, "t_eval_not_sorted", "rejected");
        return Err(IntegrateValidationError::TEvalNotSorted);
    }

    Ok(())
}

fn validate_events_with_audit(
    events: Option<&[EventSpec]>,
    audit: Option<&AuditScope<'_>>,
) -> Result<(), IntegrateValidationError> {
    let Some(events) = events else {
        return Ok(());
    };

    for (index, event) in events.iter().enumerate() {
        if !event.direction.is_finite() {
            fail_closed(audit, "event_direction_must_be_finite", "rejected");
            return Err(IntegrateValidationError::NonFiniteEventDirection { index });
        }
        if event.max_events == Some(0) {
            fail_closed(audit, "event_max_events_must_be_positive", "rejected");
            return Err(IntegrateValidationError::EventMaxEventsMustBePositive { index });
        }
    }

    Ok(())
}

/// The audit fingerprint of one `solve_ivp` request (frankenscipy-3cu8u.1): a
/// [`Fingerprinter`] for `fsci_integrate::solve_ivp` over the [`SolveIvpOptions`] fields in
/// declaration order — `t_span` (two `f64`), `y0` (`f64s`), `method` (`Debug`), `t_eval`
/// (presence flag, then `f64s`), `dense_output`, `events` (presence flag, then the count and,
/// per event, `direction` then `max_events` as presence flag and value), `rtol`, `atol`
/// (variant name, then value or values), `first_step` (presence flag, then value), `max_step`
/// and `mode` (`Debug`). The right-hand side and the event functions are code, not data, and
/// are not part of it.
fn solve_ivp_fingerprint(options: &SolveIvpOptions<'_>) -> String {
    let mut fingerprinter = Fingerprinter::new("fsci_integrate::solve_ivp");
    fingerprinter
        .f64(options.t_span.0)
        .f64(options.t_span.1)
        .f64s(options.y0)
        .str(&format!("{:?}", options.method))
        .bool(options.t_eval.is_some());
    if let Some(t_eval) = options.t_eval {
        fingerprinter.f64s(t_eval);
    }
    fingerprinter
        .bool(options.dense_output)
        .bool(options.events.is_some());
    if let Some(events) = &options.events {
        fingerprinter.usize(events.len());
        for event in events {
            fingerprinter
                .f64(event.direction)
                .bool(event.max_events.is_some());
            if let Some(max_events) = event.max_events {
                fingerprinter.usize(max_events);
            }
        }
    }
    fingerprinter.f64(options.rtol);
    fingerprint_tolerance(&mut fingerprinter, &options.atol);
    fingerprint_optional_f64(&mut fingerprinter, options.first_step);
    fingerprinter
        .f64(options.max_step)
        .str(&format!("{:?}", options.mode));
    fingerprinter.finish()
}

fn validate_event_value(index: usize, value: f64) -> Result<f64, IntegrateValidationError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(IntegrateValidationError::NonFiniteEventValue { index })
    }
}

fn interpolate_state(
    y_old: &[f64],
    y_new: &[f64],
    f_old: &[f64],
    f_new: &[f64],
    t_old: f64,
    t_new: f64,
    t_eval: f64,
) -> Vec<f64> {
    let h = t_new - t_old;
    if h.abs() == 0.0 {
        return y_old.to_vec();
    }
    let x = (t_eval - t_old) / h;
    let x2 = x * x;
    let x3 = x2 * x;

    // Hermite basis functions: cubic interpolation matching values and derivatives
    let h00 = 2.0 * x3 - 3.0 * x2 + 1.0;
    let h10 = x3 - 2.0 * x2 + x;
    let h01 = -2.0 * x3 + 3.0 * x2;
    let h11 = x3 - x2;

    y_old
        .iter()
        .zip(y_new.iter())
        .zip(f_old.iter())
        .zip(f_new.iter())
        .map(|(((y0, y1), f0), f1)| h00 * y0 + h10 * h * f0 + h01 * y1 + h11 * h * f1)
        .collect()
}

/// Sample the solution at `t_sample` inside the step `(t_old, t_new)`.
///
/// Prefers the solver's own dense output — for RK45 SciPy's quartic Dormand-Prince
/// interpolant over all seven stage derivatives, for BDF SciPy's difference polynomial —
/// and falls back to the generic cubic Hermite for solvers that do not provide one. The
/// fallback is one order lower and drifts from SciPy's samples mid-step even
/// when the step endpoints agree, which is frankenscipy-3m5ip. A solver that neither has
/// dense output nor evaluates derivatives (BDF before its first accepted step, i.e.
/// finishing at once on `t0 == t_bound`) is interpolated linearly.
// frankenscipy-3qjah. Private helper; every argument is a distinct piece of the
// step being interpolated (solver, both endpoints' states and times, the sample
// point, the derivative closure). Bundling them into a struct would add an
// indirection inside the dense-output loop and change no logic, so the lint is
// silenced here rather than the signature churned. Scoped to this function on
// purpose -- a crate-level allow would hide future cases that ARE worth fixing.
#[allow(clippy::too_many_arguments)]
fn sample_state<S, F>(
    solver: &S,
    y_old: &[f64],
    y_new: &[f64],
    f_old: Option<&[f64]>,
    f_new: Option<&[f64]>,
    t_old: f64,
    t_new: f64,
    t_sample: f64,
) -> Vec<f64>
where
    S: IvpSolver<F> + ?Sized,
{
    solver
        .dense_output_at(t_sample)
        .unwrap_or_else(|| match (f_old, f_new) {
            (Some(f_old), Some(f_new)) => {
                interpolate_state(y_old, y_new, f_old, f_new, t_old, t_new, t_sample)
            }
            _ => {
                let h = t_new - t_old;
                let x = if h == 0.0 {
                    1.0
                } else {
                    (t_sample - t_old) / h
                };
                y_old
                    .iter()
                    .zip(y_new)
                    .map(|(a, b)| a + x * (b - a))
                    .collect()
            }
        })
}

fn is_new_time_point(points: &[f64], candidate: f64) -> bool {
    points
        .last()
        .is_none_or(|&last| (last - candidate).abs() > 1e-14)
}

fn eval_time_in_range(t_eval: f64, t_old: f64, t_new: f64, direction: f64) -> bool {
    let eps = 1e-12_f64.max(1e-12 * t_new.abs());
    if direction > 0.0 {
        t_eval > t_old + eps && t_eval <= t_new + eps
    } else {
        t_eval < t_old - eps && t_eval >= t_new - eps
    }
}

/// Root of one event inside the step `(t_old, t_new)`, evaluated on the step's
/// interpolant `interp` (the solver's dense output where it has one, as SciPy's
/// `solve_event_equation` uses `sol`).
fn solve_event_equation(
    event_index: usize,
    event_fn: EventFn,
    t_old: f64,
    t_new: f64,
    interp: &dyn Fn(f64) -> Vec<f64>,
) -> Result<f64, IntegrateValidationError> {
    let saw_non_finite = std::cell::Cell::new(false);
    let f = |t: f64| {
        let y = interp(t);
        let value = event_fn(t, &y);
        if value.is_finite() {
            value
        } else {
            saw_non_finite.set(true);
            0.0
        }
    };

    let options = RootOptions {
        xtol: 1e-12,
        maxiter: 100,
        ..Default::default()
    };

    let root = match brentq(f, (t_old, t_new), options) {
        Ok(res) => res.root,
        Err(_) => 0.5 * (t_old + t_new),
    };
    if saw_non_finite.get() {
        Err(IntegrateValidationError::NonFiniteEventValue { index: event_index })
    } else {
        Ok(root)
    }
}

/// Internal trait to unify RK and BDF solver loops.
trait IvpSolver<F> {
    fn step_with(&mut self, fun: &mut F) -> Result<StepOutcome, crate::solver::StepFailure>;
    fn t(&self) -> f64;
    fn y(&self) -> &[f64];
    /// The derivative at the current point, for solvers that evaluate it. BDF does not
    /// (neither does SciPy's) and returns `None`; it provides `dense_output_at` instead.
    fn f(&self) -> Option<&[f64]>;
    fn t_old(&self) -> Option<f64>;
    fn y_old(&self) -> Option<&[f64]>;
    fn f_old(&self) -> Option<&[f64]>;
    fn nfev(&self) -> usize;
    fn njev(&self) -> usize;
    fn nlu(&self) -> usize;
    fn ivp_state(&self) -> OdeSolverState;
    /// Build the last step's dense output where that costs evaluations (SciPy's
    /// `solver.dense_output()`); only DOP853 does. Called at most once per step.
    fn prepare_dense_output(&mut self, _fun: &mut F) -> Result<(), crate::solver::StepFailure> {
        Ok(())
    }
    /// Solver-specific dense output at `t`, or `None` to fall back to the
    /// generic cubic Hermite. Every RK method, BDF and Radau provide SciPy's.
    fn dense_output_at(&self, _t: f64) -> Option<Vec<f64>> {
        None
    }
}

impl<F> IvpSolver<F> for RkSolver
where
    F: FnMut(f64, &[f64]) -> Vec<f64>,
{
    fn step_with(&mut self, fun: &mut F) -> Result<StepOutcome, crate::solver::StepFailure> {
        self.step_with(fun)
    }
    fn t(&self) -> f64 {
        self.t()
    }
    fn y(&self) -> &[f64] {
        self.y()
    }
    fn f(&self) -> Option<&[f64]> {
        Some(self.f())
    }
    fn t_old(&self) -> Option<f64> {
        self.t_old()
    }
    fn y_old(&self) -> Option<&[f64]> {
        self.y_old()
    }
    fn f_old(&self) -> Option<&[f64]> {
        self.f_old()
    }
    fn nfev(&self) -> usize {
        self.nfev()
    }
    fn njev(&self) -> usize {
        0
    }
    fn nlu(&self) -> usize {
        0
    }
    fn ivp_state(&self) -> OdeSolverState {
        self.state()
    }
    fn prepare_dense_output(&mut self, fun: &mut F) -> Result<(), crate::solver::StepFailure> {
        self.prepare_dense_output(fun)
    }
    fn dense_output_at(&self, t: f64) -> Option<Vec<f64>> {
        self.dense_output_at(t)
    }
}

impl<F> IvpSolver<F> for BdfSolver
where
    F: FnMut(f64, &[f64]) -> Vec<f64>,
{
    fn step_with(&mut self, fun: &mut F) -> Result<StepOutcome, crate::solver::StepFailure> {
        self.step_with(fun)
    }
    fn t(&self) -> f64 {
        self.t()
    }
    fn y(&self) -> &[f64] {
        self.y()
    }
    fn f(&self) -> Option<&[f64]> {
        None
    }
    fn t_old(&self) -> Option<f64> {
        self.t_old()
    }
    fn y_old(&self) -> Option<&[f64]> {
        self.y_old()
    }
    fn f_old(&self) -> Option<&[f64]> {
        None
    }
    fn nfev(&self) -> usize {
        self.nfev()
    }
    fn njev(&self) -> usize {
        self.njev()
    }
    fn nlu(&self) -> usize {
        self.nlu()
    }
    fn ivp_state(&self) -> OdeSolverState {
        self.state()
    }
    fn dense_output_at(&self, t: f64) -> Option<Vec<f64>> {
        self.dense_output_at(t)
    }
}

impl<F> IvpSolver<F> for crate::radau::RadauSolver
where
    F: FnMut(f64, &[f64]) -> Vec<f64>,
{
    fn step_with(&mut self, fun: &mut F) -> Result<StepOutcome, crate::solver::StepFailure> {
        self.step_with(fun)
    }
    fn t(&self) -> f64 {
        self.t()
    }
    fn y(&self) -> &[f64] {
        self.y()
    }
    fn f(&self) -> Option<&[f64]> {
        Some(self.f())
    }
    fn t_old(&self) -> Option<f64> {
        self.t_old()
    }
    fn y_old(&self) -> Option<&[f64]> {
        self.y_old()
    }
    fn f_old(&self) -> Option<&[f64]> {
        None
    }
    fn nfev(&self) -> usize {
        self.nfev()
    }
    fn njev(&self) -> usize {
        self.njev()
    }
    fn nlu(&self) -> usize {
        self.nlu()
    }
    fn ivp_state(&self) -> OdeSolverState {
        self.state()
    }
    fn dense_output_at(&self, t: f64) -> Option<Vec<f64>> {
        self.dense_output_at(t)
    }
}

enum LsodaMode {
    Adams(RkSolver),
    Bdf(BdfSolver),
}

/// Configuration for constructing an [`LsodaSolver`].
pub struct LsodaSolverConfig<'a> {
    pub t0: f64,
    pub y0: &'a [f64],
    pub t_bound: f64,
    pub rtol: f64,
    pub atol: ToleranceValue,
    pub max_step: f64,
    pub first_step: Option<f64>,
    pub mode: RuntimeMode,
}

/// The stepper behind `solve_ivp(method="LSODA")` and `odeint`, as a step-by-step solver
/// object like [`RkSolver`] and [`BdfSolver`] (`scipy.integrate.LSODA`).
///
/// It is NOT ODEPACK's LSODA: it starts with RK45 and switches once, for good, to BDF when a
/// stiffness estimate from consecutive steps crosses a threshold or RK45 cannot take a step.
/// ODEPACK switches both ways between Adams and BDF families of varying order, so step
/// counts and step sequences differ from SciPy's (frankenscipy-1ksfv.9).
pub struct LsodaSolver {
    mode: LsodaMode,
    t_bound: f64,
    rtol: f64,
    atol: ToleranceValue,
    max_step: f64,
    first_step: Option<f64>,
    runtime_mode: RuntimeMode,
    pending_bdf_switch: bool,
    nfev_offset: usize,
}

impl LsodaSolver {
    /// Create the solver at `(t0, y0)`, starting in its RK45 phase.
    ///
    /// # Errors
    /// The same tolerance, step and state validation as [`RkSolver::new`].
    pub fn new<F>(
        fun: &mut F,
        config: LsodaSolverConfig<'_>,
    ) -> Result<Self, IntegrateValidationError>
    where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        let rk_config = RkSolverConfig {
            t0: config.t0,
            y0: config.y0,
            t_bound: config.t_bound,
            rtol: config.rtol,
            atol: config.atol.clone(),
            max_step: config.max_step,
            first_step: config.first_step,
            mode: config.mode,
            tableau: &RK45_TABLEAU,
        };
        let solver = RkSolver::new(fun, rk_config)?;
        Ok(Self {
            mode: LsodaMode::Adams(solver),
            t_bound: config.t_bound,
            rtol: config.rtol,
            atol: config.atol,
            max_step: config.max_step,
            first_step: config.first_step,
            runtime_mode: config.mode,
            pending_bdf_switch: false,
            nfev_offset: 0,
        })
    }

    fn from_options<F>(
        fun: &mut F,
        options: &SolveIvpOptions<'_>,
    ) -> Result<Self, IntegrateValidationError>
    where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        Self::new(
            fun,
            LsodaSolverConfig {
                t0: options.t_span.0,
                y0: options.y0,
                t_bound: options.t_span.1,
                rtol: options.rtol,
                atol: options.atol.clone(),
                max_step: options.max_step,
                first_step: options.first_step,
                mode: options.mode,
            },
        )
    }

    /// Whether the solver has switched to its BDF phase.
    #[must_use]
    pub fn is_stiff_phase(&self) -> bool {
        matches!(self.mode, LsodaMode::Bdf(_))
    }

    fn should_switch_to_bdf(rk: &RkSolver, t_bound: f64) -> bool {
        let Some(t_old) = rk.t_old() else {
            return false;
        };
        let Some(y_old) = rk.y_old() else {
            return false;
        };
        let Some(f_old) = rk.f_old() else {
            return false;
        };

        let step_size = (rk.t() - t_old).abs();
        if step_size == 0.0 {
            return false;
        }

        let remaining = (t_bound - rk.t()).abs();
        let mut stiffness_indicator = 0.0_f64;
        for (((&y_prev, &y_curr), &f_prev), &f_curr) in y_old
            .iter()
            .zip(rk.y().iter())
            .zip(f_old.iter())
            .zip(rk.f().iter())
        {
            let state_delta = (y_curr - y_prev).abs();
            let slope_delta = (f_curr - f_prev).abs();
            if state_delta > 1e-14 {
                stiffness_indicator =
                    stiffness_indicator.max(step_size * slope_delta / state_delta);
            }
        }

        stiffness_indicator > 1.5 || (step_size < remaining * 1e-4 && stiffness_indicator > 0.25)
    }

    fn switch_to_bdf<F>(
        &mut self,
        fun: &mut F,
        preferred_first_step: Option<f64>,
    ) -> Result<(), StepFailure>
    where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        let (t0, y0, consumed_nfev) = match &self.mode {
            LsodaMode::Adams(rk) => (rk.t(), rk.y().to_vec(), rk.nfev()),
            LsodaMode::Bdf(_) => return Ok(()),
        };

        let config = BdfSolverConfig {
            t0,
            y0: &y0,
            t_bound: self.t_bound,
            rtol: self.rtol,
            atol: self.atol.clone(),
            max_step: self.max_step,
            first_step: preferred_first_step.or(self.first_step),
            mode: self.runtime_mode,
            max_order: 5,
        };
        let solver = BdfSolver::new(fun, config).map_err(|_| StepFailure::SolverError)?;
        self.nfev_offset += consumed_nfev;
        self.mode = LsodaMode::Bdf(solver);
        self.pending_bdf_switch = false;
        Ok(())
    }
}

impl LsodaSolver {
    /// Advance one step (switching to BDF first if the previous step flagged stiffness).
    ///
    /// # Errors
    /// The step failures of the active RK45 or BDF phase.
    pub fn step_with<F>(&mut self, fun: &mut F) -> Result<StepOutcome, crate::solver::StepFailure>
    where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        if self.pending_bdf_switch {
            let preferred_first_step = match &self.mode {
                LsodaMode::Adams(rk) => rk.t_old().map(|t_old| (rk.t() - t_old).abs()),
                LsodaMode::Bdf(_) => None,
            };
            self.switch_to_bdf(fun, preferred_first_step)?;
        }

        match &mut self.mode {
            LsodaMode::Adams(rk) => match rk.step_with(fun) {
                Ok(outcome) => {
                    if outcome.state == OdeSolverState::Running
                        && Self::should_switch_to_bdf(rk, self.t_bound)
                    {
                        self.pending_bdf_switch = true;
                    }
                    Ok(outcome)
                }
                Err(crate::solver::StepFailure::StepSizeTooSmall)
                | Err(crate::solver::StepFailure::ConvergenceFailure) => {
                    let preferred_first_step = rk
                        .t_old()
                        .map(|t_old| (rk.t() - t_old).abs())
                        .or(self.first_step);
                    self.switch_to_bdf(fun, preferred_first_step)?;
                    if let LsodaMode::Bdf(bdf) = &mut self.mode {
                        bdf.step_with(fun)
                    } else {
                        Err(crate::solver::StepFailure::ConvergenceFailure)
                    }
                }
                Err(err) => Err(err),
            },
            LsodaMode::Bdf(bdf) => bdf.step_with(fun),
        }
    }

    #[must_use]
    pub fn t(&self) -> f64 {
        match &self.mode {
            LsodaMode::Adams(rk) => rk.t(),
            LsodaMode::Bdf(bdf) => bdf.t(),
        }
    }

    #[must_use]
    pub fn y(&self) -> &[f64] {
        match &self.mode {
            LsodaMode::Adams(rk) => rk.y(),
            LsodaMode::Bdf(bdf) => bdf.y(),
        }
    }

    fn f(&self) -> Option<&[f64]> {
        match &self.mode {
            LsodaMode::Adams(rk) => Some(rk.f()),
            LsodaMode::Bdf(_) => None,
        }
    }

    #[must_use]
    pub fn t_old(&self) -> Option<f64> {
        match &self.mode {
            LsodaMode::Adams(rk) => rk.t_old(),
            LsodaMode::Bdf(bdf) => bdf.t_old(),
        }
    }

    fn y_old(&self) -> Option<&[f64]> {
        match &self.mode {
            LsodaMode::Adams(rk) => rk.y_old(),
            LsodaMode::Bdf(bdf) => bdf.y_old(),
        }
    }

    fn f_old(&self) -> Option<&[f64]> {
        match &self.mode {
            LsodaMode::Adams(rk) => rk.f_old(),
            LsodaMode::Bdf(_) => None,
        }
    }

    /// Right-hand-side evaluations over both phases.
    #[must_use]
    pub fn nfev(&self) -> usize {
        self.nfev_offset
            + match &self.mode {
                LsodaMode::Adams(rk) => rk.nfev(),
                LsodaMode::Bdf(bdf) => bdf.nfev(),
            }
    }

    #[must_use]
    pub fn njev(&self) -> usize {
        match &self.mode {
            LsodaMode::Adams(_) => 0,
            LsodaMode::Bdf(bdf) => bdf.njev(),
        }
    }

    #[must_use]
    pub fn nlu(&self) -> usize {
        match &self.mode {
            LsodaMode::Adams(_) => 0,
            LsodaMode::Bdf(bdf) => bdf.nlu(),
        }
    }

    #[must_use]
    pub fn state(&self) -> OdeSolverState {
        match &self.mode {
            LsodaMode::Adams(rk) => rk.state(),
            LsodaMode::Bdf(bdf) => bdf.state(),
        }
    }

    /// The last step's interpolant at `t` (SciPy's `solver.dense_output()(t)`): RK45's in the
    /// first phase, BDF's after the switch.
    #[must_use]
    pub fn dense_output_at(&self, t: f64) -> Option<Vec<f64>> {
        match &self.mode {
            LsodaMode::Adams(rk) => rk.dense_output_at(t),
            LsodaMode::Bdf(bdf) => bdf.dense_output_at(t),
        }
    }
}

impl<F> IvpSolver<F> for LsodaSolver
where
    F: FnMut(f64, &[f64]) -> Vec<f64>,
{
    fn step_with(&mut self, fun: &mut F) -> Result<StepOutcome, crate::solver::StepFailure> {
        self.step_with(fun)
    }
    fn t(&self) -> f64 {
        self.t()
    }
    fn y(&self) -> &[f64] {
        self.y()
    }
    fn f(&self) -> Option<&[f64]> {
        self.f()
    }
    fn t_old(&self) -> Option<f64> {
        self.t_old()
    }
    fn y_old(&self) -> Option<&[f64]> {
        self.y_old()
    }
    fn f_old(&self) -> Option<&[f64]> {
        self.f_old()
    }
    fn nfev(&self) -> usize {
        self.nfev()
    }
    fn njev(&self) -> usize {
        self.njev()
    }
    fn nlu(&self) -> usize {
        self.nlu()
    }
    fn ivp_state(&self) -> OdeSolverState {
        self.state()
    }
    fn prepare_dense_output(&mut self, fun: &mut F) -> Result<(), crate::solver::StepFailure> {
        match &mut self.mode {
            LsodaMode::Adams(rk) => rk.prepare_dense_output(fun),
            LsodaMode::Bdf(_) => Ok(()),
        }
    }
    fn dense_output_at(&self, t: f64) -> Option<Vec<f64>> {
        self.dense_output_at(t)
    }
}

fn solve_ivp_core<F, S>(
    fun: &mut F,
    mut solver: S,
    options: &SolveIvpOptions<'_>,
) -> Result<SolveIvpResult, IntegrateValidationError>
where
    F: FnMut(f64, &[f64]) -> Vec<f64>,
    S: IvpSolver<F>,
{
    let (t0, tf) = options.t_span;
    let direction = if tf >= t0 { 1.0 } else { -1.0 };

    let mut ts = Vec::new();
    let mut ys: Vec<Vec<f64>> = Vec::new();
    // Dense output uses solver-chosen knots even when t_eval is supplied.
    let mut dense_knots = vec![t0];
    let mut dense_values = vec![options.y0.to_vec()];
    let mut next_t_eval_index = 0usize;

    let mut t_events: Option<Vec<Vec<f64>>> = options
        .events
        .as_ref()
        .map(|evs| vec![Vec::new(); evs.len()]);
    let mut y_events: Option<Vec<Vec<Vec<f64>>>> = options
        .events
        .as_ref()
        .map(|evs| vec![Vec::new(); evs.len()]);
    let mut event_counts: Option<Vec<usize>> =
        options.events.as_ref().map(|evs| vec![0; evs.len()]);
    let mut event_vals = if let Some(evs) = options.events.as_ref() {
        let mut vals = Vec::with_capacity(evs.len());
        for (i, ev) in evs.iter().enumerate() {
            vals.push(validate_event_value(i, (ev.func)(t0, options.y0))?);
        }
        Some(vals)
    } else {
        None
    };

    // SciPy samples a t_eval point at t0 from the FIRST step's dense output (its value is y0
    // exactly), so that step builds dense output even when no other sample falls in it.
    let mut t0_sample_pending = false;
    if let Some(t_eval) = options.t_eval {
        if matches!(t_eval.first(), Some(&first) if (first - t0).abs() < 1e-14) {
            ts.push(t0);
            ys.push(options.y0.to_vec());
            next_t_eval_index = 1;
            t0_sample_pending = true;
        }
    } else {
        ts.push(t0);
        ys.push(options.y0.to_vec());
    }

    let mut status: i32 = -1;

    while solver.ivp_state() == OdeSolverState::Running {
        match solver.step_with(fun) {
            Ok(outcome) => {
                let t = solver.t();
                let y = solver.y().to_vec();
                let f = solver.f().map(<[f64]>::to_vec);
                let t_old = solver.t_old().unwrap_or(t0);
                let y_old = solver.y_old().unwrap_or(options.y0).to_vec();
                let f_old = solver.f_old().map(<[f64]>::to_vec).or_else(|| f.clone());

                // Each event at the new point, once per event per step.
                let new_event_vals = match options.events.as_ref() {
                    Some(evs) => {
                        let mut vals = Vec::with_capacity(evs.len());
                        for (i, ev) in evs.iter().enumerate() {
                            vals.push(validate_event_value(i, (ev.func)(t, &y))?);
                        }
                        Some(vals)
                    }
                    None => None,
                };
                // SciPy builds a step's dense output only when something reads it:
                // dense_output=True, an active event, or a t_eval point inside the step. For
                // DOP853 building it costs three counted evaluations, so prepare it on exactly
                // those steps.
                let any_event_active = match (
                    options.events.as_ref(),
                    event_vals.as_ref(),
                    new_event_vals.as_ref(),
                ) {
                    (Some(evs), Some(old), Some(new)) => evs
                        .iter()
                        .enumerate()
                        .any(|(i, ev)| event_active(old[i], new[i], ev.direction)),
                    _ => false,
                };
                let t_eval_in_step = std::mem::take(&mut t0_sample_pending)
                    || options
                        .t_eval
                        .and_then(|t_eval| t_eval.get(next_t_eval_index))
                        .is_some_and(|&te| eval_time_in_range(te, t_old, t, direction));
                if (options.dense_output || any_event_active || t_eval_in_step)
                    && solver.prepare_dense_output(fun).is_err()
                {
                    break;
                }

                // The step's interpolant, for t_eval samples AND event roots.
                let interp = |te: f64| {
                    sample_state(
                        &solver,
                        &y_old,
                        &y,
                        f_old.as_deref(),
                        f.as_deref(),
                        t_old,
                        t,
                        te,
                    )
                };

                let mut terminal_event: Option<(f64, Vec<f64>)> = None;

                if let (Some(evs), Some(old_vals), Some(counts), Some(new_vals)) = (
                    options.events.as_ref(),
                    event_vals.as_mut(),
                    event_counts.as_mut(),
                    new_event_vals.as_ref(),
                ) {
                    for (i, ev_spec) in evs.iter().enumerate() {
                        let val = new_vals[i];
                        if event_active(old_vals[i], val, ev_spec.direction) {
                            let t_ev = solve_event_equation(i, ev_spec.func, t_old, t, &interp)?;
                            let y_ev = interp(t_ev);

                            if let Some(tes) = t_events.as_mut() {
                                tes[i].push(t_ev);
                            }
                            if let Some(yes) = y_events.as_mut() {
                                yes[i].push(y_ev.clone());
                            }

                            counts[i] += 1;
                            let terminal_hit = ev_spec
                                .max_events
                                .is_some_and(|max_events| counts[i] >= max_events);
                            if terminal_hit {
                                let select = match terminal_event.as_ref() {
                                    None => true,
                                    Some((current_t_ev, _)) => {
                                        (t_ev - t_old).abs() < (current_t_ev - t_old).abs()
                                    }
                                };
                                if select {
                                    terminal_event = Some((t_ev, y_ev.clone()));
                                }
                            }
                        }
                        old_vals[i] = val;
                    }
                }

                if let Some((t_ev, y_ev)) = terminal_event {
                    if is_new_time_point(&dense_knots, t_ev) {
                        dense_knots.push(t_ev);
                        dense_values.push(y_ev.clone());
                    }

                    if let Some(t_eval) = options.t_eval {
                        while let Some(&te) = t_eval.get(next_t_eval_index) {
                            if !eval_time_in_range(te, t_old, t_ev, direction) {
                                break;
                            }
                            ts.push(te);
                            ys.push(interp(te));
                            next_t_eval_index += 1;
                        }
                    }
                    // Always add the event time itself when terminal event triggers
                    if ts
                        .last()
                        .is_none_or(|&last_t| (last_t - t_ev).abs() > 1e-14)
                    {
                        ts.push(t_ev);
                        ys.push(y_ev);
                    }
                    status = 1;
                    break;
                }

                if is_new_time_point(&dense_knots, t) {
                    dense_knots.push(t);
                    dense_values.push(y.clone());
                }

                if let Some(t_eval) = options.t_eval {
                    while let Some(&te) = t_eval.get(next_t_eval_index) {
                        if !eval_time_in_range(te, t_old, t, direction) {
                            break;
                        }

                        ts.push(te);
                        ys.push(interp(te));
                        next_t_eval_index += 1;
                    }
                } else if is_new_time_point(&ts, t) {
                    ts.push(t);
                    ys.push(y);
                }

                if outcome.state == OdeSolverState::Finished {
                    status = 0;
                }
            }
            Err(_) => {
                status = -1;
                break;
            }
        }
    }

    let message = match status {
        0 => MSG_SUCCESS.to_owned(),
        1 => "A termination event occurred.".to_owned(),
        _ => MSG_FAILED.to_owned(),
    };

    let sol = options.dense_output.then_some(OdeSolution {
        knots: dense_knots,
        values: dense_values,
        alt_segment: matches!(options.method, SolverKind::Bdf | SolverKind::Lsoda),
    });

    Ok(SolveIvpResult {
        t: ts,
        y: ys,
        sol,
        t_events,
        y_events,
        nfev: solver.nfev(),
        njev: solver.njev(),
        nlu: solver.nlu(),
        status,
        message,
        success: status >= 0,
    })
}

pub fn solve_ivp<F>(
    fun: &mut F,
    options: &SolveIvpOptions<'_>,
) -> Result<SolveIvpResult, IntegrateValidationError>
where
    F: FnMut(f64, &[f64]) -> Vec<f64>,
{
    solve_ivp_impl(fun, options, None)
}

pub fn solve_ivp_with_audit<F>(
    fun: &mut F,
    options: &SolveIvpOptions<'_>,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<SolveIvpResult, IntegrateValidationError>
where
    F: FnMut(f64, &[f64]) -> Vec<f64>,
{
    let fingerprint_of = || solve_ivp_fingerprint(options);
    let audit = AuditScope::new(audit_ledger, &fingerprint_of);
    // frankenscipy-3cu8u.2: every error is one fail-closed event, a failed integration
    // included, not only the validators' rejections.
    audit.finish(
        solve_ivp_impl(fun, options, Some(&audit)),
        IntegrateValidationError::reason_code,
    )
}

impl From<OdeSolverAction> for SolverKind {
    fn from(action: OdeSolverAction) -> Self {
        match action {
            OdeSolverAction::RK45 => Self::Rk45,
            OdeSolverAction::RK23 => Self::Rk23,
            OdeSolverAction::DOP853 => Self::Dop853,
            OdeSolverAction::Radau => Self::Radau,
            OdeSolverAction::BDF => Self::Bdf,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OdePortfolioResult {
    pub chosen_action: OdeSolverAction,
    pub posterior: [f64; 4],
    pub expected_losses: [f64; 5],
    pub result: SolveIvpResult,
    pub fallback_active: bool,
}

/// Solve an initial value problem using CASP Bayesian expected-loss portfolio selection.
///
/// Selects RK45 for non-stiff dynamics, Radau IIA for algebraic/DAE constraints,
/// and BDF for stiff systems, with online stiffness detection and conformal drift fallback.
pub fn solve_ivp_with_casp_portfolio<F>(
    fun: &mut F,
    options: &SolveIvpOptions<'_>,
    portfolio: &mut OdeSolverPortfolio,
    stiffness_ratio_estimate: f64,
    is_algebraic_dae: bool,
) -> Result<OdePortfolioResult, IntegrateValidationError>
where
    F: FnMut(f64, &[f64]) -> Vec<f64>,
{
    let (action, posterior, expected_losses, chosen_loss) =
        portfolio.select_action(stiffness_ratio_estimate, is_algebraic_dae);

    let method: SolverKind = action.into();
    let mut resolved_opts = options.clone();
    resolved_opts.method = method;

    let res = solve_ivp(fun, &resolved_opts)?;
    let fallback_active = portfolio.calibrator().should_fallback();

    portfolio.observe_step(res.success);
    portfolio.record_evidence(OdeSolverEvidenceEntry {
        component: "fsci-integrate",
        system_dim: options.y0.len(),
        stiffness_ratio_estimate,
        chosen_action: action,
        posterior: posterior.to_vec(),
        expected_losses: expected_losses.to_vec(),
        chosen_expected_loss: chosen_loss,
        fallback_active,
        step_rejections: if res.success { Some(0) } else { Some(1) },
    });

    Ok(OdePortfolioResult {
        chosen_action: action,
        posterior,
        expected_losses,
        result: res,
        fallback_active,
    })
}

/// Solve an initial value problem using CASP condition-aware portfolio selection.
///
/// Selects RK45 for non-stiff dynamics, Radau IIA for algebraic/DAE constraints,
/// and BDF for stiff systems via Bayesian expected-loss minimization, with
/// online stiffness detection and conformal drift fallback.
pub fn solve_ivp_with_casp<F>(
    fun: &mut F,
    options: &SolveIvpOptions<'_>,
    portfolio: &mut OdeSolverPortfolio,
) -> Result<OdePortfolioResult, IntegrateValidationError>
where
    F: FnMut(f64, &[f64]) -> Vec<f64>,
{
    solve_ivp_with_casp_portfolio(fun, options, portfolio, 1.0, false)
}

/// Batched ODE integration: integrate the SAME dynamics `fun` from MANY initial conditions
/// (`y0_rows`), one [`SolveIvpResult`] per row. This is the vmap-over-solver primitive SciPy
/// lacks — there you loop `solve_ivp` in Python, calling the Python RHS thousands of times per
/// solve, N solves SERIALLY; here the N independent integrations are fanned across cores and the
/// RHS is an inlined Rust closure (callback lever × N-way parallel). Result `i` is identical to
/// `solve_ivp(&mut |t, y| fun(t, y), &opts)` with `opts.y0 = &y0_rows[i]`.
///
/// `template` supplies the shared `t_span` / `method` / `t_eval` / tolerances; its `y0` field is
/// ignored (overridden per row). Common for parameter sweeps and ensemble simulation.
pub fn solve_ivp_many<'a, F>(
    fun: F,
    y0_rows: &'a [Vec<f64>],
    template: &SolveIvpOptions<'a>,
) -> Vec<Result<SolveIvpResult, IntegrateValidationError>>
where
    F: Fn(f64, &[f64]) -> Vec<f64> + Sync,
{
    let nrows = y0_rows.len();
    if nrows == 0 {
        return Vec::new();
    }
    let fun_ref = &fun;
    let solve_one = move |y0: &'a [f64]| -> Result<SolveIvpResult, IntegrateValidationError> {
        let mut opts = template.clone();
        opts.y0 = y0;
        let mut local = |t: f64, y: &[f64]| fun_ref(t, y);
        solve_ivp(&mut local, &opts)
    };

    // Each integration is an independent, expensive (~ms) adaptive solve → fan whole rows
    // across cores, capped by the row count; a tiny ensemble stays serial.
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let nthreads = cores.min(nrows);
    if nthreads <= 1 || nrows < 4 {
        return y0_rows.iter().map(|y0| solve_one(y0)).collect();
    }

    let chunk = nrows.div_ceil(nthreads);
    let solve_one = &solve_one;
    let chunk_results: Vec<Vec<Result<SolveIvpResult, IntegrateValidationError>>> =
        std::thread::scope(|scope| {
            (0..nthreads)
                .filter_map(|t| {
                    let lo = t * chunk;
                    if lo >= nrows {
                        return None;
                    }
                    let hi = (lo + chunk).min(nrows);
                    Some(scope.spawn(move || {
                        (lo..hi).map(|i| solve_one(&y0_rows[i])).collect::<Vec<_>>()
                    }))
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().expect("solve_ivp_many worker panicked"))
                .collect()
        });

    let mut out = Vec::with_capacity(nrows);
    for cr in chunk_results {
        out.extend(cr);
    }
    out
}

/// `solve_ivp`; every audit event, including those of the validators it runs, is recorded
/// under `audit`, whose fingerprint is [`solve_ivp_fingerprint`] of `options`.
fn solve_ivp_impl<F>(
    fun: &mut F,
    options: &SolveIvpOptions<'_>,
    audit: Option<&AuditScope<'_>>,
) -> Result<SolveIvpResult, IntegrateValidationError>
where
    F: FnMut(f64, &[f64]) -> Vec<f64>,
{
    let (t0, tf) = options.t_span;
    let n = options.y0.len();
    if n == 0 {
        fail_closed(audit, "empty_y0", "rejected");
        return Err(IntegrateValidationError::EmptyY0);
    }
    if !t0.is_finite() || !tf.is_finite() {
        fail_closed(audit, "non_finite_span", "rejected");
        return Err(IntegrateValidationError::NonFiniteSpan);
    }
    if options.y0.iter().any(|v| !v.is_finite()) {
        fail_closed(audit, "non_finite_y0", "rejected");
        return Err(IntegrateValidationError::NonFiniteY0);
    }

    let max_step = validate_max_step_scoped(options.max_step, audit)?;
    let first_step = options
        .first_step
        .map(|value| validate_first_step_scoped(value, t0, tf, audit))
        .transpose()?;
    let validated_tol = validate_tol_scoped(
        ToleranceValue::Scalar(options.rtol),
        options.atol.clone(),
        n,
        options.mode,
        audit,
    )?;
    let rtol = validated_tol
        .rtol
        .into_scalar()
        .expect("solve_ivp always carries scalar rtol");
    let atol = validated_tol.atol;
    let mut resolved_options = options.clone();
    resolved_options.rtol = rtol;
    resolved_options.atol = atol.clone();
    resolved_options.first_step = first_step;
    resolved_options.max_step = max_step;
    validate_events_with_audit(resolved_options.events.as_deref(), audit)?;

    // Validate the initial event values at (t0, y0) before constructing the solver, which evaluates
    // the RHS for initial step-size selection. This rejects a non-finite initial event value with
    // zero RHS calls (solve_ivp_rejects_non_finite_initial_event_value), matching the contract that
    // initial event validation runs before any stepping.
    if let Some(evs) = resolved_options.events.as_ref() {
        for (i, ev) in evs.iter().enumerate() {
            validate_event_value(i, (ev.func)(t0, options.y0))?;
        }
    }

    if let Some(t_eval) = resolved_options.t_eval {
        validate_t_eval_with_audit(t_eval, t0, tf, audit)?;
    }

    match resolved_options.method {
        SolverKind::Rk45 | SolverKind::Rk23 | SolverKind::Dop853 => {
            let tableau = match resolved_options.method {
                SolverKind::Rk45 => &RK45_TABLEAU,
                SolverKind::Rk23 => &RK23_TABLEAU,
                _ => &crate::rk::DOP853_TABLEAU,
            };
            let config = RkSolverConfig {
                t0,
                y0: resolved_options.y0,
                t_bound: tf,
                rtol,
                atol: atol.clone(),
                max_step,
                first_step,
                mode: resolved_options.mode,
                tableau,
            };
            let solver = RkSolver::new(fun, config)?;
            solve_ivp_core(fun, solver, &resolved_options)
        }
        SolverKind::Radau => {
            // Genuine Radau IIA (3-stage, order 5) — see radau.rs (frankenscipy-3y5p9).
            let config = crate::radau::RadauSolverConfig {
                t0,
                y0: resolved_options.y0,
                t_bound: tf,
                rtol,
                atol,
                max_step,
                first_step,
                mode: resolved_options.mode,
            };
            let solver = crate::radau::RadauSolver::new(fun, config)?;
            solve_ivp_core(fun, solver, &resolved_options)
        }
        SolverKind::Bdf => {
            let config = BdfSolverConfig {
                t0,
                y0: resolved_options.y0,
                t_bound: tf,
                rtol,
                atol,
                max_step,
                first_step,
                mode: resolved_options.mode,
                max_order: 5,
            };
            let solver = BdfSolver::new(fun, config)?;
            solve_ivp_core(fun, solver, &resolved_options)
        }
        SolverKind::Lsoda => {
            let solver = LsodaSolver::from_options(fun, &resolved_options)?;
            solve_ivp_core(fun, solver, &resolved_options)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsci_runtime::AuditAction;

    /// A trial step whose right-hand side goes NaN is rejected and shrunk, not a failure.
    /// SciPy 1.17.1 `solve_ivp(f, (0, 1.9), y0, method=..., first_step=0.9)`, default tolerances:
    /// - y' = -sqrt(y), y0 = 1, RK45: 14 NaN evaluations, success, nfev = 97,
    ///   y(1.9) = 0.0025021952904742124.
    /// - y' = sqrt(1 - y), y0 = 0, RK45: 14 NaN evaluations, success, nfev = 85,
    ///   y(1.9) = 0.997460458540015.
    /// - must not change (no NaN evaluated): y' = -sqrt(y) with RK23 is nfev = 52,
    ///   y = 0.002229771521795849; with DOP853 it is nfev = 25, y = 0.0024524366176797985.
    #[test]
    fn solve_ivp_rejects_a_nan_trial_step_like_scipy() {
        let cases: [(&str, fn(f64) -> f64, f64, SolverKind, usize, f64); 4] = [
            (
                "sqrt_decay RK45",
                |y| -y.sqrt(),
                1.0,
                SolverKind::Rk45,
                97,
                0.0025021952904742124,
            ),
            (
                "sqrt_1my RK45",
                |y| (1.0 - y).sqrt(),
                0.0,
                SolverKind::Rk45,
                85,
                0.997460458540015,
            ),
            (
                "sqrt_decay RK23",
                |y| -y.sqrt(),
                1.0,
                SolverKind::Rk23,
                52,
                0.002229771521795849,
            ),
            (
                "sqrt_decay DOP853",
                |y| -y.sqrt(),
                1.0,
                SolverKind::Dop853,
                25,
                0.0024524366176797985,
            ),
        ];
        for (label, rhs, y0, method, nfev, y_end) in cases {
            let y0 = [y0];
            let result = solve_ivp(
                &mut |_t, y| vec![rhs(y[0])],
                &SolveIvpOptions {
                    t_span: (0.0, 1.9),
                    y0: &y0,
                    method,
                    first_step: Some(0.9),
                    ..SolveIvpOptions::default()
                },
            )
            .expect("scipy succeeds here");
            let yf = result.y.last().expect("a final state")[0];
            assert!(
                result.success && result.status == 0,
                "{label}: {} {}",
                result.status,
                result.message
            );
            assert_eq!(result.nfev, nfev, "{label}: nfev");
            assert!(
                (yf - y_end).abs() < 1e-12,
                "{label}: y(1.9) = {yf}, scipy {y_end}"
            );
        }
    }

    #[test]
    fn solve_ivp_harmonic_oscillator_system() {
        // 2-equation system y' = [y1, -y0], y(0)=[1,0] -> [cos t, -sin t].
        // Exercises the vector ODE path; scipy.integrate.solve_ivp converges to
        // the same analytic solution at this tolerance.
        let result = solve_ivp(
            &mut |_t, y| vec![y[1], -y[0]],
            &SolveIvpOptions {
                t_span: (0.0, 2.0),
                y0: &[1.0, 0.0],
                method: SolverKind::Rk45,
                rtol: 1e-9,
                atol: ToleranceValue::Scalar(1e-12),
                ..SolveIvpOptions::default()
            },
        )
        .expect("solve_ivp should succeed");
        assert!(result.success, "integration should succeed");
        let yf = result.y.last().unwrap();
        assert!(
            (yf[0] - 2.0_f64.cos()).abs() < 1e-6,
            "y0: {} vs {}",
            yf[0],
            2.0_f64.cos()
        );
        assert!(
            (yf[1] + 2.0_f64.sin()).abs() < 1e-6,
            "y1: {} vs {}",
            yf[1],
            -2.0_f64.sin()
        );
    }

    #[test]
    fn solve_ivp_exponential_decay() {
        let result = solve_ivp(
            &mut |_t, y| vec![-0.5 * y[0]],
            &SolveIvpOptions {
                t_span: (0.0, 10.0),
                y0: &[2.0],
                method: SolverKind::Rk45,
                rtol: 1e-6,
                atol: ToleranceValue::Scalar(1e-8),
                ..SolveIvpOptions::default()
            },
        )
        .expect("solve_ivp should succeed");

        assert!(result.success, "integration should succeed");
        assert_eq!(result.status, 0);

        let y_final = result.y.last().unwrap()[0];
        let expected = 2.0 * (-5.0_f64).exp();
        assert!(
            (y_final - expected).abs() < 1e-4,
            "y(10) = {y_final}, expected ≈ {expected}"
        );
    }

    #[test]
    fn solve_ivp_many_byte_identical_to_per_member() {
        // Ensemble of Lotka-Volterra initial conditions, shared dynamics. The batched
        // result must equal looping solve_ivp per row, bit-for-bit (each solve is an
        // independent deterministic adaptive integration).
        let (a, b, c, d) = (1.5_f64, 1.0, 3.0, 1.0);
        let rhs =
            |_t: f64, y: &[f64]| vec![a * y[0] - b * y[0] * y[1], -c * y[1] + d * y[0] * y[1]];
        let t_eval: Vec<f64> = (0..60).map(|i| i as f64 * 10.0 / 59.0).collect();
        let mut s = 99u64;
        let mut rng = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            1.0 + 4.0 * ((s >> 11) as f64 / (1u64 << 53) as f64)
        };
        let nrows = 12usize; // crosses the serial->parallel gate
        let y0_rows: Vec<Vec<f64>> = (0..nrows).map(|_| vec![rng(), rng()]).collect();
        let template = SolveIvpOptions {
            t_span: (0.0, 10.0),
            y0: &[0.0, 0.0], // overridden per row
            method: SolverKind::Rk45,
            t_eval: Some(&t_eval),
            rtol: 1e-8,
            atol: ToleranceValue::Scalar(1e-10),
            ..SolveIvpOptions::default()
        };

        let batched = solve_ivp_many(rhs, &y0_rows, &template);
        assert_eq!(batched.len(), nrows);
        for (i, y0) in y0_rows.iter().enumerate() {
            let mut single_opts = template.clone();
            single_opts.y0 = y0;
            let single = solve_ivp(&mut |t, y| rhs(t, y), &single_opts).expect("single");
            let many = batched[i].as_ref().expect("batched member");
            assert_eq!(many.t.len(), single.t.len());
            for (a, b) in many.t.iter().zip(single.t.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "t mismatch row {i}");
            }
            assert_eq!(many.y.len(), single.y.len());
            for (ry, sy) in many.y.iter().zip(single.y.iter()) {
                for (a, b) in ry.iter().zip(sy.iter()) {
                    assert_eq!(a.to_bits(), b.to_bits(), "y mismatch row {i}");
                }
            }
        }
        assert!(solve_ivp_many(rhs, &[], &template).is_empty());
    }

    #[test]
    fn solve_ivp_rejects_empty_initial_state() {
        let err = solve_ivp(
            &mut |_t, _y| Vec::new(),
            &SolveIvpOptions {
                t_span: (0.0, 1.0),
                y0: &[],
                method: SolverKind::Rk45,
                ..SolveIvpOptions::default()
            },
        )
        .expect_err("empty initial state should be rejected");
        assert_eq!(err, IntegrateValidationError::EmptyY0);
    }

    #[test]
    fn solve_ivp_rejects_wrong_size_initial_rhs_output() {
        for method in [SolverKind::Rk45, SolverKind::Bdf, SolverKind::Radau] {
            let err = solve_ivp(
                &mut |_t, _y| Vec::new(),
                &SolveIvpOptions {
                    t_span: (0.0, 1.0),
                    y0: &[1.0],
                    method,
                    first_step: Some(0.1),
                    ..SolveIvpOptions::default()
                },
            )
            .expect_err("wrong-size RHS output should be rejected");
            assert_eq!(
                err,
                IntegrateValidationError::RhsWrongShape {
                    expected: 1,
                    actual: 0
                },
                "method {method:?}"
            );
        }
    }

    #[test]
    fn solve_ivp_with_audit_records_fail_closed_on_empty_initial_state() {
        let audit_ledger = crate::sync_audit_ledger();
        let err = solve_ivp_with_audit(
            &mut |_t, _y| Vec::new(),
            &SolveIvpOptions {
                t_span: (0.0, 1.0),
                y0: &[],
                method: SolverKind::Rk45,
                ..SolveIvpOptions::default()
            },
            &audit_ledger,
        )
        .expect_err("empty initial state should be rejected");
        assert_eq!(err, IntegrateValidationError::EmptyY0);

        let ledger = audit_ledger.lock().expect("lock");
        assert_eq!(ledger.len(), 1);
        assert!(matches!(
            ledger.entries()[0].action,
            AuditAction::FailClosed { ref reason } if reason == "empty_y0"
        ));
    }

    /// The documented `solve_ivp` audit fingerprint, rebuilt by hand for a request that sets
    /// `t_span`, `y0`, `t_eval` and `mode` and leaves every other option at its default.
    fn default_request_fingerprint(
        t_span: (f64, f64),
        y0: &[f64],
        t_eval: Option<&[f64]>,
        mode: &str,
    ) -> String {
        let mut fingerprinter = Fingerprinter::new("fsci_integrate::solve_ivp");
        fingerprinter
            .f64(t_span.0)
            .f64(t_span.1)
            .f64s(y0)
            .str("Rk45")
            .bool(t_eval.is_some());
        if let Some(t_eval) = t_eval {
            fingerprinter.f64s(t_eval);
        }
        fingerprinter
            .bool(false) // dense_output
            .bool(false) // events: None
            .f64(1e-3) // rtol
            .str("Scalar") // atol
            .f64(1e-6)
            .bool(false) // first_step: None
            .f64(f64::INFINITY) // max_step
            .str(mode);
        fingerprinter.finish()
    }

    /// A t_eval rejection is fingerprinted by the whole solve_ivp request (frankenscipy-3cu8u.1).
    /// It used to be a digest of `validate_t_eval`'s own `t_eval`, `t0` and `tf` alone.
    #[test]
    fn solve_ivp_with_audit_preserves_t_eval_fingerprint() {
        let t0 = 0.0;
        let tf = 1.0;
        let t_eval = [0.0, 0.5, 2.0];
        let expected_fingerprint =
            default_request_fingerprint((t0, tf), &[1.0], Some(&t_eval), "Strict");
        let audit_ledger = crate::sync_audit_ledger();

        let err = solve_ivp_with_audit(
            &mut |_t, y| vec![-y[0]],
            &SolveIvpOptions {
                t_span: (t0, tf),
                y0: &[1.0],
                t_eval: Some(&t_eval),
                ..SolveIvpOptions::default()
            },
            &audit_ledger,
        )
        .expect_err("out-of-span t_eval should fail closed");
        assert_eq!(err, IntegrateValidationError::TEvalOutOfSpan);

        let ledger = audit_ledger.lock().expect("lock");
        assert_eq!(ledger.len(), 1);
        let entry = &ledger.entries()[0];
        assert_eq!(entry.input_fingerprint, expected_fingerprint);
        assert!(matches!(
            entry.action,
            AuditAction::FailClosed { ref reason } if reason == "t_eval_out_of_span"
        ));
    }

    /// A non-finite y0 is fingerprinted by the whole request, `y0` bit for bit (`-0.0` and the
    /// NaN payload included) (frankenscipy-3cu8u.1). It used to be a digest of a `Debug`
    /// string of `y0` and `mode`, in which every NaN payload read `NaN`.
    #[test]
    fn solve_ivp_with_audit_preserves_non_finite_y0_fingerprint() {
        let y0 = [1.0, -0.0, f64::from_bits(0x7ff8_0000_0000_0042)];
        let mode = RuntimeMode::Strict;
        let expected_fingerprint = default_request_fingerprint((0.0, 1.0), &y0, None, "Strict");
        let audit_ledger = crate::sync_audit_ledger();
        let mut rhs_calls = 0usize;

        let err = solve_ivp_with_audit(
            &mut |_t, _y| {
                rhs_calls += 1;
                Vec::new()
            },
            &SolveIvpOptions {
                t_span: (0.0, 1.0),
                y0: &y0,
                mode,
                ..SolveIvpOptions::default()
            },
            &audit_ledger,
        )
        .expect_err("non-finite y0 should fail closed");
        assert_eq!(err, IntegrateValidationError::NonFiniteY0);
        assert_eq!(rhs_calls, 0, "y0 validation must precede RHS evaluation");

        let ledger = audit_ledger.lock().expect("lock");
        assert_eq!(ledger.len(), 1);
        let entry = &ledger.entries()[0];
        assert_eq!(entry.input_fingerprint, expected_fingerprint);
        assert!(matches!(
            entry.action,
            AuditAction::FailClosed { ref reason } if reason == "non_finite_y0"
        ));
    }

    #[test]
    fn solve_ivp_with_audit_records_hardened_tolerance_clamp() {
        let audit_ledger = crate::sync_audit_ledger();
        let result = solve_ivp_with_audit(
            &mut |_t, y| vec![-0.5 * y[0]],
            &SolveIvpOptions {
                t_span: (0.0, 1.0),
                y0: &[2.0],
                method: SolverKind::Rk45,
                rtol: 0.0,
                atol: ToleranceValue::Scalar(1e-8),
                first_step: Some(1e-3),
                max_step: 0.1,
                mode: RuntimeMode::Hardened,
                ..SolveIvpOptions::default()
            },
            &audit_ledger,
        )
        .expect("hardened solve should succeed with clamped tolerance");
        assert!(result.success);

        let ledger = audit_ledger.lock().expect("lock");
        assert!(
            ledger.entries().iter().any(|event| {
                matches!(
                    event.action,
                    AuditAction::BoundedRecovery {
                        ref recovery_action
                    } if recovery_action == "clamp_rtol_to_min"
                )
            }),
            "expected bounded recovery entry for rtol clamp"
        );
    }

    #[test]
    fn solve_ivp_with_audit_rejects_nan_rtol() {
        let audit_ledger = crate::sync_audit_ledger();
        let err = solve_ivp_with_audit(
            &mut |_t, y| vec![-0.5 * y[0]],
            &SolveIvpOptions {
                t_span: (0.0, 1.0),
                y0: &[2.0],
                method: SolverKind::Rk45,
                rtol: f64::NAN,
                atol: ToleranceValue::Scalar(1e-8),
                first_step: Some(1e-3),
                max_step: 0.1,
                mode: RuntimeMode::Hardened,
                ..SolveIvpOptions::default()
            },
            &audit_ledger,
        )
        .expect_err("NaN rtol should fail closed before stepping");
        assert_eq!(err, IntegrateValidationError::NonFiniteRtol);

        let ledger = audit_ledger.lock().expect("lock");
        assert_eq!(ledger.len(), 1);
        assert!(matches!(
            ledger.entries()[0].action,
            AuditAction::FailClosed { ref reason } if reason == "rtol_must_not_be_nan"
        ));
    }

    /// frankenscipy-3cu8u.1: an audit event is fingerprinted by the whole request. The old
    /// digests covered only the failing validator's own arguments (`max_step` alone, or
    /// `t_eval` with the span) or a `Debug` string in which every NaN payload reads `NaN`, so
    /// every pair compared below shared one fingerprint under them.
    #[test]
    fn audit_fingerprints_cover_every_input() {
        // Run one audited request that is rejected before any step: its error, the fingerprint
        // its events share, and how many events it recorded.
        let rejected = |options: &SolveIvpOptions<'_>| {
            let ledger = crate::sync_audit_ledger();
            let err = solve_ivp_with_audit(&mut |_t, _y| Vec::new(), options, &ledger)
                .expect_err("the request is rejected");
            let guard = ledger.lock().expect("lock");
            let entries = guard.entries();
            assert!(!entries.is_empty(), "the rejection is recorded");
            assert!(
                entries
                    .iter()
                    .all(|entry| entry.input_fingerprint == entries[0].input_fingerprint),
                "one call, one fingerprint"
            );
            (err, entries[0].input_fingerprint.clone(), entries.len())
        };
        let y0 = [1.0, 2.0];
        let other_y0 = [1.0, 3.0];

        // max_step = NaN: the old digest was `max_step` alone.
        let nan_max_step = SolveIvpOptions {
            t_span: (0.0, 1.0),
            y0: &y0,
            max_step: f64::NAN,
            ..SolveIvpOptions::default()
        };
        let (err, fingerprint, _) = rejected(&nan_max_step);
        assert_eq!(err, IntegrateValidationError::NonFiniteMaxStep);
        assert!(fingerprint.starts_with("blake3:"));
        assert_eq!(fingerprint, rejected(&nan_max_step).1);
        let changed_y0 = SolveIvpOptions {
            y0: &other_y0,
            ..nan_max_step.clone()
        };
        assert_ne!(fingerprint, rejected(&changed_y0).1);
        // The standalone validator is a different routine with the same argument.
        let ledger = crate::sync_audit_ledger();
        assert!(crate::validate_max_step_with_audit(f64::NAN, Some(&ledger)).is_err());
        let standalone = ledger.lock().expect("lock").entries()[0]
            .input_fingerprint
            .clone();
        assert_ne!(fingerprint, standalone);

        // t_eval out of span: the old digest was `t_eval`, `t0` and `tf`.
        let t_eval = [0.0, 2.0];
        let out_of_span = SolveIvpOptions {
            t_span: (0.0, 1.0),
            y0: &y0,
            t_eval: Some(&t_eval),
            ..SolveIvpOptions::default()
        };
        let (err, fingerprint, _) = rejected(&out_of_span);
        assert_eq!(err, IntegrateValidationError::TEvalOutOfSpan);
        let tighter_rtol = SolveIvpOptions {
            rtol: 1e-6,
            ..out_of_span.clone()
        };
        assert_ne!(fingerprint, rejected(&tighter_rtol).1);
        let bdf = SolveIvpOptions {
            method: SolverKind::Bdf,
            ..out_of_span.clone()
        };
        assert_ne!(fingerprint, rejected(&bdf).1);

        // Non-finite y0 differing only in the NaN payload: the old digest read `NaN` for both.
        let quiet_nan = [f64::NAN, 1.0];
        let other_nan = [f64::from_bits(f64::NAN.to_bits() ^ 1), 1.0];
        let nan_y0 = SolveIvpOptions {
            t_span: (0.0, 1.0),
            y0: &quiet_nan,
            ..SolveIvpOptions::default()
        };
        let other_nan_y0 = SolveIvpOptions {
            y0: &other_nan,
            ..nan_y0.clone()
        };
        let (err, fingerprint, _) = rejected(&nan_y0);
        assert_eq!(err, IntegrateValidationError::NonFiniteY0);
        assert_ne!(fingerprint, rejected(&other_nan_y0).1);

        // Hardened: the rtol clamp and then the atol shape rejection, two events of one call.
        let clamp_then_reject = SolveIvpOptions {
            t_span: (0.0, 1.0),
            y0: &y0,
            rtol: 0.0,
            atol: ToleranceValue::Vector(vec![1e-6]),
            mode: RuntimeMode::Hardened,
            ..SolveIvpOptions::default()
        };
        let (err, _, events) = rejected(&clamp_then_reject);
        assert_eq!(
            err,
            IntegrateValidationError::AtolWrongShape {
                expected: 2,
                actual: 1
            }
        );
        assert_eq!(events, 2, "bounded recovery, then fail-closed");

        // The standalone validate_tol: NaN rtol differing only in the payload.
        let validate_tol_fingerprint = |rtol: f64| {
            let ledger = crate::sync_audit_ledger();
            let err = crate::validate_tol_with_audit(
                ToleranceValue::Scalar(rtol),
                ToleranceValue::Scalar(1e-6),
                1,
                RuntimeMode::Strict,
                Some(&ledger),
            )
            .expect_err("NaN rtol is rejected");
            assert_eq!(err, IntegrateValidationError::NonFiniteRtol);
            let guard = ledger.lock().expect("lock");
            assert_eq!(guard.len(), 1);
            guard.entries()[0].input_fingerprint.clone()
        };
        assert_eq!(
            validate_tol_fingerprint(f64::NAN),
            validate_tol_fingerprint(f64::NAN)
        );
        assert_ne!(
            validate_tol_fingerprint(f64::NAN),
            validate_tol_fingerprint(f64::from_bits(f64::NAN.to_bits() ^ 1))
        );
    }

    #[test]
    fn solve_ivp_bdf_reports_newton_diagnostics() {
        let result = solve_ivp(
            &mut |_t, y| vec![-1000.0 * y[0]],
            &SolveIvpOptions {
                t_span: (0.0, 0.01),
                y0: &[1.0],
                method: SolverKind::Bdf,
                rtol: 1e-6,
                atol: ToleranceValue::Scalar(1e-8),
                first_step: Some(1e-6),
                max_step: 1e-3,
                ..SolveIvpOptions::default()
            },
        )
        .expect("BDF solve should succeed");

        assert!(result.success);
        assert!(result.njev > 0, "BDF should report Jacobian evaluations");
        assert!(result.nlu > 0, "BDF should report LU factorizations");
    }

    #[test]
    fn solve_ivp_with_event() {
        fn event_at_5(_t: f64, y: &[f64]) -> f64 {
            y[0] - 5.0
        }

        let result = solve_ivp(
            &mut |_t, _y| vec![1.0],
            &SolveIvpOptions {
                t_span: (0.0, 10.0),
                y0: &[0.0],
                method: SolverKind::Rk45,
                events: Some(vec![EventSpec::terminal(event_at_5)]),
                ..SolveIvpOptions::default()
            },
        )
        .expect("solve_ivp with event should succeed");

        assert!(result.success);
        assert_eq!(result.status, 1);
        assert!((result.t.last().unwrap() - 5.0).abs() < 1e-8);
        assert!((result.y.last().unwrap()[0] - 5.0).abs() < 1e-8);

        let t_events = result.t_events.unwrap();
        assert_eq!(t_events[0].len(), 1);
        assert!((t_events[0][0] - 5.0).abs() < 1e-8);
    }

    #[test]
    fn solve_ivp_terminal_event_inside_long_step_truncates_main_solution() {
        fn event_at_0_3(_t: f64, y: &[f64]) -> f64 {
            y[0] - 0.3
        }

        let t_eval = [0.1, 0.2, 0.3, 0.4, 0.5];
        let result = solve_ivp(
            &mut |_t, _y| vec![1.0],
            &SolveIvpOptions {
                t_span: (0.0, 1.0),
                y0: &[0.0],
                method: SolverKind::Rk45,
                t_eval: Some(&t_eval),
                events: Some(vec![EventSpec::terminal(event_at_0_3)]),
                ..SolveIvpOptions::default()
            },
        )
        .expect("event truncation test");

        assert_eq!(result.t.len(), 3);
        assert!((result.t[0] - 0.1).abs() < 1e-12);
        assert!((result.t[1] - 0.2).abs() < 1e-12);
        assert!((result.t[2] - 0.3).abs() < 1e-12);
    }

    #[test]
    fn solve_ivp_lsoda_handles_nonstiff_decay() {
        let result = solve_ivp(
            &mut |_t, y| vec![-0.5 * y[0]],
            &SolveIvpOptions {
                t_span: (0.0, 4.0),
                y0: &[2.0],
                method: SolverKind::Lsoda,
                rtol: 1e-6,
                atol: ToleranceValue::Scalar(1e-8),
                ..SolveIvpOptions::default()
            },
        )
        .expect("LSODA solve_ivp should succeed");

        assert!(result.success);
        assert_eq!(result.status, 0);
        let y_final = result.y.last().expect("final state")[0];
        let expected = 2.0 * (-2.0_f64).exp();
        assert!(
            (y_final - expected).abs() < 5e-4,
            "LSODA nonstiff solve drifted: got {y_final}, expected {expected}"
        );
    }

    #[test]
    fn solve_ivp_lsoda_preserves_event_handling() {
        fn event_at_0_4(_t: f64, y: &[f64]) -> f64 {
            y[0] - 0.4
        }

        let result = solve_ivp(
            &mut |_t, _y| vec![1.0],
            &SolveIvpOptions {
                t_span: (0.0, 1.0),
                y0: &[0.0],
                method: SolverKind::Lsoda,
                events: Some(vec![EventSpec::terminal(event_at_0_4)]),
                t_eval: Some(&[0.1, 0.2, 0.3, 0.4, 0.5]),
                ..SolveIvpOptions::default()
            },
        )
        .expect("LSODA event solve should succeed");

        assert!(result.success);
        assert_eq!(result.status, 1);
        assert_eq!(result.t.len(), 4);
        assert!((result.t[3] - 0.4).abs() < 1e-8);
        assert!((result.y[3][0] - 0.4).abs() < 1e-8);
    }

    #[test]
    fn solve_ivp_event_direction_filters_crossings() {
        fn event_at_half(_t: f64, y: &[f64]) -> f64 {
            y[0] - 0.5
        }

        let upward = solve_ivp(
            &mut |_t, _y| vec![-1.0],
            &SolveIvpOptions {
                t_span: (0.0, 1.0),
                y0: &[1.0],
                method: SolverKind::Rk45,
                events: Some(vec![EventSpec::terminal(event_at_half).with_direction(1.0)]),
                ..SolveIvpOptions::default()
            },
        )
        .expect("direction +1 should skip downward crossing");

        assert_eq!(upward.status, 0);
        assert!(upward.t_events.as_ref().unwrap()[0].is_empty());

        let downward = solve_ivp(
            &mut |_t, _y| vec![-1.0],
            &SolveIvpOptions {
                t_span: (0.0, 1.0),
                y0: &[1.0],
                method: SolverKind::Rk45,
                events: Some(vec![
                    EventSpec::terminal(event_at_half).with_direction(-1.0),
                ]),
                ..SolveIvpOptions::default()
            },
        )
        .expect("direction -1 should capture downward crossing");

        assert_eq!(downward.status, 1);
        assert!(
            (downward.t_events.as_ref().unwrap()[0][0] - 0.5).abs() < 1e-6,
            "expected event near t=0.5"
        );
    }

    #[test]
    fn solve_ivp_rejects_non_finite_event_direction_before_stepping() {
        fn event_at_half(_t: f64, y: &[f64]) -> f64 {
            y[0] - 0.5
        }

        for bad_direction in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut rhs_calls = 0usize;
            let mut rhs = |_t: f64, _y: &[f64]| {
                rhs_calls += 1;
                vec![1.0]
            };
            let err = solve_ivp(
                &mut rhs,
                &SolveIvpOptions {
                    t_span: (0.0, 1.0),
                    y0: &[0.0],
                    method: SolverKind::Rk45,
                    events: Some(vec![
                        EventSpec::terminal(event_at_half).with_direction(bad_direction),
                    ]),
                    ..SolveIvpOptions::default()
                },
            )
            .expect_err("non-finite event direction should fail before stepping");
            assert_eq!(
                err,
                IntegrateValidationError::NonFiniteEventDirection { index: 0 }
            );
            assert_eq!(rhs_calls, 0, "invalid metadata must fail before RHS calls");
        }
    }

    #[test]
    fn solve_ivp_rejects_zero_event_max_events_before_stepping() {
        fn event_at_half(_t: f64, y: &[f64]) -> f64 {
            y[0] - 0.5
        }

        let mut rhs_calls = 0usize;
        let mut rhs = |_t: f64, _y: &[f64]| {
            rhs_calls += 1;
            vec![1.0]
        };
        let err = solve_ivp(
            &mut rhs,
            &SolveIvpOptions {
                t_span: (0.0, 1.0),
                y0: &[0.0],
                method: SolverKind::Rk45,
                events: Some(vec![EventSpec {
                    func: event_at_half,
                    direction: 0.0,
                    max_events: Some(0),
                }]),
                ..SolveIvpOptions::default()
            },
        )
        .expect_err("zero max_events should fail before stepping");
        assert_eq!(
            err,
            IntegrateValidationError::EventMaxEventsMustBePositive { index: 0 }
        );
        assert_eq!(
            rhs_calls, 0,
            "invalid event metadata must fail before RHS calls"
        );
    }

    #[test]
    fn solve_ivp_rejects_non_finite_initial_event_value() {
        fn bad_event(_t: f64, _y: &[f64]) -> f64 {
            f64::NAN
        }

        let mut rhs_calls = 0usize;
        let mut rhs = |_t: f64, _y: &[f64]| {
            rhs_calls += 1;
            vec![1.0]
        };
        let err = solve_ivp(
            &mut rhs,
            &SolveIvpOptions {
                t_span: (0.0, 1.0),
                y0: &[0.0],
                method: SolverKind::Rk45,
                events: Some(vec![EventSpec::new(bad_event)]),
                ..SolveIvpOptions::default()
            },
        )
        .expect_err("non-finite initial event value");
        assert_eq!(
            err,
            IntegrateValidationError::NonFiniteEventValue { index: 0 }
        );
        assert_eq!(
            rhs_calls, 0,
            "initial event validation must run before stepping"
        );
    }

    #[test]
    fn solve_ivp_rejects_non_finite_stepped_event_value() {
        fn bad_after_start(t: f64, y: &[f64]) -> f64 {
            if t == 0.0 { y[0] - 0.5 } else { f64::INFINITY }
        }

        let err = solve_ivp(
            &mut |_t, _y| vec![1.0],
            &SolveIvpOptions {
                t_span: (0.0, 1.0),
                y0: &[0.0],
                method: SolverKind::Rk45,
                first_step: Some(0.1),
                events: Some(vec![EventSpec::new(bad_after_start)]),
                ..SolveIvpOptions::default()
            },
        )
        .expect_err("non-finite stepped event value");
        assert_eq!(
            err,
            IntegrateValidationError::NonFiniteEventValue { index: 0 }
        );
    }

    #[test]
    fn solve_ivp_rejects_non_finite_event_value_during_root_solve() {
        fn bad_inside_bracket(t: f64, _y: &[f64]) -> f64 {
            if t <= 0.0 {
                -1.0
            } else if t >= 1.0 {
                1.0
            } else {
                f64::NAN
            }
        }

        let err = solve_ivp(
            &mut |_t, _y| vec![0.0],
            &SolveIvpOptions {
                t_span: (0.0, 1.0),
                y0: &[0.0],
                method: SolverKind::Rk45,
                first_step: Some(1.0),
                max_step: 1.0,
                events: Some(vec![EventSpec::new(bad_inside_bracket)]),
                ..SolveIvpOptions::default()
            },
        )
        .expect_err("non-finite event value during root solve");
        assert_eq!(
            err,
            IntegrateValidationError::NonFiniteEventValue { index: 0 }
        );
    }

    #[test]
    fn solve_ivp_nonterminal_event_does_not_stop_integration() {
        fn event_at_half(_t: f64, y: &[f64]) -> f64 {
            y[0] - 0.5
        }

        let result = solve_ivp(
            &mut |_t, _y| vec![1.0],
            &SolveIvpOptions {
                t_span: (0.0, 1.0),
                y0: &[0.0],
                method: SolverKind::Rk45,
                events: Some(vec![EventSpec::new(event_at_half)]),
                ..SolveIvpOptions::default()
            },
        )
        .expect("non-terminal event should not stop integration");

        assert_eq!(result.status, 0);
        let t_events = result.t_events.expect("t_events should be populated");
        assert_eq!(t_events.len(), 1);
        assert_eq!(t_events[0].len(), 1);
        assert!(
            (t_events[0][0] - 0.5).abs() < 1e-6,
            "expected non-terminal event near t=0.5"
        );
    }

    #[test]
    fn solve_ivp_event_max_events_stops_after_limit() {
        fn periodic_event(t: f64, _y: &[f64]) -> f64 {
            (2.0 * std::f64::consts::PI * t).sin()
        }

        let result = solve_ivp(
            &mut |_t, _y| vec![0.0],
            &SolveIvpOptions {
                t_span: (0.1, 2.0),
                y0: &[0.0],
                method: SolverKind::Rk45,
                events: Some(vec![EventSpec::new(periodic_event).with_max_events(2)]),
                max_step: 0.1,
                first_step: Some(0.05),
                ..SolveIvpOptions::default()
            },
        )
        .expect("periodic event solve should succeed");

        assert_eq!(result.status, 1, "expected termination after max_events");
        let t_events = result.t_events.expect("t_events should be present");
        assert_eq!(t_events.len(), 1);
        assert_eq!(t_events[0].len(), 2);
        assert!(
            (t_events[0][0] - 0.5).abs() < 1e-6,
            "expected first event near t=0.5"
        );
        assert!(
            (t_events[0][1] - 1.0).abs() < 1e-6,
            "expected second event near t=1.0"
        );
        assert!(
            (result.t.last().unwrap() - 1.0).abs() < 1e-6,
            "solver should stop at the second event"
        );
    }

    #[test]
    fn solve_ivp_dense_output_returns_solver_knots_with_t_eval_present() {
        let result = solve_ivp(
            &mut |_t, y| vec![-0.5 * y[0]],
            &SolveIvpOptions {
                t_span: (0.0, 1.0),
                y0: &[2.0],
                method: SolverKind::Rk45,
                t_eval: Some(&[0.25, 0.5, 0.75, 1.0]),
                dense_output: true,
                rtol: 1e-6,
                atol: ToleranceValue::Scalar(1e-8),
                ..SolveIvpOptions::default()
            },
        )
        .expect("dense output solve should succeed");

        let sol = result.sol.expect("dense_output should populate result.sol");
        assert!(
            !sol.alt_segment,
            "RK dense output should not use alt_segment"
        );
        assert_eq!(sol.knots.first().copied(), Some(0.0));
        assert_eq!(sol.values.first().cloned(), Some(vec![2.0]));
        assert!(
            sol.knots.len() > result.t.len(),
            "dense-output knots should be solver-chosen, not just t_eval points"
        );
        assert_eq!(result.t, vec![0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn solve_ivp_dense_output_uses_alt_segment_for_lsoda() {
        let result = solve_ivp(
            &mut |_t, y| vec![-0.5 * y[0]],
            &SolveIvpOptions {
                t_span: (0.0, 1.0),
                y0: &[2.0],
                method: SolverKind::Lsoda,
                dense_output: true,
                rtol: 1e-6,
                atol: ToleranceValue::Scalar(1e-8),
                ..SolveIvpOptions::default()
            },
        )
        .expect("LSODA dense output solve should succeed");

        let sol = result.sol.expect("dense_output should populate result.sol");
        assert!(sol.alt_segment, "LSODA should set alt_segment");
        assert_eq!(sol.knots.first().copied(), Some(0.0));
        assert_eq!(sol.values.first().cloned(), Some(vec![2.0]));
        assert_eq!(
            sol.knots.last().copied(),
            result.t.last().copied(),
            "dense-output knots should end at the final solver time"
        );
    }

    #[test]
    fn solve_ivp_zero_length_interval_returns_single_point() {
        let result = solve_ivp(
            &mut |_t, y| vec![-0.5 * y[0]],
            &SolveIvpOptions {
                t_span: (1.0, 1.0),
                y0: &[2.0],
                method: SolverKind::Rk45,
                dense_output: true,
                ..SolveIvpOptions::default()
            },
        )
        .expect("zero-length solve should succeed");

        assert!(result.success);
        assert_eq!(result.status, 0);
        assert_eq!(result.t, vec![1.0]);
        assert_eq!(result.y, vec![vec![2.0]]);

        let sol = result.sol.expect("dense output should still be populated");
        assert_eq!(sol.knots, vec![1.0]);
        assert_eq!(sol.values, vec![vec![2.0]]);
    }

    #[test]
    fn lsoda_switches_to_bdf_for_stiff_problem() {
        let options = SolveIvpOptions {
            t_span: (0.0, 0.1),
            y0: &[1.0],
            method: SolverKind::Lsoda,
            rtol: 1e-4,
            atol: ToleranceValue::Scalar(1e-6),
            first_step: Some(1e-6),
            ..SolveIvpOptions::default()
        };
        let mut fun = |t: f64, y: &[f64]| vec![-1000.0 * (y[0] - t.cos())];
        let mut solver = LsodaSolver::from_options(&mut fun, &options).expect("LSODA init");

        let mut switched = false;
        for _ in 0..2000 {
            let outcome = solver.step_with(&mut fun).expect("LSODA step");
            if matches!(solver.mode, LsodaMode::Bdf(_)) {
                switched = true;
            }
            if outcome.state != OdeSolverState::Running {
                break;
            }
        }

        assert!(
            switched,
            "LSODA wrapper never switched to BDF on a stiff system"
        );
        let expected = 0.1_f64.cos();
        let final_y = match &solver.mode {
            LsodaMode::Adams(rk) => rk.y()[0],
            LsodaMode::Bdf(bdf) => bdf.y()[0],
        };
        assert!(
            (final_y - expected).abs() < 0.05,
            "LSODA stiff solve ended at {}, expected about {}",
            final_y,
            expected
        );
    }

    /// `scipy.integrate.LSODA` used to be an empty unit struct (frankenscipy-8dndw.1). The
    /// public handle must drive exactly the stepper `solve_ivp(method="LSODA")` runs.
    #[test]
    fn public_lsoda_handle_steps_like_solve_ivp() {
        let mut fun = |t: f64, y: &[f64]| vec![-1000.0 * (y[0] - t.cos())];
        let mut solver = crate::LSODA::new(
            &mut fun,
            LsodaSolverConfig {
                t0: 0.0,
                y0: &[1.0],
                t_bound: 0.1,
                rtol: 1e-4,
                atol: ToleranceValue::Scalar(1e-6),
                max_step: f64::INFINITY,
                first_step: Some(1e-6),
                mode: RuntimeMode::Strict,
            },
        )
        .expect("LSODA init");
        let mut ts = vec![solver.t()];
        while solver.state() == OdeSolverState::Running {
            solver.step_with(&mut fun).expect("LSODA step");
            ts.push(solver.t());
            let t_old = solver.t_old().expect("a step was taken");
            let midpoint = solver
                .dense_output_at(0.5 * (t_old + solver.t()))
                .expect("dense output inside the last step");
            assert!(midpoint[0].is_finite());
        }
        assert!(
            solver.is_stiff_phase(),
            "stiff problem must reach the BDF phase"
        );
        assert_eq!(solver.state(), OdeSolverState::Finished);

        let options = SolveIvpOptions {
            t_span: (0.0, 0.1),
            y0: &[1.0],
            method: SolverKind::Lsoda,
            rtol: 1e-4,
            atol: ToleranceValue::Scalar(1e-6),
            first_step: Some(1e-6),
            ..SolveIvpOptions::default()
        };
        let mut fun2 = |t: f64, y: &[f64]| vec![-1000.0 * (y[0] - t.cos())];
        let reference = solve_ivp(&mut fun2, &options).expect("solve_ivp LSODA");
        assert_eq!(
            ts, reference.t,
            "the handle and solve_ivp took different steps"
        );
        assert_eq!(
            solver.y()[0].to_bits(),
            reference.y.last().expect("final state")[0].to_bits()
        );
        assert_eq!(solver.nfev(), reference.nfev);
    }

    #[test]
    fn solve_ivp_lsoda_van_der_pol_matches_scipy_reference() {
        let mu = 10.0;
        let mut fun = |_t: f64, y: &[f64]| vec![y[1], mu * (1.0 - y[0] * y[0]) * y[1] - y[0]];
        let y0 = [2.0, 0.0];
        let t_eval = [0.0, 0.5, 1.0, 2.0];
        let opts = SolveIvpOptions {
            t_span: (0.0, 2.0),
            y0: &y0,
            method: SolverKind::Lsoda,
            rtol: 1e-8,
            atol: ToleranceValue::Scalar(1e-10),
            t_eval: Some(&t_eval),
            mode: RuntimeMode::Strict,
            ..Default::default()
        };
        let r = solve_ivp(&mut fun, &opts).expect("LSODA solve should succeed");
        assert!(r.success, "LSODA should succeed on Van der Pol");
        assert_eq!(r.t, vec![0.0, 0.5, 1.0, 2.0]);
        let expected_y0 = [2.0, 1.96853255, 1.93385289, 1.86106865];
        let expected_y1 = [0.0, -0.06832900, -0.07042352, -0.07532164];
        for (i, y_step) in r.y.iter().enumerate() {
            assert!(
                (y_step[0] - expected_y0[i]).abs() < 1e-4,
                "step {i} y[0]: got {}, expected {}",
                y_step[0],
                expected_y0[i]
            );
            assert!(
                (y_step[1] - expected_y1[i]).abs() < 1e-4,
                "step {i} y[1]: got {}, expected {}",
                y_step[1],
                expected_y1[i]
            );
        }
    }

    #[test]
    fn solve_ivp_exponential_decay_matches_scipy_reference_values() {
        // scipy.integrate.solve_ivp(exp_decay, [0, 4], [1.0], t_eval=[0, 1, 2, 3, 4], method='RK45')
        // where exp_decay = lambda t, y: -0.5 * y
        let result = solve_ivp(
            &mut |_t, y| vec![-0.5 * y[0]],
            &SolveIvpOptions {
                t_span: (0.0, 4.0),
                y0: &[1.0],
                method: SolverKind::Rk45,
                t_eval: Some(&[0.0, 1.0, 2.0, 3.0, 4.0]),
                rtol: 1e-6,
                atol: ToleranceValue::Scalar(1e-8),
                ..SolveIvpOptions::default()
            },
        )
        .expect("solve_ivp should succeed");

        assert!(result.success, "integration should succeed");
        let expected_y = [
            1.0,
            0.6065268298424474,
            0.3676699729783925,
            0.22325717532500178,
            0.13541609549742406,
        ];
        for (i, (got, want)) in result
            .y
            .iter()
            .map(|y| y[0])
            .zip(expected_y.iter())
            .enumerate()
        {
            assert!(
                (got - want).abs() < 1e-3,
                "y[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn solve_ivp_harmonic_oscillator_matches_scipy_reference_values() {
        // scipy.integrate.solve_ivp(harmonic, [0, 2*pi], [1.0, 0.0], t_eval=[0, pi/2, pi, 3*pi/2, 2*pi])
        // where harmonic = lambda t, state: [state[1], -state[0]]
        use std::f64::consts::PI;
        let result = solve_ivp(
            &mut |_t, state| vec![state[1], -state[0]],
            &SolveIvpOptions {
                t_span: (0.0, 2.0 * PI),
                y0: &[1.0, 0.0],
                method: SolverKind::Rk45,
                t_eval: Some(&[0.0, PI / 2.0, PI, 3.0 * PI / 2.0, 2.0 * PI]),
                rtol: 1e-6,
                atol: ToleranceValue::Scalar(1e-8),
                ..SolveIvpOptions::default()
            },
        )
        .expect("solve_ivp should succeed");

        assert!(result.success, "integration should succeed");
        let expected_y = [1.0, 0.0, -1.0, 0.0, 1.0];
        let expected_v = [0.0, -1.0, 0.0, 1.0, 0.0];
        for (i, (got_y, got_v)) in result.y.iter().map(|y| (y[0], y[1])).enumerate() {
            let want_y = expected_y[i];
            let want_v = expected_v[i];
            assert!(
                (got_y - want_y).abs() < 1e-4,
                "y[{i}] got {got_y}, expected {want_y}"
            );
            assert!(
                (got_v - want_v).abs() < 1e-4,
                "v[{i}] got {got_v}, expected {want_v}"
            );
        }
    }

    #[test]
    fn solve_ivp_rk23_exponential_decay_matches_scipy_reference() {
        // scipy.integrate.solve_ivp(exp_decay, [0, 4], [1.0], t_eval=[0, 2, 4], method='RK23')
        let result = solve_ivp(
            &mut |_t, y| vec![-0.5 * y[0]],
            &SolveIvpOptions {
                t_span: (0.0, 4.0),
                y0: &[1.0],
                method: SolverKind::Rk23,
                t_eval: Some(&[0.0, 2.0, 4.0]),
                rtol: 1e-6,
                atol: ToleranceValue::Scalar(1e-8),
                ..SolveIvpOptions::default()
            },
        )
        .expect("solve_ivp RK23 should succeed");

        assert!(result.success);
        // exp(-0.5 * t): t=0 -> 1.0, t=2 -> 0.3679, t=4 -> 0.1353
        let expected = [1.0, 0.36787944117144233, 0.1353352832366127];
        for (i, (got, want)) in result
            .y
            .iter()
            .map(|y| y[0])
            .zip(expected.iter())
            .enumerate()
        {
            assert!(
                (got - want).abs() < 1e-3,
                "y[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn solve_ivp_dop853_exponential_decay_matches_scipy_reference() {
        // scipy.integrate.solve_ivp(exp_decay, [0, 4], [1.0], t_eval=[0, 2, 4], method='DOP853')
        let result = solve_ivp(
            &mut |_t, y| vec![-0.5 * y[0]],
            &SolveIvpOptions {
                t_span: (0.0, 4.0),
                y0: &[1.0],
                method: SolverKind::Dop853,
                t_eval: Some(&[0.0, 2.0, 4.0]),
                rtol: 1e-6,
                atol: ToleranceValue::Scalar(1e-8),
                ..SolveIvpOptions::default()
            },
        )
        .expect("solve_ivp DOP853 should succeed");

        assert!(result.success);
        let expected = [1.0, 0.36787944117144233, 0.1353352832366127];
        for (i, (got, want)) in result
            .y
            .iter()
            .map(|y| y[0])
            .zip(expected.iter())
            .enumerate()
        {
            assert!(
                (got - want).abs() < 1e-3,
                "y[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn solve_ivp_bdf_stiff_exponential_decay_matches_scipy_reference() {
        // scipy.integrate.solve_ivp(stiff_decay, [0, 10], [1.0], method='BDF')
        // stiff_decay = lambda t, y: -50 * y (stiff problem)
        let result = solve_ivp(
            &mut |_t, y| vec![-50.0 * y[0]],
            &SolveIvpOptions {
                t_span: (0.0, 1.0),
                y0: &[1.0],
                method: SolverKind::Bdf,
                t_eval: Some(&[0.0, 0.5, 1.0]),
                rtol: 1e-4,
                atol: ToleranceValue::Scalar(1e-6),
                ..SolveIvpOptions::default()
            },
        )
        .expect("solve_ivp BDF should succeed for stiff problem");

        assert!(result.success);
        // True exp(-50 t): t=0.5 -> 1.93e-11, t=1.0 -> 1.93e-22. Once the solution
        // decays below atol=1e-6 the error control no longer tracks it, so the
        // tail just drifts within ~atol — scipy BDF here returns y(0.5)=3.48e-8,
        // y(1.0)=-1.99e-9. Match scipy's behaviour: the tail must stay within the
        // absolute tolerance, not pinned to the true (untracked) value. (The old
        // <1e-10 bound over-fit the previous fixed-order-1 solver; frankenscipy-3y5p9.)
        assert!(
            (result.y[0][0] - 1.0).abs() < 1e-6,
            "y[0] = {}",
            result.y[0][0]
        );
        assert!(
            result.y[1][0].abs() < 1e-6,
            "y[0.5] should be within atol of zero: {}",
            result.y[1][0]
        );
        assert!(
            result.y[2][0].abs() < 1e-6,
            "y[1.0] should be within atol of zero: {}",
            result.y[2][0]
        );
    }

    #[test]
    fn solve_ivp_radau_matches_scipy_reference() {
        // Genuine Radau IIA (frankenscipy-3y5p9), no longer a BDF alias.
        // (1) exp decay y' = -y: y(5) = exp(-5) to tolerance.
        let decay = solve_ivp(
            &mut |_t, y| vec![-y[0]],
            &SolveIvpOptions {
                t_span: (0.0, 5.0),
                y0: &[1.0],
                method: SolverKind::Radau,
                t_eval: Some(&[5.0]),
                rtol: 1e-8,
                atol: ToleranceValue::Scalar(1e-10),
                ..SolveIvpOptions::default()
            },
        )
        .expect("Radau decay");
        assert!(
            (decay.y[0][0] - (-5.0_f64).exp()).abs() < 1e-7,
            "Radau y(5) = {}, want {}",
            decay.y[0][0],
            (-5.0_f64).exp()
        );

        // (2) 2-D stiff linear y1' = -100 y1 + y2, y2' = -y2; scipy Radau golden.
        let stiff = solve_ivp(
            &mut |_t, y| vec![-100.0 * y[0] + y[1], -y[1]],
            &SolveIvpOptions {
                t_span: (0.0, 2.0),
                y0: &[1.0, 1.0],
                method: SolverKind::Radau,
                t_eval: Some(&[2.0]),
                rtol: 1e-8,
                atol: ToleranceValue::Scalar(1e-10),
                ..SolveIvpOptions::default()
            },
        )
        .expect("Radau stiff 2D");
        assert!(
            (stiff.y[0][0] - 0.001_367_023_063_007_146_6).abs() < 1e-8,
            "Radau stiff y1(2) = {}",
            stiff.y[0][0]
        );
        assert!(
            (stiff.y[0][1] - (-2.0_f64).exp()).abs() < 1e-8,
            "Radau stiff y2(2) = {}",
            stiff.y[0][1]
        );

        // (3) stiff van der Pol (mu=10): final state matches scipy Radau golden.
        let mu = 10.0;
        let vdp = solve_ivp(
            &mut |_t, y| vec![y[1], mu * (1.0 - y[0] * y[0]) * y[1] - y[0]],
            &SolveIvpOptions {
                t_span: (0.0, 20.0),
                y0: &[2.0, 0.0],
                method: SolverKind::Radau,
                rtol: 1e-6,
                atol: ToleranceValue::Scalar(1e-9),
                ..SolveIvpOptions::default()
            },
        )
        .expect("Radau vdp");
        let last = vdp.y.len() - 1;
        assert!(
            (vdp.y[last][0] - 1.939_359).abs() < 1e-4
                && (vdp.y[last][1] - (-0.070_082)).abs() < 1e-4,
            "Radau vdp final = [{}, {}], want ~[1.939359, -0.070082]",
            vdp.y[last][0],
            vdp.y[last][1]
        );
    }

    #[test]
    fn test_solve_ivp_with_casp_portfolio_nonstiff_routes_to_rk45() {
        let mut portfolio = OdeSolverPortfolio::new(RuntimeMode::Strict, 16);
        let opts = SolveIvpOptions {
            t_span: (0.0, 1.0),
            y0: &[1.0, 0.0],
            ..SolveIvpOptions::default()
        };
        let mut f = |_t: f64, y: &[f64]| vec![y[1], -y[0]];
        let res = solve_ivp_with_casp_portfolio(&mut f, &opts, &mut portfolio, 1.0, false)
            .expect("solve_ivp portfolio");

        assert_eq!(res.chosen_action, OdeSolverAction::RK45);
        assert!(res.result.success);
        assert_eq!(portfolio.evidence_len(), 1);
    }

    #[test]
    fn test_solve_ivp_with_casp_portfolio_stiff_routes_to_bdf() {
        let mut portfolio = OdeSolverPortfolio::new(RuntimeMode::Strict, 16);
        let opts = SolveIvpOptions {
            t_span: (0.0, 1.0),
            y0: &[1.0, 0.0],
            ..SolveIvpOptions::default()
        };
        let mut f = |_t: f64, y: &[f64]| vec![y[1], -y[0]];
        let res = solve_ivp_with_casp_portfolio(&mut f, &opts, &mut portfolio, 1e6, false)
            .expect("solve_ivp portfolio");

        assert_eq!(res.chosen_action, OdeSolverAction::BDF);
        assert!(res.result.success);
        assert_eq!(portfolio.evidence_len(), 1);
    }

    #[test]
    fn test_solve_ivp_with_casp_portfolio_algebraic_routes_to_radau() {
        let mut portfolio = OdeSolverPortfolio::new(RuntimeMode::Strict, 16);
        let opts = SolveIvpOptions {
            t_span: (0.0, 1.0),
            y0: &[1.0, 0.0],
            ..SolveIvpOptions::default()
        };
        let mut f = |_t: f64, y: &[f64]| vec![y[1], -y[0]];
        let res = solve_ivp_with_casp_portfolio(&mut f, &opts, &mut portfolio, 10.0, true)
            .expect("solve_ivp portfolio");

        assert_eq!(res.chosen_action, OdeSolverAction::Radau);
        assert!(res.result.success);
        assert_eq!(portfolio.evidence_len(), 1);
    }

    #[test]
    fn test_solve_ivp_with_casp_standard_entrypoint() {
        let mut portfolio = OdeSolverPortfolio::new(RuntimeMode::Strict, 16);
        let opts = SolveIvpOptions {
            t_span: (0.0, 1.0),
            y0: &[1.0, 0.0],
            ..SolveIvpOptions::default()
        };
        let mut f = |_t: f64, y: &[f64]| vec![y[1], -y[0]];
        let res = solve_ivp_with_casp(&mut f, &opts, &mut portfolio).expect("solve_ivp_with_casp");

        assert_eq!(res.chosen_action, OdeSolverAction::RK45);
        assert!(res.result.success);
        assert_eq!(portfolio.evidence_len(), 1);
    }
}
