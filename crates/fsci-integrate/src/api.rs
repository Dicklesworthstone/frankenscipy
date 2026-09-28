#![forbid(unsafe_code)]

use fsci_opt::root::brentq;
use fsci_opt::types::RootOptions;
use fsci_runtime::{
    AuditScope, Fingerprinter, OdeSolverAction, OdeSolverEvidenceEntry, OdeSolverPortfolio,
    RuntimeMode,
};

use crate::IntegrateValidationError;
use crate::bdf::{BdfSolver, BdfSolverConfig};
use crate::lsoda::{JacFn, Lsoda, LsodaMethod, LsodaSetup, Task};
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

#[derive(Debug, Clone)]
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
    /// SciPy's `jac`: `jac(t, y)[i][j] = d f_i / d y_j`. LSODA uses it in place of its
    /// finite-difference Jacobian (ODEPACK `jt = 1` instead of 2). The explicit Runge-Kutta
    /// methods ignore it, as SciPy's do (with a warning there); BDF and Radau here form their
    /// Jacobians by differences only, so a `jac` with either is refused as not implemented
    /// rather than silently not used.
    pub jac: Option<JacFn>,
}

/// Field by field, with `jac` compared by address (`fn_addr_eq`), as [`EventSpec`] compares
/// its event functions.
impl PartialEq for SolveIvpOptions<'_> {
    fn eq(&self, other: &Self) -> bool {
        let jac_eq = match (self.jac, other.jac) {
            (Some(a), Some(b)) => std::ptr::fn_addr_eq(a, b),
            (None, None) => true,
            _ => false,
        };
        self.t_span == other.t_span
            && self.y0 == other.y0
            && self.method == other.method
            && self.t_eval == other.t_eval
            && self.dense_output == other.dense_output
            && self.events == other.events
            && self.rtol == other.rtol
            && self.atol == other.atol
            && self.first_step == other.first_step
            && self.max_step == other.max_step
            && self.mode == other.mode
            && jac_eq
    }
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
            jac: None,
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
/// (variant name, then value or values), `first_step` (presence flag, then value), `max_step`,
/// `mode` (`Debug`) and `jac` (presence flag only). The right-hand side, the event functions and
/// the Jacobian are code, not data, and are not part of it.
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
        .str(&format!("{:?}", options.mode))
        .bool(options.jac.is_some());
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

/// Configuration for constructing an [`LsodaSolver`]: the arguments of
/// `scipy.integrate.LSODA(fun, t0, y0, t_bound, first_step, max_step, rtol, atol, jac)`.
pub struct LsodaSolverConfig<'a> {
    pub t0: f64,
    pub y0: &'a [f64],
    pub t_bound: f64,
    pub rtol: f64,
    pub atol: ToleranceValue,
    /// The largest |h| (ODEPACK's `hmax`); `f64::INFINITY` for no bound.
    pub max_step: f64,
    /// The first step to try (ODEPACK's `h0`, given the integration direction's sign); `None`
    /// lets ODEPACK choose it from the tolerances and `f(t0, y0)`.
    pub first_step: Option<f64>,
    /// `jac(t, y)[i][j] = d f_i / d y_j` (`jt = 1`); `None` forms the Jacobian by forward
    /// differences of `fun` (`jt = 2`), one evaluation per column.
    pub jac: Option<JacFn>,
    pub mode: RuntimeMode,
}

/// `scipy.integrate.LSODA`, the stepper behind `solve_ivp(method="LSODA")`: ODEPACK's LSODA as
/// SciPy 1.17.1 runs it (a port of SciPy's `lsoda.c`, see [`crate::lsoda`]), stepped the way
/// SciPy's wrapper steps it: one ODEPACK call per step with `itask = 5` and `tcrit = t_bound`,
/// so no step passes `t_bound` and the last one lands on it.
///
/// Variable-order implicit Adams (orders 1-12, functional iteration) and BDF (orders 1-5, chord
/// iteration on `I - h el0 J`), switching automatically in both directions;
/// [`LsodaSolver::method_used`] and [`LsodaSolver::order_used`] report what took each step.
///
/// Construction evaluates nothing, as SciPy's `LSODA.__init__` does not: the first
/// [`LsodaSolver::step_with`] evaluates `f(t0, y0)` and chooses the first step. `nfev` counts
/// every right-hand-side evaluation, those of finite-difference Jacobian columns included;
/// `njev` and `nlu` both count Jacobians, each followed by one LU factorization (SciPy reports
/// ODEPACK's NJE for both).
///
/// SciPy's `min_step` has no counterpart: SciPy 1.17.1's `lsoda.c` never passes it to the
/// stepper, so it changes nothing there either (see [`crate::lsoda`]).
pub struct LsodaSolver {
    core: Lsoda,
    t: f64,
    y: Vec<f64>,
    t_old: Option<f64>,
    y_old: Option<Vec<f64>>,
    t_bound: f64,
    direction: f64,
    state: OdeSolverState,
    mode: RuntimeMode,
}

impl LsodaSolver {
    /// The solver at `(t0, y0)`, before its first step. Nothing is evaluated here.
    ///
    /// # Errors
    /// An empty or non-finite `y0`, a non-finite `t0` or `t_bound`, and what
    /// [`crate::validate_tol`], [`crate::validate_first_step`] and [`crate::validate_max_step`]
    /// reject (SciPy's `LSODA.__init__` checks).
    pub fn new(config: LsodaSolverConfig<'_>) -> Result<Self, IntegrateValidationError> {
        let n = config.y0.len();
        if n == 0 {
            return Err(IntegrateValidationError::EmptyY0);
        }
        if !config.t0.is_finite() || !config.t_bound.is_finite() {
            return Err(IntegrateValidationError::NonFiniteSpan);
        }
        if config.y0.iter().any(|v| !v.is_finite()) {
            return Err(IntegrateValidationError::NonFiniteY0);
        }
        let max_step = crate::validate_max_step(config.max_step)?;
        let first_step = config
            .first_step
            .map(|value| crate::validate_first_step(value, config.t0, config.t_bound))
            .transpose()?;
        let tol = crate::validate_tol(
            ToleranceValue::Scalar(config.rtol),
            config.atol,
            n,
            config.mode,
        )?;
        let as_vec = |value: ToleranceValue| match value {
            ToleranceValue::Scalar(v) => vec![v],
            ToleranceValue::Vector(values) => values,
        };
        // SciPy: direction = sign(t_bound - t0), or 1 when they are equal.
        let direction = if config.t_bound == config.t0 {
            1.0
        } else {
            (config.t_bound - config.t0).signum()
        };
        let setup = LsodaSetup {
            rtol: as_vec(tol.rtol),
            atol: as_vec(tol.atol),
            // SciPy: first_step * direction into rwork[4]; max_step = inf is ODEPACK's 0.
            h0: first_step.map_or(0.0, |step| step * direction),
            hmax: if max_step.is_infinite() {
                0.0
            } else {
                max_step
            },
            mxstep: 500,
            jac: config.jac,
        };
        Ok(Self {
            core: Lsoda::new(n, setup),
            t: config.t0,
            y: config.y0.to_vec(),
            t_old: None,
            y_old: None,
            t_bound: config.t_bound,
            direction,
            state: OdeSolverState::Running,
            mode: config.mode,
        })
    }

    fn from_options(options: &SolveIvpOptions<'_>) -> Result<Self, IntegrateValidationError> {
        Self::new(LsodaSolverConfig {
            t0: options.t_span.0,
            y0: options.y0,
            t_bound: options.t_span.1,
            rtol: options.rtol,
            atol: options.atol.clone(),
            max_step: options.max_step,
            first_step: options.first_step,
            jac: options.jac,
            mode: options.mode,
        })
    }

    /// The method of the last accepted step (ODEPACK's MUSED, `iwork[18]`); `None` before the
    /// first step.
    #[must_use]
    pub fn method_used(&self) -> Option<LsodaMethod> {
        self.core.method_used()
    }

    /// The method the next step will use (MCUR, `iwork[19]`). It differs from
    /// [`LsodaSolver::method_used`] right after the step that decided to switch.
    #[must_use]
    pub fn method_current(&self) -> Option<LsodaMethod> {
        self.core.method_current()
    }

    /// The order of the last accepted step (NQU, `iwork[13]`); 0 before the first step.
    #[must_use]
    pub fn order_used(&self) -> usize {
        self.core.nqu()
    }

    /// The order the next step will attempt (NQCUR, `iwork[14]`).
    #[must_use]
    pub fn order_current(&self) -> usize {
        self.core.nq()
    }

    /// The size of the last accepted step (HU, `rwork[10]`).
    #[must_use]
    pub fn step_size_used(&self) -> f64 {
        self.core.hu()
    }

    /// The step size the next step will attempt (HCUR, `rwork[11]`).
    #[must_use]
    pub fn step_size_current(&self) -> f64 {
        self.core.h()
    }

    /// Accepted steps (NST, `iwork[10]`).
    #[must_use]
    pub fn n_steps(&self) -> usize {
        self.core.nst()
    }

    /// The time of the last method switch, `t0` if there has been none (TSW, `rwork[14]`).
    #[must_use]
    pub fn t_switch(&self) -> f64 {
        self.core.tsw()
    }
}

impl LsodaSolver {
    /// Advance one step: one ODEPACK call with `itask = 5`, `tcrit = t_bound` (SciPy's
    /// `LSODA._step_impl` inside `OdeSolver.step`).
    ///
    /// # Errors
    /// Stepping a finished or failed solver, and ODEPACK's failures (`istate < 0`: repeated
    /// error-test or convergence failures, tolerances too small, a zero error weight) or a
    /// right-hand side / Jacobian of the wrong shape. The solver is then failed and keeps its
    /// last accepted state, as SciPy's does.
    pub fn step_with<F>(&mut self, fun: &mut F) -> Result<StepOutcome, StepFailure>
    where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        if self.state != OdeSolverState::Running {
            return Err(StepFailure::RuntimeError(
                "Attempt to step on a finished or failed solver.",
            ));
        }
        if self.t == self.t_bound {
            // SciPy's corner case: no integration, no evaluation.
            self.t_old = Some(self.t);
            self.y_old = Some(self.y.clone());
            self.state = OdeSolverState::Finished;
            return Ok(StepOutcome {
                message: None,
                state: OdeSolverState::Finished,
            });
        }
        let t_start = self.t;
        let mut t = self.t;
        let mut y = self.y.clone();
        match self.core.call(
            fun,
            &mut y,
            &mut t,
            self.t_bound,
            Task::OneStepToTcrit,
            self.t_bound,
        ) {
            Ok(()) => {
                self.t_old = Some(t_start);
                self.y_old = Some(std::mem::replace(&mut self.y, y));
                self.t = t;
                if self.direction * (self.t - self.t_bound) >= 0.0 {
                    self.state = OdeSolverState::Finished;
                }
                Ok(StepOutcome {
                    message: None,
                    state: self.state,
                })
            }
            Err(failure) => {
                self.state = OdeSolverState::Failed;
                Err(failure.into_step_failure())
            }
        }
    }

    #[must_use]
    pub fn t(&self) -> f64 {
        self.t
    }

    #[must_use]
    pub fn y(&self) -> &[f64] {
        &self.y
    }

    #[must_use]
    pub fn t_old(&self) -> Option<f64> {
        self.t_old
    }

    #[must_use]
    pub fn y_old(&self) -> Option<&[f64]> {
        self.y_old.as_deref()
    }

    /// Right-hand-side evaluations (ODEPACK's NFE), finite-difference Jacobian columns
    /// included.
    #[must_use]
    pub fn nfev(&self) -> usize {
        self.core.nfe()
    }

    /// Jacobian evaluations (NJE), user-supplied or by differences.
    #[must_use]
    pub fn njev(&self) -> usize {
        self.core.nje()
    }

    /// LU factorizations: one per Jacobian, so NJE again, as SciPy reports it.
    #[must_use]
    pub fn nlu(&self) -> usize {
        self.core.nje()
    }

    #[must_use]
    pub fn state(&self) -> OdeSolverState {
        self.state
    }

    #[must_use]
    pub fn mode(&self) -> RuntimeMode {
        self.mode
    }

    /// SciPy's `LsodaDenseOutput` for the last step at `t`: the Nordsieck history up to the
    /// step's order `q` (NQU), `sum_k yh[:, k] ((t - t_n)/h)^k`, where `h` is the step size the
    /// history is scaled to (the NEXT step's, HCUR) and, when ODEPACK has already lowered the
    /// order for the next step, column `q` (left at the old scale) is first multiplied by
    /// `(h/hu)^q`, all as SciPy computes it. A step that did not integrate (`t0 == t_bound`)
    /// gives the constant `y`. `None` before the first step.
    #[must_use]
    pub fn dense_output_at(&self, t: f64) -> Option<Vec<f64>> {
        let t_old = self.t_old?;
        if self.t == t_old {
            return Some(self.y.clone());
        }
        let order = self.core.nqu();
        let h = self.core.h();
        let top_scale = (self.core.nq() < order).then(|| (h / self.core.hu()).powf(order as f64));
        let x = (t - self.t) / h;
        let mut out = vec![0.0; self.y.len()];
        for k in 0..=order {
            // numpy: ((t - t_n)/h) ** arange(order + 1), elementwise pow.
            let p = x.powf(k as f64);
            let scale = if k == order { top_scale } else { None };
            for (o, &c) in out.iter_mut().zip(self.core.yh_column(k)) {
                let c = scale.map_or(c, |s| c * s);
                *o += c * p;
            }
        }
        Some(out)
    }
}

impl<F> IvpSolver<F> for LsodaSolver
where
    F: FnMut(f64, &[f64]) -> Vec<f64>,
{
    fn step_with(&mut self, fun: &mut F) -> Result<StepOutcome, StepFailure> {
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

    // SciPy's BDF and Radau would use `jac`; these form their Jacobians by differences only.
    if resolved_options.jac.is_some()
        && matches!(resolved_options.method, SolverKind::Bdf | SolverKind::Radau)
    {
        fail_closed(audit, "not_yet_implemented", "rejected");
        return Err(IntegrateValidationError::NotYetImplemented {
            function: "solve_ivp(jac=...) with BDF or Radau",
        });
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
            let solver = LsodaSolver::from_options(&resolved_options)?;
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
            .str(mode)
            .bool(false); // jac: None
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

    fn lsoda_config(t_span: (f64, f64), y0: &[f64], jac: Option<JacFn>) -> LsodaSolverConfig<'_> {
        LsodaSolverConfig {
            t0: t_span.0,
            y0,
            t_bound: t_span.1,
            rtol: 1e-3,
            atol: ToleranceValue::Scalar(1e-6),
            max_step: f64::INFINITY,
            first_step: None,
            jac,
            mode: RuntimeMode::Strict,
        }
    }

    /// A stepped LSODA run: the steps at which the method of the accepted step (ODEPACK's
    /// MUSED) changed, with the new method, and the highest order (NQU) each method reached.
    struct LsodaHistory {
        switches: Vec<(usize, LsodaMethod)>,
        max_adams_order: usize,
        max_bdf_order: usize,
        solver: LsodaSolver,
    }

    fn lsoda_history<F>(fun: &mut F, config: LsodaSolverConfig<'_>) -> LsodaHistory
    where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        let mut solver = crate::LSODA::new(config).expect("LSODA config");
        let mut switches: Vec<(usize, LsodaMethod)> = Vec::new();
        let (mut max_adams_order, mut max_bdf_order) = (0, 0);
        while solver.state() == OdeSolverState::Running {
            solver.step_with(fun).expect("LSODA step");
            let used = solver.method_used().expect("a step was accepted");
            if switches.last().map(|&(_, method)| method) != Some(used) {
                switches.push((solver.n_steps(), used));
            }
            match used {
                LsodaMethod::Adams => max_adams_order = max_adams_order.max(solver.order_used()),
                LsodaMethod::Bdf => max_bdf_order = max_bdf_order.max(solver.order_used()),
            }
        }
        LsodaHistory {
            switches,
            max_adams_order,
            max_bdf_order,
            solver,
        }
    }

    fn vdp1000(_t: f64, y: &[f64]) -> Vec<f64> {
        vec![y[1], 1000.0 * (1.0 - y[0] * y[0]) * y[1] - y[0]]
    }

    fn vdp1000_jac(_t: f64, y: &[f64]) -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 1.0],
            vec![-2000.0 * y[0] * y[1] - 1.0, 1000.0 * (1.0 - y[0] * y[0])],
        ]
    }

    fn stiff_cos_jac(_t: f64, _y: &[f64]) -> Vec<Vec<f64>> {
        vec![vec![-1000.0]]
    }

    /// The switch history SciPy 1.17.1 reports: step by step, `(nst, iwork[18])` from
    /// `LSODA(...)._lsoda_solver._integrator.iwork` after each `step()`, reduced to the steps
    /// where MUSED changed (1 = Adams, 2 = BDF), and max(iwork[13]) per method. The ODEPACK
    /// documentation places MUSED at iwork(19) and NQU at iwork(14), 1-based.
    /// - y' = -1000 (y - cos t), [0, 10]: [(1, 1), (31, 2)], orders Adams <= 4, BDF <= 5.
    /// - van der Pol mu = 1000, [0, 3000]: [(1, 1), (23, 2), (166, 1), (250, 2), (393, 1),
    ///   (477, 2), (620, 1), (704, 2)] over 727 steps, orders Adams <= 4, BDF <= 3.
    /// - y' = -y cos t at rtol 1e-9: [(1, 1)], Adams order up to 9.
    ///
    /// A one-way RK45 -> BDF switch cannot produce the van der Pol row.
    #[test]
    fn lsoda_switch_history_is_scipys() {
        use LsodaMethod::{Adams, Bdf};
        let h = lsoda_history(
            &mut |t: f64, y: &[f64]| vec![-1000.0 * (y[0] - t.cos())],
            lsoda_config((0.0, 10.0), &[0.0], None),
        );
        assert_eq!(h.switches, vec![(1, Adams), (31, Bdf)]);
        assert_eq!((h.max_adams_order, h.max_bdf_order), (4, 5));
        assert_eq!(
            (h.solver.n_steps(), h.solver.nfev(), h.solver.njev()),
            (110, 208, 25)
        );

        let h = lsoda_history(&mut vdp1000, lsoda_config((0.0, 3000.0), &[2.0, 0.0], None));
        assert_eq!(
            h.switches,
            vec![
                (1, Adams),
                (23, Bdf),
                (166, Adams),
                (250, Bdf),
                (393, Adams),
                (477, Bdf),
                (620, Adams),
                (704, Bdf)
            ]
        );
        assert_eq!((h.max_adams_order, h.max_bdf_order), (4, 3));
        assert_eq!(h.solver.n_steps(), 727);

        let h = lsoda_history(
            &mut |t: f64, y: &[f64]| vec![-y[0] * t.cos()],
            LsodaSolverConfig {
                rtol: 1e-9,
                atol: ToleranceValue::Scalar(1e-12),
                ..lsoda_config((0.0, 10.0), &[1.0], None)
            },
        );
        assert_eq!(h.switches, vec![(1, Adams)]);
        assert_eq!((h.max_adams_order, h.max_bdf_order), (9, 0));
    }

    /// `jac` given is ODEPACK's jt = 1: the Jacobian comes from the user, so each one costs no
    /// evaluations (SciPy: stiff scalar nfev 230 with 24 Jacobians; van der Pol nfev 1543 =
    /// 1803 - 2 x 130, the same 727 steps and switch history as with differences).
    #[test]
    fn lsoda_user_jacobian_is_scipys_jt1() {
        let r = solve_ivp(
            &mut |t: f64, y: &[f64]| vec![-1000.0 * (y[0] - t.cos())],
            &SolveIvpOptions {
                jac: Some(stiff_cos_jac),
                ..lsoda_options((0.0, 10.0), &[0.0], 1e-3, 1e-6)
            },
        )
        .expect("LSODA with jac");
        assert_eq!(
            (r.status, r.t.len(), r.nfev, r.njev, r.nlu),
            (0, 121, 230, 24, 24)
        );
        assert_lsoda_bits("y(10)", &r.y[120], &[-0.8396138595348207]);

        let h = lsoda_history(
            &mut vdp1000,
            lsoda_config((0.0, 3000.0), &[2.0, 0.0], Some(vdp1000_jac)),
        );
        assert_eq!(h.switches.len(), 8);
        assert_eq!(
            (h.solver.n_steps(), h.solver.nfev(), h.solver.njev()),
            (727, 1543, 130)
        );
        let scipy = [-1.4974126251531406, 0.0012054067934727682];
        for (got, want) in h.solver.y().iter().zip(scipy) {
            assert!(
                (got - want).abs() <= 1e-9 * want.abs(),
                "y(3000): fsci {got:?}, SciPy {want:?}"
            );
        }
    }

    /// Negative arms: a Jacobian of the wrong shape fails the step (SciPy raises), a `jac` for
    /// BDF or Radau is refused rather than silently replaced by differences, and a finished
    /// solver cannot be stepped.
    #[test]
    fn lsoda_rejects_bad_jacobians_and_finished_steps() {
        fn wrong_shape(_t: f64, _y: &[f64]) -> Vec<Vec<f64>> {
            vec![vec![1.0, 2.0]]
        }
        let mut fun = |t: f64, y: &[f64]| vec![-1000.0 * (y[0] - t.cos())];
        let mut solver =
            crate::LSODA::new(lsoda_config((0.0, 10.0), &[0.0], Some(wrong_shape))).expect("init");
        let failure = loop {
            match solver.step_with(&mut fun) {
                Ok(_) => assert_eq!(solver.method_used(), Some(LsodaMethod::Adams)),
                Err(failure) => break failure,
            }
        };
        assert!(
            matches!(failure, StepFailure::RuntimeError(m) if m.contains("Jacobian")),
            "{failure:?}"
        );
        assert_eq!(solver.state(), OdeSolverState::Failed);

        for method in [SolverKind::Bdf, SolverKind::Radau] {
            let err = solve_ivp(
                &mut fun,
                &SolveIvpOptions {
                    method,
                    jac: Some(stiff_cos_jac),
                    ..lsoda_options((0.0, 1.0), &[0.0], 1e-3, 1e-6)
                },
            )
            .expect_err("jac with BDF/Radau is not implemented");
            assert!(
                matches!(err, IntegrateValidationError::NotYetImplemented { .. }),
                "{err:?}"
            );
        }

        let mut solver = crate::LSODA::new(lsoda_config((0.0, 0.1), &[0.0], None)).expect("init");
        while solver.state() == OdeSolverState::Running {
            solver.step_with(&mut fun).expect("step");
        }
        assert_eq!(
            solver.t().to_bits(),
            0.1_f64.to_bits(),
            "the last step lands on t_bound"
        );
        assert!(solver.step_with(&mut fun).is_err());
    }

    /// `scipy.integrate.LSODA` used to be an empty unit struct (frankenscipy-8dndw.1), then an
    /// RK45-then-BDF stand-in. The public handle must drive exactly the stepper
    /// `solve_ivp(method="LSODA")` runs, and evaluates nothing until its first step.
    #[test]
    fn public_lsoda_handle_steps_like_solve_ivp() {
        let mut calls = 0_usize;
        let mut fun = |t: f64, y: &[f64]| {
            calls += 1;
            vec![-1000.0 * (y[0] - t.cos())]
        };
        let mut solver = crate::LSODA::new(LsodaSolverConfig {
            rtol: 1e-4,
            first_step: Some(1e-6),
            ..lsoda_config((0.0, 0.1), &[1.0], None)
        })
        .expect("LSODA init");
        assert!(solver.dense_output_at(0.0).is_none());
        let mut ts = vec![solver.t()];
        while solver.state() == OdeSolverState::Running {
            solver.step_with(&mut fun).expect("LSODA step");
            ts.push(solver.t());
            let t_old = solver.t_old().expect("a step was taken");
            let at_end = solver.dense_output_at(solver.t()).expect("dense output");
            assert_eq!(at_end[0].to_bits(), solver.y()[0].to_bits());
            let midpoint = solver
                .dense_output_at(0.5 * (t_old + solver.t()))
                .expect("dense output inside the last step");
            assert!(midpoint[0].is_finite());
        }
        assert_eq!(solver.method_used(), Some(LsodaMethod::Bdf));
        assert_eq!(solver.state(), OdeSolverState::Finished);
        assert_eq!(solver.nfev(), calls);

        let reference = solve_ivp(
            &mut |t: f64, y: &[f64]| vec![-1000.0 * (y[0] - t.cos())],
            &SolveIvpOptions {
                first_step: Some(1e-6),
                ..lsoda_options((0.0, 0.1), &[1.0], 1e-4, 1e-6)
            },
        )
        .expect("solve_ivp LSODA");
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

    // ── solve_ivp(method="LSODA") against SciPy 1.17.1, step for step (frankenscipy-1ksfv.9) ──
    //
    // Every number below was printed by SciPy 1.17.1 / numpy 2.4.3 (`solve_ivp(fun, t_span, y0,
    // method="LSODA", ...)`, repr of each float). The right-hand sides are written with the same
    // operations in the same order as the Python lambdas. Where no LAPACK call is involved
    // (Adams-only runs, and any n = 1 problem, whose 1x1 LU is one division) the port does
    // SciPy's arithmetic operation for operation, so the step times and states are asserted
    // bit for bit. The RK45-then-BDF stand-in this port replaced fails every count here.

    fn assert_lsoda_bits(label: &str, got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len(), "{label}: length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "{label}[{i}]: fsci {g:?}, SciPy {w:?}"
            );
        }
    }

    fn lsoda_options(t_span: (f64, f64), y0: &[f64], rtol: f64, atol: f64) -> SolveIvpOptions<'_> {
        SolveIvpOptions {
            t_span,
            y0,
            method: SolverKind::Lsoda,
            rtol,
            atol: ToleranceValue::Scalar(atol),
            ..SolveIvpOptions::default()
        }
    }

    /// y' = -y cos t on [0, 10] at rtol 1e-9: nonstiff, so SciPy stays on Adams the whole way
    /// (MUSED = 1 on all 194 steps) and climbs to order 9. nfev = 416 is 1 initial evaluation
    /// plus the functional-iteration evaluations; no Jacobian is ever formed.
    #[test]
    fn lsoda_adams_cos_decay_is_scipys_step_for_step() {
        let r = solve_ivp(
            &mut |t: f64, y: &[f64]| vec![-y[0] * t.cos()],
            &lsoda_options((0.0, 10.0), &[1.0], 1e-9, 1e-12),
        )
        .expect("LSODA cos decay");
        assert_eq!((r.status, r.success), (0, true), "{}", r.message);
        assert_eq!((r.t.len(), r.nfev, r.njev, r.nlu), (195, 416, 0, 0));
        assert_lsoda_bits(
            "t[..5]",
            &r.t[..5],
            &[
                0.0,
                3.149699260936168e-05,
                6.299398521872336e-05,
                0.006933604330925288,
                0.013804214676631853,
            ],
        );
        assert_lsoda_bits("y(10)", &r.y[194], &[1.7229209951620803]);
    }

    /// Lotka-Volterra on [0, 15] at rtol 1e-8: n = 2 and still Adams-only (order up to 8), so
    /// no LU factorization, and SciPy's 682 steps are reproduced exactly.
    #[test]
    fn lsoda_adams_lotka_volterra_is_scipys_step_for_step() {
        let r = solve_ivp(
            &mut |_t: f64, y: &[f64]| vec![1.5 * y[0] - y[0] * y[1], -3.0 * y[1] + y[0] * y[1]],
            &lsoda_options((0.0, 15.0), &[10.0, 5.0], 1e-8, 1e-10),
        )
        .expect("LSODA Lotka-Volterra");
        assert_eq!((r.status, r.t.len(), r.nfev, r.njev), (0, 683, 1451, 0));
        assert_lsoda_bits(
            "y(15)",
            &r.y[682],
            &[0.7137520493090799, 0.0754077628257764],
        );
    }

    /// The same Adams run integrated backward from t = 10 to 0 (h < 0 throughout).
    #[test]
    fn lsoda_adams_backward_is_scipys_step_for_step() {
        let r = solve_ivp(
            &mut |t: f64, y: &[f64]| vec![-y[0] * t.cos()],
            &lsoda_options((10.0, 0.0), &[1.0], 1e-9, 1e-12),
        )
        .expect("LSODA backward");
        assert_eq!((r.status, r.t.len(), r.nfev, r.njev), (0, 179, 383, 0));
        assert_lsoda_bits(
            "t[..3]",
            &r.t[..3],
            &[10.0, 9.999962540117728, 9.999925080235457],
        );
        assert_lsoda_bits("y(0)", &r.y[178], &[0.5804096642119552]);
    }

    /// y' = -1000 (y - cos t) at the default tolerances: SciPy starts on Adams, switches to BDF
    /// on step 31 and stays there, forming 25 finite-difference Jacobians (one evaluation each,
    /// n = 1). nfev 208, njev = nlu = 25.
    #[test]
    fn lsoda_stiff_scalar_switches_to_bdf_like_scipy() {
        let r = solve_ivp(
            &mut |t: f64, y: &[f64]| vec![-1000.0 * (y[0] - t.cos())],
            &lsoda_options((0.0, 10.0), &[0.0], 1e-3, 1e-6),
        )
        .expect("LSODA stiff scalar");
        assert_eq!(
            (r.status, r.t.len(), r.nfev, r.njev, r.nlu),
            (0, 111, 208, 25, 25)
        );
        assert_lsoda_bits(
            "t[..5]",
            &r.t[..5],
            &[
                0.0,
                3.162277660168363e-08,
                6.324555320336727e-08,
                1.657533971189277e-05,
                3.308743387058218e-05,
            ],
        );
        assert_lsoda_bits("y(10)", &r.y[110], &[-0.8393610876664377]);
    }

    /// first_step and max_step reach ODEPACK as h0 (rwork[4]) and hmax (rwork[5]).
    #[test]
    fn lsoda_first_step_and_max_step_reach_odepack() {
        let r = solve_ivp(
            &mut |t: f64, y: &[f64]| vec![-1000.0 * (y[0] - t.cos())],
            &SolveIvpOptions {
                first_step: Some(1e-4),
                max_step: 0.5,
                ..lsoda_options((0.0, 10.0), &[0.0], 1e-3, 1e-6)
            },
        )
        .expect("LSODA first_step/max_step");
        assert_eq!((r.status, r.t.len(), r.nfev, r.njev), (0, 131, 250, 28));
        assert_lsoda_bits("t[1]", &r.t[1..2], &[1.193142358094778e-06]);
        assert_lsoda_bits("y(10)", &r.y[130], &[-0.8396154663283392]);
    }

    /// t_eval samples come from SciPy's LsodaDenseOutput over the Nordsieck history
    /// (sum_k yh[:, k] ((t - t_n)/h)^k); nfev is unchanged by sampling.
    #[test]
    fn lsoda_t_eval_samples_come_from_the_nordsieck_history() {
        let t_eval = [0.0, 0.001, 0.5, 2.5, 7.25, 10.0];
        let r = solve_ivp(
            &mut |t: f64, y: &[f64]| vec![-1000.0 * (y[0] - t.cos())],
            &SolveIvpOptions {
                t_eval: Some(&t_eval),
                ..lsoda_options((0.0, 10.0), &[0.0], 1e-3, 1e-6)
            },
        )
        .expect("LSODA t_eval");
        let scipy = [
            0.0,
            0.6321647151385436,
            0.878127537595317,
            -0.8006126583631122,
            0.5693233954013428,
            -0.8393610876664377,
        ];
        assert_eq!((r.t.len(), r.nfev), (6, 208));
        for (i, (got, want)) in r.y.iter().map(|y| y[0]).zip(scipy).enumerate() {
            // The sum over the history is numpy's dot (BLAS), whose summation order is not
            // pinned; allow a few ulps.
            assert!(
                (got - want).abs() <= 4.0 * f64::EPSILON * want.abs(),
                "sample {i} at t = {}: fsci {got:?}, SciPy {want:?}",
                t_eval[i]
            );
        }

        let t_eval = [0.3, 1.7, 4.4, 9.9];
        let r = solve_ivp(
            &mut |t: f64, y: &[f64]| vec![-y[0] * t.cos()],
            &SolveIvpOptions {
                t_eval: Some(&t_eval),
                ..lsoda_options((0.0, 10.0), &[1.0], 1e-9, 1e-12)
            },
        )
        .expect("LSODA t_eval Adams");
        let scipy = [
            0.7441443748121572,
            0.370958598381063,
            2.5898554457115757,
            1.580175452023483,
        ];
        assert_eq!((r.t.len(), r.nfev), (4, 416));
        for (i, (got, want)) in r.y.iter().map(|y| y[0]).zip(scipy).enumerate() {
            assert!(
                (got - want).abs() <= 4.0 * f64::EPSILON * want.abs(),
                "sample {i} at t = {}: fsci {got:?}, SciPy {want:?}",
                t_eval[i]
            );
        }
    }

    /// Van der Pol, mu = 1000, on [0, 3000] at the default tolerances: SciPy switches Adams ->
    /// BDF -> Adams ... seven times (see `lsoda_van_der_pol_switch_history_is_scipys`), taking
    /// 727 steps with 130 finite-difference Jacobians (two evaluations each).
    #[test]
    fn lsoda_van_der_pol_1000_counts_are_scipys() {
        let r = solve_ivp(
            &mut |_t: f64, y: &[f64]| vec![y[1], 1000.0 * (1.0 - y[0] * y[0]) * y[1] - y[0]],
            &lsoda_options((0.0, 3000.0), &[2.0, 0.0], 1e-3, 1e-6),
        )
        .expect("LSODA van der Pol");
        assert_eq!(
            (r.status, r.t.len(), r.nfev, r.njev, r.nlu),
            (0, 728, 1803, 130, 130)
        );
        let scipy = [-1.49741262484335, 0.001205406794123583];
        for (got, want) in r.y[727].iter().zip(scipy) {
            assert!(
                (got - want).abs() <= 1e-9 * want.abs(),
                "y(3000): fsci {got:?}, SciPy {want:?}"
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
