#![forbid(unsafe_code)]

use fsci_opt::root::brentq;
use fsci_opt::types::RootOptions;
use fsci_runtime::RuntimeMode;

use crate::IntegrateValidationError;
use crate::bdf::{BdfSolver, BdfSolverConfig};
use crate::rk::{RK23_TABLEAU, RK45_TABLEAU, RkSolver, RkSolverConfig};
use crate::solver::{OdeSolver, OdeSolverState, StepFailure, StepOutcome};
use crate::validation::{
    SyncSharedAuditLedger, ToleranceValue, audit_fingerprint, record_fail_closed,
    validate_first_step_with_audit, validate_max_step_with_audit, validate_tol_with_audit,
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
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<(), IntegrateValidationError> {
    let fingerprint = audit_fingerprint(
        "validate_t_eval",
        format!("t_eval={t_eval:?};t0={t0};tf={tf}"),
    );
    let t_min = t0.min(tf);
    let t_max = t0.max(tf);
    if t_eval.iter().any(|&te| te < t_min || te > t_max) {
        record_fail_closed(audit_ledger, &fingerprint, "t_eval_out_of_span", "rejected");
        return Err(IntegrateValidationError::TEvalOutOfSpan);
    }

    let is_sorted = if tf >= t0 {
        t_eval.windows(2).all(|window| window[1] > window[0])
    } else {
        t_eval.windows(2).all(|window| window[1] < window[0])
    };
    if !is_sorted {
        record_fail_closed(audit_ledger, &fingerprint, "t_eval_not_sorted", "rejected");
        return Err(IntegrateValidationError::TEvalNotSorted);
    }

    Ok(())
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

fn solve_event_equation(
    event_fn: EventFn,
    t_old: f64,
    t_new: f64,
    y_old: &[f64],
    y_new: &[f64],
    f_old: &[f64],
    f_new: &[f64],
) -> f64 {
    let f = |t: f64| {
        let y = interpolate_state(y_old, y_new, f_old, f_new, t_old, t_new, t);
        event_fn(t, &y)
    };

    let options = RootOptions {
        xtol: 1e-12,
        maxiter: 100,
        ..Default::default()
    };

    match brentq(f, (t_old, t_new), options) {
        Ok(res) => res.root,
        Err(_) => 0.5 * (t_old + t_new),
    }
}

/// Internal trait to unify RK and BDF solver loops.
trait IvpSolver<F> {
    fn step_with(&mut self, fun: &mut F) -> Result<StepOutcome, crate::solver::StepFailure>;
    fn t(&self) -> f64;
    fn y(&self) -> &[f64];
    fn f(&self) -> &[f64];
    fn t_old(&self) -> Option<f64>;
    fn y_old(&self) -> Option<&[f64]>;
    fn f_old(&self) -> Option<&[f64]>;
    fn nfev(&self) -> usize;
    fn njev(&self) -> usize;
    fn nlu(&self) -> usize;
    fn ivp_state(&self) -> OdeSolverState;
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
    fn f(&self) -> &[f64] {
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
        0
    }
    fn nlu(&self) -> usize {
        0
    }
    fn ivp_state(&self) -> OdeSolverState {
        self.state()
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
    fn f(&self) -> &[f64] {
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
}

enum LsodaMode {
    Adams(RkSolver),
    Bdf(BdfSolver),
}

struct LsodaSolver {
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
    fn new<F>(fun: &mut F, options: &SolveIvpOptions<'_>) -> Result<Self, IntegrateValidationError>
    where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        let config = RkSolverConfig {
            t0: options.t_span.0,
            y0: options.y0,
            t_bound: options.t_span.1,
            rtol: options.rtol,
            atol: options.atol.clone(),
            max_step: options.max_step,
            first_step: options.first_step,
            mode: options.mode,
            tableau: &RK45_TABLEAU,
        };
        let solver = RkSolver::new(fun, config)?;
        Ok(Self {
            mode: LsodaMode::Adams(solver),
            t_bound: options.t_span.1,
            rtol: options.rtol,
            atol: options.atol.clone(),
            max_step: options.max_step,
            first_step: options.first_step,
            runtime_mode: options.mode,
            pending_bdf_switch: false,
            nfev_offset: 0,
        })
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

impl<F> IvpSolver<F> for LsodaSolver
where
    F: FnMut(f64, &[f64]) -> Vec<f64>,
{
    fn step_with(&mut self, fun: &mut F) -> Result<StepOutcome, crate::solver::StepFailure> {
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

    fn t(&self) -> f64 {
        match &self.mode {
            LsodaMode::Adams(rk) => rk.t(),
            LsodaMode::Bdf(bdf) => bdf.t(),
        }
    }

    fn y(&self) -> &[f64] {
        match &self.mode {
            LsodaMode::Adams(rk) => rk.y(),
            LsodaMode::Bdf(bdf) => bdf.y(),
        }
    }

    fn f(&self) -> &[f64] {
        match &self.mode {
            LsodaMode::Adams(rk) => rk.f(),
            LsodaMode::Bdf(bdf) => bdf.f(),
        }
    }

    fn t_old(&self) -> Option<f64> {
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
            LsodaMode::Bdf(bdf) => bdf.f_old(),
        }
    }

    fn nfev(&self) -> usize {
        self.nfev_offset
            + match &self.mode {
                LsodaMode::Adams(rk) => rk.nfev(),
                LsodaMode::Bdf(bdf) => bdf.nfev(),
            }
    }

    fn njev(&self) -> usize {
        match &self.mode {
            LsodaMode::Adams(_) => 0,
            LsodaMode::Bdf(bdf) => bdf.njev(),
        }
    }

    fn nlu(&self) -> usize {
        match &self.mode {
            LsodaMode::Adams(_) => 0,
            LsodaMode::Bdf(bdf) => bdf.nlu(),
        }
    }

    fn ivp_state(&self) -> OdeSolverState {
        match &self.mode {
            LsodaMode::Adams(rk) => rk.state(),
            LsodaMode::Bdf(bdf) => bdf.state(),
        }
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
    let mut event_vals = options.events.as_ref().map(|evs| {
        evs.iter()
            .map(|ev| (ev.func)(t0, options.y0))
            .collect::<Vec<_>>()
    });

    if let Some(t_eval) = options.t_eval {
        if matches!(t_eval.first(), Some(&first) if (first - t0).abs() < 1e-14) {
            ts.push(t0);
            ys.push(options.y0.to_vec());
            next_t_eval_index = 1;
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
                let f = solver.f().to_vec();
                let t_old = solver.t_old().unwrap_or(t0);
                let y_old = solver.y_old().unwrap_or(options.y0).to_vec();
                let f_old = solver.f_old().unwrap_or(&f).to_vec();

                let mut terminal_event: Option<(f64, Vec<f64>)> = None;

                if let (Some(evs), Some(old_vals), Some(counts)) = (
                    options.events.as_ref(),
                    event_vals.as_mut(),
                    event_counts.as_mut(),
                ) {
                    for (i, ev_spec) in evs.iter().enumerate() {
                        let val = (ev_spec.func)(t, &y);
                        if event_active(old_vals[i], val, ev_spec.direction) {
                            let t_ev = solve_event_equation(
                                ev_spec.func,
                                t_old,
                                t,
                                &y_old,
                                &y,
                                &f_old,
                                &f,
                            );
                            let y_ev = interpolate_state(&y_old, &y, &f_old, &f, t_old, t, t_ev);

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
                            ys.push(interpolate_state(&y_old, &y, &f_old, &f, t_old, t, te));
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
                        ys.push(interpolate_state(&y_old, &y, &f_old, &f, t_old, t, te));
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
    solve_ivp_impl(fun, options, Some(audit_ledger))
}

fn solve_ivp_impl<F>(
    fun: &mut F,
    options: &SolveIvpOptions<'_>,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<SolveIvpResult, IntegrateValidationError>
where
    F: FnMut(f64, &[f64]) -> Vec<f64>,
{
    let (t0, tf) = options.t_span;
    let n = options.y0.len();
    if n == 0 {
        let fingerprint = audit_fingerprint(
            "solve_ivp",
            format!("reason=empty_y0;t_span=({t0},{tf});mode={:?}", options.mode),
        );
        record_fail_closed(audit_ledger, &fingerprint, "empty_y0", "rejected");
        return Err(IntegrateValidationError::EmptyY0);
    }
    if !t0.is_finite() || !tf.is_finite() {
        let fingerprint = audit_fingerprint(
            "solve_ivp",
            format!(
                "reason=non_finite_span;t_span=({t0},{tf});mode={:?}",
                options.mode
            ),
        );
        record_fail_closed(audit_ledger, &fingerprint, "non_finite_span", "rejected");
        return Err(IntegrateValidationError::NonFiniteSpan);
    }
    if options.y0.iter().any(|v| !v.is_finite()) {
        let fingerprint = audit_fingerprint(
            "solve_ivp",
            format!(
                "reason=non_finite_y0;y0={:?};mode={:?}",
                options.y0, options.mode
            ),
        );
        record_fail_closed(audit_ledger, &fingerprint, "non_finite_y0", "rejected");
        return Err(IntegrateValidationError::NonFiniteY0);
    }

    let max_step = validate_max_step_with_audit(options.max_step, audit_ledger)?;
    let first_step = options
        .first_step
        .map(|value| validate_first_step_with_audit(value, t0, tf, audit_ledger))
        .transpose()?;
    let validated_tol = validate_tol_with_audit(
        ToleranceValue::Scalar(options.rtol),
        options.atol.clone(),
        n,
        options.mode,
        audit_ledger,
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

    if let Some(t_eval) = resolved_options.t_eval {
        validate_t_eval_with_audit(t_eval, t0, tf, audit_ledger)?;
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
        SolverKind::Radau | SolverKind::Bdf => {
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
            let solver = LsodaSolver::new(fun, &resolved_options)?;
            solve_ivp_core(fun, solver, &resolved_options)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsci_runtime::AuditAction;

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
        let mut solver = LsodaSolver::new(&mut fun, &options).expect("LSODA init");

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
        // exp(-50 * t): t=0 -> 1.0, t=0.5 -> 1.93e-11, t=1.0 -> 1.93e-22
        // BDF solver has numerical limits - verify exponential decay is correct order of magnitude
        assert!(
            (result.y[0][0] - 1.0).abs() < 1e-6,
            "y[0] = {}",
            result.y[0][0]
        );
        assert!(
            result.y[1][0].abs() < 1e-6,
            "y[0.5] should be near zero: {}",
            result.y[1][0]
        );
        assert!(
            result.y[2][0].abs() < 1e-10,
            "y[1.0] should be near zero: {}",
            result.y[2][0]
        );
    }
}
