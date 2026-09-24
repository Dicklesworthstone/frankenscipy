#![forbid(unsafe_code)]

use std::env;
use std::time::{Duration, Instant};

use fsci_opt::{
    ConvergenceStatus, MinimizeOptions, OptError, OptimizeMethod, OptimizeResult, cg_pr_plus,
};
use fsci_runtime::RuntimeMode;

const DIM: usize = 10;
const F64_BYTES: usize = std::mem::size_of::<f64>();

#[derive(Clone, Copy)]
struct WolfeParams {
    c1: f64,
    c2: f64,
    amax: f64,
    amin: f64,
    maxiter: usize,
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

#[derive(Default, Clone)]
struct AllocationTraffic {
    vec_allocations: usize,
    vec_elements: usize,
}

impl AllocationTraffic {
    fn record_vec(&mut self, len: usize) {
        self.vec_allocations += 1;
        self.vec_elements += len;
    }
}

#[derive(Default, Clone)]
struct StageProfile {
    runs: usize,
    total_runtime: Duration,
    objective_actual_calls: usize,
    objective_reserved_calls: usize,
    initial_value_calls: usize,
    finite_diff_gradient_calls: usize,
    finite_diff_objective_calls: usize,
    finite_diff_runtime: Duration,
    wolfe_calls: usize,
    wolfe_runtime: Duration,
    wolfe_probe_evaluations: usize,
    wolfe_value_calls: usize,
    wolfe_value_runtime: Duration,
    wolfe_gradient_calls: usize,
    wolfe_gradient_runtime: Duration,
    wolfe_directional_derivative_bits_xor: u64,
    zoom_calls: usize,
    zoom_iterations: usize,
    accepted_x_materializations: usize,
    accepted_x_elements: usize,
    accepted_x_runtime: Duration,
    direction_update_runtime: Duration,
    armijo_fallback_calls: usize,
    armijo_value_calls: usize,
    allocation_traffic: AllocationTraffic,
}

impl StageProfile {
    fn add(&mut self, other: &Self) {
        self.runs += other.runs;
        self.total_runtime += other.total_runtime;
        self.objective_actual_calls += other.objective_actual_calls;
        self.objective_reserved_calls += other.objective_reserved_calls;
        self.initial_value_calls += other.initial_value_calls;
        self.finite_diff_gradient_calls += other.finite_diff_gradient_calls;
        self.finite_diff_objective_calls += other.finite_diff_objective_calls;
        self.finite_diff_runtime += other.finite_diff_runtime;
        self.wolfe_calls += other.wolfe_calls;
        self.wolfe_runtime += other.wolfe_runtime;
        self.wolfe_probe_evaluations += other.wolfe_probe_evaluations;
        self.wolfe_value_calls += other.wolfe_value_calls;
        self.wolfe_value_runtime += other.wolfe_value_runtime;
        self.wolfe_gradient_calls += other.wolfe_gradient_calls;
        self.wolfe_gradient_runtime += other.wolfe_gradient_runtime;
        self.wolfe_directional_derivative_bits_xor ^= other.wolfe_directional_derivative_bits_xor;
        self.zoom_calls += other.zoom_calls;
        self.zoom_iterations += other.zoom_iterations;
        self.accepted_x_materializations += other.accepted_x_materializations;
        self.accepted_x_elements += other.accepted_x_elements;
        self.accepted_x_runtime += other.accepted_x_runtime;
        self.direction_update_runtime += other.direction_update_runtime;
        self.armijo_fallback_calls += other.armijo_fallback_calls;
        self.armijo_value_calls += other.armijo_value_calls;
        self.allocation_traffic.vec_allocations += other.allocation_traffic.vec_allocations;
        self.allocation_traffic.vec_elements += other.allocation_traffic.vec_elements;
    }

    fn record_vec(&mut self, len: usize) {
        self.allocation_traffic.record_vec(len);
    }

    fn record_objective(&mut self, stage: ObjectiveStage) {
        self.objective_actual_calls += 1;
        match stage {
            ObjectiveStage::InitialValue => self.initial_value_calls += 1,
            ObjectiveStage::FiniteDiff => self.finite_diff_objective_calls += 1,
            ObjectiveStage::WolfeValue => self.wolfe_value_calls += 1,
            ObjectiveStage::WolfeGradient => self.finite_diff_objective_calls += 1,
            ObjectiveStage::Armijo => self.armijo_value_calls += 1,
        }
    }

    fn record_reserved_objectives(&mut self, count: usize) {
        self.objective_reserved_calls += count;
    }
}

#[derive(Clone, Copy)]
enum ObjectiveStage {
    InitialValue,
    FiniteDiff,
    WolfeValue,
    WolfeGradient,
    Armijo,
}

struct Objective<'a, F>
where
    F: Fn(&[f64]) -> f64,
{
    fun: &'a F,
    mode: RuntimeMode,
    maxfev: usize,
    nfev: usize,
}

impl<'a, F> Objective<'a, F>
where
    F: Fn(&[f64]) -> f64,
{
    fn new(fun: &'a F, mode: RuntimeMode, maxfev: usize) -> Self {
        Self {
            fun,
            mode,
            maxfev: maxfev.max(1),
            nfev: 0,
        }
    }

    fn eval(
        &mut self,
        x: &[f64],
        profile: &mut StageProfile,
        stage: ObjectiveStage,
    ) -> Result<f64, OptError> {
        if self.nfev >= self.maxfev {
            return Err(OptError::EvaluationBudgetExceeded {
                detail: format!("max function evaluations exceeded ({})", self.maxfev),
            });
        }
        let value = (self.fun)(x);
        self.nfev += 1;
        profile.record_objective(stage);
        if !value.is_finite() {
            return match self.mode {
                RuntimeMode::Strict => Err(OptError::InvalidArgument {
                    detail: String::from("objective evaluated to non-finite value"),
                }),
                RuntimeMode::Hardened => Err(OptError::NonFiniteInput {
                    detail: String::from("hardened mode rejects non-finite objective values"),
                }),
            };
        }
        Ok(value)
    }

    fn reserve_evaluations(
        &mut self,
        count: usize,
        profile: &mut StageProfile,
    ) -> Result<(), OptError> {
        let remaining = self.maxfev.saturating_sub(self.nfev);
        if remaining < count {
            self.nfev = self.maxfev;
            return Err(OptError::EvaluationBudgetExceeded {
                detail: format!("max function evaluations exceeded ({})", self.maxfev),
            });
        }
        self.nfev += count;
        profile.record_reserved_objectives(count);
        Ok(())
    }
}

struct LineSearchStep {
    alpha: f64,
    x: Vec<f64>,
    f: f64,
    accepted_gradient: Option<Vec<f64>>,
}

struct WolfeProbeResult {
    alpha: f64,
    f_at_alpha: f64,
    directional_derivative: f64,
    evaluations: usize,
    accepted_gradient: Option<Vec<f64>>,
    actual_objective_calls: usize,
}

fn rosenbrock_nd(x: &[f64]) -> f64 {
    let mut acc = 0.0;
    for i in 0..x.len() - 1 {
        let a = 1.0 - x[i];
        let b = x[i + 1] - x[i] * x[i];
        acc += a * a + 100.0 * b * b;
    }
    acc
}

fn options() -> MinimizeOptions<'static> {
    MinimizeOptions {
        method: Some(OptimizeMethod::ConjugateGradient),
        mode: RuntimeMode::Strict,
        ..MinimizeOptions::default()
    }
}

fn profiled_cg_pr_plus<F>(
    fun: &F,
    x0: &[f64],
    options: MinimizeOptions,
    profile: &mut StageProfile,
) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    let tol = options.tol.unwrap_or(1.0e-6).max(1.0e-12);
    let maxiter = options.maxiter.unwrap_or((250 * n).max(120));
    let maxfev = options.maxfev.unwrap_or((2500 * n).max(500));
    // cg_pr_plus's default step (central differences).
    let gradient_eps = options.gradient_eps.unwrap_or(1.0e-8);
    let mut objective = Objective::new(fun, options.mode, maxfev);

    profile.record_vec(x0.len());
    let mut x = x0.to_vec();
    let mut f = match objective.eval(&x, profile, ObjectiveStage::InitialValue) {
        Ok(value) => value,
        Err(err) => return Ok(result_from_error(x0, 0, 0, 0, err, profile)),
    };
    let mut njev = 0usize;
    let mut grad = match finite_diff_gradient(&mut objective, &x, gradient_eps, profile) {
        Ok(value) => {
            njev += 1;
            value
        }
        Err(err) => return Ok(result_from_error(&x, 0, objective.nfev, njev, err, profile)),
    };
    let mut direction = scale_vector(&grad, -1.0, profile);
    let mut nit = 0usize;

    for iteration in 0..maxiter {
        let grad_norm = l2_norm(&grad);
        if grad_norm <= tol {
            let result = OptimizeResult {
                x: clone_vec(&x, profile),
                fun: Some(f),
                success: true,
                status: ConvergenceStatus::Success,
                message: String::from("optimization converged (gradient norm <= tol)"),
                nfev: objective.nfev,
                njev,
                nhev: 0,
                nit,
                jac: Some(clone_vec(&grad, profile)),
                hess_inv: None,
                maxcv: None,
            };
            return Ok(result);
        }

        if let Some(callback) = options.callback
            && !callback(&x)
        {
            let result = OptimizeResult {
                x: clone_vec(&x, profile),
                fun: Some(f),
                success: false,
                status: ConvergenceStatus::CallbackStop,
                message: String::from("callback requested stop"),
                nfev: objective.nfev,
                njev,
                nhev: 0,
                nit,
                jac: Some(clone_vec(&grad, profile)),
                hess_inv: None,
                maxcv: None,
            };
            return Ok(result);
        }

        if dot(&direction, &grad) >= 0.0 {
            direction = scale_vector(&grad, -1.0, profile);
        }

        let wolfe_search = {
            let started = Instant::now();
            let res =
                line_search_wolfe2_profiled(fun, &x, &direction, f, &grad, gradient_eps, profile);
            profile.wolfe_runtime += started.elapsed();
            if let Ok(ls) = res {
                objective.nfev += ls.actual_objective_calls;
                profile.wolfe_probe_evaluations += ls.evaluations;
                profile.wolfe_directional_derivative_bits_xor ^=
                    ls.directional_derivative.to_bits();
                let materialize_started = Instant::now();
                let accepted_x = add_scaled(&x, &direction, ls.alpha, profile);
                profile.accepted_x_runtime += materialize_started.elapsed();
                profile.accepted_x_materializations += 1;
                profile.accepted_x_elements += accepted_x.len();
                Some(LineSearchStep {
                    x: accepted_x,
                    f: ls.f_at_alpha,
                    alpha: ls.alpha,
                    accepted_gradient: ls.accepted_gradient,
                })
            } else {
                None
            }
        };

        let search = match wolfe_search {
            Some(value) => value,
            None => match armijo_backtracking(&mut objective, &x, f, &grad, &direction, profile) {
                Ok(Some(value)) => value,
                Ok(None) => {
                    let result = OptimizeResult {
                        x: clone_vec(&x, profile),
                        fun: Some(f),
                        success: false,
                        status: ConvergenceStatus::PrecisionLoss,
                        message: String::from("line search failed to find a sufficient decrease"),
                        nfev: objective.nfev,
                        njev,
                        nhev: 0,
                        nit,
                        jac: Some(clone_vec(&grad, profile)),
                        hess_inv: None,
                        maxcv: None,
                    };
                    return Ok(result);
                }
                Err(err) => {
                    return Ok(result_from_error(
                        &x,
                        nit,
                        objective.nfev,
                        njev,
                        err,
                        profile,
                    ));
                }
            },
        };

        let next_grad = match search.accepted_gradient {
            Some(accepted_gradient) => {
                if let Err(err) = objective.reserve_evaluations(n * 2, profile) {
                    return Ok(result_from_error(
                        &search.x,
                        nit,
                        objective.nfev,
                        njev,
                        err,
                        profile,
                    ));
                }
                njev += 1;
                accepted_gradient
            }
            None => match finite_diff_gradient(&mut objective, &search.x, gradient_eps, profile) {
                Ok(value) => {
                    njev += 1;
                    value
                }
                Err(err) => {
                    return Ok(result_from_error(
                        &search.x,
                        nit,
                        objective.nfev,
                        njev,
                        err,
                        profile,
                    ));
                }
            },
        };

        let direction_started = Instant::now();
        let denom = dot(&grad, &grad).max(1.0e-18);
        let grad_delta = sub_vectors(&next_grad, &grad, profile);
        let beta_pr = dot(&next_grad, &grad_delta) / denom;
        let beta = beta_pr.max(0.0);
        direction = sub_vectors(
            &scale_vector(&next_grad, -1.0, profile),
            &scale_vector(&direction, -beta, profile),
            profile,
        );
        profile.direction_update_runtime += direction_started.elapsed();

        x = search.x;
        f = search.f;
        grad = next_grad;
        nit = iteration + 1;
        let _alpha = search.alpha;
    }

    let result = OptimizeResult {
        x: clone_vec(&x, profile),
        fun: Some(f),
        success: false,
        status: ConvergenceStatus::MaxIterations,
        message: String::from("maximum iterations exceeded"),
        nfev: objective.nfev,
        njev,
        nhev: 0,
        nit,
        jac: Some(grad),
        hess_inv: None,
        maxcv: None,
    };
    Ok(result)
}

fn finite_diff_gradient<F>(
    objective: &mut Objective<'_, F>,
    x: &[f64],
    gradient_eps: f64,
    profile: &mut StageProfile,
) -> Result<Vec<f64>, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    profile.finite_diff_gradient_calls += 1;
    let started = Instant::now();
    profile.record_vec(x.len());
    let mut gradient = vec![0.0; x.len()];
    profile.record_vec(x.len());
    let mut x_perturbed = x.to_vec();
    for (idx, component) in x.iter().copied().enumerate() {
        let step = gradient_eps * (1.0 + component.abs());

        let original = x_perturbed[idx];
        x_perturbed[idx] = original + step;
        let f_plus = objective.eval(&x_perturbed, profile, ObjectiveStage::FiniteDiff)?;

        x_perturbed[idx] = original - step;
        let f_minus = objective.eval(&x_perturbed, profile, ObjectiveStage::FiniteDiff)?;

        x_perturbed[idx] = original;
        gradient[idx] = (f_plus - f_minus) / (2.0 * step);
    }
    profile.finite_diff_runtime += started.elapsed();
    Ok(gradient)
}

fn line_search_wolfe2_profiled<F>(
    fun: &F,
    x: &[f64],
    d: &[f64],
    f0: f64,
    g0: &[f64],
    gradient_eps: f64,
    profile: &mut StageProfile,
) -> Result<WolfeProbeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    let params = WolfeParams::default();
    let dg0 = dot(g0, d);
    if dg0 >= 0.0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("search direction is not a descent direction"),
        });
    }

    profile.wolfe_calls += 1;
    profile.record_vec(x.len());
    let mut trial = vec![0.0; x.len()];
    profile.record_vec(x.len());
    let mut gradient = vec![0.0; x.len()];
    let mut evals = 0usize;
    let mut actual_objective_calls = 0usize;

    let mut alpha_prev = 0.0;
    let mut f_prev = f0;
    let mut alpha = 1.0_f64.min(params.amax);

    for i in 0..params.maxiter {
        let fi = eval_f_at(
            fun,
            x,
            d,
            &mut trial,
            alpha,
            &mut evals,
            &mut actual_objective_calls,
            profile,
        );

        if fi > f0 + params.c1 * alpha * dg0 || (i > 0 && fi >= f_prev) {
            return zoom_profiled(
                fun,
                x,
                d,
                f0,
                dg0,
                alpha_prev,
                alpha,
                f_prev,
                &params,
                &mut evals,
                &mut actual_objective_calls,
                &mut trial,
                &mut gradient,
                gradient_eps,
                profile,
            );
        }

        let dgi = eval_dg_current(
            fun,
            d,
            &mut trial,
            &mut gradient,
            &mut evals,
            &mut actual_objective_calls,
            gradient_eps,
            profile,
        );

        if dgi.abs() <= params.c2 * dg0.abs() {
            return Ok(WolfeProbeResult {
                alpha,
                f_at_alpha: fi,
                directional_derivative: dgi,
                evaluations: evals,
                accepted_gradient: Some(take_vec(&mut gradient)),
                actual_objective_calls,
            });
        }

        if dgi >= 0.0 {
            return zoom_profiled(
                fun,
                x,
                d,
                f0,
                dg0,
                alpha,
                alpha_prev,
                fi,
                &params,
                &mut evals,
                &mut actual_objective_calls,
                &mut trial,
                &mut gradient,
                gradient_eps,
                profile,
            );
        }

        alpha_prev = alpha;
        f_prev = fi;
        alpha = (2.0 * alpha).min(params.amax);
    }

    let fi = eval_f_at(
        fun,
        x,
        d,
        &mut trial,
        alpha,
        &mut evals,
        &mut actual_objective_calls,
        profile,
    );
    Ok(WolfeProbeResult {
        alpha,
        f_at_alpha: fi,
        directional_derivative: dg0,
        evaluations: evals,
        accepted_gradient: None,
        actual_objective_calls,
    })
}

#[allow(clippy::too_many_arguments)]
fn zoom_profiled<F>(
    fun: &F,
    x: &[f64],
    d: &[f64],
    f0: f64,
    dg0: f64,
    mut alpha_lo: f64,
    mut alpha_hi: f64,
    mut f_lo: f64,
    params: &WolfeParams,
    evals: &mut usize,
    actual_objective_calls: &mut usize,
    trial: &mut [f64],
    gradient: &mut Vec<f64>,
    gradient_eps: f64,
    profile: &mut StageProfile,
) -> Result<WolfeProbeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    profile.zoom_calls += 1;
    for _ in 0..params.maxiter {
        profile.zoom_iterations += 1;
        let alpha_j = 0.5 * (alpha_lo + alpha_hi);
        let fj = eval_f_at(
            fun,
            x,
            d,
            trial,
            alpha_j,
            evals,
            actual_objective_calls,
            profile,
        );

        if fj > f0 + params.c1 * alpha_j * dg0 || fj >= f_lo {
            alpha_hi = alpha_j;
        } else {
            let dgj = eval_dg_current(
                fun,
                d,
                trial,
                gradient,
                evals,
                actual_objective_calls,
                gradient_eps,
                profile,
            );

            if dgj.abs() <= params.c2 * dg0.abs() {
                return Ok(WolfeProbeResult {
                    alpha: alpha_j,
                    f_at_alpha: fj,
                    directional_derivative: dgj,
                    evaluations: *evals,
                    accepted_gradient: Some(take_vec(gradient)),
                    actual_objective_calls: *actual_objective_calls,
                });
            }

            if dgj * (alpha_hi - alpha_lo) >= 0.0 {
                alpha_hi = alpha_lo;
            }

            alpha_lo = alpha_j;
            f_lo = fj;
        }

        if (alpha_hi - alpha_lo).abs() < params.amin {
            break;
        }
    }

    Ok(WolfeProbeResult {
        alpha: alpha_lo,
        f_at_alpha: f_lo,
        directional_derivative: dg0,
        evaluations: *evals,
        accepted_gradient: None,
        actual_objective_calls: *actual_objective_calls,
    })
}

#[allow(clippy::too_many_arguments)]
fn eval_f_at<F>(
    fun: &F,
    x: &[f64],
    d: &[f64],
    trial: &mut [f64],
    alpha: f64,
    evals: &mut usize,
    actual_objective_calls: &mut usize,
    profile: &mut StageProfile,
) -> f64
where
    F: Fn(&[f64]) -> f64,
{
    fill_trial(trial, x, d, alpha);
    *evals += 1;
    *actual_objective_calls += 1;
    profile.record_objective(ObjectiveStage::WolfeValue);
    let started = Instant::now();
    let value = fun(trial);
    profile.wolfe_value_runtime += started.elapsed();
    value
}

#[allow(clippy::too_many_arguments)]
fn eval_dg_current<F>(
    fun: &F,
    direction: &[f64],
    trial: &mut [f64],
    gradient: &mut [f64],
    evals: &mut usize,
    actual_objective_calls: &mut usize,
    gradient_eps: f64,
    profile: &mut StageProfile,
) -> f64
where
    F: Fn(&[f64]) -> f64,
{
    profile.wolfe_gradient_calls += 1;
    *evals += 1;
    let started = Instant::now();
    for idx in 0..trial.len() {
        let step = gradient_eps * (1.0 + trial[idx].abs());
        let original = trial[idx];
        trial[idx] = original + step;
        *actual_objective_calls += 1;
        profile.record_objective(ObjectiveStage::WolfeGradient);
        let f_plus = fun(trial);
        trial[idx] = original - step;
        *actual_objective_calls += 1;
        profile.record_objective(ObjectiveStage::WolfeGradient);
        let f_minus = fun(trial);
        trial[idx] = original;
        gradient[idx] = (f_plus - f_minus) / (2.0 * step);
    }
    profile.wolfe_gradient_runtime += started.elapsed();
    dot(gradient, direction)
}

fn armijo_backtracking<F>(
    objective: &mut Objective<'_, F>,
    x: &[f64],
    fx: f64,
    grad: &[f64],
    direction: &[f64],
    profile: &mut StageProfile,
) -> Result<Option<LineSearchStep>, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    profile.armijo_fallback_calls += 1;
    let directional_derivative = dot(grad, direction);
    if directional_derivative >= 0.0 {
        return Ok(None);
    }

    let c1 = 1.0e-4;
    let mut alpha = 1.0;
    for _ in 0..24 {
        let candidate_x = add_scaled(x, direction, alpha, profile);
        let candidate_f = objective.eval(&candidate_x, profile, ObjectiveStage::Armijo)?;
        if candidate_f <= fx + c1 * alpha * directional_derivative {
            return Ok(Some(LineSearchStep {
                alpha,
                x: candidate_x,
                f: candidate_f,
                accepted_gradient: None,
            }));
        }
        alpha *= 0.5;
        if alpha < 1.0e-12 {
            break;
        }
    }
    Ok(None)
}

fn fill_trial(out: &mut [f64], x: &[f64], d: &[f64], alpha: f64) {
    for ((out_value, xi), di) in out.iter_mut().zip(x.iter()).zip(d.iter()) {
        *out_value = xi + alpha * di;
    }
}

fn add_scaled(a: &[f64], b: &[f64], alpha: f64, profile: &mut StageProfile) -> Vec<f64> {
    profile.record_vec(a.len());
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| ai + alpha * bi)
        .collect()
}

fn scale_vector(v: &[f64], alpha: f64, profile: &mut StageProfile) -> Vec<f64> {
    profile.record_vec(v.len());
    v.iter().map(|value| alpha * value).collect()
}

fn sub_vectors(a: &[f64], b: &[f64], profile: &mut StageProfile) -> Vec<f64> {
    profile.record_vec(a.len());
    a.iter().zip(b.iter()).map(|(ai, bi)| ai - bi).collect()
}

fn clone_vec(values: &[f64], profile: &mut StageProfile) -> Vec<f64> {
    profile.record_vec(values.len());
    values.to_vec()
}

fn take_vec(values: &mut Vec<f64>) -> Vec<f64> {
    std::mem::take(values)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

fn l2_norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

fn result_from_error(
    x: &[f64],
    nit: usize,
    nfev: usize,
    njev: usize,
    error: OptError,
    profile: &mut StageProfile,
) -> OptimizeResult {
    let (status, message) = match error {
        OptError::EvaluationBudgetExceeded { detail } => {
            (ConvergenceStatus::MaxEvaluations, detail)
        }
        OptError::NonFiniteInput { detail } => (ConvergenceStatus::NanEncountered, detail),
        OptError::InvalidArgument { detail } | OptError::InvalidBounds { detail } => {
            (ConvergenceStatus::InvalidInput, detail)
        }
        OptError::SignChangeRequired { detail } => (ConvergenceStatus::InvalidInput, detail),
        OptError::NotImplemented { detail } => (ConvergenceStatus::NotImplemented, detail),
        OptError::NotConverged { detail } => (ConvergenceStatus::MaxIterations, detail),
    };
    OptimizeResult {
        x: clone_vec(x, profile),
        fun: None,
        success: false,
        status,
        message,
        nfev,
        njev,
        nhev: 0,
        nit,
        jac: None,
        hess_inv: None,
        maxcv: None,
    }
}

fn results_match_bits(left: &OptimizeResult, right: &OptimizeResult) -> bool {
    left.success == right.success
        && left.status == right.status
        && left.nit == right.nit
        && left.nfev == right.nfev
        && left.njev == right.njev
        && left.nhev == right.nhev
        && option_bits(left.fun) == option_bits(right.fun)
        && vec_bits(&left.x) == vec_bits(&right.x)
        && option_vec_bits(&left.jac) == option_vec_bits(&right.jac)
}

fn option_bits(value: Option<f64>) -> Option<u64> {
    value.map(f64::to_bits)
}

fn option_vec_bits(values: &Option<Vec<f64>>) -> Option<Vec<u64>> {
    values.as_ref().map(|inner| vec_bits(inner))
}

fn vec_bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn result_digest(result: &OptimizeResult) -> u64 {
    let mut digest = 0xcbf2_9ce4_8422_2325_u64;
    for bits in vec_bits(&result.x) {
        digest ^= bits;
        digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
    }
    if let Some(fun) = result.fun {
        digest ^= fun.to_bits();
        digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
    }
    if let Some(jac) = &result.jac {
        for bits in vec_bits(jac) {
            digest ^= bits;
            digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    digest ^= result.nfev as u64;
    digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
    digest ^ result.njev as u64
}

fn print_result(prefix: &str, result: &OptimizeResult) {
    println!(
        "{prefix}.success={} {prefix}.status={:?} {prefix}.nit={} {prefix}.nfev={} {prefix}.njev={} {prefix}.fun_bits={:016x} {prefix}.digest={:016x}",
        result.success,
        result.status,
        result.nit,
        result.nfev,
        result.njev,
        result.fun.unwrap_or(f64::NAN).to_bits(),
        result_digest(result)
    );
}

fn avg_count(value: usize, runs: usize) -> f64 {
    value as f64 / runs as f64
}

fn avg_duration_ns(value: Duration, runs: usize) -> f64 {
    value.as_nanos() as f64 / runs as f64
}

fn print_profile(profile: &StageProfile) {
    let runs = profile.runs.max(1);
    let total_actual = profile.objective_actual_calls + profile.objective_reserved_calls;
    println!("runs={}", profile.runs);
    println!(
        "avg_total_runtime_ns={:.1}",
        avg_duration_ns(profile.total_runtime, runs)
    );
    println!(
        "avg_objective_actual_calls={:.3}",
        avg_count(profile.objective_actual_calls, runs)
    );
    println!(
        "avg_objective_reserved_calls={:.3}",
        avg_count(profile.objective_reserved_calls, runs)
    );
    println!(
        "avg_objective_reported_nfev={:.3}",
        avg_count(total_actual, runs)
    );
    println!(
        "avg_initial_value_calls={:.3}",
        avg_count(profile.initial_value_calls, runs)
    );
    println!(
        "avg_finite_diff_gradient_calls={:.3}",
        avg_count(
            profile.finite_diff_gradient_calls + profile.wolfe_gradient_calls,
            runs
        )
    );
    println!(
        "avg_finite_diff_objective_calls_actual={:.3}",
        avg_count(profile.finite_diff_objective_calls, runs)
    );
    println!(
        "avg_finite_diff_objective_calls_reserved={:.3}",
        avg_count(profile.objective_reserved_calls, runs)
    );
    println!(
        "avg_finite_diff_runtime_ns={:.1}",
        avg_duration_ns(
            profile.finite_diff_runtime + profile.wolfe_gradient_runtime,
            runs
        )
    );
    println!(
        "avg_wolfe_calls={:.3}",
        avg_count(profile.wolfe_calls, runs)
    );
    println!(
        "avg_wolfe_runtime_ns={:.1}",
        avg_duration_ns(profile.wolfe_runtime, runs)
    );
    println!(
        "avg_wolfe_probe_evaluations={:.3}",
        avg_count(profile.wolfe_probe_evaluations, runs)
    );
    println!(
        "avg_wolfe_value_calls={:.3}",
        avg_count(profile.wolfe_value_calls, runs)
    );
    println!(
        "avg_wolfe_value_runtime_ns={:.1}",
        avg_duration_ns(profile.wolfe_value_runtime, runs)
    );
    println!(
        "avg_wolfe_gradient_calls={:.3}",
        avg_count(profile.wolfe_gradient_calls, runs)
    );
    println!(
        "avg_wolfe_gradient_runtime_ns={:.1}",
        avg_duration_ns(profile.wolfe_gradient_runtime, runs)
    );
    println!(
        "wolfe_directional_derivative_bits_xor={:016x}",
        profile.wolfe_directional_derivative_bits_xor
    );
    println!("avg_zoom_calls={:.3}", avg_count(profile.zoom_calls, runs));
    println!(
        "avg_zoom_iterations={:.3}",
        avg_count(profile.zoom_iterations, runs)
    );
    println!(
        "avg_accepted_x_materializations={:.3}",
        avg_count(profile.accepted_x_materializations, runs)
    );
    println!(
        "avg_accepted_x_elements={:.3}",
        avg_count(profile.accepted_x_elements, runs)
    );
    println!(
        "avg_accepted_x_runtime_ns={:.1}",
        avg_duration_ns(profile.accepted_x_runtime, runs)
    );
    println!(
        "avg_direction_update_runtime_ns={:.1}",
        avg_duration_ns(profile.direction_update_runtime, runs)
    );
    println!(
        "avg_armijo_fallback_calls={:.3}",
        avg_count(profile.armijo_fallback_calls, runs)
    );
    println!(
        "avg_allocation_vec_events={:.3}",
        avg_count(profile.allocation_traffic.vec_allocations, runs)
    );
    println!(
        "avg_allocation_vec_elements={:.3}",
        avg_count(profile.allocation_traffic.vec_elements, runs)
    );
    println!(
        "avg_allocation_vec_bytes={:.3}",
        avg_count(profile.allocation_traffic.vec_elements * F64_BYTES, runs)
    );
}

fn main() {
    let runs = env::args()
        .nth(1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(200)
        .max(1);
    let x0 = vec![0.0; DIM];
    let opts = options();

    let public = cg_pr_plus(&rosenbrock_nd, &x0, opts).expect("public cg_pr_plus");
    let mut first_profile = StageProfile::default();
    let first_started = Instant::now();
    let first = profiled_cg_pr_plus(&rosenbrock_nd, &x0, opts, &mut first_profile)
        .expect("profiled cg_pr_plus");
    first_profile.total_runtime += first_started.elapsed();
    first_profile.runs = 1;
    let public_match = results_match_bits(&public, &first);

    println!("target=cg/rosenbrock/10");
    println!("profile_bin=perf_cg_stage_profile");
    println!("public_match_bits={public_match}");
    print_result("public", &public);
    print_result("profiled_first", &first);
    if !public_match {
        println!("error=profiled CG did not match public cg_pr_plus bit-for-bit");
        std::process::exit(2);
    }

    let mut aggregate = StageProfile::default();
    aggregate.add(&first_profile);
    for _ in 1..runs {
        let mut profile = StageProfile::default();
        let started = Instant::now();
        let result = profiled_cg_pr_plus(&rosenbrock_nd, &x0, opts, &mut profile)
            .expect("profiled cg_pr_plus repeat");
        profile.total_runtime += started.elapsed();
        profile.runs = 1;
        if !results_match_bits(&public, &result) {
            println!("error=repeat profile result diverged from public cg_pr_plus");
            std::process::exit(3);
        }
        aggregate.add(&profile);
    }
    print_profile(&aggregate);
}
