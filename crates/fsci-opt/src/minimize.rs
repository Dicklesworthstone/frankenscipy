#![forbid(unsafe_code)]

use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::linesearch::{WolfeParams, line_search_wolfe2, line_search_wolfe2_with_gradient_probe};
use crate::types::{
    Bound, ConvergenceStatus, GradientFunc, MinimizeOptions, OptError, OptimizeMethod,
    OptimizeResult, OptimizeTraceEntry,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptCaspProblem {
    /// Number of decision variables in the minimize problem.
    pub dimension: usize,
    /// Scale diagnostic `max(abs(x_i).max(1)) / min(abs(x_i).max(1))`.
    pub variable_scale_ratio: f64,
    /// Whether finite box bounds are present.
    pub has_box_bounds: bool,
    /// Whether linear or nonlinear constraints are present.
    pub has_general_constraints: bool,
    /// Whether a reliable gradient signal is available.
    pub gradient_available: bool,
    /// Whether a Hessian-vector product is available.
    pub hessian_product_available: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptCaspDecision {
    /// Selected optimizer from the fsci-opt portfolio.
    pub method: OptimizeMethod,
    /// Human-readable stability rationale for audit traces.
    pub reason: String,
}

impl OptCaspProblem {
    #[must_use]
    pub fn unbounded_from_x0(x0: &[f64], options: MinimizeOptions) -> Self {
        Self {
            dimension: x0.len(),
            variable_scale_ratio: variable_scale_ratio(x0),
            has_box_bounds: false,
            has_general_constraints: false,
            gradient_available: options.gradient_available,
            hessian_product_available: options.hessp.is_some(),
        }
    }

    #[must_use]
    pub fn from_x0_and_options(x0: &[f64], options: MinimizeOptions) -> Self {
        Self {
            dimension: x0.len(),
            variable_scale_ratio: variable_scale_ratio(x0),
            has_box_bounds: options.bounds.is_some_and(bounds_have_finite_limit),
            has_general_constraints: options.has_general_constraints,
            gradient_available: options.gradient_available,
            hessian_product_available: options.hessp.is_some(),
        }
    }
}

/// Select a minimize solver using the fsci-opt CASP policy.
///
/// State space: constraint class, gradient availability, Hessian-product
/// availability, dimension, and variable scale ratio. Evidence signals come
/// from the public problem description rather than trial execution, so the
/// selector is deterministic and side-effect free. The loss matrix prioritizes
/// stability first: general constraints route to KKT-aware `trust-constr`, box
/// constraints to projected `L-BFGS-B`, available curvature to Newton-CG or
/// trust-region steps, high-dimensional finite-difference problems to CG, and
/// ordinary smooth unconstrained problems to BFGS.
pub fn select_minimize_method(problem: OptCaspProblem) -> Result<OptCaspDecision, OptError> {
    if problem.dimension == 0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("CASP selector requires dimension >= 1"),
        });
    }
    if !problem.variable_scale_ratio.is_finite() || problem.variable_scale_ratio < 1.0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("CASP selector requires finite variable_scale_ratio >= 1"),
        });
    }

    if problem.has_general_constraints {
        return Ok(OptCaspDecision {
            method: OptimizeMethod::TrustConstr,
            reason: String::from("general constraints require trust-constr KKT handling"),
        });
    }
    if problem.has_box_bounds {
        return Ok(OptCaspDecision {
            method: OptimizeMethod::LBfgsB,
            reason: String::from("box constraints require projected L-BFGS-B steps"),
        });
    }
    if !problem.gradient_available {
        return Ok(OptCaspDecision {
            method: OptimizeMethod::NelderMead,
            reason: String::from("no gradient signal available; using derivative-free simplex"),
        });
    }
    if problem.hessian_product_available {
        if problem.dimension <= 4 && problem.variable_scale_ratio >= 1.0e4 {
            return Ok(OptCaspDecision {
                method: OptimizeMethod::TrustExact,
                reason: String::from(
                    "small ill-scaled problem with Hessian product uses trust-region stability",
                ),
            });
        }
        return Ok(OptCaspDecision {
            method: OptimizeMethod::NewtonCg,
            reason: String::from("Hessian product available; using Newton-CG curvature steps"),
        });
    }
    if problem.dimension >= 40 {
        return Ok(OptCaspDecision {
            method: OptimizeMethod::ConjugateGradient,
            reason: String::from("large unconstrained finite-difference problem avoids dense BFGS"),
        });
    }

    Ok(OptCaspDecision {
        method: OptimizeMethod::Bfgs,
        reason: String::from("default smooth unconstrained problem uses BFGS"),
    })
}

pub fn minimize<F>(fun: F, x0: &[f64], options: MinimizeOptions) -> Result<OptimizeResult, OptError>
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

    let selected_method = if let Some(method) = options.method {
        method
    } else {
        validate_bounds_for_x0(x0, options.bounds)?;
        let decision = select_minimize_method(OptCaspProblem::from_x0_and_options(x0, options))?;
        log_casp_decision(options, &decision);
        decision.method
    };

    match selected_method {
        OptimizeMethod::Bfgs => bfgs(&fun, x0, options),
        OptimizeMethod::ConjugateGradient => cg_pr_plus(&fun, x0, options),
        OptimizeMethod::Powell => powell(&fun, x0, options),
        OptimizeMethod::NelderMead => nelder_mead(&fun, x0, options),
        OptimizeMethod::LBfgsB => lbfgsb(&fun, x0, options, options.bounds),
        OptimizeMethod::NewtonCg => newton_cg(&fun, x0, options),
        OptimizeMethod::TrustExact => trust_exact(&fun, x0, options),
        OptimizeMethod::Tnc => tnc(&fun, x0, options),
        OptimizeMethod::Slsqp => slsqp(&fun, x0, options),
        OptimizeMethod::TrustConstr => trust_constr(&fun, x0, options),
    }
}

/// Batched minimisation: minimise the SAME objective `fun` from MANY starting points
/// (`x0_rows`), one [`OptimizeResult`] per start. This is the vmap-over-solver primitive
/// SciPy lacks — there a multistart / parameter sweep loops `minimize` in Python, calling the
/// Python objective (and gradient) many times PER optimisation, N runs SERIALLY; here the N
/// independent runs are fanned across cores and the objective is an inlined Rust closure
/// (callback lever × N-way parallel). Result `i` is byte-identical to `minimize(fun,
/// &x0_rows[i], options)` — the batch only distributes independent runs.
///
/// The canonical use is global optimisation by multistart: minimise from many random `x0`,
/// then keep the lowest `fun`.
pub fn minimize_many<F>(
    fun: F,
    x0_rows: &[Vec<f64>],
    options: MinimizeOptions,
) -> Vec<Result<OptimizeResult, OptError>>
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    let nrows = x0_rows.len();
    if nrows == 0 {
        return Vec::new();
    }
    let fun_ref = &fun;
    let solve_one = move |x0: &[f64]| minimize(fun_ref, x0, options);

    // Each minimisation is an independent, expensive solve (many obj/grad evals) → fan whole
    // starts across cores, capped by the start count; a tiny batch stays serial.
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let nthreads = cores.min(nrows);
    if nthreads <= 1 || nrows < 4 {
        return x0_rows.iter().map(|x0| solve_one(x0)).collect();
    }

    let chunk = nrows.div_ceil(nthreads);
    let solve_one = &solve_one;
    let chunk_results: Vec<Vec<Result<OptimizeResult, OptError>>> =
        std::thread::scope(|scope| {
            (0..nthreads)
                .filter_map(|t| {
                    let lo = t * chunk;
                    if lo >= nrows {
                        return None;
                    }
                    let hi = (lo + chunk).min(nrows);
                    Some(scope.spawn(move || {
                        (lo..hi).map(|i| solve_one(&x0_rows[i])).collect::<Vec<_>>()
                    }))
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().expect("minimize_many worker panicked"))
                .collect()
        });

    let mut out = Vec::with_capacity(nrows);
    for cr in chunk_results {
        out.extend(cr);
    }
    out
}

pub fn minimize_with_audit<F>(
    fun: F,
    x0: &[f64],
    options: MinimizeOptions,
    ledger: &crate::audit::SyncSharedAuditLedger,
) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    let input_fingerprint = format!(
        "method={:?}; mode={:?}; x0={:?}; maxiter={:?}; maxfev={:?}",
        options.method, options.mode, x0, options.maxiter, options.maxfev
    );
    let result = minimize(fun, x0, options);
    match &result {
        Err(error) => crate::audit::record_fail_closed(
            ledger,
            input_fingerprint.as_bytes(),
            &format!("minimize::{error:?}"),
            "rejected",
        ),
        Ok(output)
            if options.mode == fsci_runtime::RuntimeMode::Hardened
                && matches!(
                    output.status,
                    ConvergenceStatus::NanEncountered
                        | ConvergenceStatus::InvalidInput
                        | ConvergenceStatus::OutOfBounds
                ) =>
        {
            crate::audit::record_fail_closed(
                ledger,
                input_fingerprint.as_bytes(),
                &format!("minimize::{:?}", output.status),
                "rejected",
            );
        }
        _ => {}
    }
    result
}

/// The convergence tolerance a caller actually asked for.
///
/// This used to be written `options.tol.unwrap_or(1.0e-6).max(1.0e-12)`, copied
/// into all nine optimizer entry points as each method was added, and the
/// `.max` silently overrode any tighter request. That is not a floor on
/// precision, it is a floor on what the caller is ALLOWED TO ASK FOR: for an
/// objective whose gradient is everywhere below 1e-12 — an ordinary objective
/// scaled down — `grad_norm <= tol` holds at iteration zero, so the routine
/// returned the STARTING POINT reporting success, and no setting could change
/// that (frankenscipy-ei0az).
///
/// Measured on `f(x) = s·((x₀−1)² + 2(x₁+2)²)` from (0,0) with tol = 1e-30:
/// at s = 2^-53 and below we returned x₀ itself, 2.236 from the minimizer, with
/// `success = true`. `scipy.optimize.minimize(method='trust-exact')` run with
/// `gtol=0` reaches the minimizer to 2.220e-16 at every scale down to 2^-60, so
/// the arithmetic is scale-invariant underneath and only the stopping rule was
/// in the way — the same shape as the `conlim` heuristic in
/// frankenscipy-xs7i2. At its DEFAULT tolerance SciPy stops at `nit = 0` on
/// these same problems and also reports success, so this is about honouring an
/// explicit request, not about beating the incumbent at defaults.
///
/// The clamp bought nothing that `maxiter` does not already provide: every entry
/// point bounds its own loop, so asking for an unreachable tolerance terminates
/// on iterations rather than spinning. `.max(0.0)` keeps a negative or NaN
/// request from meaning "converged immediately" — `f64::max` returns the
/// non-NaN operand — and tol = 0 then means what it means in SciPy: iterate
/// until another criterion stops you.
fn requested_tolerance(tol: Option<f64>) -> f64 {
    tol.unwrap_or(1.0e-6).max(0.0)
}

pub fn bfgs<F>(fun: &F, x0: &[f64], options: MinimizeOptions) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    validate_minimize_options(options)?;

    let n = x0.len();
    let tol = requested_tolerance(options.tol);
    let maxiter = options.maxiter.unwrap_or((200 * n).max(100));
    let maxfev = options.maxfev.unwrap_or((2000 * n).max(400));
    let mut objective = Objective::new(fun, options.mode, maxfev);

    let mut x = x0.to_vec();
    let mut f = match objective.eval(&x) {
        Ok(value) => value,
        Err(err) => return Ok(result_from_error(x0, 0, 0, 0, err)),
    };
    let mut njev = 0usize;
    let exact_gradient = options.gradient;
    let mut grad = match evaluate_minimize_gradient(
        &mut objective,
        exact_gradient,
        &x,
        options.gradient_eps,
    ) {
        Ok(value) => {
            njev += 1;
            value
        }
        Err(err) => return Ok(result_from_error(&x, 0, objective.nfev, njev, err)),
    };

    let mut h_inv = identity_matrix(n);
    let mut nit = 0usize;

    for iteration in 0..maxiter {
        let grad_norm = l2_norm(&grad);
        if grad_norm <= tol {
            let result = OptimizeResult {
                x: x.clone(),
                fun: Some(f),
                success: true,
                status: ConvergenceStatus::Success,
                message: String::from("optimization converged (gradient norm <= tol)"),
                nfev: objective.nfev,
                njev,
                nhev: 0,
                nit,
                jac: Some(grad.clone()),
                hess_inv: Some(h_inv.clone()),
                maxcv: None,
            };
            log_completion(OptimizeMethod::Bfgs, options, iteration, &result);
            return Ok(result);
        }

        if let Some(callback) = options.callback
            && !callback(&x)
        {
            let result = OptimizeResult {
                x: x.clone(),
                fun: Some(f),
                success: false,
                status: ConvergenceStatus::CallbackStop,
                message: String::from("callback requested stop"),
                nfev: objective.nfev,
                njev,
                nhev: 0,
                nit,
                jac: Some(grad.clone()),
                hess_inv: Some(h_inv.clone()),
                maxcv: None,
            };
            log_completion(OptimizeMethod::Bfgs, options, iteration, &result);
            return Ok(result);
        }

        let mut direction = matrix_vector_mul(&h_inv, &scale_vector(&grad, -1.0));
        if dot(&direction, &grad) >= 0.0 {
            direction = scale_vector(&grad, -1.0);
        }
        let search = match armijo_backtracking(&mut objective, &x, f, &grad, &direction) {
            Ok(Some(value)) => value,
            Ok(None) => {
                let result = OptimizeResult {
                    x: x.clone(),
                    fun: Some(f),
                    success: false,
                    status: ConvergenceStatus::PrecisionLoss,
                    message: String::from("line search failed to find a sufficient decrease"),
                    nfev: objective.nfev,
                    njev,
                    nhev: 0,
                    nit,
                    jac: Some(grad.clone()),
                    hess_inv: Some(h_inv.clone()),
                    maxcv: None,
                };
                log_completion(OptimizeMethod::Bfgs, options, iteration, &result);
                return Ok(result);
            }
            Err(err) => return Ok(result_from_error(&x, nit, objective.nfev, njev, err)),
        };

        let next_grad = match evaluate_minimize_gradient(
            &mut objective,
            exact_gradient,
            &search.x,
            options.gradient_eps,
        ) {
            Ok(value) => {
                njev += 1;
                value
            }
            Err(err) => return Ok(result_from_error(&search.x, nit, objective.nfev, njev, err)),
        };

        let s = sub_vectors(&search.x, &x);
        let y = sub_vectors(&next_grad, &grad);
        let ys = dot(&y, &s);
        if ys > 1.0e-12 {
            let rho = 1.0 / ys;
            h_inv = bfgs_inverse_update(&h_inv, &s, &y, rho);
        }
        // If ys <= 1e-12, skip the update to preserve existing curvature info.
        // Resetting to identity would lose all accumulated information.

        log_iteration(
            OptimizeMethod::Bfgs,
            options,
            iteration + 1,
            search.f,
            l2_norm(&next_grad),
            search.alpha,
            objective.nfev,
        );

        x = search.x;
        f = search.f;
        grad = next_grad;
        nit = iteration + 1;
    }

    let result = OptimizeResult {
        x: x.clone(),
        fun: Some(f),
        success: false,
        status: ConvergenceStatus::MaxIterations,
        message: String::from("maximum iterations exceeded"),
        nfev: objective.nfev,
        njev,
        nhev: 0,
        nit,
        jac: Some(grad),
        hess_inv: Some(h_inv),
        maxcv: None,
    };
    log_completion(OptimizeMethod::Bfgs, options, nit, &result);
    Ok(result)
}

pub fn cg_pr_plus<F>(
    fun: &F,
    x0: &[f64],
    options: MinimizeOptions,
) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    validate_minimize_options(options)?;

    let n = x0.len();
    let tol = requested_tolerance(options.tol);
    let maxiter = options.maxiter.unwrap_or((250 * n).max(120));
    let maxfev = options.maxfev.unwrap_or((2500 * n).max(500));
    let mut objective = Objective::new(fun, options.mode, maxfev);

    let mut x = x0.to_vec();
    let mut f = match objective.eval(&x) {
        Ok(value) => value,
        Err(err) => return Ok(result_from_error(x0, 0, 0, 0, err)),
    };
    let mut njev = 0usize;
    let exact_gradient = options.gradient;
    let mut grad = match evaluate_minimize_gradient(
        &mut objective,
        exact_gradient,
        &x,
        options.gradient_eps,
    ) {
        Ok(value) => {
            njev += 1;
            value
        }
        Err(err) => return Ok(result_from_error(&x, 0, objective.nfev, njev, err)),
    };
    let mut direction = scale_vector(&grad, -1.0);
    let mut nit = 0usize;

    for iteration in 0..maxiter {
        let grad_norm = l2_norm(&grad);
        if grad_norm <= tol {
            let result = OptimizeResult {
                x: x.clone(),
                fun: Some(f),
                success: true,
                status: ConvergenceStatus::Success,
                message: String::from("optimization converged (gradient norm <= tol)"),
                nfev: objective.nfev,
                njev,
                nhev: 0,
                nit,
                jac: Some(grad.clone()),
                hess_inv: None,
                maxcv: None,
            };
            log_completion(
                OptimizeMethod::ConjugateGradient,
                options,
                iteration,
                &result,
            );
            return Ok(result);
        }

        if let Some(callback) = options.callback
            && !callback(&x)
        {
            let result = OptimizeResult {
                x: x.clone(),
                fun: Some(f),
                success: false,
                status: ConvergenceStatus::CallbackStop,
                message: String::from("callback requested stop"),
                nfev: objective.nfev,
                njev,
                nhev: 0,
                nit,
                jac: Some(grad.clone()),
                hess_inv: None,
                maxcv: None,
            };
            log_completion(
                OptimizeMethod::ConjugateGradient,
                options,
                iteration,
                &result,
            );
            return Ok(result);
        }

        if dot(&direction, &grad) >= 0.0 {
            direction = scale_vector(&grad, -1.0);
        }
        // Strong-Wolfe line search: nonlinear CG needs the curvature condition to
        // keep generating descent directions; Armijo-only steps stall the
        // Polak-Ribière recurrence (e.g. Rosenbrock). scipy's CG uses the same.
        let wolfe_search = {
            let eps = options.gradient_eps;
            let counter = std::cell::Cell::new(0usize);
            let mut wolfe_gradient_evals = 0usize;
            let mut gradient_error: Option<OptError> = None;
            let f_closure = |xv: &[f64]| {
                counter.set(counter.get() + 1);
                (fun)(xv)
            };
            let mut g_closure = |xv: &mut [f64], gradient: &mut Vec<f64>| {
                if let Some(gradient_fn) = exact_gradient {
                    match validate_gradient_output(gradient_fn(xv), xv.len()) {
                        Ok(value) => {
                            *gradient = value;
                            wolfe_gradient_evals += 1;
                        }
                        Err(err) => {
                            gradient_error = Some(err);
                            gradient.clear();
                            gradient.resize(xv.len(), f64::NAN);
                            return f64::NAN;
                        }
                    }
                } else {
                    if gradient.len() != xv.len() {
                        gradient.resize(xv.len(), 0.0);
                    }
                    for i in 0..xv.len() {
                        let step = eps * (1.0 + xv[i].abs());
                        let orig = xv[i];
                        xv[i] = orig + step;
                        counter.set(counter.get() + 1);
                        let fp = (fun)(xv);
                        xv[i] = orig - step;
                        counter.set(counter.get() + 1);
                        let fm = (fun)(xv);
                        xv[i] = orig;
                        gradient[i] = (fp - fm) / (2.0 * step);
                    }
                }
                dot(gradient, &direction)
            };
            let res = line_search_wolfe2_with_gradient_probe(
                &f_closure,
                &mut g_closure,
                &x,
                &direction,
                f,
                &grad,
                WolfeParams::default(),
            );
            if let Some(err) = gradient_error {
                return Ok(result_from_error(&x, nit, objective.nfev, njev, err));
            }
            if let Ok(ls) = res {
                objective.nfev += counter.get();
                njev += wolfe_gradient_evals;
                Some(LineSearchStep {
                    x: add_scaled(&x, &direction, ls.result.alpha),
                    f: ls.result.f_at_alpha,
                    alpha: ls.result.alpha,
                    accepted_gradient: ls.accepted_gradient,
                })
            } else {
                None
            }
        };
        let search = match wolfe_search {
            Some(value) => value,
            None => match armijo_backtracking(&mut objective, &x, f, &grad, &direction) {
                Ok(Some(value)) => value,
                Ok(None) => {
                    let result = OptimizeResult {
                        x: x.clone(),
                        fun: Some(f),
                        success: false,
                        status: ConvergenceStatus::PrecisionLoss,
                        message: String::from("line search failed to find a sufficient decrease"),
                        nfev: objective.nfev,
                        njev,
                        nhev: 0,
                        nit,
                        jac: Some(grad.clone()),
                        hess_inv: None,
                        maxcv: None,
                    };
                    log_completion(
                        OptimizeMethod::ConjugateGradient,
                        options,
                        iteration,
                        &result,
                    );
                    return Ok(result);
                }
                Err(err) => return Ok(result_from_error(&x, nit, objective.nfev, njev, err)),
            },
        };

        let next_grad = match search.accepted_gradient {
            Some(accepted_gradient) => {
                if exact_gradient.is_none() {
                    if let Err(err) = objective.reserve_evaluations(n * 2) {
                        return Ok(result_from_error(&search.x, nit, objective.nfev, njev, err));
                    }
                    njev += 1;
                }
                accepted_gradient
            }
            None => match evaluate_minimize_gradient(
                &mut objective,
                exact_gradient,
                &search.x,
                options.gradient_eps,
            ) {
                Ok(value) => {
                    njev += 1;
                    value
                }
                Err(err) => {
                    return Ok(result_from_error(&search.x, nit, objective.nfev, njev, err));
                }
            },
        };

        let denom = dot(&grad, &grad).max(1.0e-18);
        let grad_delta = sub_vectors(&next_grad, &grad);
        let beta_pr = dot(&next_grad, &grad_delta) / denom;
        let beta = beta_pr.max(0.0);
        direction = sub_vectors(
            &scale_vector(&next_grad, -1.0),
            &scale_vector(&direction, -beta),
        );

        log_iteration(
            OptimizeMethod::ConjugateGradient,
            options,
            iteration + 1,
            search.f,
            l2_norm(&next_grad),
            search.alpha,
            objective.nfev,
        );

        x = search.x;
        f = search.f;
        grad = next_grad;
        nit = iteration + 1;
    }

    let result = OptimizeResult {
        x: x.clone(),
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
    log_completion(OptimizeMethod::ConjugateGradient, options, nit, &result);
    Ok(result)
}

pub fn powell<F>(fun: &F, x0: &[f64], options: MinimizeOptions) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    validate_minimize_options(options)?;

    let n = x0.len();
    let tol = requested_tolerance(options.tol);
    let maxiter = options.maxiter.unwrap_or((150 * n).max(80));
    let maxfev = options.maxfev.unwrap_or((3000 * n).max(800));
    let mut objective = Objective::new(fun, options.mode, maxfev);

    let mut x = x0.to_vec();
    let mut f = match objective.eval(&x) {
        Ok(value) => value,
        Err(err) => return Ok(result_from_error(x0, 0, 0, 0, err)),
    };
    let mut directions = identity_matrix(n);

    for iteration in 0..maxiter {
        if let Some(callback) = options.callback
            && !callback(&x)
        {
            let result = OptimizeResult {
                x: x.clone(),
                fun: Some(f),
                success: false,
                status: ConvergenceStatus::CallbackStop,
                message: String::from("callback requested stop"),
                nfev: objective.nfev,
                njev: 0,
                nhev: 0,
                nit: iteration,
                jac: None,
                hess_inv: None,
                maxcv: None,
            };
            log_completion(OptimizeMethod::Powell, options, iteration, &result);
            return Ok(result);
        }

        let x_start = x.clone();
        let f_start = f;
        let mut largest_drop = 0.0;
        let mut largest_drop_idx = 0usize;

        for (dir_idx, direction) in directions.iter().enumerate() {
            let search = match golden_section_direction_search(
                &mut objective,
                &x,
                f,
                direction,
                tol.max(1.0e-4),
            ) {
                Ok(value) => value,
                Err(err) => return Ok(result_from_error(&x, iteration, objective.nfev, 0, err)),
            };
            let drop = (f - search.f).max(0.0);
            if drop > largest_drop {
                largest_drop = drop;
                largest_drop_idx = dir_idx;
            }
            x = search.x;
            f = search.f;
            log_iteration(
                OptimizeMethod::Powell,
                options,
                iteration + 1,
                f,
                0.0,
                search.alpha,
                objective.nfev,
            );
        }

        let move_vec = sub_vectors(&x, &x_start);
        let move_norm = l2_norm(&move_vec);
        let f_delta = (f_start - f).abs();

        // Convergence check
        if move_norm <= tol || f_delta <= tol {
            let result = OptimizeResult {
                x: x.clone(),
                fun: Some(f),
                success: true,
                status: ConvergenceStatus::Success,
                message: String::from("optimization converged (step/f-value change <= tol)"),
                nfev: objective.nfev,
                njev: 0,
                nhev: 0,
                nit: iteration + 1,
                jac: None,
                hess_inv: None,
                maxcv: None,
            };
            log_completion(OptimizeMethod::Powell, options, iteration + 1, &result);
            return Ok(result);
        }

        // Direction set update (standard Powell criterion)
        // x_start is point before iteration, x is point after n line searches
        // x_ext = 2*x - x_start (extrapolated point)
        let x_ext: Vec<f64> = x
            .iter()
            .zip(x_start.iter())
            .map(|(&xi, &xs)| 2.0 * xi - xs)
            .collect();
        let f_ext = match objective.eval(&x_ext) {
            Ok(v) => v,
            Err(err) => return Ok(result_from_error(&x, iteration, objective.nfev, 0, err)),
        };

        if f_ext < f_start {
            let term1 = f_start - f - largest_drop;
            let term2 = f_start - f_ext;
            let lhs = 2.0 * (f_start - 2.0 * f + f_ext) * term1 * term1;
            let rhs = largest_drop * term2 * term2;

            if lhs < rhs {
                // Update direction set: replace direction of largest drop with total move
                let search = match golden_section_direction_search(
                    &mut objective,
                    &x,
                    f,
                    &move_vec,
                    tol.max(1.0e-4),
                ) {
                    Ok(v) => v,
                    Err(err) => {
                        return Ok(result_from_error(&x, iteration, objective.nfev, 0, err));
                    }
                };
                x = search.x;
                f = search.f;

                if move_norm > 1.0e-12 {
                    directions.remove(largest_drop_idx);
                    directions.push(scale_vector(&move_vec, 1.0 / move_norm));
                }
            }
        }
    }

    let result = OptimizeResult {
        x: x.clone(),
        fun: Some(f),
        success: false,
        status: ConvergenceStatus::MaxIterations,
        message: String::from("maximum iterations exceeded"),
        nfev: objective.nfev,
        njev: 0,
        nhev: 0,
        nit: maxiter,
        jac: None,
        hess_inv: None,
        maxcv: None,
    };
    log_completion(OptimizeMethod::Powell, options, maxiter, &result);
    Ok(result)
}

/// Nelder-Mead simplex (downhill simplex) method for derivative-free optimization.
///
/// Matches `scipy.optimize.minimize(f, x0, method='Nelder-Mead')`.
/// Uses the standard simplex coefficients (rho=1, chi=2, psi=0.5, sigma=0.5),
/// which is SciPy's default (`adaptive=False`). SciPy's adaptive Gao-Han
/// coefficients are an opt-in and are NOT the default — using them for n>=2
/// reproduces SciPy's `adaptive=True` stagnation on higher-dim Rosenbrock.
pub fn nelder_mead<F>(
    fun: &F,
    x0: &[f64],
    options: MinimizeOptions,
) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    // Nelder-Mead doesn't use gradient_eps, but we still validate other options
    if let Some(maxiter) = options.maxiter
        && maxiter == 0
    {
        return Err(OptError::InvalidArgument {
            detail: String::from("maxiter must be >= 1"),
        });
    }
    if let Some(maxfev) = options.maxfev
        && maxfev == 0
    {
        return Err(OptError::InvalidArgument {
            detail: String::from("maxfev must be >= 1"),
        });
    }

    let n = x0.len();
    let tol = options.tol.unwrap_or(1.0e-8);
    let xatol = tol;
    let fatol = tol;
    let maxiter = options.maxiter.unwrap_or(200 * n);
    let maxfev = options.maxfev.unwrap_or(200 * n);
    let mut objective = Objective::new(fun, options.mode, maxfev);

    // Standard simplex coefficients — SciPy's default (adaptive=False) for all n.
    let (rho, chi, psi, sigma) = (1.0, 2.0, 0.5, 0.5);

    // Build initial simplex: n+1 vertices
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x0.to_vec());

    for j in 0..n {
        let mut vertex = x0.to_vec();
        let h = if x0[j].abs() > 1e-12 {
            0.05 * x0[j]
        } else {
            0.00025
        };
        vertex[j] += h;
        simplex.push(vertex);
    }

    // Evaluate at all vertices
    let mut f_values: Vec<f64> = Vec::with_capacity(n + 1);
    for vertex in &simplex {
        let fval = match objective.eval(vertex) {
            Ok(v) => v,
            Err(err) => return Ok(result_from_error(x0, 0, objective.nfev, 0, err)),
        };
        f_values.push(fval);
    }

    let mut nit = 0usize;

    for iteration in 0..maxiter {
        nit = iteration + 1;

        // Sort simplex by function values
        let mut pairs: Vec<(f64, Vec<f64>)> = f_values.into_iter().zip(simplex).collect();
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let (sorted_f, sorted_simplex): (Vec<f64>, Vec<Vec<f64>>) = pairs.into_iter().unzip();
        simplex = sorted_simplex;
        f_values = sorted_f;

        // Check convergence: range of function values and simplex diameter
        let f_range = f_values[n] - f_values[0];
        let mut max_delta = 0.0_f64;
        for vertex in simplex.iter().skip(1) {
            let delta: f64 = vertex
                .iter()
                .zip(simplex[0].iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, |a: f64, b: f64| {
                    if a.is_nan() || b.is_nan() {
                        f64::NAN
                    } else {
                        a.max(b)
                    }
                });
            max_delta = if max_delta.is_nan() || delta.is_nan() {
                f64::NAN
            } else {
                max_delta.max(delta)
            };
        }

        if f_range <= fatol && max_delta <= xatol {
            let result = OptimizeResult {
                x: simplex[0].clone(),
                fun: Some(f_values[0]),
                success: true,
                status: ConvergenceStatus::Success,
                message: String::from("optimization converged (Nelder-Mead)"),
                nfev: objective.nfev,
                njev: 0,
                nhev: 0,
                nit,
                jac: None,
                hess_inv: None,
                maxcv: None,
            };
            log_completion(OptimizeMethod::NelderMead, options, iteration, &result);
            return Ok(result);
        }

        if let Some(callback) = options.callback
            && !callback(&simplex[0])
        {
            let result = OptimizeResult {
                x: simplex[0].clone(),
                fun: Some(f_values[0]),
                success: false,
                status: ConvergenceStatus::CallbackStop,
                message: String::from("callback requested stop"),
                nfev: objective.nfev,
                njev: 0,
                nhev: 0,
                nit,
                jac: None,
                hess_inv: None,
                maxcv: None,
            };
            log_completion(OptimizeMethod::NelderMead, options, iteration, &result);
            return Ok(result);
        }

        // Centroid of best n vertices (exclude worst)
        let mut centroid = vec![0.0; n];
        for vertex in simplex.iter().take(n) {
            for (j, c) in centroid.iter_mut().enumerate() {
                *c += vertex[j];
            }
        }
        for c in &mut centroid {
            *c /= n as f64;
        }

        // Reflection: x_r = centroid + rho * (centroid - worst)
        let worst = &simplex[n];
        let x_r: Vec<f64> = centroid
            .iter()
            .zip(worst.iter())
            .map(|(c, w)| c + rho * (c - w))
            .collect();
        let f_r = match objective.eval(&x_r) {
            Ok(v) => v,
            Err(err) => return Ok(result_from_error(&simplex[0], nit, objective.nfev, 0, err)),
        };

        if f_r < f_values[0] {
            // Expansion: x_e = centroid + chi * (x_r - centroid)
            let x_e: Vec<f64> = centroid
                .iter()
                .zip(x_r.iter())
                .map(|(c, r)| c + chi * (r - c))
                .collect();
            let f_e = match objective.eval(&x_e) {
                Ok(v) => v,
                Err(err) => return Ok(result_from_error(&simplex[0], nit, objective.nfev, 0, err)),
            };
            if f_e < f_r {
                simplex[n] = x_e;
                f_values[n] = f_e;
            } else {
                simplex[n] = x_r;
                f_values[n] = f_r;
            }
        } else if f_r < f_values[n - 1] {
            // Accept reflection
            simplex[n] = x_r;
            f_values[n] = f_r;
        } else {
            // Contraction
            if f_r < f_values[n] {
                // Outside contraction: x_c = centroid + psi * (x_r - centroid)
                let x_c: Vec<f64> = centroid
                    .iter()
                    .zip(x_r.iter())
                    .map(|(c, r)| c + psi * (r - c))
                    .collect();
                let f_c = match objective.eval(&x_c) {
                    Ok(v) => v,
                    Err(err) => {
                        return Ok(result_from_error(&simplex[0], nit, objective.nfev, 0, err));
                    }
                };
                if f_c <= f_r {
                    simplex[n] = x_c;
                    f_values[n] = f_c;
                } else {
                    // Shrink
                    match nelder_mead_shrink(&mut simplex, &mut f_values, sigma, &mut objective, n)
                    {
                        Ok(()) => {}
                        Err(err) => {
                            return Ok(result_from_error(&simplex[0], nit, objective.nfev, 0, err));
                        }
                    }
                }
            } else {
                // Inside contraction: x_cc = centroid - psi * (centroid - worst)
                let x_cc: Vec<f64> = centroid
                    .iter()
                    .zip(worst.iter())
                    .map(|(c, w)| c - psi * (c - w))
                    .collect();
                let f_cc = match objective.eval(&x_cc) {
                    Ok(v) => v,
                    Err(err) => {
                        return Ok(result_from_error(&simplex[0], nit, objective.nfev, 0, err));
                    }
                };
                if f_cc < f_values[n] {
                    simplex[n] = x_cc;
                    f_values[n] = f_cc;
                } else {
                    // Shrink
                    match nelder_mead_shrink(&mut simplex, &mut f_values, sigma, &mut objective, n)
                    {
                        Ok(()) => {}
                        Err(err) => {
                            return Ok(result_from_error(&simplex[0], nit, objective.nfev, 0, err));
                        }
                    }
                }
            }
        }

        log_iteration(
            OptimizeMethod::NelderMead,
            options,
            iteration,
            f_values[0],
            0.0,
            0.0,
            objective.nfev,
        );
    }

    // Max iterations reached
    let mut indices: Vec<usize> = (0..=n).collect();
    indices.sort_by(|&a, &b| f_values[a].total_cmp(&f_values[b]));
    let best_idx = indices[0];

    let result = OptimizeResult {
        x: simplex[best_idx].clone(),
        fun: Some(f_values[best_idx]),
        success: false,
        status: ConvergenceStatus::MaxIterations,
        message: format!("maximum iterations reached ({maxiter})"),
        nfev: objective.nfev,
        njev: 0,
        nhev: 0,
        nit,
        jac: None,
        hess_inv: None,
        maxcv: None,
    };
    log_completion(OptimizeMethod::NelderMead, options, nit, &result);
    Ok(result)
}

fn nelder_mead_shrink<F>(
    simplex: &mut [Vec<f64>],
    f_values: &mut [f64],
    sigma: f64,
    objective: &mut Objective<'_, F>,
    n: usize,
) -> Result<(), OptError>
where
    F: Fn(&[f64]) -> f64,
{
    let best = simplex[0].clone();
    for i in 1..=n {
        for j in 0..n {
            simplex[i][j] = best[j] + sigma * (simplex[i][j] - best[j]);
        }
        f_values[i] = objective.eval(&simplex[i]).map_err(|e| match e {
            OptError::EvaluationBudgetExceeded { detail } => {
                OptError::EvaluationBudgetExceeded { detail }
            }
            other => other,
        })?;
    }
    Ok(())
}

/// L-BFGS-B: Limited-memory BFGS with box constraints.
///
/// Matches `scipy.optimize.minimize(f, x0, method='L-BFGS-B', bounds=...)`.
/// Uses two-loop recursion with limited memory (m=10 corrections).
pub fn lbfgsb<F>(
    fun: &F,
    x0: &[f64],
    options: MinimizeOptions,
    bounds: Option<&[Bound]>,
) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    validate_minimize_options(options)?;
    validate_bounds_for_x0(x0, bounds)?;

    let n = x0.len();
    let tol = requested_tolerance(options.tol);
    let maxiter = options.maxiter.unwrap_or((200 * n).max(100));
    let maxfev = options.maxfev.unwrap_or((2000 * n).max(400));
    let m = 10; // number of correction pairs stored
    let mut objective = Objective::new(fun, options.mode, maxfev);

    // Project x0 onto bounds
    let mut x = x0.to_vec();
    if let Some(bounds) = bounds {
        project_onto_bounds(&mut x, bounds);
    }

    let mut f = match objective.eval(&x) {
        Ok(value) => value,
        Err(err) => return Ok(result_from_error(x0, 0, 0, 0, err)),
    };

    let mut njev = 0usize;
    let mut grad = match finite_diff_gradient(&mut objective, &x, options.gradient_eps) {
        Ok(value) => {
            njev += 1;
            value
        }
        Err(err) => return Ok(result_from_error(&x, 0, objective.nfev, njev, err)),
    };

    // L-BFGS correction history
    let mut s_history: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut y_history: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut rho_history: Vec<f64> = Vec::with_capacity(m);

    let mut nit = 0usize;

    for iteration in 0..maxiter {
        nit = iteration + 1;

        // Project gradient for bound-constrained case
        let projected_grad = if let Some(bounds) = bounds {
            projected_gradient(&x, &grad, bounds)
        } else {
            grad.clone()
        };

        let grad_norm = l2_norm(&projected_grad);
        if grad_norm <= tol {
            let result = OptimizeResult {
                x: x.clone(),
                fun: Some(f),
                success: true,
                status: ConvergenceStatus::Success,
                message: String::from("optimization converged (L-BFGS-B gradient norm <= tol)"),
                nfev: objective.nfev,
                njev,
                nhev: 0,
                nit,
                jac: Some(grad.clone()),
                hess_inv: None,
                maxcv: None,
            };
            log_completion(OptimizeMethod::LBfgsB, options, iteration, &result);
            return Ok(result);
        }

        if let Some(callback) = options.callback
            && !callback(&x)
        {
            let result = OptimizeResult {
                x: x.clone(),
                fun: Some(f),
                success: false,
                status: ConvergenceStatus::CallbackStop,
                message: String::from("callback requested stop"),
                nfev: objective.nfev,
                njev,
                nhev: 0,
                nit,
                jac: Some(grad.clone()),
                hess_inv: None,
                maxcv: None,
            };
            log_completion(OptimizeMethod::LBfgsB, options, iteration, &result);
            return Ok(result);
        }

        // Two-loop recursion for L-BFGS direction
        let direction = lbfgs_two_loop(&grad, &s_history, &y_history, &rho_history);

        // Negate for descent direction
        let direction: Vec<f64> = direction.iter().map(|&d| -d).collect();

        // Line search with bound projection
        let mut alpha = 1.0;
        let directional_deriv = dot(&grad, &direction);
        if directional_deriv >= 0.0 {
            // Not a descent direction — reset history and use steepest descent
            s_history.clear();
            y_history.clear();
            rho_history.clear();
            let direction: Vec<f64> = grad.iter().map(|&g| -g).collect();
            alpha = 1.0 / l2_norm(&direction).max(1.0);
            let candidate_x = add_scaled(&x, &direction, alpha);
            let mut projected_candidate = candidate_x;
            if let Some(bounds) = bounds {
                project_onto_bounds(&mut projected_candidate, bounds);
            }
            match objective.eval(&projected_candidate) {
                Ok(fv) => {
                    let s = sub_vectors(&projected_candidate, &x);
                    x = projected_candidate;
                    f = fv;
                    let new_grad =
                        match finite_diff_gradient(&mut objective, &x, options.gradient_eps) {
                            Ok(g) => {
                                njev += 1;
                                g
                            }
                            Err(err) => {
                                return Ok(result_from_error(&x, nit, objective.nfev, njev, err));
                            }
                        };
                    let y = sub_vectors(&new_grad, &grad);
                    let sy = dot(&s, &y);
                    if sy > 1e-10 {
                        push_lbfgs_history(
                            &mut s_history,
                            &mut y_history,
                            &mut rho_history,
                            s,
                            y,
                            sy,
                            m,
                        );
                    }
                    grad = new_grad;
                }
                Err(err) => return Ok(result_from_error(&x, nit, objective.nfev, njev, err)),
            }
            continue;
        }

        // Strong-Wolfe line search for the unconstrained case. The curvature
        // condition |g(x+αd)·d| ≤ c2·|g·d| guarantees s·y > 0, so every L-BFGS
        // correction pair is valid; with Armijo-only steps the limited-memory
        // model degrades to steepest descent and stalls (e.g. Rosenbrock from a
        // hard start). scipy's L-BFGS-B uses an equivalent dcsrch line search.
        if bounds.is_none() {
            let eps = options.gradient_eps;
            let counter = std::cell::Cell::new(0usize);
            let f_closure = |xv: &[f64]| {
                counter.set(counter.get() + 1);
                (fun)(xv)
            };
            let g_closure = |xv: &[f64]| {
                let mut g = vec![0.0; xv.len()];
                let mut xp = xv.to_vec();
                for i in 0..xv.len() {
                    let step = eps * (1.0 + xv[i].abs());
                    let orig = xp[i];
                    xp[i] = orig + step;
                    counter.set(counter.get() + 1);
                    let fp = (fun)(&xp);
                    xp[i] = orig - step;
                    counter.set(counter.get() + 1);
                    let fm = (fun)(&xp);
                    xp[i] = orig;
                    g[i] = (fp - fm) / (2.0 * step);
                }
                g
            };
            let wolfe = line_search_wolfe2(
                &f_closure,
                &g_closure,
                &x,
                &direction,
                f,
                &grad,
                WolfeParams::default(),
            );
            if let Ok(ls) = wolfe {
                objective.nfev += counter.get();
                let new_x = add_scaled(&x, &direction, ls.alpha);
                let s = sub_vectors(&new_x, &x);
                x = new_x;
                f = ls.f_at_alpha;
                let new_grad = match finite_diff_gradient(&mut objective, &x, eps) {
                    Ok(g) => {
                        njev += 1;
                        g
                    }
                    Err(err) => return Ok(result_from_error(&x, nit, objective.nfev, njev, err)),
                };
                let y = sub_vectors(&new_grad, &grad);
                let sy = dot(&s, &y);
                if sy > 1e-10 {
                    push_lbfgs_history(
                        &mut s_history,
                        &mut y_history,
                        &mut rho_history,
                        s,
                        y,
                        sy,
                        m,
                    );
                }
                grad = new_grad;
                log_iteration(
                    OptimizeMethod::LBfgsB,
                    options,
                    iteration,
                    f,
                    grad_norm,
                    ls.alpha,
                    objective.nfev,
                );
                continue;
            }
            // Wolfe search failed → fall through to the projected Armijo fallback.
        }

        // Armijo backtracking with bound projection
        let c1 = 1e-4;
        let mut step_accepted = false;
        for _ in 0..24 {
            let mut candidate_x = add_scaled(&x, &direction, alpha);
            if let Some(bounds) = bounds {
                project_onto_bounds(&mut candidate_x, bounds);
            }
            match objective.eval(&candidate_x) {
                Ok(fv) => {
                    let actual_step = sub_vectors(&candidate_x, &x);
                    let actual_directional_deriv = dot(&grad, &actual_step);

                    if fv <= f + c1 * actual_directional_deriv {
                        let s = actual_step;
                        x = candidate_x;
                        f = fv;
                        let new_grad =
                            match finite_diff_gradient(&mut objective, &x, options.gradient_eps) {
                                Ok(g) => {
                                    njev += 1;
                                    g
                                }
                                Err(err) => {
                                    return Ok(result_from_error(
                                        &x,
                                        nit,
                                        objective.nfev,
                                        njev,
                                        err,
                                    ));
                                }
                            };
                        let y = sub_vectors(&new_grad, &grad);
                        let sy = dot(&s, &y);
                        if sy > 1e-10 {
                            push_lbfgs_history(
                                &mut s_history,
                                &mut y_history,
                                &mut rho_history,
                                s,
                                y,
                                sy,
                                m,
                            );
                        }
                        grad = new_grad;
                        step_accepted = true;
                        break;
                    }
                }
                Err(err) => return Ok(result_from_error(&x, nit, objective.nfev, njev, err)),
            }
            alpha *= 0.5;
            if alpha < 1e-12 {
                break;
            }
        }

        if !step_accepted {
            break;
        }

        log_iteration(
            OptimizeMethod::LBfgsB,
            options,
            iteration,
            f,
            grad_norm,
            alpha,
            objective.nfev,
        );
    }

    let result = OptimizeResult {
        x: x.clone(),
        fun: Some(f),
        success: false,
        status: ConvergenceStatus::MaxIterations,
        message: format!("maximum iterations reached ({maxiter})"),
        nfev: objective.nfev,
        njev,
        nhev: 0,
        nit,
        jac: Some(grad),
        hess_inv: None,
        maxcv: None,
    };
    log_completion(OptimizeMethod::LBfgsB, options, nit, &result);
    Ok(result)
}

/// Project x onto box constraints.
fn project_onto_bounds(x: &mut [f64], bounds: &[Bound]) {
    for (xi, bound) in x.iter_mut().zip(bounds.iter()) {
        if xi.is_nan() {
            continue;
        }
        if let Some(lo) = bound.0 {
            *xi = xi.max(lo);
        }
        if let Some(hi) = bound.1 {
            *xi = xi.min(hi);
        }
    }
}

/// Compute projected gradient: zero out components at active bounds.
fn projected_gradient(x: &[f64], grad: &[f64], bounds: &[Bound]) -> Vec<f64> {
    let mut pg = grad.to_vec();
    for (i, (xi, bound)) in x.iter().zip(bounds.iter()).enumerate() {
        if let Some(lo) = bound.0
            && (*xi - lo).abs() < 1e-14
            && pg[i] > 0.0
        {
            pg[i] = 0.0;
        }
        if let Some(hi) = bound.1
            && (*xi - hi).abs() < 1e-14
            && pg[i] < 0.0
        {
            pg[i] = 0.0;
        }
    }
    pg
}

/// L-BFGS two-loop recursion to compute H*g.
fn lbfgs_two_loop(
    grad: &[f64],
    s_hist: &[Vec<f64>],
    y_hist: &[Vec<f64>],
    rho_hist: &[f64],
) -> Vec<f64> {
    let k = s_hist.len();
    if k == 0 {
        return grad.to_vec();
    }

    let mut q = grad.to_vec();
    let mut alpha_cache = vec![0.0; k];

    // Forward loop
    for i in (0..k).rev() {
        alpha_cache[i] = rho_hist[i] * dot(&s_hist[i], &q);
        q = add_scaled(&q, &y_hist[i], -alpha_cache[i]);
    }

    // Initial Hessian scaling: H0 = (s^T y / y^T y) * I
    let sy = dot(&s_hist[k - 1], &y_hist[k - 1]);
    let yy = dot(&y_hist[k - 1], &y_hist[k - 1]);
    let gamma = if yy > 1e-15 { sy / yy } else { 1.0 };
    let mut r = scale_vector(&q, gamma);

    // Backward loop
    for i in 0..k {
        let beta = rho_hist[i] * dot(&y_hist[i], &r);
        r = add_scaled(&r, &s_hist[i], alpha_cache[i] - beta);
    }

    r
}

fn push_lbfgs_history(
    s_hist: &mut Vec<Vec<f64>>,
    y_hist: &mut Vec<Vec<f64>>,
    rho_hist: &mut Vec<f64>,
    s: Vec<f64>,
    y: Vec<f64>,
    sy: f64,
    max_m: usize,
) {
    if s_hist.len() >= max_m {
        s_hist.remove(0);
        y_hist.remove(0);
        rho_hist.remove(0);
    }
    s_hist.push(s);
    y_hist.push(y);
    rho_hist.push(1.0 / sy);
}

/// Newton-CG method: Newton's method with CG inner solver for the Newton equation.
///
/// Matches `scipy.optimize.minimize(f, x0, method='Newton-CG')`.
/// Uses finite-difference Hessian-vector products and CG to approximately
/// solve H*d = -g at each outer iteration.
pub fn newton_cg<F>(
    fun: &F,
    x0: &[f64],
    options: MinimizeOptions,
) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    validate_minimize_options(options)?;

    let n = x0.len();
    let tol = requested_tolerance(options.tol);
    let maxiter = options.maxiter.unwrap_or((200 * n).max(100));
    let maxfev = options.maxfev.unwrap_or((2000 * n).max(400));
    let eps = options.gradient_eps;
    let mut objective = Objective::new(fun, options.mode, maxfev);

    let mut x = x0.to_vec();
    let mut f = match objective.eval(&x) {
        Ok(v) => v,
        Err(err) => return Ok(result_from_error(x0, 0, 0, 0, err)),
    };
    let mut njev = 0usize;
    let mut nhev = 0usize;
    let mut grad = match finite_diff_gradient(&mut objective, &x, eps) {
        Ok(v) => {
            njev += 1;
            v
        }
        Err(err) => return Ok(result_from_error(&x, 0, objective.nfev, 0, err)),
    };

    for iteration in 0..maxiter {
        let grad_norm = l2_norm(&grad);
        if grad_norm <= tol {
            let result = OptimizeResult {
                x: x.clone(),
                fun: Some(f),
                success: true,
                status: ConvergenceStatus::Success,
                message: String::from("optimization converged (Newton-CG)"),
                nfev: objective.nfev,
                njev,
                nhev,
                nit: iteration,
                jac: Some(grad.clone()),
                hess_inv: None,
                maxcv: None,
            };
            log_completion(OptimizeMethod::NewtonCg, options, iteration, &result);
            return Ok(result);
        }

        if let Some(callback) = options.callback
            && !callback(&x)
        {
            let result = OptimizeResult {
                x: x.clone(),
                fun: Some(f),
                success: false,
                status: ConvergenceStatus::CallbackStop,
                message: String::from("callback requested stop"),
                nfev: objective.nfev,
                njev,
                nhev,
                nit: iteration,
                jac: Some(grad.clone()),
                hess_inv: None,
                maxcv: None,
            };
            log_completion(OptimizeMethod::NewtonCg, options, iteration, &result);
            return Ok(result);
        }

        // Inner CG loop to solve H*d = -g approximately
        // Using user-provided hessp or finite differences: H*v ≈ (∇f(x+εv) - ∇f(x)) / ε
        let cg_tol = grad_norm.min(0.5); // Eisenstat-Walker forcing term
        let (direction, nhvp) =
            match cg_newton_direction(&mut objective, &x, &grad, eps, cg_tol, n, options.hessp) {
                Ok(v) => v,
                Err(e) => return Ok(result_from_error(&x, iteration, objective.nfev, njev, e)),
            };
        if options.hessp.is_some() {
            nhev += nhvp;
        } else {
            // Each finite-difference HVP requires one gradient evaluation.
            njev += nhvp;
        }

        // Line search along direction
        let directional_deriv = dot(&grad, &direction);
        if directional_deriv >= 0.0 {
            // Not a descent direction — use steepest descent
            let neg_grad: Vec<f64> = grad.iter().map(|&g| -g).collect();
            let alpha = 1.0 / grad_norm.max(1.0);
            let candidate = add_scaled(&x, &neg_grad, alpha);
            match objective.eval(&candidate) {
                Ok(fv) => {
                    x = candidate;
                    f = fv;
                }
                Err(err) => {
                    return Ok(result_from_error(&x, iteration, objective.nfev, njev, err));
                }
            }
        } else {
            // Armijo backtracking
            let c1 = 1e-4;
            let mut alpha = 1.0;
            let mut accepted = false;
            for _ in 0..24 {
                let candidate = add_scaled(&x, &direction, alpha);
                match objective.eval(&candidate) {
                    Ok(fv) => {
                        if fv <= f + c1 * alpha * directional_deriv {
                            x = candidate;
                            f = fv;
                            accepted = true;
                            break;
                        }
                    }
                    Err(err) => {
                        return Ok(result_from_error(&x, iteration, objective.nfev, njev, err));
                    }
                }
                alpha *= 0.5;
                if alpha < 1e-12 {
                    break;
                }
            }
            if !accepted {
                break;
            }
        }

        // Update gradient
        grad = match finite_diff_gradient(&mut objective, &x, eps) {
            Ok(v) => {
                njev += 1;
                v
            }
            Err(err) => return Ok(result_from_error(&x, iteration, objective.nfev, njev, err)),
        };

        log_iteration(
            OptimizeMethod::NewtonCg,
            options,
            iteration,
            f,
            l2_norm(&grad),
            1.0,
            objective.nfev,
        );
    }

    let result = OptimizeResult {
        x,
        fun: Some(f),
        success: false,
        status: ConvergenceStatus::MaxIterations,
        message: format!("maximum iterations reached ({maxiter})"),
        nfev: objective.nfev,
        njev,
        nhev,
        nit: maxiter,
        jac: Some(grad),
        hess_inv: None,
        maxcv: None,
    };
    log_completion(OptimizeMethod::NewtonCg, options, maxiter, &result);
    Ok(result)
}

/// Trust-region exact method using finite-difference derivatives.
///
/// Matches `scipy.optimize.minimize(f, x0, method='trust-exact')` at the
/// observable API level, while approximating the missing Jacobian/Hessian
/// inputs from function evaluations.
pub fn trust_exact<F>(
    fun: &F,
    x0: &[f64],
    options: MinimizeOptions,
) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    validate_minimize_options(options)?;

    let n = x0.len();
    let tol = requested_tolerance(options.tol);
    let maxiter = options.maxiter.unwrap_or((200 * n).max(100));
    let maxfev = options.maxfev.unwrap_or((5000 * n).max(1_000));
    let eps = options.gradient_eps;
    let mut objective = Objective::new(fun, options.mode, maxfev);

    let mut x = x0.to_vec();
    let mut f = match objective.eval(&x) {
        Ok(v) => v,
        Err(err) => return Ok(result_from_error(x0, 0, 0, 0, err)),
    };
    let mut njev = 0usize;
    let mut nhev = 0usize;
    let mut grad = match finite_diff_gradient(&mut objective, &x, eps) {
        Ok(v) => {
            njev += 1;
            v
        }
        Err(err) => return Ok(result_from_error(&x, 0, objective.nfev, 0, err)),
    };

    let mut delta = l2_norm(&x).clamp(1.0, 10.0);
    let max_delta = 1_000.0;
    let acceptance_threshold = 0.1;
    let boundary_threshold = 0.9;
    let mut nit = 0usize;

    // Quasi-Newton (BFGS) Hessian approximation. The previous implementation rebuilt a
    // finite-difference Hessian every iteration — n Hessian-vector products, each a full
    // finite-difference gradient ≈ 2n function evals, so ≈ 2n² evals PER iteration, the
    // dominant cost. Instead we maintain a BFGS model updated from the gradient differences
    // already produced by the accepted steps: ZERO extra function/gradient evaluations for
    // the curvature. Started from the identity, BFGS stays symmetric positive-definite, so
    // the exact trust-region subproblem solver below always has a descent step. BFGS
    // converges to the same minimizer as the exact Hessian (verified to match the
    // finite-difference build within tolerance across a problem suite); only the
    // intermediate curvature model and the (now far smaller) eval count differ.
    // Started from ‖g₀‖/Δ₀ on the diagonal rather than from 1, which is the
    // difference between a model that knows what units this problem is in and
    // one that assumes they are order one.
    //
    // B carries units of f/x², ‖g‖ carries f/x and Δ carries x, so ‖g‖/Δ is
    // dimensionally the right diagonal — and it makes the first Newton step
    // −B⁻¹g come out with length Δ instead of length ‖g‖. That is not a
    // refinement, it is what makes the first step MEASURABLE: from the identity,
    // the first step on an objective scaled by 2^-53 has length ‖g‖ ≈ 6.7e-16,
    // so the function change over it is ≈ ‖g‖² ≈ 4.5e-31 against an ulp(f) of
    // ≈ 1.9e-31 — about two ulps, i.e. rounding noise. ρ is then meaningless,
    // the step is rejected, x never moves, and the curvature rescale below never
    // gets the first pair it needs (frankenscipy-uluc9).
    let initial_curvature = {
        let grad_norm = l2_norm(&grad);
        if grad_norm > 0.0 && delta > 0.0 {
            grad_norm / delta
        } else {
            1.0
        }
    };
    let mut hessian: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = vec![0.0; n];
            row[i] = initial_curvature;
            row
        })
        .collect();

    for iteration in 0..maxiter {
        let grad_norm = l2_norm(&grad);
        if grad_norm <= tol {
            let result = OptimizeResult {
                x: x.clone(),
                fun: Some(f),
                success: true,
                status: ConvergenceStatus::Success,
                message: String::from("optimization converged (trust-exact)"),
                nfev: objective.nfev,
                njev,
                nhev,
                nit: iteration,
                jac: Some(grad.clone()),
                hess_inv: None,
                maxcv: None,
            };
            log_completion(OptimizeMethod::TrustExact, options, iteration, &result);
            return Ok(result);
        }

        if let Some(callback) = options.callback
            && !callback(&x)
        {
            let result = OptimizeResult {
                x: x.clone(),
                fun: Some(f),
                success: false,
                status: ConvergenceStatus::CallbackStop,
                message: String::from("callback requested stop"),
                nfev: objective.nfev,
                njev,
                nhev,
                nit: iteration,
                jac: Some(grad.clone()),
                hess_inv: None,
                maxcv: None,
            };
            log_completion(OptimizeMethod::TrustExact, options, iteration, &result);
            return Ok(result);
        }

        let mut step = trust_region_exact_step(&grad, &hessian, delta);
        let mut predicted_reduction = trust_model_reduction(&grad, &hessian, &step);
        let mut step_norm = l2_norm(&step);

        if !predicted_reduction.is_finite() || predicted_reduction <= 0.0 || step_norm == 0.0 {
            step = steepest_descent_step(&grad, delta);
            predicted_reduction = trust_model_reduction(&grad, &hessian, &step);
            step_norm = l2_norm(&step);
        }

        if !predicted_reduction.is_finite() || predicted_reduction <= 0.0 || step_norm == 0.0 {
            let result = OptimizeResult {
                x: x.clone(),
                fun: Some(f),
                success: false,
                status: ConvergenceStatus::PrecisionLoss,
                message: String::from("trust-exact failed to produce a descent step"),
                nfev: objective.nfev,
                njev,
                nhev,
                nit: iteration,
                jac: Some(grad.clone()),
                hess_inv: None,
                maxcv: None,
            };
            log_completion(OptimizeMethod::TrustExact, options, iteration, &result);
            return Ok(result);
        }

        let candidate = add_scaled(&x, &step, 1.0);
        let candidate_f = match objective.eval(&candidate) {
            Ok(v) => v,
            Err(err) => return Ok(result_from_error(&x, iteration, objective.nfev, njev, err)),
        };

        let actual_reduction = f - candidate_f;
        let rho = actual_reduction / predicted_reduction;

        if rho < 0.25 {
            delta = (0.25 * delta).max(1.0e-8);
        } else if rho > 0.75 && step_norm >= boundary_threshold * delta {
            delta = (2.0 * delta).min(max_delta);
        }

        if rho > acceptance_threshold {
            let grad_old = grad.clone();
            x = candidate;
            f = candidate_f;
            grad = match finite_diff_gradient(&mut objective, &x, eps) {
                Ok(v) => {
                    njev += 1;
                    v
                }
                Err(err) => {
                    return Ok(result_from_error(&x, iteration, objective.nfev, njev, err));
                }
            };
            // BFGS curvature update from the step actually taken (s) and the resulting
            // gradient change (y) — both already in hand, so no extra evaluations.
            let y: Vec<f64> = grad
                .iter()
                .zip(grad_old.iter())
                .map(|(gn, go)| gn - go)
                .collect();
            trust_bfgs_hessian_update(&mut hessian, &step, &y, nhev == 0);
            nhev += 1;
            nit = iteration + 1;
        } else if delta <= 1.0e-8 {
            let result = OptimizeResult {
                x: x.clone(),
                fun: Some(f),
                success: false,
                status: ConvergenceStatus::PrecisionLoss,
                message: String::from("trust region radius collapsed without an acceptable step"),
                nfev: objective.nfev,
                njev,
                nhev,
                nit: iteration,
                jac: Some(grad.clone()),
                hess_inv: None,
                maxcv: None,
            };
            log_completion(OptimizeMethod::TrustExact, options, iteration, &result);
            return Ok(result);
        }

        log_iteration(
            OptimizeMethod::TrustExact,
            options,
            iteration + 1,
            f,
            l2_norm(&grad),
            step_norm,
            objective.nfev,
        );
    }

    let result = OptimizeResult {
        x,
        fun: Some(f),
        success: false,
        status: ConvergenceStatus::MaxIterations,
        message: format!("maximum iterations reached ({maxiter})"),
        nfev: objective.nfev,
        njev,
        nhev,
        nit,
        jac: Some(grad),
        hess_inv: None,
        maxcv: None,
    };
    log_completion(OptimizeMethod::TrustExact, options, nit, &result);
    Ok(result)
}

/// Inner CG solver for Newton equation H*d = -g.
/// Uses Hessian-vector products via finite differences.
/// Returns (direction, number_of_hvps).
fn cg_newton_direction<F>(
    objective: &mut Objective<'_, F>,
    x: &[f64],
    grad: &[f64],
    eps: f64,
    tol: f64,
    n: usize,
    hessp: Option<crate::types::HesspFunc>,
) -> Result<(Vec<f64>, usize), OptError>
where
    F: Fn(&[f64]) -> f64,
{
    // CG to solve H*d = -g
    let neg_g: Vec<f64> = grad.iter().map(|&g| -g).collect();
    let mut d = vec![0.0; n];
    let mut r = neg_g.clone(); // r = -g - H*d = -g (since d=0)
    let mut p = r.clone();
    let mut rs = dot(&r, &r);
    let mut nhvp = 0usize;

    let max_cg_iter = n.min(20);
    let target = tol * tol * dot(grad, grad);

    for _ in 0..max_cg_iter {
        if rs < target {
            break;
        }

        // Hessian-vector product: user-provided or via finite differences
        let hp = match hessp {
            Some(func) => {
                nhvp += 1;
                func(x, &p)
            }
            None => {
                let v = hessian_vector_product(objective, x, grad, &p, eps)?;
                nhvp += 1;
                v
            }
        };

        let p_hp = dot(&p, &hp);
        if p_hp <= 0.0 {
            // Negative curvature — use current d if nonzero, else use steepest descent
            if dot(&d, &d) > 0.0 {
                return Ok((d, nhvp));
            }
            return Ok((neg_g, nhvp));
        }

        let alpha = rs / p_hp;
        for i in 0..n {
            d[i] += alpha * p[i];
            r[i] -= alpha * hp[i];
        }

        let rs_new = dot(&r, &r);
        let beta = rs_new / rs;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rs = rs_new;
    }

    Ok((d, nhvp))
}

/// Compute Hessian-vector product H*v via finite differences of gradient.
fn hessian_vector_product<F>(
    objective: &mut Objective<'_, F>,
    x: &[f64],
    grad_at_x: &[f64],
    v: &[f64],
    eps: f64,
) -> Result<Vec<f64>, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    let v_norm = l2_norm(v);
    // Exact zero: the guard is on the division forming the perturbation step,
    // and a direction of any nonzero length has a well-defined normalization.
    if v_norm == 0.0 {
        return Ok(vec![0.0; x.len()]);
    }
    let step = eps * (1.0 + l2_norm(x)) / v_norm;
    let x_pert = add_scaled(x, v, step);
    let grad_pert = finite_diff_gradient(objective, &x_pert, eps)?;
    let mut hv = Vec::with_capacity(x.len());
    for (gp, g) in grad_pert.iter().zip(grad_at_x.iter()) {
        hv.push((gp - g) / step);
    }
    Ok(hv)
}

/// Forward BFGS update of the Hessian approximation `b`, in place, from the step
/// `s = x_new - x_old` and the gradient change `y = grad_new - grad_old`:
///
/// `B <- B − (B·s)(B·s)ᵀ / (sᵀ·B·s) + (y·yᵀ) / (yᵀ·s)`.
///
/// Started from a symmetric positive-definite matrix, BFGS keeps `b` SPD as long as the
/// curvature condition `yᵀs > 0` holds; the update is skipped otherwise (Nocedal–Wright
/// §6.1 safeguard). An SPD model guarantees the trust-region subproblem below always has a
/// descent step, which the indefinite SR1 update does not — hence BFGS here.
fn trust_bfgs_hessian_update(b: &mut [Vec<f64>], s: &[f64], y: &[f64], is_first_update: bool) {
    let n = s.len();
    let y_s = dot(y, s); // yᵀ s
    let y_norm = l2_norm(y);
    let s_norm = l2_norm(s);
    if y_s <= 1.0e-10 * y_norm * s_norm {
        return; // curvature condition fails → keep the current model
    }

    // Rescale the identity start to the curvature this problem actually has,
    // once, from the first curvature pair: B₀ ← (yᵀy / yᵀs)·I (Nocedal & Wright
    // §6.1, eq. 6.20). The identity is a statement that the curvature is order
    // one, which is a statement about the objective's SCALING — for an ordinary
    // objective multiplied by 2^-53 the true curvature is order 1e-16 and the
    // model is sixteen orders too stiff, so every trust-region step is
    // proportionally too short and the model needs thousands of updates to
    // catch up (frankenscipy-uluc9).
    //
    // Measured on worker vmi1227854 before this scaling, f(x) = s·((x₀−1)² +
    // 2(x₁+2)²) from (0,0): scale 1 converged exactly in 23 iterations and
    // 2^-40 in 53, while 2^-53 was still 2.056 from the minimizer after 200
    // iterations and 5.571e-1 after 5000 — converging, but sublinearly. A line
    // search hides this (SciPy's BFGS also starts from the identity and gets
    // away with it, because its step LENGTH adapts); a trust region does not,
    // because within the radius it is the model alone that sizes the step.
    if is_first_update {
        let gamma = dot(y, y) / y_s;
        if gamma.is_finite() && gamma > 0.0 {
            for (i, row) in b.iter_mut().enumerate() {
                for (j, entry) in row.iter_mut().enumerate() {
                    *entry = if i == j { gamma } else { 0.0 };
                }
            }
        }
    }

    let bs = matrix_vector_mul(b, s);
    let s_bs = dot(s, &bs); // sᵀ B s
    if s_bs <= 0.0 {
        return; // model curvature along s is unusable → keep the current model
    }
    let inv_sbs = 1.0 / s_bs;
    let inv_ys = 1.0 / y_s;
    for i in 0..n {
        let yi = y[i];
        let bsi = bs[i];
        let row = &mut b[i];
        for j in 0..n {
            row[j] += yi * y[j] * inv_ys - bsi * bs[j] * inv_sbs;
        }
    }
}

fn trust_region_exact_step(grad: &[f64], hessian: &[Vec<f64>], delta: f64) -> Vec<f64> {
    let n = grad.len();
    let grad_norm = l2_norm(grad);
    // Exact zero, not `f64::EPSILON`: this guards the division below, and only a
    // zero gradient makes it undefined — a zero gradient is also the one case
    // where a zero step is the right answer, because x is already stationary.
    // Against an absolute floor, an objective scaled so its gradient is ~1e-16
    // could not take a step at all, and once frankenscipy-ei0az stopped clamping
    // the caller's tolerance this became reachable: trust-exact reached 1.165e-5
    // from the minimizer at scale 2^-40 and then stopped with a zero step.
    if grad_norm == 0.0 {
        return vec![0.0; n];
    }

    let rhs = scale_vector(grad, -1.0);
    if let Some(newton_step) = solve_trust_spd_system(hessian, 0.0, &rhs)
        && l2_norm(&newton_step) <= delta
        && trust_model_reduction(grad, hessian, &newton_step) > 0.0
    {
        return newton_step;
    }

    // λ is added to the Hessian's diagonal, so it carries the Hessian's units.
    // Starting the search at a bare `1.0e-6` is therefore only sensible when the
    // model is order one: against a 2^-53-scaled objective, whose curvature is
    // order 1e-16, the first trial shift was ten orders larger than the entire
    // matrix, so every candidate solved a system that was almost purely the
    // shift (frankenscipy-uluc9). Anchoring the start to the model's largest
    // diagonal leaves a well-scaled problem starting at exactly 1e-6 as before.
    let model_scale = (0..n).fold(0.0_f64, |largest, i| largest.max(hessian[i][i].abs()));
    let mut lower = 0.0;
    let mut upper = if model_scale > 0.0 {
        1.0e-6 * model_scale
    } else {
        1.0e-6
    };
    let mut boundary_step = None;

    for _ in 0..60 {
        let candidate_opt = solve_trust_spd_system(hessian, upper, &rhs);
        if let Some(candidate) = candidate_opt {
            let norm = l2_norm(&candidate);
            if norm <= delta {
                boundary_step = Some((upper, candidate));
                break;
            }
            lower = upper;
        } else {
            lower = upper;
        }
        upper *= 2.0;
    }

    if let Some((mut hi_lambda, mut hi_step)) = boundary_step {
        let mut lo_lambda = lower;
        for _ in 0..60 {
            let mid_lambda = 0.5 * (lo_lambda + hi_lambda);
            let candidate_opt = solve_trust_spd_system(hessian, mid_lambda, &rhs);
            let Some(candidate) = candidate_opt else {
                lo_lambda = mid_lambda;
                continue;
            };
            let norm = l2_norm(&candidate);
            if (norm - delta).abs() <= 1.0e-10 * delta.max(1.0) {
                hi_step = candidate;
                break;
            }
            if norm > delta {
                lo_lambda = mid_lambda;
            } else {
                hi_lambda = mid_lambda;
                hi_step = candidate;
            }
        }

        if trust_model_reduction(grad, hessian, &hi_step) > 0.0 {
            return hi_step;
        }
    }

    steepest_descent_step(grad, delta)
}

fn steepest_descent_step(grad: &[f64], delta: f64) -> Vec<f64> {
    let grad_norm = l2_norm(grad);
    // Exact zero for the same reason as `trust_region_exact_step`: the guard is
    // on the division by `grad_norm`, and the size of the gradient is a fact
    // about the objective's scaling, not about whether a step exists.
    if grad_norm == 0.0 {
        return vec![0.0; grad.len()];
    }
    scale_vector(grad, -delta / grad_norm)
}

fn trust_model_reduction(grad: &[f64], hessian: &[Vec<f64>], step: &[f64]) -> f64 {
    let hessian_step = matrix_vector_mul(hessian, step);
    -(dot(grad, step) + 0.5 * dot(step, &hessian_step))
}

fn shifted_matrix(matrix: &[Vec<f64>], lambda: f64) -> Vec<Vec<f64>> {
    let mut shifted = matrix.to_vec();
    for (idx, row) in shifted.iter_mut().enumerate() {
        row[idx] += lambda;
    }
    shifted
}

/// When `true`, the exact trust-region subproblem builds the shifted matrix `H + λI` into a
/// fresh `Vec<Vec<f64>>` (via [`shifted_matrix`]) before solving (the ORIG behaviour); default
/// `false` folds the `+λ` diagonal shift directly into the solver's augmented matrix, dropping
/// one O(n²) copy of the Hessian per λ trial (the subproblem does up to ~120 solves per step).
/// Byte-identical — the augmented matrix is the same. `#[doc(hidden)]` — same-binary A/B knob.
#[doc(hidden)]
pub static TRUST_EXACT_FOLD_SHIFT_DISABLE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// When `true`, linear solves retain the original row-segmented `Vec<Vec<f64>>` augmented
/// matrix. The default stores the same row-major values in one contiguous allocation, avoiding
/// `n` row allocations and pointer-chasing while preserving every arithmetic operation.
/// `#[doc(hidden)]` — same-binary A/B knob.
#[doc(hidden)]
pub static TRUST_EXACT_FLAT_AUGMENTED_DISABLE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// When `true`, trust-exact retains the pivoted Gauss-Jordan solve for its BFGS Hessian
/// systems. The default first uses an SPD Cholesky factor and triangular solves, then falls
/// back to Gauss-Jordan if the factorization detects a non-positive or non-finite pivot.
/// `#[doc(hidden)]` — same-binary A/B knob.
#[doc(hidden)]
pub static TRUST_EXACT_CHOLESKY_DISABLE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Gauss-Jordan solve of a prebuilt `n × (n+1)` augmented `[A | b]` (partial pivoting +
/// back-substitution). Shared by [`solve_linear_system`] and [`solve_shifted_system`].
fn solve_augmented_nested(mut aug: Vec<Vec<f64>>, n: usize) -> Option<Vec<f64>> {
    for pivot in 0..n {
        let mut pivot_row = pivot;
        let mut pivot_abs = aug[pivot][pivot].abs();
        for (row, candidate_row) in aug.iter().enumerate().skip(pivot + 1) {
            let candidate = candidate_row[pivot].abs();
            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = row;
            }
        }

        if pivot_abs <= 1.0e-12 {
            return None;
        }

        if pivot_row != pivot {
            aug.swap(pivot, pivot_row);
        }

        let pivot_value = aug[pivot][pivot];
        let (head, tail) = aug.split_at_mut(pivot + 1);
        let pivot_row_values = &head[pivot];
        for current_row in tail.iter_mut() {
            let factor = current_row[pivot] / pivot_value;
            current_row[pivot] = 0.0;
            for (entry, pivot_entry) in current_row
                .iter_mut()
                .skip(pivot + 1)
                .zip(pivot_row_values.iter().skip(pivot + 1))
            {
                *entry -= factor * pivot_entry;
            }
        }
    }

    let mut solution = vec![0.0; n];
    for row in (0..n).rev() {
        let mut value = aug[row][n];
        for column in row + 1..n {
            value -= aug[row][column] * solution[column];
        }
        let denom = aug[row][row];
        if denom.abs() <= 1.0e-12 {
            return None;
        }
        solution[row] = value / denom;
    }

    Some(solution)
}

/// Contiguous-storage twin of [`solve_augmented_nested`]. Rows, pivot candidates, elimination
/// columns, and back-substitution columns are visited in exactly the same order.
fn solve_augmented_flat(mut aug: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    let stride = n + 1;
    debug_assert_eq!(aug.len(), n * stride);

    for pivot in 0..n {
        let mut pivot_row = pivot;
        let mut pivot_abs = aug[pivot * stride + pivot].abs();
        for row in pivot + 1..n {
            let candidate = aug[row * stride + pivot].abs();
            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = row;
            }
        }

        if pivot_abs <= 1.0e-12 {
            return None;
        }

        if pivot_row != pivot {
            let pivot_base = pivot * stride;
            let pivot_row_base = pivot_row * stride;
            for column in 0..stride {
                aug.swap(pivot_base + column, pivot_row_base + column);
            }
        }

        let pivot_base = pivot * stride;
        let pivot_value = aug[pivot_base + pivot];
        for row in pivot + 1..n {
            let row_base = row * stride;
            let factor = aug[row_base + pivot] / pivot_value;
            aug[row_base + pivot] = 0.0;
            for column in pivot + 1..stride {
                let pivot_entry = aug[pivot_base + column];
                aug[row_base + column] -= factor * pivot_entry;
            }
        }
    }

    let mut solution = vec![0.0; n];
    for row in (0..n).rev() {
        let row_base = row * stride;
        let mut value = aug[row_base + n];
        for column in row + 1..n {
            value -= aug[row_base + column] * solution[column];
        }
        let denom = aug[row_base + row];
        if denom.abs() <= 1.0e-12 {
            return None;
        }
        solution[row] = value / denom;
    }

    Some(solution)
}

fn solve_linear_system(matrix: &[Vec<f64>], rhs: &[f64]) -> Option<Vec<f64>> {
    let n = matrix.len();
    if rhs.len() != n || matrix.iter().any(|row| row.len() != n) {
        return None;
    }

    if TRUST_EXACT_FLAT_AUGMENTED_DISABLE.load(std::sync::atomic::Ordering::Relaxed) {
        let mut aug = vec![vec![0.0; n + 1]; n];
        for row in 0..n {
            aug[row][..n].copy_from_slice(&matrix[row][..n]);
            aug[row][n] = rhs[row];
        }
        return solve_augmented_nested(aug, n);
    }

    let stride = n + 1;
    let mut aug = vec![0.0; n * stride];
    for row in 0..n {
        let row_base = row * stride;
        aug[row_base..row_base + n].copy_from_slice(&matrix[row][..n]);
        aug[row_base + n] = rhs[row];
    }
    solve_augmented_flat(aug, n)
}

/// Solve `(matrix + lambda·I)·x = rhs` by folding the diagonal shift into the augmented
/// matrix directly — byte-identical to `solve_linear_system(&shifted_matrix(matrix, lambda),
/// rhs)` (same `[matrix + lambda·I | rhs]`) but without the extra full-matrix copy.
fn solve_shifted_system(matrix: &[Vec<f64>], lambda: f64, rhs: &[f64]) -> Option<Vec<f64>> {
    let n = matrix.len();
    if rhs.len() != n || matrix.iter().any(|row| row.len() != n) {
        return None;
    }
    if TRUST_EXACT_FLAT_AUGMENTED_DISABLE.load(std::sync::atomic::Ordering::Relaxed) {
        let mut aug = vec![vec![0.0; n + 1]; n];
        for row in 0..n {
            aug[row][..n].copy_from_slice(&matrix[row][..n]);
            aug[row][row] += lambda;
            aug[row][n] = rhs[row];
        }
        return solve_augmented_nested(aug, n);
    }

    let stride = n + 1;
    let mut aug = vec![0.0; n * stride];
    for row in 0..n {
        let row_base = row * stride;
        aug[row_base..row_base + n].copy_from_slice(&matrix[row][..n]);
        aug[row_base + row] += lambda;
        aug[row_base + n] = rhs[row];
    }
    solve_augmented_flat(aug, n)
}

fn solve_trust_pivoted_system(matrix: &[Vec<f64>], lambda: f64, rhs: &[f64]) -> Option<Vec<f64>> {
    if lambda == 0.0 {
        solve_linear_system(matrix, rhs)
    } else if TRUST_EXACT_FOLD_SHIFT_DISABLE.load(std::sync::atomic::Ordering::Relaxed) {
        solve_linear_system(&shifted_matrix(matrix, lambda), rhs)
    } else {
        solve_shifted_system(matrix, lambda, rhs)
    }
}

/// Solve the symmetric positive-definite BFGS system `(matrix + lambda·I) x = rhs`.
///
/// Trust-exact starts its BFGS Hessian at identity and applies only curvature-safe updates,
/// so this is the native factorization for its subproblem. A failed positivity/finite check
/// falls back to the existing pivoted solve, preserving the hardened numerical path.
fn solve_trust_spd_system(matrix: &[Vec<f64>], lambda: f64, rhs: &[f64]) -> Option<Vec<f64>> {
    if TRUST_EXACT_CHOLESKY_DISABLE.load(std::sync::atomic::Ordering::Relaxed) {
        return solve_trust_pivoted_system(matrix, lambda, rhs);
    }

    let n = matrix.len();
    if rhs.len() != n || matrix.iter().any(|row| row.len() != n) {
        return None;
    }

    // The positivity floor is relative to the model's own magnitude. A bare
    // `1.0e-12` is a claim that the Hessian's entries are order one, which is a
    // claim about the objective's SCALING: for an objective scaled by 2^-53 the
    // true curvature is order 1e-16, so every pivot looked degenerate and this
    // factorization bailed to the pivoted fallback on every single subproblem
    // solve (frankenscipy-uluc9). Multiplying by the largest diagonal leaves a
    // well-scaled model on exactly the old threshold — `max_diagonal` is order
    // one there — while making the question scale-free. With `max_diagonal` at
    // zero the floor is zero, which is still the right positive-definiteness
    // test for `value <= floor`.
    let max_diagonal = (0..n)
        .fold(0.0_f64, |largest, i| largest.max(matrix[i][i].abs()))
        .max(lambda.abs());
    let pivot_floor = 1.0e-12 * max_diagonal;
    // L's entries carry units of √B, so their floor scales as √ of the model's,
    // and at `max_diagonal` = 1 it is the same 1e-12 this line always used.
    let factor_floor = 1.0e-12 * max_diagonal.sqrt();

    let mut lower = vec![0.0; n * n];
    for row in 0..n {
        for column in 0..=row {
            let mut value = matrix[row][column];
            if row == column {
                value += lambda;
            }
            for inner in 0..column {
                value -= lower[row * n + inner] * lower[column * n + inner];
            }

            if row == column {
                if !value.is_finite() || value <= pivot_floor {
                    return solve_trust_pivoted_system(matrix, lambda, rhs);
                }
                lower[row * n + column] = value.sqrt();
            } else {
                let diagonal = lower[column * n + column];
                if !value.is_finite() || !diagonal.is_finite() || diagonal <= factor_floor {
                    return solve_trust_pivoted_system(matrix, lambda, rhs);
                }
                lower[row * n + column] = value / diagonal;
            }
        }
    }

    let mut intermediate = vec![0.0; n];
    for row in 0..n {
        let mut value = rhs[row];
        for column in 0..row {
            value -= lower[row * n + column] * intermediate[column];
        }
        value /= lower[row * n + row];
        if !value.is_finite() {
            return solve_trust_pivoted_system(matrix, lambda, rhs);
        }
        intermediate[row] = value;
    }

    let mut solution = vec![0.0; n];
    for row in (0..n).rev() {
        let mut value = intermediate[row];
        for column in row + 1..n {
            value -= lower[column * n + row] * solution[column];
        }
        value /= lower[row * n + row];
        if !value.is_finite() {
            return solve_trust_pivoted_system(matrix, lambda, rhs);
        }
        solution[row] = value;
    }
    Some(solution)
}

pub fn get_optimize_traces() -> Vec<OptimizeTraceEntry> {
    // Resolves [frankenscipy-pvrjd]: previously this returned Vec::new()
    // on poison, silently dropping the entire trace history even though
    // push_trace recovers and keeps writing through poison. Mirror the
    // write-side recovery so a single panic in trace machinery does not
    // make all subsequent reads appear empty.
    let log = trace_log();
    match log.lock() {
        Ok(guard) => guard.clone(),
        Err(poisoned) => {
            log.clear_poison();
            poisoned.into_inner().clone()
        }
    }
}

#[derive(Debug, Clone)]
struct LineSearchStep {
    alpha: f64,
    x: Vec<f64>,
    f: f64,
    accepted_gradient: Option<Vec<f64>>,
}

struct Objective<'a, F>
where
    F: Fn(&[f64]) -> f64,
{
    fun: &'a F,
    mode: fsci_runtime::RuntimeMode,
    maxfev: usize,
    nfev: usize,
}

impl<'a, F> Objective<'a, F>
where
    F: Fn(&[f64]) -> f64,
{
    fn new(fun: &'a F, mode: fsci_runtime::RuntimeMode, maxfev: usize) -> Self {
        Self {
            fun,
            mode,
            maxfev: maxfev.max(1),
            nfev: 0,
        }
    }

    fn eval(&mut self, x: &[f64]) -> Result<f64, OptError> {
        if self.nfev >= self.maxfev {
            return Err(OptError::EvaluationBudgetExceeded {
                detail: format!("max function evaluations exceeded ({})", self.maxfev),
            });
        }
        let value = (self.fun)(x);
        self.nfev += 1;
        if !value.is_finite() {
            return match self.mode {
                fsci_runtime::RuntimeMode::Strict => Err(OptError::InvalidArgument {
                    detail: String::from("objective evaluated to non-finite value"),
                }),
                fsci_runtime::RuntimeMode::Hardened => Err(OptError::NonFiniteInput {
                    detail: String::from("hardened mode rejects non-finite objective values"),
                }),
            };
        }
        Ok(value)
    }

    fn reserve_evaluations(&mut self, count: usize) -> Result<(), OptError> {
        let remaining = self.maxfev.saturating_sub(self.nfev);
        if remaining < count {
            self.nfev = self.maxfev;
            return Err(OptError::EvaluationBudgetExceeded {
                detail: format!("max function evaluations exceeded ({})", self.maxfev),
            });
        }
        self.nfev += count;
        Ok(())
    }
}

fn finite_diff_gradient<F>(
    objective: &mut Objective<'_, F>,
    x: &[f64],
    gradient_eps: f64,
) -> Result<Vec<f64>, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    let mut gradient = vec![0.0; x.len()];
    let mut x_perturbed = x.to_vec();
    for (idx, component) in x.iter().enumerate() {
        let step = gradient_eps * (1.0 + component.abs());

        let original = x_perturbed[idx];
        x_perturbed[idx] = original + step;
        let f_plus = objective.eval(&x_perturbed)?;

        x_perturbed[idx] = original - step;
        let f_minus = objective.eval(&x_perturbed)?;

        x_perturbed[idx] = original; // restore
        gradient[idx] = (f_plus - f_minus) / (2.0 * step);
    }
    Ok(gradient)
}

fn evaluate_minimize_gradient<F>(
    objective: &mut Objective<'_, F>,
    gradient: Option<GradientFunc>,
    x: &[f64],
    gradient_eps: f64,
) -> Result<Vec<f64>, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    if let Some(gradient_fn) = gradient {
        validate_gradient_output(gradient_fn(x), x.len())
    } else {
        finite_diff_gradient(objective, x, gradient_eps)
    }
}

fn validate_gradient_output(gradient: Vec<f64>, expected_len: usize) -> Result<Vec<f64>, OptError> {
    if gradient.len() != expected_len {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "gradient callback returned length {}, expected {expected_len}",
                gradient.len()
            ),
        });
    }
    if gradient.iter().any(|value| !value.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: String::from("gradient callback returned NaN or Inf"),
        });
    }
    Ok(gradient)
}

fn armijo_backtracking<F>(
    objective: &mut Objective<'_, F>,
    x: &[f64],
    fx: f64,
    grad: &[f64],
    direction: &[f64],
) -> Result<Option<LineSearchStep>, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    let directional_derivative = dot(grad, direction);
    if directional_derivative >= 0.0 {
        return Ok(None);
    }

    let c1 = 1.0e-4;
    let mut alpha = 1.0;
    for _ in 0..24 {
        let candidate_x = add_scaled(x, direction, alpha);
        let candidate_f = objective.eval(&candidate_x)?;
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

fn golden_section_direction_search<F>(
    objective: &mut Objective<'_, F>,
    x: &[f64],
    fx: f64,
    direction: &[f64],
    tolerance: f64,
) -> Result<LineSearchStep, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    const MAX_BRACKET_STEP: f64 = 1.0e10;
    let mut candidate_x = vec![0.0; x.len()];

    // 1. Bracket the minimum along the direction
    // Simple bracketing: start with small step, expand until we find an increase
    let mut b = 1.0;
    add_scaled_into(&mut candidate_x, x, direction, b);
    let mut fb = objective.eval(&candidate_x)?;

    let (mut left, mut right) = if fb > fx {
        // Try other direction
        b = -1.0;
        add_scaled_into(&mut candidate_x, x, direction, b);
        fb = objective.eval(&candidate_x)?;
        if fb > fx {
            // Already bracketed by (-1, 1)?
            (-1.0, 1.0)
        } else {
            // Decreasing in negative direction
            let mut step = -2.0;
            loop {
                add_scaled_into(&mut candidate_x, x, direction, step);
                let f_next = objective.eval(&candidate_x)?;
                if f_next > fb {
                    break (step, b);
                }
                b = step;
                fb = f_next;
                step *= 2.0;
                if step.abs() > MAX_BRACKET_STEP {
                    add_scaled_into(&mut candidate_x, x, direction, b);
                    return Ok(LineSearchStep {
                        alpha: b,
                        x: candidate_x,
                        f: fb,
                        accepted_gradient: None,
                    });
                }
            }
        }
    } else {
        // Decreasing in positive direction
        let mut step = 2.0;
        loop {
            add_scaled_into(&mut candidate_x, x, direction, step);
            let f_next = objective.eval(&candidate_x)?;
            if f_next > fb {
                // fb is already lower than fa, and f_next > fb, so we have a bracket [a_orig, step]
                // but specifically [0, step] contains a minimum since f(0) > f(1) and f(step) > f(1)
                break (0.0, step);
            }
            b = step;
            fb = f_next;
            step *= 2.0;
            if step > MAX_BRACKET_STEP {
                add_scaled_into(&mut candidate_x, x, direction, b);
                return Ok(LineSearchStep {
                    alpha: b,
                    x: candidate_x,
                    f: fb,
                    accepted_gradient: None,
                });
            }
        }
    };

    let phi = 0.5 * (5.0_f64.sqrt() - 1.0);
    let mut c = right - phi * (right - left);
    let mut d = left + phi * (right - left);

    add_scaled_into(&mut candidate_x, x, direction, c);
    let mut fc = objective.eval(&candidate_x)?;
    add_scaled_into(&mut candidate_x, x, direction, d);
    let mut fd = objective.eval(&candidate_x)?;

    for _ in 0..60 {
        if (right - left).abs() <= tolerance * (1.0 + left.abs().max(right.abs())) {
            break;
        }
        if fc < fd {
            right = d;
            d = c;
            fd = fc;
            c = right - phi * (right - left);
            add_scaled_into(&mut candidate_x, x, direction, c);
            fc = objective.eval(&candidate_x)?;
        } else {
            left = c;
            c = d;
            fc = fd;
            d = left + phi * (right - left);
            add_scaled_into(&mut candidate_x, x, direction, d);
            fd = objective.eval(&candidate_x)?;
        }
    }

    let alpha = 0.5 * (left + right);
    add_scaled_into(&mut candidate_x, x, direction, alpha);
    let candidate_f = objective.eval(&candidate_x)?;
    if candidate_f <= fx {
        return Ok(LineSearchStep {
            alpha,
            x: candidate_x,
            f: candidate_f,
            accepted_gradient: None,
        });
    }

    Ok(LineSearchStep {
        alpha: 0.0,
        x: x.to_vec(),
        f: fx,
        accepted_gradient: None,
    })
}

fn result_from_error(
    x: &[f64],
    nit: usize,
    nfev: usize,
    njev: usize,
    error: OptError,
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
    };
    OptimizeResult {
        x: x.to_vec(),
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

fn validate_minimize_options(options: MinimizeOptions) -> Result<(), OptError> {
    if let Some(maxiter) = options.maxiter
        && maxiter == 0
    {
        return Err(OptError::InvalidArgument {
            detail: String::from("maxiter must be >= 1"),
        });
    }
    if let Some(maxfev) = options.maxfev
        && maxfev == 0
    {
        return Err(OptError::InvalidArgument {
            detail: String::from("maxfev must be >= 1"),
        });
    }
    if let Some(tol) = options.tol
        && (!tol.is_finite() || tol <= 0.0)
    {
        return Err(OptError::InvalidArgument {
            detail: String::from("tol must be finite and > 0"),
        });
    }
    if !options.gradient_eps.is_finite() || options.gradient_eps <= 0.0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("gradient_eps must be finite and > 0"),
        });
    }
    Ok(())
}

fn validate_bounds_for_x0(x0: &[f64], bounds: Option<&[Bound]>) -> Result<(), OptError> {
    let Some(bounds) = bounds else {
        return Ok(());
    };
    if bounds.len() != x0.len() {
        return Err(OptError::InvalidBounds {
            detail: format!(
                "bounds length must match x0 length (got {} and {})",
                bounds.len(),
                x0.len()
            ),
        });
    }
    for (index, (lo, hi)) in bounds.iter().enumerate() {
        if let Some(lo) = lo
            && !lo.is_finite()
        {
            return Err(OptError::InvalidBounds {
                detail: format!("lower bound at index {index} must be finite when present"),
            });
        }
        if let Some(hi) = hi
            && !hi.is_finite()
        {
            return Err(OptError::InvalidBounds {
                detail: format!("upper bound at index {index} must be finite when present"),
            });
        }
        if let (Some(lo), Some(hi)) = (lo, hi)
            && lo > hi
        {
            return Err(OptError::InvalidBounds {
                detail: format!("lower bound {lo} exceeds upper bound {hi} at index {index}"),
            });
        }
    }
    Ok(())
}

fn bounds_have_finite_limit(bounds: &[Bound]) -> bool {
    bounds.iter().any(|(lo, hi)| lo.is_some() || hi.is_some())
}

fn bfgs_inverse_update(h_inv: &[Vec<f64>], s: &[f64], y: &[f64], rho: f64) -> Vec<Vec<f64>> {
    let n = s.len();

    // H_new = (I - rho*s*y^T) * H * (I - rho*y*s^T) + rho*s*s^T
    //
    // Let A = (I - rho*s*y^T) * H = H - rho*s*(y^T * H)
    // Then H_new = A * (I - rho*y*s^T) + rho*s*s^T = A - rho*(A*y)*s^T + rho*s*s^T

    // 1. v^T = y^T * H  (O(n^2))
    let mut v = vec![0.0; n];
    for j in 0..n {
        for i in 0..n {
            v[j] += y[i] * h_inv[i][j];
        }
    }

    // 2. A = H - rho * s * v^T  (O(n^2))
    let mut a = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            a[i][j] = h_inv[i][j] - rho * s[i] * v[j];
        }
    }

    // 3. u = A * y  (O(n^2))
    let mut u = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            u[i] += a[i][j] * y[j];
        }
    }

    // 4. H_new = A - rho * u * s^T + rho * s * s^T (O(n^2))
    let mut h_new = a;
    for i in 0..n {
        for j in 0..n {
            h_new[i][j] += rho * (s[i] * s[j] - u[i] * s[j]);
        }
    }

    h_new
}

fn log_iteration(
    method: OptimizeMethod,
    options: MinimizeOptions,
    iter_num: usize,
    f_val: f64,
    grad_norm: f64,
    step_size: f64,
    total_nfev: usize,
) {
    let trace = OptimizeTraceEntry {
        ts_unix_ms: now_unix_ms(),
        event: String::from("iteration"),
        method,
        iter_num,
        f_val: Some(f_val),
        grad_norm: Some(grad_norm),
        step_size: Some(step_size),
        mode: options.mode,
        reason: None,
        final_x: None,
        final_f: None,
        total_nfev,
        fixture_id: options.fixture_id.map(ToOwned::to_owned),
        seed: options.seed,
    };
    push_trace(trace);
}

fn log_completion(
    method: OptimizeMethod,
    options: MinimizeOptions,
    iter_num: usize,
    result: &OptimizeResult,
) {
    let trace = OptimizeTraceEntry {
        ts_unix_ms: now_unix_ms(),
        event: String::from("completion"),
        method,
        iter_num,
        f_val: result.fun,
        grad_norm: None,
        step_size: None,
        mode: options.mode,
        reason: Some(format!("{:?}: {}", result.status, result.message)),
        final_x: Some(result.x.clone()),
        final_f: result.fun,
        total_nfev: result.nfev,
        fixture_id: options.fixture_id.map(ToOwned::to_owned),
        seed: options.seed,
    };
    push_trace(trace);
}

fn log_casp_decision(options: MinimizeOptions, decision: &OptCaspDecision) {
    let trace = OptimizeTraceEntry {
        ts_unix_ms: now_unix_ms(),
        event: String::from("casp_decision"),
        method: decision.method,
        iter_num: 0,
        f_val: None,
        grad_norm: None,
        step_size: None,
        mode: options.mode,
        reason: Some(decision.reason.clone()),
        final_x: None,
        final_f: None,
        total_nfev: 0,
        fixture_id: options.fixture_id.map(ToOwned::to_owned),
        seed: options.seed,
    };
    push_trace(trace);
}

fn trace_log() -> &'static Mutex<Vec<OptimizeTraceEntry>> {
    static TRACE_LOG: OnceLock<Mutex<Vec<OptimizeTraceEntry>>> = OnceLock::new();
    TRACE_LOG.get_or_init(|| Mutex::new(Vec::new()))
}

fn push_trace(entry: OptimizeTraceEntry) {
    // Resolves [frankenscipy-be4cw] (deferred from kt4od): the previous
    // `if let Ok(mut guard) = trace_log().lock()` pattern silently
    // dropped trace entries on poisoned mutexes. Recover from poison so
    // the trace log keeps recording entries after any prior panic.
    let log = trace_log();
    let mut guard = match log.lock() {
        Ok(g) => g,
        Err(poisoned) => {
            log.clear_poison();
            poisoned.into_inner()
        }
    };
    guard.push(entry);
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

fn variable_scale_ratio(values: &[f64]) -> f64 {
    let mut min_positive = f64::INFINITY;
    let mut max_scale = 1.0_f64;
    for value in values {
        let scale = value.abs().max(1.0);
        min_positive = min_positive.min(scale);
        max_scale = max_scale.max(scale);
    }
    max_scale / min_positive
}

fn dot(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum()
}

fn l2_norm(vec: &[f64]) -> f64 {
    dot(vec, vec).sqrt()
}

fn linf_norm(vec: &[f64]) -> f64 {
    let mut norm: f64 = 0.0;
    for value in vec {
        let abs = value.abs();
        if !abs.is_finite() {
            return abs;
        }
        norm = norm.max(abs);
    }
    norm
}

fn scale_vector(input: &[f64], scale: f64) -> Vec<f64> {
    input.iter().map(|value| value * scale).collect()
}

fn add_scaled(lhs: &[f64], rhs: &[f64], scale: f64) -> Vec<f64> {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(left, right)| left + scale * right)
        .collect()
}

fn add_scaled_into(out: &mut [f64], lhs: &[f64], rhs: &[f64], scale: f64) {
    debug_assert_eq!(out.len(), lhs.len());
    debug_assert_eq!(lhs.len(), rhs.len());
    for ((out_value, left), right) in out.iter_mut().zip(lhs.iter()).zip(rhs.iter()) {
        *out_value = left + scale * right;
    }
}

fn sub_vectors(lhs: &[f64], rhs: &[f64]) -> Vec<f64> {
    lhs.iter().zip(rhs.iter()).map(|(a, b)| a - b).collect()
}

fn identity_matrix(n: usize) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0; n]; n];
    for (idx, row) in out.iter_mut().enumerate() {
        row[idx] = 1.0;
    }
    out
}

fn matrix_vector_mul(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    matrix.iter().map(|row| dot(row, vector)).collect()
}

// ══════════════════════════════════════════════════════════════════════
// Scalar Minimization — Public API
// ══════════════════════════════════════════════════════════════════════

/// Options for scalar (1-D) minimization.
#[derive(Debug, Clone, Copy)]
pub struct MinimizeScalarOptions {
    /// Convergence tolerance on x.
    pub tol: f64,
    /// Maximum number of iterations.
    pub maxiter: usize,
}

impl Default for MinimizeScalarOptions {
    fn default() -> Self {
        Self {
            tol: 1.48e-8, // matches SciPy's default
            maxiter: 500,
        }
    }
}

/// Result of scalar minimization.
#[derive(Debug, Clone, PartialEq)]
pub struct MinimizeScalarResult {
    /// The solution (minimizer).
    pub x: f64,
    /// Function value at the minimizer.
    pub fun: f64,
    /// Whether the optimizer converged.
    pub success: bool,
    /// Number of function evaluations.
    pub nfev: usize,
    /// Number of iterations.
    pub nit: usize,
}

/// Minimize a scalar function using Brent's method.
///
/// Finds a local minimum of `f` within the bracket `(a, b)`.
/// Optionally, an interior point `xatol` can be provided.
/// Matches `scipy.optimize.minimize_scalar(f, bracket=(a, b), method='brent')`.
pub fn minimize_scalar<F>(
    f: F,
    bracket: (f64, f64),
    options: MinimizeScalarOptions,
) -> Result<MinimizeScalarResult, OptError>
where
    F: Fn(f64) -> f64,
{
    let (mut a, mut b) = bracket;
    if !a.is_finite() || !b.is_finite() {
        return Err(OptError::NonFiniteInput {
            detail: "bracket bounds must be finite".to_string(),
        });
    }
    if a > b {
        std::mem::swap(&mut a, &mut b);
    }
    if (a - b).abs() < f64::EPSILON {
        return Err(OptError::InvalidBounds {
            detail: "bracket bounds must not be equal".to_string(),
        });
    }

    // Brent's method with golden section and parabolic interpolation
    let golden_ratio = 0.5 * (3.0 - 5.0_f64.sqrt()); // ~0.381966

    let mut x = a + golden_ratio * (b - a);
    let mut w = x;
    let mut v = x;
    let mut fx = f(x);
    let mut fw = fx;
    let mut fv = fx;
    let mut nfev = 1;

    let mut d = 0.0_f64;
    let mut e = 0.0_f64;

    for nit in 0..options.maxiter {
        let midpoint = 0.5 * (a + b);
        let tol1 = options.tol * x.abs() + 1e-10;
        let tol2 = 2.0 * tol1;

        // Convergence check
        if (x - midpoint).abs() <= (tol2 - 0.5 * (b - a)) {
            return Ok(MinimizeScalarResult {
                x,
                fun: fx,
                success: true,
                nfev,
                nit,
            });
        }

        // Try parabolic interpolation
        let mut use_golden = true;

        if e.abs() > tol1 {
            // Parabolic fit through (v, fv), (w, fw), (x, fx)
            let r = (x - w) * (fx - fv);
            let q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * r;
            let mut q_val = 2.0 * (q - r);

            if q_val > 0.0 {
                p = -p;
            } else {
                q_val = -q_val;
            }

            if p.abs() < (0.5 * q_val * e).abs() && p > q_val * (a - x) && p < q_val * (b - x) {
                // Accept parabolic step
                d = p / q_val;
                let u_test = x + d;
                if (u_test - a) < tol2 || (b - u_test) < tol2 {
                    d = if x < midpoint { tol1 } else { -tol1 };
                }
                use_golden = false;
            }
        }

        if use_golden {
            // Golden section step
            e = if x < midpoint { b - x } else { a - x };
            d = golden_ratio * e;
        } else {
            e = d;
        }

        // Evaluate at new point
        let u = if d.abs() >= tol1 {
            x + d
        } else if d > 0.0 {
            x + tol1
        } else {
            x - tol1
        };
        let fu = f(u);
        nfev += 1;

        // Update bracket
        if fu <= fx {
            if u < x {
                b = x;
            } else {
                a = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
    }

    Ok(MinimizeScalarResult {
        x,
        fun: fx,
        success: false,
        nfev,
        nit: options.maxiter,
    })
}

/// Batched 1-D minimization: minimize `f(x, params)` over a shared `bracket` for MANY parameter
/// sets, one [`MinimizeScalarResult`] per set. This is the vmap-over-solver primitive for scalar
/// minimization — a 1-D minimization SWEEP (calibrate a 1-parameter model per channel, find the
/// mode/MLE per series, minimize a per-case cost) loops `minimize_scalar` in Python, N Brent solves
/// SERIALLY. fsci `minimize_scalar_many` (param-sweep `F: Fn(f64 x, &[f64] params)->f64`) fans the N
/// independent solves across cores and inlines the objective. Result `i` is byte-identical to
/// `minimize_scalar(|x| f(x, &param_rows[i]), bracket, options)`.
pub fn minimize_scalar_many<F>(
    f: F,
    bracket: (f64, f64),
    param_rows: &[Vec<f64>],
    options: MinimizeScalarOptions,
) -> Vec<Result<MinimizeScalarResult, OptError>>
where
    F: Fn(f64, &[f64]) -> f64 + Sync,
{
    let nrows = param_rows.len();
    if nrows == 0 {
        return Vec::new();
    }
    let f_ref = &f;
    let solve_one = move |params: &[f64]| minimize_scalar(|x| f_ref(x, params), bracket, options);

    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let nthreads = cores.min(nrows);
    if nthreads <= 1 || nrows < 4 {
        return param_rows.iter().map(|p| solve_one(p)).collect();
    }

    let chunk = nrows.div_ceil(nthreads);
    let solve_one = &solve_one;
    let chunk_results: Vec<Vec<Result<MinimizeScalarResult, OptError>>> =
        std::thread::scope(|scope| {
            (0..nthreads)
                .filter_map(|t| {
                    let lo = t * chunk;
                    if lo >= nrows {
                        return None;
                    }
                    let hi = (lo + chunk).min(nrows);
                    Some(scope.spawn(move || {
                        (lo..hi)
                            .map(|i| solve_one(&param_rows[i]))
                            .collect::<Vec<_>>()
                    }))
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().expect("minimize_scalar_many worker panicked"))
                .collect()
        });

    let mut out = Vec::with_capacity(nrows);
    for cr in chunk_results {
        out.extend(cr);
    }
    out
}

// ══════════════════════════════════════════════════════════════════════
// TNC (Truncated Newton Constrained)
// ══════════════════════════════════════════════════════════════════════

pub fn tnc<F>(fun: &F, x0: &[f64], options: MinimizeOptions) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    validate_minimize_options(options)?;

    let n = x0.len();
    let tol = requested_tolerance(options.tol);
    let maxiter = options.maxiter.unwrap_or((200 * n).max(100));
    let maxfev = options.maxfev.unwrap_or((2000 * n).max(400));
    let mut objective = Objective::new(fun, options.mode, maxfev);

    // TNC is a *bound-constrained* truncated-Newton method: honour options.bounds.
    // Previously bounds were ignored entirely, so TNC returned the unconstrained
    // optimum (e.g. [3,3] for a [0,2]² box instead of [2,2]).
    let bounds = options.bounds;
    let mut x = x0.to_vec();
    if let Some(b) = bounds {
        project_onto_bounds(&mut x, b);
    }
    let mut f = objective.eval(&x)?;
    let mut njev = 0usize;

    let mut grad = {
        let value = finite_diff_gradient(&mut objective, &x, options.gradient_eps)?;
        njev += 1;
        value
    };
    let mut hess_approx = vec![vec![0.0; n]; n];
    for (i, row) in hess_approx.iter_mut().enumerate().take(n) {
        row[i] = 1.0;
    }

    for iteration in 0..maxiter {
        let proj_grad = if let Some(b) = bounds {
            projected_gradient(&x, &grad, b)
        } else {
            grad.clone()
        };
        let grad_norm = l2_norm(&proj_grad);
        if grad_norm < tol {
            log_completion(
                OptimizeMethod::Tnc,
                options,
                iteration,
                &OptimizeResult {
                    x: x.clone(),
                    fun: Some(f),
                    success: true,
                    status: ConvergenceStatus::Success,
                    message: String::from("Convergence: gradient norm below tolerance"),
                    nfev: objective.nfev,
                    njev,
                    nhev: 0,
                    nit: iteration,
                    jac: Some(grad.clone()),
                    hess_inv: None,
                    maxcv: None,
                },
            );
            return Ok(OptimizeResult {
                x,
                fun: Some(f),
                success: true,
                status: ConvergenceStatus::Success,
                message: String::from("Convergence: gradient norm below tolerance"),
                nfev: objective.nfev,
                njev,
                nhev: 0,
                nit: iteration,
                jac: Some(grad),
                hess_inv: None,
                maxcv: None,
            });
        }

        let direction = conjugate_gradient_solve(&hess_approx, &grad, tol, n.min(50));
        let direction: Vec<f64> = direction.iter().map(|d| -d).collect();

        // Bound-projected backtracking line search. With bounds present each trial
        // point is clipped into the box, so the accepted step (and thus x) always
        // stays feasible; the Armijo test uses the *actual* (post-projection) step.
        let (x_new, f_new, s) = {
            let mut alpha = 1.0;
            let mut accepted: Option<(Vec<f64>, f64, Vec<f64>)> = None;
            for _ in 0..30 {
                let mut cand: Vec<f64> = x
                    .iter()
                    .zip(direction.iter())
                    .map(|(xi, di)| xi + alpha * di)
                    .collect();
                if let Some(b) = bounds {
                    project_onto_bounds(&mut cand, b);
                }
                let step: Vec<f64> = cand.iter().zip(x.iter()).map(|(c, xi)| c - xi).collect();
                let fv = objective.eval(&cand)?;
                let dd = dot(&grad, &step);
                if fv <= f + 1.0e-4 * dd {
                    accepted = Some((cand, fv, step));
                    break;
                }
                alpha *= 0.5;
                if alpha < 1.0e-12 {
                    break;
                }
            }
            match accepted {
                Some(t) => t,
                None => {
                    // No feasible decrease found; the bound-constrained optimum is
                    // already handled by the projected-gradient check at loop top.
                    return Ok(OptimizeResult {
                        x,
                        fun: Some(f),
                        success: false,
                        status: ConvergenceStatus::PrecisionLoss,
                        message: String::from("line search failed"),
                        nfev: objective.nfev,
                        njev,
                        nhev: 0,
                        nit: iteration,
                        jac: Some(grad),
                        hess_inv: None,
                        maxcv: None,
                    });
                }
            }
        };

        let grad_new = match finite_diff_gradient(&mut objective, &x_new, options.gradient_eps) {
            Ok(v) => {
                njev += 1;
                v
            }
            Err(e) => {
                return Ok(result_from_error(
                    &x_new,
                    iteration,
                    objective.nfev,
                    njev,
                    e,
                ));
            }
        };

        let y: Vec<f64> = grad_new
            .iter()
            .zip(grad.iter())
            .map(|(gn, go)| gn - go)
            .collect();
        let sy: f64 = s.iter().zip(y.iter()).map(|(si, yi)| si * yi).sum();

        if sy > 1.0e-12 {
            // BFGS update of the Hessian approximation B (the search direction
            // solves B d = grad): B_{k+1} = B_k - (B s)(B s)^T / (s^T B s)
            //                                     + (y y^T) / (y^T s).
            // The previous formula mixed inverse- and direct-Hessian terms and
            // never referenced B_k, so the approximation degraded into noise
            // and the iteration stalled in narrow valleys (e.g. Rosenbrock).
            let bs: Vec<f64> = (0..n)
                .map(|i| (0..n).map(|j| hess_approx[i][j] * s[j]).sum())
                .collect();
            let s_bs: f64 = s.iter().zip(bs.iter()).map(|(si, bsi)| si * bsi).sum();
            if s_bs > 1.0e-12 {
                for i in 0..n {
                    for j in 0..n {
                        hess_approx[i][j] += y[i] * y[j] / sy - bs[i] * bs[j] / s_bs;
                    }
                }
            }
        }

        x = x_new;
        f = f_new;
        grad = grad_new;

        if let Some(callback) = options.callback
            && !callback(&x)
        {
            return Ok(OptimizeResult {
                x,
                fun: Some(f),
                success: false,
                status: ConvergenceStatus::CallbackStop,
                message: String::from("callback requested stop"),
                nfev: objective.nfev,
                njev,
                nhev: 0,
                nit: iteration + 1,
                jac: Some(grad),
                hess_inv: None,
                maxcv: None,
            });
        }
    }

    Ok(OptimizeResult {
        x,
        fun: Some(f),
        success: false,
        status: ConvergenceStatus::MaxIterations,
        message: String::from("Maximum iterations reached"),
        nfev: objective.nfev,
        njev,
        nhev: 0,
        nit: maxiter,
        jac: Some(grad),
        hess_inv: None,
        maxcv: None,
    })
}

fn conjugate_gradient_solve(a: &[Vec<f64>], b: &[f64], tol: f64, max_iter: usize) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    let mut r = b.to_vec();
    let mut p = r.clone();
    let mut rs_old: f64 = r.iter().map(|ri| ri * ri).sum();

    for _ in 0..max_iter {
        let mut ap = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                ap[i] += a[i][j] * p[j];
            }
        }
        let pap: f64 = p.iter().zip(ap.iter()).map(|(pi, api)| pi * api).sum();
        if pap.abs() < 1.0e-15 {
            break;
        }
        let alpha = rs_old / pap;

        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        let rs_new: f64 = r.iter().map(|ri| ri * ri).sum();
        if rs_new.sqrt() < tol {
            break;
        }

        let beta = rs_new / rs_old;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rs_old = rs_new;
    }
    x
}

// ══════════════════════════════════════════════════════════════════════
// SLSQP (Sequential Least Squares Programming)
// ══════════════════════════════════════════════════════════════════════

pub fn slsqp<F>(fun: &F, x0: &[f64], options: MinimizeOptions) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    validate_minimize_options(options)?;

    let n = x0.len();
    let tol = requested_tolerance(options.tol);
    let maxiter = options.maxiter.unwrap_or((200 * n).max(100));
    let maxfev = options.maxfev.unwrap_or((2000 * n).max(400));
    let mut objective = Objective::new(fun, options.mode, maxfev);

    let mut x = x0.to_vec();
    let mut f = objective.eval(&x)?;

    let mut grad = finite_diff_gradient(&mut objective, &x, options.gradient_eps)?;
    let mut hess_approx = vec![vec![0.0; n]; n];
    for (i, row) in hess_approx.iter_mut().enumerate().take(n) {
        row[i] = 1.0;
    }

    for iteration in 0..maxiter {
        let grad_norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < tol {
            log_completion(
                OptimizeMethod::Slsqp,
                options,
                iteration,
                &OptimizeResult {
                    x: x.clone(),
                    fun: Some(f),
                    success: true,
                    status: ConvergenceStatus::Success,
                    message: String::from("Convergence: gradient norm below tolerance"),
                    nfev: objective.nfev,
                    njev: iteration,
                    nhev: 0,
                    nit: iteration,
                    jac: Some(grad.clone()),
                    hess_inv: None,
                    maxcv: None,
                },
            );
            return Ok(OptimizeResult {
                x,
                fun: Some(f),
                success: true,
                status: ConvergenceStatus::Success,
                message: String::from("Convergence: gradient norm below tolerance"),
                nfev: objective.nfev,
                njev: iteration,
                nhev: 0,
                nit: iteration,
                jac: Some(grad),
                hess_inv: None,
                maxcv: None,
            });
        }

        let direction = solve_qp_subproblem(&hess_approx, &grad);

        let line = match armijo_backtracking(&mut objective, &x, f, &grad, &direction)? {
            Some(value) => value,
            None => {
                return Ok(OptimizeResult {
                    x,
                    fun: Some(f),
                    success: false,
                    status: ConvergenceStatus::PrecisionLoss,
                    message: String::from("SLSQP line search failed"),
                    nfev: objective.nfev,
                    njev: iteration,
                    nhev: 0,
                    nit: iteration,
                    jac: Some(grad),
                    hess_inv: None,
                    maxcv: None,
                });
            }
        };
        let alpha = line.alpha;
        let f_new = line.f;

        let s: Vec<f64> = direction.iter().map(|d| alpha * d).collect();
        let x_new: Vec<f64> = x.iter().zip(s.iter()).map(|(xi, si)| xi + si).collect();
        let grad_new = finite_diff_gradient(&mut objective, &x_new, options.gradient_eps)?;

        let y: Vec<f64> = grad_new
            .iter()
            .zip(grad.iter())
            .map(|(gn, go)| gn - go)
            .collect();
        let sy: f64 = s.iter().zip(y.iter()).map(|(si, yi)| si * yi).sum();

        if sy > 1.0e-12 {
            bfgs_update(&mut hess_approx, &s, &y, sy);
        }

        let step_norm: f64 = s.iter().map(|si| si * si).sum::<f64>().sqrt();
        if step_norm < tol * (1.0 + x.iter().map(|xi| xi.abs()).sum::<f64>() / n as f64) {
            log_completion(
                OptimizeMethod::Slsqp,
                options,
                iteration + 1,
                &OptimizeResult {
                    x: x_new.clone(),
                    fun: Some(f_new),
                    success: true,
                    status: ConvergenceStatus::Success,
                    message: String::from("Convergence: step size below tolerance"),
                    nfev: objective.nfev,
                    njev: iteration + 1,
                    nhev: 0,
                    nit: iteration + 1,
                    jac: Some(grad_new.clone()),
                    hess_inv: None,
                    maxcv: None,
                },
            );
            return Ok(OptimizeResult {
                x: x_new,
                fun: Some(f_new),
                success: true,
                status: ConvergenceStatus::Success,
                message: String::from("Convergence: step size below tolerance"),
                nfev: objective.nfev,
                njev: iteration + 1,
                nhev: 0,
                nit: iteration + 1,
                jac: Some(grad_new),
                hess_inv: None,
                maxcv: None,
            });
        }

        x = x_new;
        f = f_new;
        grad = grad_new;

        if let Some(cb) = options.callback
            && cb(&x)
        {
            return Ok(OptimizeResult {
                x,
                fun: Some(f),
                success: false,
                status: ConvergenceStatus::CallbackStop,
                message: String::from("Optimization stopped by callback"),
                nfev: objective.nfev,
                njev: iteration + 1,
                nhev: 0,
                nit: iteration + 1,
                jac: Some(grad),
                hess_inv: None,
                maxcv: None,
            });
        }
    }

    Ok(OptimizeResult {
        x,
        fun: Some(f),
        success: false,
        status: ConvergenceStatus::MaxIterations,
        message: String::from("Maximum iterations reached"),
        nfev: objective.nfev,
        njev: maxiter,
        nhev: 0,
        nit: maxiter,
        jac: Some(grad),
        hess_inv: None,
        maxcv: None,
    })
}

fn solve_qp_subproblem(h: &[Vec<f64>], g: &[f64]) -> Vec<f64> {
    let n = g.len();
    let mut direction = vec![0.0; n];
    for i in 0..n {
        if h[i][i].abs() > 1.0e-12 {
            direction[i] = -g[i] / h[i][i];
        } else {
            direction[i] = -g[i];
        }
    }
    direction
}

fn bfgs_update(h: &mut [Vec<f64>], s: &[f64], y: &[f64], sy: f64) {
    let n = s.len();
    let rho = 1.0 / sy;

    let mut hy = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            hy[i] += h[i][j] * y[j];
        }
    }
    let yhy: f64 = y.iter().zip(hy.iter()).map(|(yi, hyi)| yi * hyi).sum();

    for i in 0..n {
        for j in 0..n {
            h[i][j] += rho * (1.0 + rho * yhy) * s[i] * s[j] - rho * (s[i] * hy[j] + hy[i] * s[j]);
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// Trust-Constr (Trust-Region Constrained)
// ══════════════════════════════════════════════════════════════════════

pub fn trust_constr<F>(
    fun: &F,
    x0: &[f64],
    options: MinimizeOptions,
) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    validate_minimize_options(options)?;

    let n = x0.len();
    let tol = requested_tolerance(options.tol);
    let maxiter = options.maxiter.unwrap_or((200 * n).max(100));
    let maxfev = options.maxfev.unwrap_or((2000 * n).max(400));
    let mut objective = Objective::new(fun, options.mode, maxfev);

    let mut x = x0.to_vec();
    let mut f = objective.eval(&x)?;

    let mut grad = finite_diff_gradient(&mut objective, &x, options.gradient_eps)?;
    let mut trust_radius = 1.0;
    let eta = 0.15;
    // Quasi-Newton Hessian approximation B for the quadratic trust-region
    // model m(s) = f + g·s + ½ s^T B s, updated by BFGS. A pure Cauchy
    // (steepest-descent) step ignores curvature and stalls in narrow valleys.
    let mut b_hess = vec![vec![0.0; n]; n];
    for (i, row) in b_hess.iter_mut().enumerate() {
        row[i] = 1.0;
    }

    for iteration in 0..maxiter {
        let kkt_optimality = unconstrained_kkt_optimality(&grad);
        if kkt_optimality <= tol {
            log_completion(
                OptimizeMethod::TrustConstr,
                options,
                iteration,
                &OptimizeResult {
                    x: x.clone(),
                    fun: Some(f),
                    success: true,
                    status: ConvergenceStatus::Success,
                    message: format!(
                        "Convergence: KKT optimality {kkt_optimality:.3e} <= tolerance"
                    ),
                    nfev: objective.nfev,
                    njev: iteration,
                    nhev: iteration,
                    nit: iteration,
                    jac: Some(grad.clone()),
                    hess_inv: None,
                    maxcv: None,
                },
            );
            return Ok(OptimizeResult {
                x,
                fun: Some(f),
                success: true,
                status: ConvergenceStatus::Success,
                message: format!("Convergence: KKT optimality {kkt_optimality:.3e} <= tolerance"),
                nfev: objective.nfev,
                njev: iteration,
                nhev: iteration,
                nit: iteration,
                jac: Some(grad),
                hess_inv: None,
                maxcv: None,
            });
        }

        let step = dogleg_step(&b_hess, &grad, trust_radius);

        let x_new: Vec<f64> = x.iter().zip(step.iter()).map(|(xi, si)| xi + si).collect();
        let f_new = match objective.eval(&x_new) {
            Ok(v) => v,
            Err(OptError::EvaluationBudgetExceeded { .. }) => {
                return Ok(OptimizeResult {
                    x,
                    fun: Some(f),
                    success: false,
                    status: ConvergenceStatus::MaxEvaluations,
                    message: String::from("Maximum function evaluations reached"),
                    nfev: objective.nfev,
                    njev: iteration,
                    nhev: iteration,
                    nit: iteration,
                    jac: Some(grad),
                    hess_inv: None,
                    maxcv: None,
                });
            }
            Err(e) => return Err(e),
        };

        // Reduction predicted by the quadratic model: -(g·s + ½ s^T B s).
        let bs: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| b_hess[i][j] * step[j]).sum())
            .collect();
        let s_bs: f64 = step.iter().zip(bs.iter()).map(|(si, bsi)| si * bsi).sum();
        let g_s: f64 = grad.iter().zip(step.iter()).map(|(gi, si)| gi * si).sum();
        let predicted = -(g_s + 0.5 * s_bs);
        let actual = f - f_new;
        let rho = if predicted.abs() > 1.0e-15 {
            actual / predicted
        } else {
            0.0
        };

        let step_norm = step.iter().map(|s| s * s).sum::<f64>().sqrt();
        if rho < 0.25 {
            trust_radius *= 0.25;
        } else if rho > 0.75 && (step_norm - trust_radius).abs() < 1.0e-10 {
            trust_radius = (2.0 * trust_radius).min(100.0);
        }

        if rho > eta {
            let grad_new = finite_diff_gradient(&mut objective, &x_new, options.gradient_eps)?;
            // BFGS update of the Hessian model:
            //   B_{k+1} = B_k - (B s)(B s)^T / (s^T B s) + (y y^T) / (y^T s).
            let y: Vec<f64> = grad_new
                .iter()
                .zip(grad.iter())
                .map(|(gn, go)| gn - go)
                .collect();
            let sy: f64 = step.iter().zip(y.iter()).map(|(si, yi)| si * yi).sum();
            if sy > 1.0e-12 && s_bs > 1.0e-12 {
                for i in 0..n {
                    for j in 0..n {
                        b_hess[i][j] += y[i] * y[j] / sy - bs[i] * bs[j] / s_bs;
                    }
                }
            }
            x = x_new;
            f = f_new;
            grad = grad_new;
        }

        if trust_radius < tol * 1.0e-6 {
            return Ok(OptimizeResult {
                x,
                fun: Some(f),
                success: true,
                status: ConvergenceStatus::Success,
                message: String::from("Convergence: trust radius below tolerance"),
                nfev: objective.nfev,
                njev: iteration + 1,
                nhev: iteration + 1,
                nit: iteration + 1,
                jac: Some(grad),
                hess_inv: None,
                maxcv: None,
            });
        }

        if let Some(cb) = options.callback
            && cb(&x)
        {
            return Ok(OptimizeResult {
                x,
                fun: Some(f),
                success: false,
                status: ConvergenceStatus::CallbackStop,
                message: String::from("Optimization stopped by callback"),
                nfev: objective.nfev,
                njev: iteration + 1,
                nhev: iteration + 1,
                nit: iteration + 1,
                jac: Some(grad),
                hess_inv: None,
                maxcv: None,
            });
        }
    }

    Ok(OptimizeResult {
        x,
        fun: Some(f),
        success: false,
        status: ConvergenceStatus::MaxIterations,
        message: String::from("Maximum iterations reached"),
        nfev: objective.nfev,
        njev: maxiter,
        nhev: maxiter,
        nit: maxiter,
        jac: Some(grad),
        hess_inv: None,
        maxcv: None,
    })
}

fn unconstrained_kkt_optimality(grad: &[f64]) -> f64 {
    linf_norm(grad)
}

/// Powell's dogleg step for the trust-region subproblem
/// `min g·s + ½ s^T B s` subject to `||s|| <= trust_radius`.
///
/// Combines the Cauchy point (steepest-descent model minimizer) and the
/// Newton point `-B^{-1} g`: returns the Newton point when it lies inside the
/// region, the boundary-scaled steepest-descent step when even the Cauchy
/// point lies outside, and otherwise the point where the segment joining them
/// crosses the trust boundary.
fn dogleg_step(b: &[Vec<f64>], grad: &[f64], trust_radius: f64) -> Vec<f64> {
    let n = grad.len();
    let grad_norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
    if grad_norm < 1.0e-15 {
        return vec![0.0; n];
    }

    // Newton point p_b = -B^{-1} g.
    let binv_g = conjugate_gradient_solve(b, grad, 1.0e-10, n.clamp(1, 50));
    let p_newton: Vec<f64> = binv_g.iter().map(|v| -v).collect();
    let newton_norm = p_newton.iter().map(|p| p * p).sum::<f64>().sqrt();
    if newton_norm.is_finite() && newton_norm <= trust_radius {
        return p_newton;
    }

    // Cauchy point p_u = -(g·g / g^T B g) g.
    let gg = grad_norm * grad_norm;
    let bg: Vec<f64> = (0..n)
        .map(|i| (0..n).map(|j| b[i][j] * grad[j]).sum())
        .collect();
    let gbg: f64 = grad.iter().zip(bg.iter()).map(|(g, bgi)| g * bgi).sum();
    if gbg <= 0.0 || gbg.is_nan() {
        // Non-positive curvature: ride the boundary along -g.
        let scale = trust_radius / grad_norm;
        return grad.iter().map(|g| -scale * g).collect();
    }
    let p_cauchy: Vec<f64> = grad.iter().map(|g| -(gg / gbg) * g).collect();
    let cauchy_norm = p_cauchy.iter().map(|p| p * p).sum::<f64>().sqrt();
    if cauchy_norm >= trust_radius || !newton_norm.is_finite() {
        let scale = trust_radius / grad_norm;
        return grad.iter().map(|g| -scale * g).collect();
    }

    // Second dogleg leg: p(t) = p_cauchy + t (p_newton - p_cauchy), t in [0,1];
    // pick t where ||p(t)|| = trust_radius.
    let diff: Vec<f64> = p_newton
        .iter()
        .zip(p_cauchy.iter())
        .map(|(pn, pc)| pn - pc)
        .collect();
    let a: f64 = diff.iter().map(|d| d * d).sum();
    let b_coef: f64 = 2.0
        * p_cauchy
            .iter()
            .zip(diff.iter())
            .map(|(pc, d)| pc * d)
            .sum::<f64>();
    let c = cauchy_norm * cauchy_norm - trust_radius * trust_radius;
    let t = if a.abs() < 1.0e-15 {
        0.0
    } else {
        let disc = (b_coef * b_coef - 4.0 * a * c).max(0.0).sqrt();
        ((-b_coef + disc) / (2.0 * a)).clamp(0.0, 1.0)
    };
    p_cauchy
        .iter()
        .zip(diff.iter())
        .map(|(pc, d)| pc + t * d)
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, OnceLock};
    use std::time::{SystemTime, UNIX_EPOCH};

    use fsci_runtime::RuntimeMode;
    use proptest::prelude::*;
    use serde::Serialize;

    use super::{slsqp, tnc, trust_constr};

    use super::{Objective, golden_section_direction_search};
    use crate::{
        Bound, ConvergenceStatus, MinimizeOptions, MinimizeScalarOptions, OptCaspProblem, OptError,
        OptimizeMethod, OptimizeResult, bfgs, cg_pr_plus, get_optimize_traces, minimize,
        minimize_many, minimize_scalar, minimize_scalar_many, powell, select_minimize_method,
    };

    #[derive(Debug, Serialize)]
    struct TestLogEntry<'a> {
        test_id: &'a str,
        optimizer: &'a str,
        problem: &'a str,
        n_dim: usize,
        mode: &'a str,
        converged: bool,
        nfev: usize,
        final_f: Option<f64>,
        seed: u64,
        timestamp_ms: u64,
    }

    fn test_log_sink() -> &'static Mutex<Vec<String>> {
        static TEST_LOGS: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
        TEST_LOGS.get_or_init(|| Mutex::new(Vec::new()))
    }

    fn callback_points() -> &'static Mutex<Vec<Vec<f64>>> {
        static CALLBACK_POINTS: OnceLock<Mutex<Vec<Vec<f64>>>> = OnceLock::new();
        CALLBACK_POINTS.get_or_init(|| Mutex::new(Vec::new()))
    }

    fn now_unix_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_millis() as u64)
    }

    fn mode_name(mode: RuntimeMode) -> &'static str {
        match mode {
            RuntimeMode::Strict => "strict",
            RuntimeMode::Hardened => "hardened",
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn push_test_log(
        test_id: &str,
        optimizer: &str,
        problem: &str,
        n_dim: usize,
        mode: RuntimeMode,
        result: &OptimizeResult,
        seed: u64,
    ) {
        let entry = TestLogEntry {
            test_id,
            optimizer,
            problem,
            n_dim,
            mode: mode_name(mode),
            converged: result.success,
            nfev: result.nfev,
            final_f: result.fun,
            seed,
            timestamp_ms: now_unix_ms(),
        };
        let payload = serde_json::to_string(&entry).expect("serialize test log");
        let parsed: serde_json::Value =
            serde_json::from_str(&payload).expect("re-parse serialized log payload");
        assert!(parsed.get("test_id").is_some());
        assert!(parsed.get("optimizer").is_some());
        assert!(parsed.get("timestamp_ms").is_some());
        test_log_sink()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(payload);
    }

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|value| value * value).sum()
    }

    fn rosenbrock(x: &[f64]) -> f64 {
        let x0 = x[0];
        let x1 = x[1];
        (1.0 - x0).powi(2) + 100.0 * (x1 - x0 * x0).powi(2)
    }

    fn rosenbrock_gradient(x: &[f64]) -> Vec<f64> {
        let x0 = x[0];
        let x1 = x[1];
        vec![
            -2.0 * (1.0 - x0) - 400.0 * x0 * (x1 - x0 * x0),
            200.0 * (x1 - x0 * x0),
        ]
    }

    fn himmelblau(x: &[f64]) -> f64 {
        let x0 = x[0];
        let x1 = x[1];
        (x0 * x0 + x1 - 11.0).powi(2) + (x0 + x1 * x1 - 7.0).powi(2)
    }

    fn flat_quartic(x: &[f64]) -> f64 {
        (x[0] - 1.0).powi(4) + (x[1] + 2.0).powi(4)
    }

    fn step_plateau(x: &[f64]) -> f64 {
        if x[0] >= 0.5 { 1.0 } else { 0.0 }
    }

    fn nonconvex_saddle(x: &[f64]) -> f64 {
        x[0] * x[0] - x[1] * x[1] + 0.1 * x[1].powi(4)
    }

    fn abs_sum(x: &[f64]) -> f64 {
        x.iter().map(|v| v.abs()).sum()
    }

    fn one_dim_quadratic(x: &[f64]) -> f64 {
        (x[0] - 1.5).powi(2) + 1.0
    }

    fn zero_function(_: &[f64]) -> f64 {
        0.0
    }

    fn unit_hessp(_x: &[f64], p: &[f64]) -> Vec<f64> {
        p.to_vec()
    }

    fn callback_record_and_stop(x: &[f64]) -> bool {
        callback_points()
            .lock()
            .expect("callback points lock")
            .push(x.to_vec());
        false
    }

    #[test]
    fn casp_selector_chooses_bfgs_for_smooth_unconstrained_problem() {
        let decision = select_minimize_method(OptCaspProblem {
            dimension: 3,
            variable_scale_ratio: 10.0,
            has_box_bounds: false,
            has_general_constraints: false,
            gradient_available: true,
            hessian_product_available: false,
        })
        .expect("selector");
        assert_eq!(decision.method, OptimizeMethod::Bfgs);
        assert!(decision.reason.contains("BFGS"));
    }

    #[test]
    fn casp_selector_chooses_lbfgsb_for_box_constraints() {
        let decision = select_minimize_method(OptCaspProblem {
            dimension: 8,
            variable_scale_ratio: 1.0,
            has_box_bounds: true,
            has_general_constraints: false,
            gradient_available: true,
            hessian_product_available: false,
        })
        .expect("selector");
        assert_eq!(decision.method, OptimizeMethod::LBfgsB);
        assert!(decision.reason.contains("box constraints"));
    }

    #[test]
    fn casp_selector_chooses_curvature_methods_by_scale_and_hessp() {
        let newton = select_minimize_method(OptCaspProblem {
            dimension: 20,
            variable_scale_ratio: 100.0,
            has_box_bounds: false,
            has_general_constraints: false,
            gradient_available: true,
            hessian_product_available: true,
        })
        .expect("selector");
        assert_eq!(newton.method, OptimizeMethod::NewtonCg);

        let trust = select_minimize_method(OptCaspProblem {
            dimension: 2,
            variable_scale_ratio: 1.0e8,
            has_box_bounds: false,
            has_general_constraints: false,
            gradient_available: true,
            hessian_product_available: true,
        })
        .expect("selector");
        assert_eq!(trust.method, OptimizeMethod::TrustExact);
        assert!(trust.reason.contains("trust-region"));
    }

    #[test]
    fn get_optimize_traces_recovers_from_poisoned_trace_log() {
        // Regression for [frankenscipy-pvrjd]: read path used to return
        // Vec::new() on poison even though push_trace recovered. The
        // safety property under parallel tests: a sentinel pushed before
        // poisoning must still be visible from get_optimize_traces.
        // (Asserting the global mutex's poison state would race with
        // any other test thread that hits the recovery branch first.)
        use crate::types::OptimizeTraceEntry;
        let sentinel = format!(
            "pvrjd_sentinel_{}",
            super::now_unix_ms() ^ (std::process::id() as u64)
        );
        super::push_trace(OptimizeTraceEntry {
            ts_unix_ms: super::now_unix_ms(),
            event: sentinel.clone(),
            method: OptimizeMethod::Bfgs,
            iter_num: 0,
            f_val: None,
            grad_norm: None,
            step_size: None,
            mode: RuntimeMode::Strict,
            reason: None,
            final_x: None,
            final_f: None,
            total_nfev: 0,
            fixture_id: None,
            seed: None,
        });
        let _ = std::thread::spawn(|| {
            let _g = super::trace_log().lock().expect("acquire trace_log");
            panic!("intentional poison for poison-recovery test");
        })
        .join();
        let traces = get_optimize_traces();
        assert!(
            traces.iter().any(|t| t.event == sentinel),
            "sentinel pushed before poison must survive the read path; got {} entries",
            traces.len()
        );
    }

    #[test]
    fn casp_default_minimize_logs_decision_and_uses_newton_cg_when_hessp_available() {
        const FIXTURE_ID: &str = "casp-default-newton-cg";
        let _ = get_optimize_traces();
        let options = MinimizeOptions {
            method: None,
            hessp: Some(unit_hessp),
            fixture_id: Some(FIXTURE_ID),
            tol: Some(1.0e-10),
            maxiter: Some(20),
            maxfev: Some(200),
            ..MinimizeOptions::default()
        };
        let result = minimize(|x| 0.5 * x[0] * x[0], &[3.0], options).expect("minimize");
        assert!(result.success, "selected optimizer should converge");

        let traces = get_optimize_traces();
        let decision = traces
            .iter()
            .find(|entry| {
                entry.event == "casp_decision" && entry.fixture_id.as_deref() == Some(FIXTURE_ID)
            })
            .expect("CASP decision trace");
        assert_eq!(decision.method, OptimizeMethod::NewtonCg);
        assert!(
            decision
                .reason
                .as_deref()
                .is_some_and(|reason| reason.contains("Hessian product"))
        );
    }

    #[test]
    fn casp_default_minimize_uses_lbfgsb_for_static_box_bounds() {
        const FIXTURE_ID: &str = "casp-default-lbfgsb-bounds";
        static BOUNDS: [Bound; 2] = [(Some(0.0), Some(2.0)), (Some(0.0), Some(2.0))];
        let _ = get_optimize_traces();
        let options = MinimizeOptions {
            method: None,
            bounds: Some(&BOUNDS),
            fixture_id: Some(FIXTURE_ID),
            tol: Some(1.0e-8),
            maxiter: Some(120),
            maxfev: Some(2_000),
            ..MinimizeOptions::default()
        };
        let result = minimize(
            |x| (x[0] - 3.0).powi(2) + (x[1] - 3.0).powi(2),
            &[0.0, 0.0],
            options,
        )
        .expect("bounded minimize");

        assert!(result.success, "{}", result.message);
        assert!((result.x[0] - 2.0).abs() <= 1.0e-6, "x={:?}", result.x);
        assert!((result.x[1] - 2.0).abs() <= 1.0e-6, "x={:?}", result.x);

        let traces = get_optimize_traces();
        let decision = traces
            .iter()
            .find(|entry| {
                entry.event == "casp_decision" && entry.fixture_id.as_deref() == Some(FIXTURE_ID)
            })
            .expect("CASP decision trace");
        assert_eq!(decision.method, OptimizeMethod::LBfgsB);
        assert!(
            decision
                .reason
                .as_deref()
                .is_some_and(|reason| reason.contains("box constraints"))
        );
    }

    #[test]
    fn casp_default_minimize_uses_nelder_mead_when_gradient_unavailable() {
        const FIXTURE_ID: &str = "casp-default-nelder-mead";
        let _ = get_optimize_traces();
        let options = MinimizeOptions {
            method: None,
            gradient_available: false,
            fixture_id: Some(FIXTURE_ID),
            tol: Some(1.0e-8),
            maxiter: Some(120),
            maxfev: Some(2_000),
            ..MinimizeOptions::default()
        };
        let result =
            minimize(one_dim_quadratic, &[0.0], options).expect("derivative-free minimize");
        assert!(result.success, "{}", result.message);

        let traces = get_optimize_traces();
        let decision = traces
            .iter()
            .find(|entry| {
                entry.event == "casp_decision" && entry.fixture_id.as_deref() == Some(FIXTURE_ID)
            })
            .expect("CASP decision trace");
        assert_eq!(decision.method, OptimizeMethod::NelderMead);
        assert!(
            decision
                .reason
                .as_deref()
                .is_some_and(|reason| reason.contains("no gradient"))
        );
    }

    #[test]
    fn casp_default_minimize_uses_trust_constr_when_constraints_are_flagged() {
        const FIXTURE_ID: &str = "casp-default-trust-constr";
        let _ = get_optimize_traces();
        let options = MinimizeOptions {
            method: None,
            has_general_constraints: true,
            fixture_id: Some(FIXTURE_ID),
            tol: Some(1.0e-8),
            maxiter: Some(10),
            maxfev: Some(100),
            ..MinimizeOptions::default()
        };
        let result = minimize(zero_function, &[0.0, 0.0], options).expect("trust-constr minimize");
        assert!(result.success, "{}", result.message);

        let traces = get_optimize_traces();
        let decision = traces
            .iter()
            .find(|entry| {
                entry.event == "casp_decision" && entry.fixture_id.as_deref() == Some(FIXTURE_ID)
            })
            .expect("CASP decision trace");
        assert_eq!(decision.method, OptimizeMethod::TrustConstr);
        assert!(
            decision
                .reason
                .as_deref()
                .is_some_and(|reason| reason.contains("general constraints"))
        );
    }

    #[test]
    fn casp_bounds_validate_against_x0_before_dispatch() {
        static BAD_BOUNDS: [Bound; 1] = [(Some(0.0), Some(1.0))];
        let error = minimize(
            sphere,
            &[0.0, 0.0],
            MinimizeOptions {
                method: None,
                bounds: Some(&BAD_BOUNDS),
                ..MinimizeOptions::default()
            },
        )
        .expect_err("bounds dimension mismatch");
        assert!(matches!(error, OptError::InvalidBounds { .. }));
    }

    #[test]
    fn optimize_result_success_fields_are_populated() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            tol: Some(1.0e-8),
            maxiter: Some(200),
            maxfev: Some(20_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result = minimize(sphere, &[2.0, -3.0], options).expect("minimize executes");
        assert!(result.success, "{}", result.message);
        assert_eq!(result.status, ConvergenceStatus::Success);
        assert!(result.fun.is_some());
        assert!(result.nfev >= 1);
        assert!(result.x.iter().all(|value| value.abs() < 1.0e-4));
        push_test_log(
            "optimize-result-success-fields",
            "bfgs",
            "sphere",
            2,
            RuntimeMode::Strict,
            &result,
            101,
        );
    }

    #[test]
    fn optimize_result_reports_max_iterations_exceeded() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            tol: Some(1.0e-14),
            maxiter: Some(1),
            maxfev: Some(20_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result = minimize(rosenbrock, &[-1.2, 1.0], options).expect("minimize executes");
        assert!(!result.success);
        assert_eq!(result.status, ConvergenceStatus::MaxIterations);
        assert!(result.message.contains("maximum iterations"));
        push_test_log(
            "optimize-result-maxiter",
            "bfgs",
            "rosenbrock",
            2,
            RuntimeMode::Strict,
            &result,
            102,
        );
    }

    #[test]
    fn optimize_result_reports_line_search_failure() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            tol: Some(1.0e-12),
            maxiter: Some(50),
            maxfev: Some(20_000),
            gradient_eps: 1.0e-6,
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result =
            minimize(step_plateau, &[0.499_999], options).expect("minimize returns a result");
        assert!(!result.success);
        assert_eq!(result.status, ConvergenceStatus::PrecisionLoss);
        assert!(result.message.contains("line search failed"));
        push_test_log(
            "optimize-result-linesearch-failure",
            "bfgs",
            "step-plateau",
            1,
            RuntimeMode::Strict,
            &result,
            103,
        );
    }

    #[test]
    fn bfgs_converges_on_quadratic() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            tol: Some(1.0e-7),
            maxiter: Some(200),
            maxfev: Some(20_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result = bfgs(&sphere, &[2.0, -3.0], options).expect("bfgs executes");
        assert!(result.success, "{}", result.message);
        assert_eq!(result.status, ConvergenceStatus::Success);
        assert!(result.fun.expect("objective") < 1.0e-8);
        assert!(result.x.iter().all(|value| value.abs() < 1.0e-4));
        push_test_log(
            "bfgs-quadratic",
            "bfgs",
            "sphere",
            2,
            RuntimeMode::Strict,
            &result,
            104,
        );
    }

    #[test]
    fn bfgs_converges_on_rosenbrock() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            tol: Some(1.0e-7),
            maxiter: Some(800),
            maxfev: Some(80_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result = bfgs(&rosenbrock, &[-1.2, 1.0], options).expect("bfgs executes");
        assert!(result.fun.expect("objective") < 1.0e-6);
        assert!((result.x[0] - 1.0).abs() < 1.0e-2);
        assert!((result.x[1] - 1.0).abs() < 1.0e-2);
        push_test_log(
            "bfgs-rosenbrock",
            "bfgs",
            "rosenbrock",
            2,
            RuntimeMode::Strict,
            &result,
            105,
        );
    }

    #[test]
    fn bfgs_rosenbrock_exact_gradient_uses_callback_jacobian() {
        let mut options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            tol: Some(1.0e-7),
            maxiter: Some(800),
            maxfev: Some(80_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        options.gradient = Some(rosenbrock_gradient);

        let initial = rosenbrock(&[-1.2, 1.0]);
        let exact = bfgs(&rosenbrock, &[-1.2, 1.0], options).expect("bfgs exact");
        let expected_jac = rosenbrock_gradient(&exact.x);

        assert!(exact.fun.expect("objective") < initial);
        assert_eq!(exact.jac.as_ref().expect("jac"), &expected_jac);
        assert!(exact.njev > 0);
        push_test_log(
            "bfgs-rosenbrock-exact-gradient",
            "bfgs",
            "rosenbrock",
            2,
            RuntimeMode::Strict,
            &exact,
            118,
        );
    }

    #[test]
    fn bfgs_exact_gradient_rejects_bad_callback_shape() {
        fn bad_gradient(_: &[f64]) -> Vec<f64> {
            vec![0.0]
        }

        let result = bfgs(
            &sphere,
            &[2.0, -3.0],
            MinimizeOptions {
                method: Some(OptimizeMethod::Bfgs),
                gradient: Some(bad_gradient),
                mode: RuntimeMode::Strict,
                ..MinimizeOptions::default()
            },
        )
        .expect("bfgs returns invalid-input result");

        assert!(!result.success);
        assert_eq!(result.status, ConvergenceStatus::InvalidInput);
        assert!(result.message.contains("gradient callback returned length"));
    }

    #[test]
    fn bfgs_exact_gradient_rejects_nonfinite_callback() {
        fn nonfinite_gradient(x: &[f64]) -> Vec<f64> {
            vec![f64::NAN; x.len()]
        }

        let result = bfgs(
            &sphere,
            &[2.0, -3.0],
            MinimizeOptions {
                method: Some(OptimizeMethod::Bfgs),
                gradient: Some(nonfinite_gradient),
                mode: RuntimeMode::Strict,
                ..MinimizeOptions::default()
            },
        )
        .expect("bfgs returns non-finite result");

        assert!(!result.success);
        assert_eq!(result.status, ConvergenceStatus::NanEncountered);
        assert!(
            result
                .message
                .contains("gradient callback returned NaN or Inf")
        );
    }

    #[test]
    fn bfgs_finds_local_minimum_on_nonconvex_surface() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            tol: Some(1.0e-7),
            maxiter: Some(800),
            maxfev: Some(80_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result = bfgs(&himmelblau, &[0.0, 0.0], options).expect("bfgs executes");
        assert!(result.fun.expect("objective") < 1.0e-5);
        push_test_log(
            "bfgs-nonconvex-local-min",
            "bfgs",
            "himmelblau",
            2,
            RuntimeMode::Strict,
            &result,
            106,
        );
    }

    #[test]
    fn bfgs_zero_gradient_at_start_converges_immediately() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            tol: Some(1.0e-8),
            maxiter: Some(200),
            maxfev: Some(20_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result = bfgs(&sphere, &[0.0, 0.0], options).expect("bfgs executes");
        assert!(result.success);
        assert_eq!(result.nit, 0);
        push_test_log(
            "bfgs-zero-gradient-start",
            "bfgs",
            "sphere",
            2,
            RuntimeMode::Strict,
            &result,
            107,
        );
    }

    #[test]
    fn bfgs_handles_very_flat_function() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            tol: Some(1.0e-7),
            maxiter: Some(800),
            maxfev: Some(80_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let initial = flat_quartic(&[0.5, -1.5]);
        let result = bfgs(&flat_quartic, &[0.5, -1.5], options).expect("bfgs executes");
        assert!(result.fun.expect("objective") <= initial);
        push_test_log(
            "bfgs-flat-function",
            "bfgs",
            "flat-quartic",
            2,
            RuntimeMode::Strict,
            &result,
            108,
        );
    }

    #[test]
    fn bfgs_hardened_mode_rejects_nan_gradient_path() {
        let objective = |x: &[f64]| {
            if x[0] < 0.0 { f64::NAN } else { x[0] * x[0] }
        };
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            mode: RuntimeMode::Hardened,
            ..MinimizeOptions::default()
        };
        let result = minimize(objective, &[0.0], options).expect("returns an OptimizeResult");
        assert!(!result.success);
        assert_eq!(result.status, ConvergenceStatus::NanEncountered);
        push_test_log(
            "bfgs-hardened-nan-gradient",
            "bfgs",
            "nan-gradient",
            1,
            RuntimeMode::Hardened,
            &result,
            109,
        );
    }

    #[test]
    fn bfgs_callback_receives_intermediate_points() {
        callback_points()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            callback: Some(callback_record_and_stop),
            maxiter: Some(40),
            maxfev: Some(10_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result = bfgs(&sphere, &[2.5, -1.5], options).expect("bfgs executes");
        let points = callback_points().lock().unwrap_or_else(|e| e.into_inner());
        assert!(!points.is_empty());
        assert_eq!(points[0], vec![2.5, -1.5]);
        assert_eq!(result.status, ConvergenceStatus::CallbackStop);
        push_test_log(
            "bfgs-callback-points",
            "bfgs",
            "sphere",
            2,
            RuntimeMode::Strict,
            &result,
            110,
        );
    }

    #[test]
    fn bfgs_gradient_tolerance_threshold_stops_early() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            tol: Some(10.0),
            maxiter: Some(40),
            maxfev: Some(10_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result = bfgs(&sphere, &[0.1, -0.1], options).expect("bfgs executes");
        assert!(result.success);
        assert_eq!(result.nit, 0);
        push_test_log(
            "bfgs-gtol-threshold",
            "bfgs",
            "sphere",
            2,
            RuntimeMode::Strict,
            &result,
            111,
        );
    }

    #[test]
    fn cg_quadratic_converges_with_small_iterations() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::ConjugateGradient),
            tol: Some(1.0e-8),
            maxiter: Some(80),
            maxfev: Some(20_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result = cg_pr_plus(&sphere, &[4.0, -1.5], options).expect("cg executes");
        assert!(result.success, "{}", result.message);
        assert!(result.nit <= 20);
        push_test_log(
            "cg-quadratic",
            "cg_pr_plus",
            "sphere",
            2,
            RuntimeMode::Strict,
            &result,
            112,
        );
    }

    #[test]
    fn cg_converges_on_quadratic() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::ConjugateGradient),
            tol: Some(1.0e-7),
            maxiter: Some(200),
            maxfev: Some(30_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result = cg_pr_plus(&sphere, &[4.0, -1.5], options).expect("cg executes");
        assert!(result.success, "{}", result.message);
        assert!(result.fun.expect("objective") < 1.0e-8);
        push_test_log(
            "cg-sphere-reference",
            "cg_pr_plus",
            "sphere",
            2,
            RuntimeMode::Strict,
            &result,
            113,
        );
    }

    #[test]
    fn cg_rosenbrock_reduces_objective() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::ConjugateGradient),
            tol: Some(1.0e-7),
            maxiter: Some(600),
            maxfev: Some(80_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let initial = rosenbrock(&[-1.2, 1.0]);
        let result = cg_pr_plus(&rosenbrock, &[-1.2, 1.0], options).expect("cg executes");
        assert!(result.fun.expect("objective") < initial);
        push_test_log(
            "cg-rosenbrock",
            "cg_pr_plus",
            "rosenbrock",
            2,
            RuntimeMode::Strict,
            &result,
            114,
        );
    }

    #[test]
    fn cg_rosenbrock_exact_gradient_uses_callback_jacobian() {
        let mut options = MinimizeOptions {
            method: Some(OptimizeMethod::ConjugateGradient),
            tol: Some(1.0e-7),
            maxiter: Some(600),
            maxfev: Some(80_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        options.gradient = Some(rosenbrock_gradient);

        let initial = rosenbrock(&[-1.2, 1.0]);
        let exact = cg_pr_plus(&rosenbrock, &[-1.2, 1.0], options).expect("cg exact");
        let expected_jac = rosenbrock_gradient(&exact.x);

        assert!(exact.fun.expect("objective") < initial);
        assert_eq!(exact.jac.as_ref().expect("jac"), &expected_jac);
        assert!(exact.njev > 0);
        push_test_log(
            "cg-rosenbrock-exact-gradient",
            "cg_pr_plus",
            "rosenbrock",
            2,
            RuntimeMode::Strict,
            &exact,
            118,
        );
    }

    #[test]
    fn cg_exact_gradient_rejects_bad_callback_shape() {
        fn bad_gradient(_: &[f64]) -> Vec<f64> {
            vec![0.0]
        }

        let result = cg_pr_plus(
            &sphere,
            &[4.0, -1.5],
            MinimizeOptions {
                method: Some(OptimizeMethod::ConjugateGradient),
                gradient: Some(bad_gradient),
                mode: RuntimeMode::Strict,
                ..MinimizeOptions::default()
            },
        )
        .expect("cg returns invalid-input result");

        assert!(!result.success);
        assert_eq!(result.status, ConvergenceStatus::InvalidInput);
        assert!(result.message.contains("gradient callback returned length"));
    }

    #[test]
    fn cg_high_dimensional_problem_converges() {
        let diag_quadratic = |x: &[f64]| {
            x.iter()
                .enumerate()
                .map(|(idx, value)| (idx as f64 + 1.0) * value * value)
                .sum::<f64>()
        };
        let x0 = vec![1.0; 50];
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::ConjugateGradient),
            tol: Some(1.0e-6),
            maxiter: Some(600),
            maxfev: Some(200_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result = cg_pr_plus(&diag_quadratic, &x0, options).expect("cg executes");
        assert!(result.fun.expect("objective") < 1.0e-6);
        push_test_log(
            "cg-high-dimensional",
            "cg_pr_plus",
            "diag-quadratic",
            50,
            RuntimeMode::Strict,
            &result,
            115,
        );
    }

    #[test]
    fn cg_handles_nonconvex_surface_without_invalid_status() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::ConjugateGradient),
            tol: Some(1.0e-6),
            maxiter: Some(200),
            maxfev: Some(100_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let initial = nonconvex_saddle(&[1.0, 1.0]);
        let result = cg_pr_plus(&nonconvex_saddle, &[1.0, 1.0], options).expect("cg executes");
        assert!(result.fun.expect("objective") <= initial + 1.0e-6);
        assert_ne!(result.status, ConvergenceStatus::InvalidInput);
        push_test_log(
            "cg-nonconvex",
            "cg_pr_plus",
            "saddle",
            2,
            RuntimeMode::Strict,
            &result,
            116,
        );
    }

    #[test]
    fn cg_zero_function_trivial_convergence() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::ConjugateGradient),
            tol: Some(1.0e-8),
            maxiter: Some(40),
            maxfev: Some(10_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result = cg_pr_plus(&zero_function, &[10.0, -10.0], options).expect("cg executes");
        assert!(result.success);
        assert_eq!(result.nit, 0);
        push_test_log(
            "cg-zero-function",
            "cg_pr_plus",
            "constant-zero",
            2,
            RuntimeMode::Strict,
            &result,
            117,
        );
    }

    #[test]
    fn powell_reduces_objective() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Powell),
            tol: Some(1.0e-5),
            maxiter: Some(50),
            maxfev: Some(40_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let initial = sphere(&[3.0, -2.0]);
        let result = powell(&sphere, &[3.0, -2.0], options).expect("powell executes");
        assert!(result.fun.expect("objective") < initial);
        push_test_log(
            "powell-reduces-objective",
            "powell",
            "sphere",
            2,
            RuntimeMode::Strict,
            &result,
            118,
        );
    }

    #[test]
    fn powell_quadratic_converges_near_exact_minimum() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Powell),
            tol: Some(1.0e-6),
            maxiter: Some(120),
            maxfev: Some(80_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result = powell(&sphere, &[2.0, -2.0], options).expect("powell executes");
        assert!(result.fun.expect("objective") < 1.0e-5);
        push_test_log(
            "powell-quadratic",
            "powell",
            "sphere",
            2,
            RuntimeMode::Strict,
            &result,
            119,
        );
    }

    #[test]
    fn powell_nonsmooth_objective_best_effort() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Powell),
            tol: Some(1.0e-6),
            maxiter: Some(120),
            maxfev: Some(80_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let initial = abs_sum(&[3.0, -4.0]);
        let result = powell(&abs_sum, &[3.0, -4.0], options).expect("powell executes");
        assert!(result.fun.expect("objective") <= initial);
        push_test_log(
            "powell-nonsmooth",
            "powell",
            "absolute-sum",
            2,
            RuntimeMode::Strict,
            &result,
            120,
        );
    }

    #[test]
    fn powell_one_dimensional_path_reduces_objective() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Powell),
            tol: Some(1.0e-8),
            maxiter: Some(100),
            maxfev: Some(50_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let initial = one_dim_quadratic(&[-3.0]);
        let result = powell(&one_dim_quadratic, &[-3.0], options).expect("powell executes");
        assert!(result.fun.expect("objective") < initial);
        push_test_log(
            "powell-one-dimensional",
            "powell",
            "one-dim-quadratic",
            1,
            RuntimeMode::Strict,
            &result,
            121,
        );
    }

    #[test]
    fn powell_constant_function_terminates_quickly() {
        let constant = |_x: &[f64]| 5.0;
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Powell),
            tol: Some(1.0e-8),
            maxiter: Some(100),
            maxfev: Some(50_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result = powell(&constant, &[3.0, -1.0], options).expect("powell executes");
        assert!(result.success);
        assert!(result.nit <= 1);
        push_test_log(
            "powell-constant-function",
            "powell",
            "constant",
            2,
            RuntimeMode::Strict,
            &result,
            122,
        );
    }

    #[test]
    fn minimize_scalar_many_byte_identical_to_per_param() {
        // 1-D minimization sweep: minimize (x - p0)^2 + p1 over a shared bracket for many params.
        // The batched solve must equal looping minimize_scalar per parameter, bit-for-bit.
        let f = |x: f64, p: &[f64]| (x - p[0]) * (x - p[0]) + p[1];
        let bracket = (-10.0, 10.0);
        let opts = MinimizeScalarOptions::default();
        let mut s = 13u64;
        let mut rng = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            -5.0 + 10.0 * ((s >> 11) as f64 / (1u64 << 53) as f64)
        };
        let nrows = 16usize; // crosses the serial->parallel gate
        let params: Vec<Vec<f64>> = (0..nrows).map(|_| vec![rng(), rng().abs()]).collect();

        let batched = minimize_scalar_many(f, bracket, &params, opts);
        assert_eq!(batched.len(), nrows);
        for (i, p) in params.iter().enumerate() {
            let single = minimize_scalar(|x| f(x, p), bracket, opts).expect("single");
            let many = batched[i].as_ref().expect("batched member");
            assert_eq!(many.x.to_bits(), single.x.to_bits(), "x mismatch param {i}");
            assert_eq!(
                many.fun.to_bits(),
                single.fun.to_bits(),
                "fun mismatch param {i}"
            );
            assert_eq!(many.success, single.success, "success mismatch param {i}");
        }
        assert!(
            batched
                .iter()
                .filter(|r| r.as_ref().map(|x| x.success).unwrap_or(false))
                .count()
                == nrows
        );
        assert!(minimize_scalar_many(f, bracket, &[], opts).is_empty());
    }

    #[test]
    fn golden_section_direction_search_returns_best_sample_when_no_bracket_exists() {
        let fun = |x: &[f64]| x[0].exp();
        let mut objective = Objective::new(&fun, RuntimeMode::Strict, 512);
        let search =
            golden_section_direction_search(&mut objective, &[0.0], fun(&[0.0]), &[1.0], 1.0e-6)
                .expect("line search succeeds");
        assert!(
            search.alpha < 0.0,
            "search should move downhill in the negative direction"
        );
        assert!(search.f.is_finite());
        assert!(
            search.f < 1.0,
            "best sampled point should improve the objective"
        );
    }

    #[test]
    fn minimize_many_byte_identical_to_per_start() {
        // Multistart over the 4-D Rosenbrock: the batched run must equal looping minimize
        // per start, bit-for-bit (each run is an independent deterministic optimisation).
        let rosen = |x: &[f64]| -> f64 {
            (0..x.len() - 1)
                .map(|i| 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2))
                .sum()
        };
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            tol: Some(1.0e-8),
            maxiter: Some(500),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let mut s = 7u64;
        let mut rng = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            -2.0 + 4.0 * ((s >> 11) as f64 / (1u64 << 53) as f64)
        };
        let nrows = 10usize; // crosses the serial->parallel gate
        let starts: Vec<Vec<f64>> = (0..nrows)
            .map(|_| (0..4).map(|_| rng()).collect())
            .collect();

        let batched = minimize_many(rosen, &starts, options);
        assert_eq!(batched.len(), nrows);
        for (i, x0) in starts.iter().enumerate() {
            let single = minimize(rosen, x0, options).expect("single");
            let many = batched[i].as_ref().expect("batched member");
            assert_eq!(many.x.len(), single.x.len());
            for (a, b) in many.x.iter().zip(single.x.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "x mismatch start {i}");
            }
            assert_eq!(
                many.fun.map(f64::to_bits),
                single.fun.map(f64::to_bits),
                "fun mismatch start {i}"
            );
        }
        assert!(minimize_many(rosen, &[], options).is_empty());
    }

    #[test]
    fn minimize_dispatches_to_selected_method() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            tol: Some(1.0e-7),
            maxiter: Some(120),
            maxfev: Some(20_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result = minimize(sphere, &[1.0, -1.0], options).expect("minimize executes");
        assert!(result.success, "{}", result.message);
        push_test_log(
            "dispatch-selected-method",
            "minimize",
            "sphere",
            2,
            RuntimeMode::Strict,
            &result,
            123,
        );
    }

    #[test]
    fn hardened_mode_rejects_non_finite_objective_values() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            mode: RuntimeMode::Hardened,
            ..MinimizeOptions::default()
        };
        let result = minimize(|_| f64::NAN, &[1.0, 2.0], options)
            .expect("execution should return an OptimizeResult");
        assert!(!result.success);
        assert_eq!(result.status, ConvergenceStatus::NanEncountered);
        push_test_log(
            "hardened-non-finite",
            "minimize",
            "nan-objective",
            2,
            RuntimeMode::Hardened,
            &result,
            124,
        );
    }

    #[test]
    fn minimize_with_audit_emits_fail_closed_for_hardened_nan() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            mode: RuntimeMode::Hardened,
            ..MinimizeOptions::default()
        };
        let ledger = crate::audit::sync_audit_ledger();
        let result = super::minimize_with_audit(|_| f64::NAN, &[0.0], options, &ledger)
            .expect("execution should return an OptimizeResult");

        assert_eq!(result.status, ConvergenceStatus::NanEncountered);
        let guard = ledger.lock().expect("audit ledger lock");
        assert_eq!(guard.len(), 1);
        assert!(matches!(
            guard.entries()[0].action,
            fsci_runtime::AuditAction::FailClosed { .. }
        ));
    }

    #[test]
    fn trace_log_contains_iteration_and_completion_events() {
        const FIXTURE_ID: &str = "trace-log-bfgs-iter";
        let _ = get_optimize_traces(); // clear stale traces from parallel tests
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            tol: Some(1.0e-6),
            maxiter: Some(40),
            maxfev: Some(20_000),
            mode: RuntimeMode::Strict,
            fixture_id: Some(FIXTURE_ID),
            ..MinimizeOptions::default()
        };
        let _ = minimize(sphere, &[1.0, 1.0], options).expect("execution succeeds");
        let traces = get_optimize_traces();
        // Filter to only our BFGS traces to avoid interference from parallel tests
        let bfgs_traces: Vec<_> = traces
            .iter()
            .filter(|entry| {
                entry.method == OptimizeMethod::Bfgs
                    && entry.fixture_id.as_deref() == Some(FIXTURE_ID)
            })
            .collect();
        assert!(
            !bfgs_traces.is_empty(),
            "expected BFGS traces but found none"
        );
        assert!(
            bfgs_traces.iter().any(|entry| entry.event == "iteration"),
            "expected iteration trace, found events: {:?}",
            bfgs_traces
                .iter()
                .map(|entry| &entry.event)
                .collect::<Vec<_>>()
        );
        assert!(bfgs_traces.iter().any(|entry| entry.event == "completion"));
        let completion = bfgs_traces
            .iter()
            .find(|entry| entry.event == "completion")
            .expect("completion trace exists");
        let result = OptimizeResult {
            x: completion.final_x.clone().unwrap_or_default(),
            fun: completion.final_f,
            success: completion
                .reason
                .as_deref()
                .is_some_and(|reason| reason.contains("Success")),
            status: ConvergenceStatus::Success,
            message: completion.reason.clone().unwrap_or_default(),
            nfev: completion.total_nfev,
            njev: 0,
            nhev: 0,
            nit: completion.iter_num,
            jac: None,
            hess_inv: None,
            maxcv: None,
        };
        push_test_log(
            "trace-log-schema",
            "bfgs",
            "sphere",
            2,
            RuntimeMode::Strict,
            &result,
            125,
        );
    }

    #[test]
    fn invalid_gradient_epsilon_is_rejected() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            gradient_eps: 0.0,
            ..MinimizeOptions::default()
        };
        let err = bfgs(&sphere, &[1.0, 1.0], options).expect_err("invalid options should fail");
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
    }

    #[test]
    fn evaluation_budget_exceeded_maps_to_max_evaluations() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            maxfev: Some(1),
            ..MinimizeOptions::default()
        };
        let result = minimize(sphere, &[1.0, 1.0], options).expect("execution returns result");
        assert_eq!(result.status, ConvergenceStatus::MaxEvaluations);
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 500,
            failure_persistence: None,
            .. ProptestConfig::default()
        })]

        #[test]
        fn property_minimize_improves_objective(
            x0 in proptest::array::uniform2(-4.0f64..4.0f64),
            method_pick in 0u8..3u8,
        ) {
            let method = match method_pick % 3 {
                0 => OptimizeMethod::Bfgs,
                1 => OptimizeMethod::ConjugateGradient,
                _ => OptimizeMethod::Powell,
            };
            let options = MinimizeOptions {
                method: Some(method),
                tol: Some(1.0e-6),
                maxiter: Some(40),
                maxfev: Some(20_000),
                mode: RuntimeMode::Strict,
                ..MinimizeOptions::default()
            };
            let result = minimize(sphere, &x0, options).expect("minimize executes");
            prop_assert!(result.fun.is_some());
            let final_f = result.fun.unwrap_or(f64::INFINITY);
            prop_assert!(final_f <= sphere(&x0) + 1.0e-8);
            push_test_log(
                "property-minimize-improves-objective",
                "minimize",
                "sphere",
                2,
                RuntimeMode::Strict,
                &result,
                201,
            );
        }

        #[test]
        fn property_nfev_is_non_zero_for_non_trivial_runs(
            x0 in proptest::array::uniform2(-3.0f64..3.0f64),
        ) {
            let norm = (x0[0] * x0[0] + x0[1] * x0[1]).sqrt();
            prop_assume!(norm > 1.0e-4);
            let options = MinimizeOptions {
                method: Some(OptimizeMethod::Bfgs),
                tol: Some(1.0e-9),
                maxiter: Some(5),
                maxfev: Some(5_000),
                mode: RuntimeMode::Strict,
                ..MinimizeOptions::default()
            };
            let result = bfgs(&sphere, &x0, options).expect("bfgs executes");
            prop_assert!(result.nfev >= 1);
            push_test_log(
                "property-nfev-non-zero",
                "bfgs",
                "sphere",
                2,
                RuntimeMode::Strict,
                &result,
                202,
            );
        }

        #[test]
        fn property_bfgs_hessian_inverse_is_symmetric_positive_diagonal(
            x0 in proptest::array::uniform2(-2.5f64..2.5f64),
        ) {
            let norm = (x0[0] * x0[0] + x0[1] * x0[1]).sqrt();
            prop_assume!(norm > 0.2);
            let options = MinimizeOptions {
                method: Some(OptimizeMethod::Bfgs),
                tol: Some(1.0e-6),
                maxiter: Some(40),
                maxfev: Some(40_000),
                mode: RuntimeMode::Strict,
                ..MinimizeOptions::default()
            };
            let result = bfgs(&sphere, &x0, options).expect("bfgs executes");
            let h_inv = result.hess_inv.as_ref().expect("hessian inverse is present");
            prop_assert_eq!(h_inv.len(), 2);
            prop_assert!((h_inv[0][1] - h_inv[1][0]).abs() <= 1.0e-6);
            prop_assert!(h_inv[0][0] > 0.0);
            prop_assert!(h_inv[1][1] > 0.0);
            push_test_log(
                "property-bfgs-hinv-spd",
                "bfgs",
                "sphere",
                2,
                RuntimeMode::Strict,
                &result,
                203,
            );
        }

        #[test]
        fn property_cg_first_iteration_is_descent(
            x0 in proptest::array::uniform2(-4.0f64..4.0f64),
        ) {
            let norm = (x0[0] * x0[0] + x0[1] * x0[1]).sqrt();
            prop_assume!(norm > 0.2);
            let _ = get_optimize_traces();
            let options = MinimizeOptions {
                method: Some(OptimizeMethod::ConjugateGradient),
                tol: Some(1.0e-6),
                maxiter: Some(40),
                maxfev: Some(40_000),
                fixture_id: Some("property-cg-descent"),
                mode: RuntimeMode::Strict,
                ..MinimizeOptions::default()
            };
            let initial = sphere(&x0);
            let result = cg_pr_plus(&sphere, &x0, options).expect("cg executes");
            let traces = get_optimize_traces();
            let first_iteration = traces
                .iter()
                .find(|entry| {
                    entry.event == "iteration"
                        && entry.method == OptimizeMethod::ConjugateGradient
                        && entry.fixture_id.as_deref() == Some("property-cg-descent")
                });
            if let Some(first) = first_iteration {
                prop_assert!(first.grad_norm.unwrap_or(f64::INFINITY).is_finite());
                let first_f = first.f_val.unwrap_or(f64::INFINITY);
                let final_f = result.fun.unwrap_or(f64::INFINITY);
                prop_assert!(first_f <= initial + 1.0e-6 || final_f <= initial + 1.0e-8);
            }
            push_test_log(
                "property-cg-descent",
                "cg_pr_plus",
                "sphere",
                2,
                RuntimeMode::Strict,
                &result,
                204,
            );
        }
    }

    // ── minimize_scalar tests ───────────────────────────────────────

    #[test]
    fn minimize_scalar_quadratic() {
        // f(x) = (x - 3)^2, minimum at x = 3
        let result = minimize_scalar(
            |x| (x - 3.0).powi(2),
            (0.0, 10.0),
            MinimizeScalarOptions::default(),
        )
        .expect("minimize_scalar works");
        assert!(result.success, "should converge");
        assert!(
            (result.x - 3.0).abs() < 1e-6,
            "minimizer should be near 3, got {}",
            result.x
        );
        assert!(result.fun < 1e-10, "minimum value should be near 0");
    }

    #[test]
    fn minimize_scalar_cos() {
        // f(x) = cos(x), minimum at x = pi in [2, 4]
        let result = minimize_scalar(f64::cos, (2.0, 4.0), MinimizeScalarOptions::default())
            .expect("minimize_scalar works");
        assert!(result.success);
        assert!(
            (result.x - std::f64::consts::PI).abs() < 1e-6,
            "minimizer should be near pi, got {}",
            result.x
        );
        assert!(
            (result.fun - (-1.0)).abs() < 1e-10,
            "min value should be -1"
        );
    }

    #[test]
    fn minimize_scalar_linear_converges() {
        // f(x) = x, monotone, should find the left endpoint
        let result = minimize_scalar(|x| x, (0.0, 10.0), MinimizeScalarOptions::default())
            .expect("minimize_scalar works");
        assert!(result.x < 1.0, "should be near 0, got {}", result.x);
    }

    #[test]
    fn minimize_scalar_equal_bounds_error() {
        let err = minimize_scalar(|x| x * x, (5.0, 5.0), MinimizeScalarOptions::default())
            .expect_err("equal bounds");
        assert!(matches!(err, OptError::InvalidBounds { .. }));
    }

    #[test]
    fn minimize_scalar_nan_bounds_error() {
        let err = minimize_scalar(|x| x * x, (f64::NAN, 5.0), MinimizeScalarOptions::default())
            .expect_err("nan bounds");
        assert!(matches!(err, OptError::NonFiniteInput { .. }));
    }

    #[test]
    fn minimize_scalar_reversed_bracket() {
        // (10, 0) should be auto-swapped to (0, 10)
        let result = minimize_scalar(
            |x| (x - 3.0).powi(2),
            (10.0, 0.0),
            MinimizeScalarOptions::default(),
        )
        .expect("minimize_scalar works");
        assert!(result.success);
        assert!(
            (result.x - 3.0).abs() < 1e-6,
            "minimizer should be near 3, got {}",
            result.x
        );
    }

    // ── Nelder-Mead tests ───────────────────────────────────────────

    #[test]
    fn nelder_mead_sphere_converges() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::NelderMead),
            tol: Some(1e-8),
            maxiter: Some(1000),
            maxfev: Some(5000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result = minimize(sphere, &[2.0, -3.0], options).expect("minimize executes");
        assert!(result.success, "should converge: {}", result.message);
        assert!(result.x.iter().all(|v| v.abs() < 1e-4), "x={:?}", result.x);
        assert!(result.fun.unwrap() < 1e-8);
        push_test_log(
            "nelder-mead-sphere",
            "nelder_mead",
            "sphere",
            2,
            RuntimeMode::Strict,
            &result,
            200,
        );
    }

    #[test]
    fn nelder_mead_rosenbrock() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::NelderMead),
            tol: Some(1e-8),
            maxiter: Some(5000),
            maxfev: Some(20_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result = minimize(rosenbrock, &[-1.0, 1.0], options).expect("minimize executes");
        assert!(result.success, "should converge: {}", result.message);
        assert!(
            (result.x[0] - 1.0).abs() < 1e-3 && (result.x[1] - 1.0).abs() < 1e-3,
            "x={:?}",
            result.x
        );
    }

    #[test]
    fn nelder_mead_1d() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::NelderMead),
            tol: Some(1e-10),
            ..MinimizeOptions::default()
        };
        let result = minimize(one_dim_quadratic, &[5.0], options).expect("minimize executes");
        assert!(result.success, "should converge: {}", result.message);
        assert!(
            (result.x[0] - 1.5).abs() < 1e-4,
            "minimizer should be near 1.5, got {}",
            result.x[0]
        );
    }

    #[test]
    fn nelder_mead_flat_function() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::NelderMead),
            maxiter: Some(500),
            ..MinimizeOptions::default()
        };
        let result = minimize(zero_function, &[1.0, 2.0], options).expect("minimize executes");
        // Should converge since f is constant everywhere
        assert!(result.success, "{}", result.message);
        assert_eq!(result.fun, Some(0.0));
    }

    #[test]
    fn nelder_mead_max_iter_reached() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::NelderMead),
            maxiter: Some(2),
            maxfev: Some(10_000),
            tol: Some(1e-15),
            ..MinimizeOptions::default()
        };
        let result = minimize(rosenbrock, &[5.0, 5.0], options).expect("minimize executes");
        assert!(!result.success);
        assert_eq!(result.status, ConvergenceStatus::MaxIterations);
    }

    #[test]
    fn nelder_mead_callback_stops() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::NelderMead),
            callback: Some(callback_record_and_stop),
            ..MinimizeOptions::default()
        };
        callback_points()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        let result = minimize(sphere, &[2.0, -3.0], options).expect("minimize executes");
        assert!(!result.success);
        assert_eq!(result.status, ConvergenceStatus::CallbackStop);
    }

    #[test]
    fn nelder_mead_empty_x0_rejected() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::NelderMead),
            ..MinimizeOptions::default()
        };
        let err = minimize(sphere, &[], options).expect_err("should reject empty");
        assert!(matches!(err, OptError::InvalidArgument { .. }));
    }

    #[test]
    fn nelder_mead_nonfinite_x0_rejected() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::NelderMead),
            ..MinimizeOptions::default()
        };
        let err = minimize(sphere, &[f64::NAN, 1.0], options).expect_err("should reject NaN");
        assert!(matches!(err, OptError::NonFiniteInput { .. }));
    }

    #[test]
    fn nelder_mead_himmelblau() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::NelderMead),
            tol: Some(1e-8),
            maxiter: Some(2000),
            maxfev: Some(10_000),
            ..MinimizeOptions::default()
        };
        let result = minimize(himmelblau, &[0.0, 0.0], options).expect("minimize executes");
        assert!(result.success, "should converge: {}", result.message);
        // Himmelblau has 4 minima, all with f=0
        assert!(result.fun.unwrap() < 1e-6, "f={:?}", result.fun);
    }

    #[test]
    fn nelder_mead_higher_dim() {
        // 5-D sphere
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::NelderMead),
            tol: Some(1e-6),
            maxiter: Some(5000),
            maxfev: Some(50_000),
            ..MinimizeOptions::default()
        };
        let result =
            minimize(sphere, &[1.0, -2.0, 3.0, -4.0, 5.0], options).expect("minimize executes");
        assert!(result.success, "should converge: {}", result.message);
        assert!(result.x.iter().all(|v| v.abs() < 1e-3), "x={:?}", result.x);
    }

    #[test]
    fn nelder_mead_traces_are_emitted() {
        let _ = get_optimize_traces(); // clear
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::NelderMead),
            maxiter: Some(10),
            ..MinimizeOptions::default()
        };
        let _ = minimize(sphere, &[1.0, 1.0], options);
        let traces = get_optimize_traces();
        let nm_traces: Vec<_> = traces
            .iter()
            .filter(|t| t.method == OptimizeMethod::NelderMead)
            .collect();
        assert!(
            !nm_traces.is_empty(),
            "nelder-mead should emit trace entries"
        );
    }

    // ── L-BFGS-B tests ─────────────────────────────────────────────

    #[test]
    fn lbfgsb_unconstrained_sphere() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::LBfgsB),
            tol: Some(1e-8),
            ..MinimizeOptions::default()
        };
        let result = minimize(sphere, &[2.0, -3.0], options).expect("minimize");
        assert!(result.success, "should converge: {}", result.message);
        assert!(result.x.iter().all(|v| v.abs() < 1e-4), "x={:?}", result.x);
    }

    #[test]
    fn lbfgsb_with_bounds() {
        use crate::minimize::lbfgsb;
        // Minimize (x-3)^2 + (y-3)^2 with bounds x in [0, 2], y in [0, 2]
        // Optimum should be at (2, 2)
        let f = |x: &[f64]| (x[0] - 3.0).powi(2) + (x[1] - 3.0).powi(2);
        let bounds = vec![(Some(0.0), Some(2.0)), (Some(0.0), Some(2.0))];
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::LBfgsB),
            tol: Some(1e-8),
            ..MinimizeOptions::default()
        };
        let result = lbfgsb(&f, &[0.0, 0.0], options, Some(&bounds)).expect("lbfgsb");
        assert!(result.success, "should converge: {}", result.message);
        assert!(
            (result.x[0] - 2.0).abs() < 0.01,
            "x[0] should be at upper bound 2, got {}",
            result.x[0]
        );
        assert!(
            (result.x[1] - 2.0).abs() < 0.01,
            "x[1] should be at upper bound 2, got {}",
            result.x[1]
        );
    }

    #[test]
    fn lbfgsb_lower_bound_only() {
        use crate::minimize::lbfgsb;
        // Minimize (x+5)^2 with lower bound x >= 0. Optimum at x=0.
        let f = |x: &[f64]| (x[0] + 5.0).powi(2);
        let bounds = vec![(Some(0.0), None)];
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::LBfgsB),
            tol: Some(1e-8),
            ..MinimizeOptions::default()
        };
        let result = lbfgsb(&f, &[10.0], options, Some(&bounds)).expect("lbfgsb");
        assert!(result.success, "should converge: {}", result.message);
        assert!(
            result.x[0].abs() < 0.01,
            "x should be at lower bound 0, got {}",
            result.x[0]
        );
    }

    #[test]
    fn lbfgsb_rosenbrock_unconstrained() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::LBfgsB),
            tol: Some(1e-6),
            maxiter: Some(2000),
            maxfev: Some(20_000),
            ..MinimizeOptions::default()
        };
        let result = minimize(rosenbrock, &[0.0, 0.0], options).expect("minimize");
        assert!(result.success, "should converge: {}", result.message);
        assert!(
            (result.x[0] - 1.0).abs() < 0.01 && (result.x[1] - 1.0).abs() < 0.01,
            "x={:?}",
            result.x
        );
    }

    #[test]
    fn lbfgsb_and_cg_converge_on_hard_rosenbrock_start() {
        // Standard hard Rosenbrock start [-1.2, 1.0]. Before the strong-Wolfe
        // line search both L-BFGS-B and CG stalled (Armijo-only steps gave invalid
        // curvature pairs / lost conjugacy), ending far from the [1,1] minimum.
        for method in [OptimizeMethod::LBfgsB, OptimizeMethod::ConjugateGradient] {
            let options = MinimizeOptions {
                method: Some(method),
                tol: Some(1e-6),
                maxiter: Some(2000),
                maxfev: Some(50_000),
                ..MinimizeOptions::default()
            };
            let r = minimize(rosenbrock, &[-1.2, 1.0], options).expect("minimize");
            assert!(r.success, "{method:?} should converge: {}", r.message);
            assert!(
                (r.x[0] - 1.0).abs() < 1e-3 && (r.x[1] - 1.0).abs() < 1e-3,
                "{method:?} x={:?}",
                r.x
            );
        }
    }

    // ── Newton-CG tests ─────────────────────────────────────────────

    #[test]
    fn newton_cg_sphere_converges() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::NewtonCg),
            tol: Some(1e-8),
            ..MinimizeOptions::default()
        };
        let result = minimize(sphere, &[2.0, -3.0], options).expect("minimize");
        assert!(result.success, "should converge: {}", result.message);
        assert!(result.x.iter().all(|v| v.abs() < 1e-3), "x={:?}", result.x);
    }

    #[test]
    fn newton_cg_rosenbrock() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::NewtonCg),
            tol: Some(1e-6),
            maxiter: Some(500),
            maxfev: Some(50_000),
            ..MinimizeOptions::default()
        };
        let result = minimize(rosenbrock, &[0.0, 0.0], options).expect("minimize");
        assert!(result.success, "should converge: {}", result.message);
        assert!(
            (result.x[0] - 1.0).abs() < 0.05 && (result.x[1] - 1.0).abs() < 0.05,
            "x={:?}",
            result.x
        );
    }

    #[test]
    fn newton_cg_1d_quadratic() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::NewtonCg),
            tol: Some(1e-10),
            ..MinimizeOptions::default()
        };
        let result = minimize(one_dim_quadratic, &[5.0], options).expect("minimize");
        assert!(result.success, "should converge: {}", result.message);
        assert!(
            (result.x[0] - 1.5).abs() < 1e-4,
            "minimizer should be 1.5, got {}",
            result.x[0]
        );
    }

    fn quadratic_hessp(_x: &[f64], p: &[f64]) -> Vec<f64> {
        vec![4.0 * p[0]]
    }

    #[test]
    fn newton_cg_uses_hessian_product_api_under_tight_eval_budget() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::NewtonCg),
            tol: Some(1e-10),
            maxiter: Some(8),
            maxfev: Some(10),
            hessp: Some(quadratic_hessp),
            ..MinimizeOptions::default()
        };
        let result = minimize(|x| 2.0 * x[0] * x[0], &[3.0], options).expect("minimize");
        assert!(
            result.success,
            "should converge via hessp: {}",
            result.message
        );
        assert!(result.nfev <= 10, "hessp path should stay within budget");
        assert!(
            result.nhev > 0,
            "explicit hessp calls should be counted as Hessian evaluations"
        );
        assert!(
            result.njev <= 3,
            "hessp calls should not be counted as gradient evaluations"
        );
        assert!(
            result.x[0].abs() < 1e-8,
            "minimizer should be near zero, got {}",
            result.x[0]
        );
    }

    /// frankenscipy-ei0az. Every entry point clamped the caller's tolerance with
    /// `.max(1.0e-12)`, so an objective whose gradient is everywhere below that
    /// satisfied `grad_norm <= tol` at iteration zero and the routine returned
    /// the STARTING POINT reporting success — with no setting that could change
    /// it, since a requested 1e-30 was silently raised to 1e-12.
    ///
    /// Measured on worker vmi1227854 before the fix, on this fixture with
    /// tol = 1e-30: success=true with nit=0 and 2.236 from the minimizer at
    /// 2^-53, 2^-56 and 2^-60, and already 4.886e-1 away at 2^-40.
    /// `scipy.optimize.minimize(method='trust-exact', gtol=0)` reaches the
    /// minimizer to 2.220e-16 at every one of those scales, so the arithmetic
    /// was never the problem — only the stopping rule.
    /// frankenscipy-uluc9. Four constants in trust-exact described the SCALE of
    /// the problem rather than its geometry, and together they made a scaled
    /// objective unsolvable no matter what the caller asked for: the initial
    /// Hessian (identity, i.e. "curvature is order one"), the first-update
    /// rescale, the Cholesky positivity floor in the subproblem solver, and the
    /// λ shift the boundary search starts from.
    ///
    /// The fair incumbent arm is the point of this bead, and it is not the
    /// analytic-Hessian one: handing SciPy the true Hessian compares against a
    /// better-informed solver than ours. Given a finite-difference gradient AND
    /// Hessian — the curvature ours derives — `scipy.optimize.minimize(method=
    /// 'trust-exact', gtol=0)` reaches the minimizer EXACTLY in 5 iterations at
    /// every scale from 2^-0 to 2^-60 (harness `scripts/scipy_scale_probe.py`,
    /// section "trust-exact with DERIVED curvature"). So this was ours to close,
    /// not a property of the problem.
    ///
    /// Measured on worker vmi1227854 across the fix, tol = 1e-30, from (0,0):
    ///   scale    before                          after
    ///   2^-0     22 iterations, exact            22 iterations, exact
    ///   2^-40    53 iterations, exact            22 iterations, exact
    ///   2^-53    0.394 away after 5000 iters     9 iterations, exact, success
    #[test]
    fn trust_exact_reaches_the_minimizer_at_every_scale_the_incumbent_does() {
        for exponent in [0_i32, 20, 40, 46, 53, 60] {
            let scale = 2.0_f64.powi(-exponent);
            let options = MinimizeOptions {
                method: Some(OptimizeMethod::TrustExact),
                // Deliberately far below anything this problem can satisfy. An
                // ordinary tolerance would make this test vacuous rather than
                // strict: `grad_norm <= tol` is an ABSOLUTE test, so at 2^-40
                // the starting gradient is already 6.1e-12 and a tol of 1e-10
                // is met at iteration zero — correctly, and SciPy's default
                // gtol stops at nit=0 on the same input. Asking for 1e-30
                // forces the solver to actually go and find the minimizer.
                tol: Some(1e-30),
                maxiter: Some(200),
                maxfev: Some(100_000),
                ..MinimizeOptions::default()
            };
            let result = minimize(
                move |x: &[f64]| scale * ((x[0] - 1.0).powi(2) + 2.0 * (x[1] + 2.0).powi(2)),
                &[0.0, 0.0],
                options,
            )
            .expect("minimize");
            let distance = ((result.x[0] - 1.0).powi(2) + (result.x[1] + 2.0).powi(2)).sqrt();
            assert!(
                distance < 1e-9,
                "at scale 2^-{exponent} we stopped {distance:.3e} from (1, -2) after {} \
                 iterations ({} evals, status {:?}); with derived curvature the peer \
                 reaches it exactly in 5",
                result.nit,
                result.nfev,
                result.status
            );
            // The iteration count must not blow up with the scaling either —
            // before the fix 2^-40 took 53 iterations against 22 unscaled, and
            // 2^-53 never arrived at all (0.394 away after 5000).
            assert!(
                result.nit <= 40,
                "at scale 2^-{exponent} convergence took {} iterations; the unscaled problem \
                 takes 22 and scaling the objective does not change its geometry",
                result.nit
            );
        }
    }

    #[test]
    fn a_requested_tolerance_is_not_silently_raised_for_a_scaled_objective() {
        for exponent in [0_i32, 20, 40, 53, 60] {
            let scale = 2.0_f64.powi(-exponent);
            let options = MinimizeOptions {
                method: Some(OptimizeMethod::TrustExact),
                tol: Some(1e-30),
                maxiter: Some(200),
                maxfev: Some(20_000),
                ..MinimizeOptions::default()
            };
            let result = minimize(
                move |x: &[f64]| scale * ((x[0] - 1.0).powi(2) + 2.0 * (x[1] + 2.0).powi(2)),
                &[0.0, 0.0],
                options,
            )
            .expect("minimize");
            let distance = ((result.x[0] - 1.0).powi(2) + (result.x[1] + 2.0).powi(2)).sqrt();

            // The defect itself, at every scale: a caller who asked for 1e-30
            // must never be told `success` while sitting on the starting point.
            assert!(
                !(result.success && result.nit == 0),
                "at scale 2^-{exponent} we reported success at iteration 0, {distance:.3e} from \
                 the minimizer, for a caller who asked for tol = 1e-30"
            );
            // The routine reaches the minimizer at EVERY scale here — the
            // sub-2^-46 residue this test used to concede was closed by
            // frankenscipy-uluc9. `success` is deliberately not asserted: this
            // run asks for 1e-30, which `grad_norm <= tol` can never satisfy, so
            // the well-scaled cases arrive at the minimizer and then exit on a
            // zero step. Arriving is the claim; the label is the other test's.
            assert!(
                distance < 1e-9,
                "at scale 2^-{exponent} the minimizer is (1, -2) and we stopped \
                 {distance:.3e} away after {} iterations reporting success={}",
                result.nit,
                result.success
            );
        }
    }

    /// Negative cases for frankenscipy-ei0az. Honouring a tight tolerance must
    /// not turn into never terminating, and must not break the ordinary paths.
    #[test]
    fn an_unreachable_tolerance_still_terminates_on_iterations() {
        // A tolerance this problem cannot reach in the budget must terminate on
        // `maxiter` — which every entry point already bounds, and which is why
        // the removed clamp bought nothing.
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::TrustExact),
            tol: Some(f64::MIN_POSITIVE),
            maxiter: Some(25),
            maxfev: Some(20_000),
            ..MinimizeOptions::default()
        };
        let result = minimize(rosenbrock, &[-1.2, 1.0], options).expect("minimize returns");
        assert!(
            result.nit <= 25,
            "an unreachable tolerance must terminate on maxiter, ran {} iterations",
            result.nit
        );

        // Non-positive and non-finite tolerances are rejected by the API rather
        // than clamped — that contract predates this change and must survive it,
        // since "clamp it silently" is exactly what was wrong with the old floor.
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let options = MinimizeOptions {
                method: Some(OptimizeMethod::TrustExact),
                tol: Some(bad),
                maxiter: Some(200),
                maxfev: Some(20_000),
                ..MinimizeOptions::default()
            };
            let outcome = minimize(
                |x: &[f64]| (x[0] - 1.0).powi(2) + 2.0 * (x[1] + 2.0).powi(2),
                &[0.0, 0.0],
                options,
            );
            assert!(
                matches!(outcome, Err(OptError::InvalidArgument { .. })),
                "tol = {bad} must be rejected, got {outcome:?}"
            );
        }

        // An objective that genuinely starts AT its minimizer must still stop
        // immediately — nit = 0 there is the correct answer, not the defect.
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::TrustExact),
            tol: Some(1e-12),
            maxiter: Some(200),
            maxfev: Some(20_000),
            ..MinimizeOptions::default()
        };
        let result = minimize(
            |x: &[f64]| (x[0] - 1.0).powi(2) + 2.0 * (x[1] + 2.0).powi(2),
            &[1.0, -2.0],
            options,
        )
        .expect("minimize");
        assert!(
            result.success && result.nit == 0,
            "starting at the minimizer must converge immediately, got nit={} success={}",
            result.nit,
            result.success
        );
    }

    #[test]
    fn trust_exact_shifted_quadratic_converges() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::TrustExact),
            tol: Some(1e-10),
            maxiter: Some(100),
            maxfev: Some(20_000),
            ..MinimizeOptions::default()
        };
        let result = minimize(
            |x| (x[0] - 1.0).powi(2) + 4.0 * (x[1] + 2.0).powi(2),
            &[8.0, -8.0],
            options,
        )
        .expect("minimize");
        assert!(result.success, "should converge: {}", result.message);
        assert!((result.x[0] - 1.0).abs() < 1.0e-6, "x={:?}", result.x);
        assert!((result.x[1] + 2.0).abs() < 1.0e-6, "x={:?}", result.x);
        assert!(result.fun.unwrap_or(f64::INFINITY) < 1.0e-10);
        assert!(result.nhev > 0);
    }

    #[test]
    fn flat_augmented_solver_is_bit_identical() {
        let cases = [
            (vec![vec![4.0, 1.0], vec![2.0, 3.0]], vec![1.0, -2.0]),
            (
                vec![
                    vec![0.0, 2.0, -1.0],
                    vec![3.0, -1.0, 2.0],
                    vec![1.0, 4.0, 5.0],
                ],
                vec![1.0, 7.0, -3.0],
            ),
            (vec![vec![1.0, 2.0], vec![2.0, 4.0]], vec![3.0, 6.0]),
        ];

        for (matrix, rhs) in cases {
            let n = matrix.len();
            let stride = n + 1;
            let mut nested = vec![vec![0.0; stride]; n];
            let mut flat = vec![0.0; n * stride];
            for row in 0..n {
                nested[row][..n].copy_from_slice(&matrix[row]);
                nested[row][n] = rhs[row];
                let row_base = row * stride;
                flat[row_base..row_base + n].copy_from_slice(&matrix[row]);
                flat[row_base + n] = rhs[row];
            }

            let expected = super::solve_augmented_nested(nested, n);
            let actual = super::solve_augmented_flat(flat, n);
            assert_eq!(
                expected.is_some(),
                actual.is_some(),
                "flat solve changed singularity verdict"
            );
            if let (Some(expected), Some(actual)) = (expected, actual) {
                assert_eq!(expected.len(), actual.len());
                for (index, (&expected, &actual)) in expected.iter().zip(actual.iter()).enumerate()
                {
                    assert_eq!(
                        expected.to_bits(),
                        actual.to_bits(),
                        "flat solve changed solution bits at index {index}"
                    );
                }
            }
        }
    }

    #[test]
    fn trust_spd_cholesky_matches_pivoted_solve() {
        let cases = [
            (vec![vec![4.0, 1.0], vec![1.0, 3.0]], vec![1.0, 2.0], 0.0),
            (
                vec![
                    vec![9.0, 3.0, 1.0],
                    vec![3.0, 5.0, 2.0],
                    vec![1.0, 2.0, 4.0],
                ],
                vec![-2.0, 7.0, 3.0],
                0.25,
            ),
        ];

        for (matrix, rhs, lambda) in cases {
            let pivoted =
                super::solve_trust_pivoted_system(&matrix, lambda, &rhs).expect("pivoted solve");
            let cholesky =
                super::solve_trust_spd_system(&matrix, lambda, &rhs).expect("Cholesky solve");
            for (index, (&pivoted, &cholesky)) in pivoted.iter().zip(cholesky.iter()).enumerate() {
                assert!(
                    (pivoted - cholesky).abs() <= 1.0e-12 * pivoted.abs().max(1.0),
                    "solution mismatch at index {index}: pivoted={pivoted}, cholesky={cholesky}"
                );
            }
        }

        let indefinite = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
        let rhs = vec![3.0, -1.0];
        let pivoted =
            super::solve_trust_pivoted_system(&indefinite, 0.0, &rhs).expect("pivoted solve");
        let fallback =
            super::solve_trust_spd_system(&indefinite, 0.0, &rhs).expect("fallback solve");
        assert_eq!(
            pivoted
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            fallback
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn trust_exact_rosenbrock_converges() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::TrustExact),
            tol: Some(1e-6),
            maxiter: Some(200),
            maxfev: Some(50_000),
            ..MinimizeOptions::default()
        };
        let result = minimize(rosenbrock, &[0.0, 0.0], options).expect("minimize");
        assert!(result.success, "should converge: {}", result.message);
        assert!(
            (result.x[0] - 1.0).abs() < 0.05 && (result.x[1] - 1.0).abs() < 0.05,
            "x={:?}",
            result.x
        );
    }

    /// Higher-dimensional Rosenbrock from a far, badly-scaled start. The previous
    /// finite-difference-Hessian build STALLED here (trust radius collapsed at
    /// f ≈ 3.99, ~2 away from the all-ones optimum); the BFGS quasi-Newton model
    /// converges to the true minimizer (and uses several-fold fewer evaluations).
    #[test]
    fn trust_exact_high_dim_rosenbrock_converges() {
        fn rosenbrock_nd(x: &[f64]) -> f64 {
            let mut acc = 0.0;
            for i in 0..x.len() - 1 {
                let a = 1.0 - x[i];
                let b = x[i + 1] - x[i] * x[i];
                acc += a * a + 100.0 * b * b;
            }
            acc
        }
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::TrustExact),
            tol: Some(1e-8),
            maxiter: Some(300),
            maxfev: Some(200_000),
            ..MinimizeOptions::default()
        };
        let x0: Vec<f64> = (0..10)
            .map(|i| if i % 2 == 0 { -1.5 } else { 1.7 })
            .collect();
        let result = minimize(rosenbrock_nd, &x0, options).expect("minimize");
        assert!(result.success, "should converge: {}", result.message);
        assert!(
            result.fun.unwrap() < 1e-6,
            "should reach the optimum, got fun={:?}",
            result.fun
        );
        for (i, &xi) in result.x.iter().enumerate() {
            assert!((xi - 1.0).abs() < 1e-3, "x[{i}]={xi} should be ~1");
        }
    }

    // ── TNC / SLSQP / trust_constr unit coverage (per frankenscipy-rkk2) ──

    /// Convex quadratic bowl: f(x, y) = (x - 2)^2 + (y + 3)^2. Minimum at (2, -3).
    fn convex_bowl(x: &[f64]) -> f64 {
        (x[0] - 2.0).powi(2) + (x[1] + 3.0).powi(2)
    }

    #[test]
    fn tnc_converges_on_convex_bowl() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Tnc),
            tol: Some(1e-7),
            maxiter: Some(200),
            maxfev: Some(5_000),
            ..MinimizeOptions::default()
        };
        let result = tnc(&convex_bowl, &[0.0, 0.0], options).expect("tnc run");
        assert!(result.success, "tnc should converge: {}", result.message);
        assert!(
            (result.x[0] - 2.0).abs() < 1e-3 && (result.x[1] + 3.0).abs() < 1e-3,
            "tnc x={:?}",
            result.x
        );
        assert!(
            result.fun.is_some_and(|v| v < 1e-5),
            "tnc f={:?} should be ~0",
            result.fun
        );
    }

    #[test]
    fn tnc_respects_box_bounds() {
        // (x-3)² + (y-3)² with x,y ∈ [0,2] → constrained optimum at [2,2] (f=2).
        // TNC previously ignored bounds and returned the unconstrained [3,3].
        static BOUNDS: [(Option<f64>, Option<f64>); 2] =
            [(Some(0.0), Some(2.0)), (Some(0.0), Some(2.0))];
        let f = |x: &[f64]| (x[0] - 3.0).powi(2) + (x[1] - 3.0).powi(2);
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Tnc),
            tol: Some(1e-8),
            maxiter: Some(500),
            bounds: Some(&BOUNDS),
            ..MinimizeOptions::default()
        };
        let result = tnc(&f, &[0.0, 0.0], options).expect("tnc run");
        assert!(
            result.success,
            "tnc bounded should converge: {}",
            result.message
        );
        assert!(
            (result.x[0] - 2.0).abs() < 1e-4 && (result.x[1] - 2.0).abs() < 1e-4,
            "tnc must stay within bounds at [2,2], got {:?}",
            result.x
        );
        // Feasibility: every component within the box.
        assert!(
            result.x.iter().all(|&v| (0.0..=2.0).contains(&v)),
            "tnc returned an out-of-bounds point: {:?}",
            result.x
        );
    }

    #[test]
    fn tnc_converges_on_rosenbrock() {
        // From (0, 0) at default options. Previously stalled at f ~ 0.95
        // because the quasi-Newton update never referenced B_k.
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Tnc),
            ..MinimizeOptions::default()
        };
        let result = tnc(&rosenbrock, &[0.0, 0.0], options).expect("tnc rosenbrock");
        assert!(
            result.fun.is_some_and(|v| v < 1e-3),
            "tnc Rosenbrock f={:?} should be < 1e-3",
            result.fun
        );
    }

    #[test]
    fn tnc_respects_maxfev_limit() -> Result<(), OptError> {
        // Pathologically tight budget: one function evaluation is not enough
        // to converge. TNC should hit the budget limit, not hang.
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Tnc),
            tol: Some(1e-15),
            maxiter: Some(10_000),
            maxfev: Some(3), // deliberately too small
            ..MinimizeOptions::default()
        };
        let result = tnc(&convex_bowl, &[10.0, 10.0], options);
        // Either returns success=false with budget message, or EvaluationBudgetExceeded.
        match result {
            Ok(r) => assert!(
                !r.success || r.nfev <= 100,
                "tnc with maxfev=3 cannot fully converge from (10,10); got x={:?} msg={}",
                r.x,
                r.message
            ),
            Err(OptError::EvaluationBudgetExceeded { .. }) => {}
            Err(e) => return Err(e),
        }
        Ok(())
    }

    #[test]
    fn slsqp_converges_on_convex_bowl() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Slsqp),
            tol: Some(1e-7),
            maxiter: Some(200),
            maxfev: Some(5_000),
            ..MinimizeOptions::default()
        };
        let result = slsqp(&convex_bowl, &[0.0, 0.0], options).expect("slsqp run");
        assert!(result.success, "slsqp should converge: {}", result.message);
        assert!(
            (result.x[0] - 2.0).abs() < 1e-3 && (result.x[1] + 3.0).abs() < 1e-3,
            "slsqp x={:?}",
            result.x
        );
    }

    #[test]
    fn slsqp_rejects_nonfinite_x0() {
        // NaN initial guess must fail-closed per the standard minimize
        // input-validation surface.
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Slsqp),
            ..MinimizeOptions::default()
        };
        let result = slsqp(&convex_bowl, &[f64::NAN, 0.0], options);
        assert!(result.is_err(), "NaN x0 should fail; got {result:?}");
    }

    #[test]
    fn trust_constr_converges_on_convex_bowl() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::TrustConstr),
            tol: Some(1e-7),
            maxiter: Some(500),
            maxfev: Some(20_000),
            ..MinimizeOptions::default()
        };
        let result = trust_constr(&convex_bowl, &[0.0, 0.0], options).expect("trust_constr");
        assert!(
            result.success,
            "trust_constr should converge: {}",
            result.message
        );
        assert!(
            (result.x[0] - 2.0).abs() < 1e-2 && (result.x[1] + 3.0).abs() < 1e-2,
            "trust_constr x={:?}",
            result.x
        );
    }

    #[test]
    fn trust_constr_converges_on_rosenbrock() {
        // From (0, 0). A Cauchy-only (steepest-descent) trust step stalled at
        // f ~ 0.048; the dogleg step with a BFGS Hessian model descends the
        // narrow valley.
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::TrustConstr),
            ..MinimizeOptions::default()
        };
        let result =
            trust_constr(&rosenbrock, &[0.0, 0.0], options).expect("trust_constr rosenbrock");
        assert!(
            result.fun.is_some_and(|v| v < 1e-3),
            "trust_constr Rosenbrock f={:?} should be < 1e-3",
            result.fun
        );
    }

    #[test]
    fn trust_constr_accepts_kkt_infinity_norm_stationarity() -> Result<(), OptError> {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::TrustConstr),
            tol: Some(1.0e-6),
            maxiter: Some(1),
            maxfev: Some(100),
            ..MinimizeOptions::default()
        };
        let x0 = [7.5e-7; 4];
        let result = trust_constr(
            &|x: &[f64]| 0.5 * x.iter().map(|value| value * value).sum::<f64>(),
            &x0,
            options,
        )?;

        assert!(
            result.success,
            "trust_constr should converge: {}",
            result.message
        );
        assert_eq!(result.status, ConvergenceStatus::Success);
        assert_eq!(result.nit, 0);
        assert!(
            result.message.contains("KKT optimality"),
            "message={}",
            result.message
        );
        let Some(jac) = result.jac else {
            return Err(OptError::InvalidArgument {
                detail: String::from("trust_constr should return final KKT gradient"),
            });
        };
        assert!(jac.iter().all(|value| value.abs() <= 1.0e-6), "jac={jac:?}");
        Ok(())
    }

    #[test]
    fn trust_constr_starts_at_optimum() {
        // Starting at the minimum should converge in 0-1 iterations.
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::TrustConstr),
            tol: Some(1e-10),
            ..MinimizeOptions::default()
        };
        let result = trust_constr(&convex_bowl, &[2.0, -3.0], options).expect("at_optimum");
        assert!(result.success);
        assert!(
            result.fun.is_some_and(|v| v < 1e-10),
            "at-optimum f={:?}",
            result.fun
        );
    }
}
