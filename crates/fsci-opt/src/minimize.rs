#![forbid(unsafe_code)]

use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::bfgs::{
    self, BfgsParams, CgParams, LineObjective, NewtonCgParams, NewtonCgStop, NewtonObjective,
};
use crate::lbfgs_inv_hess::LbfgsInvHessProduct;
use crate::lbfgsb::{self, LbfgsbObjective, LbfgsbStop};
use crate::trust_region::{self, Subproblem, TrustObjective, TrustParams};
use crate::types::{
    Bound, Bounds, Constraint, ConstraintType, ConvergenceStatus, GradientFunc, HessFunc, HessInv,
    HesspFunc, MinimizeCallback, MinimizeMethodOptions, MinimizeOptions, OptError, OptimizeMethod,
    OptimizeResult, OptimizeTraceEntry,
};
use fsci_runtime::{
    Fingerprinter, OptSolverAction, OptSolverEvidenceEntry, OptSolverPortfolio, RuntimeMode,
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
    /// Whether Hessian-vector products are available (`hessp`, or `hess` which gives them).
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
            hessian_product_available: options.hessp.is_some() || options.hess.is_some(),
        }
    }

    #[must_use]
    pub fn from_x0_and_options(x0: &[f64], options: MinimizeOptions) -> Self {
        Self {
            dimension: x0.len(),
            variable_scale_ratio: variable_scale_ratio(x0),
            has_box_bounds: options.bounds.is_some_and(bounds_have_finite_limit),
            has_general_constraints: !options.constraints.is_empty(),
            gradient_available: options.gradient_available,
            hessian_product_available: options.hessp.is_some() || options.hess.is_some(),
        }
    }
}

/// Select a minimize solver using the fsci-opt CASP policy.
///
/// State space: constraint class, gradient availability, Hessian-product
/// availability, dimension, and variable scale ratio. Evidence signals come
/// from the public problem description rather than trial execution, so the
/// selector is deterministic and side-effect free. The loss matrix prioritizes
/// stability first: general constraints route to SLSQP (as `scipy.optimize.minimize`
/// does), box constraints to projected `L-BFGS-B`, available curvature to Newton-CG or
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
            method: OptimizeMethod::Slsqp,
            reason: String::from(
                "general constraints route to SLSQP, as scipy.optimize.minimize does",
            ),
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
                method: OptimizeMethod::TrustNcg,
                reason: String::from(
                    "small ill-scaled problem with Hessian products uses the trust-ncg trust region",
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
    // SciPy warns and drops the constraints for a method that cannot use them; returning that
    // unconstrained optimum as a success is exactly the silent wrong answer this refuses.
    if !options.constraints.is_empty()
        && !matches!(
            selected_method,
            OptimizeMethod::Slsqp | OptimizeMethod::TrustConstr
        )
    {
        return Err(OptError::InvalidArgument {
            detail: format!("method {selected_method:?} cannot handle constraints; use SLSQP"),
        });
    }

    match selected_method {
        OptimizeMethod::Bfgs => bfgs(&fun, x0, options),
        OptimizeMethod::ConjugateGradient => cg_pr_plus(&fun, x0, options),
        OptimizeMethod::Powell => powell(&fun, x0, options),
        OptimizeMethod::NelderMead => nelder_mead(&fun, x0, options),
        OptimizeMethod::LBfgsB => lbfgsb(&fun, x0, options, options.bounds),
        OptimizeMethod::NewtonCg => newton_cg(&fun, x0, options),
        OptimizeMethod::TrustExact => trust_exact(&fun, x0, options),
        OptimizeMethod::TrustNcg => trust_ncg(&fun, x0, options),
        OptimizeMethod::Dogleg => dogleg(&fun, x0, options),
        OptimizeMethod::Tnc => tnc(&fun, x0, options),
        OptimizeMethod::Slsqp => slsqp(&fun, x0, options),
        OptimizeMethod::TrustConstr => trust_constr(&fun, x0, options),
    }
}

#[derive(Debug, Clone)]
pub struct OptPortfolioResult {
    pub chosen_action: OptSolverAction,
    pub posterior: [f64; 4],
    pub expected_losses: [f64; 5],
    pub result: OptimizeResult,
    pub fallback_active: bool,
}

/// Minimize an objective function using CASP Bayesian expected-loss portfolio selection.
///
/// Dispatches across BFGS (smooth convex), TrustRegionNewtonCG (narrow valley / high curvature),
/// DIRECT (multimodal global exploration), Nelder-Mead (noisy non-smooth), and L-BFGS-B (bounded).
/// When `options.bounds` constrains any variable, only methods that honour bounds are
/// candidates: L-BFGS-B, plus DIRECT when every bound is finite.
pub fn minimize_with_casp_portfolio<F>(
    fun: F,
    x0: &[f64],
    options: MinimizeOptions,
    portfolio: &mut OptSolverPortfolio,
    is_noisy: bool,
    is_multimodal: bool,
) -> Result<OptPortfolioResult, OptError>
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
    // No portfolio member honours general constraints; `minimize` routes them to SLSQP.
    if !options.constraints.is_empty() {
        return Err(OptError::InvalidArgument {
            detail: String::from(
                "the CASP portfolio has no constrained solver; use minimize, which routes constraints to SLSQP",
            ),
        });
    }

    // br-szq1n.9: bounds are a feasibility constraint, not a preference. L-BFGS-B
    // and Nelder-Mead (which clips every trial point, as SciPy's does, since
    // br-szq1n.7) honour arbitrary bounds, and DIRECT needs every side finite; BFGS
    // and Newton-CG ignore `options.bounds` and would return an infeasible x
    // reported as success. Restrict the candidates BEFORE the argmin.
    let active_bounds = options
        .bounds
        .filter(|b| b.iter().any(|&(lo, hi)| lo.is_some() || hi.is_some()));
    let feasible: Vec<OptSolverAction> = match active_bounds {
        None => OptSolverAction::ALL.to_vec(),
        Some(b) => {
            let mut v = vec![OptSolverAction::LBFGSB, OptSolverAction::NelderMead];
            if b.iter().all(|&(lo, hi)| lo.is_some() && hi.is_some()) {
                v.push(OptSolverAction::DIRECT);
            }
            v
        }
    };

    let cond_estimate = variable_scale_ratio(x0);
    let (action, posterior, expected_losses, chosen_loss) =
        portfolio.select_action_among(cond_estimate, is_noisy, is_multimodal, &feasible);

    let (result, fallback_active) = match action {
        OptSolverAction::BFGS => {
            let res = bfgs(&fun, x0, options)?;
            (res, false)
        }
        OptSolverAction::LBFGSB => {
            let res = lbfgsb(&fun, x0, options, options.bounds)?;
            (res, false)
        }
        OptSolverAction::NelderMead => {
            let res = nelder_mead(&fun, x0, options)?;
            (res, false)
        }
        OptSolverAction::TrustRegionNewtonCG => {
            let res = newton_cg(&fun, x0, options)?;
            (res, false)
        }
        OptSolverAction::DIRECT => {
            let (lb, ub): (Vec<f64>, Vec<f64>) = if let Some(bnds) = options.bounds {
                bnds.iter()
                    .map(|b| (b.0.unwrap_or(-10.0), b.1.unwrap_or(10.0)))
                    .unzip()
            } else {
                x0.iter().map(|&x| (x - 10.0, x + 10.0)).unzip()
            };
            let direct_bounds = Bounds::new(lb, ub)?;
            let direct_res = crate::direct::direct(
                &fun,
                &direct_bounds,
                crate::direct::DirectOptions::default(),
            )?;
            let res = OptimizeResult {
                x: direct_res.x,
                fun: Some(direct_res.fun),
                nit: direct_res.nit,
                nfev: direct_res.nfev,
                njev: 0,
                nhev: 0,
                success: direct_res.success,
                status: if direct_res.success {
                    ConvergenceStatus::Success
                } else {
                    ConvergenceStatus::MaxIterations
                },
                message: direct_res.message,
                jac: None,
                hess_inv: None,
                maxcv: None,
            };
            (res, false)
        }
    };

    // The Hardened retry uses Nelder-Mead, which receives the same `options` and so
    // honours the same bounds (br-szq1n.7).
    let (final_res, final_fallback) = if !result.success
        && action != OptSolverAction::DIRECT
        && portfolio.mode() == RuntimeMode::Hardened
    {
        match nelder_mead(&fun, x0, options) {
            Ok(nm) => (nm, true),
            Err(_) => (result, fallback_active),
        }
    } else {
        (result, fallback_active)
    };

    portfolio.record_evidence(OptSolverEvidenceEntry {
        component: "fsci-opt",
        dimension: x0.len(),
        condition_number_estimate: cond_estimate,
        chosen_action: action,
        posterior: posterior.to_vec(),
        expected_losses: expected_losses.to_vec(),
        chosen_expected_loss: chosen_loss,
        fallback_active: final_fallback,
        gradient_norm: None,
    });

    Ok(OptPortfolioResult {
        chosen_action: action,
        posterior,
        expected_losses,
        result: final_res,
        fallback_active: final_fallback,
    })
}

/// Minimize an objective function using CASP condition-aware portfolio selection.
///
/// Dispatches among BFGS (smooth), L-BFGS-B (bounded), Nelder-Mead (noisy),
/// TrustRegionNewtonCG (narrow valley), and DIRECT (global exploration)
/// via Bayesian expected-loss minimization, with automatic condition diagnostics.
pub fn minimize_with_casp<F>(
    fun: F,
    x0: &[f64],
    options: MinimizeOptions,
    portfolio: &mut OptSolverPortfolio,
) -> Result<OptPortfolioResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    minimize_with_casp_portfolio(fun, x0, options, portfolio, false, false)
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
    let result = minimize(fun, x0, options);
    match &result {
        // frankenscipy-3cu8u.2: a machine-matchable code; it used to be the error's `Debug`
        // rendering, message and all.
        Err(error) => crate::audit::record_fail_closed(
            ledger,
            &minimize_audit_fingerprint(x0, &options),
            &format!("minimize::{}", error.reason_code()),
            &format!("rejected: {error}"),
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
                &minimize_audit_fingerprint(x0, &options),
                &format!("minimize::{:?}", output.status),
                "rejected",
            );
        }
        _ => {}
    }
    result
}

/// The audit fingerprint of a `minimize` call (frankenscipy-3cu8u.1): `x0`, then every field of
/// `options` in declaration order — `Option<f64>`/`Option<usize>`/`Option<u64>` as a presence
/// flag then the value, enums as their `Debug` rendering, `bounds` as presence, count and each
/// `(lower, upper)` pair, `constraints` as count and each one's type and `jac` presence.
///
/// The objective and the function-valued options (`callback`, `gradient`, `hess`, `hessp`, a
/// constraint's `fun`/`jac`) are code, not data: a function pointer's address is neither stable
/// across processes nor unique, so only their presence is fed. It used to be a `Debug` string of
/// `method`, `mode`, `x0`, `maxiter` and `maxfev` only, so calls differing in `tol`, `bounds`,
/// `seed` or any other option shared a fingerprint, as did `x0`s differing only in a NaN payload.
fn minimize_audit_fingerprint(x0: &[f64], options: &MinimizeOptions<'_>) -> String {
    fn optional_f64(fingerprinter: &mut Fingerprinter, value: Option<f64>) {
        fingerprinter.bool(value.is_some());
        if let Some(value) = value {
            fingerprinter.f64(value);
        }
    }
    fn optional_u64(fingerprinter: &mut Fingerprinter, value: Option<u64>) {
        fingerprinter.bool(value.is_some());
        if let Some(value) = value {
            fingerprinter.u64(value);
        }
    }

    let mut fingerprinter = Fingerprinter::new("fsci_opt::minimize");
    fingerprinter.f64s(x0).str(&format!("{:?}", options.method));
    optional_f64(&mut fingerprinter, options.tol);
    optional_u64(&mut fingerprinter, options.maxiter.map(|v| v as u64));
    optional_u64(&mut fingerprinter, options.maxfev.map(|v| v as u64));
    optional_f64(&mut fingerprinter, options.gradient_eps);
    fingerprinter
        .bool(options.callback.is_some())
        .bool(options.gradient.is_some())
        .bool(options.hess.is_some())
        .bool(options.hessp.is_some())
        .bool(options.bounds.is_some());
    if let Some(bounds) = options.bounds {
        fingerprinter.usize(bounds.len());
        for &(lower, upper) in bounds {
            optional_f64(&mut fingerprinter, lower);
            optional_f64(&mut fingerprinter, upper);
        }
    }
    fingerprinter.usize(options.constraints.len());
    for constraint in options.constraints {
        fingerprinter
            .str(&format!("{:?}", constraint.kind))
            .bool(constraint.jac.is_some());
    }
    fingerprinter
        .bool(options.gradient_available)
        .bool(options.fixture_id.is_some());
    if let Some(fixture_id) = options.fixture_id {
        fingerprinter.str(fixture_id);
    }
    optional_u64(&mut fingerprinter, options.seed);
    fingerprinter.str(&format!("{:?}", options.mode));
    fingerprinter.finish()
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

/// SciPy's `_epsilon` (√ε), the default `eps` of BFGS, CG, Newton-CG and SLSQP, whose finite
/// differences are SciPy's forward scheme.
const SCIPY_SQRT_EPS: f64 = 1.490_116_119_384_765_6e-8;

/// The default `gradient_eps` of the methods that difference with fsci's central scheme
/// (`eps·(1 + |x|)`), SciPy's own default for TNC.
const CENTRAL_DIFF_EPS: f64 = 1.0e-8;

/// `_minimize_lbfgsb`'s default `eps`: the absolute forward-difference step.
const LBFGSB_EPS: f64 = 1.0e-8;

/// `_minimize_lbfgsb`'s default `ftol` (`factr = 1e7`).
const LBFGSB_FTOL: f64 = 2.220_446_049_250_313e-9;

/// `scipy.optimize.minimize(method='BFGS')`: SciPy's `_minimize_bfgs` (see [`crate::bfgs`]),
/// its MINPACK-2 `dcsrch` line search with the Wolfe-2 fallback, and its inverse-Hessian update.
///
/// SciPy's option mapping: `tol` is `gtol` (default 1e-5, on the ∞-norm of the gradient),
/// `maxiter` defaults to 200·n, and `gradient_eps` is `eps` (default √ε), the absolute
/// forward-difference step for the gradient when `options.gradient` is absent. SciPy has no
/// evaluation cap, so one applies only when `options.maxfev` is set. Evaluations are cached at
/// the last point as SciPy's `ScalarFunction` caches them, so `nfev` / `njev` count what SciPy
/// counts. The callback runs after each iteration, as SciPy's does. In Strict mode a
/// non-finite objective or gradient value is passed to the algorithm as SciPy's is (status 2 or
/// 3 follows); Hardened mode rejects it.
pub fn bfgs<F>(fun: &F, x0: &[f64], options: MinimizeOptions) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    validate_minimize_options(options)?;
    check_method_options(OptimizeMethod::Bfgs, options)?;
    let n = x0.len();
    // An explicit option wins over `tol`, which only fills `gtol` (SciPy's `setdefault`).
    let method_options = options.method_options;
    let params = BfgsParams {
        gtol: method_options.gtol.or(options.tol).unwrap_or(1.0e-5),
        norm: method_options.norm.unwrap_or(f64::INFINITY),
        maxiter: options.maxiter.unwrap_or(200 * n),
        xrtol: method_options.xrtol.unwrap_or(0.0),
        c1: method_options.c1.unwrap_or(1.0e-4),
        c2: method_options.c2.unwrap_or(0.9),
    };
    let mut adapter = ScalarFunction::new(fun, options, OptimizeMethod::Bfgs, x0);
    let outcome = bfgs::minimize_bfgs(&mut adapter, x0, &params);
    let result = match outcome {
        Ok(outcome) => {
            let (success, status, message) =
                warnflag_result(outcome.status, outcome.stopped_by_callback);
            OptimizeResult {
                fun: Some(outcome.fun),
                success,
                status,
                message,
                nfev: adapter.nfev,
                njev: adapter.njev,
                nhev: 0,
                nit: outcome.nit,
                jac: Some(outcome.jac),
                hess_inv: Some(HessInv::Dense(
                    outcome
                        .hess_inv
                        .chunks(n.max(1))
                        .map(<[f64]>::to_vec)
                        .collect(),
                )),
                maxcv: None,
                x: outcome.x,
            }
        }
        Err(err) => result_from_error(
            &adapter.iterate,
            adapter.nit,
            adapter.nfev,
            adapter.njev,
            err,
        ),
    };
    log_completion(OptimizeMethod::Bfgs, options, result.nit, &result);
    Ok(result)
}

/// A SciPy BFGS / CG warnflag (0 success, 1 maxiter, 2 precision loss, 3 NaN) as fsci's
/// `(success, status, message)`; a callback stop is fsci's `CallbackStop`.
fn warnflag_result(warnflag: u8, stopped_by_callback: bool) -> (bool, ConvergenceStatus, String) {
    if stopped_by_callback {
        return (
            false,
            ConvergenceStatus::CallbackStop,
            String::from("callback requested stop"),
        );
    }
    let status = match warnflag {
        0 => ConvergenceStatus::Success,
        1 => ConvergenceStatus::MaxIterations,
        2 => ConvergenceStatus::PrecisionLoss,
        _ => ConvergenceStatus::NanEncountered,
    };
    (
        warnflag == 0,
        status,
        String::from(bfgs::status_message(warnflag)),
    )
}

/// [`LineObjective`] with SciPy `ScalarFunction`'s semantics: the last point's `f` and gradient
/// are cached, the gradient is the caller's or SciPy's `approx_derivative(method='2-point',
/// abs_step=eps, bounds)` from the cached `f`, and `nfev` / `njev` count as SciPy counts.
struct ScalarFunction<'a, F> {
    fun: &'a F,
    options: MinimizeOptions<'a>,
    /// Whose iterations the trace log records.
    method: OptimizeMethod,
    maxfev: usize,
    /// The absolute forward-difference step: `gradient_eps`, else the method's SciPy default.
    eps: f64,
    /// The `bounds` `approx_derivative` keeps its steps inside, infinite where absent.
    fd_bounds: Option<(Vec<f64>, Vec<f64>)>,
    nfev: usize,
    njev: usize,
    cached_x: Option<Vec<f64>>,
    cached_f: Option<f64>,
    cached_g: Option<Vec<f64>>,
    /// The last accepted iterate (`x0` before the first), reported if an evaluation fails.
    iterate: Vec<f64>,
    nit: usize,
}

impl<'a, F> ScalarFunction<'a, F>
where
    F: Fn(&[f64]) -> f64,
{
    fn new(fun: &'a F, options: MinimizeOptions<'a>, method: OptimizeMethod, x0: &[f64]) -> Self {
        let default_eps = if method == OptimizeMethod::LBfgsB {
            LBFGSB_EPS
        } else {
            SCIPY_SQRT_EPS
        };
        Self {
            fun,
            options,
            method,
            maxfev: options.maxfev.unwrap_or(usize::MAX),
            eps: options.gradient_eps.unwrap_or(default_eps),
            fd_bounds: None,
            nfev: 0,
            njev: 0,
            cached_x: None,
            cached_f: None,
            cached_g: None,
            iterate: x0.to_vec(),
            nit: 0,
        }
    }

    /// Keep the finite-difference steps inside `[lower, upper]`, as SciPy's bounded
    /// `approx_derivative` does.
    fn with_fd_bounds(mut self, lower: Vec<f64>, upper: Vec<f64>) -> Self {
        self.fd_bounds = Some((lower, upper));
        self
    }

    fn move_to(&mut self, x: &[f64]) {
        if self.cached_x.as_deref() != Some(x) {
            self.cached_x = Some(x.to_vec());
            self.cached_f = None;
            self.cached_g = None;
        }
    }

    fn eval_raw(&mut self, x: &[f64]) -> Result<f64, OptError> {
        if self.nfev >= self.maxfev {
            return Err(OptError::EvaluationBudgetExceeded {
                detail: format!("max function evaluations exceeded ({})", self.maxfev),
            });
        }
        self.nfev += 1;
        let value = (self.fun)(x);
        if !value.is_finite() && self.options.mode == RuntimeMode::Hardened {
            return Err(OptError::NonFiniteInput {
                detail: String::from("hardened mode rejects non-finite objective values"),
            });
        }
        Ok(value)
    }
}

impl<F> LineObjective for ScalarFunction<'_, F>
where
    F: Fn(&[f64]) -> f64,
{
    type Error = OptError;

    fn fun(&mut self, x: &[f64]) -> Result<f64, OptError> {
        self.move_to(x);
        if let Some(f) = self.cached_f {
            return Ok(f);
        }
        let f = self.eval_raw(x)?;
        self.cached_f = Some(f);
        Ok(f)
    }

    fn grad(&mut self, x: &[f64]) -> Result<Vec<f64>, OptError> {
        self.move_to(x);
        if let Some(g) = &self.cached_g {
            return Ok(g.clone());
        }
        self.njev += 1;
        let n = x.len();
        let g = if let Some(gradient) = self.options.gradient {
            let g = gradient(x);
            if self.options.mode == RuntimeMode::Hardened {
                validate_gradient_output(g, n)?
            } else if g.len() != n {
                return Err(OptError::InvalidArgument {
                    detail: format!(
                        "gradient callback returned length {}, expected {n}",
                        g.len()
                    ),
                });
            } else {
                g
            }
        } else {
            let f0 = self.fun(x)?;
            let steps = fd_steps_2point(
                x,
                self.eps,
                self.fd_bounds
                    .as_ref()
                    .map(|(lower, upper)| (lower.as_slice(), upper.as_slice())),
            );
            let mut xp = x.to_vec();
            let mut g = vec![0.0; n];
            for i in 0..n {
                xp[i] = x[i] + steps[i];
                let dx = xp[i] - x[i];
                g[i] = (self.eval_raw(&xp)? - f0) / dx;
                xp[i] = x[i];
            }
            g
        };
        self.cached_g = Some(g.clone());
        Ok(g)
    }

    fn callback(&mut self, x: &[f64], fun: f64) -> bool {
        self.nit += 1;
        let grad_norm = match (&self.cached_x, &self.cached_g) {
            (Some(at), Some(g)) if at.as_slice() == x => l2_norm(g),
            _ => f64::NAN,
        };
        let step = l2_norm(&sub_vectors(x, &self.iterate));
        self.iterate.clear();
        self.iterate.extend_from_slice(x);
        log_iteration(
            self.method,
            self.options,
            self.nit,
            fun,
            grad_norm,
            step,
            self.nfev,
        );
        self.options.callback.is_some_and(|callback| !callback(x))
    }
}

impl<F> NewtonObjective for ScalarFunction<'_, F>
where
    F: Fn(&[f64]) -> f64,
{
    fn has_hess(&self) -> bool {
        self.options.hess.is_some()
    }

    fn hess(&mut self, x: &[f64]) -> Result<Vec<f64>, OptError> {
        let Some(hess) = self.options.hess else {
            return Err(OptError::InvalidArgument {
                detail: String::from("Newton-CG's dense curvature needs options.hess"),
            });
        };
        validate_hessian_output(hess(x), x.len())
    }

    fn has_hessp(&self) -> bool {
        self.options.hessp.is_some()
    }

    fn hessp(&mut self, x: &[f64], p: &[f64]) -> Result<Vec<f64>, OptError> {
        let Some(hessp) = self.options.hessp else {
            return Err(OptError::InvalidArgument {
                detail: String::from("Newton-CG's Hessian products need options.hessp"),
            });
        };
        validate_hessp_output(hessp(x, p), x.len())
    }
}

impl<F> LbfgsbObjective for ScalarFunction<'_, F>
where
    F: Fn(&[f64]) -> f64,
{
    type Error = OptError;

    fn fun_and_grad(&mut self, x: &[f64]) -> Result<(f64, Vec<f64>), OptError> {
        let f = LineObjective::fun(self, x)?;
        let g = LineObjective::grad(self, x)?;
        Ok((f, g))
    }

    fn nfev(&self) -> usize {
        self.nfev
    }

    fn callback(&mut self, x: &[f64], f: f64) -> bool {
        LineObjective::callback(self, x, f)
    }
}

/// SciPy's `approx_derivative(method='2-point', abs_step=eps, bounds=(lb, ub))` steps: `eps`, or
/// `√ε·sign(x)·max(1, |x|)` where `x + eps` rounds back to `x`, then — unless every bound is
/// infinite — reversed or shortened to stay inside the box (`_adjust_scheme_to_bounds`,
/// one-sided).
fn fd_steps_2point(x: &[f64], eps: f64, bounds: Option<(&[f64], &[f64])>) -> Vec<f64> {
    let bounds = bounds.filter(|(lb, ub)| {
        !(lb.iter().all(|v| *v == f64::NEG_INFINITY) && ub.iter().all(|v| *v == f64::INFINITY))
    });
    x.iter()
        .enumerate()
        .map(|(i, &xi)| {
            let mut h = eps;
            if (xi + h) - xi == 0.0 {
                let sign = if xi >= 0.0 { 1.0 } else { -1.0 };
                h = f64::EPSILON.sqrt() * sign * xi.abs().max(1.0);
            }
            if let Some((lb, ub)) = bounds {
                let lower = xi - lb[i];
                let upper = ub[i] - xi;
                let trial = xi + h;
                let violated = trial < lb[i] || trial > ub[i];
                let fitting = h.abs() <= lower.max(upper);
                if violated && fitting {
                    h = -h;
                } else if !fitting {
                    h = if upper >= lower { upper } else { -lower };
                }
            }
            h
        })
        .collect()
}

/// `scipy.optimize.minimize(method='CG')`: SciPy's `_minimize_cg` (see [`crate::bfgs`]),
/// Polak–Ribière+ on the same `dcsrch` / Wolfe-2 line search, which must also pass Gilbert &
/// Nocedal's sufficient-descent test.
///
/// SciPy's option mapping: `tol` is `gtol` (default 1e-5, on the ∞-norm of the gradient),
/// `maxiter` defaults to 200·n, the curvature constant is SciPy's `c2 = 0.4`, and
/// `gradient_eps` is `eps` (default √ε), the absolute forward-difference step. Evaluation
/// caching, counting, the callback and the Strict / Hardened split are as in [`bfgs`].
pub fn cg_pr_plus<F>(
    fun: &F,
    x0: &[f64],
    options: MinimizeOptions,
) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    validate_minimize_options(options)?;
    check_method_options(OptimizeMethod::ConjugateGradient, options)?;
    let method_options = options.method_options;
    let params = CgParams {
        gtol: method_options.gtol.or(options.tol).unwrap_or(1.0e-5),
        norm: method_options.norm.unwrap_or(f64::INFINITY),
        maxiter: options.maxiter.unwrap_or(200 * x0.len()),
        c1: method_options.c1.unwrap_or(1.0e-4),
        c2: method_options.c2.unwrap_or(0.4),
    };
    let mut adapter = ScalarFunction::new(fun, options, OptimizeMethod::ConjugateGradient, x0);
    let outcome = bfgs::minimize_cg(&mut adapter, x0, &params);
    let result = match outcome {
        Ok(outcome) => {
            let (success, status, message) =
                warnflag_result(outcome.status, outcome.stopped_by_callback);
            OptimizeResult {
                fun: Some(outcome.fun),
                success,
                status,
                message,
                nfev: adapter.nfev,
                njev: adapter.njev,
                nhev: 0,
                nit: outcome.nit,
                jac: Some(outcome.jac),
                hess_inv: None,
                maxcv: None,
                x: outcome.x,
            }
        }
        Err(err) => result_from_error(
            &adapter.iterate,
            adapter.nit,
            adapter.nfev,
            adapter.njev,
            err,
        ),
    };
    log_completion(
        OptimizeMethod::ConjugateGradient,
        options,
        result.nit,
        &result,
    );
    Ok(result)
}

pub fn powell<F>(fun: &F, x0: &[f64], options: MinimizeOptions) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    validate_minimize_options(options)?;
    check_method_options(OptimizeMethod::Powell, options)?;
    // Bounds as SciPy's Powell applies them: x0 clipped into the box, every line search limited
    // to the feasible step interval, the extrapolated point capped at the boundary. They used to
    // be ignored (frankenscipy-szq1n.7).
    validate_bounds_for_x0(x0, options.bounds)?;
    let bounds = options.bounds.filter(|b| bounds_have_finite_limit(b));

    let n = x0.len();
    // `_minimize_powell(xtol=1e-4, ftol=1e-4)`; `minimize(method="Powell", tol=t)` fills both
    // with t unless given as options (frankenscipy-6ycp2). The line searches get xtol·100.
    let method_options = options.method_options;
    let xtol = method_options.xtol.or(options.tol).unwrap_or(1.0e-4);
    let tol = method_options.ftol.or(options.tol).unwrap_or(1.0e-4);
    let line_tol = xtol * 100.0;
    let maxiter = options.maxiter.unwrap_or((150 * n).max(80));
    let maxfev = options.maxfev.unwrap_or((3000 * n).max(800));
    let mut objective = Objective::new(fun, options.mode, maxfev);

    let mut x = x0.to_vec();
    if let Some(b) = bounds {
        project_onto_bounds(&mut x, b);
    }
    let mut f = match objective.eval(&x) {
        Ok(value) => value,
        Err(err) => return Ok(result_from_error(x0, 0, 0, 0, err)),
    };
    // SciPy's `direc`: the starting direction set, one per row (default the identity).
    let mut directions = match method_options.direc {
        Some(direc) => {
            if direc.len() != n || direc.iter().any(|row| row.len() != n) {
                return Err(OptError::InvalidArgument {
                    detail: format!("direc must be {n} x {n}, one direction per row"),
                });
            }
            direc.to_vec()
        }
        None => identity_matrix(n),
    };
    // SciPy's `x1`: where the previous sweep ENDED, before that iteration's extrapolation line
    // search moved x again. The next extrapolation direction is `x − x1`. Measuring it from the
    // start of the sweep instead (after the extrapolation search) builds a different direction
    // set from the second iteration on; on bounded Rosenbrock that stalled at f = 0.2709 where
    // SciPy reaches 0.2500169 in the same 26 iterations and 1266 evaluations it now takes.
    let mut x1 = x.clone();

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

        let f_start = f;
        let mut largest_drop = 0.0;
        let mut largest_drop_idx = 0usize;

        for (dir_idx, direction) in directions.iter().enumerate() {
            let search =
                match powell_line_search(&mut objective, &x, f, direction, line_tol, bounds) {
                    Ok(value) => value,
                    Err(err) => {
                        return Ok(result_from_error(&x, iteration, objective.nfev, 0, err));
                    }
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

        // SciPy's `_minimize_powell` stopping test, RELATIVE in f: 2(fx − fval) ≤
        // ftol·(|fx| + |fval|) + 1e-20, with `tol` mapped to ftol (and to the line-search xtol)
        // as `minimize(method="Powell", tol=...)` maps it. This was `‖Δx‖ ≤ tol || |Δf| ≤ tol`,
        // absolute, which declared an objective scaled to tiny values converged after one sweep
        // (frankenscipy-cjv9z).
        if 2.0 * (f_start - f) <= tol * (f_start.abs() + f.abs()) + 1.0e-20 {
            let result = OptimizeResult {
                x: x.clone(),
                fun: Some(f),
                // status: SciPy's relative f-decrease test over a full direction sweep
                success: true,
                status: ConvergenceStatus::Success,
                message: String::from("optimization converged (relative f decrease <= tol)"),
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

        // Direction set update (standard Powell criterion): extrapolate along the move since
        // the previous sweep ended, x_ext = x + (x − x1); with bounds, SciPy's
        // x + min(lmax, 1)*(x − x1) so the extrapolation stays feasible.
        let move_vec = sub_vectors(&x, &x1);
        x1.clone_from(&x);
        let x_ext: Vec<f64> = match bounds {
            None => x
                .iter()
                .zip(move_vec.iter())
                .map(|(&xi, &mi)| xi + mi)
                .collect(),
            Some(b) => {
                let reach = feasible_step_interval(&x, &move_vec, b).1.min(1.0);
                let mut capped: Vec<f64> = x
                    .iter()
                    .zip(move_vec.iter())
                    .map(|(&xi, &mi)| xi + reach * mi)
                    .collect();
                project_onto_bounds(&mut capped, b);
                capped
            }
        };
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
                // SciPy's direction update: search along the sweep's total move, then store
                // the step actually taken (alpha·move, unnormalized, so the next searches keep
                // its scale) as the last direction, the old last one taking the slot of the
                // largest drop.
                let search =
                    match powell_line_search(&mut objective, &x, f, &move_vec, line_tol, bounds) {
                        Ok(v) => v,
                        Err(err) => {
                            return Ok(result_from_error(&x, iteration, objective.nfev, 0, err));
                        }
                    };
                x = search.x;
                f = search.f;

                let taken = scale_vector(&move_vec, search.alpha);
                if taken.iter().any(|&component| component != 0.0) {
                    let last = directions.len() - 1;
                    directions[largest_drop_idx] = directions[last].clone();
                    directions[last] = taken;
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

/// `scipy.optimize.minimize(method='Nelder-Mead')`: SciPy 1.17's `_minimize_neldermead`,
/// transcribed (frankenscipy-6ycp2).
///
/// Options, SciPy's names and defaults: `xatol` and `fatol` 1e-4 (`tol` fills both unless given
/// as options), `adaptive` (Gao & Han's dimension-dependent coefficients, default off),
/// `initial_simplex` (the n + 1 starting vertices, replacing the ones built around x0 by
/// scaling each coordinate by 1.05, or setting it to 0.00025 when it is 0). `maxiter` and
/// `maxfev` default to 200·n when neither is given, and one given alone leaves the other
/// unlimited. Evaluations past `maxfev` are refused, ending the run with SciPy's `maxfev`
/// status, which also wins over convergence reached on the last allowed evaluation. Bounds, as
/// SciPy applies them: x0 and every trial point are clipped, and a starting vertex past an upper
/// bound is first reflected into the interior (frankenscipy-szq1n.7). In Strict mode a
/// non-finite objective value is used as SciPy uses it (an objective returning `inf` outside a
/// region is a standard device); Hardened mode rejects it.
pub fn nelder_mead<F>(
    fun: &F,
    x0: &[f64],
    options: MinimizeOptions,
) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
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
    check_method_options(OptimizeMethod::NelderMead, options)?;
    if x0.is_empty() {
        return Err(OptError::InvalidArgument {
            detail: String::from("x0 must have at least one element"),
        });
    }
    validate_bounds_for_x0(x0, options.bounds)?;
    let bounds = options.bounds.filter(|b| bounds_have_finite_limit(b));
    let clip = |point: &mut [f64]| {
        if let Some(b) = bounds {
            project_onto_bounds(point, b);
        }
    };

    let method_options = options.method_options;
    let xatol = method_options.xatol.or(options.tol).unwrap_or(1.0e-4);
    let fatol = method_options.fatol.or(options.tol).unwrap_or(1.0e-4);

    let mut start = x0.to_vec();
    clip(&mut start);
    let n = start.len();
    let (rho, chi, psi, sigma) = if method_options.adaptive == Some(true) {
        let dim = n as f64;
        (
            1.0,
            1.0 + 2.0 / dim,
            0.75 - 1.0 / (2.0 * dim),
            1.0 - 1.0 / dim,
        )
    } else {
        (1.0, 2.0, 0.5, 0.5)
    };

    let mut sim: Vec<Vec<f64>> = match method_options.initial_simplex {
        None => {
            let mut sim = Vec::with_capacity(n + 1);
            sim.push(start.clone());
            for k in 0..n {
                let mut y = start.clone();
                if y[k] != 0.0 {
                    y[k] *= 1.0 + 0.05;
                } else {
                    y[k] = 0.00025;
                }
                sim.push(y);
            }
            sim
        }
        Some(given) => {
            let width = given.first().map_or(0, Vec::len);
            if given.len() != width + 1 || given.iter().any(|row| row.len() != width) {
                return Err(OptError::InvalidArgument {
                    detail: String::from("`initial_simplex` should be an array of shape (N+1,N)"),
                });
            }
            if width != n {
                return Err(OptError::InvalidArgument {
                    detail: String::from("Size of `initial_simplex` is not consistent with `x0`"),
                });
            }
            given.to_vec()
        }
    };
    let (maxiter, maxfun) = match (options.maxiter, options.maxfev) {
        (None, None) => (200 * n, 200 * n),
        (None, Some(maxfun)) => (usize::MAX, maxfun),
        (Some(maxiter), None) => (maxiter, usize::MAX),
        (Some(maxiter), Some(maxfun)) => (maxiter, maxfun),
    };
    if let Some(b) = bounds {
        for vertex in &mut sim {
            for (value, &(_, hi)) in vertex.iter_mut().zip(b) {
                if let Some(hi) = hi
                    && *value > hi
                {
                    *value = 2.0 * hi - *value;
                }
            }
            project_onto_bounds(vertex, b);
        }
    }

    // SciPy's `_wrap_scalar_function_maxfun_validation`: an evaluation past `maxfun` is refused
    // (`None` here, `_MaxFuncCallError` there) and ends the current step.
    let mut nfev = 0_usize;
    let eval = |x: &[f64], nfev: &mut usize| -> Result<Option<f64>, OptError> {
        if *nfev >= maxfun {
            return Ok(None);
        }
        *nfev += 1;
        let value = fun(x);
        if !value.is_finite() && options.mode == RuntimeMode::Hardened {
            return Err(OptError::NonFiniteInput {
                detail: String::from("hardened mode rejects non-finite objective values"),
            });
        }
        Ok(Some(value))
    };

    let mut fsim = vec![f64::INFINITY; n + 1];
    for (vertex, value) in sim.iter().zip(fsim.iter_mut()) {
        match eval(vertex, &mut nfev)? {
            Some(f) => *value = f,
            None => break,
        }
    }
    nelder_mead_sort(&mut sim, &mut fsim);

    let mut iterations = 1_usize;
    let mut stopped_by_callback = false;
    while nfev < maxfun && iterations < maxiter {
        let x_spread = sim[1..]
            .iter()
            .flat_map(|vertex| vertex.iter().zip(&sim[0]).map(|(a, b)| (a - b).abs()))
            .fold(0.0_f64, nan_max);
        let f_spread = fsim[1..]
            .iter()
            .map(|f| (fsim[0] - f).abs())
            .fold(0.0_f64, nan_max);
        if x_spread <= xatol && f_spread <= fatol {
            break;
        }

        'step: {
            let mut xbar = vec![0.0; n];
            for vertex in &sim[..n] {
                for (acc, v) in xbar.iter_mut().zip(vertex) {
                    *acc += v;
                }
            }
            for value in &mut xbar {
                *value /= n as f64;
            }
            let along = |a: f64, b: f64| -> Vec<f64> {
                xbar.iter()
                    .zip(&sim[n])
                    .map(|(xb, worst)| a * xb - b * worst)
                    .collect()
            };

            let mut xr = along(1.0 + rho, rho);
            clip(&mut xr);
            let Some(fxr) = eval(&xr, &mut nfev)? else {
                break 'step;
            };
            let mut doshrink = false;
            if fxr < fsim[0] {
                let mut xe = along(1.0 + rho * chi, rho * chi);
                clip(&mut xe);
                let Some(fxe) = eval(&xe, &mut nfev)? else {
                    break 'step;
                };
                if fxe < fxr {
                    sim[n] = xe;
                    fsim[n] = fxe;
                } else {
                    sim[n] = xr;
                    fsim[n] = fxr;
                }
            } else if fxr < fsim[n - 1] {
                sim[n] = xr;
                fsim[n] = fxr;
            } else if fxr < fsim[n] {
                let mut xc = along(1.0 + psi * rho, psi * rho);
                clip(&mut xc);
                let Some(fxc) = eval(&xc, &mut nfev)? else {
                    break 'step;
                };
                if fxc <= fxr {
                    sim[n] = xc;
                    fsim[n] = fxc;
                } else {
                    doshrink = true;
                }
            } else {
                // `(1 - psi) * xbar + psi * sim[-1]`
                let mut xcc: Vec<f64> = xbar
                    .iter()
                    .zip(&sim[n])
                    .map(|(xb, worst)| (1.0 - psi) * xb + psi * worst)
                    .collect();
                clip(&mut xcc);
                let Some(fxcc) = eval(&xcc, &mut nfev)? else {
                    break 'step;
                };
                if fxcc < fsim[n] {
                    sim[n] = xcc;
                    fsim[n] = fxcc;
                } else {
                    doshrink = true;
                }
            }
            if doshrink {
                for j in 1..=n {
                    let shrunk: Vec<f64> = sim[0]
                        .iter()
                        .zip(&sim[j])
                        .map(|(best, v)| best + sigma * (v - best))
                        .collect();
                    sim[j] = shrunk;
                    clip(&mut sim[j]);
                    let Some(f) = eval(&sim[j], &mut nfev)? else {
                        break 'step;
                    };
                    fsim[j] = f;
                }
            }
            iterations += 1;
        }

        nelder_mead_sort(&mut sim, &mut fsim);
        log_iteration(
            OptimizeMethod::NelderMead,
            options,
            iterations,
            fsim[0],
            0.0,
            0.0,
            nfev,
        );
        if let Some(callback) = options.callback
            && !callback(&sim[0])
        {
            stopped_by_callback = true;
            break;
        }
    }

    let fval = fsim.iter().copied().fold(f64::INFINITY, nan_min);
    let (success, status, message) = if stopped_by_callback {
        (
            false,
            ConvergenceStatus::CallbackStop,
            "`callback` raised `StopIteration`.",
        )
    } else if nfev >= maxfun {
        (
            false,
            ConvergenceStatus::MaxEvaluations,
            "Maximum number of function evaluations has been exceeded.",
        )
    } else if iterations >= maxiter {
        (
            false,
            ConvergenceStatus::MaxIterations,
            "Maximum number of iterations has been exceeded.",
        )
    } else {
        (
            true,
            ConvergenceStatus::Success,
            "Optimization terminated successfully.",
        )
    };
    let result = OptimizeResult {
        x: sim[0].clone(),
        fun: Some(fval),
        success,
        status,
        message: String::from(message),
        nfev,
        njev: 0,
        nhev: 0,
        nit: iterations,
        jac: None,
        hess_inv: None,
        maxcv: None,
    };
    log_completion(OptimizeMethod::NelderMead, options, iterations, &result);
    Ok(result)
}

/// numpy's `argsort` of the simplex values applied to both arrays; NaN sorts last, as numpy's
/// does. A stable sort, as numpy's is for these sizes (insertion sort below 16 elements).
fn nelder_mead_sort(sim: &mut Vec<Vec<f64>>, fsim: &mut Vec<f64>) {
    let mut order: Vec<usize> = (0..fsim.len()).collect();
    order.sort_by(|&a, &b| {
        fsim[a]
            .partial_cmp(&fsim[b])
            .unwrap_or_else(|| fsim[a].is_nan().cmp(&fsim[b].is_nan()))
    });
    *sim = order.iter().map(|&i| sim[i].clone()).collect();
    *fsim = order.iter().map(|&i| fsim[i]).collect();
}

/// `np.max`'s propagation of NaN, as a fold.
fn nan_max(acc: f64, value: f64) -> f64 {
    if acc.is_nan() || value.is_nan() {
        f64::NAN
    } else {
        acc.max(value)
    }
}

/// `np.min`'s propagation of NaN, as a fold.
fn nan_min(acc: f64, value: f64) -> f64 {
    if acc.is_nan() || value.is_nan() {
        f64::NAN
    } else {
        acc.min(value)
    }
}

/// `scipy.optimize.minimize(method='L-BFGS-B')`: SciPy's `_minimize_lbfgsb` driving L-BFGS-B 3.0
/// (see [`crate::lbfgsb`]) — the generalized Cauchy point, the direct primal subspace
/// minimization and the MINPACK-2 line search — under `bounds`.
///
/// SciPy's option mapping: `tol` sets both `ftol` (default 2.2e-9, the relative-reduction test)
/// and `gtol` (default 1e-5, the projected-gradient test); `maxiter` and `maxfev` (`maxfun`,
/// checked between iterations as SciPy checks it, so a run can end a few evaluations past it)
/// default to 15000; `maxcor` is 10 and `maxls` 20. `gradient_eps` is `eps` (default 1e-8), the
/// absolute forward-difference step when `options.gradient` is absent, stepped backwards or
/// shortened at a bound as `approx_derivative(..., '2-point', abs_step=eps, bounds)` does. `x0`
/// is clipped into the bounds. The message is SciPy's task text. `hess_inv` is SciPy's lazy
/// `LbfgsInvHessProduct` ([`HessInv::Lbfgs`]) over the stored corrections, read from the
/// workspace as `_minimize_lbfgsb` reads them; no `n × n` matrix is formed unless `todense` is
/// asked for (frankenscipy-6ycp2). In Strict mode a non-finite objective value is passed to
/// the algorithm as SciPy's is; Hardened mode rejects it.
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
    check_method_options(OptimizeMethod::LBfgsB, options)?;
    validate_bounds_for_x0(x0, bounds)?;
    let n = x0.len();
    let (lower, upper): (Vec<f64>, Vec<f64>) = match bounds {
        Some(bounds) => bounds
            .iter()
            .map(|&(lo, hi)| (lo.unwrap_or(f64::NEG_INFINITY), hi.unwrap_or(f64::INFINITY)))
            .unzip(),
        None => (vec![f64::NEG_INFINITY; n], vec![f64::INFINITY; n]),
    };
    // `np.clip(x0, lb, ub)`, which keeps a NaN.
    let x0: Vec<f64> = x0
        .iter()
        .zip(lower.iter().zip(&upper))
        .map(|(&x, (&lo, &hi))| {
            if x < lo {
                lo
            } else if x > hi {
                hi
            } else {
                x
            }
        })
        .collect();
    let (l, u, nbd) = lbfgsb::encode_bounds(&lower, &upper);
    // An explicit option wins over `tol`, which only fills `ftol` and `gtol` (SciPy's
    // `setdefault`); SciPy's `factr` is `ftol / eps`.
    let method_options = options.method_options;
    let params = lbfgsb::LbfgsbParams {
        m: method_options.maxcor.unwrap_or(10),
        factr: method_options.ftol.or(options.tol).unwrap_or(LBFGSB_FTOL) / f64::EPSILON,
        pgtol: method_options.gtol.or(options.tol).unwrap_or(1.0e-5),
        maxfun: options.maxfev.unwrap_or(15_000),
        maxiter: options.maxiter.unwrap_or(15_000),
        maxls: method_options.maxls.unwrap_or(20),
    };
    // SciPy checks `maxfun` between iterations; the adapter must not cut an evaluation off.
    let adapter_options = MinimizeOptions {
        maxfev: None,
        ..options
    };
    let mut adapter = ScalarFunction::new(fun, adapter_options, OptimizeMethod::LBfgsB, &x0)
        .with_fd_bounds(lower, upper);
    let outcome = lbfgsb::minimize_lbfgsb(&mut adapter, &x0, &l, &u, &nbd, &params);
    let result = match outcome {
        Ok(outcome) => {
            let status = match outcome.stop {
                LbfgsbStop::ProjectedGradient | LbfgsbStop::RelativeReduction => {
                    ConvergenceStatus::Success
                }
                LbfgsbStop::MaxIter => ConvergenceStatus::MaxIterations,
                LbfgsbStop::MaxFun => ConvergenceStatus::MaxEvaluations,
                LbfgsbStop::Callback => ConvergenceStatus::CallbackStop,
                LbfgsbStop::Abnormal => ConvergenceStatus::PrecisionLoss,
            };
            // Stored pairs passed L-BFGS-B's own curvature test, so construction does not
            // fail; if it ever did, no operator is better than a wrong one.
            let (sk, yk) = outcome.corrections;
            let hess_inv = LbfgsInvHessProduct::with_dimension(sk, yk, n)
                .ok()
                .map(HessInv::Lbfgs);
            OptimizeResult {
                x: outcome.x,
                fun: Some(outcome.fun),
                // status: SciPy's warnflag 0 (projected gradient or relative reduction test)
                success: outcome.warnflag == 0,
                status,
                message: String::from(outcome.stop.message()),
                nfev: adapter.nfev,
                njev: adapter.njev,
                nhev: 0,
                nit: outcome.nit,
                jac: Some(outcome.jac),
                hess_inv,
                maxcv: None,
            }
        }
        Err(err) => result_from_error(
            &adapter.iterate,
            adapter.nit,
            adapter.nfev,
            adapter.njev,
            err,
        ),
    };
    log_completion(OptimizeMethod::LBfgsB, options, result.nit, &result);
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

/// `scipy.optimize.minimize(method='Newton-CG')`: SciPy's `_minimize_newtoncg` (see
/// [`crate::bfgs`]): truncated CG on the Newton system (at most 20·n inner steps, the forcing
/// tolerance min(0.5, √‖g‖₁)·‖g‖₁, SciPy's negative-curvature exits) and the same `dcsrch` /
/// Wolfe-2 line search as BFGS.
///
/// Curvature comes, as in SciPy, from `options.hess` (once per outer iteration), else
/// `options.hessp`, else forward differences of the gradient with step `gradient_eps` (default
/// √ε); `nhev` counts the first two as SciPy's `hcalls` does. SciPy's option mapping: `tol` is
/// `xtol` (default 1e-5; the stop is ‖update‖₁ ≤ n·xtol) and `maxiter` defaults to 200·n. SciPy
/// requires `jac`; without `options.gradient` fsci differences the objective forward, as its
/// trust-region methods do. Evaluation caching and counting, the callback and the Strict /
/// Hardened split are as in [`bfgs`]. Like SciPy's, the returned `jac` is the gradient at the
/// start of the last outer iteration.
pub fn newton_cg<F>(
    fun: &F,
    x0: &[f64],
    options: MinimizeOptions,
) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    validate_minimize_options(options)?;
    check_method_options(OptimizeMethod::NewtonCg, options)?;
    let params = NewtonCgParams {
        xtol: options.tol.unwrap_or(1.0e-5),
        maxiter: options.maxiter.unwrap_or(200 * x0.len()),
        eps: options.gradient_eps.unwrap_or(SCIPY_SQRT_EPS),
        c1: 1.0e-4,
        c2: 0.9,
    };
    let mut adapter = ScalarFunction::new(fun, options, OptimizeMethod::NewtonCg, x0);
    let outcome = bfgs::minimize_newton_cg(&mut adapter, x0, &params);
    let result = match outcome {
        Ok(outcome) => {
            let (success, status, message) = match outcome.stop {
                NewtonCgStop::Success => (
                    true,
                    ConvergenceStatus::Success,
                    "Optimization terminated successfully.",
                ),
                NewtonCgStop::MaxIterations => (
                    false,
                    ConvergenceStatus::MaxIterations,
                    "Warning: Maximum number of iterations has been exceeded.",
                ),
                NewtonCgStop::PrecisionLoss => (
                    false,
                    ConvergenceStatus::PrecisionLoss,
                    "Warning: Desired error not necessarily achieved due to precision loss.",
                ),
                NewtonCgStop::HessianNotPositiveDefinite => (
                    false,
                    ConvergenceStatus::LinAlgError,
                    "Warning: CG iterations didn't converge. The Hessian is not positive definite.",
                ),
                NewtonCgStop::Nan => (
                    false,
                    ConvergenceStatus::NanEncountered,
                    "NaN result encountered.",
                ),
                NewtonCgStop::Callback => (
                    false,
                    ConvergenceStatus::CallbackStop,
                    "callback requested stop",
                ),
            };
            OptimizeResult {
                fun: Some(outcome.fun),
                success,
                status,
                message: String::from(message),
                nfev: adapter.nfev,
                njev: adapter.njev,
                nhev: outcome.nhev,
                nit: outcome.nit,
                jac: outcome.jac,
                hess_inv: None,
                maxcv: None,
                x: outcome.x,
            }
        }
        Err(err) => result_from_error(
            &adapter.iterate,
            adapter.nit,
            adapter.nfev,
            adapter.njev,
            err,
        ),
    };
    log_completion(OptimizeMethod::NewtonCg, options, result.nit, &result);
    Ok(result)
}

/// `scipy.optimize.minimize(f, x0, method='trust-exact')`.
///
/// With `options.hess` this is SciPy's algorithm: the `_minimize_trust_region` driver with the
/// nearly-exact `IterativeSubproblem` (Moré–Sorensen λ iteration on Cholesky factors), taking
/// SciPy's iteration path. SciPy requires `hess`; without it fsci runs its own trust region on
/// a BFGS model of the Hessian (not SciPy's algorithm). Without `options.gradient` the
/// gradient is forward-differenced (SciPy requires `jac`).
pub fn trust_exact<F>(
    fun: &F,
    x0: &[f64],
    options: MinimizeOptions,
) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    check_method_options(OptimizeMethod::TrustExact, options)?;
    if options.hess.is_some() {
        return trust_region_minimize(
            fun,
            x0,
            options,
            OptimizeMethod::TrustExact,
            Subproblem::Exact { maxiter: 25 },
        );
    }
    validate_minimize_options(options)?;

    let n = x0.len();
    let tol = requested_tolerance(options.tol);
    let maxiter = options.maxiter.unwrap_or((200 * n).max(100));
    let maxfev = options.maxfev.unwrap_or((5000 * n).max(1_000));
    let eps = options.gradient_eps.unwrap_or(CENTRAL_DIFF_EPS);
    let mut objective = Objective::new(fun, options.mode, maxfev);

    let mut x = x0.to_vec();
    let mut f = match objective.eval(&x) {
        Ok(v) => v,
        Err(err) => return Ok(result_from_error(x0, 0, 0, 0, err)),
    };
    let mut njev = 0usize;
    let mut nhev = 0usize;
    let mut grad = match evaluate_minimize_gradient(&mut objective, options.gradient, &x, eps) {
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
                // status: ‖∇f‖₂ ≤ tol
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
            grad = match evaluate_minimize_gradient(&mut objective, options.gradient, &x, eps) {
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

/// `scipy.optimize.minimize(f, x0, method='trust-ncg')`: SciPy's trust-region driver with the
/// CG-Steihaug subproblem, taking SciPy's iteration path. Needs `options.hess` or
/// `options.hessp` (products use `hessp` when both are given, as in SciPy). Without
/// `options.gradient` the gradient is forward-differenced (SciPy requires `jac`).
pub fn trust_ncg<F>(
    fun: &F,
    x0: &[f64],
    options: MinimizeOptions,
) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    check_method_options(OptimizeMethod::TrustNcg, options)?;
    if options.hess.is_none() && options.hessp.is_none() {
        return Err(OptError::InvalidArgument {
            detail: String::from(
                "Either the Hessian or the Hessian-vector product is currently required for \
                 trust-region methods",
            ),
        });
    }
    trust_region_minimize(
        fun,
        x0,
        options,
        OptimizeMethod::TrustNcg,
        Subproblem::CgSteihaug,
    )
}

/// `scipy.optimize.minimize(f, x0, method='dogleg')`: SciPy's trust-region driver with the
/// dogleg subproblem, taking SciPy's iteration path. Needs `options.hess`; a Hessian that is
/// not positive definite at an iterate ends the solve with [`ConvergenceStatus::LinAlgError`]
/// (SciPy status 3). Without `options.gradient` the gradient is forward-differenced (SciPy
/// requires `jac`).
pub fn dogleg<F>(fun: &F, x0: &[f64], options: MinimizeOptions) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    check_method_options(OptimizeMethod::Dogleg, options)?;
    if options.hess.is_none() {
        return Err(OptError::InvalidArgument {
            detail: String::from("Hessian is required for dogleg minimization"),
        });
    }
    trust_region_minimize(fun, x0, options, OptimizeMethod::Dogleg, Subproblem::Dogleg)
}

/// [`TrustObjective`] over fsci's evaluation machinery: `f` through [`Objective`] (evaluation
/// budget and mode checks), the caller's gradient or forward differences, and their
/// `hess` / `hessp`.
struct TrustRegionAdapter<'o, 'f, F>
where
    F: Fn(&[f64]) -> f64,
{
    objective: &'o mut Objective<'f, F>,
    gradient: Option<GradientFunc>,
    hess: Option<HessFunc>,
    hessp: Option<HesspFunc>,
    gradient_eps: f64,
    callback: Option<MinimizeCallback>,
    njev: usize,
    nhev: usize,
    /// The last point `f` was evaluated at, reported if an evaluation fails.
    last_x: Vec<f64>,
}

impl<F> TrustObjective for TrustRegionAdapter<'_, '_, F>
where
    F: Fn(&[f64]) -> f64,
{
    type Error = OptError;

    fn fun(&mut self, x: &[f64]) -> Result<f64, OptError> {
        self.last_x.clear();
        self.last_x.extend_from_slice(x);
        self.objective.eval(x)
    }

    fn grad(&mut self, x: &[f64]) -> Result<Vec<f64>, OptError> {
        self.njev += 1;
        evaluate_minimize_gradient(self.objective, self.gradient, x, self.gradient_eps)
    }

    fn has_hess(&self) -> bool {
        self.hess.is_some()
    }

    fn hess(&mut self, x: &[f64]) -> Result<Vec<f64>, OptError> {
        let Some(hess) = self.hess else {
            return Err(OptError::InvalidArgument {
                detail: String::from("this trust-region method needs options.hess"),
            });
        };
        self.nhev += 1;
        validate_hessian_output(hess(x), x.len())
    }

    fn has_hessp(&self) -> bool {
        self.hessp.is_some()
    }

    fn hessp(&mut self, x: &[f64], p: &[f64]) -> Result<Vec<f64>, OptError> {
        let Some(hessp) = self.hessp else {
            return Err(OptError::InvalidArgument {
                detail: String::from("this trust-region method needs options.hessp"),
            });
        };
        self.nhev += 1;
        validate_hessp_output(hessp(x, p), x.len())
    }

    fn callback(&mut self, x: &[f64], _fun: f64) -> bool {
        self.callback.is_some_and(|callback| !callback(x))
    }
}

/// SciPy `_minimize_trust_region` with its default options (initial radius 1, maximum radius
/// 1000, η = 0.15, `gtol` = `tol` or 1e-4, `maxiter` = 200·n). SciPy has no evaluation cap, so
/// one applies only when `options.maxfev` is set.
fn trust_region_minimize<F>(
    fun: &F,
    x0: &[f64],
    options: MinimizeOptions,
    method: OptimizeMethod,
    subproblem: Subproblem,
) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    validate_minimize_options(options)?;
    let params = TrustParams {
        initial_trust_radius: 1.0,
        max_trust_radius: 1000.0,
        eta: 0.15,
        gtol: options.tol.unwrap_or(1.0e-4),
        maxiter: options.maxiter.unwrap_or(200 * x0.len()),
    };
    let mut objective = Objective::new(fun, options.mode, options.maxfev.unwrap_or(usize::MAX));
    let mut adapter = TrustRegionAdapter {
        objective: &mut objective,
        gradient: options.gradient,
        hess: options.hess,
        // SciPy hands `hessp` to trust-ncg only; dogleg and trust-exact use the Hessian.
        hessp: if subproblem == Subproblem::CgSteihaug {
            options.hessp
        } else {
            None
        },
        gradient_eps: options.gradient_eps.unwrap_or(CENTRAL_DIFF_EPS),
        callback: options.callback,
        njev: 0,
        nhev: 0,
        last_x: x0.to_vec(),
    };
    let outcome = trust_region::minimize(&mut adapter, x0, subproblem, params);
    let TrustRegionAdapter {
        njev,
        nhev,
        hess,
        last_x,
        ..
    } = adapter;
    // SciPy's `ScalarFunction` evaluates a stand-in Hessian once when only `hessp` is given
    // and counts it; the count is kept so `nhev` reads as SciPy's.
    let nhev = nhev + usize::from(hess.is_none());
    let outcome = match outcome {
        Ok(outcome) => outcome,
        Err(err) => {
            let mut result = result_from_error(&last_x, 0, objective.nfev, njev, err);
            result.nhev = nhev;
            log_completion(method, options, 0, &result);
            return Ok(result);
        }
    };
    let (success, status, message) = if outcome.stopped_by_callback {
        (
            false,
            ConvergenceStatus::CallbackStop,
            String::from("callback requested stop"),
        )
    } else {
        let status = match outcome.status {
            0 => ConvergenceStatus::Success,
            1 => ConvergenceStatus::MaxIterations,
            2 => ConvergenceStatus::PrecisionLoss,
            _ => ConvergenceStatus::LinAlgError,
        };
        (
            outcome.status == 0,
            status,
            String::from(trust_region::status_message(outcome.status)),
        )
    };
    let result = OptimizeResult {
        x: outcome.x,
        fun: Some(outcome.fun),
        success,
        status,
        message,
        nfev: objective.nfev,
        njev,
        nhev,
        nit: outcome.nit,
        jac: Some(outcome.jac),
        hess_inv: None,
        maxcv: None,
    };
    log_completion(method, options, result.nit, &result);
    Ok(result)
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
/// CONTRACT: BYTE-IDENTICAL either way. The switch changes only the LAYOUT of the
/// augmented matrix -- one flat `Vec<f64>` against `n` separately-allocated rows -- and
/// the elimination performs the same operations on the same values in the same order in
/// both. Nothing is reassociated, so there is no rounding difference to bound; the win is
/// allocation count and locality, not arithmetic.
pub static TRUST_EXACT_FLAT_AUGMENTED_DISABLE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// When `true`, trust-exact retains the pivoted Gauss-Jordan solve for its BFGS Hessian
/// systems. The default first uses an SPD Cholesky factor and triangular solves, then falls
/// back to Gauss-Jordan if the factorization detects a non-positive or non-finite pivot.
/// `#[doc(hidden)]` — same-binary A/B knob.
#[doc(hidden)]
/// CONTRACT: NOT byte-identical, and not merely reordered -- the two arms run DIFFERENT
/// ALGORITHMS. The fast arm factors the trust-region subproblem's matrix by Cholesky; the
/// slow arm solves it by pivoted elimination. Both are backward-stable for a symmetric
/// positive-definite system, so they agree to roughly the conditioning of that system
/// times machine epsilon -- expect ~1e-14 relative on the well-conditioned subproblems
/// this path sees, not bit equality.
///
/// There is one region where they ARE identical, and it is worth naming because it makes
/// the toggle look exact under casual testing: when the factorization detects a
/// non-positive or non-finite pivot the fast arm ABANDONS Cholesky and calls the same
/// pivoted solve the slow arm uses. On any input that trips that check both arms execute
/// identical code and return identical bits. A test that only exercised
/// nearly-singular matrices would therefore see bit equality and conclude the wrong
/// contract.
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

/// Flatten the caller's `hess(x)` to row-major `n × n`, refusing a wrong shape or a non-finite
/// entry as [`validate_gradient_output`] does for gradients.
fn validate_hessian_output(rows: Vec<Vec<f64>>, n: usize) -> Result<Vec<f64>, OptError> {
    if rows.len() != n || rows.iter().any(|row| row.len() != n) {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "hess callback returned {} rows of lengths {:?}, expected {n} x {n}",
                rows.len(),
                rows.iter().map(Vec::len).collect::<Vec<_>>()
            ),
        });
    }
    let matrix: Vec<f64> = rows.into_iter().flatten().collect();
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: String::from("hess callback returned NaN or Inf"),
        });
    }
    Ok(matrix)
}

fn validate_hessp_output(product: Vec<f64>, n: usize) -> Result<Vec<f64>, OptError> {
    if product.len() != n {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "hessp callback returned length {}, expected {n}",
                product.len()
            ),
        });
    }
    if product.iter().any(|value| !value.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: String::from("hessp callback returned NaN or Inf"),
        });
    }
    Ok(product)
}

/// Step lengths `alpha` keeping `x + alpha*direction` inside the box: SciPy's `_line_for_search`.
/// `x` must be feasible, so the interval always contains 0.
fn feasible_step_interval(x: &[f64], direction: &[f64], bounds: &[Bound]) -> (f64, f64) {
    let mut lo = f64::NEG_INFINITY;
    let mut hi = f64::INFINITY;
    for ((&xi, &di), &(lb, ub)) in x.iter().zip(direction).zip(bounds) {
        if di == 0.0 {
            continue;
        }
        let to_lb = lb.map(|l| (l - xi) / di);
        let to_ub = ub.map(|u| (u - xi) / di);
        let (low_side, high_side) = if di > 0.0 {
            (to_lb, to_ub)
        } else {
            (to_ub, to_lb)
        };
        if let Some(a) = low_side {
            lo = lo.max(a);
        }
        if let Some(a) = high_side {
            hi = hi.min(a);
        }
    }
    (lo.min(0.0), hi.max(0.0))
}

/// Powell's line search along `direction`: SciPy's `_linesearch_powell`. With no finite limit on
/// the step it brackets from (0, 1) and runs Brent ([`brent_line_minimum`], relative
/// `tolerance`); on a two-sided feasible interval it runs `fminbound`'s bounded Brent
/// ([`bounded_line_minimum`], `xatol = tolerance/100`); on a one-sided interval the same over
/// `alpha = tan(t)`, `t ∈ [atan(lo), atan(hi)]`. `tolerance` is SciPy's `xtol·100`.
///
/// This used to be a golden-section search that stopped at a bracket width of about 1e-4 in
/// ABSOLUTE alpha on unit directions, which capped Powell's accuracy near 2e-5 and stalled it
/// on curved valleys (frankenscipy-de6qs).
fn powell_line_search<F>(
    objective: &mut Objective<'_, F>,
    x: &[f64],
    fx: f64,
    direction: &[f64],
    tolerance: f64,
    bounds: Option<&[Bound]>,
) -> Result<LineSearchStep, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    let stay = LineSearchStep {
        alpha: 0.0,
        x: x.to_vec(),
        f: fx,
    };
    if direction.iter().all(|&d| d == 0.0) {
        return Ok(stay);
    }
    let (lo, hi) = bounds.map_or((f64::NEG_INFINITY, f64::INFINITY), |b| {
        feasible_step_interval(x, direction, b)
    });
    let mut candidate_x = vec![0.0; x.len()];
    let mut phi = |alpha: f64| -> Result<f64, OptError> {
        add_scaled_into(&mut candidate_x, x, direction, alpha);
        if let Some(b) = bounds {
            // Rounding in x + alpha*d can overshoot a bound by an ulp; the objective must only
            // ever see feasible points.
            project_onto_bounds(&mut candidate_x, b);
        }
        objective.eval(&candidate_x)
    };
    let (alpha, f_alpha) = if lo == f64::NEG_INFINITY && hi == f64::INFINITY {
        brent_line_minimum(&mut phi, tolerance)?
    } else if lo == hi {
        return Ok(stay);
    } else if lo.is_finite() && hi.is_finite() {
        bounded_line_minimum(&mut phi, lo, hi, tolerance / 100.0)?
    } else {
        let (t, f_t) = bounded_line_minimum(
            &mut |t: f64| phi(t.tan()),
            lo.atan(),
            hi.atan(),
            tolerance / 100.0,
        )?;
        (t.tan(), f_t)
    };
    // SciPy takes the scalar minimizer's point even when it is worse than `x` (`fminbound`
    // never evaluates the interval ends, and `x` sits on one when it is on a bound); only the
    // bracket-recovery NaN is refused here.
    if alpha.is_nan() || f_alpha.is_nan() {
        return Ok(stay);
    }
    let mut best_x = vec![0.0; x.len()];
    add_scaled_into(&mut best_x, x, direction, alpha);
    if let Some(b) = bounds {
        project_onto_bounds(&mut best_x, b);
    }
    Ok(LineSearchStep {
        alpha,
        x: best_x,
        f: f_alpha,
    })
}

/// SciPy's `_minimize_scalar_brent(f, brack=None, xtol=tol)` as `_linesearch_powell` calls it:
/// `bracket` grows a downhill bracket from (0, 1) (golden ratio, parabolic extrapolation capped
/// at 110× the last interval, ≤ 1000 iterations), then Brent's method refines it (relative
/// `tol` plus a 1e-11 floor, ≤ 500 iterations). When no valid bracket is found the best of its
/// three points is returned, as `_recover_from_bracket_error` does.
fn brent_line_minimum(
    f: &mut impl FnMut(f64) -> Result<f64, OptError>,
    tol: f64,
) -> Result<(f64, f64), OptError> {
    const GOLD: f64 = 1.618_034;
    const VERY_SMALL: f64 = 1.0e-21;
    const GROW_LIMIT: f64 = 110.0;
    const BRACKET_MAXITER: usize = 1000;
    const MINTOL: f64 = 1.0e-11;
    const CG: f64 = 0.381_966;
    const BRENT_MAXITER: usize = 500;

    let (mut xa, mut xb) = (0.0_f64, 1.0_f64);
    let (mut fa, mut fb) = (f(xa)?, f(xb)?);
    if fa < fb {
        std::mem::swap(&mut xa, &mut xb);
        std::mem::swap(&mut fa, &mut fb);
    }
    let mut xc = xb + GOLD * (xb - xa);
    let mut fc = f(xc)?;
    let mut iterations = 0;
    let mut exhausted = false;
    while fc < fb {
        let tmp1 = (xb - xa) * (fb - fc);
        let tmp2 = (xb - xc) * (fb - fa);
        let val = tmp2 - tmp1;
        let denom = if val.abs() < VERY_SMALL {
            2.0 * VERY_SMALL
        } else {
            2.0 * val
        };
        let mut w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom;
        let wlim = xb + GROW_LIMIT * (xc - xb);
        if iterations > BRACKET_MAXITER {
            exhausted = true;
            break;
        }
        iterations += 1;
        let mut fw;
        if (w - xc) * (xb - w) > 0.0 {
            fw = f(w)?;
            if fw < fc {
                xa = xb;
                xb = w;
                fa = fb;
                fb = fw;
                break;
            } else if fw > fb {
                xc = w;
                fc = fw;
                break;
            }
            w = xc + GOLD * (xc - xb);
            fw = f(w)?;
        } else if (w - wlim) * (wlim - xc) >= 0.0 {
            w = wlim;
            fw = f(w)?;
        } else if (w - wlim) * (xc - w) > 0.0 {
            fw = f(w)?;
            if fw < fc {
                xb = xc;
                xc = w;
                w = xc + GOLD * (xc - xb);
                fb = fc;
                fc = fw;
                fw = f(w)?;
            }
        } else {
            w = xc + GOLD * (xc - xb);
            fw = f(w)?;
        }
        xa = xb;
        xb = xc;
        xc = w;
        fa = fb;
        fb = fc;
        fc = fw;
    }
    let valid = !exhausted
        && ((fb < fc && fb <= fa) || (fb < fa && fb <= fc))
        && ((xa < xb && xb < xc) || (xc < xb && xb < xa))
        && xa.is_finite()
        && xb.is_finite()
        && xc.is_finite();
    if !valid {
        let points = [(xa, fa), (xb, fb), (xc, fc)];
        if points.iter().any(|(x, fx)| x.is_nan() || fx.is_nan()) {
            return Ok((f64::NAN, f64::NAN));
        }
        let mut best = points[0];
        for point in &points[1..] {
            if point.1 < best.1 {
                best = *point;
            }
        }
        return Ok(best);
    }

    let (mut x, mut w, mut v) = (xb, xb, xb);
    let (mut fx, mut fw, mut fv) = (fb, fb, fb);
    let (mut a, mut b) = if xa < xc { (xa, xc) } else { (xc, xa) };
    let mut deltax = 0.0_f64;
    let mut rat = 0.0_f64;
    for _ in 0..BRENT_MAXITER {
        let tol1 = tol * x.abs() + MINTOL;
        let tol2 = 2.0 * tol1;
        let xmid = 0.5 * (a + b);
        if (x - xmid).abs() < tol2 - 0.5 * (b - a) {
            break;
        }
        if deltax.abs() <= tol1 {
            deltax = if x >= xmid { a - x } else { b - x };
            rat = CG * deltax;
        } else {
            let tmp1 = (x - w) * (fx - fv);
            let mut tmp2 = (x - v) * (fx - fw);
            let mut p = (x - v) * tmp2 - (x - w) * tmp1;
            tmp2 = 2.0 * (tmp2 - tmp1);
            if tmp2 > 0.0 {
                p = -p;
            }
            tmp2 = tmp2.abs();
            let dx_temp = deltax;
            deltax = rat;
            if p > tmp2 * (a - x) && p < tmp2 * (b - x) && p.abs() < (0.5 * tmp2 * dx_temp).abs() {
                rat = p / tmp2;
                let u = x + rat;
                if (u - a) < tol2 || (b - u) < tol2 {
                    rat = if xmid - x >= 0.0 { tol1 } else { -tol1 };
                }
            } else {
                deltax = if x >= xmid { a - x } else { b - x };
                rat = CG * deltax;
            }
        }
        let u = if rat.abs() < tol1 {
            if rat >= 0.0 { x + tol1 } else { x - tol1 }
        } else {
            x + rat
        };
        let fu = f(u)?;
        if fu > fx {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                v = w;
                w = u;
                fv = fw;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        } else {
            if u >= x {
                a = x;
            } else {
                b = x;
            }
            v = w;
            w = x;
            x = u;
            fv = fw;
            fw = fx;
            fx = fu;
        }
    }
    Ok((x, fx))
}

/// SciPy's `_minimize_scalar_bounded` (`fminbound`) on `[x1, x2]`: Brent's method with
/// tolerance `√eps·|x| + xatol/3`, at most 500 evaluations. Returns the best point evaluated.
fn bounded_line_minimum(
    f: &mut impl FnMut(f64) -> Result<f64, OptError>,
    x1: f64,
    x2: f64,
    xatol: f64,
) -> Result<(f64, f64), OptError> {
    const MAXFUN: usize = 500;
    let sqrt_eps = 2.2e-16_f64.sqrt();
    let golden_mean = 0.5 * (3.0 - 5.0_f64.sqrt());
    // np.sign(v) + (v == 0): the sign, with 0 counted as positive.
    let sign = |v: f64| if v < 0.0 { -1.0 } else { 1.0 };

    let (mut a, mut b) = (x1, x2);
    let mut fulc = a + golden_mean * (b - a);
    let (mut nfc, mut xf) = (fulc, fulc);
    let (mut rat, mut e) = (0.0_f64, 0.0_f64);
    let mut fx = f(xf)?;
    let mut num = 1;
    let (mut ffulc, mut fnfc) = (fx, fx);
    let mut xm = 0.5 * (a + b);
    let mut tol1 = sqrt_eps * xf.abs() + xatol / 3.0;
    let mut tol2 = 2.0 * tol1;
    while (xf - xm).abs() > tol2 - 0.5 * (b - a) {
        let mut golden = true;
        if e.abs() > tol1 {
            golden = false;
            let r = (xf - nfc) * (fx - ffulc);
            let mut q = (xf - fulc) * (fx - fnfc);
            let mut p = (xf - fulc) * q - (xf - nfc) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            let r = e;
            e = rat;
            if p.abs() < (0.5 * q * r).abs() && p > q * (a - xf) && p < q * (b - xf) {
                rat = p / q;
                let x = xf + rat;
                if (x - a) < tol2 || (b - x) < tol2 {
                    rat = tol1 * sign(xm - xf);
                }
            } else {
                golden = true;
            }
        }
        if golden {
            e = if xf >= xm { a - xf } else { b - xf };
            rat = golden_mean * e;
        }
        let x = xf + sign(rat) * rat.abs().max(tol1);
        let fu = f(x)?;
        num += 1;
        if fu <= fx {
            if x >= xf {
                a = xf;
            } else {
                b = xf;
            }
            fulc = nfc;
            ffulc = fnfc;
            nfc = xf;
            fnfc = fx;
            xf = x;
            fx = fu;
        } else {
            if x < xf {
                a = x;
            } else {
                b = x;
            }
            if fu <= fnfc || nfc == xf {
                fulc = nfc;
                ffulc = fnfc;
                nfc = x;
                fnfc = fu;
            } else if fu <= ffulc || fulc == xf || fulc == nfc {
                fulc = x;
                ffulc = fu;
            }
        }
        xm = 0.5 * (a + b);
        tol1 = sqrt_eps * xf.abs() + xatol / 3.0;
        tol2 = 2.0 * tol1;
        if num >= MAXFUN {
            break;
        }
    }
    Ok((xf, fx))
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
        OptError::NotConverged { detail } | OptError::NoConvergence { detail } => {
            (ConvergenceStatus::MaxIterations, detail)
        }
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
    if let Some(eps) = options.gradient_eps
        && (!eps.is_finite() || eps <= 0.0)
    {
        return Err(OptError::InvalidArgument {
            detail: String::from("gradient_eps must be finite and > 0"),
        });
    }
    Ok(())
}

/// The SciPy option names each method reads from [`MinimizeMethodOptions`]
/// (frankenscipy-6ycp2). The other methods read none of them yet.
fn accepted_method_options(method: OptimizeMethod) -> &'static [&'static str] {
    match method {
        OptimizeMethod::Bfgs => &["gtol", "norm", "c1", "c2", "xrtol"],
        OptimizeMethod::ConjugateGradient => &["gtol", "norm", "c1", "c2"],
        OptimizeMethod::LBfgsB => &["gtol", "ftol", "maxcor", "maxls"],
        OptimizeMethod::NelderMead => &["xatol", "fatol", "adaptive", "initial_simplex"],
        OptimizeMethod::Powell => &["xtol", "ftol", "direc"],
        _ => &[],
    }
}

/// The names of the options set in `options`.
fn set_method_options(options: &MinimizeMethodOptions<'_>) -> Vec<&'static str> {
    [
        ("gtol", options.gtol.is_some()),
        ("norm", options.norm.is_some()),
        ("c1", options.c1.is_some()),
        ("c2", options.c2.is_some()),
        ("xrtol", options.xrtol.is_some()),
        ("maxcor", options.maxcor.is_some()),
        ("maxls", options.maxls.is_some()),
        ("ftol", options.ftol.is_some()),
        ("xtol", options.xtol.is_some()),
        ("xatol", options.xatol.is_some()),
        ("fatol", options.fatol.is_some()),
        ("adaptive", options.adaptive.is_some()),
        ("initial_simplex", options.initial_simplex.is_some()),
        ("direc", options.direc.is_some()),
    ]
    .into_iter()
    .filter_map(|(name, set)| set.then_some(name))
    .collect()
}

/// SciPy's `_check_unknown_options` and the per-option checks its solvers make
/// (frankenscipy-6ycp2). An option the method does not read is an `OptimizeWarning` in SciPy,
/// which then proceeds: Strict does the same and records it in the optimize trace; Hardened
/// refuses it. The values SciPy itself refuses (`0 < c1 < c2 < 1`, `maxls` and `maxcor` at
/// least 1) are refused in both modes.
fn check_method_options(method: OptimizeMethod, options: MinimizeOptions) -> Result<(), OptError> {
    let method_options = options.method_options;
    let accepted = accepted_method_options(method);
    let unknown: Vec<&str> = set_method_options(&method_options)
        .into_iter()
        .filter(|name| !accepted.contains(name))
        .collect();
    if !unknown.is_empty() {
        let detail = format!(
            "Unknown solver options for {method:?}: {}",
            unknown.join(", ")
        );
        if options.mode == RuntimeMode::Hardened {
            return Err(OptError::InvalidArgument { detail });
        }
        push_trace(OptimizeTraceEntry {
            ts_unix_ms: now_unix_ms(),
            event: String::from("unknown_solver_options"),
            method,
            iter_num: 0,
            f_val: None,
            grad_norm: None,
            step_size: None,
            mode: options.mode,
            reason: Some(detail),
            final_x: None,
            final_f: None,
            total_nfev: 0,
            fixture_id: options.fixture_id.map(ToOwned::to_owned),
            seed: options.seed,
        });
    }
    if matches!(
        method,
        OptimizeMethod::Bfgs | OptimizeMethod::ConjugateGradient
    ) {
        let c1 = method_options.c1.unwrap_or(1.0e-4);
        let c2 = method_options
            .c2
            .unwrap_or(if method == OptimizeMethod::Bfgs {
                0.9
            } else {
                0.4
            });
        if !(0.0 < c1 && c1 < c2 && c2 < 1.0) {
            return Err(OptError::InvalidArgument {
                detail: String::from("'c1' and 'c2' do not satisfy '0 < c1 < c2 < 1'."),
            });
        }
    }
    if method == OptimizeMethod::LBfgsB {
        if method_options.maxls == Some(0) {
            return Err(OptError::InvalidArgument {
                detail: String::from("maxls must be positive."),
            });
        }
        if method_options.maxcor == Some(0) {
            return Err(OptError::InvalidArgument {
                detail: String::from("maxcor must be positive."),
            });
        }
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
                // status: Brent bracket stop |x − mid| ≤ 2·tol1 − (b−a)/2
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
    check_method_options(OptimizeMethod::Tnc, options)?;

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
        let value = evaluate_minimize_gradient(
            &mut objective,
            options.gradient,
            &x,
            options.gradient_eps.unwrap_or(CENTRAL_DIFF_EPS),
        )?;
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
                    // status: ‖projected ∇f‖₂ < tol (plain ∇f when no bounds)
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
                // status: ‖projected ∇f‖₂ < tol (plain ∇f when no bounds)
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

        let grad_new = match evaluate_minimize_gradient(
            &mut objective,
            options.gradient,
            &x_new,
            options.gradient_eps.unwrap_or(CENTRAL_DIFF_EPS),
        ) {
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

/// Refuse bounds and constraints for a kernel that cannot honour them yet.
///
/// `trust_constr` below is an UNCONSTRAINED quasi-Newton kernel; the constrained algorithm SciPy
/// runs under that name is frankenscipy-1ksfv.2. SciPy honours bounds and constraints for it, so
/// ignoring them returned an infeasible optimum under `success = true` (`(x-3)^2` with bounds
/// `[(0,2)]` came back as `x = 3`; frankenscipy-szq1n.7). Bounds: SLSQP, L-BFGS-B, TNC or
/// Nelder-Mead. General constraints: SLSQP.
fn reject_unhonoured_constraints(method: &str, options: MinimizeOptions) -> Result<(), OptError> {
    if options.bounds.is_some_and(bounds_have_finite_limit) {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "{method} does not implement bounds yet; use SLSQP, L-BFGS-B, TNC or Nelder-Mead for box constraints"
            ),
        });
    }
    if !options.constraints.is_empty() {
        return Err(OptError::InvalidArgument {
            detail: format!("{method} does not implement constraints yet; use SLSQP"),
        });
    }
    Ok(())
}

/// `scipy.optimize.minimize(method='SLSQP')`: Kraft's sequential least-squares QP (see
/// [`crate::slsqp`]) under `options.bounds` and `options.constraints`.
///
/// SciPy's option mapping: `tol` is `ftol` (default 1e-6), `maxiter` defaults to 100, and
/// `gradient_eps` is `eps` (default √ε), the absolute forward-difference step for the gradient
/// (when `options.gradient` is absent) and for every constraint without `jac` — stepped
/// backwards or shortened at a bound exactly as `approx_derivative(..., '2-point',
/// abs_step=eps, bounds)` does. `x0` is clipped
/// into the bounds first. The message is SciPy's exit-mode text; `maxcv` is the largest
/// constraint violation at `x`. In Strict mode a non-finite objective value is passed to the
/// algorithm as SciPy's is (its convergence tests never accept one); Hardened mode rejects it.
pub fn slsqp<F>(fun: &F, x0: &[f64], options: MinimizeOptions) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    validate_minimize_options(options)?;
    check_method_options(OptimizeMethod::Slsqp, options)?;
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
    validate_bounds_for_x0(x0, options.bounds)?;
    let n = x0.len();
    let (lb, ub): (Vec<f64>, Vec<f64>) = match options.bounds {
        Some(bounds) => bounds
            .iter()
            .map(|&(lo, hi)| (lo.unwrap_or(f64::NEG_INFINITY), hi.unwrap_or(f64::INFINITY)))
            .unzip(),
        None => (vec![f64::NEG_INFINITY; n], vec![f64::INFINITY; n]),
    };
    let mut x: Vec<f64> = x0
        .iter()
        .zip(lb.iter().zip(&ub))
        .map(|(&v, (&lo, &hi))| v.max(lo).min(hi))
        .collect();

    // SciPy triages constraints into equalities then inequalities, each in the given order,
    // and sizes them by one evaluation at the clipped x0.
    let ordered: Vec<&Constraint<'_>> = options
        .constraints
        .iter()
        .filter(|c| c.kind == ConstraintType::Eq)
        .chain(
            options
                .constraints
                .iter()
                .filter(|c| c.kind == ConstraintType::Ineq),
        )
        .collect();
    let sizes: Vec<usize> = ordered.iter().map(|c| (c.fun)(&x).len()).collect();
    let meq: usize = ordered
        .iter()
        .zip(&sizes)
        .filter(|(c, _)| c.kind == ConstraintType::Eq)
        .map(|(_, &s)| s)
        .sum();
    let m: usize = sizes.iter().sum();
    let acc = options.tol.unwrap_or(1.0e-6);
    let maxiter = options.maxiter.unwrap_or(100);
    let xl: Vec<f64> = lb
        .iter()
        .map(|&v| if v.is_finite() { v } else { f64::NAN })
        .collect();
    let xu: Vec<f64> = ub
        .iter()
        .map(|&v| if v.is_finite() { v } else { f64::NAN })
        .collect();

    let mut driver = SlsqpDriver {
        fun,
        constraints: ordered,
        sizes,
        m,
        gradient: options.gradient,
        callback: options.callback,
        eps: options.gradient_eps.unwrap_or(SCIPY_SQRT_EPS),
        lb,
        ub,
        mode: options.mode,
        maxfev: options.maxfev.unwrap_or(usize::MAX),
        nfev: 0,
        njev: 0,
        last_f: f64::NAN,
    };
    let outcome = crate::slsqp::run(&mut driver, &mut x, &xl, &xu, m, meq, acc, maxiter);
    let maxcv = if m > 0 {
        Some(driver.max_violation(&x))
    } else {
        None
    };
    let result = match outcome {
        Ok(out) => {
            let status = if out.stopped_by_callback {
                ConvergenceStatus::CallbackStop
            } else {
                match out.mode {
                    0 => ConvergenceStatus::Success,
                    9 => ConvergenceStatus::MaxIterations,
                    4 => ConvergenceStatus::Infeasible,
                    2 => ConvergenceStatus::InvalidInput,
                    _ => ConvergenceStatus::PrecisionLoss,
                }
            };
            let message = if out.stopped_by_callback {
                String::from("Optimization stopped by callback")
            } else {
                String::from(crate::slsqp::exit_message(out.mode))
            };
            OptimizeResult {
                x,
                fun: Some(out.fx),
                // status: SciPy SLSQP exit mode 0 (KKT and step tests passed on a consistent QP)
                success: out.mode == 0 && !out.stopped_by_callback,
                status,
                message,
                nfev: driver.nfev,
                njev: driver.njev,
                nhev: 0,
                nit: out.iter,
                jac: Some(out.grad),
                hess_inv: None,
                maxcv,
            }
        }
        Err(OptError::EvaluationBudgetExceeded { detail }) => OptimizeResult {
            x,
            fun: Some(driver.last_f),
            success: false,
            status: ConvergenceStatus::MaxEvaluations,
            message: detail,
            nfev: driver.nfev,
            njev: driver.njev,
            nhev: 0,
            nit: 0,
            jac: None,
            hess_inv: None,
            maxcv,
        },
        Err(err) => return Err(err),
    };
    log_completion(OptimizeMethod::Slsqp, options, result.nit, &result);
    Ok(result)
}

/// The objective and constraints of one `slsqp` call, with SciPy's finite differences.
struct SlsqpDriver<'a, F> {
    fun: &'a F,
    constraints: Vec<&'a Constraint<'a>>,
    sizes: Vec<usize>,
    m: usize,
    gradient: Option<GradientFunc>,
    callback: Option<crate::types::MinimizeCallback>,
    eps: f64,
    lb: Vec<f64>,
    ub: Vec<f64>,
    mode: RuntimeMode,
    maxfev: usize,
    nfev: usize,
    njev: usize,
    last_f: f64,
}

impl<F> SlsqpDriver<'_, F>
where
    F: Fn(&[f64]) -> f64,
{
    fn eval_f(&mut self, x: &[f64]) -> Result<f64, OptError> {
        if self.nfev >= self.maxfev {
            return Err(OptError::EvaluationBudgetExceeded {
                detail: format!("max function evaluations exceeded ({})", self.maxfev),
            });
        }
        self.nfev += 1;
        let value = (self.fun)(x);
        if !value.is_finite() && self.mode == RuntimeMode::Hardened {
            return Err(OptError::NonFiniteInput {
                detail: String::from("hardened mode rejects non-finite objective values"),
            });
        }
        Ok(value)
    }

    /// All constraint values at `x`, equalities first.
    fn eval_constraints(&self, x: &[f64]) -> Result<Vec<f64>, OptError> {
        let mut out = Vec::with_capacity(self.m);
        for (k, (con, &size)) in self.constraints.iter().zip(&self.sizes).enumerate() {
            let values = (con.fun)(x);
            if values.len() != size {
                return Err(OptError::InvalidArgument {
                    detail: format!(
                        "constraint {k} returned {} values, {size} at x0",
                        values.len()
                    ),
                });
            }
            if self.mode == RuntimeMode::Hardened && values.iter().any(|v| !v.is_finite()) {
                return Err(OptError::NonFiniteInput {
                    detail: format!("hardened mode rejects non-finite values of constraint {k}"),
                });
            }
            out.extend(values);
        }
        Ok(out)
    }

    /// SciPy `approx_derivative(method='2-point', abs_step=eps, bounds=(lb, ub))` steps.
    fn fd_steps(&self, x: &[f64]) -> Vec<f64> {
        fd_steps_2point(x, self.eps, Some((&self.lb, &self.ub)))
    }

    fn max_violation(&self, x: &[f64]) -> f64 {
        let Ok(values) = self.eval_constraints(x) else {
            return f64::NAN;
        };
        let meq: usize = self
            .constraints
            .iter()
            .zip(&self.sizes)
            .filter(|(c, _)| c.kind == ConstraintType::Eq)
            .map(|(_, &s)| s)
            .sum();
        // NaN-propagating: `(-v).max(0.0)` and `f64::max` both drop a NaN, which reported a
        // constraint that had gone NaN as maxcv 0.
        values
            .iter()
            .enumerate()
            .map(|(i, &v)| if i < meq { v.abs() } else { -v })
            .fold(0.0, |worst: f64, v| {
                if worst.is_nan() || v.is_nan() {
                    f64::NAN
                } else {
                    worst.max(v)
                }
            })
    }
}

impl<F> crate::slsqp::SlsqpProblem for SlsqpDriver<'_, F>
where
    F: Fn(&[f64]) -> f64,
{
    type Error = OptError;

    fn eval_fc(&mut self, x: &[f64], d: &mut [f64]) -> Result<f64, OptError> {
        let f = self.eval_f(x)?;
        self.last_f = f;
        d.copy_from_slice(&self.eval_constraints(x)?);
        Ok(f)
    }

    fn eval_gc(
        &mut self,
        x: &[f64],
        fx: f64,
        g: &mut [f64],
        c: &mut [f64],
    ) -> Result<(), OptError> {
        self.njev += 1;
        let n = x.len();
        let steps = self.fd_steps(x);
        if let Some(gradient) = self.gradient {
            g.copy_from_slice(&validate_gradient_output(gradient(x), n)?);
        } else {
            let mut xp = x.to_vec();
            for i in 0..n {
                xp[i] = x[i] + steps[i];
                let dx = xp[i] - x[i];
                g[i] = (self.eval_f(&xp)? - fx) / dx;
                xp[i] = x[i];
            }
        }
        let lda = self.m.max(1);
        let mut row = 0;
        for (k, (con, &size)) in self.constraints.iter().zip(&self.sizes).enumerate() {
            let jac: Vec<Vec<f64>> = if let Some(jac) = &con.jac {
                jac(x)
            } else {
                let c0 = (con.fun)(x);
                let mut cols = vec![vec![0.0; n]; size];
                let mut xp = x.to_vec();
                for i in 0..n {
                    xp[i] = x[i] + steps[i];
                    let dx = xp[i] - x[i];
                    let c1 = (con.fun)(&xp);
                    for r in 0..size.min(c1.len()).min(c0.len()) {
                        cols[r][i] = (c1[r] - c0[r]) / dx;
                    }
                    xp[i] = x[i];
                }
                cols
            };
            if jac.len() != size || jac.iter().any(|r| r.len() != n) {
                return Err(OptError::InvalidArgument {
                    detail: format!("jacobian of constraint {k} must be {size}x{n}"),
                });
            }
            for (r, jrow) in jac.iter().enumerate() {
                for (i, &v) in jrow.iter().enumerate() {
                    c[row + r + i * lda] = v;
                }
            }
            row += size;
        }
        Ok(())
    }

    fn callback(&mut self, x: &[f64], _fx: f64) -> bool {
        self.callback.is_some_and(|cb| cb(x))
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
    check_method_options(OptimizeMethod::TrustConstr, options)?;
    reject_unhonoured_constraints("trust-constr", options)?;

    let n = x0.len();
    let tol = requested_tolerance(options.tol);
    let maxiter = options.maxiter.unwrap_or((200 * n).max(100));
    let maxfev = options.maxfev.unwrap_or((2000 * n).max(400));
    let mut objective = Objective::new(fun, options.mode, maxfev);

    let mut x = x0.to_vec();
    let mut f = objective.eval(&x)?;

    let mut grad = evaluate_minimize_gradient(
        &mut objective,
        options.gradient,
        &x,
        options.gradient_eps.unwrap_or(CENTRAL_DIFF_EPS),
    )?;
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
                    // status: KKT optimality ‖∇f‖_inf ≤ tol (unconstrained)
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
                // status: KKT optimality ‖∇f‖_inf ≤ tol (unconstrained)
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
            let grad_new = evaluate_minimize_gradient(
                &mut objective,
                options.gradient,
                &x_new,
                options.gradient_eps.unwrap_or(CENTRAL_DIFF_EPS),
            )?;
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
                // status: trust radius < tol·1e-6 (SciPy trust-constr xtol stop, status 2 = success)
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

    use fsci_runtime::{OptSolverAction, OptSolverPortfolio, RuntimeMode};
    use proptest::prelude::*;
    use serde::Serialize;

    use super::{slsqp, tnc, trust_constr};

    use super::{
        Objective, feasible_step_interval, minimize_with_casp, minimize_with_casp_portfolio,
        powell_line_search,
    };
    use crate::{
        Bound, Constraint, ConvergenceStatus, GradientFunc, HessFunc, HesspFunc, LinearConstraint,
        MinimizeOptions, MinimizeScalarOptions, NonlinearConstraint, OptCaspProblem, OptError,
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
        // trust-ncg, which uses the products that selected it; the BFGS-model trust-exact
        // this used to pick never called `hessp`.
        assert_eq!(trust.method, OptimizeMethod::TrustNcg);
        assert!(trust.reason.contains("trust-ncg"));
    }

    static SELECTED_HESSP_CALLS: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);

    fn counted_offset_quadratic_hessp(_x: &[f64], p: &[f64]) -> Vec<f64> {
        SELECTED_HESSP_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        vec![2.0 * p[0], 2.0 * p[1]]
    }

    fn offset_quadratic_gradient(x: &[f64]) -> Vec<f64> {
        vec![2.0 * (x[0] - 3.0), 2.0 * (x[1] - 1.0e8)]
    }

    #[test]
    fn casp_trust_region_route_uses_the_hessian_products_that_selected_it() {
        // x0 spans eight orders of magnitude, so CASP takes its small ill-scaled branch.
        let options = MinimizeOptions {
            gradient: Some(offset_quadratic_gradient),
            hessp: Some(counted_offset_quadratic_hessp),
            ..MinimizeOptions::default()
        };
        let x0 = [1.0, 1.0e8 + 1.0];
        let decision =
            select_minimize_method(OptCaspProblem::from_x0_and_options(&x0, options)).unwrap();
        assert_eq!(decision.method, OptimizeMethod::TrustNcg);
        let before = SELECTED_HESSP_CALLS.load(std::sync::atomic::Ordering::Relaxed);
        let result = minimize(
            |x: &[f64]| (x[0] - 3.0).powi(2) + (x[1] - 1.0e8).powi(2),
            &x0,
            options,
        )
        .expect("minimize");
        let calls = SELECTED_HESSP_CALLS.load(std::sync::atomic::Ordering::Relaxed) - before;
        assert!(calls > 0, "the selected method never called hessp");
        assert!(result.success, "{}", result.message);
        assert!(
            (result.x[0] - 3.0).abs() < 1e-4 && (result.x[1] - 1.0e8).abs() < 1e-4,
            "x = {:?}",
            result.x
        );
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

    // frankenscipy-1ksfv.1: with constraints and no method, `minimize` routes to SLSQP exactly as
    // `scipy.optimize.minimize` does, and honours them. SciPy 1.17.1:
    // minimize((x-2)^2 + (y-1)^2, [0, 0], constraints=[{'type': 'ineq', 'fun': 1 - x - y}])
    // -> x = [1, 0], fun = 2, success. The unconstrained optimum (2, 1) violates it by 2.
    #[test]
    fn casp_default_minimize_routes_constraints_to_slsqp() {
        const FIXTURE_ID: &str = "casp-default-slsqp";
        let _ = get_optimize_traces();
        let constraints = [Constraint::ineq(|v: &[f64]| vec![1.0 - v[0] - v[1]])];
        let options = MinimizeOptions {
            method: None,
            constraints: &constraints,
            fixture_id: Some(FIXTURE_ID),
            ..MinimizeOptions::default()
        };
        let r = minimize(
            |v: &[f64]| (v[0] - 2.0).powi(2) + (v[1] - 1.0).powi(2),
            &[0.0, 0.0],
            options,
        )
        .expect("slsqp run");
        assert!(r.success, "{}", r.message);
        assert!(
            (r.x[0] - 1.0).abs() < 1e-6 && r.x[1].abs() < 1e-6,
            "x = {:?} (the unconstrained optimum is (2, 1))",
            r.x
        );
        assert!((r.fun.expect("fun") - 2.0).abs() < 1e-8);
        assert!(r.maxcv.expect("maxcv") <= 1e-8, "maxcv {:?}", r.maxcv);

        let traces = get_optimize_traces();
        let decision = traces
            .iter()
            .find(|entry| {
                entry.event == "casp_decision" && entry.fixture_id.as_deref() == Some(FIXTURE_ID)
            })
            .expect("CASP decision trace");
        assert_eq!(decision.method, OptimizeMethod::Slsqp);
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
        // SciPy 1.17.1: status 1, nit 1, nfev 9, njev 3 (forward differences, eps = √ε).
        assert_eq!(
            result.message,
            "Maximum number of iterations has been exceeded."
        );
        assert_eq!((result.nit, result.nfev, result.njev), (1, 9, 3));
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
            gradient_eps: Some(1.0e-6),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result =
            minimize(step_plateau, &[0.499_999], options).expect("minimize returns a result");
        assert!(!result.success);
        assert_eq!(result.status, ConvergenceStatus::PrecisionLoss);
        // SciPy 1.17.1 (eps = 1e-6): both line searches fail at the first step.
        assert_eq!(
            result.message,
            "Desired error not necessarily achieved due to precision loss."
        );
        assert_eq!((result.nit, result.nfev, result.njev), (0, 86, 37));
        assert_eq!(result.x, vec![0.499_999]);
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
    fn bfgs_nonfinite_gradient_is_scipys_nan_status_in_strict_and_refused_in_hardened() {
        fn nonfinite_gradient(x: &[f64]) -> Vec<f64> {
            vec![f64::NAN; x.len()]
        }
        let run = |mode| {
            bfgs(
                &sphere,
                &[2.0, -3.0],
                MinimizeOptions {
                    method: Some(OptimizeMethod::Bfgs),
                    gradient: Some(nonfinite_gradient),
                    mode,
                    ..MinimizeOptions::default()
                },
            )
            .expect("bfgs returns a result")
        };

        // SciPy 1.17.1: status 3 "NaN result encountered.", nit 0, nfev 1, njev 1, fun 13.
        let strict = run(RuntimeMode::Strict);
        assert!(!strict.success);
        assert_eq!(strict.status, ConvergenceStatus::NanEncountered);
        assert_eq!(strict.message, "NaN result encountered.");
        assert_eq!((strict.nit, strict.nfev, strict.njev), (0, 1, 1));
        assert_eq!(strict.fun, Some(13.0));

        let hardened = run(RuntimeMode::Hardened);
        assert!(!hardened.success);
        assert_eq!(hardened.status, ConvergenceStatus::NanEncountered);
        assert!(
            hardened
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
        fn sphere_gradient(x: &[f64]) -> Vec<f64> {
            x.iter().map(|v| 2.0 * v).collect()
        }
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            tol: Some(1.0e-8),
            maxiter: Some(200),
            maxfev: Some(20_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let exact = bfgs(
            &sphere,
            &[0.0, 0.0],
            MinimizeOptions {
                gradient: Some(sphere_gradient),
                ..options
            },
        )
        .expect("bfgs executes");
        // SciPy 1.17.1 with jac: status 0, nit 0, nfev 1, njev 1.
        assert!(exact.success);
        assert_eq!((exact.nit, exact.nfev, exact.njev), (0, 1, 1));

        // Without jac the forward difference at the origin is h = √ε ≈ 1.49e-8 > gtol, and
        // SciPy's line searches then fail: status 2, nit 0, nfev 315, njev 101.
        let result = bfgs(&sphere, &[0.0, 0.0], options).expect("bfgs executes");
        assert!(!result.success);
        assert_eq!(result.status, ConvergenceStatus::PrecisionLoss);
        assert_eq!((result.nit, result.nfev, result.njev), (0, 315, 101));
        assert_eq!(result.x, vec![0.0, 0.0]);
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
        // From x = 1 the first trial step is 0.505 · (−2), into the NaN half-line.
        let result = minimize(objective, &[1.0], options).expect("returns an OptimizeResult");
        assert!(!result.success);
        assert_eq!(result.status, ConvergenceStatus::NanEncountered);

        // Strict hands the NaN to the line search as SciPy 1.17.1 does: status 2 after one
        // iteration at x = −1033.24, nfev 224, njev 112, fun NaN.
        let strict = minimize(
            objective,
            &[1.0],
            MinimizeOptions {
                mode: RuntimeMode::Strict,
                ..options
            },
        )
        .expect("returns an OptimizeResult");
        assert_eq!(strict.status, ConvergenceStatus::PrecisionLoss);
        assert_eq!((strict.nit, strict.nfev, strict.njev), (1, 224, 112));
        assert_eq!(strict.x, vec![-1033.24]);
        assert!(strict.fun.is_some_and(f64::is_nan));

        // From x = 0 the forward difference never leaves the domain: SciPy converges at once
        // (nit 0, nfev 2, njev 1) in either mode.
        let origin = minimize(objective, &[0.0], options).expect("returns an OptimizeResult");
        assert!(origin.success, "{}", origin.message);
        assert_eq!((origin.nit, origin.nfev, origin.njev), (0, 2, 1));
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
        // Its own recorder: `callback_points` is shared with tests running concurrently.
        static SEEN: Mutex<Vec<Vec<f64>>> = Mutex::new(Vec::new());
        fn record_and_stop(x: &[f64]) -> bool {
            SEEN.lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(x.to_vec());
            false
        }
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            callback: Some(record_and_stop),
            maxiter: Some(40),
            maxfev: Some(10_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let result = bfgs(&sphere, &[2.5, -1.5], options).expect("bfgs executes");
        let points = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        // SciPy 1.17.1 calls back after each iteration, never with x0; a StopIteration on the
        // first call ends it at nit 1, nfev 6, njev 2, x = (1.6339321450303306, −0.98035928…).
        assert_eq!(points.len(), 1);
        assert_eq!(points[0], result.x);
        assert!((result.x[0] - 1.633_932_145_030_330_6).abs() <= 1e-15);
        assert!((result.x[1] + 0.980_359_287_018_198_4).abs() <= 1e-15);
        assert_eq!((result.nit, result.nfev, result.njev), (1, 6, 2));
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

    /// `scipy.optimize.minimize(method='BFGS')` at its defaults, SciPy 1.17.1 values.
    #[test]
    fn bfgs_takes_scipys_path() {
        // SciPy's `rosen` / `rosen_der`.
        fn rosen(x: &[f64]) -> f64 {
            (0..x.len() - 1)
                .map(|i| 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2))
                .sum()
        }
        fn rosen_der(x: &[f64]) -> Vec<f64> {
            let n = x.len();
            let mut d = vec![0.0; n];
            for i in 1..n - 1 {
                d[i] = 200.0 * (x[i] - x[i - 1] * x[i - 1])
                    - 400.0 * (x[i + 1] - x[i] * x[i]) * x[i]
                    - 2.0 * (1.0 - x[i]);
            }
            d[0] = -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]);
            d[n - 1] = 200.0 * (x[n - 1] - x[n - 2] * x[n - 2]);
            d
        }
        // Indefinite at x0: two wells in x₀, two in x₂.
        fn nonconvex(x: &[f64]) -> f64 {
            0.25 * x[0].powi(4) - 0.5 * x[0] * x[0]
                + x[1] * x[1]
                + 0.1 * x[0] * x[1]
                + 0.05 * x[2].powi(4)
                - x[2] * x[2]
        }
        fn nonconvex_grad(x: &[f64]) -> Vec<f64> {
            vec![
                x[0].powi(3) - x[0] + 0.1 * x[1],
                2.0 * x[1] + 0.1 * x[0],
                0.2 * x[2].powi(3) - 2.0 * x[2],
            ]
        }
        type Case = (
            &'static str,
            fn(&[f64]) -> f64,
            Option<GradientFunc>,
            &'static [f64],
            (usize, usize, usize),
            &'static [f64],
        );
        let cases: [Case; 5] = [
            (
                "rosen2/jac",
                rosen,
                Some(rosen_der),
                &[-1.2, 1.0],
                (32, 39, 39),
                &[0.999_999_971_001_401_1, 0.999_999_946_119_096_5],
            ),
            (
                "rosen3/jac",
                rosen,
                Some(rosen_der),
                &[1.3, 0.7, 0.8],
                (13, 20, 20),
                &[
                    1.000_000_001_386_452_3,
                    1.000_000_006_876_866_6,
                    1.000_000_010_860_801_8,
                ],
            ),
            (
                "nonconvex/jac",
                nonconvex,
                Some(nonconvex_grad),
                &[0.1, 0.2, 0.3],
                (11, 18, 18),
                &[
                    1.002_496_802_905_991_7,
                    -0.050_124_628_905_726_15,
                    3.162_277_541_589_315_6,
                ],
            ),
            (
                "rosen2/fd",
                rosen,
                None,
                &[-1.2, 1.0],
                (32, 117, 39),
                &[0.999_995_501_496_128_8, 0.999_990_994_780_780_5],
            ),
            (
                "nonconvex/fd",
                nonconvex,
                None,
                &[0.1, 0.2, 0.3],
                (11, 72, 18),
                &[
                    1.002_496_817_710_361,
                    -0.050_124_627_141_102_07,
                    3.162_277_533_334_137_3,
                ],
            ),
        ];
        for (label, fun, gradient, x0, counts, want) in cases {
            let result = bfgs(
                &fun,
                x0,
                MinimizeOptions {
                    method: Some(OptimizeMethod::Bfgs),
                    gradient,
                    ..MinimizeOptions::default()
                },
            )
            .expect("bfgs executes");
            assert!(result.success, "{label}: {}", result.message);
            assert_eq!(result.message, "Optimization terminated successfully.");
            assert_eq!(
                (result.nit, result.nfev, result.njev),
                counts,
                "{label}: (nit, nfev, njev)"
            );
            // With a finite-difference gradient the path is SciPy's to the iteration, but the
            // inverse-Hessian products are numpy's BLAS in SciPy and ordered sums here, and the
            // differences amplify that rounding to ~2e-8 in x (measured 2.1e-8 and 8.3e-9).
            let tol = if gradient.is_some() { 1e-12 } else { 1e-7 };
            for (got, want) in result.x.iter().zip(want) {
                assert!(
                    (got - want).abs() <= tol * want.abs().max(1.0),
                    "{label}: x = {:?}",
                    result.x
                );
            }
        }
    }

    #[test]
    fn cg_quadratic_converges_with_small_iterations() {
        fn sphere_gradient(x: &[f64]) -> Vec<f64> {
            x.iter().map(|v| 2.0 * v).collect()
        }
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::ConjugateGradient),
            tol: Some(1.0e-8),
            maxiter: Some(80),
            maxfev: Some(20_000),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let exact = cg_pr_plus(
            &sphere,
            &[4.0, -1.5],
            MinimizeOptions {
                gradient: Some(sphere_gradient),
                ..options
            },
        )
        .expect("cg executes");
        assert!(exact.success, "{}", exact.message);
        assert!(exact.nit <= 20);

        // By forward differences the gradient cannot fall below gtol = 1e-8 (h = √ε leaves
        // ~1.5e-8), and SciPy 1.17.1 ends in precision loss after 2 iterations, (nfev, njev) =
        // (101, 30), at x within 1e-8 of the minimizer.
        let result = cg_pr_plus(&sphere, &[4.0, -1.5], options).expect("cg executes");
        assert_eq!(result.status, ConvergenceStatus::PrecisionLoss);
        assert_eq!((result.nit, result.nfev, result.njev), (2, 101, 30));
        assert!(result.x.iter().all(|v| v.abs() < 1.0e-8));
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

    /// `scipy.optimize.minimize(method='CG')` at its defaults: SciPy 1.17.1's (status, nit,
    /// nfev, njev) and x to the bit — CG has no BLAS product, so nothing rounds differently.
    #[test]
    fn cg_takes_scipys_path() {
        let cases: [(
            Option<GradientFunc>,
            Option<usize>,
            ConvergenceStatus,
            [usize; 3],
            [f64; 2],
        ); 3] = [
            (
                Some(rosenbrock_gradient),
                None,
                ConvergenceStatus::Success,
                [36, 78, 77],
                [1.000_000_005_485_334, 0.999_999_997_275_824_1],
            ),
            (
                None,
                None,
                ConvergenceStatus::Success,
                [37, 280, 93],
                [0.999_996_778_620_981_7, 0.999_993_554_925_881],
            ),
            (
                Some(rosenbrock_gradient),
                Some(3),
                ConvergenceStatus::MaxIterations,
                [3, 12, 11],
                [-0.477_789_082_671_575_05, 0.199_857_261_632_542_25],
            ),
        ];
        for (gradient, maxiter, status, counts, want) in cases {
            let result = cg_pr_plus(
                &rosenbrock,
                &[-1.2, 1.0],
                MinimizeOptions {
                    method: Some(OptimizeMethod::ConjugateGradient),
                    gradient,
                    maxiter,
                    ..MinimizeOptions::default()
                },
            )
            .expect("cg executes");
            assert_eq!(result.status, status, "{}", result.message);
            assert_eq!([result.nit, result.nfev, result.njev], counts);
            assert_eq!(
                result.x[0].to_bits(),
                want[0].to_bits(),
                "x = {:?}",
                result.x
            );
            assert_eq!(
                result.x[1].to_bits(),
                want[1].to_bits(),
                "x = {:?}",
                result.x
            );
        }
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

    /// frankenscipy-cjv9z: Powell's stopping test must be relative in f, as SciPy's is. The
    /// absolute `|Δf| ≤ tol` stopped `1e-12·rosen` after one sweep, far from x* = (1, 1).
    /// SciPy 1.17.1, same x0 and tol=1e-6: x* to 2.4e-6 at scale 1e-12 and 4e-14 unscaled.
    #[test]
    fn powell_stopping_test_is_invariant_to_objective_scale() {
        let rosen = |x: &[f64]| 100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2);
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Powell),
            tol: Some(1.0e-6),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        for scale in [1.0, 1.0e-6, 1.0e-12] {
            let scaled = |x: &[f64]| scale * rosen(x);
            let result = powell(&scaled, &[-1.2, 1.0], options).expect("powell executes");
            let err = (result.x[0] - 1.0).abs().max((result.x[1] - 1.0).abs());
            assert!(
                result.success && err < 1.0e-4,
                "scale {scale}: x = {:?}, success {}, nit {}, nfev {}",
                result.x,
                result.success,
                result.nit,
                result.nfev
            );
        }
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
    fn powell_line_search_returns_best_sample_when_no_bracket_exists() {
        let fun = |x: &[f64]| x[0].exp();
        let mut objective = Objective::new(&fun, RuntimeMode::Strict, 5000);
        let search = powell_line_search(&mut objective, &[0.0], fun(&[0.0]), &[1.0], 1.0e-4, None)
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

    /// frankenscipy-de6qs: the golden-section search this replaced resolved alpha only to about
    /// 1e-4 absolute. Brent's parabolic steps land on a quadratic's minimum; the bounded
    /// searches stop at the active bound.
    #[test]
    fn powell_line_search_is_precise_like_scipy_brent() {
        let fun = |x: &[f64]| (x[0] - 0.3).powi(2) + (x[1] + 0.7).powi(2) + 1.0;
        let run = |direction: &[f64], bounds: Option<&[Bound]>| {
            let mut objective = Objective::new(&fun, RuntimeMode::Strict, 5000);
            powell_line_search(&mut objective, &[0.0, 0.0], 1.58, direction, 1.0e-4, bounds)
                .expect("line search")
        };
        let unbounded = run(&[1.0, 0.0], None);
        assert!((unbounded.alpha - 0.3).abs() < 1e-9, "{}", unbounded.alpha);
        // Scaled direction: the step is found in units of the direction.
        let scaled = run(&[0.0, -1.0e-3], None);
        assert!((scaled.alpha - 700.0).abs() < 1e-6, "{}", scaled.alpha);
        // Bounded: fminbound stops within about xatol = 1e-6 of the active bound.
        let two_sided: [Bound; 2] = [(Some(-1.0), Some(0.2)), (None, None)];
        let capped = run(&[1.0, 0.0], Some(&two_sided));
        assert!((capped.x[0] - 0.2).abs() < 5e-6, "{:?}", capped.x);
        let one_sided: [Bound; 2] = [(None, Some(0.2)), (None, None)];
        let capped = run(&[1.0, 0.0], Some(&one_sided));
        assert!((capped.x[0] - 0.2).abs() < 5e-6, "{:?}", capped.x);
        assert!(capped.x[0] <= 0.2);
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

    /// frankenscipy-3cu8u.1: the audit fingerprint covers `x0` and every option. It used to be
    /// a `Debug` string of `method`, `mode`, `x0`, `maxiter` and `maxfev`, so calls differing
    /// only in `tol` or `seed`, or in the payload of a NaN in `x0`, shared one fingerprint.
    #[test]
    fn audit_fingerprints_cover_every_input() {
        let fingerprint_of = |x0: &[f64], options: MinimizeOptions<'_>| {
            let ledger = crate::audit::sync_audit_ledger();
            let _ = super::minimize_with_audit(|_| f64::NAN, x0, options, &ledger);
            let guard = ledger.lock().expect("audit ledger lock");
            assert_eq!(guard.len(), 1, "one rejection, one event");
            guard.entries()[0].input_fingerprint.clone()
        };
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            mode: RuntimeMode::Hardened,
            ..MinimizeOptions::default()
        };

        // Hardened NaN objective: `Ok` with `NanEncountered`.
        let base = fingerprint_of(&[0.0, 1.0], options);
        assert!(base.starts_with("blake3:"), "{base}");
        assert_eq!(base, fingerprint_of(&[0.0, 1.0], options));
        assert_ne!(base, fingerprint_of(&[0.0, 2.0], options));
        let tighter = MinimizeOptions {
            tol: Some(1.0e-3),
            ..options
        };
        assert_ne!(base, fingerprint_of(&[0.0, 1.0], tighter));
        let seeded = MinimizeOptions {
            seed: Some(7),
            ..options
        };
        assert_ne!(base, fingerprint_of(&[0.0, 1.0], seeded));

        // Non-finite `x0`: `Err(NonFiniteInput)`; both render as `NaN` under `Debug`.
        let quiet = fingerprint_of(&[f64::NAN], options);
        assert_eq!(quiet, fingerprint_of(&[f64::NAN], options));
        let payload = f64::from_bits(f64::NAN.to_bits() ^ 1);
        assert!(payload.is_nan());
        assert_ne!(quiet, fingerprint_of(&[payload], options));
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
            gradient_eps: Some(0.0),
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
            let h_inv = result.hess_inv.as_ref().expect("hessian inverse is present").todense();
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

    // frankenscipy-szq1n.7: Nelder-Mead ignored bounds and returned the unconstrained optimum
    // under success = true. SciPy 1.17.1 clips every trial point:
    //   (x0-3)^2+(x1+1)^2, x0=[0.5,0.5], bounds [(0,2),(0,1)] -> x=[2,0], fun=2.0
    //   rosen, x0=[0,0], bounds [(None,0.5),(None,None)] -> x=[0.5,0.24999], fun=0.2500000152
    #[test]
    fn nelder_mead_honours_bounds_like_scipy() {
        static BOX: [Bound; 2] = [(Some(0.0), Some(2.0)), (Some(0.0), Some(1.0))];
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::NelderMead),
            bounds: Some(&BOX),
            ..MinimizeOptions::default()
        };
        let shifted = |x: &[f64]| (x[0] - 3.0).powi(2) + (x[1] + 1.0).powi(2);
        let r = minimize(shifted, &[0.5, 0.5], options).expect("bounded nelder-mead");
        println!("NM box: x={:?} fun={:?} success={}", r.x, r.fun, r.success);
        assert!(r.success, "{}", r.message);
        assert!(
            (r.x[0] - 2.0).abs() < 1e-6 && r.x[1].abs() < 1e-6,
            "x={:?}",
            r.x
        );
        assert!((r.fun.unwrap() - 2.0).abs() < 1e-8);

        static HALF: [Bound; 2] = [(None, Some(0.5)), (None, None)];
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::NelderMead),
            bounds: Some(&HALF),
            ..MinimizeOptions::default()
        };
        let rosen = |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
        let r = minimize(rosen, &[0.0, 0.0], options).expect("bounded rosenbrock");
        println!("NM rosen x0<=0.5: x={:?} fun={:?}", r.x, r.fun);
        assert!(r.x[0] <= 0.5, "infeasible x0 {}", r.x[0]);
        assert!(
            (r.x[0] - 0.5).abs() < 1e-4 && (r.x[1] - 0.25).abs() < 1e-3,
            "x={:?}",
            r.x
        );
        assert!((r.fun.unwrap() - 0.25).abs() < 1e-6);

        // An x0 outside the box is clipped in, as SciPy does (it warns).
        let r = minimize(
            shifted,
            &[5.0, -4.0],
            MinimizeOptions {
                method: Some(OptimizeMethod::NelderMead),
                bounds: Some(&BOX),
                ..MinimizeOptions::default()
            },
        )
        .expect("clipped start");
        assert!(
            r.x.iter()
                .zip(&BOX)
                .all(|(v, (lo, hi))| { *v >= lo.unwrap() && *v <= hi.unwrap() })
        );
    }

    // frankenscipy-szq1n.7: Powell ignored bounds as well. SciPy 1.17.1 (whose own bounded
    // Powell stops loosely, so the check is feasibility plus a near-optimal value):
    //   (x0-3)^2+(x1+1)^2, x0=[0.5,0.5], bounds [(0,2),(0,1)] -> x=[2, 6.6e-5], fun=2.00013
    //   rosen, x0=[0,0], bounds [(None,0.5),(None,None)] -> x=[0.5, 0.2515], fun=0.25023
    #[test]
    fn powell_honours_bounds() {
        static BOX: [Bound; 2] = [(Some(0.0), Some(2.0)), (Some(0.0), Some(1.0))];
        let shifted = |x: &[f64]| (x[0] - 3.0).powi(2) + (x[1] + 1.0).powi(2);
        let r = minimize(
            shifted,
            &[0.5, 0.5],
            MinimizeOptions {
                method: Some(OptimizeMethod::Powell),
                bounds: Some(&BOX),
                ..MinimizeOptions::default()
            },
        )
        .expect("bounded powell");
        println!(
            "Powell box: x={:?} fun={:?} success={}",
            r.x, r.fun, r.success
        );
        assert!(
            r.x.iter()
                .zip(&BOX)
                .all(|(v, (lo, hi))| *v >= lo.unwrap() && *v <= hi.unwrap()),
            "infeasible x={:?}",
            r.x
        );
        // The old code returned the infeasible [3, -1] (fun 0); the feasibility check catches it.
        assert!(r.fun.unwrap() <= 2.001, "fun={:?} (optimum 2.0)", r.fun);

        static HALF: [Bound; 2] = [(None, Some(0.5)), (None, None)];
        let rosen = |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
        let r = minimize(
            rosen,
            &[0.0, 0.0],
            MinimizeOptions {
                method: Some(OptimizeMethod::Powell),
                bounds: Some(&HALF),
                ..MinimizeOptions::default()
            },
        )
        .expect("one-sided bounded powell");
        println!("Powell rosen x0<=0.5: x={:?} fun={:?}", r.x, r.fun);
        assert!(r.x[0] <= 0.5, "infeasible x0 {}", r.x[0]);
        assert!(r.fun.unwrap() <= 0.251, "fun={:?} (optimum 0.25)", r.fun);
    }

    #[test]
    fn feasible_step_interval_matches_scipy_line_for_search() {
        let bounds = [(Some(0.0), Some(2.0)), (None, Some(1.0))];
        // x=[1,0.5], d=[1,-1]: x0 hits 2 at alpha=1 and 0 at alpha=-1; x1 hits 1 at alpha=-0.5.
        assert_eq!(
            feasible_step_interval(&[1.0, 0.5], &[1.0, -1.0], &bounds),
            (-0.5, 1.0)
        );
        // Unconstrained along d=[0,-1] in the downward direction of x1.
        assert_eq!(
            feasible_step_interval(&[1.0, 0.5], &[0.0, -1.0], &bounds),
            (-0.5, f64::INFINITY)
        );
    }

    // frankenscipy-szq1n.7: these kernels used to ignore bounds and return x = 3 for (x-3)^2 on
    // [0,2] under success = true. SLSQP now honours them (frankenscipy-1ksfv.1); trust-constr,
    // whose constrained algorithm is frankenscipy-1ksfv.2, still refuses bounds and constraints.
    #[test]
    fn slsqp_honours_bounds_and_trust_constr_refuses_them() {
        let bounds = [(Some(0.0), Some(2.0))];
        let quad = |x: &[f64]| (x[0] - 3.0).powi(2);
        let bounded = |method| MinimizeOptions {
            method: Some(method),
            bounds: Some(&bounds),
            ..MinimizeOptions::default()
        };
        let r = minimize(quad, &[1.0], bounded(OptimizeMethod::Slsqp)).expect("slsqp");
        assert!(r.success, "{}", r.message);
        assert!((r.x[0] - 2.0).abs() < 1e-12, "x = {:?}", r.x);

        let err = minimize(quad, &[1.0], bounded(OptimizeMethod::TrustConstr))
            .expect_err("trust-constr must refuse bounds");
        assert!(
            matches!(&err, OptError::InvalidArgument { detail } if detail.contains("bounds")),
            "{err:?}"
        );
        let constraints = [Constraint::ineq(|x: &[f64]| vec![2.0 - x[0]])];
        let err = minimize(
            quad,
            &[1.0],
            MinimizeOptions {
                method: Some(OptimizeMethod::TrustConstr),
                constraints: &constraints,
                ..MinimizeOptions::default()
            },
        )
        .expect_err("trust-constr must refuse constraints");
        assert!(
            matches!(&err, OptError::InvalidArgument { detail } if detail.contains("constraints")),
            "{err:?}"
        );
        for method in [OptimizeMethod::Slsqp, OptimizeMethod::TrustConstr] {
            let free = MinimizeOptions {
                method: Some(method),
                ..MinimizeOptions::default()
            };
            let r = minimize(quad, &[1.0], free).expect("unbounded still runs");
            assert!((r.x[0] - 3.0).abs() < 1e-4, "{method:?} x={:?}", r.x);
        }
    }

    // A method that cannot use constraints refuses them rather than returning the unconstrained
    // optimum (SciPy warns and drops them).
    #[test]
    fn unconstrained_methods_refuse_constraints() {
        let constraints = [Constraint::ineq(|x: &[f64]| vec![2.0 - x[0]])];
        for method in [
            OptimizeMethod::Bfgs,
            OptimizeMethod::NelderMead,
            OptimizeMethod::LBfgsB,
            OptimizeMethod::Powell,
        ] {
            let err = minimize(
                |x: &[f64]| (x[0] - 3.0).powi(2),
                &[1.0],
                MinimizeOptions {
                    method: Some(method),
                    constraints: &constraints,
                    ..MinimizeOptions::default()
                },
            )
            .expect_err("constraints must be refused");
            assert!(
                matches!(&err, OptError::InvalidArgument { detail } if detail.contains("constraints")),
                "{method:?}: {err:?}"
            );
        }
    }

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
            if method == OptimizeMethod::ConjugateGradient {
                // SciPy 1.17.1's CG reaches (1, 1) to 3.3e-6 here but, by forward differences,
                // cannot bring the gradient under gtol = 1e-6: status 2 after 37 iterations.
                assert_eq!(r.status, ConvergenceStatus::PrecisionLoss, "{}", r.message);
                assert_eq!(r.nit, 37);
            } else {
                assert!(r.success, "{method:?} should converge: {}", r.message);
            }
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
        fn sphere_gradient(x: &[f64]) -> Vec<f64> {
            x.iter().map(|v| 2.0 * v).collect()
        }
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::NewtonCg),
            tol: Some(1e-8),
            gradient: Some(sphere_gradient),
            ..MinimizeOptions::default()
        };
        let result = minimize(sphere, &[2.0, -3.0], options).expect("minimize");
        // SciPy 1.17.1 (jac given): success, (nit, nfev, njev, nhev) = (2, 2, 3, 0), x = 0.
        assert!(result.success, "should converge: {}", result.message);
        assert_eq!(
            (result.nit, result.nfev, result.njev, result.nhev),
            (2, 2, 3, 0)
        );
        assert_eq!(result.x, vec![0.0, 0.0]);

        // Without a gradient (fsci's extension; SciPy requires jac) the objective is
        // differenced forward and the run still converges.
        let fd = minimize(
            sphere,
            &[2.0, -3.0],
            MinimizeOptions {
                gradient: None,
                ..options
            },
        )
        .expect("minimize");
        assert!(fd.success, "{}", fd.message);
        assert!(fd.x.iter().all(|v| v.abs() < 1e-6), "x={:?}", fd.x);
    }

    #[test]
    fn newton_cg_rosenbrock() {
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::NewtonCg),
            tol: Some(1e-6),
            maxiter: Some(500),
            maxfev: Some(50_000),
            gradient: Some(rosenbrock_gradient),
            ..MinimizeOptions::default()
        };
        let result = minimize(rosenbrock, &[0.0, 0.0], options).expect("minimize");
        // SciPy 1.17.1: success with (nit, nfev, njev, nhev) = (31, 49, 112, 0) and
        // x = (0.9999999097801645, 0.9999998191974939) (Hessian products by differences).
        assert!(result.success, "should converge: {}", result.message);
        assert_eq!(
            (result.nit, result.nfev, result.njev, result.nhev),
            (31, 49, 112, 0)
        );
        assert!((result.x[0] - 0.999_999_909_780_164_5).abs() < 1e-12);
        assert!((result.x[1] - 0.999_999_819_197_493_9).abs() < 1e-12);
    }

    /// `scipy.optimize.minimize(rosen, [-1.2, 1], method='Newton-CG', jac=rosen_der, ...)` with
    /// each curvature source, SciPy 1.17.1 values: (nit, nfev, njev, nhev) exactly (SciPy's
    /// `hcalls` for nhev), x to the bit through `hessp` and differences, and to 1e-10 through the
    /// dense Hessian (SciPy's `A.dot(p)` is BLAS; it lands 1.3e-11 away).
    #[test]
    fn newton_cg_takes_scipys_path() {
        fn rosen_hess(x: &[f64]) -> Vec<Vec<f64>> {
            vec![
                vec![1200.0 * x[0] * x[0] - 400.0 * x[1] + 2.0, -400.0 * x[0]],
                vec![-400.0 * x[0], 200.0],
            ]
        }
        fn rosen_hessp(x: &[f64], p: &[f64]) -> Vec<f64> {
            vec![
                (1200.0 * x[0] * x[0] - 400.0 * x[1] + 2.0) * p[0] - 400.0 * x[0] * p[1],
                -400.0 * x[0] * p[0] + 200.0 * p[1],
            ]
        }
        type Case = (
            Option<HessFunc>,
            Option<HesspFunc>,
            [usize; 4],
            [f64; 2],
            f64,
        );
        let cases: [Case; 3] = [
            (
                Some(rosen_hess),
                None,
                [83, 105, 105, 83],
                [0.999_982_602_903_939_3, 0.999_965_136_510_466_7],
                1e-10,
            ),
            (
                None,
                Some(rosen_hessp),
                [83, 105, 105, 142],
                [0.999_982_602_890_679_6, 0.999_965_136_483_894_8],
                0.0,
            ),
            (
                None,
                None,
                [86, 106, 316, 0],
                [0.999_995_330_560_910_8, 0.999_990_642_483_231_3],
                0.0,
            ),
        ];
        for (hess, hessp, counts, want, tol) in cases {
            let result = minimize(
                rosenbrock,
                &[-1.2, 1.0],
                MinimizeOptions {
                    method: Some(OptimizeMethod::NewtonCg),
                    gradient: Some(rosenbrock_gradient),
                    hess,
                    hessp,
                    ..MinimizeOptions::default()
                },
            )
            .expect("newton-cg");
            assert!(result.success, "{}", result.message);
            assert_eq!(
                [result.nit, result.nfev, result.njev, result.nhev],
                counts,
                "hess={} hessp={}",
                hess.is_some(),
                hessp.is_some()
            );
            for (got, want) in result.x.iter().zip(want) {
                assert!((got - want).abs() <= tol, "x = {:?}", result.x);
            }
        }
    }

    #[test]
    fn newton_cg_1d_quadratic() {
        fn one_dim_quadratic_gradient(x: &[f64]) -> Vec<f64> {
            vec![2.0 * (x[0] - 1.5)]
        }
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::NewtonCg),
            tol: Some(1e-10),
            gradient: Some(one_dim_quadratic_gradient),
            ..MinimizeOptions::default()
        };
        let result = minimize(one_dim_quadratic, &[5.0], options).expect("minimize");
        // SciPy 1.17.1: success, (2, 2, 3, 0), x = 1.5 exactly.
        assert!(result.success, "should converge: {}", result.message);
        assert_eq!(
            (result.nit, result.nfev, result.njev, result.nhev),
            (2, 2, 3, 0)
        );
        assert_eq!(result.x, vec![1.5]);
    }

    fn quadratic_hessp(_x: &[f64], p: &[f64]) -> Vec<f64> {
        vec![4.0 * p[0]]
    }

    #[test]
    fn newton_cg_uses_hessian_product_api_under_tight_eval_budget() {
        fn quadratic_gradient(x: &[f64]) -> Vec<f64> {
            vec![4.0 * x[0]]
        }
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::NewtonCg),
            tol: Some(1e-10),
            maxiter: Some(8),
            maxfev: Some(10),
            gradient: Some(quadratic_gradient),
            hessp: Some(quadratic_hessp),
            ..MinimizeOptions::default()
        };
        let result = minimize(|x| 2.0 * x[0] * x[0], &[3.0], options).expect("minimize");
        assert!(
            result.success,
            "should converge via hessp: {}",
            result.message
        );
        // SciPy 1.17.1: (nit, nfev, njev, nhev) = (2, 2, 2, 1) — one product, then an exact
        // zero residual.
        assert_eq!(
            (result.nit, result.nfev, result.njev, result.nhev),
            (2, 2, 2, 1)
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

    // ── trust-ncg / dogleg / trust-exact with `hess` (SciPy's trust-region family) ──

    fn rosenbrock_hess(x: &[f64]) -> Vec<Vec<f64>> {
        vec![
            vec![1200.0 * x[0] * x[0] - 400.0 * x[1] + 2.0, -400.0 * x[0]],
            vec![-400.0 * x[0], 200.0],
        ]
    }

    fn rosenbrock_hessp(x: &[f64], p: &[f64]) -> Vec<f64> {
        vec![
            (1200.0 * x[0] * x[0] - 400.0 * x[1] + 2.0) * p[0] - 400.0 * x[0] * p[1],
            -400.0 * x[0] * p[0] + 200.0 * p[1],
        ]
    }

    /// f = x₀⁴/4 − x₀²/2 + x₁² + x₀x₁/10 + x₂⁴/20 − x₂²: its Hessian is indefinite at
    /// (0.1, 0.2, 0.3), where the trust-region methods must use negative curvature.
    fn indefinite_start(x: &[f64]) -> f64 {
        0.25 * x[0].powi(4) - 0.5 * x[0] * x[0]
            + x[1] * x[1]
            + 0.1 * x[0] * x[1]
            + 0.05 * x[2].powi(4)
            - x[2] * x[2]
    }

    fn indefinite_start_gradient(x: &[f64]) -> Vec<f64> {
        vec![
            x[0].powi(3) - x[0] + 0.1 * x[1],
            2.0 * x[1] + 0.1 * x[0],
            0.2 * x[2].powi(3) - 2.0 * x[2],
        ]
    }

    fn indefinite_start_hess(x: &[f64]) -> Vec<Vec<f64>> {
        vec![
            vec![3.0 * x[0] * x[0] - 1.0, 0.1, 0.0],
            vec![0.1, 2.0, 0.0],
            vec![0.0, 0.0, 0.6 * x[2] * x[2] - 2.0],
        ]
    }

    /// Every number below is live SciPy 1.17.1 (`minimize(rosen, [-1.2, 1],
    /// method=..., jac=rosen_der, hess=rosen_hess | hessp=rosen_hess_prod)`). The counters are
    /// the path: a transcription slip anywhere in the driver or a subproblem moves them.
    #[test]
    fn trust_region_family_takes_scipys_path_on_rosenbrock() {
        type Case = (
            OptimizeMethod,
            bool,
            (usize, usize, usize, usize),
            [f64; 2],
            f64,
        );
        let cases: [Case; 4] = [
            (
                OptimizeMethod::TrustNcg,
                false,
                (29, 30, 27, 26),
                [0.9999996957772002, 0.9999993903385656],
                9.269935987707732e-14,
            ),
            (
                OptimizeMethod::TrustNcg,
                true,
                (29, 30, 27, 82),
                [0.9999996957772002, 0.9999993903385656],
                9.269935987707732e-14,
            ),
            (
                OptimizeMethod::Dogleg,
                false,
                (23, 24, 21, 20),
                [0.9999983082930026, 0.99999659792968],
                2.89668909123374e-12,
            ),
            (
                OptimizeMethod::TrustExact,
                false,
                (25, 26, 23, 26),
                [0.9999999994467651, 0.9999999988770814],
                3.331252984229145e-19,
            ),
        ];
        for (method, use_hessp, counts, x_scipy, fun_scipy) in cases {
            let options = MinimizeOptions {
                method: Some(method),
                gradient: Some(rosenbrock_gradient),
                hess: (!use_hessp).then_some(rosenbrock_hess as HessFunc),
                hessp: use_hessp.then_some(rosenbrock_hessp as HesspFunc),
                ..MinimizeOptions::default()
            };
            let result = minimize(rosenbrock, &[-1.2, 1.0], options).expect("minimize");
            let label = format!("{method:?} hessp={use_hessp}");
            assert!(result.success, "{label}: {}", result.message);
            assert_eq!(
                (result.nit, result.nfev, result.njev, result.nhev),
                counts,
                "{label}: (nit, nfev, njev, nhev) differ from SciPy's"
            );
            for (ours, theirs) in result.x.iter().zip(x_scipy) {
                assert!(
                    (ours - theirs).abs() <= 1e-12,
                    "{label}: x = {:?}",
                    result.x
                );
            }
            let fun = result.fun.unwrap();
            assert!(
                (fun - fun_scipy).abs() <= 1e-20_f64.max(1e-9 * fun_scipy),
                "{label}: f = {fun:e}"
            );
        }
    }

    /// SciPy on the indefinite start: trust-ncg and trust-exact follow the negative curvature to
    /// the minimizer near (1.0025, −0.0501, √10); dogleg's Cholesky fails at iteration 0
    /// (status 3, "A linalg error occurred, such as a non-psd Hessian").
    #[test]
    fn trust_region_family_on_an_indefinite_hessian_matches_scipy() {
        let run = |method| {
            let options = MinimizeOptions {
                method: Some(method),
                gradient: Some(indefinite_start_gradient),
                hess: Some(indefinite_start_hess),
                ..MinimizeOptions::default()
            };
            minimize(indefinite_start, &[0.1, 0.2, 0.3], options).expect("minimize")
        };
        let ncg = run(OptimizeMethod::TrustNcg);
        assert!(ncg.success, "{}", ncg.message);
        assert_eq!((ncg.nit, ncg.nfev, ncg.njev, ncg.nhev), (10, 11, 9, 8));
        let exact = run(OptimizeMethod::TrustExact);
        assert!(exact.success, "{}", exact.message);
        assert_eq!(
            (exact.nit, exact.nfev, exact.njev, exact.nhev),
            (8, 9, 8, 9)
        );
        for (result, x_scipy, fun_scipy) in [
            (
                &ncg,
                [1.0024982177076796, -0.050124205636529244, 3.162277722018637],
                -5.252506249997705,
            ),
            (
                &exact,
                [1.0024969852080294, -0.05012484926040148, 3.1622776601683795],
                -5.252506249999989,
            ),
        ] {
            for (ours, theirs) in result.x.iter().zip(x_scipy) {
                assert!((ours - theirs).abs() <= 1e-12, "x = {:?}", result.x);
            }
            assert!((result.fun.unwrap() - fun_scipy).abs() <= 1e-13);
        }

        let dogleg = run(OptimizeMethod::Dogleg);
        assert!(!dogleg.success);
        assert_eq!(dogleg.status, ConvergenceStatus::LinAlgError);
        assert_eq!(
            (dogleg.nit, dogleg.nfev, dogleg.njev, dogleg.nhev),
            (0, 1, 1, 1)
        );
        assert_eq!(dogleg.x, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn trust_region_methods_refuse_missing_curvature_like_scipy() {
        let bare = |method| MinimizeOptions {
            method: Some(method),
            gradient: Some(rosenbrock_gradient),
            ..MinimizeOptions::default()
        };
        for method in [OptimizeMethod::TrustNcg, OptimizeMethod::Dogleg] {
            let outcome = minimize(rosenbrock, &[-1.2, 1.0], bare(method));
            assert!(
                matches!(outcome, Err(OptError::InvalidArgument { .. })),
                "{method:?} without a Hessian must be refused, got {outcome:?}"
            );
        }
        // hessp is enough for trust-ncg, not for dogleg (SciPy passes it to trust-ncg only).
        let with_hessp = |method| MinimizeOptions {
            hessp: Some(rosenbrock_hessp),
            ..bare(method)
        };
        assert!(
            minimize(
                rosenbrock,
                &[-1.2, 1.0],
                with_hessp(OptimizeMethod::TrustNcg)
            )
            .is_ok()
        );
        assert!(matches!(
            minimize(rosenbrock, &[-1.2, 1.0], with_hessp(OptimizeMethod::Dogleg)),
            Err(OptError::InvalidArgument { .. })
        ));
        // A malformed Hessian is an error, not an index panic.
        fn short_hess(_x: &[f64]) -> Vec<Vec<f64>> {
            vec![vec![1.0, 0.0]]
        }
        let result = minimize(
            rosenbrock,
            &[-1.2, 1.0],
            MinimizeOptions {
                hess: Some(short_hess),
                ..bare(OptimizeMethod::TrustExact)
            },
        )
        .expect("minimize returns");
        assert_eq!(result.status, ConvergenceStatus::InvalidInput);
    }

    static USER_GRADIENT_CALLS: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);

    fn counted_rosenbrock_gradient(x: &[f64]) -> Vec<f64> {
        USER_GRADIENT_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        rosenbrock_gradient(x)
    }

    /// Newton-CG, the BFGS-model trust-exact, L-BFGS-B, TNC and trust-constr used to
    /// finite-difference the gradient even when the caller supplied one, and Newton-CG
    /// ignored `hess`.
    #[test]
    fn gradient_methods_use_the_callers_derivatives() {
        for method in [OptimizeMethod::Tnc, OptimizeMethod::TrustConstr] {
            let options = MinimizeOptions {
                method: Some(method),
                gradient: Some(counted_rosenbrock_gradient),
                ..MinimizeOptions::default()
            };
            let before = USER_GRADIENT_CALLS.load(std::sync::atomic::Ordering::Relaxed);
            let result = minimize(rosenbrock, &[-1.2, 1.0], options).expect("minimize");
            let calls = USER_GRADIENT_CALLS.load(std::sync::atomic::Ordering::Relaxed) - before;
            assert!(
                calls > 0,
                "{method:?}: the caller's gradient was never called"
            );
            assert!(
                result.x.iter().all(|v| v.is_finite()),
                "{method:?}: {:?}",
                result.x
            );
        }
        for (method, hess) in [
            (OptimizeMethod::NewtonCg, Some(rosenbrock_hess as HessFunc)),
            (OptimizeMethod::NewtonCg, None),
            (OptimizeMethod::TrustExact, None),
            (OptimizeMethod::LBfgsB, None),
        ] {
            let options = MinimizeOptions {
                method: Some(method),
                gradient: Some(counted_rosenbrock_gradient),
                hess,
                tol: Some(1e-8),
                ..MinimizeOptions::default()
            };
            let before = USER_GRADIENT_CALLS.load(std::sync::atomic::Ordering::Relaxed);
            let result = minimize(rosenbrock, &[-1.2, 1.0], options).expect("minimize");
            let calls = USER_GRADIENT_CALLS.load(std::sync::atomic::Ordering::Relaxed) - before;
            let label = format!("{method:?} hess={}", hess.is_some());
            assert!(calls > 0, "{label}: the caller's gradient was never called");
            assert!(result.success, "{label}: {}", result.message);
            // L-BFGS-B at tol = 1e-8 stops on its relative-reduction test (ftol = tol) short of
            // 1e-6: SciPy 1.17.1 with this objective and gradient ends at
            // (0.9999989453492792, 0.9999980209057279), and fsci takes SciPy's path.
            let target = if method == OptimizeMethod::LBfgsB {
                [0.999_998_945_349_279_2, 0.999_998_020_905_727_9]
            } else {
                [1.0, 1.0]
            };
            let x_tol = if method == OptimizeMethod::LBfgsB {
                1e-9
            } else {
                1e-6
            };
            assert!(
                (result.x[0] - target[0]).abs() < x_tol && (result.x[1] - target[1]).abs() < x_tol,
                "{label}: x = {:?}",
                result.x
            );
            if hess.is_some() {
                // Products come from the Hessian: no finite-difference gradient differences,
                // so the objective is evaluated only by the line search (the finite-difference
                // path costs at least 7 evaluations per iteration here).
                assert!(result.nhev > 0, "{label}: hess never evaluated");
                assert!(
                    result.nfev <= 4 * result.nit + 4,
                    "{label}: nfev = {}",
                    result.nfev
                );
            }
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

    fn hs71(v: &[f64]) -> f64 {
        v[0] * v[3] * (v[0] + v[1] + v[2]) + v[2]
    }

    // frankenscipy-1ksfv.1: Hock–Schittkowski #71 (one inequality, one equality, 1 <= x <= 5)
    // against SciPy 1.17.1 `minimize(..., method='SLSQP')` from (1, 5, 5, 1):
    // x = [1, 4.7429961, 3.8211546, 1.3794077], fun = 17.01401725, success. With SciPy's
    // default eps the port takes SciPy's 5 iterations; analytic derivatives reach the same point.
    #[test]
    fn slsqp_solves_hock_schittkowski_71_like_scipy() {
        let want = [1.0, 4.742_996_1, 3.821_154_6, 1.379_407_7];
        let bounds = [(Some(1.0), Some(5.0)); 4];
        let fd = [
            Constraint::ineq(|v: &[f64]| vec![v[0] * v[1] * v[2] * v[3] - 25.0]),
            Constraint::eq(|v: &[f64]| vec![v.iter().map(|x| x * x).sum::<f64>() - 40.0]),
        ];
        let analytic = [
            Constraint::ineq(|v: &[f64]| vec![v[0] * v[1] * v[2] * v[3] - 25.0]).with_jac(
                |v: &[f64]| {
                    vec![vec![
                        v[1] * v[2] * v[3],
                        v[0] * v[2] * v[3],
                        v[0] * v[1] * v[3],
                        v[0] * v[1] * v[2],
                    ]]
                },
            ),
            Constraint::eq(|v: &[f64]| vec![v.iter().map(|x| x * x).sum::<f64>() - 40.0])
                .with_jac(|v: &[f64]| vec![v.iter().map(|x| 2.0 * x).collect()]),
        ];
        fn hs71_grad(v: &[f64]) -> Vec<f64> {
            vec![
                v[3] * (2.0 * v[0] + v[1] + v[2]),
                v[0] * v[3],
                v[0] * v[3] + 1.0,
                v[0] * (v[0] + v[1] + v[2]),
            ]
        }
        for (label, cons, gradient) in [
            ("finite differences", &fd, None),
            (
                "analytic",
                &analytic,
                Some(hs71_grad as fn(&[f64]) -> Vec<f64>),
            ),
        ] {
            let r = minimize(
                hs71,
                &[1.0, 5.0, 5.0, 1.0],
                MinimizeOptions {
                    method: Some(OptimizeMethod::Slsqp),
                    bounds: Some(&bounds),
                    constraints: cons,
                    gradient,
                    // SciPy's default `eps` (√ε) is SLSQP's default here too.
                    ..MinimizeOptions::default()
                },
            )
            .expect("slsqp");
            assert!(r.success, "{label}: {}", r.message);
            for (xi, wi) in r.x.iter().zip(want) {
                assert!((xi - wi).abs() < 1e-6, "{label}: x = {:?}", r.x);
            }
            assert!(
                (r.fun.expect("fun") - 17.014_017_25).abs() < 1e-7,
                "{label}"
            );
            // SLSQP's success test is Σ violation < ftol (1e-6); SciPy itself ends HS71 at a
            // violation of 8.226e-8.
            assert!(
                r.maxcv.expect("maxcv") <= 1e-6,
                "{label}: maxcv {:?}",
                r.maxcv
            );
            if gradient.is_none() {
                assert_eq!(r.nit, 5, "{label}: SciPy takes 5 iterations");
            }
        }
    }

    // SciPy converts LinearConstraint / NonlinearConstraint for SLSQP with
    // `new_constraint_to_old`; these are its results (SciPy 1.17.1, default method).
    #[test]
    fn slsqp_does_not_report_a_nan_constraint_as_satisfied() {
        // SciPy: minimize(f, [0, 0], method="SLSQP", constraints=[{"type": "ineq",
        // "fun": lambda x: nan}]) fails (status 4, "Inequality constraints incompatible").
        // maxcv used to come out 0 here, the NaN dropped by `max`.
        let f = |v: &[f64]| (v[0] - 1.0).powi(2) + (v[1] - 2.0).powi(2);
        let cons = [Constraint::ineq(|_: &[f64]| vec![f64::NAN])];
        let r = minimize(
            f,
            &[0.0, 0.0],
            MinimizeOptions {
                constraints: &cons,
                ..MinimizeOptions::default()
            },
        )
        .expect("slsqp");
        assert!(!r.success, "{r:?}");
        assert!(r.maxcv.is_some_and(f64::is_nan), "maxcv {:?}", r.maxcv);
        // A satisfied constraint still reports maxcv 0.
        let ok_cons = [Constraint::ineq(|v: &[f64]| vec![2.0 - v[0]])];
        let ok = minimize(
            f,
            &[0.0, 0.0],
            MinimizeOptions {
                constraints: &ok_cons,
                ..MinimizeOptions::default()
            },
        )
        .expect("slsqp");
        assert!(ok.success, "{ok:?}");
        assert_eq!(ok.maxcv, Some(0.0));
    }

    #[test]
    fn slsqp_converts_linear_and_nonlinear_constraints_like_scipy() {
        let sphere2 = |v: &[f64]| v[0] * v[0] + v[1] * v[1];
        fn run(f: &dyn Fn(&[f64]) -> f64, x0: &[f64], cons: &[Constraint<'_>]) -> OptimizeResult {
            minimize(
                f,
                x0,
                MinimizeOptions {
                    constraints: cons,
                    ..MinimizeOptions::default()
                },
            )
            .expect("slsqp")
        }
        // lb == ub: one equality constraint.
        let lin_eq = LinearConstraint::new(vec![vec![1.0, 1.0]], vec![1.0], vec![1.0]).unwrap();
        let r = run(&sphere2, &[2.0, -1.0], &Constraint::from_linear(&lin_eq));
        assert!(r.success);
        assert!(
            (r.x[0] - 0.5).abs() < 1e-8 && (r.x[1] - 0.5).abs() < 1e-8,
            "{:?}",
            r.x
        );
        // Two-sided and one-sided rows: 0.5 <= x <= 0.8, y <= 0.3 -> (0.8, 0.3).
        let lin_box = LinearConstraint::new(
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![0.5, f64::NEG_INFINITY],
            vec![0.8, 0.3],
        )
        .unwrap();
        let converted = Constraint::from_linear(&lin_box);
        assert_eq!(
            converted.len(),
            1,
            "no lb == ub row: a single inequality constraint"
        );
        assert_eq!((converted[0].fun)(&[0.6, 0.1]).len(), 3);
        let r = run(
            &|v: &[f64]| (v[0] - 1.0).powi(2) + (v[1] - 1.0).powi(2),
            &[0.0, 0.0],
            &converted,
        );
        assert!(r.success);
        assert!(
            (r.x[0] - 0.8).abs() < 1e-8 && (r.x[1] - 0.3).abs() < 1e-8,
            "{:?}",
            r.x
        );
        assert!((r.fun.unwrap() - 0.53).abs() < 1e-10);
        // NonlinearConstraint x^2 + y^2 <= 1 on -xy -> (1/sqrt2, 1/sqrt2), fun -1/2.
        fn disc(v: &[f64]) -> Vec<f64> {
            vec![v[0] * v[0] + v[1] * v[1]]
        }
        let nl = NonlinearConstraint::new(disc, vec![f64::NEG_INFINITY], vec![1.0]).unwrap();
        let r = run(
            &|v: &[f64]| -(v[0] * v[1]),
            &[0.5, 0.5],
            &Constraint::from_nonlinear(&nl),
        );
        assert!(r.success);
        let h = std::f64::consts::FRAC_1_SQRT_2;
        assert!(
            (r.x[0] - h).abs() < 1e-6 && (r.x[1] - h).abs() < 1e-6,
            "{:?}",
            r.x
        );
        assert!((r.fun.unwrap() + 0.5).abs() < 1e-8);
    }

    // SciPy exit modes carry through: maxiter=1 is "Iteration limit reached" (mode 9, never
    // success); x >= 1 and x <= 0 together end in mode 8 "Positive directional derivative for
    // linesearch" with the point infeasible (SciPy 1.17.1: x = 0.5, 19 iterations).
    #[test]
    fn slsqp_reports_scipy_exit_modes_without_claiming_success() {
        let constraints = [Constraint::ineq(|v: &[f64]| vec![2.0 - v[0]])];
        let r = minimize(
            |v: &[f64]| (v[0] - 3.0).powi(2),
            &[1.0],
            MinimizeOptions {
                constraints: &constraints,
                maxiter: Some(1),
                ..MinimizeOptions::default()
            },
        )
        .expect("slsqp");
        assert!(!r.success);
        assert_eq!(r.status, ConvergenceStatus::MaxIterations);
        assert_eq!(r.message, "Iteration limit reached");
        assert_eq!(r.nit, 1);

        let incompatible = [
            Constraint::ineq(|v: &[f64]| vec![v[0] - 1.0]),
            Constraint::ineq(|v: &[f64]| vec![-v[0]]),
        ];
        let r = minimize(
            |v: &[f64]| v[0] * v[0],
            &[0.5],
            MinimizeOptions {
                constraints: &incompatible,
                ..MinimizeOptions::default()
            },
        )
        .expect("slsqp");
        assert!(!r.success, "an infeasible problem must not report success");
        assert_eq!(r.message, "Positive directional derivative for linesearch");
        assert!(
            r.maxcv.expect("maxcv") >= 0.5 - 1e-12,
            "maxcv {:?}",
            r.maxcv
        );
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

    #[test]
    fn test_minimize_with_casp_portfolio_smooth_convex_routes_to_bfgs() {
        let mut portfolio = OptSolverPortfolio::new(RuntimeMode::Strict, 16);
        let res = minimize_with_casp_portfolio(
            convex_bowl,
            &[1.0, 1.0],
            MinimizeOptions::default(),
            &mut portfolio,
            false,
            false,
        )
        .expect("portfolio minimize");

        assert_eq!(res.chosen_action, OptSolverAction::BFGS);
        assert!(res.result.success);
        assert_eq!(portfolio.evidence_len(), 1);
    }

    #[test]
    fn test_minimize_with_casp_portfolio_noisy_routes_to_nelder_mead() {
        let mut portfolio = OptSolverPortfolio::new(RuntimeMode::Strict, 16);
        let res = minimize_with_casp_portfolio(
            convex_bowl,
            &[1.0, 1.0],
            MinimizeOptions::default(),
            &mut portfolio,
            true,
            false,
        )
        .expect("portfolio minimize");

        assert_eq!(res.chosen_action, OptSolverAction::NelderMead);
        assert!(res.result.success);
        assert_eq!(portfolio.evidence_len(), 1);
    }

    #[test]
    fn test_minimize_with_casp_portfolio_multimodal_routes_to_direct() {
        let mut portfolio = OptSolverPortfolio::new(RuntimeMode::Strict, 16);
        let res = minimize_with_casp_portfolio(
            convex_bowl,
            &[1.0, 1.0],
            MinimizeOptions::default(),
            &mut portfolio,
            false,
            true,
        )
        .expect("portfolio minimize");

        assert_eq!(res.chosen_action, OptSolverAction::DIRECT);
        assert!(res.result.success);
        assert_eq!(portfolio.evidence_len(), 1);
    }

    // br-szq1n.9: expected optima are live SciPy 1.17.1
    // `minimize(..., method='L-BFGS-B', bounds=...)` results (and analytic).
    #[test]
    fn test_minimize_with_casp_honours_bounds_on_a_smooth_problem() {
        static BOUNDS: [crate::types::Bound; 1] = [(Some(-1.0), Some(1.0))];
        let mut portfolio = OptSolverPortfolio::new(RuntimeMode::Strict, 16);
        let options = MinimizeOptions {
            bounds: Some(&BOUNDS),
            ..MinimizeOptions::default()
        };
        let res = minimize_with_casp(
            |x: &[f64]| (x[0] - 5.0).powi(2),
            &[0.0],
            options,
            &mut portfolio,
        )
        .expect("bounded casp minimize");
        // The unconstrained argmin is BFGS, which ignores bounds and returned x = 5.
        assert_eq!(res.chosen_action, OptSolverAction::LBFGSB);
        assert!(
            (res.result.x[0] - 1.0).abs() <= 1e-8,
            "x = {:?}",
            res.result.x
        );
    }

    /// frankenscipy-szq1n.9: bounded L-BFGS-B built its step from the full gradient and
    /// projected it afterwards, with Armijo steps only. All three cases below failed: an
    /// active upper bound stopped at [0.5, 0.2519] ("maximum iterations"), a wide box stalled
    /// at f = 3.47, and an active lower bound accepted no step at f = 0.269. SciPy 1.17.1
    /// L-BFGS-B: [0.5, 0.25]; [0.99999697, 0.99999396] (f = 9.2e-12); [1.5, 2.25].
    #[test]
    fn lbfgsb_converges_with_active_and_inactive_bounds_like_scipy() {
        use crate::minimize::lbfgsb;
        let rosen = |x: &[f64]| 100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2);
        let cases: [(&[f64], [Bound; 2], [f64; 2]); 3] = [
            (
                &[-1.0, -1.0],
                [(Some(-2.0), Some(0.5)), (Some(-2.0), Some(0.5))],
                [0.5, 0.25],
            ),
            (
                &[-1.2, 1.0],
                [(Some(-5.0), Some(5.0)), (Some(-5.0), Some(5.0))],
                [1.0, 1.0],
            ),
            (
                &[2.0, 2.0],
                [(Some(1.5), Some(5.0)), (Some(-5.0), Some(5.0))],
                [1.5, 2.25],
            ),
        ];
        for (x0, bounds, expected) in cases {
            let r = lbfgsb(&rosen, x0, MinimizeOptions::default(), Some(&bounds)).expect("lbfgsb");
            assert!(r.success, "x0 {x0:?}: {} at {:?}", r.message, r.x);
            for (got, want) in r.x.iter().zip(expected) {
                assert!((got - want).abs() < 1e-5, "x0 {x0:?}: x = {:?}", r.x);
            }
            assert!(
                r.x.iter()
                    .zip(&bounds)
                    .all(|(v, (lo, hi))| *v >= lo.unwrap() && *v <= hi.unwrap()),
                "infeasible {:?}",
                r.x
            );
        }
    }

    /// frankenscipy-1ksfv.18: `lbfgsb` is SciPy 1.17.1's L-BFGS-B 3.0 to the evaluation. Each
    /// pinned result is `minimize(method='L-BFGS-B')` on the same explicit-operation objective,
    /// bit-identical under OPENBLAS_CORETYPE = Prescott / Nehalem / Sandybridge / Haswell / Zen,
    /// and fsci must reproduce x to the bit and (nit, nfev, njev) and the message exactly: a
    /// bounded Rosenbrock stopping on the projected gradient, lower bounds only, Powell's badly
    /// scaled function stopping on the relative reduction, and a `maxfev` limit that SciPy checks
    /// only between iterations (22 evaluations against a limit of 20).
    #[test]
    fn lbfgsb_takes_scipys_path() {
        use crate::minimize::lbfgsb;
        fn rosen(x: &[f64]) -> f64 {
            let mut s = 0.0;
            for i in 0..x.len() - 1 {
                let a = x[i + 1] - x[i] * x[i];
                let b = 1.0 - x[i];
                s += 100.0 * a * a + b * b;
            }
            s
        }
        fn rosen_der(x: &[f64]) -> Vec<f64> {
            let n = x.len();
            let mut g = vec![0.0; n];
            g[0] = -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]);
            for i in 1..n - 1 {
                g[i] = 200.0 * (x[i] - x[i - 1] * x[i - 1])
                    - 400.0 * x[i] * (x[i + 1] - x[i] * x[i])
                    - 2.0 * (1.0 - x[i]);
            }
            g[n - 1] = 200.0 * (x[n - 1] - x[n - 2] * x[n - 2]);
            g
        }
        fn quad(x: &[f64]) -> f64 {
            let mut s = 0.0;
            for (i, &xi) in x.iter().enumerate() {
                let c = if i.is_multiple_of(2) {
                    (i + 1) as f64
                } else {
                    -((i + 1) as f64)
                };
                let d = xi - c;
                s += ((i + 1) as f64) * d * d;
            }
            s
        }
        fn quad_der(x: &[f64]) -> Vec<f64> {
            x.iter()
                .enumerate()
                .map(|(i, &xi)| {
                    let c = if i.is_multiple_of(2) {
                        (i + 1) as f64
                    } else {
                        -((i + 1) as f64)
                    };
                    2.0 * ((i + 1) as f64) * (xi - c)
                })
                .collect()
        }
        fn powell_bs(x: &[f64]) -> f64 {
            let f1 = 10000.0 * x[0] * x[1] - 1.0;
            let f2 = (-x[0]).exp() + (-x[1]).exp() - 1.0001;
            f1 * f1 + f2 * f2
        }
        fn powell_bs_der(x: &[f64]) -> Vec<f64> {
            let f1 = 10000.0 * x[0] * x[1] - 1.0;
            let f2 = (-x[0]).exp() + (-x[1]).exp() - 1.0001;
            vec![
                2.0 * f1 * 10000.0 * x[1] - 2.0 * f2 * (-x[0]).exp(),
                2.0 * f1 * 10000.0 * x[0] - 2.0 * f2 * (-x[1]).exp(),
            ]
        }
        let alternating: Vec<f64> = (0..10)
            .map(|i| if i % 2 == 0 { -1.2 } else { 1.0 })
            .collect();
        let options = |gradient: Option<GradientFunc>, maxfev: Option<usize>| MinimizeOptions {
            method: Some(OptimizeMethod::LBfgsB),
            gradient,
            maxfev,
            ..MinimizeOptions::default()
        };
        let check = |label: &str,
                     result: OptimizeResult,
                     x: &[f64],
                     counts: (usize, usize, usize),
                     message: &str| {
            let got: Vec<u64> = result.x.iter().map(|v| v.to_bits()).collect();
            let want: Vec<u64> = x.iter().map(|v| v.to_bits()).collect();
            assert_eq!(got, want, "{label}: x = {:?}", result.x);
            assert_eq!((result.nit, result.nfev, result.njev), counts, "{label}");
            assert_eq!(result.message, message, "{label}");
        };

        let r = lbfgsb(
            &rosen,
            &[1.3, 0.7, 0.8, 1.9, 1.2],
            options(Some(rosen_der), None),
            Some(&[(Some(-2.0), Some(0.5)); 5]),
        )
        .expect("bounded rosen");
        assert_eq!(r.status, ConvergenceStatus::Success);
        check(
            "rosen5 in [-2, 0.5]",
            r,
            &[
                0.5,
                0.263_037_383_182_917_64,
                0.079_962_310_584_494_13,
                0.016_231_644_140_291_21,
                0.000_263_455_983_679_337_8,
            ],
            (12, 14, 14),
            "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL",
        );

        let r = lbfgsb(
            &quad,
            &[1.0; 8],
            options(Some(quad_der), None),
            Some(&[(Some(0.0), None); 8]),
        )
        .expect("lower-bounded quadratic");
        check(
            "quad8 >= 0",
            r,
            &[
                1.0,
                0.0,
                3.000_000_171_554_962_2,
                0.0,
                5.000_000_214_014_39,
                0.0,
                7.000_000_199_958_738_5,
                0.0,
            ],
            (7, 8, 8),
            "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL",
        );

        let r = lbfgsb(
            &powell_bs,
            &[0.0, 1.0],
            options(Some(powell_bs_der), None),
            None,
        )
        .expect("powell badly scaled");
        check(
            "powell badly scaled",
            r,
            &[0.000_100_003_676_309_729_37, 1.000_000_002_705_369_5],
            (2, 4, 4),
            "CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH",
        );

        // The evaluation cap is SciPy's: the first iteration finishes (22 evaluations with the
        // forward difference) before `nfev > maxfun` stops the run between iterations.
        let r = lbfgsb(&rosen, &alternating, options(None, Some(20)), None).expect("maxfev");
        assert_eq!(r.status, ConvergenceStatus::MaxEvaluations);
        assert!(!r.success);
        check(
            "rosen10 maxfev 20",
            r,
            &[
                -1.095_816_581_571_935,
                0.617_285_394_517_458_5,
                -0.883_197_361_187_090_2,
                0.617_285_394_517_458_5,
                -0.883_197_361_187_090_2,
                0.617_285_394_517_458_5,
                -0.883_197_339_212_537_8,
                0.617_285_394_517_458_5,
                -0.883_197_361_187_090_2,
                1.042_523_835_287_148,
            ],
            (1, 22, 2),
            "STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT",
        );
    }

    #[test]
    fn test_minimize_with_casp_bounded_rosenbrock_matches_scipy() {
        static BOUNDS: [crate::types::Bound; 2] =
            [(Some(-2.0), Some(0.5)), (Some(-2.0), Some(0.5))];
        let rosen = |x: &[f64]| 100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2);
        let mut portfolio = OptSolverPortfolio::new(RuntimeMode::Strict, 16);
        let options = MinimizeOptions {
            bounds: Some(&BOUNDS),
            ..MinimizeOptions::default()
        };
        let res = minimize_with_casp(rosen, &[-1.0, -1.0], options, &mut portfolio)
            .expect("bounded rosenbrock");
        // SciPy: x* = [0.5, 0.25], fun = 0.25 (the x <= 0.5 bound is active).
        assert!(
            (res.result.x[0] - 0.5).abs() <= 1e-6,
            "x = {:?}",
            res.result.x
        );
        assert!(
            (res.result.x[1] - 0.25).abs() <= 1e-6,
            "x = {:?}",
            res.result.x
        );
    }

    #[test]
    fn test_minimize_with_casp_results_are_always_feasible() {
        // 200 seeded convex quadratics whose unconstrained minimum is usually OUTSIDE
        // the box: every CASP answer must lie inside it.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        for case in 0..200 {
            let dim = 1 + case % 3;
            let bounds: Vec<crate::types::Bound> = (0..dim)
                .map(|_| {
                    let lo = -2.0 + 2.0 * next();
                    (Some(lo), Some(lo + 0.5 + next()))
                })
                .collect();
            let bounds: &'static [crate::types::Bound] = Box::leak(bounds.into_boxed_slice());
            let center: Vec<f64> = (0..dim).map(|_| -6.0 + 12.0 * next()).collect();
            let x0: Vec<f64> = bounds.iter().map(|b| b.0.unwrap()).collect();
            let mut portfolio = OptSolverPortfolio::new(RuntimeMode::Strict, 4);
            let options = MinimizeOptions {
                bounds: Some(bounds),
                ..MinimizeOptions::default()
            };
            let res = minimize_with_casp(
                |x: &[f64]| x.iter().zip(&center).map(|(a, c)| (a - c).powi(2)).sum(),
                &x0,
                options,
                &mut portfolio,
            )
            .expect("bounded casp minimize");
            for (i, (&xi, b)) in res.result.x.iter().zip(bounds).enumerate() {
                let (lo, hi) = (b.0.unwrap(), b.1.unwrap());
                assert!(
                    xi >= lo - 1e-12 && xi <= hi + 1e-12,
                    "case {case}: x[{i}] = {xi} outside [{lo}, {hi}] via {:?}",
                    res.chosen_action
                );
            }
        }
    }

    #[test]
    fn test_minimize_with_casp_standard_entrypoint() {
        let mut portfolio = OptSolverPortfolio::new(RuntimeMode::Strict, 16);
        let res = minimize_with_casp(
            convex_bowl,
            &[1.0, 1.0],
            MinimizeOptions::default(),
            &mut portfolio,
        )
        .expect("minimize_with_casp");

        assert_eq!(res.chosen_action, OptSolverAction::BFGS);
        assert!(res.result.success);
        assert!((res.result.x[0] - 2.0).abs() < 1e-4);
        assert!((res.result.x[1] + 3.0).abs() < 1e-4);
        assert_eq!(portfolio.evidence_len(), 1);
    }

    /// frankenscipy-6ycp2: SciPy's per-method options, each row against SciPy 1.17.1 running
    /// `minimize(rosen, x0, jac=rosen_der, method=..., tol=..., options={...})` (Nelder-Mead and
    /// Powell without `jac`), pinned interpreter, numpy 2.4.3. Every option row differs from its
    /// method's default row in SciPy and here (must-differ); `tol` fills only the unset options.
    /// (nit, nfev) must be SciPy's exactly. x agrees to 1e-7: Nelder-Mead, Powell and CG land
    /// on SciPy's x bit for bit, while L-BFGS-B and BFGS differ in the last digits of the
    /// arithmetic that SciPy itself moves by as much between OpenBLAS kernels (the live
    /// `diff_opt_*` rows use the same x tolerance).
    #[test]
    fn per_method_options_take_scipys_path() {
        use crate::MinimizeMethodOptions;
        fn rosen(x: &[f64]) -> f64 {
            (0..x.len() - 1)
                .map(|i| 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2))
                .sum()
        }
        fn rosen_der(x: &[f64]) -> Vec<f64> {
            let n = x.len();
            let mut d = vec![0.0; n];
            for i in 1..n - 1 {
                d[i] = 200.0 * (x[i] - x[i - 1] * x[i - 1])
                    - 400.0 * (x[i + 1] - x[i] * x[i]) * x[i]
                    - 2.0 * (1.0 - x[i]);
            }
            d[0] = -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]);
            d[n - 1] = 200.0 * (x[n - 1] - x[n - 2] * x[n - 2]);
            d
        }
        let x5 = [-1.2, 1.0, -1.2, 1.0, -1.2];
        let x4 = [-1.2, 1.0, -1.2, 1.0];
        let x3 = [-1.2, 1.0, 0.5];
        let simplex = vec![
            vec![-1.2, 1.0, 0.5],
            vec![-1.0, 1.0, 0.5],
            vec![-1.2, 1.2, 0.5],
            vec![-1.2, 1.0, 0.7],
        ];
        let direc = vec![
            vec![0.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 0.0, 0.0],
        ];
        let none = MinimizeMethodOptions::default();
        type Row<'a> = (
            &'static str,
            OptimizeMethod,
            &'a [f64],
            MinimizeMethodOptions<'a>,
            Option<f64>,
            (usize, usize),
            Vec<f64>,
        );
        let rows: Vec<Row<'_>> = vec![
            (
                "lbfgsb/default",
                OptimizeMethod::LBfgsB,
                &x5,
                none,
                None,
                (49, 66),
                vec![
                    1.0000002716142677,
                    1.000000476026837,
                    1.0000009965844259,
                    1.0000018855760335,
                    1.0000034963341937,
                ],
            ),
            (
                "lbfgsb/maxcor=3",
                OptimizeMethod::LBfgsB,
                &x5,
                MinimizeMethodOptions {
                    maxcor: Some(3),
                    ..none
                },
                None,
                (58, 75),
                vec![
                    0.9999996963852181,
                    0.9999992014580509,
                    0.9999977907921979,
                    0.9999962772844899,
                    0.9999915868868214,
                ],
            ),
            (
                "lbfgsb/maxls=2",
                OptimizeMethod::LBfgsB,
                &x5,
                MinimizeMethodOptions {
                    maxls: Some(2),
                    ..none
                },
                None,
                (58, 75),
                vec![
                    0.9999999937972197,
                    0.9999999925159285,
                    0.999999988884144,
                    0.999999980832507,
                    0.9999999724375375,
                ],
            ),
            (
                "lbfgsb/maxls=3",
                OptimizeMethod::LBfgsB,
                &x5,
                MinimizeMethodOptions {
                    maxls: Some(3),
                    ..none
                },
                None,
                (46, 67),
                vec![
                    1.0000006967978263,
                    1.0000009980473656,
                    1.0000015563607503,
                    1.0000034609928912,
                    1.0000068963849624,
                ],
            ),
            (
                "lbfgsb/gtol=0.1",
                OptimizeMethod::LBfgsB,
                &x5,
                MinimizeMethodOptions {
                    gtol: Some(0.1),
                    ..none
                },
                None,
                (45, 62),
                vec![
                    0.9999028732430887,
                    0.9998223816502801,
                    0.9996087746157547,
                    0.9992252427339321,
                    0.9984663374275495,
                ],
            ),
            (
                "lbfgsb/ftol=1e-4",
                OptimizeMethod::LBfgsB,
                &x5,
                MinimizeMethodOptions {
                    ftol: Some(1e-4),
                    ..none
                },
                None,
                (46, 63),
                vec![
                    1.0000200960523424,
                    1.0000594676253218,
                    1.0001004093079977,
                    1.0001680582101513,
                    1.0002674513822452,
                ],
            ),
            (
                // tol fills ftol; the explicit gtol wins over it.
                "lbfgsb/tol=1e-4,gtol=1e-9",
                OptimizeMethod::LBfgsB,
                &x5,
                MinimizeMethodOptions {
                    gtol: Some(1e-9),
                    ..none
                },
                Some(1e-4),
                (46, 63),
                vec![
                    1.0000200960523424,
                    1.0000594676253218,
                    1.0001004093079977,
                    1.0001680582101513,
                    1.0002674513822452,
                ],
            ),
            (
                "bfgs5/default",
                OptimizeMethod::Bfgs,
                &x5,
                none,
                None,
                (49, 60),
                vec![
                    0.9999999920157406,
                    0.9999999890213155,
                    0.9999999860453902,
                    0.9999999540064021,
                    0.9999999189310969,
                ],
            ),
            (
                "bfgs5/gtol=0.1",
                OptimizeMethod::Bfgs,
                &x5,
                MinimizeMethodOptions {
                    gtol: Some(0.1),
                    ..none
                },
                None,
                (44, 55),
                vec![
                    0.999814761308239,
                    0.999679921781256,
                    0.9991663802602391,
                    0.998318522772607,
                    0.9965395433960469,
                ],
            ),
            (
                "bfgs5/norm=2",
                OptimizeMethod::Bfgs,
                &x5,
                MinimizeMethodOptions {
                    norm: Some(2.0),
                    ..none
                },
                None,
                (50, 61),
                vec![
                    1.0000000001088962,
                    1.000000000283706,
                    1.0000000004844876,
                    1.0000000008672074,
                    1.0000000016702875,
                ],
            ),
            (
                "bfgs5/xrtol=1e-3",
                OptimizeMethod::Bfgs,
                &x5,
                MinimizeMethodOptions {
                    xrtol: Some(1e-3),
                    ..none
                },
                None,
                (46, 57),
                vec![
                    0.9999876361859859,
                    0.999989753822751,
                    0.9999791482727433,
                    0.9999697708313129,
                    0.9999344723508423,
                ],
            ),
            (
                "bfgs4/default",
                OptimizeMethod::Bfgs,
                &x4,
                none,
                None,
                (38, 47),
                vec![
                    0.9999999127358445,
                    0.9999998321105728,
                    0.999999686510974,
                    0.9999993800315787,
                ],
            ),
            (
                "bfgs4/c2=0.5",
                OptimizeMethod::Bfgs,
                &x4,
                MinimizeMethodOptions {
                    c2: Some(0.5),
                    ..none
                },
                None,
                (34, 48),
                vec![
                    0.9999999573089223,
                    0.9999999190622244,
                    0.9999998486148944,
                    0.999999689279727,
                ],
            ),
            (
                "cg4/default",
                OptimizeMethod::ConjugateGradient,
                &x4,
                none,
                None,
                (100, 182),
                vec![
                    1.0000008314568956,
                    1.0000016662529765,
                    1.0000033408989533,
                    1.000006698459287,
                ],
            ),
            (
                "cg4/gtol=1e-3",
                OptimizeMethod::ConjugateGradient,
                &x4,
                MinimizeMethodOptions {
                    gtol: Some(1e-3),
                    ..none
                },
                None,
                (69, 132),
                vec![
                    0.9998979459958574,
                    0.999794601969902,
                    0.999585730941003,
                    0.9991685645231335,
                ],
            ),
            (
                "nm/default",
                OptimizeMethod::NelderMead,
                &x3,
                none,
                None,
                (208, 373),
                vec![1.0000160542863308, 1.0000294440802586, 1.0000624769428637],
            ),
            (
                "nm/adaptive",
                OptimizeMethod::NelderMead,
                &x3,
                MinimizeMethodOptions {
                    adaptive: Some(true),
                    ..none
                },
                None,
                (250, 437),
                vec![0.9999994511501469, 1.0000016029593437, 1.0000012879107845],
            ),
            (
                "nm/initial_simplex",
                OptimizeMethod::NelderMead,
                &x3,
                MinimizeMethodOptions {
                    initial_simplex: Some(&simplex),
                    ..none
                },
                None,
                (205, 361),
                vec![1.0000018670271915, 1.0000050320049534, 1.0000095411892593],
            ),
            (
                "nm/xatol=fatol=1e-2",
                OptimizeMethod::NelderMead,
                &x3,
                MinimizeMethodOptions {
                    xatol: Some(1e-2),
                    fatol: Some(1e-2),
                    ..none
                },
                None,
                (28, 50),
                vec![-0.8842530566339224, 0.7952657856053376, 0.6374529897190208],
            ),
            (
                "powell/default",
                OptimizeMethod::Powell,
                &x3,
                none,
                None,
                (33, 1155),
                vec![1.0000000000000402, 1.0000000000000722, 1.0000000000001463],
            ),
            (
                "powell/direc",
                OptimizeMethod::Powell,
                &x3,
                MinimizeMethodOptions {
                    direc: Some(&direc),
                    ..none
                },
                None,
                (28, 983),
                vec![0.9999999999999847, 0.9999999999999516, 0.9999999999998933],
            ),
            (
                "powell/xtol=1e-2",
                OptimizeMethod::Powell,
                &x3,
                MinimizeMethodOptions {
                    xtol: Some(1e-2),
                    ..none
                },
                None,
                (27, 641),
                vec![1.000000000000124, 1.000000000000257, 1.0000000000005351],
            ),
            (
                "powell/ftol=1e-3",
                OptimizeMethod::Powell,
                &x3,
                MinimizeMethodOptions {
                    ftol: Some(1e-3),
                    ..none
                },
                None,
                (3, 98),
                vec![-0.8802148388065856, 0.7855771312603995, 0.6210472979630902],
            ),
        ];

        let mut mismatches = Vec::new();
        let mut defaults: Vec<(&str, (usize, usize, Vec<f64>))> = Vec::new();
        for (name, method, x0, method_options, tol, (nit, nfev), x) in &rows {
            let gradient = matches!(
                method,
                OptimizeMethod::LBfgsB | OptimizeMethod::Bfgs | OptimizeMethod::ConjugateGradient
            )
            .then_some(rosen_der as GradientFunc);
            let options = MinimizeOptions {
                method: Some(*method),
                tol: *tol,
                gradient,
                method_options: *method_options,
                ..MinimizeOptions::default()
            };
            let result = minimize(rosen, x0, options).expect(name);
            let x_error = result
                .x
                .iter()
                .zip(x)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            println!(
                "{name}: fsci (nit, nfev) = ({}, {}), SciPy ({nit}, {nfev}); max |x - x_scipy| = {x_error:e}",
                result.nit, result.nfev
            );
            if (result.nit, result.nfev) != (*nit, *nfev) || x_error > 1e-7 {
                mismatches.push(*name);
            }
            let family = name.split('/').next().expect("family");
            let run = (result.nit, result.nfev, result.x.clone());
            if name.ends_with("/default") {
                defaults.push((family, run));
            } else if let Some((_, default)) = defaults.iter().find(|(f, _)| *f == family) {
                assert_ne!(&run, default, "{name}: the option did not change the run");
            }
        }
        assert!(
            mismatches.is_empty(),
            "rows off SciPy's path: {mismatches:?}"
        );
    }

    /// frankenscipy-6ycp2: L-BFGS-B's `hess_inv` is SciPy's lazy `LbfgsInvHessProduct` over
    /// the stored corrections. SciPy 1.17.1, `minimize(rosen, x5, jac=rosen_der,
    /// method='L-BFGS-B', options={'maxls': 2, ...})`: `hess_inv.matvec([1, -2, 0.5, 3, -1])`
    /// with 10 corrections, and with `maxcor=3` (no correction stored at the end: the identity,
    /// a different operator and the must-differ arm).
    #[test]
    fn lbfgsb_hess_inv_is_scipys_operator() {
        use crate::{HessInv, MinimizeMethodOptions};
        fn rosen(x: &[f64]) -> f64 {
            (0..x.len() - 1)
                .map(|i| 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2))
                .sum()
        }
        fn rosen_der(x: &[f64]) -> Vec<f64> {
            let n = x.len();
            let mut d = vec![0.0; n];
            for i in 1..n - 1 {
                d[i] = 200.0 * (x[i] - x[i - 1] * x[i - 1])
                    - 400.0 * (x[i + 1] - x[i] * x[i]) * x[i]
                    - 2.0 * (1.0 - x[i]);
            }
            d[0] = -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]);
            d[n - 1] = 200.0 * (x[n - 1] - x[n - 2] * x[n - 2]);
            d
        }
        let v = [1.0, -2.0, 0.5, 3.0, -1.0];
        // (maxcor, (nit, nfev), SciPy's hess_inv.matvec(v)). maxls = 2 in both: there fsci's
        // iterates match SciPy's to 1e-15, so the stored corrections do too. (At the defaults
        // x agrees to 1e-10 and the operator only to ~3e-4: near the optimum the steps s_k are
        // ~1e-7 long, so a 1e-10 difference in x is a 1e-3 difference in s.) With maxcor = 3 the
        // run ends with no correction stored and SciPy's operator is the identity.
        let cases: [(Option<usize>, (usize, usize), [f64; 5]); 2] = [
            (
                None,
                (58, 75),
                [
                    1.9838301175840392,
                    4.852811425105005,
                    8.1946217963063,
                    14.358352695675844,
                    29.037623422295617,
                ],
            ),
            (Some(3), (12, 19), [1.0, -2.0, 0.5, 3.0, -1.0]),
        ];
        let mut products = Vec::new();
        for (maxcor, (nit, nfev), expected) in cases {
            let options = MinimizeOptions {
                method: Some(OptimizeMethod::LBfgsB),
                gradient: Some(rosen_der as GradientFunc),
                method_options: MinimizeMethodOptions {
                    maxcor,
                    maxls: Some(2),
                    ..MinimizeMethodOptions::default()
                },
                ..MinimizeOptions::default()
            };
            let result = minimize(rosen, &[-1.2, 1.0, -1.2, 1.0, -1.2], options).expect("L-BFGS-B");
            assert_eq!((result.nit, result.nfev), (nit, nfev), "maxcor {maxcor:?}");
            let Some(HessInv::Lbfgs(operator)) = &result.hess_inv else {
                unreachable!(
                    "L-BFGS-B returns the lazy operator, got {:?}",
                    result.hess_inv
                );
            };
            assert_eq!(operator.shape(), (5, 5));
            let product = result
                .hess_inv
                .as_ref()
                .expect("present")
                .matvec(&v)
                .expect("matvec");
            let error = product
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs() / b.abs())
                .fold(0.0_f64, f64::max);
            println!("maxcor {maxcor:?}: matvec {product:?}, max rel error vs SciPy {error:e}");
            // Measured 1.8e-7 with 10 corrections (two stored steps are ~2e-5 long, so last-bit
            // differences in the iterates reach the operator at that level). The defect this
            // must catch, a wrong correction order, moves SciPy's own product by 0.35 (rotated
            // by one slot) to 0.95 (reversed).
            assert!(
                error <= 1e-6,
                "maxcor {maxcor:?}: {product:?} vs SciPy {expected:?}"
            );
            products.push(product);
        }
        assert_ne!(products[0], products[1], "maxcor must change the operator");
    }

    /// frankenscipy-6ycp2: an option the method does not read is SciPy's unknown solver option:
    /// Strict proceeds and records it in the optimize trace, Hardened refuses it; the values
    /// SciPy itself refuses are refused in both modes.
    #[test]
    fn unknown_and_invalid_method_options() {
        use crate::MinimizeMethodOptions;
        let f = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] + 2.0).powi(2);
        let stray = MinimizeMethodOptions {
            maxcor: Some(3),
            direc: None,
            ..MinimizeMethodOptions::default()
        };
        let strict = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            method_options: stray,
            seed: Some(0x6_7C92),
            ..MinimizeOptions::default()
        };
        let result = minimize(f, &[0.0, 0.0], strict).expect("Strict proceeds");
        assert!(result.success);
        assert!(get_optimize_traces().iter().any(|t| {
            t.event == "unknown_solver_options"
                && t.seed == Some(0x6_7C92)
                && t.reason.as_deref().is_some_and(|r| r.contains("maxcor"))
        }));
        let hardened = MinimizeOptions {
            mode: RuntimeMode::Hardened,
            ..strict
        };
        assert!(matches!(
            minimize(f, &[0.0, 0.0], hardened),
            Err(OptError::InvalidArgument { detail }) if detail.contains("Unknown solver options")
        ));
        // A read option is not unknown: the same maxcor under L-BFGS-B leaves no trace.
        let known = MinimizeOptions {
            method: Some(OptimizeMethod::LBfgsB),
            mode: RuntimeMode::Hardened,
            ..strict
        };
        assert!(minimize(f, &[0.0, 0.0], known).is_ok());

        let bad_wolfe = MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            method_options: MinimizeMethodOptions {
                c1: Some(0.9),
                c2: Some(0.5),
                ..MinimizeMethodOptions::default()
            },
            ..MinimizeOptions::default()
        };
        assert!(minimize(f, &[0.0, 0.0], bad_wolfe).is_err());
        let wrong_simplex = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
        let bad_simplex = MinimizeOptions {
            method: Some(OptimizeMethod::NelderMead),
            method_options: MinimizeMethodOptions {
                initial_simplex: Some(&wrong_simplex),
                ..MinimizeMethodOptions::default()
            },
            ..MinimizeOptions::default()
        };
        assert!(minimize(f, &[0.0, 0.0], bad_simplex).is_err());
        let wrong_direc = vec![vec![1.0, 0.0]];
        let bad_direc = MinimizeOptions {
            method: Some(OptimizeMethod::Powell),
            method_options: MinimizeMethodOptions {
                direc: Some(&wrong_direc),
                ..MinimizeMethodOptions::default()
            },
            ..MinimizeOptions::default()
        };
        assert!(minimize(f, &[0.0, 0.0], bad_direc).is_err());
    }
}
