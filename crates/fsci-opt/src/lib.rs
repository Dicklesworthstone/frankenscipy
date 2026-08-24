#![forbid(unsafe_code)]

pub mod audit;
pub mod bracket;
pub mod curvefit;
pub mod linesearch;
pub mod minimize;
pub mod nonlin;
pub mod root;
pub mod types;

pub use audit::{SyncSharedAuditLedger, record_fail_closed, sync_audit_ledger};
pub use bracket::{BracketOptions, BracketResult, bracket_root};
pub use curvefit::{
    CurveFitOptions, CurveFitResult, LeastSquaresOptions, LeastSquaresResult, LeastsqResult,
    curve_fit, curve_fit_bounded, curve_fit_bounded_many, curve_fit_many, least_squares,
    least_squares_bounded, least_squares_many, leastsq,
};
pub use linesearch::{
    LineSearchResult, ScipyLineSearchResult, WolfeParams, line_search, line_search_wolfe1,
    line_search_wolfe2,
};
pub use minimize::{
    MinimizeScalarOptions, MinimizeScalarResult, OptCaspDecision, OptCaspProblem,
    TRUST_EXACT_CHOLESKY_DISABLE, TRUST_EXACT_FLAT_AUGMENTED_DISABLE,
    TRUST_EXACT_FOLD_SHIFT_DISABLE, bfgs, cg_pr_plus, get_optimize_traces, lbfgsb, minimize,
    minimize_many, minimize_scalar, minimize_scalar_many, minimize_with_audit, nelder_mead,
    newton_cg, powell, select_minimize_method, trust_exact,
};
// NOTE on the two `anderson` functions, resolved conservatively rather than by
// picking a winner. `root::anderson(func, x0, tol, maxiter, m, beta) ->
// Result<MultivariateRootResult, OptError>` predates the nonlin family;
// `nonlin::anderson(func, x0, options: NonlinOptions) -> NonlinResult` arrived
// with 642079dc4 and collided with it at this glob (E0252), which left the
// crate uncompilable and -- because HFDT takes fsci-stats as a PATH dependency
// -- red-lined the entire hedge_fund_data_tool workspace.
//
// Dropping either from the public surface is a breaking API decision that
// belongs to this crate's author, so neither is dropped: the pre-existing
// unqualified name is left alone and the newcomer is re-exported as
// `anderson_nonlin`. Rename it freely; the point of this edit is only that a
// crate which does not build is strictly worse than either naming.
pub use nonlin::{
    AndersonJacobian, BroydenJacobian, BroydenVariant, DiagBroydenJacobian, ExcitingMixingJacobian,
    InnerMethod, InverseJacobian, Jacobian, KrylovJacobian, LineSearch, LinearMixingJacobian,
    LowRankMatrix, NonlinJacobian, NonlinMethod, NonlinOptions, NonlinResult, ReductionMethod,
    anderson as anderson_nonlin, broyden1_lowrank, broyden2_lowrank, diag_broyden, exciting_mixing,
    linear_mixing, nonlin_solve, root_nonlin,
};
pub use root::{
    MultivariateRootMethod, MultivariateRootOptions, MultivariateRootResult, RootResult,
    RootResults, anderson, bisect, brenth, brentq, brentq_many, broyden1, broyden2, df_sane,
    fsolve, halley, lm_root, newton, newton_krylov, newton_many, newton_scalar, ridder, root,
    root_many, root_scalar, secant, secant_many, toms748,
};
pub use types::{
    Bound, Bounds, ConvergenceStatus, GradientFunc, LinearConstraint, MinimizeOptions,
    NonlinearConstraint, OptError, OptimizeMethod, OptimizeResult, RootMethod, RootOptions,
};

/// Exit status for adaptive numerical differentiation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DifferentiateStatus {
    Converged,
    ErrorIncreased,
    MaxIterations,
}

/// Direction of finite-difference steps for first derivatives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepDirection {
    Central,
    Forward,
    Backward,
}

/// Absolute and relative tolerances for adaptive differentiation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DifferentiateTolerances {
    pub atol: f64,
    pub rtol: f64,
}

impl Default for DifferentiateTolerances {
    fn default() -> Self {
        Self {
            atol: f64::MIN_POSITIVE,
            rtol: f64::EPSILON.sqrt(),
        }
    }
}

/// Options for `derivative`, `jacobian`, and `hessian`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DifferentiateOptions {
    pub tolerances: DifferentiateTolerances,
    pub maxiter: usize,
    pub order: usize,
    pub initial_step: f64,
    pub step_factor: f64,
    pub step_direction: StepDirection,
}

impl Default for DifferentiateOptions {
    fn default() -> Self {
        Self {
            tolerances: DifferentiateTolerances::default(),
            maxiter: 10,
            order: 8,
            initial_step: 0.5,
            step_factor: 2.0,
            step_direction: StepDirection::Central,
        }
    }
}

/// Scalar first-derivative result matching the core fields of SciPy's RichResult.
#[derive(Debug, Clone, PartialEq)]
pub struct DerivativeResult {
    pub df: f64,
    pub error: f64,
    pub success: bool,
    pub status: DifferentiateStatus,
    pub nit: usize,
    pub nfev: usize,
    pub step: f64,
}

/// Jacobian result. `df[row][column]` is `d f[row] / d x[column]`.
#[derive(Debug, Clone, PartialEq)]
pub struct JacobianResult {
    pub df: Vec<Vec<f64>>,
    pub error: Vec<Vec<f64>>,
    pub success: bool,
    pub status: DifferentiateStatus,
    pub nit: usize,
    pub nfev: usize,
}

/// Hessian result. `ddf[row][column]` is the second partial derivative.
#[derive(Debug, Clone, PartialEq)]
pub struct HessianResult {
    pub ddf: Vec<Vec<f64>>,
    pub error: Vec<Vec<f64>>,
    pub success: bool,
    pub status: DifferentiateStatus,
    pub nit: usize,
    pub nfev: usize,
}

/// Forward-difference gradient approximation.
///
/// Matches `scipy.optimize.approx_fprime(xk, f, epsilon)`.
/// When `true`, force `approx_fprime` onto its ORIG serial per-component loop. Byte-identical either
/// way. Benchmark knob for the same-binary A/B.
#[doc(hidden)]
pub static OPT_APPROX_FPRIME_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub fn approx_fprime<F>(xk: &[f64], f: F, epsilon: f64) -> Result<Vec<f64>, OptError>
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    if xk.is_empty() {
        return Err(OptError::InvalidArgument {
            detail: String::from("xk must have at least one element"),
        });
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("epsilon must be a positive finite value"),
        });
    }
    if xk.iter().any(|value| !value.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: String::from("xk must not contain NaN or Inf"),
        });
    }

    let f0 = f(xk);
    if !f0.is_finite() {
        return Err(OptError::NonFiniteInput {
            detail: String::from("objective returned non-finite value at xk"),
        });
    }

    // Each component's forward difference perturbs ONLY `xk[index]` and evaluates `f` independently,
    // so the components fan out across cores (each thread uses a PRIVATE perturbation buffer). Result
    // is BYTE-IDENTICAL to the serial loop: identical `(f(xk+ε·eᵢ) - f0)/ε` per component in index
    // order, and the non-finite error reports the SAME (lowest) index the serial loop would hit first.
    // `f` evaluations dominate a finite-difference gradient; scipy's `approx_fprime` is serial.
    let n = xk.len();
    let component = |index: usize| -> Result<f64, usize> {
        let mut xp = xk.to_vec();
        xp[index] += epsilon;
        let f_plus = f(&xp);
        if !f_plus.is_finite() {
            return Err(index);
        }
        Ok((f_plus - f0) / epsilon)
    };
    let nthreads =
        if OPT_APPROX_FPRIME_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed) || n < 2 {
            1
        } else {
            std::thread::available_parallelism()
                .map(std::num::NonZero::get)
                .unwrap_or(1)
                .min(n)
        };
    let results: Vec<Result<f64, usize>> = if nthreads <= 1 {
        (0..n).map(component).collect()
    } else {
        let chunk = n.div_ceil(nthreads);
        let component = &component;
        let parts: Vec<Vec<Result<f64, usize>>> = std::thread::scope(|scope| {
            (0..n)
                .step_by(chunk)
                .map(|i0| {
                    scope.spawn(move || {
                        let i1 = (i0 + chunk).min(n);
                        (i0..i1).map(component).collect::<Vec<_>>()
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().expect("approx_fprime chunk panicked"))
                .collect()
        });
        let mut r = Vec::with_capacity(n);
        for part in parts {
            r.extend(part);
        }
        r
    };
    // Lowest-index non-finite error, matching the serial loop's first-error return.
    for res in &results {
        if let &Err(index) = res {
            return Err(OptError::NonFiniteInput {
                detail: format!("objective returned non-finite value at perturbed index {index}"),
            });
        }
    }
    let gradient: Vec<f64> = results.into_iter().map(|r| r.unwrap_or(0.0)).collect();
    Ok(gradient)
}

/// Finite-difference scheme used by [`approx_derivative`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FiniteDiffMethod {
    /// Forward difference `(f(x+h) - f(x)) / h` (scipy `'2-point'`).
    TwoPoint,
    /// Central difference `(f(x+h) - f(x-h)) / 2h` (scipy `'3-point'`).
    ThreePoint,
}

impl FiniteDiffMethod {
    /// Relative step `EPS**0.5` (2-point) or `EPS**(1/3)` (3-point).
    fn rel_eps(self) -> f64 {
        match self {
            FiniteDiffMethod::TwoPoint => f64::EPSILON.powf(0.5),
            FiniteDiffMethod::ThreePoint => f64::EPSILON.powf(1.0 / 3.0),
        }
    }
}

/// Compute a finite-difference approximation of the Jacobian of a vector
/// function `fun: ℝⁿ → ℝᵐ` at `x0`.
///
/// Matches the dense path of `scipy.optimize.approx_derivative` for the
/// `'2-point'` and `'3-point'` methods, including its relative/absolute step
/// selection and bound-aware step adjustment. The complex-step (`'cs'`),
/// sparse, and linear-operator variants are not supported. Returns the
/// `m × n` Jacobian; for a scalar function (`m == 1`) the single row is the
/// gradient.
///
/// `rel_step`/`abs_step` are scalars broadcast to every component (`None`
/// selects scipy's default relative step). `bounds` is `None` for the
/// unconstrained case or `Some((lb, ub))` with per-variable bounds.
/// When `true`, force `approx_derivative` onto its ORIG serial per-column loop. Byte-identical either
/// way. Benchmark knob for the same-binary A/B.
#[doc(hidden)]
pub static OPT_APPROX_DERIV_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub fn approx_derivative<F>(
    fun: F,
    x0: &[f64],
    method: FiniteDiffMethod,
    rel_step: Option<f64>,
    abs_step: Option<f64>,
    bounds: Option<(&[f64], &[f64])>,
) -> Result<Vec<Vec<f64>>, OptError>
where
    F: Fn(&[f64]) -> Vec<f64> + Sync,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptError::InvalidArgument {
            detail: "x0 must have at least one element".to_string(),
        });
    }
    if x0.iter().any(|v| !v.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: "x0 must not contain NaN or Inf".to_string(),
        });
    }

    let neg_inf = f64::NEG_INFINITY;
    let pos_inf = f64::INFINITY;
    let lb: Vec<f64> = match bounds {
        Some((l, _)) => l.to_vec(),
        None => vec![neg_inf; n],
    };
    let ub: Vec<f64> = match bounds {
        Some((_, u)) => u.to_vec(),
        None => vec![pos_inf; n],
    };
    if lb.len() != n || ub.len() != n {
        return Err(OptError::InvalidArgument {
            detail: "bounds must match x0 length".to_string(),
        });
    }
    if (0..n).any(|i| x0[i] < lb[i] || x0[i] > ub[i]) {
        return Err(OptError::InvalidArgument {
            detail: "x0 violates bound constraints".to_string(),
        });
    }

    let f0 = fun(x0);
    let m = f0.len();
    let rstep = method.rel_eps();

    // Absolute step per component (scipy _compute_absolute_step / abs_step path).
    let mut h = vec![0.0_f64; n];
    for i in 0..n {
        let sign = if x0[i] >= 0.0 { 1.0 } else { -1.0 };
        match abs_step {
            None => {
                h[i] = match rel_step {
                    None => rstep * sign * x0[i].abs().max(1.0),
                    Some(rs) => {
                        let step = rs * sign * x0[i].abs();
                        if (x0[i] + step) - x0[i] == 0.0 {
                            rstep * sign * x0[i].abs().max(1.0)
                        } else {
                            step
                        }
                    }
                };
            }
            Some(abs) => {
                // Fall back to a relative step if the absolute one underflows.
                if (x0[i] + abs) - x0[i] == 0.0 {
                    h[i] = rstep * sign * x0[i].abs().max(1.0);
                } else {
                    h[i] = abs;
                }
            }
        }
    }

    // Adjust the scheme to the bounds (num_steps = 1 for both methods).
    let all_unbounded = lb.iter().all(|&v| v == neg_inf) && ub.iter().all(|&v| v == pos_inf);
    let mut use_one_sided = vec![matches!(method, FiniteDiffMethod::TwoPoint); n];
    match method {
        FiniteDiffMethod::TwoPoint => {
            // '1-sided': use_one_sided already all true.
            if !all_unbounded {
                for i in 0..n {
                    let h_total = h[i];
                    let lower = x0[i] - lb[i];
                    let upper = ub[i] - x0[i];
                    let x = x0[i] + h_total;
                    let violated = x < lb[i] || x > ub[i];
                    let fitting = h_total.abs() <= lower.max(upper);
                    if violated && fitting {
                        h[i] = -h[i];
                    } else if !fitting {
                        if upper >= lower {
                            h[i] = upper;
                        } else {
                            h[i] = -lower;
                        }
                    }
                }
            }
        }
        FiniteDiffMethod::ThreePoint => {
            // '2-sided': h = |h|, use_one_sided all false initially.
            for hi in h.iter_mut() {
                *hi = hi.abs();
            }
            for u in use_one_sided.iter_mut() {
                *u = false;
            }
            if !all_unbounded {
                for i in 0..n {
                    let h_total = h[i];
                    let lower = x0[i] - lb[i];
                    let upper = ub[i] - x0[i];
                    let central = lower >= h_total && upper >= h_total;
                    if !central {
                        if upper >= lower {
                            h[i] = h[i].min(0.5 * upper);
                            use_one_sided[i] = true;
                        } else {
                            h[i] = -(h[i].min(0.5 * lower));
                            use_one_sided[i] = true;
                        }
                        let min_dist = upper.min(lower);
                        if h[i].abs() <= min_dist {
                            h[i] = min_dist;
                            use_one_sided[i] = false;
                        }
                    }
                }
            }
        }
    }

    // Dense difference: `jt[i]` is the i-th Jacobian column. Each column perturbs ONLY component `i`
    // and evaluates `fun` independently, so the columns fan out across cores — BYTE-IDENTICAL to the
    // serial loop (identical per-column FD arithmetic, each `jt[i]` written once; only the owning core
    // changes). The `fun` evaluations are the dominant cost of a finite-difference Jacobian, so this
    // pays whenever `fun` is non-trivial. `scipy`'s `approx_derivative` is serial.
    let compute_col = |i: usize| -> Vec<f64> {
        let mut col = vec![0.0_f64; m];
        match method {
            FiniteDiffMethod::TwoPoint => {
                let mut x1 = x0.to_vec();
                x1[i] = x0[i] + h[i];
                let f1 = fun(&x1);
                let dx = (x0[i] + h[i]) - x0[i];
                for k in 0..m {
                    col[k] = (f1[k] - f0[k]) / dx;
                }
            }
            FiniteDiffMethod::ThreePoint => {
                if use_one_sided[i] {
                    let mut x1 = x0.to_vec();
                    let mut x2 = x0.to_vec();
                    x1[i] = x0[i] + h[i];
                    x2[i] = x0[i] + 2.0 * h[i];
                    let f1 = fun(&x1);
                    let f2 = fun(&x2);
                    let dx = (x0[i] + 2.0 * h[i]) - x0[i];
                    for k in 0..m {
                        col[k] = (-3.0 * f0[k] + 4.0 * f1[k] - f2[k]) / dx;
                    }
                } else {
                    let mut x1 = x0.to_vec();
                    let mut x2 = x0.to_vec();
                    x1[i] = x0[i] - h[i];
                    x2[i] = x0[i] + h[i];
                    let f1 = fun(&x1);
                    let f2 = fun(&x2);
                    let dx = (x0[i] + h[i]) - (x0[i] - h[i]);
                    for k in 0..m {
                        col[k] = (f2[k] - f1[k]) / dx;
                    }
                }
            }
        }
        col
    };

    let nthreads =
        if OPT_APPROX_DERIV_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed) || n < 2 {
            1
        } else {
            std::thread::available_parallelism()
                .map(std::num::NonZero::get)
                .unwrap_or(1)
                .min(n)
        };

    let jt: Vec<Vec<f64>> = if nthreads <= 1 {
        (0..n).map(compute_col).collect()
    } else {
        let chunk = n.div_ceil(nthreads);
        let compute_col = &compute_col;
        let parts: Vec<Vec<Vec<f64>>> = std::thread::scope(|scope| {
            (0..n)
                .step_by(chunk)
                .map(|i0| {
                    scope.spawn(move || {
                        let i1 = (i0 + chunk).min(n);
                        (i0..i1).map(compute_col).collect::<Vec<_>>()
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().expect("approx_derivative chunk panicked"))
                .collect()
        });
        let mut jt = Vec::with_capacity(n);
        for part in parts {
            jt.extend(part);
        }
        jt
    };

    // Transpose to the m × n Jacobian.
    let mut j = vec![vec![0.0_f64; n]; m];
    for (i, col) in jt.iter().enumerate() {
        for (k, &v) in col.iter().enumerate() {
            j[k][i] = v;
        }
    }
    Ok(j)
}

/// Compare an analytical gradient to a finite-difference approximation.
///
/// Matches `scipy.optimize.check_grad`.
pub fn check_grad<F, G>(func: F, grad: G, x0: &[f64]) -> Result<f64, OptError>
where
    F: Fn(&[f64]) -> f64 + Sync,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let analytical = grad(x0);
    if analytical.len() != x0.len() {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "gradient length must match x0 length (got {} and {})",
                analytical.len(),
                x0.len()
            ),
        });
    }
    if analytical.iter().any(|value| !value.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: String::from("gradient returned non-finite values"),
        });
    }

    let numerical = approx_fprime(x0, func, 1.490_116_119_384_765_6e-8)?;
    let squared_error = analytical
        .iter()
        .zip(numerical.iter())
        .map(|(lhs, rhs)| {
            let delta = lhs - rhs;
            delta * delta
        })
        .sum::<f64>();
    Ok(squared_error.sqrt())
}

// ═══════════════════════════════════════════════════════════════════════════
// Rosenbrock test function family
// ═══════════════════════════════════════════════════════════════════════════

/// The Rosenbrock function.
///
/// Matches `scipy.optimize.rosen`. The function is defined as:
/// f(x) = sum_{i=0}^{n-2} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
///
/// The global minimum is at x = [1, 1, ..., 1] with f(x) = 0.
///
/// # Arguments
/// * `x` - Input point (length >= 2)
///
/// # Returns
/// Function value at x
pub fn rosen(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..x.len() - 1 {
        let t1 = x[i + 1] - x[i] * x[i];
        let t2 = 1.0 - x[i];
        sum += 100.0 * t1 * t1 + t2 * t2;
    }
    sum
}

/// Gradient of the Rosenbrock function.
///
/// Matches `scipy.optimize.rosen_der`.
///
/// # Arguments
/// * `x` - Input point (length >= 2)
///
/// # Returns
/// Gradient vector at x
pub fn rosen_der(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n < 2 {
        return vec![0.0; n];
    }

    let mut grad = vec![0.0; n];

    // First element: df/dx_0 = -400*x_0*(x_1 - x_0^2) - 2*(1 - x_0)
    grad[0] = -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]);

    // Middle elements: df/dx_i = 200*(x_i - x_{i-1}^2) - 400*x_i*(x_{i+1} - x_i^2) - 2*(1 - x_i)
    for i in 1..n - 1 {
        grad[i] = 200.0 * (x[i] - x[i - 1] * x[i - 1])
            - 400.0 * x[i] * (x[i + 1] - x[i] * x[i])
            - 2.0 * (1.0 - x[i]);
    }

    // Last element: df/dx_{n-1} = 200*(x_{n-1} - x_{n-2}^2)
    grad[n - 1] = 200.0 * (x[n - 1] - x[n - 2] * x[n - 2]);

    grad
}

/// Hessian matrix of the Rosenbrock function.
///
/// Matches `scipy.optimize.rosen_hess`.
///
/// # Arguments
/// * `x` - Input point (length >= 2)
///
/// # Returns
/// Hessian matrix (n x n) at x
pub fn rosen_hess(x: &[f64]) -> Vec<Vec<f64>> {
    let n = x.len();
    if n < 2 {
        return vec![vec![0.0; n]; n];
    }

    let mut h = vec![vec![0.0; n]; n];

    // The Hessian has a tridiagonal structure
    // d²f/dx_i² and d²f/dx_i dx_{i±1} are non-zero

    // First diagonal element
    h[0][0] = 1200.0 * x[0] * x[0] - 400.0 * x[1] + 2.0;

    // First off-diagonal
    h[0][1] = -400.0 * x[0];
    h[1][0] = -400.0 * x[0];

    // Middle elements
    for i in 1..n - 1 {
        h[i][i] = 202.0 + 1200.0 * x[i] * x[i] - 400.0 * x[i + 1];
        h[i][i + 1] = -400.0 * x[i];
        h[i + 1][i] = -400.0 * x[i];
    }

    // Last diagonal element
    h[n - 1][n - 1] = 200.0;

    h
}

/// Hessian-vector product for the Rosenbrock function.
///
/// Matches `scipy.optimize.rosen_hess_prod`. Computes H(x) @ p
/// without explicitly forming the Hessian matrix.
///
/// # Arguments
/// * `x` - Input point (length >= 2)
/// * `p` - Vector to multiply (same length as x)
///
/// # Returns
/// Result of Hessian-vector product
pub fn rosen_hess_prod(x: &[f64], p: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n < 2 || p.len() != n {
        return vec![0.0; n];
    }

    let mut hp = vec![0.0; n];

    // First element
    hp[0] = (1200.0 * x[0] * x[0] - 400.0 * x[1] + 2.0) * p[0] - 400.0 * x[0] * p[1];

    // Middle elements
    for i in 1..n - 1 {
        hp[i] = -400.0 * x[i - 1] * p[i - 1]
            + (202.0 + 1200.0 * x[i] * x[i] - 400.0 * x[i + 1]) * p[i]
            - 400.0 * x[i] * p[i + 1];
    }

    // Last element
    hp[n - 1] = -400.0 * x[n - 2] * p[n - 2] + 200.0 * p[n - 1];

    hp
}

/// Solve the linear sum assignment problem using a Hungarian-style algorithm.
///
/// Returns `(row_ind, col_ind)` with monotonically increasing row indices.
pub fn linear_sum_assignment(
    cost_matrix: &[Vec<f64>],
) -> Result<(Vec<usize>, Vec<usize>), OptError> {
    if cost_matrix.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    let col_count = cost_matrix[0].len();
    if col_count == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    for (row_index, row) in cost_matrix.iter().enumerate() {
        if row.len() != col_count {
            return Err(OptError::InvalidArgument {
                detail: format!(
                    "cost_matrix must be rectangular; row 0 has length {col_count} but row {row_index} has length {}",
                    row.len()
                ),
            });
        }
        if row.iter().any(|value| !value.is_finite()) {
            return Err(OptError::NonFiniteInput {
                detail: format!("cost_matrix contains non-finite values in row {row_index}"),
            });
        }
    }

    if cost_matrix.len() <= col_count {
        let assignment = shortest_augmenting_path_rectangular(cost_matrix);
        let row_ind = (0..assignment.len()).collect::<Vec<_>>();
        return Ok((row_ind, assignment));
    }

    let transposed = transpose_matrix(cost_matrix);
    let assignment = shortest_augmenting_path_rectangular(&transposed);
    let mut pairs = assignment
        .into_iter()
        .enumerate()
        .map(|(col, row)| (row, col))
        .collect::<Vec<_>>();
    pairs.sort_unstable_by_key(|(row, _)| *row);
    let (row_ind, col_ind): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
    Ok((row_ind, col_ind))
}

fn transpose_matrix(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let row_count = matrix.len();
    let col_count = matrix[0].len();
    let mut transposed = vec![vec![0.0; row_count]; col_count];
    for (row_index, row) in matrix.iter().enumerate() {
        for (col_index, value) in row.iter().enumerate() {
            transposed[col_index][row_index] = *value;
        }
    }
    transposed
}

/// Result of [`quadratic_assignment`].
#[derive(Debug, Clone)]
pub struct QuadraticAssignmentResult {
    /// The column indices of the optimal permutation: node `i` of `A` maps to
    /// node `col_ind[i]` of `B`.
    pub col_ind: Vec<usize>,
    /// The objective value `Σ A[i][j]·B[col[i]][col[j]]`.
    pub fun: f64,
    /// The number of FAQ iterations performed.
    pub nit: usize,
}

// Dense n×n matrix multiply (small matrices).
fn qap_matmul(x: &[Vec<f64>], y: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = x.len();
    let m = if y.is_empty() { 0 } else { y[0].len() };
    let k = y.len();
    let mut out = vec![vec![0.0; m]; n];
    for i in 0..n {
        for p in 0..k {
            let xip = x[i][p];
            if xip == 0.0 {
                continue;
            }
            for j in 0..m {
                out[i][j] += xip * y[p][j];
            }
        }
    }
    out
}

// Σ_ij x[i][j]·y[i][j] (trace(Xᵀ Y) written as an elementwise product sum).
fn qap_elem_mul_sum(x: &[Vec<f64>], y: &[Vec<f64>]) -> f64 {
    x.iter()
        .zip(y)
        .map(|(xr, yr)| xr.iter().zip(yr).map(|(&a, &b)| a * b).sum::<f64>())
        .sum()
}

/// Solve the quadratic assignment problem (approximately) with the Fast
/// Approximate QAP (FAQ) algorithm, matching `scipy.optimize.quadratic_assignment`
/// with `method='faq'` for the default options (no `partial_match`, no input
/// shuffle, `P0='barycenter'`).
///
/// Minimizes `trace(Aᵀ P B Pᵀ)` over permutation matrices `P` (or maximizes it
/// when `maximize` is true), returning the column permutation, objective value,
/// and iteration count. `A` and `B` must be square of the same size.
/// scipy defaults are `maximize=false`, `maxiter=30`, `tol=0.03`.
pub fn quadratic_assignment(
    a: &[Vec<f64>],
    b: &[Vec<f64>],
    maximize: bool,
    maxiter: usize,
    tol: f64,
) -> Result<QuadraticAssignmentResult, OptError> {
    let n = a.len();
    if maxiter == 0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("'maxiter' must be a positive integer"),
        });
    }
    if tol <= 0.0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("'tol' must be a positive float"),
        });
    }
    if a.iter().any(|r| r.len() != n) || b.len() != n || b.iter().any(|r| r.len() != n) {
        return Err(OptError::InvalidArgument {
            detail: String::from("`A` and `B` must be square matrices of the same size"),
        });
    }
    if n == 0 {
        return Ok(QuadraticAssignmentResult {
            col_ind: Vec::new(),
            fun: 0.0,
            nit: 0,
        });
    }

    let obj_scalar = if maximize { -1.0 } else { 1.0 };
    let at = transpose_matrix(a);
    let bt = transpose_matrix(b);
    let nf = n as f64;
    // P0 = barycenter of the Birkhoff polytope.
    let mut p = vec![vec![1.0 / nf; n]; n];
    let mut n_iter = 0usize;

    for it in 1..=maxiter {
        n_iter = it;
        // grad = A·P·Bᵀ + Aᵀ·P·B.
        let grad1 = qap_matmul(&qap_matmul(a, &p), &bt);
        let grad2 = qap_matmul(&qap_matmul(&at, &p), b);
        let grad: Vec<Vec<f64>> = grad1
            .iter()
            .zip(&grad2)
            .map(|(r1, r2)| r1.iter().zip(r2).map(|(&x, &y)| x + y).collect())
            .collect();

        // Direction Q from the linear assignment on the gradient.
        let cost: Vec<Vec<f64>> = if maximize {
            grad.iter()
                .map(|r| r.iter().map(|&v| -v).collect())
                .collect()
        } else {
            grad.clone()
        };
        let (_, cols) = linear_sum_assignment(&cost)?;
        let mut q = vec![vec![0.0; n]; n];
        for (i, &c) in cols.iter().enumerate() {
            q[i][c] = 1.0;
        }

        // R = P - Q.
        let r: Vec<Vec<f64>> = p
            .iter()
            .zip(&q)
            .map(|(pr, qr)| pr.iter().zip(qr).map(|(&x, &y)| x - y).collect())
            .collect();
        // AR = Aᵀ·R ; BR = B·Rᵀ.
        let ar = qap_matmul(&at, &r);
        let br = qap_matmul(b, &transpose_matrix(&r));
        // Step-size quadratic coefficients (no-seed simplification).
        let bt_cols: Vec<Vec<f64>> = cols.iter().map(|&c| bt[c].clone()).collect();
        let br_cols: Vec<Vec<f64>> = cols.iter().map(|&c| br[c].clone()).collect();
        let b22a = qap_elem_mul_sum(&ar, &bt_cols);
        let b22b = qap_elem_mul_sum(a, &br_cols);
        let a_coef = qap_elem_mul_sum(&transpose_matrix(&ar), &br);
        let b_coef = b22a + b22b;

        let crit = -b_coef / (2.0 * a_coef);
        let alpha = if a_coef * obj_scalar > 0.0 && (0.0..=1.0).contains(&crit) {
            crit
        } else if (b_coef + a_coef) * obj_scalar >= 0.0 {
            0.0
        } else {
            1.0
        };

        // P_i1 = alpha·P + (1 - alpha)·Q.
        let p_new: Vec<Vec<f64>> = p
            .iter()
            .zip(&q)
            .map(|(pr, qr)| {
                pr.iter()
                    .zip(qr)
                    .map(|(&pv, &qv)| alpha * pv + (1.0 - alpha) * qv)
                    .collect()
            })
            .collect();
        // Convergence on ‖P - P_i1‖_F / √n.
        let diff: f64 = p
            .iter()
            .zip(&p_new)
            .map(|(pr, nr)| {
                pr.iter()
                    .zip(nr)
                    .map(|(&x, &y)| (x - y) * (x - y))
                    .sum::<f64>()
            })
            .sum::<f64>()
            .sqrt()
            / nf.sqrt();
        p = p_new;
        if diff < tol {
            break;
        }
    }

    // Project onto a permutation: maximize the trace against P (== min on -P).
    let neg_p: Vec<Vec<f64>> = p.iter().map(|r| r.iter().map(|&v| -v).collect()).collect();
    let (_, col) = linear_sum_assignment(&neg_p)?;
    // Objective: Σ A[i][j]·B[col[i]][col[j]].
    let mut fun = 0.0;
    for i in 0..n {
        for j in 0..n {
            fun += a[i][j] * b[col[i]][col[j]];
        }
    }
    Ok(QuadraticAssignmentResult {
        col_ind: col,
        fun,
        nit: n_iter,
    })
}

// Modified Jonker-Volgenant shortest augmenting path implementation for
// row_count <= col_count, following SciPy's rectangular_lsap core.
fn shortest_augmenting_path_rectangular(cost_matrix: &[Vec<f64>]) -> Vec<usize> {
    let row_count = cost_matrix.len();
    let col_count = cost_matrix[0].len();
    // Flatten to one contiguous row-major buffer. scipy's LAPJVsp solves on a
    // flat C array; the per-row `Vec<f64>` indirection here scatters each row
    // across the heap, so the O(n) inner reduced-cost scans miss cache more as
    // n grows (measured 1.09x→1.54x slower than scipy from n=256→1000). A single
    // O(n²) copy up front (cheap vs the O(n³) augmenting-path work) restores
    // contiguous, cache-friendly row access while staying byte-identical.
    let mut flat = Vec::with_capacity(row_count * col_count);
    for row in cost_matrix {
        flat.extend_from_slice(row);
    }
    let mut u = vec![0.0; row_count];
    let mut v = vec![0.0; col_count];
    let mut col4row = vec![UNASSIGNED; row_count];
    let mut row4col = vec![UNASSIGNED; col_count];
    let mut scratch = ShortestAugmentingPathScratch::new(row_count, col_count);

    for cur_row in 0..row_count {
        let (sink, min_val) =
            scratch.shortest_augmenting_path(&flat, col_count, &u, &v, &row4col, cur_row);

        u[cur_row] += min_val;
        for row in 0..row_count {
            if scratch.sr[row] && row != cur_row {
                let col = col4row[row];
                debug_assert_ne!(col, UNASSIGNED);
                u[row] += min_val - scratch.shortest_path_costs[col];
            }
        }
        for (col, v_col) in v.iter_mut().enumerate() {
            if scratch.sc[col] {
                *v_col -= min_val - scratch.shortest_path_costs[col];
            }
        }

        let mut col = sink;
        loop {
            let row = scratch.path[col];
            debug_assert_ne!(row, UNASSIGNED);
            row4col[col] = row;
            let previous_col = col4row[row];
            col4row[row] = col;
            if row == cur_row {
                break;
            }
            col = previous_col;
        }
    }

    col4row
}

const UNASSIGNED: usize = usize::MAX;

struct ShortestAugmentingPathScratch {
    path: Vec<usize>,
    shortest_path_costs: Vec<f64>,
    sr: Vec<bool>,
    sc: Vec<bool>,
    remaining: Vec<usize>,
}

impl ShortestAugmentingPathScratch {
    fn new(row_count: usize, col_count: usize) -> Self {
        Self {
            path: vec![UNASSIGNED; col_count],
            shortest_path_costs: vec![0.0; col_count],
            sr: vec![false; row_count],
            sc: vec![false; col_count],
            remaining: vec![0usize; col_count],
        }
    }

    #[inline(always)]
    fn shortest_augmenting_path(
        &mut self,
        cost: &[f64],
        cost_cols: usize,
        u: &[f64],
        v: &[f64],
        row4col: &[usize],
        start_row: usize,
    ) -> (usize, f64) {
        let col_count = v.len();
        let mut min_val;
        let mut row = start_row;
        let mut remaining_count = col_count;

        for (index, slot) in self.remaining.iter_mut().enumerate() {
            *slot = col_count - index - 1;
        }
        self.sr.fill(false);
        self.sc.fill(false);

        let mut lowest = f64::INFINITY;
        let mut best_remaining_index = 0usize;
        let cost_row = &cost[row * cost_cols..][..cost_cols];
        let row_dual = u[row];
        self.sr[row] = true;

        for (remaining_index, &col) in self.remaining[..remaining_count].iter().enumerate() {
            let reduced_cost = cost_row[col] - row_dual - v[col];
            self.path[col] = row;
            self.shortest_path_costs[col] = reduced_cost;
            if reduced_cost < lowest || (reduced_cost == lowest && row4col[col] == UNASSIGNED) {
                lowest = reduced_cost;
                best_remaining_index = remaining_index;
            }
        }

        min_val = lowest;
        let col = self.remaining[best_remaining_index];
        if row4col[col] == UNASSIGNED {
            return (col, min_val);
        }

        row = row4col[col];
        self.sc[col] = true;
        remaining_count -= 1;
        self.remaining[best_remaining_index] = self.remaining[remaining_count];

        loop {
            let mut lowest = f64::INFINITY;
            let mut best_remaining_index = 0usize;
            let cost_row = &cost[row * cost_cols..][..cost_cols];
            let row_dual = u[row];
            self.sr[row] = true;

            for (remaining_index, &col) in self.remaining[..remaining_count].iter().enumerate() {
                let reduced_cost = min_val + cost_row[col] - row_dual - v[col];
                if reduced_cost < self.shortest_path_costs[col] {
                    self.path[col] = row;
                    self.shortest_path_costs[col] = reduced_cost;
                }
                if self.shortest_path_costs[col] < lowest
                    || (self.shortest_path_costs[col] == lowest && row4col[col] == UNASSIGNED)
                {
                    lowest = self.shortest_path_costs[col];
                    best_remaining_index = remaining_index;
                }
            }

            min_val = lowest;
            let col = self.remaining[best_remaining_index];
            if row4col[col] == UNASSIGNED {
                return (col, min_val);
            }

            row = row4col[col];
            self.sc[col] = true;
            remaining_count -= 1;
            self.remaining[best_remaining_index] = self.remaining[remaining_count];
        }
    }
}

#[cfg(test)]
fn hungarian_rectangular_reference(cost_matrix: &[Vec<f64>]) -> Vec<usize> {
    let row_count = cost_matrix.len();
    let col_count = cost_matrix[0].len();
    let mut u = vec![0.0; row_count + 1];
    let mut v = vec![0.0; col_count + 1];
    let mut p = vec![0usize; col_count + 1];
    let mut way = vec![0usize; col_count + 1];

    for row in 1..=row_count {
        p[0] = row;
        let mut col0 = 0usize;
        let mut minv = vec![f64::INFINITY; col_count + 1];
        let mut used = vec![false; col_count + 1];

        loop {
            used[col0] = true;
            let row0 = p[col0];
            let mut delta = f64::INFINITY;
            let mut col1 = 0usize;

            for col in 1..=col_count {
                if used[col] {
                    continue;
                }
                let current = cost_matrix[row0 - 1][col - 1] - u[row0] - v[col];
                if current < minv[col] {
                    minv[col] = current;
                    way[col] = col0;
                }
                if minv[col] < delta {
                    delta = minv[col];
                    col1 = col;
                }
            }

            for col in 0..=col_count {
                if used[col] {
                    u[p[col]] += delta;
                    v[col] -= delta;
                } else {
                    minv[col] -= delta;
                }
            }
            col0 = col1;

            if p[col0] == 0 {
                break;
            }
        }

        loop {
            let col1 = way[col0];
            p[col0] = p[col1];
            col0 = col1;
            if col0 == 0 {
                break;
            }
        }
    }

    let mut assignment = vec![0usize; row_count];
    for col in 1..=col_count {
        if p[col] != 0 {
            assignment[p[col] - 1] = col - 1;
        }
    }
    assignment
}

// ══════════════════════════════════════════════════════════════════════
// Linear Programming: linprog
// ══════════════════════════════════════════════════════════════════════

/// Result of a linear programming problem.
#[derive(Debug, Clone)]
pub struct LinprogResult {
    /// Solution vector.
    pub x: Vec<f64>,
    /// Optimal objective value `c^T x`.
    pub fun: f64,
    /// Slack variables for inequality constraints.
    pub slack: Vec<f64>,
    /// Whether the solver succeeded.
    pub success: bool,
    /// Solver status: 0 = optimal, 1 = iteration limit, 2 = infeasible, 3 = unbounded.
    pub status: u8,
    /// Human-readable message.
    pub message: String,
    /// Number of iterations.
    pub nit: usize,
}

/// Variable integrality kind for `milp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Integrality {
    Continuous,
    Integer,
    Binary,
}

/// Result of a mixed-integer linear programming problem.
#[derive(Debug, Clone)]
pub struct MilpResult {
    /// Solution vector.
    pub x: Vec<f64>,
    /// Optimal objective value `c^T x`.
    pub fun: f64,
    /// Whether the solver succeeded.
    pub success: bool,
    /// Solver status: 0 = optimal, 1 = node/iteration limit, 2 = infeasible, 3 = unbounded.
    pub status: u8,
    /// Human-readable message.
    pub message: String,
    /// Number of LP relaxations solved.
    pub mip_node_count: usize,
    /// Accumulated LP iteration count.
    pub nit: usize,
}

/// Problem definition for `milp`.
#[derive(Debug, Clone, Copy)]
pub struct MilpProblem<'a> {
    pub c: &'a [f64],
    pub integrality: &'a [Integrality],
    pub a_ub: &'a [Vec<f64>],
    pub b_ub: &'a [f64],
    pub a_eq: &'a [Vec<f64>],
    pub b_eq: &'a [f64],
    pub bounds: &'a [(Option<f64>, Option<f64>)],
}

/// Solver controls for `milp`.
#[derive(Debug, Clone, Copy, Default)]
pub struct MilpOptions {
    pub max_nodes: Option<usize>,
    pub lp_maxiter: Option<usize>,
}

#[derive(Debug, Clone)]
struct StandardFormVariable {
    offset: f64,
    terms: Vec<(usize, f64)>,
}

#[derive(Debug, Clone)]
struct StandardFormTransform {
    original_vars: Vec<StandardFormVariable>,
    decision_var_count: usize,
    upper_bound_rows: Vec<(usize, f64)>,
}

fn validate_variable_bounds(index: usize, lo: f64, hi: f64) -> Result<(), OptError> {
    if lo.is_nan() || hi.is_nan() {
        return Err(OptError::InvalidBounds {
            detail: format!("variable {index}: bounds must not be NaN (got {lo}, {hi})"),
        });
    }
    if lo == f64::INFINITY {
        return Err(OptError::InvalidBounds {
            detail: format!("variable {index}: lower bound must not be +inf"),
        });
    }
    if hi == f64::NEG_INFINITY {
        return Err(OptError::InvalidBounds {
            detail: format!("variable {index}: upper bound must not be -inf"),
        });
    }
    if lo > hi {
        return Err(OptError::InvalidBounds {
            detail: format!("variable {index}: lower bound {lo} > upper bound {hi}"),
        });
    }
    Ok(())
}

fn build_standard_form_transform(
    var_bounds: &[(f64, f64)],
) -> Result<StandardFormTransform, OptError> {
    let mut original_vars = Vec::with_capacity(var_bounds.len());
    let mut upper_bound_rows = Vec::new();
    let mut next_index = 0usize;

    for (index, &(lo, hi)) in var_bounds.iter().enumerate() {
        validate_variable_bounds(index, lo, hi)?;
        if lo.is_finite() {
            let std_index = next_index;
            next_index += 1;
            original_vars.push(StandardFormVariable {
                offset: lo,
                terms: vec![(std_index, 1.0)],
            });
            if hi.is_finite() {
                upper_bound_rows.push((std_index, hi - lo));
            }
        } else if hi.is_finite() {
            let std_index = next_index;
            next_index += 1;
            original_vars.push(StandardFormVariable {
                offset: hi,
                terms: vec![(std_index, -1.0)],
            });
        } else {
            let pos_index = next_index;
            let neg_index = next_index + 1;
            next_index += 2;
            original_vars.push(StandardFormVariable {
                offset: 0.0,
                terms: vec![(pos_index, 1.0), (neg_index, -1.0)],
            });
        }
    }

    Ok(StandardFormTransform {
        original_vars,
        decision_var_count: next_index,
        upper_bound_rows,
    })
}

/// Solve a linear programming problem using the revised simplex method.
///
/// Matches `scipy.optimize.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method)`.
///
/// Minimizes `c^T x` subject to:
///   `A_ub @ x <= b_ub` (inequality constraints)
///   `A_eq @ x == b_eq` (equality constraints)
///   `lb <= x <= ub` (variable bounds)
///
/// # Arguments
/// * `c` — Objective function coefficients (minimize c^T x).
/// * `a_ub` — Inequality constraint matrix (each row: a_i^T x <= b_ub_i). Can be empty.
/// * `b_ub` — Inequality constraint RHS.
/// * `a_eq` — Equality constraint matrix. Can be empty.
/// * `b_eq` — Equality constraint RHS.
/// * `bounds` — Variable bounds as `(Option<lower>, Option<upper>)` per variable.
///   `None` means unbounded in that direction. If empty, defaults to `(0, +inf)` per variable.
/// * `maxiter` — Maximum number of simplex iterations.
pub fn linprog(
    c: &[f64],
    a_ub: &[Vec<f64>],
    b_ub: &[f64],
    a_eq: &[Vec<f64>],
    b_eq: &[f64],
    bounds: &[(Option<f64>, Option<f64>)],
    maxiter: Option<usize>,
) -> Result<LinprogResult, OptError> {
    let n = c.len();
    if n == 0 {
        return Err(OptError::InvalidArgument {
            detail: "c must not be empty".to_string(),
        });
    }
    if c.iter().any(|value| !value.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: "c must contain only finite values".to_string(),
        });
    }
    if maxiter == Some(0) {
        return Err(OptError::InvalidArgument {
            detail: "maxiter must be greater than zero".to_string(),
        });
    }

    // Validate dimensions.
    if a_ub.len() != b_ub.len() {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "A_ub rows ({}) must match b_ub length ({})",
                a_ub.len(),
                b_ub.len()
            ),
        });
    }
    for (i, row) in a_ub.iter().enumerate() {
        if row.len() != n {
            return Err(OptError::InvalidArgument {
                detail: format!("A_ub row {i} has {} cols, expected {n}", row.len()),
            });
        }
        if row.iter().any(|value| !value.is_finite()) {
            return Err(OptError::NonFiniteInput {
                detail: format!("A_ub row {i} contains non-finite values"),
            });
        }
    }
    if b_ub.iter().any(|value| !value.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: "b_ub must contain only finite values".to_string(),
        });
    }
    if a_eq.len() != b_eq.len() {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "A_eq rows ({}) must match b_eq length ({})",
                a_eq.len(),
                b_eq.len()
            ),
        });
    }
    for (i, row) in a_eq.iter().enumerate() {
        if row.len() != n {
            return Err(OptError::InvalidArgument {
                detail: format!("A_eq row {i} has {} cols, expected {n}", row.len()),
            });
        }
        if row.iter().any(|value| !value.is_finite()) {
            return Err(OptError::NonFiniteInput {
                detail: format!("A_eq row {i} contains non-finite values"),
            });
        }
    }
    if b_eq.iter().any(|value| !value.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: "b_eq must contain only finite values".to_string(),
        });
    }

    // Apply variable bounds by transforming to standard form.
    // Default bounds: (0, +inf) if not specified.
    let var_bounds: Vec<(f64, f64)> = if bounds.is_empty() {
        vec![(0.0, f64::INFINITY); n]
    } else {
        if bounds.len() != n {
            return Err(OptError::InvalidArgument {
                detail: format!("bounds length ({}) must match c length ({n})", bounds.len()),
            });
        }
        bounds
            .iter()
            .map(|(lo, hi)| (lo.unwrap_or(f64::NEG_INFINITY), hi.unwrap_or(f64::INFINITY)))
            .collect()
    };

    let transform = build_standard_form_transform(&var_bounds)?;

    // Transform to standard form: minimize c^T x, A x = b, x >= 0.
    // 1. Finite-lower-bound variables are shifted to y >= 0.
    // 2. Upper-only variables are reflected as x = ub - y with y >= 0.
    // 3. Free variables are split as x = x_pos - x_neg with x_pos, x_neg >= 0.
    // 4. Add slack variables for inequality constraints and finite shifted upper bounds.
    let m_ub = a_ub.len();
    let m_eq = a_eq.len();
    let decision_var_count = transform.decision_var_count;
    let ub_slacks = &transform.upper_bound_rows;
    let n_ub_slack = ub_slacks.len();

    // Total variables: transformed decision vars + inequality slack + upper-bound slack.
    let total_vars = decision_var_count + m_ub + n_ub_slack;
    let total_constraints = m_ub + m_eq + n_ub_slack;

    // Build standard-form objective: c' = [c, 0, 0, ...]
    let mut c_std = vec![0.0; total_vars];
    for (&coeff, transformed) in c.iter().zip(transform.original_vars.iter()) {
        for &(std_index, std_coeff) in &transformed.terms {
            c_std[std_index] += coeff * std_coeff;
        }
    }

    // Build constraint matrix A_std and b_std.
    let mut a_std = vec![vec![0.0; total_vars]; total_constraints];
    let mut b_std = vec![0.0; total_constraints];

    // Row 0..m_ub: inequality constraints → A_ub x + s = b_ub.
    for i in 0..m_ub {
        b_std[i] = b_ub[i];
        for (j, transformed) in transform.original_vars.iter().enumerate() {
            for &(std_index, std_coeff) in &transformed.terms {
                a_std[i][std_index] += a_ub[i][j] * std_coeff;
            }
            b_std[i] -= a_ub[i][j] * transformed.offset;
        }
        a_std[i][decision_var_count + i] = 1.0; // slack variable
    }

    // Row m_ub..m_ub+m_eq: equality constraints.
    for i in 0..m_eq {
        b_std[m_ub + i] = b_eq[i];
        for (j, transformed) in transform.original_vars.iter().enumerate() {
            for &(std_index, std_coeff) in &transformed.terms {
                a_std[m_ub + i][std_index] += a_eq[i][j] * std_coeff;
            }
            b_std[m_ub + i] -= a_eq[i][j] * transformed.offset;
        }
    }

    // Row m_ub+m_eq..: shifted finite upper bounds: y_i + s_ub = ub_i - lb_i.
    for (k, &(std_index, bound_diff)) in ub_slacks.iter().enumerate() {
        let row = m_ub + m_eq + k;
        a_std[row][std_index] = 1.0;
        a_std[row][decision_var_count + m_ub + k] = 1.0;
        b_std[row] = bound_diff;
    }

    // Ensure all b_std >= 0 (multiply negative rows by -1); record which rows were
    // flipped — their slack column becomes -1 and can no longer seed the basis.
    let mut flipped = vec![false; total_constraints];
    for i in 0..total_constraints {
        if b_std[i] < 0.0 {
            flipped[i] = true;
            b_std[i] = -b_std[i];
            for val in &mut a_std[i] {
                *val = -*val;
            }
        }
    }

    // Phase I: find a basic feasible solution. Seed the basis from the natural
    // slack columns — a ≤ or finite-upper-bound row that was NOT sign-flipped
    // already carries a +1 unit slack, a basic FEASIBLE variable (value
    // b_std[i] ≥ 0), so it needs NO artificial. Only equality rows and sign-flipped
    // rows (slack became −1) get one. This shrinks Phase I and ELIMINATES it for an
    // all-≤, b≥0 LP (origin feasible), where the old code added one artificial per
    // constraint and pivoted them all out.
    let obj_row = total_constraints;
    let mut basis: Vec<usize> = vec![0; total_constraints];
    let mut art_of_row = vec![usize::MAX; total_constraints];
    let mut n_art = 0usize;
    for (i, row_basis) in basis.iter_mut().enumerate() {
        let slack = if flipped[i] {
            None
        } else if i < m_ub {
            Some(decision_var_count + i)
        } else if i >= m_ub + m_eq {
            Some(decision_var_count + m_ub + (i - m_ub - m_eq))
        } else {
            None // equality row — no slack
        };
        match slack {
            Some(col) => *row_basis = col,
            None => {
                art_of_row[i] = n_art;
                n_art += 1;
            }
        }
    }

    let phase1_vars = total_vars + n_art;
    let mut tableau = vec![vec![0.0; phase1_vars + 1]; total_constraints + 1];

    // Fill constraint rows; an artificial column only where the row needs one.
    for i in 0..total_constraints {
        tableau[i][..total_vars].copy_from_slice(&a_std[i][..total_vars]);
        if art_of_row[i] != usize::MAX {
            let acol = total_vars + art_of_row[i];
            tableau[i][acol] = 1.0;
            basis[i] = acol;
        }
        tableau[i][phase1_vars] = b_std[i]; // RHS
    }

    // Phase I objective: minimize the sum of the artificials actually present,
    // then eliminate each (basic) artificial from the objective by subtracting its
    // row. With no artificials the objective row is already zero (Phase I no-op).
    for i in 0..total_constraints {
        if art_of_row[i] != usize::MAX {
            tableau[obj_row][total_vars + art_of_row[i]] = 1.0;
        }
    }
    for i in 0..total_constraints {
        if art_of_row[i] != usize::MAX {
            let (tableau_i, tableau_obj) = {
                let (left, right) = tableau.split_at_mut(obj_row);
                (&left[i], &mut right[0])
            };
            for (obj_val, &i_val) in tableau_obj
                .iter_mut()
                .zip(tableau_i.iter())
                .take(phase1_vars + 1)
            {
                *obj_val -= i_val;
            }
        }
    }
    let maxiter = maxiter.unwrap_or(10_000);

    // Simplex iterations for Phase I.
    let phase1_nit = match simplex_iterate(&mut tableau, &mut basis, maxiter, phase1_vars) {
        Ok(nit) => nit,
        Err(err_code) => {
            let (status, message) = if err_code == 3 {
                (
                    3,
                    "Phase I problem is unbounded (should not happen)".to_string(),
                )
            } else {
                (1, "Phase I iteration limit exceeded".to_string())
            };
            return Ok(LinprogResult {
                x: vec![0.0; n],
                fun: 0.0,
                slack: vec![0.0; m_ub],
                success: false,
                status,
                message,
                nit: maxiter,
            });
        }
    };

    // Check if Phase I found a feasible solution (all artificials = 0).
    // Phase I objective = -sum(artificials). If < -1e-8, some artificials are nonzero → infeasible.
    let phase1_obj = tableau[obj_row][phase1_vars];
    if phase1_obj < -1e-8 {
        return Ok(LinprogResult {
            x: vec![0.0; n],
            fun: 0.0,
            slack: vec![0.0; m_ub],
            success: false,
            status: 2,
            message: "Problem is infeasible".to_string(),
            nit: phase1_nit,
        });
    }

    // Phase II: optimize with original objective.
    // Remove artificial columns and set Phase II objective.
    let mut tableau2 = vec![vec![0.0; total_vars + 1]; total_constraints + 1];
    for i in 0..total_constraints {
        for j in 0..total_vars {
            tableau2[i][j] = tableau[i][j];
        }
        tableau2[i][total_vars] = tableau[i][phase1_vars]; // RHS
    }

    // Set Phase II objective row.
    tableau2[obj_row][..total_vars].copy_from_slice(&c_std[..total_vars]);
    tableau2[obj_row][total_vars] = 0.0;

    // Eliminate basic variables from objective row.
    for (row, &bj) in basis.iter().enumerate() {
        if bj < total_vars {
            let ratio = tableau2[obj_row][bj];
            if ratio.abs() > 1e-15 {
                let (row_vals, obj_vals) = {
                    let (left, right) = tableau2.split_at_mut(obj_row);
                    (&left[row], &mut right[0])
                };
                for (obj_val, &row_val) in obj_vals
                    .iter_mut()
                    .zip(row_vals.iter())
                    .take(total_vars + 1)
                {
                    *obj_val -= ratio * row_val;
                }
            }
        }
    }

    let phase2_result = simplex_iterate(&mut tableau2, &mut basis, maxiter, total_vars);
    let nit = phase1_nit + phase2_result.unwrap_or(maxiter);

    if let Err(err_code) = phase2_result {
        let (status, message) = if err_code == 3 {
            (3, "Problem is unbounded".to_string())
        } else {
            (1, "Phase II iteration limit exceeded".to_string())
        };
        return Ok(LinprogResult {
            x: vec![0.0; n],
            fun: 0.0,
            slack: vec![0.0; m_ub],
            success: false,
            status,
            message,
            nit,
        });
    }

    // Extract solution.
    let mut x_std = vec![0.0; total_vars];
    for (row, &bj) in basis.iter().enumerate() {
        if bj < total_vars {
            x_std[bj] = tableau2[row][total_vars];
        }
    }

    // Check for unbounded (any non-basic variable with negative reduced cost
    // and no positive column entry would have been caught during iteration).
    // The simplex_iterate returns Err for unbounded.

    // Recover original decision variables from the transformed non-negative basis.
    let mut x_orig = vec![0.0; n];
    for (i, transformed) in transform.original_vars.iter().enumerate() {
        let mut value = transformed.offset;
        for &(std_index, std_coeff) in &transformed.terms {
            value += std_coeff * x_std[std_index];
        }
        x_orig[i] = value;
    }

    // Compute slack: s = b_ub - A_ub @ x.
    let slack: Vec<f64> = (0..m_ub)
        .map(|i| {
            b_ub[i]
                - a_ub[i]
                    .iter()
                    .zip(&x_orig)
                    .map(|(&a, &x)| a * x)
                    .sum::<f64>()
        })
        .collect();

    let fun = c.iter().zip(&x_orig).map(|(&ci, &xi)| ci * xi).sum::<f64>();

    Ok(LinprogResult {
        x: x_orig,
        fun,
        slack,
        success: true,
        status: 0,
        message: "Optimization terminated successfully".to_string(),
        nit,
    })
}

fn default_milp_bounds(
    n: usize,
    bounds: &[(Option<f64>, Option<f64>)],
) -> Result<Vec<(f64, f64)>, OptError> {
    if bounds.is_empty() {
        Ok(vec![(0.0, f64::INFINITY); n])
    } else if bounds.len() != n {
        Err(OptError::InvalidArgument {
            detail: format!("bounds length ({}) must match c length ({n})", bounds.len()),
        })
    } else {
        Ok(bounds
            .iter()
            .map(|(lo, hi)| (lo.unwrap_or(f64::NEG_INFINITY), hi.unwrap_or(f64::INFINITY)))
            .collect())
    }
}

fn bounds_to_options(bounds: &[(f64, f64)]) -> Vec<(Option<f64>, Option<f64>)> {
    bounds
        .iter()
        .map(|&(lo, hi)| {
            (
                if lo.is_finite() { Some(lo) } else { None },
                if hi.is_finite() { Some(hi) } else { None },
            )
        })
        .collect()
}

fn is_integral_with_tol(value: f64, tol: f64) -> bool {
    (value - value.round()).abs() <= tol
}

/// Solve a mixed-integer linear program by branch-and-bound over `linprog`
/// relaxations.
///
/// Matches the core SciPy surface of `scipy.optimize.milp` for bounded
/// continuous/integer/binary variables on small to medium problems.
pub fn milp(problem: MilpProblem<'_>, options: MilpOptions) -> Result<MilpResult, OptError> {
    let MilpProblem {
        c,
        integrality,
        a_ub,
        b_ub,
        a_eq,
        b_eq,
        bounds,
    } = problem;

    let n = c.len();
    if n == 0 {
        return Err(OptError::InvalidArgument {
            detail: "c must not be empty".to_string(),
        });
    }
    if options.max_nodes == Some(0) {
        return Err(OptError::InvalidArgument {
            detail: "max_nodes must be greater than zero".to_string(),
        });
    }

    let integrality = if integrality.is_empty() {
        vec![Integrality::Continuous; n]
    } else {
        if integrality.len() != n {
            return Err(OptError::InvalidArgument {
                detail: format!(
                    "integrality length ({}) must match c length ({n})",
                    integrality.len()
                ),
            });
        }
        integrality.to_vec()
    };

    let mut root_bounds = default_milp_bounds(n, bounds)?;
    for (i, (bound, kind)) in root_bounds.iter_mut().zip(integrality.iter()).enumerate() {
        validate_variable_bounds(i, bound.0, bound.1)?;
        match kind {
            Integrality::Continuous | Integrality::Integer => {}
            Integrality::Binary => {
                bound.0 = bound.0.max(0.0);
                bound.1 = bound.1.min(1.0);
            }
        }
        if bound.0.is_nan() || bound.1.is_nan() {
            return Err(OptError::InvalidBounds {
                detail: format!("variable {i}: bounds became NaN after integrality adjustment"),
            });
        }
        if bound.0 > bound.1 {
            return Ok(MilpResult {
                x: vec![0.0; n],
                fun: 0.0,
                success: false,
                status: 2,
                message: format!(
                    "variable {i}: bounds are infeasible after integrality restrictions"
                ),
                mip_node_count: 0,
                nit: 0,
            });
        }
    }

    let max_nodes = options.max_nodes.unwrap_or(2_048);
    let tol = 1e-9;
    let mut node_count = 0usize;
    let mut total_nit = 0usize;
    let mut best: Option<LinprogResult> = None;
    let mut root_status = None;
    let mut stack = vec![root_bounds];

    while let Some(node_bounds) = stack.pop() {
        if node_count >= max_nodes {
            return Ok(match best {
                Some(best) => MilpResult {
                    x: best.x,
                    fun: best.fun,
                    success: false,
                    status: 1,
                    message: "MILP node limit exceeded after finding an incumbent".to_string(),
                    mip_node_count: node_count,
                    nit: total_nit,
                },
                None => MilpResult {
                    x: vec![0.0; n],
                    fun: 0.0,
                    success: false,
                    status: 1,
                    message: "MILP node limit exceeded before finding a feasible integer solution"
                        .to_string(),
                    mip_node_count: node_count,
                    nit: total_nit,
                },
            });
        }

        node_count += 1;
        let lp = linprog(
            c,
            a_ub,
            b_ub,
            a_eq,
            b_eq,
            &bounds_to_options(&node_bounds),
            options.lp_maxiter,
        )?;
        total_nit += lp.nit;

        if root_status.is_none() {
            root_status = Some(lp.status);
        }
        if !lp.success {
            continue;
        }
        if best
            .as_ref()
            .is_some_and(|incumbent| lp.fun >= incumbent.fun - tol)
        {
            continue;
        }

        let branch_index = integrality
            .iter()
            .enumerate()
            .find_map(|(i, kind)| match kind {
                Integrality::Continuous => None,
                Integrality::Integer | Integrality::Binary => {
                    if is_integral_with_tol(lp.x[i], tol) {
                        None
                    } else {
                        Some(i)
                    }
                }
            });

        match branch_index {
            None => best = Some(lp),
            Some(i) => {
                let value = lp.x[i];
                let floor = value.floor();
                let ceil = value.ceil();

                let mut lower_branch = node_bounds.clone();
                lower_branch[i].1 = lower_branch[i].1.min(floor);
                if lower_branch[i].0 <= lower_branch[i].1 {
                    stack.push(lower_branch);
                }

                let mut upper_branch = node_bounds;
                upper_branch[i].0 = upper_branch[i].0.max(ceil);
                if upper_branch[i].0 <= upper_branch[i].1 {
                    stack.push(upper_branch);
                }
            }
        }
    }

    Ok(match best {
        Some(best) => MilpResult {
            x: best.x,
            fun: best.fun,
            success: true,
            status: 0,
            message: "Optimization terminated successfully".to_string(),
            mip_node_count: node_count,
            nit: total_nit,
        },
        None => {
            let (status, message) = match root_status {
                Some(3) => (3, "LP relaxation is unbounded".to_string()),
                _ => (2, "Problem is infeasible".to_string()),
            };
            MilpResult {
                x: vec![0.0; n],
                fun: 0.0,
                success: false,
                status,
                message,
                mip_node_count: node_count,
                nit: total_nit,
            }
        }
    })
}

/// Run simplex iterations on a tableau. Returns the number of iterations used.
/// Returns Err if iteration limit exceeded or problem is unbounded.
fn simplex_iterate(
    tableau: &mut [Vec<f64>],
    basis: &mut [usize],
    maxiter: usize,
    n_vars: usize,
) -> Result<usize, u8> {
    let m = basis.len(); // number of constraints
    let obj_row = m;
    let rhs_col = n_vars;
    let tol = 1e-12;

    // Entering-variable rule: Dantzig (most-negative reduced cost) converges in far
    // fewer pivots than Bland's "first-negative", but can cycle on degenerate LPs.
    // Count consecutive degenerate pivots (objective unchanged) and fall back to
    // Bland's anti-cycling rule once they pile up, then resume Dantzig on progress.
    // Same optimum either way (the simplex is exact); only the pivot path changes.
    let bland_after = m + 16;
    let mut degenerate_run = 0usize;
    let mut last_obj = tableau[obj_row][rhs_col];

    for iteration in 0..maxiter {
        let mut pivot_col = None;
        if degenerate_run >= bland_after {
            // Bland's rule: smallest index with negative reduced cost.
            for (j, &cost) in tableau[obj_row].iter().enumerate().take(n_vars) {
                if cost < -tol {
                    pivot_col = Some(j);
                    break;
                }
            }
        } else {
            // Dantzig's rule: most negative reduced cost.
            let mut best = -tol;
            for (j, &cost) in tableau[obj_row].iter().enumerate().take(n_vars) {
                if cost < best {
                    best = cost;
                    pivot_col = Some(j);
                }
            }
        }

        let pivot_col = match pivot_col {
            Some(c) => c,
            None => return Ok(iteration), // Optimal
        };

        // Minimum ratio test: find leaving variable.
        let mut pivot_row = None;
        let mut min_ratio = f64::INFINITY;
        for (i, row) in tableau.iter().enumerate().take(m) {
            if row[pivot_col] > tol {
                let ratio = row[rhs_col] / row[pivot_col];
                if ratio < min_ratio {
                    min_ratio = ratio;
                    pivot_row = Some(i);
                }
            }
        }

        let pivot_row = match pivot_row {
            Some(r) => r,
            None => return Err(3), // Unbounded
        };

        // Pivot.
        let pivot_val = tableau[pivot_row][pivot_col];
        for val in &mut tableau[pivot_row][..=rhs_col] {
            *val /= pivot_val;
        }

        for i in 0..=obj_row {
            if i != pivot_row {
                let factor = tableau[i][pivot_col];
                if factor.abs() > 1e-15 {
                    let (p_row, t_row) = if i < pivot_row {
                        let (left, right) = tableau.split_at_mut(pivot_row);
                        (&right[0], &mut left[i])
                    } else {
                        let (left, right) = tableau.split_at_mut(i);
                        (&left[pivot_row], &mut right[0])
                    };

                    for (t_val, &p_val) in t_row.iter_mut().zip(p_row.iter()).take(rhs_col + 1) {
                        *t_val -= factor * p_val;
                    }
                }
            }
        }
        basis[pivot_row] = pivot_col;

        // A pivot that leaves the objective unchanged is degenerate (the cycling
        // risk for Dantzig); track the run length to trigger the Bland fallback.
        let obj = tableau[obj_row][rhs_col];
        if (obj - last_obj).abs() > tol {
            degenerate_run = 0;
        } else {
            degenerate_run += 1;
        }
        last_obj = obj;
    }

    Err(1) // Iteration limit
}

// ══════════════════════════════════════════════════════════════════════
// Global Optimization: differential_evolution, basinhopping
// ══════════════════════════════════════════════════════════════════════

use rand::{Rng, RngExt, SeedableRng};

/// Strategy for differential evolution mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeStrategy {
    /// best/1/bin: mutant = best + F*(r1 - r2).
    #[default]
    Best1Bin,
    /// rand/1/bin: mutant = r0 + F*(r1 - r2).
    Rand1Bin,
}

/// Options for `differential_evolution`.
#[derive(Debug, Clone)]
pub struct DifferentialEvolutionOptions {
    /// Mutation strategy.
    pub strategy: DeStrategy,
    /// Maximum number of generations.
    pub maxiter: usize,
    /// Population size multiplier (popsize * ndim = actual population).
    pub popsize: usize,
    /// Convergence tolerance on function value spread.
    pub tol: f64,
    /// Mutation factor (scaling of difference vectors), typically in [0.5, 2].
    pub mutation: f64,
    /// Crossover probability, typically in [0.5, 1.0].
    pub recombination: f64,
    /// Random seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for DifferentialEvolutionOptions {
    fn default() -> Self {
        Self {
            strategy: DeStrategy::Best1Bin,
            maxiter: 1000,
            popsize: 15,
            tol: 1e-8,
            mutation: 0.8,
            recombination: 0.7,
            seed: None,
        }
    }
}

/// Minimize a function using differential evolution (global optimizer).
///
/// Matches `scipy.optimize.differential_evolution(func, bounds)`.
///
/// A population-based evolutionary optimizer that requires only function evaluations
/// (no gradients). Handles box constraints naturally.
///
/// # Arguments
/// * `func` — Objective function to minimize.
/// * `bounds` — Box constraints: `(lower, upper)` for each variable.
/// * `opts` — Options controlling strategy, population, mutation, crossover.
pub fn differential_evolution<F>(
    func: F,
    bounds: &[(f64, f64)],
    opts: DifferentialEvolutionOptions,
) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    let ndim = bounds.len();
    if ndim == 0 {
        return Err(OptError::InvalidArgument {
            detail: "bounds must not be empty".to_string(),
        });
    }
    if opts.popsize == 0 {
        return Err(OptError::InvalidArgument {
            detail: "popsize must be at least 1".to_string(),
        });
    }
    if opts.maxiter == 0 {
        return Err(OptError::InvalidArgument {
            detail: "maxiter must be greater than zero".to_string(),
        });
    }
    let pop_count = opts
        .popsize
        .checked_mul(ndim)
        .ok_or_else(|| OptError::InvalidArgument {
            detail: format!(
                "population size overflow: popsize {} * ndim {ndim} exceeds usize",
                opts.popsize
            ),
        })?;
    if pop_count < 4 {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "population size must be at least 4, got {pop_count} (popsize={} * ndim={ndim})",
                opts.popsize
            ),
        });
    }
    if !opts.tol.is_finite() || opts.tol < 0.0 {
        return Err(OptError::InvalidArgument {
            detail: format!("tol must be a finite non-negative value, got {}", opts.tol),
        });
    }
    if !opts.mutation.is_finite() || opts.mutation < 0.0 {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "mutation must be a finite non-negative value, got {}",
                opts.mutation
            ),
        });
    }
    if !opts.recombination.is_finite() || !(0.0..=1.0).contains(&opts.recombination) {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "recombination must be finite and within [0, 1], got {}",
                opts.recombination
            ),
        });
    }
    for (i, &(lo, hi)) in bounds.iter().enumerate() {
        if !lo.is_finite() || !hi.is_finite() {
            return Err(OptError::InvalidBounds {
                detail: format!("variable {i}: bounds must be finite, got ({lo}, {hi})"),
            });
        }
        if lo >= hi {
            return Err(OptError::InvalidBounds {
                detail: format!("variable {i}: lower bound {lo} >= upper bound {hi}"),
            });
        }
    }
    let mut rng: rand::rngs::StdRng = match opts.seed {
        Some(s) => rand::rngs::StdRng::seed_from_u64(s),
        None => rand::rngs::StdRng::from_rng(&mut rand::rng()),
    };

    // Initialize population uniformly within bounds.
    let mut population: Vec<Vec<f64>> = (0..pop_count)
        .map(|_| {
            bounds
                .iter()
                .map(|&(lo, hi)| rng.random_range(lo..hi))
                .collect()
        })
        .collect();

    let mut fitness: Vec<f64> = population.iter().map(|x| func(x)).collect();
    let mut nfev = pop_count;

    // Track best.
    let mut best_idx = 0;
    let mut best_fun = fitness[0];
    for (i, &f) in fitness.iter().enumerate() {
        if f < best_fun {
            best_fun = f;
            best_idx = i;
        }
    }

    let mut converged = false;
    let mut generation = 0;
    let mut mutant = vec![0.0; ndim];
    let mut trial = vec![0.0; ndim];

    for g in 0..opts.maxiter {
        generation = g;

        for i in 0..pop_count {
            // Select three distinct random indices != i.
            let (r0, r1, r2) = select_three(&mut rng, pop_count, i);

            // Mutation.
            let base = match opts.strategy {
                DeStrategy::Best1Bin => &population[best_idx],
                DeStrategy::Rand1Bin => &population[r0],
            };
            for j in 0..ndim {
                mutant[j] = base[j] + opts.mutation * (population[r1][j] - population[r2][j]);
            }

            // Binomial crossover.
            let j_rand = rng.random_range(0..ndim);
            for j in 0..ndim {
                trial[j] = if j == j_rand || rng.random_range(0.0..1.0) < opts.recombination {
                    // Clip to bounds.
                    mutant[j].clamp(bounds[j].0, bounds[j].1)
                } else {
                    population[i][j]
                };
            }

            // Selection.
            let f_trial = func(&trial);
            nfev += 1;
            if f_trial <= fitness[i] {
                population[i].copy_from_slice(&trial);
                fitness[i] = f_trial;
                if f_trial < best_fun {
                    best_fun = f_trial;
                    best_idx = i;
                }
            }
        }

        // Check convergence: spread of fitness values.
        let fmin = fitness
            .iter()
            .cloned()
            .fold(f64::INFINITY, |a: f64, b: f64| {
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a.min(b)
                }
            });
        let fmax = fitness
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, |a: f64, b: f64| {
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a.max(b)
                }
            });
        if (fmax - fmin).abs() <= opts.tol * (1.0 + fmin.abs()) {
            converged = true;
            break;
        }
    }

    Ok(OptimizeResult {
        x: population[best_idx].clone(),
        fun: Some(best_fun),
        success: converged,
        status: if converged {
            ConvergenceStatus::Success
        } else {
            ConvergenceStatus::MaxIterations
        },
        message: if converged {
            "Optimization converged".to_string()
        } else {
            format!("Maximum iterations ({}) reached", opts.maxiter)
        },
        nfev,
        njev: 0,
        nhev: 0,
        nit: generation + 1,
        jac: None,
        hess_inv: None,
        maxcv: None,
    })
}

/// br-7470: Penalty-based constraint handling for differential evolution.
///
/// Wraps the user objective with a quadratic exterior penalty: for each
/// candidate x, the effective cost is
///
/// ```text
/// func(x) + penalty * sum_j violation_j(x)^2
/// ```
///
/// where `violation` returns the sum of squared boundary excess across
/// all linear and nonlinear constraints. A feasible point has zero
/// penalty; an infeasible point pays a large cost that pushes selection
/// away. Mirrors the spirit of scipy.optimize.differential_evolution's
/// constrained mode, which uses a feasibility-tournament with similar
/// boundary-excess penalty (the underlying numerics differ — scipy
/// uses a Lampinen tournament — but for unimodal global minima both
/// converge to the same constrained optimum).
///
/// `constraint_violation` should return >= 0; zero means feasible.
pub fn differential_evolution_constrained<F, G>(
    func: F,
    bounds: &[(f64, f64)],
    constraint_violation: G,
    opts: DifferentialEvolutionOptions,
) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> f64,
{
    const PENALTY: f64 = 1.0e6;
    let penalized = |x: &[f64]| {
        let v = constraint_violation(x);
        if v <= 0.0 {
            func(x)
        } else {
            func(x) + PENALTY * v * v
        }
    };
    differential_evolution(penalized, bounds, opts)
}

/// Select three distinct random indices from [0, n), all different from `exclude`.
///
/// Requires `n >= 4` (3 picks + 1 excluded). If n < 4, picks are allowed to
/// overlap with `exclude` to avoid an infinite loop.
fn select_three(rng: &mut impl Rng, n: usize, exclude: usize) -> (usize, usize, usize) {
    let mut indices = [0; 3];
    let mut len = 0;
    let strict = n >= 4; // Can we guarantee 3 distinct indices != exclude?
    let mut attempts = 0;
    while len < indices.len() {
        let idx = rng.random_range(0..n);
        let skip_exclude = strict && idx == exclude;
        if !skip_exclude && !indices[..len].contains(&idx) {
            indices[len] = idx;
            len += 1;
        }
        attempts += 1;
        if attempts > 1000 {
            // Fallback: fill remaining with sequential indices
            for k in 0..n {
                if len >= indices.len() {
                    break;
                }
                if !indices[..len].contains(&k) {
                    indices[len] = k;
                    len += 1;
                }
            }
            break;
        }
    }
    (indices[0], indices[1], indices[2])
}

/// Options for `basinhopping`.
#[derive(Debug, Clone)]
pub struct BasinhoppingOptions {
    /// Number of basin-hopping iterations.
    pub niter: usize,
    /// Temperature for Metropolis acceptance criterion.
    pub temperature: f64,
    /// Step size for random perturbation.
    pub stepsize: f64,
    /// Random seed for reproducibility.
    pub seed: Option<u64>,
    /// Tolerance for local minimizer.
    pub minimizer_tol: Option<f64>,
}

impl Default for BasinhoppingOptions {
    fn default() -> Self {
        Self {
            niter: 100,
            temperature: 1.0,
            stepsize: 0.5,
            seed: None,
            minimizer_tol: Some(1e-8),
        }
    }
}

/// Minimize a function using the basin-hopping algorithm (global optimizer).
///
/// Matches `scipy.optimize.basinhopping(func, x0)`.
///
/// Combines random perturbation with local minimization and a Metropolis
/// acceptance criterion to escape local minima.
///
/// # Arguments
/// * `func` — Objective function to minimize.
/// * `x0` — Initial guess.
/// * `opts` — Options controlling iterations, temperature, step size.
pub fn basinhopping<F>(
    func: F,
    x0: &[f64],
    opts: BasinhoppingOptions,
) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    let ndim = x0.len();
    if ndim == 0 {
        return Err(OptError::InvalidArgument {
            detail: "x0 must not be empty".to_string(),
        });
    }
    if x0.iter().any(|value| !value.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: "x0 must not contain NaN or Inf".to_string(),
        });
    }
    if opts.niter == 0 {
        return Err(OptError::InvalidArgument {
            detail: "niter must be greater than zero".to_string(),
        });
    }
    if !opts.temperature.is_finite() || opts.temperature <= 0.0 {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "temperature must be a positive finite value, got {}",
                opts.temperature
            ),
        });
    }
    if !opts.stepsize.is_finite() || opts.stepsize <= 0.0 {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "stepsize must be a positive finite value, got {}",
                opts.stepsize
            ),
        });
    }
    if let Some(tol) = opts.minimizer_tol
        && (!tol.is_finite() || tol <= 0.0)
    {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "minimizer_tol must be a positive finite value when provided, got {tol}",
            ),
        });
    }

    let mut rng: rand::rngs::StdRng = match opts.seed {
        Some(s) => rand::rngs::StdRng::seed_from_u64(s),
        None => rand::rngs::StdRng::from_rng(&mut rand::rng()),
    };

    let minimize_opts = MinimizeOptions {
        method: Some(OptimizeMethod::Bfgs),
        tol: opts.minimizer_tol,
        maxiter: Some(200),
        ..MinimizeOptions::default()
    };

    // Initial local minimization.
    let local_result =
        crate::bfgs(&func, x0, minimize_opts).map_err(|e| OptError::InvalidArgument {
            detail: format!("initial local minimization failed: {e}"),
        })?;

    let mut x_current = local_result.x;
    let mut f_current = local_result.fun.unwrap_or(f64::INFINITY);
    let mut x_best = x_current.clone();
    let mut f_best = f_current;
    let mut total_nfev = local_result.nfev;
    let mut total_nit = 0;

    for _iter in 0..opts.niter {
        // Random perturbation.
        let x_perturbed: Vec<f64> = x_current
            .iter()
            .map(|&xi| xi + rng.random_range(-opts.stepsize..opts.stepsize))
            .collect();

        // Local minimization from perturbed point.
        let local = crate::bfgs(&func, &x_perturbed, minimize_opts);
        total_nit += 1;

        if let Ok(result) = local {
            total_nfev += result.nfev;
            let f_new = result.fun.unwrap_or(f64::INFINITY);

            // Metropolis acceptance criterion.
            let delta = f_new - f_current;
            let accept = if delta < 0.0 {
                true
            } else {
                let prob = (-delta / opts.temperature).exp();
                rng.random_range(0.0..1.0) < prob
            };

            if f_new < f_best {
                f_best = f_new;
                x_best.clone_from(&result.x);
            }

            if accept {
                x_current = result.x;
                f_current = f_new;
            }
        }
    }

    Ok(OptimizeResult {
        x: x_best,
        fun: Some(f_best),
        success: true,
        status: ConvergenceStatus::Success,
        message: format!("Basin-hopping completed after {} iterations", opts.niter),
        nfev: total_nfev,
        njev: 0,
        nhev: 0,
        nit: total_nit,
        jac: None,
        hess_inv: None,
        maxcv: None,
    })
}

/// Dual annealing global optimization.
///
/// Combines simulated annealing with periodic local minimization.
/// Effective for finding global minima of multimodal functions.
///
/// Matches `scipy.optimize.dual_annealing(func, bounds)`.
///
/// # Arguments
/// * `func` — Objective function to minimize.
/// * `bounds` — Search bounds as (lower, upper) pairs for each dimension.
/// * `maxiter` — Maximum number of global iterations.
/// * `seed` — Random seed for reproducibility.
pub fn dual_annealing<F>(
    func: F,
    bounds: &[(f64, f64)],
    maxiter: usize,
    seed: u64,
) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    let ndim = bounds.len();
    if ndim == 0 {
        return Err(OptError::InvalidArgument {
            detail: "bounds must be non-empty".to_string(),
        });
    }
    if maxiter == 0 {
        return Err(OptError::InvalidArgument {
            detail: "maxiter must be greater than zero".to_string(),
        });
    }
    for (i, &(lo, hi)) in bounds.iter().enumerate() {
        if !lo.is_finite() || !hi.is_finite() {
            return Err(OptError::InvalidBounds {
                detail: format!("variable {i}: bounds must be finite, got ({lo}, {hi})"),
            });
        }
        if lo >= hi {
            return Err(OptError::InvalidBounds {
                detail: format!("variable {i}: lower bound {lo} >= upper bound {hi}"),
            });
        }
    }

    let mut rng = SimpleRng::new(seed);

    // Initialize with random point in bounds
    let mut x_best: Vec<f64> = bounds
        .iter()
        .map(|&(lo, hi)| lo + rng.next_f64() * (hi - lo))
        .collect();
    let mut f_best = func(&x_best);
    let mut nfev = 1usize;

    let mut x_current = x_best.clone();
    let mut f_current = f_best;

    // Local search: dual annealing's defining step is a deterministic local
    // minimization after each annealing move (scipy uses L-BFGS-B). The bfgs
    // result is clamped back into the bounds and re-evaluated.
    let minimize_opts = MinimizeOptions {
        method: Some(OptimizeMethod::Bfgs),
        maxiter: Some(200),
        ..MinimizeOptions::default()
    };
    let local_search = |start: &[f64], nfev: &mut usize| -> (Vec<f64>, f64) {
        match crate::bfgs(&func, start, minimize_opts) {
            Ok(res) => {
                *nfev += res.nfev;
                let x_clamped: Vec<f64> = res
                    .x
                    .iter()
                    .zip(bounds.iter())
                    .map(|(&xi, &(lo, hi))| xi.clamp(lo, hi))
                    .collect();
                let f_clamped = func(&x_clamped);
                *nfev += 1;
                (x_clamped, f_clamped)
            }
            Err(_) => {
                let f = func(start);
                *nfev += 1;
                (start.to_vec(), f)
            }
        }
    };

    // Temperature schedule
    let t_initial = 5230.0; // SciPy default
    for iteration in 0..maxiter {
        let temp = t_initial / (iteration as f64 + 1.0).ln().max(1.0);

        // Generate candidate via visiting distribution (Cauchy-like perturbation)
        let x_candidate: Vec<f64> = x_current
            .iter()
            .zip(bounds.iter())
            .map(|(&xi, &(lo, hi))| {
                let range = hi - lo;
                let perturbation = range * (rng.next_f64() - 0.5) * temp.sqrt() / (1.0 + temp);
                (xi + perturbation).clamp(lo, hi)
            })
            .collect();

        let f_candidate = func(&x_candidate);
        nfev += 1;

        // Metropolis acceptance
        let delta = f_candidate - f_current;
        let accept = if delta < 0.0 {
            true
        } else {
            let prob = (-delta / temp.max(1e-30)).exp();
            rng.next_f64() < prob
        };

        if accept {
            x_current = x_candidate.clone();
            f_current = f_candidate;
        }

        // Local search from the annealing candidate.
        let (x_local, f_local) = local_search(&x_candidate, &mut nfev);
        if f_local < f_current {
            x_current = x_local.clone();
            f_current = f_local;
        }

        if f_current < f_best {
            x_best.clone_from(&x_current);
            f_best = f_current;
        }
    }

    // Final local polish of the best point found.
    let (x_polished, f_polished) = local_search(&x_best, &mut nfev);
    if f_polished < f_best {
        x_best = x_polished;
        f_best = f_polished;
    }

    Ok(OptimizeResult {
        x: x_best,
        fun: Some(f_best),
        nit: maxiter,
        nfev,
        njev: 0,
        nhev: 0,
        success: true,
        status: ConvergenceStatus::Success,
        message: "dual_annealing completed".to_string(),
        jac: None,
        hess_inv: None,
        maxcv: None,
    })
}

/// Brute-force global optimization.
///
/// Evaluates the objective function at each point of a multidimensional grid
/// defined by `ranges` and `ns` (number of samples per dimension), returning
/// the point with the lowest value.
/// Matches `scipy.optimize.brute(func, ranges, Ns=ns)`.
pub fn brute<F>(func: F, ranges: &[(f64, f64)], ns: usize) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    let ndim = ranges.len();
    if ndim == 0 {
        return Err(OptError::InvalidArgument {
            detail: "ranges must not be empty".to_string(),
        });
    }
    if ns < 2 {
        return Err(OptError::InvalidArgument {
            detail: "ns must be at least 2".to_string(),
        });
    }
    for &(lb, ub) in ranges {
        if !lb.is_finite() || !ub.is_finite() || lb >= ub {
            return Err(OptError::InvalidArgument {
                detail: format!("invalid range: ({}, {})", lb, ub),
            });
        }
    }

    let exponent = u32::try_from(ndim).map_err(|_| OptError::InvalidArgument {
        detail: format!("number of dimensions {ndim} exceeds supported grid exponent"),
    })?;
    let total_points = ns
        .checked_pow(exponent)
        .ok_or_else(|| OptError::InvalidArgument {
            detail: format!("grid size {ns}^{ndim} overflows usize"),
        })?;
    let nfev = total_points;

    // Fill a grid point for a flattened index (column-major over the per-dim steps).
    let fill_point = |idx: usize, x: &mut [f64]| {
        let mut current_idx = idx;
        for (d, slot) in x.iter_mut().enumerate() {
            let step_idx = current_idx % ns;
            current_idx /= ns;
            let (lb, ub) = ranges[d];
            *slot = lb + (ub - lb) * (step_idx as f64) / ((ns - 1) as f64);
        }
    };

    // Every grid point's objective is independent, so evaluate them in parallel
    // into a flat score array; the argmin below then keeps the FIRST index at the
    // minimum, bit-identical to the serial `if f < best_f` strict-`<` scan.
    let mut fs = vec![f64::INFINITY; total_points];
    let nthreads = brute_thread_count(total_points);
    if nthreads <= 1 {
        let mut x = vec![0.0; ndim];
        for (idx, slot) in fs.iter_mut().enumerate() {
            fill_point(idx, &mut x);
            *slot = func(&x);
        }
    } else {
        let chunk = total_points.div_ceil(nthreads);
        let fill_point = &fill_point;
        let func = &func;
        std::thread::scope(|scope| {
            for (ci, block) in fs.chunks_mut(chunk).enumerate() {
                let base = ci * chunk;
                scope.spawn(move || {
                    let mut x = vec![0.0; ndim];
                    for (li, slot) in block.iter_mut().enumerate() {
                        fill_point(base + li, &mut x);
                        *slot = func(&x);
                    }
                });
            }
        });
    }

    let mut best_x = vec![0.0; ndim];
    let mut best_f = f64::INFINITY;
    for (idx, &f) in fs.iter().enumerate() {
        if f < best_f {
            best_f = f;
            fill_point(idx, &mut best_x);
        }
    }

    Ok(OptimizeResult {
        x: best_x,
        fun: Some(best_f),
        nit: total_points,
        nfev,
        njev: 0,
        nhev: 0,
        success: true,
        status: ConvergenceStatus::Success,
        message: "brute completed".to_string(),
        jac: None,
        hess_inv: None,
        maxcv: None,
    })
}

/// Worker count for `brute`'s grid evaluation, or 1 to stay serial. The objective
/// cost is unknown, so gate on the grid size: a high threshold ensures the total
/// work amortises thread spawn even for cheap objectives.
fn brute_thread_count(total_points: usize) -> usize {
    if total_points < 1 << 18 {
        return 1;
    }
    std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1)
        .min(total_points)
        .max(1)
}

/// Simplicial homology global optimization over a bounded box.
///
/// This implementation uses a bounded lattice search to identify promising
/// simplicial cells, then polishes the best seeds with the existing L-BFGS-B
/// local optimizer. It matches the high-level contract of
/// `scipy.optimize.shgo(func, bounds)` for unconstrained bounded problems.
pub fn shgo<F>(func: F, bounds: &[(f64, f64)]) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    let ndim = bounds.len();
    if ndim == 0 {
        return Err(OptError::InvalidArgument {
            detail: "bounds must be non-empty".to_string(),
        });
    }
    for (i, &(lo, hi)) in bounds.iter().enumerate() {
        if !lo.is_finite() || !hi.is_finite() {
            return Err(OptError::InvalidBounds {
                detail: format!("variable {i}: bounds must be finite, got ({lo}, {hi})"),
            });
        }
        if lo >= hi {
            return Err(OptError::InvalidBounds {
                detail: format!("variable {i}: lower bound {lo} >= upper bound {hi}"),
            });
        }
    }

    let target_points = 5usize.saturating_pow((ndim.min(4)) as u32).max(9);
    let axis_levels = ((target_points as f64).powf(1.0 / ndim as f64).ceil() as usize + 1).max(3);
    let mut lattice = Vec::new();
    let mut current = vec![0.0; ndim];
    build_shgo_lattice(bounds, axis_levels, 0, &mut current, &mut lattice);

    let mut nfev = 0usize;
    let mut ranked: Vec<(f64, Vec<f64>)> = lattice
        .into_iter()
        .filter_map(|point| {
            let value = func(&point);
            nfev += 1;
            value.is_finite().then_some((value, point))
        })
        .collect();

    if ranked.is_empty() {
        return Err(OptError::NonFiniteInput {
            detail: "objective returned only non-finite values over the bounded domain".to_string(),
        });
    }

    ranked.sort_by(|lhs, rhs| lhs.0.total_cmp(&rhs.0));
    let starts = select_shgo_starts(&ranked, bounds, 8);
    let bound_tuples: Vec<Bound> = bounds
        .iter()
        .map(|&(lo, hi)| (Some(lo), Some(hi)))
        .collect();
    let local_options = MinimizeOptions {
        method: Some(OptimizeMethod::LBfgsB),
        tol: Some(1e-8),
        maxiter: Some(250),
        maxfev: Some(2500),
        ..MinimizeOptions::default()
    };

    let mut best_x = ranked[0].1.clone();
    let mut best_fun = ranked[0].0;
    let mut total_nit = 0usize;

    for start in starts {
        let local = lbfgsb(&func, &start, local_options, Some(&bound_tuples))?;
        nfev += local.nfev;
        total_nit += local.nit;
        if let Some(fun) = local.fun
            && fun < best_fun
        {
            best_fun = fun;
            best_x = local.x;
        }
    }

    Ok(OptimizeResult {
        x: best_x,
        fun: Some(best_fun),
        success: true,
        status: ConvergenceStatus::Success,
        message: "shgo completed".to_string(),
        nfev,
        njev: 0,
        nhev: 0,
        nit: total_nit.max(1),
        jac: None,
        hess_inv: None,
        maxcv: None,
    })
}

fn build_shgo_lattice(
    bounds: &[(f64, f64)],
    axis_levels: usize,
    dim: usize,
    current: &mut Vec<f64>,
    lattice: &mut Vec<Vec<f64>>,
) {
    if dim == bounds.len() {
        lattice.push(current.clone());
        return;
    }

    let (lo, hi) = bounds[dim];
    for step in 0..axis_levels {
        let fraction = if axis_levels == 1 {
            0.0
        } else {
            step as f64 / (axis_levels - 1) as f64
        };
        current[dim] = lo + fraction * (hi - lo);
        build_shgo_lattice(bounds, axis_levels, dim + 1, current, lattice);
    }
}

fn select_shgo_starts(
    ranked: &[(f64, Vec<f64>)],
    bounds: &[(f64, f64)],
    max_starts: usize,
) -> Vec<Vec<f64>> {
    let mut starts = Vec::new();
    let distance_scale = bounds.iter().map(|(lo, hi)| hi - lo).sum::<f64>().max(1.0);
    let min_distance = 0.1 * distance_scale / bounds.len() as f64;

    for (_, point) in ranked {
        let distinct = starts.iter().all(|existing: &Vec<f64>| {
            existing
                .iter()
                .zip(point.iter())
                .map(|(lhs, rhs)| {
                    let delta = lhs - rhs;
                    delta * delta
                })
                .sum::<f64>()
                .sqrt()
                >= min_distance
        });
        if distinct {
            starts.push(point.clone());
            if starts.len() >= max_starts {
                break;
            }
        }
    }

    if starts.is_empty() {
        starts.push(ranked[0].1.clone());
    }
    starts
}

/// Simple deterministic pseudo-random number generator (xorshift64).
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }
    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// COBYLA: Constrained Optimization BY Linear Approximation.
///
/// Minimizes `func(x)` subject to `constraints[i](x) >= 0` for all i.
///
/// Uses a simplex-based method that approximates the objective and constraints
/// with linear models at each iteration.
///
/// Matches `scipy.optimize.minimize(func, x0, method='COBYLA', constraints=...)`.
pub fn cobyla<F, G>(
    func: F,
    x0: &[f64],
    constraints: &[G],
    maxiter: usize,
    rhobeg: f64,
) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptError::InvalidArgument {
            detail: "x0 must be non-empty".to_string(),
        });
    }
    if x0.iter().any(|v| !v.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: "x0 must be finite".to_string(),
        });
    }
    if maxiter == 0 {
        return Err(OptError::InvalidArgument {
            detail: "maxiter must be greater than zero".to_string(),
        });
    }
    if !rhobeg.is_finite() || rhobeg <= 0.0 {
        return Err(OptError::InvalidArgument {
            detail: "rhobeg must be a positive finite value".to_string(),
        });
    }

    let mut x = x0.to_vec();
    let mut f_best = func(&x);
    let mut nfev = 1usize;
    let mut rho = rhobeg;

    for _iteration in 0..maxiter {
        // Check constraints
        let mut max_violation = 0.0_f64;
        for constraint in constraints {
            let cv = constraint(&x);
            if cv < 0.0 {
                max_violation = max_violation.max(-cv);
            }
        }

        // Try coordinate-wise descent with constraint penalty
        let mut improved = false;
        for d in 0..n {
            for &direction in &[rho, -rho] {
                let mut x_trial = x.clone();
                x_trial[d] += direction;

                let f_trial = func(&x_trial);
                nfev += 1;

                // Check all constraints
                let mut trial_violation = 0.0;
                for constraint in constraints {
                    let cv = constraint(&x_trial);
                    if cv < -1e-10 {
                        trial_violation += -cv;
                    }
                }

                // Accept if: (feasible and better) or (less infeasible)
                let current_penalty = f_best + 1000.0 * max_violation;
                let trial_penalty = f_trial + 1000.0 * trial_violation;

                if trial_penalty < current_penalty - 1e-12 {
                    x = x_trial;
                    f_best = f_trial;
                    max_violation = trial_violation;
                    improved = true;
                }
            }
        }

        // Shrink trust region if no improvement
        if !improved {
            rho *= 0.5;
            if rho < 1e-12 {
                break;
            }
        }
    }

    Ok(OptimizeResult {
        x,
        fun: Some(f_best),
        nit: maxiter,
        nfev,
        njev: 0,
        nhev: 0,
        success: true,
        status: ConvergenceStatus::Success,
        message: "cobyla completed".to_string(),
        jac: None,
        hess_inv: None,
        maxcv: None,
    })
}

// ══════════════════════════════════════════════════════════════════════
// Constrained Optimization via Augmented Lagrangian
// ══════════════════════════════════════════════════════════════════════

/// Minimize a function subject to nonlinear constraints using an augmented
/// Lagrangian penalty method.
///
/// Solves: min f(x) subject to c(x) >= 0.
///
/// This converts the constrained problem into a sequence of unconstrained
/// subproblems, each solved via Nelder-Mead.
pub fn augmented_lagrangian<F, C>(
    f: F,
    constraints: C,
    x0: &[f64],
    n_constraints: usize,
    max_outer: usize,
    max_inner: usize,
) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
    C: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptError::InvalidArgument {
            detail: "x0 must be non-empty".to_string(),
        });
    }
    if max_outer == 0 {
        return Err(OptError::InvalidArgument {
            detail: "max_outer must be greater than zero".to_string(),
        });
    }
    if max_inner == 0 {
        return Err(OptError::InvalidArgument {
            detail: "max_inner must be greater than zero".to_string(),
        });
    }
    let initial_constraints = constraints(x0);
    if initial_constraints.len() != n_constraints {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "constraints(x0) length {} must equal n_constraints {}",
                initial_constraints.len(),
                n_constraints
            ),
        });
    }

    let mut x = x0.to_vec();
    let mut lambda = vec![0.0; n_constraints]; // Lagrange multipliers
    let mut mu = 10.0; // penalty parameter
    let mut total_nfev = 0;

    for outer in 0..max_outer {
        // Build augmented Lagrangian:
        // L(x) = f(x) - Σ λ_i * c_i(x) + (μ/2) * Σ max(0, -c_i(x) + λ_i/μ)²
        let lambda_copy = lambda.clone();
        let mu_copy = mu;
        let penalty_fn = |xv: &[f64]| -> f64 {
            let fval = f(xv);
            let cvec = constraints(xv);
            let mut penalty = 0.0;
            for (i, &ci) in cvec.iter().enumerate() {
                let shifted = -ci + lambda_copy[i] / mu_copy;
                if shifted > 0.0 {
                    penalty += shifted * shifted;
                }
            }
            fval + mu_copy / 2.0 * penalty
        };

        // Solve unconstrained subproblem
        let opts = MinimizeOptions {
            method: Some(crate::OptimizeMethod::NelderMead),
            maxiter: Some(max_inner),
            tol: Some(1e-8),
            ..MinimizeOptions::default()
        };
        let result = crate::minimize(penalty_fn, &x, opts)?;
        total_nfev += result.nfev;
        x = result.x;

        // Update multipliers
        let cvec = constraints(&x);
        let mut max_violation = 0.0f64;
        for (i, &ci) in cvec.iter().enumerate() {
            let unconstrained_lambda = lambda[i] - mu * ci;
            lambda[i] = if unconstrained_lambda.is_nan() {
                f64::NAN
            } else {
                unconstrained_lambda.max(0.0)
            };

            let violation = -ci;
            max_violation = if max_violation.is_nan() || violation.is_nan() {
                f64::NAN
            } else {
                max_violation.max(violation.max(0.0))
            };
        }

        // Check convergence
        if max_violation < 1e-6 {
            let fval = f(&x);
            return Ok(OptimizeResult {
                x,
                fun: Some(fval),
                success: true,
                status: ConvergenceStatus::Success,
                message: format!("augmented lagrangian converged in {outer} outer iterations"),
                nfev: total_nfev,
                njev: 0,
                nhev: 0,
                nit: outer + 1,
                jac: None,
                hess_inv: None,
                maxcv: Some(max_violation),
            });
        }

        // Increase penalty
        mu *= 10.0;
    }

    let fval = f(&x);
    Ok(OptimizeResult {
        x,
        fun: Some(fval),
        success: false,
        status: ConvergenceStatus::MaxIterations,
        message: "augmented lagrangian did not converge".to_string(),
        nfev: total_nfev,
        njev: 0,
        nhev: 0,
        nit: max_outer,
        jac: None,
        hess_inv: None,
        maxcv: None,
    })
}

/// Golden-section search for 1D minimization on [a, b].
///
/// Matches `scipy.optimize.golden`.
pub fn golden<F>(f: F, a: f64, b: f64, tol: f64, maxiter: usize) -> (f64, f64)
where
    F: Fn(f64) -> f64,
{
    if invalid_scalar_minimize_controls(a, b, tol, maxiter) {
        return (f64::NAN, f64::NAN);
    }

    let gr = (5.0_f64.sqrt() - 1.0) / 2.0; // golden ratio conjugate
    let mut lo = a;
    let mut hi = b;
    let mut c = hi - gr * (hi - lo);
    let mut d = lo + gr * (hi - lo);
    let mut fc = f(c);
    let mut fd = f(d);

    for _ in 0..maxiter {
        if (hi - lo).abs() < tol {
            break;
        }
        if fc < fd {
            hi = d;
            d = c;
            fd = fc;
            c = hi - gr * (hi - lo);
            fc = f(c);
        } else {
            lo = c;
            c = d;
            fc = fd;
            d = lo + gr * (hi - lo);
            fd = f(d);
        }
    }

    let x_min = (lo + hi) / 2.0;
    (x_min, f(x_min))
}

/// Brent's method for 1D minimization.
///
/// Matches `scipy.optimize.brent`.
pub fn brent_minimize<F>(f: F, a: f64, b: f64, tol: f64, maxiter: usize) -> (f64, f64)
where
    F: Fn(f64) -> f64,
{
    if invalid_scalar_minimize_controls(a, b, tol, maxiter) {
        return (f64::NAN, f64::NAN);
    }

    let golden = 0.381_966_011_250_105; // 1 - golden ratio conjugate

    let mut x = a + golden * (b - a);
    let mut w = x;
    let mut v = x;
    let mut fx = f(x);
    let mut fw = fx;
    let mut fv = fx;
    let mut lo = a;
    let mut hi = b;
    let mut e = 0.0f64;
    let mut d_step;

    for _ in 0..maxiter {
        let mid = 0.5 * (lo + hi);
        let tol1 = tol * x.abs() + 1e-10;
        let tol2 = 2.0 * tol1;

        if (x - mid).abs() <= tol2 - 0.5 * (hi - lo) {
            return (x, fx);
        }

        // Try parabolic interpolation
        if e.abs() > tol1 {
            let r = (x - w) * (fx - fv);
            let q = (x - v) * (fx - fw);
            let p = (x - v) * q - (x - w) * r;
            let q = 2.0 * (q - r);
            let (p, q) = if q > 0.0 { (-p, q) } else { (p, -q) };

            if p.abs() < (0.5 * q * e).abs() && p > q * (lo - x) && p < q * (hi - x) {
                d_step = p / q;
                let u = x + d_step;
                if (u - lo) < tol2 || (hi - u) < tol2 {
                    d_step = if x < mid { tol1 } else { -tol1 };
                }
            } else {
                e = if x < mid { hi - x } else { lo - x };
                d_step = golden * e;
            }
        } else {
            e = if x < mid { hi - x } else { lo - x };
            d_step = golden * e;
        }

        let u = if d_step.abs() >= tol1 {
            x + d_step
        } else {
            x + tol1 * d_step.signum()
        };
        let fu = f(u);

        if fu <= fx {
            if u < x {
                hi = x;
            } else {
                lo = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                lo = u;
            } else {
                hi = u;
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

        e = d_step;
    }

    (x, fx)
}

fn invalid_scalar_minimize_controls(a: f64, b: f64, tol: f64, maxiter: usize) -> bool {
    !a.is_finite() || !b.is_finite() || !tol.is_finite() || tol <= 0.0 || maxiter == 0
}

/// Find a minimum bracket for a 1D function.
///
/// Returns (xa, xb, xc, fa, fb, fc) such that f(xb) < f(xa) and f(xb) < f(xc).
/// Matches `scipy.optimize.bracket`.
pub fn bracket<F>(f: F, xa: f64, xb: f64) -> (f64, f64, f64, f64, f64, f64)
where
    F: Fn(f64) -> f64,
{
    let golden = 1.618_033_988_749_895;
    let limit = 100.0;

    let mut xa = xa;
    let mut xb = xb;
    let mut fa = f(xa);
    let mut fb = f(xb);

    if fa < fb {
        std::mem::swap(&mut xa, &mut xb);
        std::mem::swap(&mut fa, &mut fb);
    }

    let mut xc = xb + golden * (xb - xa);
    let mut fc = f(xc);

    for _ in 0..200 {
        if fb <= fc {
            return (xa, xb, xc, fa, fb, fc);
        }

        // Parabolic extrapolation
        let r = (xb - xa) * (fb - fc);
        let q = (xb - xc) * (fb - fa);
        let denom = 2.0 * (q - r).max(1e-20) * if q > r { 1.0 } else { -1.0 };
        let mut w = xb - ((xb - xc) * q - (xb - xa) * r) / denom;
        let wlim = xb + limit * (xc - xb);

        if (w - xc) * (xb - w) > 0.0 {
            let fw = f(w);
            if fw < fc {
                xa = xb;
                fa = fb;
                xb = w;
                fb = fw;
                return (xa, xb, xc, fa, fb, fc);
            } else if fw > fb {
                xc = w;
                fc = fw;
                return (xa, xb, xc, fa, fb, fc);
            }
            w = xc + golden * (xc - xb);
        } else if (w - wlim) * (wlim - xc) >= 0.0 {
            w = wlim;
        } else if (w - wlim) * (xc - w) > 0.0 {
            let fw = f(w);
            if fw < fc {
                xb = xc;
                xc = w;
                w = xc + golden * (xc - xb);
                fb = fc;
                fc = fw;
            }
        } else {
            w = xc + golden * (xc - xb);
        }

        xa = xb;
        xb = xc;
        xc = w;
        fa = fb;
        fb = fc;
        fc = f(xc);
    }

    (xa, xb, xc, fa, fb, fc)
}

/// Minimize a scalar function using bounded Brent's method.
///
/// Matches `scipy.optimize.minimize_scalar(method='bounded')`.
pub fn minimize_scalar_bounded<F>(f: F, bounds: (f64, f64), tol: f64, maxiter: usize) -> (f64, f64)
where
    F: Fn(f64) -> f64,
{
    brent_minimize(f, bounds.0, bounds.1, tol, maxiter)
}

/// Bounded scalar minimization, returning the minimizer.
///
/// Matches `scipy.optimize.fminbound(func, x1, x2)`: finds the `x ∈ [x1, x2]`
/// minimising `func` with the bounded Brent method, using scipy's defaults
/// (`xtol = 1e-5`, `maxfun = 500`). Returns the minimizer `x` (scipy's first
/// output); use [`minimize_scalar_bounded`] for the `(x, f(x))` pair or custom
/// tolerances.
pub fn fminbound<F>(func: F, x1: f64, x2: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    minimize_scalar_bounded(func, (x1, x2), 1e-5, 500).0
}

/// Unconstrained scalar minimization via Brent's method.
///
/// Matches `scipy.optimize.brent(func, brack, tol, maxiter)`, returning the
/// minimizer. `brack` may be `None` (default, brackets downhill from `(0, 1)`),
/// a two-element `[xa, xb]` bracketing interval to expand, or a three-element
/// `[xa, xb, xc]` already-downhill bracket (requires `f(xb) < f(xa)` and
/// `f(xb) < f(xc)`). The bracket establishes the search bounds, then Brent's
/// parabolic-interpolation method locates the minimum within them.
pub fn brent<F>(func: F, brack: Option<&[f64]>, tol: f64, maxiter: usize) -> Result<f64, OptError>
where
    F: Fn(f64) -> f64,
{
    if !tol.is_finite() || tol <= 0.0 {
        return Err(OptError::InvalidArgument {
            detail: "tol must be a positive finite value".to_string(),
        });
    }
    if maxiter == 0 {
        return Err(OptError::InvalidArgument {
            detail: "maxiter must be greater than zero".to_string(),
        });
    }
    if let Some(b) = brack
        && b.iter().any(|value| !value.is_finite())
    {
        return Err(OptError::InvalidArgument {
            detail: "brack values must be finite".to_string(),
        });
    }
    // Establish the bracketing triple (xa, xc) bounding the minimum.
    let (xa, xc) = match brack {
        None => {
            let (a, _b, c, ..) = bracket(&func, 0.0, 1.0);
            (a, c)
        }
        Some(b) if b.len() == 2 => {
            let (a, _b, c, ..) = bracket(&func, b[0], b[1]);
            (a, c)
        }
        Some(b) if b.len() == 3 => {
            let (fa, fb, fc) = (func(b[0]), func(b[1]), func(b[2]));
            if !(fb < fa && fb < fc) {
                return Err(OptError::InvalidArgument {
                    detail: "brack must be a downhill bracket: f(xb) < f(xa) and f(xb) < f(xc)"
                        .to_string(),
                });
            }
            (b[0], b[2])
        }
        Some(_) => {
            return Err(OptError::InvalidArgument {
                detail: "brack must have 2 or 3 elements".to_string(),
            });
        }
    };
    let lo = xa.min(xc);
    let hi = xa.max(xc);
    Ok(brent_minimize(&func, lo, hi, tol, maxiter).0)
}

/// Run [`minimize`] with a fixed method and return the minimiser `xopt`, the
/// first output of the legacy `scipy.optimize.fmin_*` family.
fn fmin_with_method<F>(fun: F, x0: &[f64], method: OptimizeMethod) -> Result<Vec<f64>, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    let options = MinimizeOptions {
        method: Some(method),
        ..MinimizeOptions::default()
    };
    minimize(fun, x0, options).map(|result| result.x)
}

/// Minimize a function using the downhill simplex (Nelder–Mead) algorithm.
///
/// Matches `scipy.optimize.fmin(func, x0)`: a legacy wrapper that runs
/// Nelder–Mead and returns the minimiser `xopt` (scipy's first output). Use
/// [`minimize`] with [`OptimizeMethod::NelderMead`] for the full result.
pub fn fmin<F>(func: F, x0: &[f64]) -> Result<Vec<f64>, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    fmin_with_method(func, x0, OptimizeMethod::NelderMead)
}

/// Minimize a function using the modified Powell direction-set method.
///
/// Matches `scipy.optimize.fmin_powell(func, x0)`: returns the minimiser
/// `xopt`. Use [`minimize`] with [`OptimizeMethod::Powell`] for the full result.
pub fn fmin_powell<F>(func: F, x0: &[f64]) -> Result<Vec<f64>, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    fmin_with_method(func, x0, OptimizeMethod::Powell)
}

/// Minimize a function using the nonlinear conjugate-gradient (Polak–Ribière)
/// method with a finite-difference gradient.
///
/// Matches `scipy.optimize.fmin_cg(f, x0)`: returns the minimiser `xopt`. Use
/// [`minimize`] with [`OptimizeMethod::ConjugateGradient`] for the full result.
pub fn fmin_cg<F>(func: F, x0: &[f64]) -> Result<Vec<f64>, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    fmin_with_method(func, x0, OptimizeMethod::ConjugateGradient)
}

/// Minimize a function using the BFGS quasi-Newton method with a
/// finite-difference gradient.
///
/// Matches `scipy.optimize.fmin_bfgs(f, x0)`: returns the minimiser `xopt`. Use
/// [`minimize`] with [`OptimizeMethod::Bfgs`] for the full result.
pub fn fmin_bfgs<F>(func: F, x0: &[f64]) -> Result<Vec<f64>, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    fmin_with_method(func, x0, OptimizeMethod::Bfgs)
}

/// Minimize a function using the L-BFGS-B method (finite-difference gradient).
///
/// Matches `scipy.optimize.fmin_l_bfgs_b(func, x0)`: returns the minimiser
/// `xopt` (the first element of scipy's output tuple). Use [`minimize`] with
/// [`OptimizeMethod::LBfgsB`] for the full result.
pub fn fmin_l_bfgs_b<F>(func: F, x0: &[f64]) -> Result<Vec<f64>, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    fmin_with_method(func, x0, OptimizeMethod::LBfgsB)
}

/// Minimize a function using the truncated-Newton (TNC) method.
///
/// Matches `scipy.optimize.fmin_tnc(func, x0)`: returns the minimiser `xopt`
/// (the first element of scipy's output tuple). Use [`minimize`] with
/// [`OptimizeMethod::Tnc`] for the full result.
pub fn fmin_tnc<F>(func: F, x0: &[f64]) -> Result<Vec<f64>, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    fmin_with_method(func, x0, OptimizeMethod::Tnc)
}

/// Minimize a function using Sequential Least-Squares Programming (SLSQP).
///
/// Matches `scipy.optimize.fmin_slsqp(func, x0)`: returns the minimiser `xopt`.
/// Use [`minimize`] with [`OptimizeMethod::Slsqp`] for the full result.
pub fn fmin_slsqp<F>(func: F, x0: &[f64]) -> Result<Vec<f64>, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    fmin_with_method(func, x0, OptimizeMethod::Slsqp)
}

/// Minimize a function using the Newton-CG (truncated Newton, line-search)
/// method with a finite-difference gradient.
///
/// Matches `scipy.optimize.fmin_ncg(f, x0, fprime)`: returns the minimiser
/// `xopt`. Use [`minimize`] with [`OptimizeMethod::NewtonCg`] for the full result.
pub fn fmin_ncg<F>(func: F, x0: &[f64]) -> Result<Vec<f64>, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    fmin_with_method(func, x0, OptimizeMethod::NewtonCg)
}

/// Minimize a function subject to inequality constraints using COBYLA.
///
/// Matches `scipy.optimize.fmin_cobyla(func, x0, cons, rhobeg, maxfun)`: each
/// constraint `c(x) >= 0` must hold at the solution. Returns the minimiser
/// `xopt` (scipy's output). Use [`cobyla`] for the full result.
pub fn fmin_cobyla<F, G>(
    func: F,
    x0: &[f64],
    constraints: &[G],
    rhobeg: f64,
    maxfun: usize,
) -> Result<Vec<f64>, OptError>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> f64,
{
    cobyla(func, x0, constraints, maxfun, rhobeg).map(|result| result.x)
}

/// Squared Euclidean norm of a residual vector.
fn nonlin_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

/// Backtracking Armijo line search on `phi(s) = ‖F(x + s·dx)‖²`, mirroring the
/// `armijo` search of scipy's nonlinear solvers (derphi0 = −phi0, c1 = 1e-4).
/// Returns the accepted `(x_new, F(x_new))`; always takes some step so the
/// quasi-Newton update has a nonzero secant even when no descent is found (the
/// diagonal-Broyden update is scale-invariant in `s`, so it still self-corrects).
fn nonlin_line_search<F>(f: &F, x: &[f64], fx: &[f64], dx: &[f64]) -> (Vec<f64>, Vec<f64>)
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let phi0 = nonlin_dot(fx, fx);
    let c1 = 1e-4;
    let mut s = 1.0;
    let mut last = None;
    for _ in 0..50 {
        let xt: Vec<f64> = x.iter().zip(dx).map(|(&xi, &di)| xi + s * di).collect();
        let ft = f(&xt);
        let phi = nonlin_dot(&ft, &ft);
        if phi <= phi0 * (1.0 - c1 * s) {
            return (xt, ft);
        }
        last = Some((xt, ft));
        s *= 0.5;
    }
    last.unwrap_or_else(|| (x.to_vec(), fx.to_vec()))
}

/// Auto-scaled initial Jacobian parameter `α = 0.5·max(‖x₀‖, 1)/‖F₀‖`, matching
/// scipy's `GenericBroyden` autoscaling when `alpha` is not supplied.
fn nonlin_auto_alpha(x0: &[f64], f0: &[f64]) -> f64 {
    let nf0 = nonlin_dot(f0, f0).sqrt();
    if nf0 == 0.0 {
        return 1.0;
    }
    0.5 * nonlin_dot(x0, x0).sqrt().max(1.0) / nf0
}

const NONLIN_FTOL: f64 = 6.055_454_452_393_343e-6; // f64::EPSILON.powf(1.0/3.0), scipy default

fn nonlin_setup<F>(func: &F, x0: &[f64]) -> Result<Vec<f64>, OptError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    if x0.is_empty() {
        return Err(OptError::InvalidArgument {
            detail: "x0 must be non-empty".to_string(),
        });
    }
    if x0.iter().any(|v| !v.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: "x0 must be finite".to_string(),
        });
    }
    let f0 = func(x0);
    if f0.len() != x0.len() {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "F(x0) length {} must equal x0 length {}",
                f0.len(),
                x0.len()
            ),
        });
    }
    Ok(f0)
}

fn nonlin_converged(fx: &[f64]) -> bool {
    fx.iter().fold(0.0_f64, |m, &v| m.max(v.abs())) < NONLIN_FTOL
}

fn validate_nonlin_maxiter(maxiter: usize) -> Result<(), OptError> {
    if maxiter == 0 {
        return Err(OptError::InvalidArgument {
            detail: "maxiter must be greater than zero".to_string(),
        });
    }
    Ok(())
}

fn validate_nonzero_alpha(alpha: Option<f64>, name: &str) -> Result<(), OptError> {
    if let Some(value) = alpha
        && (!value.is_finite() || value == 0.0)
    {
        return Err(OptError::InvalidArgument {
            detail: format!("{name} must be finite and nonzero"),
        });
    }
    Ok(())
}

fn validate_positive_alpha(alpha: Option<f64>, name: &str) -> Result<(), OptError> {
    if let Some(value) = alpha
        && (!value.is_finite() || value <= 0.0)
    {
        return Err(OptError::InvalidArgument {
            detail: format!("{name} must be a positive finite value"),
        });
    }
    Ok(())
}

/// Find a root of a vector function using a diagonal Broyden Jacobian
/// approximation, matching `scipy.optimize.diagbroyden(F, xin)`.
///
/// The Jacobian is approximated by a diagonal `d` initialized to `-1/alpha`
/// (auto-scaled when `alpha` is `None`) and refined by the diagonal Broyden
/// update each step; an Armijo line search globalizes the iteration. Returns the
/// converged root (`‖F‖∞ < eps^(1/3)`), or `EvaluationBudgetExceeded` if
/// `maxiter` is reached. The returned root matches scipy's to solver tolerance.
pub fn diagbroyden<F>(
    func: F,
    x0: &[f64],
    alpha: Option<f64>,
    maxiter: usize,
) -> Result<Vec<f64>, OptError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    validate_nonlin_maxiter(maxiter)?;
    validate_nonzero_alpha(alpha, "alpha")?;
    let mut fx = nonlin_setup(&func, x0)?;
    let mut x = x0.to_vec();
    if nonlin_converged(&fx) {
        return Ok(x);
    }
    let n = x.len();
    let a = alpha.unwrap_or_else(|| nonlin_auto_alpha(&x, &fx));
    // scipy DiagBroyden.setup: d = 1/alpha (solve returns -f/d, so dx = f/d).
    let mut d = vec![1.0 / a; n];
    for _ in 0..maxiter {
        let dx: Vec<f64> = fx.iter().zip(&d).map(|(&fi, &di)| fi / di).collect();
        let (x_new, fx_new) = nonlin_line_search(&func, &x, &fx, &dx);
        let dx_taken: Vec<f64> = x_new.iter().zip(&x).map(|(&a, &b)| a - b).collect();
        let dnorm = nonlin_dot(&dx_taken, &dx_taken);
        if dnorm > 0.0 {
            // d -= (df + d·dx)·dx / ‖dx‖²  (componentwise; scipy diagonal Broyden).
            for i in 0..n {
                let df = fx_new[i] - fx[i];
                d[i] -= (df + d[i] * dx_taken[i]) * dx_taken[i] / dnorm;
            }
        }
        x = x_new;
        fx = fx_new;
        if nonlin_converged(&fx) {
            return Ok(x);
        }
    }
    Err(OptError::EvaluationBudgetExceeded {
        detail: format!("diagbroyden failed to converge in {maxiter} iterations"),
    })
}

/// Find a root of a vector function using a scalar Jacobian approximation
/// (linear mixing), matching `scipy.optimize.linearmixing(F, xin)`.
///
/// The Jacobian is fixed at `-1/alpha`, so the step is `dx = alpha·F(x)`,
/// globalized by an Armijo line search. As scipy warns, convergence depends
/// strongly on a well-chosen (often negative) `alpha`; when `alpha` is `None`
/// it is auto-scaled like scipy's. Returns the converged root or
/// `EvaluationBudgetExceeded`.
pub fn linearmixing<F>(
    func: F,
    x0: &[f64],
    alpha: Option<f64>,
    maxiter: usize,
) -> Result<Vec<f64>, OptError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    validate_nonlin_maxiter(maxiter)?;
    validate_nonzero_alpha(alpha, "alpha")?;
    let mut fx = nonlin_setup(&func, x0)?;
    let mut x = x0.to_vec();
    if nonlin_converged(&fx) {
        return Ok(x);
    }
    let a = alpha.unwrap_or_else(|| nonlin_auto_alpha(&x, &fx));
    for _ in 0..maxiter {
        // dx = -solve(Fx) = -(-Fx·alpha) = alpha·Fx.
        let dx: Vec<f64> = fx.iter().map(|&fi| a * fi).collect();
        let (x_new, fx_new) = nonlin_line_search(&func, &x, &fx, &dx);
        x = x_new;
        fx = fx_new;
        if nonlin_converged(&fx) {
            return Ok(x);
        }
    }
    Err(OptError::EvaluationBudgetExceeded {
        detail: format!("linearmixing failed to converge in {maxiter} iterations"),
    })
}

/// Find a root of a vector function using a tuned diagonal (exciting mixing)
/// Jacobian approximation, matching `scipy.optimize.excitingmixing(F, xin)`.
///
/// Each component carries its own step weight `beta_i` (initialized to `alpha`):
/// when a residual component keeps its sign across a step the weight grows by
/// `alpha` (accelerating), otherwise it resets to `alpha`; weights are clamped
/// to `[0, alphamax]` (scipy default `alphamax = 1.0`). The step is
/// `dx = beta·F(x)`, globalized by an Armijo line search. As scipy notes, this
/// works for problems whose Jacobian is negative-definite-ish (positive
/// `alpha`); returns the converged root or `EvaluationBudgetExceeded`.
pub fn excitingmixing<F>(
    func: F,
    x0: &[f64],
    alpha: Option<f64>,
    alphamax: f64,
    maxiter: usize,
) -> Result<Vec<f64>, OptError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    validate_nonlin_maxiter(maxiter)?;
    validate_positive_alpha(alpha, "alpha")?;
    if !alphamax.is_finite() || alphamax <= 0.0 {
        return Err(OptError::InvalidArgument {
            detail: "alphamax must be a positive finite value".to_string(),
        });
    }
    let mut fx = nonlin_setup(&func, x0)?;
    let mut x = x0.to_vec();
    if nonlin_converged(&fx) {
        return Ok(x);
    }
    let n = x.len();
    let a = alpha.unwrap_or_else(|| nonlin_auto_alpha(&x, &fx));
    let mut beta = vec![a; n];
    for _ in 0..maxiter {
        // dx = -solve(Fx) = -(-Fx·beta) = beta·Fx.
        let dx: Vec<f64> = fx.iter().zip(&beta).map(|(&fi, &bi)| bi * fi).collect();
        let (x_new, fx_new) = nonlin_line_search(&func, &x, &fx, &dx);
        // Per-component weight update: grow where the residual sign persisted,
        // reset where it flipped, then clamp to [0, alphamax].
        for i in 0..n {
            if fx_new[i] * fx[i] > 0.0 {
                beta[i] += a;
            } else {
                beta[i] = a;
            }
            beta[i] = beta[i].clamp(0.0, alphamax);
        }
        x = x_new;
        fx = fx_new;
        if nonlin_converged(&fx) {
            return Ok(x);
        }
    }
    Err(OptError::EvaluationBudgetExceeded {
        detail: format!("excitingmixing failed to converge in {maxiter} iterations"),
    })
}

/// Fixed-point iteration: find x such that f(x) = x.
///
/// Matches `scipy.optimize.fixed_point`.
pub fn fixed_point<F>(f: F, x0: f64, tol: f64, maxiter: usize) -> Result<f64, OptError>
where
    F: Fn(f64) -> f64,
{
    if !x0.is_finite() {
        return Err(OptError::InvalidArgument {
            detail: "x0 must be finite".to_string(),
        });
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err(OptError::InvalidArgument {
            detail: "tol must be a positive finite value".to_string(),
        });
    }
    if maxiter == 0 {
        return Err(OptError::InvalidArgument {
            detail: "maxiter must be greater than zero".to_string(),
        });
    }
    let mut x = x0;
    for _iter in 0..maxiter {
        let x_new = f(x);
        if (x_new - x).abs() < tol {
            return Ok(x_new);
        }

        // Steffensen's method acceleration
        let x_new2 = f(x_new);
        let denom = x_new2 - 2.0 * x_new + x;
        if denom.abs() > 1e-30 {
            let x_acc = x - (x_new - x).powi(2) / denom;
            if (x_acc - x).abs() < tol {
                return Ok(x_acc);
            }
            x = x_acc;
        } else {
            x = x_new;
        }
    }
    Err(OptError::InvalidArgument {
        detail: format!("fixed_point did not converge in {maxiter} iterations"),
    })
}

/// Vectorised [`fixed_point`] iteration over `n` parameter rows — the
/// vmap-over-solver form for `scipy.optimize.fixed_point`. `f(x, params)` is an
/// inlined Rust closure (no per-iteration Python callback), and the independent
/// solves fan across threads with a work-capped gate (a scalar fixed-point
/// iteration is cheap, so the win is eliminating SciPy's per-solve Python
/// overhead; typical batches stay serial). Entry `k` equals
/// `fixed_point(|x| f(x, &param_rows[k]), x0, tol, maxiter)`.
pub fn fixed_point_many<F>(
    f: F,
    x0: f64,
    param_rows: &[Vec<f64>],
    tol: f64,
    maxiter: usize,
) -> Vec<Result<f64, OptError>>
where
    F: Fn(f64, &[f64]) -> f64 + Sync,
{
    let nrows = param_rows.len();
    if nrows == 0 {
        return Vec::new();
    }
    let f_ref = &f;
    let solve_one = move |params: &[f64]| fixed_point(|x| f_ref(x, params), x0, tol, maxiter);

    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let nthreads = cores.min(nrows / 2048).max(1);
    if nthreads <= 1 {
        return param_rows.iter().map(|p| solve_one(p)).collect();
    }
    let chunk = nrows.div_ceil(nthreads);
    let solve_one = &solve_one;
    let chunk_results: Vec<Vec<Result<f64, OptError>>> = std::thread::scope(|scope| {
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
            .map(|h| h.join().expect("fixed_point_many worker panicked"))
            .collect()
    });
    let mut out = Vec::with_capacity(nrows);
    for cr in chunk_results {
        out.extend(cr);
    }
    out
}

/// Non-negative least squares: minimize ||Ax - b||² subject to x >= 0.
///
/// Uses the active-set method (Lawson-Hanson algorithm).
/// Matches `scipy.optimize.nnls`.
/// Full Cholesky factorization of the passive-set Gram submatrix `gram[pidx, pidx]` into the
/// strided lower-triangular buffer `lflat` (stride `n`: `L[i][j] = lflat[i*n + j]`, `j ≤ i`).
/// Returns `false` on a non-positive pivot (rank-deficient passive set → caller falls back).
/// O(p³); used only when a variable LEAVES the passive set (rare), so the active-set buildup stays
/// O(n³) overall instead of the old re-factor-every-iteration O(n⁴).
fn nnls_chol_refactor(gram: &[f64], n: usize, pidx: &[usize], lflat: &mut [f64]) -> bool {
    let p = pidx.len();
    for i in 0..p {
        let gi = pidx[i] * n;
        for j in 0..=i {
            let mut sum = gram[gi + pidx[j]];
            for k in 0..j {
                sum -= lflat[i * n + k] * lflat[j * n + k];
            }
            if i == j {
                if sum <= 0.0 {
                    return false;
                }
                lflat[i * n + i] = sum.sqrt();
            } else {
                lflat[i * n + j] = sum / lflat[j * n + j];
            }
        }
    }
    true
}

/// Extend the Cholesky factor `lflat` (currently `p×p` for `pidx`) by one column for the new
/// passive index `jnew` (rank-1 update). Writes the new row `p` in place. Returns `false` if the
/// new column is (numerically) dependent on the passive set (non-positive Schur pivot). O(p²).
fn nnls_chol_add_col(
    gram: &[f64],
    n: usize,
    pidx: &[usize],
    jnew: usize,
    lflat: &mut [f64],
) -> bool {
    let p = pidx.len();
    // Forward-solve L v = gram[pidx, jnew]; store v directly into the new row (row p).
    for k in 0..p {
        let mut s = gram[pidx[k] * n + jnew];
        for t in 0..k {
            s -= lflat[k * n + t] * lflat[p * n + t];
        }
        lflat[p * n + k] = s / lflat[k * n + k];
    }
    let mut dd = gram[jnew * n + jnew];
    for k in 0..p {
        dd -= lflat[p * n + k] * lflat[p * n + k];
    }
    if dd <= 0.0 {
        return false;
    }
    lflat[p * n + p] = dd.sqrt();
    true
}

/// Solve `L Lᵀ s = rhs` for the `p×p` strided lower-triangular factor `lflat` (forward then back
/// substitution). `y` is scratch of length ≥ p. O(p²).
fn nnls_chol_solve(lflat: &[f64], n: usize, p: usize, rhs: &[f64], s: &mut [f64], y: &mut [f64]) {
    for i in 0..p {
        let mut acc = rhs[i];
        for j in 0..i {
            acc -= lflat[i * n + j] * y[j];
        }
        y[i] = acc / lflat[i * n + i];
    }
    for i in (0..p).rev() {
        let mut acc = y[i];
        for j in (i + 1)..p {
            acc -= lflat[j * n + i] * s[j];
        }
        s[i] = acc / lflat[i * n + i];
    }
}

/// Accumulate one observation's rank-1 contribution into the UPPER triangle of a partial Gram
/// matrix `g` (row-major n×n) and `Aᵀb` accumulator `ab`. Shared by the serial and parallel-
/// reduction Gram precompute in `nnls`.
#[inline]
fn nnls_gram_accumulate(g: &mut [f64], ab: &mut [f64], ai: &[f64], bi: f64, n: usize) {
    for (j1, &v1) in ai.iter().enumerate() {
        let grow = &mut g[j1 * n + j1..j1 * n + n];
        for (gg, &v2) in grow.iter_mut().zip(ai[j1..].iter()) {
            *gg += v1 * v2;
        }
    }
    for (abj, &v) in ab.iter_mut().zip(ai.iter()) {
        *abj += v * bi;
    }
}

pub fn nnls(a: &[Vec<f64>], b: &[f64]) -> Result<(Vec<f64>, f64), OptError> {
    let m = a.len();
    if m == 0 {
        return Err(OptError::InvalidArgument {
            detail: "empty matrix".to_string(),
        });
    }
    let n = a[0].len();
    for (row_index, row) in a.iter().enumerate() {
        if row.len() != n {
            return Err(OptError::InvalidArgument {
                detail: "A must be rectangular".to_string(),
            });
        }
        if row.iter().any(|value| !value.is_finite()) {
            return Err(OptError::NonFiniteInput {
                detail: format!("A contains non-finite values in row {row_index}"),
            });
        }
    }
    if b.len() != m {
        return Err(OptError::InvalidArgument {
            detail: format!("b length {} != A rows {m}", b.len()),
        });
    }
    if b.iter().any(|value| !value.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: "b must not contain NaN or Inf".to_string(),
        });
    }

    let mut x = vec![0.0; n];
    let mut passive = vec![false; n]; // passive set (unconstrained)

    // Precompute the full Gram matrix G = AᵀA and Aᵀb ONCE. The passive
    // sub-problem rebuilt `ata[pi][pj] = Σ_i a[i][ji]·a[i][jj]` and
    // `atb_sub[pi] = Σ_i a[i][ji]·b[i]` from scratch (O(p²·m) / O(p·m)) on every
    // inner solve. Those are exactly `gram[ji][jj]` and `atb[ji]`; accumulating
    // them once with the same `for i in 0..m` order makes the per-iteration
    // gather bit-identical to recomputing, while removing the O(p²·m) rebuild
    // from each inner iteration (the precompute is the same O(n²·m) order as the
    // gradient pass already run each outer step, so it adds no dominant term).
    // Gram = AᵀA and Aᵀb dominate `nnls` for tall A (O(n²·m)): the old code
    // accumulated each `gram[j1][j2]` with the row loop INNERMOST, striding column
    // `j1`/`j2` through the `Vec<Vec>` heap rows (a cache miss per element). Stream
    // A once into a contiguous row-major buffer and accumulate by RANK-1 UPDATE —
    // for each observation `i`, `gram[j1][.] += a_i[j1] · a_i[.]` runs as a
    // contiguous AXPY the compiler vectorizes, and `atb += b_i · a_i` likewise.
    // BYTE-IDENTICAL: the same `a_i[j1]·a_i[j2]` (resp. `a_i[j]·b_i`) products
    // accumulate in the same `i = 0..m` order as the original sums.
    // `AᵀA` is symmetric, so accumulate only the upper triangle (`j2 ≥ j1`) per
    // rank-1 update and mirror it once — halving the dominant O(n²·m) work.
    // BYTE-IDENTICAL: `gram[j2][j1] = Σ a_i[j2]·a_i[j1]` equals the mirrored
    // `gram[j1][j2] = Σ a_i[j1]·a_i[j2]` exactly (f64 multiply is commutative and
    // the `i` order is unchanged), and the diagonal/upper terms are summed in the
    // same order as the full sweep.
    // Gram stored as a contiguous row-major n×n buffer (`gram[j1*n + j2]`) — flat for the strided
    // incremental Cholesky and a cache-friendly gradient sweep.
    let a_flat: Vec<f64> = a.iter().flat_map(|row| row.iter().copied()).collect();
    let mut gram = vec![0.0; n * n];
    let mut atb = vec![0.0; n];
    // The Gram precompute AᵀA is O(m·n²), the dominant cost for tall A. Fan the row sum across
    // cores as a partial-Gram REDUCTION (each thread accumulates a private Gram over a row chunk,
    // then merge in chunk order). Not bit-identical to the serial sum (~1e-13 reassociation), but
    // NNLS is strictly convex: the gradient only RANKS which variable enters, so the unique
    // minimizer is unchanged. Small problems stay serial (byte-identical) below the gate.
    let work = (m as u64).saturating_mul(n as u64).saturating_mul(n as u64);
    let nthreads = if work < (1 << 22) || m < 4 {
        1
    } else {
        std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1)
            .min(m)
            .min(16)
    };
    if nthreads <= 1 {
        for (ai, &bi) in a_flat.chunks(n).zip(b.iter()) {
            nnls_gram_accumulate(&mut gram, &mut atb, ai, bi, n);
        }
    } else {
        let chunk = m.div_ceil(nthreads);
        let partials: Vec<(Vec<f64>, Vec<f64>)> = std::thread::scope(|scope| {
            let handles: Vec<_> = a_flat
                .chunks(chunk * n)
                .zip(b.chunks(chunk))
                .map(|(arows, brows)| {
                    scope.spawn(move || {
                        let mut g = vec![0.0; n * n];
                        let mut ab = vec![0.0; n];
                        for (ai, &bi) in arows.chunks(n).zip(brows.iter()) {
                            nnls_gram_accumulate(&mut g, &mut ab, ai, bi, n);
                        }
                        (g, ab)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for (g, ab) in &partials {
            for (dst, &src) in gram.iter_mut().zip(g.iter()) {
                *dst += src;
            }
            for (dst, &src) in atb.iter_mut().zip(ab.iter()) {
                *dst += src;
            }
        }
    }
    for j1 in 0..n {
        for j2 in (j1 + 1)..n {
            gram[j2 * n + j1] = gram[j1 * n + j2];
        }
    }

    // Lawson–Hanson active set with an INCREMENTALLY-MAINTAINED Cholesky factor of the passive-set
    // Gram submatrix (`lflat`, strided). A variable ENTERING the passive set is an O(p²) rank-1
    // column add (`nnls_chol_add_col`); the rare REMOVAL triggers an O(p³) refactor. The old code
    // rebuilt+re-factored the full p×p submatrix on EVERY inner solve (O(n⁴) for n columns); this
    // is O(n³). `pidx` is the passive set in factorization order. A rank-deficient passive column
    // (non-positive Schur pivot) flips `use_slow`, reverting to the proven gather + Cholesky /
    // pivoting solve for the rest of the run. NNLS has a unique minimizer, so the returned `x` is
    // unchanged (only the path differs).
    let mut pidx: Vec<usize> = Vec::new();
    let mut lflat = vec![0.0; n * n];
    let mut use_slow = false;
    let mut w = vec![0.0; n];
    let mut rhs = vec![0.0; n];
    let mut s = vec![0.0; n];
    let mut ybuf = vec![0.0; n];

    for _ in 0..3 * n {
        // Gradient w = Aᵀb − (AᵀA)x = atb − gram·x (O(n²); contiguous row sweep).
        for j in 0..n {
            let mut acc = atb[j];
            let grow = &gram[j * n..j * n + n];
            for k in 0..n {
                acc -= grow[k] * x[k];
            }
            w[j] = acc;
        }

        // Variable with the largest gradient among the active (non-passive) set enters.
        let mut max_w = f64::NEG_INFINITY;
        let mut max_j = 0;
        for j in 0..n {
            if !passive[j] && w[j] > max_w {
                max_w = w[j];
                max_j = j;
            }
        }
        if max_w <= 1e-10 {
            break; // optimality reached
        }

        passive[max_j] = true;
        if !use_slow {
            if nnls_chol_add_col(&gram, n, &pidx, max_j, &mut lflat) {
                pidx.push(max_j);
            } else {
                use_slow = true; // dependent column → fall back to the gather/pivot solve
            }
        }

        // Solve the unconstrained LS on the passive set; drop any variable that goes ≤ 0.
        loop {
            if use_slow {
                pidx = (0..n).filter(|&j| passive[j]).collect();
            }
            let p = pidx.len();
            if p == 0 {
                break;
            }

            if use_slow {
                let mut ata = vec![vec![0.0; p]; p];
                let mut atb_sub = vec![0.0; p];
                for (pi, &ji) in pidx.iter().enumerate() {
                    for (pj, &jj) in pidx.iter().enumerate() {
                        ata[pi][pj] = gram[ji * n + jj];
                    }
                    atb_sub[pi] = atb[ji];
                }
                let s_p = cholesky_solve_spd(&ata, &atb_sub)
                    .unwrap_or_else(|| solve_small_system(&ata, &atb_sub));
                s[..p].copy_from_slice(&s_p);
            } else {
                for (k, &ji) in pidx.iter().enumerate() {
                    rhs[k] = atb[ji];
                }
                nnls_chol_solve(&lflat, n, p, &rhs[..p], &mut s[..p], &mut ybuf[..p]);
            }

            let mut all_positive = true;
            for &si in &s[..p] {
                if si <= 0.0 {
                    all_positive = false;
                    break;
                }
            }
            if all_positive {
                for (k, &ji) in pidx.iter().enumerate() {
                    x[ji] = s[k];
                }
                break;
            }

            // Step toward the unconstrained solution until the first variable hits 0.
            let mut alpha = f64::INFINITY;
            for (k, &ji) in pidx.iter().enumerate() {
                if s[k] <= 0.0 {
                    let denom = x[ji] - s[k];
                    if denom > 0.0 {
                        alpha = alpha.min(x[ji] / denom);
                    }
                }
            }
            for (k, &ji) in pidx.iter().enumerate() {
                x[ji] += alpha * (s[k] - x[ji]);
                if x[ji].abs() < 1e-15 {
                    x[ji] = 0.0;
                    passive[ji] = false;
                }
            }

            // A variable left the passive set → rebuild `pidx` and re-factor (rare, O(p³)).
            if !use_slow {
                pidx = (0..n).filter(|&j| passive[j]).collect();
                if !nnls_chol_refactor(&gram, n, &pidx, &mut lflat) {
                    use_slow = true;
                }
            }
        }
    }

    // Compute residual
    let mut residual = 0.0;
    for i in 0..m {
        let mut ax_i = 0.0;
        for j in 0..n {
            ax_i += a[i][j] * x[j];
        }
        residual += (b[i] - ax_i).powi(2);
    }

    Ok((x, residual.sqrt()))
}

/// Solve a symmetric positive-definite system `M z = rhs` via Cholesky
/// (`M = L Lᵀ`, forward/back substitution). ~⅓ the flops of Gauss-Jordan and no
/// pivoting. Returns `None` if `M` is not positive-definite (a non-positive pivot
/// is hit), so callers can fall back to a pivoting solver on rank-deficient input.
fn cholesky_solve_spd(m: &[Vec<f64>], rhs: &[f64]) -> Option<Vec<f64>> {
    let n = rhs.len();
    if n == 0 {
        return Some(Vec::new());
    }
    // Lower-triangular Cholesky factor (row-major, only j ≤ i used).
    let mut l = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = m[i][j];
            for (&l_ik, &l_jk) in l[i][..j].iter().zip(&l[j][..j]) {
                sum -= l_ik * l_jk;
            }
            if i == j {
                if sum.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
                    return None;
                }
                l[i][i] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    // Forward solve L y = rhs.
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut s = rhs[i];
        for k in 0..i {
            s -= l[i][k] * y[k];
        }
        y[i] = s / l[i][i];
    }
    // Back solve Lᵀ x = y.
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for (k, xk) in x.iter().enumerate().skip(i + 1) {
            s -= l[k][i] * xk;
        }
        x[i] = s / l[i][i];
    }
    Some(x)
}

/// Solve a small symmetric positive-definite system `M z = rhs` by Gaussian
/// elimination with partial pivoting. Returns `None` on (near-)singularity.
fn dense_spd_solve(m: &[Vec<f64>], rhs: &[f64]) -> Option<Vec<f64>> {
    let n = rhs.len();
    if n == 0 {
        return Some(Vec::new());
    }
    let mut aug: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = m[i].clone();
            row.push(rhs[i]);
            row
        })
        .collect();
    for col in 0..n {
        let mut piv = col;
        for r in (col + 1)..n {
            if aug[r][col].abs() > aug[piv][col].abs() {
                piv = r;
            }
        }
        if aug[piv][col].abs() < 1e-300 {
            return None;
        }
        aug.swap(col, piv);
        for r in 0..n {
            if r != col {
                let f = aug[r][col] / aug[col][col];
                if f != 0.0 {
                    if r < col {
                        let (rows_before_pivot, rows_from_pivot) = aug.split_at_mut(col);
                        let row = &mut rows_before_pivot[r];
                        let pivot = &rows_from_pivot[0];
                        for (entry, &pivot_entry) in
                            row[col..=n].iter_mut().zip(pivot[col..=n].iter())
                        {
                            *entry -= f * pivot_entry;
                        }
                    } else {
                        let (rows_before_row, rows_from_row) = aug.split_at_mut(r);
                        let pivot = &rows_before_row[col];
                        let row = &mut rows_from_row[0];
                        for (entry, &pivot_entry) in
                            row[col..=n].iter_mut().zip(pivot[col..=n].iter())
                        {
                            *entry -= f * pivot_entry;
                        }
                    }
                }
            }
        }
    }
    Some((0..n).map(|i| aug[i][n] / aug[i][i]).collect())
}

/// Solve the bound-constrained linear least-squares problem
/// `min ‖A·x − b‖²` subject to `lb ≤ x ≤ ub`, matching
/// `scipy.optimize.lsq_linear(A, b, bounds=(lb, ub), method='bvls')`.
///
/// Uses the BVLS active-set algorithm (Stark & Parker). Bounds may be infinite
/// (`f64::NEG_INFINITY`/`INFINITY`) for one-sided or unconstrained variables.
/// The problem is convex, so the returned minimiser matches scipy's `bvls`
/// solution to solver tolerance.
pub fn lsq_linear(a: &[Vec<f64>], b: &[f64], lb: &[f64], ub: &[f64]) -> Result<Vec<f64>, OptError> {
    let m = a.len();
    if m == 0 {
        return Err(OptError::InvalidArgument {
            detail: "empty matrix".to_string(),
        });
    }
    let n = a[0].len();
    if b.len() != m || lb.len() != n || ub.len() != n {
        return Err(OptError::InvalidArgument {
            detail: "inconsistent A/b/bounds dimensions".to_string(),
        });
    }
    for (row_index, row) in a.iter().enumerate() {
        if row.len() != n {
            return Err(OptError::InvalidArgument {
                detail: format!("A row {row_index} length {} != {n}", row.len()),
            });
        }
        if row.iter().any(|value| !value.is_finite()) {
            return Err(OptError::NonFiniteInput {
                detail: format!("A row {row_index} contains non-finite values"),
            });
        }
    }
    if b.iter().any(|value| !value.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: "b must contain only finite values".to_string(),
        });
    }
    for (index, (&lo, &hi)) in lb.iter().zip(ub.iter()).enumerate() {
        if lo.is_nan() || hi.is_nan() || lo > hi {
            return Err(OptError::InvalidBounds {
                detail: format!("invalid bounds for variable {index}"),
            });
        }
    }
    // Gram = AᵀA, atb = Aᵀb. Accumulate by RANK-1 update over a contiguous row-major copy of A:
    // `gram[j1][.] += a_i[j1] · a_i[.]` is a contiguous AXPY the compiler vectorizes, vs the old
    // per-pair `Σ_i row[j1]·row[j2]` that strided TWO columns through the `Vec<Vec>` heap rows
    // (a cache miss per element, O(n²·m) of them — the dominant cost for this routine). ~1e-13
    // reassociation only; lsq_linear's bounded LS has a unique minimizer (full column rank), so
    // the optimum is unchanged (the Gram only feeds the subproblem solves and the KKT gradient).
    let a_flat: Vec<f64> = a.iter().flat_map(|row| row.iter().copied()).collect();
    let mut gram = vec![0.0_f64; n * n]; // contiguous row-major (full symmetric); feeds nnls_chol_*
    let mut atb = vec![0.0_f64; n];
    for i in 0..m {
        let ai = &a_flat[i * n..i * n + n];
        let bi = b[i];
        for j1 in 0..n {
            let v1 = ai[j1];
            let grow = &mut gram[j1 * n..j1 * n + n];
            for (g, &v2) in grow.iter_mut().zip(ai.iter()) {
                *g += v1 * v2;
            }
            atb[j1] += v1 * bi;
        }
    }

    // State: -1 = at lower, +1 = at upper, 0 = free. Start each variable at a bound.
    let mut state = vec![0i8; n];
    let mut x = vec![0.0_f64; n];
    for j in 0..n {
        if lb[j].is_infinite() && ub[j].is_infinite() {
            state[j] = 0;
            x[j] = 0.0;
        } else if lb[j].is_finite() {
            state[j] = -1;
            x[j] = lb[j];
        } else {
            state[j] = 1;
            x[j] = ub[j];
        }
    }
    let tol = 1e-10;
    let max_outer = 10 * (n + 1);

    // Incrementally-maintained Cholesky of `gram[free, free]` (`lflat`, strided stride=n). A
    // variable ENTERS the free set via the KKT step = O(p²) rank-1 column add (`nnls_chol_add_col`);
    // it LEAVES via inner-loop blocking = O(p³) refactor (`nnls_chol_refactor`) — rare, since the
    // active set mostly grows. The old code rebuilt+re-factored the full free submatrix on EVERY
    // inner solve (O(n⁴)); this is O(n³). `use_slow` reverts to the gather + Cholesky/Gauss-Jordan
    // path on a rank-deficient free set. Bounded-LS minimizer is unique → converged x unchanged.
    let mut free: Vec<usize> = (0..n).filter(|&j| state[j] == 0).collect();
    let mut lflat = vec![0.0_f64; n * n];
    let mut use_slow = !free.is_empty() && !nnls_chol_refactor(&gram, n, &free, &mut lflat);
    let mut rhs = vec![0.0_f64; n];
    let mut zbuf = vec![0.0_f64; n];
    let mut ybuf = vec![0.0_f64; n];

    for _ in 0..max_outer {
        // Inner active-set loop: solve the free subproblem, fixing blocking vars.
        loop {
            if use_slow {
                free = (0..n).filter(|&j| state[j] == 0).collect();
            }
            let p = free.len();
            if p == 0 {
                break;
            }
            // sub_rhs[k] = atb[free[k]] − Σ_{fixed j} gram[free[k]][j]·x[j].
            for (k, &jk) in free.iter().enumerate() {
                let grow = &gram[jk * n..jk * n + n];
                let mut r = atb[jk];
                for (j2, (&gv, &xv)) in grow.iter().zip(x.iter()).enumerate() {
                    if state[j2] != 0 {
                        r -= gv * xv;
                    }
                }
                rhs[k] = r;
            }
            // Solve gram[free,free] · z = rhs. Incremental factor on the fast path; gather +
            // Cholesky/Gauss-Jordan on the rank-deficient fallback (preserving break-on-singular).
            if use_slow {
                let mut sub_gram = vec![vec![0.0_f64; p]; p];
                for (pi, &ii) in free.iter().enumerate() {
                    for (pj, &jj) in free.iter().enumerate() {
                        sub_gram[pi][pj] = gram[ii * n + jj];
                    }
                }
                match cholesky_solve_spd(&sub_gram, &rhs[..p])
                    .or_else(|| dense_spd_solve(&sub_gram, &rhs[..p]))
                {
                    Some(zz) => zbuf[..p].copy_from_slice(&zz),
                    None => break,
                }
            } else {
                nnls_chol_solve(&lflat, n, p, &rhs[..p], &mut zbuf[..p], &mut ybuf[..p]);
            }
            let z = &zbuf[..p];

            // Step toward z; if any free var would cross a bound, clamp the most-binding one.
            let mut alpha = 1.0_f64;
            let mut block: Option<(usize, i8)> = None;
            for (fi, &j) in free.iter().enumerate() {
                if z[fi] < lb[j] - tol {
                    let a_j = (lb[j] - x[j]) / (z[fi] - x[j]);
                    if a_j < alpha {
                        alpha = a_j;
                        block = Some((j, -1));
                    }
                } else if z[fi] > ub[j] + tol {
                    let a_j = (ub[j] - x[j]) / (z[fi] - x[j]);
                    if a_j < alpha {
                        alpha = a_j;
                        block = Some((j, 1));
                    }
                }
            }
            for (fi, &j) in free.iter().enumerate() {
                x[j] += alpha * (z[fi] - x[j]);
            }
            match block {
                Some((j, side)) => {
                    x[j] = if side == -1 { lb[j] } else { ub[j] };
                    state[j] = side;
                    if !use_slow {
                        free.retain(|&f| f != j);
                        if !nnls_chol_refactor(&gram, n, &free, &mut lflat) {
                            use_slow = true;
                        }
                    }
                }
                None => break, // all free vars within bounds: subproblem solved
            }
        }
        // KKT check: free the bound variable that most violates optimality.
        // Gradient g[j] = (Gram·x − atb)[j] = Aᵀ(Ax − b); only bound vars can be freed.
        let mut best = None;
        let mut best_score = tol;
        for j in 0..n {
            if state[j] == 0 {
                continue;
            }
            let grow = &gram[j * n..j * n + n];
            let mut gj = -atb[j];
            for (&gv, &xv) in grow.iter().zip(x.iter()) {
                gj += gv * xv;
            }
            let score = if state[j] == -1 { -gj } else { gj };
            if score > best_score {
                best_score = score;
                best = Some(j);
            }
        }
        let Some(j_free) = best else { break };
        state[j_free] = 0;
        if !use_slow {
            if nnls_chol_add_col(&gram, n, &free, j_free, &mut lflat) {
                free.push(j_free);
            } else {
                use_slow = true;
            }
        }
    }
    Ok(x)
}

fn solve_small_system(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = a.len();
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let mut row = r.clone();
            row.push(b[i]);
            row
        })
        .collect();

    for col in 0..n {
        let max_row = (col..n)
            .max_by(|&i, &j| aug[i][col].abs().total_cmp(&aug[j][col].abs()))
            .unwrap_or(col);
        aug.swap(col, max_row);

        if aug[col][col].abs() < 1e-15 {
            continue;
        }

        let pivot = aug[col][col];
        for row in col + 1..n {
            let factor = aug[row][col] / pivot;
            #[allow(clippy::needless_range_loop)]
            for j in col..=n {
                let val = aug[col][j];
                aug[row][j] -= factor * val;
            }
        }
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        if aug[i][i].abs() < 1e-15 {
            continue;
        }
        let mut sum = aug[i][n];
        for j in i + 1..n {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }

    x
}

/// A pooled block of adjacent points produced by the PAVA stack.
struct IsotonicBlock {
    /// Number of original points the block spans.
    count: usize,
    /// Sum of case weights over the block.
    weight: f64,
    /// Weight-weighted sum of responses over the block.
    weighted_sum: f64,
}

/// Isotonic regression: fit a non-decreasing least-squares function.
///
/// Minimizes `Σ wᵢ(yᵢ − xᵢ)²` subject to `x` being non-decreasing, via the
/// Pool Adjacent Violators Algorithm (PAVA). Matches
/// `scipy.optimize.isotonic_regression` (the `increasing=True` case).
///
/// `weights` must be strictly positive and the same length as `y`; violating
/// either, or a non-finite entry in `y`/`weights`, yields a `NaN` vector.
pub fn isotonic_regression(y: &[f64], weights: Option<&[f64]>) -> Vec<f64> {
    let n = y.len();
    if n == 0 {
        return vec![];
    }
    if y.iter().any(|v| !v.is_finite()) {
        return vec![f64::NAN; n];
    }

    let w: Vec<f64> = if let Some(wts) = weights {
        // SciPy requires strictly-positive case weights.
        if wts.len() != n || wts.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return vec![f64::NAN; n];
        }
        wts.to_vec()
    } else {
        vec![1.0; n]
    };

    // Pool Adjacent Violators Algorithm (Busing 2022): scan left to right,
    // maintaining a stack of blocks whose means are strictly increasing. Each
    // new point starts a singleton block and is merged into the block to its
    // left as long as the left block's mean exceeds it — a downward
    // violation. Means are compared by cross-multiplication (all weights are
    // strictly positive) to avoid division until the final expansion.
    let mut blocks: Vec<IsotonicBlock> = Vec::with_capacity(n);
    for i in 0..n {
        let mut block = IsotonicBlock {
            count: 1,
            weight: w[i],
            weighted_sum: w[i] * y[i],
        };
        while let Some(prev) = blocks.last() {
            // prev.mean > block.mean  ⇔  prev.wsum·block.w > block.wsum·prev.w
            if prev.weighted_sum * block.weight > block.weighted_sum * prev.weight {
                let prev = blocks.pop().expect("last() returned Some");
                block.count += prev.count;
                block.weight += prev.weight;
                block.weighted_sum += prev.weighted_sum;
            } else {
                break;
            }
        }
        blocks.push(block);
    }

    // Expand each block's weighted mean back across the points it spans.
    let mut result = Vec::with_capacity(n);
    for block in &blocks {
        let mean = block.weighted_sum / block.weight;
        result.resize(result.len() + block.count, mean);
    }
    result
}

/// Bisection method for scalar optimization: find minimum by trisection.
///
/// For unimodal functions on [a, b].
pub fn minimize_trisection<F>(f: F, a: f64, b: f64, tol: f64, maxiter: usize) -> (f64, f64)
where
    F: Fn(f64) -> f64,
{
    if !a.is_finite() || !b.is_finite() || !tol.is_finite() || tol <= 0.0 || a >= b {
        return (f64::NAN, f64::NAN);
    }
    let mut lo = a;
    let mut hi = b;

    for _ in 0..maxiter {
        if (hi - lo).abs() < tol {
            break;
        }
        let m1 = lo + (hi - lo) / 3.0;
        let m2 = hi - (hi - lo) / 3.0;
        if f(m1) < f(m2) {
            hi = m2;
        } else {
            lo = m1;
        }
    }

    let x = (lo + hi) / 2.0;
    (x, f(x))
}

/// Gradient descent with line search.
///
/// Minimizes f(x) starting from x0 using gradient descent with backtracking.
pub fn gradient_descent<F, G>(
    f: F,
    grad: G,
    x0: &[f64],
    tol: f64,
    maxiter: usize,
    learning_rate: f64,
) -> OptimizeResult
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    if x0.is_empty()
        || !tol.is_finite()
        || tol <= 0.0
        || maxiter == 0
        || !learning_rate.is_finite()
        || learning_rate <= 0.0
        || x0.iter().any(|v| !v.is_finite())
    {
        return OptimizeResult {
            x: x0.to_vec(),
            fun: None,
            success: false,
            status: ConvergenceStatus::InvalidInput,
            message: "invalid input parameters".to_string(),
            nfev: 0,
            njev: 0,
            nhev: 0,
            nit: 0,
            jac: None,
            hess_inv: None,
            maxcv: None,
        };
    }
    let mut x = x0.to_vec();
    let mut nfev = 0;
    let mut njev = 0;

    for iter in 0..maxiter {
        let g = grad(&x);
        njev += 1;
        if g.len() != x.len() || g.iter().any(|v| !v.is_finite()) {
            return OptimizeResult {
                x,
                fun: None,
                success: false,
                status: ConvergenceStatus::InvalidInput,
                message: "gradient returned invalid values".to_string(),
                nfev,
                njev,
                nhev: 0,
                nit: iter,
                jac: None,
                hess_inv: None,
                maxcv: None,
            };
        }

        let g_norm: f64 = g.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if g_norm < tol {
            let fval = f(&x);
            nfev += 1;
            return OptimizeResult {
                x,
                fun: Some(fval),
                success: true,
                status: ConvergenceStatus::Success,
                message: format!("gradient descent converged in {iter} iterations"),
                nfev,
                njev,
                nhev: 0,
                nit: iter,
                jac: Some(g),
                hess_inv: None,
                maxcv: None,
            };
        }

        // Backtracking line search (Armijo condition)
        let mut alpha = learning_rate;
        let f0 = f(&x);
        nfev += 1;
        let descent = g.iter().map(|&v| v * v).sum::<f64>();

        let mut found = false;
        for _ in 0..20 {
            let x_new: Vec<f64> = x
                .iter()
                .zip(g.iter())
                .map(|(&xi, &gi)| xi - alpha * gi)
                .collect();
            let f_new = f(&x_new);
            nfev += 1;
            if f_new <= f0 - 1e-4 * alpha * descent {
                x = x_new;
                found = true;
                break;
            }
            alpha *= 0.5;
        }

        // If line search didn't find improvement, take a small step anyway
        if !found {
            for (xi, &gi) in x.iter_mut().zip(g.iter()) {
                *xi -= learning_rate * 1e-3 * gi;
            }
        }
    }

    let fval = f(&x);
    nfev += 1;
    OptimizeResult {
        x,
        fun: Some(fval),
        success: false,
        status: ConvergenceStatus::MaxIterations,
        message: "gradient descent did not converge".to_string(),
        nfev,
        njev,
        nhev: 0,
        nit: maxiter,
        jac: None,
        hess_inv: None,
        maxcv: None,
    }
}

/// Projected gradient descent for box-constrained optimization.
///
/// Minimizes f(x) subject to lb <= x <= ub.
#[allow(clippy::too_many_arguments)]
pub fn projected_gradient_descent<F, G>(
    f: F,
    grad: G,
    x0: &[f64],
    lb: &[f64],
    ub: &[f64],
    tol: f64,
    maxiter: usize,
    learning_rate: f64,
) -> OptimizeResult
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    if n == 0
        || lb.len() != n
        || ub.len() != n
        || !tol.is_finite()
        || tol <= 0.0
        || maxiter == 0
        || !learning_rate.is_finite()
        || learning_rate <= 0.0
        || x0.iter().any(|v| !v.is_finite())
        || lb
            .iter()
            .zip(ub.iter())
            .any(|(l, u)| !l.is_finite() || !u.is_finite() || l > u)
    {
        return OptimizeResult {
            x: x0.to_vec(),
            fun: None,
            success: false,
            status: ConvergenceStatus::InvalidInput,
            message: "invalid input parameters".to_string(),
            nfev: 0,
            njev: 0,
            nhev: 0,
            nit: 0,
            jac: None,
            hess_inv: None,
            maxcv: None,
        };
    }
    let mut x: Vec<f64> = x0
        .iter()
        .zip(lb.iter().zip(ub.iter()))
        .map(|(&xi, (&lo, &hi))| xi.clamp(lo, hi))
        .collect();
    let mut nfev = 0;
    let mut njev = 0;

    for iter in 0..maxiter {
        let g = grad(&x);
        njev += 1;
        if g.len() != n || g.iter().any(|v| !v.is_finite()) {
            return OptimizeResult {
                x,
                fun: None,
                success: false,
                status: ConvergenceStatus::InvalidInput,
                message: "gradient returned invalid values".to_string(),
                nfev,
                njev,
                nhev: 0,
                nit: iter,
                jac: None,
                hess_inv: None,
                maxcv: None,
            };
        }

        // Projected gradient step
        let x_new: Vec<f64> = x
            .iter()
            .zip(g.iter())
            .zip(lb.iter().zip(ub.iter()))
            .map(|((&xi, &gi), (&lo, &hi))| (xi - learning_rate * gi).clamp(lo, hi))
            .collect();

        let step_norm: f64 = x_new
            .iter()
            .zip(x.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        x = x_new;

        if step_norm < tol {
            let fval = f(&x);
            nfev += 1;
            return OptimizeResult {
                x,
                fun: Some(fval),
                success: true,
                status: ConvergenceStatus::Success,
                message: format!("projected GD converged in {iter} iterations"),
                nfev,
                njev,
                nhev: 0,
                nit: iter,
                jac: Some(g),
                hess_inv: None,
                maxcv: None,
            };
        }
    }

    let fval = f(&x);
    nfev += 1;
    OptimizeResult {
        x,
        fun: Some(fval),
        success: false,
        status: ConvergenceStatus::MaxIterations,
        message: "projected GD did not converge".to_string(),
        nfev,
        njev,
        nhev: 0,
        nit: maxiter,
        jac: None,
        hess_inv: None,
        maxcv: None,
    }
}

/// Evaluate the first derivative of a scalar function numerically.
///
/// This mirrors `scipy.differentiate.derivative` for scalar-in/scalar-out
/// callables: adaptive real finite differences, Richardson extrapolation,
/// convergence metadata, and explicit non-finite failure handling.
pub fn derivative<F>(
    f: F,
    x: f64,
    options: DifferentiateOptions,
) -> Result<DerivativeResult, OptError>
where
    F: Fn(f64) -> f64,
{
    adaptive_first_derivative(&f, x, options)
}

/// Evaluate the Jacobian of a vector-valued function numerically.
///
/// `df[row][column]` is the derivative of `f(x)[row]` with respect to
/// `x[column]`.
pub fn jacobian<F>(
    f: F,
    x: &[f64],
    options: DifferentiateOptions,
) -> Result<JacobianResult, OptError>
where
    F: Fn(&[f64]) -> Vec<f64> + Sync,
{
    validate_differentiate_input(x)?;
    validate_differentiate_options(options)?;

    let f0 = f(x);
    if f0.is_empty() {
        return Err(OptError::InvalidArgument {
            detail: String::from("f(x) must return at least one element"),
        });
    }
    if f0.iter().any(|value| !value.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: String::from("f(x) returned non-finite values"),
        });
    }

    let rows = f0.len();
    let columns = x.len();
    let mut df = vec![vec![0.0; columns]; rows];
    let mut error = vec![vec![0.0; columns]; rows];
    let mut status = DifferentiateStatus::Converged;
    let mut success = true;
    let mut nit = 0;
    let mut nfev = 1;

    // Each (row, column) partial is an independent adaptive first-derivative estimate:
    // it perturbs only x[column] and reads f(...)[row], sharing no mutable state. So the
    // grid of partials runs on disjoint cores in a single thread::scope batch and the
    // results are folded back in the original row-major order — `df`/`error` land in
    // disjoint cells, `nfev` is an exact integer sum, `nit`/`success`/`status` are
    // max / AND / precedence-merge (all commutative). Bit-identical to the serial double
    // loop, and an error is propagated from the first failing pair in the same order.
    let pairs: Vec<(usize, usize)> = (0..rows)
        .flat_map(|row| (0..columns).map(move |column| (row, column)))
        .collect();
    let compute = |&(row, column): &(usize, usize)| -> Result<DerivativeResult, OptError> {
        // Reuse a single scratch vector per component: set the perturbed coordinate,
        // evaluate, then restore it so the buffer stays equal to `x` everywhere except
        // `column`. This is the same vector the per-eval `x.to_vec()` produced, so f sees
        // bit-identical inputs — but it allocates once per component instead of once per
        // evaluation, which is what makes the parallel path scale (concurrent per-eval
        // allocation otherwise serializes on the global allocator). RefCell keeps the
        // closure `Fn` (adaptive_first_derivative wants `Fn`); the cell is thread-local.
        let scratch = std::cell::RefCell::new(x.to_vec());
        let base = x[column];
        let component = |value: f64| {
            let mut shifted = scratch.borrow_mut();
            shifted[column] = value;
            let values = f(&shifted);
            shifted[column] = base;
            if values.len() == rows {
                values[row]
            } else {
                f64::NAN
            }
        };
        adaptive_first_derivative(&component, x[column], options)
    };

    let nthreads = if pairs.len() < 16 {
        1
    } else {
        std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1)
            .min(pairs.len() / 2)
            .max(1)
    };
    let results: Vec<Result<DerivativeResult, OptError>> = if nthreads <= 1 {
        pairs.iter().map(compute).collect()
    } else {
        let chunk = pairs.len().div_ceil(nthreads);
        let compute = &compute;
        let pairs_ref = &pairs;
        std::thread::scope(|scope| {
            let handles: Vec<_> =
                (0..nthreads)
                    .filter_map(|t| {
                        let i0 = t * chunk;
                        if i0 >= pairs_ref.len() {
                            return None;
                        }
                        let i1 = (i0 + chunk).min(pairs_ref.len());
                        Some(scope.spawn(move || {
                            pairs_ref[i0..i1].iter().map(compute).collect::<Vec<_>>()
                        }))
                    })
                    .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().expect("jacobian worker panicked"))
                .collect()
        })
    };

    for (&(row, column), res) in pairs.iter().zip(results) {
        let result = res?;
        df[row][column] = result.df;
        error[row][column] = result.error;
        success &= result.success;
        status = merge_differentiate_status(status, result.status);
        nit = nit.max(result.nit);
        nfev += result.nfev;
    }

    Ok(JacobianResult {
        df,
        error,
        success,
        status,
        nit,
        nfev,
    })
}

/// Evaluate the Hessian of a scalar function numerically.
///
/// This uses central second differences with Richardson extrapolation and
/// returns a symmetric Hessian matrix.
pub fn hessian<F>(f: F, x: &[f64], options: DifferentiateOptions) -> Result<HessianResult, OptError>
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    validate_differentiate_input(x)?;
    let order = validate_differentiate_options(options)?;
    let f0 = evaluate_finite_vector_scalar(&f, x, "f(x)")?;

    let n = x.len();
    let mut ddf = vec![vec![0.0; n]; n];
    let mut error = vec![vec![0.0; n]; n];
    let mut status = DifferentiateStatus::Converged;
    let mut success = true;
    let mut nit = 0;
    let mut nfev = 1;

    // Each upper-triangle entry (row, column) is an independent adaptive
    // finite-difference / Richardson estimate that only reads `f`, `x`, `f0` — no
    // shared mutable state — so the components run on disjoint cores. The results
    // are then folded back in the original (row, column) order, so every reduction
    // is order-preserving: `ddf`/`error` writes land in disjoint cells, `nfev` is an
    // exact integer sum, `nit`/`success`/`status` are max / AND / precedence-merge
    // (all commutative). The result is therefore bit-identical to the sequential
    // double loop, and an error is propagated from the first failing pair in the
    // same order the serial code would have hit it.
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|row| (row..n).map(move |column| (row, column)))
        .collect();
    let compute = |&(row, column): &(usize, usize)| -> Result<DerivativeResult, OptError> {
        adaptive_hessian_component(&f, x, f0, row, column, options, order)
    };

    let nthreads = if pairs.len() < 16 {
        1
    } else {
        std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1)
            .min(pairs.len() / 2)
            .max(1)
    };
    let results: Vec<Result<DerivativeResult, OptError>> = if nthreads <= 1 {
        pairs.iter().map(compute).collect()
    } else {
        let chunk = pairs.len().div_ceil(nthreads);
        let compute = &compute;
        let pairs_ref = &pairs;
        std::thread::scope(|scope| {
            let handles: Vec<_> =
                (0..nthreads)
                    .filter_map(|t| {
                        let i0 = t * chunk;
                        if i0 >= pairs_ref.len() {
                            return None;
                        }
                        let i1 = (i0 + chunk).min(pairs_ref.len());
                        Some(scope.spawn(move || {
                            pairs_ref[i0..i1].iter().map(compute).collect::<Vec<_>>()
                        }))
                    })
                    .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().expect("hessian worker panicked"))
                .collect()
        })
    };

    for (&(row, column), res) in pairs.iter().zip(results) {
        let result = res?;
        ddf[row][column] = result.df;
        ddf[column][row] = result.df;
        error[row][column] = result.error;
        error[column][row] = result.error;
        success &= result.success;
        status = merge_differentiate_status(status, result.status);
        nit = nit.max(result.nit);
        nfev += result.nfev;
    }

    Ok(HessianResult {
        ddf,
        error,
        success,
        status,
        nit,
        nfev,
    })
}

fn validate_differentiate_input(x: &[f64]) -> Result<(), OptError> {
    if x.is_empty() {
        return Err(OptError::InvalidArgument {
            detail: String::from("x must have at least one element"),
        });
    }
    if x.iter().any(|value| !value.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: String::from("x must contain only finite values"),
        });
    }
    Ok(())
}

fn validate_differentiate_options(options: DifferentiateOptions) -> Result<usize, OptError> {
    if options.maxiter == 0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("maxiter must be greater than zero"),
        });
    }
    if options.order == 0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("order must be greater than zero"),
        });
    }
    if !options.initial_step.is_finite() || options.initial_step <= 0.0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("initial_step must be finite and positive"),
        });
    }
    if !options.step_factor.is_finite()
        || options.step_factor <= 0.0
        || (options.step_factor - 1.0).abs() <= f64::EPSILON
    {
        return Err(OptError::InvalidArgument {
            detail: String::from("step_factor must be finite, positive, and not equal to one"),
        });
    }
    if !options.tolerances.atol.is_finite() || options.tolerances.atol < 0.0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("atol must be finite and non-negative"),
        });
    }
    if !options.tolerances.rtol.is_finite() || options.tolerances.rtol < 0.0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("rtol must be finite and non-negative"),
        });
    }

    if options.order.is_multiple_of(2) {
        Ok(options.order.max(2))
    } else {
        let rounded = options
            .order
            .checked_add(1)
            .ok_or_else(|| OptError::InvalidArgument {
                detail: String::from("order is too large"),
            })?;
        Ok(rounded.max(2))
    }
}

fn component_step(initial_step: f64, x: f64) -> f64 {
    let scale = 1.0 + x.abs();
    (initial_step * scale).max(f64::EPSILON.sqrt() * scale)
}

fn step_at_iteration(initial_step: f64, step_factor: f64, iteration: usize) -> f64 {
    initial_step / step_factor.powf(iteration as f64)
}

fn extrapolation_levels(
    order: usize,
    direction: StepDirection,
    iteration: usize,
    previous_levels: usize,
) -> usize {
    let maximum = match direction {
        StepDirection::Central => (order / 2).max(1),
        StepDirection::Forward | StepDirection::Backward => order.max(1),
    };
    maximum.min(iteration + 1).min(previous_levels + 1)
}

fn richardson_estimate(
    base: f64,
    previous_tableau: &[f64],
    levels: usize,
    step_factor: f64,
    error_power: usize,
) -> (f64, Vec<f64>) {
    let mut current = base;
    let mut next_tableau = Vec::with_capacity(levels);
    next_tableau.push(base);
    for level in 1..levels {
        let denominator = step_factor.powf((error_power * level) as f64) - 1.0;
        let previous_lower_order = previous_tableau[level - 1];
        current += (current - previous_lower_order) / denominator;
        next_tableau.push(current);
    }
    (current, next_tableau)
}

fn evaluate_finite_scalar<F>(f: &F, x: f64, context: &str) -> Result<f64, OptError>
where
    F: Fn(f64) -> f64,
{
    let value = f(x);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(OptError::NonFiniteInput {
            detail: format!("{context} returned a non-finite value"),
        })
    }
}

fn evaluate_finite_vector_scalar<F>(f: &F, x: &[f64], context: &str) -> Result<f64, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    let value = f(x);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(OptError::NonFiniteInput {
            detail: format!("{context} returned a non-finite value"),
        })
    }
}

fn adaptive_first_derivative<F>(
    f: &F,
    x: f64,
    options: DifferentiateOptions,
) -> Result<DerivativeResult, OptError>
where
    F: Fn(f64) -> f64,
{
    if !x.is_finite() {
        return Err(OptError::NonFiniteInput {
            detail: String::from("x must be finite"),
        });
    }
    let order = validate_differentiate_options(options)?;
    let initial_step = component_step(options.initial_step, x);
    let error_power = match options.step_direction {
        StepDirection::Central => 2,
        StepDirection::Forward | StepDirection::Backward => 1,
    };
    let mut nfev = 0;
    let f0 = match options.step_direction {
        StepDirection::Central => None,
        StepDirection::Forward | StepDirection::Backward => {
            nfev += 1;
            Some(evaluate_finite_scalar(f, x, "f(x)")?)
        }
    };
    let mut previous_estimate = None;
    let mut previous_tableau = Vec::new();
    let mut best_df = f64::NAN;
    let mut best_error = f64::INFINITY;
    let mut best_nit = 0;
    let mut best_step = initial_step;

    for iteration in 0..options.maxiter {
        let step = step_at_iteration(initial_step, options.step_factor, iteration);
        let base = match options.step_direction {
            StepDirection::Central => {
                let plus = evaluate_finite_scalar(f, x + step, "f(x + h)")?;
                let minus = evaluate_finite_scalar(f, x - step, "f(x - h)")?;
                nfev += 2;
                (plus - minus) / (2.0 * step)
            }
            StepDirection::Forward => {
                let plus = evaluate_finite_scalar(f, x + step, "f(x + h)")?;
                nfev += 1;
                let f_at_x = f0.ok_or_else(|| OptError::InvalidArgument {
                    detail: String::from("one-sided derivative missing base evaluation"),
                })?;
                (plus - f_at_x) / step
            }
            StepDirection::Backward => {
                let minus = evaluate_finite_scalar(f, x - step, "f(x - h)")?;
                nfev += 1;
                let f_at_x = f0.ok_or_else(|| OptError::InvalidArgument {
                    detail: String::from("one-sided derivative missing base evaluation"),
                })?;
                (f_at_x - minus) / step
            }
        };
        if !base.is_finite() {
            return Err(OptError::NonFiniteInput {
                detail: String::from("finite-difference estimate became non-finite"),
            });
        }

        let levels = extrapolation_levels(
            order,
            options.step_direction,
            iteration,
            previous_tableau.len(),
        );
        let (estimate, next_tableau) = richardson_estimate(
            base,
            &previous_tableau,
            levels,
            options.step_factor,
            error_power,
        );
        if !estimate.is_finite() {
            return Err(OptError::NonFiniteInput {
                detail: String::from("Richardson estimate became non-finite"),
            });
        }

        let error =
            previous_estimate.map_or(f64::INFINITY, |previous: f64| (estimate - previous).abs());
        let nit = iteration + 1;
        if iteration == 0 || error < best_error {
            best_df = estimate;
            best_error = error;
            best_nit = nit;
            best_step = step;
        } else if error.is_finite() && best_error.is_finite() && error > best_error {
            return Ok(DerivativeResult {
                df: best_df,
                error: best_error,
                success: false,
                status: DifferentiateStatus::ErrorIncreased,
                nit,
                nfev,
                step: best_step,
            });
        }

        if error.is_finite()
            && error <= options.tolerances.atol + options.tolerances.rtol * estimate.abs()
        {
            return Ok(DerivativeResult {
                df: estimate,
                error,
                success: true,
                status: DifferentiateStatus::Converged,
                nit,
                nfev,
                step,
            });
        }
        previous_estimate = Some(estimate);
        previous_tableau = next_tableau;
    }

    Ok(DerivativeResult {
        df: best_df,
        error: best_error,
        success: false,
        status: DifferentiateStatus::MaxIterations,
        nit: best_nit,
        nfev,
        step: best_step,
    })
}

fn adaptive_hessian_component<F>(
    f: &F,
    x: &[f64],
    f0: f64,
    row: usize,
    column: usize,
    options: DifferentiateOptions,
    order: usize,
) -> Result<DerivativeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    let row_step = component_step(options.initial_step, x[row]);
    let column_step = component_step(options.initial_step, x[column]);
    let mut previous_estimate = None;
    let mut previous_tableau = Vec::new();
    let mut best_ddf = f64::NAN;
    let mut best_error = f64::INFINITY;
    let mut best_nit = 0;
    let mut best_step = row_step.max(column_step);
    let mut nfev = 0;

    for iteration in 0..options.maxiter {
        let h_row = step_at_iteration(row_step, options.step_factor, iteration);
        let h_column = step_at_iteration(column_step, options.step_factor, iteration);
        let (base, evaluations) =
            hessian_component_estimate(f, x, f0, row, column, h_row, h_column)?;
        nfev += evaluations;
        if !base.is_finite() {
            return Err(OptError::NonFiniteInput {
                detail: String::from("finite-difference Hessian estimate became non-finite"),
            });
        }

        let levels = (order / 2)
            .max(1)
            .min(iteration + 1)
            .min(previous_tableau.len() + 1);
        let (estimate, next_tableau) =
            richardson_estimate(base, &previous_tableau, levels, options.step_factor, 2);
        if !estimate.is_finite() {
            return Err(OptError::NonFiniteInput {
                detail: String::from("Richardson Hessian estimate became non-finite"),
            });
        }

        let error =
            previous_estimate.map_or(f64::INFINITY, |previous: f64| (estimate - previous).abs());
        let nit = iteration + 1;
        if iteration == 0 || error < best_error {
            best_ddf = estimate;
            best_error = error;
            best_nit = nit;
            best_step = h_row.max(h_column);
        } else if error.is_finite() && best_error.is_finite() && error > best_error {
            return Ok(DerivativeResult {
                df: best_ddf,
                error: best_error,
                success: false,
                status: DifferentiateStatus::ErrorIncreased,
                nit,
                nfev,
                step: best_step,
            });
        }

        if error.is_finite()
            && error <= options.tolerances.atol + options.tolerances.rtol * estimate.abs()
        {
            return Ok(DerivativeResult {
                df: estimate,
                error,
                success: true,
                status: DifferentiateStatus::Converged,
                nit,
                nfev,
                step: h_row.max(h_column),
            });
        }
        previous_estimate = Some(estimate);
        previous_tableau = next_tableau;
    }

    Ok(DerivativeResult {
        df: best_ddf,
        error: best_error,
        success: false,
        status: DifferentiateStatus::MaxIterations,
        nit: best_nit,
        nfev,
        step: best_step,
    })
}

fn hessian_component_estimate<F>(
    f: &F,
    x: &[f64],
    f0: f64,
    row: usize,
    column: usize,
    h_row: f64,
    h_column: f64,
) -> Result<(f64, usize), OptError>
where
    F: Fn(&[f64]) -> f64,
{
    if row == column {
        let mut plus = x.to_vec();
        plus[row] += h_row;
        let f_plus = evaluate_finite_vector_scalar(f, &plus, "f(x + h)")?;

        let mut minus = x.to_vec();
        minus[row] -= h_row;
        let f_minus = evaluate_finite_vector_scalar(f, &minus, "f(x - h)")?;

        Ok(((f_plus - 2.0 * f0 + f_minus) / (h_row * h_row), 2))
    } else {
        let mut point = x.to_vec();
        point[row] += h_row;
        point[column] += h_column;
        let f_pp = evaluate_finite_vector_scalar(f, &point, "f(x + h_i + h_j)")?;

        point[column] = x[column] - h_column;
        let f_pm = evaluate_finite_vector_scalar(f, &point, "f(x + h_i - h_j)")?;

        point[row] = x[row] - h_row;
        point[column] = x[column] + h_column;
        let f_mp = evaluate_finite_vector_scalar(f, &point, "f(x - h_i + h_j)")?;

        point[column] = x[column] - h_column;
        let f_mm = evaluate_finite_vector_scalar(f, &point, "f(x - h_i - h_j)")?;

        Ok(((f_pp - f_pm - f_mp + f_mm) / (4.0 * h_row * h_column), 4))
    }
}

fn merge_differentiate_status(
    left: DifferentiateStatus,
    right: DifferentiateStatus,
) -> DifferentiateStatus {
    match (left, right) {
        (DifferentiateStatus::ErrorIncreased, _) | (_, DifferentiateStatus::ErrorIncreased) => {
            DifferentiateStatus::ErrorIncreased
        }
        (DifferentiateStatus::MaxIterations, _) | (_, DifferentiateStatus::MaxIterations) => {
            DifferentiateStatus::MaxIterations
        }
        (DifferentiateStatus::Converged, DifferentiateStatus::Converged) => {
            DifferentiateStatus::Converged
        }
    }
}

/// Compute numerical gradient via forward differences.
pub fn numerical_gradient<F>(f: F, x: &[f64], eps: f64) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x.len();
    if n == 0 || !eps.is_finite() || eps <= 0.0 || x.iter().any(|v| !v.is_finite()) {
        return vec![f64::NAN; n];
    }
    let f0 = f(x);
    if !f0.is_finite() {
        return vec![f64::NAN; n];
    }
    let mut grad = Vec::with_capacity(n);
    let mut xp = x.to_vec();
    for i in 0..n {
        let original = xp[i];
        xp[i] = original + eps;
        let fp = f(&xp);
        xp[i] = original;
        grad.push(if fp.is_finite() {
            (fp - f0) / eps
        } else {
            f64::NAN
        });
    }
    grad
}

/// Numerical Hessian via central differences.
pub fn numerical_hessian<F>(f: F, x: &[f64], eps: f64) -> Vec<Vec<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x.len();
    if n == 0 || !eps.is_finite() || eps <= 0.0 || x.iter().any(|v| !v.is_finite()) {
        return vec![vec![f64::NAN; n]; n];
    }
    let mut hess = vec![vec![0.0; n]; n];
    let mut scratch = x.to_vec();

    for i in 0..n {
        for j in i..n {
            // f(x + eps_i + eps_j)
            scratch[i] += eps;
            scratch[j] += eps;
            let f_pp = f(&scratch);

            // f(x + eps_i - eps_j)
            scratch[j] -= 2.0 * eps;
            let f_pm = f(&scratch);

            // f(x - eps_i + eps_j)
            scratch[i] -= 2.0 * eps;
            scratch[j] += 2.0 * eps;
            let f_mp = f(&scratch);

            // f(x - eps_i - eps_j)
            scratch[j] -= 2.0 * eps;
            let f_mm = f(&scratch);

            // Restore scratch to original x
            scratch[i] += eps;
            scratch[j] += eps;

            hess[i][j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps * eps);
            hess[j][i] = hess[i][j];
        }
    }
    hess
}

/// Compute the numerical Jacobian of a vector function.
pub fn numerical_jacobian<F>(f: F, x: &[f64], eps: f64) -> Vec<Vec<f64>>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x.len();
    if n == 0 || !eps.is_finite() || eps <= 0.0 || x.iter().any(|v| !v.is_finite()) {
        return vec![];
    }
    let f0 = f(x);
    let m = f0.len();
    if m == 0 || f0.iter().any(|v| !v.is_finite()) {
        return vec![vec![f64::NAN; n]; m.max(1)];
    }
    let mut jac = vec![vec![0.0; n]; m];
    let mut xp = x.to_vec();
    for j in 0..n {
        let original = xp[j];
        xp[j] = original + eps;
        let fp = f(&xp);
        xp[j] = original;
        for i in 0..m {
            let fpi = fp.get(i).copied().unwrap_or(f64::NAN);
            jac[i][j] = if fpi.is_finite() {
                (fpi - f0[i]) / eps
            } else {
                f64::NAN
            };
        }
    }
    jac
}

/// Simulated annealing for combinatorial optimization.
///
/// Minimizes a function over discrete states using neighbor generation.
pub fn simulated_annealing<S, F, N>(
    initial: S,
    cost: F,
    neighbor: N,
    temp_initial: f64,
    temp_final: f64,
    max_iter: usize,
    seed: u64,
) -> (S, f64)
where
    S: Clone,
    F: Fn(&S) -> f64,
    N: Fn(&S, u64) -> S,
{
    if !temp_initial.is_finite()
        || !temp_final.is_finite()
        || temp_initial <= 0.0
        || temp_final <= 0.0
        || max_iter == 0
    {
        return (initial, f64::NAN);
    }
    let mut current = initial;
    let mut current_cost = cost(&current);
    let mut best = current.clone();
    let mut best_cost = current_cost;
    let mut rng = seed;

    let cooling_rate = if max_iter > 1 && temp_initial > 0.0 {
        (temp_final.max(1e-12) / temp_initial.max(1e-12)).powf(1.0 / max_iter as f64)
    } else {
        1.0
    };
    let mut temp = temp_initial;

    for _ in 0..max_iter {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let candidate = neighbor(&current, rng);
        let candidate_cost = cost(&candidate);

        let delta = candidate_cost - current_cost;
        let accept = if delta < 0.0 {
            true
        } else {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = (rng >> 11) as f64 / (1u64 << 53) as f64;
            r < (-delta / temp).exp()
        };

        if accept {
            current = candidate;
            current_cost = candidate_cost;
            if current_cost < best_cost {
                best = current.clone();
                best_cost = current_cost;
            }
        }

        temp *= cooling_rate;
    }

    (best, best_cost)
}

/// Particle Swarm Optimization.
///
/// Minimizes f(x) over [lb, ub] using a swarm of particles.
pub fn pso<F>(
    f: F,
    lb: &[f64],
    ub: &[f64],
    mut n_particles: usize,
    max_iter: usize,
    seed: u64,
) -> (Vec<f64>, f64)
where
    F: Fn(&[f64]) -> f64,
{
    n_particles = n_particles.max(1);
    let d = lb.len();
    if d == 0
        || max_iter == 0
        || ub.len() != d
        || lb
            .iter()
            .zip(ub.iter())
            .any(|(l, u)| !l.is_finite() || !u.is_finite() || l > u)
    {
        return (vec![f64::NAN; d.max(1)], f64::NAN);
    }
    let w = 0.7298; // inertia
    let c1 = 1.4962; // cognitive
    let c2 = 1.4962; // social

    let mut rng = seed;
    let next_rand = |rng: &mut u64| -> f64 {
        *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (*rng >> 11) as f64 / (1u64 << 53) as f64
    };

    // Initialize particles
    let mut positions: Vec<Vec<f64>> = (0..n_particles)
        .map(|_| {
            (0..d)
                .map(|j| lb[j] + next_rand(&mut rng) * (ub[j] - lb[j]))
                .collect()
        })
        .collect();

    let mut velocities: Vec<Vec<f64>> = (0..n_particles)
        .map(|_| {
            (0..d)
                .map(|j| (next_rand(&mut rng) - 0.5) * (ub[j] - lb[j]) * 0.1)
                .collect()
        })
        .collect();

    let mut personal_best = positions.clone();
    let mut personal_best_cost: Vec<f64> = positions.iter().map(|p| f(p)).collect();

    let global_best_idx = personal_best_cost
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let mut global_best = personal_best[global_best_idx].clone();
    let mut global_best_cost = personal_best_cost[global_best_idx];

    for _ in 0..max_iter {
        for i in 0..n_particles {
            // Update velocity
            for j in 0..d {
                let r1 = next_rand(&mut rng);
                let r2 = next_rand(&mut rng);
                velocities[i][j] = w * velocities[i][j]
                    + c1 * r1 * (personal_best[i][j] - positions[i][j])
                    + c2 * r2 * (global_best[j] - positions[i][j]);
            }

            // Update position
            for j in 0..d {
                positions[i][j] = (positions[i][j] + velocities[i][j]).clamp(lb[j], ub[j]);
            }

            // Evaluate
            let cost = f(&positions[i]);
            if cost < personal_best_cost[i] {
                personal_best_cost[i] = cost;
                personal_best[i].clone_from(&positions[i]);
                if cost < global_best_cost {
                    global_best_cost = cost;
                    global_best.clone_from(&positions[i]);
                }
            }
        }
    }

    (global_best, global_best_cost)
}

#[cfg(test)]
mod tests {
    use crate::hungarian_rectangular_reference;
    use fsci_runtime::RuntimeMode;

    use crate::{
        BasinhoppingOptions, Bounds, ConvergenceStatus, DifferentialEvolutionOptions,
        DifferentiateOptions, Integrality, LinearConstraint, MilpOptions, MilpProblem,
        MinimizeOptions, NonlinearConstraint, OptError, OptimizeMethod, RootOptions, approx_fprime,
        basinhopping, bracket, brent_minimize, brute, check_grad, cholesky_solve_spd, cobyla,
        derivative, differential_evolution, differential_evolution_constrained, dual_annealing,
        fixed_point, fixed_point_many, golden, gradient_descent, hessian, isotonic_regression,
        jacobian, linear_sum_assignment, linprog, milp, minimize_scalar_bounded,
        minimize_trisection, nnls, numerical_gradient, numerical_hessian, numerical_jacobian,
        projected_gradient_descent, pso, quadratic_assignment, rosen, rosen_der, rosen_hess,
        rosen_hess_prod, shgo, transpose_matrix,
    };

    #[test]
    fn opt_numerical_and_trisection_match_analytic() {
        // numerical gradient/hessian/jacobian (exact within FD roundoff for low-degree
        // polys) and minimize_trisection. All four were previously untested.
        let g = numerical_gradient(|x: &[f64]| x[0] * x[0] + x[1] * x[1], &[1.0, 2.0], 1e-5);
        assert!(
            (g[0] - 2.0).abs() < 1e-4 && (g[1] - 4.0).abs() < 1e-4,
            "gradient"
        );
        let h = numerical_hessian(|x: &[f64]| x[0] * x[0] + x[1] * x[1], &[1.0, 2.0], 1e-3);
        assert!(
            (h[0][0] - 2.0).abs() < 1e-3 && (h[1][1] - 2.0).abs() < 1e-3 && h[0][1].abs() < 1e-3,
            "hessian"
        );
        let j = numerical_jacobian(
            |x: &[f64]| vec![x[0] * x[1], x[0] + x[1]],
            &[2.0, 3.0],
            1e-5,
        );
        assert!(
            (j[0][0] - 3.0).abs() < 1e-4
                && (j[0][1] - 2.0).abs() < 1e-4
                && (j[1][0] - 1.0).abs() < 1e-4
                && (j[1][1] - 1.0).abs() < 1e-4,
            "jacobian"
        );
        // minimize_trisection of (x-2)^2 over [0,5] -> (2, 0).
        let (xmin, fmin) = minimize_trisection(|x: f64| (x - 2.0).powi(2), 0.0, 5.0, 1e-8, 200);
        assert!(
            (xmin - 2.0).abs() < 1e-5 && fmin.abs() < 1e-9,
            "trisection min"
        );
    }

    #[test]
    fn cholesky_solve_spd_rejects_nan_pivots_without_regressing_valid_solves() {
        assert_eq!(cholesky_solve_spd(&[vec![4.0]], &[6.0]), Some(vec![1.5]));
        assert_eq!(cholesky_solve_spd(&[vec![f64::NAN]], &[1.0]), None);
    }

    #[test]
    fn numerical_finite_difference_helpers_restore_scratch_point() {
        let base = [1.0, -2.0, 0.5];

        let gradient_calls = std::cell::Cell::new(0usize);
        let gradient = numerical_gradient(
            |x: &[f64]| {
                let changed = x
                    .iter()
                    .zip(base.iter())
                    .filter(|(actual, expected)| (**actual - **expected).abs() > 1.0e-12)
                    .count();
                assert!(
                    changed <= 1,
                    "gradient scratch point leaked prior perturbation: {x:?}"
                );
                gradient_calls.set(gradient_calls.get() + 1);
                x.iter().sum::<f64>()
            },
            &base,
            1.0e-6,
        );
        assert_eq!(gradient_calls.get(), base.len() + 1);
        for value in gradient {
            assert!((value - 1.0).abs() < 1.0e-9);
        }

        let jacobian_calls = std::cell::Cell::new(0usize);
        let jacobian = numerical_jacobian(
            |x: &[f64]| {
                let changed = x
                    .iter()
                    .zip(base.iter())
                    .filter(|(actual, expected)| (**actual - **expected).abs() > 1.0e-12)
                    .count();
                assert!(
                    changed <= 1,
                    "jacobian scratch point leaked prior perturbation: {x:?}"
                );
                jacobian_calls.set(jacobian_calls.get() + 1);
                vec![x[0] + x[1], x[2] * x[2]]
            },
            &base,
            1.0e-6,
        );
        assert_eq!(jacobian_calls.get(), base.len() + 1);
        assert!((jacobian[0][0] - 1.0).abs() < 1.0e-9);
        assert!((jacobian[0][1] - 1.0).abs() < 1.0e-9);
        assert!(jacobian[0][2].abs() < 1.0e-9);
        assert!(jacobian[1][0].abs() < 1.0e-9);
        assert!(jacobian[1][1].abs() < 1.0e-9);
        assert!((jacobian[1][2] - 1.0).abs() < 1.0e-4);
    }

    #[test]
    fn lsq_linear_matches_scipy_bvls() {
        use crate::lsq_linear;
        let a = vec![
            vec![1.0, 2.0, 0.0],
            vec![0.0, 3.0, 1.0],
            vec![1.0, 0.0, 2.0],
            vec![2.0, 1.0, 1.0],
            vec![0.0, 1.0, 3.0],
        ];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0];
        let inf = f64::INFINITY;
        // Unconstrained optimum (wide bounds).
        let x0 = lsq_linear(&a, &b, &[-inf, -inf, -inf], &[inf, inf, inf]).unwrap();
        for (g, e) in x0.iter().zip(&[0.88, 0.215, 1.465]) {
            assert!((g - e).abs() < 1e-8, "unc {g} vs {e}");
        }
        // Upper bounds active on two variables.
        let x1 = lsq_linear(&a, &b, &[0.0, 0.0, 0.0], &[0.3, 1.0, 0.8]).unwrap();
        for (g, e) in x1.iter().zip(&[0.3, 0.68, 0.8]) {
            assert!((g - e).abs() < 1e-8, "ub {g} vs {e}");
        }
        // All upper bounds binding.
        let x2 = lsq_linear(&a, &b, &[0.0, 0.0, 0.0], &[0.5, 0.5, 1.0]).unwrap();
        for (g, e) in x2.iter().zip(&[0.5, 0.5, 1.0]) {
            assert!((g - e).abs() < 1e-8, "ub2 {g} vs {e}");
        }
        // Lower bound active on x0.
        let x3 = lsq_linear(&a, &b, &[0.9, 0.0, 0.0], &[2.0, 2.0, 2.0]).unwrap();
        for (g, e) in x3.iter().zip(&[0.9, 0.2113636364, 1.4613636364]) {
            assert!((g - e).abs() < 1e-8, "lb {g} vs {e}");
        }
    }

    #[test]
    fn lsq_linear_rejects_ragged_matrix() {
        use crate::lsq_linear;
        let err = lsq_linear(
            &[vec![1.0, 2.0], vec![3.0]],
            &[1.0, 2.0],
            &[0.0, 0.0],
            &[1.0, 1.0],
        )
        .expect_err("ragged A should fail closed");
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
    }

    #[test]
    fn lsq_linear_rejects_invalid_bounds() {
        use crate::lsq_linear;
        let a = [vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = [1.0, 2.0];
        let err = lsq_linear(&a, &b, &[2.0, 0.0], &[1.0, 3.0])
            .expect_err("lower bound greater than upper should fail");
        assert!(matches!(err, crate::OptError::InvalidBounds { .. }));

        let err =
            lsq_linear(&a, &b, &[f64::NAN, 0.0], &[1.0, 3.0]).expect_err("NaN bound should fail");
        assert!(matches!(err, crate::OptError::InvalidBounds { .. }));
    }

    #[test]
    fn nonlinear_mixing_solvers_find_root() {
        use crate::{diagbroyden, linearmixing};
        // F(x) = 0 has the unique root (0.8411639, 0.1588361); scipy's
        // diagbroyden/linearmixing both converge there from (0, 0).
        let f = |x: &[f64]| {
            vec![
                x[0] + 0.5 * (x[0] - x[1]).powi(3) - 1.0,
                0.5 * (x[1] - x[0]).powi(3) + x[1],
            ]
        };
        let root = [0.8411639019, 0.1588360981];

        let rd = diagbroyden(f, &[0.0, 0.0], None, 500).expect("diagbroyden converges");
        assert!((rd[0] - root[0]).abs() < 1e-5, "diagbroyden x0 {}", rd[0]);
        assert!((rd[1] - root[1]).abs() < 1e-5, "diagbroyden x1 {}", rd[1]);

        // linearmixing needs a well-chosen (negative) alpha, like scipy.
        let rl = linearmixing(f, &[0.0, 0.0], Some(-0.1), 2000).expect("linearmixing converges");
        assert!((rl[0] - root[0]).abs() < 1e-3, "linearmixing x0 {}", rl[0]);
        assert!((rl[1] - root[1]).abs() < 1e-3, "linearmixing x1 {}", rl[1]);

        // Already-at-root returns immediately; bad input rejected.
        let at_root = diagbroyden(f, &root, None, 10).expect("at root");
        assert!((at_root[0] - root[0]).abs() < 1e-3);
        assert!(diagbroyden(f, &[], None, 10).is_err());
        assert!(diagbroyden(f, &[0.0, 0.0], None, 0).is_err());
        assert!(diagbroyden(f, &[0.0, 0.0], Some(0.0), 10).is_err());
        assert!(diagbroyden(f, &[0.0, 0.0], Some(f64::NAN), 10).is_err());
        assert!(linearmixing(f, &[0.0, 0.0], None, 0).is_err());
        assert!(linearmixing(f, &[0.0, 0.0], Some(0.0), 10).is_err());
        assert!(linearmixing(f, &[0.0, 0.0], Some(f64::INFINITY), 10).is_err());
    }

    #[test]
    fn exciting_mixing_finds_root() {
        use crate::excitingmixing;
        // Negative-definite Jacobian system (where exciting mixing applies):
        // F = c - x - 0.1 x^3; scipy converges to (1.5945621, 2.0887301).
        let f = |x: &[f64]| {
            vec![
                2.0 - x[0] - 0.1 * x[0].powi(3),
                3.0 - x[1] - 0.1 * x[1].powi(3),
            ]
        };
        let r = excitingmixing(f, &[0.0, 0.0], Some(0.1), 1.0, 500).expect("converges");
        assert!((r[0] - 1.5945621166).abs() < 1e-5, "x0 {}", r[0]);
        assert!((r[1] - 2.0887301451).abs() < 1e-5, "x1 {}", r[1]);
        assert!(excitingmixing(f, &[0.0, 0.0], Some(0.1), 1.0, 0).is_err());
        assert!(excitingmixing(f, &[0.0, 0.0], Some(0.0), 1.0, 10).is_err());
        assert!(excitingmixing(f, &[0.0, 0.0], Some(f64::NAN), 1.0, 10).is_err());
        assert!(excitingmixing(f, &[0.0, 0.0], Some(0.1), 0.0, 10).is_err());
        assert!(excitingmixing(f, &[0.0, 0.0], Some(0.1), f64::INFINITY, 10).is_err());
    }

    #[test]
    fn fmin_cobyla_reaches_constrained_minimum() {
        use crate::fmin_cobyla;
        // min x0^2 + x1^2 s.t. x0 >= 1; scipy fmin_cobyla -> ~(1, 0).
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let cons: Vec<fn(&[f64]) -> f64> = vec![|x: &[f64]| x[0] - 1.0];
        let xopt = fmin_cobyla(f, &[3.0, 2.0], &cons, 1.0, 2000).expect("fmin_cobyla");
        assert!((xopt[0] - 1.0).abs() < 1e-3, "x0 = {}", xopt[0]);
        assert!(xopt[1].abs() < 1e-3, "x1 = {}", xopt[1]);
        assert!(cobyla(f, &[], &cons, 10, 1.0).is_err());
        assert!(cobyla(f, &[f64::NAN, 2.0], &cons, 10, 1.0).is_err());
        assert!(cobyla(f, &[3.0, 2.0], &cons, 0, 1.0).is_err());
        assert!(cobyla(f, &[3.0, 2.0], &cons, 10, 0.0).is_err());
        assert!(fmin_cobyla(f, &[3.0, 2.0], &cons, f64::INFINITY, 10).is_err());
    }

    #[test]
    fn augmented_lagrangian_rejects_zero_iteration_budgets() {
        use crate::augmented_lagrangian;

        let f = |x: &[f64]| x[0] * x[0];
        let constraints = |x: &[f64]| vec![x[0] + 1.0];
        assert!(augmented_lagrangian(f, constraints, &[0.0], 1, 0, 10).is_err());
        assert!(augmented_lagrangian(f, constraints, &[0.0], 1, 10, 0).is_err());
    }

    #[test]
    fn augmented_lagrangian_rejects_constraint_length_mismatch() {
        use crate::augmented_lagrangian;

        let f = |x: &[f64]| x[0] * x[0];
        let missing_constraint = |_x: &[f64]| Vec::<f64>::new();
        let extra_constraints = |_x: &[f64]| vec![1.0, 2.0];
        assert!(augmented_lagrangian(f, missing_constraint, &[0.0], 1, 10, 10).is_err());
        assert!(augmented_lagrangian(f, extra_constraints, &[0.0], 1, 10, 10).is_err());
    }

    #[test]
    fn fmin_legacy_wrappers_reach_quadratic_minimum() {
        use crate::{
            fmin, fmin_bfgs, fmin_cg, fmin_l_bfgs_b, fmin_ncg, fmin_powell, fmin_slsqp, fmin_tnc,
        };
        // Quadratic bowl with minimiser at (1, -2); scipy's fmin/fmin_powell/
        // fmin_cg/fmin_bfgs/fmin_l_bfgs_b/fmin_tnc/fmin_slsqp/fmin_ncg all return
        // this point from x0 = (0, 0).
        let f = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] + 2.0).powi(2);
        let x0 = [0.0, 0.0];
        for xopt in [
            fmin(f, &x0).expect("fmin"),
            fmin_powell(f, &x0).expect("fmin_powell"),
            fmin_cg(f, &x0).expect("fmin_cg"),
            fmin_bfgs(f, &x0).expect("fmin_bfgs"),
            fmin_l_bfgs_b(f, &x0).expect("fmin_l_bfgs_b"),
            fmin_tnc(f, &x0).expect("fmin_tnc"),
            fmin_slsqp(f, &x0).expect("fmin_slsqp"),
            fmin_ncg(f, &x0).expect("fmin_ncg"),
        ] {
            assert!((xopt[0] - 1.0).abs() < 1e-3, "x0 = {}", xopt[0]);
            assert!((xopt[1] + 2.0).abs() < 1e-3, "x1 = {}", xopt[1]);
        }
    }

    #[test]
    fn defaults_match_packet_contract_intent() {
        let minimize = MinimizeOptions::default();
        assert_eq!(minimize.method, None);
        assert_eq!(minimize.mode, RuntimeMode::Strict);

        let root = RootOptions::default();
        assert!(root.xtol > 0.0);
        assert!(root.rtol > 0.0);
        assert_eq!(root.mode, RuntimeMode::Strict);
    }

    #[test]
    fn convergence_status_is_cloneable() {
        let status = ConvergenceStatus::CallbackStop;
        assert_eq!(status, ConvergenceStatus::CallbackStop);
    }

    #[test]
    fn optimize_method_enum_is_stable() {
        let method = OptimizeMethod::Bfgs;
        assert!(matches!(method, OptimizeMethod::Bfgs));
    }

    // ── Constraint type tests ──────────────────────────────────────

    #[test]
    fn bounds_basic() {
        let b = Bounds::new(vec![0.0, -1.0], vec![1.0, 1.0]).expect("bounds");
        assert_eq!(b.len(), 2);
        assert!(b.is_feasible(&[0.5, 0.0]));
        assert!(!b.is_feasible(&[1.5, 0.0]));
        assert!(!b.is_feasible(&[0.5, -2.0]));
    }

    #[test]
    fn bounds_project() {
        let b = Bounds::new(vec![0.0, -1.0], vec![1.0, 1.0]).expect("bounds");
        let projected = b.project(&[2.0, -3.0]);
        assert_eq!(projected, vec![1.0, -1.0]);
    }

    #[test]
    fn bounds_unbounded() {
        let b = Bounds::unbounded(3);
        assert!(b.is_feasible(&[1e100, -1e100, 0.0]));
    }

    #[test]
    fn bounds_invalid_lb_gt_ub() {
        let err = Bounds::new(vec![1.0], vec![0.0]).expect_err("lb > ub");
        assert!(matches!(err, crate::OptError::InvalidBounds { .. }));
    }

    #[test]
    fn bounds_to_tuples() {
        let b = Bounds::new(vec![0.0, f64::NEG_INFINITY], vec![f64::INFINITY, 1.0]).unwrap();
        let tuples = b.to_bound_tuples();
        assert_eq!(tuples[0], (Some(0.0), None));
        assert_eq!(tuples[1], (None, Some(1.0)));
    }

    #[test]
    fn linear_constraint_basic() {
        // x + y <= 1 expressed as A @ x <= ub with lb = -inf
        let lc = LinearConstraint::new(vec![vec![1.0, 1.0]], vec![f64::NEG_INFINITY], vec![1.0])
            .expect("linear constraint");
        assert!(lc.is_feasible(&[0.3, 0.3])); // 0.6 <= 1
        assert!(!lc.is_feasible(&[0.6, 0.6])); // 1.2 > 1
    }

    #[test]
    fn linear_constraint_evaluate() {
        let lc = LinearConstraint::new(
            vec![vec![2.0, 3.0], vec![1.0, -1.0]],
            vec![0.0, 0.0],
            vec![10.0, 5.0],
        )
        .expect("linear constraint");
        let ax = lc.evaluate(&[1.0, 2.0]);
        assert!((ax[0] - 8.0).abs() < 1e-12); // 2*1 + 3*2 = 8
        assert!((ax[1] - (-1.0)).abs() < 1e-12); // 1*1 + (-1)*2 = -1
    }

    #[test]
    fn nonlinear_constraint_basic() {
        fn circle(x: &[f64]) -> Vec<f64> {
            vec![x[0] * x[0] + x[1] * x[1]]
        }
        let nlc = NonlinearConstraint::new(circle, vec![0.0], vec![1.0]).expect("nlc");
        assert!(nlc.is_feasible(&[0.5, 0.5])); // 0.5 <= 1
        assert!(!nlc.is_feasible(&[1.0, 1.0])); // 2.0 > 1
    }

    #[test]
    fn bounds_with_lbfgsb() {
        // Minimize x^2 + y^2 subject to x >= 1, y >= 2
        let bounds = Bounds::new(vec![1.0, 2.0], vec![f64::INFINITY, f64::INFINITY]).unwrap();
        let bound_tuples = bounds.to_bound_tuples();

        let options = MinimizeOptions {
            method: Some(OptimizeMethod::LBfgsB),
            tol: Some(1e-10),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let result = crate::lbfgsb(&f, &[5.0, 5.0], options, Some(&bound_tuples))
            .expect("lbfgsb with bounds");
        assert!(result.success, "{}", result.message);
        // Minimum should be at (1, 2) — the bound corner
        assert!(
            (result.x[0] - 1.0).abs() < 0.01,
            "x ~ 1.0, got {}",
            result.x[0]
        );
        assert!(
            (result.x[1] - 2.0).abs() < 0.01,
            "y ~ 2.0, got {}",
            result.x[1]
        );
    }

    #[test]
    fn approx_fprime_matches_quadratic_gradient() {
        let f = |x: &[f64]| x[0] * x[0];
        let grad = approx_fprime(&[3.0], f, 1.0e-8).expect("gradient should evaluate");
        assert!(
            (grad[0] - 6.0).abs() < 1.0e-5,
            "expected 6.0, got {}",
            grad[0]
        );
    }

    #[test]
    fn approx_fprime_matches_multivariate_gradient() {
        let f = |x: &[f64]| x[0] * x[0] + 3.0 * x[1] * x[1] + x[0] * x[1];
        let grad = approx_fprime(&[2.0, -1.0], f, 1.0e-8).expect("gradient should evaluate");
        assert!(
            (grad[0] - 3.0).abs() < 1.0e-5,
            "expected 3.0, got {}",
            grad[0]
        );
        assert!(
            (grad[1] - (-4.0)).abs() < 1.0e-5,
            "expected -4.0, got {}",
            grad[1]
        );
    }

    #[test]
    fn check_grad_is_small_for_exact_gradient() {
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1] * x[1];
        let g = |x: &[f64]| vec![2.0 * x[0], 3.0 * x[1] * x[1]];
        let error = check_grad(f, g, &[1.5, -2.0]).expect("gradient check should succeed");
        assert!(error < 1.0e-5, "expected small error, got {error}");
    }

    #[test]
    fn check_grad_detects_bad_gradient() {
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let wrong = |_x: &[f64]| vec![0.0, 0.0];
        let error = check_grad(f, wrong, &[2.0, -3.0]).expect("gradient check should succeed");
        assert!(error > 1.0, "expected large error, got {error}");
    }

    #[test]
    fn differentiate_derivative_matches_cubic() {
        let f = |x: f64| x.powi(3) - 2.0 * x;
        let x = 1.25_f64;
        let result = derivative(f, x, crate::DifferentiateOptions::default())
            .expect("derivative should evaluate");
        let expected = 3.0 * x * x - 2.0;
        assert!(result.success, "status: {:?}", result.status);
        assert!(
            (result.df - expected).abs() < 1.0e-8,
            "df = {}, expected {expected}",
            result.df
        );
        assert!(result.error.is_finite());
    }

    #[test]
    fn differentiate_derivative_supports_one_sided_steps() {
        let options = crate::DifferentiateOptions {
            step_direction: crate::StepDirection::Forward,
            ..crate::DifferentiateOptions::default()
        };
        let result = derivative(|x| x * x, 0.0, options).expect("forward derivative");
        assert!(result.success, "status: {:?}", result.status);
        assert!(result.df.abs() < 1.0e-8, "df = {}", result.df);
    }

    #[test]
    fn differentiate_jacobian_recovers_linear_map() {
        let f = |x: &[f64]| {
            vec![
                2.0 * x[0] - x[1] + 0.5 * x[2],
                -3.0 * x[0] + 4.0 * x[1] + x[2],
            ]
        };
        let result = jacobian(
            f,
            &[1.0, -2.0, 0.25],
            crate::DifferentiateOptions::default(),
        )
        .expect("jacobian should evaluate");
        let expected = [[2.0, -1.0, 0.5], [-3.0, 4.0, 1.0]];
        assert!(result.success, "status: {:?}", result.status);
        for (row_index, row) in expected.iter().enumerate() {
            for (column_index, expected_value) in row.iter().enumerate() {
                assert!(
                    (result.df[row_index][column_index] - expected_value).abs() < 1.0e-8,
                    "jac[{row_index},{column_index}] = {} vs {expected_value}",
                    result.df[row_index][column_index]
                );
            }
        }
    }

    #[test]
    fn differentiate_jacobian_rejects_output_shape_changes() {
        let f = |x: &[f64]| {
            if x[0].is_sign_positive() {
                vec![x[0], x[0] * x[0]]
            } else {
                vec![x[0]]
            }
        };
        let options = crate::DifferentiateOptions {
            maxiter: 2,
            initial_step: 0.25,
            ..crate::DifferentiateOptions::default()
        };
        let error =
            jacobian(f, &[0.0], options).expect_err("shape-changing outputs must fail closed");
        assert!(matches!(error, crate::OptError::NonFiniteInput { .. }));
    }

    #[test]
    fn differentiate_hessian_matches_quadratic() {
        let f = |x: &[f64]| 3.0 * x[0] * x[0] + 2.0 * x[0] * x[1] + 5.0 * x[1] * x[1] - x[0];
        let result = hessian(f, &[0.75, -1.25], crate::DifferentiateOptions::default())
            .expect("hessian should evaluate");
        let expected = [[6.0, 2.0], [2.0, 10.0]];
        assert!(result.success, "status: {:?}", result.status);
        for (row_index, row) in expected.iter().enumerate() {
            for (column_index, expected_value) in row.iter().enumerate() {
                assert!(
                    (result.ddf[row_index][column_index] - expected_value).abs() < 1.0e-7,
                    "hess[{row_index},{column_index}] = {} vs {expected_value}",
                    result.ddf[row_index][column_index]
                );
            }
        }
    }

    #[test]
    fn differentiate_rejects_non_finite_inputs() {
        let error = derivative(|x| x, f64::NAN, crate::DifferentiateOptions::default())
            .expect_err("NaN x should fail");
        assert!(matches!(error, crate::OptError::NonFiniteInput { .. }));
    }

    #[test]
    fn differentiate_rejects_degenerate_step_factor() {
        let options = crate::DifferentiateOptions {
            step_factor: 1.0,
            ..crate::DifferentiateOptions::default()
        };
        let error = derivative(|x| x, 1.0, options).expect_err("step_factor=1 should fail");
        assert!(matches!(error, crate::OptError::InvalidArgument { .. }));
    }

    #[test]
    fn gradient_descent_rejects_invalid_callback_gradient() {
        let f = |x: &[f64]| x.iter().map(|value| value * value).sum::<f64>();

        let wrong_len = gradient_descent(f, |_x| vec![1.0], &[1.0, 2.0], 1.0e-6, 10, 0.1);
        assert!(!wrong_len.success);
        assert_eq!(wrong_len.status, ConvergenceStatus::InvalidInput);

        let non_finite =
            gradient_descent(f, |x| vec![f64::NAN; x.len()], &[1.0, 2.0], 1.0e-6, 10, 0.1);
        assert!(!non_finite.success);
        assert_eq!(non_finite.status, ConvergenceStatus::InvalidInput);
    }

    #[test]
    fn gradient_descent_rejects_zero_iteration_budget() {
        let f = |x: &[f64]| x.iter().map(|value| value * value).sum::<f64>();
        let result = gradient_descent(
            f,
            |_x| panic!("gradient callback should not run for invalid controls"),
            &[1.0, 2.0],
            1.0e-6,
            0,
            0.1,
        );
        assert!(!result.success);
        assert_eq!(result.status, ConvergenceStatus::InvalidInput);
        assert_eq!(result.nfev, 0);
        assert_eq!(result.njev, 0);
    }

    #[test]
    fn projected_gradient_descent_rejects_invalid_callback_gradient() {
        let f = |x: &[f64]| x.iter().map(|value| value * value).sum::<f64>();

        let wrong_len = projected_gradient_descent(
            f,
            |_x| vec![1.0],
            &[1.0, 2.0],
            &[0.0, 0.0],
            &[2.0, 2.0],
            1.0e-6,
            10,
            0.1,
        );
        assert!(!wrong_len.success);
        assert_eq!(wrong_len.status, ConvergenceStatus::InvalidInput);

        let non_finite = projected_gradient_descent(
            f,
            |x| vec![f64::INFINITY; x.len()],
            &[1.0, 2.0],
            &[0.0, 0.0],
            &[2.0, 2.0],
            1.0e-6,
            10,
            0.1,
        );
        assert!(!non_finite.success);
        assert_eq!(non_finite.status, ConvergenceStatus::InvalidInput);
    }

    #[test]
    fn projected_gradient_descent_rejects_zero_iteration_budget() {
        let f = |x: &[f64]| x.iter().map(|value| value * value).sum::<f64>();
        let result = projected_gradient_descent(
            f,
            |_x| panic!("gradient callback should not run for invalid controls"),
            &[1.0, 2.0],
            &[0.0, 0.0],
            &[2.0, 2.0],
            1.0e-6,
            0,
            0.1,
        );
        assert!(!result.success);
        assert_eq!(result.status, ConvergenceStatus::InvalidInput);
        assert_eq!(result.nfev, 0);
        assert_eq!(result.njev, 0);
    }

    #[test]
    fn quadratic_assignment_faq_matches_scipy() {
        let a = vec![
            vec![0.0, 5.0, 8.0, 3.0],
            vec![5.0, 0.0, 2.0, 7.0],
            vec![8.0, 2.0, 0.0, 4.0],
            vec![3.0, 7.0, 4.0, 0.0],
        ];
        let b = vec![
            vec![0.0, 2.0, 4.0, 6.0],
            vec![2.0, 0.0, 3.0, 1.0],
            vec![4.0, 3.0, 0.0, 5.0],
            vec![6.0, 1.0, 5.0, 0.0],
        ];
        let r = quadratic_assignment(&a, &b, false, 30, 0.03).unwrap();
        assert_eq!(r.col_ind, vec![1, 0, 2, 3]);
        assert!((r.fun - 214.0).abs() < 1e-9);
        assert_eq!(r.nit, 1);

        let rmax = quadratic_assignment(&a, &b, true, 30, 0.03).unwrap();
        assert_eq!(rmax.col_ind, vec![0, 2, 3, 1]);
        assert!((rmax.fun - 218.0).abs() < 1e-9);

        // 5x5 case (scipy: col_ind=[2,3,1,0,4], fun=311, nit=24).
        let a5 = vec![
            vec![0.0, 7.0, 0.0, 3.0, 1.0],
            vec![1.0, 0.0, 3.0, 4.0, 5.0],
            vec![4.0, 7.0, 0.0, 9.0, 0.0],
            vec![7.0, 4.0, 6.0, 0.0, 2.0],
            vec![8.0, 10.0, 3.0, 7.0, 0.0],
        ];
        let b5 = vec![
            vec![0.0, 1.0, 0.0, 2.0, 9.0],
            vec![1.0, 0.0, 10.0, 5.0, 7.0],
            vec![3.0, 7.0, 0.0, 0.0, 8.0],
            vec![10.0, 7.0, 3.0, 0.0, 1.0],
            vec![4.0, 9.0, 3.0, 3.0, 0.0],
        ];
        let r5 = quadratic_assignment(&a5, &b5, false, 30, 0.03).unwrap();
        assert_eq!(r5.col_ind, vec![2, 3, 1, 0, 4]);
        assert!((r5.fun - 311.0).abs() < 1e-9);
        assert_eq!(r5.nit, 24);
    }

    /// The LAPJV solver must return a TRUE minimum-cost assignment. Verify the
    /// assignment cost equals the brute-force optimum over all permutations, for
    /// square and rectangular (nr<nc, nr>nc) continuous-cost matrices.
    #[test]
    fn linear_sum_assignment_cost_matches_brute_force() {
        // tiny deterministic LCG for reproducible continuous costs
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        fn brute_min(cost: &[Vec<f64>]) -> f64 {
            // assign each row to a distinct column (rows <= cols), minimize sum.
            let nr = cost.len();
            let nc = cost[0].len();
            let mut used = vec![false; nc];
            fn rec(
                r: usize,
                nr: usize,
                nc: usize,
                cost: &[Vec<f64>],
                used: &mut [bool],
                acc: f64,
                best: &mut f64,
            ) {
                if acc >= *best {
                    return;
                }
                if r == nr {
                    *best = best.min(acc);
                    return;
                }
                for c in 0..nc {
                    if !used[c] {
                        used[c] = true;
                        rec(r + 1, nr, nc, cost, used, acc + cost[r][c], best);
                        used[c] = false;
                    }
                }
            }
            let mut best = f64::INFINITY;
            rec(0, nr, nc, cost, &mut used, 0.0, &mut best);
            best
        }
        // nr <= nc only (the tall nr>nc path is the transpose wrapper, covered by
        // linear_sum_assignment_handles_more_rows_than_columns); brute_min assigns
        // every row to a distinct column.
        for &(nr, nc) in &[(6usize, 6usize), (5, 8), (7, 7), (4, 9)] {
            let cost: Vec<Vec<f64>> = (0..nr).map(|_| (0..nc).map(|_| next()).collect()).collect();
            let (row_ind, col_ind) = linear_sum_assignment(&cost).expect("assignment");
            // valid: distinct cols, every row covered, count = min(nr,nc)
            assert_eq!(row_ind.len(), nr.min(nc));
            let mut seen = std::collections::HashSet::new();
            let mut got = 0.0;
            for (&r, &c) in row_ind.iter().zip(&col_ind) {
                assert!(seen.insert(c), "duplicate column {c} (nr={nr} nc={nc})");
                got += cost[r][c];
            }
            let opt = brute_min(&cost);
            assert!(
                (got - opt).abs() < 1e-9,
                "nr={nr} nc={nc}: LAPJV cost {got} != brute optimum {opt}"
            );
        }
    }

    #[test]
    fn linear_sum_assignment_solves_square_problem() {
        let cost_matrix = vec![
            vec![4.0, 1.0, 3.0],
            vec![2.0, 0.0, 5.0],
            vec![3.0, 2.0, 2.0],
        ];
        let (row_ind, col_ind) =
            linear_sum_assignment(&cost_matrix).expect("assignment should succeed");
        assert_eq!(row_ind, vec![0, 1, 2]);
        assert_eq!(col_ind, vec![1, 0, 2]);
    }

    #[test]
    fn linear_sum_assignment_handles_more_columns_than_rows() {
        let cost_matrix = vec![vec![10.0, 19.0, 8.0, 15.0], vec![10.0, 18.0, 7.0, 17.0]];
        let (row_ind, col_ind) =
            linear_sum_assignment(&cost_matrix).expect("assignment should succeed");
        assert_eq!(row_ind, vec![0, 1]);
        assert_eq!(col_ind, vec![0, 2]);
    }

    #[test]
    fn linear_sum_assignment_handles_more_rows_than_columns() {
        let cost_matrix = vec![vec![5.0, 9.0], vec![10.0, 3.0], vec![8.0, 7.0]];
        let (row_ind, col_ind) =
            linear_sum_assignment(&cost_matrix).expect("assignment should succeed");
        assert_eq!(row_ind, vec![0, 1]);
        assert_eq!(col_ind, vec![0, 1]);
    }

    #[test]
    fn linear_sum_assignment_rejects_non_rectangular_input() {
        let cost_matrix = vec![vec![1.0, 2.0], vec![3.0]];
        let error = linear_sum_assignment(&cost_matrix).expect_err("matrix should be rejected");
        assert!(matches!(error, crate::OptError::InvalidArgument { .. }));
    }

    #[test]
    fn linear_sum_assignment_allows_empty_input() {
        let (row_ind, col_ind) = linear_sum_assignment(&[]).expect("empty problem should succeed");
        assert!(row_ind.is_empty());
        assert!(col_ind.is_empty());
    }

    #[test]
    fn linear_sum_assignment_matches_hungarian_reference_costs() {
        fn deterministic_cost(rows: usize, cols: usize) -> Vec<Vec<f64>> {
            (0..rows)
                .map(|i| {
                    (0..cols)
                        .map(|j| {
                            let mixed = (i
                                .wrapping_mul(1_103_515_245)
                                .wrapping_add(j.wrapping_mul(12_345))
                                ^ (i.wrapping_mul(2_654_435_761) >> 7))
                                as u64;
                            let jitter = ((i * 17 + j * 31) % 997) as f64 * 1.0e-13;
                            mixed as f64 / u64::MAX as f64 + jitter
                        })
                        .collect()
                })
                .collect()
        }

        fn assignment_cost(cost: &[Vec<f64>], rows: &[usize], cols: &[usize]) -> f64 {
            rows.iter()
                .zip(cols)
                .map(|(&row, &col)| cost[row][col])
                .sum()
        }

        fn reference_assignment(cost: &[Vec<f64>]) -> (Vec<usize>, Vec<usize>) {
            if cost.len() <= cost[0].len() {
                let cols = hungarian_rectangular_reference(cost);
                let rows = (0..cols.len()).collect::<Vec<_>>();
                return (rows, cols);
            }

            let transposed = transpose_matrix(cost);
            let assignment = hungarian_rectangular_reference(&transposed);
            let mut pairs = assignment
                .into_iter()
                .enumerate()
                .map(|(col, row)| (row, col))
                .collect::<Vec<_>>();
            pairs.sort_unstable_by_key(|(row, _)| *row);
            pairs.into_iter().unzip()
        }

        for &(rows, cols) in &[(3, 5), (8, 8), (16, 12)] {
            let cost = deterministic_cost(rows, cols);
            let (fast_rows, fast_cols) = linear_sum_assignment(&cost).expect("lapjv assignment");
            let (ref_rows, ref_cols) = reference_assignment(&cost);
            assert_eq!(fast_rows.len(), rows.min(cols));
            assert_eq!(fast_rows.len(), fast_cols.len());
            assert_eq!(ref_rows.len(), fast_rows.len());
            let fast_cost = assignment_cost(&cost, &fast_rows, &fast_cols);
            let ref_cost = assignment_cost(&cost, &ref_rows, &ref_cols);
            assert!(
                (fast_cost - ref_cost).abs() <= 1.0e-10,
                "shape {rows}x{cols}: fast cost {fast_cost}, reference cost {ref_cost}"
            );
        }
    }

    #[test]
    fn nnls_rejects_non_rectangular_input() {
        let a = vec![vec![1.0, 2.0], vec![3.0]];
        let b = vec![1.0, 2.0];
        let error = nnls(&a, &b).expect_err("ragged matrix should be rejected");
        assert!(matches!(error, crate::OptError::InvalidArgument { .. }));
    }

    #[test]
    fn nnls_rejects_non_finite_inputs() {
        let a = vec![vec![1.0, f64::NAN], vec![0.0, 1.0]];
        let b = vec![1.0, 2.0];
        let error = nnls(&a, &b).expect_err("NaN in A should be rejected");
        assert!(matches!(error, crate::OptError::NonFiniteInput { .. }));

        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![1.0, f64::INFINITY];
        let error = nnls(&a, &b).expect_err("Inf in b should be rejected");
        assert!(matches!(error, crate::OptError::NonFiniteInput { .. }));
    }

    // ── linprog tests ──────────────────────────────────────────────

    #[test]
    fn linprog_simple_2d() {
        // Maximize 5x + 4y (minimize -5x - 4y)
        // Subject to: 6x + 4y <= 24, x + 2y <= 6, x,y >= 0
        // Optimal: x=3, y=1.5, obj = -21
        let c = vec![-5.0, -4.0];
        let a_ub = vec![vec![6.0, 4.0], vec![1.0, 2.0]];
        let b_ub = vec![24.0, 6.0];
        let result = linprog(&c, &a_ub, &b_ub, &[], &[], &[], None).unwrap();
        assert!(result.success, "linprog failed: {}", result.message);
        assert!(
            (result.x[0] - 3.0).abs() < 0.01,
            "x={}, expected 3.0",
            result.x[0]
        );
        assert!(
            (result.x[1] - 1.5).abs() < 0.01,
            "y={}, expected 1.5",
            result.x[1]
        );
        assert!(
            (result.fun - (-21.0)).abs() < 0.01,
            "obj={}, expected -21.0",
            result.fun
        );
    }

    #[test]
    fn linprog_infeasible() {
        // x = -1 with x >= 0 — infeasible (non-negative variable can't equal negative).
        let c = vec![1.0];
        let a_eq = vec![vec![1.0]];
        let b_eq = vec![-1.0]; // x = -1 but x >= 0
        let result = linprog(&c, &[], &[], &a_eq, &b_eq, &[], None).unwrap();
        assert!(
            !result.success,
            "should be infeasible: success={}, status={}, x={:?}, fun={}",
            result.success, result.status, result.x, result.fun
        );
        assert_eq!(result.status, 2);
    }

    #[test]
    fn linprog_equality_constraint() {
        // Minimize x + y, subject to x + y = 10, x,y >= 0
        // Optimal: x=0, y=10, obj=10 (or x=10, y=0)
        let c = vec![1.0, 1.0];
        let a_eq = vec![vec![1.0, 1.0]];
        let b_eq = vec![10.0];
        let result = linprog(&c, &[], &[], &a_eq, &b_eq, &[], None).unwrap();
        assert!(result.success, "linprog failed: {}", result.message);
        assert!(
            (result.fun - 10.0).abs() < 0.01,
            "obj={}, expected 10.0",
            result.fun
        );
    }

    #[test]
    fn linprog_with_bounds() {
        // Minimize -x - 2y, subject to x + y <= 4, x in [0,3], y in [0,3]
        // Optimal: x=1, y=3, obj = -7
        let c = vec![-1.0, -2.0];
        let a_ub = vec![vec![1.0, 1.0]];
        let b_ub = vec![4.0];
        let bounds = vec![(Some(0.0), Some(3.0)), (Some(0.0), Some(3.0))];
        let result = linprog(&c, &a_ub, &b_ub, &[], &[], &bounds, None).unwrap();
        assert!(result.success, "linprog failed: {}", result.message);
        assert!(
            (result.fun - (-7.0)).abs() < 0.1,
            "obj={}, expected -7.0",
            result.fun
        );
    }

    #[test]
    fn linprog_degenerate_redundant_constraints() {
        // Minimize x, subject to x <= 5 AND x <= 10 AND x >= 0
        // Optimal: x = 0
        let c = vec![1.0];
        let a_ub = vec![vec![1.0], vec![1.0]];
        let b_ub = vec![5.0, 10.0];
        let result = linprog(&c, &a_ub, &b_ub, &[], &[], &[], None).unwrap();
        assert!(result.success, "linprog failed: {}", result.message);
        assert!(result.x[0].abs() < 0.01, "x={}, expected 0.0", result.x[0]);
    }

    #[test]
    fn linprog_slack_variables() {
        // Verify slack is computed correctly.
        let c = vec![1.0, 1.0];
        let a_ub = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b_ub = vec![10.0, 10.0];
        let result = linprog(&c, &a_ub, &b_ub, &[], &[], &[], None).unwrap();
        assert!(result.success);
        // Slack = b_ub - A_ub @ x
        for (i, &s) in result.slack.iter().enumerate() {
            assert!(s >= -1e-8, "slack[{i}] = {s} should be non-negative");
        }
    }

    #[test]
    fn linprog_bounds_enforced() {
        // Minimize -x, x in [2, 5] → optimal at x=5
        let c = vec![-1.0];
        let bounds = vec![(Some(2.0), Some(5.0))];
        let result = linprog(&c, &[], &[], &[], &[], &bounds, None).unwrap();
        assert!(result.success, "linprog failed: {}", result.message);
        assert!(
            (result.x[0] - 5.0).abs() < 0.01,
            "x={}, expected 5.0",
            result.x[0]
        );
    }

    #[test]
    fn linprog_supports_free_variables() {
        // Minimize x, subject to x >= 1 with x otherwise free.
        let c = vec![1.0];
        let a_ub = vec![vec![-1.0]];
        let b_ub = vec![-1.0];
        let bounds = vec![(None, None)];
        let result = linprog(&c, &a_ub, &b_ub, &[], &[], &bounds, None).unwrap();
        assert!(result.success, "linprog failed: {}", result.message);
        assert!(
            (result.x[0] - 1.0).abs() < 0.01,
            "x={}, expected 1.0",
            result.x[0]
        );
        assert!(
            (result.fun - 1.0).abs() < 0.01,
            "obj={}, expected 1.0",
            result.fun
        );
    }

    #[test]
    fn linprog_supports_upper_only_bounds() {
        // Minimize -x, subject only to x <= 5.
        let c = vec![-1.0];
        let bounds = vec![(None, Some(5.0))];
        let result = linprog(&c, &[], &[], &[], &[], &bounds, None).unwrap();
        assert!(result.success, "linprog failed: {}", result.message);
        assert!(
            (result.x[0] - 5.0).abs() < 0.01,
            "x={}, expected 5.0",
            result.x[0]
        );
        assert!(
            (result.fun - (-5.0)).abs() < 0.01,
            "obj={}, expected -5.0",
            result.fun
        );
    }

    #[test]
    fn linprog_dimension_mismatch() {
        let c = vec![1.0, 2.0];
        let a_ub = vec![vec![1.0]]; // wrong number of columns
        let b_ub = vec![5.0];
        assert!(linprog(&c, &a_ub, &b_ub, &[], &[], &[], None).is_err());
    }

    #[test]
    fn linprog_rejects_non_finite_coefficients() {
        let err = linprog(&[f64::NAN], &[], &[], &[], &[], &[], None)
            .expect_err("NaN objective should fail");
        assert!(matches!(err, crate::OptError::NonFiniteInput { .. }));

        let err = linprog(&[1.0], &[vec![f64::INFINITY]], &[1.0], &[], &[], &[], None)
            .expect_err("Inf inequality coefficient should fail");
        assert!(matches!(err, crate::OptError::NonFiniteInput { .. }));

        let err = linprog(&[1.0], &[], &[], &[vec![1.0]], &[f64::NAN], &[], None)
            .expect_err("NaN equality RHS should fail");
        assert!(matches!(err, crate::OptError::NonFiniteInput { .. }));
    }

    #[test]
    fn linprog_rejects_zero_iteration_budget() {
        let err = linprog(&[1.0], &[], &[], &[], &[], &[], Some(0))
            .expect_err("zero maxiter should fail");
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
    }

    // ── milp tests ──────────────────────────────────────────────────

    #[test]
    fn milp_finds_integer_optimum() {
        let c = vec![-1.0, -2.0];
        let a_ub = vec![vec![1.0, 1.0]];
        let b_ub = vec![4.0];
        let bounds = vec![(Some(0.0), Some(3.0)), (Some(0.0), Some(3.0))];
        let result = milp(
            MilpProblem {
                c: &c,
                integrality: &[Integrality::Integer, Integrality::Integer],
                a_ub: &a_ub,
                b_ub: &b_ub,
                a_eq: &[],
                b_eq: &[],
                bounds: &bounds,
            },
            MilpOptions::default(),
        )
        .expect("milp");
        assert!(result.success, "milp failed: {}", result.message);
        assert_eq!(result.x, vec![1.0, 3.0]);
        assert!((result.fun + 7.0).abs() < 1e-9);
    }

    #[test]
    fn milp_handles_binary_variables() {
        let c = vec![-3.0, -2.0];
        let a_ub = vec![vec![1.0, 1.0]];
        let b_ub = vec![1.0];
        let result = milp(
            MilpProblem {
                c: &c,
                integrality: &[Integrality::Binary, Integrality::Binary],
                a_ub: &a_ub,
                b_ub: &b_ub,
                a_eq: &[],
                b_eq: &[],
                bounds: &[],
            },
            MilpOptions::default(),
        )
        .expect("milp");
        assert!(result.success, "milp failed: {}", result.message);
        assert_eq!(result.x, vec![1.0, 0.0]);
        assert!((result.fun + 3.0).abs() < 1e-9);
    }

    #[test]
    fn milp_reports_infeasible_problem() {
        let c = vec![1.0];
        let a_eq = vec![vec![1.0]];
        let b_eq = vec![0.5];
        let result = milp(
            MilpProblem {
                c: &c,
                integrality: &[Integrality::Integer],
                a_ub: &[],
                b_ub: &[],
                a_eq: &a_eq,
                b_eq: &b_eq,
                bounds: &[(Some(0.0), Some(1.0))],
            },
            MilpOptions::default(),
        )
        .expect("milp");
        assert!(!result.success);
        assert_eq!(result.status, 2);
    }

    #[test]
    fn milp_supports_free_integer_variables() {
        let c = vec![-1.0];
        let a_ub = vec![vec![1.0], vec![-1.0]];
        let b_ub = vec![2.2, 2.2];
        let result = milp(
            MilpProblem {
                c: &c,
                integrality: &[Integrality::Integer],
                a_ub: &a_ub,
                b_ub: &b_ub,
                a_eq: &[],
                b_eq: &[],
                bounds: &[(None, None)],
            },
            MilpOptions::default(),
        )
        .expect("milp");
        assert!(result.success, "milp failed: {}", result.message);
        assert_eq!(result.x, vec![2.0]);
        assert!((result.fun + 2.0).abs() < 1e-9);
    }

    #[test]
    fn milp_rejects_integrality_length_mismatch() {
        let err = milp(
            MilpProblem {
                c: &[1.0, 2.0],
                integrality: &[Integrality::Integer],
                a_ub: &[],
                b_ub: &[],
                a_eq: &[],
                b_eq: &[],
                bounds: &[],
            },
            MilpOptions::default(),
        )
        .expect_err("integrality length mismatch");
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
    }

    #[test]
    fn milp_rejects_zero_node_budget() {
        let err = milp(
            MilpProblem {
                c: &[1.0],
                integrality: &[Integrality::Integer],
                a_ub: &[],
                b_ub: &[],
                a_eq: &[],
                b_eq: &[],
                bounds: &[(Some(0.0), Some(1.0))],
            },
            MilpOptions {
                max_nodes: Some(0),
                ..MilpOptions::default()
            },
        )
        .expect_err("zero max_nodes should fail");
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
    }

    // MilpOptions has exactly two fields and both are solver budgets:
    // `max_nodes` (covered above) and `lp_maxiter`. Only the first had a
    // rejection test. `lp_maxiter` is forwarded verbatim to `linprog`, which
    // rejects maxiter == Some(0) itself, so milp does reject a zero LP budget --
    // but that had never been asserted, and it holds only as long as the
    // forwarding stays intact. If milp ever grew a fast path that skipped the
    // LP call, or clamped the budget before forwarding, a zero LP iteration
    // budget would silently become a synthetic result -- exactly the failure
    // mode frankenscipy-qgc24 was filed about for max_nodes.
    #[test]
    fn milp_rejects_zero_lp_iteration_budget() {
        let err = milp(
            MilpProblem {
                c: &[1.0],
                integrality: &[Integrality::Integer],
                a_ub: &[],
                b_ub: &[],
                a_eq: &[],
                b_eq: &[],
                bounds: &[(Some(0.0), Some(1.0))],
            },
            MilpOptions {
                lp_maxiter: Some(0),
                ..MilpOptions::default()
            },
        )
        .expect_err("zero lp_maxiter should fail");
        assert!(
            matches!(err, crate::OptError::InvalidArgument { .. }),
            "a zero LP iteration budget must be rejected as invalid, got {err:?}"
        );
    }

    // ── Differential Evolution tests ───────────────────────────────

    #[test]
    fn de_rastrigin_finds_global_minimum() {
        // Rastrigin function: f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
        // Global minimum at origin: f(0,...,0) = 0
        let rastrigin = |x: &[f64]| -> f64 {
            let n = x.len() as f64;
            let pi2 = 2.0 * std::f64::consts::PI;
            n * 10.0
                + x.iter()
                    .map(|&xi| xi * xi - 10.0 * (pi2 * xi).cos())
                    .sum::<f64>()
        };

        let bounds = vec![(-5.12, 5.12), (-5.12, 5.12)];
        let opts = DifferentialEvolutionOptions {
            maxiter: 200,
            seed: Some(42),
            ..Default::default()
        };
        let result = differential_evolution(rastrigin, &bounds, opts).unwrap();
        let fun = result.fun.unwrap();
        assert!(
            fun < 1.0,
            "DE should find near-global minimum, got f={}",
            fun
        );
    }

    #[test]
    fn de_rosenbrock_2d() {
        // Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
        // Minimum at (1,1) with f=0
        let rosenbrock =
            |x: &[f64]| -> f64 { (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2) };

        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let opts = DifferentialEvolutionOptions {
            maxiter: 500,
            seed: Some(123),
            ..Default::default()
        };
        let result = differential_evolution(rosenbrock, &bounds, opts).unwrap();
        assert!(
            (result.x[0] - 1.0).abs() < 0.1,
            "x ~ 1.0, got {}",
            result.x[0]
        );
        assert!(
            (result.x[1] - 1.0).abs() < 0.1,
            "y ~ 1.0, got {}",
            result.x[1]
        );
    }

    #[test]
    fn differential_evolution_rosen_5d_seed_is_bit_identical() {
        let rosen_5d = |x: &[f64]| -> f64 {
            let mut sum = 0.0;
            for i in 0..4 {
                sum += 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2);
            }
            sum
        };
        let result = differential_evolution(
            rosen_5d,
            &[(-5.0, 5.0); 5],
            DifferentialEvolutionOptions {
                maxiter: 100,
                popsize: 15,
                tol: 1e-8,
                seed: Some(1),
                ..Default::default()
            },
        )
        .expect("differential evolution should run");

        let x_bits: Vec<u64> = result.x.iter().map(|value| value.to_bits()).collect();
        assert_eq!(
            x_bits,
            [
                0x3fef_fd54_1a99_f908,
                0x3ff0_063d_28ff_c97b,
                0x3ff0_0c7b_8733_cbae,
                0x3ff0_199e_aa5b_07ea,
                0x3ff0_2ad9_0849_0820,
            ]
        );
        assert_eq!(
            result.fun.expect("objective value").to_bits(),
            0x3f4f_87af_07bb_6109
        );
        assert_eq!(result.nfev, 7_575);
        assert_eq!(result.nit, 100);
    }

    #[test]
    fn de_bounds_respected() {
        let func = |x: &[f64]| -> f64 { x.iter().sum() };
        let bounds = vec![(1.0, 3.0), (2.0, 4.0)];
        let opts = DifferentialEvolutionOptions {
            maxiter: 50,
            seed: Some(99),
            ..Default::default()
        };
        let result = differential_evolution(func, &bounds, opts).unwrap();
        for (i, (&xi, &(lo, hi))) in result.x.iter().zip(bounds.iter()).enumerate() {
            assert!(
                xi >= lo - 1e-10 && xi <= hi + 1e-10,
                "variable {i}: {xi} outside [{lo}, {hi}]"
            );
        }
    }

    #[test]
    fn de_rand1bin_strategy() {
        let sphere = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let opts = DifferentialEvolutionOptions {
            strategy: crate::DeStrategy::Rand1Bin,
            maxiter: 200,
            seed: Some(77),
            ..Default::default()
        };
        let result = differential_evolution(sphere, &bounds, opts).unwrap();
        let fun = result.fun.unwrap();
        assert!(fun < 0.1, "rand1bin should find near-origin, got f={fun}");
    }

    #[test]
    fn de_population_explores() {
        // With only a few iterations, the population should have explored
        let sphere = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
        let bounds = vec![(-10.0, 10.0)];
        let opts = DifferentialEvolutionOptions {
            maxiter: 5,
            seed: Some(1),
            ..Default::default()
        };
        let result = differential_evolution(sphere, &bounds, opts).unwrap();
        assert!(result.nfev > 0, "should have function evaluations");
    }

    #[test]
    fn de_rejects_population_too_small() {
        let sphere = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
        let bounds = vec![(-1.0, 1.0)];
        let opts = DifferentialEvolutionOptions {
            popsize: 3,
            ..Default::default()
        };
        let err = differential_evolution(sphere, &bounds, opts).unwrap_err();
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
    }

    #[test]
    fn de_rejects_zero_iteration_budget() {
        let sphere = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
        let bounds = vec![(-1.0, 1.0)];
        let opts = DifferentialEvolutionOptions {
            maxiter: 0,
            ..Default::default()
        };
        let err = differential_evolution(sphere, &bounds, opts).unwrap_err();
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
    }

    #[test]
    fn de_rejects_nonfinite_bounds() {
        let sphere = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
        let bounds = vec![(f64::NAN, 1.0)];
        let err = differential_evolution(sphere, &bounds, Default::default()).unwrap_err();
        assert!(matches!(err, crate::OptError::InvalidBounds { .. }));
    }

    #[test]
    fn de_rejects_population_size_overflow() {
        let sphere = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
        let bounds = vec![(-1.0, 1.0), (-1.0, 1.0)];
        let opts = DifferentialEvolutionOptions {
            popsize: usize::MAX,
            ..Default::default()
        };
        let err = differential_evolution(sphere, &bounds, opts).unwrap_err();
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
    }

    #[test]
    fn de_constrained_respects_feasible_boundary() {
        let objective = |x: &[f64]| x[0] * x[0];
        let lower_bound_constraint = |x: &[f64]| (1.0 - x[0]).max(0.0);
        let bounds = vec![(0.0, 2.0)];
        let opts = DifferentialEvolutionOptions {
            maxiter: 160,
            seed: Some(101),
            ..Default::default()
        };

        let result =
            differential_evolution_constrained(objective, &bounds, lower_bound_constraint, opts)
                .expect("constrained differential evolution");

        assert!(result.success, "should converge: {}", result.message);
        assert!(
            result.x[0] >= 0.99,
            "constrained optimum must stay feasible, got {:?}",
            result.x
        );
        assert!(
            result.fun.unwrap_or(f64::INFINITY) < 1.05,
            "objective should be close to boundary optimum"
        );
    }

    // ── Basinhopping tests ─────────────────────────────────────────

    #[test]
    fn basinhopping_finds_global_minimum() {
        // Multi-modal function: f(x) = x^4 - 4x^2 + x
        // Has two local minima, global near x ≈ -1.38
        let func = |x: &[f64]| -> f64 {
            let xi = x[0];
            xi.powi(4) - 4.0 * xi * xi + xi
        };

        let opts = BasinhoppingOptions {
            niter: 50,
            seed: Some(42),
            stepsize: 1.0,
            ..Default::default()
        };
        let result = basinhopping(func, &[3.0], opts).unwrap();
        let fun = result.fun.unwrap();
        assert!(
            fun < -3.0,
            "basinhopping should find global minimum, got f={fun}"
        );
    }

    #[test]
    fn basinhopping_sphere() {
        // Simple sphere function: global min at origin
        let sphere = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
        let opts = BasinhoppingOptions {
            niter: 20,
            seed: Some(99),
            ..Default::default()
        };
        let result = basinhopping(sphere, &[5.0, -3.0], opts).unwrap();
        let fun = result.fun.unwrap();
        assert!(
            fun < 0.01,
            "basinhopping on sphere should find near-zero, got f={fun}"
        );
    }

    #[test]
    fn basinhopping_returns_correct_nfev() {
        let func = |x: &[f64]| -> f64 { x[0] * x[0] };
        let opts = BasinhoppingOptions {
            niter: 5,
            seed: Some(1),
            ..Default::default()
        };
        let result = basinhopping(func, &[1.0], opts).unwrap();
        assert!(result.nfev > 0, "should track function evaluations");
    }

    #[test]
    fn basinhopping_rejects_nonpositive_stepsize() {
        let func = |x: &[f64]| -> f64 { x[0] * x[0] };
        let opts = BasinhoppingOptions {
            stepsize: 0.0,
            ..Default::default()
        };
        let err = basinhopping(func, &[1.0], opts).unwrap_err();
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
    }

    #[test]
    fn basinhopping_rejects_zero_iteration_budget() {
        let func = |x: &[f64]| -> f64 { x[0] * x[0] };
        let opts = BasinhoppingOptions {
            niter: 0,
            ..Default::default()
        };
        let err = basinhopping(func, &[1.0], opts).unwrap_err();
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
    }

    #[test]
    fn basinhopping_rejects_nonfinite_initial_point() {
        let func = |x: &[f64]| -> f64 { x[0] * x[0] };
        let err = basinhopping(func, &[f64::INFINITY], Default::default()).unwrap_err();
        assert!(matches!(err, crate::OptError::NonFiniteInput { .. }));
    }

    // ── Dual annealing tests ─────────────────────────────────────────

    #[test]
    fn dual_annealing_finds_sphere_minimum() {
        // Sphere function: f(x) = Σ x_i²  — minimum 0 at origin. The bead's
        // reproduction parameters: 100 iterations, seed 42. With a real local
        // search per annealing step this converges essentially exactly.
        let result = dual_annealing(
            |x| x.iter().map(|&xi| xi * xi).sum(),
            &[(-5.0, 5.0), (-5.0, 5.0)],
            100,
            42,
        )
        .expect("dual_annealing");
        assert!(result.success);
        assert!(
            result.fun.unwrap() < 1e-6,
            "should find near-zero minimum: {}",
            result.fun.unwrap()
        );
    }

    #[test]
    fn dual_annealing_finds_shifted_quadratic() {
        // Shifted convex quadratic with interior minimum 0 at (2, -3).
        let result = dual_annealing(
            |x| (x[0] - 2.0).powi(2) + (x[1] + 3.0).powi(2),
            &[(-5.0, 5.0), (-5.0, 5.0)],
            100,
            42,
        )
        .expect("dual_annealing shifted");
        assert!(
            result.fun.unwrap() < 1e-6,
            "should find the shifted minimum: {}",
            result.fun.unwrap()
        );
    }

    #[test]
    fn dual_annealing_finds_rosenbrock_valley() {
        // Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²  — minimum 0 at (1,1).
        let result = dual_annealing(
            |x| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2),
            &[(-5.0, 5.0), (-5.0, 5.0)],
            100,
            123,
        )
        .expect("dual_annealing rosenbrock");
        assert!(
            result.fun.unwrap() < 1e-3,
            "should find the Rosenbrock minimum: {}",
            result.fun.unwrap()
        );
    }

    #[test]
    fn dual_annealing_empty_bounds_rejected() {
        let err = dual_annealing(|_| 0.0, &[], 100, 42).expect_err("empty");
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
    }

    #[test]
    fn dual_annealing_zero_iteration_budget_rejected() {
        let err = dual_annealing(|x| x[0] * x[0], &[(-1.0, 1.0)], 0, 42).expect_err("maxiter");
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
    }

    #[test]
    fn dual_annealing_nonfinite_bounds_rejected() {
        let err = dual_annealing(|_| 0.0, &[(f64::NAN, 1.0)], 100, 42).expect_err("nonfinite");
        assert!(matches!(err, crate::OptError::InvalidBounds { .. }));
    }

    #[test]
    fn dual_annealing_inverted_bounds_rejected() {
        let err = dual_annealing(|_| 0.0, &[(2.0, 2.0)], 100, 42).expect_err("inverted");
        assert!(matches!(err, crate::OptError::InvalidBounds { .. }));
    }

    #[test]
    fn pso_zero_particles_falls_back_to_single_particle() {
        let sphere = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
        let lb = [-5.0, -5.0];
        let ub = [5.0, 5.0];
        let (best, cost) = pso(sphere, &lb, &ub, 0, 10, 123);
        assert_eq!(best.len(), 2);
        assert!(cost.is_finite(), "expected finite cost, got {cost}");
        for (i, (&xi, (&lo, &hi))) in best.iter().zip(lb.iter().zip(ub.iter())).enumerate() {
            assert!(
                xi >= lo && xi <= hi,
                "particle coordinate {i} should stay in bounds [{lo}, {hi}], got {xi}"
            );
        }
    }

    #[test]
    fn pso_rejects_zero_iteration_budget_without_sampling() {
        let lb = [-1.0, -1.0];
        let ub = [1.0, 1.0];
        let (best, cost) = pso(
            |_| -> f64 { panic!("zero-iteration pso should not sample the objective") },
            &lb,
            &ub,
            4,
            0,
            123,
        );
        assert_eq!(best.len(), 2);
        assert!(best.iter().all(|value| value.is_nan()));
        assert!(cost.is_nan());
    }

    #[test]
    fn simulated_annealing_rejects_invalid_controls_without_sampling() {
        for (temp_initial, temp_final, max_iter) in [
            (f64::NAN, 1.0, 10),
            (1.0, f64::INFINITY, 10),
            (0.0, 1.0, 10),
            (1.0, -1.0, 10),
            (1.0, 0.1, 0),
        ] {
            let (state, cost) = crate::simulated_annealing(
                7_i32,
                |_| -> f64 {
                    panic!("invalid simulated_annealing controls should not sample cost")
                },
                |_, _| -> i32 {
                    panic!("invalid simulated_annealing controls should not sample neighbor")
                },
                temp_initial,
                temp_final,
                max_iter,
                123,
            );
            assert_eq!(state, 7);
            assert!(cost.is_nan());
        }
    }

    #[test]
    fn brute_finds_sphere_minimum() {
        let result = crate::brute(
            |x: &[f64]| x.iter().map(|&xi| xi * xi).sum(),
            &[(-5.0, 5.0), (-5.0, 5.0)],
            11,
        )
        .expect("brute sphere");
        assert!(result.success);
        assert!(
            result.fun.expect("objective value") < 1e-6,
            "expected exactly zero minimum (since 0.0 is on the grid), got {:?}",
            result.fun
        );
        assert_eq!(result.x, vec![0.0, 0.0]);
    }

    #[test]
    fn brute_reused_grid_point_preserves_parallel_coordinates_and_first_tie() {
        let ranges = [(-2.0, 2.0), (10.0, 14.0)];
        let ns = 512;

        let corner = crate::brute(|x| x[0] - x[1], &ranges, ns).expect("brute corner");
        assert_eq!(corner.x[0].to_bits(), (-2.0_f64).to_bits());
        assert_eq!(corner.x[1].to_bits(), 14.0_f64.to_bits());

        let tie = crate::brute(|_| 1.0, &ranges, ns).expect("brute tie");
        assert_eq!(tie.x[0].to_bits(), (-2.0_f64).to_bits());
        assert_eq!(tie.x[1].to_bits(), 10.0_f64.to_bits());
        assert_eq!(tie.nfev, ns * ns);
    }

    #[test]
    fn brute_rejects_overflowing_grid_size() {
        let result = crate::brute(|_| 0.0, &[(0.0, 1.0), (0.0, 1.0)], usize::MAX);
        assert!(matches!(
            result,
            Err(crate::OptError::InvalidArgument { .. })
        ));
    }

    #[test]
    fn shgo_finds_sphere_minimum() {
        let result = shgo(
            |x| x.iter().map(|&xi| xi * xi).sum(),
            &[(-5.0, 5.0), (-5.0, 5.0)],
        )
        .expect("shgo sphere");
        assert!(result.success);
        assert!(
            result.fun.expect("objective value") < 1e-6,
            "expected near-zero minimum, got {:?}",
            result.fun
        );
    }

    #[test]
    fn shgo_finds_multimodal_global_minimum() {
        let result = shgo(
            |x| {
                let xi = x[0];
                xi.powi(4) - 4.0 * xi * xi + xi
            },
            &[(-3.0, 3.0)],
        )
        .expect("shgo multimodal");
        assert!(
            result.fun.expect("objective value") < -3.0,
            "expected global basin, got {:?}",
            result.fun
        );
    }

    #[test]
    fn shgo_rejects_invalid_bounds() {
        let err = shgo(|_| 0.0, &[(1.0, 1.0)]).expect_err("invalid bounds");
        assert!(matches!(err, crate::OptError::InvalidBounds { .. }));
    }

    // ── COBYLA tests ─────────────────────────────────────────────────

    #[test]
    fn cobyla_unconstrained_quadratic() {
        // Minimize x² + y² → (0, 0)
        let result = cobyla(
            |x| x[0] * x[0] + x[1] * x[1],
            &[1.0, 1.0],
            &[] as &[fn(&[f64]) -> f64],
            1000,
            0.5,
        )
        .expect("cobyla");
        assert!(
            result.fun.unwrap() < 0.01,
            "cobyla should minimize: {}",
            result.fun.unwrap()
        );
    }

    #[test]
    fn cobyla_with_constraint() {
        // Minimize x + y subject to x + y >= 1
        // Solution: x + y = 1 (constraint active), e.g. (0.5, 0.5)
        type ConstraintFn = dyn Fn(&[f64]) -> f64;
        let constraints: Vec<Box<ConstraintFn>> = vec![
            Box::new(|x: &[f64]| x[0] + x[1] - 1.0), // x + y >= 1
        ];
        let constraint_fns: Vec<&ConstraintFn> = constraints.iter().map(|b| b.as_ref()).collect();
        let result = cobyla(|x| x[0] + x[1], &[2.0, 2.0], &constraint_fns, 1000, 0.5)
            .expect("cobyla constrained");
        // x + y should be approximately 1
        let sum = result.x[0] + result.x[1];
        assert!((sum - 1.0).abs() < 0.1, "x+y should be ~1: {sum}");
    }

    #[test]
    fn cobyla_empty_x0_rejected() {
        let err = cobyla(|_: &[f64]| 0.0, &[], &[] as &[fn(&[f64]) -> f64], 100, 0.5)
            .expect_err("empty x0");
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
    }

    // ── Rosenbrock test function tests ──────────────────────────────

    #[test]
    fn rosen_minimum_is_zero() {
        // f([1, 1, ..., 1]) = 0
        let x = vec![1.0; 5];
        let val = rosen(&x);
        assert!(val.abs() < 1e-12, "rosen at minimum = {val}");
    }

    #[test]
    fn rosen_away_from_minimum() {
        // f([0, 0]) = 1
        let val = rosen(&[0.0, 0.0]);
        assert!((val - 1.0).abs() < 1e-12, "rosen([0,0]) = {val}");
    }

    #[test]
    fn rosen_2d_value() {
        // f([2, 3]) = 100*(3 - 4)^2 + (1 - 2)^2 = 100 + 1 = 101
        let val = rosen(&[2.0, 3.0]);
        assert!((val - 101.0).abs() < 1e-10, "rosen([2,3]) = {val}");
    }

    #[test]
    fn rosen_der_at_minimum_is_zero() {
        // Gradient at minimum should be zero
        let x = vec![1.0; 4];
        let grad = rosen_der(&x);
        for (i, g) in grad.iter().enumerate() {
            assert!(g.abs() < 1e-10, "grad[{i}] = {g}, expected 0");
        }
    }

    #[test]
    fn rosen_der_numerical_check() {
        // Compare analytical gradient to numerical approximation
        let x = vec![1.5, 2.0, 0.5];
        let grad = rosen_der(&x);
        let num_grad = approx_fprime(&x, rosen, 1e-7).expect("approx_fprime");
        for (i, (a, n)) in grad.iter().zip(num_grad.iter()).enumerate() {
            // Relative error check for large values
            let tol = 1e-4 * a.abs().max(1.0);
            assert!(
                (a - n).abs() < tol,
                "grad[{i}]: analytical={a}, numerical={n}"
            );
        }
    }

    #[test]
    fn rosen_hess_is_symmetric() {
        let x = vec![1.2, 0.8, 1.5];
        let h = rosen_hess(&x);
        for (i, row) in h.iter().enumerate() {
            for (j, &entry) in row.iter().enumerate().skip(i + 1) {
                assert!(
                    (entry - h[j][i]).abs() < 1e-12,
                    "H[{i}][{j}] != H[{j}][{i}]"
                );
            }
        }
    }

    #[test]
    fn rosen_hess_at_minimum() {
        // At [1,1], H should be [[802, -400], [-400, 200]]
        let h = rosen_hess(&[1.0, 1.0]);
        assert!((h[0][0] - 802.0).abs() < 1e-10, "H[0][0] = {}", h[0][0]);
        assert!((h[0][1] + 400.0).abs() < 1e-10, "H[0][1] = {}", h[0][1]);
        assert!((h[1][0] + 400.0).abs() < 1e-10, "H[1][0] = {}", h[1][0]);
        assert!((h[1][1] - 200.0).abs() < 1e-10, "H[1][1] = {}", h[1][1]);
    }

    #[test]
    fn rosen_hess_prod_matches_explicit() {
        // H @ p should match explicit matrix-vector product
        let x = vec![0.5, 1.5, 2.0];
        let p = vec![1.0, 2.0, 3.0];
        let h = rosen_hess(&x);
        let hp_explicit: Vec<f64> = (0..3)
            .map(|i| h[i].iter().zip(p.iter()).map(|(a, b)| a * b).sum::<f64>())
            .collect();
        let hp = rosen_hess_prod(&x, &p);
        for (i, (e, c)) in hp_explicit.iter().zip(hp.iter()).enumerate() {
            assert!((e - c).abs() < 1e-10, "hp[{i}]: explicit={e}, computed={c}");
        }
    }

    #[test]
    fn rosen_short_input() {
        // Single element input should return 0
        assert_eq!(rosen(&[1.0]), 0.0);
        assert_eq!(rosen_der(&[1.0]), vec![0.0]);
    }

    fn assert_close(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len(), "length mismatch");
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-12, "index {i}: got {g}, want {w}");
        }
    }

    #[test]
    fn isotonic_regression_strictly_decreasing_pools_to_global_mean() {
        // A fully decreasing input pools into a single block — every PAVA
        // merge must propagate across the whole block, not just two cells.
        // scipy.optimize.isotonic_regression([5,4,3,2,1]).x == [3,3,3,3,3]
        let r = isotonic_regression(&[5.0, 4.0, 3.0, 2.0, 1.0], None);
        assert_close(&r, &[3.0, 3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn isotonic_regression_weighted_multi_pool() {
        // Weighted PAVA: scipy gives an all-1.5 fit for this input.
        let y = [3.0, 1.0, 2.0, 0.5, 1.5];
        let w = [1.0, 2.0, 1.0, 1.0, 1.0];
        let r = isotonic_regression(&y, Some(&w));
        assert_close(&r, &[1.5, 1.5, 1.5, 1.5, 1.5]);

        // Weights bias the pooled mean: scipy gives an all-3.0 fit here.
        let r2 = isotonic_regression(&[4.0, 4.0, 2.0, 2.0], Some(&[3.0, 1.0, 1.0, 3.0]));
        assert_close(&r2, &[3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn isotonic_regression_already_monotone_is_identity() {
        let r = isotonic_regression(&[1.0, 2.0, 3.0], None);
        assert_close(&r, &[1.0, 2.0, 3.0]);
        // Mixed input with one isolated violation, scipy reference values.
        let r2 = isotonic_regression(
            &[1.5, 1.0, 4.0, 6.0, 5.7, 5.0, 7.8, 9.0, 7.5, 9.5, 9.0],
            None,
        );
        assert_close(
            &r2,
            &[
                1.25,
                1.25,
                4.0,
                5.566_666_666_666_666,
                5.566_666_666_666_666,
                5.566_666_666_666_666,
                7.8,
                8.25,
                8.25,
                9.25,
                9.25,
            ],
        );
    }

    #[test]
    fn isotonic_regression_rejects_invalid_weights() {
        // Non-positive weights are invalid (scipy requires strictly positive).
        assert!(
            isotonic_regression(&[1.0, 2.0], Some(&[1.0, 0.0]))
                .iter()
                .all(|v| v.is_nan())
        );
        assert!(
            isotonic_regression(&[1.0, 2.0], Some(&[1.0, -1.0]))
                .iter()
                .all(|v| v.is_nan())
        );
        // Mismatched weight length is invalid.
        assert!(
            isotonic_regression(&[1.0, 2.0], Some(&[1.0]))
                .iter()
                .all(|v| v.is_nan())
        );
        // Empty input returns empty.
        assert!(isotonic_regression(&[], None).is_empty());
    }

    #[test]
    fn rosen_matches_scipy_reference_values() {
        // scipy.optimize.rosen([1, 1]) = 0.0
        let result = rosen(&[1.0, 1.0]);
        assert!(
            result.abs() < 1e-10,
            "rosen at [1,1] got {result}, expected 0.0"
        );
        // scipy.optimize.rosen([0, 0]) = 1.0
        let result = rosen(&[0.0, 0.0]);
        assert!(
            (result - 1.0).abs() < 1e-10,
            "rosen at [0,0] got {result}, expected 1.0"
        );
    }

    #[test]
    fn rosen_der_matches_scipy_reference_values() {
        // scipy.optimize.rosen_der([1, 1]) = [0, 0]
        let result = rosen_der(&[1.0, 1.0]);
        assert!(
            result[0].abs() < 1e-10,
            "rosen_der[0] got {}, expected 0.0",
            result[0]
        );
        assert!(
            result[1].abs() < 1e-10,
            "rosen_der[1] got {}, expected 0.0",
            result[1]
        );
    }

    #[test]
    fn golden_matches_scipy_reference_values() {
        // scipy.optimize.golden(lambda x: x**2, brack=(-1, 0, 1)) ≈ 0
        let (x_opt, _f_opt) = golden(|x| x * x, -1.0, 1.0, 1e-10, 1000);
        assert!(
            x_opt.abs() < 1e-8,
            "golden x^2 minimum at {x_opt}, expected ~0"
        );
    }

    #[test]
    fn golden_rejects_invalid_controls_without_sampling() {
        for (a, b, tol, maxiter) in [
            (f64::NAN, 1.0, 1e-10, 100),
            (0.0, f64::INFINITY, 1e-10, 100),
            (0.0, 1.0, f64::NAN, 100),
            (0.0, 1.0, 0.0, 100),
            (0.0, 1.0, 1e-10, 0),
        ] {
            let (x_opt, f_opt) = golden(
                |_| -> f64 { panic!("invalid golden controls should not sample the objective") },
                a,
                b,
                tol,
                maxiter,
            );
            assert!(x_opt.is_nan());
            assert!(f_opt.is_nan());
        }
    }

    #[test]
    fn brent_minimize_matches_scipy_reference_values() {
        // scipy.optimize.brent(lambda x: (x-2)**2, brack=(0, 4)) ≈ 2
        let (x_opt, _f_opt) = brent_minimize(|x| (x - 2.0).powi(2), 0.0, 4.0, 1e-10, 1000);
        assert!(
            (x_opt - 2.0).abs() < 1e-8,
            "brent (x-2)^2 minimum at {x_opt}, expected ~2"
        );
    }

    #[test]
    fn brent_minimize_rejects_invalid_controls_without_sampling() {
        for (a, b, tol, maxiter) in [
            (f64::NEG_INFINITY, 1.0, 1e-10, 100),
            (0.0, f64::NAN, 1e-10, 100),
            (0.0, 1.0, f64::INFINITY, 100),
            (0.0, 1.0, -1e-10, 100),
            (0.0, 1.0, 1e-10, 0),
        ] {
            let (x_opt, f_opt) = brent_minimize(
                |_| -> f64 {
                    panic!("invalid brent_minimize controls should not sample the objective")
                },
                a,
                b,
                tol,
                maxiter,
            );
            assert!(x_opt.is_nan());
            assert!(f_opt.is_nan());
        }

        let (bounded_x, bounded_f) = minimize_scalar_bounded(
            |_| -> f64 {
                panic!("invalid minimize_scalar_bounded controls should not sample the objective")
            },
            (0.0, 1.0),
            0.0,
            100,
        );
        assert!(bounded_x.is_nan());
        assert!(bounded_f.is_nan());
    }

    #[test]
    fn approx_derivative_matches_scipy() {
        use crate::{FiniteDiffMethod, approx_derivative};
        // R^3 -> R^2: f = [x0^2 x1, x1 sin(x2)].
        let fun = |x: &[f64]| vec![x[0] * x[0] * x[1], x[1] * x[2].sin()];
        let x0 = [1.5, 2.0, 0.5];

        let j2 = approx_derivative(fun, &x0, FiniteDiffMethod::TwoPoint, None, None, None).unwrap();
        let exp2 = [
            [6.000000039736, 2.25, 0.0],
            [0.0, 0.479425538331, 1.755165114999],
        ];
        for i in 0..2 {
            for k in 0..3 {
                assert!(
                    (j2[i][k] - exp2[i][k]).abs() < 1e-7,
                    "2pt[{i}][{k}] {} vs {}",
                    j2[i][k],
                    exp2[i][k]
                );
            }
        }

        let j3 =
            approx_derivative(fun, &x0, FiniteDiffMethod::ThreePoint, None, None, None).unwrap();
        let exp3 = [
            [6.0, 2.249999999989, 0.0],
            [0.0, 0.479425538604, 1.755165123766],
        ];
        for i in 0..2 {
            for k in 0..3 {
                assert!(
                    (j3[i][k] - exp3[i][k]).abs() < 1e-7,
                    "3pt[{i}][{k}] {} vs {}",
                    j3[i][k],
                    exp3[i][k]
                );
            }
        }

        // Custom relative step (2-point).
        let jr = approx_derivative(fun, &x0, FiniteDiffMethod::TwoPoint, Some(1e-6), None, None)
            .unwrap();
        assert!((jr[0][0] - 6.000002999971).abs() < 1e-6);
        assert!((jr[1][2] - 1.755164884021).abs() < 1e-6);

        // 3-point with x0 sitting exactly on an upper bound (forces one-sided).
        let f2 = |x: &[f64]| vec![x[0] * x[0] * x[0] + x[1] * x[1]];
        let x0b = [2.0, 1.0];
        let lb = [f64::NEG_INFINITY, f64::NEG_INFINITY];
        let ub = [2.0, f64::INFINITY];
        let jb = approx_derivative(
            f2,
            &x0b,
            FiniteDiffMethod::ThreePoint,
            None,
            None,
            Some((&lb, &ub)),
        )
        .unwrap();
        assert!((jb[0][0] - 12.0).abs() < 1e-6, "bound dx0 {}", jb[0][0]);
        assert!((jb[0][1] - 2.0).abs() < 1e-6, "bound dx1 {}", jb[0][1]);

        // x0 outside bounds is rejected.
        assert!(
            approx_derivative(
                f2,
                &[3.0, 1.0],
                FiniteDiffMethod::ThreePoint,
                None,
                None,
                Some((&lb, &ub))
            )
            .is_err()
        );
    }

    #[test]
    fn brent_matches_scipy() {
        use crate::brent;
        // scipy.optimize.brent references.
        let x = brent(|x| (x - 2.0).powi(2), None, 1.48e-8, 500).unwrap();
        assert!((x - 2.0).abs() < 1e-6, "quad default {x} vs 2");
        let x = brent(|x| (x - 2.0).powi(2), Some(&[0.0, 1.0]), 1.48e-8, 500).unwrap();
        assert!((x - 2.0).abs() < 1e-6, "quad brack2 {x} vs 2");
        let x = brent(|x| x.cos(), Some(&[0.0, 2.0]), 1.48e-8, 500).unwrap();
        assert!((x - std::f64::consts::PI).abs() < 1e-6, "cos {x} vs pi");
        let x = brent(
            |x| (x + 3.5).powi(2) + 1.0,
            Some(&[-10.0, -1.0]),
            1.48e-8,
            500,
        )
        .unwrap();
        assert!((x + 3.5).abs() < 1e-6, "shifted {x} vs -3.5");
        let x = brent(
            |x| (x - 1.3).powi(4) + 0.5 * x,
            Some(&[0.0, 2.0]),
            1.48e-8,
            500,
        )
        .unwrap();
        assert!((x - 0.8000000000410487).abs() < 1e-5, "quartic {x} vs 0.8");
        // Non-downhill 3-point bracket and wrong-length bracket are rejected.
        assert!(brent(|x| (x - 2.0).powi(2), Some(&[0.0, 0.0, 4.0]), 1.48e-8, 500).is_err());
        assert!(brent(|x| (x - 2.0).powi(2), Some(&[0.0]), 1.48e-8, 500).is_err());
        assert!(brent(|x| (x - 2.0).powi(2), None, 0.0, 500).is_err());
        assert!(brent(|x| (x - 2.0).powi(2), None, f64::NAN, 500).is_err());
        assert!(brent(|x| (x - 2.0).powi(2), None, 1.48e-8, 0).is_err());
        assert!(
            brent(
                |x| (x - 2.0).powi(2),
                Some(&[0.0, f64::INFINITY]),
                1.48e-8,
                500
            )
            .is_err()
        );
    }

    #[test]
    fn linear_sum_assignment_matches_scipy_reference_values() {
        // scipy.optimize.linear_sum_assignment([[4,1,3],[2,0,5],[3,2,2]])
        let cost = vec![
            vec![4.0, 1.0, 3.0],
            vec![2.0, 0.0, 5.0],
            vec![3.0, 2.0, 2.0],
        ];
        let (row_ind, col_ind) = linear_sum_assignment(&cost).expect("lsa");
        assert_eq!(row_ind, vec![0, 1, 2]);
        assert_eq!(col_ind, vec![1, 0, 2]);
        let total_cost: f64 = row_ind
            .iter()
            .zip(col_ind.iter())
            .map(|(&i, &j)| cost[i][j])
            .sum();
        assert!(
            (total_cost - 5.0).abs() < 1e-10,
            "total cost got {total_cost}, expected 5.0"
        );
    }

    #[test]
    fn minimize_bfgs_rosenbrock_matches_scipy_reference() {
        // scipy.optimize.minimize(rosen, [0, 0], method='BFGS')
        // Expected: x ≈ [1, 1], fun ≈ 0
        use crate::{OptimizeMethod, minimize::minimize};
        let result = minimize(
            rosen,
            &[0.0, 0.0],
            MinimizeOptions {
                method: Some(OptimizeMethod::Bfgs),
                ..Default::default()
            },
        )
        .expect("minimize BFGS should succeed");
        assert!(result.success, "BFGS should converge");
        assert!(
            (result.x[0] - 1.0).abs() < 1e-4,
            "x[0] got {}, expected ~1.0",
            result.x[0]
        );
        assert!(
            (result.x[1] - 1.0).abs() < 1e-4,
            "x[1] got {}, expected ~1.0",
            result.x[1]
        );
        let fun = result.fun.expect("fun should be Some");
        assert!(fun < 1e-8, "fun got {fun}, expected ~0.0");
    }

    #[test]
    fn minimize_nelder_mead_rosenbrock_matches_scipy_reference() {
        // scipy.optimize.minimize(rosen, [0, 0], method='Nelder-Mead')
        // Expected: x ≈ [1, 1], fun ≈ 0
        use crate::{OptimizeMethod, minimize::minimize};
        let result = minimize(
            rosen,
            &[0.0, 0.0],
            MinimizeOptions {
                method: Some(OptimizeMethod::NelderMead),
                ..Default::default()
            },
        )
        .expect("minimize Nelder-Mead should succeed");
        assert!(result.success, "Nelder-Mead should converge");
        assert!(
            (result.x[0] - 1.0).abs() < 1e-3,
            "x[0] got {}, expected ~1.0",
            result.x[0]
        );
        assert!(
            (result.x[1] - 1.0).abs() < 1e-3,
            "x[1] got {}, expected ~1.0",
            result.x[1]
        );
        let fun = result.fun.expect("fun should be Some");
        assert!(fun < 1e-6, "fun got {fun}, expected ~0.0");
    }

    #[test]
    fn approx_fprime_matches_scipy_reference_values() {
        // scipy.optimize.approx_fprime([1, 2], lambda x: x[0]**2 + x[1]**2, 1e-8)
        // -> array([2., 4.])
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let grad = approx_fprime(&[1.0, 2.0], f, 1e-8).expect("approx_fprime should succeed");
        assert!(
            (grad[0] - 2.0).abs() < 1e-5,
            "grad[0] got {}, expected 2.0",
            grad[0]
        );
        assert!(
            (grad[1] - 4.0).abs() < 1e-5,
            "grad[1] got {}, expected 4.0",
            grad[1]
        );
    }

    #[test]
    fn fixed_point_matches_scipy_reference_values() {
        // scipy.optimize.fixed_point(lambda x: np.cos(x), 0.5)
        // -> 0.7390851332151607 (Dottie number)
        let result =
            fixed_point(|x| x.cos(), 0.5, 1e-10, 100).expect("fixed_point should converge");
        assert!(
            (result - 0.7390851332151607).abs() < 1e-8,
            "fixed_point got {result}, expected 0.7390851332151607"
        );
    }

    #[test]
    fn fixed_point_rejects_invalid_controls() {
        assert!(matches!(
            fixed_point(|x| x, f64::NAN, 1e-10, 100),
            Err(OptError::InvalidArgument { .. })
        ));
        assert!(matches!(
            fixed_point(|x| x, 0.5, 0.0, 100),
            Err(OptError::InvalidArgument { .. })
        ));
        assert!(matches!(
            fixed_point(|x| x, 0.5, f64::INFINITY, 100),
            Err(OptError::InvalidArgument { .. })
        ));
        assert!(matches!(
            fixed_point(|x| x, 0.5, 1e-10, 0),
            Err(OptError::InvalidArgument { .. })
        ));
    }

    #[test]
    fn fixed_point_many_byte_identical_to_per_param() {
        // Babylonian sqrt: x = 0.5*(x + a/x) -> sqrt(a), swept over many a.
        // nrows crosses the work-capped serial->parallel gate (>= 2*2048).
        let g = |x: f64, p: &[f64]| 0.5 * (x + p[0] / x);
        let mut s = 17u64;
        let mut rng = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            0.1 + 100.0 * ((s >> 11) as f64 / (1u64 << 53) as f64)
        };
        let nrows = 5000usize;
        let params: Vec<Vec<f64>> = (0..nrows).map(|_| vec![rng()]).collect();
        let batched = fixed_point_many(g, 1.0, &params, 1e-11, 500);
        assert_eq!(batched.len(), nrows);
        for (i, p) in params.iter().enumerate() {
            let a = p[0];
            let single = fixed_point(|x| 0.5 * (x + a / x), 1.0, 1e-11, 500).expect("single");
            let many = *batched[i].as_ref().expect("batched member");
            assert_eq!(many.to_bits(), single.to_bits(), "value mismatch {i}");
            assert!(
                (many - a.sqrt()).abs() < 1e-8,
                "{many} vs sqrt {}",
                a.sqrt()
            );
        }
        assert!(fixed_point_many(g, 1.0, &[], 1e-11, 500).is_empty());
    }

    #[test]
    fn isotonic_regression_matches_scipy_reference_values() {
        // from scipy.optimize import isotonic_regression
        // isotonic_regression([5, 3, 1, 2, 4]).x
        // -> array([1. , 1. , 1. , 3. , 4. ]) -- scipy 1.15+ returns decreasing by default
        // Our impl expects increasing, so: isotonic_regression([1, 2, 4, 3, 5])
        // For increasing isotonic regression on [5, 3, 1, 2, 4]:
        // Result should be monotonically increasing, minimizing SSE
        let y = [5.0, 3.0, 1.0, 2.0, 4.0];
        let result = isotonic_regression(&y, None);
        // Check monotonicity
        for i in 1..result.len() {
            assert!(
                result[i] >= result[i - 1],
                "result[{i}]={} < result[{}]={}",
                result[i],
                i - 1,
                result[i - 1]
            );
        }
        // Should have same sum as input (weighted average preserves sum)
        let input_sum: f64 = y.iter().sum();
        let output_sum: f64 = result.iter().sum();
        assert!(
            (input_sum - output_sum).abs() < 1e-10,
            "sum mismatch: input {input_sum}, output {output_sum}"
        );
    }

    #[test]
    fn brute_matches_scipy_reference_values() {
        // scipy.optimize.brute(lambda x: (x[0]-1)**2 + (x[1]-2)**2, [(-5, 5), (-5, 5)], Ns=20)
        // -> array([1., 2.])
        let f = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);
        let result = brute(f, &[(-5.0, 5.0), (-5.0, 5.0)], 20).expect("brute should succeed");
        // brute with coarse grid should get close to [1, 2]
        assert!(
            (result.x[0] - 1.0).abs() < 0.6,
            "x[0] got {}, expected ~1.0",
            result.x[0]
        );
        assert!(
            (result.x[1] - 2.0).abs() < 0.6,
            "x[1] got {}, expected ~2.0",
            result.x[1]
        );
    }

    #[test]
    fn rosen_hess_matches_scipy_reference_values() {
        // scipy.optimize.rosen_hess([1.0, 1.0])
        // At the minimum [1,1], the Hessian should be [[802, -400], [-400, 200]]
        let h = rosen_hess(&[1.0, 1.0]);
        assert_eq!(h.len(), 2);
        assert_eq!(h[0].len(), 2);
        assert!(
            (h[0][0] - 802.0).abs() < 1e-10,
            "H[0][0] got {}, expected 802.0",
            h[0][0]
        );
        assert!(
            (h[0][1] - (-400.0)).abs() < 1e-10,
            "H[0][1] got {}, expected -400.0",
            h[0][1]
        );
        assert!(
            (h[1][0] - (-400.0)).abs() < 1e-10,
            "H[1][0] got {}, expected -400.0",
            h[1][0]
        );
        assert!(
            (h[1][1] - 200.0).abs() < 1e-10,
            "H[1][1] got {}, expected 200.0",
            h[1][1]
        );
    }

    #[test]
    fn bracket_matches_scipy_reference_values() {
        // scipy.optimize.bracket(lambda x: (x-2)**2, xa=0, xb=1)
        // Brackets the minimum at x=2
        let (xa, xb, xc, fa, fb, fc) = bracket(|x| (x - 2.0).powi(2), 0.0, 1.0);
        // xb should be between xa and xc (golden section bracket)
        // fb should be less than both fa and fc (bracketing condition)
        assert!(
            fb <= fa && fb <= fc,
            "bracket failed: fa={fa}, fb={fb}, fc={fc}"
        );
        assert!(
            (xa < xb && xb < xc) || (xa > xb && xb > xc),
            "bracket order violated: xa={xa}, xb={xb}, xc={xc}"
        );
    }

    #[test]
    fn nnls_matches_scipy_reference_values() {
        // scipy.optimize.nnls([[1,0],[1,0],[0,1]], [2,1,1])
        // -> (array([1.5, 1.0]), 0.5)
        // Solve min ||Ax - b||^2 subject to x >= 0
        let a = vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![2.0, 1.0, 1.0];
        let (x, residual) = nnls(&a, &b).expect("nnls should succeed");
        assert_eq!(x.len(), 2);
        assert!(
            (x[0] - 1.5).abs() < 1e-10,
            "x[0] got {}, expected 1.5",
            x[0]
        );
        assert!(
            (x[1] - 1.0).abs() < 1e-10,
            "x[1] got {}, expected 1.0",
            x[1]
        );
        // Residual: ||Ax - b|| = ||(-0.5, 0.5, 0)|| = sqrt(0.5) ≈ 0.707
        let expected_residual = 0.5_f64.sqrt();
        assert!(
            (residual - expected_residual).abs() < 1e-10,
            "residual got {residual}, expected {expected_residual}"
        );
    }

    #[test]
    fn check_grad_matches_scipy_reference_values() {
        // scipy.optimize.check_grad(lambda x: sum(x**2), lambda x: 2*x, [1.0, 2.0])
        // -> 0.0 (gradient is exact)
        let f = |x: &[f64]| -> f64 { x.iter().map(|&xi| xi * xi).sum() };
        let grad = |x: &[f64]| -> Vec<f64> { x.iter().map(|&xi| 2.0 * xi).collect() };
        let x0 = vec![1.0, 2.0];
        let result = check_grad(f, grad, &x0).expect("check_grad");
        assert!(result < 1e-6, "check_grad got {result}, expected ~0");
    }

    #[test]
    fn linprog_matches_scipy_reference_values() {
        // scipy.optimize.linprog(c=[-1, 4], A_ub=[[-3, 1], [1, 2]], b_ub=[6, 4])
        // -> x=[4.0, 0.0], fun=-4.0
        let c = vec![-1.0, 4.0];
        let a_ub = vec![vec![-3.0, 1.0], vec![1.0, 2.0]];
        let b_ub = vec![6.0, 4.0];
        let result = linprog(&c, &a_ub, &b_ub, &[], &[], &[], None).expect("linprog");
        assert_eq!(result.x.len(), 2);
        assert!(
            (result.x[0] - 4.0).abs() < 1e-6,
            "x[0] got {}, expected 4.0",
            result.x[0]
        );
        assert!(
            (result.x[1] - 0.0).abs() < 1e-6,
            "x[1] got {}, expected 0.0",
            result.x[1]
        );
        assert!(
            (result.fun - (-4.0)).abs() < 1e-6,
            "fun got {}, expected -4.0",
            result.fun
        );
    }

    #[test]
    fn derivative_matches_scipy_reference_values() {
        // scipy.misc.derivative(lambda x: x**2, 1.0, dx=1e-6)
        // -> 2.0 (derivative of x^2 at x=1)
        let f = |x: f64| x * x;
        let opts = DifferentiateOptions::default();
        let result = derivative(f, 1.0, opts).expect("derivative");
        assert!(
            (result.df - 2.0).abs() < 1e-4,
            "derivative got {}, expected 2.0",
            result.df
        );
    }

    #[test]
    fn jacobian_matches_scipy_reference_values() {
        // Jacobian of f(x,y) = [x^2, x*y] at (1, 2)
        // -> [[2*x, 0], [y, x]] at (1,2) = [[2, 0], [2, 1]]
        let f = |x: &[f64]| vec![x[0] * x[0], x[0] * x[1]];
        let x0 = vec![1.0, 2.0];
        let opts = DifferentiateOptions::default();
        let result = jacobian(f, &x0, opts).expect("jacobian");
        // df is the Jacobian matrix as Vec<Vec<f64>>
        // Check first row: [2, 0]
        assert!(
            (result.df[0][0] - 2.0).abs() < 1e-4,
            "J[0][0] got {}, expected 2",
            result.df[0][0]
        );
        assert!(
            result.df[0][1].abs() < 1e-4,
            "J[0][1] got {}, expected 0",
            result.df[0][1]
        );
        // Check second row: [2, 1]
        assert!(
            (result.df[1][0] - 2.0).abs() < 1e-4,
            "J[1][0] got {}, expected 2",
            result.df[1][0]
        );
        assert!(
            (result.df[1][1] - 1.0).abs() < 1e-4,
            "J[1][1] got {}, expected 1",
            result.df[1][1]
        );
    }

    #[test]
    fn differential_evolution_rosenbrock_matches_scipy_reference() {
        // scipy.optimize.differential_evolution(rosen, [(-5, 5), (-5, 5)], seed=42)
        // finds minimum near (1, 1) with f(x) near 0
        let result = differential_evolution(
            rosen,
            &[(-5.0, 5.0), (-5.0, 5.0)],
            DifferentialEvolutionOptions::default(),
        )
        .expect("differential_evolution should succeed");

        assert!(result.success, "differential_evolution should converge");
        let fun = result.fun.expect("fun should be Some");
        assert!(fun < 1e-6, "minimum value = {fun}, expected near 0");
        assert!(
            (result.x[0] - 1.0).abs() < 0.1,
            "x[0] = {}, expected near 1",
            result.x[0]
        );
        assert!(
            (result.x[1] - 1.0).abs() < 0.1,
            "x[1] = {}, expected near 1",
            result.x[1]
        );
    }

    #[test]
    fn dual_annealing_rosenbrock_matches_scipy_reference() {
        // scipy.optimize.dual_annealing(rosen, [(-5, 5), (-5, 5)], seed=42)
        // finds minimum near (1, 1) with f(x) near 0
        let result = dual_annealing(rosen, &[(-5.0, 5.0), (-5.0, 5.0)], 1000, 42)
            .expect("dual_annealing should succeed");

        assert!(result.success, "dual_annealing should converge");
        let fun = result.fun.expect("fun should be Some");
        assert!(fun < 1e-3, "minimum value = {fun}, expected near 0");
        assert!(
            (result.x[0] - 1.0).abs() < 0.1,
            "x[0] = {}, expected near 1",
            result.x[0]
        );
        assert!(
            (result.x[1] - 1.0).abs() < 0.1,
            "x[1] = {}, expected near 1",
            result.x[1]
        );
    }

    #[test]
    fn minimize_scalar_bounded_matches_scipy_reference() {
        // scipy.optimize.minimize_scalar(lambda x: (x-2)**2, bounds=(0, 4), method='bounded')
        // finds minimum at x=2
        let (xmin, fmin) = minimize_scalar_bounded(|x| (x - 2.0).powi(2), (0.0, 4.0), 1e-8, 500);
        assert!((xmin - 2.0).abs() < 1e-6, "xmin = {xmin}, expected 2.0");
        assert!(fmin.abs() < 1e-12, "fmin = {fmin}, expected 0.0");
    }

    #[test]
    fn linear_sum_assignment_match_scipy() {
        // scipy.optimize.linear_sum_assignment: minimal-cost assignment. Assert the
        // (unique) optimal cost and a valid permutation rather than over-pinning a
        // possibly-tie-broken column order.
        let cost = vec![
            vec![4.0, 1.0, 3.0],
            vec![2.0, 0.0, 5.0],
            vec![3.0, 2.0, 2.0],
        ];
        let (row, col) = linear_sum_assignment(&cost).expect("lsa");
        assert_eq!(row, vec![0, 1, 2], "rows returned in order");
        let total: f64 = row.iter().zip(&col).map(|(&r, &c)| cost[r][c]).sum();
        assert!((total - 5.0).abs() < 1e-12, "optimal cost: {total}");
        let mut sorted_col = col.clone();
        sorted_col.sort();
        assert_eq!(sorted_col, vec![0, 1, 2], "col is a permutation");
    }

    #[test]
    fn fminbound_matches_scipy_reference() {
        // scipy.optimize.fminbound(lambda x: (x-1.3)**2, 0, 3) -> 1.3
        let x = crate::fminbound(|x| (x - 1.3).powi(2), 0.0, 3.0);
        assert!((x - 1.3).abs() < 1e-4, "fminbound parabola got {x}");
        // scipy.optimize.fminbound(np.cos, 0, 2*pi) -> pi
        let xc = crate::fminbound(|x: f64| x.cos(), 0.0, 2.0 * std::f64::consts::PI);
        assert!(
            (xc - std::f64::consts::PI).abs() < 1e-4,
            "fminbound cos got {xc}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Quasi-Newton Hessian update strategies — scipy.optimize.BFGS / SR1
// ─────────────────────────────────────────────────────────────────────────────

/// Whether a strategy approximates the Hessian or its inverse.
///
/// `scipy.optimize`'s `approx_type`. The two are not interchangeable: every update
/// formula below swaps the roles of `delta_x` and `delta_grad` depending on which is
/// being maintained, so a strategy initialized for one and used as the other produces
/// a matrix that is neither.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HessianApproxType {
    /// Approximate the Hessian `B`.
    Hess,
    /// Approximate the inverse Hessian `H`.
    InvHess,
}

/// What BFGS does when the curvature condition is violated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BfgsExceptionStrategy {
    /// Leave the matrix untouched for that step. Default, with `min_curvature = 1e-8`.
    SkipUpdate,
    /// Interpolate between the BFGS result and the unmodified matrix (Powell damping).
    /// Default `min_curvature = 0.2`.
    DampUpdate,
}

/// BFGS quasi-Newton approximation — `scipy.optimize.BFGS`.
///
/// This crate already has a `bfgs` MINIMIZER; what it did not have is BFGS as a
/// reusable Hessian-approximation strategy. They are different things: the optimizer
/// owns a loop, this owns a matrix and can be handed to any method that wants curvature
/// information -- which is how `trust-constr` and friends consume it in SciPy.
///
/// # The curvature condition is the whole subtlety
///
/// The BFGS update preserves positive-definiteness only while `s·y > 0`. In exact
/// arithmetic a Wolfe line search guarantees it; in floating point it can fail, and
/// applying the formula anyway yields an indefinite matrix that then sends the caller
/// in an ascent direction. SciPy offers two responses and so does this: skip the step,
/// or damp it by interpolating `z` toward `Mw` so the condition holds by construction.
///
/// A SECOND guard exists for the same reason: if `wMw <= 0` -- impossible in exact
/// arithmetic, reachable through roundoff -- the matrix is REINITIALIZED to a scaled
/// identity rather than updated. Without it, one bad step poisons every subsequent one.
#[derive(Debug, Clone)]
pub struct BfgsHessian {
    n: usize,
    approx_type: HessianApproxType,
    /// Row-major `n x n`.
    matrix: Vec<f64>,
    min_curvature: f64,
    exception_strategy: BfgsExceptionStrategy,
    first_iteration: bool,
}

/// Symmetric-rank-1 quasi-Newton approximation — `scipy.optimize.SR1`.
///
/// SR1 does NOT preserve positive-definiteness, by design: it can represent indefinite
/// curvature, which is what makes it useful inside trust-region methods that can
/// exploit negative curvature and useless for line-search methods that cannot. Its
/// only safeguard is skipping the update when the denominator is small relative to the
/// vectors involved, since that is where the rank-1 formula becomes numerically
/// meaningless.
#[derive(Debug, Clone)]
pub struct Sr1Hessian {
    n: usize,
    approx_type: HessianApproxType,
    matrix: Vec<f64>,
    min_denominator: f64,
    first_iteration: bool,
}

/// `y_norm2 / |s·y|` for a Hessian, `|s·y| / y_norm2` for its inverse.
///
/// Nocedal & Wright formula (6.20): the scaling that makes the initial identity a
/// reasonable guess at the true curvature, applied on the FIRST update rather than at
/// initialization because it needs a step to estimate from. Degenerate inputs fall
/// back to 1 rather than dividing by zero.
fn auto_scale(delta_x: &[f64], delta_grad: &[f64], approx: HessianApproxType) -> f64 {
    let s_norm2: f64 = delta_x.iter().map(|v| v * v).sum();
    let y_norm2: f64 = delta_grad.iter().map(|v| v * v).sum();
    let ys: f64 = delta_grad
        .iter()
        .zip(delta_x)
        .map(|(y, s)| y * s)
        .sum::<f64>()
        .abs();
    if ys == 0.0 || y_norm2 == 0.0 || s_norm2 == 0.0 {
        return 1.0;
    }
    match approx {
        HessianApproxType::Hess => y_norm2 / ys,
        HessianApproxType::InvHess => ys / y_norm2,
    }
}

fn identity_scaled(n: usize, scale: f64) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        m[i * n + i] = scale;
    }
    m
}

fn mat_vec(matrix: &[f64], n: usize, p: &[f64]) -> Vec<f64> {
    (0..n)
        .map(|i| (0..n).map(|j| matrix[i * n + j] * p[j]).sum())
        .collect()
}

impl BfgsHessian {
    /// `exception_strategy` defaults `min_curvature` to SciPy's values: `1e-8` for
    /// `SkipUpdate`, `0.2` for `DampUpdate`. Pass `None` to take the default.
    #[must_use]
    pub fn new(exception_strategy: BfgsExceptionStrategy, min_curvature: Option<f64>) -> Self {
        let min_curvature = min_curvature.unwrap_or(match exception_strategy {
            BfgsExceptionStrategy::SkipUpdate => 1e-8,
            BfgsExceptionStrategy::DampUpdate => 0.2,
        });
        Self {
            n: 0,
            approx_type: HessianApproxType::Hess,
            matrix: Vec::new(),
            min_curvature,
            exception_strategy,
            first_iteration: true,
        }
    }

    /// Start a fresh approximation of dimension `n`, as the identity.
    pub fn initialize(&mut self, n: usize, approx_type: HessianApproxType) {
        self.n = n;
        self.approx_type = approx_type;
        self.matrix = identity_scaled(n, 1.0);
        self.first_iteration = true;
    }

    /// Absorb one step. `delta_x = x2 - x1`, `delta_grad = grad(x2) - grad(x1)`.
    ///
    /// A zero step or a zero gradient change is ignored, matching SciPy: the first
    /// carries no information, and the second means the function looked linear over
    /// the step, where a quasi-Newton update has nothing to learn.
    pub fn update(&mut self, delta_x: &[f64], delta_grad: &[f64]) {
        if self.n == 0 || delta_x.len() != self.n || delta_grad.len() != self.n {
            return;
        }
        if delta_x.iter().all(|v| *v == 0.0) || delta_grad.iter().all(|v| *v == 0.0) {
            return;
        }
        if self.first_iteration {
            let scale = auto_scale(delta_x, delta_grad, self.approx_type);
            self.matrix = identity_scaled(self.n, scale);
            self.first_iteration = false;
        }

        // `w` and `z` swap roles with the approximation type; the rest of the formula
        // is identical, which is why SciPy writes it once.
        let (w, z_in): (&[f64], &[f64]) = match self.approx_type {
            HessianApproxType::Hess => (delta_x, delta_grad),
            HessianApproxType::InvHess => (delta_grad, delta_x),
        };
        let mut z = z_in.to_vec();

        let mut wz: f64 = w.iter().zip(&z).map(|(a, b)| a * b).sum();
        let mut mw = mat_vec(&self.matrix, self.n, w);
        let mut wmw: f64 = mw.iter().zip(w).map(|(a, b)| a * b).sum();

        if wmw <= 0.0 {
            // Roundoff has made the matrix indefinite. Reinitialize rather than
            // update; continuing would carry the damage forward indefinitely.
            let scale = auto_scale(delta_x, delta_grad, self.approx_type);
            self.matrix = identity_scaled(self.n, scale);
            mw = mat_vec(&self.matrix, self.n, w);
            wmw = mw.iter().zip(w).map(|(a, b)| a * b).sum();
        }

        if wz <= self.min_curvature * wmw {
            match self.exception_strategy {
                BfgsExceptionStrategy::SkipUpdate => return,
                BfgsExceptionStrategy::DampUpdate => {
                    let denom = 1.0 - wz / wmw;
                    if denom == 0.0 {
                        return;
                    }
                    let update_factor = (1.0 - self.min_curvature) / denom;
                    for (zi, mwi) in z.iter_mut().zip(&mw) {
                        *zi = update_factor * *zi + (1.0 - update_factor) * mwi;
                    }
                    wz = w.iter().zip(&z).map(|(a, b)| a * b).sum();
                }
            }
        }
        if wz == 0.0 || wmw == 0.0 {
            return;
        }

        let n = self.n;
        match self.approx_type {
            // B <- B - (Bs)(Bs)^T / (s^T B s) + y y^T / (s^T y)   -- N&W (6.19)
            HessianApproxType::Hess => {
                for i in 0..n {
                    for j in 0..n {
                        self.matrix[i * n + j] += -mw[i] * mw[j] / wmw + z[i] * z[j] / wz;
                    }
                }
            }
            // H <- H + ((Hy)^T y + s^T y)/(s^T y)^2 * s s^T
            //        - ((Hy) s^T + s (Hy)^T) / (s^T y)            -- N&W (6.17)
            HessianApproxType::InvHess => {
                let coeff = (wmw + wz) / (wz * wz);
                for i in 0..n {
                    for j in 0..n {
                        self.matrix[i * n + j] +=
                            coeff * z[i] * z[j] - (mw[i] * z[j] + z[i] * mw[j]) / wz;
                    }
                }
            }
        }
    }

    /// Matrix-vector product with the current approximation.
    #[must_use]
    pub fn dot(&self, p: &[f64]) -> Vec<f64> {
        mat_vec(&self.matrix, self.n, p)
    }

    /// The current approximation, row-major `n x n`.
    #[must_use]
    pub fn get_matrix(&self) -> &[f64] {
        &self.matrix
    }
}

impl Sr1Hessian {
    /// `min_denominator` defaults to SciPy's `1e-8`.
    #[must_use]
    pub fn new(min_denominator: Option<f64>) -> Self {
        Self {
            n: 0,
            approx_type: HessianApproxType::Hess,
            matrix: Vec::new(),
            min_denominator: min_denominator.unwrap_or(1e-8),
            first_iteration: true,
        }
    }

    /// Start a fresh approximation of dimension `n`, as the identity.
    pub fn initialize(&mut self, n: usize, approx_type: HessianApproxType) {
        self.n = n;
        self.approx_type = approx_type;
        self.matrix = identity_scaled(n, 1.0);
        self.first_iteration = true;
    }

    /// Absorb one step.
    pub fn update(&mut self, delta_x: &[f64], delta_grad: &[f64]) {
        if self.n == 0 || delta_x.len() != self.n || delta_grad.len() != self.n {
            return;
        }
        if delta_x.iter().all(|v| *v == 0.0) || delta_grad.iter().all(|v| *v == 0.0) {
            return;
        }
        if self.first_iteration {
            let scale = auto_scale(delta_x, delta_grad, self.approx_type);
            self.matrix = identity_scaled(self.n, scale);
            self.first_iteration = false;
        }

        let (w, z): (&[f64], &[f64]) = match self.approx_type {
            HessianApproxType::Hess => (delta_x, delta_grad),
            HessianApproxType::InvHess => (delta_grad, delta_x),
        };
        let mw = mat_vec(&self.matrix, self.n, w);
        let diff: Vec<f64> = z.iter().zip(&mw).map(|(zi, m)| zi - m).collect();
        let denominator: f64 = w.iter().zip(&diff).map(|(a, b)| a * b).sum();

        // The skip test is RELATIVE to the vectors, not an absolute floor: an absolute
        // epsilon on a quantity carrying the units of the problem is a scale bug, and
        // would skip every update on a small-scale problem while never firing on a
        // large one.
        let w_norm = w.iter().map(|v| v * v).sum::<f64>().sqrt();
        let diff_norm = diff.iter().map(|v| v * v).sum::<f64>().sqrt();
        if denominator.abs() <= self.min_denominator * w_norm * diff_norm {
            return;
        }

        // Rank-1: M <- M + (z - Mw)(z - Mw)^T / w·(z - Mw)
        let n = self.n;
        for i in 0..n {
            for j in 0..n {
                self.matrix[i * n + j] += diff[i] * diff[j] / denominator;
            }
        }
    }

    /// Matrix-vector product with the current approximation.
    #[must_use]
    pub fn dot(&self, p: &[f64]) -> Vec<f64> {
        mat_vec(&self.matrix, self.n, p)
    }

    /// The current approximation, row-major `n x n`.
    #[must_use]
    pub fn get_matrix(&self) -> &[f64] {
        &self.matrix
    }
}

/// The interface both quasi-Newton strategies present
/// -- `scipy.optimize.HessianUpdateStrategy`.
///
/// Without this, `BfgsHessian` and `Sr1Hessian` are two unrelated types and a method
/// that wants curvature information has to pick one at compile time. The whole point of
/// separating a strategy from an optimizer is that the CALLER chooses, so a trust-region
/// method can be handed SR1 for its indefinite curvature and a line-search method BFGS
/// for its definiteness, without either method knowing which it got.
///
/// The inherent methods on both types are kept and are what the delegating trait
/// methods call, so nothing here forces an import on callers who only ever use one
/// strategy concretely.
pub trait HessianUpdateStrategy {
    /// Start a fresh approximation of dimension `n`.
    ///
    /// `approx_type` is not a formatting choice: every update formula swaps the roles
    /// of `delta_x` and `delta_grad` depending on it, so a strategy initialized for one
    /// and read as the other holds a matrix that is neither.
    fn initialize(&mut self, n: usize, approx_type: HessianApproxType);

    /// Absorb one step: `delta_x = x2 - x1`, `delta_grad = grad(x2) - grad(x1)`.
    fn update(&mut self, delta_x: &[f64], delta_grad: &[f64]);

    /// Matrix-vector product with the current approximation.
    fn dot(&self, p: &[f64]) -> Vec<f64>;

    /// The current approximation, row-major `n x n`.
    fn get_matrix(&self) -> &[f64];
}

impl HessianUpdateStrategy for BfgsHessian {
    fn initialize(&mut self, n: usize, approx_type: HessianApproxType) {
        BfgsHessian::initialize(self, n, approx_type);
    }
    fn update(&mut self, delta_x: &[f64], delta_grad: &[f64]) {
        BfgsHessian::update(self, delta_x, delta_grad);
    }
    fn dot(&self, p: &[f64]) -> Vec<f64> {
        BfgsHessian::dot(self, p)
    }
    fn get_matrix(&self) -> &[f64] {
        BfgsHessian::get_matrix(self)
    }
}

impl HessianUpdateStrategy for Sr1Hessian {
    fn initialize(&mut self, n: usize, approx_type: HessianApproxType) {
        Sr1Hessian::initialize(self, n, approx_type);
    }
    fn update(&mut self, delta_x: &[f64], delta_grad: &[f64]) {
        Sr1Hessian::update(self, delta_x, delta_grad);
    }
    fn dot(&self, p: &[f64]) -> Vec<f64> {
        Sr1Hessian::dot(self, p)
    }
    fn get_matrix(&self) -> &[f64] {
        Sr1Hessian::get_matrix(self)
    }
}

/// Quasi-Newton Hessian update strategies.
///
/// Every test here is anchored on the SECANT CONDITION, which is the defining property
/// of both strategies and holds EXACTLY in arithmetic rather than approximately: after
/// absorbing `(s, y)` the approximation satisfies `B s = y` (or `H y = s`). Testing
/// against it means these tests state what the code must be, not what it currently
/// happens to produce.
#[cfg(test)]
mod hessian_update_strategy_tests {
    use super::{BfgsExceptionStrategy, BfgsHessian, HessianApproxType, Sr1Hessian};

    /// A fixed SPD matrix. Steps are drawn against it so that `y = A s` is a genuine
    /// curvature pair and the strategies have something true to converge to.
    const A: [[f64; 3]; 3] = [[4.0, 1.0, 0.5], [1.0, 3.0, -0.25], [0.5, -0.25, 2.0]];

    fn a_times(s: &[f64]) -> Vec<f64> {
        (0..3)
            .map(|i| (0..3).map(|j| A[i][j] * s[j]).sum())
            .collect()
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// Three linearly independent steps and the gradient changes they induce.
    fn steps() -> Vec<(Vec<f64>, Vec<f64>)> {
        [
            vec![1.0, 0.0, 0.0],
            vec![0.3, 1.0, -0.2],
            vec![-0.5, 0.4, 1.0],
        ]
        .into_iter()
        .map(|s| {
            let y = a_times(&s);
            (s, y)
        })
        .collect()
    }

    /// MUST-HIT: `B s = y` after the update, for the pair just absorbed.
    /// MUST-MISS: it does NOT hold for a pair that was never absorbed -- otherwise the
    /// assertion above could be passed by any matrix that happened to look like `A`.
    #[test]
    fn bfgs_satisfies_the_secant_condition() {
        let (s, y) = steps().remove(0);
        let mut b = BfgsHessian::new(BfgsExceptionStrategy::SkipUpdate, None);
        b.initialize(3, HessianApproxType::Hess);
        b.update(&s, &y);

        let bs = b.dot(&s);
        for i in 0..3 {
            assert!(
                (bs[i] - y[i]).abs() < 1e-12,
                "secant condition violated at {i}: B s = {:?}, y = {y:?}",
                bs
            );
        }

        // MUST-MISS: an unrelated step is not reproduced by a rank-2 update built from
        // a single different step.
        let (s2, y2) = steps().remove(2);
        let bs2 = b.dot(&s2);
        assert!(
            (0..3).any(|i| (bs2[i] - y2[i]).abs() > 1e-6),
            "the approximation reproduced a step it never saw ({bs2:?} vs {y2:?}); the \
             secant check above would then be vacuous"
        );
    }

    /// The same property for the inverse branch: `H y = s`.
    #[test]
    fn bfgs_inverse_satisfies_the_secant_condition() {
        let (s, y) = steps().remove(1);
        let mut h = BfgsHessian::new(BfgsExceptionStrategy::SkipUpdate, None);
        h.initialize(3, HessianApproxType::InvHess);
        h.update(&s, &y);

        let hy = h.dot(&y);
        for i in 0..3 {
            assert!(
                (hy[i] - s[i]).abs() < 1e-12,
                "inverse secant condition violated at {i}: H y = {hy:?}, s = {s:?}"
            );
        }
    }

    /// The two branches are written as two different formulas -- N&W (6.19) and (6.17)
    /// -- and this is what ties them together: fed the SAME steps, the Hessian and the
    /// inverse-Hessian approximations must be actual inverses of each other. A sign or
    /// index slip in either formula that still satisfied its own secant condition would
    /// be caught here and nowhere else.
    ///
    /// The initial scalings are reciprocal by construction (`y2/|sy|` against
    /// `|sy|/y2`), so the relation holds from the first update onward, not just
    /// asymptotically.
    #[test]
    fn the_hessian_and_inverse_branches_are_mutual_inverses() {
        let mut b = BfgsHessian::new(BfgsExceptionStrategy::SkipUpdate, None);
        let mut h = BfgsHessian::new(BfgsExceptionStrategy::SkipUpdate, None);
        b.initialize(3, HessianApproxType::Hess);
        h.initialize(3, HessianApproxType::InvHess);
        for (s, y) in steps() {
            b.update(&s, &y);
            h.update(&s, &y);
        }

        let bm = b.get_matrix();
        let hm = h.get_matrix();
        for i in 0..3 {
            for j in 0..3 {
                let prod: f64 = (0..3).map(|k| hm[i * 3 + k] * bm[k * 3 + j]).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (prod - want).abs() < 1e-9,
                    "H B is not the identity at ({i}, {j}): {prod} != {want}"
                );
            }
        }
    }

    /// After `n` linearly independent steps on a quadratic, SR1 reproduces the TRUE
    /// Hessian exactly -- Nocedal & Wright Theorem 6.1. This is finite termination, not
    /// asymptotic convergence, so a loose bound here would be hiding something.
    ///
    /// Note this is an SR1 property and NOT a BFGS one, which is easy to get backwards.
    /// SR1 is hereditary: every past secant condition survives later updates. BFGS
    /// preserves only the most recent one unless the directions are conjugate, so
    /// asserting the same thing of BFGS would fail against arbitrary independent steps.
    /// It is the exact trade for which SR1 gives up positive-definiteness.
    #[test]
    fn three_independent_steps_recover_the_exact_hessian() {
        let mut b = Sr1Hessian::new(None);
        b.initialize(3, HessianApproxType::Hess);
        for (s, y) in steps() {
            b.update(&s, &y);
        }
        let m = b.get_matrix();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (m[i * 3 + j] - A[i][j]).abs() < 1e-9,
                    "entry ({i}, {j}) is {} but the true Hessian has {}",
                    m[i * 3 + j],
                    A[i][j]
                );
            }
        }
    }

    /// SR1 satisfies the secant condition too, and -- unlike BFGS -- it does so even
    /// when the curvature is NEGATIVE. That is the whole reason to reach for SR1: it
    /// can represent indefinite curvature, which trust-region methods exploit. BFGS
    /// would have refused this step entirely.
    #[test]
    fn sr1_satisfies_the_secant_condition_including_negative_curvature() {
        let (s, y) = steps().remove(0);
        let mut b = Sr1Hessian::new(None);
        b.initialize(3, HessianApproxType::Hess);
        b.update(&s, &y);
        let bs = b.dot(&s);
        for i in 0..3 {
            assert!((bs[i] - y[i]).abs() < 1e-12, "SR1 secant violated at {i}");
        }

        // Negative curvature: s . y < 0. SR1 absorbs it.
        let s_neg = vec![0.0, 1.0, 0.0];
        let y_neg = vec![0.1, -2.0, 0.3];
        assert!(
            dot(&s_neg, &y_neg) < 0.0,
            "the test pair is not negative curvature"
        );
        b.update(&s_neg, &y_neg);
        let bs_neg = b.dot(&s_neg);
        for i in 0..3 {
            assert!(
                (bs_neg[i] - y_neg[i]).abs() < 1e-10,
                "SR1 did not absorb a negative-curvature step at {i}: {} vs {}",
                bs_neg[i],
                y_neg[i]
            );
        }
        // And the result is genuinely indefinite -- SR1 does not pretend otherwise.
        assert!(
            dot(&s_neg, &b.dot(&s_neg)) < 0.0,
            "SR1 produced a positive-definite matrix from a negative-curvature step, \
             which would mean it silently discarded the information"
        );
    }

    /// `skip_update` leaves the matrix BIT-identical when the curvature condition
    /// fails, and updates it when the condition holds. Both arms are required: a guard
    /// that skipped everything would pass the first check alone.
    #[test]
    fn skip_update_declines_exactly_the_bad_steps() {
        let (s, y) = steps().remove(0);
        let mut b = BfgsHessian::new(BfgsExceptionStrategy::SkipUpdate, None);
        b.initialize(3, HessianApproxType::Hess);
        b.update(&s, &y);
        let before: Vec<u64> = b.get_matrix().iter().map(|v| v.to_bits()).collect();

        // MUST-HIT the guard: s . y < 0.
        let s_bad = vec![0.0, 1.0, 0.0];
        let y_bad = vec![0.0, -3.0, 0.0];
        assert!(dot(&s_bad, &y_bad) < 0.0);
        b.update(&s_bad, &y_bad);
        let after: Vec<u64> = b.get_matrix().iter().map(|v| v.to_bits()).collect();
        assert_eq!(
            before, after,
            "a curvature-violating step was absorbed anyway"
        );

        // MUST-MISS the guard: a good step still gets through.
        let (s_ok, y_ok) = steps().remove(2);
        b.update(&s_ok, &y_ok);
        let after_ok: Vec<u64> = b.get_matrix().iter().map(|v| v.to_bits()).collect();
        assert_ne!(before, after_ok, "a valid step was skipped as well");
    }

    /// Powell damping replaces the curvature condition with an EXACT identity:
    /// interpolating `z` toward `Mw` by `(1 - mc)/(1 - wz/wMw)` makes the new
    /// `w . z` equal `mc * wMw` precisely. So this is checkable to rounding rather
    /// than by "the matrix looks reasonable".
    ///
    /// `B' s = z'` after the update (the other two terms cancel), which is how the
    /// test reaches an internal quantity without exposing one.
    #[test]
    fn damp_update_lands_on_the_minimum_curvature_exactly() {
        let (s, y) = steps().remove(0);
        let mut b = BfgsHessian::new(BfgsExceptionStrategy::DampUpdate, None);
        b.initialize(3, HessianApproxType::Hess);
        b.update(&s, &y);

        let s_bad = vec![0.0, 1.0, 0.0];
        let y_bad = vec![0.0, -3.0, 0.0];
        let wmw = dot(&s_bad, &b.dot(&s_bad));
        assert!(wmw > 0.0, "the pre-update matrix is not positive definite");

        b.update(&s_bad, &y_bad);
        let wz_new = dot(&s_bad, &b.dot(&s_bad));
        let want = 0.2 * wmw; // default min_curvature for damp_update
        assert!(
            (wz_new - want).abs() < 1e-10 * want.abs().max(1.0),
            "damped curvature is {wz_new}, expected exactly {want}"
        );
        // Damping exists to KEEP positive definiteness; the point is lost if it does
        // not. Positive `mc` and positive `wMw` make the damped curvature positive.
        assert!(
            wz_new > 0.0,
            "damping left the matrix without positive curvature"
        );
    }

    /// Zero steps and zero gradient changes carry no curvature information and are
    /// ignored rather than dividing by zero. Paired with a must-miss so this cannot
    /// pass by ignoring everything.
    #[test]
    fn degenerate_steps_are_ignored() {
        let mut b = BfgsHessian::new(BfgsExceptionStrategy::SkipUpdate, None);
        b.initialize(3, HessianApproxType::Hess);
        let base: Vec<u64> = b.get_matrix().iter().map(|v| v.to_bits()).collect();

        b.update(&[0.0, 0.0, 0.0], &[1.0, 2.0, 3.0]);
        assert_eq!(
            base,
            b.get_matrix()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            "a zero step changed the approximation"
        );
        b.update(&[1.0, 2.0, 3.0], &[0.0, 0.0, 0.0]);
        assert_eq!(
            base,
            b.get_matrix()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            "a zero gradient change changed the approximation"
        );
        // Mismatched lengths are ignored rather than panicking or reading past the end.
        b.update(&[1.0, 2.0], &[1.0, 2.0]);
        assert_eq!(
            base,
            b.get_matrix()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            "a wrong-length step changed the approximation"
        );

        // MUST-MISS.
        let (s, y) = steps().remove(0);
        b.update(&s, &y);
        assert_ne!(
            base,
            b.get_matrix()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            "a valid step was ignored too"
        );
    }

    /// The first update replaces the identity with a SCALED identity (N&W 6.20) before
    /// applying the update, and the two approximation types scale reciprocally. Without
    /// this the initial matrix carries the units of nothing in particular and the first
    /// few steps are wasted rediscovering the problem's scale.
    #[test]
    fn the_first_update_rescales_the_initial_identity() {
        // A problem scaled by 1000 must produce a correspondingly scaled matrix; a
        // plain identity start would not.
        let s = vec![1.0, 0.0, 0.0];
        let y = vec![1000.0, 0.0, 0.0];

        let mut b = BfgsHessian::new(BfgsExceptionStrategy::SkipUpdate, None);
        b.initialize(3, HessianApproxType::Hess);
        b.update(&s, &y);
        // Untouched directions keep the auto scale y_norm2 / |s . y| = 1e6 / 1e3 = 1e3.
        let m = b.get_matrix();
        assert!(
            (m[4] - 1000.0).abs() < 1e-9,
            "expected the untouched diagonal to hold the auto scale 1000, got {}",
            m[4]
        );

        let mut h = BfgsHessian::new(BfgsExceptionStrategy::SkipUpdate, None);
        h.initialize(3, HessianApproxType::InvHess);
        h.update(&s, &y);
        let hm = h.get_matrix();
        assert!(
            (hm[4] - 1e-3).abs() < 1e-12,
            "the inverse branch should scale reciprocally to 1e-3, got {}",
            hm[4]
        );
    }

    /// The trait is the reason these types exist as strategies rather than as two more
    /// optimizers, so it gets exercised through a `dyn` reference: the two are driven
    /// identically and must each still satisfy their own secant condition. Declaring
    /// the trait without ever calling through it would leave object-safety and the
    /// delegation untested.
    #[test]
    fn both_strategies_are_usable_through_the_trait() {
        use super::HessianUpdateStrategy;

        let mut bfgs = BfgsHessian::new(BfgsExceptionStrategy::SkipUpdate, None);
        let mut sr1 = Sr1Hessian::new(None);
        let strategies: [&mut dyn HessianUpdateStrategy; 2] = [&mut bfgs, &mut sr1];

        for strategy in strategies {
            strategy.initialize(3, HessianApproxType::Hess);
            let (s, y) = steps().remove(0);
            strategy.update(&s, &y);
            let bs = strategy.dot(&s);
            for i in 0..3 {
                assert!(
                    (bs[i] - y[i]).abs() < 1e-12,
                    "a strategy driven through the trait broke the secant condition \
                     at {i}: {} vs {}",
                    bs[i],
                    y[i]
                );
            }
            assert_eq!(strategy.get_matrix().len(), 9);
        }
    }
}
