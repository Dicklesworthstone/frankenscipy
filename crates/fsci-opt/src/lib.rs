// Index walks over parallel slices by a single counter; an iterator rewrite would
// obscure the arithmetic without changing behaviour.
#![allow(clippy::needless_range_loop)]
#![forbid(unsafe_code)]

pub mod audit;
mod bfgs;
pub mod bracket;
pub mod chandrupatla;
mod cobyla;
pub mod curvefit;
pub mod direct;
mod gsa;
pub mod lbfgs_inv_hess;
mod lbfgsb;
mod linear_lsq;
pub mod linesearch;
mod linprog;
mod lsq_matrix;
pub mod minimize;
pub mod nonlin;
mod npx;
pub mod root;
mod slsqp;
mod trf;
mod trf_lsmr;
pub mod trust_constr;
mod trust_region;
pub mod types;

pub use audit::{SyncSharedAuditLedger, record_fail_closed, sync_audit_ledger};
pub use bracket::{
    BracketOptions, BracketResult, MinimumBracketOptions, MinimumBracketResult,
    MinimumBracketStatus, bracket_minimum, bracket_root,
};
pub use cobyla::{
    CONSTRMAX as COBYLA_NAN_VIOLATION, CobylaInfo, FminCobylaOptions, cobyla, fmin_cobyla,
    minimize_cobyla,
};
pub use direct::{DirectOptions, DirectResult, direct, direct_with_callback};
pub use lbfgs_inv_hess::LbfgsInvHessProduct;

pub use chandrupatla::{
    FindMinimumOptions, FindMinimumResult, FindMinimumStatus, FindRootOptions, FindRootResult,
    FindRootStatus, find_minimum, find_root,
};
pub use curvefit::{
    CurveFitNanPolicy, CurveFitOptions, CurveFitResult, CurveFitSigma, JacobianSparsity,
    LeastSquaresCallback, LeastSquaresIteration, LeastSquaresJacobian, LeastSquaresLoss,
    LeastSquaresMethod, LeastSquaresOptions, LeastSquaresResult, LeastsqResult, curve_fit,
    curve_fit_bounded, curve_fit_bounded_many, curve_fit_many, least_squares,
    least_squares_bounded, least_squares_many, leastsq,
};
pub use linear_lsq::{
    LinearLeastSquaresOptions, LinearLeastSquaresResult, lsq_linear_with_options,
};
pub use linesearch::{
    LineSearchResult, ScipyLineSearchResult, WolfeParams, line_search, line_search_wolfe1,
    line_search_wolfe2,
};
pub use linprog::{LinprogMethod, LinprogOptions, LinprogResult, LinprogSensitivity, linprog};
pub use lsq_matrix::{LeastSquaresMatrix, LeastSquaresOperator};
pub use minimize::{
    MinimizeScalarOptions, MinimizeScalarResult, ObjectiveLandscapeProbe, OptCaspDecision,
    OptCaspProblem, OptPortfolioResult, TRUST_EXACT_CHOLESKY_DISABLE,
    TRUST_EXACT_FLAT_AUGMENTED_DISABLE, TRUST_EXACT_FOLD_SHIFT_DISABLE, bfgs, cg_pr_plus, dogleg,
    get_optimize_traces, lbfgsb, minimize, minimize_many, minimize_scalar, minimize_scalar_many,
    minimize_with_audit, minimize_with_casp, minimize_with_casp_portfolio,
    minimize_with_casp_portfolio_and_audit, nelder_mead, newton_cg, powell,
    probe_objective_landscape, select_minimize_method, trust_exact, trust_krylov, trust_ncg,
};
pub use trf::{LossKind, XScale};
pub use trf_lsmr::{TrustRegionLsmrOptions, TrustRegionSolver};
pub use trust_constr::{
    FactorizationMethod, TrustBounds, TrustConstrMethod, TrustConstrResult, TrustConstraint,
    trust_constr_full,
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
    AndersonJacobian, BroydenFirst, BroydenJacobian, BroydenSecond, BroydenVariant,
    DiagBroydenJacobian, ExcitingMixingJacobian, InnerMethod, InverseJacobian, Jacobian,
    KrylovJacobian, LineSearch, LinearMixingJacobian, LowRankMatrix, NonlinJacobian, NonlinMethod,
    NonlinOptions, NonlinResult, ReductionMethod, anderson as anderson_nonlin, broyden1_lowrank,
    broyden2_lowrank, diag_broyden, exciting_mixing, linear_mixing, nonlin_solve, root_nonlin,
};
pub use root::{
    MultivariateRootMethod, MultivariateRootOptions, MultivariateRootResult, NewtonKrylovOptions,
    RootResult, RootResults, anderson, bisect, brenth, brentq, brentq_many, broyden1, broyden2,
    df_sane, fsolve, halley, lm_root, newton, newton_krylov, newton_krylov_with_options,
    newton_many, newton_scalar, ridder, root, root_many, root_scalar, secant, secant_many, toms748,
};
pub use types::{
    Bound, Bounds, Constraint, ConstraintFn, ConstraintJacFn, ConstraintOrigin, ConstraintType,
    ConvergenceStatus, GradientFunc, HessFunc, HessInv, HesspFunc, LinearConstraint,
    MinimizeMethodOptions, MinimizeOptions, NewConstraint, NonlinearConstraint, OptError,
    OptimizeMethod, OptimizeResult, RootMethod, RootOptions,
};

// SciPy's `OptimizeWarning` is `WarningCategory::OptimizeWarning` (raised by `curve_fit` when
// the covariance cannot be estimated), and its `NoConvergence` exception is
// `OptError::NoConvergence`.
pub use fsci_runtime::{Warning, WarningCategory, catch_warnings};

/// Verbose display callback for linear programming, matching `scipy.optimize.linprog_verbose_callback`.
pub fn linprog_verbose_callback(res: &OptimizeResult) {
    let nit = res.nit;
    let fun = res.fun.unwrap_or(f64::NAN);
    eprintln!("linprog iteration {nit}: objective = {fun:.6e}");
}

// `scipy.optimize.show_options` (an interactive documentation printer) is not applicable; the
// option structs' rustdoc is the reference. A template-string stand-in used to be counted as
// covered (frankenscipy-8dndw.1).

/// Exit status for adaptive numerical differentiation, `scipy.differentiate`'s `status`:
/// `Converged` 0, `ErrorIncreased` −1, `MaxIterations` −2, `NonFiniteValue` −3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DifferentiateStatus {
    Converged,
    ErrorIncreased,
    MaxIterations,
    /// A non-finite `x` or derivative estimate (a non-finite function value in the stencil);
    /// `df` is NaN.
    NonFiniteValue,
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

    // Use the displacement that is actually representable, as SciPy does.
    // A fixed epsilon can round away entirely for a large coordinate.
    let points = xk
        .iter()
        .enumerate()
        .map(|(index, &x)| {
            let mut step = epsilon;
            if x + step == x {
                let sign = if x >= 0.0 { 1.0 } else { -1.0 };
                step = f64::EPSILON.sqrt() * sign * x.abs().max(1.0);
            }
            let mut point = x + step;
            if !point.is_finite() {
                point = x - step;
            }
            let dx = point - x;
            if !point.is_finite() || !dx.is_finite() || dx == 0.0 {
                return Err(OptError::InvalidArgument {
                    detail: format!("no finite difference step at index {index}"),
                });
            }
            Ok((point, dx))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let f0 = f(xk);
    if !f0.is_finite() {
        return Err(OptError::NonFiniteInput {
            detail: String::from("objective returned non-finite value at xk"),
        });
    }

    // Every component owns its perturbation buffer. Collect in coordinate order
    // so both serial and parallel execution report the lowest-index error.
    let n = xk.len();
    let component = |index: usize| -> Result<f64, OptError> {
        let mut xp = xk.to_vec();
        let (point, dx) = points[index];
        xp[index] = point;
        let f_plus = f(&xp);
        if !f_plus.is_finite() {
            return Err(OptError::NonFiniteInput {
                detail: format!("objective returned non-finite value at perturbed index {index}"),
            });
        }
        let slope = finite_difference_slope(f_plus, f0, dx);
        if !slope.is_finite() {
            return Err(OptError::NonFiniteInput {
                detail: format!("finite-difference gradient is not representable at index {index}"),
            });
        }
        Ok(slope)
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
    let results: Vec<Result<f64, OptError>> = if nthreads <= 1 {
        (0..n).map(component).collect()
    } else {
        let chunk = n.div_ceil(nthreads);
        let component = &component;
        let parts: Vec<Vec<Result<f64, OptError>>> = std::thread::scope(|scope| {
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
        let mut results = Vec::with_capacity(n);
        for part in parts {
            results.extend(part);
        }
        results
    };
    results.into_iter().collect()
}

/// Divide a difference of finite samples without first requiring their
/// difference to fit in f64. The usual arithmetic is retained when it is finite.
fn finite_difference_slope(next: f64, base: f64, dx: f64) -> f64 {
    let difference = next - base;
    if difference.is_finite() {
        difference / dx
    } else {
        (0.5 * next - 0.5 * base) / (0.5 * dx)
    }
}

/// Finite-difference scheme used by [`approx_derivative`] and
/// [`LeastSquaresOptions::diff_method`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FiniteDiffMethod {
    /// Forward difference `(f(x+h) - f(x)) / h` (scipy `'2-point'`).
    TwoPoint,
    /// Central difference `(f(x+h) - f(x-h)) / 2h`, switching to a second-order
    /// one-sided stencil near bounds (SciPy `'3-point'`).
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
/// Callback outputs must keep their length and contain only finite values.
/// Invalid controls, an unrepresentable derivative, or bounds too narrow for
/// distinct stencil points return a typed error. Perturbations remain finite
/// even at the limits of the floating-point range.
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
    if rel_step.is_some_and(|step| !step.is_finite())
        || abs_step.is_some_and(|step| !step.is_finite())
    {
        return Err(OptError::InvalidArgument {
            detail: "finite-difference steps must be finite".to_string(),
        });
    }

    let mut lb = match bounds {
        Some((lower, _)) => lower.to_vec(),
        None => vec![f64::NEG_INFINITY; n],
    };
    let mut ub = match bounds {
        Some((_, upper)) => upper.to_vec(),
        None => vec![f64::INFINITY; n],
    };
    if lb.len() != n || ub.len() != n {
        return Err(OptError::InvalidArgument {
            detail: "bounds must match x0 length".to_string(),
        });
    }
    if (0..n).any(|i| lb[i].is_nan() || ub[i].is_nan() || lb[i] >= ub[i]) {
        return Err(OptError::InvalidArgument {
            detail: "finite-difference bounds must be ordered and have positive width".to_string(),
        });
    }
    if (0..n).any(|i| x0[i] < lb[i] || x0[i] > ub[i]) {
        return Err(OptError::InvalidArgument {
            detail: "x0 violates bound constraints".to_string(),
        });
    }
    // The representable range is also a bound. This lets the existing scheme
    // adjustment select backward differences at +MAX, and forward ones at -MAX.
    for i in 0..n {
        lb[i] = lb[i].max(-f64::MAX);
        ub[i] = ub[i].min(f64::MAX);
    }

    let rstep = method.rel_eps();
    let mut h = vec![0.0_f64; n];
    for i in 0..n {
        let sign = if x0[i] >= 0.0 { 1.0 } else { -1.0 };
        let default_step = rstep * sign * x0[i].abs().max(1.0);
        let requested = match abs_step {
            Some(step) => step,
            None => match rel_step {
                Some(step) => step * sign * x0[i].abs(),
                None => default_step,
            },
        };
        h[i] = if x0[i] + requested == x0[i] {
            default_step
        } else {
            requested
        };
        if !h[i].is_finite() {
            return Err(OptError::InvalidArgument {
                detail: format!("finite-difference step is not representable at column {i}"),
            });
        }
    }

    let mut use_one_sided = vec![matches!(method, FiniteDiffMethod::TwoPoint); n];
    match method {
        FiniteDiffMethod::TwoPoint => {
            for i in 0..n {
                let lower = x0[i] - lb[i];
                let upper = ub[i] - x0[i];
                let point = x0[i] + h[i];
                let violated = point < lb[i] || point > ub[i];
                let fitting = h[i].abs() <= lower.max(upper);
                if violated && fitting {
                    h[i] = -h[i];
                } else if !fitting {
                    h[i] = if upper >= lower { upper } else { -lower };
                }
            }
        }
        FiniteDiffMethod::ThreePoint => {
            for hi in &mut h {
                *hi = hi.abs();
            }
            use_one_sided.fill(false);
            for i in 0..n {
                let lower = x0[i] - lb[i];
                let upper = ub[i] - x0[i];
                if lower < h[i] || upper < h[i] {
                    h[i] = if upper >= lower {
                        h[i].min(0.5 * upper)
                    } else {
                        -h[i].min(0.5 * lower)
                    };
                    use_one_sided[i] = true;
                    let min_dist = upper.min(lower);
                    if h[i].abs() <= min_dist {
                        h[i] = min_dist;
                        use_one_sided[i] = false;
                    }
                }
            }
        }
    }

    let f0 = fun(x0);
    if f0.iter().any(|value| !value.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: "finite-difference callback returned NaN or infinity at x0".to_string(),
        });
    }
    let m = f0.len();

    // Each column owns its evaluation buffers. Results are collected in index
    // order, preserving deterministic error selection under parallel execution.
    let compute_col = |i: usize| -> Result<Vec<f64>, OptError> {
        let invalid_step = || OptError::InvalidArgument {
            detail: format!("cannot form a distinct finite-difference stencil at column {i}"),
        };
        let valid_point = |point: f64| {
            point.is_finite() && point >= lb[i] && point <= ub[i]
        };
        let evaluate = |x: &[f64]| -> Result<Vec<f64>, OptError> {
            let values = fun(x);
            if values.len() != m {
                return Err(OptError::InvalidArgument {
                    detail: format!(
                        "finite-difference callback changed output length at column {i}: expected {m}, got {}",
                        values.len()
                    ),
                });
            }
            if values.iter().any(|value| !value.is_finite()) {
                return Err(OptError::NonFiniteInput {
                    detail: format!(
                        "finite-difference callback returned NaN or infinity at column {i}"
                    ),
                });
            }
            Ok(values)
        };
        let mut col = vec![0.0_f64; m];
        match method {
            FiniteDiffMethod::TwoPoint => {
                let mut x1 = x0.to_vec();
                x1[i] = x0[i] + h[i];
                let dx = x1[i] - x0[i];
                if !valid_point(x1[i]) || !dx.is_finite() || dx == 0.0 {
                    return Err(invalid_step());
                }
                let f1 = evaluate(&x1)?;
                for k in 0..m {
                    col[k] = finite_difference_slope(f1[k], f0[k], dx);
                }
            }
            FiniteDiffMethod::ThreePoint => {
                let mut x1 = x0.to_vec();
                let mut x2 = x0.to_vec();
                if use_one_sided[i] {
                    x1[i] = x0[i] + h[i];
                    x2[i] = x0[i] + 2.0 * h[i];
                    let dx1 = x1[i] - x0[i];
                    let dx2 = x2[i] - x0[i];
                    let adjacent = x2[i] - x1[i];
                    if !valid_point(x1[i])
                        || !valid_point(x2[i])
                        || [dx1, dx2, adjacent].iter().any(|dx| !dx.is_finite() || *dx == 0.0)
                    {
                        return Err(invalid_step());
                    }
                    let f1 = evaluate(&x1)?;
                    let f2 = evaluate(&x2)?;
                    for k in 0..m {
                        let ordinary = (-3.0 * f0[k] + 4.0 * f1[k] - f2[k]) / dx2;
                        col[k] = if dx1 == 0.5 * dx2
                            && ordinary.is_finite()
                            && (ordinary != 0.0 || (f0[k] == f1[k] && f1[k] == f2[k]))
                        {
                            ordinary
                        } else {
                            // Divided differences remove a large common offset before
                            // weighting samples. Actual node distances also account for
                            // rounding when the first point is not exactly the midpoint.
                            let first = finite_difference_slope(f1[k], f0[k], dx1);
                            let second = finite_difference_slope(f2[k], f1[k], adjacent);
                            let ratio = dx1 / dx2;
                            let difference = first - second;
                            let first_difference = f1[k] - f0[k];
                            let second_difference = f2[k] - f1[k];
                            let numerator = first_difference
                                + 0.5 * (first_difference - second_difference);
                            let before_division = numerator / dx1;
                            if dx1 == 0.5 * dx2 && before_division.is_finite() {
                                // Cancel differences before dividing by a tiny
                                // step; individual slopes may be unrepresentable.
                                before_division
                            } else if difference.is_finite() {
                                first + ratio * difference
                            } else {
                                first + (ratio * first - ratio * second)
                            }
                        };
                    }
                } else {
                    x1[i] = x0[i] - h[i];
                    x2[i] = x0[i] + h[i];
                    let left = x0[i] - x1[i];
                    let right = x2[i] - x0[i];
                    let total = x2[i] - x1[i];
                    if !valid_point(x1[i])
                        || !valid_point(x2[i])
                        || [left, right, total].iter().any(|dx| !dx.is_finite() || *dx == 0.0)
                    {
                        return Err(invalid_step());
                    }
                    let f1 = evaluate(&x1)?;
                    let f2 = evaluate(&x2)?;
                    for k in 0..m {
                        col[k] = if left == right {
                            finite_difference_slope(f2[k], f1[k], total)
                        } else {
                            (right / total) * finite_difference_slope(f0[k], f1[k], left)
                                + (left / total) * finite_difference_slope(f2[k], f0[k], right)
                        };
                    }
                }
            }
        }
        if col.iter().any(|value| !value.is_finite()) {
            return Err(OptError::NonFiniteInput {
                detail: format!("finite-difference Jacobian is not representable at column {i}"),
            });
        }
        Ok(col)
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
    let columns: Vec<Result<Vec<f64>, OptError>> = if nthreads <= 1 {
        (0..n).map(compute_col).collect()
    } else {
        let chunk = n.div_ceil(nthreads);
        let compute_col = &compute_col;
        let parts: Vec<Vec<Result<Vec<f64>, OptError>>> = std::thread::scope(|scope| {
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
        let mut columns = Vec::with_capacity(n);
        for part in parts {
            columns.extend(part);
        }
        columns
    };
    let columns = columns.into_iter().collect::<Result<Vec<_>, _>>()?;

    let mut jacobian = vec![vec![0.0_f64; n]; m];
    for (i, column) in columns.iter().enumerate() {
        for (k, &value) in column.iter().enumerate() {
            jacobian[k][i] = value;
        }
    }
    Ok(jacobian)
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
/// A positive infinite cost forbids that pairing. An error is returned if the
/// smaller side cannot be fully matched using finite costs.
pub fn linear_sum_assignment(
    cost_matrix: &[Vec<f64>],
) -> Result<(Vec<usize>, Vec<usize>), OptError> {
    linear_sum_assignment_with_maximize(cost_matrix, false)
}

/// Solve the rectangular assignment problem with SciPy's `maximize` option.
///
/// Maximization treats negative infinity as a forbidden edge. Minimization
/// treats positive infinity as forbidden. NaN and infinity of the opposite
/// sign are invalid in both square and rectangular problems.
pub fn linear_sum_assignment_with_maximize(
    cost_matrix: &[Vec<f64>],
    maximize: bool,
) -> Result<(Vec<usize>, Vec<usize>), OptError> {
    let col_count = cost_matrix.first().map_or(0, Vec::len);
    let invalid_infinity = if maximize {
        f64::INFINITY
    } else {
        f64::NEG_INFINITY
    };
    for (row_index, row) in cost_matrix.iter().enumerate() {
        if row.len() != col_count {
            return Err(OptError::InvalidArgument {
                detail: format!(
                    "cost_matrix must be rectangular; row 0 has length {col_count} but row {row_index} has length {}",
                    row.len()
                ),
            });
        }
        if row.iter().any(|&x| x.is_nan() || x == invalid_infinity) {
            return Err(OptError::NonFiniteInput {
                detail: format!("cost_matrix contains invalid numeric entries in row {row_index}"),
            });
        }
    }
    if col_count == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    if cost_matrix.len() <= col_count {
        let assignment = shortest_augmenting_path_rectangular(cost_matrix, maximize)?;
        let row_ind = (0..assignment.len()).collect::<Vec<_>>();
        return Ok((row_ind, assignment));
    }

    let transposed = transpose_matrix(cost_matrix);
    let assignment = shortest_augmenting_path_rectangular(&transposed, maximize)?;
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
fn shortest_augmenting_path_rectangular(
    cost_matrix: &[Vec<f64>],
    maximize: bool,
) -> Result<Vec<usize>, OptError> {
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
        if maximize {
            flat.extend(row.iter().map(|&cost| -cost));
        } else {
            flat.extend_from_slice(row);
        }
    }
    let mut u = vec![0.0; row_count];
    let mut v = vec![0.0; col_count];
    let mut col4row = vec![UNASSIGNED; row_count];
    let mut row4col = vec![UNASSIGNED; col_count];
    let mut scratch = ShortestAugmentingPathScratch::new(row_count, col_count);

    for cur_row in 0..row_count {
        let (sink, min_val) = scratch
            .shortest_augmenting_path(&flat, col_count, &u, &v, &row4col, cur_row)
            .ok_or_else(|| OptError::InvalidArgument {
                detail: String::from("cost matrix is infeasible"),
            })?;

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

    Ok(col4row)
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
    ) -> Option<(usize, f64)> {
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

        if !lowest.is_finite() {
            return None;
        }
        min_val = lowest;
        let col = self.remaining[best_remaining_index];
        if row4col[col] == UNASSIGNED {
            return Some((col, min_val));
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

            if !lowest.is_finite() {
                return None;
            }
            min_val = lowest;
            let col = self.remaining[best_remaining_index];
            if row4col[col] == UNASSIGNED {
                return Some((col, min_val));
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
// Mixed-integer linear programming: milp (linprog lives in linprog.rs)
// ══════════════════════════════════════════════════════════════════════

/// Variable integrality kind for `milp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Integrality {
    Continuous,
    Integer,
    Binary,
    /// Either zero or a real value in the supplied bounds (SciPy integrality 2).
    SemiContinuous,
    /// Either zero or an integer in the supplied bounds (SciPy integrality 3).
    SemiInteger,
}

/// Result of a mixed-integer linear programming problem.
#[derive(Debug, Clone)]
pub struct MilpResult {
    /// Best feasible solution, or empty when no incumbent was found.
    pub x: Vec<f64>,
    /// Incumbent objective `c^T x`, or NaN when there is no incumbent.
    pub fun: f64,
    /// Whether the search proved optimality to the requested gap tolerances.
    pub success: bool,
    /// 0 = optimal (within requested gap), 1 = node/iteration limit,
    /// 2 = infeasible, 3 = unbounded, 4 = unresolved numerical failure.
    pub status: u8,
    /// Human-readable message.
    pub message: String,
    /// Number of branch-and-bound nodes processed. An unbounded relaxation
    /// may additionally require a feasibility LP at the same node.
    pub mip_node_count: usize,
    /// Accumulated LP iteration count.
    pub nit: usize,
    /// Global lower bound on the minimum objective, including pending and
    /// unresolved nodes. Negative infinity means no finite bound is available;
    /// positive infinity certifies infeasibility. At an exhausted, solved tree
    /// this equals `fun`, subject to the LP solver's feasibility tolerances.
    pub mip_dual_bound: f64,
    /// `(fun - mip_dual_bound) / abs(fun)`. Zero when the bounds coincide,
    /// infinity without an incumbent or with a nonzero gap at a zero objective.
    pub mip_gap: f64,
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
    /// Stop when the relative incumbent/global-bound gap is at most this
    /// value. Must be finite and nonnegative. The default zero retains the
    /// exhaustive-search behavior; a positive value permits an approximate optimum.
    pub mip_rel_gap: f64,
    /// Absolute objective gap accepted in addition to `mip_rel_gap`. Useful
    /// when the optimum is zero. Must be finite and nonnegative; default zero.
    pub mip_abs_gap: f64,
}

/// A pending subproblem carries the bound already proved by its ancestors.
/// Best-bound ordering keeps that evidence available even before its LP runs.
#[derive(Debug)]
struct MilpNode {
    bounds: Vec<(f64, f64)>,
    lower_bound: f64,
    depth: usize,
    sequence: usize,
}

impl PartialEq for MilpNode {
    fn eq(&self, other: &Self) -> bool {
        self.sequence == other.sequence
    }
}

impl Eq for MilpNode {}

impl PartialOrd for MilpNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MilpNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // BinaryHeap is a max-heap. Lower bounds come first; depth-first ties
        // find an incumbent promptly, with insertion order as a stable tiebreaker.
        other
            .lower_bound
            .total_cmp(&self.lower_bound)
            .then_with(|| self.depth.cmp(&other.depth))
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

fn milp_relative_gap(upper: f64, lower: f64) -> f64 {
    if upper == lower {
        return 0.0;
    }
    if !upper.is_finite() || !lower.is_finite() || upper == 0.0 {
        return f64::INFINITY;
    }
    let difference = upper - lower;
    if difference.is_finite() {
        (difference / upper.abs()).max(0.0)
    } else {
        // Opposite-sign finite endpoints may have an overflowing difference
        // even though the relative gap itself is representable.
        (upper / upper.abs() - lower / upper.abs()).max(0.0)
    }
}

fn milp_result(
    best: Option<LinprogResult>,
    status: u8,
    message: String,
    lower_bound: f64,
    node_count: usize,
    nit: usize,
) -> MilpResult {
    let (x, fun) = best.map_or((Vec::new(), f64::NAN), |best| (best.x, best.fun));
    let lower_bound = if fun.is_finite() {
        lower_bound.min(fun)
    } else {
        lower_bound
    };
    MilpResult {
        x,
        fun,
        // status: exhausted search or a certified global gap met the requested tolerance
        success: status == 0,
        status,
        message,
        mip_node_count: node_count,
        nit,
        mip_dual_bound: lower_bound,
        mip_gap: milp_relative_gap(fun, lower_bound),
    }
}

/// Bound the Lagrangian over a variable box. Any nonpositive inequality
/// multipliers and unrestricted equality multipliers give a valid lower bound;
/// they need not satisfy the LP's reduced-cost tolerances. Outward rounding
/// retains the residual coefficients instead of silently treating them as zero.
/// This matters for infinite bounds and for objectives much smaller than 1e-9.
fn milp_lower_bound(
    problem: MilpProblem<'_>,
    bounds: &[(f64, f64)],
    dual: Option<&LinprogResult>,
) -> f64 {
    let mut reduced: Vec<(f64, f64)> = problem.c.iter().map(|&c| (c, c)).collect();
    let mut value = 0.0;
    if let Some(dual) = dual {
        for (rows, rhs, multipliers, inequality) in [
            (problem.a_ub, problem.b_ub, &dual.ineqlin.marginals, true),
            (problem.a_eq, problem.b_eq, &dual.eqlin.marginals, false),
        ] {
            if multipliers.len() != rows.len() {
                return f64::NEG_INFINITY;
            }
            for ((row, &rhs), &multiplier) in rows.iter().zip(rhs).zip(multipliers) {
                if !multiplier.is_finite() {
                    return f64::NEG_INFINITY;
                }
                let multiplier = if inequality {
                    multiplier.min(0.0)
                } else {
                    multiplier
                };
                if multiplier == 0.0 {
                    continue;
                }
                if rhs != 0.0 {
                    value = (value + (multiplier * rhs).next_down()).next_down();
                }
                for ((low, high), &coefficient) in reduced.iter_mut().zip(row) {
                    if coefficient != 0.0 {
                        let product = coefficient * multiplier;
                        *low = (*low - product.next_up()).next_down();
                        *high = (*high - product.next_down()).next_up();
                    }
                }
            }
        }
    }
    for ((low, high), &(lo, hi)) in reduced.into_iter().zip(bounds) {
        if !low.is_finite()
            || !high.is_finite()
            || (low < 0.0 && hi == f64::INFINITY)
            || (high > 0.0 && lo == f64::NEG_INFINITY)
        {
            return f64::NEG_INFINITY;
        }
        let mut contribution = f64::INFINITY;
        for coefficient in [low, high] {
            if coefficient == 0.0 {
                contribution = contribution.min(0.0);
            } else {
                for endpoint in [lo, hi] {
                    if endpoint.is_finite() {
                        let product = coefficient * endpoint;
                        let lower = if endpoint == 0.0 {
                            0.0
                        } else {
                            product.next_down()
                        };
                        contribution = contribution.min(lower);
                    }
                }
            }
        }
        if contribution != 0.0 {
            value = (value + contribution).next_down();
        }
        if !value.is_finite() {
            return f64::NEG_INFINITY;
        }
    }
    if value.is_finite() {
        value
    } else {
        f64::NEG_INFINITY
    }
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
/// continuous/integer/binary and semi-continuous/semi-integer variables on
/// small to medium problems. A semi-variable may be zero even when its lower
/// bound is positive. Its active interval requires a finite nonnegative lower
/// bound; an infinite upper bound is allowed. An empty active interval leaves
/// only zero, including an interval containing no integer for `SemiInteger`.
///
/// Pending nodes are visited in best-bound order. The result retains a global
/// lower bound and an optimality gap even when a node or LP iteration budget is
/// exhausted. Optional relative/absolute gaps allow an early, quantified stop.
/// An unbounded LP relaxation first triggers an integer-feasibility search:
/// relaxation unboundedness alone never certifies an unbounded integer problem.
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
    linprog::validate_problem(c, a_ub, b_ub, a_eq, b_eq)?;
    if !options.mip_rel_gap.is_finite()
        || options.mip_rel_gap < 0.0
        || !options.mip_abs_gap.is_finite()
        || options.mip_abs_gap < 0.0
    {
        return Err(OptError::InvalidArgument {
            detail: "MILP gap tolerances must be finite and nonnegative".to_string(),
        });
    }
    if options.max_nodes == Some(0) {
        return Err(OptError::InvalidArgument {
            detail: "max_nodes must be greater than zero".to_string(),
        });
    }
    // `linprog` follows SciPy and answers `maxiter = 0` with an iteration-limit result; a
    // branch-and-bound whose every node LP is starved that way could prove nothing.
    if options.lp_maxiter == Some(0) {
        return Err(OptError::InvalidArgument {
            detail: "lp_maxiter must be greater than zero".to_string(),
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
    let mut semi_lower = vec![0.0; n];
    for (i, (bound, kind)) in root_bounds.iter_mut().zip(integrality.iter()).enumerate() {
        if matches!(kind, Integrality::SemiContinuous | Integrality::SemiInteger) {
            // The LP sees the convex hull of {0} union [lo, hi]. Keep the
            // activation threshold separately so a point in (0, lo) is never
            // accepted as a feasible incumbent merely because it solves the LP.
            if !bound.0.is_finite()
                || bound.0 < 0.0
                || bound.1.is_nan()
                || bound.1 == f64::NEG_INFINITY
            {
                return Err(OptError::InvalidBounds {
                    detail: format!(
                        "semi-variable {i}: lower bound must be finite and nonnegative, and upper bound must not be NaN or -inf"
                    ),
                });
            }
            if *kind == Integrality::SemiInteger {
                bound.0 = bound.0.ceil();
                bound.1 = bound.1.floor();
            }
            if bound.0 > bound.1 {
                // There is no active value, but the isolated zero remains in
                // the domain. This is not an infeasible-variable certificate.
                *bound = (0.0, 0.0);
            } else {
                semi_lower[i] = bound.0;
                bound.0 = 0.0;
            }
            continue;
        }
        validate_variable_bounds(i, bound.0, bound.1)?;
        if *kind == Integrality::Binary {
            bound.0 = bound.0.max(0.0);
            bound.1 = bound.1.min(1.0);
        }
        if matches!(kind, Integrality::Integer | Integrality::Binary) {
            bound.0 = bound.0.ceil();
            bound.1 = bound.1.floor();
        }
        if bound.0.is_nan() || bound.1.is_nan() {
            return Err(OptError::InvalidBounds {
                detail: format!("variable {i}: bounds became NaN after integrality adjustment"),
            });
        }
        if bound.0 > bound.1 {
            return Ok(milp_result(
                None,
                2,
                format!("variable {i}: bounds are infeasible after integrality restrictions"),
                f64::INFINITY,
                0,
                0,
            ));
        }
    }

    let max_nodes = options.max_nodes.unwrap_or(2_048);
    let mut node_count = 0usize;
    let mut total_nit = 0usize;
    let mut best: Option<LinprogResult> = None;
    // A failed LP does not discharge its subtree. Retain its inherited bound
    // in the global certificate instead of silently pruning it as infeasible.
    let mut unresolved_bound = f64::INFINITY;
    let mut unresolved_status = 0;
    let mut sequence = 0usize;
    let mut pending = std::collections::BinaryHeap::from([MilpNode {
        lower_bound: milp_lower_bound(problem, &root_bounds, None),
        bounds: root_bounds,
        depth: 0,
        sequence,
    }]);
    let zero_objective = vec![0.0; n];

    loop {
        let lower_bound = pending
            .peek()
            .map_or(f64::INFINITY, |node| node.lower_bound)
            .min(unresolved_bound);
        if let Some(incumbent) = &best {
            let lower_bound = lower_bound.min(incumbent.fun);
            let gap = milp_relative_gap(incumbent.fun, lower_bound);
            if gap <= options.mip_rel_gap || incumbent.fun - lower_bound <= options.mip_abs_gap {
                return Ok(milp_result(
                    best,
                    0,
                    "Optimization terminated successfully within the requested MIP gap".to_string(),
                    lower_bound,
                    node_count,
                    total_nit,
                ));
            }
        }
        if pending.is_empty() {
            let (status, message) = if unresolved_status != 0 {
                (
                    unresolved_status,
                    "MILP search has unresolved LP subproblems".to_string(),
                )
            } else {
                (2, "Problem is infeasible".to_string())
            };
            return Ok(milp_result(
                best,
                status,
                message,
                lower_bound,
                node_count,
                total_nit,
            ));
        }
        if node_count >= max_nodes {
            return Ok(milp_result(
                best,
                1,
                "MILP node limit reached; the incumbent and global bound are retained".to_string(),
                lower_bound,
                node_count,
                total_nit,
            ));
        }
        let node = pending.pop().expect("pending node checked above");
        if best
            .as_ref()
            .is_some_and(|incumbent| node.lower_bound >= incumbent.fun)
        {
            continue;
        }
        let node_bounds = node.bounds;
        let bounds = bounds_to_options(&node_bounds);
        node_count += 1;
        let mut lp = linprog(
            c,
            a_ub,
            b_ub,
            a_eq,
            b_eq,
            &bounds,
            LinprogOptions {
                maxiter: options.lp_maxiter,
                ..LinprogOptions::default()
            },
        )?;
        total_nit += lp.nit;
        let unbounded = lp.status == 3;
        if unbounded && node.lower_bound.is_finite() {
            // A finite independent lower bound contradicts an unbounded LP
            // verdict. Keep the subproblem unresolved instead of turning a
            // numerical inconsistency into a certificate of unboundedness.
            unresolved_bound = unresolved_bound.min(node.lower_bound);
            unresolved_status = 4;
            continue;
        }
        if unbounded {
            // For rational input data, a mixed-integer feasible point plus an
            // unbounded continuous relaxation implies a mixed-integer recession
            // direction (scale the rational ray to integral coordinates). First
            // establish that feasible point, branching when necessary. A point
            // such as x=0.5 on an integer equality must never certify unboundedness.
            lp = linprog(
                &zero_objective,
                a_ub,
                b_ub,
                a_eq,
                b_eq,
                &bounds,
                LinprogOptions {
                    maxiter: options.lp_maxiter,
                    ..LinprogOptions::default()
                },
            )?;
            total_nit += lp.nit;
        }
        if !lp.success {
            if lp.status != 2 || unbounded {
                unresolved_bound = unresolved_bound.min(node.lower_bound);
                unresolved_status = unresolved_status.max(if lp.status == 1 { 1 } else { 4 });
            }
            continue;
        }
        if !lp.fun.is_finite() || lp.x.len() != n || lp.x.iter().any(|x| !x.is_finite()) {
            unresolved_bound = unresolved_bound.min(node.lower_bound);
            unresolved_status = 4;
            continue;
        }
        let lower_bound = if unbounded {
            node.lower_bound
        } else {
            node.lower_bound
                .max(milp_lower_bound(problem, &node_bounds, Some(&lp)))
        };
        if best
            .as_ref()
            .is_some_and(|incumbent| lower_bound >= incumbent.fun)
        {
            continue;
        }

        // A semi-variable has two different branch decisions: isolate zero
        // from its active interval first, then enforce integrality inside the
        // active interval when requested. Both children intersect the current
        // node bounds, so other integer branches cannot resurrect a removed arm.
        let branch = integrality.iter().enumerate().find_map(|(i, kind)| {
            if semi_lower[i] > 0.0
                && node_bounds[i].0 < semi_lower[i]
                && node_bounds[i].1 > 0.0
                && lp.x[i] != 0.0
                && lp.x[i] < semi_lower[i]
            {
                // Only unresolved disjunctions may be split. An LP endpoint
                // rounded just outside an already selected arm must not
                // recreate the same node indefinitely.
                Some((i, 0.0, semi_lower[i]))
            } else if matches!(
                kind,
                Integrality::Integer | Integrality::Binary | Integrality::SemiInteger
            ) && !is_integral_with_tol(lp.x[i], 1e-9)
            {
                Some((i, lp.x[i].floor(), lp.x[i].ceil()))
            } else {
                None
            }
        });

        match branch {
            None if unbounded => {
                return Ok(milp_result(
                    None,
                    3,
                    "Problem is unbounded after establishing mixed-integer feasibility".to_string(),
                    f64::NEG_INFINITY,
                    node_count,
                    total_nit,
                ));
            }
            None => {
                if best.as_ref().is_none_or(|incumbent| lp.fun < incumbent.fun) {
                    best = Some(lp);
                }
            }
            Some((i, left_upper, right_lower)) => {
                let mut lower_branch = node_bounds.clone();
                lower_branch[i].1 = lower_branch[i].1.min(left_upper);
                let mut upper_branch = node_bounds.clone();
                upper_branch[i].0 = upper_branch[i].0.max(right_lower);
                // Retain upper-first depth ties from the previous DFS while
                // allowing a stronger lower bound elsewhere to order the queue.
                for child in [upper_branch, lower_branch] {
                    if child[i].0 <= child[i].1 && child[i] != node_bounds[i] {
                        sequence += 1;
                        pending.push(MilpNode {
                            bounds: child,
                            lower_bound,
                            depth: node.depth + 1,
                            sequence,
                        });
                    } else if child[i] == node_bounds[i] {
                        // A numerical endpoint outside its selected branch must
                        // not recreate the same node or disappear from the proof.
                        unresolved_bound = unresolved_bound.min(lower_bound);
                        unresolved_status = 4;
                    }
                }
            }
        }
    }
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
    /// Relative convergence tolerance on the population's objective values:
    /// `std(fitness) <= atol + tol * abs(mean(fitness))`.
    pub tol: f64,
    /// Absolute convergence tolerance on the same spread (default zero).
    /// Infeasible members or nonfinite objective values prevent convergence.
    pub atol: f64,
    /// Mutation factor (scaling of difference vectors), typically in [0.5, 2].
    pub mutation: f64,
    /// Crossover probability, typically in [0.5, 1.0].
    pub recombination: f64,
    /// Random seed for reproducibility.
    pub seed: Option<u64>,
    /// Integer decision variables. An empty vector means all variables are real;
    /// otherwise it must contain one flag per bound. Both the objective and
    /// constraint callback receive integer values at flagged coordinates.
    /// Bounds are intersected with the integer lattice before optimization.
    /// Integer endpoints must lie in the exactly representable range ±(2^53 - 1).
    pub integrality: Vec<bool>,
}

impl Default for DifferentialEvolutionOptions {
    fn default() -> Self {
        Self {
            strategy: DeStrategy::Best1Bin,
            maxiter: 1000,
            popsize: 15,
            tol: 1e-8,
            atol: 0.0,
            mutation: 0.8,
            recombination: 0.7,
            seed: None,
            integrality: Vec::new(),
        }
    }
}

/// Minimize a function using differential evolution (global optimizer).
///
/// Matches `scipy.optimize.differential_evolution(func, bounds)`.
///
/// A population-based evolutionary optimizer that requires only function evaluations
/// (no gradients). Handles box constraints and mixed integer/real variables.
/// Equal bounds fix a variable exactly. Integer variables retain a continuous
/// search coordinate internally, so rounding an accepted trial does not collapse
/// the mutation population onto the integer lattice.
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
    // Every point is feasible, so the tournament below reduces exactly to `f_trial <= f_orig`.
    differential_evolution_core(|x| (func(x), 0.0), bounds, opts).map(|(result, _)| result)
}

/// SciPy's `_accept_trial` for one scalar violation `v >= 0` (0 = feasible): a feasible trial
/// needs a lower-or-equal objective against a feasible original and always beats an infeasible
/// one; an infeasible trial needs a lower-or-equal violation.
fn de_accepts(trial: (f64, f64), orig: (f64, f64)) -> bool {
    match (trial.1 == 0.0, orig.1 == 0.0) {
        (true, true) => trial.0 <= orig.0,
        (true, false) => true,
        (false, _) => trial.1 <= orig.1,
    }
}

/// Strict "ranks before" for the best member (SciPy's `_promote_lowest_energy`): the lowest
/// objective among feasible members, else the least violation.
fn de_ranks_before(a: (f64, f64), b: (f64, f64)) -> bool {
    match (a.1 == 0.0, b.1 == 0.0) {
        (true, true) => a.0 < b.0,
        (true, false) => true,
        (false, true) => false,
        (false, false) => a.1 < b.1,
    }
}

/// SciPy's population standard-deviation test, scaled before taking moments.
/// Squaring raw energies can underflow for small objectives or overflow even
/// when their standard deviation is representable. Normalizing also keeps an
/// overflowing sum from turning a finite constant objective's mean into infinity.
fn de_population_converged(fitness: &[f64], tol: f64, atol: f64) -> bool {
    if fitness.iter().any(|value| !value.is_finite()) {
        return false;
    }
    let scale = fitness
        .iter()
        .fold(0.0_f64, |scale, value| scale.max(value.abs()));
    if scale == 0.0 {
        return true;
    }
    let count = fitness.len() as f64;
    let mean = fitness.iter().map(|value| value / scale).sum::<f64>() / count;
    let variance = fitness
        .iter()
        .map(|value| (value / scale - mean).powi(2))
        .sum::<f64>()
        / count;
    variance.sqrt() <= atol / scale + tol * mean.abs()
}

/// Differential evolution over `evaluate(x) = (objective, violation)`. Returns the result and
/// the violation of the reported point. The evaluator calls the objective only
/// when the constraint violation is zero; infeasible points have infinite energy.
fn differential_evolution_core<E>(
    evaluate: E,
    bounds: &[(f64, f64)],
    opts: DifferentialEvolutionOptions,
) -> Result<(OptimizeResult, f64), OptError>
where
    E: Fn(&[f64]) -> (f64, f64),
{
    let ndim = bounds.len();
    if ndim == 0 {
        return Err(OptError::InvalidArgument {
            detail: "bounds must not be empty".to_string(),
        });
    }
    if !opts.integrality.is_empty() && opts.integrality.len() != ndim {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "integrality length ({}) must match bounds length ({ndim})",
                opts.integrality.len()
            ),
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
    if !opts.atol.is_finite() || opts.atol < 0.0 {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "atol must be a finite non-negative value, got {}",
                opts.atol
            ),
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
    let mut search_bounds = bounds.to_vec();
    let mut integer_bounds = vec![None; ndim];
    for (i, &(lo, hi)) in bounds.iter().enumerate() {
        if !lo.is_finite() || !hi.is_finite() {
            return Err(OptError::InvalidBounds {
                detail: format!("variable {i}: bounds must be finite, got ({lo}, {hi})"),
            });
        }
        if lo > hi {
            return Err(OptError::InvalidBounds {
                detail: format!("variable {i}: lower bound {lo} > upper bound {hi}"),
            });
        }
        if opts.integrality.get(i).copied().unwrap_or(false) {
            let lower = lo.ceil();
            let upper = hi.floor();
            if lower > upper {
                return Err(OptError::InvalidBounds {
                    detail: format!("variable {i}: bounds ({lo}, {hi}) contain no integer"),
                });
            }
            const MAX_EXACT_INTEGER: f64 = 9_007_199_254_740_991.0;
            if lower < -MAX_EXACT_INTEGER || upper > MAX_EXACT_INTEGER {
                return Err(OptError::InvalidBounds {
                    detail: format!(
                        "variable {i}: integer bounds exceed the exactly representable range"
                    ),
                });
            }
            integer_bounds[i] = Some((lower, upper));
            search_bounds[i] = if lower == upper {
                (lower, upper)
            } else {
                // Give endpoint integers a full rounding cell, like SciPy.
                // Exclude the half-integer endpoints where ties-to-even could
                // round outside the feasible lattice. At large magnitudes a
                // half step can round away; preserve the endpoint itself.
                (
                    (lower - 0.5).next_up().min(lower),
                    (upper + 0.5).next_down().max(upper),
                )
            };
        }
    }
    let mut rng: rand::rngs::StdRng = match opts.seed {
        Some(s) => rand::rngs::StdRng::seed_from_u64(s),
        None => rand::rngs::StdRng::from_rng(&mut rand::rng()),
    };

    // Initialize population uniformly within bounds.
    let mut population: Vec<Vec<f64>> = (0..pop_count)
        .map(|_| {
            search_bounds
                .iter()
                .map(|&(lo, hi)| {
                    if lo == hi {
                        lo
                    } else if (hi - lo).is_finite() {
                        rng.random_range(lo..hi)
                    } else {
                        // Opposite-sign finite bounds can have an overflowing
                        // width even though every convex combination is finite.
                        let weight: f64 = rng.random();
                        (1.0 - weight) * lo + weight * hi
                    }
                })
                .collect()
        })
        .collect();

    let has_integers = integer_bounds.iter().any(Option::is_some);
    let mut evaluation_point = vec![0.0; ndim];
    let mut evaluate_point = |point: &[f64]| {
        if !has_integers {
            return evaluate(point);
        }
        for (j, &value) in point.iter().enumerate() {
            evaluation_point[j] = match integer_bounds[j] {
                Some((lower, upper)) => value.round_ties_even().clamp(lower, upper),
                None => value,
            };
        }
        evaluate(&evaluation_point)
    };
    let (mut fitness, mut violation): (Vec<f64>, Vec<f64>) =
        population.iter().map(|x| evaluate_point(x)).unzip();
    let mut nfev = violation.iter().filter(|&&value| value == 0.0).count();

    // Track best.
    let mut best_idx = 0;
    for i in 1..pop_count {
        if de_ranks_before(
            (fitness[i], violation[i]),
            (fitness[best_idx], violation[best_idx]),
        ) {
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
                mutant[j] = if opts.mutation == 0.0 {
                    base[j]
                } else {
                    base[j] + opts.mutation * (population[r1][j] - population[r2][j])
                };
            }

            // Binomial crossover.
            let j_rand = rng.random_range(0..ndim);
            for j in 0..ndim {
                trial[j] = if j == j_rand || rng.random_range(0.0..1.0) < opts.recombination {
                    // Clip to bounds.
                    mutant[j].clamp(search_bounds[j].0, search_bounds[j].1)
                } else {
                    population[i][j]
                };
            }

            // Selection.
            let scored_trial = evaluate_point(&trial);
            nfev += usize::from(scored_trial.1 == 0.0);
            if de_accepts(scored_trial, (fitness[i], violation[i])) {
                population[i].copy_from_slice(&trial);
                (fitness[i], violation[i]) = scored_trial;
                if de_ranks_before(scored_trial, (fitness[best_idx], violation[best_idx])) {
                    best_idx = i;
                }
            }
        }

        // As SciPy, whose infeasible members carry infinite energy: no spread test passes while
        // any member is infeasible.
        if violation.iter().all(|&v| v == 0.0)
            && de_population_converged(&fitness, opts.tol, opts.atol)
        {
            // status: all members feasible and std(fitness) ≤ atol + tol·|mean(fitness)|
            converged = true;
            break;
        }
    }

    let best_violation = violation[best_idx];
    let mut best_point = population[best_idx].clone();
    for (j, value) in best_point.iter_mut().enumerate() {
        if let Some((lower, upper)) = integer_bounds[j] {
            *value = value.round_ties_even().clamp(lower, upper);
        }
    }
    let result = OptimizeResult {
        x: best_point,
        fun: Some(fitness[best_idx]),
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
    };
    Ok((result, best_violation))
}

/// Differential evolution under a constraint, with SciPy's feasibility tournament.
///
/// `constraint_violation(x)` returns the total violation, `>= 0`, zero meaning feasible (a NaN
/// counts as infeasible). Selection follows scipy.optimize.differential_evolution's constrained
/// mode (Lampinen): a feasible point beats an infeasible one, feasible points compare by
/// objective, infeasible ones by violation; the population converges only once every member is
/// feasible. `maxcv` reports the violation of the returned point, and an infeasible point is
/// never a success ("The solution does not satisfy the constraints, MAXCV = ...").
/// The objective is evaluated only for feasible points, so it may be defined
/// solely on the constraint domain. `nfev` counts actual objective calls; an
/// entirely infeasible population returns an infinite objective and `nfev == 0`.
///
/// This replaced a quadratic exterior penalty (br-7470), which could only discourage
/// infeasibility: it put the optimum of `x^2` under `x >= 1` at `x = 1 - 1e-6` and reported
/// that infeasible point as a success (frankenscipy-szq1n.7).
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
    let scored = |x: &[f64]| {
        let v = constraint_violation(x);
        let violation = if v.is_nan() {
            f64::INFINITY
        } else {
            v.max(0.0)
        };
        let objective = if violation == 0.0 {
            func(x)
        } else {
            f64::INFINITY
        };
        (objective, violation)
    };
    let (mut result, violation) = differential_evolution_core(scored, bounds, opts)?;
    result.maxcv = Some(violation);
    if violation > 0.0 {
        result.success = false;
        result.status = ConvergenceStatus::Infeasible;
        result.message =
            format!("The solution does not satisfy the constraints, MAXCV = {violation:e}");
    }
    Ok(result)
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
    /// Tolerance for the local minimizer (`minimizer_kwargs={'tol': ...}`); `None` is SciPy's
    /// default, the local BFGS's own `gtol` of 1e-5.
    pub minimizer_tol: Option<f64>,
}

impl Default for BasinhoppingOptions {
    fn default() -> Self {
        Self {
            niter: 100,
            temperature: 1.0,
            stepsize: 0.5,
            seed: None,
            minimizer_tol: None,
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

    // SciPy's default `minimizer_kwargs` is `minimize`'s own default for an unconstrained
    // problem: BFGS with its default `maxiter` (200 n).
    let minimize_opts = MinimizeOptions {
        method: Some(OptimizeMethod::Bfgs),
        tol: opts.minimizer_tol,
        ..MinimizeOptions::default()
    };

    // Initial local minimization.
    let local_result =
        crate::bfgs(&func, x0, minimize_opts).map_err(|e| OptError::InvalidArgument {
            detail: format!("initial local minimization failed: {e}"),
        })?;

    // SciPy: `res.success = res.lowest_optimization_result.success` -- the verdict belongs to
    // the local minimisation that produced the reported point (frankenscipy-szq1n.7).
    let mut best_success = local_result.success;
    let mut best_status = local_result.status;
    let mut best_message = local_result.message.clone();
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

            // SciPy's `Storage.update`: only a SUCCESSFUL local minimisation becomes the new
            // lowest (or replaces a stored one that failed). A run that ends in precision loss
            // often stops LOWER than the successful ones — 27-34% of SciPy's own BFGS runs at
            // tol 1e-8 by finite differences do — and must not be reported as the minimum.
            if result.success && (f_new < f_best || !best_success) {
                f_best = f_new;
                x_best.clone_from(&result.x);
                best_success = result.success;
                best_status = result.status;
                best_message.clone_from(&result.message);
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
        success: best_success,
        status: best_status,
        message: if best_success {
            format!("Basin-hopping completed after {} iterations", opts.niter)
        } else {
            format!(
                "Basin-hopping completed after {} iterations; the lowest local minimisation failed: {best_message}",
                opts.niter
            )
        },
        nfev: total_nfev,
        njev: 0,
        nhev: 0,
        nit: total_nit,
        jac: None,
        hess_inv: None,
        maxcv: None,
    })
}

/// Dual annealing global optimization: `scipy.optimize.dual_annealing(func, bounds,
/// maxiter=maxiter, rng=seed)` with SciPy's other defaults.
///
/// SciPy's generalized simulated annealing as SciPy 1.17.1 implements it (see [`crate::gsa`]):
/// Tsallis–Stariolo visits (`visit = 2.62`) of every coordinate and then of one coordinate at a
/// time along a chain of 2·n steps, the generalized Metropolis test (`accept = -5`), the
/// temperature `5230·(2^1.62 − 1)/((k + 2)^1.62 − 1)` with a restart below `5230·2e-5`, and a
/// local search — L-BFGS-B inside the bounds, `maxiter = min(max(6 n, 100), 1000)` — from the
/// best point whenever the chain improved it and from the chain's minimum after 1000 iterations
/// without improvement. The random stream is fsci's own (xorshift), so a seed does not
/// reproduce SciPy's iterates, but the search is SciPy's: on Rastrigin in 2-D it finds the
/// global minimum from 10 of 20 seeds at `maxiter = 80` (SciPy: 9 of 20) and from 20 of 20 at
/// SciPy's default 1000 (SciPy: 20 of 20).
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
    let lower: Vec<f64> = bounds.iter().map(|&(lo, _)| lo).collect();
    let upper: Vec<f64> = bounds.iter().map(|&(_, hi)| hi).collect();

    // SciPy's `LocalSearchWrapper`: L-BFGS-B inside the bounds with
    // `maxiter = min(max(6 n, 100), 1000)`.
    let local_bounds: Vec<Bound> = bounds
        .iter()
        .map(|&(lo, hi)| (Some(lo), Some(hi)))
        .collect();
    let minimize_opts = MinimizeOptions {
        method: Some(OptimizeMethod::LBfgsB),
        maxiter: Some((6 * ndim).clamp(100, 1000)),
        bounds: Some(&local_bounds),
        ..MinimizeOptions::default()
    };
    let mut local = |start: &[f64]| -> Option<(f64, Vec<f64>, usize, usize)> {
        let res = crate::lbfgsb(&func, start, minimize_opts, Some(&local_bounds)).ok()?;
        Some((res.fun.unwrap_or(f64::NAN), res.x, res.nfev, res.njev))
    };

    // SciPy's `EnergyState.reset` redraws a start whose energy is not finite up to 1000 times,
    // then raises; a NaN start would otherwise stick and come back as the answer
    // (frankenscipy-0v9od).
    let outcome = gsa::anneal(&func, &lower, &upper, maxiter, &mut rng, Some(&mut local))
        .map_err(|detail| OptError::NonFiniteInput { detail })?;
    Ok(OptimizeResult {
        x: outcome.x,
        fun: Some(outcome.fun),
        nit: outcome.nit,
        nfev: outcome.nfev,
        njev: outcome.njev,
        nhev: 0,
        // status: SciPy's `success` is false only when maxfun stopped the search
        success: outcome.success,
        status: if outcome.success {
            ConvergenceStatus::Success
        } else {
            ConvergenceStatus::MaxEvaluations
        },
        message: outcome.message,
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

    // Start from the first grid point, which is SciPy's `argmin` when no value is finite; the
    // old `[0.0; ndim]` start could lie outside every range (frankenscipy-szq1n.7).
    let mut best_x = vec![0.0; ndim];
    fill_point(0, &mut best_x);
    let mut best_f = fs[0];
    for (idx, &f) in fs.iter().enumerate() {
        if f < best_f || (best_f.is_nan() && !f.is_nan()) {
            best_f = f;
            fill_point(idx, &mut best_x);
        }
    }
    // No finite objective value anywhere on the grid is not a minimum.
    let success = best_f.is_finite();

    Ok(OptimizeResult {
        x: best_x,
        fun: Some(best_f),
        nit: total_points,
        nfev,
        njev: 0,
        nhev: 0,
        success,
        status: if success {
            ConvergenceStatus::Success
        } else {
            ConvergenceStatus::NanEncountered
        },
        message: if success {
            "brute completed".to_string()
        } else {
            "brute: the objective is not finite anywhere on the grid".to_string()
        },
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

/// Controls for simplicial global optimization.
///
/// Each coordinate is initially divided into `initial_subdivisions` intervals.
/// Subsequent iterations bisect these intervals, retaining previously sampled
/// vertices. The Cartesian grid is triangulated by the Freudenthal construction:
/// vertices belong to a common simplex when they are coordinatewise ordered in
/// the same grid cell. Local solves start at the minima of this neighbor graph,
/// including one representative of each flat minimum component.
///
/// The sample and edge limits are checked before allocating a complex. This
/// method has exponential dimension dependence; an infeasible initial grid is
/// rejected before either the objective or a constraint is called.
#[derive(Debug, Clone, Copy)]
pub struct ShgoOptions<'a> {
    pub initial_subdivisions: usize,
    pub iterations: usize,
    /// Maximum number of unique sampling vertices, including infeasible ones.
    pub max_samples: usize,
    /// Maximum number of edges in any refined simplicial complex.
    pub max_complex_edges: usize,
    /// Hard limit on actual objective calls, including numerical gradients.
    pub max_evaluations: usize,
    pub local_max_iterations: usize,
    /// Hard objective-call limit for each local solve.
    pub local_max_evaluations: usize,
    pub local_tolerance: f64,
    /// Two local solutions within this fraction of every box width are merged.
    pub minimizer_tolerance: f64,
    /// Equalities and inequalities use the same contract as `minimize`.
    /// Inequalities filter the sampling complex. Equalities, which can define
    /// a lower-dimensional set with no grid vertices, are enforced by SLSQP.
    pub constraints: &'a [Constraint<'a>],
    pub constraint_tolerance: f64,
}

impl Default for ShgoOptions<'_> {
    fn default() -> Self {
        Self {
            initial_subdivisions: 4,
            iterations: 1,
            max_samples: 100_000,
            max_complex_edges: 5_000_000,
            max_evaluations: 1_000_000,
            local_max_iterations: 250,
            local_max_evaluations: 2_500,
            local_tolerance: 1e-8,
            minimizer_tolerance: 1e-7,
            constraints: &[],
            constraint_tolerance: 1e-7,
        }
    }
}

/// Global result and the distinct, successfully polished local minima.
///
/// `xl` and `funl` are sorted by objective value. On success, `optimum` is the
/// lowest converged local minimum. A budget or local-solver failure instead
/// retains the best finite feasible point, including unpolished samples.
/// Its iteration count is the number of completed sampling rounds,
/// and `nfev == sampling_nfev + nlfev` counts actual objective calls.
#[derive(Debug, Clone, PartialEq)]
pub struct ShgoResult {
    pub optimum: OptimizeResult,
    pub xl: Vec<Vec<f64>>,
    pub funl: Vec<f64>,
    pub sampling_nfev: usize,
    pub nlfev: usize,
    pub nljev: usize,
    pub nlhev: usize,
    pub samples: usize,
    pub local_minimizations: usize,
    pub failed_local_minimizations: usize,
}

/// Simplicial global optimization over a bounded box with default controls.
///
/// See [`shgo_with_options`] for refinement, constraints, resource limits and
/// the complete local-minimum pool. A finite sampling run is not a certificate
/// that an arbitrary black-box objective has no narrower unsampled basin.
pub fn shgo<F>(func: F, bounds: &[(f64, f64)]) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    shgo_with_options(func, bounds, ShgoOptions::default()).map(|result| result.optimum)
}

/// Minimize using the minima of a refined simplicial complex.
///
/// Unlike a fixed number of value-ranked restarts, every sampled minimum
/// component receives a local solve. Unconstrained problems use L-BFGS-B in
/// the sampled vertex's star, with a full-box polish if an artificial star
/// boundary is reached. Constrained problems use SLSQP over the original
/// bounds so a local star cannot exclude the equality-feasible manifold.
/// Infeasible inequality vertices are retained with infinite value and never
/// passed to the objective. Nonfinite objective values likewise create holes
/// in the complex; nonfinite or shape-changing constraints are errors.
///
/// Success requires completed sampling rounds, a converged feasible best
/// point, and no failed local solves. Limits retain the best feasible point
/// with `success = false`. Neither a finite grid value nor a failed local
/// optimizer is reported as convergence.
pub fn shgo_with_options<F>(
    func: F,
    bounds: &[(f64, f64)],
    options: ShgoOptions<'_>,
) -> Result<ShgoResult, OptError>
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
    if options.initial_subdivisions == 0
        || options.iterations == 0
        || options.max_samples == 0
        || options.max_complex_edges == 0
        || options.max_evaluations == 0
    {
        return Err(OptError::InvalidArgument {
            detail: "shgo: subdivisions, iterations and global resource limits must be positive"
                .into(),
        });
    }
    for (name, tolerance) in [
        ("local_tolerance", options.local_tolerance),
        ("minimizer_tolerance", options.minimizer_tolerance),
        ("constraint_tolerance", options.constraint_tolerance),
    ] {
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(OptError::InvalidArgument {
                detail: format!("shgo: {name} must be positive and finite"),
            });
        }
    }
    shgo_complex_size(ndim, options.initial_subdivisions, options)?;

    let evaluations = std::cell::Cell::new(0usize);
    let evaluation_limit = std::cell::Cell::new(false);
    let evaluate = |point: &[f64]| {
        if evaluations.get() >= options.max_evaluations {
            evaluation_limit.set(true);
            f64::NAN
        } else {
            evaluations.set(evaluations.get() + 1);
            func(point)
        }
    };
    let global_bounds: Vec<Bound> = bounds
        .iter()
        .map(|&(lo, hi)| (Some(lo), Some(hi)))
        .collect();
    let mut cache = std::collections::BTreeMap::<Vec<u64>, usize>::new();
    let mut vertices: Vec<(Vec<f64>, f64)> = Vec::new();
    let mut constraint_sizes = vec![None; options.constraints.len()];
    let mut minima: Vec<(Vec<f64>, f64)> = Vec::new();
    let mut best: Option<(Vec<f64>, f64)> = None;
    let mut sampling_nfev = 0usize;
    let mut nlfev = 0usize;
    let mut nljev = 0usize;
    let mut nlhev = 0usize;
    let mut local_minimizations = 0usize;
    let mut failed_local_minimizations = 0usize;
    let mut local_failure_status = None;
    let mut completed = 0usize;
    let mut stop: Option<(ConvergenceStatus, String)> = None;
    let mut subdivisions = options.initial_subdivisions;

    'rounds: for iteration in 0..options.iterations {
        let count = match shgo_complex_size(ndim, subdivisions, options) {
            Ok(count) => count,
            Err(error) => {
                stop = Some((ConvergenceStatus::MaxEvaluations, error.to_string()));
                break;
            }
        };
        let levels = subdivisions + 1;
        let mut strides = vec![1usize; ndim];
        for dimension in 1..ndim {
            strides[dimension] = strides[dimension - 1] * levels;
        }
        let mut indices = Vec::with_capacity(count);
        let mut values = Vec::with_capacity(count);
        for grid_index in 0..count {
            let point: Vec<f64> = bounds
                .iter()
                .zip(&strides)
                .map(|(&(lo, hi), &stride)| {
                    shgo_grid_coordinate(lo, hi, (grid_index / stride) % levels, subdivisions)
                })
                .collect();
            let key: Vec<u64> = point.iter().map(|value| value.to_bits()).collect();
            let vertex = if let Some(&vertex) = cache.get(&key) {
                vertex
            } else {
                let (inequality_violation, violation) =
                    shgo_constraint_state(&point, options.constraints, &mut constraint_sizes)?;
                let value = if inequality_violation == 0.0 {
                    let before = evaluations.get();
                    let value = evaluate(&point);
                    sampling_nfev += evaluations.get() - before;
                    if evaluation_limit.get() {
                        stop = Some((
                            ConvergenceStatus::MaxEvaluations,
                            "shgo: objective evaluation limit reached during sampling".into(),
                        ));
                        break 'rounds;
                    }
                    if value.is_finite() {
                        value
                    } else {
                        f64::INFINITY
                    }
                } else {
                    f64::INFINITY
                };
                if value.is_finite()
                    && violation <= options.constraint_tolerance
                    && best.as_ref().is_none_or(|(_, current)| value < *current)
                {
                    best = Some((point.clone(), value));
                }
                let vertex = vertices.len();
                vertices.push((point, value));
                cache.insert(key, vertex);
                vertex
            };
            indices.push(vertex);
            values.push(vertices[vertex].1);
        }

        let pool = shgo_minimizer_pool(&values, levels, &strides);
        for (grid_index, plateau) in pool {
            if options.local_max_iterations == 0 || options.local_max_evaluations == 0 {
                stop = Some((
                    if options.local_max_iterations == 0 {
                        ConvergenceStatus::MaxIterations
                    } else {
                        ConvergenceStatus::MaxEvaluations
                    },
                    "shgo: local optimization budget is zero; returning sampling results".into(),
                ));
                break 'rounds;
            }
            let start = &vertices[indices[grid_index]].0;
            let local_bounds = if plateau || !options.constraints.is_empty() {
                global_bounds.clone()
            } else {
                bounds
                    .iter()
                    .zip(&strides)
                    .map(|(&(lo, hi), &stride)| {
                        let coordinate = (grid_index / stride) % levels;
                        (
                            Some(shgo_grid_coordinate(
                                lo,
                                hi,
                                coordinate.saturating_sub(1),
                                subdivisions,
                            )),
                            Some(shgo_grid_coordinate(
                                lo,
                                hi,
                                (coordinate + 1).min(subdivisions),
                                subdivisions,
                            )),
                        )
                    })
                    .collect()
            };
            let before = evaluations.get();
            let local_limit = std::cell::Cell::new(false);
            let local_objective = |point: &[f64]| {
                if evaluations.get() - before >= options.local_max_evaluations {
                    local_limit.set(true);
                    f64::NAN
                } else {
                    evaluate(point)
                }
            };
            let local_options = MinimizeOptions {
                method: Some(if options.constraints.is_empty() {
                    OptimizeMethod::LBfgsB
                } else {
                    OptimizeMethod::Slsqp
                }),
                bounds: Some(&local_bounds),
                constraints: options.constraints,
                tol: Some(options.local_tolerance),
                maxiter: Some(options.local_max_iterations),
                maxfev: Some(options.local_max_evaluations),
                ..MinimizeOptions::default()
            };
            let mut local = minimize(local_objective, start, local_options);
            local_minimizations += 1;
            if let Ok(first) = &local
                && first.success
                && first.x.len() == ndim
                && first.x.iter().zip(&local_bounds).zip(bounds).any(
                    |((&x, &(lower, upper)), &(lo, hi))| {
                        let lower = lower.unwrap_or(lo);
                        let upper = upper.unwrap_or(hi);
                        (lower > lo && x <= lower + 64.0 * f64::EPSILON * lower.abs().max(1.0))
                            || (upper < hi
                                && x >= upper - 64.0 * f64::EPSILON * upper.abs().max(1.0))
                    },
                )
            {
                // A constrained optimum on an artificial star boundary is
                // not necessarily a local minimum of the original box.
                // Keep its finite value if the full-box polish later stops,
                // but only the full-box result may enter the minima pool.
                if let Some(value) = first.fun.filter(|value| value.is_finite())
                    && first
                        .x
                        .iter()
                        .zip(bounds)
                        .all(|(&x, &(lo, hi))| x.is_finite() && x >= lo && x <= hi)
                    && best.as_ref().is_none_or(|(_, current)| value < *current)
                {
                    best = Some((first.x.clone(), value));
                }
                nljev += first.njev;
                nlhev += first.nhev;
                local = minimize(
                    local_objective,
                    &first.x,
                    MinimizeOptions {
                        bounds: Some(&global_bounds),
                        ..local_options
                    },
                );
                local_minimizations += 1;
            }
            nlfev += evaluations.get() - before;
            let local = match local {
                Ok(local) => local,
                Err(error) => {
                    let status = match error {
                        OptError::NonFiniteInput { .. } => ConvergenceStatus::NanEncountered,
                        OptError::EvaluationBudgetExceeded { .. } => {
                            ConvergenceStatus::MaxEvaluations
                        }
                        OptError::NotConverged { .. } | OptError::NoConvergence { .. } => {
                            ConvergenceStatus::PrecisionLoss
                        }
                        _ => return Err(error),
                    };
                    failed_local_minimizations += 1;
                    local_failure_status.get_or_insert(if local_limit.get() {
                        ConvergenceStatus::MaxEvaluations
                    } else {
                        status
                    });
                    if evaluation_limit.get() {
                        stop = Some((
                            ConvergenceStatus::MaxEvaluations,
                            "shgo: objective evaluation limit reached during local optimization"
                                .into(),
                        ));
                        break 'rounds;
                    }
                    continue;
                }
            };
            nljev += local.njev;
            nlhev += local.nhev;
            let finite_point = local.x.len() == ndim
                && local
                    .x
                    .iter()
                    .zip(bounds)
                    .all(|(&x, &(lo, hi))| x.is_finite() && x >= lo && x <= hi);
            let feasible = finite_point
                && shgo_constraint_state(&local.x, options.constraints, &mut constraint_sizes)?.1
                    <= options.constraint_tolerance;
            let converged = local.success
                && feasible
                && local.fun.is_some_and(f64::is_finite)
                && !local_limit.get()
                && !evaluation_limit.get();
            if !converged {
                failed_local_minimizations += 1;
                local_failure_status.get_or_insert(if local_limit.get() {
                    ConvergenceStatus::MaxEvaluations
                } else if !local.success {
                    local.status
                } else {
                    ConvergenceStatus::Infeasible
                });
            }
            if feasible && let Some(value) = local.fun.filter(|value| value.is_finite()) {
                if best.as_ref().is_none_or(|(_, current)| value <= *current) {
                    best = Some((local.x.clone(), value));
                }
                if converged {
                    let duplicate = minima.iter().position(|(point, _)| {
                        point.iter().zip(&local.x).zip(bounds).all(
                            |((&left, &right), &(lo, hi))| {
                                (shgo_unit_coordinate(left, lo, hi)
                                    - shgo_unit_coordinate(right, lo, hi))
                                .abs()
                                    <= options.minimizer_tolerance
                            },
                        )
                    });
                    if let Some(index) = duplicate {
                        if value < minima[index].1 {
                            minima[index] = (local.x, value);
                        }
                    } else {
                        minima.push((local.x, value));
                    }
                }
            }
            if evaluation_limit.get() {
                stop = Some((
                    ConvergenceStatus::MaxEvaluations,
                    "shgo: objective evaluation limit reached during local optimization".into(),
                ));
                break 'rounds;
            }
        }
        completed += 1;
        if iteration + 1 < options.iterations {
            let Some(refined) = subdivisions.checked_mul(2) else {
                stop = Some((
                    ConvergenceStatus::MaxEvaluations,
                    "shgo: refinement would overflow the sampling grid".into(),
                ));
                break;
            };
            subdivisions = refined;
        }
    }

    let success = stop.is_none()
        && completed == options.iterations
        && !minima.is_empty()
        && failed_local_minimizations == 0;
    let (status, message) = if success {
        (
            ConvergenceStatus::Success,
            "shgo: simplicial sampling and local minimizations completed successfully".into(),
        )
    } else if let Some(stop) = stop {
        stop
    } else if failed_local_minimizations > 0 {
        (
            local_failure_status.unwrap_or(ConvergenceStatus::PrecisionLoss),
            format!(
                "shgo: {failed_local_minimizations} local minimization(s) failed to converge to a finite feasible point"
            ),
        )
    } else {
        (
            if options.constraints.is_empty() {
                ConvergenceStatus::NanEncountered
            } else {
                ConvergenceStatus::Infeasible
            },
            "shgo: sampling did not find a converged finite feasible minimizer".into(),
        )
    };
    minima.sort_by(|left, right| left.1.total_cmp(&right.1));
    if success {
        // An exactly represented sampled optimum can have a slightly lower
        // value than a converged point a few ULP away. Return the point whose
        // convergence was actually established, not a falsely failed sample.
        best = minima.first().cloned();
    }
    let (xl, funl) = minima.into_iter().unzip();
    let (x, fun) = best.map_or_else(|| (Vec::new(), None), |(x, value)| (x, Some(value)));
    let maxcv = if options.constraints.is_empty() || x.is_empty() {
        None
    } else {
        Some(shgo_constraint_state(&x, options.constraints, &mut constraint_sizes)?.1)
    };
    Ok(ShgoResult {
        optimum: OptimizeResult {
            x,
            fun,
            success,
            status,
            message,
            nfev: evaluations.get(),
            njev: nljev,
            nhev: nlhev,
            nit: completed,
            jac: None,
            hess_inv: None,
            maxcv,
        },
        xl,
        funl,
        sampling_nfev,
        nlfev,
        nljev,
        nlhev,
        samples: vertices.len(),
        local_minimizations,
        failed_local_minimizations,
    })
}

fn shgo_complex_size(
    dimension: usize,
    subdivisions: usize,
    options: ShgoOptions<'_>,
) -> Result<usize, OptError> {
    let invalid = || {
        OptError::InvalidArgument {
        detail: "shgo: simplicial complex exceeds its sample/edge budget; reduce dimension or subdivisions, or increase the explicit limits".into(),
    }
    };
    let levels = subdivisions.checked_add(1).ok_or_else(invalid)?;
    let edge_base = subdivisions
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(invalid)?;
    let mut points = 1usize;
    let mut edges_with_self = 1usize;
    for _ in 0..dimension {
        points = points.checked_mul(levels).ok_or_else(invalid)?;
        if points > options.max_samples {
            return Err(invalid());
        }
        // Sum over all coordinatewise comparable neighboring vertex pairs:
        // (2*levels-1)^dimension, including the points themselves.
        edges_with_self = edges_with_self.checked_mul(edge_base).ok_or_else(invalid)?;
        if edges_with_self - points > options.max_complex_edges {
            return Err(invalid());
        }
    }
    Ok(points)
}

fn shgo_grid_coordinate(lo: f64, hi: f64, index: usize, subdivisions: usize) -> f64 {
    if index == 0 {
        lo
    } else if index == subdivisions {
        hi
    } else {
        let fraction = index as f64 / subdivisions as f64;
        // A finite interval may have an overflowing width. Convex interpolation
        // keeps every node finite without first forming hi-lo.
        ((1.0 - fraction) * lo + fraction * hi).clamp(lo, hi)
    }
}

fn shgo_unit_coordinate(value: f64, lo: f64, hi: f64) -> f64 {
    let width = hi - lo;
    if width.is_finite() {
        (value - lo) / width
    } else {
        (0.5 * value - 0.5 * lo) / (0.5 * hi - 0.5 * lo)
    }
}

fn shgo_constraint_state(
    point: &[f64],
    constraints: &[Constraint<'_>],
    sizes: &mut [Option<usize>],
) -> Result<(f64, f64), OptError> {
    let mut inequality_violation = 0.0_f64;
    let mut violation = 0.0_f64;
    for (index, constraint) in constraints.iter().enumerate() {
        let values = (constraint.fun)(point);
        if let Some(expected) = sizes[index] {
            if values.len() != expected {
                return Err(OptError::InvalidArgument {
                    detail: format!("shgo: constraint {index} changed its output length"),
                });
            }
        } else {
            sizes[index] = Some(values.len());
        }
        for value in values {
            if !value.is_finite() {
                return Err(OptError::NonFiniteInput {
                    detail: format!("shgo: constraint {index} returned a nonfinite value"),
                });
            }
            match constraint.kind {
                ConstraintType::Eq => violation = violation.max(value.abs()),
                ConstraintType::Ineq => {
                    inequality_violation = inequality_violation.max(-value);
                    violation = violation.max(-value);
                }
            }
        }
    }
    Ok((inequality_violation, violation))
}

/// One representative per minimal connected plateau in the simplicial graph.
/// Each edge is visited once, without storing an exponential adjacency list.
fn shgo_minimizer_pool(values: &[f64], levels: usize, strides: &[usize]) -> Vec<(usize, bool)> {
    fn representative(parent: &mut [usize], mut index: usize) -> usize {
        while parent[index] != index {
            parent[index] = parent[parent[index]];
            index = parent[index];
        }
        index
    }
    let mut parent: Vec<usize> = (0..values.len()).collect();
    let mut has_lower = vec![false; values.len()];
    let mut plateau_size = vec![1usize; values.len()];
    let mut offsets = Vec::new();
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            continue;
        }
        offsets.clear();
        offsets.push(0usize);
        for &stride in strides {
            if (index / stride) % levels + 1 < levels {
                let count = offsets.len();
                for offset_index in 0..count {
                    offsets.push(offsets[offset_index] + stride);
                }
            }
        }
        for &offset in &offsets[1..] {
            let neighbor = index + offset;
            if !values[neighbor].is_finite() {
                continue;
            }
            let left = representative(&mut parent, index);
            let right = representative(&mut parent, neighbor);
            if value < values[neighbor] {
                has_lower[right] = true;
            } else if value > values[neighbor] {
                has_lower[left] = true;
            } else if left != right {
                let (root, child) = if left < right {
                    (left, right)
                } else {
                    (right, left)
                };
                parent[child] = root;
                has_lower[root] |= has_lower[child];
                plateau_size[root] += plateau_size[child];
            }
        }
    }
    let mut pool: Vec<(usize, bool)> = (0..values.len())
        .filter(|&index| parent[index] == index && values[index].is_finite() && !has_lower[index])
        .map(|index| (index, plateau_size[index] > 1))
        .collect();
    pool.sort_by(|&(left, _), &(right, _)| {
        values[left]
            .total_cmp(&values[right])
            .then(left.cmp(&right))
    });
    pool
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

impl gsa::Uniform for SimpleRng {
    fn next_f64(&mut self) -> f64 {
        Self::next_f64(self)
    }
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
        let inner_converged = result.success;
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

        // Check convergence. Feasibility alone is not optimality: a feasible x0 with a small
        // `max_inner` used to be reported as the solution on the first pass, far from the optimum
        // (frankenscipy-szq1n.7). A feasible point whose subproblem did not converge is
        // warm-started into another outer pass instead.
        if max_violation < 1e-6 && inner_converged {
            let fval = f(&x);
            return Ok(OptimizeResult {
                x,
                fun: Some(fval),
                // status: max constraint violation < 1e-6 and inner subproblem success
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
/// Returns (xa, xb, xc, fa, fb, fc). A valid bracket has xb strictly between
/// xa and xc, with fb no greater than either endpoint and below at least one.
/// If the downhill search fails, the tuple contains its last three samples.
/// Matches `scipy.optimize.bracket`.
pub fn bracket<F>(f: F, xa: f64, xb: f64) -> (f64, f64, f64, f64, f64, f64)
where
    F: Fn(f64) -> f64,
{
    let Ok(samples) = minimize::downhill_bracket(
        &mut |x| Ok::<_, std::convert::Infallible>(f(x)),
        xa,
        xb,
    );
    let [(xa, fa), (xb, fb), (xc, fc)] = samples.points;
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
    let mut eval = |x| Ok(func(x));
    let points = match *brack.unwrap_or(&[0.0, 1.0]) {
        [xa, xb] => {
            let samples = minimize::downhill_bracket(&mut eval, xa, xb)?;
            if !samples.valid {
                return Err(OptError::InvalidArgument {
                    detail: "could not find a valid downhill bracket".to_string(),
                });
            }
            samples.points
        }
        [xa, xb, xc] => {
            let (xa, xc) = (xa.min(xc), xa.max(xc));
            if !(xa < xb && xb < xc) {
                return Err(OptError::InvalidArgument {
                    detail: "middle bracketing point must lie between the endpoints".to_string(),
                });
            }
            let (fa, fb, fc) = (func(xa), func(xb), func(xc));
            if !(fb < fa && fb < fc) {
                return Err(OptError::InvalidArgument {
                    detail: "brack must be a downhill bracket: f(xb) < f(xa) and f(xb) < f(xc)"
                        .to_string(),
                });
            }
            [(xa, fa), (xb, fb), (xc, fc)]
        }
        _ => {
            return Err(OptError::InvalidArgument {
                detail: "brack must have 2 or 3 elements".to_string(),
            });
        }
    };
    Ok(minimize::refine_brent(&mut eval, points, tol, maxiter)?.0)
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

/// Every component within `NONLIN_FTOL`. A NaN component is not: the max-norm fold this
/// replaced let `f64::max` drop the NaN, so a residual gone NaN (x^2 + 1 = 0 from
/// linearmixing) was returned as a root where SciPy raises NoConvergence.
fn nonlin_converged(fx: &[f64]) -> bool {
    fx.iter().all(|v| v.abs() < NONLIN_FTOL)
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
    Err(OptError::NoConvergence {
        detail: format!("diagbroyden failed to converge in {maxiter} iterations"),
    })
}

/// Find a root of a vector function using a scalar Jacobian approximation
/// (linear mixing), matching `scipy.optimize.linearmixing(F, xin)`.
///
/// The Jacobian is fixed at `-1/alpha`, so the step is `dx = alpha·F(x)`,
/// globalized by an Armijo line search. As scipy warns, convergence depends
/// strongly on a well-chosen (often negative) `alpha`; when `alpha` is `None`
/// it is auto-scaled like scipy's. Returns the converged root or, where SciPy
/// raises `NoConvergence`, [`OptError::NoConvergence`].
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
    Err(OptError::NoConvergence {
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
    Err(OptError::NoConvergence {
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
    let mut p0 = x0;
    for _iter in 0..maxiter {
        let p1 = f(p0);
        let p2 = f(p1);
        let d = p2 - 2.0 * p1 + p0;
        let p = if d != 0.0 {
            p0 - (p1 - p0).powi(2) / d
        } else {
            p2
        };
        let relerr = if p0 != 0.0 { (p - p0) / p0 } else { p };
        if relerr.abs() < tol {
            return Ok(p);
        }
        p0 = p;
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

/// Column-block width for the blocked Gram accumulation. A `B x B` `f64` tile is
/// `B²·8` bytes and must stay resident in L1 across the whole row sweep; 64 gives 32 KB,
/// which fits alongside the two streamed `A` rows on every core this runs on.
const NNLS_GRAM_BLOCK: usize = 64;

/// Use the blocked Gram accumulation instead of the per-row rank-1 sweep.
///
/// **MEASURED AND REJECTED — SHIPS OFF.** Bit-identical either way (see
/// `nnls_gram_blocked`), so this is purely a cost knob, and it costs. Both arms in one
/// binary, alternated, `n = m = 256`:
///
/// ```text
/// blocked ON   fsci 3.084 / 3.078 ms   scipy/fsci 0.451x / 0.453x
/// blocked OFF  fsci 2.925 / 2.940 ms   scipy/fsci 0.478x / 0.473x
/// 1.05x SLOWER, ranges disjoint
/// ```
///
/// The premise was that the rank-1 sweep touches the whole 512 KB upper triangle once per
/// observation and so streams ~64 MB over 256 rows. The premise is arithmetically right and
/// operationally irrelevant: at `n = 256` the Gram is 512 KB and FITS IN L2, so those
/// re-reads are L2 hits, not memory traffic, and there was no bandwidth problem to solve.
/// Blocking then only adds loop nesting and the per-row triangle test, and shortens the
/// vectorized inner run from up to `n` down to the 64-wide block.
///
/// Kept, defaulted off, so the next attempt starts from a measured dead end. It would
/// become the right shape again at an `n` where the Gram genuinely leaves cache — around
/// `n >= 512` (2 MB) — which is not the size this op is used at here.
pub static NNLS_GRAM_BLOCKED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Use the direct Lawson-Hanson QR active-set solve instead of the normal-equations path.
///
/// The shipping arm never forms `AᵀA`: it maintains the passive factorization as columns enter
/// and uses Givens rotations to downdate it when a trial least-squares solution expels columns.
/// This avoids both the normal-equations conditioning penalty and the dense Gram precompute.
pub static NNLS_DIRECT_QR: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Counts NNLS calls that entered the direct QR arm. The perf driver checks this around both
/// arms so an apparently clean A/B cannot be a switch wired to nothing.
pub static NNLS_DIRECT_QR_HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Insert an original column into the passive QR factorization using two-pass modified
/// Gram-Schmidt. `q` owns orthonormal columns and `r` is an `n`-stride upper triangle.
fn nnls_qr_add_column(
    a: &[Vec<f64>],
    b: &[f64],
    column: usize,
    q: &mut Vec<Vec<f64>>,
    r: &mut [f64],
    qtb: &mut Vec<f64>,
    n: usize,
) -> bool {
    let p = q.len();
    let mut v: Vec<f64> = a.iter().map(|row| row[column]).collect();

    // `r` is capacity-sized so a prior Givens downdate leaves retired cells behind. An
    // insertion into that logical column must start from zero, not accumulate stale rotations.
    for k in 0..p {
        r[k * n + p] = 0.0;
    }

    // Reorthogonalization is material here: NNLS can select nearly dependent columns, and a
    // one-pass projection can make a correct positive passive solution look negative.
    for _ in 0..2 {
        for k in 0..p {
            let projection: f64 = q[k].iter().zip(v.iter()).map(|(&u, &z)| u * z).sum();
            r[k * n + p] += projection;
            for (entry, &basis) in v.iter_mut().zip(q[k].iter()) {
                *entry -= projection * basis;
            }
        }
    }

    let norm = v.iter().map(|value| value * value).sum::<f64>().sqrt();
    if !norm.is_finite() || norm <= f64::EPSILON.sqrt() {
        return false;
    }
    for value in &mut v {
        *value /= norm;
    }
    r[p * n + p] = norm;
    qtb.push(v.iter().zip(b.iter()).map(|(&u, &rhs)| u * rhs).sum());
    q.push(v);
    true
}

/// Remove passive column `removed` from a QR factorization.
///
/// Deleting a column from an upper-triangular `R` leaves one subdiagonal. Chase that bulge down
/// with Givens rotations and apply the inverse rotations to `Q`; the remaining prefix is again a
/// QR factorization of exactly the surviving original columns. This is the downdate that keeps
/// Lawson-Hanson's inner feasibility loop incremental instead of refactorizing after every exit.
fn nnls_qr_remove_column(
    removed: usize,
    q: &mut Vec<Vec<f64>>,
    r: &mut [f64],
    qtb: &mut Vec<f64>,
    n: usize,
) {
    let p = q.len();
    debug_assert!(removed < p);

    for row in 0..p {
        for col in removed..(p - 1) {
            r[row * n + col] = r[row * n + col + 1];
        }
    }
    for pivot in removed..(p - 1) {
        let upper = r[pivot * n + pivot];
        let lower = r[(pivot + 1) * n + pivot];
        let hypot = upper.hypot(lower);
        if hypot == 0.0 {
            continue;
        }
        let cosine = upper / hypot;
        let sine = lower / hypot;
        for col in pivot..(p - 1) {
            let top = r[pivot * n + col];
            let bottom = r[(pivot + 1) * n + col];
            r[pivot * n + col] = cosine * top + sine * bottom;
            r[(pivot + 1) * n + col] = -sine * top + cosine * bottom;
        }
        let top_rhs = qtb[pivot];
        let bottom_rhs = qtb[pivot + 1];
        qtb[pivot] = cosine * top_rhs + sine * bottom_rhs;
        qtb[pivot + 1] = -sine * top_rhs + cosine * bottom_rhs;
        for row in 0..q[pivot].len() {
            let left = q[pivot][row];
            let right = q[pivot + 1][row];
            q[pivot][row] = cosine * left + sine * right;
            q[pivot + 1][row] = -sine * left + cosine * right;
        }
    }
    q.remove(p - 1);
    qtb.pop();
}

/// Direct Lawson-Hanson NNLS using an incrementally-maintained passive QR factorization.
fn nnls_direct_qr(a: &[Vec<f64>], b: &[f64], n: usize) -> Result<(Vec<f64>, f64), OptError> {
    let mut x = vec![0.0; n];
    let mut residual = b.to_vec();
    let mut passive = vec![false; n];
    let mut passive_indices = Vec::new();
    let mut q = Vec::new();
    let mut r = vec![0.0; n * n];
    let mut qtb = Vec::new();
    let mut candidate = vec![0.0; n];
    let tolerance = 1.0e-10;

    for _ in 0..(3 * n) {
        let mut gradient = vec![0.0; n];
        for (row, &residual_value) in a.iter().zip(residual.iter()) {
            for (gradient_value, &aij) in gradient.iter_mut().zip(row.iter()) {
                *gradient_value += aij * residual_value;
            }
        }
        let mut entering = None;
        let mut best_gradient = tolerance;
        for (index, &gradient_value) in gradient.iter().enumerate() {
            if !passive[index] && gradient_value > best_gradient {
                best_gradient = gradient_value;
                entering = Some(index);
            }
        }
        let Some(entering) = entering else {
            break;
        };

        if !nnls_qr_add_column(a, b, entering, &mut q, &mut r, &mut qtb, n) {
            // A dependent column has zero projected norm. It cannot improve the passive solve;
            // leave it active so the outer gradient test does not select it again.
            passive[entering] = true;
            continue;
        }
        passive[entering] = true;
        passive_indices.push(entering);

        loop {
            candidate.fill(0.0);
            let p = passive_indices.len();
            for local in (0..p).rev() {
                let mut value = qtb[local];
                for next in (local + 1)..p {
                    value -= r[local * n + next] * candidate[passive_indices[next]];
                }
                candidate[passive_indices[local]] = value / r[local * n + local];
            }

            let mut alpha = 1.0_f64;
            let mut has_nonpositive = false;
            for &index in &passive_indices {
                let next = candidate[index];
                if next <= tolerance {
                    has_nonpositive = true;
                    let denominator = x[index] - next;
                    if denominator > 0.0 {
                        alpha = alpha.min(x[index] / denominator);
                    }
                }
            }
            if !has_nonpositive {
                x.copy_from_slice(&candidate);
                break;
            }

            for &index in &passive_indices {
                x[index] += alpha * (candidate[index] - x[index]);
            }
            let mut local = 0;
            while local < passive_indices.len() {
                let index = passive_indices[local];
                if x[index] <= tolerance {
                    x[index] = 0.0;
                    passive[index] = false;
                    passive_indices.remove(local);
                    nnls_qr_remove_column(local, &mut q, &mut r, &mut qtb, n);
                } else {
                    local += 1;
                }
            }
            if passive_indices.is_empty() {
                break;
            }
        }

        for (row_index, row) in a.iter().enumerate() {
            residual[row_index] = b[row_index]
                - row
                    .iter()
                    .zip(x.iter())
                    .map(|(&aij, &xj)| aij * xj)
                    .sum::<f64>();
        }
    }
    let residual_norm = residual
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    Ok((x, residual_norm))
}

/// Accumulate `AᵀA` (upper triangle) and `Aᵀb` with the GRAM TILE held in cache instead of
/// the row.
///
/// WHY. The rank-1 sweep visits the entire upper triangle of the Gram once per observation:
/// at `n = 256` that is 256 KB touched per row and 64 MB streamed over `m = 256` rows, for
/// a Gram that is only 512 KB and an `A` that is only 512 KB. The arithmetic is `O(m·n²)`
/// either way; the traffic is not. SciPy gets this matrix from one OpenBLAS `dgemm`, which
/// is exactly the dependency this project does not take, so the answer has to be blocking.
///
/// Holding a `B x B` tile of the Gram in L1 and streaming `A`'s two column panels through
/// it inverts the residency: each tile pair reads `m·2B` elements of `A` and writes its
/// tile once, so total traffic falls from `O(m·n²)` to `O(m·n²/B)` — about 2.6 MB here
/// against 64 MB.
///
/// BIT-IDENTICAL to the rank-1 sweep. Every `gram[j1][j2]` still accumulates its `m`
/// products in ASCENDING `i`, because the row loop stays innermost within a tile; only the
/// ORDER IN WHICH INDEPENDENT ENTRIES are visited changes, and separate entries never
/// interact. `Aᵀb` is accumulated in its own single pass over the rows, again in ascending
/// `i`, so it is unchanged too.
fn nnls_gram_blocked(a_flat: &[f64], b: &[f64], g: &mut [f64], ab: &mut [f64], n: usize) {
    for (ai, &bi) in a_flat.chunks_exact(n).zip(b.iter()) {
        for (abj, &v) in ab.iter_mut().zip(ai.iter()) {
            *abj += v * bi;
        }
    }
    let mut j1_block = 0usize;
    while j1_block < n {
        let j1_end = (j1_block + NNLS_GRAM_BLOCK).min(n);
        let mut j2_block = j1_block;
        while j2_block < n {
            let j2_end = (j2_block + NNLS_GRAM_BLOCK).min(n);
            for ai in a_flat.chunks_exact(n) {
                for j1 in j1_block..j1_end {
                    let v1 = ai[j1];
                    // Upper triangle only: on the diagonal tile the row starts at `j1`.
                    let lo = j2_block.max(j1);
                    if lo >= j2_end {
                        continue;
                    }
                    let grow = &mut g[j1 * n + lo..j1 * n + j2_end];
                    for (gg, &v2) in grow.iter_mut().zip(ai[lo..j2_end].iter()) {
                        *gg += v1 * v2;
                    }
                }
            }
            j2_block += NNLS_GRAM_BLOCK;
        }
        j1_block += NNLS_GRAM_BLOCK;
    }
}

/// Accumulate one observation's rank-1 contribution into the UPPER triangle of a partial Gram
/// matrix `g` (row-major n×n) and `Aᵀb` accumulator `ab`. Used by the parallel-reduction Gram
/// precompute in `nnls`, and as the A/B reference arm for `nnls_gram_blocked`.
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

    if NNLS_DIRECT_QR.load(std::sync::atomic::Ordering::Relaxed) {
        NNLS_DIRECT_QR_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return nnls_direct_qr(a, b, n);
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
        if NNLS_GRAM_BLOCKED.load(std::sync::atomic::Ordering::Relaxed) {
            nnls_gram_blocked(&a_flat, b, &mut gram, &mut atb, n);
        } else {
            for (ai, &bi) in a_flat.chunks(n).zip(b.iter()) {
                nnls_gram_accumulate(&mut gram, &mut atb, ai, bi, n);
            }
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

/// Minimum-norm least-squares solution without squaring the condition number.
/// The relative singular-value cutoff follows NumPy's `lstsq`: machine precision
/// for the initial solve, and `max(m, n) * eps` for BVLS free-set solves.
fn lsq_linear_minimum_norm(a: &[Vec<f64>], b: &[f64], rcond: f64) -> Result<Vec<f64>, OptError> {
    let n = a[0].len();
    let mut x = vec![0.0; n];
    if n == 0 {
        return Ok(x);
    }
    let (left, singular, right) = trf::svd_thin(a, n);
    if singular.iter().any(|value| !value.is_finite()) {
        return Err(OptError::NotConverged {
            detail: "lsq_linear: non-finite singular value in the free least-squares problem"
                .to_string(),
        });
    }
    let largest = singular[0];
    let b_scale = b
        .iter()
        .fold(0.0_f64, |scale, value| scale.max(value.abs()));
    if largest == 0.0 || b_scale == 0.0 {
        return Ok(x);
    }
    for ((u, &sigma), v) in left.iter().zip(&singular).zip(&right) {
        if sigma / largest <= rcond {
            continue;
        }
        let projection: f64 = u.iter().zip(b).map(|(&ui, &bi)| ui * (bi / b_scale)).sum();
        let ratio = b_scale / sigma;
        let coefficient = if ratio.is_finite() && ratio != 0.0 {
            projection * ratio
        } else {
            (projection / sigma) * b_scale
        };
        for (xi, &vi) in x.iter_mut().zip(v) {
            *xi += vi * coefficient;
        }
    }
    if x.iter().any(|value| !value.is_finite()) {
        return Err(OptError::NotConverged {
            detail: "lsq_linear: the free least-squares solution is not finite".to_string(),
        });
    }
    Ok(x)
}

/// Solve for the free variables against the residual of the fixed variables.
fn lsq_linear_free_solution(
    a: &[Vec<f64>],
    b: &[f64],
    x: &[f64],
    state: &[i8],
    free: &[usize],
) -> Result<Vec<f64>, OptError> {
    let free_a: Vec<Vec<f64>> = a
        .iter()
        .map(|row| free.iter().map(|&column| row[column]).collect())
        .collect();
    let free_b: Vec<f64> = a
        .iter()
        .zip(b)
        .map(|(row, &bi)| {
            bi - row
                .iter()
                .zip(x)
                .zip(state)
                .filter(|&(_, &side)| side != 0)
                .map(|((&aij, &xj), _)| aij * xj)
                .sum::<f64>()
        })
        .collect();
    if free_b.iter().any(|value| !value.is_finite()) {
        return Err(OptError::NotConverged {
            detail: "lsq_linear: non-finite residual in the free least-squares problem".to_string(),
        });
    }
    lsq_linear_minimum_norm(
        &free_a,
        &free_b,
        f64::EPSILON * a.len().max(free.len()) as f64,
    )
}

/// Residual-space projected gradient, normalized by the matrix and data norms.
/// This checks free-variable stationarity as well as the signs at active bounds.
/// Common scaling of A and b leaves the stopping test and the minimizer unchanged.
/// Dot-product roundoff is bounded using `|A| |x|`, so cancellation in Ax does
/// not demand accuracy beyond what the supplied floating-point data can provide.
fn lsq_linear_optimality(
    a: &[Vec<f64>],
    b: &[f64],
    x: &[f64],
    state: &[i8],
    a_norm: f64,
) -> Result<(f64, Option<usize>), OptError> {
    let model: Vec<f64> = a
        .iter()
        .map(|row| row.iter().zip(x).map(|(&aij, &xj)| aij * xj).sum())
        .collect();
    if model.iter().any(|value| !value.is_finite()) {
        return Err(OptError::NotConverged {
            detail: "lsq_linear: non-finite fitted values prevent an optimality check".to_string(),
        });
    }
    let data_scale = model
        .iter()
        .chain(b)
        .fold(0.0_f64, |scale, value| scale.max(value.abs()));
    if a_norm == 0.0 || data_scale == 0.0 {
        return Ok((0.0, None));
    }
    let model_norm = model
        .iter()
        .fold(0.0_f64, |norm, &value| norm.hypot(value / data_scale));
    let b_norm = b
        .iter()
        .fold(0.0_f64, |norm, &value| norm.hypot(value / data_scale));
    let reference_norm = model_norm.max(b_norm);
    let mut gradient = vec![0.0; x.len()];
    let mut activity_norm = 0.0_f64;
    let mut residual_norm = 0.0_f64;
    for ((row, &prediction), &bi) in a.iter().zip(&model).zip(b) {
        let residual = (prediction / data_scale - bi / data_scale) / reference_norm;
        residual_norm = residual_norm.hypot(residual);
        let activity: f64 = row
            .iter()
            .zip(x)
            .map(|(&aij, &xj)| (aij * xj).abs() / data_scale)
            .sum();
        activity_norm = activity_norm.hypot(activity);
        for (gj, &aij) in gradient.iter_mut().zip(row) {
            *gj += (aij / a_norm) * residual;
        }
    }
    // For a floating-point dot product, |fl(a·x) - a·x| <= gamma_n |a|·|x|,
    // gamma_n = n*eps/(1-n*eps). Include residual subtraction and the gradient
    // dot products as well. This is an absolute backward-error allowance in the
    // same normalized units as the gradient, not a larger solver tolerance.
    let model_roundoff = x.len() as f64 * f64::EPSILON;
    let gradient_roundoff = (a.len() as f64 + 2.0) * f64::EPSILON;
    let roundoff = (model_roundoff / (1.0 - model_roundoff)) * activity_norm / reference_norm
        + f64::EPSILON * (model_norm + b_norm) / reference_norm
        + (gradient_roundoff / (1.0 - gradient_roundoff)) * residual_norm;
    if !roundoff.is_finite() {
        return Err(OptError::NotConverged {
            detail: "lsq_linear: the roundoff bound prevents a finite optimality check".to_string(),
        });
    }
    let mut optimality = 0.0_f64;
    let mut best_bound_score = 0.0;
    let mut release = None;
    for (j, (&gj, &side)) in gradient.iter().zip(state).enumerate() {
        let violation = match side {
            -1 => -gj,
            0 => gj.abs(),
            1 => gj,
            _ => 0.0, // A variable with equal finite bounds is fixed permanently.
        };
        optimality = optimality.max(violation);
        if matches!(side, -1 | 1) && violation > best_bound_score {
            best_bound_score = violation;
            release = Some(j);
        }
    }
    Ok(((optimality - roundoff).max(0.0), release))
}

/// Solve the bound-constrained linear least-squares problem
/// `min ‖A·x − b‖²` subject to `lb ≤ x ≤ ub`, matching
/// `scipy.optimize.lsq_linear(A, b, bounds=(lb, ub), method='bvls')`.
///
/// Uses the BVLS active-set algorithm (Stark & Parker), with direct SVD solves
/// for rank-deficient and underdetermined free-variable problems. Bounds may be
/// infinite (`f64::NEG_INFINITY`/`INFINITY`) for one-sided or unconstrained
/// variables. Equal finite bounds fix a variable. When minimizers are not
/// unique, the initial unconstrained solve returns the minimum-norm solution;
/// the active-set path may select another minimizer when bounds bind.
///
/// The projected gradient is checked in residual space to relative tolerance
/// `1e-10`, accounting for the roundoff bound of the matrix-vector products.
/// Numerical failure or exhaustion of the iteration budget returns
/// [`OptError::NotConverged`], rather than an unchecked feasible point.
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
        if lo.is_nan() || hi.is_nan() || lo > hi || lo == f64::INFINITY || hi == f64::NEG_INFINITY {
            return Err(OptError::InvalidBounds {
                detail: format!("invalid bounds for variable {index}"),
            });
        }
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    // A common data scale protects the SVD and the residual calculations at both
    // ends of the floating-point range without changing x or its bounds. Divide
    // instead of taking a reciprocal, which can overflow for subnormal data.
    let scale = a
        .iter()
        .flatten()
        .chain(b)
        .fold(0.0_f64, |scale, value| scale.max(value.abs()));
    let scale = if scale == 0.0 { 1.0 } else { scale };
    if a.iter()
        .flatten()
        .chain(b)
        .any(|&value| value != 0.0 && value / scale == 0.0)
    {
        return Err(OptError::NotConverged {
            detail: "lsq_linear: the data range cannot be scaled without underflow".to_string(),
        });
    }
    let a: Vec<Vec<f64>> = a
        .iter()
        .map(|row| row.iter().map(|value| value / scale).collect())
        .collect();
    let b: Vec<f64> = b.iter().map(|value| value / scale).collect();
    let a_norm = a
        .iter()
        .flatten()
        .fold(0.0_f64, |norm, &value| norm.hypot(value));
    let mut x = lsq_linear_minimum_norm(&a, &b, f64::EPSILON)?;

    // State: -1 = lower, +1 = upper, 0 = free, 2 = fixed at equal bounds.
    let mut state = vec![0i8; n];
    for j in 0..n {
        if lb[j] == ub[j] {
            state[j] = 2;
            x[j] = lb[j];
        } else if x[j] <= lb[j] {
            state[j] = -1;
            x[j] = lb[j];
        } else if x[j] >= ub[j] {
            state[j] = 1;
            x[j] = ub[j];
        }
    }

    // Find a feasible free-set least-squares solution before beginning BVLS.
    // Each unsuccessful pass fixes at least one additional variable.
    if state.iter().any(|&side| side != 0) {
        loop {
            let free: Vec<usize> = (0..n).filter(|&j| state[j] == 0).collect();
            if free.is_empty() {
                break;
            }
            let z = lsq_linear_free_solution(&a, &b, &x, &state, &free)?;
            let mut blocked = false;
            for (&j, &zj) in free.iter().zip(&z) {
                if zj < lb[j] {
                    x[j] = lb[j];
                    state[j] = -1;
                    blocked = true;
                } else if zj > ub[j] {
                    x[j] = ub[j];
                    state[j] = 1;
                    blocked = true;
                } else {
                    x[j] = zj;
                }
            }
            if !blocked {
                break;
            }
        }
    }

    let tolerance = 1e-10;
    let max_outer = 10 * (n + 1);
    for _ in 0..max_outer {
        let (optimality, release) = lsq_linear_optimality(&a, &b, &x, &state, a_norm)?;
        if optimality <= tolerance {
            return Ok(x);
        }
        let Some(j_free) = release else {
            return Err(OptError::NotConverged {
                detail: format!(
                    "lsq_linear: free-variable optimality {optimality:e} exceeds tolerance"
                ),
            });
        };
        state[j_free] = 0;

        // Each blocking step removes a free variable. The final pass either
        // accepts a feasible free solution or exhausts the finite free set.
        let mut solved = false;
        for _ in 0..=n {
            let free: Vec<usize> = (0..n).filter(|&j| state[j] == 0).collect();
            if free.is_empty() {
                solved = true;
                break;
            }
            let z = lsq_linear_free_solution(&a, &b, &x, &state, &free)?;
            let mut alpha = 1.0_f64;
            let mut block = None;
            for (&j, &zj) in free.iter().zip(&z) {
                let side = if zj < lb[j] {
                    -1
                } else if zj > ub[j] {
                    1
                } else {
                    0
                };
                if side != 0 {
                    let bound = if side == -1 { lb[j] } else { ub[j] };
                    let difference = zj - x[j];
                    let fraction = if difference.is_finite() {
                        (bound - x[j]) / difference
                    } else {
                        (0.5 * bound - 0.5 * x[j]) / (0.5 * zj - 0.5 * x[j])
                    };
                    if block.is_none() || fraction < alpha {
                        alpha = fraction.clamp(0.0, 1.0);
                        block = Some((j, side));
                    }
                }
            }
            for (&j, &zj) in free.iter().zip(&z) {
                x[j] = ((1.0 - alpha) * x[j] + alpha * zj).clamp(lb[j], ub[j]);
            }
            match block {
                Some((j, side)) => {
                    x[j] = if side == -1 { lb[j] } else { ub[j] };
                    state[j] = side;
                }
                None => {
                    solved = true;
                    break;
                }
            }
        }
        if !solved {
            return Err(OptError::NotConverged {
                detail: "lsq_linear: the free-set feasibility loop did not converge".to_string(),
            });
        }
    }
    Err(OptError::NotConverged {
        detail: format!("lsq_linear: exceeded {max_outer} BVLS iterations"),
    })
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

/// A finite value with an extended binary exponent. Block weights and weighted
/// sums may exceed f64's range even though every fitted mean is representable.
#[derive(Clone, Copy)]
struct IsotonicNumber {
    mantissa: f64,
    exponent: i32,
}

impl IsotonicNumber {
    const SIGN_AND_FRACTION: u64 = 0x800f_ffff_ffff_ffff;

    fn new(value: f64) -> Self {
        if value == 0.0 {
            return Self {
                mantissa: value,
                exponent: 0,
            };
        }
        let mut bits = value.to_bits();
        let mut exponent = ((bits >> 52) & 0x7ff) as i32 - 1023;
        if exponent == -1023 {
            bits = (value * 4_503_599_627_370_496.0).to_bits();
            exponent = ((bits >> 52) & 0x7ff) as i32 - 1023 - 52;
        }
        Self {
            mantissa: f64::from_bits((bits & Self::SIGN_AND_FRACTION) | (1023_u64 << 52)),
            exponent,
        }
    }

    fn from_parts(mantissa: f64, exponent: i32) -> Self {
        let mut value = Self::new(mantissa);
        if value.mantissa != 0.0 {
            value.exponent += exponent;
        }
        value
    }

    fn add(self, other: Self) -> Self {
        if self.mantissa == 0.0 {
            return other;
        }
        if other.mantissa == 0.0 {
            return self;
        }
        let (large, small) = if self.exponent >= other.exponent {
            (self, other)
        } else {
            (other, self)
        };
        let difference = large.exponent - small.exponent;
        if difference > 54 {
            return large;
        }
        let scale = f64::from_bits(((1023 - difference) as u64) << 52);
        Self::from_parts(large.mantissa + small.mantissa * scale, large.exponent)
    }

    fn sub(self, other: Self) -> Self {
        self.add(Self {
            mantissa: -other.mantissa,
            exponent: other.exponent,
        })
    }

    fn abs_ge(self, other: Self) -> bool {
        if self.mantissa == 0.0 || other.mantissa == 0.0 {
            return self.mantissa.abs() >= other.mantissa.abs();
        }
        self.exponent > other.exponent
            || (self.exponent == other.exponent && self.mantissa.abs() >= other.mantissa.abs())
    }

    fn ratio(self, denominator: Self) -> f64 {
        let value = Self::from_parts(
            self.mantissa / denominator.mantissa,
            self.exponent - denominator.exponent,
        );
        if value.mantissa == 0.0 {
            return value.mantissa;
        }
        if value.exponent > 1023 {
            return f64::INFINITY.copysign(value.mantissa);
        }
        if value.exponent >= -1022 {
            return f64::from_bits(
                (value.mantissa.to_bits() & Self::SIGN_AND_FRACTION)
                    | (((value.exponent + 1023) as u64) << 52),
            );
        }
        if value.exponent < -1075 {
            return 0.0_f64.copysign(value.mantissa);
        }
        let scale = f64::from_bits(((value.exponent + 2045) as u64) << 52);
        (value.mantissa * f64::MIN_POSITIVE) * scale
    }
}

/// Compensated block sums also retain small observations when large responses
/// of opposite signs cancel during a later merge.
#[derive(Clone, Copy)]
struct IsotonicSum {
    value: IsotonicNumber,
    correction: IsotonicNumber,
}

impl IsotonicSum {
    fn product(weight: f64, response: f64) -> Self {
        let weight = IsotonicNumber::new(weight);
        let response = IsotonicNumber::new(response);
        let product = weight.mantissa * response.mantissa;
        let exponent = weight.exponent + response.exponent;
        Self {
            value: IsotonicNumber::from_parts(product, exponent),
            correction: IsotonicNumber::from_parts(
                weight.mantissa.mul_add(response.mantissa, -product),
                exponent,
            ),
        }
    }

    fn accumulate(&mut self, value: IsotonicNumber) {
        let sum = self.value.add(value);
        let error = if self.value.abs_ge(value) {
            self.value.sub(sum).add(value)
        } else {
            value.sub(sum).add(self.value)
        };
        self.correction = self.correction.add(error);
        self.value = sum;
    }

    fn merge(&mut self, other: Self) {
        self.accumulate(other.value);
        self.accumulate(other.correction);
    }

    fn total(self) -> IsotonicNumber {
        self.value.add(self.correction)
    }
}

/// A pooled block of adjacent points produced by the PAVA stack.
struct IsotonicBlock {
    count: usize,
    weight: IsotonicNumber,
    weighted_sum: IsotonicSum,
    mean: f64,
}

/// Isotonic regression: fit a non-decreasing least-squares function.
///
/// Minimizes `Σ wᵢ(yᵢ − xᵢ)²` subject to `x` being non-decreasing, via the
/// Pool Adjacent Violators Algorithm (PAVA). Matches
/// `scipy.optimize.isotonic_regression` (the `increasing=True` case).
///
/// `weights` must be strictly positive and the same length as `y`; violating
/// either, or a non-finite entry in `y`/`weights`, yields a `NaN` vector.
/// Finite positive weights retain their relative influence without a global
/// normalization that could round a small weight to zero.
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

    // PAVA remains linear in the number of observations: each block is pushed
    // and popped once. Compare finite means directly; cross multiplication of
    // weighted sums can overflow or underflow and hide a decreasing pair.
    let mut blocks: Vec<IsotonicBlock> = Vec::with_capacity(n);
    for i in 0..n {
        let mut block = IsotonicBlock {
            count: 1,
            weight: IsotonicNumber::new(w[i]),
            weighted_sum: IsotonicSum::product(w[i], y[i]),
            mean: y[i],
        };
        while let Some(prev) = blocks.last() {
            if prev.mean > block.mean {
                let prev = blocks.pop().expect("last() returned Some");
                block.count += prev.count;
                block.weight = block.weight.add(prev.weight);
                block.weighted_sum.merge(prev.weighted_sum);
                // A positive weighted mean lies between its two block means.
                // This also bounds a final rounding overshoot at f64::MAX.
                block.mean = block
                    .weighted_sum
                    .total()
                    .ratio(block.weight)
                    .clamp(block.mean, prev.mean);
            } else {
                break;
            }
        }
        blocks.push(block);
    }

    let mut result = Vec::with_capacity(n);
    for block in &blocks {
        result.resize(result.len() + block.count, block.mean);
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
                // status: ‖∇f‖₂ < tol (gradient checked finite above)
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
                // status: projected step ‖clamp(x − lr·g) − x‖₂ < tol
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
    // status: AND identity; `&=` with every partial's success below (rows, cols ≥ 1)
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
        // A changed output length is an invalid `f`, not a non-finite value: it fails the
        // whole Jacobian rather than ending this partial with status NonFiniteValue.
        let reshaped = std::cell::Cell::new(false);
        let component = |value: f64| {
            let mut shifted = scratch.borrow_mut();
            shifted[column] = value;
            let values = f(&shifted);
            shifted[column] = base;
            if values.len() == rows {
                values[row]
            } else {
                reshaped.set(true);
                f64::NAN
            }
        };
        let partial = adaptive_first_derivative(&component, x[column], options)?;
        if reshaped.get() {
            return Err(OptError::NonFiniteInput {
                detail: format!("f(x) changed its output length from {rows}"),
            });
        }
        Ok(partial)
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
    // status: AND identity; `&=` with every component's success below (n ≥ 1 validated)
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
    // SciPy 1.17.1's `scipy.differentiate.derivative` for one element (frankenscipy-3vcjx):
    // a fixed-order stencil (`_derivative_weights`) over steps h, h/c, …, h/c^(n−1) (central)
    // or h, h/√c, … (one-sided), the step divided by c each iteration so that two new
    // abscissae reuse every stored value, `error = |df − df_last|`, and SciPy's termination
    // order. The step is absolute (`initial_step`, not scaled by |x|), and a non-finite value
    // is a status (−3), not an error. The previous Richardson scheme scaled the step by
    // 1 + |x| and refused any non-finite evaluation, so log at x = 1 (x − h = 0) failed.
    let order = validate_differentiate_options(options)?;
    let n = order / 2;
    let fac = options.step_factor;
    let (central_weights, right_weights) = derivative_weights(n, fac);
    let root = fac.sqrt();
    let shrink = fac.powf((n - 1) as f64);
    let direction = options.step_direction;
    let tolerances = options.tolerances;

    // f(x) is evaluated for every direction, as SciPy's initialisation does: it is the
    // one-sided stencils' first point and the central stencil's zero-weighted middle, and a
    // non-finite f(x) makes the estimate NaN through that zero weight as in SciPy.
    let mut stored = vec![f(x)];
    let mut nfev = 1;
    let mut h = options.initial_step;
    let mut df = f64::NAN;
    let mut error = f64::NAN;
    let mut step = h;
    for iteration in 0..options.maxiter {
        let first = iteration == 0;
        let offsets: Vec<f64> = match (direction, first) {
            (StepDirection::Central, true) => {
                let steps: Vec<f64> = (0..n).map(|k| h / fac.powf(k as f64)).collect();
                steps
                    .iter()
                    .rev()
                    .map(|&v| -v)
                    .chain(steps.iter().copied())
                    .collect()
            }
            (StepDirection::Central, false) => vec![-h / shrink, h / shrink],
            (_, true) => (0..2 * n).map(|k| h / root.powf(k as f64)).collect(),
            (_, false) => vec![h / shrink, h / root / shrink],
        };
        let values: Vec<f64> = offsets
            .iter()
            .map(|&offset| match direction {
                StepDirection::Backward => f(x - offset),
                StepDirection::Central | StepDirection::Forward => f(x + offset),
            })
            .collect();
        nfev += values.len();

        let estimate = if direction == StepDirection::Central {
            // The stored values stay in abscissa order: the new outermost-small pair wraps
            // them. From the second iteration SciPy's middle slot is `work_fc[n]`, which is
            // f(x − h) rather than f(x); its weight is 0 either way, but a non-finite value
            // there still makes the estimate NaN, as it does in SciPy.
            let fresh = if first { n } else { 1 };
            let mut wrapped = Vec::with_capacity(stored.len() + 2 * fresh);
            wrapped.extend_from_slice(&values[..fresh]);
            wrapped.extend_from_slice(&stored);
            wrapped.extend_from_slice(&values[values.len() - fresh..]);
            stored = wrapped;
            let last = stored.len();
            let stencil: Vec<f64> = if first {
                stored.clone()
            } else {
                stored[..=n]
                    .iter()
                    .chain(&stored[last - n..])
                    .copied()
                    .collect()
            };
            dot_product(&stencil, &central_weights) / h
        } else {
            stored.extend_from_slice(&values);
            let last = stored.len();
            let stencil: Vec<f64> = if first {
                stored.clone()
            } else {
                std::iter::once(stored[0])
                    .chain(stored[last - 2 * n..].iter().copied())
                    .collect()
            };
            let one_sided = dot_product(&stencil, &right_weights) / h;
            if direction == StepDirection::Backward {
                -one_sided
            } else {
                one_sided
            }
        };
        let df_last = df;
        df = estimate;
        step = h;
        h /= fac;
        let error_last = error;
        error = (df - df_last).abs();
        let nit = iteration + 1;

        if error < tolerances.atol + tolerances.rtol * df.abs() {
            return Ok(DerivativeResult {
                df,
                error,
                // status: SciPy's convergence test, error < atol + rtol·|df|
                success: true,
                status: DifferentiateStatus::Converged,
                nit,
                nfev,
                step,
            });
        }
        if !(x.is_finite() && df.is_finite()) {
            return Ok(DerivativeResult {
                df: f64::NAN,
                error,
                success: false,
                status: DifferentiateStatus::NonFiniteValue,
                nit,
                nfev,
                step,
            });
        }
        if error > error_last * 10.0 {
            return Ok(DerivativeResult {
                df,
                error,
                success: false,
                status: DifferentiateStatus::ErrorIncreased,
                nit,
                nfev,
                step,
            });
        }
    }

    Ok(DerivativeResult {
        df,
        error,
        success: false,
        status: DifferentiateStatus::MaxIterations,
        nit: options.maxiter,
        nfev,
        step,
    })
}

/// `scipy.differentiate`'s `_derivative_weights` (frankenscipy-3vcjx): central weights over
/// the step multiples `sign(i)/c^(|i|−1)` for `i = −n..=n`, with the middle weight set to 0 and
/// the rest made exactly antisymmetric as SciPy enforces, and right one-sided weights over
/// `0, 1, 1/√c, …, 1/√c^(2n−1)`.
fn derivative_weights(n: usize, fac: f64) -> (Vec<f64>, Vec<f64>) {
    let m = 2 * n + 1;
    let central: Vec<f64> = (0..m)
        .map(|j| {
            let i = j as f64 - n as f64;
            if i == 0.0 {
                0.0
            } else {
                i.signum() / fac.powf(i.abs() - 1.0)
            }
        })
        .collect();
    let mut central_weights = first_derivative_weights(&central);
    central_weights[n] = 0.0;
    for k in 0..n {
        central_weights[m - 1 - k] = -central_weights[k];
    }
    let root = fac.sqrt();
    let right: Vec<f64> = (0..m)
        .map(|j| {
            if j == 0 {
                0.0
            } else {
                1.0 / root.powf(j as f64 - 1.0)
            }
        })
        .collect();
    (central_weights, first_derivative_weights(&right))
}

/// Weights `w` with `Σ_j w_j·t_j^k = [k == 1]` for `k = 0..m−1` over the `m` nodes `t`: SciPy's
/// `np.linalg.solve(np.vander(t, increasing=True).T, e_1)`, here by Gaussian elimination with
/// partial pivoting (LAPACK `gesv`'s algorithm). The nodes are distinct, so the system is
/// nonsingular.
fn first_derivative_weights(nodes: &[f64]) -> Vec<f64> {
    let m = nodes.len();
    // Row k holds t_j^k, built by repeated multiplication as `np.vander` does.
    let mut a = vec![vec![0.0; m]; m];
    for (j, &t) in nodes.iter().enumerate() {
        let mut power = 1.0;
        for row in &mut a {
            row[j] = power;
            power *= t;
        }
    }
    let mut b = vec![0.0; m];
    if m > 1 {
        b[1] = 1.0;
    }
    for col in 0..m {
        let pivot = (col..m).fold(col, |best, row| {
            if a[row][col].abs() > a[best][col].abs() {
                row
            } else {
                best
            }
        });
        a.swap(col, pivot);
        b.swap(col, pivot);
        let inverse = 1.0 / a[col][col];
        for row in col + 1..m {
            let factor = a[row][col] * inverse;
            if factor != 0.0 {
                for k in col..m {
                    a[row][k] -= factor * a[col][k];
                }
                b[row] -= factor * b[col];
            }
        }
    }
    let mut w = vec![0.0; m];
    for col in (0..m).rev() {
        let tail: f64 = (col + 1..m).map(|k| a[col][k] * w[k]).sum();
        w[col] = (b[col] - tail) / a[col][col];
    }
    w
}

fn dot_product(values: &[f64], weights: &[f64]) -> f64 {
    values.iter().zip(weights).map(|(v, w)| v * w).sum()
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
                // status: finite successive-estimate error ≤ atol + rtol·|estimate|
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
        (DifferentiateStatus::NonFiniteValue, _) | (_, DifferentiateStatus::NonFiniteValue) => {
            DifferentiateStatus::NonFiniteValue
        }
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

    static NNLS_MODE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct NnlsModeRestore {
        gram_blocked: bool,
        direct_qr: bool,
    }

    impl NnlsModeRestore {
        fn capture() -> Self {
            use std::sync::atomic::Ordering;

            Self {
                gram_blocked: crate::NNLS_GRAM_BLOCKED.load(Ordering::Relaxed),
                direct_qr: crate::NNLS_DIRECT_QR.load(Ordering::Relaxed),
            }
        }
    }

    impl Drop for NnlsModeRestore {
        fn drop(&mut self) {
            use std::sync::atomic::Ordering;

            crate::NNLS_GRAM_BLOCKED.store(self.gram_blocked, Ordering::Relaxed);
            crate::NNLS_DIRECT_QR.store(self.direct_qr, Ordering::Relaxed);
        }
    }

    use crate::{
        BasinhoppingOptions, Bounds, ConvergenceStatus, DifferentialEvolutionOptions,
        DifferentiateOptions, Integrality, LinearConstraint, LinprogOptions, MilpOptions,
        MilpProblem, MinimizeOptions, NonlinearConstraint, OptError, OptimizeMethod, RootOptions,
        approx_fprime, basinhopping, bracket, brent_minimize, brute, check_grad,
        cholesky_solve_spd, cobyla, derivative, differential_evolution,
        differential_evolution_constrained, dual_annealing, fixed_point, fixed_point_many, golden,
        gradient_descent, hessian, isotonic_regression, jacobian, linear_sum_assignment, linprog,
        milp, minimize_scalar_bounded, minimize_trisection, nnls, numerical_gradient,
        numerical_hessian, numerical_jacobian, projected_gradient_descent, pso,
        quadratic_assignment, rosen, rosen_der, rosen_hess, rosen_hess_prod, shgo,
        transpose_matrix,
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

    /// frankenscipy-vfs3g: `lsq_linear` formed AᵀA from A as given, so A = [[1e160], [-1e160]]
    /// overflowed it and x came back [NaN] unbounded and [0] on [0, 1]. SciPy 1.17.1 (bvls and
    /// trf, both through lstsq on A), b = [1e160, 0]: x = 0.5 (status 3) unbounded and on [0, 1];
    /// bvls x = 0.2 on (-inf, 0.2]. The mirror A = [[1e-170], [-1e-170]], b = [1e-170, 0]
    /// underflowed AᵀA to 0 (x = [0]); SciPy: 0.49999999999999994. Two columns,
    /// A = [[1e160, 0], [-1e160, 1e160], [0, 2e160]], b = [1e160, 0, 3e160]: SciPy
    /// [1.2222222222222225, 1.4444444444444449] unbounded, bvls [0.3, 1.0] on [0, 0.3]×[0, 1].
    #[test]
    fn lsq_linear_scales_an_overflowing_gram_like_scipy() -> Result<(), crate::OptError> {
        use crate::lsq_linear;
        let inf = f64::INFINITY;
        let close = |got: &[f64], want: &[f64]| {
            got.len() == want.len() && got.iter().zip(want).all(|(g, w)| (g - w).abs() <= 1e-15)
        };
        let huge = [vec![1.0e160], vec![-1.0e160]];
        for (lb, ub, want) in [(-inf, inf, 0.5), (0.0, 1.0, 0.5), (-inf, 0.2, 0.2)] {
            let x = lsq_linear(&huge, &[1.0e160, 0.0], &[lb], &[ub])?;
            assert!(
                close(&x, &[want]),
                "bounds [{lb}, {ub}]: {x:?}, SciPy {want}"
            );
        }
        let tiny = [vec![1.0e-170], vec![-1.0e-170]];
        let x = lsq_linear(&tiny, &[1.0e-170, 0.0], &[0.0], &[1.0])?;
        assert!(close(&x, &[0.49999999999999994]), "{x:?}");
        let two = [
            vec![1.0e160, 0.0],
            vec![-1.0e160, 1.0e160],
            vec![0.0, 2.0e160],
        ];
        let b2 = [1.0e160, 0.0, 3.0e160];
        let x = lsq_linear(&two, &b2, &[-inf, -inf], &[inf, inf])?;
        assert!(
            close(&x, &[1.2222222222222225, 1.4444444444444449]),
            "{x:?}"
        );
        let x = lsq_linear(&two, &b2, &[0.0, 0.0], &[0.3, 1.0])?;
        assert!(close(&x, &[0.3, 1.0]), "{x:?}");

        // Must not change: at unit scale A enters as given (SciPy 0.4999999999999998).
        let x = lsq_linear(&[vec![1.0], vec![-1.0]], &[1.0, 0.0], &[-inf], &[inf])?;
        assert!(close(&x, &[0.4999999999999998]), "{x:?}");
        Ok(())
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

        // Running out of iterations is SciPy's NoConvergence, not an evaluation-budget error.
        let no_root = |x: &[f64]| vec![x[0] * x[0] + 1.0];
        for outcome in [
            diagbroyden(no_root, &[1.0], None, 5),
            linearmixing(no_root, &[1.0], None, 5),
            crate::excitingmixing(no_root, &[1.0], None, 1.0, 5),
        ] {
            assert!(
                matches!(outcome, Err(OptError::NoConvergence { .. })),
                "{outcome:?}"
            );
        }
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
        use crate::{FminCobylaOptions, fmin_cobyla};
        // min x0^2 + x1^2 s.t. x0 >= 1; scipy fmin_cobyla -> ~(1, 0).
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let cons: Vec<fn(&[f64]) -> f64> = vec![|x: &[f64]| x[0] - 1.0];
        let opts = FminCobylaOptions {
            maxfun: 2000,
            ..FminCobylaOptions::default()
        };
        let xopt = fmin_cobyla(f, &[3.0, 2.0], &cons, opts).expect("fmin_cobyla");
        assert!((xopt[0] - 1.0).abs() < 1e-3, "x0 = {}", xopt[0]);
        assert!(xopt[1].abs() < 1e-3, "x1 = {}", xopt[1]);
        assert!(cobyla(f, &[], &cons, opts).is_err());
        assert!(cobyla(f, &[f64::NAN, 2.0], &cons, opts).is_err());
        let no_budget = FminCobylaOptions { maxfun: 0, ..opts };
        assert!(cobyla(f, &[3.0, 2.0], &cons, no_budget).is_err());
        let bad_rhoend = FminCobylaOptions {
            rhoend: f64::INFINITY,
            ..opts
        };
        assert!(fmin_cobyla(f, &[3.0, 2.0], &cons, bad_rhoend).is_err());
    }

    #[test]
    fn augmented_lagrangian_rejects_zero_iteration_budgets() {
        use crate::augmented_lagrangian;

        let f = |x: &[f64]| x[0] * x[0];
        let constraints = |x: &[f64]| vec![x[0] + 1.0];
        assert!(augmented_lagrangian(f, constraints, &[0.0], 1, 0, 10).is_err());
        assert!(augmented_lagrangian(f, constraints, &[0.0], 1, 10, 0).is_err());
    }

    // frankenscipy-szq1n.7: feasibility alone was treated as convergence, so a feasible x0 with a
    // one-iteration inner budget was reported as the optimum of (x-3)^2 under x <= 5.
    #[test]
    fn augmented_lagrangian_needs_a_converged_subproblem_not_just_feasibility() {
        use crate::augmented_lagrangian;

        let f = |x: &[f64]| (x[0] - 3.0).powi(2);
        let inactive = |x: &[f64]| vec![5.0 - x[0]];
        let starved = augmented_lagrangian(f, inactive, &[0.0], 1, 1, 1).expect("runs");
        println!(
            "starved AL: x={:?} fun={:?} success={}",
            starved.x, starved.fun, starved.success
        );
        assert!(
            !starved.success,
            "x={:?} is feasible but not optimal",
            starved.x
        );
        let solved = augmented_lagrangian(f, inactive, &[0.0], 1, 20, 1000).expect("runs");
        assert!(solved.success, "{}", solved.message);
        assert!((solved.x[0] - 3.0).abs() < 1e-3, "x={:?}", solved.x);
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

    /// d/dx x² at 0 with forward steps. scipy 1.17.1 `derivative(lambda t: t*t, 0.0,
    /// step_direction=1)` gives df 4.119968255444917e-18, status −2 (maxiter), success False,
    /// nit 10, nfev 27: with df ≈ 0 the default atol (the smallest normal) never admits
    /// convergence. (This test used to assert success, which SciPy does not report here.)
    #[test]
    fn differentiate_derivative_supports_one_sided_steps() {
        let options = crate::DifferentiateOptions {
            step_direction: crate::StepDirection::Forward,
            ..crate::DifferentiateOptions::default()
        };
        let result = derivative(|x| x * x, 0.0, options).expect("forward derivative");
        assert_eq!(result.status, crate::DifferentiateStatus::MaxIterations);
        assert!(!result.success);
        assert_eq!((result.nit, result.nfev), (10, 27));
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

    /// A non-finite x or function value is SciPy's status −3, not an error: scipy 1.17.1
    /// `derivative(lambda t: t, nan)` and `derivative(np.log, 0.5)` (x − h = 0) both give
    /// df nan, status −3, success False, nit 1, nfev 9. (This test used to expect Err.)
    #[test]
    fn differentiate_reports_non_finite_values_like_scipy() {
        for (label, x, f) in [
            ("identity at NaN", f64::NAN, (|t: f64| t) as fn(f64) -> f64),
            ("log at 0.5", 0.5, f64::ln),
        ] {
            let result = derivative(f, x, crate::DifferentiateOptions::default())
                .expect("a non-finite value is a status, not an error");
            assert_eq!(
                result.status,
                crate::DifferentiateStatus::NonFiniteValue,
                "{label}"
            );
            assert!(!result.success && result.df.is_nan(), "{label}: {result:?}");
            assert_eq!((result.nit, result.nfev), (1, 9), "{label}");
        }
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
    fn linear_sum_assignment_handles_forbidden_edges_in_both_directions() {
        let inf = f64::INFINITY;
        // The second row can only use column 0. Its augmenting path must
        // displace row 0 to column 1, even though column 2 is free.
        let cost = vec![vec![0.0, 3.0, inf], vec![1.0, inf, inf]];
        assert_eq!(
            linear_sum_assignment(&cost).unwrap(),
            (vec![0, 1], vec![1, 0])
        );

        let weight = vec![vec![4.0, 6.0], vec![1.0, -inf], vec![2.0, -inf]];
        assert_eq!(
            super::linear_sum_assignment_with_maximize(&weight, true).unwrap(),
            (vec![0, 2], vec![1, 0])
        );
        assert_eq!(
            super::linear_sum_assignment_with_maximize(&[vec![1.0, 3.0]], true).unwrap(),
            (vec![0], vec![1])
        );
    }

    #[test]
    fn linear_sum_assignment_reports_infeasibility_and_bad_inputs() {
        let inf = f64::INFINITY;
        // Each row and column has a candidate, but rows 0 and 1 both
        // require column 0.
        let cost = vec![
            vec![1.0, inf, inf],
            vec![0.0, inf, inf],
            vec![inf, 1.0, 2.0],
        ];
        assert_eq!(
            linear_sum_assignment(&cost).unwrap_err().to_string(),
            "cost matrix is infeasible"
        );
        let invalid = [(f64::NAN, false), (f64::NAN, true), (-inf, false), (inf, true)];
        for (value, maximize) in invalid {
            assert!(matches!(
                super::linear_sum_assignment_with_maximize(&[vec![value]], maximize),
                Err(crate::OptError::NonFiniteInput { .. })
            ));
        }
        assert!(linear_sum_assignment(&[vec![], vec![1.0]]).is_err());
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

    /// The blocked Gram accumulation must produce EXACTLY the solution the rank-1 sweep
    /// produces.
    ///
    /// Bit equality is the right assertion: blocking changes only the ORDER IN WHICH
    /// INDEPENDENT Gram entries are visited, never the order of the `m` products summed
    /// into any one entry, and separate entries never interact. Any difference means a tile
    /// boundary dropped or double-counted a term. A tolerance would hide exactly that,
    /// because NNLS is stable enough that a slightly wrong Gram still yields a
    /// plausible-looking solution.
    ///
    /// `n` deliberately straddles `NNLS_GRAM_BLOCK` (64) — below it there is a single tile
    /// and the test proves nothing about tiling, so sizes just above and well above are
    /// what matter, including ones that leave a RAGGED final tile in both index directions.
    /// The diagonal tile is the delicate one, since it is the only place the upper-triangle
    /// bound `j2 >= j1` cuts inside a block rather than between blocks.
    #[test]
    fn nnls_gram_blocked_matches_rank1_sweep_bits() {
        use std::sync::atomic::Ordering;
        let _mode_lock = NNLS_MODE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _restore = NnlsModeRestore::capture();
        crate::NNLS_DIRECT_QR.store(false, Ordering::Relaxed);
        let value = |i: usize, salt: usize| -> f64 {
            let k = (i * 2_654_435_761usize).wrapping_add(salt * 40_503) % 100_003;
            k as f64 / 100_003.0
        };
        let mut compared = 0usize;
        for &(m, n) in &[
            (12usize, 5usize),
            (40, 64),
            (40, 65),
            (90, 100),
            (70, 128),
            (70, 129),
        ] {
            let a: Vec<Vec<f64>> = (0..m)
                .map(|i| (0..n).map(|j| value(i * n + j, 7)).collect())
                .collect();
            let b: Vec<f64> = (0..m).map(|i| value(i, 29)).collect();

            crate::NNLS_GRAM_BLOCKED.store(false, Ordering::Relaxed);
            let (x_ref, r_ref) = nnls(&a, &b).expect("rank-1 sweep");
            crate::NNLS_GRAM_BLOCKED.store(true, Ordering::Relaxed);
            let (x_new, r_new) = nnls(&a, &b).expect("blocked");

            assert_eq!(x_ref.len(), x_new.len(), "m={m} n={n}: length");
            for (idx, (want, got)) in x_ref.iter().zip(x_new.iter()).enumerate() {
                assert_eq!(
                    want.to_bits(),
                    got.to_bits(),
                    "m={m} n={n}: x[{idx}] differs ({want} vs {got})"
                );
            }
            assert_eq!(
                r_ref.to_bits(),
                r_new.to_bits(),
                "m={m} n={n}: residual norm differs ({r_ref} vs {r_new})"
            );
            compared += 1;
        }
        // Guards a silently-empty comparison: an early return in `nnls` would skip every
        // assertion above while the test still reported success.
        assert_eq!(compared, 6, "expected 6 compared cases, ran {compared}");
    }

    #[test]
    fn nnls_direct_qr_matches_normal_equations_and_scipy_pin() {
        use std::sync::atomic::Ordering;

        let _mode_lock = NNLS_MODE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _restore = NnlsModeRestore::capture();
        let cases = [
            (
                vec![
                    vec![1.0, 0.0, 1.0],
                    vec![1.0, 1.0, 0.0],
                    vec![0.0, 1.0, 1.0],
                    vec![2.0, 1.0, 1.0],
                ],
                vec![1.0, 2.0, 1.5, 3.0],
            ),
            (
                vec![
                    vec![1.0, 0.2, 0.7],
                    vec![0.1, 1.0, 0.3],
                    vec![0.4, 0.6, 1.0],
                    vec![0.9, 0.3, 0.2],
                    vec![0.2, 0.8, 0.5],
                ],
                vec![0.7, 0.4, 1.2, 0.5, 0.9],
            ),
        ];

        for (a, b) in cases {
            crate::NNLS_DIRECT_QR.store(false, Ordering::Relaxed);
            let (normal_x, normal_residual) = nnls(&a, &b).expect("normal-equations arm");
            crate::NNLS_DIRECT_QR.store(true, Ordering::Relaxed);
            let (qr_x, qr_residual) = nnls(&a, &b).expect("direct QR arm");
            for (index, (&normal, &qr)) in normal_x.iter().zip(qr_x.iter()).enumerate() {
                assert!(
                    (normal - qr).abs() <= 1.0e-9,
                    "coefficient {index}: normal={normal:.17e} qr={qr:.17e}"
                );
                assert!(qr >= -1.0e-12, "coefficient {index} is negative: {qr:.17e}");
            }
            assert!(
                (normal_residual - qr_residual).abs() <= 1.0e-9,
                "residual: normal={normal_residual:.17e} qr={qr_residual:.17e}"
            );
        }
    }

    #[test]
    fn nnls_direct_qr_matches_normal_equations_on_perf_fixture() {
        use std::sync::atomic::Ordering;

        let _mode_lock = NNLS_MODE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _restore = NnlsModeRestore::capture();

        let n = 256;
        let unit = |i: usize, salt: usize| -> f64 {
            let k = (i * 2_654_435_761usize).wrapping_add(salt * 40_503) % 1_000_003;
            k as f64 / 1_000_003.0
        };
        let a: Vec<Vec<f64>> = (0..n * n)
            .map(|i| 0.05 + unit(i, 7))
            .collect::<Vec<_>>()
            .chunks_exact(n)
            .map(<[f64]>::to_vec)
            .collect();
        let b: Vec<f64> = (0..n).map(|i| 0.05 + unit(i, 29)).collect();
        crate::NNLS_DIRECT_QR.store(false, Ordering::Relaxed);
        let (normal_x, normal_residual) = nnls(&a, &b).expect("normal-equations arm");
        crate::NNLS_DIRECT_QR.store(true, Ordering::Relaxed);
        let (qr_x, qr_residual) = nnls(&a, &b).expect("direct QR arm");

        for (index, (&normal, &qr)) in normal_x.iter().zip(qr_x.iter()).enumerate() {
            assert!(
                (normal - qr).abs() <= 1.0e-8,
                "coefficient {index}: normal={normal:.17e} qr={qr:.17e}"
            );
        }
        assert!(
            (normal_residual - qr_residual).abs() <= 1.0e-8,
            "residual: normal={normal_residual:.17e} qr={qr_residual:.17e}"
        );
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
        let result = linprog(&c, &a_ub, &b_ub, &[], &[], &[], LinprogOptions::default()).unwrap();
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
        let result = linprog(&c, &[], &[], &a_eq, &b_eq, &[], LinprogOptions::default()).unwrap();
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
        let result = linprog(&c, &[], &[], &a_eq, &b_eq, &[], LinprogOptions::default()).unwrap();
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
        let result = linprog(
            &c,
            &a_ub,
            &b_ub,
            &[],
            &[],
            &bounds,
            LinprogOptions::default(),
        )
        .unwrap();
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
        let result = linprog(&c, &a_ub, &b_ub, &[], &[], &[], LinprogOptions::default()).unwrap();
        assert!(result.success, "linprog failed: {}", result.message);
        assert!(result.x[0].abs() < 0.01, "x={}, expected 0.0", result.x[0]);
    }

    #[test]
    fn linprog_slack_variables() {
        // Verify slack is computed correctly.
        let c = vec![1.0, 1.0];
        let a_ub = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b_ub = vec![10.0, 10.0];
        let result = linprog(&c, &a_ub, &b_ub, &[], &[], &[], LinprogOptions::default()).unwrap();
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
        let result = linprog(&c, &[], &[], &[], &[], &bounds, LinprogOptions::default()).unwrap();
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
        let result = linprog(
            &c,
            &a_ub,
            &b_ub,
            &[],
            &[],
            &bounds,
            LinprogOptions::default(),
        )
        .unwrap();
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
        let result = linprog(&c, &[], &[], &[], &[], &bounds, LinprogOptions::default()).unwrap();
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
        assert!(linprog(&c, &a_ub, &b_ub, &[], &[], &[], LinprogOptions::default()).is_err());
    }

    #[test]
    fn linprog_rejects_non_finite_coefficients() {
        let options = LinprogOptions::default();
        let err = linprog(&[f64::NAN], &[], &[], &[], &[], &[], options)
            .expect_err("NaN objective should fail");
        assert!(matches!(err, crate::OptError::NonFiniteInput { .. }));

        let err = linprog(
            &[1.0],
            &[vec![f64::INFINITY]],
            &[1.0],
            &[],
            &[],
            &[],
            options,
        )
        .expect_err("Inf inequality coefficient should fail");
        assert!(matches!(err, crate::OptError::NonFiniteInput { .. }));

        let err = linprog(&[1.0], &[], &[], &[vec![1.0]], &[f64::NAN], &[], options)
            .expect_err("NaN equality RHS should fail");
        assert!(matches!(err, crate::OptError::NonFiniteInput { .. }));
    }

    // SciPy 1.17.1 answers `options={'maxiter': 0}` on an LP that needs an iteration with
    // status 1 ("Iteration limit reached. (HiGHS Status 14: ...)") rather than an exception.
    #[test]
    fn linprog_zero_iteration_budget_is_an_iteration_limit() {
        let c = vec![-5.0, -4.0];
        let a_ub = vec![vec![6.0, 4.0], vec![1.0, 2.0]];
        let b_ub = vec![24.0, 6.0];
        let result = linprog(
            &c,
            &a_ub,
            &b_ub,
            &[],
            &[],
            &[],
            LinprogOptions {
                maxiter: Some(0),
                ..LinprogOptions::default()
            },
        )
        .expect("a zero budget is a result, not an error");
        assert_eq!(result.status, 1, "{}", result.message);
        assert!(!result.success);
        assert_eq!(result.nit, 0);
        assert!(result.x.is_empty());
    }

    // ── milp tests ──────────────────────────────────────────────────

    #[test]
    fn milp_preserves_global_bounds_and_gap_at_node_limits() {
        for max_nodes in [1, 2, 3] {
            let result = one_variable_milp(
                Integrality::SemiContinuous,
                1.0,
                (Some(2.0), Some(5.0)),
                Some(1.0),
                None,
                MilpOptions {
                    max_nodes: Some(max_nodes),
                    ..Default::default()
                },
            );
            assert!(result.mip_dual_bound <= 2.0, "{result:?}");
            if max_nodes < 3 {
                assert_eq!(result.status, 1, "{result:?}");
                assert!(!result.success);
                assert!((result.mip_dual_bound - 1.0).abs() < 1e-13, "{result:?}");
            }
            match max_nodes {
                1 => {
                    assert!(result.x.is_empty());
                    assert!(result.fun.is_nan());
                    assert!(result.mip_gap.is_infinite());
                }
                2 => {
                    assert_eq!(result.x, vec![2.0]);
                    assert!((result.mip_gap - 0.5).abs() < 1e-13, "{result:?}");
                }
                _ => {
                    assert!(result.success, "{result:?}");
                    assert_eq!(result.mip_dual_bound, result.fun);
                    assert_eq!(result.mip_gap, 0.0);
                }
            }
        }
    }

    #[test]
    fn milp_requested_gap_stops_with_quantified_incumbent() {
        for options in [
            MilpOptions {
                mip_rel_gap: 0.51,
                ..Default::default()
            },
            MilpOptions {
                mip_abs_gap: 1.01,
                ..Default::default()
            },
        ] {
            let result = one_variable_milp(
                Integrality::SemiContinuous,
                1.0,
                (Some(2.0), Some(5.0)),
                Some(1.0),
                None,
                options,
            );
            assert!(result.success, "{result:?}");
            assert_eq!(result.mip_node_count, 2);
            assert_eq!(result.fun, 2.0);
            assert!(
                result.mip_gap > 0.49 && result.mip_gap <= 0.51,
                "{result:?}"
            );
        }
        let zero = milp(
            MilpProblem {
                c: &[-1.0, 1.0],
                integrality: &[Integrality::Integer, Integrality::Continuous],
                a_ub: &[vec![2.0, -1.0]],
                b_ub: &[1.0],
                a_eq: &[],
                b_eq: &[],
                bounds: &[(Some(0.0), Some(1.0)), (Some(0.0), Some(2.0))],
            },
            MilpOptions {
                mip_abs_gap: 0.51,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(zero.success, "{zero:?}");
        assert_eq!(zero.fun, 0.0);
        assert!(zero.mip_gap.is_infinite());
        assert!((zero.mip_dual_bound + 0.5).abs() < 1e-13);
        assert_eq!(zero.mip_node_count, 2);
    }

    #[test]
    fn milp_unbounded_relaxation_requires_integer_feasibility() {
        for fixed_integer in [0.5, 1.0] {
            let result = milp(
                MilpProblem {
                    c: &[0.0, -1.0],
                    integrality: &[Integrality::Integer, Integrality::Continuous],
                    a_ub: &[],
                    b_ub: &[],
                    a_eq: &[vec![1.0, 0.0]],
                    b_eq: &[fixed_integer],
                    bounds: &[(Some(0.0), Some(1.0)), (Some(0.0), None)],
                },
                MilpOptions::default(),
            )
            .unwrap();
            assert!(!result.success, "{result:?}");
            assert_eq!(
                result.status,
                if fixed_integer == 0.5 { 2 } else { 3 },
                "{result:?}"
            );
            assert!(result.x.is_empty());
            assert!(result.fun.is_nan());
            assert_eq!(
                result.mip_dual_bound,
                if fixed_integer == 0.5 {
                    f64::INFINITY
                } else {
                    f64::NEG_INFINITY
                }
            );
        }
    }

    #[test]
    fn milp_does_not_prune_a_better_incumbent_with_an_absolute_epsilon() {
        let result = milp(
            MilpProblem {
                c: &[-1.0, -1.000_000_000_1],
                integrality: &[Integrality::Integer; 2],
                a_ub: &[vec![1.0, 1.0]],
                b_ub: &[1.5],
                a_eq: &[],
                b_eq: &[],
                bounds: &[(Some(0.0), Some(1.0)); 2],
            },
            MilpOptions::default(),
        )
        .unwrap();
        assert!(result.success, "{result:?}");
        assert_eq!(result.x, vec![0.0, 1.0], "{result:?}");
        assert_eq!(result.fun, -1.000_000_000_1);
    }

    #[test]
    fn milp_gap_controls_and_empty_integer_domains_are_validated() {
        for gap in [-1.0, f64::NAN, f64::INFINITY] {
            for options in [
                MilpOptions {
                    mip_rel_gap: gap,
                    ..Default::default()
                },
                MilpOptions {
                    mip_abs_gap: gap,
                    ..Default::default()
                },
            ] {
                assert!(
                    milp(
                        MilpProblem {
                            c: &[1.0],
                            integrality: &[],
                            a_ub: &[],
                            b_ub: &[],
                            a_eq: &[],
                            b_eq: &[],
                            bounds: &[],
                        },
                        options
                    )
                    .is_err()
                );
            }
        }
        let empty = one_variable_milp(
            Integrality::Integer,
            1.0,
            (Some(0.2), Some(0.8)),
            None,
            None,
            MilpOptions::default(),
        );
        assert_eq!(empty.status, 2);
        assert_eq!(empty.mip_node_count, 0);
        assert!(empty.x.is_empty());
        assert!(empty.fun.is_nan());
        assert_eq!(super::milp_relative_gap(1e308, -1e308), 2.0);
        assert_eq!(super::milp_relative_gap(0.0, 0.0), 0.0);
        assert!(super::milp_relative_gap(0.0, -1.0).is_infinite());
    }

    fn one_variable_milp(
        kind: Integrality,
        cost: f64,
        bounds: (Option<f64>, Option<f64>),
        required: Option<f64>,
        capacity: Option<f64>,
        options: MilpOptions,
    ) -> crate::MilpResult {
        let mut a_ub = Vec::new();
        let mut b_ub = Vec::new();
        if let Some(minimum) = required {
            a_ub.push(vec![-1.0]);
            b_ub.push(-minimum);
        }
        if let Some(maximum) = capacity {
            a_ub.push(vec![1.0]);
            b_ub.push(maximum);
        }
        milp(
            MilpProblem {
                c: &[cost],
                integrality: &[kind],
                a_ub: &a_ub,
                b_ub: &b_ub,
                a_eq: &[],
                b_eq: &[],
                bounds: &[bounds],
            },
            options,
        )
        .expect("one-variable MILP")
    }

    #[test]
    fn milp_semi_variables_branch_across_the_activation_gap() {
        // The ordinary LP relaxation returns x=1. A semi-continuous activity
        // must instead activate at 2.2; its semi-integer counterpart starts at 3.
        let relaxation = one_variable_milp(
            Integrality::Continuous,
            1.0,
            (Some(0.0), Some(5.5)),
            Some(1.0),
            None,
            MilpOptions::default(),
        );
        assert!(relaxation.success);
        assert!((relaxation.x[0] - 1.0).abs() < 1e-9);
        for (kind, expected) in [
            (Integrality::SemiContinuous, 2.2),
            (Integrality::SemiInteger, 3.0),
        ] {
            let result = one_variable_milp(
                kind,
                1.0,
                (Some(2.2), Some(5.5)),
                Some(1.0),
                None,
                MilpOptions::default(),
            );
            assert!(result.success, "{kind:?}: {}", result.message);
            assert_eq!(result.status, 0);
            assert!((result.x[0] - expected).abs() < 1e-9, "{result:?}");
            assert!((result.fun - expected).abs() < 1e-9);
            assert!(result.mip_node_count > 1);
        }
        let integer_active = one_variable_milp(
            Integrality::SemiInteger,
            1.0,
            (Some(2.2), Some(8.7)),
            Some(3.5),
            None,
            MilpOptions::default(),
        );
        assert!(integer_active.success);
        assert_eq!(integer_active.x, vec![4.0]);
    }

    #[test]
    fn milp_semi_variables_allow_zero_and_fractional_active_values() {
        for (kind, active) in [
            (Integrality::SemiContinuous, 5.5),
            (Integrality::SemiInteger, 5.0),
        ] {
            for (cost, capacity, expected) in [
                (1.0, None, 0.0),
                (-1.0, Some(1.0), 0.0),
                (-1.0, None, active),
            ] {
                let result = one_variable_milp(
                    kind,
                    cost,
                    (Some(2.0), Some(5.5)),
                    None,
                    capacity,
                    MilpOptions::default(),
                );
                assert!(result.success, "{result:?}");
                assert!((result.x[0] - expected).abs() < 1e-9, "{result:?}");
                assert!((result.fun - cost * expected).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn milp_semi_bounds_preserve_the_isolated_zero() {
        for kind in [Integrality::SemiContinuous, Integrality::SemiInteger] {
            for bounds in [
                (Some(5.0), Some(2.0)),
                (Some(2.0), Some(-1.0)),
                (Some(0.0), Some(0.0)),
            ] {
                let result =
                    one_variable_milp(kind, -1.0, bounds, None, None, MilpOptions::default());
                assert!(result.success, "{kind:?} {bounds:?}: {result:?}");
                assert_eq!(result.x, vec![0.0]);
            }
            let gap_only = one_variable_milp(
                kind,
                1.0,
                (Some(2.0), Some(5.0)),
                Some(0.5),
                Some(1.5),
                MilpOptions::default(),
            );
            assert!(!gap_only.success);
            assert_eq!(gap_only.status, 2, "{gap_only:?}");
        }
        for (kind, expected) in [
            (Integrality::SemiContinuous, 2.5),
            (Integrality::SemiInteger, 0.0),
        ] {
            let result = one_variable_milp(
                kind,
                -1.0,
                (Some(2.5), Some(2.5)),
                None,
                None,
                MilpOptions::default(),
            );
            assert!(result.success, "{result:?}");
            assert_eq!(result.x, vec![expected]);
        }
        let no_active_integer = one_variable_milp(
            Integrality::SemiInteger,
            1.0,
            (Some(2.2), Some(2.8)),
            Some(0.1),
            None,
            MilpOptions::default(),
        );
        assert!(!no_active_integer.success);
        assert_eq!(no_active_integer.status, 2);
    }

    #[test]
    fn milp_semi_zero_lower_and_infinite_upper_bounds() {
        for (kind, zero_lower, infinite_upper) in [
            (Integrality::SemiContinuous, 0.5, 3.5),
            (Integrality::SemiInteger, 1.0, 4.0),
        ] {
            for (bounds, required, expected) in [
                ((Some(0.0), Some(5.0)), 0.5, zero_lower),
                ((Some(2.0), None), 3.5, infinite_upper),
            ] {
                let result = one_variable_milp(
                    kind,
                    1.0,
                    bounds,
                    Some(required),
                    None,
                    MilpOptions::default(),
                );
                assert!(result.success, "{result:?}");
                assert_eq!(result.x, vec![expected]);
            }
            let default_bounds = milp(
                MilpProblem {
                    c: &[1.0],
                    integrality: &[kind],
                    a_ub: &[vec![-1.0]],
                    b_ub: &[-0.5],
                    a_eq: &[],
                    b_eq: &[],
                    bounds: &[],
                },
                MilpOptions::default(),
            )
            .expect("default semi bounds");
            assert!(default_bounds.success);
            assert_eq!(default_bounds.x, vec![zero_lower]);
        }
    }

    #[test]
    fn milp_combines_semi_and_continuous_activity() {
        // The LP mixes a disallowed 1.5 units of the second activity with
        // 5 units of the first. Activating the second at 3 instead gives the
        // true optimum (3.5, 3, 0), objective 8.
        let result = milp(
            MilpProblem {
                c: &[1.0, 1.5, 3.0],
                integrality: &[
                    Integrality::SemiContinuous,
                    Integrality::SemiInteger,
                    Integrality::Continuous,
                ],
                a_ub: &[],
                b_ub: &[],
                a_eq: &[vec![1.0, 1.0, 1.0]],
                b_eq: &[6.5],
                bounds: &[
                    (Some(2.0), Some(5.0)),
                    (Some(3.0), Some(6.0)),
                    (Some(0.0), Some(10.0)),
                ],
            },
            MilpOptions::default(),
        )
        .expect("mixed semi domains");
        assert!(result.success, "{result:?}");
        for (&actual, expected) in result.x.iter().zip([3.5, 3.0, 0.0]) {
            assert!((actual - expected).abs() < 1e-9, "{result:?}");
        }
        assert!((result.fun - 8.0).abs() < 1e-9);
    }

    #[test]
    fn milp_semi_node_and_lp_limits_never_certify_the_relaxation() {
        for max_nodes in [1, 2] {
            let limited = one_variable_milp(
                Integrality::SemiContinuous,
                1.0,
                (Some(2.0), Some(5.0)),
                Some(1.0),
                None,
                MilpOptions {
                    max_nodes: Some(max_nodes),
                    ..MilpOptions::default()
                },
            );
            assert!(!limited.success, "{limited:?}");
            assert_eq!(limited.status, 1);
            assert_eq!(limited.mip_node_count, max_nodes);
            if max_nodes == 2 {
                assert_eq!(limited.x, vec![2.0]);
                assert_eq!(limited.fun, 2.0);
            }
        }
        let problem = MilpProblem {
            c: &[-5.0, -4.0],
            integrality: &[Integrality::SemiContinuous, Integrality::SemiInteger],
            a_ub: &[vec![6.0, 4.0], vec![1.0, 2.0]],
            b_ub: &[24.0, 6.0],
            a_eq: &[],
            b_eq: &[],
            bounds: &[(Some(2.0), Some(3.0)); 2],
        };
        let limited = milp(
            problem,
            MilpOptions {
                lp_maxiter: Some(1),
                ..MilpOptions::default()
            },
        )
        .expect("limited semi relaxation");
        assert!(!limited.success);
        assert_eq!(limited.status, 1, "{limited:?}");
        let result = milp(problem, MilpOptions::default()).expect("semi optimum");
        assert!(result.success, "{result:?}");
        assert!((result.fun + 18.0).abs() < 1e-9);
    }

    #[test]
    fn milp_semi_bounds_reject_invalid_activation_thresholds() {
        for kind in [Integrality::SemiContinuous, Integrality::SemiInteger] {
            for bounds in [
                (Some(-1.0), Some(5.0)),
                (None, Some(5.0)),
                (Some(f64::INFINITY), Some(5.0)),
                (Some(f64::NAN), Some(5.0)),
                (Some(2.0), Some(f64::NAN)),
                (Some(2.0), Some(f64::NEG_INFINITY)),
            ] {
                let result = milp(
                    MilpProblem {
                        c: &[1.0],
                        integrality: &[kind],
                        a_ub: &[],
                        b_ub: &[],
                        a_eq: &[],
                        b_eq: &[],
                        bounds: &[bounds],
                    },
                    MilpOptions::default(),
                );
                assert!(
                    matches!(result, Err(OptError::InvalidBounds { .. })),
                    "{kind:?} {bounds:?}: {result:?}"
                );
            }
        }
    }

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

    // frankenscipy-szq1n.7: a node LP stopped at its iteration limit was pruned as if infeasible.
    // At the root that turned "LP budget too small" into "Problem is infeasible" (status 2); below
    // the root it let the search finish with status 0 on an unproven incumbent. SciPy reports an
    // iteration limit as status 1. The root LP needs several simplex iterations (its
    // starting vertex (3, 3) violates both rows), so `lp_maxiter = 1` starves it.
    #[test]
    fn milp_lp_iteration_limit_is_not_infeasibility() {
        let c = vec![-5.0, -4.0];
        let a_ub = vec![vec![6.0, 4.0], vec![1.0, 2.0]];
        let b_ub = vec![24.0, 6.0];
        let bounds = vec![(Some(0.0), Some(3.0)), (Some(0.0), Some(3.0))];
        let problem = MilpProblem {
            c: &c,
            integrality: &[Integrality::Integer, Integrality::Integer],
            a_ub: &a_ub,
            b_ub: &b_ub,
            a_eq: &[],
            b_eq: &[],
            bounds: &bounds,
        };
        let starved = milp(
            problem,
            MilpOptions {
                lp_maxiter: Some(1),
                ..MilpOptions::default()
            },
        )
        .expect("milp");
        println!(
            "milp lp_maxiter=1: status={} success={} msg={}",
            starved.status, starved.success, starved.message
        );
        assert!(!starved.success);
        assert_eq!(starved.status, 1, "{}", starved.message);
        let solved = milp(problem, MilpOptions::default()).expect("milp");
        assert_eq!(solved.status, 0, "{}", solved.message);
        // LP relaxation optimum (3, 1.5); the integer optimum is (3, 1) with fun −19.
        assert_eq!(solved.x, vec![3.0, 1.0]);
        assert_eq!(solved.fun, -19.0);
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

    // `max_nodes` (covered above) and `lp_maxiter` are solver budgets. `linprog` follows SciPy and
    // answers maxiter == Some(0) with an iteration-limit result (status 1), so milp
    // rejects a zero LP budget itself. Without that check a zero LP iteration
    // budget would silently become a synthetic result -- exactly the failure mode
    // frankenscipy-qgc24 was filed about for max_nodes.
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
    fn de_small_objective_scale_does_not_cause_false_convergence() {
        // Multiplying an objective by an exact power of two changes neither
        // its minimizer nor the relative population spread. In particular a
        // small objective must not gain an implicit absolute tolerance.
        for scale in [2.0_f64.powi(-500), 1.0, 2.0_f64.powi(500)] {
            let result = differential_evolution(
                |x| scale * (x[0] - 0.25).powi(2),
                &[(-5.0, 5.0)],
                DifferentialEvolutionOptions {
                    seed: Some(7),
                    ..Default::default()
                },
            )
            .expect("scaled objective");
            assert!(result.success, "scale={scale:e}: {}", result.message);
            assert!(
                (result.x[0] - 0.25).abs() < 1e-10,
                "scale={scale:e}: x={:?}, nit={}",
                result.x,
                result.nit
            );
            assert!(
                result.nit > 1,
                "a scattered initial population is not converged"
            );
        }
    }

    #[test]
    fn de_constrained_only_evaluates_the_feasible_objective_domain() {
        let objective_calls = std::cell::Cell::new(0);
        let constraint_calls = std::cell::Cell::new(0);
        let result = differential_evolution_constrained(
            |x| {
                assert!(x[0] >= 1.0, "infeasible objective evaluation at {x:?}");
                objective_calls.set(objective_calls.get() + 1);
                x[0].ln() + 1.0
            },
            &[(-2.0, 2.0)],
            |x| {
                constraint_calls.set(constraint_calls.get() + 1);
                (1.0 - x[0]).max(0.0)
            },
            DifferentialEvolutionOptions {
                seed: Some(7),
                ..Default::default()
            },
        )
        .expect("objective defined only on feasible points");
        assert!(result.success, "{}", result.message);
        assert!((result.x[0] - 1.0).abs() < 1e-6, "{:?}", result.x);
        assert_eq!(result.maxcv, Some(0.0));
        assert_eq!(result.nfev, objective_calls.get());
        assert!(objective_calls.get() > 0);
        assert!(constraint_calls.get() > objective_calls.get());
    }

    #[test]
    fn de_infeasible_population_has_no_objective_evaluations() {
        let result = differential_evolution_constrained(
            |_| panic!("an infeasible problem must not evaluate the objective"),
            &[(0.0, 2.0)],
            |x| (3.0 - x[0]).max(0.0),
            DifferentialEvolutionOptions {
                maxiter: 30,
                seed: Some(5),
                ..Default::default()
            },
        )
        .expect("infeasible problem");
        assert!(!result.success);
        assert_eq!(result.status, ConvergenceStatus::Infeasible);
        assert_eq!(result.nfev, 0);
        assert_eq!(result.fun, Some(f64::INFINITY));
        assert!(result.maxcv.unwrap() > 0.0);
    }

    #[test]
    fn de_nonfinite_feasible_objectives_are_still_counted() {
        for value in [f64::INFINITY, f64::NAN] {
            let calls = std::cell::Cell::new(0);
            let result = differential_evolution_constrained(
                |_| {
                    calls.set(calls.get() + 1);
                    value
                },
                &[(-1.0, 1.0)],
                |_| 0.0,
                DifferentialEvolutionOptions {
                    maxiter: 2,
                    seed: Some(7),
                    ..Default::default()
                },
            )
            .expect("feasible objective evaluations");
            assert!(!result.success);
            assert_eq!(result.nfev, calls.get());
            assert_eq!(result.nfev, 45);
        }
    }

    #[test]
    fn de_population_stopping_uses_standard_deviation_without_range_loss() {
        use crate::de_population_converged;

        // Mean = 2/3, population std = sqrt(8/9): the standard-deviation
        // criterion passes, whereas a maximum-minus-minimum test would fail.
        assert!(de_population_converged(&[0.0, 0.0, 2.0], 1.5, 0.0));
        assert!(!de_population_converged(&[0.0, 0.0, 2.0], 1.0, 0.0));
        assert!(de_population_converged(&[0.0; 15], 0.0, 0.0));
        assert!(de_population_converged(&[f64::MAX; 15], 1e-8, 0.0));
        assert!(!de_population_converged(&[-f64::MAX, f64::MAX], 1e-8, 0.0));
        assert!(de_population_converged(
            &[-f64::MAX, f64::MAX],
            0.0,
            f64::MAX
        ));
        assert!(!de_population_converged(&[1e-300, 2e-300], 0.1, 0.0));
        assert!(de_population_converged(&[1e-300, 2e-300], 0.0, 1e-300));
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(!de_population_converged(&[1.0, invalid], 1.0, 1.0));
        }
    }

    #[test]
    fn de_absolute_tolerance_is_explicit_and_validated_before_callbacks() {
        let result = differential_evolution(
            |x| 1e-20 * (x[0] - 0.25).powi(2),
            &[(-5.0, 5.0)],
            DifferentialEvolutionOptions {
                atol: 1e-12,
                seed: Some(7),
                ..Default::default()
            },
        )
        .expect("absolute tolerance");
        assert!(result.success);
        assert_eq!(result.nit, 1);
        for atol in [-1.0, f64::NAN, f64::INFINITY] {
            let error = differential_evolution(
                |_| panic!("invalid tolerances must not reach the objective"),
                &[(-1.0, 1.0)],
                DifferentialEvolutionOptions {
                    atol,
                    ..Default::default()
                },
            )
            .expect_err("invalid absolute tolerance");
            assert!(matches!(error, OptError::InvalidArgument { .. }));
        }
    }

    #[test]
    fn de_mixed_integer_callbacks_and_solution_are_feasible() {
        let calls = std::cell::Cell::new(0);
        let objective = |x: &[f64]| {
            calls.set(calls.get() + 1);
            assert_eq!(x[0], x[0].round_ties_even());
            assert!((-4.2..=5.8).contains(&x[0]));
            assert_eq!(x[2], 7.0);
            (x[0] - 2.3).powi(2) + (x[1] + 0.4 * x[0]).powi(2)
        };
        for strategy in [crate::DeStrategy::Best1Bin, crate::DeStrategy::Rand1Bin] {
            let result = differential_evolution(
                objective,
                &[(-4.2, 5.8), (-4.0, 4.0), (7.0, 7.0)],
                DifferentialEvolutionOptions {
                    strategy,
                    integrality: vec![true, false, true],
                    seed: Some(713),
                    tol: 1e-12,
                    ..Default::default()
                },
            )
            .expect("mixed integer evolution");
            assert!(result.success, "{}", result.message);
            assert_eq!(result.x[0], 2.0);
            assert_eq!(result.x[2], 7.0);
            assert!((result.x[1] + 0.8).abs() < 2e-6, "{:?}", result.x);
            assert!((result.fun.unwrap() - 0.09).abs() < 1e-11);
        }
        assert!(calls.get() > 0);
    }

    #[test]
    fn de_integer_constraints_receive_the_same_point_as_the_objective() {
        let objective = |x: &[f64]| {
            assert_eq!(x[0].fract(), 0.0);
            x[0] * x[0] + x[1] * x[1]
        };
        let constraint = |x: &[f64]| {
            assert_eq!(x[0].fract(), 0.0);
            (2.2 - x[0]).max(0.0)
        };
        let result = differential_evolution_constrained(
            objective,
            &[(0.2, 6.8), (-2.0, 2.0)],
            constraint,
            DifferentialEvolutionOptions {
                integrality: vec![true, false],
                seed: Some(81),
                tol: 1e-12,
                ..Default::default()
            },
        )
        .expect("integer constraint");
        assert!(result.success);
        assert_eq!(result.x[0], 3.0);
        assert_eq!(result.maxcv, Some(0.0));
        assert!(result.x[1].abs() < 4e-6);
        assert!((result.fun.unwrap() - objective(&result.x)).abs() < 1e-14);
    }

    #[test]
    fn de_integer_singletons_fixed_parameters_and_endpoints_are_exact() {
        for (bounds, integrality, expected) in [
            (vec![(1.2, 2.8)], vec![true], vec![2.0]),
            (vec![(-3.8, -2.2)], vec![true], vec![-3.0]),
            (vec![(0.25, 0.25)], vec![false], vec![0.25]),
            (vec![(0.0, 3.0)], vec![true], vec![0.0]),
            (vec![(-3.0, 0.0)], vec![true], vec![-3.0]),
        ] {
            let result = differential_evolution(
                |x| x[0],
                &bounds,
                DifferentialEvolutionOptions {
                    integrality,
                    seed: Some(52),
                    ..Default::default()
                },
            )
            .expect("fixed or endpoint optimum");
            assert!(result.success);
            assert_eq!(result.x, expected);
            assert_eq!(result.fun, Some(expected[0]));
        }
    }

    #[test]
    fn de_rejects_invalid_integer_domains_before_user_callbacks() {
        for (bounds, integrality) in [
            (vec![(0.0, 1.0); 2], vec![true]),
            (vec![(0.2, 0.8)], vec![true]),
            (vec![(1.5, 1.5)], vec![true]),
            (vec![(3.0, 2.0)], vec![true]),
            (vec![(0.0, f64::INFINITY)], vec![true]),
            (vec![(0.0, 9_007_199_254_740_992.0)], vec![true]),
        ] {
            let error = differential_evolution(
                |_| panic!("invalid bounds must not reach the objective"),
                &bounds,
                DifferentialEvolutionOptions {
                    integrality,
                    ..Default::default()
                },
            )
            .expect_err("invalid integer domain");
            assert!(matches!(
                error,
                OptError::InvalidBounds { .. } | OptError::InvalidArgument { .. }
            ));
        }
    }

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
        // frankenscipy-szq1n.7: a success must be feasible EXACTLY. The quadratic penalty this
        // replaced converged to x = 1 - 1e-6, violation 1e-6, under success = true.
        assert_eq!(result.maxcv, Some(0.0), "x = {:?}", result.x);
        assert!(result.x[0] >= 1.0, "x = {:?}", result.x);
    }

    // frankenscipy-szq1n.7: an infeasible answer is reported as one. x >= 3 cannot be met inside
    // the box [0, 2]; the least-violating point x = 2 is returned with MAXCV 1 and success false.
    #[test]
    fn de_constrained_reports_an_unsatisfiable_constraint() {
        let objective = |x: &[f64]| x[0] * x[0];
        let impossible = |x: &[f64]| (3.0 - x[0]).max(0.0);
        let opts = DifferentialEvolutionOptions {
            maxiter: 100,
            seed: Some(5),
            ..Default::default()
        };
        let result = differential_evolution_constrained(objective, &[(0.0, 2.0)], impossible, opts)
            .expect("runs");
        println!(
            "DE impossible: x={:?} maxcv={:?} success={} msg={}",
            result.x, result.maxcv, result.success, result.message
        );
        assert!(!result.success);
        assert_eq!(result.status, ConvergenceStatus::Infeasible);
        assert!(
            (result.x[0] - 2.0).abs() < 1e-6,
            "least violation is at x = 2"
        );
        assert!((result.maxcv.unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn de_tournament_matches_scipy_accept_trial() {
        use crate::{de_accepts, de_ranks_before};
        // both feasible: objective decides, ties accepted
        assert!(de_accepts((1.0, 0.0), (1.0, 0.0)));
        assert!(!de_accepts((2.0, 0.0), (1.0, 0.0)));
        // feasible trial always beats an infeasible original, whatever the objective
        assert!(de_accepts((1e9, 0.0), (-1e9, 0.5)));
        // infeasible trial never replaces a feasible original
        assert!(!de_accepts((-1e9, 1e-12), (1e9, 0.0)));
        // infeasible vs infeasible: violation decides
        assert!(de_accepts((5.0, 0.1), (0.0, 0.2)));
        assert!(!de_accepts((0.0, 0.3), (5.0, 0.2)));
        assert!(de_ranks_before((0.0, 0.1), (0.0, 0.2)));
        assert!(!de_ranks_before((-5.0, 0.1), (5.0, 0.0)));
    }

    // ── Basinhopping tests ─────────────────────────────────────────

    #[test]
    fn basinhopping_finds_global_minimum() {
        // Multi-modal function: f(x) = x^4 - 4x^2 + x, local minima at x ≈ 1.347 (f ≈ -2.6186)
        // and x ≈ -1.473 (global, f ≈ -5.4442), separated by the maximum at x ≈ 0.126.
        let func = |x: &[f64]| -> f64 {
            let xi = x[0];
            xi.powi(4) - 4.0 * xi * xi + xi
        };
        let run = |stepsize| {
            let opts = BasinhoppingOptions {
                niter: 50,
                seed: Some(42),
                stepsize,
                ..Default::default()
            };
            basinhopping(func, &[3.0], opts).unwrap().fun.unwrap()
        };

        // SciPy 1.17.1 basinhopping(f, [3.0], niter=50, stepsize=2.0) reaches the global minimum
        // (x = -1.472998, f = -5.444192) for seeds 0-3 and 42.
        let fun = run(2.0);
        assert!(
            (fun - -5.444_192).abs() < 1e-5,
            "basinhopping should find global minimum, got f={fun}"
        );

        // With stepsize=1.0 no hop from x ≈ 1.347 crosses the barrier at 0.126, and SciPy stays
        // at the local minimum (f = -2.618556) for every one of those seeds; a local search that
        // overshoots into the other basin is not SciPy's.
        let fun = run(1.0);
        assert!(
            (fun - -2.618_556).abs() < 1e-5,
            "stepsize 1 stays in the start basin as SciPy's does, got f={fun}"
        );
    }

    #[test]
    fn basinhopping_reports_only_successful_local_minima() {
        // Local BFGS at tol 1e-8 by forward differences ends in precision loss in about a third
        // of SciPy's own runs, and those stop LOWER (f ~ 1e-18) than the successful ones
        // (f ~ 1e-17 .. 1e-16). SciPy's Storage keeps only successful minimisations, so
        // basinhopping(sphere, [4, -3], niter=40, minimizer_kwargs={'tol': 1e-8}) reports
        // success for every seed tried (42, 7, 0, 1, 2). Keeping the lowest run regardless
        // reported a failed one here (seed 7).
        let sphere = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let opts = BasinhoppingOptions {
            niter: 40,
            seed: Some(7),
            minimizer_tol: Some(1e-8),
            ..Default::default()
        };
        let result = basinhopping(sphere, &[4.0, -3.0], opts).unwrap();
        assert!(result.success, "{}", result.message);
        assert!(
            result.x.iter().all(|v| v.abs() < 1e-6),
            "x = {:?}",
            result.x
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

    // frankenscipy-szq1n.7: basinhopping reported success = true unconditionally. SciPy uses
    // `lowest_optimization_result.success`. The fixture needs local runs that cannot succeed:
    // any objective with an attainable minimum lets a finite-difference gradient reach exactly
    // 0 (the kink |x - 1/3| + 1 did, BFGS landing on x = 1/3 where the central difference
    // vanishes). A linear objective has a constant gradient of 1 and no stationary point, so
    // every BFGS run ends at its iteration limit (SciPy: success False as well).
    #[test]
    fn basinhopping_success_is_the_lowest_local_minimisations() {
        let linear = |x: &[f64]| -> f64 { x[0] };
        let unreachable = BasinhoppingOptions {
            niter: 5,
            seed: Some(7),
            minimizer_tol: Some(f64::MIN_POSITIVE),
            ..Default::default()
        };
        let failed = basinhopping(linear, &[2.0], unreachable).unwrap();
        println!(
            "basinhopping, unreachable local tol: success={} fun={:?} msg={}",
            failed.success, failed.fun, failed.message
        );
        assert!(!failed.success, "every local minimisation failed");
        assert_ne!(failed.status, ConvergenceStatus::Success);
        // The positive arm: a smooth bowl, default tolerance, every local run converges.
        let bowl = |x: &[f64]| -> f64 { (x[0] - 1.0 / 3.0).powi(2) + 1.0 };
        let fine = BasinhoppingOptions {
            niter: 5,
            seed: Some(7),
            ..Default::default()
        };
        let ok = basinhopping(bowl, &[2.0], fine).unwrap();
        assert!(ok.success, "{}", ok.message);
        assert!((ok.fun.unwrap() - 1.0).abs() < 1e-10);
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

    /// frankenscipy-0v9od: a NaN at the seeded starting point used to stick -- no later `<`
    /// against NaN succeeds -- and came back as `fun = NaN` with `success = true`. SciPy redraws
    /// a non-finite starting energy.
    #[test]
    fn dual_annealing_redraws_a_nan_starting_point() {
        let bounds = [(-10.0, 10.0), (-10.0, 10.0)];
        // A seed whose first draw lands in the NaN region x[0] < -1.
        let seed = (0..1000_u64)
            .find(|&s| {
                let mut rng = crate::SimpleRng::new(s);
                bounds[0].0 + rng.next_f64() * (bounds[0].1 - bounds[0].0) < -1.0
            })
            .expect("a seed starting in the NaN region");
        let objective = |x: &[f64]| {
            if x[0] < -1.0 {
                f64::NAN
            } else {
                (x[0] - 0.5).powi(2) + (x[1] + 0.25).powi(2)
            }
        };
        let result = dual_annealing(objective, &bounds, 50, seed).expect("dual_annealing");
        let fun = result.fun.expect("fun");
        assert!(
            fun.is_finite() && fun < 1e-10,
            "fun = {fun}, x = {:?}",
            result.x
        );
        assert!((result.x[0] - 0.5).abs() < 1e-5 && (result.x[1] + 0.25).abs() < 1e-5);

        // NaN everywhere: SciPy raises after MAX_REINIT_COUNT redraws.
        let err = dual_annealing(|_| f64::NAN, &bounds, 50, seed).expect_err("all NaN");
        assert!(
            matches!(err, crate::OptError::NonFiniteInput { .. }),
            "{err:?}"
        );
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

    // frankenscipy-szq1n.7: an objective finite nowhere on the grid returned x = [0.0] (outside
    // the range) with success = true. It now returns the first grid point, as SciPy's argmin
    // does, and success = false.
    #[test]
    fn brute_with_no_finite_value_is_not_a_success() {
        let nan = crate::brute(|_| f64::NAN, &[(1.0, 2.0)], 3).expect("brute runs");
        assert!(!nan.success);
        assert_eq!(nan.x, vec![1.0]);
        let inf = crate::brute(|_| f64::INFINITY, &[(1.0, 2.0)], 3).expect("brute runs");
        assert!(!inf.success);
        assert_eq!(inf.x, vec![1.0]);
        // A single finite value among NaNs is found.
        let one = crate::brute(
            |x| if x[0] == 1.5 { 4.0 } else { f64::NAN },
            &[(1.0, 2.0)],
            3,
        )
        .expect("brute runs");
        assert!(one.success);
        assert_eq!(one.x, vec![1.5]);
        assert_eq!(one.fun, Some(4.0));
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
    fn isotonic_regression_extreme_weight_scales_preserve_kkt() {
        let y = [3.0, 1.0, 4.0, 0.0, 6.0, 5.0];
        let weights = [1.0, 3.0, 2.0, 2.0, 1.0, 1.0];
        let expected = [1.5, 1.5, 2.0, 2.0, 5.5, 5.5];
        for reversed in [false, true] {
            let (y, weights, expected): (Vec<f64>, Vec<f64>, Vec<f64>) = if reversed {
                (
                    y.iter().rev().map(|value| -value).collect(),
                    weights.iter().rev().copied().collect(),
                    expected.iter().rev().map(|value| -value).collect(),
                )
            } else {
                (y.to_vec(), weights.to_vec(), expected.to_vec())
            };
            for factor in [1e-300, 1e-200, 1.0, 1e200, 1e300] {
                let scaled_weights: Vec<f64> =
                    weights.iter().map(|weight| weight * factor).collect();
                let fit = isotonic_regression(&y, Some(&scaled_weights));
                assert_close(&fit, &expected);
                assert!(fit.iter().all(|value| value.is_finite()));
                assert!(fit.windows(2).all(|pair| pair[0] <= pair[1]));

                // The prefix residuals are the nonnegative KKT multipliers.
                // Use the original weights, since a common positive scale
                // leaves the minimizer and the signs of these conditions intact.
                let mut prefix = 0.0;
                for i in 0..fit.len() {
                    prefix += weights[i] * (y[i] - fit[i]);
                    assert!(prefix >= -2e-13, "factor {factor:e}: {fit:?}");
                    if i + 1 < fit.len() {
                        assert!(
                            (prefix * (fit[i + 1] - fit[i])).abs() <= 2e-13,
                            "complementarity at {i}, factor {factor:e}: {fit:?}"
                        );
                    }
                }
                assert!(prefix.abs() <= 2e-13, "factor {factor:e}: {fit:?}");
            }
        }
    }

    #[test]
    fn isotonic_regression_finite_near_maximum_responses() {
        let largest = f64::MAX;
        for weights in [None, Some([largest, largest])] {
            let fit = isotonic_regression(
                &[largest, 0.5 * largest],
                weights.as_ref().map(|values| values.as_slice()),
            );
            assert!(fit.iter().all(|value| value.is_finite()));
            for value in fit {
                assert!((value / largest - 0.75).abs() <= 2.0 * f64::EPSILON);
            }
        }
        assert_eq!(
            isotonic_regression(&[largest; 8], Some(&[largest; 8])),
            vec![largest; 8]
        );

        let y = [-largest, -1e-300, 0.0, 1e-300, largest];
        let weights = [
            largest,
            f64::from_bits(1),
            1.0,
            f64::MIN_POSITIVE,
            largest,
        ];
        assert_eq!(isotonic_regression(&y, Some(&weights)), y.to_vec());
    }

    #[test]
    fn isotonic_regression_retains_tiny_positive_weight_effects() {
        let large = f64::from_bits(2046_u64 << 52);
        for tiny in [f64::MIN_POSITIVE, f64::from_bits(1)] {
            // Normalizing the weights by their maximum would turn tiny into
            // zero, although the pooled mean tiny*large/(tiny+large) is tiny.
            assert_eq!(tiny / large, 0.0);
            let fit = isotonic_regression(&[large, 0.0], Some(&[tiny, large]));
            assert_eq!(fit, vec![tiny; 2]);
        }
    }

    #[test]
    fn isotonic_regression_preserves_cancellation_across_merges() {
        // The middle observation must survive the first pool with f64::MAX
        // and the later cancellation against -f64::MAX.
        let fit = isotonic_regression(&[f64::MAX, 1.0, -f64::MAX], None);
        assert_eq!(fit, vec![1.0 / 3.0; 3]);
        let fit = isotonic_regression(&[f64::MAX, -1.0, -f64::MAX], None);
        assert_eq!(fit, vec![-1.0 / 3.0; 3]);
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
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                isotonic_regression(&[1.0, 2.0], Some(&[1.0, invalid]))
                    .iter()
                    .all(|value| value.is_nan())
            );
            assert!(
                isotonic_regression(&[1.0, invalid], None)
                    .iter()
                    .all(|value| value.is_nan())
            );
        }
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
    fn approx_fprime_uses_actual_representable_steps() {
        for x in [1.0, -1.0, 1e16, -1e16, 1e100, -1e100] {
            let gradient = approx_fprime(&[x], |point| point[0], 1e-8).unwrap();
            assert_eq!(gradient, vec![1.0], "identity gradient at {x}");
            let discrepancy = crate::check_grad(|point| point[0], |_| vec![1.0], &[x]).unwrap();
            assert_eq!(discrepancy, 0.0, "identity gradient check at {x}");
        }
    }

    #[test]
    fn approx_fprime_preserves_finite_extreme_differences() {
        for x in [f64::MAX, -f64::MAX] {
            let gradient = approx_fprime(
                &[x],
                |point| {
                    assert!(point[0].is_finite(), "callback received an infinite coordinate");
                    point[0] - x
                },
                1e-8,
            )
            .unwrap();
            assert_eq!(gradient, vec![1.0]);
        }

        let center = 2.0_f64.powi(1000);
        let step = 2.0_f64.powi(974);
        let offset = 2.0_f64.powi(1022);
        let half_rate = 2.0_f64.powi(49);
        let gradient = approx_fprime(
            &[center],
            |point| {
                let value = 2.0 * (offset - half_rate * (point[0] - center));
                assert!(value.is_finite());
                value
            },
            step,
        )
        .unwrap();
        assert_eq!(gradient, vec![-2.0_f64.powi(50)]);
    }

    #[test]
    fn approx_derivative_rejects_invalid_controls_before_evaluation() {
        use crate::{FiniteDiffMethod, approx_derivative};
        for method in [FiniteDiffMethod::TwoPoint, FiniteDiffMethod::ThreePoint] {
            for (relative, absolute) in [
                (Some(f64::NAN), None),
                (Some(f64::INFINITY), None),
                (None, Some(f64::NAN)),
                (None, Some(f64::NEG_INFINITY)),
            ] {
                let result = approx_derivative(
                    |_| panic!("invalid controls must not evaluate the callback"),
                    &[0.0],
                    method,
                    relative,
                    absolute,
                    None,
                );
                assert!(matches!(result, Err(crate::OptError::InvalidArgument { .. })));
            }
            for (lower, upper) in [
                (f64::NAN, 1.0),
                (-1.0, f64::NAN),
                (1.0, -1.0),
                (0.0, 0.0),
            ] {
                let result = approx_derivative(
                    |_| panic!("invalid bounds must not evaluate the callback"),
                    &[0.0],
                    method,
                    None,
                    None,
                    Some((&[lower], &[upper])),
                );
                assert!(matches!(result, Err(crate::OptError::InvalidArgument { .. })));
            }
        }
    }

    #[test]
    fn approx_derivative_rejects_changed_shapes_and_nonfinite_samples() {
        use crate::{FiniteDiffMethod, approx_derivative};
        for method in [FiniteDiffMethod::TwoPoint, FiniteDiffMethod::ThreePoint] {
            for dimensions in [1, 2] {
                for length in [0, 2] {
                    let result = approx_derivative(
                        |point| {
                            if point.iter().all(|value| *value == 0.0) {
                                vec![0.0]
                            } else {
                                vec![0.0; length]
                            }
                        },
                        &vec![0.0; dimensions],
                        method,
                        None,
                        None,
                        None,
                    );
                    assert!(matches!(result, Err(crate::OptError::InvalidArgument { .. })));
                }
            }
            for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                for at_base in [false, true] {
                    let result = approx_derivative(
                        |point| {
                            vec![if (point[0] == 0.0) == at_base {
                                invalid
                            } else {
                                point[0]
                            }]
                        },
                        &[0.0],
                        method,
                        None,
                        None,
                        None,
                    );
                    assert!(matches!(result, Err(crate::OptError::NonFiniteInput { .. })));
                }
            }
        }
    }

    #[test]
    fn approx_derivative_keeps_finite_coordinates_and_large_offsets() {
        use crate::{FiniteDiffMethod, approx_derivative};
        for method in [FiniteDiffMethod::TwoPoint, FiniteDiffMethod::ThreePoint] {
            for x in [f64::MAX, -f64::MAX] {
                let jacobian = approx_derivative(
                    |point| {
                        assert!(point[0].is_finite(), "callback received an infinite coordinate");
                        vec![point[0]]
                    },
                    &[x],
                    method,
                    None,
                    None,
                    None,
                )
                .unwrap();
                assert_eq!(jacobian, vec![vec![1.0]], "{method:?} at {x}");
            }
        }
        // The ordinary one-sided formula overflows its weighted samples,
        // despite the exact derivative of this constant being zero.
        let constant = approx_derivative(
            |_| vec![1e308, -1e308],
            &[0.0],
            FiniteDiffMethod::ThreePoint,
            None,
            None,
            Some((&[0.0], &[f64::INFINITY])),
        )
        .unwrap();
        assert_eq!(constant, vec![vec![0.0], vec![0.0]]);

        // Removing a large common offset before weighting retains its slope.
        let linear = approx_derivative(
            |point| vec![1e308 * (1.0 + point[0])],
            &[0.0],
            FiniteDiffMethod::ThreePoint,
            None,
            Some(1e-4),
            Some((&[0.0], &[f64::INFINITY])),
        )
        .unwrap();
        assert!((linear[0][0] / 1e308 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn approx_derivative_retains_cancellation_before_tiny_step_division() {
        let jacobian = crate::approx_derivative(
            |point| vec![2.0 * (point[0] / 1e-308).powi(2)],
            &[0.0],
            crate::FiniteDiffMethod::ThreePoint,
            None,
            Some(1e-308),
            Some((&[0.0], &[f64::INFINITY])),
        )
        .unwrap();
        // The samples are [0, 2, 8]. The quadratic's derivative is zero;
        // dividing individual sample differences first would overflow.
        assert_eq!(jacobian, vec![vec![0.0]]);
    }

    #[test]
    fn approx_derivative_respects_narrow_bounds_and_distinct_nodes() {
        use crate::{FiniteDiffMethod, approx_derivative};
        let lower = 1.0_f64;
        let upper = f64::from_bits(lower.to_bits() + 1);
        let jacobian = approx_derivative(
            |point| {
                assert!(point[0] >= lower && point[0] <= upper);
                vec![point[0]]
            },
            &[lower],
            FiniteDiffMethod::TwoPoint,
            None,
            None,
            Some((&[lower], &[upper])),
        )
        .unwrap();
        assert_eq!(jacobian, vec![vec![1.0]]);
        let result = approx_derivative(
            |point| vec![point[0]],
            &[lower],
            FiniteDiffMethod::ThreePoint,
            None,
            None,
            Some((&[lower], &[upper])),
        );
        assert!(matches!(result, Err(crate::OptError::InvalidArgument { .. })));

        // Three nodes near a binade boundary need their actual, unequal gaps.
        let x = f64::from_bits(1.0_f64.to_bits() - 1);
        let jacobian = approx_derivative(
            |point| vec![point[0] * point[0]],
            &[x],
            FiniteDiffMethod::ThreePoint,
            None,
            Some(f64::EPSILON),
            Some((&[x], &[2.0])),
        )
        .unwrap();
        assert!((jacobian[0][0] - 2.0 * x).abs() < 1e-14);
    }

    #[test]
    fn approx_derivative_rejects_unrepresentable_slopes() {
        use crate::{FiniteDiffMethod, approx_derivative};
        let scalar = |point: &[f64]| {
            if point[0] <= 0.0 {
                -f64::MAX
            } else {
                f64::MAX
            }
        };
        assert!(matches!(
            approx_fprime(&[0.0], scalar, 1e-8),
            Err(crate::OptError::NonFiniteInput { .. })
        ));
        for method in [FiniteDiffMethod::TwoPoint, FiniteDiffMethod::ThreePoint] {
            let result = approx_derivative(
                |point| vec![scalar(point)],
                &[0.0],
                method,
                None,
                None,
                None,
            );
            assert!(matches!(result, Err(crate::OptError::NonFiniteInput { .. })));
        }
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
    fn brent_preserves_bracket_minimum_and_rejects_invalid_brackets() {
        // Only the supplied middle sample lies in the narrow well. Discarding
        // it and running a bounded search from scratch loses the minimum.
        let well = |x: f64| ((x - 0.25).abs() * 1.0e10).min(1.0);
        for points in [[-1.0, 0.25, 2.0], [2.0, 0.25, -1.0]] {
            let x = crate::brent(well, Some(&points), 1.0e-8, 500).unwrap();
            assert_eq!(x, 0.25);
        }
        // Reject bad ordering before invoking the objective.
        let invalid = [0.0, 3.0, 1.0];
        assert!(crate::brent(|_| panic!("bad bracket"), Some(&invalid), 1e-8, 500).is_err());
        assert!(crate::brent(|_| 1.0, None, 1e-8, 500).is_err());
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
        let result =
            linprog(&c, &a_ub, &b_ub, &[], &[], &[], LinprogOptions::default()).expect("linprog");
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

    /// frankenscipy-3vcjx: `derivative` follows scipy.differentiate.derivative's iteration
    /// (stencil, absolute initial step, step division, termination order, status codes).
    /// Values from scipy 1.17.1, live (scratchpad deriv/pins.py and find_status.py; a float64
    /// transliteration matched SciPy bit for bit on 168 cases, deriv/translit.py). Before, fsci
    /// scaled the step by 1 + |x| and refused any non-finite evaluation, so log at x = 1 and
    /// sqrt at x = 0.5 returned Err.
    #[test]
    fn derivative_follows_scipy_iteration() {
        use crate::{DifferentiateStatus, StepDirection};
        struct Row {
            label: &'static str,
            f: fn(f64) -> f64,
            x: f64,
            options: DifferentiateOptions,
            df: f64,
            df_rtol: f64,
            status: DifferentiateStatus,
            nit: usize,
            nfev: usize,
        }
        let defaults = DifferentiateOptions::default();
        let rows = [
            Row {
                label: "log at 1",
                f: f64::ln,
                x: 1.0,
                options: defaults,
                df: 0.999_999_999_998_354_8,
                df_rtol: 1e-13,
                status: DifferentiateStatus::Converged,
                nit: 3,
                nfev: 13,
            },
            Row {
                label: "sqrt at 0.5",
                f: f64::sqrt,
                x: 0.5,
                options: defaults,
                df: 0.707_106_781_186_311_7,
                df_rtol: 1e-13,
                status: DifferentiateStatus::Converged,
                nit: 4,
                nfev: 15,
            },
            Row {
                label: "exp at 1, forward",
                f: f64::exp,
                x: 1.0,
                options: DifferentiateOptions {
                    step_direction: StepDirection::Forward,
                    ..defaults
                },
                // One-sided weights are large, so the last bits of the weight solve and the
                // dot product move df by ~1e-13: fsci gives 2.718281828459112, 6.7e-14 from
                // e, where SciPy's value is 1.4e-13 from it.
                df: 2.718_281_828_458_657_4,
                df_rtol: 1e-12,
                status: DifferentiateStatus::Converged,
                nit: 2,
                nfev: 11,
            },
            Row {
                label: "exp at 1, backward, order 4",
                f: f64::exp,
                x: 1.0,
                options: DifferentiateOptions {
                    step_direction: StepDirection::Backward,
                    order: 4,
                    ..defaults
                },
                df: 2.718_281_825_794_747_4,
                df_rtol: 1e-12,
                status: DifferentiateStatus::Converged,
                nit: 5,
                nfev: 13,
            },
            Row {
                label: "sin at 1, maxiter 3",
                f: f64::sin,
                x: 1.0,
                options: DifferentiateOptions {
                    maxiter: 3,
                    ..defaults
                },
                df: 0.540_302_305_868_131_3,
                df_rtol: 1e-13,
                status: DifferentiateStatus::Converged,
                nit: 2,
                nfev: 11,
            },
            Row {
                // sqrt(t) plus 1 at exactly t = 0.515625 = 0.5 + 2⁻⁶, the abscissa the third
                // iteration adds: the error goes from 4.8e-5 to 46.2, SciPy's error-increase
                // stop. (Rounding-noise cases such as log at 1e8 with step_factor 5 flip
                // between the two implementations and cannot be pinned.)
                label: "sqrt with a jump at 0.515625 (error increases)",
                f: |t: f64| {
                    t.sqrt()
                        + if t.to_bits() == 0.515_625_f64.to_bits() {
                            1.0
                        } else {
                            0.0
                        }
                },
                x: 0.5,
                options: defaults,
                df: 46.940_616_481_301_82,
                df_rtol: 1e-12,
                status: DifferentiateStatus::ErrorIncreased,
                nit: 3,
                nfev: 13,
            },
        ];
        for row in rows {
            let result = derivative(row.f, row.x, row.options).expect(row.label);
            let label = row.label;
            assert!(
                (result.df - row.df).abs() <= row.df_rtol * row.df.abs(),
                "{label}: df {} vs scipy {}",
                result.df,
                row.df
            );
            assert_eq!(result.status, row.status, "{label}");
            assert_eq!(
                result.success,
                row.status == DifferentiateStatus::Converged,
                "{label}"
            );
            assert_eq!((result.nit, result.nfev), (row.nit, row.nfev), "{label}");
        }
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

/// SciPy-compatible type alias for [`BfgsHessian`], matching `scipy.optimize.BFGS`.
pub type BFGS = BfgsHessian;

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

/// SciPy-compatible type alias for [`Sr1Hessian`], matching `scipy.optimize.SR1`.
pub type SR1 = Sr1Hessian;

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

    /// SciPy parity for `scipy.optimize` entry points with no differential coverage
    /// (frankenscipy-ivxx6): `lsq_linear`, `fmin_bfgs`, `fmin_powell`, `fmin_cg`.
    ///
    /// TWO DIFFERENT CONTRACTS, because these are two different kinds of function.
    ///
    /// `lsq_linear` is a deterministic bounded solve, so it is pinned tightly against SciPy
    /// 1.17.1: 5.551e-17 with the bounds binding, 3.362e-11 with them slack. The slack arm is a
    /// CONTROL -- it must reproduce ordinary least squares, and if it did not the bounded arm
    /// would be measuring a broken solver rather than bound handling. The bounds in the binding
    /// arm were chosen so the unconstrained solution lies outside them; two components come back
    /// exactly on their bounds.
    ///
    /// The MINIMISERS are not pinned to SciPy's iterate, and doing so would be wrong: two descent
    /// algorithms with different stopping rules do not agree to machine precision, and SciPy's
    /// stopping point is not more canonical than ours. They are pinned instead to the ANALYTIC
    /// minimiser, which is implementation-independent. Measured distances from truth:
    ///
    /// | case | this crate | SciPy 1.17.1 |
    /// |---|---|---|
    /// | bfgs/quadratic | 2.54e-08 | 5.90e-07 |
    /// | cg/quadratic | 1.22e-07 | 2.98e-09 |
    /// | powell/quadratic | 2.53e-05 | 0.00e+00 |
    /// | bfgs/Rosenbrock | 1.45e-08 | 9.01e-06 |
    /// | powell/Rosenbrock | 1.94e-05 | 1.92e-13 |
    ///
    /// Both reach the same minimiser; neither is uniformly tighter. So these are COVERAGE gaps,
    /// not defects.
    #[test]
    fn lsq_linear_and_fmin_family_against_scipy_1_17_1() {
        use super::{fmin_bfgs, fmin_cg, fmin_powell, lsq_linear};

        fn close(name: &str, got: &[f64], want: &[f64], tol: f64) {
            assert_eq!(got.len(), want.len(), "{name}: length");
            for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
                assert!(
                    (a - b).abs() <= tol,
                    "{name}[{i}]: got {a:.17e}, expected {b:.17e}"
                );
            }
        }

        let a: Vec<Vec<f64>> = vec![
            vec![1.0, 0.5, -0.25],
            vec![0.0, 2.0, 1.0],
            vec![3.0, -1.0, 0.5],
            vec![-1.0, 1.5, 2.0],
            vec![0.5, 0.5, -3.0],
        ];
        let b = vec![1.0, -2.0, 3.0, 0.5, -1.5];

        // CONTROL: bounds too wide to bind -> ordinary least squares.
        let x = lsq_linear(&a, &b, &[-1.0e6; 3], &[1.0e6; 3]).unwrap();
        close(
            "lsq_linear/slack",
            &x,
            &[0.6470480849976267, -0.7417454695924158, 0.5825939854199373],
            1e-8,
        );

        // Bounds that BIND: two components land exactly on a bound.
        let lo = vec![-0.2, -0.5, 0.0];
        let hi = vec![0.5, 0.4, 1.0];
        let x = lsq_linear(&a, &b, &lo, &hi).unwrap();
        close(
            "lsq_linear/binding",
            &x,
            &[
                0.49999999999999994,
                -0.49999999999999994,
                0.5109170305676856,
            ],
            1e-12,
        );
        for (i, (&v, (&l, &u))) in x.iter().zip(lo.iter().zip(hi.iter())).enumerate() {
            assert!(
                v >= l - 1e-12 && v <= u + 1e-12,
                "lsq_linear: component {i} escaped its bound"
            );
        }
        assert!(
            x.iter()
                .zip(lo.iter().zip(hi.iter()))
                .any(|(&v, (&l, &u))| (v - l).abs() < 1e-9 || (v - u).abs() < 1e-9),
            "the bounded fixture must actually BIND, or it tests nothing the slack arm does not"
        );

        // Minimisers, against the ANALYTIC minimiser rather than SciPy's iterate.
        let quad = |x: &[f64]| {
            (x[0] - 1.0).powi(2) + 2.0 * (x[1] + 2.0).powi(2) + 3.0 * (x[2] - 0.5).powi(2)
        };
        let rosen = |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
        let q0 = [0.0, 0.0, 0.0];
        let r0 = [-1.2, 1.0];
        let quad_min = [1.0, -2.0, 0.5];
        let rosen_min = [1.0, 1.0];

        close(
            "fmin_bfgs/quadratic",
            &fmin_bfgs(quad, &q0).unwrap(),
            &quad_min,
            1e-5,
        );
        close(
            "fmin_cg/quadratic",
            &fmin_cg(quad, &q0).unwrap(),
            &quad_min,
            1e-5,
        );
        close(
            "fmin_powell/quadratic",
            &fmin_powell(quad, &q0).unwrap(),
            &quad_min,
            1e-3,
        );
        close(
            "fmin_bfgs/rosenbrock",
            &fmin_bfgs(rosen, &r0).unwrap(),
            &rosen_min,
            1e-5,
        );
        close(
            "fmin_powell/rosenbrock",
            &fmin_powell(rosen, &r0).unwrap(),
            &rosen_min,
            1e-3,
        );
    }
}
