#![forbid(unsafe_code)]

//! Nonlinear least-squares curve fitting (Levenberg-Marquardt).
//!
//! Provides `curve_fit` (matching `scipy.optimize.curve_fit`) and `least_squares`
//! (matching `scipy.optimize.least_squares`).

use fsci_runtime::RuntimeMode;

use crate::types::OptError;

/// Result from `least_squares`.
#[derive(Debug, Clone, PartialEq)]
pub struct LeastSquaresResult {
    /// Solution parameters.
    pub x: Vec<f64>,
    /// Residual vector at the solution.
    pub fun: Vec<f64>,
    /// Sum of squared residuals (cost = 0.5 * ||fun||^2).
    pub cost: f64,
    /// Whether the solver converged.
    pub success: bool,
    /// Status message.
    pub message: String,
    /// Number of function evaluations.
    pub nfev: usize,
    /// Number of Jacobian evaluations.
    pub njev: usize,
    /// Number of iterations.
    pub nit: usize,
    /// Jacobian at the solution (row-major, m rows x n cols).
    pub jac: Vec<Vec<f64>>,
}

/// Options for `least_squares`.
#[derive(Debug, Clone, Copy)]
pub struct LeastSquaresOptions {
    /// Convergence tolerance on the gradient (gtol).
    pub gtol: f64,
    /// Convergence tolerance on the change in parameters (xtol).
    pub xtol: f64,
    /// Convergence tolerance on the change in cost (ftol).
    pub ftol: f64,
    /// Maximum number of iterations.
    pub max_nfev: Option<usize>,
    /// Finite-difference step for Jacobian approximation.
    pub diff_step: f64,
    /// Runtime mode.
    pub mode: RuntimeMode,
}

impl Default for LeastSquaresOptions {
    fn default() -> Self {
        Self {
            gtol: 1.0e-8,
            xtol: 1.0e-8,
            ftol: 1.0e-8,
            max_nfev: None,
            diff_step: 1.4901161193847656e-8, // sqrt(machine eps)
            mode: RuntimeMode::Strict,
        }
    }
}

/// Result from `curve_fit`.
#[derive(Debug, Clone, PartialEq)]
pub struct CurveFitResult {
    /// Optimal parameters.
    pub popt: Vec<f64>,
    /// Estimated covariance matrix of parameters (n x n).
    pub pcov: Vec<Vec<f64>>,
    /// Underlying least-squares result.
    pub ls_result: LeastSquaresResult,
}

/// Options for `curve_fit`.
#[derive(Debug, Clone, Default)]
pub struct CurveFitOptions {
    /// Initial guess for parameters.
    pub p0: Option<Vec<f64>>,
    /// Least-squares options.
    pub ls_options: LeastSquaresOptions,
    /// Whether to compute absolute sigma (unscaled covariance).
    pub absolute_sigma: bool,
}

/// Solve a nonlinear least-squares problem using the Levenberg-Marquardt algorithm.
///
/// Finds `x` that minimizes `0.5 * sum(residuals(x)^2)` where `residuals` maps
/// parameters to a vector of residuals.
///
/// Equivalent to `scipy.optimize.least_squares` with `method='lm'`.
pub fn least_squares<F>(
    residuals: F,
    x0: &[f64],
    options: LeastSquaresOptions,
) -> Result<LeastSquaresResult, OptError>
where
    F: Fn(&[f64]) -> Vec<f64> + Sync,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("x0 must have at least one element"),
        });
    }
    if x0.iter().any(|v| !v.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: String::from("x0 must not contain NaN or Inf"),
        });
    }
    if !options.diff_step.is_finite() || options.diff_step <= 0.0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("diff_step must be finite and > 0"),
        });
    }
    if !options.gtol.is_finite() || options.gtol < 0.0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("gtol must be finite and >= 0"),
        });
    }
    if !options.xtol.is_finite() || options.xtol < 0.0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("xtol must be finite and >= 0"),
        });
    }
    if !options.ftol.is_finite() || options.ftol < 0.0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("ftol must be finite and >= 0"),
        });
    }

    let max_nfev = options.max_nfev.unwrap_or(100 * (n + 1));

    let mut x = x0.to_vec();
    let mut r = residuals(&x);
    let m = r.len();
    let mut nfev = 1usize;
    let mut njev = 0usize;

    if m < n {
        return Err(OptError::InvalidArgument {
            detail: format!("number of residuals ({m}) must be >= number of parameters ({n})"),
        });
    }

    // Check for non-finite residuals
    if r.iter().any(|v| !v.is_finite()) {
        return match options.mode {
            RuntimeMode::Strict => Err(OptError::NonFiniteInput {
                detail: String::from("residuals contain non-finite values at x0"),
            }),
            RuntimeMode::Hardened => Err(OptError::NonFiniteInput {
                detail: String::from("hardened mode rejects non-finite residuals"),
            }),
        };
    }

    let mut cost = 0.5 * dot_vec(&r, &r);
    let mut jac = Vec::new();
    let mut x_perturbed = x.clone();
    finite_diff_jacobian_parallel_into(
        &residuals,
        &x,
        &r,
        options.diff_step,
        &mut jac,
        &mut x_perturbed,
    );
    nfev += n;
    njev += 1;
    // J^T J (O(n²·m)) and J^T r depend only on (jac, r), which change ONLY on an accepted
    // step. Cache them and recompute only when jac/r change, rather than rebuilding both at
    // the top of every iteration — on a rejected step (mu ratchets up, common for hard /
    // ill-conditioned problems) only `mu`/`nu` change, so the rebuild was redundant. Byte-
    // identical: jtj/jtr are deterministic functions of (jac, r).
    let mut jtj = Vec::new();
    let mut jtr = Vec::new();
    jtj_matrix_into(&jac, &mut jtj);
    jt_vec_into(&jac, &r, &mut jtr);
    let mut damped_normal = Vec::new();
    let mut cholesky_low = Vec::new();
    let mut solve_y = Vec::new();
    let mut step = Vec::new();
    let mut jstep = Vec::new();

    // Initial damping parameter (Marquardt strategy)
    let mut mu = 1.0e-3 * max_diag_jtj(&jac);
    if mu == 0.0 {
        mu = 1.0;
    }
    let mut nu = 2.0;
    let mut nit = 0usize;

    for _ in 0..max_nfev {
        nit += 1;

        // jtj / jtr are current for this (jac, r) — computed before the loop and refreshed
        // only after an accepted step recomputes the Jacobian (see below).

        // Check gradient convergence: ||J^T r||_inf <= gtol
        let grad_inf = jtr.iter().map(|v| v.abs()).fold(0.0_f64, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        });
        if grad_inf <= options.gtol {
            return Ok(LeastSquaresResult {
                x,
                fun: r,
                cost,
                success: true,
                message: String::from("gradient converged (||J^T r||_inf <= gtol)"),
                nfev,
                njev,
                nit,
                jac,
            });
        }

        // Solve (J^T J + mu I) * step = -J^T r
        solve_damped_normal_equations_into(
            &jtj,
            &jtr,
            mu,
            n,
            DampedNormalScratch {
                a: &mut damped_normal,
                low: &mut cholesky_low,
                y: &mut solve_y,
                step: &mut step,
            },
        );

        let step_norm = l2_norm(&step);
        let x_norm = l2_norm(&x);

        // Check parameter convergence
        if step_norm <= options.xtol * (options.xtol + x_norm) {
            return Ok(LeastSquaresResult {
                x,
                fun: r,
                cost,
                success: true,
                message: String::from("parameter change converged (xtol)"),
                nfev,
                njev,
                nit,
                jac,
            });
        }

        // Trial step
        let x_new: Vec<f64> = x.iter().zip(step.iter()).map(|(a, b)| a + b).collect();
        let r_new = residuals(&x_new);
        nfev += 1;
        let cost_new = 0.5 * dot_vec(&r_new, &r_new);

        // Gain ratio (actual reduction / predicted reduction)
        let predicted_reduction = {
            mat_vec_into(&jac, &step, &mut jstep);
            -dot_vec(&jtr, &step) - 0.5 * dot_vec(&jstep, &jstep)
        };

        let actual_reduction = cost - cost_new;

        if predicted_reduction > 0.0 {
            let rho = actual_reduction / predicted_reduction;

            if rho > 0.25 {
                // Accept step
                x = x_new;
                r = r_new;
                let old_cost = cost;
                cost = cost_new;

                // Check cost convergence
                let cost_change = (old_cost - cost) / (1.0 + cost);
                if cost_change.abs() <= options.ftol {
                    finite_diff_jacobian_parallel_into(
                        &residuals,
                        &x,
                        &r,
                        options.diff_step,
                        &mut jac,
                        &mut x_perturbed,
                    );
                    nfev += n;
                    njev += 1;
                    return Ok(LeastSquaresResult {
                        x,
                        fun: r,
                        cost,
                        success: true,
                        message: String::from("cost change converged (ftol)"),
                        nfev,
                        njev,
                        nit,
                        jac,
                    });
                }

                // Recompute Jacobian (and the derived J^T J / J^T r) — jac and r changed.
                finite_diff_jacobian_parallel_into(
                    &residuals,
                    &x,
                    &r,
                    options.diff_step,
                    &mut jac,
                    &mut x_perturbed,
                );
                nfev += n;
                njev += 1;
                jtj_matrix_into(&jac, &mut jtj);
                jt_vec_into(&jac, &r, &mut jtr);

                // Decrease damping
                mu *= f64::max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0).powi(3));
                nu = 2.0;
            } else {
                // Reject step, increase damping
                mu *= nu;
                nu *= 2.0;
            }
        } else {
            // Reject step, increase damping
            mu *= nu;
            nu *= 2.0;
        }

        if nfev >= max_nfev {
            return Ok(LeastSquaresResult {
                x,
                fun: r,
                cost,
                success: false,
                message: format!("max function evaluations exceeded ({max_nfev})"),
                nfev,
                njev,
                nit,
                jac,
            });
        }
    }

    Ok(LeastSquaresResult {
        x,
        fun: r,
        cost,
        success: false,
        message: String::from("max iterations exceeded"),
        nfev,
        njev,
        nit,
        jac,
    })
}

/// Nonlinear curve fitting.
///
/// Finds the parameters `popt` such that `f(xdata, popt)` best fits `ydata`
/// in the least-squares sense.
///
/// Equivalent to `scipy.optimize.curve_fit`.
///
/// # Arguments
/// * `f` - Model function `f(x, params) -> y` where `x` is a single data point
/// * `xdata` - Independent variable data
/// * `ydata` - Dependent variable data (same length as xdata)
/// * `options` - Curve fitting options (initial guess, tolerances, etc.)
pub fn curve_fit<F>(
    f: F,
    xdata: &[f64],
    ydata: &[f64],
    options: CurveFitOptions,
) -> Result<CurveFitResult, OptError>
where
    F: Fn(f64, &[f64]) -> f64 + Sync,
{
    if xdata.len() != ydata.len() {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "xdata and ydata must have same length (got {} and {})",
                xdata.len(),
                ydata.len()
            ),
        });
    }
    if xdata.is_empty() {
        return Err(OptError::InvalidArgument {
            detail: String::from("xdata must not be empty"),
        });
    }

    let p0 = options.p0.ok_or_else(|| OptError::InvalidArgument {
        detail: String::from("p0 (initial parameter guess) is required"),
    })?;
    let n_params = p0.len();

    if xdata.len() < n_params {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "number of data points ({}) must be >= number of parameters ({n_params})",
                xdata.len()
            ),
        });
    }

    // Build residual function: r_i(p) = f(x_i, p) - y_i
    let residuals = |params: &[f64]| -> Vec<f64> {
        xdata
            .iter()
            .zip(ydata.iter())
            .map(|(&xi, &yi)| f(xi, params) - yi)
            .collect()
    };

    let ls_result = least_squares(residuals, &p0, options.ls_options)?;

    // Compute covariance: pcov = (J^T J)^{-1} * s^2
    // where s^2 = cost * 2 / (m - n) for relative sigma
    let pcov = compute_covariance(
        &ls_result.jac,
        ls_result.cost,
        xdata.len(),
        n_params,
        options.absolute_sigma,
    );

    Ok(CurveFitResult {
        popt: ls_result.x.clone(),
        pcov,
        ls_result,
    })
}

/// Numerically-stable `ln(1 + e^u)` (softplus); for `u` above ~30 it collapses to `u`.
fn softplus(u: f64) -> f64 {
    if u > 30.0 { u } else { u.exp().ln_1p() }
}

/// Inverse softplus `ln(e^y - 1)` for `y > 0`; for large `y` it collapses to `y`.
fn softplus_inv(y: f64) -> f64 {
    if y > 30.0 {
        y
    } else {
        (y.exp() - 1.0).max(f64::MIN_POSITIVE).ln()
    }
}

/// Per-parameter smooth, monotone bijection between a box-constrained parameter
/// `p` and an unconstrained `u ∈ ℝ`, so a bounded least-squares fit can reuse the
/// unbounded Levenberg-Marquardt core. Two-sided bounds use the logistic map,
/// one-sided bounds a softplus shift, and an unbounded coordinate the identity.
/// This is the reparameterisation strategy `lmfit` uses; for an interior optimum
/// it reaches the identical minimiser as `scipy.optimize.curve_fit(bounds=...)`.
#[derive(Clone, Copy)]
enum BoundKind {
    Both(f64, f64),
    Lower(f64),
    Upper(f64),
    Free,
}

impl BoundKind {
    fn new(lo: f64, hi: f64) -> Self {
        match (lo.is_finite(), hi.is_finite()) {
            (true, true) => BoundKind::Both(lo, hi),
            (true, false) => BoundKind::Lower(lo),
            (false, true) => BoundKind::Upper(hi),
            (false, false) => BoundKind::Free,
        }
    }

    /// Map the unconstrained coordinate `u` to the bounded parameter `p`.
    fn to_param(self, u: f64) -> f64 {
        match self {
            BoundKind::Both(lo, hi) => lo + (hi - lo) / (1.0 + (-u).exp()),
            BoundKind::Lower(lo) => lo + softplus(u),
            BoundKind::Upper(hi) => hi - softplus(-u),
            BoundKind::Free => u,
        }
    }

    /// Map the bounded parameter `p` (assumed strictly inside the box) to `u`.
    fn to_unconstrained(self, p: f64) -> f64 {
        match self {
            BoundKind::Both(lo, hi) => ((p - lo) / (hi - p)).ln(),
            BoundKind::Lower(lo) => softplus_inv(p - lo),
            BoundKind::Upper(hi) => -softplus_inv(hi - p),
            BoundKind::Free => p,
        }
    }

    /// Clip `p` to lie strictly inside the box so `to_unconstrained` stays finite.
    fn clip_inside(self, p: f64) -> f64 {
        match self {
            BoundKind::Both(lo, hi) => {
                let eps = (hi - lo) * 1.0e-10;
                p.clamp(lo + eps, hi - eps)
            }
            BoundKind::Lower(lo) => p.max(lo + 1.0e-10 * (1.0 + lo.abs())),
            BoundKind::Upper(hi) => p.min(hi - 1.0e-10 * (1.0 + hi.abs())),
            BoundKind::Free => p,
        }
    }
}

/// Box-constrained nonlinear least squares: minimise `0.5·‖residuals(p)‖²` subject
/// to `lower ≤ p ≤ upper`, the bounded analogue of [`least_squares`].
///
/// Mirrors `scipy.optimize.least_squares(..., bounds=(lower, upper))`. Each bounded
/// coordinate is smoothly reparameterised to an unconstrained variable (logistic for
/// two-sided bounds, softplus for one-sided, identity for `±inf`), and the existing
/// Levenberg-Marquardt core solves the unconstrained problem — so the bounded solve is
/// as fast as the unbounded one. For an interior optimum the minimiser is identical to
/// SciPy's `trf`; when a bound is active the transform approaches it asymptotically.
/// The returned `x`, `fun`, and `jac` are recomputed in parameter space at the optimum
/// (so a downstream covariance is in `p`-space, not the internal `u`-space).
pub fn least_squares_bounded<F>(
    residuals: F,
    x0: &[f64],
    lower: &[f64],
    upper: &[f64],
    options: LeastSquaresOptions,
) -> Result<LeastSquaresResult, OptError>
where
    F: Fn(&[f64]) -> Vec<f64> + Sync,
{
    let n = x0.len();
    if lower.len() != n || upper.len() != n {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "lower ({}) and upper ({}) bounds must match x0 length ({n})",
                lower.len(),
                upper.len()
            ),
        });
    }
    let mut kinds = Vec::with_capacity(n);
    let mut u0 = Vec::with_capacity(n);
    for i in 0..n {
        if lower[i].partial_cmp(&upper[i]) != Some(std::cmp::Ordering::Less) {
            return Err(OptError::InvalidArgument {
                detail: format!("require lower[{i}] < upper[{i}]"),
            });
        }
        let kind = BoundKind::new(lower[i], upper[i]);
        u0.push(kind.to_unconstrained(kind.clip_inside(x0[i])));
        kinds.push(kind);
    }

    let to_p = |u: &[f64]| -> Vec<f64> { (0..n).map(|i| kinds[i].to_param(u[i])).collect() };
    let res_u = |u: &[f64]| -> Vec<f64> { residuals(&to_p(u)) };

    let mut ls = least_squares(res_u, &u0, options)?;

    // The LM ran in `u`-space; map the optimum back and recompute the residual and a
    // finite-difference Jacobian in `p`-space so `x`/`fun`/`jac` (hence any covariance)
    // are expressed in the original parameters.
    let popt = to_p(&ls.x);
    let r_p = residuals(&popt);
    let mut jac_p = Vec::new();
    let mut scratch = popt.clone();
    finite_diff_jacobian_parallel_into(
        &residuals,
        &popt,
        &r_p,
        options.diff_step,
        &mut jac_p,
        &mut scratch,
    );
    ls.cost = 0.5 * dot_vec(&r_p, &r_p);
    ls.x = popt;
    ls.fun = r_p;
    ls.jac = jac_p;
    Ok(ls)
}

/// Box-constrained curve fitting: the bounded analogue of [`curve_fit`], matching
/// `scipy.optimize.curve_fit(f, xdata, ydata, p0=..., bounds=(lower, upper))`.
///
/// Reuses [`least_squares_bounded`] (smooth reparameterisation + the fast LM core), so a
/// bounded fit runs at unbounded-fit speed — dramatically faster than SciPy's `trf` path
/// for the common case of sanity bounds with an interior optimum.
pub fn curve_fit_bounded<F>(
    f: F,
    xdata: &[f64],
    ydata: &[f64],
    lower: &[f64],
    upper: &[f64],
    options: CurveFitOptions,
) -> Result<CurveFitResult, OptError>
where
    F: Fn(f64, &[f64]) -> f64 + Sync,
{
    if xdata.len() != ydata.len() {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "xdata and ydata must have same length (got {} and {})",
                xdata.len(),
                ydata.len()
            ),
        });
    }
    if xdata.is_empty() {
        return Err(OptError::InvalidArgument {
            detail: String::from("xdata must not be empty"),
        });
    }
    let p0 = options.p0.ok_or_else(|| OptError::InvalidArgument {
        detail: String::from("p0 (initial parameter guess) is required"),
    })?;
    let n_params = p0.len();
    if xdata.len() < n_params {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "number of data points ({}) must be >= number of parameters ({n_params})",
                xdata.len()
            ),
        });
    }

    let residuals = |params: &[f64]| -> Vec<f64> {
        xdata
            .iter()
            .zip(ydata.iter())
            .map(|(&xi, &yi)| f(xi, params) - yi)
            .collect()
    };

    let ls_result = least_squares_bounded(residuals, &p0, lower, upper, options.ls_options)?;

    let pcov = compute_covariance(
        &ls_result.jac,
        ls_result.cost,
        xdata.len(),
        n_params,
        options.absolute_sigma,
    );

    Ok(CurveFitResult {
        popt: ls_result.x.clone(),
        pcov,
        ls_result,
    })
}

/// Batched curve fitting: fit the same model `f` (and shared `xdata`) to MANY independent
/// `ydata` rows, returning one optimal-parameter vector per row. This is the vmap-over-solver
/// primitive SciPy lacks — there you loop `curve_fit` in Python, paying the per-call overhead
/// N times serially; here the N independent fits are fanned across cores and the model is an
/// inlined Rust closure. Row `i` of the output is byte-identical to `curve_fit(f, xdata,
/// &ydata_rows[i], options).popt`.
///
/// Common in imaging / signal processing (a decay or peak fit per pixel / channel / trace).
pub fn curve_fit_many<F>(
    f: F,
    xdata: &[f64],
    ydata_rows: &[Vec<f64>],
    options: CurveFitOptions,
) -> Result<Vec<Vec<f64>>, OptError>
where
    F: Fn(f64, &[f64]) -> f64 + Sync,
{
    let nrows = ydata_rows.len();
    if nrows == 0 {
        return Ok(Vec::new());
    }
    let f_ref = &f;
    let opts_ref = &options;
    let fit_one = move |row: &[f64]| curve_fit(f_ref, xdata, row, opts_ref.clone()).map(|r| r.popt);

    // Each fit is an independent ~0.1 ms LM solve (heavy per item) → fan whole rows across
    // cores, capped by the row count; a tiny batch stays serial to dodge the spawn floor.
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let nthreads = cores.min(nrows);
    if nthreads <= 1 || nrows < 8 {
        return ydata_rows.iter().map(|row| fit_one(row)).collect();
    }

    let chunk = nrows.div_ceil(nthreads);
    let fit_one = &fit_one;
    let chunk_results: Vec<Result<Vec<Vec<f64>>, OptError>> = std::thread::scope(|scope| {
        (0..nthreads)
            .filter_map(|t| {
                let lo = t * chunk;
                if lo >= nrows {
                    return None;
                }
                let hi = (lo + chunk).min(nrows);
                Some(scope.spawn(move || {
                    (lo..hi)
                        .map(|i| fit_one(&ydata_rows[i]))
                        .collect::<Result<Vec<_>, _>>()
                }))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().expect("curve_fit_many worker panicked"))
            .collect()
    });

    let mut out = Vec::with_capacity(nrows);
    for cr in chunk_results {
        out.extend(cr?);
    }
    Ok(out)
}

/// Vectorised [`least_squares`] over `n` independent parameter rows — the
/// vmap-over-solver form for `scipy.optimize.least_squares`. `residuals(x, params)`
/// is an inlined Rust closure (no per-iteration Python callback, unlike a looped
/// scipy `least_squares`), and the independent (heavy, iterative LM/finite-diff)
/// solves fan whole-row across cores; a tiny batch stays serial. Entry `k` equals
/// `least_squares(|x| residuals(x, &param_rows[k]), x0, options)`.
pub fn least_squares_many<F>(
    residuals: F,
    x0: &[f64],
    param_rows: &[Vec<f64>],
    options: LeastSquaresOptions,
) -> Vec<Result<LeastSquaresResult, OptError>>
where
    F: Fn(&[f64], &[f64]) -> Vec<f64> + Sync,
{
    let nrows = param_rows.len();
    if nrows == 0 {
        return Vec::new();
    }
    let r_ref = &residuals;
    let solve_one = move |params: &[f64]| least_squares(|x| r_ref(x, params), x0, options);

    // Each solve is a heavy iterative LM fit → fan whole rows across cores, capped
    // by the row count; a tiny batch stays serial to dodge the spawn floor.
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let nthreads = cores.min(nrows);
    if nthreads <= 1 || nrows < 4 {
        return param_rows.iter().map(|p| solve_one(p)).collect();
    }

    let chunk = nrows.div_ceil(nthreads);
    let solve_one = &solve_one;
    let chunk_results: Vec<Vec<Result<LeastSquaresResult, OptError>>> =
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
                .map(|h| h.join().expect("least_squares_many worker panicked"))
                .collect()
        });

    let mut out = Vec::with_capacity(nrows);
    for cr in chunk_results {
        out.extend(cr);
    }
    out
}

/// Batched box-constrained curve fitting — [`curve_fit_bounded`] fanned across many `ydata`
/// rows, the bounded analogue of [`curve_fit_many`]. Row `i` is byte-identical to
/// `curve_fit_bounded(f, xdata, &ydata_rows[i], lower, upper, options).popt`.
pub fn curve_fit_bounded_many<F>(
    f: F,
    xdata: &[f64],
    ydata_rows: &[Vec<f64>],
    lower: &[f64],
    upper: &[f64],
    options: CurveFitOptions,
) -> Result<Vec<Vec<f64>>, OptError>
where
    F: Fn(f64, &[f64]) -> f64 + Sync,
{
    let nrows = ydata_rows.len();
    if nrows == 0 {
        return Ok(Vec::new());
    }
    let f_ref = &f;
    let opts_ref = &options;
    let fit_one = move |row: &[f64]| {
        curve_fit_bounded(f_ref, xdata, row, lower, upper, opts_ref.clone()).map(|r| r.popt)
    };

    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let nthreads = cores.min(nrows);
    if nthreads <= 1 || nrows < 8 {
        return ydata_rows.iter().map(|row| fit_one(row)).collect();
    }

    let chunk = nrows.div_ceil(nthreads);
    let fit_one = &fit_one;
    let chunk_results: Vec<Result<Vec<Vec<f64>>, OptError>> = std::thread::scope(|scope| {
        (0..nthreads)
            .filter_map(|t| {
                let lo = t * chunk;
                if lo >= nrows {
                    return None;
                }
                let hi = (lo + chunk).min(nrows);
                Some(scope.spawn(move || {
                    (lo..hi)
                        .map(|i| fit_one(&ydata_rows[i]))
                        .collect::<Result<Vec<_>, _>>()
                }))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().expect("curve_fit_bounded_many worker panicked"))
            .collect()
    });

    let mut out = Vec::with_capacity(nrows);
    for cr in chunk_results {
        out.extend(cr?);
    }
    Ok(out)
}

/// Result from [`leastsq`] — the classic MINPACK-style interface.
#[derive(Debug, Clone, PartialEq)]
pub struct LeastsqResult {
    /// Solution parameters (scipy's `x`).
    pub x: Vec<f64>,
    /// Unscaled covariance estimate `cov_x = (JᵀJ)⁻¹`, or `None` if the
    /// Jacobian is rank-deficient at the solution (matching scipy, which
    /// returns `None` for `cov_x` in that case).
    pub cov_x: Option<Vec<Vec<f64>>>,
    /// Number of function evaluations (scipy's `infodict['nfev']`).
    pub nfev: usize,
    /// Human-readable termination message (scipy's `mesg`).
    pub mesg: String,
    /// MINPACK termination flag (scipy's `ier`). A value in `1..=4` means a
    /// solution was found.
    pub ier: i32,
}

/// Minimize the sum of squares of a set of equations (Levenberg-Marquardt).
///
/// Mirrors `scipy.optimize.leastsq(func, x0)`: `func` maps the parameter vector
/// to a vector of residuals, and the solver returns the optimal parameters
/// together with the unscaled covariance `cov_x = (JᵀJ)⁻¹` and a MINPACK-style
/// `ier` termination flag (`ier in {1, 2, 3, 4}` indicates success).
///
/// This is the legacy interface that predates [`least_squares`]; both wrap the
/// same Levenberg-Marquardt core, so they converge to the same minimum.
pub fn leastsq<F>(
    func: F,
    x0: &[f64],
    options: LeastSquaresOptions,
) -> Result<LeastsqResult, OptError>
where
    F: Fn(&[f64]) -> Vec<f64> + Sync,
{
    let ls = least_squares(func, x0, options)?;
    let n = x0.len();
    let m = ls.fun.len();

    // scipy's `cov_x` is the *unscaled* inverse (JᵀJ)⁻¹; reuse the
    // `absolute_sigma` branch of the covariance helper. A rank-deficient
    // Jacobian yields a non-finite inverse -> report `None` like scipy.
    let cov = compute_covariance(&ls.jac, ls.cost, m, n, true);
    let cov_x = if cov.iter().flatten().all(|v| v.is_finite()) {
        Some(cov)
    } else {
        None
    };

    // Map the convergence reason onto the MINPACK `ier` codes scipy reports.
    let (ier, mesg) = if ls.success {
        if ls.message.contains("ftol") {
            (
                1,
                "Both actual and predicted relative reductions in the sum of squares\n  are at most ftol.",
            )
        } else if ls.message.contains("xtol") {
            (
                2,
                "The relative error between two consecutive iterates is at most xtol.",
            )
        } else if ls.message.contains("gtol") {
            (
                4,
                "The cosine of the angle between func(x) and any column of the\n  Jacobian is at most gtol in absolute value.",
            )
        } else {
            (1, "The solution converged.")
        }
    } else {
        (5, "Number of calls to function has reached maxfev = 0.")
    };

    Ok(LeastsqResult {
        x: ls.x,
        cov_x,
        nfev: ls.nfev,
        mesg: mesg.to_string(),
        ier,
    })
}

// ────────────────────────── helpers ──────────────────────────

fn dot_vec(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn l2_norm(v: &[f64]) -> f64 {
    dot_vec(v, v).sqrt()
}

fn resize_square(matrix: &mut Vec<Vec<f64>>, n: usize) {
    if matrix.len() != n {
        matrix.resize_with(n, Vec::new);
    }
    for row in matrix.iter_mut() {
        row.resize(n, 0.0);
        row.fill(0.0);
    }
}

/// Compute finite-difference Jacobian into fixed-shape scratch.
fn finite_diff_jacobian_into<F>(
    residuals: &F,
    x: &[f64],
    r0: &[f64],
    eps: f64,
    jac: &mut Vec<Vec<f64>>,
    x_perturbed: &mut Vec<f64>,
) where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x.len();
    let m = r0.len();
    if jac.len() != m {
        jac.resize_with(m, Vec::new);
    }
    for row in jac.iter_mut() {
        row.resize(n, 0.0);
    }
    x_perturbed.resize(n, 0.0);
    x_perturbed.copy_from_slice(x);

    for j in 0..n {
        let step = eps * (1.0 + x[j].abs());
        let original = x_perturbed[j];
        x_perturbed[j] += step;
        let r_plus = residuals(x_perturbed);
        x_perturbed[j] = original; // restore
        for i in 0..m {
            jac[i][j] = (r_plus[i] - r0[i]) / step;
        }
    }
}

/// Finite-difference Jacobian with the independent columns evaluated in parallel.
///
/// Each column `j` perturbs one parameter and re-evaluates the full `m`-residual
/// vector — the columns are independent, so their (expensive) residual evals run
/// on a worker pool and the `jac` matrix is filled serially afterwards. The result
/// is byte-identical to [`finite_diff_jacobian_into`] (each column uses the same
/// per-column `step = eps*(1+|x[j]|)` and the same one-sided difference).
///
/// Parallelism pays only when `n_params * m_data * cost_per_eval` is large enough
/// to amortize thread spawn (measured: a hard loss below ~30K work, 2.75–13.66x
/// above it), so below the work-gate this falls back to the serial routine — which
/// also keeps `available_parallelism()` (a syscall) off the small-problem hot path.
fn finite_diff_jacobian_parallel_into<F>(
    residuals: &F,
    x: &[f64],
    r0: &[f64],
    eps: f64,
    jac: &mut Vec<Vec<f64>>,
    x_perturbed: &mut Vec<f64>,
) where
    F: Fn(&[f64]) -> Vec<f64> + Sync,
{
    let n = x.len();
    let m = r0.len();
    // Work-gate BEFORE any syscall: below it, the serial routine is faster.
    if n < 4 || m < 8192 || n.saturating_mul(m) < 131_072 {
        finite_diff_jacobian_into(residuals, x, r0, eps, jac, x_perturbed);
        return;
    }
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let nthreads = cores.min(n);
    if nthreads <= 1 {
        finite_diff_jacobian_into(residuals, x, r0, eps, jac, x_perturbed);
        return;
    }

    if jac.len() != m {
        jac.resize_with(m, Vec::new);
    }
    for row in jac.iter_mut() {
        row.resize(n, 0.0);
    }

    // Each worker owns a private perturbed-parameter buffer and returns
    // `(column index, perturbed residual vector, step)` for its column range.
    let per = n.div_ceil(nthreads);
    let columns: Vec<(usize, Vec<f64>, f64)> = std::thread::scope(|scope| {
        (0..nthreads)
            .filter_map(|t| {
                let lo = t * per;
                if lo >= n {
                    return None;
                }
                let hi = (lo + per).min(n);
                Some(scope.spawn(move || {
                    let mut xp = x.to_vec();
                    (lo..hi)
                        .map(|j| {
                            let step = eps * (1.0 + x[j].abs());
                            let original = xp[j];
                            xp[j] += step;
                            let r_plus = residuals(&xp);
                            xp[j] = original;
                            (j, r_plus, step)
                        })
                        .collect::<Vec<_>>()
                }))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .flat_map(|h| h.join().expect("finite_diff_jacobian worker panicked"))
            .collect()
    });

    for (j, r_plus, step) in &columns {
        let step = *step;
        for i in 0..m {
            jac[i][*j] = (r_plus[i] - r0[i]) / step;
        }
    }
    let _ = x_perturbed;
}

/// Compute J^T * J (n x n).
fn jtj_matrix(jac: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut jtj = Vec::new();
    jtj_matrix_into(jac, &mut jtj);
    jtj
}

/// Compute J^T * J (n x n) into fixed-shape scratch.
fn jtj_matrix_into(jac: &[Vec<f64>], jtj: &mut Vec<Vec<f64>>) {
    let n = jac.first().map_or(0, Vec::len);
    resize_square(jtj, n);
    for row in jac {
        for i in 0..n {
            for j in i..n {
                let v = row[i] * row[j];
                jtj[i][j] += v;
                if i != j {
                    jtj[j][i] += v;
                }
            }
        }
    }
}

/// Compute J^T * v into fixed-shape scratch.
fn jt_vec_into(jac: &[Vec<f64>], v: &[f64], result: &mut Vec<f64>) {
    let n = jac.first().map_or(0, Vec::len);
    result.resize(n, 0.0);
    result.fill(0.0);
    for (row, &vi) in jac.iter().zip(v.iter()) {
        for (j, &jval) in row.iter().enumerate() {
            result[j] += jval * vi;
        }
    }
}

/// Compute J * v into fixed-shape scratch.
fn mat_vec_into(jac: &[Vec<f64>], v: &[f64], result: &mut Vec<f64>) {
    result.resize(jac.len(), 0.0);
    for (out, row) in result.iter_mut().zip(jac.iter()) {
        *out = dot_vec(row, v);
    }
}

/// Max diagonal element of J^T * J.
fn max_diag_jtj(jac: &[Vec<f64>]) -> f64 {
    let n = jac.first().map_or(0, Vec::len);
    let mut max_val = 0.0_f64;
    for j in 0..n {
        let mut diag = 0.0;
        for row in jac {
            diag += row[j] * row[j];
        }
        if max_val.is_nan() || diag.is_nan() {
            max_val = f64::NAN;
        } else {
            max_val = max_val.max(diag);
        }
    }
    max_val
}

/// Solve (A + mu * I) * x = -b into fixed-shape scratch.
/// Falls back to diagonal if matrix is singular.
struct DampedNormalScratch<'a> {
    a: &'a mut Vec<Vec<f64>>,
    low: &'a mut Vec<Vec<f64>>,
    y: &'a mut Vec<f64>,
    step: &'a mut Vec<f64>,
}

fn solve_damped_normal_equations_into(
    jtj: &[Vec<f64>],
    jtr: &[f64],
    mu: f64,
    n: usize,
    scratch: DampedNormalScratch<'_>,
) {
    resize_square(scratch.a, n);
    for (dst_row, src_row) in scratch.a.iter_mut().zip(jtj.iter()) {
        dst_row.copy_from_slice(src_row);
    }
    for (i, row) in scratch.a.iter_mut().enumerate() {
        row[i] += mu;
    }

    // Solve via Cholesky decomposition
    if cholesky_decompose_into(scratch.a, n, scratch.low) {
        cholesky_solve_with_l_into(scratch.low, jtr, n, scratch.y, scratch.step);
        // Return -step (we solve A*step = -jtr, but passed jtr not -jtr).
        for value in scratch.step.iter_mut() {
            *value = -*value;
        }
    } else {
        // Fallback: diagonal solve
        scratch.step.resize(n, 0.0);
        for i in 0..n {
            scratch.step[i] = -jtr[i] / (jtj[i][i] + mu);
        }
    }
}

/// Compute Cholesky decomposition A = L * L^T.
fn cholesky_decompose(a: &[Vec<f64>], n: usize) -> Option<Vec<Vec<f64>>> {
    let mut low = Vec::new();
    if cholesky_decompose_into(a, n, &mut low) {
        Some(low)
    } else {
        None
    }
}

/// Compute Cholesky decomposition A = L * L^T into fixed-shape scratch.
fn cholesky_decompose_into(a: &[Vec<f64>], n: usize, low: &mut Vec<Vec<f64>>) -> bool {
    resize_square(low, n);
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for (&low_ik, &low_jk) in low[i].iter().zip(low[j].iter()).take(j) {
                sum -= low_ik * low_jk;
            }
            if i == j {
                if sum <= 0.0 {
                    return false; // Not positive definite
                }
                low[i][j] = sum.sqrt();
            } else {
                low[i][j] = sum / low[j][j];
            }
        }
    }
    true
}

/// Solve L * L^T * x = b given L.
fn cholesky_solve_with_l(low: &[Vec<f64>], b: &[f64], n: usize) -> Vec<f64> {
    let mut y = Vec::new();
    let mut x = Vec::new();
    cholesky_solve_with_l_into(low, b, n, &mut y, &mut x);
    x
}

/// Solve L * L^T * x = b given L into fixed-shape scratch.
fn cholesky_solve_with_l_into(
    low: &[Vec<f64>],
    b: &[f64],
    n: usize,
    y: &mut Vec<f64>,
    x: &mut Vec<f64>,
) {
    // Forward substitution: L * y = b
    y.resize(n, 0.0);
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= low[i][j] * y[j];
        }
        y[i] = sum / low[i][i];
    }

    // Back substitution: L^T * x = y
    x.resize(n, 0.0);
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= low[j][i] * x[j];
        }
        x[i] = sum / low[i][i];
    }
}

/// Compute parameter covariance matrix from Jacobian.
fn compute_covariance(
    jac: &[Vec<f64>],
    cost: f64,
    m: usize,
    n: usize,
    absolute_sigma: bool,
) -> Vec<Vec<f64>> {
    if !absolute_sigma && m <= n {
        return vec![vec![f64::INFINITY; n]; n];
    }

    let jtj = jtj_matrix(jac);

    // Invert J^T J via Cholesky
    let mut pcov = vec![vec![0.0; n]; n];
    if let Some(low) = cholesky_decompose(&jtj, n) {
        for col in 0..n {
            let mut e = vec![0.0; n];
            e[col] = 1.0;
            let column = cholesky_solve_with_l(&low, &e, n);
            for (row, val) in column.iter().enumerate() {
                pcov[row][col] = *val;
            }
        }
    } else {
        // If J^T J is singular, return infinity on diagonal
        for row in &mut pcov {
            for val in row.iter_mut() {
                *val = f64::INFINITY;
            }
        }
        return pcov;
    }

    // Scale by residual variance if not absolute_sigma
    if !absolute_sigma && m > n {
        let s2 = 2.0 * cost / (m - n) as f64;
        for row in &mut pcov {
            for val in row.iter_mut() {
                *val *= s2;
            }
        }
    }

    pcov
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_fd_jacobian_byte_identical_to_serial() {
        // Exercise the parallel path (m >= 8192, n >= 4 clears the work-gate) and
        // prove it produces a bit-for-bit identical Jacobian to the serial routine.
        let n = 6usize;
        let m = 9000usize;
        let params: Vec<f64> = (0..n).map(|k| 0.3 + 0.13 * k as f64).collect();
        let xs: Vec<f64> = (0..m).map(|i| i as f64 * 0.001).collect();
        let residuals = move |p: &[f64]| -> Vec<f64> {
            xs.iter()
                .map(|&x| {
                    p.iter()
                        .enumerate()
                        .map(|(k, &c)| c * ((k as f64 + 1.0) * x).sin())
                        .sum()
                })
                .collect()
        };
        let r0 = residuals(&params);
        let eps = 1.4901161193847656e-8;

        let mut jac_serial = Vec::new();
        let mut scratch_s = Vec::new();
        finite_diff_jacobian_into(
            &residuals,
            &params,
            &r0,
            eps,
            &mut jac_serial,
            &mut scratch_s,
        );

        let mut jac_par = Vec::new();
        let mut scratch_p = Vec::new();
        finite_diff_jacobian_parallel_into(
            &residuals,
            &params,
            &r0,
            eps,
            &mut jac_par,
            &mut scratch_p,
        );

        assert_eq!(jac_serial.len(), jac_par.len());
        for (rs, rp) in jac_serial.iter().zip(&jac_par) {
            assert_eq!(rs.len(), rp.len());
            for (a, b) in rs.iter().zip(rp) {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "parallel Jacobian must be byte-identical"
                );
            }
        }
    }

    #[test]
    fn least_squares_linear_fit() {
        // Fit y = a*x + b to data
        let xdata = [1.0, 2.0, 3.0, 4.0, 5.0];
        let ydata = [2.1, 3.9, 6.1, 7.9, 10.1]; // approx y = 2x + 0.1

        let residuals = |p: &[f64]| -> Vec<f64> {
            xdata
                .iter()
                .zip(ydata.iter())
                .map(|(&x, &y)| p[0] * x + p[1] - y)
                .collect()
        };

        let result = least_squares(residuals, &[1.0, 0.0], LeastSquaresOptions::default())
            .expect("should converge");
        assert!(result.success, "{}", result.message);
        assert!(
            (result.x[0] - 2.0).abs() < 0.1,
            "slope ~ 2.0, got {}",
            result.x[0]
        );
        assert!(
            (result.x[1] - 0.1).abs() < 0.2,
            "intercept ~ 0.1, got {}",
            result.x[1]
        );
    }

    #[test]
    fn least_squares_exponential_decay() {
        // Fit y = a * exp(-b * x)
        let xdata: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
        let ydata: Vec<f64> = xdata.iter().map(|&x| 3.0 * (-0.5 * x).exp()).collect();

        let residuals = |p: &[f64]| -> Vec<f64> {
            xdata
                .iter()
                .zip(ydata.iter())
                .map(|(&x, &y)| p[0] * (-p[1] * x).exp() - y)
                .collect()
        };

        let result = least_squares(residuals, &[1.0, 1.0], LeastSquaresOptions::default())
            .expect("should converge");
        assert!(result.success, "{}", result.message);
        assert!(
            (result.x[0] - 3.0).abs() < 1.0e-4,
            "a ~ 3.0, got {}",
            result.x[0]
        );
        assert!(
            (result.x[1] - 0.5).abs() < 1.0e-4,
            "b ~ 0.5, got {}",
            result.x[1]
        );
    }

    #[test]
    fn least_squares_rejects_empty_x0() {
        let residuals = |_: &[f64]| -> Vec<f64> { vec![] };
        let err = least_squares(residuals, &[], LeastSquaresOptions::default())
            .expect_err("should reject empty x0");
        assert!(matches!(err, OptError::InvalidArgument { .. }));
    }

    #[test]
    fn least_squares_rejects_underdetermined() {
        let residuals = |_: &[f64]| -> Vec<f64> { vec![1.0] };
        let err = least_squares(residuals, &[1.0, 2.0, 3.0], LeastSquaresOptions::default())
            .expect_err("should reject underdetermined");
        assert!(matches!(err, OptError::InvalidArgument { .. }));
    }

    #[test]
    fn curve_fit_quadratic() {
        // Fit y = a*x^2 + b*x + c
        let xdata: Vec<f64> = (-10..=10).map(|i| i as f64).collect();
        let ydata: Vec<f64> = xdata.iter().map(|&x| 2.0 * x * x - 3.0 * x + 1.0).collect();

        let model = |x: f64, p: &[f64]| p[0] * x * x + p[1] * x + p[2];

        let options = CurveFitOptions {
            p0: Some(vec![1.0, 1.0, 0.0]),
            ..CurveFitOptions::default()
        };

        let result = curve_fit(model, &xdata, &ydata, options).expect("should converge");
        assert!(result.ls_result.success, "{}", result.ls_result.message);
        assert!(
            (result.popt[0] - 2.0).abs() < 1.0e-6,
            "a ~ 2.0, got {}",
            result.popt[0]
        );
        assert!(
            (result.popt[1] - (-3.0)).abs() < 1.0e-6,
            "b ~ -3.0, got {}",
            result.popt[1]
        );
        assert!(
            (result.popt[2] - 1.0).abs() < 1.0e-6,
            "c ~ 1.0, got {}",
            result.popt[2]
        );
    }

    #[test]
    fn curve_fit_bounded_interior_optimum_and_active_bound() {
        // Exponential-plus-offset; noiseless data => the least-squares minimum is the
        // exact generating parameters, so a correct bounded solve recovers them.
        let xdata: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let model = |x: f64, p: &[f64]| p[0] * (-p[1] * x).exp() + p[2];
        let truep = [3.0_f64, 0.7, 1.0];
        let ydata: Vec<f64> = xdata.iter().map(|&x| model(x, &truep)).collect();

        // (1) Interior optimum: bounds strictly contain the truth -> exact recovery,
        //     identical to scipy's trf minimiser.
        let lower = [0.0, 0.0, -5.0];
        let upper = [10.0, 5.0, 5.0];
        let opts = CurveFitOptions {
            p0: Some(vec![1.0, 1.0, 0.0]),
            ..CurveFitOptions::default()
        };
        let r = curve_fit_bounded(model, &xdata, &ydata, &lower, &upper, opts).expect("converges");
        for k in 0..3 {
            assert!(
                (r.popt[k] - truep[k]).abs() < 1.0e-5,
                "interior popt[{k}] = {} expected {}",
                r.popt[k],
                truep[k]
            );
            assert!(
                r.popt[k] >= lower[k] && r.popt[k] <= upper[k],
                "popt[{k}] out of bounds"
            );
        }
        // pcov is finite and symmetric (recomputed in parameter space).
        assert!(r.pcov.iter().flatten().all(|v| v.is_finite()));

        // (2) Active bound: cap the amplitude below the truth -> the fit pins it at the
        //     bound (the transform approaches it from below) and stays feasible.
        let upper2 = [2.0, 5.0, 5.0];
        let opts2 = CurveFitOptions {
            p0: Some(vec![1.0, 1.0, 0.0]),
            ..CurveFitOptions::default()
        };
        let r2 =
            curve_fit_bounded(model, &xdata, &ydata, &lower, &upper2, opts2).expect("converges");
        assert!(
            r2.popt[0] <= upper2[0] + 1.0e-9 && r2.popt[0] > 1.5,
            "active-bound popt[0] = {} (want just below 2.0)",
            r2.popt[0]
        );
    }

    #[test]
    fn curve_fit_many_byte_identical_to_per_row() {
        // Batched fit must equal looping curve_fit per row, bit-for-bit (each fit is
        // independent; parallelism only distributes them).
        let x: Vec<f64> = (0..80).map(|i| i as f64 * 5.0 / 79.0).collect();
        let model = |xi: f64, p: &[f64]| p[0] * (-p[1] * xi).exp() + p[2];
        let mut s = 12345u64;
        let mut rng = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let nrows = 40usize; // crosses the serial->parallel gate
        let rows: Vec<Vec<f64>> = (0..nrows)
            .map(|_| {
                let (a, b, c) = (1.0 + 2.0 * rng(), 0.3 + rng(), rng());
                x.iter()
                    .map(|&xi| a * (-b * xi).exp() + c + 0.02 * (rng() - 0.5))
                    .collect()
            })
            .collect();
        let opts = || CurveFitOptions {
            p0: Some(vec![1.0, 1.0, 0.0]),
            ..CurveFitOptions::default()
        };
        let batched = curve_fit_many(model, &x, &rows, opts()).expect("batched");
        assert_eq!(batched.len(), nrows);
        for (i, row) in rows.iter().enumerate() {
            let single = curve_fit(model, &x, row, opts()).expect("single").popt;
            for k in 0..3 {
                assert_eq!(
                    batched[i][k].to_bits(),
                    single[k].to_bits(),
                    "row {i} param {k}: batched {} vs single {}",
                    batched[i][k],
                    single[k]
                );
            }
        }
        // bounded batched is likewise bit-identical to per-row bounded.
        let lo = [0.0, 0.0, -2.0];
        let hi = [6.0, 4.0, 2.0];
        let bb = curve_fit_bounded_many(model, &x, &rows, &lo, &hi, opts()).expect("batched bnd");
        for (i, row) in rows.iter().enumerate() {
            let single = curve_fit_bounded(model, &x, row, &lo, &hi, opts())
                .expect("single bnd")
                .popt;
            for k in 0..3 {
                assert_eq!(
                    bb[i][k].to_bits(),
                    single[k].to_bits(),
                    "bnd row {i} param {k}"
                );
            }
        }
        assert!(curve_fit_many(model, &x, &[], opts()).unwrap().is_empty());
    }

    #[test]
    fn least_squares_many_byte_identical_to_per_param() {
        // Fit y = a*exp(-b*t): residuals(x, y) = a*exp(-b*t) - y, swept over many
        // noisy datasets. The batched solve must equal looping least_squares per
        // param, bit-for-bit, and converge near the generating (a, b).
        let t: Vec<f64> = (0..40).map(|i| 3.0 * i as f64 / 39.0).collect();
        let resid = |x: &[f64], y: &[f64]| -> Vec<f64> {
            t.iter()
                .zip(y)
                .map(|(&ti, &yi)| x[0] * (-x[1] * ti).exp() - yi)
                .collect()
        };
        let mut s = 19u64;
        let mut rng = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let nrows = 40usize;
        let truth: Vec<(f64, f64)> = (0..nrows)
            .map(|_| (1.0 + 2.0 * rng(), 0.5 + 1.5 * rng()))
            .collect();
        let ys: Vec<Vec<f64>> = truth
            .iter()
            .map(|&(a, b)| t.iter().map(|&ti| a * (-b * ti).exp()).collect())
            .collect();
        let opt = LeastSquaresOptions::default();

        let batched = least_squares_many(resid, &[1.5, 1.0], &ys, opt);
        assert_eq!(batched.len(), nrows);
        for (i, y) in ys.iter().enumerate() {
            let single = least_squares(|x| resid(x, y), &[1.5, 1.0], opt).expect("single");
            let many = batched[i].as_ref().expect("batched member");
            assert_eq!(many.x[0].to_bits(), single.x[0].to_bits(), "a mismatch {i}");
            assert_eq!(many.x[1].to_bits(), single.x[1].to_bits(), "b mismatch {i}");
            // noiseless data => recovers the generating parameters.
            assert!((many.x[0] - truth[i].0).abs() < 1e-6, "a {i}");
            assert!((many.x[1] - truth[i].1).abs() < 1e-6, "b {i}");
        }
        assert!(least_squares_many(resid, &[1.5, 1.0], &[], opt).is_empty());
    }

    #[test]
    fn curve_fit_bounded_validates_bounds() {
        let xdata = [0.0, 1.0, 2.0, 3.0];
        let ydata = [0.0, 1.0, 2.0, 3.0];
        let model = |x: f64, p: &[f64]| p[0] * x + p[1];
        // lower >= upper is rejected.
        let opts = CurveFitOptions {
            p0: Some(vec![1.0, 0.0]),
            ..CurveFitOptions::default()
        };
        let err = curve_fit_bounded(model, &xdata, &ydata, &[1.0, 0.0], &[1.0, 5.0], opts);
        assert!(err.is_err(), "equal lower/upper must be rejected");
    }

    #[test]
    fn curve_fit_sinusoidal() {
        // Fit y = A * sin(omega * x + phi)
        let xdata: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let ydata: Vec<f64> = xdata.iter().map(|&x| 2.5 * (1.5 * x + 0.3).sin()).collect();

        let model = |x: f64, p: &[f64]| p[0] * (p[1] * x + p[2]).sin();

        let options = CurveFitOptions {
            p0: Some(vec![2.0, 1.0, 0.0]),
            ..CurveFitOptions::default()
        };

        let result = curve_fit(model, &xdata, &ydata, options).expect("should converge");
        assert!(result.ls_result.success, "{}", result.ls_result.message);
        assert!(
            (result.popt[0] - 2.5).abs() < 0.1,
            "A ~ 2.5, got {}",
            result.popt[0]
        );
    }

    #[test]
    fn curve_fit_returns_covariance() {
        let xdata: Vec<f64> = (0..20).map(|i| i as f64).collect();
        // Linear: y = 2x + 1 with noise
        let ydata: Vec<f64> = xdata.iter().map(|&x| 2.0 * x + 1.0).collect();

        let model = |x: f64, p: &[f64]| p[0] * x + p[1];

        let options = CurveFitOptions {
            p0: Some(vec![1.0, 0.0]),
            absolute_sigma: true,
            ..CurveFitOptions::default()
        };

        let result = curve_fit(model, &xdata, &ydata, options).expect("should converge");
        // Covariance should be 2x2 and finite
        assert_eq!(result.pcov.len(), 2);
        assert_eq!(result.pcov[0].len(), 2);
        assert!(result.pcov[0][0].is_finite());
        assert!(result.pcov[1][1].is_finite());
    }

    #[test]
    fn curve_fit_zero_dof_returns_infinite_covariance() {
        let xdata = [0.0, 1.0];
        let ydata = [1.0, 3.0];
        let model = |x: f64, p: &[f64]| p[0] * x + p[1];

        let result = curve_fit(
            model,
            &xdata,
            &ydata,
            CurveFitOptions {
                p0: Some(vec![1.0, 1.0]),
                absolute_sigma: false,
                ..CurveFitOptions::default()
            },
        )
        .expect("should fit exactly");

        assert!(
            result
                .pcov
                .iter()
                .flatten()
                .all(|entry| entry.is_infinite())
        );
    }

    #[test]
    fn curve_fit_xdata_ydata_length_mismatch() {
        let model = |_x: f64, _p: &[f64]| 0.0;
        let err = curve_fit(
            model,
            &[1.0, 2.0],
            &[1.0],
            CurveFitOptions {
                p0: Some(vec![1.0]),
                ..CurveFitOptions::default()
            },
        )
        .expect_err("should reject mismatched lengths");
        assert!(matches!(err, OptError::InvalidArgument { .. }));
    }

    #[test]
    fn curve_fit_requires_p0() {
        let model = |_x: f64, _p: &[f64]| 0.0;
        let err = curve_fit(model, &[1.0], &[1.0], CurveFitOptions::default())
            .expect_err("should require p0");
        assert!(matches!(err, OptError::InvalidArgument { .. }));
    }

    #[test]
    fn least_squares_rejects_non_positive_diff_step() {
        let residuals = |x: &[f64]| vec![x[0] - 1.0];
        let err = least_squares(
            residuals,
            &[0.0],
            LeastSquaresOptions {
                diff_step: 0.0,
                ..LeastSquaresOptions::default()
            },
        )
        .expect_err("should reject zero diff_step");
        assert!(matches!(err, OptError::InvalidArgument { .. }));
    }

    #[test]
    fn least_squares_bounded_rejects_unordered_bounds_before_evaluating() {
        for (lower, upper) in [(f64::NAN, 1.0), (0.0, 0.0)] {
            let err = least_squares_bounded(
                |_| -> Vec<f64> { panic!("invalid bounds must be rejected before evaluation") },
                &[0.0],
                &[lower],
                &[upper],
                LeastSquaresOptions::default(),
            )
            .expect_err("unordered bounds must fail");
            assert!(matches!(err, OptError::InvalidArgument { .. }));
        }
    }

    #[test]
    fn least_squares_gaussian_fit() {
        // Fit y = A * exp(-((x - mu)/sigma)^2 / 2)
        let xdata: Vec<f64> = (-30..=30).map(|i| i as f64 * 0.2).collect();
        let ydata: Vec<f64> = xdata
            .iter()
            .map(|&x| 5.0 * (-((x - 1.0) / 2.0).powi(2) / 2.0).exp())
            .collect();

        let residuals = |p: &[f64]| -> Vec<f64> {
            xdata
                .iter()
                .zip(ydata.iter())
                .map(|(&x, &y)| p[0] * (-((x - p[1]) / p[2]).powi(2) / 2.0).exp() - y)
                .collect()
        };

        let result = least_squares(residuals, &[3.0, 0.0, 1.0], LeastSquaresOptions::default())
            .expect("should converge");
        assert!(result.success, "{}", result.message);
        assert!(
            (result.x[0] - 5.0).abs() < 0.1,
            "A ~ 5.0, got {}",
            result.x[0]
        );
        assert!(
            (result.x[1] - 1.0).abs() < 0.1,
            "mu ~ 1.0, got {}",
            result.x[1]
        );
        assert!(
            (result.x[2].abs() - 2.0).abs() < 0.1,
            "|sigma| ~ 2.0, got {}",
            result.x[2]
        );
    }

    #[test]
    fn leastsq_matches_scipy_rosenbrock_and_linear() {
        // Rosenbrock residuals; minimum at (1, 1) — scipy.optimize.leastsq agrees.
        let r1 = |x: &[f64]| vec![10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]];
        let res = leastsq(r1, &[-1.2, 1.0], LeastSquaresOptions::default()).unwrap();
        assert!((res.x[0] - 1.0).abs() < 1e-6, "x0 = {}", res.x[0]);
        assert!((res.x[1] - 1.0).abs() < 1e-6, "x1 = {}", res.x[1]);
        // A converged run reports a MINPACK success flag (1..=4).
        assert!((1..=4).contains(&res.ier), "ier = {}", res.ier);
        // cov_x is the unscaled (JᵀJ)⁻¹ and is finite for a full-rank Jacobian.
        let cov = res
            .cov_x
            .expect("cov_x should be Some for full-rank Jacobian");
        assert_eq!(cov.len(), 2);
        assert!(cov.iter().flatten().all(|v| v.is_finite()));

        // Overdetermined linear fit r_i = a*u_i + b - v_i; scipy gives a≈0.97714286.
        let us = [0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0];
        let vs = [0.1_f64, 0.9, 2.1, 2.9, 4.2, 4.8];
        let r3 = move |p: &[f64]| -> Vec<f64> {
            us.iter()
                .zip(vs.iter())
                .map(|(&u, &v)| p[0] * u + p[1] - v)
                .collect()
        };
        let lin = leastsq(r3, &[0.0, 0.0], LeastSquaresOptions::default()).unwrap();
        assert!(
            (lin.x[0] - 0.977_142_857).abs() < 1e-6,
            "slope = {}",
            lin.x[0]
        );
        assert!(
            (lin.x[1] - 0.057_142_857).abs() < 1e-6,
            "intercept = {}",
            lin.x[1]
        );
    }
}
