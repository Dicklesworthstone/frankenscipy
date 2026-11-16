#![forbid(unsafe_code)]

//! Bounded linear least squares with sparse and matrix-free design matrices.
//!
//! The linear Trust Region Reflective method uses Coleman–Li scaling and an
//! LSMR solve of the augmented, scaled least-squares problem. A full Newton
//! step, a reflected step, and a scaled gradient step compete using the exact
//! quadratic model. No normal equations or dense Jacobian are constructed.
//! Reference: Branch, Coleman & Li (1999), and SciPy 1.17.1 `_lsq/trf_linear.py`.

use crate::OptError;
use crate::lsq_matrix::LeastSquaresMatrix;
use crate::trf::{
    build_quadratic_1d, cl_scaling_vector, evaluate_quadratic, find_active_constraints,
    in_bounds, make_strictly_feasible, minimize_quadratic_1d, step_size_to_bound,
};
use crate::trf_lsmr::AugmentedJacobian;
use fsci_runtime::RuntimeMode;
use fsci_sparse::LinearOperator;
use fsci_sparse::linalg::{LsmrOptions, LsmrResult, lsmr_with_options};

/// Controls for [`lsq_linear_with_options`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearLeastSquaresOptions {
    /// Coleman–Li scaled gradient and relative cost-change tolerance.
    pub tol: f64,
    /// Maximum outer iterations. Each iteration uses only matrix-vector products.
    pub max_iter: usize,
    /// LSMR's `atol` and `btol`; `None` selects `0.01 * tol`, as in SciPy.
    pub lsmr_tol: Option<f64>,
    /// Maximum LSMR iterations per subproblem; `None` uses min(rows, columns).
    pub lsmr_max_iter: Option<usize>,
    pub mode: RuntimeMode,
}

impl Default for LinearLeastSquaresOptions {
    fn default() -> Self {
        Self {
            tol: 1e-10,
            max_iter: 100,
            lsmr_tol: None,
            lsmr_max_iter: None,
            mode: RuntimeMode::Strict,
        }
    }
}

/// Solution and convergence diagnostics for bounded linear least squares.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearLeastSquaresResult {
    pub x: Vec<f64>,
    /// Residual `A*x - b` at the returned point.
    pub fun: Vec<f64>,
    pub cost: f64,
    pub grad: Vec<f64>,
    /// Infinity norm of the Coleman–Li scaled gradient at the returned point.
    pub optimality: f64,
    /// -1: lower bound, 0: inactive, 1: upper bound.
    pub active_mask: Vec<i8>,
    pub nit: usize,
    /// -1: no decreasing step; 0: iteration limit; 1: first-order optimality;
    /// 2: relative cost change; 3: unconstrained least-squares solution is feasible.
    pub status: i32,
    pub success: bool,
    pub message: String,
    /// Initial unconstrained solve, including LSMR stopping and condition diagnostics.
    pub unbounded_sol: LsmrResult,
}

fn finite(values: &[f64], what: &str) -> Result<(), OptError> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: format!("lsq_linear: {what} contains NaN or infinity"),
        });
    }
    Ok(())
}

fn residual(a: &LeastSquaresMatrix, x: &[f64], b: &[f64]) -> Result<Vec<f64>, OptError> {
    let mut r = a.matvec(x)?;
    for (value, &target) in r.iter_mut().zip(b) {
        *value -= target;
    }
    finite(&r, "residual")?;
    Ok(r)
}

fn cost(r: &[f64]) -> Result<f64, OptError> {
    let norm = r.iter().fold(0.0_f64, |acc, &value| acc.hypot(value));
    let value = (0.5 * norm) * norm;
    if !value.is_finite() {
        return Err(OptError::NotConverged {
            detail: String::from("lsq_linear: residual cost is not representable"),
        });
    }
    Ok(value)
}

fn optimality(x: &[f64], g: &[f64], lb: &[f64], ub: &[f64]) -> Result<f64, OptError> {
    let (v, _) = cl_scaling_vector(x, g, lb, ub);
    let scaled: Vec<f64> = g.iter().zip(v).map(|(gi, vi)| gi * vi).collect();
    finite(&scaled, "scaled gradient")?;
    Ok(scaled.iter().fold(0.0_f64, |acc, value| acc.max(value.abs())))
}

/// Reflect into each interval. At extreme ranges where the period is not
/// representable, projection still provides a finite feasible starting point.
fn reflect(x: &[f64], lb: &[f64], ub: &[f64]) -> Vec<f64> {
    x.iter()
        .zip(lb.iter().zip(ub))
        .map(|(&value, (&lower, &upper))| {
            if lower <= value && value <= upper {
                return value;
            }
            let reflected = if !upper.is_finite() {
                lower + (lower - value)
            } else if !lower.is_finite() {
                upper - (value - upper)
            } else {
                let width = upper - lower;
                let period = 2.0 * width;
                let offset = (value - lower).rem_euclid(period);
                lower + offset.min(period - offset)
            };
            if reflected.is_finite() {
                reflected.clamp(lower, upper)
            } else {
                value.clamp(lower, upper)
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn select_step(
    x: &[f64],
    scaled: &LeastSquaresMatrix,
    gradient: &[f64],
    diagonal: &[f64],
    mut step_h: Vec<f64>,
    scale: &[f64],
    lb: &[f64],
    ub: &[f64],
    theta: f64,
) -> Result<Vec<f64>, OptError> {
    let mut step: Vec<f64> = scale.iter().zip(&step_h).map(|(d, p)| d * p).collect();
    finite(&step, "linear step")?;
    let trial: Vec<f64> = x.iter().zip(&step).map(|(x, p)| x + p).collect();
    if in_bounds(&trial, lb, ub) {
        return Ok(step);
    }

    let (stride, hits) = step_size_to_bound(x, &step, lb, ub);
    let reflection_h: Vec<f64> = step_h
        .iter()
        .zip(hits)
        .map(|(&p, hit)| if hit == 0 { p } else { -p })
        .collect();
    let mut reflection: Vec<f64> = scale
        .iter()
        .zip(&reflection_h)
        .map(|(d, r)| d * r)
        .collect();
    for (p, ph) in step.iter_mut().zip(&mut step_h) {
        *p *= stride;
        *ph *= stride;
    }
    let on_bound: Vec<f64> = x.iter().zip(&step).map(|(x, p)| x + p).collect();
    let (reflection_stride, _) = step_size_to_bound(&on_bound, &reflection, lb, ub);
    let mut reflection_value = f64::INFINITY;
    if reflection_stride > 0.0 {
        let (a, b, c) = build_quadratic_1d(
            scaled,
            gradient,
            &reflection_h,
            Some(diagonal),
            Some(&step_h),
        )?;
        let lower = if reflection_stride.is_finite() {
            (1.0 - theta) * reflection_stride
        } else {
            0.0
        };
        let (t, value) = minimize_quadratic_1d(a, b, lower, theta * reflection_stride, c);
        reflection_value = value;
        for (((r, &ph), &rh), &d) in reflection
            .iter_mut()
            .zip(&step_h)
            .zip(&reflection_h)
            .zip(scale)
        {
            *r = d * (ph + t * rh);
        }
    }

    for (p, ph) in step.iter_mut().zip(&mut step_h) {
        *p *= theta;
        *ph *= theta;
    }
    let step_value = evaluate_quadratic(scaled, gradient, &step_h, Some(diagonal))?;
    let descent_h: Vec<f64> = gradient.iter().map(|g| -g).collect();
    let mut descent: Vec<f64> = scale.iter().zip(&descent_h).map(|(d, g)| d * g).collect();
    let (descent_stride, _) = step_size_to_bound(x, &descent, lb, ub);
    let (a, b, c) = build_quadratic_1d(scaled, gradient, &descent_h, Some(diagonal), None)?;
    let (t, descent_value) = minimize_quadratic_1d(a, b, 0.0, theta * descent_stride, c);
    for g in &mut descent {
        *g *= t;
    }
    let selected = if step_value < reflection_value && step_value < descent_value {
        step
    } else if reflection_value < step_value && reflection_value < descent_value {
        reflection
    } else {
        descent
    };
    finite(&selected, "selected step")?;
    Ok(selected)
}

/// Minimize `0.5 * ||A*x - b||²` subject to `lb <= x <= ub`, retaining dense,
/// CSR, or matrix-free storage throughout the solve.
///
/// This is the linear TRF algorithm with LSMR subproblems (SciPy's
/// `method="trf", lsq_solver="lsmr"`). The existing [`crate::lsq_linear`]
/// uses dense BVLS. Supply CSR with [`LeastSquaresMatrix::Sparse`] or products
/// with [`LeastSquaresMatrix::from_operator`]. Both forward and transpose
/// products are required; no basis-vector probing or dense conversion occurs.
///
/// Bounds may be infinite but must be strictly ordered, as in SciPy's TRF API.
/// A matrix with rows but no columns has the empty solution and residual `-b`.
/// Invalid inputs and operator failures return errors. Exhausted iteration
/// budgets return a result with `success=false`; inspect `status`, `optimality`
/// and `unbounded_sol` before accepting a fit. `lsmr_tol="auto"` is not exposed;
/// an explicit fixed tolerance or the default `0.01 * tol` is used.
pub fn lsq_linear_with_options(
    a: &LeastSquaresMatrix,
    b: &[f64],
    lb: &[f64],
    ub: &[f64],
    options: LinearLeastSquaresOptions,
) -> Result<LinearLeastSquaresResult, OptError> {
    let shape = a.shape();
    let (m, n) = (shape.rows, shape.cols);
    if m == 0 || b.len() != m || lb.len() != n || ub.len() != n || m.checked_add(n).is_none() {
        return Err(OptError::InvalidArgument {
            detail: String::from("lsq_linear: A must have rows and consistent A/b/bounds dimensions"),
        });
    }
    if let Some(rows) = a.as_dense()
        && rows.iter().any(|row| row.len() != n)
    {
        return Err(OptError::InvalidArgument {
            detail: String::from("lsq_linear: ragged matrix"),
        });
    }
    if !options.tol.is_finite()
        || options.tol <= 0.0
        || options.max_iter == 0
        || options
            .lsmr_tol
            .is_some_and(|tol| !tol.is_finite() || tol <= 0.0)
        || options.lsmr_max_iter == Some(0)
    {
        return Err(OptError::InvalidArgument {
            detail: String::from(
                "lsq_linear: tolerances and iteration limits must be positive and finite",
            ),
        });
    }
    for (index, (&lower, &upper)) in lb.iter().zip(ub).enumerate() {
        if lower.is_nan() || upper.is_nan() || lower >= upper {
            return Err(OptError::InvalidBounds {
                detail: format!("lsq_linear: invalid bounds at {index}"),
            });
        }
    }
    finite(b, "right-hand side")?;
    if !a.all_finite() {
        return Err(OptError::NonFiniteInput {
            detail: String::from("lsq_linear: non-finite matrix entries"),
        });
    }
    let linear_options = LsmrOptions {
        mode: options.mode,
        atol: options.lsmr_tol.unwrap_or(0.01 * options.tol),
        btol: options.lsmr_tol.unwrap_or(0.01 * options.tol),
        max_iter: options.lsmr_max_iter,
        ..LsmrOptions::default()
    };
    let unbounded_sol = lsmr_with_options(a, b, linear_options)?;
    finite(&unbounded_sol.solution, "unconstrained solution")?;
    let feasible = in_bounds(&unbounded_sol.solution, lb, ub);
    // A truncated unconstrained solve is not a certificate of optimality merely
    // because it happens to be feasible. Continue refining it with the outer loop.
    let solved = matches!(unbounded_sol.istop, 0 | 1 | 2 | 4 | 5);
    let mut x = if feasible && solved {
        unbounded_sol.solution.clone()
    } else {
        make_strictly_feasible(&reflect(&unbounded_sol.solution, lb, ub), lb, ub, 0.1)
    };
    finite(&x, "initial feasible point")?;
    let mut r = residual(a, &x, b)?;
    let mut g = a.rmatvec(&r)?;
    let mut value = cost(&r)?;
    let mut status = if feasible && solved { 3 } else { 0 };
    let mut nit = 0;

    if status == 0 {
        for iteration in 0..options.max_iter {
            nit = iteration + 1;
            let g_norm = optimality(&x, &g, lb, ub)?;
            if g_norm < options.tol {
                status = 1;
                break;
            }
            let (v, dv) = cl_scaling_vector(&x, &g, lb, ub);
            let diagonal: Vec<f64> = g.iter().zip(dv).map(|(gi, vi)| gi * vi).collect();
            let diagonal_root: Vec<f64> = diagonal.iter().map(|value| value.sqrt()).collect();
            let scale: Vec<f64> = v.iter().map(|value| value.sqrt()).collect();
            let gradient_h: Vec<f64> = scale.iter().zip(&g).map(|(d, g)| d * g).collect();
            finite(&diagonal_root, "regularization")?;
            finite(&scale, "bound scaling")?;
            finite(&gradient_h, "transformed gradient")?;
            let scaled = a.scaled_columns(&scale);
            if !scaled.all_finite() {
                return Err(OptError::NonFiniteInput {
                    detail: String::from("lsq_linear: scaled matrix overflow"),
                });
            }
            let operator = AugmentedJacobian {
                matrix: &scaled,
                diagonal: &diagonal_root,
                columns: n,
            };
            let mut rhs = r.clone();
            rhs.resize(m + n, 0.0);
            let direction = lsmr_with_options(&operator, &rhs, linear_options)?;
            let step_h: Vec<f64> = direction.solution.iter().map(|value| -value).collect();
            let theta = 1.0 - 0.005_f64.min(g_norm);
            let step = select_step(
                &x,
                &scaled,
                &gradient_h,
                &diagonal,
                step_h,
                &scale,
                lb,
                ub,
                theta,
            )?;

            // The quadratic is exact for a linear residual. Backtracking is
            // bounded, and a rounded zero step must never count as convergence.
            let mut accepted = None;
            let mut alpha = 1.0;
            for _ in 0..64 {
                let trial: Vec<f64> = x.iter().zip(&step).map(|(x, p)| x + alpha * p).collect();
                let trial = make_strictly_feasible(&reflect(&trial, lb, ub), lb, ub, 0.0);
                finite(&trial, "trial point")?;
                let actual_step: Vec<f64> =
                    trial.iter().zip(&x).map(|(new, old)| new - old).collect();
                let reduction = -evaluate_quadratic(a, &g, &actual_step, None)?;
                if reduction.is_finite()
                    && reduction > 0.0
                    && actual_step.iter().any(|s| *s != 0.0)
                {
                    accepted = Some((trial, reduction));
                    break;
                }
                alpha *= 0.5;
            }
            let Some((trial, reduction)) = accepted else {
                status = -1;
                break;
            };
            x = trial;
            r = residual(a, &x, b)?;
            g = a.rmatvec(&r)?;
            let previous_cost = value;
            value = cost(&r)?;
            if optimality(&x, &g, lb, ub)? < options.tol {
                status = 1;
                break;
            }
            if reduction < options.tol * previous_cost {
                status = 2;
                break;
            }
        }
    }
    let optimality = optimality(&x, &g, lb, ub)?;
    let active_mask = find_active_constraints(&x, lb, ub, options.tol);
    let message = match status {
        -1 => "No decreasing representable step was found.",
        0 => "Maximum number of iterations exceeded.",
        1 => "First-order optimality is below tolerance.",
        2 => "Relative cost change is below tolerance.",
        3 => "The unconstrained least-squares solution is feasible.",
        _ => unreachable!(),
    }
    .to_string();
    Ok(LinearLeastSquaresResult {
        x,
        fun: r,
        cost: value,
        grad: g,
        optimality,
        active_mask,
        nit,
        status,
        success: status > 0,
        message,
        unbounded_sol,
    })
}
