#![forbid(unsafe_code)]

use crate::types::OptError;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WolfeParams {
    pub c1: f64,
    pub c2: f64,
    pub amax: f64,
    pub amin: f64,
    pub maxiter: usize,
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineSearchResult {
    pub alpha: f64,
    pub f_at_alpha: f64,
    pub directional_derivative: f64,
    pub evaluations: usize,
}

pub fn validate_wolfe_params(params: WolfeParams) -> Result<(), OptError> {
    if !(0.0 < params.c1 && params.c1 < params.c2 && params.c2 < 1.0) {
        return Err(OptError::InvalidArgument {
            detail: String::from("Wolfe constants must satisfy 0 < c1 < c2 < 1"),
        });
    }
    if !params.amin.is_finite()
        || !params.amax.is_finite()
        || params.amin <= 0.0
        || params.amax <= 0.0
        || params.amin >= params.amax
    {
        return Err(OptError::InvalidArgument {
            detail: String::from(
                "line-search alpha bounds must be finite and satisfy 0 < amin < amax",
            ),
        });
    }
    if params.maxiter == 0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("line-search maxiter must be >= 1"),
        });
    }
    Ok(())
}

fn validate_line_search_inputs(x: &[f64], direction: &[f64], g0: &[f64]) -> Result<(), OptError> {
    let n = x.len();
    if n == 0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("line-search x must not be empty"),
        });
    }
    if direction.len() != n || g0.len() != n {
        return Err(OptError::InvalidArgument {
            detail: String::from(
                "line-search x, direction, and gradient must have matching lengths",
            ),
        });
    }
    Ok(())
}

fn validate_gradient_output_len(gradient: &[f64], expected: usize) -> Result<(), OptError> {
    if gradient.len() != expected {
        return Err(OptError::InvalidArgument {
            detail: String::from("line-search gradient output length must match x length"),
        });
    }
    Ok(())
}

/// Weak Wolfe line search (Armijo + curvature).
///
/// Finds alpha satisfying:
/// - Armijo:    f(x + alpha*d) <= f(x) + c1*alpha*g'*d
/// - Curvature: g(x + alpha*d)'*d >= c2*g(x)'*d
///
/// Matches `scipy.optimize.line_search` with `old_old_fval=None`.
pub fn line_search_wolfe1<F, G>(
    f: &F,
    grad: &G,
    x: &[f64],
    direction: &[f64],
    f0: f64,
    g0: &[f64],
    params: WolfeParams,
) -> Result<LineSearchResult, OptError>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    validate_wolfe_params(params)?;
    validate_line_search_inputs(x, direction, g0)?;
    let dg0 = dot(g0, direction);
    if dg0 >= 0.0 {
        return Err(OptError::InvalidArgument {
            detail: "search direction is not a descent direction".to_string(),
        });
    }

    line_search_wolfe_impl(f, grad, x, direction, f0, dg0, params, false)
}

/// Strong Wolfe line search.
///
/// Finds alpha satisfying:
/// - Armijo:       f(x + alpha*d) <= f(x) + c1*alpha*g'*d
/// - Strong Wolfe: |g(x + alpha*d)'*d| <= c2*|g(x)'*d|
///
/// Uses the zoom phase from Nocedal & Wright Algorithm 3.6.
/// Matches `scipy.optimize.line_search`.
pub fn line_search_wolfe2<F, G>(
    f: &F,
    grad: &G,
    x: &[f64],
    direction: &[f64],
    f0: f64,
    g0: &[f64],
    params: WolfeParams,
) -> Result<LineSearchResult, OptError>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    validate_wolfe_params(params)?;
    validate_line_search_inputs(x, direction, g0)?;
    let dg0 = dot(g0, direction);
    if dg0 >= 0.0 {
        return Err(OptError::InvalidArgument {
            detail: "search direction is not a descent direction".to_string(),
        });
    }

    line_search_wolfe_impl(f, grad, x, direction, f0, dg0, params, true)
}

/// Core Wolfe line search implementation (Nocedal & Wright Algorithm 3.5 + 3.6).
#[allow(clippy::too_many_arguments)]
fn line_search_wolfe_impl<F, G>(
    f: &F,
    grad: &G,
    x: &[f64],
    d: &[f64],
    f0: f64,
    dg0: f64,
    params: WolfeParams,
    strong: bool,
) -> Result<LineSearchResult, OptError>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x.len();
    let mut evals = 0;

    let eval_f = |alpha: f64, evals: &mut usize| -> f64 {
        let xp: Vec<f64> = (0..n).map(|i| x[i] + alpha * d[i]).collect();
        *evals += 1;
        f(&xp)
    };

    let eval_dg = |alpha: f64, evals: &mut usize| -> Result<f64, OptError> {
        let xp: Vec<f64> = (0..n).map(|i| x[i] + alpha * d[i]).collect();
        let gp = grad(&xp);
        *evals += 1;
        validate_gradient_output_len(&gp, n)?;
        Ok(dot(&gp, d))
    };

    // Bracketing phase (Algorithm 3.5)
    let mut alpha_prev = 0.0;
    let mut f_prev = f0;
    let mut alpha = 1.0_f64.min(params.amax);

    for i in 0..params.maxiter {
        let fi = eval_f(alpha, &mut evals);

        // Armijo violation or function not decreasing
        if fi > f0 + params.c1 * alpha * dg0 || (i > 0 && fi >= f_prev) {
            return zoom(
                f, grad, x, d, f0, dg0, alpha_prev, alpha, f_prev, fi, &params, strong, &mut evals,
            );
        }

        let dgi = eval_dg(alpha, &mut evals)?;

        // Curvature condition satisfied
        let curvature_ok = if strong {
            dgi.abs() <= params.c2 * dg0.abs()
        } else {
            dgi >= params.c2 * dg0
        };

        if curvature_ok {
            return Ok(LineSearchResult {
                alpha,
                f_at_alpha: fi,
                directional_derivative: dgi,
                evaluations: evals,
            });
        }

        // Positive slope means minimum is between alpha_prev and alpha
        if dgi >= 0.0 {
            return zoom(
                f, grad, x, d, f0, dg0, alpha, alpha_prev, fi, f_prev, &params, strong, &mut evals,
            );
        }

        alpha_prev = alpha;
        f_prev = fi;
        alpha = (2.0 * alpha).min(params.amax);
    }

    // Failed to find a step satisfying Wolfe — return best so far
    let fi = eval_f(alpha, &mut evals);
    Ok(LineSearchResult {
        alpha,
        f_at_alpha: fi,
        directional_derivative: dg0,
        evaluations: evals,
    })
}

/// Zoom phase (Nocedal & Wright Algorithm 3.6).
/// Finds a step size in [alpha_lo, alpha_hi] satisfying Strong Wolfe conditions.
#[allow(clippy::too_many_arguments)]
fn zoom<F, G>(
    f: &F,
    grad: &G,
    x: &[f64],
    d: &[f64],
    f0: f64,
    dg0: f64,
    mut alpha_lo: f64,
    mut alpha_hi: f64,
    mut f_lo: f64,
    _f_hi: f64,
    params: &WolfeParams,
    strong: bool,
    evals: &mut usize,
) -> Result<LineSearchResult, OptError>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x.len();

    for _ in 0..params.maxiter {
        // Bisection (could use cubic interpolation for speed, but bisection is robust)
        let alpha_j = 0.5 * (alpha_lo + alpha_hi);

        let xj: Vec<f64> = (0..n).map(|i| x[i] + alpha_j * d[i]).collect();
        *evals += 1;
        let fj = f(&xj);

        if fj > f0 + params.c1 * alpha_j * dg0 || fj >= f_lo {
            alpha_hi = alpha_j;
        } else {
            let gj = grad(&xj);
            *evals += 1;
            validate_gradient_output_len(&gj, n)?;
            let dgj = dot(&gj, d);

            let curvature_ok = if strong {
                dgj.abs() <= params.c2 * dg0.abs()
            } else {
                dgj >= params.c2 * dg0
            };

            if curvature_ok {
                return Ok(LineSearchResult {
                    alpha: alpha_j,
                    f_at_alpha: fj,
                    directional_derivative: dgj,
                    evaluations: *evals,
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

    // Return best found
    Ok(LineSearchResult {
        alpha: alpha_lo,
        f_at_alpha: f_lo,
        directional_derivative: dg0,
        evaluations: *evals,
    })
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

/// Result of [`line_search`], mirroring `scipy.optimize.line_search`'s tuple.
#[derive(Debug, Clone, PartialEq)]
pub struct ScipyLineSearchResult {
    /// Step length satisfying the strong Wolfe conditions, or `None` if the
    /// search did not converge.
    pub alpha: Option<f64>,
    /// Number of function evaluations.
    pub fc: usize,
    /// Number of gradient evaluations.
    pub gc: usize,
    /// `f(xk + alpha·pk)` (the value at the accepted/last point).
    pub new_fval: f64,
    /// `f(xk)` (the starting value).
    pub old_fval: f64,
    /// Gradient at `xk + alpha·pk`, or `None` if the search did not converge.
    pub new_grad: Option<Vec<f64>>,
}

// Minimizer of the cubic through (a,fa),(b,fb),(c,fc) with derivative fpa at a;
// `None` if it cannot be computed (matches scipy `_cubicmin`).
fn ls_cubicmin(a: f64, fa: f64, fpa: f64, b: f64, fb: f64, c: f64, fc: f64) -> Option<f64> {
    let cc = fpa;
    let db = b - a;
    let dc = c - a;
    let denom = (db * dc).powi(2) * (db - dc);
    let v0 = fb - fa - cc * db;
    let v1 = fc - fa - cc * dc;
    let a_coef = (dc.powi(2) * v0 - db.powi(2) * v1) / denom;
    let b_coef = (-dc.powi(3) * v0 + db.powi(3) * v1) / denom;
    let radical = b_coef * b_coef - 3.0 * a_coef * cc;
    let xmin = a + (-b_coef + radical.sqrt()) / (3.0 * a_coef);
    if xmin.is_finite() { Some(xmin) } else { None }
}

// Minimizer of the quadratic through (a,fa),(b,fb) with derivative fpa at a;
// `None` if it cannot be computed (matches scipy `_quadmin`).
fn ls_quadmin(a: f64, fa: f64, fpa: f64, b: f64, fb: f64) -> Option<f64> {
    let d = fa;
    let cc = fpa;
    let db = b - a;
    let b_coef = (fb - d - cc * db) / (db * db);
    let xmin = a - cc / (2.0 * b_coef);
    if xmin.is_finite() { Some(xmin) } else { None }
}

#[allow(clippy::too_many_arguments)]
fn ls_zoom(
    mut a_lo: f64,
    mut a_hi: f64,
    mut phi_lo: f64,
    mut phi_hi: f64,
    mut derphi_lo: f64,
    phi: &dyn Fn(f64) -> f64,
    derphi: &dyn Fn(f64) -> f64,
    phi0: f64,
    derphi0: f64,
    c1: f64,
    c2: f64,
) -> (Option<f64>, Option<f64>, Option<f64>) {
    let maxiter = 10;
    let mut i = 0;
    let delta1 = 0.2;
    let delta2 = 0.1;
    let mut phi_rec = phi0;
    let mut a_rec = 0.0;
    loop {
        let dalpha = a_hi - a_lo;
        let (a, b) = if dalpha < 0.0 {
            (a_hi, a_lo)
        } else {
            (a_lo, a_hi)
        };

        let cchk = delta1 * dalpha;
        let mut a_j: Option<f64> = None;
        if i > 0 {
            a_j = ls_cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec);
        }
        let use_quad =
            i == 0 || a_j.is_none() || a_j.unwrap() > b - cchk || a_j.unwrap() < a + cchk;
        if use_quad {
            let qchk = delta2 * dalpha;
            a_j = ls_quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi);
            if a_j.is_none() || a_j.unwrap() > b - qchk || a_j.unwrap() < a + qchk {
                a_j = Some(a_lo + 0.5 * dalpha);
            }
        }
        let a_j = a_j.unwrap();

        let phi_aj = phi(a_j);
        if (phi_aj > phi0 + c1 * a_j * derphi0) || (phi_aj >= phi_lo) {
            phi_rec = phi_hi;
            a_rec = a_hi;
            a_hi = a_j;
            phi_hi = phi_aj;
        } else {
            let derphi_aj = derphi(a_j);
            if derphi_aj.abs() <= -c2 * derphi0 {
                return (Some(a_j), Some(phi_aj), Some(derphi_aj));
            }
            if derphi_aj * (a_hi - a_lo) >= 0.0 {
                phi_rec = phi_hi;
                a_rec = a_hi;
                a_hi = a_lo;
                phi_hi = phi_lo;
            } else {
                phi_rec = phi_lo;
                a_rec = a_lo;
            }
            a_lo = a_j;
            phi_lo = phi_aj;
            derphi_lo = derphi_aj;
        }
        i += 1;
        if i > maxiter {
            return (None, None, None);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn scalar_search_wolfe2(
    phi: &dyn Fn(f64) -> f64,
    derphi: &dyn Fn(f64) -> f64,
    phi0_opt: Option<f64>,
    old_phi0: Option<f64>,
    derphi0: f64,
    c1: f64,
    c2: f64,
    amax: Option<f64>,
    maxiter: usize,
) -> (Option<f64>, f64, f64, Option<f64>) {
    let mut phi0 = phi0_opt.unwrap_or_else(|| phi(0.0));

    let mut alpha0 = 0.0_f64;
    let mut alpha1 = match old_phi0 {
        Some(op0) if derphi0 != 0.0 => (1.0_f64).min(1.01 * 2.0 * (phi0 - op0) / derphi0),
        _ => 1.0,
    };
    if alpha1 < 0.0 {
        alpha1 = 1.0;
    }
    if let Some(am) = amax {
        alpha1 = alpha1.min(am);
    }

    let mut phi_a1 = phi(alpha1);
    let mut phi_a0 = phi0;
    let mut derphi_a0 = derphi0;

    let mut i = 0;
    while i < maxiter {
        if alpha1 == 0.0 || (amax.is_some() && alpha0 > amax.unwrap()) {
            // Rounding/amax failure: report no convergence.
            let phi_star = phi0;
            phi0 = old_phi0.unwrap_or(phi0);
            return (None, phi_star, phi0, None);
        }

        let not_first = i > 0;
        if (phi_a1 > phi0 + c1 * alpha1 * derphi0) || (phi_a1 >= phi_a0 && not_first) {
            let (a, p, dp) = ls_zoom(
                alpha0, alpha1, phi_a0, phi_a1, derphi_a0, phi, derphi, phi0, derphi0, c1, c2,
            );
            return (a, p.unwrap_or(phi_a1), phi0, dp);
        }

        let derphi_a1 = derphi(alpha1);
        if derphi_a1.abs() <= -c2 * derphi0 {
            return (Some(alpha1), phi_a1, phi0, Some(derphi_a1));
        }
        if derphi_a1 >= 0.0 {
            let (a, p, dp) = ls_zoom(
                alpha1, alpha0, phi_a1, phi_a0, derphi_a1, phi, derphi, phi0, derphi0, c1, c2,
            );
            return (a, p.unwrap_or(phi_a1), phi0, dp);
        }

        let mut alpha2 = 2.0 * alpha1;
        if let Some(am) = amax {
            alpha2 = alpha2.min(am);
        }
        alpha0 = alpha1;
        alpha1 = alpha2;
        phi_a0 = phi_a1;
        phi_a1 = phi(alpha1);
        derphi_a0 = derphi_a1;
        i += 1;
    }
    // maxiter reached without converging.
    (Some(alpha1), phi_a1, phi0, None)
}

/// Find a step length satisfying the strong Wolfe conditions, matching
/// `scipy.optimize.line_search` (`line_search_wolfe2`).
///
/// `phi(s) = f(xk + s·pk)`, `derphi(s) = ∇f(xk + s·pk)·pk`. `gfk` is the
/// gradient at `xk` (computed if `None`); `old_fval`/`old_old_fval` are `f` at
/// `xk` and the previous point (used to pick the initial step). Faithful port of
/// scipy's `scalar_search_wolfe2` (cubic→quadratic→bisection zoom). Defaults:
/// `c1 = 1e-4`, `c2 = 0.9`, `maxiter = 10`, `amax = None`.
#[allow(clippy::too_many_arguments)]
pub fn line_search<F, G>(
    f: &F,
    grad: &G,
    xk: &[f64],
    pk: &[f64],
    gfk: Option<&[f64]>,
    old_fval: Option<f64>,
    old_old_fval: Option<f64>,
    c1: f64,
    c2: f64,
    amax: Option<f64>,
    maxiter: usize,
) -> ScipyLineSearchResult
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    use std::cell::{Cell, RefCell};
    let n = xk.len();
    let fc = Cell::new(0usize);
    let gc = Cell::new(0usize);
    let gval: RefCell<Option<Vec<f64>>> = RefCell::new(None);

    let at = |alpha: f64| -> Vec<f64> { (0..n).map(|i| xk[i] + alpha * pk[i]).collect() };
    let phi = |alpha: f64| -> f64 {
        fc.set(fc.get() + 1);
        f(&at(alpha))
    };
    let derphi = |alpha: f64| -> f64 {
        gc.set(gc.get() + 1);
        let g = grad(&at(alpha));
        let d = dot(&g, pk);
        *gval.borrow_mut() = Some(g);
        d
    };

    let gfk_vec: Vec<f64> = match gfk {
        Some(g) => g.to_vec(),
        None => grad(xk), // direct call, not counted in gc (matches scipy)
    };
    let derphi0 = dot(&gfk_vec, pk);

    let (alpha_star, phi_star, old_fval_out, derphi_star) = scalar_search_wolfe2(
        &phi,
        &derphi,
        old_fval,
        old_old_fval,
        derphi0,
        c1,
        c2,
        amax,
        maxiter,
    );

    let new_grad = if derphi_star.is_some() {
        gval.into_inner()
    } else {
        None
    };

    ScipyLineSearchResult {
        alpha: alpha_star,
        fc: fc.get(),
        gc: gc.get(),
        new_fval: phi_star,
        old_fval: old_fval_out,
        new_grad,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quadratic(x: &[f64]) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }

    fn quadratic_grad(x: &[f64]) -> Vec<f64> {
        x.iter().map(|xi| 2.0 * xi).collect()
    }

    fn rosenbrock(x: &[f64]) -> f64 {
        let (a, b) = (x[0], x[1]);
        (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2)
    }

    fn rosenbrock_grad(x: &[f64]) -> Vec<f64> {
        let (a, b) = (x[0], x[1]);
        vec![
            -2.0 * (1.0 - a) + 200.0 * (b - a * a) * (-2.0 * a),
            200.0 * (b - a * a),
        ]
    }

    #[test]
    fn line_search_matches_scipy() {
        // scipy.optimize.line_search docstring example: f=x0^2+x1^2.
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let grad = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
        let r = line_search(
            &f,
            &grad,
            &[1.8, 1.7],
            &[-1.0, -1.0],
            None,
            None,
            None,
            1e-4,
            0.9,
            None,
            10,
        );
        assert_eq!(r.alpha, Some(1.0));
        assert_eq!((r.fc, r.gc), (2, 1));
        assert!((r.new_fval - 1.13).abs() < 1e-9);
        assert!((r.old_fval - 6.13).abs() < 1e-9);
        let ng = r.new_grad.unwrap();
        assert!((ng[0] - 1.6).abs() < 1e-9 && (ng[1] - 1.4).abs() < 1e-9);

        // Rosenbrock steepest-descent step (exercises bracket expansion): scipy
        // alpha=0.00093831027587526, fc=11, gc=1.
        let f2 = |x: &[f64]| 100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2);
        let g2 = |x: &[f64]| {
            vec![
                -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]),
                200.0 * (x[1] - x[0] * x[0]),
            ]
        };
        let xk = [-1.2, 1.0];
        let gk = g2(&xk);
        let pk = [-gk[0], -gk[1]];
        let r2 = line_search(&f2, &g2, &xk, &pk, None, None, None, 1e-4, 0.9, None, 10);
        assert!((r2.alpha.unwrap() - 0.00093831027587526).abs() < 1e-12);
        assert_eq!((r2.fc, r2.gc), (11, 1));
        assert!((r2.new_fval - 4.75058732).abs() < 1e-6);
    }

    #[test]
    fn wolfe2_quadratic_descent() {
        let x = vec![5.0, 3.0];
        let g = quadratic_grad(&x);
        let f0 = quadratic(&x);
        let d: Vec<f64> = g.iter().map(|gi| -gi).collect(); // steepest descent

        let result = line_search_wolfe2(
            &quadratic,
            &quadratic_grad,
            &x,
            &d,
            f0,
            &g,
            WolfeParams::default(),
        )
        .expect("wolfe2 works");

        assert!(result.alpha > 0.0, "alpha should be positive");
        assert!(
            result.f_at_alpha < f0,
            "function should decrease: {} vs {f0}",
            result.f_at_alpha
        );
        // Armijo condition
        let dg0 = dot(&g, &d);
        assert!(
            result.f_at_alpha <= f0 + 1e-4 * result.alpha * dg0,
            "Armijo condition violated"
        );
        // Strong Wolfe curvature
        assert!(
            result.directional_derivative.abs() <= 0.9 * dg0.abs(),
            "Strong Wolfe curvature violated"
        );
    }

    #[test]
    fn wolfe1_quadratic_descent() {
        let x = vec![5.0, 3.0];
        let g = quadratic_grad(&x);
        let f0 = quadratic(&x);
        let d: Vec<f64> = g.iter().map(|gi| -gi).collect();

        let result = line_search_wolfe1(
            &quadratic,
            &quadratic_grad,
            &x,
            &d,
            f0,
            &g,
            WolfeParams::default(),
        )
        .expect("wolfe1 works");

        assert!(result.alpha > 0.0);
        assert!(result.f_at_alpha < f0);
    }

    #[test]
    fn wolfe2_rosenbrock_descent() {
        let x = vec![0.0, 0.0];
        let g = rosenbrock_grad(&x);
        let f0 = rosenbrock(&x);
        let d: Vec<f64> = g.iter().map(|gi| -gi).collect();

        let result = line_search_wolfe2(
            &rosenbrock,
            &rosenbrock_grad,
            &x,
            &d,
            f0,
            &g,
            WolfeParams::default(),
        )
        .expect("wolfe2 on rosenbrock works");

        assert!(result.alpha > 0.0);
        assert!(result.f_at_alpha < f0);
    }

    #[test]
    fn wolfe2_rejects_ascent_direction() {
        let x = vec![1.0, 2.0];
        let g = quadratic_grad(&x);
        let f0 = quadratic(&x);
        let d = g.clone(); // ascent direction (same as gradient)

        let err = line_search_wolfe2(
            &quadratic,
            &quadratic_grad,
            &x,
            &d,
            f0,
            &g,
            WolfeParams::default(),
        )
        .expect_err("ascent should fail");
        assert!(matches!(err, OptError::InvalidArgument { .. }));
    }

    #[test]
    fn wolfe2_invalid_params() {
        let err = validate_wolfe_params(WolfeParams {
            c1: 0.5,
            c2: 0.1, // c2 < c1 is invalid
            ..WolfeParams::default()
        })
        .expect_err("invalid params");
        assert!(matches!(err, OptError::InvalidArgument { .. }));
    }

    #[test]
    fn wolfe_params_reject_non_finite_alpha_bounds() {
        for (amin, amax) in [
            (f64::NAN, 50.0),
            (f64::INFINITY, 50.0),
            (1.0e-8, f64::NAN),
            (1.0e-8, f64::INFINITY),
        ] {
            let err = validate_wolfe_params(WolfeParams {
                amin,
                amax,
                ..WolfeParams::default()
            })
            .expect_err("non-finite alpha bounds should fail");
            assert!(matches!(err, OptError::InvalidArgument { .. }));
        }
    }

    #[test]
    fn wolfe_apis_reject_mismatched_dimensions() {
        let x = vec![1.0, 2.0];
        let g = quadratic_grad(&x);
        let f0 = quadratic(&x);
        let short_direction = vec![-1.0];
        let full_direction = vec![-1.0, -1.0];
        let short_gradient = vec![1.0];

        let err = line_search_wolfe2(
            &quadratic,
            &quadratic_grad,
            &x,
            &short_direction,
            f0,
            &g,
            WolfeParams::default(),
        )
        .expect_err("short direction should fail before trial-point indexing");
        assert!(matches!(err, OptError::InvalidArgument { .. }));

        let err = line_search_wolfe1(
            &quadratic,
            &quadratic_grad,
            &x,
            &full_direction,
            f0,
            &short_gradient,
            WolfeParams::default(),
        )
        .expect_err("short initial gradient should fail before dot truncation");
        assert!(matches!(err, OptError::InvalidArgument { .. }));
    }

    #[test]
    fn wolfe_apis_reject_mismatched_gradient_outputs() {
        let x = vec![1.0, 2.0];
        let g = quadratic_grad(&x);
        let f0 = quadratic(&x);
        let direction = vec![-0.2, -0.4];
        let short_grad = |_x: &[f64]| vec![1.0];

        let err = line_search_wolfe2(
            &quadratic,
            &short_grad,
            &x,
            &direction,
            f0,
            &g,
            WolfeParams::default(),
        )
        .expect_err("short gradient output should fail before dot truncation");
        assert!(matches!(err, OptError::InvalidArgument { .. }));
    }

    #[test]
    fn wolfe2_exact_minimizer_for_quadratic() {
        // For f(x) = x^2, starting at x=2 with d=-1, exact minimizer is alpha=2
        let x = vec![2.0];
        let g = vec![4.0]; // 2*x
        let f0 = 4.0; // x^2
        let d = vec![-1.0];

        let result = line_search_wolfe2(
            &|x: &[f64]| x[0] * x[0],
            &|x: &[f64]| vec![2.0 * x[0]],
            &x,
            &d,
            f0,
            &g,
            WolfeParams::default(),
        )
        .expect("1d quadratic works");

        // Should find a valid step that decreases the function
        assert!(result.alpha > 0.0, "alpha should be positive");
        assert!(
            result.f_at_alpha < f0,
            "f should decrease: {} vs {f0}",
            result.f_at_alpha
        );
    }

    #[test]
    fn wolfe_params_default_is_valid() {
        validate_wolfe_params(WolfeParams::default()).expect("default should be valid");
    }
}
