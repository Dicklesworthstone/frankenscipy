#![forbid(unsafe_code)]
//! SciPy's public line searches over plain closures: [`line_search`]
//! (`scipy.optimize.line_search`) and the typed [`line_search_wolfe1`] (MINPACK-2 `dcsrch`) /
//! [`line_search_wolfe2`] (strong Wolfe) wrappers. All three run the transcriptions in
//! [`crate::bfgs`] that BFGS, CG and Newton-CG use, so each search has one implementation.

use crate::bfgs::{self, LineObjective, Wolfe2};
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

/// The caller's closures as a [`LineObjective`], counting evaluations as SciPy's `phi` /
/// `derphi` count them (`fc`, `gc`).
struct Closures<'a, F, G> {
    f: &'a F,
    grad: &'a G,
    n: usize,
    fc: usize,
    gc: usize,
}

impl<F, G> LineObjective for Closures<'_, F, G>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    type Error = OptError;

    fn fun(&mut self, x: &[f64]) -> Result<f64, OptError> {
        self.fc += 1;
        Ok((self.f)(x))
    }

    fn grad(&mut self, x: &[f64]) -> Result<Vec<f64>, OptError> {
        self.gc += 1;
        let g = (self.grad)(x);
        validate_gradient_output_len(&g, self.n)?;
        Ok(g)
    }
}

fn no_wolfe_step(search: &str) -> OptError {
    OptError::NotConverged {
        detail: format!("{search} found no step satisfying the Wolfe conditions"),
    }
}

/// SciPy's `line_search_wolfe1`: MINPACK-2's `dcsrch` (the transcription BFGS, CG and
/// Newton-CG use, see [`crate::bfgs`]) on `[amin, amax]` with `xtol = 1e-14` and SciPy's 100
/// iterations. The step satisfies the STRONG Wolfe conditions
/// - Armijo:    f(x + α·d) ≤ f(x) + c1·α·g·d
/// - Curvature: |g(x + α·d)·d| ≤ c2·|g·d|
///
/// and a search that finds none is an error, where SciPy returns `stp = None`.
/// `params.maxiter` is not used (it is `line_search_wolfe2`'s). `evaluations` counts function
/// and gradient evaluations together.
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
    let mut obj = Closures {
        f,
        grad,
        n: x.len(),
        fc: 0,
        gc: 0,
    };
    let found = bfgs::line_search_wolfe1(
        &mut obj,
        x,
        direction,
        g0,
        f0,
        None,
        params.c1,
        params.c2,
        params.amax,
        params.amin,
        1e-14,
    )?
    .ok_or_else(|| no_wolfe_step("dcsrch"))?;
    // dcsrch evaluates f and ∇f together at every trial step, so the gradient is there.
    let g = found.grad.ok_or_else(|| no_wolfe_step("dcsrch"))?;
    Ok(LineSearchResult {
        alpha: found.alpha,
        f_at_alpha: found.fval,
        directional_derivative: dot(&g, direction),
        evaluations: obj.fc + obj.gc,
    })
}

/// SciPy's `line_search_wolfe2` (`scalar_search_wolfe2`: bracketing, then the
/// cubic/quadratic/bisection zoom; the transcription BFGS and CG fall back on, see
/// [`crate::bfgs`]) with `amax = params.amax` and at most `params.maxiter` bracketing steps.
/// The step satisfies the strong Wolfe conditions
/// - Armijo:       f(x + α·d) ≤ f(x) + c1·α·g·d
/// - Strong Wolfe: |g(x + α·d)·d| ≤ c2·|g·d|
///
/// and every way SciPy ends without one is an error: `alpha_star = None`, and SciPy's
/// "did not converge" last trial step, whose curvature it never checked. `params.amin` is
/// validated but unused (SciPy's Wolfe-2 search has none). `evaluations` counts function and
/// gradient evaluations together.
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
    let mut obj = Closures {
        f,
        grad,
        n: x.len(),
        fc: 0,
        gc: 0,
    };
    let outcome = bfgs::line_search_wolfe2(
        &mut obj,
        x,
        direction,
        g0,
        f0,
        None,
        params.c1,
        params.c2,
        Some(params.amax),
        params.maxiter,
        None,
    )?;
    let Wolfe2::Step(found) = outcome else {
        return Err(no_wolfe_step("the Wolfe-2 search"));
    };
    // No gradient: SciPy's "did not converge" last trial step, whose curvature was never
    // checked.
    let g = found
        .grad
        .ok_or_else(|| no_wolfe_step("the Wolfe-2 search"))?;
    Ok(LineSearchResult {
        alpha: found.alpha,
        f_at_alpha: found.fval,
        directional_derivative: dot(&g, direction),
        evaluations: obj.fc + obj.gc,
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
    /// `f(xk + alpha·pk)` at the returned step; `None` where SciPy returns `None` (the zoom
    /// failed), and `f(xk)` when the step rounded to zero or passed `amax` (SciPy's
    /// `phi_star = phi0`).
    pub new_fval: Option<f64>,
    /// `f(xk)`, except that SciPy hands back `old_old_fval` here when the step rounded to zero
    /// or passed `amax`.
    pub old_fval: Option<f64>,
    /// Gradient at `xk + alpha·pk`, or `None` if the search did not converge.
    pub new_grad: Option<Vec<f64>>,
}

/// `scipy.optimize.line_search` (`line_search_wolfe2`): a step satisfying the strong Wolfe
/// conditions, by the same transcription of `scalar_search_wolfe2` BFGS and CG use (see
/// [`crate::bfgs`]).
///
/// `phi(s) = f(xk + s·pk)` and `derphi(s) = ∇f(xk + s·pk)·pk` are counted in `fc` / `gc` as
/// SciPy counts them. `gfk` is the gradient at `xk`, computed (and not counted, as in SciPy)
/// when `None`; `old_fval` is `f(xk)`, evaluated as `phi(0)` (counted) when `None`;
/// `old_old_fval` picks the first trial step. Defaults in SciPy: `c1 = 1e-4`, `c2 = 0.9`,
/// `maxiter = 10`, `amax = None`. The result mirrors SciPy's tuple, `None`s included; the only
/// error is a gradient of the wrong length.
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
) -> Result<ScipyLineSearchResult, OptError>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = xk.len();
    let gfk = match gfk {
        Some(g) => g.to_vec(),
        None => {
            let g = grad(xk);
            validate_gradient_output_len(&g, n)?;
            g
        }
    };
    let mut obj = Closures {
        f,
        grad,
        n,
        fc: 0,
        gc: 0,
    };
    let phi0 = match old_fval {
        Some(value) => value,
        None => {
            let origin: Vec<f64> = xk.iter().zip(pk).map(|(x, p)| x + 0.0 * p).collect();
            obj.fun(&origin)?
        }
    };
    let outcome = bfgs::line_search_wolfe2(
        &mut obj,
        xk,
        pk,
        &gfk,
        phi0,
        old_old_fval,
        c1,
        c2,
        amax,
        maxiter,
        None,
    )?;
    let (alpha, new_fval, old_fval, new_grad) = match outcome {
        Wolfe2::Step(found) => (Some(found.alpha), Some(found.fval), Some(phi0), found.grad),
        Wolfe2::ZoomFailed => (None, None, Some(phi0), None),
        Wolfe2::Stalled => (None, Some(phi0), old_old_fval, None),
    };
    Ok(ScipyLineSearchResult {
        alpha,
        fc: obj.fc,
        gc: obj.gc,
        new_fval,
        old_fval,
        new_grad,
    })
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
        )
        .expect("line_search");
        assert_eq!(r.alpha, Some(1.0));
        assert_eq!((r.fc, r.gc), (2, 1));
        assert!((r.new_fval.unwrap() - 1.13).abs() < 1e-9);
        assert!((r.old_fval.unwrap() - 6.13).abs() < 1e-9);
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
        let r2 = line_search(&f2, &g2, &xk, &pk, None, None, None, 1e-4, 0.9, None, 10)
            .expect("line_search");
        assert!((r2.alpha.unwrap() - 0.00093831027587526).abs() < 1e-12);
        assert_eq!((r2.fc, r2.gc), (11, 1));
        assert!((r2.new_fval.unwrap() - 4.75058732).abs() < 1e-6);
    }

    /// SciPy 1.17.1's two ways of ending without a strong-Wolfe step, on a linear objective
    /// (the bracket only expands). With no `amax` it returns its last trial step, α = 1024,
    /// without a gradient (fc = 12, gc = 10, "did not converge"); with `amax = 3` the zoom
    /// fails and α and `new_fval` are `None` (fc = 16, gc = 3). The typed
    /// `line_search_wolfe2` reports both as errors.
    #[test]
    fn line_search_reports_scipys_non_convergence() {
        let f = |x: &[f64]| -x[0] - x[1];
        let g = |_x: &[f64]| vec![-1.0, -1.0];
        let origin = [0.0, 0.0];
        let pk = [1.0, 1.0];
        let r = line_search(&f, &g, &origin, &pk, None, None, None, 1e-4, 0.9, None, 10)
            .expect("line_search");
        assert_eq!(r.alpha, Some(1024.0));
        assert_eq!((r.fc, r.gc), (12, 10));
        assert_eq!(r.new_fval, Some(-2048.0));
        assert_eq!(r.old_fval.map(f64::to_bits), Some((-0.0_f64).to_bits()));
        assert!(r.new_grad.is_none());

        let r = line_search(
            &f,
            &g,
            &origin,
            &pk,
            None,
            None,
            None,
            1e-4,
            0.9,
            Some(3.0),
            10,
        )
        .expect("line_search");
        assert_eq!(r.alpha, None);
        assert_eq!((r.fc, r.gc), (16, 3));
        assert_eq!(r.new_fval, None);
        assert!(r.new_grad.is_none());

        let err = line_search_wolfe2(
            &f,
            &g,
            &origin,
            &pk,
            0.0,
            &[-1.0, -1.0],
            WolfeParams::default(),
        )
        .expect_err("no strong-Wolfe step on a linear objective");
        assert!(matches!(err, OptError::NotConverged { .. }));
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
