//! Complex-valued ODE integration — `scipy.integrate.complex_ode`.
//!
//! SciPy's `complex_ode` integrates a complex system `dy/dt = f(t, y)` by
//! reformulating it as a real system of twice the dimension. We follow the same
//! recipe, layering a thin wrapper over the real [`solve_ivp`] integrator: the
//! complex state `y` of length `n` is packed into a real vector
//! `[Re(y_0), …, Re(y_{n-1}), Im(y_0), …, Im(y_{n-1})]`, integrated, and
//! unpacked back into complex form.
//!
//! FrankenSciPy is functional rather than class-based, so this is exposed as a
//! single call instead of SciPy's stateful `complex_ode(...).integrate(t)`
//! object; the numerical result matches SciPy's `complex_ode` to tolerance.

use crate::api::{SolveIvpOptions, SolverKind, solve_ivp};
use crate::validation::{
    IntegrateValidationError, ToleranceValue, validate_rhs_shape, validate_tol,
};

/// A complex number as a `(real, imaginary)` pair, matching the workspace
/// `Complex64` convention used elsewhere (e.g. `fsci-fft`).
pub type Complex64 = (f64, f64);

/// Result of [`complex_ode`].
#[derive(Clone, Debug, PartialEq)]
pub struct ComplexOdeResult {
    /// Time points at which the solution was evaluated.
    pub t: Vec<f64>,
    /// Complex state at each time point; `y[i]` is the state vector at `t[i]`.
    pub y: Vec<Vec<Complex64>>,
    /// Whether the integration reached the end of the interval.
    pub success: bool,
    /// Solver status message.
    pub message: String,
    /// Number of (real) right-hand-side evaluations.
    pub nfev: usize,
}

/// Integrate a complex-valued initial value problem `dy/dt = f(t, y)`.
///
/// Mirrors `scipy.integrate.complex_ode`: `fun(t, y)` maps the current time and
/// complex state vector to the complex derivative, `y0` is the complex initial
/// state, and the system is integrated over `t_span = (t0, tf)`. If `t_eval` is
/// given, the solution is reported at those points; otherwise at the solver's
/// own steps. `rtol`/`atol` are the relative/absolute tolerances.
///
/// Uses an explicit Runge-Kutta (`RK45`) integrator on the real reformulation,
/// matching SciPy's `complex_ode(...).set_integrator('dopri5')` to tolerance.
///
/// # Errors
/// Propagates [`IntegrateValidationError`] from the underlying real solver
/// (e.g. empty initial state or a non-finite span).
pub fn complex_ode<F>(
    mut fun: F,
    y0: &[Complex64],
    t_span: (f64, f64),
    t_eval: Option<&[f64]>,
    rtol: f64,
    atol: f64,
) -> Result<ComplexOdeResult, IntegrateValidationError>
where
    F: FnMut(f64, &[Complex64]) -> Vec<Complex64>,
{
    let n = y0.len();
    validate_complex_ode_inputs(y0, t_span, t_eval, rtol, atol)?;

    let initial_dyc = fun(t_span.0, y0);
    validate_rhs_shape(initial_dyc.len(), n)?;
    validate_complex_rhs_finite(&initial_dyc)?;

    // Pack the complex initial state into [Re.., Im..].
    let mut real_y0 = Vec::with_capacity(2 * n);
    real_y0.extend(y0.iter().map(|c| c.0));
    real_y0.extend(y0.iter().map(|c| c.1));

    // Wrap the complex RHS as a real RHS on the doubled state.
    let initial_real_y0 = real_y0.clone();
    let mut cached_initial_dyc = Some(initial_dyc);
    let mut rhs_shape_error = None;
    let mut rhs_value_error = false;
    let mut real_fun = |t: f64, u: &[f64]| -> Vec<f64> {
        let yc: Vec<Complex64> = (0..n).map(|j| (u[j], u[n + j])).collect();
        let dyc = if cached_initial_dyc.is_some()
            && t.to_bits() == t_span.0.to_bits()
            && u == initial_real_y0.as_slice()
        {
            cached_initial_dyc
                .take()
                .expect("initial derivative cache checked above")
        } else {
            fun(t, &yc)
        };
        if dyc.len() != n {
            rhs_shape_error = Some((n, dyc.len()));
            return vec![f64::NAN; 2 * n];
        }
        if complex_rhs_has_non_finite(&dyc) {
            rhs_value_error = true;
            return vec![f64::NAN; 2 * n];
        }
        let mut out = vec![0.0; 2 * n];
        for (j, d) in dyc.iter().enumerate() {
            out[j] = d.0;
            out[n + j] = d.1;
        }
        out
    };

    let opts = SolveIvpOptions {
        t_span,
        y0: &real_y0,
        method: SolverKind::Rk45,
        t_eval,
        rtol,
        atol: ToleranceValue::Scalar(atol),
        ..Default::default()
    };

    let res = solve_ivp(&mut real_fun, &opts);
    if let Some((expected, actual)) = rhs_shape_error {
        return Err(IntegrateValidationError::RhsWrongShape { expected, actual });
    }
    if rhs_value_error {
        return Err(IntegrateValidationError::NonFiniteF0);
    }
    let res = res?;

    // Unpack each real state row back into complex form.
    let y: Vec<Vec<Complex64>> = res
        .y
        .iter()
        .map(|row| (0..n).map(|j| (row[j], row[n + j])).collect())
        .collect();

    Ok(ComplexOdeResult {
        t: res.t,
        y,
        success: res.success,
        message: res.message,
        nfev: res.nfev,
    })
}

fn validate_complex_ode_inputs(
    y0: &[Complex64],
    t_span: (f64, f64),
    t_eval: Option<&[f64]>,
    rtol: f64,
    atol: f64,
) -> Result<(), IntegrateValidationError> {
    if y0.is_empty() {
        return Err(IntegrateValidationError::EmptyY0);
    }
    if !t_span.0.is_finite() || !t_span.1.is_finite() {
        return Err(IntegrateValidationError::NonFiniteSpan);
    }
    if y0
        .iter()
        .any(|value| !value.0.is_finite() || !value.1.is_finite())
    {
        return Err(IntegrateValidationError::NonFiniteY0);
    }
    validate_tol(
        ToleranceValue::Scalar(rtol),
        ToleranceValue::Scalar(atol),
        y0.len() * 2,
        fsci_runtime::RuntimeMode::Strict,
    )?;
    if let Some(t_eval) = t_eval {
        validate_complex_t_eval(t_eval, t_span.0, t_span.1)?;
    }
    Ok(())
}

fn validate_complex_rhs_finite(values: &[Complex64]) -> Result<(), IntegrateValidationError> {
    if complex_rhs_has_non_finite(values) {
        return Err(IntegrateValidationError::NonFiniteF0);
    }
    Ok(())
}

fn complex_rhs_has_non_finite(values: &[Complex64]) -> bool {
    values
        .iter()
        .any(|value| !value.0.is_finite() || !value.1.is_finite())
}

fn validate_complex_t_eval(
    t_eval: &[f64],
    t0: f64,
    tf: f64,
) -> Result<(), IntegrateValidationError> {
    let t_min = t0.min(tf);
    let t_max = t0.max(tf);
    if t_eval.iter().any(|&te| te < t_min || te > t_max) {
        return Err(IntegrateValidationError::TEvalOutOfSpan);
    }
    let sorted = if tf >= t0 {
        t_eval.windows(2).all(|window| window[1] > window[0])
    } else {
        t_eval.windows(2).all(|window| window[1] < window[0])
    };
    if !sorted {
        return Err(IntegrateValidationError::TEvalNotSorted);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cmul(a: Complex64, b: Complex64) -> Complex64 {
        (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
    }

    #[test]
    fn complex_ode_exponential_matches_analytic_and_scipy() {
        // y' = i*y, y(0) = 1 -> y(t) = exp(i t); at t = 2: (cos 2, sin 2).
        let f = |_t: f64, y: &[Complex64]| vec![cmul((0.0, 1.0), y[0])];
        let r = complex_ode(f, &[(1.0, 0.0)], (0.0, 2.0), Some(&[2.0]), 1e-9, 1e-12).unwrap();
        assert!(r.success);
        let y = r.y.last().unwrap()[0];
        let (re, im) = (2.0_f64.cos(), 2.0_f64.sin());
        assert!((y.0 - re).abs() < 1e-7, "re = {}, expected {re}", y.0);
        assert!((y.1 - im).abs() < 1e-7, "im = {}, expected {im}", y.1);
    }

    #[test]
    fn complex_ode_decaying_spiral() {
        // y' = (-1 + 2i)*y, y(0) = 1 -> exp((-1+2i) t); at t = 1.5.
        let f = |_t: f64, y: &[Complex64]| vec![cmul((-1.0, 2.0), y[0])];
        let r = complex_ode(f, &[(1.0, 0.0)], (0.0, 1.5), Some(&[1.5]), 1e-9, 1e-12).unwrap();
        let y = r.y.last().unwrap()[0];
        // exp(-1.5) * (cos 3, sin 3)
        let mag = (-1.5_f64).exp();
        assert!((y.0 - mag * 3.0_f64.cos()).abs() < 1e-7, "re = {}", y.0);
        assert!((y.1 - mag * 3.0_f64.sin()).abs() < 1e-7, "im = {}", y.1);
    }

    #[test]
    fn complex_ode_coupled_pair() {
        // y0' = i*y1, y1' = i*y0; y(0) = (1, 0). y0(t) = cos t, y1(t) = i sin t.
        let f = |_t: f64, y: &[Complex64]| vec![cmul((0.0, 1.0), y[1]), cmul((0.0, 1.0), y[0])];
        let r = complex_ode(
            f,
            &[(1.0, 0.0), (0.0, 0.0)],
            (0.0, 1.0),
            Some(&[1.0]),
            1e-9,
            1e-12,
        )
        .unwrap();
        let last = r.y.last().unwrap();
        assert!((last[0].0 - 1.0_f64.cos()).abs() < 1e-7);
        assert!(last[0].1.abs() < 1e-7);
        assert!(last[1].0.abs() < 1e-7);
        assert!((last[1].1 - 1.0_f64.sin()).abs() < 1e-7);
    }

    #[test]
    fn complex_ode_rejects_short_rhs_output() {
        let err = complex_ode(
            |_t, _y| Vec::new(),
            &[(1.0, 0.0)],
            (0.0, 1.0),
            Some(&[1.0]),
            1e-6,
            1e-9,
        )
        .expect_err("short complex RHS output");
        assert_eq!(
            err,
            IntegrateValidationError::RhsWrongShape {
                expected: 1,
                actual: 0
            }
        );
    }

    #[test]
    fn complex_ode_rejects_long_rhs_output() {
        let err = complex_ode(
            |_t, _y| vec![(0.0, 1.0), (1.0, 0.0)],
            &[(1.0, 0.0)],
            (0.0, 1.0),
            Some(&[1.0]),
            1e-6,
            1e-9,
        )
        .expect_err("long complex RHS output");
        assert_eq!(
            err,
            IntegrateValidationError::RhsWrongShape {
                expected: 1,
                actual: 2
            }
        );
    }

    #[test]
    fn complex_ode_rejects_non_finite_rhs_values() {
        let err = complex_ode(
            |_t, _y| vec![(f64::NAN, 0.0)],
            &[(1.0, 0.0)],
            (0.0, 1.0),
            Some(&[1.0]),
            1e-6,
            1e-9,
        )
        .expect_err("non-finite complex RHS output");
        assert_eq!(err, IntegrateValidationError::NonFiniteF0);
    }
}
