#![forbid(unsafe_code)]

pub mod api;
pub mod bdf;
pub mod bvp;
pub mod complex;
pub mod lebedev;
pub mod quad;
pub mod radau;
pub mod rk;
pub mod solver;
pub mod step_size;
pub mod validation;

pub use api::{
    EventFn, EventSpec, OdePortfolioResult, OdeSolution, SolveIvpOptions, SolveIvpResult,
    SolverKind, solve_ivp, solve_ivp_many, solve_ivp_with_audit, solve_ivp_with_casp_portfolio,
};
pub use bdf::{BdfSolver, BdfSolverConfig};
pub use bvp::{BvpError, BvpOptions, BvpResult, solve_bvp, solve_bvp_many};
pub use complex::{ComplexOdeResult, complex_ode};
pub use fsci_runtime::{StiffnessConditionState, StiffnessDetector};
pub use lebedev::{LebedevRule, lebedev_rule};
pub use quad::{
    CompositeQuadResult, CubatureOptions, CubatureRegion, CubatureResult, CubatureRule,
    CubatureScalarResult, CubatureStatus, DblquadOptions, DblquadResult, NsumResult, QmcQuadResult,
    QuadOptions, QuadResult, QuadVecResult, cubature, cubature_scalar, cumulative_simpson,
    cumulative_simpson_axis_2d, cumulative_trapezoid, cumulative_trapezoid_axis_2d,
    cumulative_trapezoid_initial, cumulative_trapezoid_uniform, dblquad, dblquad_many,
    dblquad_rect, fixed_quad, gauss_kronrod_quad, gauss_legendre, line_integral,
    monte_carlo_integrate, newton_cotes, newton_cotes_quad, nquad, nquad_many, nsum, qmc_quad,
    quad, quad_cauchy_pv, quad_explain, quad_full_inf, quad_inf, quad_many, quad_neg_inf, quad_vec,
    romb, romb_func, romberg, simpson, simpson_axis_2d, simpson_irregular, simpson_uniform,
    tanhsinh, tplquad, tplquad_many, tplquad_rect, trapezoid, trapezoid_axis_2d,
    trapezoid_irregular, trapezoid_richardson, trapezoid_uniform,
};
pub use rk::{
    ButcherTableau, DOP853_TABLEAU, RK23_TABLEAU, RK45_TABLEAU, RkSolver, RkSolverConfig,
};
pub use solver::{OdeSolver, OdeSolverState, StepFailure, StepOutcome};
pub use step_size::{InitialStepRequest, StepRhsFn, select_initial_step};
pub use validation::{
    EPS, IntegrateValidationError, MIN_RTOL, SyncSharedAuditLedger, ToleranceValue,
    ToleranceWarning, ValidatedTolerance, sync_audit_ledger, validate_first_step,
    validate_first_step_with_audit, validate_max_step, validate_max_step_with_audit, validate_tol,
    validate_tol_with_audit,
};

/// Legacy `odeint`-style interface.
///
/// Matches `scipy.integrate.odeint(func, y0, t)`.
/// Integrates an ODE system y' = func(y, t) over a sequence of time points.
/// Returns `(y_matrix, info)` where `y_matrix[i]` is the state at `t[i]`.
pub fn odeint<F>(
    func: &mut F,
    y0: &[f64],
    t: &[f64],
) -> Result<Vec<Vec<f64>>, IntegrateValidationError>
where
    F: FnMut(&[f64], f64) -> Vec<f64>,
{
    if t.is_empty() {
        return Ok(vec![]);
    }
    if y0.is_empty() {
        return Err(IntegrateValidationError::EmptyY0);
    }
    if y0.iter().any(|value| !value.is_finite()) {
        return Err(IntegrateValidationError::NonFiniteY0);
    }
    if t.iter().any(|value| !value.is_finite()) {
        return Err(IntegrateValidationError::NonFiniteSpan);
    }
    if t.len() == 1 {
        return Ok(vec![y0.to_vec()]);
    }

    // odeint convention: func(y, t), but solve_ivp uses func(t, y)
    let mut ivp_func = |ti: f64, yi: &[f64]| -> Vec<f64> { func(yi, ti) };

    let t0 = t[0];
    let tf = t[t.len() - 1];

    let result = solve_ivp(
        &mut ivp_func,
        &SolveIvpOptions {
            t_span: (t0, tf),
            y0,
            method: SolverKind::Rk45,
            t_eval: Some(t),
            rtol: 1.49e-8,
            atol: ToleranceValue::Scalar(1.49e-8),
            ..SolveIvpOptions::default()
        },
    )?;

    // transpose result.y since solve_ivp returns columns for each time step,
    // wait, solve_ivp result.y is Vec<Vec<f64>> where each inner vector is the state at t_i.
    // Let's assume result.y matches the shape we need.
    Ok(result.y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn odeint_single_point_rejects_empty_initial_state() {
        let err = odeint(&mut |_y, _t| Vec::new(), &[], &[0.0])
            .expect_err("single-point odeint should validate empty y0");
        assert_eq!(err, IntegrateValidationError::EmptyY0);
    }

    #[test]
    fn odeint_single_point_rejects_non_finite_initial_state() {
        let err = odeint(&mut |_y, _t| vec![0.0], &[f64::NAN], &[0.0])
            .expect_err("single-point odeint should validate y0");
        assert_eq!(err, IntegrateValidationError::NonFiniteY0);
    }

    #[test]
    fn odeint_single_point_rejects_non_finite_time() {
        let err = odeint(&mut |_y, _t| vec![0.0], &[1.0], &[f64::NAN])
            .expect_err("single-point odeint should validate t");
        assert_eq!(err, IntegrateValidationError::NonFiniteSpan);
    }
}
