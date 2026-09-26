#![forbid(unsafe_code)]

pub mod api;
pub mod bdf;
pub mod bvp;
mod collocation;
pub mod complex;
pub mod lebedev;
pub mod quad;
mod quadpack;
pub mod radau;
pub mod rk;
pub mod solver;
pub mod step_size;
pub mod validation;

pub use api::{
    EventFn, EventSpec, LsodaSolver, LsodaSolverConfig, OdePortfolioResult, OdeSolution,
    SolveIvpOptions, SolveIvpResult, SolverKind, solve_ivp, solve_ivp_many, solve_ivp_with_audit,
    solve_ivp_with_casp, solve_ivp_with_casp_portfolio,
};
pub use bdf::{BdfSolver, BdfSolverConfig};
pub use bvp::{BvpBcJac, BvpError, BvpFunJac, BvpOptions, BvpResult, solve_bvp, solve_bvp_many};
pub use complex::{ComplexOdeResult, complex_ode};
pub use fsci_runtime::{StiffnessConditionState, StiffnessDetector};
pub use lebedev::{LebedevRule, lebedev_rule};
pub use quad::{
    CompositeQuadResult, CubatureOptions, CubatureRegion, CubatureResult, CubatureRule,
    CubatureScalarResult, CubatureStatus, DblquadOptions, DblquadResult, NsumResult, QmcQuadResult,
    QuadInfo, QuadOptions, QuadResult, QuadVecResult, QuadWeight, QuadWeightOptions, cubature,
    cubature_scalar, cumulative_simpson, cumulative_simpson_axis_2d, cumulative_trapezoid,
    cumulative_trapezoid_axis_2d, cumulative_trapezoid_initial, cumulative_trapezoid_uniform,
    dblquad, dblquad_many, dblquad_rect, fixed_quad, gauss_kronrod_quad, gauss_legendre,
    line_integral, monte_carlo_integrate, newton_cotes, newton_cotes_quad, nquad, nquad_many, nsum,
    qmc_quad, quad, quad_explain, quad_full_inf, quad_full_output, quad_inf, quad_many,
    quad_neg_inf, quad_points, quad_vec, quad_weighted, quad_weighted_full_output, romb, romb_func,
    romberg, simpson, simpson_axis_2d, simpson_irregular, simpson_uniform, tanhsinh, tplquad,
    tplquad_many, tplquad_rect, trapezoid, trapezoid_axis_2d, trapezoid_irregular,
    trapezoid_richardson, trapezoid_uniform,
};
pub use radau::{RadauSolver, RadauSolverConfig};
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

/// SciPy-compatible alias for explicit Runge-Kutta 5(4) solver, matching `scipy.integrate.RK45`.
pub type RK45 = RkSolver;

/// SciPy-compatible alias for explicit Runge-Kutta 3(2) solver, matching `scipy.integrate.RK23`.
pub type RK23 = RkSolver;

/// SciPy-compatible alias for Dormand-Prince 8(5,3) solver, matching `scipy.integrate.DOP853`.
pub type DOP853 = RkSolver;

/// SciPy-compatible alias for Radau IIA implicit Runge-Kutta solver, matching `scipy.integrate.Radau`.
pub type Radau = RadauSolver;

/// SciPy-compatible alias for BDF multi-step solver, matching `scipy.integrate.BDF`.
pub type BDF = BdfSolver;

/// SciPy-compatible alias for continuous solution output, matching `scipy.integrate.DenseOutput`.
pub type DenseOutput = OdeSolution;

/// SciPy-compatible alias for the LSODA stepper, matching `scipy.integrate.LSODA` (see
/// [`LsodaSolver`] for how it differs from ODEPACK's).
pub type LSODA = LsodaSolver;

/// Object-oriented ODE integrator interface, matching `scipy.integrate.ode`.
#[derive(Debug)]
pub struct Ode<F> {
    pub fun: F,
    pub t: f64,
    pub y: Vec<f64>,
    /// Outcome of the last `integrate` call; read through [`Ode::successful`].
    success: bool,
}

impl<F: FnMut(f64, &[f64]) -> Vec<f64>> Ode<F> {
    pub fn new(fun: F) -> Self {
        Self {
            fun,
            t: 0.0,
            y: Vec::new(),
            // status: no integrate call yet; SciPy's vode __init__ sets success = 1
            success: true,
        }
    }

    pub fn set_initial_value(&mut self, y: &[f64], t: f64) -> &mut Self {
        self.y = y.to_vec();
        self.t = t;
        // status: reset on set_initial_value, as SciPy's vode reset() sets success = 1
        self.success = true;
        self
    }

    /// Integrate to `t_target`. On failure the state is left at the last time the
    /// solver reached and [`Ode::successful`] returns false, as in SciPy.
    pub fn integrate(&mut self, t_target: f64) -> Result<&[f64], IntegrateValidationError> {
        let options = SolveIvpOptions {
            t_span: (self.t, t_target),
            y0: &self.y,
            ..SolveIvpOptions::default()
        };
        let res = match solve_ivp(&mut self.fun, &options) {
            Ok(res) => res,
            Err(err) => {
                self.success = false;
                return Err(err);
            }
        };
        // br-szq1n.7: this used to ignore `res.success` and always advance `t` to
        // `t_target`, while `successful()` returned a literal `true`.
        self.success = res.success;
        if let (Some(&t_reached), Some(last)) = (res.t.last(), res.y.last()) {
            self.y = last.clone();
            self.t = t_reached;
        }
        Ok(&self.y)
    }

    /// Whether the last `integrate` call reached its target time.
    pub fn successful(&self) -> bool {
        self.success
    }
}

/// SciPy-compatible alias for object-oriented ODE integrator, matching `scipy.integrate.ode`.
#[allow(non_camel_case_types)]
pub type ode<F> = Ode<F>;

// SciPy's `IntegrationWarning` is `WarningCategory::IntegrationWarning`, raised by `quad` (and
// the routines built on it) when QUADPACK stops short. SciPy's `ODEintWarning` has no
// counterpart: where SciPy warns and returns the rows it could not compute, `odeint` returns
// `IntegrateValidationError::IntegrationFailed`.
pub use fsci_runtime::{Warning, WarningCategory, catch_warnings};

/// Legacy `odeint`-style interface.
///
/// Matches `scipy.integrate.odeint(func, y0, t)`: integrates y' = func(y, t) with
/// LSODA (automatic nonstiff/stiff switching) at `rtol = atol = 1.49e-8` and
/// returns one state per requested time, `y_matrix[i]` at `t[i]`. If the
/// integrator cannot reach every requested time, this returns
/// [`IntegrateValidationError::IntegrationFailed`] instead of a truncated matrix.
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
            // SciPy's odeint is ODEPACK LSODA; a nonstiff-only method here made
            // stiff odeint problems crawl or fail where SciPy switches to BDF.
            method: SolverKind::Lsoda,
            t_eval: Some(t),
            rtol: 1.49e-8,
            atol: ToleranceValue::Scalar(1.49e-8),
            ..SolveIvpOptions::default()
        },
    )?;

    // br-szq1n.7: `result.y[i]` is the state at `t[i]` (t_eval), but only for the
    // times the solver reached. This used to return the result unchecked, so a
    // failed integration silently produced fewer rows than requested times.
    if !result.success || result.y.len() != t.len() {
        return Err(IntegrateValidationError::IntegrationFailed {
            message: format!(
                "{} (reached {} of {} requested times)",
                result.message,
                result.y.len(),
                t.len()
            ),
        });
    }
    Ok(result.y)
}

#[cfg(test)]
mod tests {
    use super::*;

    // br-szq1n.7: y' = y^2, y(0) = 1 blows up at t = 1, so integrating to t = 2 must
    // fail. `successful()` used to be a literal `true`.
    #[test]
    fn ode_successful_reports_failure_on_finite_time_blowup() {
        let mut ok = Ode::new(|_t: f64, y: &[f64]| vec![-y[0]]);
        ok.set_initial_value(&[1.0], 0.0);
        ok.integrate(1.0).expect("decay integrates");
        assert!(ok.successful(), "exponential decay must succeed");
        assert!((ok.t - 1.0).abs() < 1e-12);

        let mut blowup = Ode::new(|_t: f64, y: &[f64]| vec![y[0] * y[0]]);
        blowup.set_initial_value(&[1.0], 0.0);
        let _ = blowup.integrate(2.0);
        assert!(
            !blowup.successful(),
            "y' = y^2 from y(0)=1 cannot reach t = 2"
        );
        assert!(
            blowup.t < 1.0 + 1e-6,
            "state must stay at the time reached, got t = {}",
            blowup.t
        );
    }

    #[test]
    fn odeint_errors_instead_of_truncating_on_failure() {
        let err = odeint(
            &mut |y: &[f64], _t| vec![y[0] * y[0]],
            &[1.0],
            &[0.0, 0.5, 2.0],
        )
        .expect_err("odeint cannot reach t = 2 past the blow-up at t = 1");
        assert!(
            matches!(err, IntegrateValidationError::IntegrationFailed { .. }),
            "unexpected error {err:?}"
        );
        // Positive arm: every requested time is returned for a solvable problem.
        let ok = odeint(&mut |y: &[f64], _t| vec![-y[0]], &[1.0], &[0.0, 0.5, 2.0]).expect("decay");
        assert_eq!(ok.len(), 3);
        assert!(
            (ok[2][0] - (-2.0_f64).exp()).abs() < 1e-6,
            "y(2) = {}",
            ok[2][0]
        );
    }

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
