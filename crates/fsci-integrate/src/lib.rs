#![forbid(unsafe_code)]

pub mod api;
pub mod bdf;
pub mod bvp;
mod collocation;
pub mod complex;
pub mod lebedev;
pub mod lsoda;
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
pub use lsoda::{JacFn, LsodaMethod};
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

/// SciPy-compatible alias for the LSODA stepper, matching `scipy.integrate.LSODA`: ODEPACK's
/// LSODA as SciPy 1.17.1 runs it (see [`LsodaSolver`] and [`lsoda`]).
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

/// `odeint`'s default `rtol` and `atol` (`_odepackmodule.c`: `tol = 1.49012e-8`).
const ODEINT_TOL: f64 = 1.49012e-8;
/// `odeint`'s default `mxstep`: steps allowed per output time.
const ODEINT_MXSTEP: usize = 500;

/// Legacy `odeint`-style interface.
///
/// `scipy.integrate.odeint(func, y0, t)` at its defaults: ODEPACK's LSODA (the port in
/// [`lsoda`], automatic Adams/BDF switching, finite-difference Jacobian) at
/// `rtol = atol = 1.49012e-8`, `mxstep = 500`, called once per output time with `itask = 1`,
/// so the solver steps past each `t[i]` and interpolates `y(t[i])` from its Nordsieck history,
/// exactly as SciPy's `_odepackmodule.c` drives it. The first row is `y0` (repeated for every
/// leading time equal to `t[0]`); `t` may increase or decrease, with repeats.
///
/// Where SciPy warns (`ODEintWarning`) and returns rows it could not compute, this returns
/// [`IntegrateValidationError::IntegrationFailed`] naming ODEPACK's reason.
///
/// # Errors
/// An empty or non-finite `y0`, a non-finite or non-monotonic `t`
/// ([`IntegrateValidationError::TEvalNotSorted`], SciPy's `ValueError`), a right-hand side of
/// the wrong length, and ODEPACK failures.
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
    let increasing = t.windows(2).all(|w| w[1] >= w[0]);
    let decreasing = t.windows(2).all(|w| w[1] <= w[0]);
    if !increasing && !decreasing {
        return Err(IntegrateValidationError::TEvalNotSorted);
    }

    // SciPy copies y0 into every row whose time equals t[0], then calls LSODA for the rest.
    let t0count = t.iter().take_while(|&&ti| ti == t[0]).count();
    let mut rows = vec![y0.to_vec(); t0count];
    let mut core = lsoda::Lsoda::new(
        y0.len(),
        lsoda::LsodaSetup {
            rtol: vec![ODEINT_TOL],
            atol: vec![ODEINT_TOL],
            h0: 0.0,
            hmax: 0.0,
            mxstep: ODEINT_MXSTEP,
            jac: None,
        },
    );
    // odeint's func(y, t) as ODEPACK's f(t, y).
    let mut rhs = |ti: f64, yi: &[f64]| func(yi, ti);
    let mut y = y0.to_vec();
    let mut t_reached = t[0];
    for &tout in &t[t0count..] {
        if let Err(failure) = core.call(
            &mut rhs,
            &mut y,
            &mut t_reached,
            tout,
            lsoda::Task::Interpolate,
            tout,
        ) {
            return Err(IntegrateValidationError::IntegrationFailed {
                message: format!(
                    "{} (reached {} of {} requested times; stopped at t = {t_reached})",
                    failure.message(),
                    rows.len(),
                    t.len()
                ),
            });
        }
        rows.push(y.clone());
    }
    Ok(rows)
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

    fn assert_rows_bits(label: &str, rows: &[Vec<f64>], want: &[f64]) {
        assert_eq!(rows.len(), want.len(), "{label}: rows");
        for (i, (row, w)) in rows.iter().zip(want).enumerate() {
            assert_eq!(
                row[0].to_bits(),
                w.to_bits(),
                "{label} row {i}: fsci {:?}, SciPy {w:?}",
                row[0]
            );
        }
    }

    /// `scipy.integrate.odeint` 1.17.1 at its defaults (rtol = atol = 1.49012e-8, LSODA with
    /// itask = 1 per output time, finite-difference Jacobian): every row is ODEPACK's
    /// interpolation (DINTDY) at that time, printed by SciPy and asserted bit for bit (n = 1,
    /// so no LAPACK arithmetic is involved).
    #[test]
    fn odeint_rows_are_scipys_lsoda_interpolants() {
        let t = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0];
        let rows = odeint(&mut |y: &[f64], t: f64| vec![-y[0] * t.cos()], &[1.0], &t)
            .expect("nonstiff odeint");
        assert_rows_bits(
            "cos decay",
            &rows,
            &[
                1.0,
                0.6191388879730072,
                0.43107590913850996,
                0.4028070605001435,
                2.608887530046163,
                1.7229206092686842,
            ],
        );
        let rows = odeint(
            &mut |y: &[f64], t: f64| vec![-1000.0 * (y[0] - t.cos())],
            &[0.0],
            &t,
        )
        .expect("stiff odeint");
        assert_rows_bits(
            "stiff cos",
            &rows,
            &[
                0.0,
                0.8780611083804218,
                0.541143235584491,
                -0.41523712836919385,
                0.2827029769136198,
                -0.8396147102545395,
            ],
        );
    }

    /// Repeated output times are allowed (SciPy copies y0 for every leading t0 and
    /// interpolates again for a repeated later time), and so is a decreasing grid.
    #[test]
    fn odeint_repeated_and_decreasing_times_match_scipy() {
        let rows = odeint(
            &mut |y: &[f64], t: f64| vec![-1000.0 * (y[0] - t.cos())],
            &[0.0],
            &[0.0, 0.0, 1.0, 1.0, 3.0],
        )
        .expect("repeated times");
        assert_rows_bits(
            "repeated",
            &rows,
            &[
                0.0,
                0.0,
                0.5411432355845051,
                0.5411432355845051,
                -0.9898503878100214,
            ],
        );
        let rows = odeint(
            &mut |y: &[f64], t: f64| vec![-y[0] * t.cos()],
            &[1.0],
            &[3.0, 2.0, 0.0],
        )
        .expect("decreasing times");
        assert_rows_bits(
            "decreasing",
            &rows,
            &[1.0, 0.4638577416108029, 1.1515629173033617],
        );
    }

    #[test]
    fn odeint_single_point_rejects_non_finite_time() {
        let err = odeint(&mut |_y, _t| vec![0.0], &[1.0], &[f64::NAN])
            .expect_err("single-point odeint should validate t");
        assert_eq!(err, IntegrateValidationError::NonFiniteSpan);
    }
}
