//! Regression coverage for SciPy odeint's per-component error controls and stiff solver options.

use fsci_integrate::{IntegrateValidationError, OdeintOptions, ToleranceValue, odeint, odeint_with_options};
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn configurable_budget_reaches_distant_output_without_extra_samples() {
    let mut oscillator = |y: &[f64], _: f64| vec![y[1], -y[0]];
    let times = [0.0, 200.0];
    let err = odeint(&mut oscillator, &[1.0, 0.0], &times).unwrap_err();
    assert!(err.to_string().contains("Excess work"));
    let result = odeint_with_options(
        &mut oscillator, &[1.0, 0.0], &times,
        &OdeintOptions { mxstep: 5000, ..Default::default() },
    ).unwrap();
    assert_eq!(result.len(), times.len());
    assert!((result[1][0] - 200_f64.cos()).abs() < 5e-6);
    assert!((result[1][1] + 200_f64.sin()).abs() < 5e-6);
}

fn multiscale(y: &[f64], _: f64) -> Vec<f64> {
    vec![-y[0], -2000.0 * y[1], -0.5 * y[2]]
}

#[test]
fn component_tolerances_resolve_small_fast_state_and_use_banded_jacobian() {
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    fn diagonal(_t: f64, _y: &[f64]) -> Vec<Vec<f64>> {
        CALLS.fetch_add(1, Ordering::Relaxed);
        vec![vec![-1.0, -2000.0, -0.5]]
    }
    let y0 = [1.0, 1e-14, 1e8];
    let times = [0.0, 0.0001, 0.002, 0.1, 1.0];
    let opts = OdeintOptions {
        rtol: ToleranceValue::Vector(vec![0.0, 1e-8, 1e-10]),
        atol: ToleranceValue::Vector(vec![1e-10, 1e-26, 0.0]),
        jac: Some(diagonal), bandwidth: Some((0, 0)), ..Default::default()
    };
    let result = odeint_with_options(&mut multiscale, &y0, &times, &opts).unwrap();
    assert!(CALLS.load(Ordering::Relaxed) > 0);
    for (&t, row) in times.iter().zip(result) {
        assert!((row[0] - (-t).exp()).abs() < 2e-9);
        assert!((row[1] - y0[1] * (-2000.0*t).exp()).abs() < 4e-24 + 1e-7*y0[1]*(-2000.0*t).exp());
        assert!((row[2] / y0[2] - (-0.5*t).exp()).abs() < 2e-9);
    }
}

#[test]
fn backward_steps_repeats_and_interval_step_caps() {
    let grid = [1.0, 1.0, 0.5, 0.5, 0.0];
    let mut decay = |y: &[f64], _: f64| vec![-y[0]];
    let opts = OdeintOptions { h0: -0.001, hmax: 0.01, mxstep: 100, ..Default::default() };
    let out = odeint_with_options(&mut decay, &[1.0], &grid, &opts).unwrap();
    assert_eq!(out[0], out[1]);
    assert_eq!(out[2], out[3]);
    for (row, t) in out.iter().zip(grid) {
        assert!((row[0] - (1.0-t).exp()).abs() < 1e-6);
    }
    let limited = OdeintOptions { mxstep: 10, ..opts.clone() };
    assert!(odeint_with_options(&mut decay, &[1.0], &grid, &limited).unwrap_err().to_string().contains("Excess work"));
    assert!(odeint_with_options(&mut decay, &[1.0], &grid,
        &OdeintOptions { h0: 0.001, ..opts }).is_err());
}

#[test]
fn invalid_options_fail_before_rhs_and_bad_packed_jacobian_is_reported() {
    let invalid = [
        OdeintOptions { rtol: ToleranceValue::Vector(vec![1e-7; 2]), ..Default::default() },
        OdeintOptions { atol: ToleranceValue::Vector(vec![]), ..Default::default() },
        OdeintOptions { rtol: ToleranceValue::Scalar(f64::INFINITY), ..Default::default() },
        OdeintOptions { atol: ToleranceValue::Scalar(-1.0), ..Default::default() },
        OdeintOptions { bandwidth: Some((3,0)), ..Default::default() },
        OdeintOptions { hmax: -1.0, ..Default::default() },
    ];
    for opt in invalid {
        assert!(odeint_with_options(&mut |_,_| panic!("invalid option evaluated RHS"), &[1.0;3], &[0.0,1.0], &opt).is_err());
    }
    let opt = OdeintOptions { jac: Some(|_,_| vec![]), bandwidth: Some((0, 0)), ..Default::default() };
    assert!(matches!(
        odeint_with_options(&mut multiscale, &[1.0;3], &[0.0, 1.0], &opt),
        Err(IntegrateValidationError::BandedJacobianWrongShape { expected_rows: 1, expected_cols: 3 })
    ));
}
