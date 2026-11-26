//! LSODA must not certify a state containing a non-finite component.

use fsci_integrate::{SolveIvpOptions, SolverKind, odeint, solve_ivp};

#[test]
fn nonfinite_rhs_fails_at_start_or_after_accepted_steps() {
    let times = [0.0, 0.1, 0.2, 0.5, 1.0];
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for cutoff in [0.0, 0.35] {
            let rhs = |t: f64| vec![1.0, if t >= cutoff { invalid } else { 0.0 }];
            let err = odeint(&mut |_, t| rhs(t), &[0.0, 2.0], &times)
                .expect_err("odeint must fail without returning non-finite output rows");
            assert!(err.to_string().contains("non-finite"));

            let result = solve_ivp(
                &mut |t, _| rhs(t),
                &SolveIvpOptions {
                    method: SolverKind::Lsoda,
                    y0: &[0.0, 2.0],
                    t_span: (0.0, 1.0),
                    t_eval: Some(&times),
                    first_step: Some(0.1),
                    max_step: 0.1,
                    ..SolveIvpOptions::default()
                },
            )
            .unwrap();
            assert_eq!(result.status, -1);
            assert!(!result.success);
            assert!(result.y.iter().flatten().all(|v| v.is_finite()));
            if cutoff == 0.0 {
                assert_eq!(result.t, [0.0]);
            } else {
                assert_eq!(result.t, [0.0, 0.1, 0.2]);
            }
        }
    }
}

#[test]
fn finite_components_including_zero_slope_integrate_in_both_directions() {
    for (start, end) in [(0.0, 1.0), (1.0, 0.0)] {
        let times = [start, 0.5, end];
        let rhs = |_: f64, _: &[f64]| vec![1.0, 0.0];
        let rows = odeint(&mut |y, t| rhs(t, y), &[start, 2.0], &times).unwrap();
        let ivp = solve_ivp(
            &mut |t, y| rhs(t, y),
            &SolveIvpOptions {
                method: SolverKind::Lsoda,
                t_span: (start, end),
                y0: &[start, 2.0],
                t_eval: Some(&times),
                ..SolveIvpOptions::default()
            },
        )
        .unwrap();
        assert!(ivp.success);
        for result in [&rows, &ivp.y] {
            assert_eq!(result.len(), times.len());
            for (state, &t) in result.iter().zip(&times) {
                assert!((state[0] - t).abs() < 1e-12);
                assert_eq!(state[1], 2.0);
            }
        }
    }
}
