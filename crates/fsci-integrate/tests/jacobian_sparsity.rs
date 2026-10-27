//! Public BDF/Radau sparsity regressions and an opt-in live SciPy comparison.

use fsci_integrate::{
    IntegrateValidationError, JacobianSparsity, SolveIvpOptions, SolverKind, ToleranceValue,
    solve_ivp, solve_ivp_with_audit, sync_audit_ledger,
};
use fsci_runtime::{AuditAction, RuntimeMode};

#[test]
fn zero_jacobian_sparsity_rejects_nonfinite_initial_rhs() {
    let zero = JacobianSparsity::from_rows(&[vec![]]).unwrap();
    for method in [SolverKind::Bdf, SolverKind::Radau] {
        for jac_sparsity in [None, Some(&zero)] {
            let options = SolveIvpOptions {
                t_span: (0.0, 1.0),
                y0: &[1.0],
                method,
                // Bypass the initial-step probe: the Jacobian path must validate
                // the baseline even when the pattern needs no perturbation calls.
                first_step: Some(0.1),
                jac_sparsity,
                ..SolveIvpOptions::default()
            };
            for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let mut calls = 0;
                let error = solve_ivp(
                    &mut |_t, _y| {
                        calls += 1;
                        vec![value]
                    },
                    &options,
                )
                .unwrap_err();
                assert_eq!(error, IntegrateValidationError::NonFiniteJacobian);
                assert_eq!(calls, 1, "reject the baseline before perturbing the RHS");
            }
            // A genuinely constant, finite RHS has the same zero structure and
            // remains valid; the exact solution is y(t) = 1 + t.
            let result = solve_ivp(&mut |_t, _y| vec![1.0], &options).unwrap();
            assert!(result.success, "{method:?}: {}", result.message);
            assert!((result.y.last().unwrap()[0] - 2.0).abs() < 1.0e-12);
        }
    }
}

#[test]
fn numerical_jacobian_retries_reject_overflow_before_rhs_callback() {
    let diagonal = JacobianSparsity::from_rows(&[vec![0]]).unwrap();
    for method in [SolverKind::Bdf, SolverKind::Radau] {
        for mode in [RuntimeMode::Strict, RuntimeMode::Hardened] {
            for jac_sparsity in [None, Some(&diagonal)] {
                for direction in [-1.0, 1.0] {
                    // The first relative perturbation remains finite. A constant
                    // RHS then forces the tenfold cancellation retry beyond the
                    // finite range, in either direction.
                    let y0 = [direction * (f64::MAX * (1.0 - 1.0e-7))];
                    let mut calls = 0;
                    let error = solve_ivp(
                        &mut |_t, y| {
                            assert!(
                                y.iter().all(|value| value.is_finite()),
                                "{method:?}/{mode:?}: RHS received a non-finite state"
                            );
                            calls += 1;
                            vec![direction]
                        },
                        &SolveIvpOptions {
                            t_span: (0.0, 1.0),
                            y0: &y0,
                            method,
                            mode,
                            // Avoid the initial-step probe so this specifically
                            // exercises Jacobian initialization and its retry.
                            first_step: Some(0.1),
                            jac_sparsity,
                            ..SolveIvpOptions::default()
                        },
                    )
                    .unwrap_err();
                    assert_eq!(error, IntegrateValidationError::NonFiniteJacobian);
                    assert_eq!(calls, 2, "baseline and finite first perturbation only");
                }
            }
        }
    }
}

#[test]
fn implicit_jacobian_sparsity_reduces_rhs_calls_with_identical_solutions() {
    let n = 48_usize;
    let rows: Vec<Vec<usize>> = (0..n)
        .map(|i| (i.saturating_sub(1)..=(i + 1).min(n - 1)).collect())
        .collect();
    let pattern = JacobianSparsity::from_rows(&rows).unwrap();
    let y0: Vec<f64> = (0..n)
        .map(|i| {
            let theta = (i + 1) as f64 * std::f64::consts::PI / (n + 1) as f64;
            theta.sin() + 0.25 * (3.0 * theta).sin()
        })
        .collect();
    let times = [0.0, 0.05, 0.1, 0.25, 0.5];
    for method in [SolverKind::Bdf, SolverKind::Radau] {
        for mode in [RuntimeMode::Strict, RuntimeMode::Hardened] {
            let run = |jac_sparsity| {
                let mut calls = 0;
                let mut rhs = |_t: f64, y: &[f64]| {
                    calls += 1;
                    (0..n)
                        .map(|i| {
                            let left = if i > 0 { y[i - 1] } else { 0.0 };
                            let right = if i + 1 < n { y[i + 1] } else { 0.0 };
                            100.0 * (left - 2.0 * y[i] + right) - 0.5 * y[i]
                        })
                        .collect()
                };
                let result = solve_ivp(
                    &mut rhs,
                    &SolveIvpOptions {
                        t_span: (0.0, 0.5),
                        y0: &y0,
                        method,
                        mode,
                        t_eval: Some(&times),
                        rtol: 1e-7,
                        atol: ToleranceValue::Scalar(1e-10),
                        jac_sparsity,
                        ..SolveIvpOptions::default()
                    },
                )
                .unwrap();
                assert!(result.success, "{method:?}: {}", result.message);
                (result, calls)
            };
            let (dense, dense_calls) = run(None);
            let (grouped, grouped_calls) = run(Some(&pattern));
            assert_eq!(dense, grouped, "same arithmetic after Jacobian assembly");
            assert!(
                dense_calls - grouped_calls >= (n - pattern.group_count()) * grouped.njev,
                "{method:?}: dense={dense_calls}, grouped={grouped_calls}, njev={}",
                grouped.njev
            );
            for (&t, row) in grouped.t.iter().zip(&grouped.y) {
                let theta = std::f64::consts::PI / (n + 1) as f64;
                let lambda1 = -0.5 - 200.0 * (1.0 - theta.cos());
                let lambda3 = -0.5 - 200.0 * (1.0 - (3.0 * theta).cos());
                for (i, &actual) in row.iter().enumerate() {
                    let angle = (i + 1) as f64 * theta;
                    let exact = angle.sin() * (lambda1 * t).exp()
                        + 0.25 * (3.0 * angle).sin() * (lambda3 * t).exp();
                    assert!((actual - exact).abs() < 2e-7, "{method:?}, t={t}, i={i}");
                }
            }
        }
    }
}

#[test]
fn implicit_jacobian_sparsity_is_reused_on_nonlinear_refreshes() {
    let n = 16;
    let pattern =
        JacobianSparsity::from_rows(&(0..n).map(|i| vec![i]).collect::<Vec<_>>()).unwrap();
    let y0 = vec![2.0; n];
    for method in [SolverKind::Bdf, SolverKind::Radau] {
        let run = |jac_sparsity| {
            let mut calls = 0;
            let result = solve_ivp(
                &mut |_t, y| {
                    calls += 1;
                    y.iter()
                        .enumerate()
                        .map(|(i, &yi)| -((i + 1) as f64) * yi * yi)
                        .collect()
                },
                &SolveIvpOptions {
                    t_span: (0.0, 10.0),
                    y0: &y0,
                    method,
                    t_eval: Some(&[0.1, 1.0, 10.0]),
                    rtol: 1e-7,
                    atol: ToleranceValue::Scalar(1e-10),
                    jac_sparsity,
                    ..SolveIvpOptions::default()
                },
            )
            .unwrap();
            (result, calls)
        };
        let (dense, dense_calls) = run(None);
        let (grouped, grouped_calls) = run(Some(&pattern));
        assert!(grouped.success);
        assert!(grouped.njev > 1, "{method:?} must refresh its Jacobian");
        assert_eq!(dense, grouped);
        assert!(dense_calls - grouped_calls >= (n - 1) * grouped.njev);
        for (&t, row) in grouped.t.iter().zip(&grouped.y) {
            for (i, &actual) in row.iter().enumerate() {
                let exact = 2.0 / (1.0 + 2.0 * (i + 1) as f64 * t);
                assert!((actual - exact).abs() <= 2e-6 * exact + 1e-10);
            }
        }
    }
}

#[test]
fn implicit_jacobian_sparsity_validation_precedence_and_audit() {
    let wrong_size = JacobianSparsity::from_rows(&[vec![0], vec![1]]).unwrap();
    let diagonal = JacobianSparsity::from_rows(&[vec![0]]).unwrap();
    let zero = JacobianSparsity::from_rows(&[vec![]]).unwrap();
    fn jac(_t: f64, _y: &[f64]) -> Vec<Vec<f64>> {
        vec![vec![-1.0]]
    }
    // The public audit API must distinguish requests differing only in sparsity.
    let fingerprint = |options: &SolveIvpOptions<'_>| {
        let ledger = sync_audit_ledger();
        let error = solve_ivp_with_audit(
            &mut |_t, _y| panic!("invalid max_step fails before evaluation"),
            &SolveIvpOptions {
                max_step: f64::NAN,
                ..options.clone()
            },
            &ledger,
        )
        .unwrap_err();
        assert_eq!(error, IntegrateValidationError::NonFiniteMaxStep);
        let guard = ledger.lock().unwrap();
        guard.entries()[0].input_fingerprint.clone()
    };
    for method in [SolverKind::Bdf, SolverKind::Radau] {
        let options = SolveIvpOptions {
            t_span: (0.0, 1.0),
            y0: &[1.0],
            method,
            jac_sparsity: Some(&wrong_size),
            ..SolveIvpOptions::default()
        };
        let ledger = sync_audit_ledger();
        let error = solve_ivp_with_audit(
            &mut |_t, _y| panic!("invalid sparsity must fail before RHS evaluation"),
            &options,
            &ledger,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            IntegrateValidationError::InvalidJacobianSparsity { .. }
        ));
        assert!(ledger.lock().unwrap().entries().iter().any(|entry| {
            matches!(
                &entry.action,
                AuditAction::FailClosed { reason } if reason == "invalid_jacobian_sparsity"
            )
        }));

        let analytic = solve_ivp(
            &mut |_t, y| vec![-y[0]],
            &SolveIvpOptions {
                jac: Some(jac),
                ..options.clone()
            },
        )
        .unwrap();
        assert!(analytic.success, "analytic Jacobians ignore jac_sparsity");
        let with_diagonal = SolveIvpOptions {
            jac_sparsity: Some(&diagonal),
            ..options.clone()
        };
        let with_zero = SolveIvpOptions {
            jac_sparsity: Some(&zero),
            ..options
        };
        assert_ne!(with_diagonal, with_zero);
        assert_ne!(fingerprint(&with_diagonal), fingerprint(&with_zero));
    }
    let explicit = solve_ivp(
        &mut |_t, y| vec![-y[0]],
        &SolveIvpOptions {
            t_span: (0.0, 1.0),
            y0: &[1.0],
            method: SolverKind::Rk45,
            jac_sparsity: Some(&wrong_size),
            ..SolveIvpOptions::default()
        },
    )
    .unwrap();
    assert!(explicit.success, "explicit methods ignore jac_sparsity");
}

/// Run explicitly with FSCI_SCIPY_PYTHON pointing to the pinned oracle environment.
/// Both dense and grouped paths are compared, so a broken coloring cannot hide behind
/// a reference generated with the same implementation.
#[test]
#[ignore = "requires Python with SciPy 1.17.1 (set FSCI_SCIPY_PYTHON)"]
fn implicit_jacobian_sparsity_live_scipy() {
    let python = std::env::var_os("FSCI_SCIPY_PYTHON")
        .unwrap_or_else(|| std::ffi::OsString::from("python3"));
    let output = std::process::Command::new(python)
        .args([
            "-c",
            r#"
import json
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix
assert scipy.__version__ == '1.17.1', scipy.__version__
results = {}
n = 32
for case in ['stencil', 'nonlinear']:
    y0 = np.array([0.25 + 0.5*np.sin((i+1)*np.pi/(n+1)) for i in range(n)]) if case == 'stencil' else np.full(n, 2.)
    times = [0., 0.1, 0.5, 1., 2.] if case == 'stencil' else [0.1, 1., 10.]
    mask = np.eye(n, dtype=bool)
    if case == 'stencil':
        mask |= np.eye(n, k=1, dtype=bool) | np.eye(n, k=-1, dtype=bool)
    for method in ['BDF', 'Radau']:
        for colored in [False, True]:
            calls = [0]
            def fun(t, y):
                calls[0] += 1
                if case == 'nonlinear':
                    return np.array([-(i+1.)*y[i]*y[i] for i in range(n)])
                return np.array([100.*((y[i-1] if i else 0.) - 2.*y[i] + (y[i+1] if i+1<n else 0.)) + y[i]*(1.-y[i]) for i in range(n)])
            sol = solve_ivp(fun, (0., times[-1]), y0, method=method, t_eval=times,
                            rtol=1e-7, atol=1e-10, jac_sparsity=csr_matrix(mask) if colored else None)
            assert sol.success, sol.message
            results[f'{case}_{method}_{colored}'] = dict(y=sol.y.T.tolist(), calls=calls[0],
                nfev=sol.nfev, njev=sol.njev, nlu=sol.nlu)
print(json.dumps(dict(version=scipy.__version__, numpy=np.__version__, results=results), allow_nan=False))
"#,
        ])
        .output()
        .expect("launch the SciPy oracle");
    assert!(
        output.status.success(),
        "SciPy oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let oracle: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    println!("SciPy {} NumPy {}", oracle["version"], oracle["numpy"]);
    let n = 32_usize;
    let mut compared = 0;
    for nonlinear in [false, true] {
        let case = if nonlinear { "nonlinear" } else { "stencil" };
        let y0: Vec<f64> = (0..n)
            .map(|i| {
                if nonlinear {
                    2.0
                } else {
                    0.25 + 0.5 * ((i + 1) as f64 * std::f64::consts::PI / (n + 1) as f64).sin()
                }
            })
            .collect();
        let times = if nonlinear {
            vec![0.1, 1.0, 10.0]
        } else {
            vec![0.0, 0.1, 0.5, 1.0, 2.0]
        };
        let rows: Vec<Vec<usize>> = (0..n)
            .map(|i| {
                if nonlinear {
                    vec![i]
                } else {
                    (i.saturating_sub(1)..=(i + 1).min(n - 1)).collect()
                }
            })
            .collect();
        let pattern = JacobianSparsity::from_rows(&rows).unwrap();
        for (method, method_name) in [(SolverKind::Bdf, "BDF"), (SolverKind::Radau, "Radau")] {
            let mut rust_calls = [0; 2];
            for (colored, suffix) in [(false, "False"), (true, "True")] {
                let mut calls = 0;
                let result = solve_ivp(
                    &mut |_t, y| {
                        calls += 1;
                        (0..n)
                            .map(|i| {
                                if nonlinear {
                                    -((i + 1) as f64) * y[i] * y[i]
                                } else {
                                    let left = if i > 0 { y[i - 1] } else { 0.0 };
                                    let right = if i + 1 < n { y[i + 1] } else { 0.0 };
                                    100.0 * (left - 2.0 * y[i] + right) + y[i] * (1.0 - y[i])
                                }
                            })
                            .collect()
                    },
                    &SolveIvpOptions {
                        t_span: (0.0, *times.last().unwrap()),
                        y0: &y0,
                        method,
                        t_eval: Some(&times),
                        rtol: 1e-7,
                        atol: ToleranceValue::Scalar(1e-10),
                        jac_sparsity: colored.then_some(&pattern),
                        ..SolveIvpOptions::default()
                    },
                )
                .unwrap();
                assert!(result.success, "{case} {method_name}: {}", result.message);
                let key = format!("{case}_{method_name}_{suffix}");
                let expected = &oracle["results"][&key];
                let expected_y = expected["y"].as_array().unwrap();
                assert_eq!(result.y.len(), expected_y.len());
                let mut max_error = 0.0_f64;
                for (actual_row, expected_row) in result.y.iter().zip(expected_y) {
                    let expected_row = expected_row.as_array().unwrap();
                    assert_eq!(actual_row.len(), expected_row.len());
                    for (&actual, expected) in actual_row.iter().zip(expected_row) {
                        let expected = expected.as_f64().unwrap();
                        let error = (actual - expected).abs();
                        max_error = max_error.max(error);
                        assert!(
                            error <= 3e-7 * expected.abs().max(1.0),
                            "{key}: {actual} vs {expected}"
                        );
                        compared += 1;
                    }
                }
                rust_calls[usize::from(colored)] = calls;
                println!(
                    "{key}: max_abs={max_error:.3e}; calls Rust={calls} SciPy={}; nfev/njev/nlu Rust={}/{}/{} SciPy={}/{}/{}",
                    expected["calls"],
                    result.nfev,
                    result.njev,
                    result.nlu,
                    expected["nfev"],
                    expected["njev"],
                    expected["nlu"]
                );
            }
            assert!(
                rust_calls[1] < rust_calls[0],
                "{case} {method_name}: grouping must execute"
            );
        }
    }
    assert_eq!(compared, 1024, "all eight solutions must be compared");
}
