//! Public BDF/Radau sparsity and LSODA banded-Jacobian regressions, including
//! live SciPy comparisons.

use fsci_integrate::{
    IntegrateValidationError, JacobianSparsity, SolveIvpOptions, SolverKind, ToleranceValue,
    solve_ivp, solve_ivp_with_audit, sync_audit_ledger,
};
use fsci_runtime::{AuditAction, RuntimeMode};

fn sparse_graph_rhs(y: &[f64], reaction: bool, direction: f64) -> Vec<f64> {
    let n = y.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        let row = (17 * i) % n;
        let left = if i > 0 { y[(17 * (i - 1)) % n] } else { 0.0 };
        let right = if i + 1 < n {
            y[(17 * (i + 1)) % n]
        } else {
            0.0
        };
        let local = if reaction {
            y[row] * (1.0 - y[row])
        } else {
            -0.5 * y[row]
        };
        result[row] = direction * (80.0 * (left - 2.0 * y[row] + right) + local);
    }
    result
}

fn sparse_graph_jacobian(y: &[f64], reaction: bool, direction: f64) -> fsci_sparse::CscMatrix {
    let n = y.len();
    let mut inverse = vec![0; n];
    for i in 0..n {
        inverse[(17 * i) % n] = i;
    }
    let mut values = Vec::with_capacity(3 * n - 2);
    let mut indices = Vec::with_capacity(3 * n - 2);
    let mut indptr = vec![0];
    for column in 0..n {
        let i = inverse[column];
        let diagonal = -160.0
            + if reaction {
                1.0 - 2.0 * y[column]
            } else {
                -0.5
            };
        let mut entries = vec![(column, direction * diagonal)];
        if i > 0 {
            entries.push(((17 * (i - 1)) % n, direction * 80.0));
        }
        if i + 1 < n {
            entries.push(((17 * (i + 1)) % n, direction * 80.0));
        }
        entries.sort_unstable_by_key(|entry| entry.0);
        for (row, value) in entries {
            indices.push(row);
            values.push(value);
        }
        indptr.push(values.len());
    }
    fsci_sparse::CscMatrix::from_components(
        fsci_sparse::Shape2D::new(n, n),
        values,
        indices,
        indptr,
        true,
    )
    .unwrap()
}

fn sparse_heat_jacobian(_t: f64, y: &[f64]) -> fsci_sparse::CscMatrix {
    sparse_graph_jacobian(y, false, 1.0)
}

fn sparse_heat_reverse_jacobian(_t: f64, y: &[f64]) -> fsci_sparse::CscMatrix {
    sparse_graph_jacobian(y, false, -1.0)
}

fn sparse_reaction_jacobian(_t: f64, y: &[f64]) -> fsci_sparse::CscMatrix {
    sparse_graph_jacobian(y, true, 1.0)
}

#[test]
fn bdf_sparse_jacobians_match_live_scipy() {
    let python = std::env::var("FSCI_PYTHON").unwrap_or_else(|_| "python3".into());
    let required = std::env::var("FSCI_REQUIRE_SCIPY_ORACLE").as_deref() == Ok("1");
    let available = std::process::Command::new(&python)
        .args(["-c", "import scipy, numpy"])
        .output()
        .is_ok_and(|output| output.status.success());
    if !available {
        assert!(!required, "required SciPy oracle unavailable via {python}");
        eprintln!("SKIP sparse BDF oracle: missing SciPy; compared cases = 0");
        return;
    }
    let output = std::process::Command::new(&python)
        .args([
            "-c",
            r#"
import json, os
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.sparse import csc_matrix, diags
if os.environ.get('FSCI_REQUIRE_SCIPY_ORACLE') == '1':
    assert (scipy.__version__, np.__version__) == ('1.17.1', '2.4.3')
n = 64
permutation = 17*np.arange(n) % n
inverse = np.argsort(permutation)
theta = (np.arange(n)+1)*np.pi/(n+1)
initial = np.sin(theta) + .2*np.sin(11*theta)
y0 = initial[inverse]
laplacian = diags([80.*np.ones(n-1), -160.*np.ones(n), 80.*np.ones(n-1)], [-1, 0, 1], format='csc')
matrix = csc_matrix(laplacian[inverse, :][:, inverse])
mask = matrix.astype(bool)
rows = []
for kind in ['linear_forward', 'linear_backward', 'reaction']:
    reaction = kind == 'reaction'
    direction = -1. if kind == 'linear_backward' else 1.
    start = .5 if direction < 0 else 0.
    stop = start + direction*.5
    times = start + direction*np.array([0., .001, .01, .05, .2, .5])
    samples = start + direction*np.array([.003, .03, .12, .42])
    def fun(t, y):
        return direction*(matrix@y + (y*(1-y) if reaction else -.5*y))
    def jac(t, y):
        return direction*(matrix + diags(1.-2.*y if reaction else np.full(n, -.5)))
    def event(t, y):
        return y[permutation[n//2]] - .75
    event.terminal = True
    event.direction = -1
    for analytic in [False, True]:
        kwargs = dict(jac=jac) if analytic else dict(jac_sparsity=mask)
        sol = solve_ivp(fun, (start,stop), y0, method='BDF', rtol=1e-8, atol=1e-10,
                        t_eval=times, dense_output=True, **kwargs)
        assert sol.success and sol.nlu > 0 and sol.njev > 0, sol.message
        row = dict(kind=kind, analytic=analytic, y=sol.y.T.tolist(),
                   dense=sol.sol(samples).T.tolist(), nfev=int(sol.nfev),
                   njev=int(sol.njev), nlu=int(sol.nlu))
        if not reaction:
            stopped = solve_ivp(fun, (start,stop), y0, method='BDF', rtol=1e-8, atol=1e-10,
                                events=event, **kwargs)
            assert stopped.status == 1 and len(stopped.t_events[0]) == 1
            row['event_time'] = float(stopped.t_events[0][0])
        rows.append(row)
print(json.dumps(dict(scipy=scipy.__version__, numpy=np.__version__, rows=rows), allow_nan=False))
"#,
        ])
        .output()
        .expect("launch sparse BDF SciPy oracle");
    assert!(
        output.status.success(),
        "SciPy failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let oracle: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let rows = oracle["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 6);
    let n = 64;
    let mut y0 = vec![0.0; n];
    let mut supports = vec![Vec::new(); n];
    for i in 0..n {
        let row = (17 * i) % n;
        let theta = (i + 1) as f64 * std::f64::consts::PI / (n + 1) as f64;
        y0[row] = theta.sin() + 0.2 * (11.0 * theta).sin();
        supports[row] = (i.saturating_sub(1)..=(i + 1).min(n - 1))
            .map(|j| (17 * j) % n)
            .collect();
    }
    let pattern = JacobianSparsity::from_rows(&supports).unwrap();
    let mut compared_values = 0;
    let mut compared_events = 0;
    for row in rows {
        let kind = row["kind"].as_str().unwrap();
        let analytic = row["analytic"].as_bool().unwrap();
        let reaction = kind == "reaction";
        let direction = if kind == "linear_backward" { -1.0 } else { 1.0 };
        let start = if direction < 0.0 { 0.5 } else { 0.0 };
        let times = [0.0, 0.001, 0.01, 0.05, 0.2, 0.5].map(|t| start + direction * t);
        let samples = [0.003, 0.03, 0.12, 0.42].map(|t| start + direction * t);
        let callback: fsci_integrate::SparseJacFn = if reaction {
            sparse_reaction_jacobian
        } else if direction < 0.0 {
            sparse_heat_reverse_jacobian
        } else {
            sparse_heat_jacobian
        };
        let options = SolveIvpOptions {
            t_span: (start, start + direction * 0.5),
            y0: &y0,
            method: SolverKind::Bdf,
            t_eval: Some(&times),
            dense_output: true,
            rtol: 1e-8,
            atol: ToleranceValue::Scalar(1e-10),
            jac_sparse: analytic.then_some(callback),
            jac_sparsity: (!analytic).then_some(&pattern),
            ..SolveIvpOptions::default()
        };
        let mut calls = 0;
        let solution = solve_ivp(
            &mut |_t, y| {
                calls += 1;
                sparse_graph_rhs(y, reaction, direction)
            },
            &options,
        )
        .unwrap();
        assert!(solution.success, "{kind}/{analytic}: {}", solution.message);
        assert!(solution.nlu > 0 && solution.njev > 0);
        if analytic {
            assert_eq!(
                calls, solution.nfev,
                "analytic Jacobians need no hidden RHS calls"
            );
        } else {
            assert!(
                calls > solution.nfev,
                "the sparse numerical Jacobian must be evaluated"
            );
        }
        let dense = solution
            .sol
            .as_ref()
            .unwrap()
            .evaluate_many(&samples)
            .unwrap();
        for (actual, expected, sample_times) in [
            (&solution.y, &row["y"], times.as_slice()),
            (&dense, &row["dense"], samples.as_slice()),
        ] {
            let expected = expected.as_array().unwrap();
            assert_eq!(actual.len(), expected.len());
            for ((actual_row, expected_row), &t) in actual.iter().zip(expected).zip(sample_times) {
                let expected_row = expected_row.as_array().unwrap();
                assert_eq!(actual_row.len(), expected_row.len());
                for (i, (&actual, expected)) in actual_row.iter().zip(expected_row).enumerate() {
                    let expected = expected.as_f64().unwrap();
                    assert!(
                        (actual - expected).abs() <= 3e-7 * expected.abs().max(1.0),
                        "{kind}/{analytic} t={t} component={i}: Rust={actual} SciPy={expected}"
                    );
                    compared_values += 1;
                }
                if !reaction {
                    let elapsed = direction * (t - start);
                    for i in 0..n {
                        let theta = (i + 1) as f64 * std::f64::consts::PI / (n + 1) as f64;
                        let eigenvalue = |k: f64| {
                            -0.5 - 320.0
                                * (k * std::f64::consts::PI / (2 * (n + 1)) as f64)
                                    .sin()
                                    .powi(2)
                        };
                        let exact = theta.sin() * (eigenvalue(1.0) * elapsed).exp()
                            + 0.2 * (11.0 * theta).sin() * (eigenvalue(11.0) * elapsed).exp();
                        assert!(
                            (actual_row[(17 * i) % n] - exact).abs() <= 3e-7,
                            "independent eigenmode solution disagrees for {kind}/{analytic} t={t}"
                        );
                    }
                }
            }
        }
        if !reaction {
            let event = fsci_integrate::EventSpec::terminal(|_t, y| {
                y[(17 * (y.len() / 2)) % y.len()] - 0.75
            })
            .with_direction(-1.0);
            let events = vec![event];
            let stopped = solve_ivp(
                &mut |_t, y| sparse_graph_rhs(y, false, direction),
                &SolveIvpOptions {
                    t_eval: None,
                    events: Some(events),
                    ..options
                },
            )
            .unwrap();
            assert!(stopped.success && stopped.status == 1);
            let event_time = stopped.t_events.as_ref().unwrap()[0][0];
            let expected_time = row["event_time"].as_f64().unwrap();
            assert!(
                (event_time - expected_time).abs() < 1e-6,
                "{kind}/{analytic}: event {event_time} vs {expected_time}"
            );
            compared_events += 1;
        }
        println!(
            "sparse BDF {kind}/{analytic}: nfev/njev/nlu={}/{}/{}; SciPy={}/{}/{}",
            solution.nfev, solution.njev, solution.nlu, row["nfev"], row["njev"], row["nlu"]
        );
    }
    assert_eq!(compared_values, 6 * 10 * 64);
    assert_eq!(compared_events, 4);
    println!(
        "sparse BDF: 6 cases, {compared_values} trajectory/dense comparisons, {compared_events} event comparisons"
    );
}

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
            if method == SolverKind::Bdf {
                // CSC LU changes the elimination order. Preserve every result
                // field other than the trajectory bits, and admit only roundoff
                // there; the independent analytical error bound below is unchanged.
                assert_eq!(dense.t, grouped.t);
                assert_eq!(dense.sol, grouped.sol);
                assert_eq!(dense.t_events, grouped.t_events);
                assert_eq!(dense.y_events, grouped.y_events);
                assert_eq!(dense.message, grouped.message);
                assert_eq!(dense.success, grouped.success);
                assert_eq!(
                    (dense.nfev, dense.njev, dense.nlu, dense.status),
                    (grouped.nfev, grouped.njev, grouped.nlu, grouped.status)
                );
                let within_roundoff = |a: f64, b: f64| {
                    (a - b).abs() <= 8.0 * f64::EPSILON * a.abs().max(b.abs()).max(1.0)
                };
                assert!(!within_roundoff(1.0, 1.0 + 32.0 * f64::EPSILON));
                assert_eq!(dense.y.len(), grouped.y.len());
                for (dense_row, sparse_row) in dense.y.iter().zip(&grouped.y) {
                    assert_eq!(dense_row.len(), sparse_row.len());
                    for (&a, &b) in dense_row.iter().zip(sparse_row) {
                        assert!(within_roundoff(a, b), "dense={a}, sparse={b}");
                    }
                }
            } else {
                assert_eq!(dense, grouped, "Radau retains dense Jacobian arithmetic");
            }
            assert!(
                dense_calls - grouped_calls >= (n - pattern.group_count()) * grouped.njev,
                "{method:?}: dense={dense_calls}, grouped={grouped_calls}, njev={}",
                grouped.njev
            );
            for (&t, row) in dense
                .t
                .iter()
                .zip(&dense.y)
                .chain(grouped.t.iter().zip(&grouped.y))
            {
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

fn banded_rhs(problem: &str, y: &[f64]) -> Vec<f64> {
    let n = y.len();
    (0..n)
        .map(|i| match problem {
            "diagonal" => -10.0_f64.powf(i as f64 / 7.0) * y[i],
            "lower" => (20.0 + i as f64).powi(2) * (if i == 0 { 1.0 } else { y[i - 1] } - y[i]),
            "upper" => (20.0 + i as f64).powi(2) * (if i + 1 == n { 1.0 } else { y[i + 1] } - y[i]),
            _ => {
                let left = if i == 0 { 0.0 } else { y[i - 1] };
                let right = if i + 1 == n { 0.0 } else { y[i + 1] };
                200.0 * (left - 2.0 * y[i] + right)
                    + if problem == "nonlinear" {
                        y[i] * (1.0 - y[i])
                    } else {
                        -0.5 * y[i]
                    }
            }
        })
        .collect()
}

fn packed_band_jacobian(problem: &str, y: &[f64]) -> Vec<Vec<f64>> {
    let (lower, upper) = match problem {
        "diagonal" => (0, 0),
        "lower" => (1, 0),
        "upper" => (0, 1),
        _ => (1, 1),
    };
    let n = y.len();
    let mut matrix = vec![vec![0.0; n]; lower + upper + 1];
    for j in 0..n {
        matrix[upper][j] = match problem {
            "diagonal" => -10.0_f64.powf(j as f64 / 7.0),
            "lower" | "upper" => -(20.0 + j as f64).powi(2),
            "nonlinear" => -400.0 + (1.0 - 2.0 * y[j]),
            _ => -400.5,
        };
        if lower > 0 && j + 1 < n {
            matrix[upper + 1][j] = if problem == "lower" {
                (21.0 + j as f64).powi(2)
            } else {
                200.0
            };
        }
        if upper > 0 && j > 0 {
            matrix[upper - 1][j] = if problem == "upper" {
                (19.0 + j as f64).powi(2)
            } else {
                200.0
            };
        }
    }
    matrix
}

/// The diagonal and both asymmetric bands exercise every packed-storage edge.
/// The nonlinear reaction/diffusion system forces Jacobian refreshes. Every
/// case also samples retained dense output between the requested output times.
#[test]
fn lsoda_banded_jacobians_match_live_scipy() {
    let python = std::env::var("FSCI_PYTHON").unwrap_or_else(|_| "python3".into());
    let required = std::env::var("FSCI_REQUIRE_SCIPY_ORACLE").as_deref() == Ok("1");
    let available = std::process::Command::new(&python)
        .args(["-c", "import scipy, numpy"])
        .output()
        .is_ok_and(|output| output.status.success());
    if !available {
        assert!(!required, "required SciPy oracle unavailable via {python}");
        eprintln!("SKIP LSODA banded oracle: missing SciPy; compared cases = 0");
        return;
    }
    let output = std::process::Command::new(&python)
        .args([
            "-c",
            r#"
import json, os
import numpy as np
import scipy
from scipy.integrate import solve_ivp
if os.environ.get('FSCI_REQUIRE_SCIPY_ORACLE') == '1':
    assert (scipy.__version__, np.__version__) == ('1.17.1', '2.4.3')
n = 32
times = [0., .001, .01, .1, .5, 1., 2.]
samples = [.003, .03, .3, .9, 1.7]
rows = []
for kind in ['heat', 'lower', 'upper', 'diagonal', 'nonlinear']:
    rates = 10.**(np.arange(n)/7.)
    y0 = (1 + np.arange(n)/n if kind == 'diagonal' else
          np.zeros(n) if kind in ['lower', 'upper'] else
          np.array([np.sin((i+1)*np.pi/(n+1)) + .2*np.sin(17*(i+1)*np.pi/(n+1)) for i in range(n)]))
    lb, ub = {'heat': (1,1), 'nonlinear': (1,1), 'lower': (1,0), 'upper': (0,1), 'diagonal': (0,0)}[kind]
    def fun(t, y):
        if kind == 'diagonal': return -rates*y
        if kind == 'lower': return np.array([(20+i)**2*((y[i-1] if i else 1.)-y[i]) for i in range(n)])
        if kind == 'upper': return np.array([(20+i)**2*((y[i+1] if i+1<n else 1.)-y[i]) for i in range(n)])
        return np.array([200.*((y[i-1] if i else 0.)-2*y[i]+(y[i+1] if i+1<n else 0.)) +
                         (y[i]*(1-y[i]) if kind == 'nonlinear' else -.5*y[i]) for i in range(n)])
    def jac(t, y):
        a = np.zeros((lb+ub+1, n))
        for j in range(n):
            a[ub,j] = (-rates[j] if kind == 'diagonal' else -(20+j)**2 if kind in ['lower','upper'] else
                       -400. + (1.-2*y[j] if kind == 'nonlinear' else -.5))
            if lb and j+1<n: a[ub+1,j] = (21+j)**2 if kind == 'lower' else 200.
            if ub and j: a[ub-1,j] = (19+j)**2 if kind == 'upper' else 200.
        return a
    for analytic in [False, True]:
        r = solve_ivp(fun, (0.,2.), y0, method='LSODA', rtol=1e-8, atol=1e-10,
                      lband=lb, uband=ub, jac=jac if analytic else None,
                      t_eval=times, dense_output=True)
        assert r.success and r.nlu > 0, (kind, analytic, r.message)
        row = dict(problem=kind, analytic=analytic, y=r.y.T.tolist(), dense=r.sol(samples).T.tolist(),
                   nfev=int(r.nfev), njev=int(r.njev), nlu=int(r.nlu))
        if kind == 'heat':
            def event(t,y): return y[n//2]-.1
            event.terminal = True
            event.direction = -1
            stopped = solve_ivp(fun, (0.,2.), y0, method='LSODA', rtol=1e-8, atol=1e-10,
                                lband=lb, uband=ub, jac=jac if analytic else None, events=event)
            assert stopped.status == 1 and len(stopped.t_events[0]) == 1
            row['event_time'] = stopped.t_events[0][0]
        rows.append(row)
print(json.dumps(dict(scipy=scipy.__version__, numpy=np.__version__, rows=rows), allow_nan=False))
"#,
        ])
        .output()
        .expect("launch banded LSODA SciPy oracle");
    assert!(
        output.status.success(),
        "SciPy failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let oracle: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let rows = oracle["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 10);
    let n = 32;
    let times = [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0];
    let samples = [0.003, 0.03, 0.3, 0.9, 1.7];
    let mut compared = 0;
    let mut compared_values = 0;
    for row in rows {
        let problem = row["problem"].as_str().unwrap();
        let analytic = row["analytic"].as_bool().unwrap();
        let (lower, upper) = match problem {
            "diagonal" => (0, 0),
            "lower" => (1, 0),
            "upper" => (0, 1),
            _ => (1, 1),
        };
        let y0: Vec<_> = (0..n)
            .map(|i| match problem {
                "diagonal" => 1.0 + i as f64 / n as f64,
                "lower" | "upper" => 0.0,
                _ => {
                    let angle = (i + 1) as f64 * std::f64::consts::PI / (n + 1) as f64;
                    angle.sin() + 0.2 * (17.0 * angle).sin()
                }
            })
            .collect();
        let jac: fsci_integrate::JacFn = match problem {
            "diagonal" => |_, y| packed_band_jacobian("diagonal", y),
            "lower" => |_, y| packed_band_jacobian("lower", y),
            "upper" => |_, y| packed_band_jacobian("upper", y),
            "nonlinear" => |_, y| packed_band_jacobian("nonlinear", y),
            _ => |_, y| packed_band_jacobian("heat", y),
        };
        let options = SolveIvpOptions {
            t_span: (0.0, 2.0),
            y0: &y0,
            method: SolverKind::Lsoda,
            t_eval: Some(&times),
            dense_output: true,
            rtol: 1e-8,
            atol: ToleranceValue::Scalar(1e-10),
            lband: Some(lower),
            uband: Some(upper),
            jac: analytic.then_some(jac),
            ..SolveIvpOptions::default()
        };
        let result = solve_ivp(&mut |_, y| banded_rhs(problem, y), &options).unwrap();
        assert!(result.success, "{problem}/{analytic}: {}", result.message);
        assert!(
            result.nlu > 0 && result.njev > 0,
            "banded stiff path must execute"
        );
        let dense = result
            .sol
            .as_ref()
            .unwrap()
            .evaluate_many(&samples)
            .unwrap();
        let mut max_scaled_error = 0.0_f64;
        for (actual, key) in [(&result.y, "y"), (&dense, "dense")] {
            let expected = row[key].as_array().unwrap();
            assert_eq!(actual.len(), expected.len());
            for (actual_row, expected_row) in actual.iter().zip(expected) {
                let expected_row = expected_row.as_array().unwrap();
                assert_eq!(actual_row.len(), expected_row.len());
                for (&actual, expected) in actual_row.iter().zip(expected_row) {
                    let expected = expected.as_f64().unwrap();
                    let scaled = (actual - expected).abs() / (1e-10 + 1e-8 * expected.abs());
                    // Global error accumulates over many locally controlled
                    // steps; both solutions must stay within 50 local scales.
                    assert!(
                        actual.is_finite() && scaled <= 50.0,
                        "{problem}/{analytic}/{key}: {actual} vs {expected}, scaled={scaled}"
                    );
                    max_scaled_error = max_scaled_error.max(scaled);
                    compared_values += 1;
                }
            }
        }
        if problem == "heat" {
            let stopped = solve_ivp(
                &mut |_, y| banded_rhs(problem, y),
                &SolveIvpOptions {
                    t_eval: None,
                    events: Some(vec![
                        fsci_integrate::EventSpec::terminal(|_, y| y[y.len() / 2] - 0.1)
                            .with_direction(-1.0),
                    ]),
                    ..options
                },
            )
            .unwrap();
            assert_eq!(stopped.status, 1);
            let event_time = stopped.t_events.as_ref().unwrap()[0][0];
            let expected = row["event_time"].as_f64().unwrap();
            // At this root |dy/dt| exceeds 0.2; the state-error contract
            // gives a time error below 1e-6.
            assert!((event_time - expected).abs() < 1e-6);
        }
        eprintln!(
            "LSODA {problem}/analytic={analytic}: max_scaled={max_scaled_error:e}, nfev/njev/nlu={}/{}/{}; SciPy {}/{}/{}",
            result.nfev, result.njev, result.nlu, row["nfev"], row["njev"], row["nlu"]
        );
        compared += 1;
    }
    assert_eq!(compared, 10);
    assert_eq!(compared_values, 3840);
    eprintln!(
        "Banded LSODA compared cases={compared}, values={compared_values}; SciPy {}, NumPy {}",
        oracle["scipy"], oracle["numpy"]
    );
}

/// The lower coupling makes the Newton matrix pivot once c exceeds about 1e-6.
/// Diagonally dominant diffusion fixtures cannot exercise that pivot path. Reversing
/// both the RHS sign and the integration direction preserves the stable trajectory,
/// so every dense sample and terminal root has an independent closed-form answer.
#[test]
fn lsoda_banded_pivots_preserve_forward_backward_dense_output_and_events() {
    fn forward_jacobian(_t: f64, _y: &[f64]) -> Vec<Vec<f64>> {
        vec![vec![-1.0, -1000.0], vec![1e6, 0.0]]
    }
    fn backward_jacobian(_t: f64, _y: &[f64]) -> Vec<Vec<f64>> {
        vec![vec![1.0, 1000.0], vec![-1e6, 0.0]]
    }
    fn exact(elapsed: f64) -> [f64; 2] {
        let slow = (-elapsed).exp();
        [slow, (1e6 / 999.0) * (slow - (-1000.0 * elapsed).exp())]
    }
    let elapsed = [0.0, 1e-6, 1e-4, 0.003, 0.03, 0.2, 0.8, 1.0];
    let sample_elapsed = [2e-6, 0.0003, 0.012, 0.17, 0.61, 0.97];
    for direction in [1.0, -1.0] {
        let start = if direction > 0.0 { 0.0 } else { 1.0 };
        let times: Vec<_> = elapsed.iter().map(|&s| start + direction * s).collect();
        let samples: Vec<_> = sample_elapsed
            .iter()
            .map(|&s| start + direction * s)
            .collect();
        let jacobian: fsci_integrate::JacFn = if direction > 0.0 {
            forward_jacobian
        } else {
            backward_jacobian
        };
        for analytic in [false, true] {
            let options = SolveIvpOptions {
                t_span: (start, start + direction),
                y0: &[1.0, 0.0],
                method: SolverKind::Lsoda,
                t_eval: Some(&times),
                dense_output: true,
                rtol: 1e-9,
                atol: ToleranceValue::Vector(vec![1e-12, 1e-9]),
                lband: Some(1),
                uband: Some(0),
                jac: analytic.then_some(jacobian),
                ..SolveIvpOptions::default()
            };
            let mut rhs = |_t: f64, y: &[f64]| {
                vec![-direction * y[0], direction * (1e6 * y[0] - 1000.0 * y[1])]
            };
            let result = solve_ivp(&mut rhs, &options).unwrap();
            assert!(result.success, "direction={direction}, analytic={analytic}");
            assert!(result.njev > 0 && result.nlu > 0, "must use banded BDF");
            assert_eq!(result.t, times);
            let dense = result
                .sol
                .as_ref()
                .unwrap()
                .evaluate_many(&samples)
                .unwrap();
            for (values, query) in [
                (&result.y, elapsed.as_slice()),
                (&dense, sample_elapsed.as_slice()),
            ] {
                for (actual, &s) in values.iter().zip(query) {
                    for (j, expected) in exact(s).into_iter().enumerate() {
                        let atol = if j == 0 { 1e-12 } else { 1e-9 };
                        let error = (actual[j] - expected).abs();
                        assert!(
                            error <= 50.0 * (atol + 1e-9 * expected.abs()),
                            "direction={direction}, analytic={analytic}, s={s}, component={j}: {} vs {expected}",
                            actual[j]
                        );
                    }
                }
            }
            let stopped = solve_ivp(
                &mut rhs,
                &SolveIvpOptions {
                    t_eval: None,
                    events: Some(vec![
                        fsci_integrate::EventSpec::terminal(|_, y| y[0] - 0.5).with_direction(-1.0),
                    ]),
                    ..options
                },
            )
            .unwrap();
            assert_eq!(stopped.status, 1);
            let event_time = stopped.t_events.as_ref().unwrap()[0][0];
            let expected_time = start + direction * std::f64::consts::LN_2;
            assert!(
                (event_time - expected_time).abs() < 1e-7,
                "direction={direction}, analytic={analytic}: event {event_time} vs {expected_time}"
            );
            assert!((stopped.y.last().unwrap()[0] - 0.5).abs() < 1e-10);
        }
    }
}

#[test]
fn all_solvers_refine_backward_events_away_from_step_midpoints() {
    for method in [
        SolverKind::Rk23,
        SolverKind::Rk45,
        SolverKind::Dop853,
        SolverKind::Bdf,
        SolverKind::Radau,
        SolverKind::Lsoda,
    ] {
        let result = solve_ivp(
            &mut |_t, _y| vec![1.0],
            &SolveIvpOptions {
                t_span: (1.0, 0.0),
                y0: &[1.0],
                method,
                first_step: Some(1.0),
                max_step: 1.0,
                rtol: 1e-9,
                atol: ToleranceValue::Scalar(1e-12),
                events: Some(vec![
                    fsci_integrate::EventSpec::terminal(|_, y| y[0] - 0.3).with_direction(-1.0),
                ]),
                ..SolveIvpOptions::default()
            },
        )
        .unwrap();
        assert_eq!(result.status, 1, "{method:?}");
        let root = result.t_events.as_ref().unwrap()[0][0];
        assert!((root - 0.3).abs() < 5e-12, "{method:?}: root={root}");
        assert!((result.y.last().unwrap()[0] - 0.3).abs() < 5e-12, "{method:?}");
    }
}
