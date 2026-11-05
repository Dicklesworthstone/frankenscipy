#![forbid(unsafe_code)]
//! Retained continuous solutions and terminal-event sampling against live SciPy.
//!
//! Every method runs forward and backward. Harmonic-oscillator queries deliberately differ
//! from `t_eval`, arrive out of order, and repeat; retaining output samples or interpolating
//! linearly between accepted steps cannot meet the declared error scale. Constant-derivative
//! cases put three event roots in one accepted step, ordered late/terminal/early in the
//! callback list. Only roots encountered before termination may be reported, and a terminal
//! root must not be appended to a user-specified output grid. A second grid includes times
//! one ULP on either side of termination and samples very near the initial time.
//!
//! SciPy's contracts: `scipy.integrate.solve_ivp` and `scipy.integrate.OdeSolution` in the
//! version reported by the shared oracle resolver. Dense arrays are transposed from SciPy's
//! component-major layout into this crate's one-state-per-time layout before comparison.

use std::io::Write;
use std::process::Stdio;

use fsci_conformance::CompareLedger;
use fsci_integrate::{EventSpec, SolveIvpOptions, SolverKind, ToleranceValue, solve_ivp};
use serde::{Deserialize, Serialize};

const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
const METHODS: [&str; 6] = ["RK23", "RK45", "DOP853", "Radau", "BDF", "LSODA"];
const RTOL: f64 = 1.0e-9;
const ATOL: f64 = 1.0e-11;
// These are solution-error envelopes, not per-step local-error promises. Compare the two
// integrations within five requested error units, and each with the known oscillator
// solution within fifty units of its fixed unit state norm to cover accumulated error.
// Scaling that global check by one component would spuriously demand atol-only accuracy
// at a zero crossing, where even the pinned SciPy solution has accumulated phase error.
const SCIPY_SCALE_FACTOR: f64 = 5.0;
const EXACT_SCALE_FACTOR: f64 = 50.0;
const EVENT_ABS_TOL: f64 = 1.0e-12;

#[derive(Debug, Clone, Serialize)]
struct Case {
    name: String,
    method: String,
    terminal: bool,
    backward: bool,
    t_span: [f64; 2],
    y0: Vec<f64>,
    t_eval: Vec<f64>,
    queries: Vec<f64>,
    first_step: Option<f64>,
    max_step: f64,
}

#[derive(Debug, Deserialize)]
struct Answer {
    name: String,
    status: i32,
    t: Vec<f64>,
    y: Vec<Vec<f64>>,
    dense: Vec<Vec<f64>>,
    t_events: Option<Vec<Vec<f64>>>,
    y_events: Option<Vec<Vec<Vec<f64>>>>,
}

#[derive(Debug, Deserialize)]
struct OracleReport {
    available: bool,
    error: Option<String>,
    #[serde(default)]
    answers: Vec<Answer>,
}

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for method in METHODS {
        for backward in [false, true] {
            let direction = if backward { "backward" } else { "forward" };
            let t0: f64 = if backward { 3.0 } else { 0.0 };
            cases.push(Case {
                name: format!("{method}_harmonic_{direction}"),
                method: method.to_string(),
                terminal: false,
                backward,
                t_span: [t0, 3.0 - t0],
                y0: vec![t0.cos(), -t0.sin()],
                t_eval: if backward {
                    vec![2.7, 1.5, 0.3]
                } else {
                    vec![0.3, 1.5, 2.7]
                },
                queries: vec![2.67, 0.13, 1.31, 1.31, 0.0, 3.0, 2.27],
                first_step: None,
                max_step: 0.25,
            });
            for adjacent in [false, true] {
                let t_eval = match (backward, adjacent) {
                    (false, false) => vec![0.0, 0.25, 0.75, 1.0],
                    (true, false) => vec![1.0, 0.75, 0.25, 0.0],
                    (false, true) => vec![
                        1.0e-15,
                        1.0e-14,
                        0.5_f64.next_down(),
                        0.5,
                        0.5_f64.next_up(),
                        1.0,
                    ],
                    (true, true) => vec![
                        1.0 - 1.0e-15,
                        1.0 - 1.0e-14,
                        0.5_f64.next_up(),
                        0.5,
                        0.5_f64.next_down(),
                        0.0,
                    ],
                };
                let grid = if adjacent { "adjacent" } else { "sparse" };
                cases.push(Case {
                    name: format!("{method}_terminal_{direction}_{grid}"),
                    method: method.to_string(),
                    terminal: true,
                    backward,
                    t_span: if backward { [1.0, 0.0] } else { [0.0, 1.0] },
                    y0: vec![if backward { 1.0 } else { 0.0 }],
                    t_eval,
                    // Include points beyond both ends of the retained domain: SciPy uses
                    // the first/last local polynomial for extrapolation. This constant RHS
                    // keeps that behavior independently checkable against y(t) = t.
                    queries: if backward {
                        vec![0.625, 0.95, 0.5, 0.625, 0.375, 1.125, 1.0]
                    } else {
                        vec![0.375, 0.05, 0.5, 0.375, -0.125, 0.625, 0.0]
                    },
                    first_step: Some(1.0),
                    max_step: 1.0,
                });
            }
        }
    }
    cases
}

fn scipy_answers(cases: &[Case]) -> Option<Vec<Answer>> {
    let script = r#"
import json, sys
try:
    import numpy as np
    from scipy.integrate import solve_ivp
except ImportError as exc:
    print(json.dumps({"available": False, "error": str(exc)}))
    sys.exit(0)

def low(t, y):
    return t - 0.2

def terminal(t, y):
    return t - 0.5

def high(t, y):
    return t - 0.8

answers = []
for c in json.load(sys.stdin):
    events = None
    if c["terminal"]:
        for event in (low, terminal, high):
            event.direction = -1.0 if c["backward"] else 1.0
        terminal.terminal = True
        events = [low, terminal, high] if c["backward"] else [high, terminal, low]
        fun = lambda t, y: [1.0]
    else:
        fun = lambda t, y: [y[1], -y[0]]
    r = solve_ivp(fun, c["t_span"], c["y0"], method=c["method"],
                  t_eval=c["t_eval"], dense_output=True, events=events,
                  rtol=1.0e-9, atol=1.0e-11,
                  first_step=c["first_step"], max_step=c["max_step"])
    if not r.success or r.sol is None:
        raise RuntimeError(c["name"] + ": " + r.message)
    answers.append({
        "name": c["name"], "status": int(r.status),
        "t": r.t.tolist(), "y": r.y.T.tolist(),
        "dense": r.sol(np.array(c["queries"])).T.tolist(),
        "t_events": None if r.t_events is None else [t.tolist() for t in r.t_events],
        "y_events": None if r.y_events is None else [y.tolist() for y in r.y_events],
    })
print(json.dumps({"available": True, "error": None, "answers": answers}))
"#;
    let query = serde_json::to_vec(cases).expect("serialize dense/event cases");
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "cannot spawn the dense/event oracle: {error}"
            );
            eprintln!("skipping dense/event oracle: interpreter unavailable ({error})");
            return None;
        }
    };
    child
        .stdin
        .take()
        .expect("oracle stdin")
        .write_all(&query)
        .expect("write dense/event oracle query");
    let output = child
        .wait_with_output()
        .expect("wait for dense/event oracle");
    assert!(
        output.status.success(),
        "dense/event oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: OracleReport =
        serde_json::from_slice(&output.stdout).expect("parse dense/event oracle");
    if !report.available {
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "SciPy is unavailable: {:?}",
            report.error
        );
        eprintln!("skipping dense/event oracle: {:?}", report.error);
        return None;
    }
    Some(report.answers)
}

fn method_of(name: &str) -> SolverKind {
    match name {
        "RK23" => SolverKind::Rk23,
        "RK45" => SolverKind::Rk45,
        "DOP853" => SolverKind::Dop853,
        "Radau" => SolverKind::Radau,
        "BDF" => SolverKind::Bdf,
        "LSODA" => SolverKind::Lsoda,
        _ => panic!("unknown solver {name}"),
    }
}

fn events(backward: bool) -> Vec<EventSpec> {
    let low = EventSpec::new(|t, _| t - 0.2);
    let terminal = EventSpec::terminal(|t, _| t - 0.5);
    let high = EventSpec::new(|t, _| t - 0.8);
    let ordered = if backward {
        [low, terminal, high]
    } else {
        [high, terminal, low]
    };
    ordered
        .into_iter()
        .map(|event| event.with_direction(if backward { -1.0 } else { 1.0 }))
        .collect()
}

fn compare_states(
    case: &Case,
    label: &str,
    times: &[f64],
    actual: &[Vec<f64>],
    reference: &[Vec<f64>],
    failures: &mut Vec<String>,
) {
    if actual.len() != times.len() || reference.len() != times.len() {
        failures.push(format!(
            "{label} shape: {} states / {} reference states / {} times",
            actual.len(),
            reference.len(),
            times.len()
        ));
        return;
    }
    for ((&time, actual), reference) in times.iter().zip(actual).zip(reference) {
        let exact = if case.terminal {
            vec![time]
        } else {
            vec![time.cos(), -time.sin()]
        };
        if actual.len() != exact.len() || reference.len() != exact.len() {
            failures.push(format!("{label} at {time}: wrong state dimension"));
            continue;
        }
        for (component, ((&actual, &reference), &exact)) in
            actual.iter().zip(reference).zip(&exact).enumerate()
        {
            let (oracle_tol, exact_tol) = if case.terminal {
                (EVENT_ABS_TOL, EVENT_ABS_TOL)
            } else {
                (
                    SCIPY_SCALE_FACTOR * (ATOL + RTOL * reference.abs()),
                    EXACT_SCALE_FACTOR * (ATOL + RTOL),
                )
            };
            if !actual.is_finite()
                || !reference.is_finite()
                || (actual - reference).abs() > oracle_tol
                || (actual - exact).abs() > exact_tol
                || (reference - exact).abs() > exact_tol
            {
                failures.push(format!(
                    "{label}[{component}] at {time}: fsci={actual:.16e}, \
                     scipy={reference:.16e}, exact={exact:.16e}"
                ));
            }
        }
    }
}

#[test]
fn diff_integrate_solve_ivp_dense_events() {
    let cases = cases();
    let Some(answers) = scipy_answers(&cases) else {
        return;
    };
    assert_eq!(cases.len(), 36);
    assert_eq!(
        answers.len(),
        cases.len(),
        "every case needs an oracle answer"
    );
    let mut ledger = CompareLedger::new("diff_integrate_solve_ivp_dense_events", &METHODS);
    let mut failures = Vec::new();
    for (case, answer) in cases.iter().zip(&answers) {
        assert_eq!(answer.name, case.name, "oracle response order");
        let mut fun = |_t: f64, y: &[f64]| {
            if case.terminal {
                vec![1.0]
            } else {
                vec![y[1], -y[0]]
            }
        };
        let options = SolveIvpOptions {
            t_span: (case.t_span[0], case.t_span[1]),
            y0: &case.y0,
            method: method_of(&case.method),
            t_eval: Some(&case.t_eval),
            dense_output: true,
            events: case.terminal.then(|| events(case.backward)),
            rtol: RTOL,
            atol: ToleranceValue::Scalar(ATOL),
            first_step: case.first_step,
            max_step: case.max_step,
            ..SolveIvpOptions::default()
        };
        let result = match solve_ivp(&mut fun, &options) {
            Ok(result) => result,
            Err(error) => {
                ledger.rust_failed(&case.method, &case.name, &error.to_string());
                failures.push(format!("{}: {error}", case.name));
                continue;
            }
        };
        let mut problems = Vec::new();
        if !result.success || result.status != answer.status {
            problems.push(format!("status {}/{}", result.status, answer.status));
        }
        if result.t != answer.t {
            problems.push(format!("t_eval: {:?} != {:?}", result.t, answer.t));
        }
        compare_states(
            case,
            "t_eval",
            &answer.t,
            &result.y,
            &answer.y,
            &mut problems,
        );

        match result.sol.as_ref() {
            None => problems.push("dense_output=True returned no solution".to_string()),
            Some(solution) => {
                match solution.evaluate_many(&case.queries) {
                    Err(error) => problems.push(format!("dense batch: {error}")),
                    Ok(dense) => {
                        compare_states(
                            case,
                            "dense",
                            &case.queries,
                            &dense,
                            &answer.dense,
                            &mut problems,
                        );
                        for (&time, batch) in case.queries.iter().zip(&dense) {
                            match solution.evaluate(time) {
                                Ok(scalar) if &scalar == batch => {}
                                other => problems.push(format!(
                                    "scalar/batch disagree at {time}: {other:?} vs {batch:?}"
                                )),
                            }
                        }
                    }
                }
                if !matches!(solution.evaluate_many(&[]), Ok(empty) if empty.is_empty()) {
                    problems.push("empty query batch did not return an empty result".to_string());
                }
                let terminal_time = if case.terminal { 0.5 } else { case.t_span[1] };
                if solution.knots.first().copied() != Some(case.t_span[0])
                    || solution
                        .knots
                        .last()
                        .is_none_or(|end| (end - terminal_time).abs() > EVENT_ABS_TOL)
                {
                    problems.push(format!("dense knot endpoints: {:?}", solution.knots));
                }
            }
        }

        match (
            &result.t_events,
            &answer.t_events,
            &result.y_events,
            &answer.y_events,
        ) {
            (None, None, None, None) if !case.terminal => {}
            (Some(times), Some(reference), Some(states), Some(reference_states))
                if times.len() == 3
                    && reference.len() == 3
                    && states.len() == 3
                    && reference_states.len() == 3 =>
            {
                for i in 0..3 {
                    if times[i].len() != reference[i].len()
                        || times[i]
                            .iter()
                            .zip(&reference[i])
                            .any(|(actual, expected)| {
                                !actual.is_finite() || (actual - expected).abs() > EVENT_ABS_TOL
                            })
                    {
                        problems.push(format!("event {i}: {:?} vs {:?}", times[i], reference[i]));
                    }
                    compare_states(
                        case,
                        &format!("event {i}"),
                        &reference[i],
                        &states[i],
                        &reference_states[i],
                        &mut problems,
                    );
                }
            }
            other => problems.push(format!("event output shapes differ: {other:?}")),
        }
        ledger.compared(&case.method, &case.name, problems.is_empty());
        if problems.is_empty() {
            println!(
                "{}: compared samples, retained dense output, and events",
                case.name
            );
        } else {
            failures.push(format!("{}: {}", case.name, problems.join("; ")));
        }
    }
    assert!(
        failures.is_empty(),
        "dense/event conformance failures:\n{}",
        failures.join("\n")
    );
    let counts = ledger.finish(6);
    assert_eq!(
        counts.values().map(|arm| arm.compared_cases).sum::<usize>(),
        36
    );
}
