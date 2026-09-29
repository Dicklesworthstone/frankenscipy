#![forbid(unsafe_code)]
//! Live SciPy diff for the implicit `solve_ivp` methods, `BDF` and `Radau`.
//!
//! Until this file nothing compared either method with the incumbent (the one other file naming
//! them compares fsci with fsci). Problems: Robertson's kinetics on [0, 40], Van der Pol with
//! mu = 100 on [0, 200], the linear stiff system y1' = -1000 y1 + y2, y2' = -y2 on [0, 10]
//! (exact solution known), and y' = -y·cos t on [0, 10] (exact exp(-sin t)); rtol 1e-4 and 1e-8
//! with atol = 1e-3·rtol, no Jacobian (both sides use finite differences). The right-hand sides
//! are written term by term identically on both sides.
//!
//! As in diff_integrate_dop853.rs, the oracle measures each row's conditioning: SciPy's own
//! y(t_end) is rerun with every component of y0 moved by one ulp, and the largest move in units
//! of rtol·|y| + atol is the row's envelope. Every row here measured below 3e-3, so each is held
//! to SciPy's answer within Y_SCALE_FACTOR_TOL of those units and to SciPy's nfev, njev, nlu and
//! accepted-step count within COUNT_REL_TOL; a row whose envelope exceeds 1 fails outright
//! rather than being skipped. Exact references: the linear system and y' = -y·cos t within
//! EXACT_SCALE_FACTOR_TOL units of rtol·|exact| + atol per component (a relative test would
//! judge the 4.5e-8 component of the linear system, where atol dominates, by the wrong scale).
//!
//! Both solvers are SciPy's algorithms step for step (frankenscipy-szq1n / the stiff parity
//! port): 15 of the 16 rows reproduce SciPy's nfev, njev, nlu and step count exactly on the
//! pinned incumbent, and the 16th (Radau, y' = -y·cos t, rtol 1e-8) spends one more Newton
//! iteration (nfev 2731 vs 2728, everything else equal). That is a rounding-level difference:
//! SciPy's LAPACK/OpenBLAS complex factor-and-divide against nalgebra's flips one borderline
//! convergence test. COUNT_REL_TOL is sized to that and to BLAS-kernel variation on other
//! hosts, far below the 30-110% count gaps of the solvers this port replaced. The compared
//! count is asserted.

use std::io::Write;
use std::process::Stdio;

use fsci_conformance::CompareLedger;
use fsci_integrate::{SolveIvpOptions, SolverKind, ToleranceValue, solve_ivp};
use serde::{Deserialize, Serialize};

const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
/// One ledger arm per `solve_ivp` method; a row's verdict is every check on it passing.
const METHODS: [&str; 2] = ["BDF", "Radau"];
/// |fsci - SciPy| <= Y_SCALE_FACTOR_TOL * (rtol * |y_scipy| + atol) per component of y(t_end).
const Y_SCALE_FACTOR_TOL: f64 = 1.0;
/// Exact solutions: |y - exact| <= EXACT_SCALE_FACTOR_TOL * (rtol * |exact| + atol) per component.
const EXACT_SCALE_FACTOR_TOL: f64 = 50.0;
/// nfev, njev, nlu and accepted steps: |fsci - SciPy| <= COUNT_REL_TOL * SciPy.
const COUNT_REL_TOL: f64 = 0.01;
/// A row is comparable when SciPy's 1-ulp envelope is at most this many units.
const WELL_CONDITIONED_ENVELOPE: f64 = 1.0;

#[derive(Debug, Clone, Serialize)]
struct Case {
    name: String,
    problem: String,
    method: String,
    t_end: f64,
    y0: Vec<f64>,
    rtol: f64,
    atol: f64,
}

#[derive(Debug, Deserialize)]
struct Answer {
    status: i32,
    nfev: usize,
    njev: usize,
    nlu: usize,
    n_t: usize,
    y_end: Vec<f64>,
    envelope: f64,
}

fn rhs(problem: &str, t: f64, y: &[f64]) -> Vec<f64> {
    match problem {
        "robertson" => vec![
            -0.04 * y[0] + 1.0e4 * y[1] * y[2],
            0.04 * y[0] - 1.0e4 * y[1] * y[2] - 3.0e7 * y[1] * y[1],
            3.0e7 * y[1] * y[1],
        ],
        "vdp100" => vec![y[1], 100.0 * (1.0 - y[0] * y[0]) * y[1] - y[0]],
        "linear" => vec![-1000.0 * y[0] + y[1], -y[1]],
        _ => vec![-y[0] * t.cos()],
    }
}

/// Exact y(t_end) where one is known.
fn exact(problem: &str, t: f64) -> Option<Vec<f64>> {
    match problem {
        // y2 = e^-t; y1 = e^-t / 999 + (998 / 999) e^-1000t from y(0) = (1, 1).
        "linear" => Some(vec![
            (-t).exp() / 999.0 + (998.0 / 999.0) * (-1000.0 * t).exp(),
            (-t).exp(),
        ]),
        "cos_decay" => Some(vec![(-t.sin()).exp()]),
        _ => None,
    }
}

fn cases() -> Vec<Case> {
    let problems: [(&str, f64, Vec<f64>); 4] = [
        ("robertson", 40.0, vec![1.0, 0.0, 0.0]),
        ("vdp100", 200.0, vec![2.0, 0.0]),
        ("linear", 10.0, vec![1.0, 1.0]),
        ("cos_decay", 10.0, vec![1.0]),
    ];
    let mut out = Vec::new();
    for method in ["BDF", "Radau"] {
        for (problem, t_end, y0) in &problems {
            for rtol in [1e-4, 1e-8] {
                out.push(Case {
                    name: format!("{method}_{problem}_rtol{rtol:e}"),
                    problem: (*problem).to_string(),
                    method: method.to_string(),
                    t_end: *t_end,
                    y0: y0.clone(),
                    rtol,
                    atol: 1e-3 * rtol,
                });
            }
        }
    }
    out
}

fn scipy_answers(cases: &[Case]) -> Option<Vec<Answer>> {
    let script = r#"
import json, sys, math
import numpy as np
from scipy.integrate import solve_ivp

def rhs(problem):
    if problem == "robertson":
        return lambda t, y: [-0.04 * y[0] + 1.0e4 * y[1] * y[2],
                             0.04 * y[0] - 1.0e4 * y[1] * y[2] - 3.0e7 * y[1] * y[1],
                             3.0e7 * y[1] * y[1]]
    if problem == "vdp100":
        return lambda t, y: [y[1], 100.0 * (1.0 - y[0] * y[0]) * y[1] - y[0]]
    if problem == "linear":
        return lambda t, y: [-1000.0 * y[0] + y[1], -y[1]]
    return lambda t, y: [-y[0] * math.cos(t)]

def run(c, y0):
    return solve_ivp(rhs(c["problem"]), (0.0, c["t_end"]), y0, method=c["method"],
                     rtol=c["rtol"], atol=c["atol"])

out = []
for c in json.load(sys.stdin):
    r = run(c, c["y0"])
    y_end = r.y[:, -1]
    scale = c["rtol"] * np.abs(y_end) + c["atol"]
    envelope = 0.0
    for k in range(len(c["y0"])):
        for direction in (math.inf, -math.inf):
            y0 = list(c["y0"])
            y0[k] = float(np.nextafter(y0[k], direction))
            p = run(c, y0)
            envelope = max(envelope, float(np.max(np.abs(p.y[:, -1] - y_end) / scale)))
    out.append({"status": int(r.status), "nfev": int(r.nfev), "njev": int(r.njev),
                "nlu": int(r.nlu), "n_t": int(len(r.t)),
                "y_end": [float(v) for v in y_end], "envelope": envelope})
print(json.dumps(out))
"#;
    let query = serde_json::to_string(cases).expect("serialize cases");
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn the stiff solve_ivp oracle: {e}"
            );
            eprintln!("skipping stiff solve_ivp oracle: python not available ({e})");
            return None;
        }
    };
    child
        .stdin
        .as_mut()
        .expect("oracle stdin")
        .write_all(query.as_bytes())
        .expect("write oracle query");
    let output = child
        .wait_with_output()
        .expect("wait for the stiff solve_ivp oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "stiff solve_ivp oracle failed: {stderr}"
        );
        eprintln!("skipping stiff solve_ivp oracle: scipy not available\n{stderr}");
        return None;
    }
    Some(serde_json::from_slice(&output.stdout).expect("parse stiff solve_ivp oracle JSON"))
}

fn count_gap(fsci: usize, scipy: usize) -> f64 {
    if fsci == scipy {
        0.0
    } else {
        (fsci as f64 - scipy as f64).abs() / scipy.max(1) as f64
    }
}

#[test]
fn diff_integrate_solve_ivp_stiff() {
    let cases = cases();
    let Some(answers) = scipy_answers(&cases) else {
        return;
    };
    assert_eq!(
        answers.len(),
        cases.len(),
        "the oracle must answer every case"
    );
    let mut compared = 0;
    let mut worst_scaled = 0.0_f64;
    let mut failures = Vec::new();
    let mut ledger = CompareLedger::new("diff_integrate_solve_ivp_stiff", &METHODS);
    for (case, answer) in cases.iter().zip(&answers) {
        let problem = case.problem.clone();
        let mut fun = move |t: f64, y: &[f64]| rhs(&problem, t, y);
        let options = SolveIvpOptions {
            t_span: (0.0, case.t_end),
            y0: &case.y0,
            method: if case.method == "BDF" {
                SolverKind::Bdf
            } else {
                SolverKind::Radau
            },
            rtol: case.rtol,
            atol: ToleranceValue::Scalar(case.atol),
            ..SolveIvpOptions::default()
        };
        compared += 1;
        let arm = case.method.as_str();
        let failures_before = failures.len();
        let result = solve_ivp(&mut fun, &options);
        if let Err(e) = &result {
            failures.push(format!("{}: fsci Err({e:?})", case.name));
        }
        let Some((answer, result)) = ledger.both(arm, &case.name, Some(answer), result.ok()) else {
            continue;
        };
        let fsci_y_end = result.y.last().cloned().unwrap_or_default();
        if result.status != 0 || answer.status != 0 {
            failures.push(format!(
                "{}: status fsci {} SciPy {}, y_end {fsci_y_end:?}",
                case.name, result.status, answer.status
            ));
        }
        // Length, and a NaN in any component, which the max folds below would swallow.
        let Some((_, y_end)) = ledger.slices(
            arm,
            &case.name,
            Some(answer.y_end.as_slice()),
            Some(fsci_y_end.as_slice()),
        ) else {
            failures.push(format!(
                "{}: y_end {fsci_y_end:?} against SciPy {:?}",
                case.name, answer.y_end
            ));
            continue;
        };
        let scaled = y_end
            .iter()
            .zip(&answer.y_end)
            .map(|(f, s)| (f - s).abs() / (case.rtol * s.abs() + case.atol))
            .fold(0.0_f64, f64::max);
        worst_scaled = worst_scaled.max(scaled);
        let n_t = result.t.len();
        println!(
            "{}: max |dy|/(rtol|y|+atol) {scaled:.3e} (SciPy 1-ulp envelope {:.3e}) | nfev {} / {} | njev {} / {} | nlu {} / {} | n_t {n_t} / {} (fsci / SciPy) | y_end fsci {y_end:?} SciPy {:?}",
            case.name,
            answer.envelope,
            result.nfev,
            answer.nfev,
            result.njev,
            answer.njev,
            result.nlu,
            answer.nlu,
            answer.n_t,
            answer.y_end
        );
        let counts = [
            ("nfev", result.nfev, answer.nfev),
            ("njev", result.njev, answer.njev),
            ("nlu", result.nlu, answer.nlu),
            ("n_t", n_t, answer.n_t),
        ];
        if answer.envelope > WELL_CONDITIONED_ENVELOPE {
            failures.push(format!(
                "{}: SciPy's own 1-ulp envelope {:e} is too wide to compare against",
                case.name, answer.envelope
            ));
        }
        if scaled.is_nan() || scaled > Y_SCALE_FACTOR_TOL {
            failures.push(format!("{}: scaled y gap {scaled:e}", case.name));
        }
        for (what, fsci, scipy) in counts {
            if count_gap(fsci, scipy) > COUNT_REL_TOL {
                failures.push(format!("{}: {what} fsci {fsci} SciPy {scipy}", case.name));
            }
        }
        if let Some(exact) = exact(&case.problem, case.t_end) {
            let scaled_exact = y_end
                .iter()
                .zip(&exact)
                .map(|(y, e)| (y - e).abs() / (case.rtol * e.abs() + case.atol))
                .fold(0.0_f64, f64::max);
            println!(
                "{}: error vs exact {scaled_exact:.3e} x (rtol|exact| + atol)",
                case.name
            );
            if scaled_exact.is_nan() || scaled_exact > EXACT_SCALE_FACTOR_TOL {
                failures.push(format!(
                    "{}: {scaled_exact:e} error-scale units from the exact solution",
                    case.name
                ));
            }
        }
        ledger.compared(arm, &case.name, failures.len() == failures_before);
    }
    println!("{compared} integrations compared; worst scaled y gap {worst_scaled:.3e}");
    assert_eq!(compared, cases.len());
    assert!(
        failures.is_empty(),
        "stiff solve_ivp disagrees: {failures:#?}"
    );
    let min_per_arm = METHODS
        .iter()
        .map(|m| cases.iter().filter(|c| c.method == *m).count())
        .min()
        .expect("METHODS is non-empty");
    ledger.finish(min_per_arm);
}
