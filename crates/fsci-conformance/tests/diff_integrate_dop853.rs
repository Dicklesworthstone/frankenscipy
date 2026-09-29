#![forbid(unsafe_code)]
//! Live SciPy diff for `solve_ivp(method="DOP853")` (frankenscipy-szq1n.1), with RK45 and RK23
//! on the same problems as the arms that fix must not have moved.
//!
//! szq1n.1: DOP853 reused its last STAGE derivative k[11] as f(t + h, y_new), which is only valid
//! for FSAL tableaus (RK23, RK45), and estimated its error from E5 alone where SciPy blends E5
//! and E3. The unit tests in `fsci-integrate` pin the tableau flag and the fixed-step order; this
//! file compares whole adaptive integrations with the incumbent: y' = -y·cos t on [0, 10]
//! (exact exp(-sin t)), Lorenz on [0, 1], and one period of the Arenstorf orbit (exact: it returns
//! to y0), at rtol 1e-6 and 1e-10 with atol = 1e-3·rtol (RK23 at 1e-6 only). The right-hand sides
//! are written term by term identically on both sides.
//!
//! How far a row can be compared with SciPy is MEASURED, not assumed: the oracle also reruns each
//! case with every component of y0 moved by one ulp either way and reports how far SciPy's own
//! y(t_end) moves, in units of its error scale `rtol·|y| + atol` (the row's envelope).
//! - Envelope <= 1 (well conditioned): every component of y(t_end) within Y_SCALE_FACTOR_TOL of
//!   those units of SciPy's.
//! - Envelope > 1: SciPy's answer is not reproducible at its own error scale. One Arenstorf period
//!   at rtol 1e-10 moves SciPy's DOP853 y(t_end) by ~1e6 units and its nfev from 3326 to 3350
//!   under a 1-ulp y0 change. Such a row MUST have an exact solution to be held to; the number of
//!   well-conditioned rows is asserted so the set cannot shrink silently.
//!
//! Counts, every row: the nfev accounting must be SciPy's exactly, i.e. fsci's nfev differs from
//! SciPy's by a whole number of step attempts (12 evaluations for DOP853, 6 for RK45, 3 for RK23:
//! nfev = 2 + that × attempts), and nfev and the accepted-step count lie within COUNT_REL_TOL of
//! SciPy's. On the pinned host fsci's counts equal SciPy's on every well-conditioned row, but
//! SciPy's own step decisions move with its BLAS kernel (OPENBLAS_CORETYPE=Prescott: DOP853 at
//! rtol 1e-10 on y' = -y·cos t takes nfev 662 where Haswell takes 674), so equality is not a
//! host-independent contract. The off-by-one nfev this file first exposed (the uncounted
//! initial-step probe) is not a whole number of attempts and fails.
//!
//! Exact references: y' = -y·cos t within EXACT_RTOL_FACTOR_TOL·rtol relative of exp(-sin t_end);
//! the Arenstorf return gap max|y(T) - y0| within RETURN_GAP_VS_SCIPY_TOL times SciPy's (no rtol
//! bound holds there for anyone: SciPy's DOP853 misses by 5e-3 at rtol 1e-6).
//! The compared count is asserted.

use std::io::Write;
use std::process::Stdio;

use fsci_conformance::CompareLedger;
use fsci_integrate::{SolveIvpOptions, SolverKind, ToleranceValue, solve_ivp};
use serde::{Deserialize, Serialize};

const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
/// One ledger arm per `solve_ivp` method; a row's verdict is every check on it passing.
const METHODS: [&str; 3] = ["DOP853", "RK45", "RK23"];
/// Well-conditioned rows: |fsci - SciPy| <= Y_SCALE_FACTOR_TOL * (rtol * |y_scipy| + atol) per
/// component, i.e. inside SciPy's own error-control scale (the acceptance allowed 50).
const Y_SCALE_FACTOR_TOL: f64 = 1.0;
/// y' = -y cos t: relative error against exp(-sin t_end) <= EXACT_RTOL_FACTOR_TOL * rtol.
/// RK23 at rtol 1e-6 sits at 8.9 rtol, in SciPy as in fsci.
const EXACT_RTOL_FACTOR_TOL: f64 = 50.0;
/// Arenstorf: fsci's return gap <= RETURN_GAP_VS_SCIPY_TOL * SciPy's return gap.
const RETURN_GAP_VS_SCIPY_TOL: f64 = 2.0;
/// nfev and accepted steps: |fsci - SciPy| <= COUNT_REL_TOL * SciPy (the acceptance allowed 15%;
/// one flipped DOP853 step decision is 12 of ~660 evaluations, 1.8%).
const COUNT_REL_TOL: f64 = 0.05;
/// A row is well conditioned when SciPy's 1-ulp envelope is at most this many units.
const WELL_CONDITIONED_ENVELOPE: f64 = 1.0;
/// 15 rows; only the two Arenstorf rows at rtol 1e-10 (DOP853, RK45) are ill conditioned.
const MIN_WELL_CONDITIONED_ROWS: usize = 13;

const MU: f64 = 0.012_277_471;
/// The Arenstorf period 17.0652165601579625588917206249 and initial velocity
/// -2.00158510637908252240537862224, as the nearest doubles.
const T_ARENSTORF: f64 = 17.065_216_560_157_964;
const ARENSTORF_Y0: [f64; 4] = [0.994, 0.0, 0.0, -2.001_585_106_379_082_4];

#[derive(Debug, Clone, Serialize)]
struct Case {
    name: String,
    problem: String,
    method: String,
    t_end: f64,
    y0: Vec<f64>,
    rtol: f64,
    atol: f64,
    /// Sample times; `None` for the step-by-step rows.
    t_eval: Option<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct Answer {
    status: i32,
    nfev: usize,
    n_t: usize,
    y_end: Vec<f64>,
    /// Largest move of SciPy's y(t_end), in units of rtol·|y| + atol, over the 1-ulp y0 changes.
    envelope: f64,
    /// SciPy's y at every output time (its `t_eval` samples when given).
    ys: Vec<Vec<f64>>,
}

fn rhs(problem: &str, t: f64, y: &[f64]) -> Vec<f64> {
    match problem {
        "cos_decay" => vec![-y[0] * t.cos()],
        "lorenz" => vec![
            10.0 * (y[1] - y[0]),
            y[0] * (28.0 - y[2]) - y[1],
            y[0] * y[1] - (8.0 / 3.0) * y[2],
        ],
        _ => {
            let mup = 1.0 - MU;
            let (x, yy, vx, vy) = (y[0], y[1], y[2], y[3]);
            let (a, b) = (x + MU, x - mup);
            let r1 = (a * a + yy * yy).sqrt();
            let r2 = (b * b + yy * yy).sqrt();
            let (d1, d2) = (r1 * r1 * r1, r2 * r2 * r2);
            vec![
                vx,
                vy,
                x + 2.0 * vy - mup * a / d1 - MU * b / d2,
                yy - 2.0 * vx - mup * yy / d1 - MU * yy / d2,
            ]
        }
    }
}

fn cases() -> Vec<Case> {
    let problems: [(&str, f64, Vec<f64>); 3] = [
        ("cos_decay", 10.0, vec![1.0]),
        ("lorenz", 1.0, vec![1.0, 1.0, 1.0]),
        ("arenstorf", T_ARENSTORF, ARENSTORF_Y0.to_vec()),
    ];
    let mut out = Vec::new();
    for method in ["DOP853", "RK45", "RK23"] {
        for (problem, t_end, y0) in &problems {
            for rtol in [1e-6, 1e-10] {
                if method == "RK23" && rtol < 1e-8 {
                    continue;
                }
                out.push(Case {
                    name: format!("{method}_{problem}_rtol{rtol:e}"),
                    problem: (*problem).to_string(),
                    method: method.to_string(),
                    t_end: *t_end,
                    y0: y0.clone(),
                    rtol,
                    atol: 1e-3 * rtol,
                    t_eval: None,
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

MU = 0.012277471

def rhs(problem):
    if problem == "cos_decay":
        return lambda t, y: [-y[0] * math.cos(t)]
    if problem == "lorenz":
        return lambda t, y: [10.0 * (y[1] - y[0]), y[0] * (28.0 - y[2]) - y[1],
                             y[0] * y[1] - (8.0 / 3.0) * y[2]]
    def arenstorf(t, y):
        mup = 1.0 - MU
        x, yy, vx, vy = y[0], y[1], y[2], y[3]
        a, b = x + MU, x - mup
        r1 = math.sqrt(a * a + yy * yy)
        r2 = math.sqrt(b * b + yy * yy)
        d1, d2 = r1 * r1 * r1, r2 * r2 * r2
        return [vx, vy, x + 2.0 * vy - mup * a / d1 - MU * b / d2,
                yy - 2.0 * vx - mup * yy / d1 - MU * yy / d2]
    return arenstorf

def run(c, y0):
    return solve_ivp(rhs(c["problem"]), (0.0, c["t_end"]), y0, method=c["method"],
                     rtol=c["rtol"], atol=c["atol"], t_eval=c["t_eval"])

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
    out.append({"status": int(r.status), "nfev": int(r.nfev), "n_t": int(len(r.t)),
                "y_end": [float(v) for v in y_end], "envelope": envelope,
                "ys": [[float(v) for v in col] for col in r.y.T]})
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
                "failed to spawn the solve_ivp oracle: {e}"
            );
            eprintln!("skipping solve_ivp oracle: python not available ({e})");
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
        .expect("wait for the solve_ivp oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "solve_ivp oracle failed: {stderr}"
        );
        eprintln!("skipping solve_ivp oracle: scipy not available\n{stderr}");
        return None;
    }
    Some(serde_json::from_slice(&output.stdout).expect("parse solve_ivp oracle JSON"))
}

fn method_of(name: &str) -> SolverKind {
    match name {
        "DOP853" => SolverKind::Dop853,
        "RK45" => SolverKind::Rk45,
        _ => SolverKind::Rk23,
    }
}

fn count_gap(fsci: usize, scipy: usize) -> f64 {
    (fsci as f64 - scipy as f64).abs() / scipy as f64
}

/// Right-hand-side evaluations per attempted step (SciPy's `n_stages`, plus the separate
/// `f(t + h, y_new)` that DOP853 is not FSAL enough to reuse).
fn evals_per_attempt(method: &str) -> usize {
    match method {
        "DOP853" => 12,
        "RK45" => 6,
        _ => 3,
    }
}

fn return_gap(y_end: &[f64]) -> f64 {
    y_end
        .iter()
        .zip(ARENSTORF_Y0)
        .map(|(y, y0)| (y - y0).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn diff_integrate_dop853() {
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
    let mut dop853_compared = 0;
    let mut well_conditioned = 0;
    let mut worst_scaled = 0.0_f64;
    let mut failures = Vec::new();
    let mut ledger = CompareLedger::new("diff_integrate_dop853", &METHODS);
    for (case, answer) in cases.iter().zip(&answers) {
        let problem = case.problem.clone();
        let mut fun = move |t: f64, y: &[f64]| rhs(&problem, t, y);
        let options = SolveIvpOptions {
            t_span: (0.0, case.t_end),
            y0: &case.y0,
            method: method_of(&case.method),
            rtol: case.rtol,
            atol: ToleranceValue::Scalar(case.atol),
            ..SolveIvpOptions::default()
        };
        compared += 1;
        if case.method == "DOP853" {
            dop853_compared += 1;
        }
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
        // Largest |fsci - SciPy| in units of SciPy's own error scale rtol*|y| + atol.
        let scaled = y_end
            .iter()
            .zip(&answer.y_end)
            .map(|(f, s)| (f - s).abs() / (case.rtol * s.abs() + case.atol))
            .fold(0.0_f64, f64::max);
        let n_t = result.t.len();
        println!(
            "{}: max |dy|/(rtol|y|+atol) {scaled:.3e} (SciPy 1-ulp envelope {:.3e}) | nfev fsci {} SciPy {} | n_t fsci {n_t} SciPy {} | y_end fsci {y_end:?} SciPy {:?}",
            case.name, answer.envelope, result.nfev, answer.nfev, answer.n_t, answer.y_end
        );
        let has_exact = case.problem == "cos_decay" || case.problem == "arenstorf";
        if answer.envelope <= WELL_CONDITIONED_ENVELOPE {
            well_conditioned += 1;
            worst_scaled = worst_scaled.max(scaled);
            if scaled.is_nan() || scaled > Y_SCALE_FACTOR_TOL {
                failures.push(format!("{}: scaled y gap {scaled:e}", case.name));
            }
        } else if !has_exact {
            failures.push(format!(
                "{}: SciPy's envelope {:e} rules out comparing y, and there is no exact solution",
                case.name, answer.envelope
            ));
        }
        if result.nfev.abs_diff(answer.nfev) % evals_per_attempt(&case.method) != 0
            || count_gap(result.nfev, answer.nfev) > COUNT_REL_TOL
            || count_gap(n_t, answer.n_t) > COUNT_REL_TOL
        {
            failures.push(format!(
                "{}: nfev {} vs {} (a whole number of {}-evaluation attempts apart?), n_t {n_t} vs {}",
                case.name,
                result.nfev,
                answer.nfev,
                evals_per_attempt(&case.method),
                answer.n_t
            ));
        }
        if case.problem == "cos_decay" {
            let exact = (-case.t_end.sin()).exp();
            let rel = (y_end[0] - exact).abs() / exact;
            println!("{}: relative error vs exp(-sin t) {rel:.3e}", case.name);
            if rel.is_nan() || rel > EXACT_RTOL_FACTOR_TOL * case.rtol {
                failures.push(format!("{}: {rel:e} from the exact solution", case.name));
            }
        }
        if case.problem == "arenstorf" {
            let (fsci_gap, scipy_gap) = (return_gap(y_end), return_gap(&answer.y_end));
            println!(
                "{}: return gap max|y(T) - y0| fsci {fsci_gap:.3e} SciPy {scipy_gap:.3e}",
                case.name
            );
            if fsci_gap.is_nan() || fsci_gap > RETURN_GAP_VS_SCIPY_TOL * scipy_gap {
                failures.push(format!(
                    "{}: return gap {fsci_gap:e} vs SciPy {scipy_gap:e}",
                    case.name
                ));
            }
        }
        ledger.compared(arm, &case.name, failures.len() == failures_before);
    }
    println!(
        "{compared} integrations compared ({dop853_compared} DOP853, {well_conditioned} well \
         conditioned); worst scaled y gap on those {worst_scaled:.3e}"
    );
    assert_eq!(compared, cases.len());
    assert!(dop853_compared > 0, "no DOP853 row was compared");
    assert!(
        well_conditioned >= MIN_WELL_CONDITIONED_ROWS,
        "only {well_conditioned} rows were well conditioned enough to compare y"
    );
    assert!(failures.is_empty(), "solve_ivp disagrees: {failures:#?}");
    // Each method has its own rows (RK23 runs at rtol 1e-6 only); each must compare all of them.
    let min_per_arm = METHODS
        .iter()
        .map(|m| cases.iter().filter(|c| c.method == *m).count())
        .min()
        .expect("METHODS is non-empty");
    ledger.finish(min_per_arm);
}

/// `t_eval` samples come from each method's own interpolant: SciPy's `RkDenseOutput` cubic
/// (RK23) and quartic (RK45), and DOP853's 7th-order polynomial over three extra stages, which
/// SciPy evaluates, and counts in nfev, on every step that holds a sample. Each row samples 21
/// evenly spaced times; every sample must lie within Y_SCALE_FACTOR_TOL units of SciPy's
/// `rtol·|y| + atol`, and nfev must differ from SciPy's by whole evaluation groups (3 for
/// DOP853, whose steps cost 12 and whose dense outputs cost 3; 6 for RK45; 3 for RK23) and by
/// at most COUNT_REL_TOL. Leaving DOP853's dense-output evaluations uncounted would miss
/// SciPy's nfev by ~15% on these rows.
#[test]
fn diff_integrate_rk_t_eval_samples() {
    let problems: [(&str, f64, Vec<f64>); 2] = [
        ("cos_decay", 10.0, vec![1.0]),
        ("lorenz", 1.0, vec![1.0, 1.0, 1.0]),
    ];
    let mut cases = Vec::new();
    for method in ["DOP853", "RK45", "RK23"] {
        for (problem, t_end, y0) in &problems {
            for rtol in [1e-6, 1e-10] {
                if method == "RK23" && rtol < 1e-8 {
                    continue;
                }
                cases.push(Case {
                    name: format!("{method}_{problem}_rtol{rtol:e}_t_eval"),
                    problem: (*problem).to_string(),
                    method: method.to_string(),
                    t_end: *t_end,
                    y0: y0.clone(),
                    rtol,
                    atol: 1e-3 * rtol,
                    t_eval: Some((0..=20).map(|i| t_end * f64::from(i) / 20.0).collect()),
                });
            }
        }
    }
    let Some(answers) = scipy_answers(&cases) else {
        return;
    };
    assert_eq!(
        answers.len(),
        cases.len(),
        "the oracle must answer every case"
    );
    let mut samples_compared = 0;
    let mut failures = Vec::new();
    let mut ledger = CompareLedger::new("diff_integrate_rk_t_eval_samples", &METHODS);
    for (case, answer) in cases.iter().zip(&answers) {
        let problem = case.problem.clone();
        let mut fun = move |t: f64, y: &[f64]| rhs(&problem, t, y);
        let t_eval = case.t_eval.as_deref().expect("t_eval rows");
        let options = SolveIvpOptions {
            t_span: (0.0, case.t_end),
            y0: &case.y0,
            method: method_of(&case.method),
            rtol: case.rtol,
            atol: ToleranceValue::Scalar(case.atol),
            t_eval: Some(t_eval),
            ..SolveIvpOptions::default()
        };
        let arm = case.method.as_str();
        let failures_before = failures.len();
        let result = solve_ivp(&mut fun, &options);
        if let Err(e) = &result {
            failures.push(format!("{}: fsci Err({e:?})", case.name));
        }
        let Some((answer, result)) = ledger.both(arm, &case.name, Some(answer), result.ok()) else {
            continue;
        };
        if result.y.len() != answer.ys.len() || result.status != 0 || answer.status != 0 {
            failures.push(format!(
                "{}: {} samples (SciPy {}), status fsci {} SciPy {}",
                case.name,
                result.y.len(),
                answer.ys.len(),
                result.status,
                answer.status
            ));
        }
        // Every sample flattened: the total length, and a NaN in any component, which the max
        // fold below would swallow.
        let scipy_flat: Vec<f64> = answer.ys.iter().flatten().copied().collect();
        let fsci_flat: Vec<f64> = result.y.iter().flatten().copied().collect();
        if ledger
            .slices(
                arm,
                &case.name,
                Some(scipy_flat.as_slice()),
                Some(fsci_flat.as_slice()),
            )
            .is_none()
        {
            failures.push(format!(
                "{}: samples {:?} against SciPy {:?}",
                case.name, result.y, answer.ys
            ));
            continue;
        }
        let mut worst = 0.0_f64;
        for (ours, theirs) in result.y.iter().zip(&answer.ys) {
            samples_compared += 1;
            for (f, s) in ours.iter().zip(theirs) {
                worst = worst.max((f - s).abs() / (case.rtol * s.abs() + case.atol));
            }
        }
        let group = if case.method == "RK45" { 6 } else { 3 };
        println!(
            "{}: worst sample gap {worst:.3e} units (SciPy 1-ulp envelope {:.3e}) | nfev fsci {} SciPy {}",
            case.name, answer.envelope, result.nfev, answer.nfev
        );
        if answer.envelope > WELL_CONDITIONED_ENVELOPE {
            failures.push(format!(
                "{}: SciPy's envelope {:e} is too wide to compare samples",
                case.name, answer.envelope
            ));
        }
        if worst.is_nan() || worst > Y_SCALE_FACTOR_TOL {
            failures.push(format!("{}: sample gap {worst:e} units", case.name));
        }
        if result.nfev.abs_diff(answer.nfev) % group != 0
            || count_gap(result.nfev, answer.nfev) > COUNT_REL_TOL
        {
            failures.push(format!(
                "{}: nfev {} vs SciPy {}",
                case.name, result.nfev, answer.nfev
            ));
        }
        ledger.compared(arm, &case.name, failures.len() == failures_before);
    }
    println!(
        "{samples_compared} t_eval samples compared over {} rows",
        cases.len()
    );
    assert_eq!(samples_compared, 21 * cases.len());
    assert!(
        failures.is_empty(),
        "t_eval samples disagree: {failures:#?}"
    );
    // Each method has its own rows (RK23 runs at rtol 1e-6 only); each must compare all of them.
    let min_per_arm = METHODS
        .iter()
        .map(|m| cases.iter().filter(|c| c.method == *m).count())
        .min()
        .expect("METHODS is non-empty");
    ledger.finish(min_per_arm);
}
